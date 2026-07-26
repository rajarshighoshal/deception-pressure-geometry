"""Full-field local deformation selectors for action-response control.

The model treats each candidate action as a direction in the same ambient
activation space as the state.  It does not expose the 4096 coordinates to a
large neural net; instead it learns over constrained full-field interactions
between the state, the action direction, and train-fold local response
summaries.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from geoprobe.control import geometry_map as gm
from geoprobe.control.action_fiber_models import resolve_device


RAW_METHOD_FOR_POLICY = {
    "global_mean_gated": "global_mean",
    "global_probe_gated": "global_probe",
    "random_gated": "random_global",
}


@dataclass(frozen=True)
class LocalDeformationConfig:
    hidden_dim: int = 32
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    harm_penalty: float = 2.0
    listwise_weight: float = 1.0
    listwise_temperature: float = 0.25
    top_k: int = 12
    neighbor_pool_multiplier: int = 4
    geometry_dim: int = 0
    include_metadata: bool = True
    use_vector_geometry: bool = True
    use_local_geometry: bool = True
    potential_only: bool = False
    objective: str = "reward"
    device: str = "cpu"
    seed: int = 0


@dataclass
class _ResponseSummary:
    count: int
    fix_sum: float
    harm_sum: float
    reward_sum: float
    reward_vec_sum: np.ndarray
    fix_vec_sum: np.ndarray
    harm_vec_sum: np.ndarray

    @classmethod
    def empty(cls, dim: int, *, with_vectors: bool = True) -> "_ResponseSummary":
        if not with_vectors:
            z = np.zeros(0, dtype=np.float32)
            return cls(0, 0.0, 0.0, 0.0, z.copy(), z.copy(), z.copy())
        z = np.zeros(dim, dtype=np.float32)
        return cls(0, 0.0, 0.0, 0.0, z.copy(), z.copy(), z.copy())

    def add(self, row: dict, direction: np.ndarray, objective: str) -> None:
        fix = gm.fix_value(row)
        harm = gm.harm_value(row)
        reward = gm.target_value(row, objective)
        self.count += 1
        self.fix_sum += fix
        self.harm_sum += harm
        self.reward_sum += reward
        if self.reward_vec_sum.size:
            self.reward_vec_sum += float(reward) * direction
            self.fix_vec_sum += float(fix) * direction
            self.harm_vec_sum += float(harm) * direction


class SteeringSpecLookup:
    """Lazy lookup for full steering vectors from a precomputed spec bank."""

    def __init__(self, bank_dir: Path) -> None:
        self.bank_dir = Path(bank_dir)
        manifest_path = self.bank_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing steering spec bank manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("kind") != "steering_spec_bank":
            raise ValueError(f"{self.bank_dir} is not a steering_spec_bank")
        self.manifest = manifest
        self._vectors: dict[tuple[str, int, str, str], np.ndarray] = {}
        self._meta: dict[tuple[str, int, str, str], dict] = {}
        self._load_all()

    def _load_all(self) -> None:
        specs_dir = self.bank_dir / "specs"
        for shard in self.manifest.get("shards", []):
            stem = str(shard["stem"])
            vectors = np.load(specs_dir / f"{stem}.npz")["vectors"]
            with (specs_dir / f"{stem}.jsonl").open() as handle:
                for idx, line in enumerate(handle):
                    meta = json.loads(line)
                    key = (
                        str(meta["conversation_id"]),
                        int(meta["layer"]),
                        str(meta["method"]),
                        str(meta["target_status"]),
                    )
                    self._vectors[key] = np.asarray(vectors[idx], dtype=np.float32)
                    self._meta[key] = meta

    def key_for_row(self, row: dict) -> tuple[str, int, str, str] | None:
        if gm.is_baseline(row) or row.get("layer") is None or row.get("target_status") not in {"PASS", "FAIL"}:
            return None
        method = str(row.get("original_method") or row.get("method") or "")
        method = RAW_METHOD_FOR_POLICY.get(method, method)
        return (
            gm.state_id(row),
            int(row["layer"]),
            method,
            str(row["target_status"]),
        )

    def vector_for(self, row: dict) -> np.ndarray | None:
        key = self.key_for_row(row)
        return None if key is None else self._vectors.get(key)

    def meta_for(self, row: dict) -> dict | None:
        key = self.key_for_row(row)
        return None if key is None else self._meta.get(key)


class _FullFieldProjector:
    def __init__(self, geometry_dim: int, seed: int) -> None:
        self.geometry_dim = int(geometry_dim)
        self.seed = int(seed)

    def fit(self, keys: list[gm.StateLayerKey], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> "_FullFieldProjector":
        self.keys_ = [key for key in keys if key in state_vectors]
        if not self.keys_:
            raise ValueError("no state vectors for full-field projector")
        raw = np.vstack([state_vectors[key] for key in self.keys_]).astype(np.float64)
        self.raw_dim_ = int(raw.shape[1])
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(raw).astype(np.float32)
        self.projector_ = None
        if self.geometry_dim > 0 and self.geometry_dim < scaled.shape[1] and scaled.shape[0] > 1:
            n_components = min(self.geometry_dim, scaled.shape[0] - 1, scaled.shape[1])
            self.projector_ = PCA(n_components=n_components, svd_solver="randomized", random_state=self.seed)
            self.projector_.fit(scaled)
        return self

    def transform_state(self, vec: np.ndarray) -> np.ndarray:
        scaled = self.scaler_.transform(np.asarray(vec, dtype=np.float64)[None, :]).astype(np.float32)
        if self.projector_ is not None:
            scaled = self.projector_.transform(scaled).astype(np.float32)
        postmap = getattr(self, "z_postmap_", None)  # covariant transfer hook (default off)
        return scaled[0] if postmap is None else postmap(scaled)[0].astype(np.float32)

    def transform_direction(self, vec: np.ndarray) -> np.ndarray:
        scale = np.asarray(self.scaler_.scale_, dtype=np.float32)
        scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
        out = np.asarray(vec, dtype=np.float32) / scale
        if self.projector_ is not None:
            out = out @ np.asarray(self.projector_.components_, dtype=np.float32).T
        return out.astype(np.float32)


class _MetadataEncoder:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def fit(self, rows: list[dict]) -> "_MetadataEncoder":
        self.vectorizer_ = DictVectorizer(sparse=False)
        if not self.enabled:
            self.dim_ = 0
            return self
        raw = self.vectorizer_.fit_transform([self._features(row) for row in rows])
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(raw)
        self.dim_ = int(raw.shape[1])
        return self

    def _features(self, row: dict) -> dict[str, str | float]:
        out = gm.action_features(row, include_response=False)
        for key, value in gm.route_features(row).items():
            out[f"route::{key}"] = value
        return out

    def transform(self, rows: list[dict]) -> np.ndarray:
        if not self.enabled:
            return np.zeros((len(rows), 0), dtype=np.float32)
        raw = self.vectorizer_.transform([self._features(row) for row in rows])
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler_.transform(raw).astype(np.float32)


class _DeformationNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fix_head = nn.Linear(hidden_dim, 1)
        self.harm_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.fix_head(h).squeeze(-1), self.harm_head(h).squeeze(-1)


class FullFieldLocalDeformationSelector:
    """Low-capacity selector over full-field local deformation features."""

    def __init__(self, spec_lookup: SteeringSpecLookup, config: LocalDeformationConfig | None = None) -> None:
        self.spec_lookup = spec_lookup
        self.config = config or LocalDeformationConfig()
        self.device_ = resolve_device(self.config.device)

    def fit(
        self,
        rows: list[dict],
        state_vectors: dict[gm.StateLayerKey, np.ndarray],
    ) -> "FullFieldLocalDeformationSelector":
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        train = [
            row for row in gm.candidate_rows(rows)
            if gm.state_layer_key(row) in state_vectors and self.spec_lookup.vector_for(row) is not None
        ]
        if not train:
            raise ValueError("no train rows with full-field state/action vectors")
        self.train_rows_ = train
        keys = sorted({gm.state_layer_key(row) for row in train if gm.state_layer_key(row) is not None})
        self.projector_ = _FullFieldProjector(self.config.geometry_dim, self.config.seed).fit(keys, state_vectors)
        self.state_by_key_ = {key: self.projector_.transform_state(state_vectors[key]) for key in keys}
        self.dir_by_row_id_: dict[int, np.ndarray] = {}
        for row in train:
            vec = self.spec_lookup.vector_for(row)
            if vec is not None:
                self.dir_by_row_id_[id(row)] = self.projector_.transform_direction(vec)
        self._fit_neighbors_and_summaries(train)
        self.metadata_ = _MetadataEncoder(enabled=self.config.include_metadata and not self.config.potential_only).fit(train)
        features, valid = self._features(train, state_vectors, exclude_same_state=True)
        if not bool(valid.all()):
            raise ValueError("internal error: fit-time full-field validity diverged from train filtering")
        self.model_ = _DeformationNet(features.shape[1], self.config.hidden_dim).to(self.device_)
        self._fit_arrays(train, features)
        return self

    def _fit_neighbors_and_summaries(self, rows: list[dict]) -> None:
        self.state_ids_: list[str] = []
        state_keys: list[gm.StateLayerKey] = []
        seen = set()
        for row in rows:
            key = gm.state_layer_key(row)
            if key is None or key in seen:
                continue
            seen.add(key)
            state_keys.append(key)
            self.state_ids_.append(key[0])
        self.state_x_ = np.vstack([self.state_by_key_[key] for key in state_keys]).astype(np.float32)
        self.state_index_by_id_ = {state_id: idx for idx, state_id in enumerate(self.state_ids_)}
        self.group_to_state_indices_: dict[tuple[str, str], np.ndarray] = defaultdict(list)
        self.route_to_state_indices_: dict[str, np.ndarray] = defaultdict(list)
        for row in rows:
            key = gm.state_layer_key(row)
            if key is None:
                continue
            idx = self.state_index_by_id_.get(key[0])
            if idx is None:
                continue
            route = str(row.get("route_action") or "unknown")
            target = str(row.get("target_status") or "NONE")
            self.group_to_state_indices_[(route, target)].append(idx)
            self.route_to_state_indices_[route].append(idx)
        self.group_to_state_indices_ = {
            key: np.asarray(sorted(set(value)), dtype=np.int64)
            for key, value in self.group_to_state_indices_.items()
        }
        self.route_to_state_indices_ = {
            key: np.asarray(sorted(set(value)), dtype=np.int64)
            for key, value in self.route_to_state_indices_.items()
        }
        self.group_models_ = {
            key: _fit_nn(self.state_x_[indices])
            for key, indices in self.group_to_state_indices_.items()
        }
        self.route_models_ = {
            key: _fit_nn(self.state_x_[indices])
            for key, indices in self.route_to_state_indices_.items()
        }
        self.all_model_ = _fit_nn(self.state_x_)
        self.summary_any_: dict[tuple[str, str, str], _ResponseSummary] = {}
        self.summary_method_: dict[tuple[str, str, str, str], _ResponseSummary] = {}
        self.summary_action_: dict[tuple[str, str, str, tuple], _ResponseSummary] = {}
        dim = int(self.state_x_.shape[1])
        for row in rows:
            key = gm.state_layer_key(row)
            direction = self.dir_by_row_id_.get(id(row))
            if key is None or direction is None:
                continue
            state_id = key[0]
            route = str(row.get("route_action") or "unknown")
            target = str(row.get("target_status") or "NONE")
            method = str(row.get("method") or "unknown")
            action = gm.compact_action_key(row)
            for store_key, store in [
                ((state_id, route, target), self.summary_any_),
                ((state_id, route, target, method), self.summary_method_),
                ((state_id, route, target, action), self.summary_action_),
            ]:
                if store_key not in store:
                    store[store_key] = _ResponseSummary.empty(
                        dim,
                        with_vectors=store is not self.summary_action_,
                    )
                store[store_key].add(row, _unit(direction), self.config.objective)

    def _candidate_state_model(self, route: str, target: str) -> tuple[np.ndarray, NearestNeighbors]:
        group = (route, target)
        if group in self.group_to_state_indices_:
            return self.group_to_state_indices_[group], self.group_models_[group]
        if route in self.route_to_state_indices_:
            return self.route_to_state_indices_[route], self.route_models_[route]
        return np.arange(len(self.state_ids_), dtype=np.int64), self.all_model_

    def _neighbor_state_ids(self, row: dict, state_x: np.ndarray, *, exclude_same_state: bool) -> tuple[list[str], np.ndarray]:
        route = str(row.get("route_action") or "unknown")
        target = str(row.get("target_status") or "NONE")
        candidate_indices, model = self._candidate_state_model(route, target)
        width = max(1, int(self.config.top_k))
        n_pool = min(len(candidate_indices), max(width * int(self.config.neighbor_pool_multiplier), width + 1))
        if n_pool <= 0:
            return [], np.asarray([], dtype=np.float32)
        dist, local = model.kneighbors(state_x[None, :], n_neighbors=n_pool)
        ids = []
        vals = []
        query_id = gm.state_id(row)
        for local_idx, d in zip(local[0], dist[0], strict=True):
            global_idx = int(candidate_indices[int(local_idx)])
            state_id = self.state_ids_[global_idx]
            if exclude_same_state and state_id == query_id:
                continue
            ids.append(state_id)
            vals.append(float(d))
            if len(ids) >= width:
                break
        if not ids and exclude_same_state:
            for local_idx, d in zip(local[0], dist[0], strict=True):
                global_idx = int(candidate_indices[int(local_idx)])
                ids.append(self.state_ids_[global_idx])
                vals.append(float(d))
                if len(ids) >= width:
                    break
        return ids, np.asarray(vals, dtype=np.float32)

    def _features(
        self,
        rows: list[dict],
        state_vectors: dict[gm.StateLayerKey, np.ndarray],
        *,
        exclude_same_state: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        numeric = []
        valid = np.zeros(len(rows), dtype=bool)
        for idx, row in enumerate(rows):
            feats, ok = self._numeric_features_one(row, state_vectors, exclude_same_state=exclude_same_state)
            numeric.append(feats)
            valid[idx] = ok
        numeric_x = np.asarray(numeric, dtype=np.float32)
        if not hasattr(self, "numeric_scaler_"):
            self.numeric_scaler_ = StandardScaler()
            self.numeric_scaler_.fit(numeric_x[valid] if bool(valid.any()) else numeric_x)
        numeric_x = self.numeric_scaler_.transform(numeric_x).astype(np.float32)
        meta_x = self.metadata_.transform(rows)
        out = np.hstack([numeric_x, meta_x]).astype(np.float32)
        return out, valid

    def _numeric_features_one(
        self,
        row: dict,
        state_vectors: dict[gm.StateLayerKey, np.ndarray],
        *,
        exclude_same_state: bool,
    ) -> tuple[list[float], bool]:
        key = gm.state_layer_key(row)
        raw_d = self.spec_lookup.vector_for(row)
        if key is None or key not in state_vectors or raw_d is None:
            return [0.0] * 42, False
        x = self.projector_.transform_state(state_vectors[key])
        d = self.projector_.transform_direction(raw_d)
        d_u = _unit(d)
        raw_x = np.asarray(state_vectors[key], dtype=np.float32)
        raw_d = np.asarray(raw_d, dtype=np.float32)
        route = str(row.get("route_action") or "unknown")
        target = str(row.get("target_status") or "NONE")
        method = str(row.get("method") or "unknown")
        action = gm.compact_action_key(row)
        neighbor_ids, distances = self._neighbor_state_ids(row, x, exclude_same_state=exclude_same_state)
        local_d = d_u if self.config.use_vector_geometry or self.config.potential_only else np.zeros_like(d_u)
        local = self._local_features(
            neighbor_ids,
            distances,
            local_d,
            route,
            target,
            method,
            action,
            include_vector_features=self.config.use_vector_geometry or self.config.potential_only,
        )
        raw_cos = _cos(raw_x, raw_d) if self.config.geometry_dim == 0 else 0.0
        base = [
            _norm(x),
            _norm(d),
            _cos(x, d),
            raw_cos,
            float(row.get("alpha") or 0.0),
            0.0,
        ]
        if self.config.potential_only:
            # Keep only the scalar-field normal analogue plus route metadata.
            keep = base[:4] + local[:10]
            return _pad(keep, 42), True
        if not self.config.use_vector_geometry:
            base = [0.0 for _ in base]
        if not self.config.use_local_geometry:
            local = [0.0 for _ in local]
        return _pad(base + local, 42), True

    def _local_features(
        self,
        neighbor_ids: list[str],
        distances: np.ndarray,
        d_u: np.ndarray,
        route: str,
        target: str,
        method: str,
        action: tuple,
        *,
        include_vector_features: bool,
    ) -> list[float]:
        if not neighbor_ids:
            return [0.0] * 36
        weights = _distance_weights(distances)
        buckets = [
            [self.summary_any_.get((sid, route, target)) for sid in neighbor_ids],
            [self.summary_method_.get((sid, route, target, method)) for sid in neighbor_ids],
            [self.summary_action_.get((sid, route, target, action)) for sid in neighbor_ids],
        ]
        out = [
            float(len(neighbor_ids)),
            float(np.mean(distances)) if len(distances) else 0.0,
            float(np.min(distances)) if len(distances) else 0.0,
        ]
        for summaries in buckets:
            out.extend(_bucket_features(summaries, weights, d_u, include_vector_features=include_vector_features))
        return _pad(out, 36)

    def _fit_arrays(self, rows: list[dict], features: np.ndarray) -> None:
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        x = torch.tensor(features, device=self.device_)
        for _ in range(int(self.config.epochs)):
            opt.zero_grad()
            fix_logits, harm_logits = self.model_(x)
            loss = self._loss(rows, fix_logits, harm_logits)
            loss.backward()
            opt.step()
        self.model_.eval()

    def _labels(self, rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fix = torch.tensor([gm.fix_value(row) for row in rows], device=self.device_)
        harm = torch.tensor([gm.harm_value(row) for row in rows], device=self.device_)
        reward = torch.tensor([gm.target_value(row, self.config.objective) for row in rows], device=self.device_)
        fix_mask = torch.tensor([str(row.get("status_class", "")).startswith("false_") for row in rows], device=self.device_, dtype=torch.bool)
        harm_mask = torch.tensor([str(row.get("status_class", "")).startswith("honest_") for row in rows], device=self.device_, dtype=torch.bool)
        group_names = [gm.state_id(row) for row in rows]
        group_ids = {name: idx for idx, name in enumerate(sorted(set(group_names)))}
        groups = torch.tensor([group_ids[name] for name in group_names], device=self.device_, dtype=torch.long)
        return fix, harm, reward, fix_mask, harm_mask, groups

    def _selection_scores(self, fix_logits: torch.Tensor, harm_logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(fix_logits) - float(self.config.harm_penalty) * torch.sigmoid(harm_logits)

    def _loss(self, rows: list[dict], fix_logits: torch.Tensor, harm_logits: torch.Tensor) -> torch.Tensor:
        fix, harm, reward, fix_mask, harm_mask, groups = self._labels(rows)
        zero = fix_logits.sum() * 0.0
        fix_loss = F.binary_cross_entropy_with_logits(fix_logits[fix_mask], fix[fix_mask]) if bool(fix_mask.any()) else zero
        harm_loss = F.binary_cross_entropy_with_logits(harm_logits[harm_mask], harm[harm_mask]) if bool(harm_mask.any()) else zero
        rank_loss = self._listwise_loss(self._selection_scores(fix_logits, harm_logits), reward, groups)
        return fix_loss + float(self.config.harm_penalty) * harm_loss + float(self.config.listwise_weight) * rank_loss

    def _listwise_loss(self, scores: torch.Tensor, reward: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        losses = []
        temp = max(float(self.config.listwise_temperature), 1e-4)
        for group in torch.unique(groups):
            idx = torch.where(groups == group)[0]
            if int(idx.numel()) < 2:
                continue
            local_reward = reward[idx]
            if float((local_reward.max() - local_reward.min()).detach().cpu()) <= 1e-8:
                continue
            target = torch.softmax(local_reward / temp, dim=0)
            pred = torch.log_softmax(scores[idx] / temp, dim=0)
            losses.append(-(target * pred).sum())
        if not losses:
            return scores.sum() * 0.0
        return torch.stack(losses).mean()

    def score(self, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> np.ndarray:
        if not rows:
            return np.asarray([], dtype=np.float64)
        features, valid = self._features(rows, state_vectors, exclude_same_state=False)
        with torch.no_grad():
            fix_logits, harm_logits = self.model_(torch.tensor(features, device=self.device_))
            scores = self._selection_scores(fix_logits, harm_logits).detach().cpu().numpy().astype(np.float64)
        scores[~valid] = -1e9
        return scores

    def explain_choice(self, row: dict, state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> dict:
        key = gm.state_layer_key(row)
        raw_d = self.spec_lookup.vector_for(row)
        if key is None or key not in state_vectors or raw_d is None:
            return {"reason": "missing_full_field_vector"}
        x = self.projector_.transform_state(state_vectors[key])
        neighbor_ids, distances = self._neighbor_state_ids(row, x, exclude_same_state=False)
        return {
            "model": "full_field_local_deformation",
            "geometry_dim": int(self.config.geometry_dim),
            "feature_mode": self._feature_mode(),
            "neighbor_state_ids": neighbor_ids[:5],
            "neighbor_distances": [float(v) for v in distances[:5]],
            "spec_key": self.spec_lookup.key_for_row(row),
        }

    def _feature_mode(self) -> str:
        if self.config.potential_only:
            return "option1_potential_only"
        if not self.config.use_vector_geometry:
            return "no_vector_geometry"
        if not self.config.use_local_geometry:
            return "no_local_geometry"
        if not self.config.include_metadata:
            return "no_metadata"
        if self.config.geometry_dim > 0:
            return f"pca{self.config.geometry_dim}_deformation"
        return "full_field_deformation"


def _fit_nn(x: np.ndarray) -> NearestNeighbors:
    model = NearestNeighbors(metric="euclidean", algorithm="auto")
    model.fit(x)
    return model


def _bucket_features(
    summaries: list[_ResponseSummary | None],
    weights: np.ndarray,
    d_u: np.ndarray,
    *,
    include_vector_features: bool,
) -> list[float]:
    vals = [(summary, float(weight)) for summary, weight in zip(summaries, weights, strict=True) if summary is not None and summary.count > 0]
    if not vals:
        return [0.0] * 11
    total_w = sum(weight for _summary, weight in vals)
    total_count = sum(summary.count for summary, _weight in vals)
    fix = sum(weight * summary.fix_sum / max(summary.count, 1) for summary, weight in vals) / max(total_w, 1e-12)
    harm = sum(weight * summary.harm_sum / max(summary.count, 1) for summary, weight in vals) / max(total_w, 1e-12)
    reward = sum(weight * summary.reward_sum / max(summary.count, 1) for summary, weight in vals) / max(total_w, 1e-12)
    if include_vector_features:
        vector_vals = [
            (summary, weight)
            for summary, weight in vals
            if summary.reward_vec_sum.size == d_u.size
        ]
    else:
        vector_vals = []
    if vector_vals:
        vector_w = sum(weight for _summary, weight in vector_vals)
        reward_vec = sum(
            (weight * summary.reward_vec_sum / max(summary.count, 1) for summary, weight in vector_vals),
            start=np.zeros_like(d_u),
        ) / max(vector_w, 1e-12)
        fix_vec = sum(
            (weight * summary.fix_vec_sum / max(summary.count, 1) for summary, weight in vector_vals),
            start=np.zeros_like(d_u),
        ) / max(vector_w, 1e-12)
        harm_vec = sum(
            (weight * summary.harm_vec_sum / max(summary.count, 1) for summary, weight in vector_vals),
            start=np.zeros_like(d_u),
        ) / max(vector_w, 1e-12)
    else:
        reward_vec = np.zeros_like(d_u)
        fix_vec = np.zeros_like(d_u)
        harm_vec = np.zeros_like(d_u)
    return [
        float(total_count),
        float(total_w),
        float(fix),
        float(harm),
        float(reward),
        _norm(reward_vec),
        _cos(d_u, reward_vec),
        _cos(d_u, fix_vec),
        _cos(d_u, harm_vec),
        _cos(fix_vec, harm_vec),
        _norm(fix_vec - harm_vec),
    ]


def _distance_weights(distances: np.ndarray) -> np.ndarray:
    if len(distances) == 0:
        return distances.astype(np.float32)
    scale = float(np.median(distances)) if len(distances) else 1.0
    scale = max(scale, 1e-6)
    return np.exp(-distances.astype(np.float32) / scale)


def _unit(x: np.ndarray) -> np.ndarray:
    n = _norm(x)
    if n <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (np.asarray(x, dtype=np.float32) / n).astype(np.float32)


def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float32)))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    an = _norm(a)
    bn = _norm(b)
    if an <= 1e-12 or bn <= 1e-12:
        return 0.0
    return float(np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)) / (an * bn))


def _pad(values: list[float], width: int) -> list[float]:
    out = [float(v) if np.isfinite(float(v)) else 0.0 for v in values[:width]]
    if len(out) < width:
        out.extend([0.0] * (width - len(out)))
    return out
