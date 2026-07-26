"""Small learned controllers over activation-state/action fibers.

These are deliberately low-capacity and inspectable.  They use measured
train-fold action responses as neighbor evidence, but test-query responses never
enter the context memory.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from geoprobe.control import geometry_map as gm


@dataclass(frozen=True)
class ActionFiberModelConfig:
    hidden_dim: int = 32
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    harm_penalty: float = 2.0
    listwise_weight: float = 1.0
    listwise_temperature: float = 0.25
    top_k: int = 12
    neighbor_pool_multiplier: int = 4
    state_dim: int = 64
    include_response: bool = False
    use_relation_features: bool = True
    objective: str = "reward"
    device: str = "cpu"
    seed: int = 0


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


class ActionFiberEncoder:
    def __init__(self, *, include_response: bool) -> None:
        self.include_response = include_response

    def fit(self, rows: list[dict]) -> "ActionFiberEncoder":
        self.vectorizer_ = DictVectorizer(sparse=False)
        raw = self.vectorizer_.fit_transform([self.features(row) for row in rows])
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(raw)
        self.dim_ = int(raw.shape[1])
        return self

    def features(self, row: dict) -> dict[str, str | float]:
        out = gm.action_features(row, include_response=self.include_response)
        for key, value in gm.route_features(row).items():
            out[f"route::{key}"] = value
        return out

    def transform(self, rows: list[dict]) -> np.ndarray:
        raw = self.vectorizer_.transform([self.features(row) for row in rows])
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler_.transform(raw).astype(np.float32)


class TrainNeighborIndex:
    """Route-conditioned nearest-neighbor memory over train action rows."""

    def __init__(self, *, top_k: int, pool_multiplier: int, state_dim: int, seed: int) -> None:
        self.top_k = int(top_k)
        self.pool_multiplier = max(int(pool_multiplier), 1)
        self.state_dim = max(int(state_dim), 1)
        self.seed = int(seed)

    def fit(
        self,
        rows: list[dict],
        state_vectors: dict[gm.StateLayerKey, np.ndarray],
    ) -> "TrainNeighborIndex":
        self.rows_ = [row for row in rows if _row_state_vector(row, state_vectors) is not None]
        if not self.rows_:
            raise ValueError("no train rows with state vectors for neighbor memory")
        unique_raw: dict[gm.StateLayerKey, np.ndarray] = {}
        for row in self.rows_:
            key = gm.state_layer_key(row)
            vec = _row_state_vector(row, state_vectors)
            if key is not None and vec is not None and key not in unique_raw:
                unique_raw[key] = np.asarray(vec, dtype=np.float64)
        if not unique_raw:
            raise ValueError("no unique train state vectors for neighbor memory")
        keys = list(unique_raw)
        raw = np.vstack([unique_raw[key] for key in keys]).astype(np.float64)
        self.raw_dim_ = int(raw.shape[1])
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(raw).astype(np.float32)
        self.projector_ = None
        if scaled.shape[0] > 1 and self.state_dim < scaled.shape[1]:
            n_components = min(self.state_dim, scaled.shape[0] - 1, scaled.shape[1])
            self.projector_ = PCA(
                n_components=n_components,
                svd_solver="randomized",
                random_state=self.seed,
            )
            projected = self.projector_.fit_transform(scaled).astype(np.float32)
        else:
            projected = scaled
        self.key_to_x_ = {key: projected[idx] for idx, key in enumerate(keys)}
        self.x_ = np.vstack([
            self.key_to_x_[gm.state_layer_key(row)]
            for row in self.rows_
        ]).astype(np.float32)
        self.response_ = np.asarray([
            [gm.fix_value(row), gm.harm_value(row), gm.target_value(row, "reward")]
            for row in self.rows_
        ], dtype=np.float32)
        self.group_to_indices_: dict[tuple[str, str], np.ndarray] = defaultdict(list)
        self.route_to_indices_: dict[str, np.ndarray] = defaultdict(list)
        for idx, row in enumerate(self.rows_):
            route = str(row.get("route_action") or "unknown")
            target = str(row.get("target_status") or "NONE")
            self.group_to_indices_[(route, target)].append(idx)
            self.route_to_indices_[route].append(idx)
        self.group_to_indices_ = {
            key: np.asarray(value, dtype=np.int64)
            for key, value in self.group_to_indices_.items()
        }
        self.route_to_indices_ = {
            key: np.asarray(value, dtype=np.int64)
            for key, value in self.route_to_indices_.items()
        }
        self.models_: dict[tuple[str, str], NearestNeighbors] = {}
        self.route_models_: dict[str, NearestNeighbors] = {}
        for key, indices in self.group_to_indices_.items():
            self.models_[key] = _fit_nn(self.x_[indices])
        for key, indices in self.route_to_indices_.items():
            self.route_models_[key] = _fit_nn(self.x_[indices])
        self.all_model_ = _fit_nn(self.x_)
        return self

    def transform_states(
        self,
        rows: list[dict],
        state_vectors: dict[gm.StateLayerKey, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        dim = int(self.x_.shape[1])
        out = np.zeros((len(rows), dim), dtype=np.float32)
        valid = np.zeros(len(rows), dtype=bool)
        raw_rows = []
        raw_positions = []
        for idx, row in enumerate(rows):
            vec = _row_state_vector(row, state_vectors)
            if vec is None:
                continue
            raw_rows.append(np.asarray(vec, dtype=np.float64))
            raw_positions.append(idx)
            valid[idx] = True
        if raw_rows:
            raw = np.vstack(raw_rows).astype(np.float64)
            scaled = self.scaler_.transform(raw).astype(np.float32)
            transformed = self.projector_.transform(scaled).astype(np.float32) if self.projector_ is not None else scaled
            for pos, value in zip(raw_positions, transformed, strict=True):
                out[pos] = value
        return out, valid

    def neighbor_indices(
        self,
        rows: list[dict],
        state_x: np.ndarray,
        *,
        exclude_same_state: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        width = max(1, self.top_k)
        indices = np.zeros((len(rows), width), dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=bool)
        for row_idx, row in enumerate(rows):
            route = str(row.get("route_action") or "unknown")
            target = str(row.get("target_status") or "NONE")
            candidate_indices, model = self._candidate_model(route, target)
            query = state_x[row_idx][None, :]
            n_pool = min(len(candidate_indices), max(width * self.pool_multiplier, width + 1))
            if n_pool <= 0:
                continue
            _dist, local = model.kneighbors(query, n_neighbors=n_pool)
            chosen = []
            query_state = gm.state_id(row)
            for local_idx in local[0]:
                global_idx = int(candidate_indices[int(local_idx)])
                if exclude_same_state and gm.state_id(self.rows_[global_idx]) == query_state:
                    continue
                chosen.append(global_idx)
                if len(chosen) >= width:
                    break
            if not chosen and exclude_same_state:
                for local_idx in local[0]:
                    chosen.append(int(candidate_indices[int(local_idx)]))
                    if len(chosen) >= width:
                        break
            for col, global_idx in enumerate(chosen[:width]):
                indices[row_idx, col] = global_idx
                mask[row_idx, col] = True
        return indices, mask

    def _candidate_model(self, route: str, target: str) -> tuple[np.ndarray, NearestNeighbors]:
        group = (route, target)
        if group in self.group_to_indices_:
            return self.group_to_indices_[group], self.models_[group]
        if route in self.route_to_indices_:
            return self.route_to_indices_[route], self.route_models_[route]
        return np.arange(len(self.rows_), dtype=np.int64), self.all_model_


class LocalActionAttentionNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.state_q = nn.Linear(state_dim, hidden_dim)
        self.action_q = nn.Linear(action_dim, hidden_dim)
        self.state_k = nn.Linear(state_dim, hidden_dim)
        self.action_k = nn.Linear(action_dim, hidden_dim)
        self.value = nn.Sequential(nn.Linear(hidden_dim + 3, hidden_dim), nn.ReLU())
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fix_head = nn.Linear(hidden_dim, 1)
        self.harm_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state_x: torch.Tensor,
        action_x: torch.Tensor,
        mem_state: torch.Tensor,
        mem_action: torch.Tensor,
        mem_response: torch.Tensor,
        mem_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.tanh(self.state_q(state_x) + self.action_q(action_x))
        k = torch.tanh(self.state_k(mem_state) + self.action_k(mem_action))
        value = self.value(torch.cat([k, mem_response], dim=-1))
        logits = torch.einsum("bd,bkd->bk", q, k) / max(float(q.shape[-1]) ** 0.5, 1.0)
        logits = torch.where(mem_mask, logits, torch.full_like(logits, -1e9))
        weights = torch.softmax(logits, dim=-1)
        evidence = torch.einsum("bk,bkd->bd", weights, value)
        hidden = self.out(torch.cat([q, evidence, q * evidence], dim=-1))
        return self.fix_head(hidden).squeeze(-1), self.harm_head(hidden).squeeze(-1), weights


class TypedActionFiberNet(nn.Module):
    RELATION_NAMES = [
        "same_full_action",
        "same_method_adjacent_layer",
        "same_method_adjacent_alpha",
        "same_method_any",
        "soft_target_partner",
        "route_matched_any",
    ]

    def __init__(self, state_dim: int, action_dim: int, relation_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.state = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.action = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.ReLU())
        self.relations = nn.ModuleList([
            nn.Sequential(nn.Linear(relation_dim, hidden_dim), nn.ReLU())
            for _ in self.RELATION_NAMES
        ])
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * (2 + len(self.RELATION_NAMES)), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fix_head = nn.Linear(hidden_dim, 1)
        self.harm_head = nn.Linear(hidden_dim, 1)

    def forward(self, state_x: torch.Tensor, action_x: torch.Tensor, relation_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts = [self.state(state_x), self.action(action_x)]
        for rel_idx, layer in enumerate(self.relations):
            parts.append(layer(relation_x[:, rel_idx, :]))
        hidden = self.out(torch.cat(parts, dim=-1))
        return self.fix_head(hidden).squeeze(-1), self.harm_head(hidden).squeeze(-1)


class _BaseActionFiberSelector:
    def __init__(self, config: ActionFiberModelConfig | None = None) -> None:
        self.config = config or ActionFiberModelConfig()
        self.device_ = resolve_device(self.config.device)

    def _fit_shared(self, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> list[dict]:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        train = [row for row in gm.candidate_rows(rows) if _row_state_vector(row, state_vectors) is not None]
        if not train:
            raise ValueError("no action-fiber train rows with state vectors")
        self.train_rows_ = train
        self.neighbors_ = TrainNeighborIndex(
            top_k=self.config.top_k,
            pool_multiplier=self.config.neighbor_pool_multiplier,
            state_dim=self.config.state_dim,
            seed=self.config.seed,
        ).fit(train, state_vectors)
        self.action_encoder_ = ActionFiberEncoder(include_response=self.config.include_response).fit(train)
        self.train_action_x_ = self.action_encoder_.transform(self.neighbors_.rows_)
        return train

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

    def _loss(
        self,
        rows: list[dict],
        fix_logits: torch.Tensor,
        harm_logits: torch.Tensor,
    ) -> torch.Tensor:
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

    def _state_action_arrays(
        self,
        rows: list[dict],
        state_vectors: dict[gm.StateLayerKey, np.ndarray],
        *,
        exclude_same_state: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state_x, valid = self.neighbors_.transform_states(rows, state_vectors)
        action_x = self.action_encoder_.transform(rows)
        neighbor_idx, neighbor_mask = self.neighbors_.neighbor_indices(
            rows,
            state_x,
            exclude_same_state=exclude_same_state,
        )
        return state_x, action_x, neighbor_idx, neighbor_mask, valid


class LocalActionAttentionSelector(_BaseActionFiberSelector):
    """Flexible baseline: action-conditioned attention over train neighbors."""

    def fit(self, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> "LocalActionAttentionSelector":
        train = self._fit_shared(rows, state_vectors)
        state_x, action_x, neighbor_idx, neighbor_mask, _valid = self._state_action_arrays(
            train,
            state_vectors,
            exclude_same_state=True,
        )
        self.model_ = LocalActionAttentionNet(
            state_dim=state_x.shape[1],
            action_dim=action_x.shape[1],
            hidden_dim=self.config.hidden_dim,
        ).to(self.device_)
        self._fit_arrays(train, state_x, action_x, neighbor_idx, neighbor_mask)
        return self

    def _fit_arrays(
        self,
        rows: list[dict],
        state_x: np.ndarray,
        action_x: np.ndarray,
        neighbor_idx: np.ndarray,
        neighbor_mask: np.ndarray,
    ) -> None:
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        sx = torch.tensor(state_x, device=self.device_)
        ax = torch.tensor(action_x, device=self.device_)
        ms = torch.tensor(self.neighbors_.x_[neighbor_idx], device=self.device_)
        ma = torch.tensor(self.train_action_x_[neighbor_idx], device=self.device_)
        mr = torch.tensor(self.neighbors_.response_[neighbor_idx], device=self.device_)
        mm = torch.tensor(neighbor_mask, device=self.device_, dtype=torch.bool)
        for _ in range(int(self.config.epochs)):
            opt.zero_grad()
            fix_logits, harm_logits, _weights = self.model_(sx, ax, ms, ma, mr, mm)
            loss = self._loss(rows, fix_logits, harm_logits)
            loss.backward()
            opt.step()
        self.model_.eval()

    def score(self, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> np.ndarray:
        if not rows:
            return np.asarray([], dtype=np.float64)
        state_x, action_x, neighbor_idx, neighbor_mask, valid = self._state_action_arrays(
            rows,
            state_vectors,
            exclude_same_state=False,
        )
        with torch.no_grad():
            fix_logits, harm_logits, _weights = self.model_(
                torch.tensor(state_x, device=self.device_),
                torch.tensor(action_x, device=self.device_),
                torch.tensor(self.neighbors_.x_[neighbor_idx], device=self.device_),
                torch.tensor(self.train_action_x_[neighbor_idx], device=self.device_),
                torch.tensor(self.neighbors_.response_[neighbor_idx], device=self.device_),
                torch.tensor(neighbor_mask, device=self.device_, dtype=torch.bool),
            )
            scores = self._selection_scores(fix_logits, harm_logits).detach().cpu().numpy().astype(np.float64)
        scores[~valid] = -1e9
        return scores

    def explain_choice(self, row: dict, state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> dict:
        state_x, action_x, neighbor_idx, neighbor_mask, valid = self._state_action_arrays(
            [row],
            state_vectors,
            exclude_same_state=False,
        )
        if not bool(valid[0]):
            return {"reason": "missing_state"}
        with torch.no_grad():
            _fix, _harm, weights = self.model_(
                torch.tensor(state_x, device=self.device_),
                torch.tensor(action_x, device=self.device_),
                torch.tensor(self.neighbors_.x_[neighbor_idx], device=self.device_),
                torch.tensor(self.train_action_x_[neighbor_idx], device=self.device_),
                torch.tensor(self.neighbors_.response_[neighbor_idx], device=self.device_),
                torch.tensor(neighbor_mask, device=self.device_, dtype=torch.bool),
            )
        weights_np = weights.detach().cpu().numpy()[0]
        order = np.argsort(-weights_np)
        top = []
        for pos in order[: min(5, len(order))]:
            if not bool(neighbor_mask[0, pos]):
                continue
            neighbor = self.neighbors_.rows_[int(neighbor_idx[0, pos])]
            top.append({
                "weight": float(weights_np[pos]),
                "conversation_id": gm.state_id(neighbor),
                "method": neighbor.get("method"),
                "target_status": neighbor.get("target_status"),
                "layer": neighbor.get("layer"),
                "alpha": neighbor.get("alpha"),
                "reward": gm.target_value(neighbor, self.config.objective),
                "fix": gm.fix_value(neighbor),
                "harm": gm.harm_value(neighbor),
            })
        return {"model": "local_action_attention", "top_neighbors": top}


class TypedActionFiberControlNetSelector(_BaseActionFiberSelector):
    """Structured model: relation-specific evidence pooling over action fibers."""

    def fit(self, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> "TypedActionFiberControlNetSelector":
        train = self._fit_shared(rows, state_vectors)
        state_x, action_x, neighbor_idx, neighbor_mask, _valid = self._state_action_arrays(
            train,
            state_vectors,
            exclude_same_state=True,
        )
        relation_x = self._relation_features(train, neighbor_idx, neighbor_mask)
        self.model_ = TypedActionFiberNet(
            state_dim=state_x.shape[1],
            action_dim=action_x.shape[1],
            relation_dim=relation_x.shape[-1],
            hidden_dim=self.config.hidden_dim,
        ).to(self.device_)
        self._fit_arrays(train, state_x, action_x, relation_x)
        return self

    def _relation_features(self, rows: list[dict], neighbor_idx: np.ndarray, neighbor_mask: np.ndarray) -> np.ndarray:
        relation_names = TypedActionFiberNet.RELATION_NAMES
        out = np.zeros((len(rows), len(relation_names), 4), dtype=np.float32)
        if not self.config.use_relation_features:
            return out
        for row_idx, row in enumerate(rows):
            buckets = {name: [] for name in relation_names}
            for pos in range(neighbor_idx.shape[1]):
                if not bool(neighbor_mask[row_idx, pos]):
                    continue
                neighbor = self.neighbors_.rows_[int(neighbor_idx[row_idx, pos])]
                for relation in _relations_between(row, neighbor):
                    if relation in buckets:
                        buckets[relation].append(neighbor)
            for rel_idx, name in enumerate(relation_names):
                bucket = buckets[name]
                if not bucket:
                    continue
                out[row_idx, rel_idx] = np.asarray([
                    min(len(bucket) / max(float(self.config.top_k), 1.0), 1.0),
                    float(np.mean([gm.fix_value(item) for item in bucket])),
                    float(np.mean([gm.harm_value(item) for item in bucket])),
                    float(np.mean([gm.target_value(item, self.config.objective) for item in bucket])),
                ], dtype=np.float32)
        return out

    def _fit_arrays(self, rows: list[dict], state_x: np.ndarray, action_x: np.ndarray, relation_x: np.ndarray) -> None:
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        sx = torch.tensor(state_x, device=self.device_)
        ax = torch.tensor(action_x, device=self.device_)
        rx = torch.tensor(relation_x, device=self.device_)
        for _ in range(int(self.config.epochs)):
            opt.zero_grad()
            fix_logits, harm_logits = self.model_(sx, ax, rx)
            loss = self._loss(rows, fix_logits, harm_logits)
            loss.backward()
            opt.step()
        self.model_.eval()

    def score(self, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> np.ndarray:
        if not rows:
            return np.asarray([], dtype=np.float64)
        state_x, action_x, neighbor_idx, neighbor_mask, valid = self._state_action_arrays(
            rows,
            state_vectors,
            exclude_same_state=False,
        )
        relation_x = self._relation_features(rows, neighbor_idx, neighbor_mask)
        with torch.no_grad():
            fix_logits, harm_logits = self.model_(
                torch.tensor(state_x, device=self.device_),
                torch.tensor(action_x, device=self.device_),
                torch.tensor(relation_x, device=self.device_),
            )
            scores = self._selection_scores(fix_logits, harm_logits).detach().cpu().numpy().astype(np.float64)
        scores[~valid] = -1e9
        return scores

    def explain_choice(self, row: dict, state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> dict:
        state_x, _action_x, neighbor_idx, neighbor_mask, valid = self._state_action_arrays(
            [row],
            state_vectors,
            exclude_same_state=False,
        )
        if not bool(valid[0]):
            return {"reason": "missing_state"}
        relation_x = self._relation_features([row], neighbor_idx, neighbor_mask)[0]
        names = TypedActionFiberNet.RELATION_NAMES
        return {
            "model": "typed_action_fiber_control_net",
            "relations": {
                name: {
                    "support_fraction": float(relation_x[idx, 0]),
                    "mean_fix": float(relation_x[idx, 1]),
                    "mean_harm": float(relation_x[idx, 2]),
                    "mean_reward": float(relation_x[idx, 3]),
                }
                for idx, name in enumerate(names)
            },
        }


def _fit_nn(x: np.ndarray) -> NearestNeighbors:
    model = NearestNeighbors(metric="euclidean", algorithm="auto")
    model.fit(x)
    return model


def _row_state_vector(row: dict, state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> np.ndarray | None:
    key = gm.state_layer_key(row)
    if key is None:
        return None
    vec = state_vectors.get(key)
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32)


def _relations_between(row: dict, neighbor: dict) -> list[str]:
    out = []
    if str(row.get("route_action") or "unknown") == str(neighbor.get("route_action") or "unknown"):
        out.append("route_matched_any")
    if gm.compact_action_key(row) == gm.compact_action_key(neighbor):
        out.append("same_full_action")
    if str(row.get("method")) == str(neighbor.get("method")):
        out.append("same_method_any")
        if str(row.get("target_status")) == str(neighbor.get("target_status")):
            if _adjacent_layer(row.get("layer"), neighbor.get("layer")):
                out.append("same_method_adjacent_layer")
            if _adjacent_alpha(row.get("alpha"), neighbor.get("alpha")):
                out.append("same_method_adjacent_alpha")
    if gm.z2_partner_key(row) == gm.compact_action_key(neighbor):
        out.append("soft_target_partner")
    return out


def _adjacent_layer(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    try:
        return 0.0 < abs(float(left) - float(right)) <= 8.0
    except (TypeError, ValueError):
        return False


def _adjacent_alpha(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    try:
        return 0.0 < abs(float(left) - float(right)) <= 48.0
    except (TypeError, ValueError):
        return False
