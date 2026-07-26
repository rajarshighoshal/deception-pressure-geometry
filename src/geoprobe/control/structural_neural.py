"""Neural structural controllers over activation-map/action fields."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from geoprobe.control import geometry_map as gm
from geoprobe.control.action_fiber_models import resolve_device
from geoprobe.control.chart_atlas import ChartAtlas, ChartAtlasConfig
from geoprobe.control.product_z2_graph import build_product_z2_graph


@dataclass(frozen=True)
class StructuralNeuralConfig:
    chart: ChartAtlasConfig = field(default_factory=ChartAtlasConfig)
    model: Literal[
        "equivariant_chart_action_field",
        "neural_atlas_control_field",
        "typed_transport_product_graph_controller",
    ] = "equivariant_chart_action_field"
    include_response: bool = False
    hidden_dim: int = 48
    latent_dim: int = 16
    pre_pca_dim: int = 24
    graph_layers: int = 2
    epochs: int = 40
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    harm_penalty: float = 2.0
    ae_weight: float = 0.2
    listwise_weight: float = 1.0
    listwise_temperature: float = 0.25
    chart_prior_weight: float = 1.0
    chart_distill_weight: float = 0.1
    use_state: bool = True
    use_chart: bool = True
    use_action: bool = True
    use_route: bool = True
    use_z2: bool = True
    transport_mode: Literal["typed", "shared", "scalar"] = "typed"
    seed: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class PaddedGroupIndex:
    indices: torch.Tensor
    mask: torch.Tensor


class StateProjector:
    def __init__(self, *, pca_dim: int, seed: int) -> None:
        self.pca_dim = int(pca_dim)
        self.seed = int(seed)

    def fit(self, keys: list[gm.StateLayerKey], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> "StateProjector":
        self.keys_ = [key for key in keys if key in state_vectors]
        x = np.vstack([state_vectors[key] for key in self.keys_]).astype(np.float64)
        self.scaler_ = StandardScaler()
        xs = self.scaler_.fit_transform(x)
        n_components = max(1, min(self.pca_dim, xs.shape[0] - 1, xs.shape[1]))
        self.pca_ = PCA(n_components=n_components, whiten=True, random_state=self.seed)
        self.pca_.fit(xs)
        return self

    def transform(self, keys: list[gm.StateLayerKey], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> np.ndarray:
        x = np.vstack([state_vectors[key] for key in keys]).astype(np.float64)
        out = self.pca_.transform(self.scaler_.transform(x))
        postmap = getattr(self, "z_postmap_", None)  # covariant transfer hook (default off)
        if postmap is not None:
            out = postmap(out)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class TabularEncoder:
    def fit(self, rows: list[dict], *, kind: Literal["action", "route"], include_response: bool) -> "TabularEncoder":
        if kind == "action":
            feats = [gm.action_features(row, include_response=include_response) for row in rows]
        else:
            feats = [gm.route_features(row) for row in rows]
        self.kind_ = kind
        self.include_response_ = include_response
        self.vectorizer_ = DictVectorizer(sparse=False)
        raw = self.vectorizer_.fit_transform(feats)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(raw)
        return self

    def transform(self, rows: list[dict]) -> np.ndarray:
        if self.kind_ == "action":
            feats = [gm.action_features(row, include_response=self.include_response_) for row in rows]
        else:
            feats = [gm.route_features(row) for row in rows]
        raw = self.vectorizer_.transform(feats)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler_.transform(raw).astype(np.float32)


class ChartActionFieldNet(nn.Module):
    def __init__(self, chart_dim: int, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.chart_net = nn.Sequential(nn.Linear(chart_dim, hidden_dim), nn.ReLU())
        self.state_net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.action_net = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.ReLU())
        self.body = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fix_head = nn.Linear(hidden_dim, 1)
        self.harm_head = nn.Linear(hidden_dim, 1)

    def forward(self, chart_x: torch.Tensor, state_x: torch.Tensor, action_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ch = self.chart_net(chart_x)
        st = self.state_net(state_x)
        ac = self.action_net(action_x)
        h = self.body(torch.cat([ch, st, ac, ch * ac], dim=-1))
        return self.fix_head(h).squeeze(-1), self.harm_head(h).squeeze(-1)


class NeuralAtlasNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.action_net = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.ReLU())
        self.action_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.body = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fix_head = nn.Linear(hidden_dim, 1)
        self.harm_head = nn.Linear(hidden_dim, 1)

    def forward(self, state_x: torch.Tensor, action_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(state_x)
        recon = self.decoder(z)
        ac = self.action_net(action_x)
        ac_latent = self.action_to_latent(ac)
        h = self.body(torch.cat([z, ac, z * ac_latent], dim=-1))
        return self.fix_head(h).squeeze(-1), self.harm_head(h).squeeze(-1), recon

    def encode(self, state_x: torch.Tensor) -> torch.Tensor:
        return self.encoder(state_x)


class TypedTransportProductGraphNet(nn.Module):
    """Typed product-graph net with relation-specific linear transports.

    This is a typed transport prototype, not a full sheaf/gauge model: each
    edge type carries its own matrix-valued transport, and messages are
    normalized per destination before node updates.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        route_dim: int,
        chart_dim: int,
        action_dim: int,
        hidden_dim: int,
        graph_layers: int,
        transport_mode: Literal["typed", "shared", "scalar"] = "typed",
    ) -> None:
        super().__init__()
        self.transport_mode = transport_mode
        self.state_in = nn.Linear(state_dim, hidden_dim)
        self.route_in = nn.Linear(route_dim, hidden_dim)
        self.chart_in = nn.Linear(chart_dim, hidden_dim)
        self.action_in = nn.Linear(action_dim, hidden_dim)
        edge_types = ["state_action", "route_action", "chart_action", "z2_partner"]
        self.transports = nn.ModuleDict({name: nn.Linear(hidden_dim, hidden_dim, bias=False) for name in edge_types})
        self.shared_transport = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.transport_gates = nn.ParameterDict({name: nn.Parameter(torch.zeros(())) for name in edge_types})
        self.updates = nn.ModuleList([nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(graph_layers)])
        self.fix_head = nn.Linear(hidden_dim, 1)
        self.harm_head = nn.Linear(hidden_dim, 1)

    def initial(
        self,
        state_x: torch.Tensor,
        route_x: torch.Tensor,
        chart_x: torch.Tensor,
        action_x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([
            F.relu(self.state_in(state_x)),
            F.relu(self.route_in(route_x)),
            F.relu(self.chart_in(chart_x)),
            F.relu(self.action_in(action_x)),
        ], dim=0)

    def forward(
        self,
        state_x: torch.Tensor,
        route_x: torch.Tensor,
        chart_x: torch.Tensor,
        action_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: list[str],
        action_nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.initial(state_x, route_x, chart_x, action_x)
        if edge_index.numel() == 0:
            ah = h[action_nodes]
            return self.fix_head(ah).squeeze(-1), self.harm_head(ah).squeeze(-1)
        src = edge_index[0]
        dst = edge_index[1]
        for update in self.updates:
            agg = torch.zeros_like(h)
            degree = torch.zeros((h.shape[0],), dtype=h.dtype, device=h.device)
            for edge_type in sorted(set(edge_types)):
                mask = torch.tensor([item == edge_type for item in edge_types], device=h.device, dtype=torch.bool)
                if not bool(mask.any()):
                    continue
                source_h = h[src[mask]]
                if self.transport_mode == "typed":
                    transported = self.transports[edge_type](source_h)
                elif self.transport_mode == "shared":
                    transported = self.shared_transport(source_h)
                elif self.transport_mode == "scalar":
                    transported = source_h
                else:
                    raise ValueError(f"unknown transport_mode {self.transport_mode!r}")
                msg = torch.sigmoid(self.transport_gates[edge_type]) * transported
                agg.index_add_(0, dst[mask], msg)
                degree.index_add_(0, dst[mask], torch.ones(mask.sum(), dtype=h.dtype, device=h.device))
            agg = agg / degree.clamp_min(1.0).unsqueeze(-1)
            h = F.relu(update(torch.cat([h, agg], dim=-1))) + h
        ah = h[action_nodes]
        return self.fix_head(ah).squeeze(-1), self.harm_head(ah).squeeze(-1)


class _BaseNeuralSelector:
    def __init__(self, config: StructuralNeuralConfig | None = None) -> None:
        self.config = config or StructuralNeuralConfig()
        self.device_ = resolve_device(self.config.device)

    def _train_labels(self, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        fix = np.asarray([gm.fix_value(row) for row in rows], dtype=np.float32)
        harm = np.asarray([gm.harm_value(row) for row in rows], dtype=np.float32)
        reward = np.asarray([gm.target_value(row, self.config.chart.objective) for row in rows], dtype=np.float32)
        fix_mask = np.asarray([str(row.get("status_class", "")).startswith("false_") for row in rows], dtype=bool)
        harm_mask = np.asarray([str(row.get("status_class", "")).startswith("honest_") for row in rows], dtype=bool)
        group_names = [gm.state_id(row) for row in rows]
        group_index = {name: idx for idx, name in enumerate(sorted(set(group_names)))}
        groups = np.asarray([group_index[name] for name in group_names], dtype=np.int64)
        return fix, harm, reward, fix_mask, harm_mask, groups

    @staticmethod
    def _chart_prior_scores(chart: ChartAtlas, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> np.ndarray:
        priors = []
        for row in rows:
            score, _meta = chart.score_one(row, state_vectors)
            if not np.isfinite(score) or score < -1e8:
                score = 0.0
            priors.append(float(score))
        return np.asarray(priors, dtype=np.float32)

    def _maybe_zero_prior(self, prior: np.ndarray) -> np.ndarray:
        if self.config.use_chart:
            return prior
        return np.zeros_like(prior, dtype=np.float32)

    def _maybe_zero_features(
        self,
        *,
        chart_x: np.ndarray | None = None,
        state_x: np.ndarray | None = None,
        action_x: np.ndarray | None = None,
        route_x: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if chart_x is not None and not self.config.use_chart:
            chart_x = np.zeros_like(chart_x, dtype=np.float32)
        if state_x is not None and not self.config.use_state:
            state_x = np.zeros_like(state_x, dtype=np.float32)
        if action_x is not None and not self.config.use_action:
            action_x = np.zeros_like(action_x, dtype=np.float32)
        if route_x is not None and not self.config.use_route:
            route_x = np.zeros_like(route_x, dtype=np.float32)
        return chart_x, state_x, action_x, route_x

    def _selection_scores(
        self,
        fix_logits: torch.Tensor,
        harm_logits: torch.Tensor,
        score_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        score = torch.sigmoid(fix_logits) - self.config.harm_penalty * torch.sigmoid(harm_logits)
        if score_prior is not None and self.config.chart_prior_weight != 0.0:
            score = score + float(self.config.chart_prior_weight) * score_prior
        return score

    def _head_loss(
        self,
        fix_logits: torch.Tensor,
        harm_logits: torch.Tensor,
        fix_y: torch.Tensor,
        harm_y: torch.Tensor,
        fix_mask: torch.Tensor,
        harm_mask: torch.Tensor,
    ) -> torch.Tensor:
        zero = fix_logits.sum() * 0.0
        fix_loss = F.binary_cross_entropy_with_logits(fix_logits[fix_mask], fix_y[fix_mask]) if bool(fix_mask.any()) else zero
        harm_loss = F.binary_cross_entropy_with_logits(harm_logits[harm_mask], harm_y[harm_mask]) if bool(harm_mask.any()) else zero
        return fix_loss + self.config.harm_penalty * harm_loss

    def _listwise_loss(
        self,
        scores: torch.Tensor,
        reward_y: torch.Tensor,
        group_ids: torch.Tensor | list[torch.Tensor] | PaddedGroupIndex,
    ) -> torch.Tensor:
        if isinstance(group_ids, PaddedGroupIndex):
            if group_ids.indices.numel() == 0:
                return scores.sum() * 0.0
            temperature = max(float(self.config.listwise_temperature), 1e-4)
            grouped_scores = scores[group_ids.indices] / temperature
            grouped_rewards = reward_y[group_ids.indices] / temperature
            fill = torch.full_like(grouped_scores, -1e9)
            grouped_scores = torch.where(group_ids.mask, grouped_scores, fill)
            grouped_rewards = torch.where(group_ids.mask, grouped_rewards, fill)
            target = torch.softmax(grouped_rewards, dim=1)
            pred = torch.log_softmax(grouped_scores, dim=1)
            return -torch.where(group_ids.mask, target * pred, torch.zeros_like(pred)).sum(dim=1).mean()
        losses = []
        temperature = max(float(self.config.listwise_temperature), 1e-4)
        groups = group_ids if isinstance(group_ids, list) else [torch.where(group_ids == group)[0] for group in torch.unique(group_ids)]
        for idx in groups:
            if int(idx.numel()) < 2:
                continue
            rewards = reward_y[idx]
            if float((rewards.max() - rewards.min()).detach().cpu()) <= 1e-8:
                continue
            target = torch.softmax(rewards / temperature, dim=0)
            pred = torch.log_softmax(scores[idx] / temperature, dim=0)
            losses.append(-(target * pred).sum())
        if not losses:
            return scores.sum() * 0.0
        return torch.stack(losses).mean()

    def _padded_group_index(self, groups: np.ndarray, reward: np.ndarray) -> PaddedGroupIndex:
        group_indices = []
        for group in np.unique(groups):
            idx = np.flatnonzero(groups == group).astype(np.int64)
            if idx.size < 2 or float(np.ptp(reward[idx])) <= 1e-8:
                continue
            group_indices.append(idx)
        if not group_indices:
            return PaddedGroupIndex(
                indices=torch.zeros((0, 1), device=self.device_, dtype=torch.long),
                mask=torch.zeros((0, 1), device=self.device_, dtype=torch.bool),
            )
        width = max(int(idx.size) for idx in group_indices)
        padded = np.zeros((len(group_indices), width), dtype=np.int64)
        mask = np.zeros((len(group_indices), width), dtype=bool)
        for row_idx, idx in enumerate(group_indices):
            padded[row_idx, : idx.size] = idx
            mask[row_idx, : idx.size] = True
        return PaddedGroupIndex(
            indices=torch.tensor(padded, device=self.device_, dtype=torch.long),
            mask=torch.tensor(mask, device=self.device_, dtype=torch.bool),
        )

    def _combined_loss(
        self,
        fix_logits: torch.Tensor,
        harm_logits: torch.Tensor,
        fix_y: torch.Tensor,
        harm_y: torch.Tensor,
        reward_y: torch.Tensor,
        fix_mask: torch.Tensor,
        harm_mask: torch.Tensor,
        group_ids: torch.Tensor | list[torch.Tensor] | PaddedGroupIndex,
        score_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self._selection_scores(fix_logits, harm_logits, score_prior=score_prior)
        loss = self._head_loss(fix_logits, harm_logits, fix_y, harm_y, fix_mask, harm_mask) + self.config.listwise_weight * self._listwise_loss(scores, reward_y, group_ids)
        if score_prior is not None and self.config.chart_distill_weight > 0.0:
            loss = loss + float(self.config.chart_distill_weight) * F.mse_loss(scores, score_prior)
        return loss

    def _score_from_logits(
        self,
        fix_logits: torch.Tensor,
        harm_logits: torch.Tensor,
        score_prior: torch.Tensor | None = None,
    ) -> np.ndarray:
        return self._selection_scores(fix_logits, harm_logits, score_prior=score_prior).detach().cpu().numpy().astype(np.float64)


class EquivariantChartActionFieldSelector(_BaseNeuralSelector):
    """Learns a chart-local action field with shared action/target parameters."""

    def fit(self, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> "EquivariantChartActionFieldSelector":
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        train = [row for row in gm.candidate_rows(rows) if gm.state_layer_key(row) in state_vectors]
        self.train_rows_ = train
        self.chart_ = ChartAtlas(self.config.chart).fit(train, state_vectors)
        self.action_encoder_ = TabularEncoder().fit(train, kind="action", include_response=self.config.include_response)
        chart_x, state_x, action_x, prior, valid = self._features(train, state_vectors)
        if not bool(valid.all()):
            keep = valid.astype(bool)
            train = [row for row, ok in zip(train, keep) if ok]
            chart_x, state_x, action_x, prior = chart_x[keep], state_x[keep], action_x[keep], prior[keep]
        fix, harm, reward, fix_mask, harm_mask, groups = self._train_labels(train)
        self.model_ = ChartActionFieldNet(chart_x.shape[1], state_x.shape[1], action_x.shape[1], self.config.hidden_dim).to(self.device_)
        self._fit_arrays(chart_x, state_x, action_x, prior, fix, harm, reward, fix_mask, harm_mask, groups)
        return self

    def _features(self, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        chart_rows = []
        state_rows = []
        valid = []
        for row in rows:
            key = gm.state_layer_key(row)
            ok = key in state_vectors if key is not None else False
            valid.append(ok)
            if not ok:
                chart_rows.append(np.zeros(self.chart_.chart_count_, dtype=np.float32))
                state_rows.append(np.zeros(self.chart_.pca_.n_components_, dtype=np.float32))
                continue
            z = self.chart_.state_to_z(state_vectors[key])
            chart_rows.append(self.chart_.membership_from_z(z)[0].astype(np.float32))
            state_rows.append(z.astype(np.float32))
        action_x = self.action_encoder_.transform(rows)
        chart_x = np.vstack(chart_rows).astype(np.float32)
        state_x = np.vstack(state_rows).astype(np.float32)
        chart_x, state_x, action_x, _route_x = self._maybe_zero_features(
            chart_x=chart_x,
            state_x=state_x,
            action_x=action_x,
        )
        prior = self._maybe_zero_prior(self._chart_prior_scores(self.chart_, rows, state_vectors))
        return (
            chart_x,
            state_x,
            action_x,
            prior,
            np.asarray(valid, dtype=bool),
        )

    def _fit_arrays(
        self,
        chart_x: np.ndarray,
        state_x: np.ndarray,
        action_x: np.ndarray,
        prior: np.ndarray,
        fix: np.ndarray,
        harm: np.ndarray,
        reward: np.ndarray,
        fix_mask: np.ndarray,
        harm_mask: np.ndarray,
        groups: np.ndarray,
    ) -> None:
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        cx = torch.tensor(chart_x, device=self.device_)
        sx = torch.tensor(state_x, device=self.device_)
        ax = torch.tensor(action_x, device=self.device_)
        sp = torch.tensor(prior, device=self.device_)
        fy = torch.tensor(fix, device=self.device_)
        hy = torch.tensor(harm, device=self.device_)
        ry = torch.tensor(reward, device=self.device_)
        fm = torch.tensor(fix_mask, device=self.device_, dtype=torch.bool)
        hm = torch.tensor(harm_mask, device=self.device_, dtype=torch.bool)
        gi = self._padded_group_index(groups, reward)
        for _ in range(self.config.epochs):
            opt.zero_grad()
            fix_logits, harm_logits = self.model_(cx, sx, ax)
            loss = self._combined_loss(fix_logits, harm_logits, fy, hy, ry, fm, hm, gi, score_prior=sp)
            loss.backward()
            opt.step()
        self.model_.eval()

    def score(self, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> np.ndarray:
        if not rows:
            return np.asarray([], dtype=np.float64)
        chart_x, state_x, action_x, prior, valid = self._features(rows, state_vectors)
        with torch.no_grad():
            fix_logits, harm_logits = self.model_(
                torch.tensor(chart_x, device=self.device_),
                torch.tensor(state_x, device=self.device_),
                torch.tensor(action_x, device=self.device_),
            )
        scores = self._score_from_logits(fix_logits, harm_logits, score_prior=torch.tensor(prior, device=self.device_))
        scores[~valid] = -1e9
        return scores

    def explain_choice(self, row: dict, state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> dict:
        return self.chart_.features_one(row, state_vectors)


class NeuralAtlasControlFieldSelector(_BaseNeuralSelector):
    """Learns latent atlas coordinates before learning the action-control field."""

    def fit(self, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> "NeuralAtlasControlFieldSelector":
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        train = [row for row in gm.candidate_rows(rows) if gm.state_layer_key(row) in state_vectors]
        self.train_rows_ = train
        self.chart_prior_ = ChartAtlas(self.config.chart).fit(train, state_vectors)
        keys = sorted({gm.state_layer_key(row) for row in train if gm.state_layer_key(row) is not None})
        self.state_projector_ = StateProjector(pca_dim=self.config.pre_pca_dim, seed=self.config.seed).fit(keys, state_vectors)
        self.action_encoder_ = TabularEncoder().fit(train, kind="action", include_response=self.config.include_response)
        state_x, action_x, prior, valid = self._features(train, state_vectors)
        train = [row for row, ok in zip(train, valid) if ok]
        state_x, action_x, prior = state_x[valid], action_x[valid], prior[valid]
        fix, harm, reward, fix_mask, harm_mask, groups = self._train_labels(train)
        self.model_ = NeuralAtlasNet(state_x.shape[1], action_x.shape[1], self.config.hidden_dim, self.config.latent_dim).to(self.device_)
        self._fit_arrays(state_x, action_x, prior, fix, harm, reward, fix_mask, harm_mask, groups)
        return self

    def _features(self, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        keys = [gm.state_layer_key(row) for row in rows]
        valid = np.asarray([key in state_vectors if key is not None else False for key in keys], dtype=bool)
        state_x = np.zeros((len(rows), self.state_projector_.pca_.n_components_), dtype=np.float32)
        valid_keys = [key for key, ok in zip(keys, valid) if ok]
        if valid_keys:
            state_x[valid] = self.state_projector_.transform(valid_keys, state_vectors)
        action_x = self.action_encoder_.transform(rows)
        _chart_x, state_x, action_x, _route_x = self._maybe_zero_features(
            state_x=state_x,
            action_x=action_x,
        )
        prior = self._maybe_zero_prior(self._chart_prior_scores(self.chart_prior_, rows, state_vectors))
        return state_x, action_x, prior, valid

    def _fit_arrays(
        self,
        state_x: np.ndarray,
        action_x: np.ndarray,
        prior: np.ndarray,
        fix: np.ndarray,
        harm: np.ndarray,
        reward: np.ndarray,
        fix_mask: np.ndarray,
        harm_mask: np.ndarray,
        groups: np.ndarray,
    ) -> None:
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        sx = torch.tensor(state_x, device=self.device_)
        ax = torch.tensor(action_x, device=self.device_)
        sp = torch.tensor(prior, device=self.device_)
        fy = torch.tensor(fix, device=self.device_)
        hy = torch.tensor(harm, device=self.device_)
        ry = torch.tensor(reward, device=self.device_)
        fm = torch.tensor(fix_mask, device=self.device_, dtype=torch.bool)
        hm = torch.tensor(harm_mask, device=self.device_, dtype=torch.bool)
        gi = self._padded_group_index(groups, reward)
        for _ in range(self.config.epochs):
            opt.zero_grad()
            fix_logits, harm_logits, recon = self.model_(sx, ax)
            loss = self._combined_loss(fix_logits, harm_logits, fy, hy, ry, fm, hm, gi, score_prior=sp) + self.config.ae_weight * F.mse_loss(recon, sx)
            loss.backward()
            opt.step()
        self.model_.eval()

    def score(self, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> np.ndarray:
        if not rows:
            return np.asarray([], dtype=np.float64)
        state_x, action_x, prior, valid = self._features(rows, state_vectors)
        with torch.no_grad():
            fix_logits, harm_logits, _ = self.model_(
                torch.tensor(state_x, device=self.device_),
                torch.tensor(action_x, device=self.device_),
            )
        scores = self._score_from_logits(fix_logits, harm_logits, score_prior=torch.tensor(prior, device=self.device_))
        scores[~valid] = -1e9
        return scores

    def explain_choice(self, row: dict, state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> dict:
        key = gm.state_layer_key(row)
        if key not in state_vectors:
            return {"neural_atlas_reason": "missing_state"}
        state_x = self.state_projector_.transform([key], state_vectors)
        with torch.no_grad():
            z = self.model_.encode(torch.tensor(state_x, device=self.device_)).detach().cpu().numpy()[0]
        return {"neural_atlas_latent_norm": float(np.linalg.norm(z)), "neural_atlas_dim": int(z.shape[0])}


class TypedTransportProductGraphController(_BaseNeuralSelector):
    """Typed transport product-graph controller over state/chart/action fibers."""

    def fit(self, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> "TypedTransportProductGraphController":
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        train = [row for row in gm.candidate_rows(rows) if gm.state_layer_key(row) in state_vectors]
        self.train_rows_ = train
        self.chart_ = ChartAtlas(self.config.chart).fit(train, state_vectors)
        keys = sorted({gm.state_layer_key(row) for row in train if gm.state_layer_key(row) is not None})
        self.state_projector_ = StateProjector(pca_dim=self.config.pre_pca_dim, seed=self.config.seed).fit(keys, state_vectors)
        self.action_encoder_ = TabularEncoder().fit(train, kind="action", include_response=self.config.include_response)
        self.route_encoder_ = TabularEncoder().fit(train, kind="route", include_response=False)
        sample_groups = [group for group in gm.grouped_by_state(train).values() if group]
        graph = self._graph_tensors(sample_groups[0], state_vectors)
        self.model_ = TypedTransportProductGraphNet(
            state_dim=graph["state_x"].shape[1],
            route_dim=graph["route_x"].shape[1],
            chart_dim=graph["chart_x"].shape[1],
            action_dim=graph["action_x"].shape[1],
            hidden_dim=self.config.hidden_dim,
            graph_layers=self.config.graph_layers,
            transport_mode=self.config.transport_mode,
        ).to(self.device_)
        self._fit_groups(sample_groups, state_vectors)
        return self

    def _state_chart_features(self, row: dict, state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> tuple[np.ndarray, np.ndarray, bool]:
        key = gm.state_layer_key(row)
        if key not in state_vectors:
            return (
                np.zeros(self.state_projector_.pca_.n_components_, dtype=np.float32),
                np.zeros(self.chart_.chart_count_ + self.chart_.pca_.n_components_, dtype=np.float32),
                False,
            )
        state_x = self.state_projector_.transform([key], state_vectors)[0]
        z = self.chart_.state_to_z(state_vectors[key]).astype(np.float32)
        membership = self.chart_.membership_from_z(z)[0].astype(np.float32)
        return state_x, np.concatenate([membership, z]).astype(np.float32), True

    def _graph_tensors(self, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> dict:
        candidates = [row for row in rows if not gm.is_baseline(row)]
        if not candidates:
            candidates = rows
        state_x, chart_x, valid = self._state_chart_features(candidates[0], state_vectors)
        route_x = self.route_encoder_.transform([candidates[0]])[0]
        action_x = self.action_encoder_.transform(candidates)
        chart_x, state_x, action_x, route_x = self._maybe_zero_features(
            chart_x=chart_x,
            state_x=state_x,
            action_x=action_x,
            route_x=route_x,
        )
        graph = build_product_z2_graph(candidates, include_z2=self.config.use_z2)
        action_nodes = np.asarray([graph.action_node_for_row[idx] for idx in range(len(candidates))], dtype=np.int64)
        fix, harm, reward, fix_mask, harm_mask, groups = self._train_labels(candidates)
        return {
            "rows": candidates,
            "valid": valid,
            "state_x": state_x[None, :].astype(np.float32),
            "route_x": route_x[None, :].astype(np.float32),
            "chart_x": chart_x[None, :].astype(np.float32),
            "action_x": action_x.astype(np.float32),
            "edge_index": graph.edge_index.astype(np.int64),
            "edge_types": graph.edge_types,
            "action_nodes": action_nodes,
            "fix": fix,
            "harm": harm,
            "reward": reward,
            "fix_mask": fix_mask,
            "harm_mask": harm_mask,
            "groups": groups,
            "score_prior": self._maybe_zero_prior(self._chart_prior_scores(self.chart_, candidates, state_vectors)),
        }

    def _fit_groups(self, groups: list[list[dict]], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> None:
        graphs = [self._graph_tensors(group, state_vectors) for group in groups]
        graphs = [graph for graph in graphs if graph["valid"] and len(graph["rows"]) > 0]
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        rng = np.random.default_rng(self.config.seed)
        for _ in range(self.config.epochs):
            order = rng.permutation(len(graphs))
            for idx in order:
                graph = graphs[int(idx)]
                opt.zero_grad()
                fix_logits, harm_logits = self._forward_graph(graph)
                fy = torch.tensor(graph["fix"], device=self.device_)
                hy = torch.tensor(graph["harm"], device=self.device_)
                ry = torch.tensor(graph["reward"], device=self.device_)
                fm = torch.tensor(graph["fix_mask"], device=self.device_, dtype=torch.bool)
                hm = torch.tensor(graph["harm_mask"], device=self.device_, dtype=torch.bool)
                gi = torch.tensor(graph["groups"], device=self.device_, dtype=torch.long)
                sp = torch.tensor(graph["score_prior"], device=self.device_)
                loss = self._combined_loss(fix_logits, harm_logits, fy, hy, ry, fm, hm, gi, score_prior=sp)
                loss.backward()
                opt.step()
        self.model_.eval()

    def _forward_graph(self, graph: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model_(
            torch.tensor(graph["state_x"], device=self.device_),
            torch.tensor(graph["route_x"], device=self.device_),
            torch.tensor(graph["chart_x"], device=self.device_),
            torch.tensor(graph["action_x"], device=self.device_),
            torch.tensor(graph["edge_index"], device=self.device_, dtype=torch.long),
            graph["edge_types"],
            torch.tensor(graph["action_nodes"], device=self.device_, dtype=torch.long),
        )

    def score(self, rows: list[dict], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> np.ndarray:
        candidates = [row for row in rows if not gm.is_baseline(row)]
        if not candidates:
            return np.full(len(rows), -1e9, dtype=np.float64)
        graph = self._graph_tensors(candidates, state_vectors)
        if not graph["valid"]:
            return np.full(len(rows), -1e9, dtype=np.float64)
        with torch.no_grad():
            fix_logits, harm_logits = self._forward_graph(graph)
        candidate_scores = self._score_from_logits(
            fix_logits,
            harm_logits,
            score_prior=torch.tensor(graph["score_prior"], device=self.device_),
        )
        scores = np.full(len(rows), -1e9, dtype=np.float64)
        cursor = 0
        for idx, row in enumerate(rows):
            if gm.is_baseline(row):
                continue
            scores[idx] = candidate_scores[cursor]
            cursor += 1
        return scores

    def explain_choice(self, row: dict, state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> dict:
        return self.chart_.features_one(row, state_vectors)


HeteroProductGraphController = TypedTransportProductGraphController
SheafGaugeProductGraphController = TypedTransportProductGraphController
