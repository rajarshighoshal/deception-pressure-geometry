"""Local control-flow estimator over activation states and steering vectors.

This is the option-3 object: train rows define a local vector field of useful
control directions, and held-out states query that field before any measured
held-out response is consulted.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from geoprobe.control import geometry_map as gm
from geoprobe.control.local_deformation_field import SteeringSpecLookup, _norm, _unit


@dataclass(frozen=True)
class LocalControlFlowConfig:
    top_k: int = 12
    neighbor_pool_multiplier: int = 4
    geometry_dim: int = 64
    objective: str = "reward"
    target_route_only: bool = True
    use_local_neighbors: bool = True
    seed: int = 0


@dataclass
class FlowPrediction:
    vector: np.ndarray
    confidence: float
    support_count: int
    neighbor_state_ids: list[str]
    neighbor_distances: list[float]
    reason: str = "ok"


@dataclass(frozen=True)
class BoundaryDoseConfig:
    """Linearized boundary-crossing dose from a train-fold detection surface."""

    target_margin: float = 0.0
    max_alpha: float = 128.0
    min_directional_slope: float = 1e-6
    ridge_alpha: float = 10.0


@dataclass(frozen=True)
class BoundaryDosePrediction:
    alpha: float | None
    raw_alpha: float | None
    current_margin: float
    predicted_margin: float
    target_status: str
    target_sign: float
    directional_slope: float
    signed_current_margin: float
    required_margin_gain: float
    reason: str


@dataclass(frozen=True)
class CausalDoseConfig:
    """Train-fold causal dose from measured fixed-alpha intervention outcomes."""

    target_margin: float = 0.0
    max_alpha: float = 128.0
    min_gain: float = 1e-4
    ridge_alpha: float = 3.0
    fixed_alphas: tuple[float, ...] = (48.0, 72.0, 96.0)


@dataclass(frozen=True)
class CausalDosePrediction:
    alpha: float | None
    raw_alpha: float | None
    current_margin: float
    target_status: str
    target_sign: float
    signed_current_margin: float
    required_margin_gain: float
    predicted_gain: float | None
    reason: str
    n_train: int


@dataclass
class _FlowSummary:
    count: int
    abs_weight_sum: float
    reward_sum: float
    vector_sum: np.ndarray

    @classmethod
    def empty(cls, dim: int) -> "_FlowSummary":
        return cls(0, 0.0, 0.0, np.zeros(dim, dtype=np.float32))

    def add(self, row: dict, direction: np.ndarray, objective: str) -> None:
        weight = gm.target_value(row, objective)
        alpha = abs(float(row.get("alpha") or 0.0))
        action_vector = alpha * _unit(direction)
        self.count += 1
        self.abs_weight_sum += abs(float(weight))
        self.reward_sum += float(weight)
        self.vector_sum += float(weight) * action_vector

    def vector(self) -> np.ndarray:
        if self.abs_weight_sum <= 1e-12:
            return np.zeros_like(self.vector_sum, dtype=np.float32)
        return (self.vector_sum / self.abs_weight_sum).astype(np.float32)


class _StateProjector:
    def __init__(self, geometry_dim: int, seed: int) -> None:
        self.geometry_dim = int(geometry_dim)
        self.seed = int(seed)

    def fit(self, keys: list[gm.StateLayerKey], state_vectors: dict[gm.StateLayerKey, np.ndarray]) -> "_StateProjector":
        self.keys_ = [key for key in keys if key in state_vectors]
        if not self.keys_:
            raise ValueError("no state vectors for local control-flow projector")
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


class LocalControlFlowEstimator:
    """Train-fold local reward-vector field over real steering directions."""

    def __init__(self, spec_lookup: SteeringSpecLookup, config: LocalControlFlowConfig | None = None) -> None:
        self.spec_lookup = spec_lookup
        self.config = config or LocalControlFlowConfig()

    def fit(
        self,
        rows: list[dict],
        state_vectors: dict[gm.StateLayerKey, np.ndarray],
    ) -> "LocalControlFlowEstimator":
        train = [
            row for row in gm.candidate_rows(rows)
            if gm.state_layer_key(row) in state_vectors and self.spec_lookup.vector_for(row) is not None
        ]
        if not train:
            raise ValueError("no train rows with state vectors and steering vectors")
        keys = sorted({gm.state_layer_key(row) for row in train if gm.state_layer_key(row) is not None})
        self.projector_ = _StateProjector(self.config.geometry_dim, self.config.seed).fit(keys, state_vectors)
        self.state_x_by_key_ = {key: self.projector_.transform_state(state_vectors[key]) for key in keys}
        self.raw_dim_ = len(next(iter(state_vectors.values())))
        self.summary_by_state_group_: dict[tuple[str, int, str, str], _FlowSummary] = {}
        self.global_summary_by_group_: dict[tuple[int, str, str], _FlowSummary] = {}
        for row in train:
            key = gm.state_layer_key(row)
            raw_d = self.spec_lookup.vector_for(row)
            if key is None or raw_d is None:
                continue
            state_id, layer = key
            route = str(row.get("route_action") or "unknown")
            target = str(row.get("target_status") or "NONE")
            for store_key, store in [
                ((state_id, layer, route, target), self.summary_by_state_group_),
                ((layer, route, target), self.global_summary_by_group_),
            ]:
                if store_key not in store:
                    store[store_key] = _FlowSummary.empty(self.raw_dim_)
                store[store_key].add(row, np.asarray(raw_d, dtype=np.float32), self.config.objective)
        self._fit_neighbor_models()
        return self

    def _fit_neighbor_models(self) -> None:
        by_group: dict[tuple[int, str, str], list[tuple[str, np.ndarray]]] = defaultdict(list)
        for (state_id, layer, route, target), summary in self.summary_by_state_group_.items():
            if summary.count <= 0:
                continue
            key = (state_id, layer)
            if key in self.state_x_by_key_:
                by_group[(layer, route, target)].append((state_id, self.state_x_by_key_[key]))
        self.group_state_ids_: dict[tuple[int, str, str], list[str]] = {}
        self.group_state_x_: dict[tuple[int, str, str], np.ndarray] = {}
        self.group_models_: dict[tuple[int, str, str], NearestNeighbors] = {}
        for group, items in by_group.items():
            state_ids = [state_id for state_id, _x in items]
            x = np.vstack([x for _state_id, x in items]).astype(np.float32)
            self.group_state_ids_[group] = state_ids
            self.group_state_x_[group] = x
            model = NearestNeighbors(metric="euclidean", algorithm="auto")
            model.fit(x)
            self.group_models_[group] = model

    def predict(
        self,
        row: dict,
        state_vectors: dict[gm.StateLayerKey, np.ndarray],
        *,
        exclude_same_state: bool,
    ) -> FlowPrediction:
        key = gm.state_layer_key(row)
        if key is None or key not in state_vectors:
            return FlowPrediction(np.zeros(self.raw_dim_, dtype=np.float32), 0.0, 0, [], [])
        state_id, layer = key
        route = str(row.get("route_action") or "unknown")
        target = str(row.get("target_status") or "NONE")
        group = (int(layer), route, target)
        if not self.config.use_local_neighbors:
            summary = self.global_summary_by_group_.get(group)
            vector = summary.vector() if summary is not None else np.zeros(self.raw_dim_, dtype=np.float32)
            return FlowPrediction(vector, _norm(vector), 0 if summary is None else summary.count, [], [])
        model = self.group_models_.get(group)
        state_ids = self.group_state_ids_.get(group, [])
        if model is None or not state_ids:
            summary = self.global_summary_by_group_.get(group)
            vector = summary.vector() if summary is not None else np.zeros(self.raw_dim_, dtype=np.float32)
            return FlowPrediction(vector, _norm(vector), 0 if summary is None else summary.count, [], [])
        x = self.projector_.transform_state(state_vectors[key])
        width = max(1, int(self.config.top_k))
        n_pool = min(len(state_ids), max(width * int(self.config.neighbor_pool_multiplier), width + 1))
        distances, indices = model.kneighbors(x[None, :], n_neighbors=n_pool)
        # covariant transfer hook (default off): refuse neighbors farther than the source
        # self-distance threshold — off-support queries must abstain, not extrapolate.
        distance_threshold = getattr(self, "neighbor_distance_threshold_", None)
        coverage_filtered = 0
        kept_ids: list[str] = []
        kept_distances: list[float] = []
        vectors: list[np.ndarray] = []
        for local_idx, distance in zip(indices[0], distances[0], strict=True):
            neighbor_state = state_ids[int(local_idx)]
            if exclude_same_state and neighbor_state == state_id:
                continue
            if distance_threshold is not None and float(distance) > float(distance_threshold):
                coverage_filtered += 1
                continue
            summary = self.summary_by_state_group_.get((neighbor_state, int(layer), route, target))
            if summary is None or summary.count <= 0:
                continue
            kept_ids.append(neighbor_state)
            kept_distances.append(float(distance))
            vectors.append(summary.vector())
            if len(kept_ids) >= width:
                break
        if not vectors:
            if coverage_filtered > 0:
                # every candidate neighbor was beyond source support: abstain (no steering),
                # do NOT fall back to the global summary vector.
                return FlowPrediction(np.zeros(self.raw_dim_, dtype=np.float32), 0.0, 0, [], [],
                                      reason="coverage_abstain")
            summary = self.global_summary_by_group_.get(group)
            vector = summary.vector() if summary is not None else np.zeros(self.raw_dim_, dtype=np.float32)
            return FlowPrediction(vector, _norm(vector), 0 if summary is None else summary.count, [], [])
        weight_arr = _distance_weights(np.asarray(kept_distances, dtype=np.float32))
        vector = np.sum([float(w) * v for w, v in zip(weight_arr, vectors, strict=True)], axis=0) / max(float(np.sum(weight_arr)), 1e-12)
        vector = np.asarray(vector, dtype=np.float32)
        return FlowPrediction(vector, _norm(vector), int(len(vectors)), kept_ids, kept_distances)

    def score(
        self,
        rows: list[dict],
        state_vectors: dict[gm.StateLayerKey, np.ndarray],
        *,
        exclude_same_state: bool,
    ) -> np.ndarray:
        scores = np.full(len(rows), -1e9, dtype=np.float64)
        for idx, row in enumerate(rows):
            if gm.is_baseline(row):
                continue
            if self.config.target_route_only:
                route_target = _target_from_route(str(row.get("route_action") or ""))
                if route_target is not None and str(row.get("target_status")) != route_target:
                    continue
            raw_d = self.spec_lookup.vector_for(row)
            if raw_d is None:
                continue
            pred = self.predict(row, state_vectors, exclude_same_state=exclude_same_state)
            if pred.confidence <= 1e-12:
                continue
            alpha = abs(float(row.get("alpha") or 0.0))
            candidate = alpha * _unit(np.asarray(raw_d, dtype=np.float32))
            # Equivalent to choosing the candidate with minimum squared distance
            # to the predicted deformation vector, dropping the constant ||pred||^2.
            scores[idx] = float(2.0 * np.dot(pred.vector, candidate) - np.dot(candidate, candidate))
        return scores

    def oracle_flow(self, rows: list[dict]) -> np.ndarray:
        summary = _FlowSummary.empty(self.raw_dim_)
        for row in gm.candidate_rows(rows):
            raw_d = self.spec_lookup.vector_for(row)
            if raw_d is not None:
                summary.add(row, np.asarray(raw_d, dtype=np.float32), self.config.objective)
        return summary.vector()


class LinearBoundaryDoseEstimator:
    """Estimate alpha from the detector boundary rather than from flow norm.

    The fitted surface is a train-fold approximation to the model's base
    PASS-minus-FAIL margin: ``s(x) = w dot x + b``. Given a target-aligned
    steering direction ``d``, alpha is the linearized distance needed for
    ``s(x + alpha d)`` to cross the target boundary.
    """

    def __init__(self, config: BoundaryDoseConfig | None = None) -> None:
        self.config = config or BoundaryDoseConfig()

    def fit(
        self,
        rows: list[dict],
        state_vectors: dict[gm.StateLayerKey, np.ndarray],
    ) -> "LinearBoundaryDoseEstimator":
        by_key: dict[gm.StateLayerKey, dict] = {}
        for row in rows:
            key = gm.state_layer_key(row)
            if key is None or key not in state_vectors or key in by_key:
                continue
            margin = _finite_float(row.get("base_margin"))
            if margin is not None:
                by_key[key] = row
        if not by_key:
            raise ValueError("no train states with base_margin for boundary-dose estimator")
        keys = sorted(by_key)
        raw_x = np.vstack([state_vectors[key] for key in keys]).astype(np.float64)
        y = np.asarray([_finite_float(by_key[key].get("base_margin")) for key in keys], dtype=np.float64)
        self.scaler_ = StandardScaler()
        z = self.scaler_.fit_transform(raw_x)
        self.model_ = Ridge(alpha=float(self.config.ridge_alpha), fit_intercept=True)
        self.model_.fit(z, y)
        scale = np.asarray(self.scaler_.scale_, dtype=np.float64)
        scale[scale == 0.0] = 1.0
        self.raw_coef_ = (np.asarray(self.model_.coef_, dtype=np.float64) / scale).astype(np.float64)
        self.raw_dim_ = int(raw_x.shape[1])
        self.n_train_ = int(len(keys))
        return self

    def predict_margin(self, state: np.ndarray) -> float:
        z = self.scaler_.transform(np.asarray(state, dtype=np.float64)[None, :])
        return float(self.model_.predict(z)[0])

    def alpha_for(
        self,
        row: dict,
        state_vectors: dict[gm.StateLayerKey, np.ndarray],
        direction: np.ndarray,
    ) -> BoundaryDosePrediction:
        key = gm.state_layer_key(row)
        target = str(row.get("target_status") or "")
        target_sign = _target_sign(target)
        if key is None or key not in state_vectors or target_sign == 0.0:
            return self._empty(row, target, target_sign, "missing_state_or_target")
        predicted_margin = self.predict_margin(state_vectors[key])
        current_margin = _finite_float(row.get("base_margin"))
        if current_margin is None:
            current_margin = predicted_margin
        unit_direction = _unit(np.asarray(direction, dtype=np.float32))
        directional_slope = float(target_sign * np.dot(self.raw_coef_, unit_direction.astype(np.float64)))
        signed_current = float(target_sign * current_margin)
        required = float(self.config.target_margin - signed_current)
        if required <= 0.0:
            return BoundaryDosePrediction(
                alpha=0.0,
                raw_alpha=0.0,
                current_margin=float(current_margin),
                predicted_margin=predicted_margin,
                target_status=target,
                target_sign=target_sign,
                directional_slope=directional_slope,
                signed_current_margin=signed_current,
                required_margin_gain=required,
                reason="already_beyond_boundary",
            )
        if directional_slope <= float(self.config.min_directional_slope):
            return BoundaryDosePrediction(
                alpha=None,
                raw_alpha=None,
                current_margin=float(current_margin),
                predicted_margin=predicted_margin,
                target_status=target,
                target_sign=target_sign,
                directional_slope=directional_slope,
                signed_current_margin=signed_current,
                required_margin_gain=required,
                reason="non_positive_boundary_slope",
            )
        raw_alpha = required / directional_slope
        alpha = min(max(float(raw_alpha), 0.0), float(self.config.max_alpha))
        reason = "boundary_crossing"
        if raw_alpha > float(self.config.max_alpha):
            reason = "clipped_to_max_alpha"
        return BoundaryDosePrediction(
            alpha=alpha,
            raw_alpha=float(raw_alpha),
            current_margin=float(current_margin),
            predicted_margin=predicted_margin,
            target_status=target,
            target_sign=target_sign,
            directional_slope=directional_slope,
            signed_current_margin=signed_current,
            required_margin_gain=required,
            reason=reason,
        )

    def _empty(self, row: dict, target: str, target_sign: float, reason: str) -> BoundaryDosePrediction:
        margin = _finite_float(row.get("base_margin"))
        return BoundaryDosePrediction(
            alpha=None,
            raw_alpha=None,
            current_margin=float("nan") if margin is None else float(margin),
            predicted_margin=float("nan"),
            target_status=target,
            target_sign=target_sign,
            directional_slope=float("nan"),
            signed_current_margin=float("nan"),
            required_margin_gain=float("nan"),
            reason=reason,
        )


class CausalGainDoseEstimator:
    """Learn a causal margin gain, then derive alpha from required/gain.

    Training labels come from train-family fixed-alpha responses. The target is
    not a memorized alpha table: each state/action group contributes an effective
    gain ``required_margin / first_successful_alpha``. At inference the learned
    gain is combined with the held-out row's current margin to produce a dose.
    """

    def __init__(self, config: CausalDoseConfig | None = None) -> None:
        self.config = config or CausalDoseConfig()

    def fit(self, rows: list[dict]) -> "CausalGainDoseEstimator":
        examples: list[tuple[np.ndarray, float]] = []
        for group in _group_causal_score_rows(rows).values():
            rep = _representative_causal_feature_row(group)
            if rep is None:
                continue
            target_sign = _target_sign(str(rep.get("target_status") or ""))
            base_margin = _finite_float(rep.get("base_margin"))
            if target_sign == 0.0 or base_margin is None:
                continue
            signed_current = float(target_sign * base_margin)
            required = float(self.config.target_margin - signed_current)
            if required <= 0.0:
                continue
            crossing_alpha = _first_successful_fixed_alpha(group, self.config.fixed_alphas, self.config.target_margin)
            if crossing_alpha is None:
                crossing_alpha = float(self.config.max_alpha)
            if crossing_alpha <= 0.0:
                continue
            gain = max(required / float(crossing_alpha), float(self.config.min_gain))
            examples.append((_causal_feature_vector(rep, self.config), float(np.log(gain))))
        if not examples:
            raise ValueError("no causal-dose training examples with positive required margin and fixed-alpha outcomes")
        x = np.vstack([feat for feat, _y in examples]).astype(np.float64)
        y = np.asarray([y for _feat, y in examples], dtype=np.float64)
        self.scaler_ = StandardScaler()
        z = self.scaler_.fit_transform(x)
        self.model_ = Ridge(alpha=float(self.config.ridge_alpha), fit_intercept=True)
        self.model_.fit(z, y)
        self.n_train_ = int(len(examples))
        self.feature_names_ = list(CAUSAL_DOSE_FEATURE_NAMES)
        return self

    def predict_gain(self, row: dict) -> float:
        feat = _causal_feature_vector(row, self.config).astype(np.float64)[None, :]
        z = self.scaler_.transform(feat)
        log_gain = float(self.model_.predict(z)[0])
        return max(float(np.exp(log_gain)), float(self.config.min_gain))

    def alpha_for(self, row: dict) -> CausalDosePrediction:
        target = str(row.get("target_status") or "")
        target_sign = _target_sign(target)
        current_margin = _finite_float(row.get("base_margin"))
        if target_sign == 0.0 or current_margin is None:
            return self._empty(row, target, target_sign, "missing_margin_or_target")
        signed_current = float(target_sign * current_margin)
        required = float(self.config.target_margin - signed_current)
        if required <= 0.0:
            return CausalDosePrediction(
                alpha=0.0,
                raw_alpha=0.0,
                current_margin=float(current_margin),
                target_status=target,
                target_sign=target_sign,
                signed_current_margin=signed_current,
                required_margin_gain=required,
                predicted_gain=None,
                reason="already_beyond_causal_target",
                n_train=int(getattr(self, "n_train_", 0)),
            )
        gain = self.predict_gain(row)
        raw_alpha = required / max(gain, float(self.config.min_gain))
        alpha = min(max(float(raw_alpha), 0.0), float(self.config.max_alpha))
        reason = "causal_gain_crossing"
        if raw_alpha > float(self.config.max_alpha):
            reason = "clipped_to_max_alpha"
        return CausalDosePrediction(
            alpha=alpha,
            raw_alpha=float(raw_alpha),
            current_margin=float(current_margin),
            target_status=target,
            target_sign=target_sign,
            signed_current_margin=signed_current,
            required_margin_gain=required,
            predicted_gain=float(gain),
            reason=reason,
            n_train=int(getattr(self, "n_train_", 0)),
        )

    def _empty(self, row: dict, target: str, target_sign: float, reason: str) -> CausalDosePrediction:
        margin = _finite_float(row.get("base_margin"))
        return CausalDosePrediction(
            alpha=None,
            raw_alpha=None,
            current_margin=float("nan") if margin is None else float(margin),
            target_status=target,
            target_sign=target_sign,
            signed_current_margin=float("nan"),
            required_margin_gain=float("nan"),
            predicted_gain=None,
            reason=reason,
            n_train=int(getattr(self, "n_train_", 0)),
        )


def flow_alignment(pred: np.ndarray, oracle: np.ndarray) -> dict[str, float]:
    return {
        "pred_norm": _norm(pred),
        "oracle_norm": _norm(oracle),
        "cosine": _cos(pred, oracle),
    }


def _target_from_route(route_action: str) -> str | None:
    if route_action == "steer_to_PASS":
        return "PASS"
    if route_action == "steer_to_FAIL":
        return "FAIL"
    return None


def _distance_weights(distances: np.ndarray) -> np.ndarray:
    if len(distances) == 0:
        return distances.astype(np.float32)
    scale = max(float(np.median(distances)), 1e-6)
    return np.exp(-distances.astype(np.float32) / scale)


def _cos(left: np.ndarray, right: np.ndarray) -> float:
    denom = _norm(left) * _norm(right)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)) / denom)


def _target_sign(target_status: str) -> float:
    if target_status == "PASS":
        return 1.0
    if target_status == "FAIL":
        return -1.0
    return 0.0


CAUSAL_DOSE_FEATURE_NAMES = (
    "target_is_pass",
    "signed_base_margin",
    "required_margin_gain",
    "abs_base_margin",
    "pc_predicted_flow_norm",
    "pc_n_train",
    "pc_neighbor_distance_mean",
    "pc_neighbor_distance_min",
    "pc_neighbor_distance_max",
    "pc_boundary_directional_slope",
    "pc_boundary_alpha",
    "pc_boundary_alpha_raw",
    "pc_boundary_positive_slope",
)


def causal_dose_context_row(row: dict, prediction: FlowPrediction, boundary: BoundaryDosePrediction) -> dict:
    """Build the context-only feature row shared by export and gain training."""
    out = dict(row)
    out.update({
        "pc_n_train": float(prediction.support_count),
        "pc_neighbor_distance_mean": float(np.mean(prediction.neighbor_distances)) if prediction.neighbor_distances else -1.0,
        "pc_neighbor_distance_min": float(np.min(prediction.neighbor_distances)) if prediction.neighbor_distances else -1.0,
        "pc_neighbor_distance_max": float(np.max(prediction.neighbor_distances)) if prediction.neighbor_distances else -1.0,
        "pc_predicted_flow_norm": float(prediction.confidence),
        "pc_boundary_alpha": boundary.alpha,
        "pc_boundary_alpha_raw": boundary.raw_alpha,
        "pc_boundary_directional_slope": boundary.directional_slope,
        "pc_boundary_reason": boundary.reason,
    })
    return out


def _group_causal_score_rows(rows: list[dict]) -> dict[tuple[str, int, str, str], list[dict]]:
    grouped: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if str(row.get("method")) == "abstain":
            continue
        cid = row.get("conversation_id")
        layer = row.get("layer")
        method = row.get("method")
        target = row.get("target_status")
        if cid is None or layer is None or method is None or target is None:
            continue
        grouped[(str(cid), int(layer), str(method), str(target))].append(row)
    return dict(grouped)


def _representative_causal_feature_row(group: list[dict]) -> dict | None:
    for row in group:
        alpha = _finite_float(row.get("alpha"))
        if alpha is not None and alpha > 0.0:
            return row
    return group[0] if group else None


def _first_successful_fixed_alpha(group: list[dict], fixed_alphas: tuple[float, ...], target_margin: float) -> float | None:
    by_alpha: dict[float, dict] = {}
    for row in group:
        alpha = _finite_float(row.get("alpha"))
        if alpha is None:
            continue
        for fixed in fixed_alphas:
            if abs(alpha - float(fixed)) <= 1e-7:
                by_alpha[float(fixed)] = row
                break
    for alpha in sorted(float(a) for a in fixed_alphas):
        row = by_alpha.get(alpha)
        if row is None:
            continue
        target_sign = _target_sign(str(row.get("target_status") or ""))
        final_margin = _finite_float(row.get("final_margin"))
        if target_sign != 0.0 and final_margin is not None and target_sign * final_margin >= float(target_margin):
            return alpha
    return None


def _feature_float(row: dict, key: str, default: float = 0.0) -> float:
    value = _finite_float(row.get(key))
    if value is not None:
        return float(value)
    context = row.get("context")
    if isinstance(context, dict):
        value = _finite_float(context.get(key))
        if value is not None:
            return float(value)
    return float(default)


def _causal_feature_vector(row: dict, config: CausalDoseConfig) -> np.ndarray:
    target_sign = _target_sign(str(row.get("target_status") or ""))
    base_margin = _feature_float(row, "base_margin", 0.0)
    signed_base = float(target_sign * base_margin)
    required = max(float(config.target_margin - signed_base), 0.0)
    boundary_slope = _feature_float(row, "pc_boundary_directional_slope", 0.0)
    values = [
        1.0 if str(row.get("target_status") or "") == "PASS" else 0.0,
        signed_base,
        required,
        abs(base_margin),
        _feature_float(row, "pc_predicted_flow_norm", 0.0),
        _feature_float(row, "pc_n_train", 0.0),
        _feature_float(row, "pc_neighbor_distance_mean", -1.0),
        _feature_float(row, "pc_neighbor_distance_min", -1.0),
        _feature_float(row, "pc_neighbor_distance_max", -1.0),
        boundary_slope,
        _feature_float(row, "pc_boundary_alpha", -1.0),
        _feature_float(row, "pc_boundary_alpha_raw", -1.0),
        1.0 if boundary_slope > 0.0 else 0.0,
    ]
    return np.asarray(values, dtype=np.float32)


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None
