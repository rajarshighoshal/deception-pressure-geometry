"""Gauge-covariant horizontal lifts from intrinsic motion to residual fibers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np


class RelationalHorizontalLiftError(ValueError):
    """A lift sample, fit, or query violates the local fiber contract."""


def _readonly(
    value: object,
    *,
    name: str,
    ndim: int,
    dtype: np.dtype[np.floating] = np.dtype(np.float64),
) -> np.ndarray:
    try:
        array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    except (TypeError, ValueError) as error:
        raise RelationalHorizontalLiftError(f"{name} is not numeric") from error
    if array.ndim != ndim or not array.size or not np.isfinite(array).all():
        raise RelationalHorizontalLiftError(
            f"{name} must be a finite non-empty {ndim}-dimensional array"
        )
    array = array.copy()
    array.flags.writeable = False
    return array


def _positive(value: object, name: str, *, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RelationalHorizontalLiftError(f"{name} is not numeric") from error
    if not math.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise RelationalHorizontalLiftError(f"{name} must be finite and {qualifier}")
    return result


def _orthogonal(value: object, rank: int, name: str) -> np.ndarray:
    matrix = _readonly(value, name=name, ndim=2)
    if matrix.shape != (rank, rank) or not np.allclose(
        matrix.T @ matrix, np.eye(rank), atol=1e-8, rtol=1e-8
    ):
        raise RelationalHorizontalLiftError(
            f"{name} must be an orthogonal [{rank}, {rank}] matrix"
        )
    return matrix


@dataclass(frozen=True, slots=True)
class GaugeLiftSample:
    """One matched natural transition expressed in a source-chart gauge."""

    sample_id: str
    chart_id: str
    family_fold: str
    tangent: np.ndarray
    fiber_delta: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("sample_id", "chart_id", "family_fold"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise RelationalHorizontalLiftError(f"{name} must be non-empty")
        tangent = _readonly(self.tangent, name="tangent", ndim=1)
        fiber = _readonly(
            self.fiber_delta,
            name="fiber_delta",
            ndim=2,
            dtype=np.dtype(np.float32),
        )
        weight = _positive(self.weight, "weight")
        object.__setattr__(self, "tangent", tangent)
        object.__setattr__(self, "fiber_delta", fiber)
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class HorizontalLiftPatch:
    """A local linear lift with a natural-transition metric and trust bounds."""

    chart_id: str
    matrix: np.ndarray
    tangent_metric: np.ndarray
    intrinsic_trust_radius: float
    layer_fiber_norm_caps: np.ndarray
    support_sample_ids: tuple[str, ...]
    support_family_folds: tuple[str, ...]
    weighted_relative_fit_error: float
    weighted_relative_roundtrip_error: float
    condition_number: float

    def __post_init__(self) -> None:
        if not isinstance(self.chart_id, str) or not self.chart_id:
            raise RelationalHorizontalLiftError("chart_id must be non-empty")
        matrix = _readonly(self.matrix, name="matrix", ndim=3)
        rank = matrix.shape[2]
        metric = _readonly(self.tangent_metric, name="tangent_metric", ndim=2)
        if metric.shape != (rank, rank) or not np.allclose(
            metric, metric.T, atol=1e-10, rtol=1e-10
        ):
            raise RelationalHorizontalLiftError(
                "tangent_metric must be symmetric and match lift rank"
            )
        if float(np.min(np.linalg.eigvalsh(metric))) <= 0.0:
            raise RelationalHorizontalLiftError("tangent_metric must be positive definite")
        caps = _readonly(
            self.layer_fiber_norm_caps,
            name="layer_fiber_norm_caps",
            ndim=1,
        )
        if caps.shape != (matrix.shape[0],) or np.any(caps <= 0.0):
            raise RelationalHorizontalLiftError(
                "layer_fiber_norm_caps must be positive and match layer count"
            )
        if not self.support_sample_ids or len(set(self.support_sample_ids)) != len(
            self.support_sample_ids
        ):
            raise RelationalHorizontalLiftError("support sample IDs must be unique")
        if not self.support_family_folds:
            raise RelationalHorizontalLiftError("support family folds must be non-empty")
        for name in (
            "weighted_relative_fit_error",
            "weighted_relative_roundtrip_error",
            "condition_number",
        ):
            _positive(getattr(self, name), name, allow_zero=True)
        object.__setattr__(
            self,
            "intrinsic_trust_radius",
            _positive(self.intrinsic_trust_radius, "intrinsic_trust_radius"),
        )
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "tangent_metric", metric)
        object.__setattr__(self, "layer_fiber_norm_caps", caps)

    @property
    def layer_count(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def rank(self) -> int:
        return int(self.matrix.shape[2])

    def intrinsic_norm(self, tangent: np.ndarray) -> float:
        vector = _readonly(tangent, name="tangent", ndim=1)
        if vector.shape != (self.rank,):
            raise RelationalHorizontalLiftError("tangent rank does not match lift")
        return float(math.sqrt(max(float(vector @ self.tangent_metric @ vector), 0.0)))

    def in_support(self, tangent: np.ndarray, *, atol: float = 1e-10) -> bool:
        return self.intrinsic_norm(tangent) <= self.intrinsic_trust_radius + atol

    def lift(self, tangent: np.ndarray, *, require_in_support: bool = True) -> np.ndarray:
        vector = _readonly(tangent, name="tangent", ndim=1)
        if vector.shape != (self.rank,):
            raise RelationalHorizontalLiftError("tangent rank does not match lift")
        if require_in_support and not self.in_support(vector):
            raise RelationalHorizontalLiftError("tangent lies outside the learned trust region")
        result = np.einsum("lhr,r->lh", self.matrix, vector, optimize=True)
        return np.asarray(result, dtype=np.float32)

    def infer_tangent(self, fiber_delta: np.ndarray) -> np.ndarray:
        fiber = _readonly(
            fiber_delta,
            name="fiber_delta",
            ndim=2,
            dtype=np.dtype(np.float32),
        )
        if fiber.shape != (self.layer_count, self.hidden_size):
            raise RelationalHorizontalLiftError("fiber shape does not match lift")
        design = self.matrix.reshape(-1, self.rank)
        result = np.linalg.pinv(design, rcond=1e-10) @ fiber.reshape(-1)
        return np.asarray(result, dtype=np.float64)

    def project_horizontal(self, fiber_delta: np.ndarray) -> np.ndarray:
        tangent = self.infer_tangent(fiber_delta)
        return self.lift(tangent, require_in_support=False)

    def transformed_gauge(self, basis: np.ndarray) -> HorizontalLiftPatch:
        """Return the same physical lift in coordinates ``v' = Q v``."""
        q = _orthogonal(basis, self.rank, "basis")
        return HorizontalLiftPatch(
            chart_id=self.chart_id,
            matrix=np.einsum("lhr,sr->lhs", self.matrix, q, optimize=True),
            tangent_metric=q @ self.tangent_metric @ q.T,
            intrinsic_trust_radius=self.intrinsic_trust_radius,
            layer_fiber_norm_caps=self.layer_fiber_norm_caps,
            support_sample_ids=self.support_sample_ids,
            support_family_folds=self.support_family_folds,
            weighted_relative_fit_error=self.weighted_relative_fit_error,
            weighted_relative_roundtrip_error=self.weighted_relative_roundtrip_error,
            condition_number=self.condition_number,
        )


def fit_horizontal_lift_patch(
    samples: Sequence[GaugeLiftSample],
    *,
    chart_id: str,
    ridge: float = 1e-3,
    metric_ridge: float = 1e-3,
    trust_quantile: float = 0.9,
    fiber_cap_quantile: float = 0.9,
    minimum_samples: int | None = None,
) -> HorizontalLiftPatch:
    """Fit a through-origin local lift from matched training transitions only."""
    rows = tuple(samples)
    if not rows or any(row.chart_id != chart_id for row in rows):
        raise RelationalHorizontalLiftError("lift samples must share the requested chart")
    if len({row.sample_id for row in rows}) != len(rows):
        raise RelationalHorizontalLiftError("lift sample IDs must be unique")
    rank = rows[0].tangent.size
    fiber_shape = rows[0].fiber_delta.shape
    if any(row.tangent.shape != (rank,) or row.fiber_delta.shape != fiber_shape for row in rows):
        raise RelationalHorizontalLiftError("lift samples have incompatible shapes")
    minimum = max(rank + 1, 3) if minimum_samples is None else int(minimum_samples)
    if minimum < rank or len(rows) < minimum:
        raise RelationalHorizontalLiftError("insufficient samples for a full-rank local lift")
    ridge = _positive(ridge, "ridge")
    metric_ridge = _positive(metric_ridge, "metric_ridge")
    for value, name in (
        (trust_quantile, "trust_quantile"),
        (fiber_cap_quantile, "fiber_cap_quantile"),
    ):
        if not 0.5 <= float(value) <= 1.0:
            raise RelationalHorizontalLiftError(f"{name} must lie in [0.5, 1]")

    x = np.stack([row.tangent for row in rows], axis=0).astype(np.float64)
    if np.linalg.matrix_rank(x, tol=1e-10) < rank:
        raise RelationalHorizontalLiftError("local tangent samples do not span the chart rank")
    y = np.stack([row.fiber_delta for row in rows], axis=0).astype(np.float64)
    weights = np.asarray([row.weight for row in rows], dtype=np.float64)
    sqrt_weights = np.sqrt(weights / weights.mean())
    xw = x * sqrt_weights[:, None]
    yw = y.reshape(len(rows), -1) * sqrt_weights[:, None]

    gram = xw.T @ xw
    scale = max(float(np.trace(gram) / rank), 1e-12)
    regularized = gram + ridge * scale * np.eye(rank)
    coefficients = np.linalg.solve(regularized, xw.T @ yw)
    matrix = coefficients.T.reshape(fiber_shape[0], fiber_shape[1], rank)

    covariance = gram / float(np.sum(sqrt_weights * sqrt_weights))
    covariance_scale = max(float(np.trace(covariance) / rank), 1e-12)
    tangent_metric = np.linalg.inv(
        covariance + metric_ridge * covariance_scale * np.eye(rank)
    )
    intrinsic_norms = np.sqrt(
        np.maximum(np.einsum("nr,rs,ns->n", x, tangent_metric, x), 0.0)
    )
    trust_radius = float(np.quantile(intrinsic_norms, trust_quantile))
    layer_norms = np.linalg.norm(y, axis=2)
    layer_caps = np.quantile(layer_norms, fiber_cap_quantile, axis=0)
    if np.any(layer_caps <= 1e-12) or trust_radius <= 1e-12:
        raise RelationalHorizontalLiftError("training transitions have degenerate trust scale")

    predicted = x @ coefficients
    target = y.reshape(len(rows), -1)
    weighted_error = np.linalg.norm((predicted - target) * sqrt_weights[:, None])
    weighted_target = max(np.linalg.norm(target * sqrt_weights[:, None]), 1e-12)
    fit_error = float(weighted_error / weighted_target)
    design = matrix.reshape(-1, rank)
    inferred = predicted @ np.linalg.pinv(design, rcond=1e-10).T
    roundtrip_error = float(
        np.linalg.norm((inferred - x) * sqrt_weights[:, None])
        / max(np.linalg.norm(x * sqrt_weights[:, None]), 1e-12)
    )
    condition = float(np.linalg.cond(regularized))
    return HorizontalLiftPatch(
        chart_id=chart_id,
        matrix=matrix,
        tangent_metric=tangent_metric,
        intrinsic_trust_radius=trust_radius,
        layer_fiber_norm_caps=layer_caps,
        support_sample_ids=tuple(sorted(row.sample_id for row in rows)),
        support_family_folds=tuple(sorted({row.family_fold for row in rows})),
        weighted_relative_fit_error=fit_error,
        weighted_relative_roundtrip_error=roundtrip_error,
        condition_number=condition,
    )


class HorizontalLift:
    """A model-specific bank of local gauge-covariant lift patches."""

    def __init__(self, patches: Mapping[str, HorizontalLiftPatch]) -> None:
        values = dict(patches)
        if not values or any(key != patch.chart_id for key, patch in values.items()):
            raise RelationalHorizontalLiftError("patch keys must bind non-empty chart IDs")
        shapes = {(patch.layer_count, patch.hidden_size, patch.rank) for patch in values.values()}
        if len(shapes) != 1:
            raise RelationalHorizontalLiftError("horizontal-lift patches have mixed fiber shapes")
        self._patches = MappingProxyType(dict(sorted(values.items())))

    @property
    def patches(self) -> Mapping[str, HorizontalLiftPatch]:
        return self._patches

    @property
    def rank(self) -> int:
        return next(iter(self._patches.values())).rank

    @property
    def layer_count(self) -> int:
        return next(iter(self._patches.values())).layer_count

    @property
    def hidden_size(self) -> int:
        return next(iter(self._patches.values())).hidden_size

    def patch(self, chart_id: str) -> HorizontalLiftPatch:
        try:
            return self._patches[chart_id]
        except KeyError as error:
            raise RelationalHorizontalLiftError("no horizontal lift for chart") from error

    def lift(
        self,
        chart_id: str,
        tangent: np.ndarray,
        *,
        require_in_support: bool = True,
    ) -> np.ndarray:
        return self.patch(chart_id).lift(
            tangent,
            require_in_support=require_in_support,
        )


__all__ = [
    "GaugeLiftSample",
    "HorizontalLift",
    "HorizontalLiftPatch",
    "RelationalHorizontalLiftError",
    "fit_horizontal_lift_patch",
]
