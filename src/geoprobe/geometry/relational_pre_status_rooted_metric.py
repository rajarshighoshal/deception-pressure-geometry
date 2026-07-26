"""Outcome-blind metric for action-free rooted pre-status relational stars."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment


class RelationalPreStatusRootedMetricError(ValueError):
    """A rooted-star metric input or fitted scale is invalid."""


@dataclass(frozen=True, slots=True)
class RootedStarMetricInput:
    """The only arrays visible to the label-free scientific distance."""

    residual_root_distances: np.ndarray
    incoming_attention: np.ndarray
    typed_annotations: np.ndarray

    def __post_init__(self) -> None:
        residual = _finite_array(
            self.residual_root_distances,
            "residual_root_distances",
            dimensions=2,
        )
        attention = _finite_array(
            self.incoming_attention,
            "incoming_attention",
            dimensions=3,
        )
        annotations = _finite_array(
            self.typed_annotations,
            "typed_annotations",
            dimensions=2,
        )
        token_count = residual.shape[1]
        if residual.shape[0] < 1 or token_count < 1:
            raise RelationalPreStatusRootedMetricError(
                "rooted residual profiles must contain layers and tokens"
            )
        if attention.shape[0] != residual.shape[0] or attention.shape[2] != token_count:
            raise RelationalPreStatusRootedMetricError(
                "residual and attention rooted stars disagree on layers or tokens"
            )
        if attention.shape[1] < 1:
            raise RelationalPreStatusRootedMetricError(
                "rooted attention requires at least one head"
            )
        if annotations.shape[0] != token_count or annotations.shape[1] < 2:
            raise RelationalPreStatusRootedMetricError(
                "typed annotations must provide normalized position and token types"
            )
        if np.any(residual < 0.0) or np.any(attention < 0.0):
            raise RelationalPreStatusRootedMetricError(
                "rooted distances and attention weights must be non-negative"
            )
        masses = attention.sum(axis=2)
        if not np.allclose(masses, 1.0, atol=2e-3, rtol=2e-3):
            raise RelationalPreStatusRootedMetricError(
                "every retained incoming attention row must be normalized"
            )
        for value in (residual, attention, annotations):
            value.flags.writeable = False
        object.__setattr__(self, "residual_root_distances", residual)
        object.__setattr__(self, "incoming_attention", attention)
        object.__setattr__(self, "typed_annotations", annotations)


@dataclass(frozen=True, slots=True)
class RootedStarDistance:
    residual: float
    attention_head_set: float

    def __post_init__(self) -> None:
        for name in ("residual", "attention_head_set"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise RelationalPreStatusRootedMetricError(
                    f"{name} distance must be finite and non-negative"
                )
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class RootedStarMetricScaler:
    """Training-fold-only positive scales for equal-view fusion."""

    residual_scale: float
    attention_scale: float

    def __post_init__(self) -> None:
        for name in ("residual_scale", "attention_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise RelationalPreStatusRootedMetricError(
                    f"{name} must be finite and positive"
                )
            object.__setattr__(self, name, value)

    @classmethod
    def fit(cls, distances: Sequence[RootedStarDistance]) -> RootedStarMetricScaler:
        rows = tuple(distances)
        if not rows:
            raise RelationalPreStatusRootedMetricError(
                "rooted-star metric scaling requires training distances"
            )

        def positive_median(name: str) -> float:
            values = np.asarray([getattr(row, name) for row in rows], dtype=np.float64)
            positive = values[np.isfinite(values) & (values > 0.0)]
            if positive.size == 0:
                raise RelationalPreStatusRootedMetricError(
                    f"{name} training distances have no positive scale"
                )
            return float(np.median(positive))

        return cls(
            residual_scale=positive_median("residual"),
            attention_scale=positive_median("attention_head_set"),
        )

    def components(self, distance: RootedStarDistance) -> tuple[float, float]:
        return (
            float(distance.residual / self.residual_scale),
            float(distance.attention_head_set / self.attention_scale),
        )

    def transform(self, distance: RootedStarDistance) -> float:
        residual, attention = self.components(distance)
        return float(math.sqrt((residual * residual + attention * attention) / 2.0))


def _finite_array(value: object, name: str, *, dimensions: int) -> np.ndarray:
    try:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    except (TypeError, ValueError) as error:
        raise RelationalPreStatusRootedMetricError(f"{name} is not numeric") from error
    if array.ndim != dimensions or not array.size or not np.isfinite(array).all():
        raise RelationalPreStatusRootedMetricError(
            f"{name} must be a finite {dimensions}-dimensional array"
        )
    return array


def _typed_cost(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    normalized_position = np.abs(left[:, None, 0] - right[None, :, 0])
    categorical = np.mean(left[:, None, 1:] != right[None, :, 1:], axis=-1)
    return np.asarray((normalized_position + categorical) / 2.0, dtype=np.float64)


def _balanced_coupling(cost: np.ndarray) -> np.ndarray:
    if cost.ndim != 2 or min(cost.shape) < 1 or not np.isfinite(cost).all():
        raise RelationalPreStatusRootedMetricError("coupling cost is invalid")
    source_count, target_count = cost.shape
    if source_count == target_count:
        rows, columns = linear_sum_assignment(cost)
        coupling = np.zeros_like(cost, dtype=np.float64)
        coupling[rows, columns] = 1.0 / source_count
        return coupling
    source = np.full(source_count, 1.0 / source_count, dtype=np.float64)
    target = np.full(target_count, 1.0 / target_count, dtype=np.float64)
    coupling = np.asarray(
        ot.emd(source, target, cost, numItermax=100_000),
        dtype=np.float64,
    )
    residual = max(
        float(np.max(np.abs(coupling.sum(axis=1) - source))),
        float(np.max(np.abs(coupling.sum(axis=0) - target))),
    )
    if coupling.shape != cost.shape or np.any(coupling < 0.0) or residual > 1e-9:
        raise RelationalPreStatusRootedMetricError("balanced coupling failed validation")
    return coupling


def _coupled_squared_profile_distance(
    left: np.ndarray,
    right: np.ndarray,
    coupling: np.ndarray,
) -> float:
    source_mass = coupling.sum(axis=1)
    target_mass = coupling.sum(axis=0)
    left_energy = np.sum((left * left) * source_mass[None, :])
    right_energy = np.sum((right * right) * target_mass[None, :])
    cross = np.sum((left @ coupling) * right)
    value = float((left_energy + right_energy - 2.0 * cross) / left.shape[0])
    if abs(value) <= 1e-15:
        return 0.0
    return float(max(value, 0.0))


def _coupled_head_costs(
    left: np.ndarray,
    right: np.ndarray,
    coupling: np.ndarray,
) -> np.ndarray:
    """Compute every head-pair cost in one contraction over the token coupling."""
    source_mass = coupling.sum(axis=1)
    target_mass = coupling.sum(axis=0)
    left_energy = (left * left) @ source_mass
    right_energy = (right * right) @ target_mass
    cross = left @ coupling @ right.T
    squared = left_energy[:, None] + right_energy[None, :] - 2.0 * cross
    squared[np.abs(squared) <= 1e-15] = 0.0
    return np.sqrt(np.maximum(squared, 0.0))


def rooted_star_distance(
    left: RootedStarMetricInput,
    right: RootedStarMetricInput,
) -> RootedStarDistance:
    """Compare two action-free rooted stars without using ambient coordinates."""
    if not isinstance(left, RootedStarMetricInput) or not isinstance(
        right, RootedStarMetricInput
    ):
        raise RelationalPreStatusRootedMetricError(
            "rooted_star_distance requires RootedStarMetricInput values"
        )
    if left.residual_root_distances.shape[0] != right.residual_root_distances.shape[0]:
        raise RelationalPreStatusRootedMetricError(
            "rooted stars must contain the same captured layers"
        )
    coupling = _balanced_coupling(
        _typed_cost(left.typed_annotations, right.typed_annotations)
    )
    residual_squared = _coupled_squared_profile_distance(
        left.residual_root_distances,
        right.residual_root_distances,
        coupling,
    )
    layer_attention_squared: list[float] = []
    for layer_index in range(left.incoming_attention.shape[0]):
        left_heads = left.incoming_attention[layer_index]
        right_heads = right.incoming_attention[layer_index]
        head_cost = _coupled_head_costs(left_heads, right_heads, coupling)
        head_coupling = _balanced_coupling(head_cost)
        layer_attention_squared.append(float(np.sum(head_coupling * head_cost**2)))
    attention_squared = float(np.mean(layer_attention_squared))
    return RootedStarDistance(
        residual=math.sqrt(residual_squared),
        attention_head_set=math.sqrt(max(attention_squared, 0.0)),
    )


def rooted_star_energy_quotient(
    cross: Sequence[RootedStarDistance],
    left_within: Sequence[RootedStarDistance],
    right_within: Sequence[RootedStarDistance],
    *,
    left_size: int,
    right_size: int,
) -> RootedStarDistance:
    """Quotient replay realizations with a component-wise empirical energy law.

    ``cross`` contains the full Cartesian product. Each within-fibre sequence
    contains unordered distinct pairs; zero self-distances are included by the
    empirical expectation here rather than materialized by callers.
    """
    if (
        left_size < 1
        or right_size < 1
        or len(cross) != left_size * right_size
        or len(left_within) != left_size * (left_size - 1) // 2
        or len(right_within) != right_size * (right_size - 1) // 2
    ):
        raise RelationalPreStatusRootedMetricError(
            "energy-quotient realization inventory is inconsistent"
        )

    def component(name: str) -> float:
        between = float(
            np.mean([float(getattr(value, name)) ** 2 for value in cross])
        )
        left_noise = 2.0 * sum(
            float(getattr(value, name)) ** 2 for value in left_within
        ) / (left_size * left_size)
        right_noise = 2.0 * sum(
            float(getattr(value, name)) ** 2 for value in right_within
        ) / (right_size * right_size)
        squared = between - 0.5 * left_noise - 0.5 * right_noise
        return math.sqrt(max(squared, 0.0))

    return RootedStarDistance(
        residual=component("residual"),
        attention_head_set=component("attention_head_set"),
    )


def rooted_star_descriptor(value: RootedStarMetricInput) -> np.ndarray:
    """Return a fixed label-free retrieval sketch; it never defines final edges."""
    quantiles = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    pieces: list[np.ndarray] = []
    for layer in value.residual_root_distances:
        pieces.append(np.quantile(layer, quantiles))
        pieces.append(np.asarray([layer.mean(), layer.std()], dtype=np.float64))
    for layer in value.incoming_attention:
        entropy = -np.sum(layer * np.log(np.maximum(layer, 1e-12)), axis=1)
        maximum = np.max(layer, axis=1)
        for summary in (entropy, maximum):
            pieces.append(np.quantile(summary, (0.0, 0.25, 0.5, 0.75, 1.0)))
            pieces.append(np.asarray([summary.mean(), summary.std()], dtype=np.float64))
    for column in value.typed_annotations.T:
        pieces.append(np.quantile(column, (0.0, 0.25, 0.5, 0.75, 1.0)))
        pieces.append(np.asarray([column.mean(), column.std()], dtype=np.float64))
    result = np.ascontiguousarray(np.concatenate(pieces), dtype=np.float64)
    if not np.isfinite(result).all():
        raise RelationalPreStatusRootedMetricError(
            "rooted-star retrieval descriptor is non-finite"
        )
    result.flags.writeable = False
    return result


__all__ = [
    "RelationalPreStatusRootedMetricError",
    "RootedStarDistance",
    "RootedStarMetricInput",
    "RootedStarMetricScaler",
    "rooted_star_descriptor",
    "rooted_star_distance",
    "rooted_star_energy_quotient",
]
