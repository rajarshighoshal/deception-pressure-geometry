"""Pressure-matched multinomial risk fields on local intrinsic gauge charts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
import math

import numpy as np

from geoprobe.geometry.relational_gauge_atlas import GaugeChartInterface


OUTCOME_CLASSES = (
    "HONEST",
    "DECEPTIVE",
    "SKIP",
    "NO_ACTION",
    "WRONG_WITHOUT_BASELINE_KNOWLEDGE",
)
FIELD_KINDS = ("absolute_deception", "pressure_residual_deception")


class RelationalIntrinsicRiskFieldError(ValueError):
    """A pressure-matched field input or local estimate is invalid."""


def _readonly(value: object, *, name: str, ndim: int) -> np.ndarray:
    try:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    except (TypeError, ValueError) as error:
        raise RelationalIntrinsicRiskFieldError(f"{name} is not numeric") from error
    if array.ndim != ndim or not array.size or not np.isfinite(array).all():
        raise RelationalIntrinsicRiskFieldError(
            f"{name} must be a finite non-empty {ndim}-dimensional array"
        )
    array = array.copy()
    array.flags.writeable = False
    return array


def _logit(probability: float) -> float:
    value = float(np.clip(probability, 1e-9, 1.0 - 1e-9))
    return float(math.log(value / (1.0 - value)))


@dataclass(frozen=True, slots=True)
class GaugeRiskObservation:
    """One repeated action draw attached to an outcome-blind quotient node."""

    observation_id: str
    node_id: Hashable
    family_fold: str
    nuisance_key: tuple[str, ...]
    outcome_class: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise RelationalIntrinsicRiskFieldError("observation_id must be non-empty")
        try:
            hash(self.node_id)
        except TypeError as error:
            raise RelationalIntrinsicRiskFieldError("node_id must be hashable") from error
        if not isinstance(self.family_fold, str) or not self.family_fold:
            raise RelationalIntrinsicRiskFieldError("family_fold must be non-empty")
        if not self.nuisance_key or any(
            not isinstance(value, str) or not value for value in self.nuisance_key
        ):
            raise RelationalIntrinsicRiskFieldError(
                "nuisance_key must contain non-empty strings"
            )
        if self.outcome_class not in OUTCOME_CLASSES:
            raise RelationalIntrinsicRiskFieldError("outcome_class is invalid")
        try:
            weight = float(self.weight)
        except (TypeError, ValueError) as error:
            raise RelationalIntrinsicRiskFieldError("weight is not numeric") from error
        if not math.isfinite(weight) or weight <= 0.0:
            raise RelationalIntrinsicRiskFieldError("weight must be finite and positive")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class PressureMatchedFieldConfig:
    minimum_support_nodes: int = 4
    minimum_effective_observations: float = 4.0
    ridge: float = 1e-3
    dirichlet_alpha: float = 0.5
    bandwidth_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.minimum_support_nodes < 2:
            raise RelationalIntrinsicRiskFieldError(
                "minimum_support_nodes must be at least two"
            )
        for name in (
            "minimum_effective_observations",
            "ridge",
            "dirichlet_alpha",
            "bandwidth_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise RelationalIntrinsicRiskFieldError(
                    f"{name} must be finite and positive"
                )
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class PressureMatchedFieldEvaluation:
    chart_id: str
    nuisance_key: tuple[str, ...]
    defined: bool
    reason: str
    outcome_probabilities: np.ndarray
    nuisance_probabilities: np.ndarray
    absolute_deception_log_odds: float
    pressure_residual_deception_log_odds: float
    absolute_gradient: np.ndarray
    pressure_residual_gradient: np.ndarray
    support_node_ids: tuple[Hashable, ...]
    support_observation_count: int
    effective_observation_count: float
    bandwidth: float

    def __post_init__(self) -> None:
        if not isinstance(self.chart_id, str) or not self.chart_id:
            raise RelationalIntrinsicRiskFieldError("chart_id must be non-empty")
        if not isinstance(self.reason, str) or not self.reason:
            raise RelationalIntrinsicRiskFieldError("reason must be non-empty")
        local = _readonly(
            self.outcome_probabilities,
            name="outcome_probabilities",
            ndim=1,
        )
        nuisance = _readonly(
            self.nuisance_probabilities,
            name="nuisance_probabilities",
            ndim=1,
        )
        if local.shape != (len(OUTCOME_CLASSES),) or nuisance.shape != local.shape:
            raise RelationalIntrinsicRiskFieldError(
                "probabilities must match the registered outcome alphabet"
            )
        if np.any(local < 0.0) or np.any(nuisance < 0.0):
            raise RelationalIntrinsicRiskFieldError("probabilities cannot be negative")
        if not np.isclose(local.sum(), 1.0) or not np.isclose(nuisance.sum(), 1.0):
            raise RelationalIntrinsicRiskFieldError("probabilities must sum to one")
        absolute = _readonly(self.absolute_gradient, name="absolute_gradient", ndim=1)
        residual = _readonly(
            self.pressure_residual_gradient,
            name="pressure_residual_gradient",
            ndim=1,
        )
        if residual.shape != absolute.shape:
            raise RelationalIntrinsicRiskFieldError("field gradients disagree on rank")
        for name in (
            "absolute_deception_log_odds",
            "pressure_residual_deception_log_odds",
            "effective_observation_count",
            "bandwidth",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise RelationalIntrinsicRiskFieldError(f"{name} must be finite")
        object.__setattr__(self, "outcome_probabilities", local)
        object.__setattr__(self, "nuisance_probabilities", nuisance)
        object.__setattr__(self, "absolute_gradient", absolute)
        object.__setattr__(self, "pressure_residual_gradient", residual)

    def depth(self, field_kind: str) -> float:
        if field_kind == "absolute_deception":
            return float(self.absolute_deception_log_odds)
        if field_kind == "pressure_residual_deception":
            return float(self.pressure_residual_deception_log_odds)
        raise RelationalIntrinsicRiskFieldError("field_kind is invalid")

    def gradient(self, field_kind: str) -> np.ndarray:
        if field_kind == "absolute_deception":
            return self.absolute_gradient.copy()
        if field_kind == "pressure_residual_deception":
            return self.pressure_residual_gradient.copy()
        raise RelationalIntrinsicRiskFieldError("field_kind is invalid")


class PressureMatchedRiskField:
    """A transparent local field that never mixes different pressure cells."""

    def __init__(
        self,
        observations: Sequence[GaugeRiskObservation],
        *,
        config: PressureMatchedFieldConfig | None = None,
        held_out_family_fold: str | None = None,
    ) -> None:
        rows = tuple(observations)
        if not rows or len({row.observation_id for row in rows}) != len(rows):
            raise RelationalIntrinsicRiskFieldError(
                "risk observations must be non-empty and uniquely identified"
            )
        if held_out_family_fold is not None and any(
            row.family_fold == held_out_family_fold for row in rows
        ):
            raise RelationalIntrinsicRiskFieldError(
                "held-out outcome entered pressure-matched field training"
            )
        self.config = config or PressureMatchedFieldConfig()
        self.held_out_family_fold = held_out_family_fold
        self._rows = rows
        grouped: dict[tuple[str, ...], list[GaugeRiskObservation]] = defaultdict(list)
        for row in rows:
            grouped[row.nuisance_key].append(row)
        self._by_nuisance = {
            key: tuple(sorted(values, key=lambda row: row.observation_id))
            for key, values in grouped.items()
        }

    @property
    def observations(self) -> tuple[GaugeRiskObservation, ...]:
        return self._rows

    def _undefined(
        self,
        chart: GaugeChartInterface,
        nuisance_key: tuple[str, ...],
        reason: str,
    ) -> PressureMatchedFieldEvaluation:
        rank = int(np.asarray(chart.coordinates).shape[1])
        uniform = np.full(len(OUTCOME_CLASSES), 1.0 / len(OUTCOME_CLASSES))
        return PressureMatchedFieldEvaluation(
            chart_id=chart.chart_id,
            nuisance_key=nuisance_key,
            defined=False,
            reason=reason,
            outcome_probabilities=uniform,
            nuisance_probabilities=uniform,
            absolute_deception_log_odds=0.0,
            pressure_residual_deception_log_odds=0.0,
            absolute_gradient=np.zeros(rank),
            pressure_residual_gradient=np.zeros(rank),
            support_node_ids=(),
            support_observation_count=0,
            effective_observation_count=0.0,
            bandwidth=0.0,
        )

    def evaluate(
        self,
        chart: GaugeChartInterface,
        coordinate: np.ndarray,
        *,
        nuisance_key: tuple[str, ...],
    ) -> PressureMatchedFieldEvaluation:
        coordinates = _readonly(chart.coordinates, name="chart coordinates", ndim=2)
        query = _readonly(coordinate, name="coordinate", ndim=1)
        if query.shape != (coordinates.shape[1],):
            raise RelationalIntrinsicRiskFieldError(
                "query coordinate rank does not match chart"
            )
        rows = self._by_nuisance.get(tuple(nuisance_key), ())
        if not rows:
            return self._undefined(chart, tuple(nuisance_key), "unseen_nuisance_cell")
        index = {node: position for position, node in enumerate(chart.support_ids)}
        support_rows = tuple(row for row in rows if row.node_id in index)
        support_nodes = tuple(sorted({row.node_id for row in support_rows}, key=repr))
        if len(support_nodes) < self.config.minimum_support_nodes:
            return self._undefined(chart, tuple(nuisance_key), "insufficient_matched_nodes")

        node_coordinates = np.stack(
            [coordinates[index[node]] for node in support_nodes], axis=0
        )
        distances = np.linalg.norm(node_coordinates - query[None, :], axis=1)
        positive = distances[distances > 1e-12]
        if positive.size == 0:
            return self._undefined(chart, tuple(nuisance_key), "zero_bandwidth_support")
        bandwidth = max(
            float(np.median(positive) * self.config.bandwidth_scale), 1e-9
        )
        kernel = np.exp(-0.5 * (distances / bandwidth) ** 2)

        class_index = {name: i for i, name in enumerate(OUTCOME_CLASSES)}
        node_counts = np.zeros((len(support_nodes), len(OUTCOME_CLASSES)))
        node_lookup = {node: i for i, node in enumerate(support_nodes)}
        nuisance_counts = np.full(
            len(OUTCOME_CLASSES), self.config.dirichlet_alpha, dtype=np.float64
        )
        for row in rows:
            class_id = class_index[row.outcome_class]
            nuisance_counts[class_id] += row.weight
            local_index = node_lookup.get(row.node_id)
            if local_index is not None:
                node_counts[local_index, class_id] += row.weight
        nuisance_probabilities = nuisance_counts / nuisance_counts.sum()
        node_totals = node_counts.sum(axis=1)
        effective = float(
            (np.sum(kernel * node_totals) ** 2)
            / max(np.sum((kernel * node_totals) ** 2), 1e-12)
        )
        if effective < self.config.minimum_effective_observations:
            return self._undefined(
                chart, tuple(nuisance_key), "insufficient_effective_observations"
            )

        smoothed_counts = node_counts + self.config.dirichlet_alpha
        node_probabilities = smoothed_counts / smoothed_counts.sum(axis=1, keepdims=True)
        weighted_counts = np.sum(kernel[:, None] * smoothed_counts, axis=0)
        local_probabilities = weighted_counts / weighted_counts.sum()
        deceptive_index = class_index["DECEPTIVE"]
        node_absolute = np.asarray(
            [_logit(value) for value in node_probabilities[:, deceptive_index]]
        )
        nuisance_log_odds = _logit(nuisance_probabilities[deceptive_index])
        node_residual = node_absolute - nuisance_log_odds

        offsets = node_coordinates - query[None, :]
        design = np.column_stack([np.ones(len(support_nodes)), offsets])
        observation_weights = kernel * np.maximum(node_totals, 1.0)
        weighted_design = design * np.sqrt(observation_weights)[:, None]
        scale = max(
            float(np.trace(weighted_design[:, 1:].T @ weighted_design[:, 1:]))
            / coordinates.shape[1],
            1e-12,
        )
        penalty = np.eye(design.shape[1]) * self.config.ridge * scale
        penalty[0, 0] = 0.0
        normal = weighted_design.T @ weighted_design + penalty

        def local_linear(values: np.ndarray) -> np.ndarray:
            target = values * np.sqrt(observation_weights)
            return np.linalg.solve(normal, weighted_design.T @ target)

        absolute_fit = local_linear(node_absolute)
        residual_fit = local_linear(node_residual)
        return PressureMatchedFieldEvaluation(
            chart_id=chart.chart_id,
            nuisance_key=tuple(nuisance_key),
            defined=True,
            reason="defined",
            outcome_probabilities=local_probabilities,
            nuisance_probabilities=nuisance_probabilities,
            absolute_deception_log_odds=float(absolute_fit[0]),
            pressure_residual_deception_log_odds=float(residual_fit[0]),
            absolute_gradient=absolute_fit[1:],
            pressure_residual_gradient=residual_fit[1:],
            support_node_ids=support_nodes,
            support_observation_count=len(support_rows),
            effective_observation_count=effective,
            bandwidth=bandwidth,
        )


__all__ = [
    "FIELD_KINDS",
    "OUTCOME_CLASSES",
    "GaugeRiskObservation",
    "PressureMatchedFieldConfig",
    "PressureMatchedFieldEvaluation",
    "PressureMatchedRiskField",
    "RelationalIntrinsicRiskFieldError",
]
