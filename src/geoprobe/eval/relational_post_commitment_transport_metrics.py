"""Scale-aware reconstruction metrics for held-out relational transport orbits."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from math import comb
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


MetricValue = float | None
METRIC_NAMES = (
    "cosine",
    "normalized_squared_error",
    "norm_ratio",
    "dot",
    "alignment",
)


@dataclass(frozen=True)
class ReconstructionMetricSummary:
    """Metrics for one pooled vector or one layer."""

    cosine: MetricValue
    normalized_squared_error: MetricValue
    norm_ratio: MetricValue
    dot: float
    alignment: MetricValue

    def as_dict(self) -> dict[str, MetricValue]:
        return {name: getattr(self, name) for name in METRIC_NAMES}


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Immutable pooled and per-layer reconstruction summaries."""

    pooled: ReconstructionMetricSummary
    per_layer: tuple[ReconstructionMetricSummary, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pooled": self.pooled.as_dict(),
            "per_layer": [summary.as_dict() for summary in self.per_layer],
        }


@dataclass(frozen=True)
class TransportMetricRow:
    """One planned held-out row and its model-level reconstruction metrics."""

    model: str
    scenario_id: str
    family: str
    fold: str
    metrics: ReconstructionMetrics


@dataclass(frozen=True)
class PairedDifferenceRow:
    """A planned local/comparator pair, including undefined values explicitly."""

    scenario_id: str
    family: str
    fold: str
    local: MetricValue
    comparator: MetricValue
    difference: MetricValue


def _as_finite_matrix(values: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have nonempty shape [layer, hidden]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_inputs(
    actual: Any, predicted: Any, layer_scales: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual_array = _as_finite_matrix(actual, name="actual")
    predicted_array = _as_finite_matrix(predicted, name="predicted")
    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted must have identical [layer, hidden] shapes")
    try:
        scales = np.asarray(layer_scales, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("layer_scales must be a numeric one-dimensional array") from error
    if scales.ndim != 1 or scales.shape[0] != actual_array.shape[0]:
        raise ValueError("layer_scales must have shape [layer]")
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("layer_scales must be finite and strictly positive")
    return actual_array, predicted_array, scales


def _summary(actual: np.ndarray, predicted: np.ndarray) -> ReconstructionMetricSummary:
    target_energy = float(np.dot(actual, actual))
    predicted_energy = float(np.dot(predicted, predicted))
    error = actual - predicted
    error_energy = float(np.dot(error, error))
    dot = float(np.dot(actual, predicted))
    if target_energy == 0.0:
        return ReconstructionMetricSummary(None, None, None, dot, None)
    target_norm = float(np.sqrt(target_energy))
    predicted_norm = float(np.sqrt(predicted_energy))
    cosine = None if predicted_energy == 0.0 else dot / (target_norm * predicted_norm)
    return ReconstructionMetricSummary(
        cosine=cosine,
        normalized_squared_error=error_energy / target_energy,
        norm_ratio=predicted_norm / target_norm,
        dot=dot,
        alignment=dot / target_energy,
    )


def reconstruction_metrics(
    actual: Any, predicted: Any, layer_scales: Any
) -> ReconstructionMetrics:
    """Score a held-out orbit without fitting or standardizing its coordinates.

    Each layer is divided by its positive train-only scale before pooling.  A
    zero-energy target has undefined relative metrics (``None``), rather than a
    misleading zero or an infinite ratio; its dot product remains well-defined.
    """
    actual_array, predicted_array, scales = _validated_inputs(actual, predicted, layer_scales)
    scaled_actual = actual_array / scales[:, None]
    scaled_predicted = predicted_array / scales[:, None]
    return ReconstructionMetrics(
        pooled=_summary(scaled_actual.reshape(-1), scaled_predicted.reshape(-1)),
        per_layer=tuple(
            _summary(scaled_actual[layer], scaled_predicted[layer])
            for layer in range(actual_array.shape[0])
        ),
    )


def score_models(
    actual: Any, predictions: Mapping[str, Any], layer_scales: Any
) -> Mapping[str, ReconstructionMetrics]:
    """Score named predictions against one held-out target with shared scales."""
    if not isinstance(predictions, Mapping):
        raise ValueError("predictions must be a mapping from model name to array")
    actual_array, _, scales = _validated_inputs(actual, actual, layer_scales)
    scored: dict[str, ReconstructionMetrics] = {}
    for model, prediction in predictions.items():
        if not isinstance(model, str) or not model:
            raise ValueError("prediction model names must be nonempty strings")
        prediction_array = _as_finite_matrix(prediction, name=f"predictions[{model!r}]")
        if prediction_array.shape != actual_array.shape:
            raise ValueError(f"predictions[{model!r}] does not match actual shape")
        scored[model] = reconstruction_metrics(actual_array, prediction_array, scales)
    return MappingProxyType(scored)


def _row_value(row: Mapping[str, Any] | TransportMetricRow, metric: str) -> MetricValue:
    if metric not in METRIC_NAMES:
        raise ValueError(f"unknown metric {metric!r}")
    if isinstance(row, TransportMetricRow):
        return getattr(row.metrics.pooled, metric)
    if metric in row:
        value = row[metric]
    else:
        metrics = row.get("metrics")
        if isinstance(metrics, ReconstructionMetrics):
            value = getattr(metrics.pooled, metric)
        elif isinstance(metrics, Mapping):
            pooled = metrics.get("pooled", metrics)
            value = pooled.get(metric) if isinstance(pooled, Mapping) else None
        else:
            value = None
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"row metric {metric!r} must be finite or None")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"row metric {metric!r} must be finite or None")
    return result


def _field(row: Mapping[str, Any] | TransportMetricRow, name: str) -> str:
    value = getattr(row, name) if isinstance(row, TransportMetricRow) else row.get(name)
    if value is None:
        raise ValueError(f"row is missing required {name!r}")
    return str(value)


def _scalar_summary(values: Sequence[MetricValue]) -> dict[str, Any]:
    defined = [value for value in values if value is not None]
    count = len(values)
    return {
        "count": count,
        "defined_count": len(defined),
        "coverage": len(defined) / count if count else 0.0,
        "mean": float(np.mean(defined)) if defined else None,
        "median": float(np.median(defined)) if defined else None,
    }


def aggregate_rows(
    rows: Sequence[Mapping[str, Any] | TransportMetricRow], metric: str
) -> dict[str, Any]:
    """Aggregate planned rows, retaining undefined values in every denominator."""
    values = [_row_value(row, metric) for row in rows]
    report = _scalar_summary(values)
    by_family: dict[str, list[MetricValue]] = defaultdict(list)
    by_fold: dict[str, list[MetricValue]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        by_family[_field(row, "family")].append(value)
        by_fold[_field(row, "fold")].append(value)
    family = {name: _scalar_summary(by_family[name]) for name in sorted(by_family)}
    folds = {name: _scalar_summary(by_fold[name]) for name in sorted(by_fold)}
    family_means = [entry["mean"] for entry in family.values() if entry["mean"] is not None]
    fold_means = [entry["mean"] for entry in folds.values() if entry["mean"] is not None]
    report.update(
        {
            "family": family,
            "family_macro_mean": float(np.mean(family_means)) if family_means else None,
            "fold": folds,
            "fold_macro_mean": float(np.mean(fold_means)) if fold_means else None,
        }
    )
    return report


def _pair_key(row: Mapping[str, Any] | TransportMetricRow, pair_fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(_field(row, field) for field in pair_fields)


def paired_local_minus_comparator(
    rows: Sequence[Mapping[str, Any] | TransportMetricRow],
    metric: str,
    *,
    local_model: str,
    comparator_model: str,
    pair_fields: Sequence[str] = ("scenario_id", "family", "fold"),
) -> dict[str, Any]:
    """Pair planned model rows while exposing missing or undefined local outcomes."""
    if not pair_fields:
        raise ValueError("pair_fields must not be empty")
    local: dict[tuple[str, ...], Mapping[str, Any] | TransportMetricRow] = {}
    comparator: dict[tuple[str, ...], Mapping[str, Any] | TransportMetricRow] = {}
    for row in rows:
        model = _field(row, "model")
        key = _pair_key(row, pair_fields)
        destination = local if model == local_model else comparator if model == comparator_model else None
        if destination is not None:
            if key in destination:
                raise ValueError(f"duplicate {model!r} row for pair key {key!r}")
            destination[key] = row
    paired_rows: list[PairedDifferenceRow] = []
    for key in sorted(local):
        local_row = local[key]
        comparison_row = comparator.get(key)
        local_value = _row_value(local_row, metric)
        comparator_value = _row_value(comparison_row, metric) if comparison_row else None
        difference = (
            local_value - comparator_value
            if local_value is not None and comparator_value is not None
            else None
        )
        paired_rows.append(
            PairedDifferenceRow(
                scenario_id=_field(local_row, "scenario_id"),
                family=_field(local_row, "family"),
                fold=_field(local_row, "fold"),
                local=local_value,
                comparator=comparator_value,
                difference=difference,
            )
        )
    differences = [row.difference for row in paired_rows]
    report = _scalar_summary(differences)
    report.update(
        {
            "local_model": local_model,
            "comparator_model": comparator_model,
            "local_defined_count": sum(row.local is not None for row in paired_rows),
            "comparator_defined_count": sum(row.comparator is not None for row in paired_rows),
            "paired_rows": tuple(paired_rows),
        }
    )
    return report


def _paired_difference(row: PairedDifferenceRow | Mapping[str, Any]) -> MetricValue:
    if isinstance(row, PairedDifferenceRow):
        return row.difference
    value = row.get("difference")
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("paired differences must be finite or None")
    return result


def _scenario_id(row: PairedDifferenceRow | Mapping[str, Any]) -> str:
    return row.scenario_id if isinstance(row, PairedDifferenceRow) else _field(row, "scenario_id")


def scenario_cluster_bootstrap_ci(
    paired_rows: Sequence[PairedDifferenceRow | Mapping[str, Any]],
    *,
    seed: int = 0,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Deterministic percentile CI resampling complete scenario clusters."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    clusters: dict[str, list[MetricValue]] = defaultdict(list)
    for row in paired_rows:
        clusters[_scenario_id(row)].append(_paired_difference(row))
    if not clusters:
        raise ValueError("paired_rows must not be empty")
    cluster_values = [clusters[scenario_id] for scenario_id in sorted(clusters)]
    defined_cluster_indices = [
        index for index, cluster in enumerate(cluster_values) if any(value is not None for value in cluster)
    ]
    if not defined_cluster_indices:
        raise ValueError("paired_rows have no defined differences")
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    completed = 0
    attempts = 0
    max_attempts = max(resamples * 100, resamples)
    while completed < resamples and attempts < max_attempts:
        draw = rng.integers(0, len(cluster_values), size=len(cluster_values))
        retained = [value for cluster_index in draw for value in cluster_values[cluster_index]]
        defined = [value for value in retained if value is not None]
        attempts += 1
        if defined:
            samples[completed] = float(np.mean(defined))
            completed += 1
    conditional_completion_count = 0
    if completed < resamples:
        undefined_cluster_indices = np.setdiff1d(
            np.arange(len(cluster_values), dtype=np.int64),
            np.asarray(defined_cluster_indices, dtype=np.int64),
            assume_unique=True,
        )
        probability_defined = len(defined_cluster_indices) / len(cluster_values)
        counts = np.arange(1, len(cluster_values) + 1, dtype=np.int64)
        probabilities = np.asarray(
            [
                comb(len(cluster_values), int(count))
                * probability_defined**count
                * (1.0 - probability_defined) ** (len(cluster_values) - count)
                for count in counts
            ],
            dtype=np.float64,
        )
        probabilities /= probabilities.sum()
        while completed < resamples:
            defined_draws = int(rng.choice(counts, p=probabilities))
            draw = np.empty(len(cluster_values), dtype=np.int64)
            positions = rng.choice(len(cluster_values), size=defined_draws, replace=False)
            draw[positions] = rng.choice(defined_cluster_indices, size=defined_draws, replace=True)
            remaining = np.setdiff1d(np.arange(len(cluster_values)), positions, assume_unique=True)
            if remaining.size:
                draw[remaining] = rng.choice(undefined_cluster_indices, size=remaining.size, replace=True)
            retained = [value for cluster_index in draw for value in cluster_values[cluster_index]]
            samples[completed] = float(np.mean([value for value in retained if value is not None]))
            completed += 1
            conditional_completion_count += 1
    alpha = (1.0 - confidence) / 2.0
    return {
        "row_count": sum(len(values) for values in cluster_values),
        "defined_count": sum(value is not None for values in cluster_values for value in values),
        "scenario_count": len(cluster_values),
        "seed": seed,
        "resamples": resamples,
        "attempts": attempts,
        "skipped_empty_resamples": attempts - (resamples - conditional_completion_count),
        "conditional_completion_count": conditional_completion_count,
        "confidence": confidence,
        "point": float(np.mean([value for values in cluster_values for value in values if value is not None])),
        "interval": [float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))],
    }


def exact_sign_flip_test(
    differences: Sequence[float], *, max_exact_n: int = 16
) -> dict[str, Any]:
    """Two-sided exact sign-flip test for a small set of finite paired effects."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("differences must be a nonempty finite one-dimensional sequence")
    if not isinstance(max_exact_n, int) or isinstance(max_exact_n, bool) or max_exact_n < 1:
        raise ValueError("max_exact_n must be a positive integer")
    if values.size > max_exact_n:
        raise ValueError(f"exact sign-flip test supports at most {max_exact_n} differences")
    observed = float(np.mean(values))
    null = np.fromiter(
        (float(np.mean(np.asarray(signs, dtype=np.float64) * values)) for signs in product((-1, 1), repeat=values.size)),
        dtype=np.float64,
        count=2 ** values.size,
    )
    extreme = int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-15))
    return {
        "n": int(values.size),
        "observed": observed,
        "permutations": int(null.size),
        "p_value_two_sided": extreme / null.size,
    }


exact_sign_flip_permutation = exact_sign_flip_test


__all__ = [
    "METRIC_NAMES",
    "PairedDifferenceRow",
    "ReconstructionMetrics",
    "ReconstructionMetricSummary",
    "TransportMetricRow",
    "aggregate_rows",
    "exact_sign_flip_permutation",
    "exact_sign_flip_test",
    "paired_local_minus_comparator",
    "reconstruction_metrics",
    "scenario_cluster_bootstrap_ci",
    "score_models",
]
