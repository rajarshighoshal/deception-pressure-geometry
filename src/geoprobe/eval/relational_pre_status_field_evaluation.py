"""Held-out evaluation of pre-status risk and shared honestward fields."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import math
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from geoprobe.eval.relational_outcome_events import OUTCOME_CLASSES
from geoprobe.eval.relational_post_commitment_transport_metrics import (
    aggregate_rows,
    reconstruction_metrics,
    scenario_cluster_bootstrap_ci,
)
from geoprobe.eval.relational_pre_status_risk_field import (
    FoldSafePreStatusRiskField,
    multiclass_brier,
    multiclass_log_loss,
)
from geoprobe.eval.relational_pre_status_risk_diagnostics import (
    descriptive_train_fold_mixed_root_proper_score_floor,
    multiclass_calibration_summary,
    shuffle_train_fold_outcome_labels,
    shuffle_train_fold_root_identities,
)
from geoprobe.eval.relational_pre_status_supervision import (
    RelationalPreStatusSupervision,
)
from geoprobe.geometry.relational_pre_status_honestward import (
    HonestwardCrossingObservation,
    SharedPreStatusHonestwardField,
)
from geoprobe.geometry.relational_pre_status_rooted_graph import (
    FOLDS,
    FoldExactRootedGraph,
)


GraphInventory = Mapping[
    str,
    Mapping[str, Mapping[str, FoldExactRootedGraph]],
]


class RelationalPreStatusFieldEvaluationError(ValueError):
    """A held-out field evaluation violates its inventory contract."""


def _graph(
    graphs: GraphInventory,
    view: str,
    variant: str,
    fold: str,
) -> FoldExactRootedGraph:
    try:
        graph = graphs[view][variant][fold]
    except KeyError as error:
        raise RelationalPreStatusFieldEvaluationError(
            f"missing graph {view}/{variant}/{fold}"
        ) from error
    if graph.held_out_family_fold != fold:
        raise RelationalPreStatusFieldEvaluationError(
            "graph held-out fold disagrees with its inventory key"
        )
    return graph


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return sha256(array.tobytes(order="C")).hexdigest()


def _probability_sha256(value: Mapping[str, float]) -> str:
    payload = json.dumps(
        {label: float(value[label]) for label in OUTCOME_CLASSES},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _cluster_ci(rows: Sequence[Mapping[str, Any]], *, seed: int) -> Mapping[str, Any] | None:
    if not rows:
        return None
    return scenario_cluster_bootstrap_ci(rows, seed=seed, resamples=2_000)


def _risk_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    models = tuple(sorted(rows[0]["probabilities"])) if rows else ()
    result: dict[str, Any] = {
        "event_count": len(rows),
        "outcome_counts": {
            label: sum(row["outcome_class"] == label for row in rows)
            for label in OUTCOME_CLASSES
        },
    }
    for model in models:
        log_losses = [float(row["scores"][model]["log_loss"]) for row in rows]
        briers = [float(row["scores"][model]["brier"]) for row in rows]
        honest_deceptive = [
            row for row in rows if row["outcome_class"] in {"HONEST", "DECEPTIVE"}
        ]
        probabilities = [
            row["probabilities"][model]["DECEPTIVE"]
            / (
                row["probabilities"][model]["DECEPTIVE"]
                + row["probabilities"][model]["HONEST"]
            )
            for row in honest_deceptive
        ]
        labels = [row["outcome_class"] == "DECEPTIVE" for row in honest_deceptive]
        conditional_log_losses = [
            -math.log(probability if label else 1.0 - probability)
            for probability, label in zip(probabilities, labels, strict=True)
        ]
        result[model] = {
            "mean_log_loss": _mean(log_losses),
            "mean_brier": _mean(briers),
            "honest_deceptive_count": len(honest_deceptive),
            "honest_deceptive_conditional_log_loss": _mean(
                conditional_log_losses
            ),
            "honest_deceptive_auroc": (
                float(roc_auc_score(labels, probabilities))
                if labels and len(set(labels)) == 2
                else None
            ),
        }
        result[model]["calibration"] = multiclass_calibration_summary(
            [str(row["outcome_class"]) for row in rows],
            [row["probabilities"][model] for row in rows],
        )
    paired = [
        {
            "scenario_id": row["scenario_id"],
            "difference": row["scores"]["nuisance"]["log_loss"]
            - row["scores"]["local"]["log_loss"],
        }
        for row in rows
    ]
    result["local_log_loss_gain_over_nuisance"] = {
        "mean": _mean([float(row["difference"]) for row in paired]),
        "scenario_cluster_ci": _cluster_ci(paired, seed=20260721),
    }
    result["support"] = {
        "mean_event_count": _mean([float(row["support_count"]) for row in rows]),
        "zero_support_count": sum(int(row["support_count"]) == 0 for row in rows),
    }
    cohorts: dict[str, dict[str, Any]] = {}
    cohort_rows = {
        "pressure_exposed": [row for row in rows if row["pressure_exposed"]],
        "no_pressure": [row for row in rows if not row["pressure_exposed"]],
        "true_PASS": [row for row in rows if row["true_status"] == "PASS"],
        "true_FAIL": [row for row in rows if row["true_status"] == "FAIL"],
        **{
            f"turn_{turn}": [row for row in rows if row["turn_index"] == turn]
            for turn in sorted({int(row["turn_index"]) for row in rows})
        },
    }
    for name, selected in cohort_rows.items():
        differences = [
            float(row["scores"]["nuisance"]["log_loss"])
            - float(row["scores"]["local"]["log_loss"])
            for row in selected
        ]
        cohorts[name] = {
            "event_count": len(selected),
            "local_log_loss_gain_over_nuisance": _mean(differences),
            "outcome_counts": {
                label: sum(row["outcome_class"] == label for row in selected)
                for label in OUTCOME_CLASSES
            },
        }
    result["cohorts"] = cohorts
    return result


def evaluate_pre_status_risk_fields(
    supervision: RelationalPreStatusSupervision,
    graphs: GraphInventory,
) -> dict[str, Any]:
    """Score five-class held-out risk predictions for every graph view/variant."""
    report: dict[str, Any] = {
        "views": {},
        "rows": [],
        "pre_score_prediction_inventory": [],
        "descriptive_train_fold_mixed_root_floors": [],
    }
    for view, events in sorted(supervision.risk_events_by_view.items()):
        variants = tuple(sorted(graphs.get(view, {})))
        if not variants:
            raise RelationalPreStatusFieldEvaluationError(
                f"view {view} has no graph variants"
            )
        view_rows: dict[str, list[dict[str, Any]]] = {
            variant: [] for variant in variants
        }
        for fold in FOLDS:
            training = tuple(event for event in events if event.family_fold != fold)
            heldout = tuple(event for event in events if event.family_fold == fold)
            if not training or not heldout:
                raise RelationalPreStatusFieldEvaluationError(
                    "risk fold lacks training or held-out events"
                )
            field = FoldSafePreStatusRiskField.fit(
                training,
                held_out_family_fold=fold,
            )
            report["descriptive_train_fold_mixed_root_floors"].append(
                {
                    "view": view,
                    **descriptive_train_fold_mixed_root_proper_score_floor(
                        training,
                        held_out_family_fold=fold,
                    ),
                }
            )
            for variant in variants:
                graph = _graph(graphs, view, variant, fold)
                fold_seed = 20260721 + FOLDS.index(fold)
                label_shuffle = shuffle_train_fold_outcome_labels(
                    training,
                    held_out_family_fold=fold,
                    seed=fold_seed,
                )
                root_shuffle = shuffle_train_fold_root_identities(
                    training,
                    graph,
                    held_out_family_fold=fold,
                    seed=fold_seed + 100,
                )
                control_fields = {
                    "label_shuffle": FoldSafePreStatusRiskField.fit(
                        label_shuffle.events,
                        held_out_family_fold=fold,
                    ),
                    "root_identity_shuffle": FoldSafePreStatusRiskField.fit(
                        root_shuffle.events,
                        held_out_family_fold=fold,
                    ),
                }
                predictions: dict[str, dict[str, Any]] = {}
                for event in heldout:
                    edges = graph.query_edges.get(event.root_id)
                    if edges is None:
                        raise RelationalPreStatusFieldEvaluationError(
                            "held-out risk root is absent from its graph"
                        )
                    primary = field.predict(
                        event_id=event.event_id,
                        root_id=event.root_id,
                        nuisance_key=event.nuisance_key,
                        edges=edges,
                    )
                    predictions[event.event_id] = {
                        "local": primary,
                        **{
                            name: control.predict(
                                event_id=event.event_id,
                                root_id=event.root_id,
                                nuisance_key=event.nuisance_key,
                                edges=edges,
                            )
                            for name, control in control_fields.items()
                        },
                    }
                for event in heldout:
                    event_predictions = predictions[event.event_id]
                    prediction = event_predictions["local"]
                    report["pre_score_prediction_inventory"].append(
                        {
                            "view": view,
                            "variant": variant,
                            "fold": fold,
                            "event_id": event.event_id,
                            "root_id": event.root_id,
                            "probability_sha256": {
                                name: _probability_sha256(
                                    value.local_probabilities
                                )
                                for name, value in event_predictions.items()
                            },
                            "support_count": prediction.support_count,
                            "support_root_ids": list(prediction.support_root_ids),
                        }
                    )
                for event in heldout:
                    event_predictions = predictions[event.event_id]
                    prediction = event_predictions["local"]
                    outcome = supervision.outcomes_by_event_id[event.event_id]
                    probabilities = {
                        "local": dict(prediction.local_probabilities),
                        "nuisance": dict(prediction.nuisance_probabilities),
                        "base": dict(prediction.base_probabilities),
                    }
                    for name, control_prediction in event_predictions.items():
                        if name != "local":
                            probabilities[name] = dict(
                                control_prediction.local_probabilities
                            )
                    scores = {
                        model: {
                            "log_loss": multiclass_log_loss(
                                event.outcome_class, values
                            ),
                            "brier": multiclass_brier(
                                event.outcome_class, values
                            ),
                        }
                        for model, values in probabilities.items()
                    }
                    row = {
                        "view": view,
                        "variant": variant,
                        "fold": fold,
                        "event_id": event.event_id,
                        "root_id": event.root_id,
                        "family": event.family,
                        "scenario_id": outcome.scenario_id,
                        "turn_index": outcome.turn_index,
                        "true_status": outcome.true_status,
                        "pressure_exposed": outcome.pressure_exposed,
                        "outcome_class": event.outcome_class,
                        "probabilities": probabilities,
                        "scores": scores,
                        "support_count": prediction.support_count,
                        "support_root_ids": list(prediction.support_root_ids),
                    }
                    view_rows[variant].append(row)
                    report["rows"].append(row)
        report["views"][view] = {
            variant: _risk_summary(rows)
            for variant, rows in view_rows.items()
        }
    return report


def _layer_scales(
    observations: Sequence[HonestwardCrossingObservation],
) -> np.ndarray:
    norms = np.stack(
        [np.linalg.norm(row.delta.astype(np.float64), axis=1) for row in observations]
    )
    scales = np.median(norms, axis=0)
    positive = norms[norms > 0.0]
    fallback = float(np.median(positive)) if positive.size else 1.0
    return np.asarray(np.where(scales > 0.0, scales, fallback), dtype=np.float64)


def _root_balanced_observations(
    observations: Sequence[HonestwardCrossingObservation],
) -> tuple[HonestwardCrossingObservation, ...]:
    grouped: dict[str, list[HonestwardCrossingObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.deceptive_root_id].append(observation)
    result: list[HonestwardCrossingObservation] = []
    for root_id, rows in sorted(grouped.items()):
        identities = {
            (
                row.family,
                row.family_fold,
                row.scenario_id,
                row.true_status,
            )
            for row in rows
        }
        if len(identities) != 1:
            raise RelationalPreStatusFieldEvaluationError(
                "one deceptive root spans incompatible target identities"
            )
        targets: dict[str, list[np.ndarray]] = defaultdict(list)
        for row in rows:
            targets[row.honest_root_id].append(row.delta)
        unique_deltas: list[np.ndarray] = []
        for honest_root_id, values in sorted(targets.items()):
            reference = values[0]
            if any(
                value.shape != reference.shape
                or not np.allclose(value, reference, atol=1e-6, rtol=1e-6)
                for value in values[1:]
            ):
                raise RelationalPreStatusFieldEvaluationError(
                    f"duplicate honest target {honest_root_id} has inconsistent deltas"
                )
            unique_deltas.append(reference)
        family, fold, scenario_id, true_status = next(iter(identities))
        contrast_ids = tuple(sorted({item for row in rows for item in row.contrast_ids}))
        source_pair_ids = tuple(
            sorted({item for row in rows for item in row.source_pair_ids})
        )
        honest_root_ids = tuple(sorted(targets))
        result.append(
            HonestwardCrossingObservation(
                pair_id=f"root-mean:{root_id}",
                deceptive_root_id=root_id,
                honest_root_id=f"target-mean:{root_id}",
                family=family,
                family_fold=fold,
                scenario_id=scenario_id,
                contrast_id=(
                    contrast_ids[0] if len(contrast_ids) == 1 else "MULTI_CONTRAST"
                ),
                true_status=true_status,
                delta=np.asarray(
                    np.mean(np.stack(unique_deltas).astype(np.float64), axis=0),
                    dtype=np.float32,
                ),
                contrast_ids=contrast_ids,
                source_pair_ids=source_pair_ids,
                honest_root_ids=honest_root_ids,
            )
        )
    return tuple(result)


def _contrast_global(
    observations: Sequence[HonestwardCrossingObservation],
    contrast_ids: Sequence[str],
    shape: tuple[int, int],
) -> np.ndarray:
    selected = _root_balanced_observations(
        tuple(
            row
            for row in observations
            if set(row.contrast_ids).intersection(contrast_ids)
        )
    )
    rows = [row.delta for row in selected]
    if not rows:
        return np.zeros(shape, dtype=np.float32)
    return np.asarray(np.mean(np.stack(rows).astype(np.float64), axis=0), dtype=np.float32)


def _fit_optional(
    observations: Sequence[HonestwardCrossingObservation],
    *,
    fold: str,
    training_edges: Mapping[str, Sequence[object]],
) -> SharedPreStatusHonestwardField | None:
    return (
        SharedPreStatusHonestwardField.fit(
            observations,
            held_out_family_fold=fold,
            training_edges=training_edges,
        )
        if observations
        else None
    )


def _metric_row(
    *,
    observation: HonestwardCrossingObservation,
    fold: str,
    model: str,
    prediction: np.ndarray,
    layer_scales: np.ndarray,
    defined: bool,
    support_count: int,
) -> dict[str, Any]:
    metrics = reconstruction_metrics(
        observation.delta,
        prediction,
        layer_scales,
    )
    return {
        "pair_id": observation.pair_id,
        "source_pair_ids": list(observation.source_pair_ids),
        "deceptive_root_id": observation.deceptive_root_id,
        "honest_root_ids": list(observation.honest_root_ids),
        "scenario_id": observation.scenario_id,
        "family": observation.family,
        "fold": fold,
        "contrast_id": observation.contrast_id,
        "contrast_ids": list(observation.contrast_ids),
        "true_status": observation.true_status,
        "model": model,
        "defined": defined,
        "support_count": support_count,
        "metrics": metrics.as_dict(),
    }


def _root_metric_rows(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["deceptive_root_id"])].append(row)
    result: list[dict[str, Any]] = []
    for root_id, root_rows in sorted(grouped.items()):
        identities = {
            (str(row["scenario_id"]), str(row["family"]), str(row["fold"]))
            for row in root_rows
        }
        if len(identities) != 1:
            raise RelationalPreStatusFieldEvaluationError(
                "one deceptive root spans scenario, family, or fold identities"
            )
        values = [row["metrics"]["pooled"][metric] for row in root_rows]
        defined = [float(value) for value in values if value is not None]
        scenario_id, family, fold = next(iter(identities))
        result.append(
            {
                "deceptive_root_id": root_id,
                "scenario_id": scenario_id,
                "family": family,
                "fold": fold,
                metric: _mean(defined),
            }
        )
    return result


def _target_coherence(
    observations: Sequence[HonestwardCrossingObservation],
) -> dict[str, Any]:
    grouped: dict[str, list[HonestwardCrossingObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.deceptive_root_id].append(observation)
    rows: list[dict[str, Any]] = []
    for root_id, crossings in sorted(grouped.items()):
        targets: dict[str, np.ndarray] = {}
        for crossing in crossings:
            prior = targets.get(crossing.honest_root_id)
            if prior is not None and not np.allclose(
                prior, crossing.delta, atol=1e-6, rtol=1e-6
            ):
                raise RelationalPreStatusFieldEvaluationError(
                    "duplicate honest target has inconsistent coherence delta"
                )
            targets[crossing.honest_root_id] = crossing.delta
        target_values = list(targets.values())
        pairwise: list[float] = []
        for left_index, left in enumerate(target_values):
            left_vector = left.astype(np.float64).reshape(-1)
            for right in target_values[left_index + 1 :]:
                right_vector = right.astype(np.float64).reshape(-1)
                denominator = float(
                    np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
                )
                if denominator > 0.0:
                    pairwise.append(float(np.dot(left_vector, right_vector) / denominator))
        rows.append(
            {
                "deceptive_root_id": root_id,
                "source_crossing_count": len(crossings),
                "target_count": len(targets),
                "pairwise_defined_count": len(pairwise),
                "mean_pairwise_cosine": _mean(pairwise),
            }
        )
    repeated = [row for row in rows if row["target_count"] > 1]
    return {
        "deceptive_root_count": len(rows),
        "crossing_count": len(observations),
        "multi_target_root_count": len(repeated),
        "multi_target_mean_pairwise_cosine": _mean(
            [
                float(row["mean_pairwise_cosine"])
                for row in repeated
                if row["mean_pairwise_cosine"] is not None
            ]
        ),
        "roots": rows,
    }


def _lift_stratum(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["model"] == "local_calibrated"]
    return {
        "deceptive_root_count": len(
            {str(row["deceptive_root_id"]) for row in selected}
        ),
        "cosine": aggregate_rows(_root_metric_rows(selected, "cosine"), "cosine"),
        "normalized_squared_error": aggregate_rows(
            _root_metric_rows(selected, "normalized_squared_error"),
            "normalized_squared_error",
        ),
        "defined_rate": _mean([float(row["defined"]) for row in selected]),
    }


def _lift_summary(
    rows: Sequence[Mapping[str, Any]],
    observations: Sequence[HonestwardCrossingObservation],
) -> dict[str, Any]:
    models = sorted({str(row["model"]) for row in rows})
    summary: dict[str, Any] = {
        "crossing_count": len(observations),
        "deceptive_root_count": len(
            {str(row["deceptive_root_id"]) for row in rows}
        ),
        "target_coherence": _target_coherence(observations),
        "models": {},
    }
    for model in models:
        selected = [row for row in rows if row["model"] == model]
        summary["models"][model] = {
            "cosine": aggregate_rows(
                _root_metric_rows(selected, "cosine"), "cosine"
            ),
            "normalized_squared_error": aggregate_rows(
                _root_metric_rows(selected, "normalized_squared_error"),
                "normalized_squared_error",
            ),
            "defined_rate": _mean(
                [
                    max(float(row["defined"]) for row in root_rows)
                    for root_rows in _group_by_root(selected).values()
                ]
            ),
            "mean_support_count": _mean(
                [
                    _mean([float(row["support_count"]) for row in root_rows])
                    for root_rows in _group_by_root(selected).values()
                ]
            ),
        }
    comparisons: dict[str, Any] = {}
    for comparator in (
        "global_mean",
        "shuffled",
        "nearest",
        "random_local_span",
        "contrast_global_oracle",
    ):
        paired: list[dict[str, Any]] = []
        selected = [
            row
            for row in rows
            if row["model"] in {"local_calibrated", comparator}
        ]
        by_key = _group_by_root_and_model(selected)
        for root_id in sorted({key[0] for key in by_key}):
            local_rows = by_key.get((root_id, "local_calibrated"), ())
            control_rows = by_key.get((root_id, comparator), ())
            if not local_rows or not control_rows:
                continue
            local_cosine = _mean(
                [
                    float(row["metrics"]["pooled"]["cosine"])
                    for row in local_rows
                    if row["metrics"]["pooled"]["cosine"] is not None
                ]
            )
            control_cosine = _mean(
                [
                    float(row["metrics"]["pooled"]["cosine"])
                    for row in control_rows
                    if row["metrics"]["pooled"]["cosine"] is not None
                ]
            )
            local_nse = _mean(
                [
                    float(row["metrics"]["pooled"]["normalized_squared_error"])
                    for row in local_rows
                    if row["metrics"]["pooled"]["normalized_squared_error"]
                    is not None
                ]
            )
            control_nse = _mean(
                [
                    float(row["metrics"]["pooled"]["normalized_squared_error"])
                    for row in control_rows
                    if row["metrics"]["pooled"]["normalized_squared_error"]
                    is not None
                ]
            )
            if None in (local_cosine, control_cosine, local_nse, control_nse):
                continue
            assert local_cosine is not None and control_cosine is not None
            assert local_nse is not None and control_nse is not None
            paired.append(
                {
                    "deceptive_root_id": root_id,
                    "scenario_id": local_rows[0]["scenario_id"],
                    "cosine_difference": float(local_cosine - control_cosine),
                    "nse_improvement": float(control_nse - local_nse),
                }
            )
        comparisons[comparator] = {
            "pair_count": len(paired),
            "mean_cosine_difference": _mean(
                [row["cosine_difference"] for row in paired]
            ),
            "mean_nse_improvement": _mean(
                [row["nse_improvement"] for row in paired]
            ),
            "cosine_scenario_cluster_ci": _cluster_ci(
                [
                    {
                        "scenario_id": row["scenario_id"],
                        "difference": row["cosine_difference"],
                    }
                    for row in paired
                ],
                seed=20260722,
            ),
            "nse_scenario_cluster_ci": _cluster_ci(
                [
                    {
                        "scenario_id": row["scenario_id"],
                        "difference": row["nse_improvement"],
                    }
                    for row in paired
                ],
                seed=20260723,
            ),
        }
    summary["local_calibrated_comparisons"] = comparisons
    local_rows = [row for row in rows if row["model"] == "local_calibrated"]
    summary["reuse_strata"] = {
        "by_true_status": {
            truth: _lift_stratum(
                [row for row in local_rows if row["true_status"] == truth]
            )
            for truth in ("PASS", "FAIL")
        },
        "by_contrast": {
            contrast: _lift_stratum(
                [row for row in local_rows if contrast in row["contrast_ids"]]
            )
            for contrast in sorted(
                {contrast for row in local_rows for contrast in row["contrast_ids"]}
            )
        },
    }
    return summary


def _group_by_root(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["deceptive_root_id"])].append(row)
    return {root: tuple(values) for root, values in grouped.items()}


def _group_by_root_and_model(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[tuple[str, str], tuple[Mapping[str, Any], ...]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["deceptive_root_id"]), str(row["model"]))].append(row)
    return {key: tuple(values) for key, values in grouped.items()}


def evaluate_pre_status_honestward_fields(
    supervision: RelationalPreStatusSupervision,
    graphs: GraphInventory,
    *,
    graph_variant: str = "joint",
) -> dict[str, Any]:
    """Score one shared D→H local lift and its frozen controls/reuse tests."""
    report: dict[str, Any] = {
        "views": {},
        "rows": [],
        "pre_score_prediction_inventory": [],
    }
    for view, observations in sorted(
        supervision.honestward_observations_by_view.items()
    ):
        view_rows: list[dict[str, Any]] = []
        for fold in FOLDS:
            raw_training = tuple(
                row for row in observations if row.family_fold != fold
            )
            raw_heldout = tuple(row for row in observations if row.family_fold == fold)
            training = _root_balanced_observations(raw_training)
            heldout = _root_balanced_observations(raw_heldout)
            if not training or not heldout:
                raise RelationalPreStatusFieldEvaluationError(
                    "honestward fold lacks training or held-out crossings"
                )
            graph = _graph(graphs, view, graph_variant, fold)
            field = SharedPreStatusHonestwardField.fit(
                training,
                held_out_family_fold=fold,
                training_edges=graph.training_edges,
            )
            scales = _layer_scales(training)
            contrast_sets = {row.contrast_ids for row in heldout}
            contrast_fields = {
                contrasts: _fit_optional(
                    _root_balanced_observations(
                        tuple(
                            row
                            for row in raw_training
                            if not set(row.contrast_ids).intersection(contrasts)
                        )
                    ),
                    fold=fold,
                    training_edges=graph.training_edges,
                )
                for contrasts in contrast_sets
            }
            truth_fields = {
                truth: _fit_optional(
                    _root_balanced_observations(
                        tuple(
                            row
                            for row in raw_training
                            if row.true_status != truth
                        )
                    ),
                    fold=fold,
                    training_edges=graph.training_edges,
                )
                for truth in {row.true_status for row in heldout}
            }
            primary_predictions = {
                root_id: field.predict(root_id, edges)
                for root_id, edges in sorted(graph.query_edges.items())
            }
            for root_id, prediction in primary_predictions.items():
                report["pre_score_prediction_inventory"].append(
                    {
                        "view": view,
                        "graph_variant": graph_variant,
                        "fold": fold,
                        "root_id": root_id,
                        "defined": prediction.defined,
                        "support_count": prediction.support_count,
                        "support_root_ids": list(prediction.support_root_ids),
                        "array_sha256": {
                            "local": _array_sha256(prediction.local),
                            "local_calibrated": _array_sha256(
                                prediction.dose_calibrated_local
                            ),
                            "global_mean": _array_sha256(prediction.global_mean),
                            "nearest": _array_sha256(prediction.nearest),
                            "shuffled": _array_sha256(prediction.shuffled),
                            "sign_flipped": _array_sha256(
                                prediction.sign_flipped
                            ),
                            "random_local_span": _array_sha256(
                                prediction.random_local_span
                            ),
                        },
                    }
                )
            for observation in heldout:
                prediction = primary_predictions.get(observation.deceptive_root_id)
                if prediction is None:
                    raise RelationalPreStatusFieldEvaluationError(
                        "held-out deceptive root is absent from its graph"
                    )
                edges = graph.query_edges[observation.deceptive_root_id]
                arrays: dict[str, tuple[np.ndarray, bool, int]] = {
                    "zero": (
                        np.zeros_like(observation.delta),
                        True,
                        0,
                    ),
                    "local": (
                        prediction.local,
                        prediction.defined,
                        prediction.support_count,
                    ),
                    "local_calibrated": (
                        prediction.dose_calibrated_local,
                        prediction.defined,
                        prediction.support_count,
                    ),
                    "global_mean": (prediction.global_mean, True, 0),
                    "nearest": (
                        prediction.nearest,
                        prediction.defined,
                        min(prediction.support_count, 1),
                    ),
                    "shuffled": (
                        prediction.shuffled,
                        prediction.defined,
                        prediction.support_count,
                    ),
                    "sign_flipped": (
                        prediction.sign_flipped,
                        prediction.defined,
                        prediction.support_count,
                    ),
                    "random_local_span": (
                        prediction.random_local_span,
                        prediction.defined,
                        prediction.support_count,
                    ),
                    "contrast_global_oracle": (
                        _contrast_global(
                            raw_training,
                            observation.contrast_ids,
                            observation.delta.shape,
                        ),
                        True,
                        0,
                    ),
                }
                restricted = contrast_fields[observation.contrast_ids]
                if restricted is not None:
                    value = restricted.predict(
                        observation.deceptive_root_id,
                        edges,
                    )
                    arrays["leave_contrast_out"] = (
                        value.dose_calibrated_local,
                        value.defined,
                        value.support_count,
                    )
                opposite = truth_fields[observation.true_status]
                if opposite is not None:
                    value = opposite.predict(
                        observation.deceptive_root_id,
                        edges,
                    )
                    arrays["opposite_truth_only"] = (
                        value.dose_calibrated_local,
                        value.defined,
                        value.support_count,
                    )
                for model, (array, defined, support_count) in arrays.items():
                    row = _metric_row(
                        observation=observation,
                        fold=fold,
                        model=model,
                        prediction=array,
                        layer_scales=scales,
                        defined=defined,
                        support_count=support_count,
                    )
                    view_rows.append(row)
                    report["rows"].append({"view": view, **row})
        report["views"][view] = _lift_summary(view_rows, observations)
    return report


__all__ = [
    "GraphInventory",
    "RelationalPreStatusFieldEvaluationError",
    "evaluate_pre_status_honestward_fields",
    "evaluate_pre_status_risk_fields",
]
