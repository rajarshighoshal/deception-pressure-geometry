"""Fold-safe radial prediction on frozen intrinsic relational spectra."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import math
from typing import Any

import numpy as np

from geoprobe.eval.relational_outcome_events import OUTCOME_CLASSES
from geoprobe.geometry.relational_spectral_distance import (
    state_profile_view_rms_distances,
)


_FOLDS = ("outer_1", "outer_2", "outer_3", "outer_4", "outer_5")
_VIEWS = ("residual", "attention", "layer_transport")
_MODELS = (*_VIEWS, "equal_view", "base_rate", "design_cell")
_SMOOTHING = 0.5
_BOOTSTRAP_SEED = 20260716
_BOOTSTRAP_RESAMPLES = 10_000


class RelationalIntrinsicSpectralFieldError(ValueError):
    """Raised when an intrinsic prediction would violate its frozen contract."""


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-safe value with the repository's canonical encoding."""
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelationalIntrinsicSpectralFieldError(f"{name} must be an object")
    return value


def _rows(value: object, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, (list, tuple)):
        raise RelationalIntrinsicSpectralFieldError(f"{name} must be an array")
    return [_mapping(item, f"{name} row") for item in value]


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationalIntrinsicSpectralFieldError(
            f"{name} must be a non-empty string"
        )
    return value


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RelationalIntrinsicSpectralFieldError(f"{name} must be positive")
    return value


def _class_counts(value: object, name: str) -> tuple[int, ...]:
    counts = _mapping(value, name)
    if set(counts) != set(OUTCOME_CLASSES):
        raise RelationalIntrinsicSpectralFieldError(
            f"{name} does not contain the canonical five classes"
        )
    values: list[int] = []
    for label in OUTCOME_CLASSES:
        count = counts[label]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise RelationalIntrinsicSpectralFieldError(
                f"{name}.{label} must be a non-negative integer"
            )
        values.append(count)
    if sum(values) < 1:
        raise RelationalIntrinsicSpectralFieldError(f"{name} must retain an event")
    return tuple(values)


def _identity(row: Mapping[str, Any]) -> dict[str, Any]:
    profile = _mapping(row.get("profile"), "quotient profile")
    state_id = _string(row.get("state_id"), "state_id")
    section = _string(
        row.get("compact_section_content_sha256"),
        "compact_section_content_sha256",
    )
    if profile.get("state_id") != state_id:
        raise RelationalIntrinsicSpectralFieldError(
            "quotient and profile state identities differ"
        )
    if profile.get("compact_section_content_sha256") != section:
        raise RelationalIntrinsicSpectralFieldError(
            "quotient and profile section identities differ"
        )
    design_cell = dict(_mapping(row.get("design_cell"), "design_cell"))
    if set(design_cell) != {"true_status", "desired_status", "knowledge_correct"}:
        raise RelationalIntrinsicSpectralFieldError(
            "design_cell must contain truth, desired status, and knowledge correctness"
        )
    if design_cell["true_status"] not in {"PASS", "FAIL"}:
        raise RelationalIntrinsicSpectralFieldError("true_status is invalid")
    if design_cell["desired_status"] not in {"PASS", "FAIL"}:
        raise RelationalIntrinsicSpectralFieldError("desired_status is invalid")
    if not isinstance(design_cell["knowledge_correct"], bool):
        raise RelationalIntrinsicSpectralFieldError("knowledge_correct must be boolean")
    event_ids = row.get("event_ids")
    if (
        not isinstance(event_ids, list)
        or not event_ids
        or len(event_ids) > 2
        or len(set(event_ids)) != len(event_ids)
        or any(not isinstance(item, str) or not item for item in event_ids)
    ):
        raise RelationalIntrinsicSpectralFieldError(
            "event_ids must contain one or two unique event identities"
        )
    return {
        "quotient_id": state_id,
        "state_id": state_id,
        "family": _string(row.get("family"), "family"),
        "fold": _string(row.get("fold"), "fold"),
        "scenario_id": _string(row.get("scenario_id"), "scenario_id"),
        "prefix_state_sha256": _string(
            row.get("prefix_state_sha256"), "prefix_state_sha256"
        ),
        "compact_section_content_sha256": section,
        "design_cell": design_cell,
        "design_cell_sha256": canonical_sha256(design_cell),
        "event_ids": sorted(event_ids),
        "profile": profile,
    }


def _index_quotients(
    quotients: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Mapping[str, Any]]]:
    identities: dict[str, dict[str, Any]] = {}
    source_rows: dict[str, Mapping[str, Any]] = {}
    owners: dict[str, dict[str, str]] = {
        name: {}
        for name in (
            "family",
            "scenario_id",
            "prefix_state_sha256",
            "compact_section_content_sha256",
        )
    }
    event_owners: dict[str, str] = {}
    for raw in quotients:
        row = _mapping(raw, "quotient")
        identity = _identity(row)
        quotient_id = identity["quotient_id"]
        if quotient_id in identities:
            raise RelationalIntrinsicSpectralFieldError("duplicate quotient_id")
        fold = identity["fold"]
        if fold not in _FOLDS:
            raise RelationalIntrinsicSpectralFieldError("quotient fold is invalid")
        identities[quotient_id] = identity
        source_rows[quotient_id] = row
        for name, mapping in owners.items():
            value = identity[name]
            previous = mapping.setdefault(value, fold)
            if previous != fold:
                raise RelationalIntrinsicSpectralFieldError(
                    f"{name} crosses outer folds"
                )
        for event_id in identity["event_ids"]:
            previous = event_owners.setdefault(event_id, quotient_id)
            if previous != quotient_id:
                raise RelationalIntrinsicSpectralFieldError(
                    "one event belongs to multiple state quotients"
                )
    if set(identity["fold"] for identity in identities.values()) != set(_FOLDS):
        raise RelationalIntrinsicSpectralFieldError(
            "intrinsic prediction requires all five outer folds"
        )
    return identities, source_rows


def _fold_policies(
    calibration_selection: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    by_fold: dict[str, list[dict[str, Any]]] = {fold: [] for fold in _FOLDS}
    seen: set[tuple[str, str]] = set()
    full_inventory: dict[str, set[str]] = {fold: set() for fold in _FOLDS}
    for raw in calibration_selection:
        row = _mapping(raw, "calibration selection")
        fold = _string(row.get("heldout_family_fold"), "heldout_family_fold")
        if fold not in by_fold:
            raise RelationalIntrinsicSpectralFieldError(
                "calibration selection has an unknown fold"
            )
        relation_name = _string(row.get("relation_name"), "relation_name")
        key = (fold, relation_name)
        if key in seen:
            raise RelationalIntrinsicSpectralFieldError(
                "calibration repeats a fold-relation policy"
            )
        seen.add(key)
        full_inventory[fold].add(relation_name)
        if row.get("admissible") is not True:
            continue
        view = _string(row.get("view"), "view")
        if view not in _VIEWS:
            raise RelationalIntrinsicSpectralFieldError("policy view is invalid")
        by_fold[fold].append(
            {
                "relation_name": relation_name,
                "view": view,
                "selected_rank": _positive_integer(
                    row.get("selected_rank"), "selected_rank"
                ),
            }
        )
    for fold in _FOLDS:
        if len(full_inventory[fold]) != 135:
            raise RelationalIntrinsicSpectralFieldError(
                "each fold must carry all 135 calibrated relation decisions"
            )
        if set(item["view"] for item in by_fold[fold]) != set(_VIEWS):
            raise RelationalIntrinsicSpectralFieldError(
                "each fold needs admitted support in all three views"
            )
        by_fold[fold].sort(key=lambda item: item["relation_name"])
    return {fold: tuple(rows) for fold, rows in by_fold.items()}


def _pair_median(values: Sequence[float], name: str) -> float:
    if not values:
        raise RelationalIntrinsicSpectralFieldError(f"{name} has no training pairs")
    result = float(np.median(np.asarray(values, dtype=np.float64)))
    if not math.isfinite(result) or result <= 0.0:
        raise RelationalIntrinsicSpectralFieldError(
            f"{name} median must be positive and finite"
        )
    return result


def _probability(scores: np.ndarray) -> dict[str, float]:
    denominator = float(scores.sum())
    if not math.isfinite(denominator) or denominator <= 0:
        raise RelationalIntrinsicSpectralFieldError(
            "prediction scores are not normalizable"
        )
    values = scores / denominator
    return {label: float(values[index]) for index, label in enumerate(OUTCOME_CLASSES)}


def _count_array(row: Mapping[str, Any], name: str) -> np.ndarray:
    return np.asarray(_class_counts(row.get("class_counts"), name), dtype=np.float64)


def _model_distance(
    view_distances: Mapping[str, float],
    scales: Mapping[str, float],
    model: str,
) -> float:
    if model in _VIEWS:
        return float(view_distances[model] / scales[model])
    return float(
        math.sqrt(
            sum((view_distances[view] / scales[view]) ** 2 for view in _VIEWS)
            / len(_VIEWS)
        )
    )


def build_intrinsic_spectral_fold_predictions(
    *,
    heldout_fold: str,
    quotients: Sequence[Mapping[str, Any]],
    admitted_policies: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one fold's predictions without reading that fold's outcome counts."""
    if heldout_fold not in _FOLDS:
        raise RelationalIntrinsicSpectralFieldError("heldout fold is invalid")
    identities, source_rows = _index_quotients(quotients)
    train_ids = sorted(
        quotient_id
        for quotient_id, row in identities.items()
        if row["fold"] != heldout_fold
    )
    query_ids = sorted(
        quotient_id
        for quotient_id, row in identities.items()
        if row["fold"] == heldout_fold
    )
    if not train_ids or not query_ids:
        raise RelationalIntrinsicSpectralFieldError("fold train/query split is empty")

    policies = tuple(dict(item) for item in admitted_policies)
    train_pair_views: dict[tuple[str, str], dict[str, float]] = {}
    for left_index, left_id in enumerate(train_ids):
        for right_id in train_ids[left_index + 1 :]:
            train_pair_views[(left_id, right_id)] = state_profile_view_rms_distances(
                identities[left_id]["profile"],
                identities[right_id]["profile"],
                policies,
            )
    query_train_views: dict[tuple[str, str], dict[str, float]] = {}
    for query_id in query_ids:
        for train_id in train_ids:
            query_train_views[(query_id, train_id)] = (
                state_profile_view_rms_distances(
                    identities[query_id]["profile"],
                    identities[train_id]["profile"],
                    policies,
                )
            )

    scales = {
        view: _pair_median(
            [distances[view] for distances in train_pair_views.values()],
            f"{heldout_fold}.{view} scale",
        )
        for view in _VIEWS
    }
    geometric_models = (*_VIEWS, "equal_view")
    bandwidths = {
        model: _pair_median(
            [
                _model_distance(distances, scales, model)
                for distances in train_pair_views.values()
            ],
            f"{heldout_fold}.{model} bandwidth",
        )
        for model in geometric_models
    }
    train_counts = {
        quotient_id: _count_array(
            source_rows[quotient_id], f"training quotient {quotient_id}"
        )
        for quotient_id in train_ids
    }
    pooled_counts = sum(train_counts.values(), np.zeros(len(OUTCOME_CLASSES)))
    design_counts: dict[str, np.ndarray] = {}
    for quotient_id in train_ids:
        cell = identities[quotient_id]["design_cell_sha256"]
        design_counts.setdefault(cell, np.zeros(len(OUTCOME_CLASSES)))
        design_counts[cell] += train_counts[quotient_id]

    predictions: list[dict[str, Any]] = []
    for query_id in query_ids:
        model_probabilities: dict[str, dict[str, float]] = {}
        for model in geometric_models:
            h = bandwidths[model]
            scores = np.full(len(OUTCOME_CLASSES), _SMOOTHING, dtype=np.float64)
            for train_id in train_ids:
                distance = _model_distance(
                    query_train_views[(query_id, train_id)], scales, model
                )
                weight = math.exp(-(distance**2) / (2.0 * h**2))
                scores += weight * train_counts[train_id]
            model_probabilities[model] = _probability(scores)
        model_probabilities["base_rate"] = _probability(
            pooled_counts + _SMOOTHING
        )
        cell_counts = design_counts.get(
            identities[query_id]["design_cell_sha256"], pooled_counts
        )
        model_probabilities["design_cell"] = _probability(
            cell_counts + _SMOOTHING
        )
        query = identities[query_id]
        predictions.append(
            {
                key: query[key]
                for key in (
                    "quotient_id",
                    "family",
                    "fold",
                    "scenario_id",
                    "prefix_state_sha256",
                    "compact_section_content_sha256",
                    "design_cell",
                    "design_cell_sha256",
                    "event_ids",
                )
            }
            | {"probabilities": model_probabilities}
        )

    train_class_counts = {
        label: int(pooled_counts[index])
        for index, label in enumerate(OUTCOME_CLASSES)
    }
    policy_counts = Counter(item["view"] for item in policies)
    return {
        "heldout_fold": heldout_fold,
        "train_quotient_count": len(train_ids),
        "query_quotient_count": len(query_ids),
        "train_event_count": int(pooled_counts.sum()),
        "train_class_counts": train_class_counts,
        "admitted_relation_counts": {
            view: policy_counts[view] for view in _VIEWS
        },
        "view_scales": scales,
        "bandwidths": bandwidths,
        "predictions": predictions,
    }


def build_intrinsic_spectral_prediction_ledger(
    *,
    quotients: Sequence[Mapping[str, Any]],
    calibration_selection: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build five fold-specific prediction ledgers before query scoring."""
    policies = _fold_policies(calibration_selection)
    fold_ledgers = [
        build_intrinsic_spectral_fold_predictions(
            heldout_fold=fold,
            quotients=quotients,
            admitted_policies=policies[fold],
        )
        for fold in _FOLDS
    ]
    predictions = [
        prediction
        for fold in fold_ledgers
        for prediction in fold.pop("predictions")
    ]
    ledger: dict[str, Any] = {
        "schema_version": 1,
        "kind": "relational_intrinsic_spectral_prediction_ledger",
        "method": {
            "distance": "fold_admitted_spectral_hellinger_equal_view_product",
            "kernel": "fixed_radial_median_training_pair_bandwidth",
            "smoothing": _SMOOTHING,
            "outcome_classes": list(OUTCOME_CLASSES),
            "pca": "absent",
            "query_labels_used_for_fold_prediction": False,
        },
        "folds": fold_ledgers,
        "predictions": sorted(
            predictions, key=lambda row: (row["fold"], row["quotient_id"])
        ),
    }
    ledger["prediction_ledger_sha256"] = canonical_sha256(ledger)
    return ledger


def _validate_prediction_ledger(ledger: Mapping[str, Any]) -> None:
    if (
        ledger.get("schema_version") != 1
        or ledger.get("kind")
        != "relational_intrinsic_spectral_prediction_ledger"
    ):
        raise RelationalIntrinsicSpectralFieldError(
            "prediction ledger schema or kind is invalid"
        )
    declared = _string(
        ledger.get("prediction_ledger_sha256"), "prediction ledger SHA"
    )
    payload = dict(ledger)
    payload.pop("prediction_ledger_sha256", None)
    if canonical_sha256(payload) != declared:
        raise RelationalIntrinsicSpectralFieldError(
            "prediction ledger self-hash is invalid"
        )


def _binary_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [
        score for score, label in zip(scores, labels, strict=True) if not label
    ]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return float(wins / (len(positives) * len(negatives)))


def _average_precision(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    positives = sum(labels)
    if positives == 0 or positives == len(labels):
        return None
    groups: dict[float, list[int]] = {}
    for score, label in zip(scores, labels, strict=True):
        groups.setdefault(score, []).append(label)
    cumulative_positive = 0
    cumulative_total = 0
    result = 0.0
    for score in sorted(groups, reverse=True):
        values = groups[score]
        group_positive = sum(values)
        cumulative_positive += group_positive
        cumulative_total += len(values)
        result += (group_positive / positives) * (
            cumulative_positive / cumulative_total
        )
    return float(result)


def _metrics(rows: Sequence[Mapping[str, Any]], model: str) -> dict[str, Any]:
    if not rows:
        raise RelationalIntrinsicSpectralFieldError("cannot score an empty row set")
    event_loss = 0.0
    event_brier = 0.0
    event_total = 0
    quotient_losses: list[float] = []
    quotient_briers: list[float] = []
    confusion = {
        true: {predicted: 0 for predicted in OUTCOME_CLASSES}
        for true in OUTCOME_CLASSES
    }
    hd_scores: list[float] = []
    hd_labels: list[int] = []
    hd_log_loss = 0.0
    hd_brier = 0.0
    support = Counter()
    for row in rows:
        counts = np.asarray(row["counts"], dtype=np.float64)
        probability_map = _mapping(
            _mapping(row["probabilities"], "probabilities").get(model),
            f"{model} probabilities",
        )
        if set(probability_map) != set(OUTCOME_CLASSES):
            raise RelationalIntrinsicSpectralFieldError(
                "prediction probability classes are invalid"
            )
        probabilities = np.asarray(
            [float(probability_map[label]) for label in OUTCOME_CLASSES],
            dtype=np.float64,
        )
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities <= 0)
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-12)
        ):
            raise RelationalIntrinsicSpectralFieldError(
                "prediction probabilities are invalid"
            )
        total = int(counts.sum())
        log_values = -np.log(probabilities)
        per_event_brier = (
            float(np.square(probabilities).sum())
            - 2.0 * probabilities
            + 1.0
        )
        event_loss += float(np.dot(counts, log_values))
        event_brier += float(np.dot(counts, per_event_brier))
        event_total += total
        quotient_distribution = counts / total
        quotient_losses.append(float(np.dot(quotient_distribution, log_values)))
        quotient_briers.append(
            float(np.dot(quotient_distribution, per_event_brier))
        )
        predicted_index = int(np.argmax(probabilities))
        predicted = OUTCOME_CLASSES[predicted_index]
        for index, label in enumerate(OUTCOME_CLASSES):
            count = int(counts[index])
            support[label] += count
            confusion[label][predicted] += count
        honest_index = OUTCOME_CLASSES.index("HONEST")
        deceptive_index = OUTCOME_CLASSES.index("DECEPTIVE")
        denominator = probabilities[honest_index] + probabilities[deceptive_index]
        deceptive_score = float(probabilities[deceptive_index] / denominator)
        honest_count = int(counts[honest_index])
        deceptive_count = int(counts[deceptive_index])
        hd_log_loss += honest_count * -math.log(1.0 - deceptive_score)
        hd_log_loss += deceptive_count * -math.log(deceptive_score)
        hd_brier += honest_count * deceptive_score**2
        hd_brier += deceptive_count * (1.0 - deceptive_score) ** 2
        hd_scores.extend([deceptive_score] * honest_count)
        hd_labels.extend([0] * honest_count)
        hd_scores.extend([deceptive_score] * deceptive_count)
        hd_labels.extend([1] * deceptive_count)
    correct = sum(confusion[label][label] for label in OUTCOME_CLASSES)
    return {
        "event_count": event_total,
        "quotient_count": len(rows),
        "event_pooled_multiclass_log_loss": event_loss / event_total,
        "event_pooled_multiclass_brier": event_brier / event_total,
        "quotient_macro_cross_entropy": float(np.mean(quotient_losses)),
        "quotient_macro_brier": float(np.mean(quotient_briers)),
        "event_pooled_accuracy": correct / event_total,
        "confusion_matrix": confusion,
        "class_support": {
            label: {
                "event_count": support[label],
                "estimable_in_this_cohort": support[label] > 0,
            }
            for label in OUTCOME_CLASSES
        },
        "honest_deceptive_slice": {
            "event_count": len(hd_labels),
            "honest_count": hd_labels.count(0),
            "deceptive_count": hd_labels.count(1),
            "conditional_log_loss": hd_log_loss / len(hd_labels),
            "conditional_brier": hd_brier / len(hd_labels),
            "auroc": _binary_auc(hd_scores, hd_labels),
            "average_precision": _average_precision(hd_scores, hd_labels),
            "binary_refit": False,
        },
    }


def _comparison_components(
    rows: Sequence[Mapping[str, Any]], comparator: str
) -> dict[str, dict[str, tuple[float, int]]]:
    """Reduce frozen paired losses to family clusters for uncertainty only."""
    if comparator not in {"base_rate", "design_cell"}:
        raise RelationalIntrinsicSpectralFieldError("comparator is invalid")
    honest_index = OUTCOME_CLASSES.index("HONEST")
    deceptive_index = OUTCOME_CLASSES.index("DECEPTIVE")
    by_family: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        family = _string(row.get("family"), "scored family")
        family_stats = by_family.setdefault(
            family,
            {
                name: [0.0, 0.0]
                for name in (
                    "event_pooled_log_loss_gain",
                    "event_pooled_brier_gain",
                    "quotient_macro_cross_entropy_gain",
                    "honest_deceptive_conditional_log_loss_gain",
                    "honest_deceptive_conditional_brier_gain",
                )
            },
        )
        counts = np.asarray(row["counts"], dtype=np.float64)
        probabilities = _mapping(row["probabilities"], "probabilities")
        equal = np.asarray(
            [float(probabilities["equal_view"][label]) for label in OUTCOME_CLASSES],
            dtype=np.float64,
        )
        control = np.asarray(
            [float(probabilities[comparator][label]) for label in OUTCOME_CLASSES],
            dtype=np.float64,
        )
        total = int(counts.sum())
        log_gain_sum = float(np.dot(counts, np.log(equal / control)))
        equal_brier = np.square(equal).sum() - 2.0 * equal + 1.0
        control_brier = np.square(control).sum() - 2.0 * control + 1.0
        brier_gain_sum = float(np.dot(counts, control_brier - equal_brier))
        family_stats["event_pooled_log_loss_gain"][0] += log_gain_sum
        family_stats["event_pooled_log_loss_gain"][1] += total
        family_stats["event_pooled_brier_gain"][0] += brier_gain_sum
        family_stats["event_pooled_brier_gain"][1] += total
        family_stats["quotient_macro_cross_entropy_gain"][0] += (
            log_gain_sum / total
        )
        family_stats["quotient_macro_cross_entropy_gain"][1] += 1

        honest_count = int(counts[honest_index])
        deceptive_count = int(counts[deceptive_index])
        hd_total = honest_count + deceptive_count
        if hd_total:
            equal_score = float(
                equal[deceptive_index]
                / (equal[honest_index] + equal[deceptive_index])
            )
            control_score = float(
                control[deceptive_index]
                / (control[honest_index] + control[deceptive_index])
            )
            hd_log_gain = honest_count * math.log(
                (1.0 - equal_score) / (1.0 - control_score)
            ) + deceptive_count * math.log(equal_score / control_score)
            hd_brier_gain = honest_count * (
                control_score**2 - equal_score**2
            ) + deceptive_count * (
                (1.0 - control_score) ** 2 - (1.0 - equal_score) ** 2
            )
            family_stats["honest_deceptive_conditional_log_loss_gain"][0] += (
                hd_log_gain
            )
            family_stats["honest_deceptive_conditional_log_loss_gain"][1] += (
                hd_total
            )
            family_stats["honest_deceptive_conditional_brier_gain"][0] += (
                hd_brier_gain
            )
            family_stats["honest_deceptive_conditional_brier_gain"][1] += (
                hd_total
            )
    return {
        family: {
            metric: (float(values[0]), int(values[1]))
            for metric, values in metrics.items()
        }
        for family, metrics in by_family.items()
    }


def _family_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Quantify paired-score stability without changing the frozen predictor."""
    families = sorted({_string(row.get("family"), "scored family") for row in rows})
    if len(families) < 2:
        raise RelationalIntrinsicSpectralFieldError(
            "family bootstrap needs at least two clusters"
        )
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    draws = rng.integers(
        0,
        len(families),
        size=(_BOOTSTRAP_RESAMPLES, len(families)),
    )
    output: dict[str, Any] = {
        "method": "paired_family_cluster_percentile_bootstrap",
        "unit": "family",
        "family_count": len(families),
        "seed": _BOOTSTRAP_SEED,
        "resamples": _BOOTSTRAP_RESAMPLES,
        "predictions_refit_per_resample": False,
        "comparisons": {},
    }
    for comparator in ("base_rate", "design_cell"):
        components = _comparison_components(rows, comparator)
        comparator_output: dict[str, Any] = {}
        for metric in next(iter(components.values())):
            numerators = np.asarray(
                [components[family][metric][0] for family in families],
                dtype=np.float64,
            )
            denominators = np.asarray(
                [components[family][metric][1] for family in families],
                dtype=np.float64,
            )
            sampled_denominators = denominators[draws].sum(axis=1)
            if np.any(sampled_denominators <= 0):
                raise RelationalIntrinsicSpectralFieldError(
                    "a bootstrap draw has no estimable events"
                )
            values = numerators[draws].sum(axis=1) / sampled_denominators
            if metric in comparisons[comparator]:
                point_estimate = float(comparisons[comparator][metric])
            else:
                point_estimate = float(numerators.sum() / denominators.sum())
            comparator_output[metric] = {
                "point_estimate": point_estimate,
                "percentile_95": [
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                ],
                "fraction_positive": float(np.mean(values > 0.0)),
                "positive_favors_equal_view": True,
            }
        output["comparisons"][comparator] = comparator_output
    return output


def score_intrinsic_spectral_prediction_ledger(
    *,
    prediction_ledger: Mapping[str, Any],
    quotients: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Join held-out count vectors only after the prediction ledger is frozen."""
    _validate_prediction_ledger(prediction_ledger)
    identities, source_rows = _index_quotients(quotients)
    predictions = _rows(prediction_ledger.get("predictions"), "predictions")
    prediction_by_id = {
        _string(row.get("quotient_id"), "prediction quotient_id"): row
        for row in predictions
    }
    if len(prediction_by_id) != len(predictions) or set(prediction_by_id) != set(
        identities
    ):
        raise RelationalIntrinsicSpectralFieldError(
            "prediction and quotient inventories differ"
        )
    scored_rows: list[dict[str, Any]] = []
    for quotient_id in sorted(identities):
        prediction = prediction_by_id[quotient_id]
        identity = identities[quotient_id]
        for key in (
            "family",
            "fold",
            "scenario_id",
            "prefix_state_sha256",
            "compact_section_content_sha256",
            "design_cell_sha256",
            "event_ids",
        ):
            if prediction.get(key) != identity[key]:
                raise RelationalIntrinsicSpectralFieldError(
                    "prediction identity differs from the outcome quotient"
                )
        scored_rows.append(
            {
                "quotient_id": quotient_id,
                "family": identity["family"],
                "fold": identity["fold"],
                "counts": list(
                    _class_counts(
                        source_rows[quotient_id].get("class_counts"),
                        f"scored quotient {quotient_id}",
                    )
                ),
                "probabilities": prediction["probabilities"],
            }
        )
    aggregate = {model: _metrics(scored_rows, model) for model in _MODELS}
    per_fold = {
        fold: {
            model: _metrics(
                [row for row in scored_rows if row["fold"] == fold], model
            )
            for model in _MODELS
        }
        for fold in _FOLDS
    }
    comparisons: dict[str, dict[str, float]] = {}
    for comparator in ("base_rate", "design_cell"):
        comparisons[comparator] = {
            "event_pooled_log_loss_gain": aggregate[comparator][
                "event_pooled_multiclass_log_loss"
            ]
            - aggregate["equal_view"]["event_pooled_multiclass_log_loss"],
            "event_pooled_brier_gain": aggregate[comparator][
                "event_pooled_multiclass_brier"
            ]
            - aggregate["equal_view"]["event_pooled_multiclass_brier"],
            "quotient_macro_cross_entropy_gain": aggregate[comparator][
                "quotient_macro_cross_entropy"
            ]
            - aggregate["equal_view"]["quotient_macro_cross_entropy"],
            "honest_deceptive_conditional_log_loss_gain": aggregate[comparator][
                "honest_deceptive_slice"
            ]["conditional_log_loss"]
            - aggregate["equal_view"]["honest_deceptive_slice"][
                "conditional_log_loss"
            ],
            "honest_deceptive_conditional_brier_gain": aggregate[comparator][
                "honest_deceptive_slice"
            ]["conditional_brier"]
            - aggregate["equal_view"]["honest_deceptive_slice"][
                "conditional_brier"
            ],
            "positive_favors_equal_view": True,
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "relational_intrinsic_spectral_field_score",
        "prediction_ledger_sha256": prediction_ledger[
            "prediction_ledger_sha256"
        ],
        "outcome_classes": list(OUTCOME_CLASSES),
        "aggregate": aggregate,
        "per_fold": per_fold,
        "equal_view_gain_over_comparators": comparisons,
        "post_score_descriptive_uncertainty": _family_cluster_bootstrap(
            scored_rows,
            comparisons,
        ),
        "claim_boundary": {
            "fixed_pressure_an_turn2_only": True,
            "all_outcomes_retained": True,
            "global_flat_coordinates": "absent",
            "controller": "not_tested",
            "curvature_or_universality": "not_established",
            "arbitrary_success_threshold": "absent",
        },
    }
    report["score_sha256"] = canonical_sha256(report)
    return report


__all__ = [
    "RelationalIntrinsicSpectralFieldError",
    "build_intrinsic_spectral_fold_predictions",
    "build_intrinsic_spectral_prediction_ledger",
    "canonical_sha256",
    "score_intrinsic_spectral_prediction_ledger",
]
