"""Fold-safe negative controls and descriptive diagnostics for risk fields.

These helpers intentionally operate on a training fold only.  They make no
held-out prediction claim; callers must fit a fresh risk field with the
returned training events before evaluating it on the untouched held-out fold.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from geoprobe.eval.relational_outcome_events import OUTCOME_CLASSES
from geoprobe.eval.relational_pre_status_risk_field import (
    PreStatusRiskEvent,
    multiclass_brier,
    multiclass_log_loss,
)
from geoprobe.geometry.relational_pre_status_rooted_graph import FoldExactRootedGraph


class RelationalPreStatusRiskDiagnosticsError(ValueError):
    """A risk diagnostic violates its fold or probability contract."""


@dataclass(frozen=True, slots=True)
class TrainFoldOutcomeLabelShuffle:
    """A deterministic event-label permutation for one training fold."""

    held_out_family_fold: str
    seed: int
    events: tuple[PreStatusRiskEvent, ...]
    outcome_by_event_id: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class TrainFoldRootIdentityShuffle:
    """A deterministic root-group permutation for one training fold."""

    held_out_family_fold: str
    seed: int
    events: tuple[PreStatusRiskEvent, ...]
    root_identity_map: Mapping[str, str]


def _training_events(
    events: Sequence[PreStatusRiskEvent], *, held_out_family_fold: str
) -> tuple[PreStatusRiskEvent, ...]:
    rows = tuple(sorted(events, key=lambda event: event.event_id))
    if not rows:
        raise RelationalPreStatusRiskDiagnosticsError("training events are empty")
    if len({event.event_id for event in rows}) != len(rows):
        raise RelationalPreStatusRiskDiagnosticsError("training event IDs are not unique")
    if any(event.family_fold == held_out_family_fold for event in rows):
        raise RelationalPreStatusRiskDiagnosticsError(
            "held-out event entered a train-fold diagnostic"
        )
    return rows


def _generator(seed: int) -> np.random.Generator:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RelationalPreStatusRiskDiagnosticsError("shuffle seed is invalid")
    return np.random.default_rng(seed)


def _nonidentity_permutation(size: int, generator: np.random.Generator) -> np.ndarray:
    permutation = np.asarray(generator.permutation(size), dtype=int)
    if size > 1 and np.array_equal(permutation, np.arange(size)):
        permutation = np.roll(permutation, 1)
    return permutation


def shuffle_train_fold_outcome_labels(
    events: Sequence[PreStatusRiskEvent],
    *,
    held_out_family_fold: str,
    seed: int,
) -> TrainFoldOutcomeLabelShuffle:
    """Permute complete five-class labels across training events only.

    The label multiset is exactly preserved.  The input must already exclude
    the held-out family fold, making accidental outcome leakage fail loudly.
    """
    rows = _training_events(events, held_out_family_fold=held_out_family_fold)
    permutation = _nonidentity_permutation(len(rows), _generator(seed))
    shuffled = tuple(
        replace(event, outcome_class=rows[int(source_index)].outcome_class)
        for event, source_index in zip(rows, permutation, strict=True)
    )
    return TrainFoldOutcomeLabelShuffle(
        held_out_family_fold=held_out_family_fold,
        seed=seed,
        events=shuffled,
        outcome_by_event_id=MappingProxyType(
            {event.event_id: event.outcome_class for event in shuffled}
        ),
    )


def shuffle_train_fold_root_identities(
    events: Sequence[PreStatusRiskEvent],
    graph: FoldExactRootedGraph,
    *,
    held_out_family_fold: str,
    seed: int,
) -> TrainFoldRootIdentityShuffle:
    """Permute whole training root identities while retaining each root's labels.

    Whole-root permutation preserves repeated-event outcome mixtures, while
    breaking their assignment to the graph's rooted neighbourhood identities.
    """
    rows = _training_events(events, held_out_family_fold=held_out_family_fold)
    if graph.held_out_family_fold != held_out_family_fold:
        raise RelationalPreStatusRiskDiagnosticsError(
            "graph held-out fold disagrees with root-identity shuffle"
        )
    roots = tuple(sorted({event.root_id for event in rows}))
    graph_roots = set(graph.training_edges)
    if not set(roots).issubset(graph_roots):
        raise RelationalPreStatusRiskDiagnosticsError(
            "training event root is absent from graph training identities"
        )
    permutation = _nonidentity_permutation(len(roots), _generator(seed))
    root_identity_map = MappingProxyType(
        {
            root: roots[int(target_index)]
            for root, target_index in zip(roots, permutation, strict=True)
        }
    )
    return TrainFoldRootIdentityShuffle(
        held_out_family_fold=held_out_family_fold,
        seed=seed,
        events=tuple(
            replace(event, root_id=root_identity_map[event.root_id]) for event in rows
        ),
        root_identity_map=root_identity_map,
    )


def _probability_vector(probabilities: Mapping[str, float]) -> np.ndarray:
    if set(probabilities) != set(OUTCOME_CLASSES):
        raise RelationalPreStatusRiskDiagnosticsError(
            "probabilities must contain every outcome class"
        )
    values = np.asarray([probabilities[label] for label in OUTCOME_CLASSES], dtype=float)
    if (
        not np.isfinite(values).all()
        or np.any(values < 0.0)
        or not np.isclose(values.sum(), 1.0)
    ):
        raise RelationalPreStatusRiskDiagnosticsError("probability vector is invalid")
    return values


def multiclass_calibration_summary(
    outcomes: Sequence[str],
    probabilities: Sequence[Mapping[str, float]],
    *,
    bin_count: int = 10,
) -> dict[str, Any]:
    """Return fixed-bin one-vs-rest calibration for every outcome class."""
    labels = tuple(outcomes)
    vectors = tuple(probabilities)
    if not labels or len(labels) != len(vectors):
        raise RelationalPreStatusRiskDiagnosticsError(
            "outcomes and probability vectors must be non-empty and aligned"
        )
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count < 2:
        raise RelationalPreStatusRiskDiagnosticsError("calibration bin count is invalid")
    if any(label not in OUTCOME_CLASSES for label in labels):
        raise RelationalPreStatusRiskDiagnosticsError("outcome class is invalid")
    matrix = np.stack([_probability_vector(vector) for vector in vectors])
    classes: dict[str, dict[str, Any]] = {}
    for class_index, outcome_class in enumerate(OUTCOME_CLASSES):
        predicted = matrix[:, class_index]
        observed = np.asarray(
            [float(label == outcome_class) for label in labels], dtype=float
        )
        bins: list[dict[str, Any]] = []
        ece = 0.0
        for index in range(bin_count):
            lower = index / bin_count
            upper = (index + 1) / bin_count
            mask = (
                (predicted >= lower) & (predicted <= upper)
                if index == bin_count - 1
                else (predicted >= lower) & (predicted < upper)
            )
            count = int(mask.sum())
            mean_predicted = float(predicted[mask].mean()) if count else None
            observed_rate = float(observed[mask].mean()) if count else None
            if count:
                assert mean_predicted is not None and observed_rate is not None
                ece += (count / len(labels)) * abs(observed_rate - mean_predicted)
            bins.append(
                {
                    "bin_index": index,
                    "lower": lower,
                    "upper": upper,
                    "count": count,
                    "mean_predicted_probability": mean_predicted,
                    "observed_frequency": observed_rate,
                }
            )
        classes[outcome_class] = {"ece": float(ece), "bins": bins}
    return {
        "event_count": len(labels),
        "bin_count": bin_count,
        "classes": classes,
        "macro_ece": float(np.mean([entry["ece"] for entry in classes.values()])),
    }


def descriptive_train_fold_mixed_root_proper_score_floor(
    events: Sequence[PreStatusRiskEvent], *, held_out_family_fold: str
) -> dict[str, Any]:
    """Describe empirical same-root ambiguity in one training fold.

    This fits and scores each root's empirical outcome distribution on those
    very training events.  It is therefore descriptive, not a held-out model,
    and must not be reported as predictive performance or a generalization
    lower bound.
    """
    rows = _training_events(events, held_out_family_fold=held_out_family_fold)
    by_root: dict[str, list[PreStatusRiskEvent]] = defaultdict(list)
    for event in rows:
        by_root[event.root_id].append(event)
    probabilities_by_root: dict[str, Mapping[str, float]] = {}
    mixed_roots: list[str] = []
    for root_id, root_rows in sorted(by_root.items()):
        counts = Counter(event.outcome_class for event in root_rows)
        if len(counts) > 1:
            mixed_roots.append(root_id)
        probabilities_by_root[root_id] = MappingProxyType(
            {
                outcome_class: counts[outcome_class] / len(root_rows)
                for outcome_class in OUTCOME_CLASSES
            }
        )
    log_losses = [
        multiclass_log_loss(event.outcome_class, probabilities_by_root[event.root_id])
        for event in rows
    ]
    briers = [
        multiclass_brier(event.outcome_class, probabilities_by_root[event.root_id])
        for event in rows
    ]
    mixed_root_event_count = sum(len(by_root[root_id]) for root_id in mixed_roots)
    return {
        "label": "descriptive_train_fold_mixed_root_irreducible_proper_score_floor",
        "held_out_family_fold": held_out_family_fold,
        "held_out_predictor": False,
        "interpretation": (
            "Training-fold empirical same-root outcome ambiguity only; fitted and "
            "scored on the same labels, so it is not held-out predictive performance "
            "or a generalization lower bound."
        ),
        "event_count": len(rows),
        "root_count": len(by_root),
        "mixed_root_count": len(mixed_roots),
        "mixed_root_event_count": mixed_root_event_count,
        "outcome_counts": {
            outcome_class: sum(event.outcome_class == outcome_class for event in rows)
            for outcome_class in OUTCOME_CLASSES
        },
        "mean_empirical_root_log_loss": float(np.mean(log_losses)),
        "mean_empirical_root_brier": float(np.mean(briers)),
    }


__all__ = [
    "RelationalPreStatusRiskDiagnosticsError",
    "TrainFoldOutcomeLabelShuffle",
    "TrainFoldRootIdentityShuffle",
    "descriptive_train_fold_mixed_root_proper_score_floor",
    "multiclass_calibration_summary",
    "shuffle_train_fold_outcome_labels",
    "shuffle_train_fold_root_identities",
]
