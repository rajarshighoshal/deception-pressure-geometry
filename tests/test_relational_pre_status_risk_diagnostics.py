from __future__ import annotations

from collections import Counter

import pytest

from geoprobe.eval.relational_outcome_events import OUTCOME_CLASSES
from geoprobe.eval.relational_pre_status_risk_diagnostics import (
    RelationalPreStatusRiskDiagnosticsError,
    descriptive_train_fold_mixed_root_proper_score_floor,
    multiclass_calibration_summary,
    shuffle_train_fold_outcome_labels,
    shuffle_train_fold_root_identities,
)
from geoprobe.eval.relational_pre_status_risk_field import PreStatusRiskEvent
from geoprobe.geometry.relational_pre_status_rooted_graph import FoldExactRootedGraph
from geoprobe.geometry.relational_pre_status_rooted_metric import RootedStarMetricScaler


def _event(event_id: str, root_id: str, outcome_class: str) -> PreStatusRiskEvent:
    return PreStatusRiskEvent(
        event_id=event_id,
        root_id=root_id,
        family="train-family",
        family_fold="outer_1",
        outcome_class=outcome_class,
        nuisance_key=("turn", "pressure"),
    )


def _graph(*roots: str) -> FoldExactRootedGraph:
    return FoldExactRootedGraph(
        held_out_family_fold="outer_5",
        graph_width=1,
        query_edges={},
        training_edges={root: () for root in roots},
        scaler=RootedStarMetricScaler(1.0, 1.0),
        candidate_pair_count=0,
        exact_pair_count=0,
    )


def _events() -> tuple[PreStatusRiskEvent, ...]:
    return (
        _event("event-1", "root-a", "HONEST"),
        _event("event-2", "root-a", "DECEPTIVE"),
        _event("event-3", "root-b", "SKIP"),
        _event("event-4", "root-c", "NO_ACTION"),
        _event("event-5", "root-c", "WRONG_WITHOUT_BASELINE_KNOWLEDGE"),
    )


def test_train_fold_label_shuffle_is_deterministic_and_preserves_all_classes() -> None:
    first = shuffle_train_fold_outcome_labels(
        _events(), held_out_family_fold="outer_5", seed=19
    )
    second = shuffle_train_fold_outcome_labels(
        tuple(reversed(_events())), held_out_family_fold="outer_5", seed=19
    )
    assert first.events == second.events
    assert Counter(event.outcome_class for event in first.events) == Counter(
        event.outcome_class for event in _events()
    )
    assert set(first.outcome_by_event_id.values()) == set(OUTCOME_CLASSES)


def test_train_fold_root_identity_shuffle_preserves_whole_root_mixtures() -> None:
    result = shuffle_train_fold_root_identities(
        _events(), _graph("root-a", "root-b", "root-c"), held_out_family_fold="outer_5", seed=8
    )
    original_by_root = {
        root: Counter(event.outcome_class for event in _events() if event.root_id == root)
        for root in ("root-a", "root-b", "root-c")
    }
    shuffled_by_root = {
        root: Counter(event.outcome_class for event in result.events if event.root_id == root)
        for root in original_by_root
    }
    assert result.root_identity_map != {root: root for root in original_by_root}
    assert sorted(shuffled_by_root.values(), key=repr) == sorted(
        original_by_root.values(), key=repr
    )


def test_train_fold_controls_reject_held_out_events_and_unknown_graph_identity() -> None:
    held_out = PreStatusRiskEvent(
        event_id="held-out",
        root_id="held-out-root",
        family="held-out-family",
        family_fold="outer_5",
        outcome_class="HONEST",
        nuisance_key=("turn",),
    )
    with pytest.raises(RelationalPreStatusRiskDiagnosticsError, match="held-out"):
        shuffle_train_fold_outcome_labels(
            (*_events(), held_out), held_out_family_fold="outer_5", seed=1
        )
    with pytest.raises(RelationalPreStatusRiskDiagnosticsError, match="absent"):
        shuffle_train_fold_root_identities(
            _events(), _graph("root-a"), held_out_family_fold="outer_5", seed=1
        )


def test_multiclass_calibration_summary_keeps_fixed_bins_and_all_classes() -> None:
    probabilities = {
        "HONEST": 0.6,
        "DECEPTIVE": 0.1,
        "SKIP": 0.1,
        "NO_ACTION": 0.1,
        "WRONG_WITHOUT_BASELINE_KNOWLEDGE": 0.1,
    }
    summary = multiclass_calibration_summary(
        ("HONEST", "SKIP"), (probabilities, probabilities), bin_count=4
    )
    assert set(summary["classes"]) == set(OUTCOME_CLASSES)
    assert all(len(entry["bins"]) == 4 for entry in summary["classes"].values())
    honest = summary["classes"]["HONEST"]
    assert honest["bins"][2]["mean_predicted_probability"] == pytest.approx(0.6)
    assert honest["bins"][2]["observed_frequency"] == pytest.approx(0.5)
    assert honest["ece"] == pytest.approx(0.1)


def test_mixed_root_floor_is_explicitly_descriptive_and_not_heldout() -> None:
    floor = descriptive_train_fold_mixed_root_proper_score_floor(
        _events(), held_out_family_fold="outer_5"
    )
    assert floor["label"] == "descriptive_train_fold_mixed_root_irreducible_proper_score_floor"
    assert floor["held_out_predictor"] is False
    assert floor["mixed_root_count"] == 2
    assert floor["mixed_root_event_count"] == 4
    assert set(floor["outcome_counts"]) == set(OUTCOME_CLASSES)
    assert floor["mean_empirical_root_log_loss"] > 0.0
