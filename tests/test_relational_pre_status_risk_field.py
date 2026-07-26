from __future__ import annotations

import pytest

from geoprobe.eval.relational_pre_status_risk_field import (
    FoldSafePreStatusRiskField,
    PreStatusRiskEvent,
    RelationalPreStatusRiskFieldError,
    multiclass_brier,
    multiclass_log_loss,
)
from geoprobe.geometry.relational_pre_status_honestward import HonestwardSupportEdge


def _event(
    event_id: str,
    root_id: str,
    outcome: str,
    *,
    fold: str = "outer_1",
    nuisance: tuple[str, ...] = ("turn2", "AN", "pressure"),
) -> PreStatusRiskEvent:
    return PreStatusRiskEvent(
        event_id=event_id,
        root_id=root_id,
        family=f"family-{fold}",
        family_fold=fold,
        outcome_class=outcome,
        nuisance_key=nuisance,
    )


def _edges(*roots: str) -> tuple[HonestwardSupportEdge, ...]:
    return tuple(
        HonestwardSupportEdge(root, index, float(index))
        for index, root in enumerate(roots, 1)
    )


def test_fit_rejects_heldout_event() -> None:
    with pytest.raises(RelationalPreStatusRiskFieldError, match="held-out"):
        FoldSafePreStatusRiskField.fit(
            (_event("e", "r", "HONEST", fold="outer_5"),),
            held_out_family_fold="outer_5",
        )


def test_mixed_exact_root_is_retained_as_stochastic_counts() -> None:
    field = FoldSafePreStatusRiskField.fit(
        (
            _event("e1", "r1", "HONEST"),
            _event("e2", "r1", "DECEPTIVE"),
            _event("e3", "r2", "DECEPTIVE"),
        ),
        held_out_family_fold="outer_5",
    )
    prediction = field.predict(
        event_id="query",
        root_id="query-root",
        nuisance_key=("turn2", "AN", "pressure"),
        edges=_edges("r1"),
    )
    assert prediction.support_event_ids == ("e1", "e2")
    assert prediction.local_probabilities["HONEST"] == pytest.approx(
        prediction.local_probabilities["DECEPTIVE"]
    )


def test_query_deduplicates_neighbor_roots_and_events() -> None:
    field = FoldSafePreStatusRiskField.fit(
        (_event("e1", "r1", "HONEST"), _event("e2", "r2", "DECEPTIVE")),
        held_out_family_fold="outer_5",
    )
    prediction = field.predict(
        event_id="query",
        root_id="q",
        nuisance_key=("missing",),
        edges=_edges("r1", "r1", "r2"),
    )
    assert prediction.support_count == 2
    assert prediction.support_root_ids == ("r1", "r2")


def test_proper_scores_accept_complete_probability_vector() -> None:
    field = FoldSafePreStatusRiskField.fit(
        (_event("e1", "r1", "HONEST"), _event("e2", "r2", "DECEPTIVE")),
        held_out_family_fold="outer_5",
    )
    probabilities = field.predict(
        event_id="q", root_id="q", nuisance_key=("missing",), edges=_edges("r1")
    ).local_probabilities
    assert multiclass_log_loss("HONEST", probabilities) > 0.0
    assert multiclass_brier("HONEST", probabilities) >= 0.0
