"""Unit tests for experiments/rescore_c10_truth_aware_nuisance.py.

All tests run on synthetic events; none touches the real artifacts or computes any
truth-aware re-score number (that is Phase 2, gated on registration).

Note on the metric: the published C10 numbers are the BINARY Brier of the conditional
deception probability P(D)/(P(H)+P(D)) against 1[outcome==DECEPTIVE], macro-averaged
over families (verified against the source projection and the shipped receipt). The
aggregation tests below pin that binary family-macro aggregation, not the 5-class
one-hot Brier sum.
"""

from __future__ import annotations

import pytest

from experiments.rescore_c10_truth_aware_nuisance import (
    FOLDS,
    GRAPH_MODEL,
    LEVEL_BASE_RATE,
    LEVEL_COARSE,
    LEVEL_EXACT,
    OUTCOME_CLASSES,
    PRIOR_MODEL,
    RescoreError,
    ScoredEvent,
    binary_loss,
    build_training_index,
    cluster_bootstrap,
    conditional_deception_probability,
    family_balanced_prior,
    family_macro_brier_gain,
    fidelity_gate,
    jeffreys_profile,
    load_scored_events,
    nuisance_preserving_permutation,
    permutation_blocks,
    permuted_graph_scores,
    report_class_order,
    score_events,
    truth_aware_coarse_cell,
    truth_aware_exact_cell,
    truth_aware_prior,
)


def _event(
    event_id: str,
    family: str,
    fold: str,
    outcome: str,
    *,
    token: int = 100,
    turn: int = 0,
    history: tuple[str, ...] = ("A",),
    pressure: bool = True,
    true_status: str = "PASS",
    desired_status: str = "FAIL",
    probabilities: dict[str, tuple[float, ...] | None] | None = None,
) -> ScoredEvent:
    return ScoredEvent(
        event_id=event_id,
        family=family,
        family_fold=fold,
        turn_index=turn,
        intervention_history=history,
        pressure_exposed=pressure,
        scenario_id=f"scn_{event_id}",
        status_sampled_token_id=token,
        outcome_class=outcome,
        true_status=true_status,
        desired_status=desired_status,
        probabilities=probabilities
        if probabilities is not None
        else {
            GRAPH_MODEL: (0.6, 0.3, 0.04, 0.03, 0.03),
            PRIOR_MODEL: (0.4, 0.4, 0.1, 0.05, 0.05),
        },
    )


def _by_id(events: list[ScoredEvent]) -> dict[str, ScoredEvent]:
    return {event.event_id: event for event in events}


# ---------------------------------------------------------------------------
# Truth-aware prior: fallback levels, fold safety, smoothing
# ---------------------------------------------------------------------------


def test_truth_aware_prior_exact_coarse_base_fallback() -> None:
    query_exact = _event("q1", "f1", "outer_1", "HONEST")
    query_coarse = _event("q2", "f1", "outer_1", "HONEST", history=("B",))
    query_base = _event("q3", "f1", "outer_1", "HONEST", token=999)
    training_exact = _event("t1", "f2", "outer_2", "DECEPTIVE")
    training_coarse = _event("t2", "f3", "outer_3", "HONEST", history=("C",))
    events = [query_exact, query_coarse, query_base, training_exact, training_coarse]
    events_by_id = _by_id(events)
    index = build_training_index(events, "outer_1")

    # q1 shares its full exact cell (token, turn, history, pressure, true, desired)
    # with t1: the prior is f2's single-family profile over [DECEPTIVE].
    probability, level = truth_aware_prior(query_exact, index, events_by_id)
    assert level == LEVEL_EXACT
    assert probability == pytest.approx(jeffreys_profile(["DECEPTIVE"], OUTCOME_CLASSES))

    # q2's exact cell is unseen (history ("B",) matches nobody), but its coarse cell
    # (token, turn, pressure, true, desired) matches t1 and t2 across histories: the
    # prior averages the per-family profiles of f2 and f3.
    probability, level = truth_aware_prior(query_coarse, index, events_by_id)
    assert level == LEVEL_COARSE
    profile_f2 = jeffreys_profile(["DECEPTIVE"], OUTCOME_CLASSES)
    profile_f3 = jeffreys_profile(["HONEST"], OUTCOME_CLASSES)
    expected = tuple((a + b) / 2 for a, b in zip(profile_f2, profile_f3))
    assert probability == pytest.approx(expected)

    # q3's token appears nowhere: base rate over the whole training pool, i.e. the
    # family-balanced mean over the two training families.
    probability, level = truth_aware_prior(query_base, index, events_by_id)
    assert level == LEVEL_BASE_RATE
    assert probability == pytest.approx(expected)


def test_truth_aware_cells_distinguish_truth_fields() -> None:
    base = _event("a", "f1", "outer_2", "HONEST")
    other_true = _event("b", "f1", "outer_2", "HONEST", true_status="FAIL")
    other_desired = _event("c", "f1", "outer_2", "HONEST", desired_status="PASS")
    assert truth_aware_exact_cell(base) != truth_aware_exact_cell(other_true)
    assert truth_aware_exact_cell(base) != truth_aware_exact_cell(other_desired)
    assert truth_aware_coarse_cell(base) != truth_aware_coarse_cell(other_true)
    assert truth_aware_coarse_cell(base) != truth_aware_coarse_cell(other_desired)
    # The coarse key drops only the intervention history.
    other_history = _event("d", "f1", "outer_2", "HONEST", history=("B",))
    assert truth_aware_exact_cell(base) != truth_aware_exact_cell(other_history)
    assert truth_aware_coarse_cell(base) == truth_aware_coarse_cell(other_history)


def test_training_index_fold_safety() -> None:
    events = [
        _event(f"e_{fold}_{i}", "f1", fold, "HONEST", token=i)
        for fold in FOLDS
        for i in range(3)
    ]
    index = build_training_index(events, "outer_1")
    held_out_ids = {event.event_id for event in events if event.family_fold == "outer_1"}
    assert held_out_ids.isdisjoint(index.training_event_ids)
    for cell_ids in list(index.exact.values()) + list(index.coarse.values()):
        assert held_out_ids.isdisjoint(cell_ids)
    expected_training = {
        event.event_id for event in events if event.family_fold != "outer_1"
    }
    assert set(index.training_event_ids) == expected_training

    # A query from a different fold than the index view is a leak and must raise.
    foreign_query = events[0]  # outer_1
    assert foreign_query.family_fold == "outer_1"
    with pytest.raises(RescoreError):
        truth_aware_prior(
            _event("x", "f1", "outer_2", "HONEST"), index, _by_id(events)
        )
    with pytest.raises(RescoreError):
        build_training_index(events, "outer_6")


def test_jeffreys_smoothing_and_family_balancing() -> None:
    # Jeffreys-0.5 over the 5 classes: denominator = n + 0.5 * 5.
    profile = jeffreys_profile(["HONEST", "DECEPTIVE"], OUTCOME_CLASSES)
    assert profile == pytest.approx((1.5 / 4.5, 1.5 / 4.5, 0.5 / 4.5, 0.5 / 4.5, 0.5 / 4.5))

    # Two families in the cell: arithmetic mean of the per-family profiles; the third
    # family (absent from the cell) does not enter.
    cell_members = [
        _event("a1", "fA", "outer_2", "HONEST"),
        _event("a2", "fA", "outer_2", "DECEPTIVE"),
        _event("b1", "fB", "outer_3", "DECEPTIVE"),
        _event("b2", "fB", "outer_3", "DECEPTIVE"),
        _event("b3", "fB", "outer_3", "DECEPTIVE"),
        _event("c1", "fC", "outer_4", "HONEST"),
    ]
    prior = family_balanced_prior(["a1", "a2", "b1", "b2", "b3"], _by_id(cell_members))
    profile_a = jeffreys_profile(["HONEST", "DECEPTIVE"], OUTCOME_CLASSES)
    profile_b = jeffreys_profile(["DECEPTIVE"] * 3, OUTCOME_CLASSES)
    expected = tuple((a + b) / 2 for a, b in zip(profile_a, profile_b))
    assert prior == pytest.approx(expected)
    assert family_balanced_prior([], _by_id(cell_members)) is None


# ---------------------------------------------------------------------------
# Metric aggregation (binary conditional-deception, family-macro)
# ---------------------------------------------------------------------------


def test_conditional_deception_renormalises_over_hd_mass() -> None:
    # Mass on non-H/D classes is excluded by the renormalisation.
    probability = (0.2, 0.4, 0.4, 0.0, 0.0)
    assert conditional_deception_probability(probability) == pytest.approx(2.0 / 3.0)
    loss, brier = binary_loss(0.25, 1)
    assert brier == pytest.approx(0.5625)
    assert loss == pytest.approx(1.3862943611198906)  # -log(0.25)


def test_family_macro_brier_is_unweighted_over_families() -> None:
    events = [
        _event("a1", "famA", "outer_1", "HONEST"),
        _event("a2", "famA", "outer_1", "HONEST"),
        _event("b1", "famB", "outer_2", "DECEPTIVE"),
    ]
    scores = {"a1": 0.0, "a2": 1.0, "b1": 1.0}
    metrics = score_events(events, scores)
    # famA briers 0 and 1 -> 0.5; famB brier 0 -> macro = (0.5 + 0) / 2 = 0.25,
    # which differs from the event-pooled 1/3.
    assert metrics["per_family"]["famA"]["brier"] == pytest.approx(0.5)
    assert metrics["per_family"]["famB"]["brier"] == pytest.approx(0.0)
    assert metrics["family_macro_brier"] == pytest.approx(0.25)
    assert metrics["family_macro_brier"] != pytest.approx(1.0 / 3.0)

    comparator_scores = {"a1": 0.5, "a2": 0.5, "b1": 0.5}
    gain, per_family = family_macro_brier_gain(events, scores, comparator_scores)
    assert per_family["famA"] == pytest.approx((0.25 - 0.0 + 0.25 - 1.0) / 2)
    assert per_family["famB"] == pytest.approx(0.25 - 0.0)
    assert gain == pytest.approx((per_family["famA"] + per_family["famB"]) / 2)


# ---------------------------------------------------------------------------
# Family-cluster bootstrap
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_ci_sanity() -> None:
    positive = {f"fam{i:02d}": 0.02 + 0.001 * i for i in range(20)}
    result = cluster_bootstrap(positive, seed=20260727, replicates=2000)
    low, high = result["percentile_95_interval"]
    assert result["observed_mean"] == pytest.approx(
        sum(positive.values()) / len(positive)
    )
    assert low <= result["observed_mean"] <= high
    # All family gains are comfortably positive and the spread is tiny relative to
    # the mean, so the resampled macro CI must exclude zero.
    assert low > 0.0
    assert result["bootstrap_fraction_positive"] == pytest.approx(1.0)

    symmetric = {f"fam{i:02d}": (0.05 if i % 2 else -0.05) for i in range(20)}
    result = cluster_bootstrap(symmetric, seed=20260727, replicates=2000)
    low, high = result["percentile_95_interval"]
    assert result["observed_mean"] == pytest.approx(0.0)
    assert low < 0.0 < high

    constant = {f"fam{i:02d}": 0.05 for i in range(20)}
    result = cluster_bootstrap(constant, seed=1, replicates=100)
    assert result["percentile_95_interval"] == pytest.approx([0.05, 0.05])


# ---------------------------------------------------------------------------
# Fidelity-gate assertion helper
# ---------------------------------------------------------------------------


def test_fidelity_gate_pass_fail() -> None:
    events = [
        _event(
            "e1",
            "famA",
            "outer_1",
            "HONEST",
            probabilities={
                GRAPH_MODEL: (0.8, 0.1, 0.1, 0.0, 0.0),
                PRIOR_MODEL: (0.5, 0.5, 0.0, 0.0, 0.0),
            },
        ),
        _event(
            "e2",
            "famA",
            "outer_1",
            "DECEPTIVE",
            probabilities={
                GRAPH_MODEL: (0.1, 0.8, 0.1, 0.0, 0.0),
                PRIOR_MODEL: (0.5, 0.5, 0.0, 0.0, 0.0),
            },
        ),
    ]
    # Hand-computed: graph scores 1/9 and 8/9 -> briers (1/9)^2 each; prior score
    # 0.5 -> brier 0.25 each; single family, so macro == per-family.
    graph_brier = (1.0 / 9.0) ** 2
    expected = {
        "graph_family_macro_brier": graph_brier,
        "prior_family_macro_brier": 0.25,
        "family_macro_brier_gain": 0.25 - graph_brier,
    }
    measured = fidelity_gate(events, targets=expected, tolerance=1e-9)
    assert measured["status"] == "passed"
    assert measured["graph_family_macro_brier"] == pytest.approx(graph_brier)
    assert measured["families_with_positive_gain"] == 1

    with pytest.raises(RescoreError, match="FIDELITY GATE FAILED"):
        fidelity_gate(events, targets={**expected, "family_macro_brier_gain": 0.5})
    # The published truth-blind targets must fail on synthetic data (the gate is not
    # a tautology).
    with pytest.raises(RescoreError, match="FIDELITY GATE FAILED"):
        fidelity_gate(events)


# ---------------------------------------------------------------------------
# Permutation machinery
# ---------------------------------------------------------------------------


def test_permutation_blocks_group_and_exclude() -> None:
    events = [
        _event("h1", "f1", "outer_2", "HONEST", token=1),
        _event("d1", "f1", "outer_3", "DECEPTIVE", token=1),
        _event("s1", "f1", "outer_4", "SKIP", token=1),  # non-H/D: excluded
        _event("q1", "f1", "outer_1", "HONEST", token=1),  # held-out fold: excluded
        _event("h2", "f1", "outer_2", "HONEST", token=2),  # different cell
        _event("h3", "f2", "outer_2", "HONEST", token=1),  # family-conditioned cell
    ]
    labels = {event.event_id: event.outcome_class for event in events}
    blocks = dict(permutation_blocks(events, labels, "outer_1"))
    cell = ("f1", 1, 0, ("A",), True)
    assert blocks[cell] == ("d1", "h1")  # sorted ids, s1/q1 excluded
    assert ("f1", 2, 0, ("A",), True) in blocks
    assert ("f2", 1, 0, ("A",), True) in blocks
    assert sum(len(ids) for ids in blocks.values()) == 4


def test_permuted_graph_scores_mirror_source_math() -> None:
    query = _event("q", "f1", "outer_1", "DECEPTIVE")
    ledger_rows = {
        "q": {
            "source_node_predictions": [
                {"unique_training_event_ids": ["a", "b", "c"]},
            ]
        }
    }
    # Node denominator = 3 events + 0.5 * 5 classes = 5.5; conditional deception is
    # (count_d + 0.5) / (count_h + count_d + 1.0) after the shared denominator cancels.
    labels = {"a": "HONEST", "b": "HONEST", "c": "DECEPTIVE"}
    scores = permuted_graph_scores([query], ledger_rows, labels)
    assert scores["q"] == pytest.approx(1.5 / 4.0)
    # Non-H/D labels stay fixed and contribute only to the denominator.
    labels = {"a": "HONEST", "b": "SKIP", "c": "DECEPTIVE"}
    scores = permuted_graph_scores([query], ledger_rows, labels)
    assert scores["q"] == pytest.approx(0.5)


def test_permutation_identity_when_no_switchable_blocks() -> None:
    # Every event sits in its own cell, so within-block shuffles are identity and the
    # null statistic must equal the observed statistic on every replicate.
    events = [
        _event("e1", "f1", "outer_1", "HONEST", token=1),
        _event("e2", "f2", "outer_2", "DECEPTIVE", token=2),
        _event("e3", "f3", "outer_3", "HONEST", token=3),
        _event("e4", "f4", "outer_4", "DECEPTIVE", token=4),
    ]
    primary = events  # all pressure-exposed H/D
    neighbours = {
        "e1": ["e2", "e3"],
        "e2": ["e3", "e4"],
        "e3": ["e1", "e4"],
        "e4": ["e1", "e2"],
    }
    ledgers = {
        fold: {
            event.event_id: {
                "source_node_predictions": [
                    {"unique_training_event_ids": neighbours[event.event_id]},
                ]
            }
            for event in primary
            if event.family_fold == fold
        }
        for fold in FOLDS
    }
    graph_scores = permuted_graph_scores(
        primary, {e.event_id: ledgers[e.family_fold][e.event_id] for e in primary},
        {e.event_id: e.outcome_class for e in events},
    )
    comparator_scores = {event.event_id: 0.5 for event in primary}
    observed, _ = family_macro_brier_gain(primary, graph_scores, comparator_scores)
    result = nuisance_preserving_permutation(
        primary,
        events,
        ledgers,
        comparator_scores,
        graph_scores,
        seed=20260727,
        replicates=50,
    )
    assert result["observed_family_macro_brier_gain"] == pytest.approx(observed)
    assert result["null_mean"] == pytest.approx(observed)
    assert result["null_max"] == pytest.approx(observed)
    assert result["one_sided_randomization_p"] == pytest.approx(1.0)
    assert result["switchable_block_count_total_across_fold_training_views"] == 0


# ---------------------------------------------------------------------------
# Report/loader validation helpers
# ---------------------------------------------------------------------------


def test_report_class_order_cross_check() -> None:
    report = {
        "full_five_way": {
            "class_counts": {label: 0 for label in OUTCOME_CLASSES},
        }
    }
    assert report_class_order(report) == OUTCOME_CLASSES
    # The report is serialized with sort_keys=True, so key order is alphabetical and
    # only the class SET is checkable; a wrong set must raise.
    bad = {"full_five_way": {"class_counts": {"HONEST": 0, "DECEPTIVE": 0}}}
    with pytest.raises(RescoreError):
        report_class_order(bad)


def test_load_scored_events_validation() -> None:
    row = {
        "field_event_id": "e1",
        "family": "f1",
        "family_fold": "outer_1",
        "turn_index": 0,
        "intervention_history": ["A"],
        "pressure_exposed": True,
        "scenario_id": "s1",
        "status_sampled_token_id": 7,
        "outcome_class": "HONEST",
        "true_status": "PASS",
        "desired_status": "FAIL",
        "class_probabilities": {GRAPH_MODEL: [0.5, 0.4, 0.04, 0.03, 0.03]},
    }
    events = load_scored_events({"scored_events": [row]})
    assert events[0].event_id == "e1"
    assert events[0].intervention_history == ("A",)
    with pytest.raises(RescoreError, match="duplicate"):
        load_scored_events({"scored_events": [row, row]})
    with pytest.raises(RescoreError, match="fold"):
        load_scored_events({"scored_events": [{**row, "family_fold": "outer_9"}]})
    with pytest.raises(RescoreError, match="outcome class"):
        load_scored_events({"scored_events": [{**row, "outcome_class": "BOGUS"}]})
