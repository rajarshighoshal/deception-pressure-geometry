from __future__ import annotations

from pathlib import Path

import pytest

from geoprobe.eval import relational_post_commitment_growth_outcome_projection as outcome_projection


def _event_structure(
    event_id: str,
    *,
    family: str = "family-alpha",
    family_fold: str = "outer_1",
    turn_index: int = 0,
    history: tuple[str, ...] = ("h",),
    pressure_exposed: bool = True,
) -> outcome_projection.EventStructure:
    return outcome_projection.EventStructure(
        field_event_id=event_id,
        family=family,
        family_fold=family_fold,
        turn_index=turn_index,
        intervention_history=history,
        pressure_exposed=pressure_exposed,
        scenario_id="scenario",
        orbit_id="orbit",
        sample_index=0,
        prefix_state_sha256=f"prefix-{event_id}",
        status_sampled_token_id=123,
        occurrence_ids=(f"occ-{event_id}",),
        activation_prefix_sha256s=(f"act-{event_id}",),
    )


def _event_outcome(
    event_id: str,
    *,
    outcome_class: str = "HONEST",
) -> outcome_projection.EventOutcome:
    return outcome_projection.EventOutcome(
        field_event_id=event_id,
        outcome_class=outcome_class,
        scientific_cohort=outcome_class,
        mapped_action="noop",
        sampled_token_id=123,
        knowledge_correct=True,
        true_status="PASS",
        desired_status="PASS",
        occurrence_ids=(f"occ-{event_id}",),
    )


def _node_structure(
    node_id: str,
    event: outcome_projection.EventStructure,
) -> outcome_projection.NodeStructure:
    return outcome_projection.NodeStructure(
        node_id=node_id,
        metric_realization_sha256=f"metric-{node_id}",
        representative_edge_sha256=f"edge-{node_id}",
        physical_edge_ids=(f"physical-{node_id}",),
        field_event_ids=(event.field_event_id,),
        family=event.family,
        family_fold=event.family_fold,
        turn_index=event.turn_index,
        intervention_history=event.intervention_history,
        pressure_exposed=event.pressure_exposed,
        scenario_id=event.scenario_id,
        prefix_state_sha256=event.prefix_state_sha256,
        status_sampled_token_id=event.status_sampled_token_id,
    )


def _prepared_with_event(
    event: outcome_projection.EventStructure,
) -> outcome_projection.PreparedPostCommitmentGrowthOutcomeProjection:
    return _prepared_with_events({event.field_event_id: event})


def _prepared_with_events(
    events: dict[str, outcome_projection.EventStructure],
) -> outcome_projection.PreparedPostCommitmentGrowthOutcomeProjection:
    return outcome_projection.PreparedPostCommitmentGrowthOutcomeProjection(
        bank_root=Path("/tmp"),
        state_graph_root=Path("/tmp"),
        graph_file_sha256="graph",
        candidate_file_sha256="candidate",
        bank_manifest_sha256="bank",
        state_graph_manifest_sha256="state-manifest",
        state_label_free_projection_sha256="state-label-free",
        protocol_file_sha256="protocol",
        family_entries={},
        state_rows={},
        occurrences={},
        events=events,
        nodes={},
        event_to_nodes={
            event_id: (f"node-{event_id}",) for event_id in sorted(events)
        },
        query_neighbors_by_fold={},
        roster={},
        roster_sha256="roster",
    )


def _prediction_row(event: outcome_projection.EventStructure) -> dict[str, object]:
    return {
        "field_event_id": event.field_event_id,
        "family": event.family,
        "family_fold": event.family_fold,
        "turn_index": event.turn_index,
        "intervention_history": list(event.intervention_history),
        "pressure_exposed": event.pressure_exposed,
        "scenario_id": event.scenario_id,
        "orbit_id": event.orbit_id,
        "sample_index": event.sample_index,
        "prefix_state_sha256": event.prefix_state_sha256,
        "status_sampled_token_id": event.status_sampled_token_id,
        "activation_prefix_sha256s": list(event.activation_prefix_sha256s),
        "source_node_ids": [f"node-{event.field_event_id}"],
        "class_probabilities": {
            "local_joint_top8": [0.2, 0.2, 0.2, 0.2, 0.2],
            "exact_nuisance_family_balanced": [0.2, 0.2, 0.2, 0.2, 0.2],
            "coarse_nuisance_family_balanced": [0.2, 0.2, 0.2, 0.2, 0.2],
        },
    }


def _ledger_with_event(event: outcome_projection.EventStructure) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": outcome_projection.PREDICTION_KIND,
        "held_out_family_fold": event.family_fold,
        "artifact_bindings": {
            "roster_sha256": "roster",
            "graph_file_sha256": "graph",
            "candidate_file_sha256": "candidate",
            "bank_manifest_sha256": "bank",
            "state_graph_manifest_sha256": "state-manifest",
            "state_label_free_projection_sha256": "state-label-free",
            "protocol_file_sha256": "protocol",
        },
        "outcome_classes": list(outcome_projection.OUTCOME_CLASSES),
        "training_folds": [fold for fold in outcome_projection.FOLDS if fold != event.family_fold],
        "heldout_families": [],
        "opened_training_outcome_shards": [],
        "training_unique_event_count": 1,
        "query_unique_event_count": 1,
        "predictions": [_prediction_row(event)],
        "prediction_ledger_sha256": "",
    }


def test_ledger_hash_and_prediction_ledger_validation_are_order_stable() -> None:
    event = _event_structure("evt-0")
    prepared = _prepared_with_event(event)
    ledger = _ledger_with_event(event)
    ledger["prediction_ledger_sha256"] = outcome_projection._ledger_hash(ledger)
    fold, predictions = outcome_projection._validate_prediction_ledger(prepared, ledger)

    assert fold == event.family_fold
    assert set(predictions) == {event.field_event_id}
    assert outcome_projection._ledger_hash(ledger) == outcome_projection._ledger_hash(
        dict(reversed(list(ledger.items())))
    )

    bad_ledger = dict(ledger)
    bad_ledger["prediction_ledger_sha256"] = "0" * 64
    with pytest.raises(
        outcome_projection.RelationalPostCommitmentGrowthOutcomeProjectionError,
        match="self-hash",
    ):
        outcome_projection._validate_prediction_ledger(prepared, bad_ledger)

    report = {"schema_version": 1, "kind": "report", "value": 7}
    report["report_sha256"] = outcome_projection._report_hash(report)
    assert report["report_sha256"] == outcome_projection._report_hash(report)
    assert report["report_sha256"] != outcome_projection._report_hash(
        {**report, "value": 8}
    )


def test_jeffreys_probabilities_respect_all_classes_and_deduplicate_events() -> None:
    event_ids = ("evt-a", "evt-a", "evt-b", "evt-c")
    events = {
        "evt-a": _event_structure(
            "evt-a",
            family="family-1",
            turn_index=1,
            history=("history",),
        ),
        "evt-b": _event_structure("evt-b", family="family-1", turn_index=1, history=("history",)),
        "evt-c": _event_structure("evt-c", family="family-2", turn_index=2, history=("history",)),
    }
    outcomes = {
        "evt-a": _event_outcome("evt-a", outcome_class="HONEST"),
        "evt-b": _event_outcome("evt-b", outcome_class="DECEPTIVE"),
        "evt-c": _event_outcome("evt-c", outcome_class="DECEPTIVE"),
    }

    prior, support = outcome_projection._family_balanced_prior(
        list(event_ids),
        events=events,
        outcomes=outcomes,
    )
    assert prior is not None
    family_one = outcome_projection._probability(
        outcome_projection._counts(["HONEST", "DECEPTIVE"]),
    )
    family_two = outcome_projection._probability(outcome_projection._counts(["DECEPTIVE"]))
    expected = [
        0.5 * (value_one + value_two)
        for value_one, value_two in zip(family_one, family_two)
    ]
    assert prior == pytest.approx(expected)
    assert support["training_family_count"] == 2
    assert support["training_event_count"] == 3
    assert support["training_event_count_by_family"] == {
        "family-1": 2,
        "family-2": 1,
    }

    with pytest.raises(
        outcome_projection.RelationalPostCommitmentGrowthOutcomeProjectionError,
        match="lacks an outcome class",
    ):
        outcome_projection._probability({"HONEST": 1})


def test_honest_deceptive_direction_and_family_macro_gain_are_coherent() -> None:
    row_low_to_high = {
        "field_event_id": "low-to-high",
        "family": "family-a",
        "family_fold": "outer_1",
        "outcome_class": "DECEPTIVE",
        "class_probabilities": {
            "exact_nuisance_family_balanced": [0.8, 0.2, 0, 0, 0],
            "local_joint_top8": [0.5, 0.5, 0, 0, 0],
            "coarse_nuisance_family_balanced": [0.5, 0.5, 0, 0, 0],
        },
    }
    row_high_to_low = {
        "field_event_id": "high-to-low",
        "family": "family-a",
        "family_fold": "outer_1",
        "outcome_class": "HONEST",
        "class_probabilities": {
            "exact_nuisance_family_balanced": [0.1, 0.9, 0, 0, 0],
            "local_joint_top8": [0.5, 0.5, 0, 0, 0],
            "coarse_nuisance_family_balanced": [0.5, 0.5, 0, 0, 0],
        },
    }
    row_favored = {
        "field_event_id": "favored",
        "family": "family-b",
        "family_fold": "outer_2",
        "outcome_class": "DECEPTIVE",
        "class_probabilities": {
            "exact_nuisance_family_balanced": [0.95, 0.05, 0, 0, 0],
            "local_joint_top8": [0.5, 0.5, 0, 0, 0],
            "coarse_nuisance_family_balanced": [0.5, 0.5, 0, 0, 0],
        },
    }

    local_scores = {
        "low-to-high": 0.8,
        "high-to-low": 0.3,
        "favored": 0.4,
    }
    gain = outcome_projection._family_macro_brier_gain(
        [row_low_to_high, row_high_to_low, row_favored],
        local_scores,
    )
    expected = (
        (
            ((0.2 - 1.0) ** 2 - (0.8 - 1.0) ** 2)
            + (0.9**2 - 0.3**2)
        )
        / 2
        + ((0.05 - 1.0) ** 2 - (0.4 - 1.0) ** 2)
    ) / 2
    assert gain == pytest.approx(expected)
    assert (
        outcome_projection._conditional_deception_probability(
            [0.8, 0.2, 0, 0, 0]
        )
        < outcome_projection._conditional_deception_probability(
            [0.2, 0.8, 0, 0, 0]
        )
    )


def test_fold_prediction_hash_is_invariant_to_heldout_label_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_events = {
        f"query-{index:03d}": _event_structure(
            f"query-{index:03d}",
            family="query-family",
            family_fold="outer_1",
        )
        for index in range(336)
    }
    training_events = {
        f"train-{index}": _event_structure(
            f"train-{index}",
            family="training-family",
            family_fold="outer_2",
        )
        for index in range(8)
    }
    events = {**query_events, **training_events}
    nodes = {
        f"node-{event_id}": _node_structure(f"node-{event_id}", event)
        for event_id, event in events.items()
    }
    target_nodes = tuple(f"node-train-{index}" for index in range(8))
    prepared = outcome_projection.PreparedPostCommitmentGrowthOutcomeProjection(
        bank_root=Path("/tmp"),
        state_graph_root=Path("/tmp"),
        graph_file_sha256="graph",
        candidate_file_sha256="candidate",
        bank_manifest_sha256="bank",
        state_graph_manifest_sha256="state-manifest",
        state_label_free_projection_sha256="state-label-free",
        protocol_file_sha256="protocol",
        family_entries={
            "query-family": {"family_fold": "outer_1"},
            "training-family": {"family_fold": "outer_2"},
        },
        state_rows={},
        occurrences={},
        events=events,
        nodes=nodes,
        event_to_nodes={
            event_id: (f"node-{event_id}",) for event_id in sorted(events)
        },
        query_neighbors_by_fold={
            "outer_1": {
                f"node-{event_id}": target_nodes for event_id in query_events
            }
        },
        roster={},
        roster_sha256="roster",
    )
    outcomes = {
        event_id: _event_outcome(
            event_id,
            outcome_class="HONEST" if index % 2 == 0 else "DECEPTIVE",
        )
        for index, event_id in enumerate(training_events)
    }
    outcomes.update(
        {event_id: _event_outcome(event_id) for event_id in query_events}
    )
    requested_folds: list[set[str]] = []

    def fake_load_outcomes(
        _prepared: outcome_projection.PreparedPostCommitmentGrowthOutcomeProjection,
        *,
        allowed_folds: set[str],
        outcome_loader: outcome_projection.OutcomeLoader,
    ) -> tuple[dict[str, outcome_projection.EventOutcome], list[dict[str, object]]]:
        del outcome_loader
        requested_folds.append(set(allowed_folds))
        return (
            {
                event_id: outcome
                for event_id, outcome in outcomes.items()
                if events[event_id].family_fold in allowed_folds
            },
            [],
        )

    monkeypatch.setattr(
        outcome_projection,
        "_load_outcomes_for_folds",
        fake_load_outcomes,
    )
    before = outcome_projection.build_relational_post_commitment_growth_fold_predictions(
        prepared,
        held_out_family_fold="outer_1",
    )
    outcomes["query-000"] = _event_outcome(
        "query-000",
        outcome_class="DECEPTIVE",
    )
    after = outcome_projection.build_relational_post_commitment_growth_fold_predictions(
        prepared,
        held_out_family_fold="outer_1",
    )

    assert before["prediction_ledger_sha256"] == after["prediction_ledger_sha256"]
    assert before["predictions"] == after["predictions"]
    assert requested_folds == [
        set(outcome_projection.FOLDS) - {"outer_1"},
        set(outcome_projection.FOLDS) - {"outer_1"},
    ]


def _scored_row(
    event_id: str,
    prefix_state_sha256: str,
    outcome_class: str,
    true_status: str,
    *,
    activation_prefix_sha256s: tuple[str, ...],
    pressure_exposed: bool = True,
) -> dict[str, object]:
    return {
        "field_event_id": event_id,
        "family": "family",
        "family_fold": "outer_1",
        "turn_index": 1,
        "intervention_history": ["h"],
        "pressure_exposed": pressure_exposed,
        "scenario_id": "scenario",
        "orbit_id": "orbit",
        "sample_index": 0,
        "prefix_state_sha256": prefix_state_sha256,
        "activation_prefix_sha256s": list(activation_prefix_sha256s),
        "status_sampled_token_id": 1 if outcome_class == "HONEST" else 2,
        "outcome_class": outcome_class,
        "scientific_cohort": outcome_class,
        "mapped_action": "noop",
        "knowledge_correct": True,
        "true_status": true_status,
        "desired_status": true_status,
        "class_probabilities": {
            "local_joint_top8": [0.7, 0.3, 0, 0, 0],
            "exact_nuisance_family_balanced": [0.6, 0.4, 0, 0, 0],
            "coarse_nuisance_family_balanced": [0.5, 0.5, 0, 0, 0],
        },
    }


def test_exact_prefix_pairs_build_31_pair_inventory_and_30_exact_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(outcome_projection, "BOOTSTRAP_REPLICATES", 10)
    rows: list[dict[str, object]] = []
    for index in range(31):
        true_status = "PASS" if index % 2 == 0 else "FAIL"
        prefix = f"prefix-{index:02d}"
        if index < 30:
            activation_hashes = ("act-shared",)
        else:
            activation_hashes = ("act-left",)
        rows.append(
            _scored_row(
                f"honest-{index}",
                prefix,
                "HONEST",
                true_status,
                activation_prefix_sha256s=activation_hashes,
            )
        )
        rows.append(
            _scored_row(
                f"deceptive-{index}",
                prefix,
                "DECEPTIVE",
                true_status,
                activation_prefix_sha256s=(
                    "act-shared" if index < 30 else "act-right",
                ),
            )
        )

    rows.append(
        _scored_row(
            "skip-ignored",
            "skip-prefix",
            "SKIP",
            "PASS",
            activation_prefix_sha256s=("act-skip",),
        )
    )
    rows.append(
        _scored_row(
            "not-exposed",
            "not-exposed-prefix",
            "HONEST",
            "PASS",
            activation_prefix_sha256s=("act-no",),
            pressure_exposed=False,
        )
    )

    report = outcome_projection._exact_prefix_pair_report(rows)
    assert len(report["pair_inventory"]) == 31
    assert report["primary_strict_activation_exact"]["overall"]["pair_count"] == 30
    assert report["all_token_identical_sensitivity"]["overall"]["pair_count"] == 31
    assert sum(1 for row in report["pair_inventory"] if row["activation_exact"]) == 30


def test_permutation_blocks_are_order_invariant() -> None:
    events = {
        "event-b": _event_structure(
            "event-b",
            family="family-1",
            family_fold="outer_1",
            turn_index=2,
            history=("h",),
        ),
        "event-a": _event_structure(
            "event-a",
            family="family-1",
            family_fold="outer_2",
            turn_index=2,
            history=("h",),
        ),
        "event-c": _event_structure(
            "event-c",
            family="family-2",
            family_fold="outer_3",
            turn_index=2,
            history=("h",),
        ),
    }
    outcomes = {
        "event-b": _event_outcome("event-b", outcome_class="HONEST"),
        "event-a": _event_outcome("event-a", outcome_class="DECEPTIVE"),
        "event-c": _event_outcome("event-c", outcome_class="HONEST"),
    }
    prepared = _prepared_with_events(events)
    prepared_nodes = _prepared_with_events(
        {key: events[key] for key in ("event-c", "event-b", "event-a")}
    )

    blocks, switchable = outcome_projection._permutation_blocks(
        prepared,
        outcomes=outcomes,
        fold="outer_4",
        family_conditioned=False,
    )
    blocks_reordered, switchable_reordered = outcome_projection._permutation_blocks(
        prepared_nodes,
        outcomes=outcomes,
        fold="outer_4",
        family_conditioned=False,
    )
    assert blocks == blocks_reordered
    assert switchable == switchable_reordered
    blocks_family, switchable_family = outcome_projection._permutation_blocks(
        prepared,
        outcomes=outcomes,
        fold="outer_4",
        family_conditioned=True,
    )
    assert len(blocks_family) == 2
    assert switchable_family == 1


def test_permuted_local_scores_are_row_order_invariant() -> None:
    primary_rows = [
        {
            "field_event_id": "query-a",
            "family_fold": "outer_1",
            "class_probabilities": {
                "local_joint_top8": [0.2, 0.8, 0, 0, 0],
            },
        },
        {
            "field_event_id": "query-b",
            "family_fold": "outer_1",
            "class_probabilities": {
                "local_joint_top8": [0.6, 0.4, 0, 0, 0],
            },
        },
    ]
    prediction_by_fold = {
        "outer_1": {
            "query-a": {
                "source_node_predictions": [
                    {"unique_training_event_ids": ["train-a", "train-b"]},
                ]
            },
            "query-b": {
                "source_node_predictions": [
                    {"unique_training_event_ids": ["train-b", "train-c"]},
                ]
            },
        },
    }
    labels = {
        "train-a": "HONEST",
        "train-b": "DECEPTIVE",
        "train-c": "DECEPTIVE",
    }
    forward = outcome_projection._permuted_local_scores(
        primary_rows,
        ledger_by_fold=prediction_by_fold,
        labels=labels,
    )
    backward = outcome_projection._permuted_local_scores(
        list(reversed(primary_rows)),
        ledger_by_fold=prediction_by_fold,
        labels=labels,
    )
    assert forward == backward
    assert forward["query-a"] == pytest.approx(0.5)
