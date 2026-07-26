from __future__ import annotations

from copy import deepcopy

import pytest

from geoprobe.eval.relational_connection_path_bank import (
    RelationalConnectionPathBankError,
    build_complete_path_bank,
)


def _sides() -> list[dict[str, object]]:
    rows = [
        {"relation_name": f"residual.L{layer}", "view": "residual", "side": "symmetric", "rank": 1}
        for layer in (12, 16, 19, 20)
    ]
    rows.extend(
        {"relation_name": f"attention.L{layer}.H{head}", "view": "attention", "side": side, "rank": 1}
            for layer, head in ((12, relation_index) if relation_index < 32 else (16, relation_index - 32) for relation_index in range(57))
        for side in ("left", "right")
    )
    rows.extend(
        {"relation_name": f"transport.L{left}->L{right}", "view": "layer_transport", "side": "symmetric", "rank": 1}
        for left, right in ((12, 16), (16, 19))
    )
    assert len(rows) == 120
    return rows


def _relation_attempts(sides: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            **{key: item[key] for key in ("relation_name", "view", "side")},
            "policy_status": "admitted",
            "forward": {
                "status": "supported",
                "principal_angle_cosines": [1.0],
                "normalized_transported_projector_discrepancy": 0.1,
                "normalized_polar_residual": 0.2,
                "polar_min_singular_value": 1.0,
            },
        }
        for item in sides
    ]


def _fixtures() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    sides = _sides()
    labels = ["DECEPTIVE"] * 34 + ["HONEST"] * 18 + ["SKIP"] * 4 + ["WRONG_WITHOUT_BASELINE_KNOWLEDGE"] * 4
    events: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    checkpoints: dict[str, object] = {}
    for index, outcome in enumerate(labels):
        scenario, event_id = f"s{index:02d}", f"event-{index:02d}"
        family, fold = f"family-{index % 20:02d}", f"outer_{index % 5 + 1}"
        event = {
            "field_event_id": event_id, "scenario_id": scenario, "family": family, "fold": fold,
            "turn_index": 2, "intervention_history": ["A", "N"], "prefix_state_sha256": f"prefix-{index}",
            "true_status": "PASS", "desired_status": "FAIL", "knowledge_correct": outcome != "WRONG_WITHOUT_BASELINE_KNOWLEDGE",
            "outcome_class": outcome,
        }
        prior_event = {
            **event,
            "field_event_id": f"prior-{index}",
            "turn_index": 1,
            "intervention_history": ["A"],
            "prefix_state_sha256": f"prior-prefix-{index}",
            "outcome_class": "DECEPTIVE",
        }
        events.extend((prior_event, event))
        attempts: list[dict[str, object]] = []
        targets = [f"target-{index}-{clone}" for clone in range(3)]
        group = f"replay-{index}"
        for clone, target in enumerate(targets):
            row = {
                "primary_realization_pair_id": f"in-{index}-{clone}", "scenario_id": scenario, "family": family, "fold": fold,
                "sample_index": 0, "edge_pair_id": f"in-edge-{index}", "source_reference_id": f"source-{index}", "target_reference_id": target,
                "source_program": "NN", "target_program": "AN", "turn_index": 1,
                "source_field_event_id": f"nn-{index}", "target_field_event_id": f"prior-{index}",
            }
            if clone == 0:
                rows.append(row)
            attempts.append({
                **{key: row[key] for key in ("edge_pair_id", "source_reference_id", "target_reference_id", "source_program", "target_program", "turn_index")},
                "roster_class": "primary", "attempt_kind": "forward_primary", "sample_index": 0, "matched_replay_group_id": group,
                "replay_match_status": "matched_exact_target_endpoint", "source_section_sha256": f"source-section-{index}",
                "relation_attempts": _relation_attempts(sides),
            })
        for target_program in ("AA", "AB"):
            row = {
                "primary_realization_pair_id": f"out-{index}-{target_program}", "scenario_id": scenario, "family": family, "fold": fold,
                "sample_index": 0, "edge_pair_id": f"out-edge-{index}-{target_program}", "source_reference_id": f"an-{index}", "target_reference_id": f"out-target-{index}-{target_program}",
                "source_program": "AN", "target_program": target_program, "turn_index": 2,
                "source_field_event_id": event_id, "target_field_event_id": f"next-{index}-{target_program}",
                "source_prefix_state_sha256": f"prefix-{index}",
            }
            rows.append(row)
            attempts.append({
                **{key: row[key] for key in ("edge_pair_id", "source_reference_id", "target_reference_id", "source_program", "target_program", "turn_index")},
                "roster_class": "primary", "attempt_kind": "forward_primary", "sample_index": 0, "source_section_sha256": f"an-section-{index}",
                "relation_attempts": _relation_attempts(sides),
            })
        checkpoints[scenario] = {
            "scenario_id": scenario, "attempts": attempts,
            "replay_attempts": [{
                "replay_group_id": group, "attempt_kind": "replay_control", "affirmative_claim_eligible": False, "pair_count": 3,
                "member_reference_ids": targets,
                "pairs": [
                    {"source_reference_id": targets[clone], "target_reference_id": targets[(clone + 1) % 3], "relation_attempts": _relation_attempts(sides)}
                    for clone in range(3)
                ],
            }],
        }
    return (
        {"schema_version": 1, "kind": "relational_partial_frame_outcome_join_report", "endpoint_events": events, "exact_realization_pairings": rows},
        checkpoints,
        {"relation_sides": sides, "inventory_hash": "b" * 64},
    )


def test_extracts_exact_label_blind_complete_path_bank() -> None:
    outcome_join, checkpoints, sides = _fixtures()
    result = build_complete_path_bank(outcome_join=outcome_join, checkpoints_by_scenario=checkpoints, stable_relation_sides=sides)
    assert result["policy_contract"] == "artifact_only_cross_fitted"
    assert result["coverage"]["class_counts"] == {
        "HONEST": 18, "DECEPTIVE": 34, "SKIP": 4, "NO_ACTION": 0, "WRONG_WITHOUT_BASELINE_KNOWLEDGE": 4,
    }
    assert len(result["paths"]) == 60
    path = result["paths"][0]
    assert path["class_counts"] == {"HONEST": 0, "DECEPTIVE": 1, "SKIP": 0, "NO_ACTION": 0, "WRONG_WITHOUT_BASELINE_KNOWLEDGE": 0}
    assert len(path["signature"]["relation_sides"]) == 120
    assert path["signature"]["relation_sides"][0]["I"]["coordinates"] == [0.0, 0.0, 0.0]


def test_rejects_confirmatory_and_broken_replay_binding() -> None:
    outcome_join, checkpoints, sides = _fixtures()
    with pytest.raises(RelationalConnectionPathBankError, match="confirmatory"):
        build_complete_path_bank(outcome_join=outcome_join, checkpoints_by_scenario=checkpoints, stable_relation_sides=sides, confirmatory=True)
    broken = deepcopy(checkpoints)
    broken["s00"]["replay_attempts"][0]["pairs"].pop()
    with pytest.raises(RelationalConnectionPathBankError, match="three physical pairs"):
        build_complete_path_bank(outcome_join=outcome_join, checkpoints_by_scenario=broken, stable_relation_sides=sides)
    negative = deepcopy(checkpoints)
    negative["s00"]["attempts"][0]["relation_attempts"][0]["forward"][
        "principal_angle_cosines"
    ] = [-0.1]
    with pytest.raises(RelationalConnectionPathBankError, match="outside"):
        build_complete_path_bank(
            outcome_join=outcome_join,
            checkpoints_by_scenario=negative,
            stable_relation_sides=sides,
        )
