from __future__ import annotations

from copy import deepcopy

import pytest

import geoprobe.data.relational_structured_capture as capture
from geoprobe.models.relational_structured_action import EOT_TOKEN_ID, canonical_json_sha256


def _sha(character: str) -> str:
    return character * 64


def _row(scenario_index: int, row_index: int) -> dict:
    scenario_id = f"scenario-{scenario_index:02d}"
    conversation_id = f"{scenario_id}:row-{row_index}"
    token_ids = [EOT_TOKEN_ID] * 4
    turns = []
    anchors = []
    spans = []
    for turn_index in range(4):
        fields = []
        for field_index, field_name in enumerate(("status", "caveat")):
            event_id = f"{conversation_id}:turn-{turn_index}:{field_name}"
            fields.append({
                "field_name": field_name,
                "field_event_id": event_id,
                "rng_domain": f"domain:{field_name}",
                "rng_stream_id": f"stream:{event_id}",
                "rng_seed": turn_index * 10 + field_index,
                "anchor_token_index": turn_index,
                "sampled_token_index": turn_index,
                "raw_token_id": EOT_TOKEN_ID,
                "raw_token_origin": "model_sample",
            })
            anchors.append({
                "turn_index": turn_index,
                "field_name": field_name,
                "field_event_id": event_id,
                "anchor_token_index": turn_index,
                "anchor_token_id": EOT_TOKEN_ID,
                "sampled_token_index": turn_index,
                "anchor_semantic_position": (
                    "immediately_before_status_sample"
                    if field_name == "status"
                    else "immediately_before_caveat_sample_status_visible"
                ),
            })
            spans.append({
                "turn_index": turn_index,
                "field_name": field_name,
                "token_start": turn_index,
                "token_end": turn_index + 1,
                "token_origin": "model_sample",
            })
        turns.append({
            "turn_index": turn_index,
            "turn_event_id": f"{conversation_id}:turn-{turn_index}",
            "transcript_before_action_token_count": turn_index,
            "transcript_before_action_token_ids_sha256": _sha("a"),
            "environment_eot": {"token_id": EOT_TOKEN_ID, "token_index": turn_index},
            "fields": fields,
        })
    return {
        "conversation_id": conversation_id,
        "scenario_id": scenario_id,
        "spec_sha256": _sha("b"),
        "source_spec_sha256": _sha("b"),
        "row_sha256": _sha("c"),
        "token_sha256": _sha("d"),
        "token_ids": token_ids,
        "token_origins": ["environment"] * 4,
        "token_origin_details": ["environment_eot"] * 4,
        "token_role_ids": [2] * 4,
        "token_turn_ids": list(range(4)),
        "token_message_ids": [2, 4, 6, 8],
        "token_span_flags": [0] * 4,
        "assistant_action_turn_records": turns,
        "action_anchors": anchors,
        "action_token_spans": spans,
        "assistant_eot_token_id": EOT_TOKEN_ID,
        "stop_token_id": EOT_TOKEN_ID,
        "stop_token_index": 3,
        "stop_reason": "environment_eot",
        "truncated": False,
    }


def _rows() -> list[dict]:
    return [_row(scenario_index, row_index) for scenario_index in range(60) for row_index in range(10)]


def _patch_live_validators(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"protocol": 0, "row": 0, "scenario": 0}

    def protocol_validator(_protocol: object) -> None:
        calls["protocol"] += 1

    def row_validator(_row: object, _protocol: object) -> None:
        calls["row"] += 1

    def scenario_validator(_rows: object, _protocol: object) -> None:
        calls["scenario"] += 1

    monkeypatch.setattr(capture, "validate_structured_action_protocol", protocol_validator)
    monkeypatch.setattr(capture, "validate_relational_structured_action_row", row_validator)
    monkeypatch.setattr(capture, "validate_relational_structured_action_scenario_rows", scenario_validator)
    return calls


def _manifest(rows_sha256: str, protocol_sha256: str) -> dict:
    contract_body = {
        "protocol_sha256": protocol_sha256,
        "model_artifact_sha256": _sha("e"),
    }
    return {
        "schema_version": 2,
        "kind": "relational_structured_action_rollout_manifest",
        "status": "success",
        "inputs": {"protocol_sha256": protocol_sha256},
        "output": {"rows_sha256": rows_sha256},
        "rollout_contract": {
            "schema_version": 2,
            "kind": "relational_structured_action_rollout_contract",
            **contract_body,
            "contract_sha256": canonical_json_sha256(contract_body),
        },
    }


def _gate(rows_sha256: str, protocol_sha256: str) -> dict:
    return {
        "schema_version": 2,
        "kind": "relational_structured_action_gate_report",
        "protocol_id": "relational-structured-action-v2-live-sequential",
        "protocol_sha256": protocol_sha256,
        "verdict": "not-found-under-this-task",
        "primary_gate_passed": False,
        "capture_eligible": False,
        "capture_allowed": False,
        "capture_authorization": "requires_separate_paid_capture_contract",
        "scope": {
            "development_mode": False,
            "confirmatory": True,
            "row_selection_performed": False,
            "row_dropping_performed": False,
            "prose_or_sidecar_borrowing_performed": False,
            "all_planned_events_retained": True,
        },
        "inputs": {
            "rows": {"sha256": rows_sha256},
            "protocol": {"sha256": protocol_sha256},
        },
    }


def test_source_requires_complete_bank_and_calls_existing_validators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_live_validators(monkeypatch)
    source = capture.validate_structured_action_capture_source(_rows(), {})

    assert source["row_count"] == 600
    assert source["scenario_count"] == 60
    assert len(source["token_lengths"]) == 600
    assert calls == {"protocol": 1, "row": 600, "scenario": 60}


def test_source_rejects_action_anchor_corruption_even_when_row_validator_is_external(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_live_validators(monkeypatch)
    rows = _rows()
    rows[0]["action_anchors"][0]["anchor_token_id"] = 7

    with pytest.raises(ValueError, match="action anchors"):
        capture.validate_structured_action_capture_source(rows, {})


def test_binding_keeps_behavioral_gate_descriptive_but_rejects_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_live_validators(monkeypatch)
    rows = _rows()
    rows_sha256 = _sha("1")
    protocol_sha256 = _sha("2")
    manifest = _manifest(rows_sha256, protocol_sha256)
    gate = _gate(rows_sha256, protocol_sha256)

    binding = capture.build_structured_action_capture_structural_binding(
        rows,
        manifest,
        gate,
        {},
        rows_sha256=rows_sha256,
        rollout_manifest_sha256=_sha("3"),
        gate_sha256=_sha("4"),
        protocol_sha256=protocol_sha256,
    )
    assert binding["scope"] == capture.STRUCTURED_CAPTURE_SCOPE
    assert binding["gate"]["capture_eligible"] is False
    assert binding["gate"]["capture_allowed"] is False
    assert binding["behavioral_admission"]["used_for_capture_authorization"] is False

    tampered = deepcopy(manifest)
    tampered["output"]["rows_sha256"] = _sha("5")
    with pytest.raises(ValueError, match="exact supplied rows"):
        capture.build_structured_action_capture_structural_binding(
            rows,
            tampered,
            gate,
            {},
            rows_sha256=rows_sha256,
            rollout_manifest_sha256=_sha("3"),
            gate_sha256=_sha("4"),
            protocol_sha256=protocol_sha256,
        )


def test_binding_rejects_incoherent_gate_without_making_pass_an_admission_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_live_validators(monkeypatch)
    rows = _rows()
    rows_sha256 = _sha("1")
    protocol_sha256 = _sha("2")
    manifest = _manifest(rows_sha256, protocol_sha256)
    gate = _gate(rows_sha256, protocol_sha256)
    gate["verdict"] = "pass"

    with pytest.raises(ValueError, match="verdict tuple"):
        capture.build_structured_action_capture_structural_binding(
            rows,
            manifest,
            gate,
            {},
            rows_sha256=rows_sha256,
            rollout_manifest_sha256=_sha("3"),
            gate_sha256=_sha("4"),
            protocol_sha256=protocol_sha256,
        )
