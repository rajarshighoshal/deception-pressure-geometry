"""Structural source binding for exploratory schema-v2 relational capture."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from geoprobe.eval.relational_structured_action_gate import GATE_REPORT_KIND
from geoprobe.models.relational_structured_action import (
    EOT_TOKEN_ID,
    FIELD_ORDER,
    PROTOCOL_ID,
    canonical_json_sha256,
    validate_structured_action_protocol,
)
from geoprobe.models.relational_structured_action_rollout import (
    ORIGIN_ENVIRONMENT_EOT,
    validate_relational_structured_action_row,
    validate_relational_structured_action_scenario_rows,
)

STRUCTURED_CAPTURE_PLAN_KIND = "relational_structured_action_capture_plan"
STRUCTURED_CAPTURE_SCOPE = "exploratory_joint_relational_gdl"
_SOURCE_KIND = "relational_structured_action_capture_source"
_BINDING_KIND = "relational_structured_action_capture_structural_binding"


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256") from error
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _field_binding(field: Mapping[str, Any], *, row_name: str, turn_index: int) -> dict[str, Any]:
    field_name = field.get("field_name")
    if field_name not in FIELD_ORDER:
        raise ValueError(f"{row_name}: turn {turn_index} has an unknown action field")
    for key in (
        "field_event_id",
        "rng_domain",
        "rng_stream_id",
        "rng_seed",
        "anchor_token_index",
        "sampled_token_index",
        "raw_token_id",
        "raw_token_origin",
    ):
        if key not in field:
            raise ValueError(f"{row_name}: action field lacks {key}")
    return {
        "field_name": str(field_name),
        "field_event_id": str(field["field_event_id"]),
        "rng_domain": str(field["rng_domain"]),
        "rng_stream_id": str(field["rng_stream_id"]),
        "rng_seed": int(field["rng_seed"]),
        "anchor_token_index": int(field["anchor_token_index"]),
        "sampled_token_index": int(field["sampled_token_index"]),
        "raw_token_id": int(field["raw_token_id"]),
        "raw_token_origin": str(field["raw_token_origin"]),
    }


def _expected_anchors_and_spans(
    row: Mapping[str, Any], *, row_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tokens = row["token_ids"]
    expected_anchors: list[dict[str, Any]] = []
    expected_spans: list[dict[str, Any]] = []
    turn_bindings: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(row["assistant_action_turn_records"]):
        turn = _require_mapping(turn, f"{row_name}: turn {turn_index}")
        fields = turn.get("fields")
        if not isinstance(fields, list) or [field.get("field_name") for field in fields] != list(FIELD_ORDER):
            raise ValueError(f"{row_name}: turn {turn_index} fields are not in frozen order")
        eot = _require_mapping(turn.get("environment_eot"), f"{row_name}: turn {turn_index} EOT")
        eot_index = eot.get("token_index")
        if (
            not isinstance(eot_index, int)
            or isinstance(eot_index, bool)
            or eot_index < 0
            or eot_index >= len(tokens)
            or tokens[eot_index] != EOT_TOKEN_ID
            or eot.get("token_id") != EOT_TOKEN_ID
            or row["token_origin_details"][eot_index] != ORIGIN_ENVIRONMENT_EOT
        ):
            raise ValueError(f"{row_name}: turn {turn_index} lacks the exact environment EOT")
        field_bindings = [
            _field_binding(_require_mapping(field, f"{row_name}: field"), row_name=row_name, turn_index=turn_index)
            for field in fields
        ]
        for field in field_bindings:
            anchor_index = field["anchor_token_index"]
            sampled_index = field["sampled_token_index"]
            if not 0 <= anchor_index < len(tokens) or not 0 <= sampled_index < len(tokens):
                raise ValueError(f"{row_name}: action anchor lies outside the transcript")
            expected_anchors.append({
                "turn_index": turn_index,
                "field_name": field["field_name"],
                "field_event_id": field["field_event_id"],
                "anchor_token_index": anchor_index,
                "anchor_token_id": tokens[anchor_index],
                "sampled_token_index": sampled_index,
                "anchor_semantic_position": (
                    "immediately_before_status_sample"
                    if field["field_name"] == "status"
                    else "immediately_before_caveat_sample_status_visible"
                ),
            })
            expected_spans.append({
                "turn_index": turn_index,
                "field_name": field["field_name"],
                "token_start": sampled_index,
                "token_end": sampled_index + 1,
                "token_origin": field["raw_token_origin"],
            })
        turn_bindings.append({
            "turn_index": turn_index,
            "turn_event_id": str(turn["turn_event_id"]),
            "transcript_before_action_token_count": int(
                turn["transcript_before_action_token_count"]
            ),
            "transcript_before_action_token_ids_sha256": str(
                turn["transcript_before_action_token_ids_sha256"]
            ),
            "environment_eot": deepcopy(dict(eot)),
            "fields": field_bindings,
        })
    if row.get("action_anchors") != expected_anchors:
        raise ValueError(f"{row_name}: action anchors differ from the exact action records")
    if row.get("action_token_spans") != expected_spans:
        raise ValueError(f"{row_name}: action spans differ from the exact action records")
    return expected_anchors, expected_spans, turn_bindings


def validate_structured_action_capture_source(
    rows: Sequence[dict[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and describe the complete, non-behaviorally-gated v2 source bank."""
    validate_structured_action_protocol(protocol)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("structured-action capture rows must be a sequence")
    if len(rows) != 600:
        raise ValueError("structured-action capture requires exactly 600 rows")

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conversation_ids: set[str] = set()
    row_bindings: list[dict[str, Any]] = []
    token_lengths: list[int] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"structured-action capture row {row_index} must be an object")
        try:
            validate_relational_structured_action_row(row, protocol)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"structured-action capture row {row_index} is structurally invalid") from error
        conversation_id = str(row["conversation_id"])
        scenario_id = str(row["scenario_id"])
        if not conversation_id or conversation_id in conversation_ids or not scenario_id:
            raise ValueError("structured-action capture conversation/scenario identities are invalid")
        conversation_ids.add(conversation_id)
        by_scenario[scenario_id].append(row)
        anchors, spans, turns = _expected_anchors_and_spans(row, row_name=conversation_id)
        final_eot = {
            "token_id": row.get("assistant_eot_token_id"),
            "stop_token_id": row.get("stop_token_id"),
            "token_index": row.get("stop_token_index"),
            "stop_reason": row.get("stop_reason"),
            "truncated": row.get("truncated"),
        }
        if final_eot != {
            "token_id": EOT_TOKEN_ID,
            "stop_token_id": EOT_TOKEN_ID,
            "token_index": len(row["token_ids"]) - 1,
            "stop_reason": "environment_eot",
            "truncated": False,
        }:
            raise ValueError(f"{conversation_id}: row does not end at the exact environment EOT")
        token_lengths.append(len(row["token_ids"]))
        token_annotations = {
            "token_origins": row["token_origins"],
            "token_origin_details": row["token_origin_details"],
            "token_role_ids": row["token_role_ids"],
            "token_turn_ids": row["token_turn_ids"],
            "token_message_ids": row["token_message_ids"],
            "token_span_flags": row["token_span_flags"],
        }
        action_structure = {
            "action_anchors": anchors,
            "action_token_spans": spans,
            "turns": turns,
            "environment_eot": final_eot,
        }
        row_bindings.append({
            "row_index": row_index,
            "conversation_id": conversation_id,
            "scenario_id": scenario_id,
            "spec_sha256": _require_sha256(row["spec_sha256"], f"{conversation_id}: spec_sha256"),
            "source_spec_sha256": _require_sha256(
                row["source_spec_sha256"], f"{conversation_id}: source_spec_sha256"
            ),
            "row_sha256": _require_sha256(row["row_sha256"], f"{conversation_id}: row_sha256"),
            "token_sha256": _require_sha256(row["token_sha256"], f"{conversation_id}: token_sha256"),
            "token_length": len(row["token_ids"]),
            "token_annotations_sha256": canonical_json_sha256(token_annotations),
            "action_structure_sha256": canonical_json_sha256(action_structure),
        })

    if len(by_scenario) != 60 or set(map(len, by_scenario.values())) != {10}:
        raise ValueError("structured-action capture requires exactly 60 scenarios with ten rows each")
    for scenario_id, scenario_rows in by_scenario.items():
        try:
            validate_relational_structured_action_scenario_rows(scenario_rows, protocol)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"structured-action capture scenario {scenario_id} is structurally invalid") from error
    scenario_counts = Counter(binding["scenario_id"] for binding in row_bindings)
    return {
        "schema_version": 2,
        "kind": _SOURCE_KIND,
        "scope": STRUCTURED_CAPTURE_SCOPE,
        "protocol_id": PROTOCOL_ID,
        "row_count": len(rows),
        "scenario_count": len(by_scenario),
        "scenario_row_counts": dict(sorted(scenario_counts.items())),
        "token_lengths": token_lengths,
        "rows": row_bindings,
    }


def _validate_rollout_manifest(
    rollout_manifest: Mapping[str, Any], *, rows_sha256: str, protocol_sha256: str
) -> dict[str, Any]:
    if (
        rollout_manifest.get("schema_version") != 2
        or rollout_manifest.get("kind") != "relational_structured_action_rollout_manifest"
        or rollout_manifest.get("status") != "success"
    ):
        raise ValueError("structured-action rollout manifest is not a successful v2 artifact")
    output = _require_mapping(rollout_manifest.get("output"), "rollout manifest output")
    if output.get("rows_sha256") != rows_sha256:
        raise ValueError("rollout manifest does not bind the exact supplied rows")
    inputs = _require_mapping(rollout_manifest.get("inputs"), "rollout manifest inputs")
    if inputs.get("protocol_sha256") != protocol_sha256:
        raise ValueError("rollout manifest does not bind the exact supplied protocol")
    contract = _require_mapping(rollout_manifest.get("rollout_contract"), "rollout contract")
    if contract.get("schema_version") != 2 or contract.get("kind") != "relational_structured_action_rollout_contract":
        raise ValueError("rollout manifest lacks the frozen v2 rollout contract")
    body = {key: value for key, value in contract.items() if key not in {"schema_version", "kind", "contract_sha256"}}
    if contract.get("contract_sha256") != canonical_json_sha256(body):
        raise ValueError("rollout contract aggregate digest is invalid")
    if contract.get("protocol_sha256") != protocol_sha256:
        raise ValueError("rollout contract does not bind the exact supplied protocol")
    model_artifact_sha256 = _require_sha256(
        contract.get("model_artifact_sha256"), "rollout contract model_artifact_sha256"
    )
    return {
        "kind": str(rollout_manifest["kind"]),
        "status": str(rollout_manifest["status"]),
        "rows_sha256": rows_sha256,
        "protocol_sha256": protocol_sha256,
        "rollout_contract_sha256": str(contract["contract_sha256"]),
        "model_artifact_sha256": model_artifact_sha256,
    }


def _validate_gate(
    gate: Mapping[str, Any], *, rows_sha256: str, protocol_sha256: str
) -> dict[str, Any]:
    if (
        gate.get("schema_version") != 2
        or gate.get("kind") != GATE_REPORT_KIND
        or gate.get("protocol_id") != PROTOCOL_ID
        or gate.get("protocol_sha256") != protocol_sha256
    ):
        raise ValueError("structured-action gate is not a v2 report for the supplied protocol")
    inputs = _require_mapping(gate.get("inputs"), "structured-action gate inputs")
    rows_input = _require_mapping(inputs.get("rows"), "structured-action gate rows input")
    protocol_input = _require_mapping(inputs.get("protocol"), "structured-action gate protocol input")
    if rows_input.get("sha256") != rows_sha256 or protocol_input.get("sha256") != protocol_sha256:
        raise ValueError("structured-action gate does not bind the exact supplied rows/protocol")
    scope = _require_mapping(gate.get("scope"), "structured-action gate scope")
    development_mode = scope.get("development_mode")
    confirmatory = scope.get("confirmatory")
    primary_passed = gate.get("primary_gate_passed")
    capture_eligible = gate.get("capture_eligible")
    verdict = gate.get("verdict")
    if (
        development_mode is not False
        or confirmatory is not True
        or any(
            scope.get(key) is not expected
            for key, expected in {
                "row_selection_performed": False,
                "row_dropping_performed": False,
                "prose_or_sidecar_borrowing_performed": False,
                "all_planned_events_retained": True,
            }.items()
        )
        or not isinstance(primary_passed, bool)
        or not isinstance(capture_eligible, bool)
        or verdict not in {"pass", "not-found-under-this-task"}
        or (verdict == "pass") is not primary_passed
        or capture_eligible is not primary_passed
        or gate.get("capture_allowed") is not False
        or gate.get("capture_authorization")
        != "requires_separate_paid_capture_contract"
    ):
        raise ValueError("structured-action gate verdict tuple is incoherent")
    return {
        "kind": str(gate["kind"]),
        "protocol_id": str(gate["protocol_id"]),
        "protocol_sha256": protocol_sha256,
        "behavioral_verdict": verdict,
        "primary_gate_passed": primary_passed,
        "capture_eligible": capture_eligible,
        "capture_allowed": False,
    }


def build_structured_action_capture_structural_binding(
    rows: Sequence[dict[str, Any]],
    rollout_manifest: Mapping[str, Any],
    gate: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    rows_sha256: str,
    rollout_manifest_sha256: str,
    gate_sha256: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Return an exact structural binding without making a behavioral admission decision."""
    rows_sha256 = _require_sha256(rows_sha256, "rows_sha256")
    rollout_manifest_sha256 = _require_sha256(
        rollout_manifest_sha256, "rollout_manifest_sha256"
    )
    gate_sha256 = _require_sha256(gate_sha256, "gate_sha256")
    protocol_sha256 = _require_sha256(protocol_sha256, "protocol_sha256")
    source = validate_structured_action_capture_source(rows, protocol)
    manifest_binding = _validate_rollout_manifest(
        rollout_manifest, rows_sha256=rows_sha256, protocol_sha256=protocol_sha256
    )
    gate_binding = _validate_gate(gate, rows_sha256=rows_sha256, protocol_sha256=protocol_sha256)
    body = {
        "schema_version": 2,
        "kind": _BINDING_KIND,
        "plan_kind": STRUCTURED_CAPTURE_PLAN_KIND,
        "scope": STRUCTURED_CAPTURE_SCOPE,
        "inputs": {
            "rows_sha256": rows_sha256,
            "rollout_manifest_sha256": rollout_manifest_sha256,
            "gate_sha256": gate_sha256,
            "protocol_sha256": protocol_sha256,
        },
        "source": source,
        "rollout_manifest": manifest_binding,
        "gate": gate_binding,
        "behavioral_admission": {
            "used_for_capture_authorization": False,
            "reason": "exploratory_scope_binds_behavioral_gate_descriptively_only",
        },
    }
    return {**body, "binding_sha256": canonical_json_sha256(body)}


__all__ = [
    "STRUCTURED_CAPTURE_PLAN_KIND",
    "STRUCTURED_CAPTURE_SCOPE",
    "build_structured_action_capture_structural_binding",
    "validate_structured_action_capture_source",
]
