"""Build the frozen, label-blind complete-path connection-response bank.

This module deliberately consumes already-materialized checkpoint projections.  It
does not open artifacts, load a model, or score an outcome: it only validates the
60 sample-0 paths and attaches the five-way outcome at the final join boundary.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
import json
from math import acos, isfinite, pi
from typing import Any

from geoprobe.eval.relational_outcome_events import OUTCOME_CLASSES

_POLICY_CONTRACT = "artifact_only_cross_fitted"
_EXPECTED_COUNTS = {
    "DECEPTIVE": 34,
    "HONEST": 18,
    "SKIP": 4,
    "NO_ACTION": 0,
    "WRONG_WITHOUT_BASELINE_KNOWLEDGE": 4,
}
_DEFINED_STATUSES = frozenset({"supported", "ill_conditioned"})


class RelationalConnectionPathBankError(ValueError):
    """Raised when a frozen complete-path artifact is not exactly bindable."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelationalConnectionPathBankError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise RelationalConnectionPathBankError(f"{name} must be an array")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationalConnectionPathBankError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RelationalConnectionPathBankError(f"{name} must be an integer")
    return value


def _finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
        raise RelationalConnectionPathBankError(f"{name} must be finite")
    return float(value)


def _side_key(value: Mapping[str, Any], name: str) -> tuple[str, str, str]:
    return (
        _string(value.get("relation_name"), f"{name} relation_name"),
        _string(value.get("view"), f"{name} view"),
        _string(value.get("side"), f"{name} side"),
    )


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stable_sides(
    value: Any,
) -> tuple[tuple[tuple[str, str, str, int | None], ...], str]:
    """Normalize the small adapter seam for calibration's frozen side inventory."""
    if not isinstance(value, Mapping):
        raise RelationalConnectionPathBankError(
            "stable_relation_sides must be the full hash-bound inventory object"
        )
    inventory_hash = value.get("inventory_hash")
    value = value.get(
        "relation_sides", value.get("stable_relation_sides", value.get("inventory"))
    )
    rows = _sequence(value, "stable_relation_sides")
    result: list[tuple[str, str, str, int | None]] = []
    for raw in rows:
        if isinstance(raw, Mapping):
            relation, view, side = _side_key(raw, "stable relation side")
            rank_raw = raw.get("rank", raw.get("selected_rank"))
        elif isinstance(raw, (tuple, list)) and len(raw) in {3, 4}:
            relation, view, side = (_string(raw[0], "stable relation name"), _string(raw[1], "stable relation view"), _string(raw[2], "stable relation side"))
            rank_raw = raw[3] if len(raw) == 4 else None
        else:
            raise RelationalConnectionPathBankError("stable relation side is invalid")
        rank = None if rank_raw is None else _integer(rank_raw, "stable relation selected rank")
        if rank is not None and rank < 1:
            raise RelationalConnectionPathBankError("stable relation selected rank must be positive")
        result.append((relation, view, side, rank))
    if len(result) != 120 or len({item[:3] for item in result}) != 120:
        raise RelationalConnectionPathBankError("frozen stable relation-side inventory must contain exactly 120 sides")
    stable = tuple(sorted(result))
    return stable, _string(inventory_hash, "stable relation-side inventory hash")


def _event_index(outcome_join: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if outcome_join.get("schema_version") != 1 or outcome_join.get("kind") != "relational_partial_frame_outcome_join_report":
        raise RelationalConnectionPathBankError("outcome_join schema/kind is invalid")
    events: dict[str, Mapping[str, Any]] = {}
    for raw in _sequence(outcome_join.get("endpoint_events"), "outcome_join endpoint_events"):
        event = _mapping(raw, "endpoint event")
        event_id = _string(event.get("field_event_id"), "endpoint event field_event_id")
        if event_id in events:
            raise RelationalConnectionPathBankError("outcome_join duplicates endpoint event")
        if _string(event.get("outcome_class"), "endpoint event outcome_class") not in OUTCOME_CLASSES:
            raise RelationalConnectionPathBankError("endpoint event has an unsupported outcome class")
        events[event_id] = event
    return events


def _primary_rows(outcome_join: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], set[str]]:
    rows: list[Mapping[str, Any]] = []
    references: set[str] = set()
    seen: set[str] = set()
    for raw in _sequence(outcome_join.get("exact_realization_pairings"), "outcome_join exact_realization_pairings"):
        row = _mapping(raw, "exact realization pairing")
        pair_id = _string(row.get("primary_realization_pair_id"), "primary realization pair ID")
        if pair_id in seen:
            raise RelationalConnectionPathBankError("outcome_join duplicates a primary realization pair")
        seen.add(pair_id)
        references.add(_string(row.get("source_reference_id"), "pairing source_reference_id"))
        references.add(_string(row.get("target_reference_id"), "pairing target_reference_id"))
        rows.append(row)
    return rows, references


def _attempt_matches(
    attempt: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    match_target_reference: bool,
) -> bool:
    keys = [
        "edge_pair_id",
        "source_reference_id",
        "source_program",
        "target_program",
        "turn_index",
    ]
    if match_target_reference:
        keys.append("target_reference_id")
    return all(attempt.get(key) == row.get(key) for key in keys)


def _bound_attempts(
    checkpoint: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    expected_count: int,
    match_target_reference: bool,
) -> list[Mapping[str, Any]]:
    matches: list[Mapping[str, Any]] = []
    for raw in _sequence(checkpoint.get("attempts"), "scenario checkpoint attempts"):
        attempt = _mapping(raw, "scenario checkpoint attempt")
        if _attempt_matches(
            attempt, row, match_target_reference=match_target_reference
        ):
            if attempt.get("roster_class") != "primary" or attempt.get("attempt_kind", "forward_primary") != "forward_primary":
                raise RelationalConnectionPathBankError("complete-path primary realization has an invalid attempt class")
            matches.append(attempt)
    if len(matches) != expected_count:
        raise RelationalConnectionPathBankError(
            "semantic primary edge does not bind the required physical checkpoint realizations"
        )
    return matches


def _bound_attempt(
    checkpoint: Mapping[str, Any], row: Mapping[str, Any]
) -> Mapping[str, Any]:
    return _bound_attempts(
        checkpoint,
        row,
        expected_count=1,
        match_target_reference=True,
    )[0]


def _numeric_vector(direction: Mapping[str, Any], *, expected_rank: int | None) -> tuple[list[float] | None, str]:
    status = _string(direction.get("status"), "alignment status")
    if status not in _DEFINED_STATUSES:
        forbidden = ("principal_angle_cosines", "normalized_transported_projector_discrepancy", "normalized_polar_residual")
        if any(direction.get(name) not in (None, []) for name in forbidden):
            raise RelationalConnectionPathBankError("unsupported alignment retains a numeric invariant")
        return None, status
    cosines = _sequence(direction.get("principal_angle_cosines"), "alignment principal_angle_cosines")
    if not cosines or (expected_rank is not None and len(cosines) != expected_rank):
        raise RelationalConnectionPathBankError("alignment principal-angle rank violates frozen policy")
    angles: list[float] = []
    for raw in cosines:
        cosine = _finite(raw, "principal-angle cosine")
        if cosine < -1e-12 or cosine > 1.0 + 1e-12:
            raise RelationalConnectionPathBankError("principal-angle cosine is outside validated tolerance")
        angles.append((2.0 / pi) * acos(max(0.0, min(1.0, cosine))))
    vector = angles + [
        _finite(direction.get("normalized_transported_projector_discrepancy"), "normalized transported-projector discrepancy"),
        _finite(direction.get("normalized_polar_residual"), "normalized polar residual"),
    ]
    _finite(direction.get("polar_min_singular_value"), "polar minimum singular value")
    return vector, status


def _edge_signature(
    attempt: Mapping[str, Any], stable: tuple[tuple[str, str, str, int | None], ...]
) -> dict[tuple[str, str, str], tuple[list[float] | None, str]]:
    observed: dict[tuple[str, str, str], tuple[list[float] | None, str]] = {}
    expected = {item[:3] for item in stable}
    for raw in _sequence(attempt.get("relation_attempts"), "attempt relation_attempts"):
        relation = _mapping(raw, "relation attempt")
        key = _side_key(relation, "relation attempt")
        if key not in expected:
            continue
        if key in observed:
            raise RelationalConnectionPathBankError("attempt duplicates a relation side")
        if relation.get("policy_admitted") is False or relation.get("policy_status") not in (None, "admitted"):
            raise RelationalConnectionPathBankError("complete-path relation side is not admitted")
        direction = _mapping(relation.get("forward"), "relation forward alignment")
        rank = next((item[3] for item in stable if item[:3] == key), None)
        observed[key] = _numeric_vector(direction, expected_rank=rank)
    if set(observed) != expected:
        raise RelationalConnectionPathBankError("attempt relation-side inventory differs from the frozen stable inventory")
    return observed


def _component(
    vectors: Sequence[list[float] | None], statuses: Sequence[str], *, operation: str
) -> dict[str, Any]:
    defined = all(vector is not None for vector in vectors)
    if defined:
        length = len(vectors[0] or [])
        if any(len(vector or []) != length for vector in vectors):
            raise RelationalConnectionPathBankError("physical alignment vectors have inconsistent dimensions")
        if operation == "incoming":
            left, right = vectors[:3], vectors[3:]
            values = [sorted(vector[index] for vector in left if vector is not None)[1] - sorted(vector[index] for vector in right if vector is not None)[1] for index in range(length)]
        elif operation == "common":
            values = [((vectors[0] or [])[index] + (vectors[1] or [])[index]) / 2.0 for index in range(length)]
        else:
            values = [(vectors[1] or [])[index] - (vectors[0] or [])[index] for index in range(length)]
    else:
        values = None
    return {"vector": values, "defined": defined, "missing_mask": [vector is None for vector in vectors], "status_multiset": sorted(statuses)}


def _replay_group(checkpoint: Mapping[str, Any], group_id: str, references: set[str]) -> list[Mapping[str, Any]]:
    groups = [
        _mapping(raw, "replay group") for raw in _sequence(checkpoint.get("replay_attempts"), "scenario checkpoint replay_attempts")
        if _mapping(raw, "replay group").get("replay_group_id") == group_id
    ]
    if len(groups) != 1:
        raise RelationalConnectionPathBankError("incoming attempts must bind one exact replay group")
    group = groups[0]
    if group.get("attempt_kind") != "replay_control" or group.get("affirmative_claim_eligible") is not False:
        raise RelationalConnectionPathBankError("replay group contract is invalid")
    pairs = [_mapping(raw, "replay pair") for raw in _sequence(group.get("pairs"), "replay pairs")]
    if group.get("pair_count") != 3 or len(pairs) != 3:
        raise RelationalConnectionPathBankError("complete path requires one replay group with exactly three physical pairs")
    member_references = {
        _string(value, "replay member reference")
        for value in _sequence(
            group.get("member_reference_ids"), "replay member_reference_ids"
        )
    }
    if member_references != references:
        raise RelationalConnectionPathBankError(
            "replay members differ from the three incoming physical targets"
        )
    observed_pairs: set[tuple[str, str]] = set()
    for pair in pairs:
        source = _string(
            pair.get("source_reference_id"), "replay source_reference_id"
        )
        target = _string(
            pair.get("target_reference_id"), "replay target_reference_id"
        )
        if source not in references or target not in references or source == target:
            raise RelationalConnectionPathBankError("replay reference ID is absent from exact realization pairings")
        observed_pairs.add(tuple(sorted((source, target))))
    expected_pairs = {
        tuple(sorted((left, right)))
        for left_index, left in enumerate(sorted(references))
        for right in sorted(references)[left_index + 1 :]
    }
    if observed_pairs != expected_pairs:
        raise RelationalConnectionPathBankError(
            "replay pairs are not the exact three-member complete graph"
        )
    return pairs


def _label_blind_path(
    *, event: Mapping[str, Any], incoming: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    replay_pairs: Sequence[Mapping[str, Any]], outgoing_a: tuple[Mapping[str, Any], Mapping[str, Any]],
    outgoing_b: tuple[Mapping[str, Any], Mapping[str, Any]], stable: tuple[tuple[str, str, str, int | None], ...],
    inventory_hash: str,
) -> dict[str, Any]:
    incoming_signatures = [_edge_signature(attempt, stable) for _, attempt in incoming]
    replay_signatures = [_edge_signature(pair, stable) for pair in replay_pairs]
    aa_signature, ab_signature = _edge_signature(outgoing_a[1], stable), _edge_signature(outgoing_b[1], stable)
    components: list[dict[str, Any]] = []
    for relation, view, side, _rank in stable:
        key = (relation, view, side)
        incoming_values = [signature[key] for signature in incoming_signatures]
        replay_values = [signature[key] for signature in replay_signatures]
        aa_value, ab_value = aa_signature[key], ab_signature[key]
        raw_components = {
            "I": _component(
                [value[0] for value in incoming_values + replay_values],
                [value[1] for value in incoming_values + replay_values],
                operation="incoming",
            ),
            "C": _component(
                [aa_value[0], ab_value[0]],
                [aa_value[1], ab_value[1]],
                operation="common",
            ),
            "O": _component(
                [aa_value[0], ab_value[0]],
                [aa_value[1], ab_value[1]],
                operation="asymmetry",
            ),
        }
        components.append(
            {
                "relation_name": relation,
                "view": view,
                "side": side,
                "selected_rank": _rank,
                **{
                    name: {
                        "defined": payload["defined"],
                        "status": payload["status_multiset"],
                        "coordinates": payload["vector"],
                        "missing_mask": payload["missing_mask"],
                    }
                    for name, payload in raw_components.items()
                },
            }
        )
    outgoing_source_references = {
        str(outgoing_a[0]["source_reference_id"]),
        str(outgoing_b[0]["source_reference_id"]),
    }
    outgoing_source_sections = {
        _string(outgoing_a[1].get("source_section_sha256"), "AA source section"),
        _string(outgoing_b[1].get("source_section_sha256"), "AB source section"),
    }
    if len(outgoing_source_references) != 1 or len(outgoing_source_sections) != 1:
        raise RelationalConnectionPathBankError(
            "outgoing branches must share one source reference and compact section"
        )
    event_id = _string(event.get("field_event_id"), "event field_event_id")
    return {
        "scenario_id": _string(event.get("scenario_id"), "event scenario_id"),
        "event_id": event_id,
        "field_event_id": event_id,
        "family": _string(event.get("family"), "event family"),
        "fold": _string(event.get("fold"), "event fold"),
        "prefix_state_sha256": _string(event.get("prefix_state_sha256"), "event prefix_state_sha256"),
        "source_identities": {
            "incoming_source_field_event_ids": sorted(
                str(row["source_field_event_id"]) for row, _ in incoming
            ),
            "outgoing_source_field_event_id": str(outgoing_a[0]["source_field_event_id"]),
            "incoming_target_field_event_ids": sorted(
                str(row["target_field_event_id"]) for row, _ in incoming
            ),
        },
        "design_cell": {
            "true_status": deepcopy(event.get("true_status")),
            "desired_status": deepcopy(event.get("desired_status")),
            "baseline_knowledge_correct": event.get("knowledge_correct"),
        },
        "source_reference_id": next(iter(outgoing_source_references)),
        "source_section_sha256": next(iter(outgoing_source_sections)),
        "reference_bindings": {
            "incoming_semantic_primary_realization_pair_id": str(
                incoming[0][0]["primary_realization_pair_id"]
            ),
            "incoming_physical_target_reference_ids": sorted(
                str(attempt["target_reference_id"]) for _, attempt in incoming
            ),
            "outgoing_aa_primary_realization_pair_id": str(outgoing_a[0]["primary_realization_pair_id"]),
            "outgoing_ab_primary_realization_pair_id": str(outgoing_b[0]["primary_realization_pair_id"]),
            "incoming_source_reference_id": str(incoming[0][0]["source_reference_id"]),
            "incoming_target_reference_ids": sorted(
                str(attempt["target_reference_id"]) for _, attempt in incoming
            ),
            "replay_group_id": str(incoming[0][1]["matched_replay_group_id"]),
            "replay_reference_pairs": [
                {"source_reference_id": str(pair["source_reference_id"]), "target_reference_id": str(pair["target_reference_id"])}
                for pair in replay_pairs
            ],
        },
        "signature": {
            "inventory_hash": inventory_hash,
            "relation_sides": components,
        },
    }


def build_complete_path_bank(
    *, outcome_join: Mapping[str, Any], checkpoints_by_scenario: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    stable_relation_sides: Any, confirmatory: bool = False,
) -> dict[str, Any]:
    """Validate and extract the exact sample-0 60-path artifact-only cohort."""
    if confirmatory:
        raise RelationalConnectionPathBankError("artifact-only cross-fitted paths cannot be used in confirmatory mode")
    stable, inventory_hash = _stable_sides(stable_relation_sides)
    events = _event_index(outcome_join)
    rows, _all_references = _primary_rows(outcome_join)
    paths: list[dict[str, Any]] = []
    seen_scenarios: set[str] = set()
    checkpoint_values: Iterable[Mapping[str, Any]] = (
        checkpoints_by_scenario.values()
        if isinstance(checkpoints_by_scenario, Mapping)
        else checkpoints_by_scenario
    )
    for raw_checkpoint in checkpoint_values:
        checkpoint = _mapping(raw_checkpoint, "scenario checkpoint")
        scenario_id = _string(
            checkpoint.get("scenario_id"), "scenario checkpoint scenario_id"
        )
        if scenario_id in seen_scenarios:
            raise RelationalConnectionPathBankError("duplicate scenario checkpoint")
        seen_scenarios.add(scenario_id)
        scenario_rows = [row for row in rows if row.get("scenario_id") == scenario_id and row.get("sample_index") == 0]
        incoming_rows = [row for row in scenario_rows if row.get("source_program") == "NN" and row.get("target_program") == "AN" and row.get("turn_index") == 1]
        outgoing_a_rows = [row for row in scenario_rows if row.get("source_program") == "AN" and row.get("target_program") == "AA" and row.get("turn_index") == 2]
        outgoing_b_rows = [row for row in scenario_rows if row.get("source_program") == "AN" and row.get("target_program") == "AB" and row.get("turn_index") == 2]
        if len(incoming_rows) != 1 or len(outgoing_a_rows) != 1 or len(outgoing_b_rows) != 1:
            raise RelationalConnectionPathBankError("each sample-0 scenario requires 1 semantic NN→AN, 1 AN→AA, and 1 AN→AB path")
        scenario_family = next(row.get("family") for row in scenario_rows)
        scenario_fold = next(row.get("fold") for row in scenario_rows)
        if checkpoint.get("family") not in (None, scenario_family) or checkpoint.get(
            "heldout_family_fold"
        ) not in (None, scenario_fold):
            raise RelationalConnectionPathBankError("checkpoint family/fold identity disagrees with exact realization pairings")
        checkpoint_primary = [
            _mapping(raw, "scenario checkpoint attempt")
            for raw in _sequence(checkpoint.get("attempts"), "scenario checkpoint attempts")
            if _mapping(raw, "scenario checkpoint attempt").get("roster_class") == "primary"
            and _mapping(raw, "scenario checkpoint attempt").get("sample_index") == 0
        ]
        checkpoint_path_counts = Counter(
            (attempt.get("source_program"), attempt.get("target_program"), attempt.get("turn_index"))
            for attempt in checkpoint_primary
        )
        if (
            checkpoint_path_counts[("NN", "AN", 1)] != 3
            or checkpoint_path_counts[("AN", "AA", 2)] != 1
            or checkpoint_path_counts[("AN", "AB", 2)] != 1
        ):
            raise RelationalConnectionPathBankError("checkpoint does not contain the exact 3/1/1 complete-path primary attempts")
        incoming_attempts = _bound_attempts(
            checkpoint,
            incoming_rows[0],
            expected_count=3,
            match_target_reference=False,
        )
        incoming = [(incoming_rows[0], attempt) for attempt in incoming_attempts]
        if len({str(attempt["source_reference_id"]) for _, attempt in incoming}) != 1 or len({str(attempt["target_reference_id"]) for _, attempt in incoming}) != 3:
            raise RelationalConnectionPathBankError("incoming NN→AN edge must have one source and three clone target realizations")
        group_ids = {attempt.get("matched_replay_group_id") for _, attempt in incoming}
        if len(group_ids) != 1 or None in group_ids or any(attempt.get("replay_match_status") != "matched_exact_target_endpoint" for _, attempt in incoming):
            raise RelationalConnectionPathBankError("three incoming primary attempts must share one matched replay group")
        incoming_target_references = {
            str(attempt["target_reference_id"]) for _, attempt in incoming
        }
        replay_pairs = _replay_group(
            checkpoint,
            _string(next(iter(group_ids)), "matched replay group ID"),
            incoming_target_references,
        )
        outgoing_a, outgoing_b = (outgoing_a_rows[0], _bound_attempt(checkpoint, outgoing_a_rows[0])), (outgoing_b_rows[0], _bound_attempt(checkpoint, outgoing_b_rows[0]))
        incoming_endpoint_ids = {
            str(row.get("target_field_event_id")) for row, _ in incoming
        }
        outgoing_source_ids = {
            str(outgoing_a[0].get("source_field_event_id")),
            str(outgoing_b[0].get("source_field_event_id")),
        }
        if (
            len(incoming_endpoint_ids) != 1
            or len(outgoing_source_ids) != 1
            or not incoming_endpoint_ids <= set(events)
            or not outgoing_source_ids <= set(events)
        ):
            raise RelationalConnectionPathBankError(
                "complete path does not bind its prior and later AN events"
            )
        prior_event = events[next(iter(incoming_endpoint_ids))]
        event = events[next(iter(outgoing_source_ids))]
        if (
            prior_event.get("scenario_id") != scenario_id
            or prior_event.get("turn_index") != 1
            or tuple(prior_event.get("intervention_history", ())) != ("A",)
        ):
            raise RelationalConnectionPathBankError(
                "incoming path endpoint is not the sample-0 first-A event"
            )
        if event.get("scenario_id") != scenario_id or event.get("turn_index") != 2 or tuple(event.get("intervention_history", ())) != ("A", "N"):
            raise RelationalConnectionPathBankError("complete path endpoint is not the sample-0 AN turn-2 outcome")
        outgoing_source_prefixes = {
            _string(
                outgoing_a[0].get("source_prefix_state_sha256"),
                "AA source prefix",
            ),
            _string(
                outgoing_b[0].get("source_prefix_state_sha256"),
                "AB source prefix",
            ),
        }
        if outgoing_source_prefixes != {
            _string(event.get("prefix_state_sha256"), "AN event prefix")
        }:
            raise RelationalConnectionPathBankError(
                "outgoing source prefixes differ from the predicted AN event"
            )
        if any(row.get("family") != event.get("family") or row.get("fold") != event.get("fold") for row in scenario_rows):
            raise RelationalConnectionPathBankError("complete-path event family/fold identity disagrees with exact realization pairings")
        label_blind = _label_blind_path(
            event=event,
            incoming=incoming,
            replay_pairs=replay_pairs,
            outgoing_a=outgoing_a,
            outgoing_b=outgoing_b,
            stable=stable,
            inventory_hash=inventory_hash,
        )
        outcome = _string(event.get("outcome_class"), "complete-path outcome class")
        paths.append(label_blind | {"outcome_class": outcome, "class_counts": {name: int(name == outcome) for name in OUTCOME_CLASSES}})
    if len(paths) != 60:
        raise RelationalConnectionPathBankError("complete-path cohort must contain exactly 60 sample-0 scenarios")
    counts = Counter(str(path["outcome_class"]) for path in paths)
    actual_counts = {name: counts[name] for name in OUTCOME_CLASSES}
    if actual_counts != _EXPECTED_COUNTS:
        raise RelationalConnectionPathBankError("complete-path five-way outcome counts differ from the frozen 34D/18H/4SKIP/0NO_ACTION/4error cohort")
    result = {
        "schema_version": 1,
        "kind": "relational_complete_path_bank",
        "policy_contract": _POLICY_CONTRACT,
        "confirmatory": False,
        "inventory_hash": inventory_hash,
        "paths": sorted(paths, key=lambda path: str(path["event_id"])),
        "coverage": {"scenario_count": 60, "path_count": 60, "relation_side_count": 120, "class_counts": actual_counts},
    }
    result["bank_sha256"] = _canonical_sha256(result)
    return result


__all__ = ["RelationalConnectionPathBankError", "build_complete_path_bank"]
