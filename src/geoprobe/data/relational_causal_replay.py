"""Exact structured-action status prefixes for independent causal replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from geoprobe.models.relational_structured_action import int32_token_sha256
from geoprobe.models.relational_structured_action_rollout import (
    validate_relational_structured_action_row,
)


class RelationalCausalReplayError(ValueError):
    """A rollout row or vector join violates exact-prefix replay identity."""


@dataclass(frozen=True, slots=True)
class RelationalCausalReplayEvent:
    event_id: str
    scenario_id: str
    family: str
    turn_index: int
    true_status: str
    desired_status: str
    knowledge_correct: bool
    knowledge_status: str
    prefix_token_ids: tuple[int, ...]
    prefix_token_ids_sha256: str
    rng_seed: int
    historical_raw_token_id: int
    historical_raw_decoded_exact: str
    historical_mapped_action: str
    source_conversation_ids: tuple[str, ...]
    source_row_sha256s: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RelationalCausalReplayPlanEvent:
    replay: RelationalCausalReplayEvent
    root_id: str
    family_fold: str
    vector_tensor_row_index: int


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationalCausalReplayError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RelationalCausalReplayError(f"{label} must be a nonnegative integer")
    return value


def _sha(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RelationalCausalReplayError(f"{label} must be a lowercase SHA-256")
    return text


def _status_records(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    records = row.get("assistant_action_turn_records")
    if not isinstance(records, list) or len(records) != 4:
        raise RelationalCausalReplayError("structured row lacks four action turns")
    result: list[Mapping[str, Any]] = []
    for turn_index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("turn_index") != turn_index:
            raise RelationalCausalReplayError("structured action turns are malformed")
        fields = record.get("fields")
        if not isinstance(fields, list) or len(fields) != 2:
            raise RelationalCausalReplayError("structured action turn lacks two fields")
        status = fields[0]
        if not isinstance(status, Mapping) or status.get("field_name") != "status":
            raise RelationalCausalReplayError("status is not the first structured field")
        result.append(status)
    return tuple(result)


def _event_from_row(
    row: Mapping[str, Any], status: Mapping[str, Any], turn_index: int
) -> RelationalCausalReplayEvent:
    tokens = row.get("token_ids")
    if not isinstance(tokens, list) or not tokens:
        raise RelationalCausalReplayError("structured row has no token IDs")
    sampled = _integer(status.get("sampled_token_index"), "sampled-token index")
    anchor = _integer(status.get("anchor_token_index"), "anchor-token index")
    if sampled != anchor + 1 or sampled >= len(tokens):
        raise RelationalCausalReplayError("status anchor does not immediately precede its token")
    prefix = tuple(_integer(token, "prefix token ID") for token in tokens[:sampled])
    prefix_sha = int32_token_sha256(prefix)
    if prefix_sha != _sha(
        status.get("anchor_prefix_token_ids_sha256"), "recorded prefix token hash"
    ):
        raise RelationalCausalReplayError("status prefix tokens differ from their recorded hash")
    raw_token_id = _integer(status.get("raw_token_id"), "historical raw token ID")
    if int(tokens[sampled]) != raw_token_id:
        raise RelationalCausalReplayError("historical status token differs from the transcript")
    true_status = _string(row.get("true_status"), "true status")
    desired_status = _string(row.get("desired_status"), "desired status")
    if true_status not in {"PASS", "FAIL"} or desired_status not in {"PASS", "FAIL"}:
        raise RelationalCausalReplayError("structured row has invalid status semantics")
    knowledge_correct = row.get("knowledge_correct")
    if not isinstance(knowledge_correct, bool):
        raise RelationalCausalReplayError(
            "structured row knowledge-correct flag must be Boolean"
        )
    knowledge_status = _string(row.get("knowledge_status"), "knowledge status")
    if knowledge_status not in {"PASS", "FAIL"}:
        raise RelationalCausalReplayError("structured row has invalid knowledge status")
    if (knowledge_status == true_status) != knowledge_correct:
        raise RelationalCausalReplayError(
            "structured row knowledge status disagrees with its correctness flag"
        )
    return RelationalCausalReplayEvent(
        event_id=_string(status.get("field_event_id"), "field-event ID"),
        scenario_id=_string(row.get("scenario_id"), "scenario ID"),
        family=_string(row.get("family"), "family"),
        turn_index=turn_index,
        true_status=true_status,
        desired_status=desired_status,
        knowledge_correct=knowledge_correct,
        knowledge_status=knowledge_status,
        prefix_token_ids=prefix,
        prefix_token_ids_sha256=prefix_sha,
        rng_seed=_integer(status.get("rng_seed"), "status RNG seed"),
        historical_raw_token_id=raw_token_id,
        historical_raw_decoded_exact=_string(
            status.get("raw_decoded_exact"), "historical decoded token"
        ),
        historical_mapped_action=_string(
            status.get("mapped_action"), "historical mapped action"
        ),
        source_conversation_ids=(
            _string(row.get("conversation_id"), "conversation ID"),
        ),
        source_row_sha256s=(_sha(row.get("row_sha256"), "source row hash"),),
    )


def _event_signature(event: RelationalCausalReplayEvent) -> tuple[object, ...]:
    return (
        event.event_id,
        event.scenario_id,
        event.family,
        event.turn_index,
        event.true_status,
        event.desired_status,
        event.knowledge_correct,
        event.knowledge_status,
        event.prefix_token_ids,
        event.prefix_token_ids_sha256,
        event.rng_seed,
        event.historical_raw_token_id,
        event.historical_raw_decoded_exact,
        event.historical_mapped_action,
    )


def build_relational_causal_replay_inventory(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[RelationalCausalReplayEvent, ...]:
    """Collapse exact structured-row clones into one immutable status draw event."""
    if not rows:
        raise RelationalCausalReplayError("causal replay requires structured rollout rows")
    grouped: dict[str, list[RelationalCausalReplayEvent]] = {}
    for row in rows:
        validate_relational_structured_action_row(row)
        for turn_index, status in enumerate(_status_records(row)):
            event = _event_from_row(row, status, turn_index)
            grouped.setdefault(event.event_id, []).append(event)
    result: list[RelationalCausalReplayEvent] = []
    for event_id, copies in sorted(grouped.items()):
        signatures = {_event_signature(event) for event in copies}
        if len(signatures) != 1:
            raise RelationalCausalReplayError(
                f"structured clones disagree for status event {event_id}"
            )
        first = copies[0]
        result.append(
            RelationalCausalReplayEvent(
                event_id=first.event_id,
                scenario_id=first.scenario_id,
                family=first.family,
                turn_index=first.turn_index,
                true_status=first.true_status,
                desired_status=first.desired_status,
                knowledge_correct=first.knowledge_correct,
                knowledge_status=first.knowledge_status,
                prefix_token_ids=first.prefix_token_ids,
                prefix_token_ids_sha256=first.prefix_token_ids_sha256,
                rng_seed=first.rng_seed,
                historical_raw_token_id=first.historical_raw_token_id,
                historical_raw_decoded_exact=first.historical_raw_decoded_exact,
                historical_mapped_action=first.historical_mapped_action,
                source_conversation_ids=tuple(
                    sorted({value for event in copies for value in event.source_conversation_ids})
                ),
                source_row_sha256s=tuple(
                    sorted({value for event in copies for value in event.source_row_sha256s})
                ),
            )
        )
    return tuple(result)


def join_relational_causal_replay_plan(
    inventory: Sequence[RelationalCausalReplayEvent],
    vector_rows: Sequence[Mapping[str, Any]],
) -> tuple[RelationalCausalReplayPlanEvent, ...]:
    """Join label-free vector roots to every exact replay event they represent."""
    events = {event.event_id: event for event in inventory}
    if len(events) != len(inventory):
        raise RelationalCausalReplayError("replay inventory contains duplicate event IDs")
    planned: dict[str, RelationalCausalReplayPlanEvent] = {}
    for row in vector_rows:
        root_id = _sha(row.get("root_id"), "vector root ID")
        fold = _string(row.get("family_fold"), "vector family fold")
        family = _string(row.get("family"), "vector family")
        turn_index = _integer(row.get("turn_index"), "vector turn index")
        prefix_sha = _sha(
            row.get("prefix_token_ids_sha256"), "vector prefix-token hash"
        )
        tensor_index = _integer(
            row.get("tensor_row_index"), "vector tensor-row index"
        )
        event_ids = row.get("event_ids")
        if (
            not isinstance(event_ids, (list, tuple))
            or not event_ids
        ):
            raise RelationalCausalReplayError("vector root has no replay event IDs")
        for raw_event_id in event_ids:
            event_id = _string(raw_event_id, "vector field-event ID")
            event = events.get(event_id)
            if event is None:
                raise RelationalCausalReplayError(
                    f"vector event {event_id} is absent from the rollout inventory"
                )
            if (
                event.family != family
                or event.turn_index != turn_index
                or event.prefix_token_ids_sha256 != prefix_sha
            ):
                raise RelationalCausalReplayError(
                    f"vector identity disagrees with replay event {event_id}"
                )
            plan_event = RelationalCausalReplayPlanEvent(
                replay=event,
                root_id=root_id,
                family_fold=fold,
                vector_tensor_row_index=tensor_index,
            )
            prior = planned.get(event_id)
            if prior is not None and prior != plan_event:
                raise RelationalCausalReplayError(
                    f"replay event {event_id} maps to multiple vector roots"
                )
            planned[event_id] = plan_event
    if not planned:
        raise RelationalCausalReplayError("vector bank produced no causal replay plan")
    return tuple(planned[event_id] for event_id in sorted(planned))


def replay_inventory_by_event_id(
    inventory: Sequence[RelationalCausalReplayEvent],
) -> Mapping[str, RelationalCausalReplayEvent]:
    result = {event.event_id: event for event in inventory}
    if len(result) != len(inventory):
        raise RelationalCausalReplayError("replay inventory contains duplicate event IDs")
    return MappingProxyType(result)


__all__ = [
    "RelationalCausalReplayError",
    "RelationalCausalReplayEvent",
    "RelationalCausalReplayPlanEvent",
    "build_relational_causal_replay_inventory",
    "join_relational_causal_replay_plan",
    "replay_inventory_by_event_id",
]
