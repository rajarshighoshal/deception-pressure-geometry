"""Treatment-blind token grammar for canonical status-prefix connections."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from numbers import Integral
from typing import Any

from geoprobe.geometry.relational_partial_connection import (
    CanonicalTokenCore,
    PartialConnectionError,
    PartialIsometryEdge,
    build_canonical_token_core,
    build_partial_isometry_edge,
)


CANONICAL_KEY_SCHEMA = "relational-structured-action-v2/message-local-token/v1"
STATUS_PREFIX_TOKEN_IDS = (2583, 25)
STATUS_ANCHOR_TOKEN_ID = 25
STATUS_SPAN_FLAG = 16
ASSISTANT_ROLE_ID = 2
STATUS_PREFIX_ORIGIN = "environment"
STATUS_PREFIX_DETAIL = "environment_status_prefix"
SAMPLED_ACTION_DETAILS = frozenset(
    {
        "model_sampled_status_action",
        "model_sampled_caveat_action",
    }
)


class RelationalPartialGrammarError(PartialConnectionError):
    """Raised when raw status-prefix annotations violate the frozen grammar."""


def _contrast_mapping() -> dict[tuple[str, str, int], tuple[int, ...]]:
    directed: dict[tuple[str, str, int], tuple[int, ...]] = {}

    def bind(
        left: str,
        right: str,
        turn_index: int,
        excluded: tuple[int, ...],
    ) -> None:
        directed[(left, right, turn_index)] = excluded
        directed[(right, left, turn_index)] = excluded

    for target in ("AN", "BA", "D2N"):
        bind("NN", target, 1, (3,))
    for target in ("AA", "AB"):
        bind("AN", target, 2, (5,))
    for turn_index in (2, 3):
        bind("AB", "BA", turn_index, (3, 5))
        bind("AA", "D2N", turn_index, (3, 5))
    return directed


CONTRAST_EXCLUDED_MESSAGE_IDS: Mapping[tuple[str, str, int], tuple[int, ...]] = _contrast_mapping()


def excluded_message_ids_for_contrast(
    source_program: str,
    target_program: str,
    turn_index: int,
) -> tuple[int, ...]:
    """Return the frozen treatment-message mask for one roster contrast."""

    if not isinstance(source_program, str) or not source_program:
        raise RelationalPartialGrammarError("source_program must be a non-empty string")
    if not isinstance(target_program, str) or not target_program:
        raise RelationalPartialGrammarError("target_program must be a non-empty string")
    turn = _integer(turn_index, "turn_index", minimum=0)
    try:
        return CONTRAST_EXCLUDED_MESSAGE_IDS[(source_program, target_program, turn)]
    except KeyError as error:
        raise RelationalPartialGrammarError(
            "contrast has no frozen treatment-message mapping"
        ) from error


def _sequence(value: Sequence[Any], name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise RelationalPartialGrammarError(f"{name} must be a token-aligned sequence")
    try:
        return tuple(value)
    except TypeError as error:
        raise RelationalPartialGrammarError(f"{name} must be a token-aligned sequence") from error


def _integer(value: object, name: str, *, minimum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < minimum:
        raise RelationalPartialGrammarError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _integer_sequence(
    value: Sequence[int],
    name: str,
    *,
    minimum: int,
) -> tuple[int, ...]:
    raw = _sequence(value, name)
    return tuple(_integer(item, f"{name} item", minimum=minimum) for item in raw)


def _decoded_sequence(value: Sequence[str], name: str) -> tuple[str, ...]:
    raw = _sequence(value, name)
    if any(not isinstance(item, str) or not item for item in raw):
        raise RelationalPartialGrammarError(f"{name} must contain decoded non-empty strings")
    return tuple(raw)


def _anchor_value(anchor: object, key: str) -> object:
    if isinstance(anchor, Mapping):
        if key not in anchor:
            raise RelationalPartialGrammarError(f"anchor is missing {key}")
        return anchor[key]
    try:
        return getattr(anchor, key)
    except AttributeError as error:
        raise RelationalPartialGrammarError(f"anchor is missing {key}") from error


def _message_local_offsets(message_ids: Sequence[int]) -> tuple[int, ...]:
    counts: Counter[int] = Counter()
    offsets: list[int] = []
    closed: set[int] = set()
    active: int | None = None
    for message_id in message_ids:
        if active is not None and message_id != active:
            closed.add(active)
        if message_id in closed:
            raise RelationalPartialGrammarError(
                "tokens for one message_id must form one contiguous raw block"
            )
        active = message_id
        offsets.append(counts[message_id])
        counts[message_id] += 1
    return tuple(offsets)


def _validate_aligned_lengths(
    token_ids: Sequence[int],
    annotations: Mapping[str, Sequence[object]],
) -> None:
    length = len(token_ids)
    if length < len(STATUS_PREFIX_TOKEN_IDS):
        raise RelationalPartialGrammarError("status prefix is too short")
    mismatched = [name for name, values in annotations.items() if len(values) != length]
    if mismatched:
        raise RelationalPartialGrammarError(
            f"token annotations are not aligned: {sorted(mismatched)}"
        )


def _validate_status_anchor(
    *,
    anchor: object,
    token_ids: tuple[int, ...],
    token_role_ids: tuple[int, ...],
    token_turn_ids: tuple[int, ...],
    token_message_ids: tuple[int, ...],
    token_span_flags: tuple[int, ...],
    token_origins: tuple[str, ...],
    token_origin_details: tuple[str, ...],
    raw_message_local_offsets: tuple[int, ...],
) -> tuple[int, int]:
    field_name = _anchor_value(anchor, "field_name")
    if field_name != "status":
        raise RelationalPartialGrammarError("anchor field_name must be status")
    turn_index = _integer(_anchor_value(anchor, "turn_index"), "anchor turn_index", minimum=0)
    anchor_index = _integer(
        _anchor_value(anchor, "anchor_token_index"),
        "anchor_token_index",
        minimum=0,
    )
    if anchor_index >= len(token_ids):
        raise RelationalPartialGrammarError("anchor_token_index is outside the prefix")
    if anchor_index != len(token_ids) - 1:
        raise RelationalPartialGrammarError("status-prefix anchor must terminate the raw prefix")
    declared_token_id = _integer(
        _anchor_value(anchor, "anchor_token_id"),
        "anchor_token_id",
        minimum=0,
    )
    if declared_token_id != STATUS_ANCHOR_TOKEN_ID or token_ids[anchor_index] != (
        STATUS_ANCHOR_TOKEN_ID
    ):
        raise RelationalPartialGrammarError("status anchor token ID must equal 25")
    if token_ids[-len(STATUS_PREFIX_TOKEN_IDS) :] != STATUS_PREFIX_TOKEN_IDS:
        raise RelationalPartialGrammarError(
            "status-prefix anchor must end in exact token IDs [2583, 25]"
        )
    expected_message_id = 2 * turn_index + 2
    if (
        token_message_ids[anchor_index],
        raw_message_local_offsets[anchor_index],
    ) != (expected_message_id, 5):
        raise RelationalPartialGrammarError(
            "status anchor must have raw message-local key (2*turn+2, 5)"
        )
    if token_turn_ids[anchor_index] != turn_index:
        raise RelationalPartialGrammarError("status anchor has the wrong turn annotation")
    if token_role_ids[anchor_index] != ASSISTANT_ROLE_ID:
        raise RelationalPartialGrammarError("status anchor must have assistant role 2")
    if token_span_flags[anchor_index] != STATUS_SPAN_FLAG:
        raise RelationalPartialGrammarError("status anchor must have span flag 16")
    if token_origins[anchor_index] != STATUS_PREFIX_ORIGIN:
        raise RelationalPartialGrammarError("status anchor must have environment origin")
    if token_origin_details[anchor_index] != STATUS_PREFIX_DETAIL:
        raise RelationalPartialGrammarError(
            "status anchor must have environment_status_prefix detail"
        )
    for index in range(len(token_ids) - len(STATUS_PREFIX_TOKEN_IDS), len(token_ids)):
        if (
            token_message_ids[index] != expected_message_id
            or token_turn_ids[index] != turn_index
            or token_role_ids[index] != ASSISTANT_ROLE_ID
            or token_span_flags[index] != STATUS_SPAN_FLAG
            or token_origins[index] != STATUS_PREFIX_ORIGIN
            or token_origin_details[index] != STATUS_PREFIX_DETAIL
        ):
            raise RelationalPartialGrammarError("status-prefix suffix annotations are inconsistent")
    return turn_index, anchor_index


def build_canonical_status_prefix_core(
    *,
    object_id: str,
    token_ids: Sequence[int],
    position_ids: Sequence[int],
    token_role_ids: Sequence[int],
    token_turn_ids: Sequence[int],
    token_message_ids: Sequence[int],
    token_span_flags: Sequence[int],
    token_origins: Sequence[str],
    token_origin_details: Sequence[str],
    anchor: object,
    excluded_message_ids: Sequence[int],
    correspondence_metadata: Mapping[str, Hashable],
) -> CanonicalTokenCore:
    """Build the exact raw-order canonical core for one pre-status prefix."""

    ids = _integer_sequence(token_ids, "token_ids", minimum=0)
    positions = _integer_sequence(position_ids, "position_ids", minimum=0)
    roles = _integer_sequence(token_role_ids, "token_role_ids", minimum=0)
    turns = _integer_sequence(token_turn_ids, "token_turn_ids", minimum=-1)
    messages = _integer_sequence(token_message_ids, "token_message_ids", minimum=0)
    spans = _integer_sequence(token_span_flags, "token_span_flags", minimum=0)
    origins = _decoded_sequence(token_origins, "token_origins")
    origin_details = _decoded_sequence(token_origin_details, "token_origin_details")
    annotations: dict[str, Sequence[object]] = {
        "position_ids": positions,
        "token_role_ids": roles,
        "token_turn_ids": turns,
        "token_message_ids": messages,
        "token_span_flags": spans,
        "token_origins": origins,
        "token_origin_details": origin_details,
    }
    _validate_aligned_lengths(ids, annotations)
    if positions != tuple(range(len(ids))):
        raise RelationalPartialGrammarError("position_ids must equal the canonical raw range")
    offsets = _message_local_offsets(messages)
    turn_index, anchor_index = _validate_status_anchor(
        anchor=anchor,
        token_ids=ids,
        token_role_ids=roles,
        token_turn_ids=turns,
        token_message_ids=messages,
        token_span_flags=spans,
        token_origins=origins,
        token_origin_details=origin_details,
        raw_message_local_offsets=offsets,
    )
    excluded_messages = _integer_sequence(excluded_message_ids, "excluded_message_ids", minimum=0)
    if len(set(excluded_messages)) != len(excluded_messages):
        raise RelationalPartialGrammarError("excluded_message_ids must be unique")
    absent_messages = set(excluded_messages) - set(messages)
    if absent_messages:
        raise RelationalPartialGrammarError(
            "every excluded message ID must occur in the raw prefix"
        )
    if messages[anchor_index] in set(excluded_messages):
        raise RelationalPartialGrammarError("the status-anchor message may not be excluded")

    action_indices = tuple(
        index for index, detail in enumerate(origin_details) if detail in SAMPLED_ACTION_DETAILS
    )
    action_counts = Counter(origin_details[index] for index in action_indices)
    if len(action_indices) != 2 * turn_index or any(
        action_counts[detail] != turn_index for detail in SAMPLED_ACTION_DETAILS
    ):
        raise RelationalPartialGrammarError(
            "sampled action details must contain exactly one status and one caveat "
            "action per completed turn"
        )
    excluded_message_set = set(excluded_messages)
    treatment_indices = tuple(
        index for index, message_id in enumerate(messages) if message_id in excluded_message_set
    )
    if set(treatment_indices) & set(action_indices):
        raise RelationalPartialGrammarError(
            "excluded treatment messages may not contain sampled actions"
        )
    excluded_indices = set(treatment_indices) | set(action_indices)
    retained_indices = tuple(index for index in range(len(ids)) if index not in excluded_indices)
    raw_keys = tuple(
        (CANONICAL_KEY_SCHEMA, message_id, offset)
        for message_id, offset in zip(messages, offsets, strict=True)
    )
    token_type_keys = tuple(
        (role_id, turn_id, message_id, span_flag, origin, detail)
        for role_id, turn_id, message_id, span_flag, origin, detail in zip(
            roles,
            turns,
            messages,
            spans,
            origins,
            origin_details,
            strict=True,
        )
    )
    expected_anchor_key = (CANONICAL_KEY_SCHEMA, 2 * turn_index + 2, 5)
    if raw_keys[anchor_index] != expected_anchor_key:
        raise RelationalPartialGrammarError("derived status anchor key is inconsistent")
    return build_canonical_token_core(
        object_id=object_id,
        token_ids=ids,
        token_type_keys=token_type_keys,
        canonical_keys=tuple(raw_keys[index] for index in retained_indices),
        token_indices=retained_indices,
        anchor_index=anchor_index,
        correspondence_metadata=correspondence_metadata,
        excluded_treatment_indices=treatment_indices,
        excluded_action_indices=action_indices,
    )


def build_exact_status_prefix_edge(
    source: CanonicalTokenCore,
    target: CanonicalTokenCore,
) -> PartialIsometryEdge:
    """Build an edge only when the complete retained grammars are identical."""

    if source.rank != target.rank:
        raise RelationalPartialGrammarError(
            "exact status-prefix endpoints must have equal retained rank"
        )
    if source.canonical_keys != target.canonical_keys:
        raise RelationalPartialGrammarError(
            "exact status-prefix endpoints must have full canonical-key equality"
        )
    source_ids = tuple(source.token_ids[index] for index in source.token_indices)
    target_ids = tuple(target.token_ids[index] for index in target.token_indices)
    if source_ids != target_ids:
        raise RelationalPartialGrammarError(
            "exact status-prefix endpoints have different retained token IDs"
        )
    source_types = tuple(source.token_type_keys[index] for index in source.token_indices)
    target_types = tuple(target.token_type_keys[index] for index in target.token_indices)
    if source_types != target_types:
        raise RelationalPartialGrammarError(
            "exact status-prefix endpoints have different retained token types"
        )
    edge = build_partial_isometry_edge(source, target)
    if edge.rank != source.rank or edge.canonical_keys != source.canonical_keys:
        raise RelationalPartialGrammarError(
            "generic connection builder did not preserve the exact grammar"
        )
    return edge


__all__ = [
    "ASSISTANT_ROLE_ID",
    "CANONICAL_KEY_SCHEMA",
    "CONTRAST_EXCLUDED_MESSAGE_IDS",
    "RelationalPartialGrammarError",
    "SAMPLED_ACTION_DETAILS",
    "STATUS_ANCHOR_TOKEN_ID",
    "STATUS_PREFIX_DETAIL",
    "STATUS_PREFIX_ORIGIN",
    "STATUS_PREFIX_TOKEN_IDS",
    "STATUS_SPAN_FLAG",
    "build_canonical_status_prefix_core",
    "build_exact_status_prefix_edge",
    "excluded_message_ids_for_contrast",
]
