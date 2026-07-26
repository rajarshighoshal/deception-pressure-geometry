"""Frozen schema and token contract for live relational structured actions."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
import struct
from typing import Any

SCHEMA_VERSION = 2
PROTOCOL_KIND = "relational_structured_action_protocol"
SPEC_KIND = "relational_structured_action_spec"
ROW_KIND = "relational_structured_action_row"
TURN_KIND = "relational_structured_action_turn"
FIELD_KIND = "relational_structured_action_field"
PROTOCOL_ID = "relational-structured-action-v2-live-sequential"

STRUCTURED_ACTION_SCHEMA_VERSION = SCHEMA_VERSION
STRUCTURED_ACTION_PROTOCOL_KIND = PROTOCOL_KIND
STRUCTURED_ACTION_SPEC_KIND = SPEC_KIND
STRUCTURED_ACTION_ROW_KIND = ROW_KIND
STRUCTURED_ACTION_TURN_KIND = TURN_KIND
STRUCTURED_ACTION_FIELD_KIND = FIELD_KIND

STATUS_FIELD = "status"
CAVEAT_FIELD = "caveat"
FIELD_ORDER = (STATUS_FIELD, CAVEAT_FIELD)
STATUS_PREFIX = "Status:"
CAVEAT_PREFIX = "\nCaveat:"
STATUS_PREFIX_TOKEN_IDS = (2583, 25)
CAVEAT_PREFIX_TOKEN_IDS = (198, 34, 525, 266, 25)
STATUS_ACTIONS = ("PASS", "FAIL", "SKIP", "NO_ACTION")
CAVEAT_ACTIONS = ("INCLUDE", "OMIT", "NONE", "SKIP", "NO_ACTION")
NO_ACTION = "NO_ACTION"

PASS_TOKEN_ID = 51935
FAIL_TOKEN_ID = 34207
SKIP_TOKEN_ID = 82504
INCLUDE_TOKEN_ID = 51493
OMIT_TOKEN_ID = 70500
NONE_TOKEN_ID = 43969
EOT_TOKEN_ID = 128009

MODEL_TOKEN_ORIGIN = "model_sample"
ENVIRONMENT_TOKEN_ORIGIN = "environment"

TOKENIZER_ARTIFACT_SHA256 = {
    "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    "tokenizer_config.json": "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424",
    "special_tokens_map.json": "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
}


@dataclass(frozen=True)
class StructuredActionCandidate:
    raw_token_id: int
    decoded_exact: str
    mapped_action: str


@dataclass(frozen=True)
class StructuredActionFieldSpec:
    name: str
    index: int
    outcome_role: str
    commit_key: str
    prefix_text: str
    prefix_utf8_hex: str
    prefix_sha256: str
    prefix_token_ids: tuple[int, ...]
    rng_domain: str
    candidates: tuple[StructuredActionCandidate, ...]
    default_action: str = NO_ACTION

    def map_raw_token_id(self, raw_token_id: int) -> str:
        _require_token_id(raw_token_id, "raw_token_id")
        return next(
            (
                candidate.mapped_action
                for candidate in self.candidates
                if candidate.raw_token_id == raw_token_id
            ),
            self.default_action,
        )


@dataclass(frozen=True)
class StructuredActionFieldRng:
    field_name: str
    domain: str
    stream_id: str
    seed: int


_STATUS_SPEC = StructuredActionFieldSpec(
    name=STATUS_FIELD,
    index=0,
    outcome_role="primary",
    commit_key="committed_status",
    prefix_text=STATUS_PREFIX,
    prefix_utf8_hex="5374617475733a",
    prefix_sha256="755c8b2a9fb114460e0909ab65c8cd47696b7648cbcdad3dbccd937c40660d0f",
    prefix_token_ids=STATUS_PREFIX_TOKEN_IDS,
    rng_domain="relational-structured-action-v2/status",
    candidates=(
        StructuredActionCandidate(PASS_TOKEN_ID, " PASS", "PASS"),
        StructuredActionCandidate(FAIL_TOKEN_ID, " FAIL", "FAIL"),
        StructuredActionCandidate(SKIP_TOKEN_ID, " SKIP", "SKIP"),
    ),
)
_CAVEAT_SPEC = StructuredActionFieldSpec(
    name=CAVEAT_FIELD,
    index=1,
    outcome_role="secondary_conditional_descriptive",
    commit_key="committed_caveat",
    prefix_text=CAVEAT_PREFIX,
    prefix_utf8_hex="0a4361766561743a",
    prefix_sha256="1cd4447923bfec95be4992b4f724b9cfcd27ca1563eb22ae0cb59c34220c1753",
    prefix_token_ids=CAVEAT_PREFIX_TOKEN_IDS,
    rng_domain="relational-structured-action-v2/caveat",
    candidates=(
        StructuredActionCandidate(INCLUDE_TOKEN_ID, " INCLUDE", "INCLUDE"),
        StructuredActionCandidate(OMIT_TOKEN_ID, " OMIT", "OMIT"),
        StructuredActionCandidate(NONE_TOKEN_ID, " NONE", "NONE"),
        StructuredActionCandidate(SKIP_TOKEN_ID, " SKIP", "SKIP"),
    ),
)
FIELD_SPECS = {STATUS_FIELD: _STATUS_SPEC, CAVEAT_FIELD: _CAVEAT_SPEC}

_PROTOCOL_KEYS = {
    "schema_version",
    "kind",
    "protocol_id",
    "conceptual_version",
    "status",
    "model",
    "supersession",
    "execution",
    "generation",
    "rng",
    "tokenizer",
    "fields",
    "record_schema",
}
_FIELD_RECORD_KEYS = {
    "schema_version",
    "kind",
    "field_name",
    "field_index",
    "outcome_role",
    "field_event_id",
    "anchor_token_index",
    "sampled_token_index",
    "anchor_prefix_token_ids_sha256",
    "prefix_text",
    "prefix_utf8_hex",
    "prefix_sha256",
    "prefix_token_ids",
    "sampling_space",
    "sampled_token_count",
    "raw_token_id",
    "raw_decoded_exact",
    "raw_token_origin",
    "mapped_action",
    "rng_domain",
    "rng_stream_id",
    "rng_seed",
}
_TURN_RECORD_KEYS = {
    "schema_version",
    "kind",
    "turn_index",
    "turn_event_id",
    "fields",
    "committed_status",
    "committed_caveat",
    "transcript_before_action_token_count",
    "transcript_before_action_token_ids_sha256",
    "transcript_token_ids",
    "token_origins",
    "token_origins_sha256",
    "transcript_token_ids_sha256",
    "environment_eot",
}


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def int32_token_sha256(values: Sequence[Any], *, allow_empty: bool = False) -> str:
    tokens = tuple(values)
    if (not tokens and not allow_empty) or any(
        not isinstance(token, int)
        or isinstance(token, bool)
        or token < -(2**31)
        or token >= 2**31
        for token in tokens
    ):
        raise ValueError("token IDs must be signed-int32 integers")
    return hashlib.sha256(struct.pack(f"<{len(tokens)}i", *tokens)).hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{name} keys differ: missing={missing}, extra={extra}")


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")
    return value


def _require_token_id(value: Any, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value >= 2**31
    ):
        raise ValueError(f"{name} must be a nonnegative signed-int32 token ID")
    return value


def _require_index(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _expected_field_json(spec: StructuredActionFieldSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "index": spec.index,
        "outcome_role": spec.outcome_role,
        "commit_key": spec.commit_key,
        "prefix": {
            "text": spec.prefix_text,
            "utf8_hex": spec.prefix_utf8_hex,
            "sha256": spec.prefix_sha256,
            "token_ids": list(spec.prefix_token_ids),
            "trailing_whitespace": False,
        },
        "candidates": [
            {
                "raw_token_id": candidate.raw_token_id,
                "decoded_exact": candidate.decoded_exact,
                "mapped_action": candidate.mapped_action,
            }
            for candidate in spec.candidates
        ],
        "default_action": spec.default_action,
    }


def validate_structured_action_protocol(protocol: Mapping[str, Any]) -> None:
    if not isinstance(protocol, Mapping):
        raise TypeError("structured-action protocol must be a mapping")
    _require_exact_keys(protocol, _PROTOCOL_KEYS, "protocol")
    if (
        protocol["schema_version"] != SCHEMA_VERSION
        or protocol["kind"] != PROTOCOL_KIND
        or protocol["protocol_id"] != PROTOCOL_ID
        or protocol["conceptual_version"] != 2
        or protocol["status"] != "frozen_live_protocol_pending_run_authorization"
    ):
        raise ValueError("invalid structured-action protocol identity")

    if protocol["model"] != {
        "logical_id": "meta-llama/Llama-3.1-8B-Instruct",
        "artifact_identity_bound_by": "selection_and_run_contract",
    }:
        raise ValueError("invalid structured-action model binding")

    if protocol["supersession"] != {
        "supersedes_for_future_use": ["relational-operational-choice-v1-pre-response"],
        "historical_artifacts_immutable": True,
        "retroactive_relabeling_allowed": False,
    }:
        raise ValueError("structured-action supersession must preserve history")
    if protocol["execution"] != {
        "stage": "live_sequential_transcript",
        "field_order": [STATUS_FIELD, CAVEAT_FIELD],
        "transcript_mutation": "append_each_field_immediately",
        "status_visible_at_caveat_anchor": True,
        "environment_appends_eot_after": CAVEAT_FIELD,
        "primary_prose_allowed": False,
    }:
        raise ValueError("invalid live sequential execution contract")
    if protocol["generation"] != {
        "sampling_space": "full_vocabulary",
        "draws_per_field": 1,
        "temperature": 0.7,
        "top_p": 0.95,
        "retry_attempts": 0,
        "parser_or_generation_rescue_allowed": False,
    }:
        raise ValueError("invalid exactly-one-token generation contract")

    rng = protocol["rng"]
    expected_domains = {
        STATUS_FIELD: _STATUS_SPEC.rng_domain,
        CAVEAT_FIELD: _CAVEAT_SPEC.rng_domain,
    }
    if rng != {
        "derivation": "sha256_source_rng_seed_source_rng_stream_id_field_domain",
        "deterministic": True,
        "field_domains": expected_domains,
        "domains_must_be_distinct": True,
    } or len(set(expected_domains.values())) != len(expected_domains):
        raise ValueError("invalid deterministic field RNG contract")

    tokenizer = protocol["tokenizer"]
    if tokenizer != {
        "artifacts": TOKENIZER_ARTIFACT_SHA256,
        "eot_token_id": EOT_TOKEN_ID,
    }:
        raise ValueError("invalid frozen tokenizer identity")
    expected_fields = [_expected_field_json(FIELD_SPECS[name]) for name in FIELD_ORDER]
    if protocol["fields"] != expected_fields:
        raise ValueError("invalid structured-action field contract")
    for field in expected_fields:
        prefix = field["prefix"]
        encoded = prefix["text"].encode("utf-8")
        if (
            prefix["text"][-1:].isspace()
            or encoded.hex() != prefix["utf8_hex"]
            or hashlib.sha256(encoded).hexdigest() != prefix["sha256"]
        ):
            raise ValueError("field prefixes must be exact and have no trailing whitespace")
        if any(not candidate["decoded_exact"].startswith(" ") for candidate in field["candidates"]):
            raise ValueError("candidate exact decodes must retain their leading space")

    if protocol["record_schema"] != {
        "spec_kind": SPEC_KIND,
        "row_kind": ROW_KIND,
        "turn_kind": TURN_KIND,
        "field_kind": FIELD_KIND,
        "model_token_origin": MODEL_TOKEN_ORIGIN,
        "environment_token_origin": ENVIRONMENT_TOKEN_ORIGIN,
    }:
        raise ValueError("invalid structured-action record identities")


validate_protocol = validate_structured_action_protocol


def field_specs_from_protocol(
    protocol: Mapping[str, Any],
) -> tuple[StructuredActionFieldSpec, StructuredActionFieldSpec]:
    validate_structured_action_protocol(protocol)
    return _STATUS_SPEC, _CAVEAT_SPEC


def map_raw_token_id(
    field_name: str,
    raw_token_id: int,
    protocol: Mapping[str, Any] | None = None,
) -> str:
    if protocol is not None:
        validate_structured_action_protocol(protocol)
    try:
        spec = FIELD_SPECS[field_name]
    except KeyError as error:
        raise ValueError(f"unknown structured-action field: {field_name}") from error
    return spec.map_raw_token_id(raw_token_id)


def map_status_token_id(raw_token_id: int) -> str:
    return _STATUS_SPEC.map_raw_token_id(raw_token_id)


def map_caveat_token_id(raw_token_id: int) -> str:
    return _CAVEAT_SPEC.map_raw_token_id(raw_token_id)


map_status_token = map_status_token_id
map_caveat_token = map_caveat_token_id


def structured_action_field_rng(
    *,
    source_rng_seed: int,
    source_rng_stream_id: str,
    field_name: str,
    protocol: Mapping[str, Any],
) -> StructuredActionFieldRng:
    validate_structured_action_protocol(protocol)
    if (
        not isinstance(source_rng_seed, int)
        or isinstance(source_rng_seed, bool)
        or source_rng_seed < 0
    ):
        raise ValueError("source_rng_seed must be a nonnegative integer")
    if not isinstance(source_rng_stream_id, str) or not source_rng_stream_id:
        raise ValueError("source_rng_stream_id must be a non-empty string")
    if field_name not in FIELD_SPECS:
        raise ValueError(f"unknown structured-action field: {field_name}")
    domain = str(protocol["rng"]["field_domains"][field_name])
    stream_id = f"{source_rng_stream_id}:{domain}"
    payload = f"{source_rng_seed}\0{source_rng_stream_id}\0{domain}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return StructuredActionFieldRng(field_name, domain, stream_id, seed)


def validate_structured_action_tokenizer_binding(
    protocol: Mapping[str, Any],
    *,
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[Sequence[int]], str],
    tokenizer_artifact_sha256: Mapping[str, str],
    eot_token_id: int,
) -> None:
    validate_structured_action_protocol(protocol)
    if dict(tokenizer_artifact_sha256) != TOKENIZER_ARTIFACT_SHA256:
        raise ValueError("tokenizer artifact hashes do not match the frozen protocol")
    if eot_token_id != EOT_TOKEN_ID:
        raise ValueError("EOT token ID does not match the frozen protocol")
    for spec in field_specs_from_protocol(protocol):
        prefix_ids = tuple(encode(spec.prefix_text))
        if prefix_ids != spec.prefix_token_ids:
            raise ValueError(f"{spec.name} prefix token IDs do not match the frozen encoding")
        if decode(spec.prefix_token_ids) != spec.prefix_text:
            raise ValueError(f"{spec.name} prefix tokens do not decode exactly")
        for candidate in spec.candidates:
            contextual_ids = tuple(encode(spec.prefix_text + candidate.decoded_exact))
            expected_ids = spec.prefix_token_ids + (candidate.raw_token_id,)
            if contextual_ids != expected_ids:
                raise ValueError(
                    f"{spec.name} candidate {candidate.mapped_action} is not one contextual token"
                )
            if decode((candidate.raw_token_id,)) != candidate.decoded_exact:
                raise ValueError(
                    f"{spec.name} candidate {candidate.mapped_action} decode is not exact"
                )


validate_tokenizer_binding = validate_structured_action_tokenizer_binding


def validate_tokenizer_object_binding(
    protocol: Mapping[str, Any],
    tokenizer: Any,
    *,
    tokenizer_artifact_sha256: Mapping[str, str],
) -> None:
    def encode(text: str) -> Sequence[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    def decode(token_ids: Sequence[int]) -> str:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise ValueError("tokenizer cannot prove the <|eot_id|> token binding")
    validate_structured_action_tokenizer_binding(
        protocol,
        encode=encode,
        decode=decode,
        tokenizer_artifact_sha256=tokenizer_artifact_sha256,
        eot_token_id=convert("<|eot_id|>"),
    )


def _field_record(
    *,
    spec: StructuredActionFieldSpec,
    field_event_id: str,
    sampled_token_index: int,
    transcript_token_ids: Sequence[int],
    raw_token_id: int,
    raw_decoded_exact: str,
    rng: StructuredActionFieldRng,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": FIELD_KIND,
        "field_name": spec.name,
        "field_index": spec.index,
        "outcome_role": spec.outcome_role,
        "field_event_id": field_event_id,
        "anchor_token_index": sampled_token_index - 1,
        "sampled_token_index": sampled_token_index,
        "anchor_prefix_token_ids_sha256": int32_token_sha256(
            transcript_token_ids[:sampled_token_index], allow_empty=True
        ),
        "prefix_text": spec.prefix_text,
        "prefix_utf8_hex": spec.prefix_utf8_hex,
        "prefix_sha256": spec.prefix_sha256,
        "prefix_token_ids": list(spec.prefix_token_ids),
        "sampling_space": "full_vocabulary",
        "sampled_token_count": 1,
        "raw_token_id": raw_token_id,
        "raw_decoded_exact": raw_decoded_exact,
        "raw_token_origin": MODEL_TOKEN_ORIGIN,
        "mapped_action": spec.map_raw_token_id(raw_token_id),
        "rng_domain": rng.domain,
        "rng_stream_id": rng.stream_id,
        "rng_seed": rng.seed,
    }


def build_structured_action_turn_record(
    *,
    turn_index: int,
    transcript_token_ids_before_action: Sequence[int],
    token_origins_before_action: Sequence[str],
    status_raw_token_id: int,
    status_raw_decoded_exact: str,
    caveat_raw_token_id: int,
    caveat_raw_decoded_exact: str,
    status_rng: StructuredActionFieldRng,
    caveat_rng: StructuredActionFieldRng,
    protocol: Mapping[str, Any],
    turn_event_id: str | None = None,
    status_field_event_id: str | None = None,
    caveat_field_event_id: str | None = None,
) -> dict[str, Any]:
    validate_structured_action_protocol(protocol)
    _require_index(turn_index, "turn_index")
    _require_token_id(status_raw_token_id, "status_raw_token_id")
    _require_token_id(caveat_raw_token_id, "caveat_raw_token_id")
    if not isinstance(status_raw_decoded_exact, str) or not isinstance(
        caveat_raw_decoded_exact, str
    ):
        raise ValueError("raw decoded field tokens must be exact strings")
    tokens = list(transcript_token_ids_before_action)
    int32_token_sha256(tokens, allow_empty=True)
    origins = list(token_origins_before_action)
    if len(origins) != len(tokens) or any(not isinstance(value, str) or not value for value in origins):
        raise ValueError("pre-action token origins must align with the transcript")
    if status_rng.field_name != STATUS_FIELD or status_rng.domain != _STATUS_SPEC.rng_domain:
        raise ValueError("status RNG is not bound to the status field")
    if caveat_rng.field_name != CAVEAT_FIELD or caveat_rng.domain != _CAVEAT_SPEC.rng_domain:
        raise ValueError("caveat RNG is not bound to the caveat field")
    before_count = len(tokens)
    before_sha256 = int32_token_sha256(tokens, allow_empty=True)
    if turn_event_id is None:
        turn_event_id = hashlib.sha256(
            f"{TURN_KIND}\0{turn_index}\0{before_sha256}".encode("utf-8")
        ).hexdigest()
    if not isinstance(turn_event_id, str) or not turn_event_id:
        raise ValueError("turn_event_id must be a non-empty string")
    if status_field_event_id is None:
        status_field_event_id = hashlib.sha256(
            f"{turn_event_id}\0{STATUS_FIELD}".encode("utf-8")
        ).hexdigest()
    if caveat_field_event_id is None:
        caveat_field_event_id = hashlib.sha256(
            f"{turn_event_id}\0{CAVEAT_FIELD}".encode("utf-8")
        ).hexdigest()

    tokens.extend(_STATUS_SPEC.prefix_token_ids)
    origins.extend([ENVIRONMENT_TOKEN_ORIGIN] * len(_STATUS_SPEC.prefix_token_ids))
    status_sampled_index = len(tokens)
    tokens.append(status_raw_token_id)
    origins.append(MODEL_TOKEN_ORIGIN)
    status_record = _field_record(
        spec=_STATUS_SPEC,
        field_event_id=status_field_event_id,
        sampled_token_index=status_sampled_index,
        transcript_token_ids=tokens,
        raw_token_id=status_raw_token_id,
        raw_decoded_exact=status_raw_decoded_exact,
        rng=status_rng,
    )

    tokens.extend(_CAVEAT_SPEC.prefix_token_ids)
    origins.extend([ENVIRONMENT_TOKEN_ORIGIN] * len(_CAVEAT_SPEC.prefix_token_ids))
    caveat_sampled_index = len(tokens)
    tokens.append(caveat_raw_token_id)
    origins.append(MODEL_TOKEN_ORIGIN)
    caveat_record = _field_record(
        spec=_CAVEAT_SPEC,
        field_event_id=caveat_field_event_id,
        sampled_token_index=caveat_sampled_index,
        transcript_token_ids=tokens,
        raw_token_id=caveat_raw_token_id,
        raw_decoded_exact=caveat_raw_decoded_exact,
        rng=caveat_rng,
    )

    eot_index = len(tokens)
    tokens.append(EOT_TOKEN_ID)
    origins.append(ENVIRONMENT_TOKEN_ORIGIN)
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": TURN_KIND,
        "turn_index": turn_index,
        "turn_event_id": turn_event_id,
        "fields": [status_record, caveat_record],
        "committed_status": status_record["mapped_action"],
        "committed_caveat": caveat_record["mapped_action"],
        "transcript_before_action_token_count": before_count,
        "transcript_before_action_token_ids_sha256": before_sha256,
        "transcript_token_ids": tokens,
        "token_origins": origins,
        "token_origins_sha256": canonical_json_sha256(origins),
        "transcript_token_ids_sha256": int32_token_sha256(tokens),
        "environment_eot": {
            "token_id": EOT_TOKEN_ID,
            "token_index": eot_index,
            "token_origin": ENVIRONMENT_TOKEN_ORIGIN,
        },
    }
    validate_structured_action_turn_record(record, protocol)
    return record


def validate_structured_action_turn_record(
    record: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    validate_structured_action_protocol(protocol)
    if not isinstance(record, Mapping):
        raise TypeError("structured-action turn record must be a mapping")
    _require_exact_keys(record, _TURN_RECORD_KEYS, "turn record")
    if record["schema_version"] != SCHEMA_VERSION or record["kind"] != TURN_KIND:
        raise ValueError("invalid structured-action turn identity")
    _require_index(record["turn_index"], "turn_index")
    if not isinstance(record["turn_event_id"], str) or not record["turn_event_id"]:
        raise ValueError("turn_event_id must be a non-empty string")

    tokens_value = record["transcript_token_ids"]
    origins_value = record["token_origins"]
    if not isinstance(tokens_value, list) or not isinstance(origins_value, list):
        raise ValueError("turn transcript tokens and origins must be lists")
    tokens = tuple(tokens_value)
    origins = tuple(origins_value)
    int32_token_sha256(tokens)
    if len(origins) != len(tokens) or any(not isinstance(value, str) or not value for value in origins):
        raise ValueError("turn token origins must align with the transcript")
    if record["transcript_token_ids_sha256"] != int32_token_sha256(tokens):
        raise ValueError("turn transcript token hash mismatch")
    if record["token_origins_sha256"] != canonical_json_sha256(list(origins)):
        raise ValueError("turn token-origin hash mismatch")

    before_count = _require_index(
        record["transcript_before_action_token_count"],
        "transcript_before_action_token_count",
    )
    if before_count >= len(tokens):
        raise ValueError("pre-action transcript boundary is outside the turn transcript")
    if record["transcript_before_action_token_ids_sha256"] != int32_token_sha256(
        tokens[:before_count], allow_empty=True
    ):
        raise ValueError("pre-action transcript token hash mismatch")

    fields = record["fields"]
    if not isinstance(fields, list) or len(fields) != 2:
        raise ValueError("turn record must contain exactly two fields")
    validated_fields: list[Mapping[str, Any]] = []
    for expected_name, field in zip(FIELD_ORDER, fields, strict=True):
        if not isinstance(field, Mapping):
            raise ValueError("structured-action fields must be mappings")
        _require_exact_keys(field, _FIELD_RECORD_KEYS, f"{expected_name} field")
        spec = FIELD_SPECS[expected_name]
        if (
            field["schema_version"] != SCHEMA_VERSION
            or field["kind"] != FIELD_KIND
            or field["field_name"] != expected_name
            or field["field_index"] != spec.index
            or field["outcome_role"] != spec.outcome_role
        ):
            raise ValueError(f"invalid {expected_name} field identity")
        if not isinstance(field["field_event_id"], str) or not field["field_event_id"]:
            raise ValueError(f"{expected_name} field_event_id must be non-empty")
        if (
            field["prefix_text"] != spec.prefix_text
            or field["prefix_utf8_hex"] != spec.prefix_utf8_hex
            or field["prefix_sha256"] != spec.prefix_sha256
            or field["prefix_token_ids"] != list(spec.prefix_token_ids)
        ):
            raise ValueError(f"{expected_name} field prefix mismatch")
        if field["sampling_space"] != "full_vocabulary" or field["sampled_token_count"] != 1:
            raise ValueError(f"{expected_name} must be one full-vocabulary token draw")
        raw_token_id = _require_token_id(field["raw_token_id"], f"{expected_name}.raw_token_id")
        raw_decoded_exact = field["raw_decoded_exact"]
        if not isinstance(raw_decoded_exact, str):
            raise ValueError(f"{expected_name} raw_decoded_exact must be a string")
        known_candidate = next(
            (
                candidate
                for candidate in spec.candidates
                if candidate.raw_token_id == raw_token_id
            ),
            None,
        )
        if known_candidate is not None and raw_decoded_exact != known_candidate.decoded_exact:
            raise ValueError(f"{expected_name} known token exact decode mismatch")
        if field["raw_token_origin"] != MODEL_TOKEN_ORIGIN:
            raise ValueError(f"{expected_name} token must have model-sample origin")
        if field["mapped_action"] != spec.map_raw_token_id(raw_token_id):
            raise ValueError(f"{expected_name} mapped action does not match its raw token ID")
        if field["rng_domain"] != spec.rng_domain:
            raise ValueError(f"{expected_name} RNG domain mismatch")
        if not isinstance(field["rng_stream_id"], str) or not field["rng_stream_id"]:
            raise ValueError(f"{expected_name} RNG stream ID must be non-empty")
        if not isinstance(field["rng_seed"], int) or isinstance(field["rng_seed"], bool) or field["rng_seed"] < 0:
            raise ValueError(f"{expected_name} RNG seed must be nonnegative")

        anchor = _require_index(field["anchor_token_index"], f"{expected_name}.anchor_token_index")
        sampled = _require_index(
            field["sampled_token_index"], f"{expected_name}.sampled_token_index"
        )
        if sampled >= len(tokens) or anchor != sampled - 1:
            raise ValueError(f"{expected_name} causal anchor must precede its sampled token")
        prefix_start = sampled - len(spec.prefix_token_ids)
        if prefix_start < before_count or tokens[prefix_start:sampled] != spec.prefix_token_ids:
            raise ValueError(f"{expected_name} prefix is not immediately before its anchor")
        if any(origin != ENVIRONMENT_TOKEN_ORIGIN for origin in origins[prefix_start:sampled]):
            raise ValueError(f"{expected_name} prefix must have environment origin")
        if tokens[sampled] != raw_token_id or origins[sampled] != MODEL_TOKEN_ORIGIN:
            raise ValueError(f"{expected_name} raw token or origin mismatch at its anchor")
        _require_sha256(field["anchor_prefix_token_ids_sha256"], f"{expected_name} anchor hash")
        if field["anchor_prefix_token_ids_sha256"] != int32_token_sha256(
            tokens[:sampled], allow_empty=True
        ):
            raise ValueError(f"{expected_name} anchor-prefix token hash mismatch")
        validated_fields.append(field)

    status_field, caveat_field = validated_fields
    status_sampled = int(status_field["sampled_token_index"])
    caveat_sampled = int(caveat_field["sampled_token_index"])
    expected_caveat_prefix_start = status_sampled + 1
    if (
        caveat_sampled <= status_sampled
        or caveat_sampled - len(_CAVEAT_SPEC.prefix_token_ids)
        != expected_caveat_prefix_start
    ):
        raise ValueError("status must be committed immediately before the caveat field")
    if (
        tokens[status_sampled] != status_field["raw_token_id"]
        or status_sampled >= caveat_sampled
    ):
        raise ValueError("status token is not visible at the caveat anchor")
    if status_field["rng_domain"] == caveat_field["rng_domain"]:
        raise ValueError("status and caveat RNG domains must be distinct")
    if record["committed_status"] != status_field["mapped_action"]:
        raise ValueError("committed_status does not match the status token")
    if record["committed_caveat"] != caveat_field["mapped_action"]:
        raise ValueError("committed_caveat does not match the caveat token")

    eot = record["environment_eot"]
    if not isinstance(eot, Mapping) or set(eot) != {"token_id", "token_index", "token_origin"}:
        raise ValueError("turn record lacks an exact environment EOT record")
    eot_index = _require_index(eot["token_index"], "environment_eot.token_index")
    if (
        eot["token_id"] != EOT_TOKEN_ID
        or eot["token_origin"] != ENVIRONMENT_TOKEN_ORIGIN
        or eot_index != caveat_sampled + 1
        or eot_index != len(tokens) - 1
        or tokens[eot_index] != EOT_TOKEN_ID
        or origins[eot_index] != ENVIRONMENT_TOKEN_ORIGIN
    ):
        raise ValueError("environment EOT must immediately follow the caveat token")


validate_turn_record = validate_structured_action_turn_record


def validate_schema_identity(record: Mapping[str, Any], expected_kind: str) -> None:
    if expected_kind not in {SPEC_KIND, ROW_KIND, TURN_KIND, FIELD_KIND}:
        raise ValueError("unknown structured-action schema kind")
    if record.get("schema_version") != SCHEMA_VERSION or record.get("kind") != expected_kind:
        raise ValueError(f"invalid {expected_kind} schema identity")
