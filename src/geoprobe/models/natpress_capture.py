"""Natpress activation-capture planning + row validation (torch-free module).

Plan precondition (3) of the privately retained pressure ledger of the program: the structured
relational capture validator rejects free-prose rows BY DESIGN, so natpress gets its own
row-manifest validator and capture-plan builder in NEW files. This module is import-safe
without torch: everything here runs on plain Python so the ``plan`` and ``verify``
subcommands of ``experiments/capture_natpress_activations.py`` stay $0.

Frozen capture decisions (pinned 2026-07-16, closing plan preconditions 1 and 2):
  * ``batch_size=1`` for every capture row -- each production forward IS the scalar
    forward, so the matched-shape clone parity check (atol 0.0) becomes a strict re-run
    reproducibility gate and the v1 "layer 12 scalar/batch residual drift 2.0" question
    (one bf16 ulp on an outlier channel under batch-4 tiling) is closed by construction.
  * Residual layers [12, 16, 19, 20] on ALL conversations of both banks.
  * Attention layers [16, 20] ONLY on the pre-registered smooth subset: per scenario the
    smooth conversation with the LOWEST sample_index from each bank (16 x 2 = 32 rows).
  * Hard 38 GiB cap on projected artifact bytes, sized from the MEASURED per-row token
    counts (``len(row["token_ids"])`` -- the banks store the exact replayed ids).

Byte formulas (bf16 = 2 bytes; hidden 4096; 32 heads; lower-triangle attention packing is
the entire saving -- no codec):
    residual_bytes(T)  = n_residual_layers * T * 4096 * 2
    attention_bytes(T) = n_attention_layers * 32 * T*(T+1)/2 * 2
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

NATPRESS_CAPTURE_PLAN_SCHEMA = "natpress_capture_plan_v1"
NATPRESS_CAPTURE_ROW_SCHEMA = "natpress_capture_row_record_v1"
CAPTURE_MAX_BYTES = 38 * (1 << 30)  # RELATIONAL_CAPTURE_MAX_BYTES convention
RESIDUAL_LAYERS = (12, 16, 19, 20)
ATTENTION_LAYERS = (16, 20)
HIDDEN_SIZE = 4096
N_ATTENTION_HEADS = 32
BF16_BYTES = 2
BATCH_SIZE = 1
VERIFY_SCALAR_ROWS = 2
RESIDUAL_PARITY_ATOL = 0.0
ATTENTION_PARITY_ATOL = 0.0
PARITY_MODE = "matched_shape_clone_same_index"
TOKENIZATION_MODE = "stored_token_ids_verbatim"
ALLOWED_PROTOCOLS = ("natpress_v1", "natpress_v3_adaptive")
ARMS = ("smooth", "benign", "step", "latedump")
EXPECTED_TURNS = 7

# Llama-3.1 chat-template marker ids (fixed by the frozen banks' tokenizer).
_BEGIN_OF_TEXT = 128000
_START_HEADER = 128006
_END_HEADER = 128007
_EOT = 128009
_ROLE_TOKEN_TO_NAME = {9125: "system", 882: "user", 78191: "assistant"}
# role ids for token_role_ids: 0 template/special, 1 system, 2 user, 3 assistant
_ROLE_NAME_TO_ID = {"system": 1, "user": 2, "assistant": 3}


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def residual_bytes(token_length: int, *, n_layers: int = len(RESIDUAL_LAYERS)) -> int:
    if token_length < 1:
        raise ValueError("token_length must be positive")
    return n_layers * token_length * HIDDEN_SIZE * BF16_BYTES


def attention_bytes(token_length: int, *, n_layers: int = len(ATTENTION_LAYERS)) -> int:
    if token_length < 1:
        raise ValueError("token_length must be positive")
    return n_layers * N_ATTENTION_HEADS * (token_length * (token_length + 1) // 2) * BF16_BYTES


def peak_attention_scores_bytes(token_length: int) -> int:
    """Transient dense [32, T, T] bf16 attention probabilities retained per hooked layer."""
    return N_ATTENTION_HEADS * token_length * token_length * BF16_BYTES


def parse_conversation_id(conversation_id: str) -> dict[str, str]:
    parts = str(conversation_id).split(":")
    if len(parts) != 5 or parts[3] not in ARMS:
        raise ValueError(f"unparseable conversation_id: {conversation_id!r}")
    return {
        "prefix": parts[0],
        "family": parts[1],
        "scenario_id": parts[2],
        "arm": parts[3],
        "sample": parts[4],
    }


def _is_subsequence(needle: Sequence[int], haystack: Sequence[int]) -> bool:
    if not needle:
        return False
    n, h = list(needle), list(haystack)
    first = n[0]
    for start in range(len(h) - len(n) + 1):
        if h[start] == first and h[start:start + len(n)] == n:
            return True
    return False


def _assistant_segments_from_annotations(
    token_ids: Sequence[int], annotations: Mapping[str, list[int]]
) -> list[list[int]]:
    """Assistant-message content token runs (header/eot included), in message order."""
    segments: dict[int, list[int]] = {}
    for token, role, turn in zip(
        token_ids, annotations["token_role_ids"], annotations["token_turn_ids"]
    ):
        if role == _ROLE_NAME_TO_ID["assistant"]:
            segments.setdefault(turn, []).append(int(token))
    return [segments[turn] for turn in sorted(segments)]


def derive_token_annotations(token_ids: Sequence[int]) -> dict[str, list[int]]:
    """Per-token role/turn ids derived from the Llama-3.1 template markers.

    role ids: 0 template/special (begin_of_text only), 1 system, 2 user, 3 assistant.
    turn ids: message ordinal (system=0, then alternating user/assistant); the
    begin_of_text token carries turn -1. Header and eot tokens belong to their message.
    """
    ids = [int(token) for token in token_ids]
    if not ids or ids[0] != _BEGIN_OF_TEXT:
        raise ValueError("token_ids must start with <|begin_of_text|>")
    roles = [0]
    turns = [-1]
    message_index = -1
    message_roles: list[str] = []
    position = 1
    current_role_id = 0
    while position < len(ids):
        token = ids[position]
        if token == _START_HEADER:
            if position + 2 >= len(ids) or ids[position + 2] != _END_HEADER:
                raise ValueError(f"malformed message header at token {position}")
            role_name = _ROLE_TOKEN_TO_NAME.get(ids[position + 1])
            if role_name is None:
                raise ValueError(f"unknown role token {ids[position + 1]} at {position + 1}")
            message_index += 1
            message_roles.append(role_name)
            current_role_id = _ROLE_NAME_TO_ID[role_name]
            for _ in range(3):
                roles.append(current_role_id)
                turns.append(message_index)
            position += 3
            continue
        if message_index < 0:
            raise ValueError(f"content token {token} before any message header")
        roles.append(current_role_id)
        turns.append(message_index)
        position += 1
    if not message_roles or message_roles[0] != "system":
        raise ValueError("first message must be the system message")
    body = message_roles[1:]
    if len(body) != 2 * EXPECTED_TURNS:
        raise ValueError(f"expected {2 * EXPECTED_TURNS} user/assistant messages, got {len(body)}")
    for index, role_name in enumerate(body):
        expected = "user" if index % 2 == 0 else "assistant"
        if role_name != expected:
            raise ValueError(f"message {index + 1} role {role_name!r} != {expected!r}")
    return {"token_role_ids": roles, "token_turn_ids": turns}


def validate_natpress_capture_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one frozen bank row for capture; returns the capture-relevant summary."""
    if not isinstance(row, Mapping):
        raise ValueError("row must be a mapping")
    protocol = row.get("protocol")
    if protocol not in ALLOWED_PROTOCOLS:
        raise ValueError(f"unsupported protocol: {protocol!r}")
    fields = parse_conversation_id(row["conversation_id"])
    if str(row.get("scenario_id")) != fields["scenario_id"]:
        raise ValueError(f"scenario_id mismatch in {row['conversation_id']}")
    if str(row.get("arm")) != fields["arm"]:
        raise ValueError(f"arm mismatch in {row['conversation_id']}")
    sample_index = row.get("sample_index")
    if not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError(f"bad sample_index in {row['conversation_id']}")
    token_ids = row.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids or not all(
        isinstance(token, int) for token in token_ids
    ):
        raise ValueError(f"missing/invalid token_ids in {row['conversation_id']}")
    turns = row.get("turns")
    if not isinstance(turns, list) or len(turns) != EXPECTED_TURNS:
        raise ValueError(
            f"{row['conversation_id']}: expected {EXPECTED_TURNS} turns, "
            f"got {len(turns) if isinstance(turns, list) else type(turns).__name__}"
        )
    annotations = derive_token_annotations(token_ids)  # raises on malformed structure
    segments = _assistant_segments_from_annotations(token_ids, annotations)
    if len(segments) != EXPECTED_TURNS:
        raise ValueError(f"{row['conversation_id']}: {len(segments)} assistant messages")
    sampling = row.get("sampling") or {}
    max_new_tokens = int(sampling.get("max_new_tokens_per_turn", 0))
    n_boundary_retokenized = 0
    for index, turn in enumerate(turns):
        if turn.get("turn_index") != index:
            raise ValueError(f"{row['conversation_id']}: turn_index {turn.get('turn_index')} != {index}")
        assistant_ids = turn.get("assistant_token_ids")
        if not isinstance(assistant_ids, list) or not assistant_ids:
            raise ValueError(f"{row['conversation_id']}: turn {index} lacks assistant_token_ids")
        if _is_subsequence(assistant_ids, segments[index]):
            continue
        # The rollout re-tokenizes the growing transcript each turn; a reply truncated at
        # the generation cap can merge boundary tokens differently on re-encode. That is
        # the ONLY sanctioned mismatch: anything else marks a genuinely foreign reply.
        if max_new_tokens and len(assistant_ids) == max_new_tokens:
            n_boundary_retokenized += 1
            continue
        raise ValueError(
            f"{row['conversation_id']}: turn {index} assistant_token_ids not found in its "
            "assistant segment and the turn is not length-capped"
        )
    return {
        "conversation_id": str(row["conversation_id"]),
        "protocol": str(protocol),
        "family": str(row.get("family", fields["family"])),
        "scenario_id": fields["scenario_id"],
        "arm": fields["arm"],
        "sample_index": sample_index,
        "token_length": len(token_ids),
        "token_ids_sha256": canonical_json_sha256([int(token) for token in token_ids]),
        "n_boundary_retokenized_turns": n_boundary_retokenized,
    }


def select_attention_subset(
    scripted_rows: Sequence[Mapping[str, Any]],
    adaptive_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Pre-registered smooth subset: lowest-sample_index smooth row per scenario per bank."""
    subset: list[str] = []
    for bank in (scripted_rows, adaptive_rows):
        best: dict[str, tuple[int, str]] = {}
        for row in bank:
            if str(row.get("arm")) != "smooth":
                continue
            scenario = str(row["scenario_id"])
            candidate = (int(row["sample_index"]), str(row["conversation_id"]))
            if scenario not in best or candidate < best[scenario]:
                best[scenario] = candidate
        subset.extend(conversation_id for _, conversation_id in sorted(best.values()))
    if len(subset) != len(set(subset)):
        raise ValueError("attention subset contains duplicate conversation ids")
    return sorted(subset)


def build_natpress_capture_plan(
    scripted_rows: Sequence[Mapping[str, Any]],
    adaptive_rows: Sequence[Mapping[str, Any]],
    *,
    residual_layers: Sequence[int] = RESIDUAL_LAYERS,
    attention_layers: Sequence[int] = ATTENTION_LAYERS,
    max_bytes: int = CAPTURE_MAX_BYTES,
    scripted_rows_sha256: str | None = None,
    adaptive_rows_sha256: str | None = None,
) -> dict[str, Any]:
    residual_layers = tuple(int(layer) for layer in residual_layers)
    attention_layers = tuple(int(layer) for layer in attention_layers)
    if not set(attention_layers) <= set(residual_layers):
        raise ValueError("attention layers must be a subset of residual (capture) layers")
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bank_label, bank in (("scripted", scripted_rows), ("adaptive", adaptive_rows)):
        for row in bank:
            summary = validate_natpress_capture_row(row)
            if summary["conversation_id"] in seen:
                raise ValueError(f"duplicate conversation_id: {summary['conversation_id']}")
            seen.add(summary["conversation_id"])
            summary["bank"] = bank_label
            summaries.append(summary)
    if not summaries:
        raise ValueError("no capture rows")
    attention_subset = set(select_attention_subset(scripted_rows, adaptive_rows))
    missing = attention_subset - seen
    if missing:
        raise ValueError(f"attention subset rows missing from banks: {sorted(missing)}")

    total_residual = total_attention = annotation_overhead = 0
    for summary in summaries:
        length = summary["token_length"]
        summary["residual_bytes"] = residual_bytes(length, n_layers=len(residual_layers))
        capture_attention = summary["conversation_id"] in attention_subset
        summary["capture_attention"] = capture_attention
        summary["attention_bytes"] = (
            attention_bytes(length, n_layers=len(attention_layers)) if capture_attention else 0
        )
        # token_ids/position_ids int32 + role uint8 + turn int16 per token
        summary["annotation_bytes"] = length * (4 + 4 + 1 + 2)
        total_residual += summary["residual_bytes"]
        total_attention += summary["attention_bytes"]
        annotation_overhead += summary["annotation_bytes"]

    projected_total = total_residual + total_attention + annotation_overhead
    if projected_total > max_bytes:
        raise ValueError(
            f"projected capture bytes {projected_total} exceed the cap {max_bytes}; "
            "sharding to slip under the cap is forbidden without an explicit user decision"
        )
    max_token_length = max(summary["token_length"] for summary in summaries)
    plan_body: dict[str, Any] = {
        "schema_version": NATPRESS_CAPTURE_PLAN_SCHEMA,
        "tokenization_mode": TOKENIZATION_MODE,
        "residual_layers": list(residual_layers),
        "attention_layers": list(attention_layers),
        "hidden_size": HIDDEN_SIZE,
        "n_attention_heads": N_ATTENTION_HEADS,
        "dtype": "bfloat16",
        "batch_size": BATCH_SIZE,
        "parity": {
            "mode": PARITY_MODE,
            "verify_scalar_rows": VERIFY_SCALAR_ROWS,
            "residual_parity_atol": RESIDUAL_PARITY_ATOL,
            "attention_parity_atol": ATTENTION_PARITY_ATOL,
            "note": (
                "batch_size=1: every production forward is the scalar forward; parity is a "
                "bitwise re-run reproducibility gate, closing plan precondition (1)"
            ),
        },
        "byte_projection": {
            "formula_residual": "n_residual_layers*T*4096*2",
            "formula_attention": "n_attention_layers*32*T*(T+1)/2*2",
            "total_residual_bytes": total_residual,
            "total_attention_bytes": total_attention,
            "annotation_overhead_bytes": annotation_overhead,
            "projected_total_bytes": projected_total,
            "cap_bytes": max_bytes,
            "cap_fraction": projected_total / max_bytes,
        },
        "eager_attention_memory": {
            "max_token_length": max_token_length,
            "peak_scores_bytes_per_layer_at_max_T": peak_attention_scores_bytes(max_token_length),
            "hooked_layers": len(residual_layers),
            "peak_retained_scores_bytes_at_max_T": (
                len(residual_layers) * peak_attention_scores_bytes(max_token_length)
            ),
            "batch_1_feasible": True,
        },
        "attention_subset": sorted(attention_subset),
        "n_rows": len(summaries),
        "n_attention_rows": len(attention_subset),
        "source_banks": {
            "scripted_rows_sha256": scripted_rows_sha256,
            "adaptive_rows_sha256": adaptive_rows_sha256,
            "n_scripted": sum(1 for s in summaries if s["bank"] == "scripted"),
            "n_adaptive": sum(1 for s in summaries if s["bank"] == "adaptive"),
        },
        "rows": sorted(summaries, key=lambda s: (s["token_length"], s["conversation_id"])),
    }
    plan_body["plan_sha256"] = canonical_json_sha256(plan_body)
    return plan_body


def validate_natpress_capture_plan(plan: Mapping[str, Any]) -> None:
    """Structural validation + sha re-check for a loaded plan artifact."""
    if plan.get("schema_version") != NATPRESS_CAPTURE_PLAN_SCHEMA:
        raise ValueError("unknown capture-plan schema")
    # git_provenance is emission metadata added AFTER the sha is computed (cmd_plan), and
    # it legitimately differs between the local emission and a pod-side rebuild -- exclude
    # it from the hash alongside the sha field itself (20260721T112209Z failure: every
    # emitted artifact failed its own validation because provenance was hashed on load).
    body = {
        key: value
        for key, value in plan.items()
        if key not in ("plan_sha256", "git_provenance")
    }
    if canonical_json_sha256(body) != plan.get("plan_sha256"):
        raise ValueError("capture plan sha256 mismatch (plan edited after emission)")
    if plan["batch_size"] != BATCH_SIZE:
        raise ValueError("capture plan must pin batch_size=1")
    projection = plan["byte_projection"]
    if projection["projected_total_bytes"] > projection["cap_bytes"]:
        raise ValueError("capture plan exceeds its byte cap")
    lengths = [row["token_length"] for row in plan["rows"]]
    if lengths != sorted(lengths):
        raise ValueError("capture plan rows must be sorted by token length")
    ids = [row["conversation_id"] for row in plan["rows"]]
    if len(ids) != len(set(ids)):
        raise ValueError("capture plan contains duplicate conversation ids")
