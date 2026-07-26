"""Exact full-token residual and causal-attention capture for HF decoder models."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from geoprobe.data.relational_capture import pack_causal_lower_triangle
from geoprobe.io import file_sha256
from geoprobe.models.hf_capture import _transformer_layers


MATCHED_SHAPE_CLONE_PARITY = "matched_shape_clone_same_index"


@dataclass(frozen=True)
class CapturedRelationalRow:
    token_ids: torch.Tensor
    residuals: dict[int, torch.Tensor]
    attentions: dict[int, torch.Tensor]


def matched_shape_parity_roster(
    rows: Sequence[dict[str, Any]], *, batch_size: int, count: int
) -> list[dict[str, Any]]:
    """Select audited rows across distinct realized equal-length batch shapes."""
    if batch_size < 1 or count < 1 or count > len(rows):
        raise ValueError("matched-shape parity roster settings are invalid")
    lengths = [len(row["token_ids"]) for row in rows]
    if lengths != sorted(lengths):
        raise ValueError("matched-shape parity rows must be ordered by token length")
    batches: list[list[dict[str, Any]]] = []
    start = 0
    while start < len(rows):
        stop = start + 1
        while stop < len(rows) and lengths[stop] == lengths[start]:
            stop += 1
        group = list(rows[start:stop])
        batches.extend(
            group[index:index + batch_size]
            for index in range(0, len(group), batch_size)
        )
        start = stop

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    seen_batch_sizes: set[int] = set()
    for batch in batches:
        realized_size = len(batch)
        if realized_size in seen_batch_sizes:
            continue
        row = batch[0]
        conversation_id = str(row["conversation_id"])
        selected.append({
            "conversation_id": conversation_id,
            "batch_size": realized_size,
            "batch_index": 0,
            "token_length": len(row["token_ids"]),
        })
        selected_ids.add(conversation_id)
        seen_batch_sizes.add(realized_size)
        if len(selected) == count:
            return selected
    for batch in batches:
        for batch_index, row in enumerate(batch):
            conversation_id = str(row["conversation_id"])
            if conversation_id in selected_ids:
                continue
            selected.append({
                "conversation_id": conversation_id,
                "batch_size": len(batch),
                "batch_index": batch_index,
                "token_length": len(row["token_ids"]),
            })
            selected_ids.add(conversation_id)
            if len(selected) == count:
                return selected
    raise ValueError("could not construct the matched-shape parity roster")


def matched_shape_clone_batch(
    token_ids_batch: Sequence[Sequence[int]], *, batch_index: int
) -> list[list[int]]:
    """Keep batch geometry fixed while replacing every companion with the audited row."""
    if not token_ids_batch or not 0 <= batch_index < len(token_ids_batch):
        raise ValueError("matched-shape clone batch index is invalid")
    lengths = {len(token_ids) for token_ids in token_ids_batch}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("matched-shape clone parity requires equal non-empty lengths")
    target = [int(token_id) for token_id in token_ids_batch[batch_index]]
    return [list(target) for _ in token_ids_batch]


def set_eager_attention(model: Any) -> None:
    if not hasattr(model, "set_attn_implementation"):
        raise ValueError("model does not expose set_attn_implementation")
    model.set_attn_implementation("eager")
    implementation = getattr(model.config, "_attn_implementation", None)
    if implementation != "eager":
        raise ValueError(f"failed to select eager attention: {implementation}")


def _right_pad(token_ids_batch: Sequence[Sequence[int]], pad_token_id: int, device: torch.device):
    if not token_ids_batch or any(not token_ids for token_ids in token_ids_batch):
        raise ValueError("token batches must be non-empty")
    max_length = max(len(token_ids) for token_ids in token_ids_batch)
    input_ids = torch.full(
        (len(token_ids_batch), max_length), int(pad_token_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, token_ids in enumerate(token_ids_batch):
        length = len(token_ids)
        input_ids[index, :length] = torch.as_tensor(token_ids, dtype=torch.long, device=device)
        attention_mask[index, :length] = 1
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    return input_ids, attention_mask, position_ids


def _validate_attention(weights: torch.Tensor, *, row_sum_atol: float) -> None:
    if not torch.isfinite(weights).all().item():
        raise ValueError("attention contains non-finite values")
    if torch.any(weights < 0).item():
        raise ValueError("attention contains negative causal weights")
    if torch.count_nonzero(torch.triu(weights, diagonal=1)).item():
        raise ValueError("attention contains future-token mass")
    row_sums = weights.float().sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=row_sum_atol, rtol=0.0):
        error = float((row_sums - 1.0).abs().max())
        raise ValueError(f"attention row sums exceed tolerance: max_abs={error}")


def capture_relational_batch(
    model: Any,
    token_ids_batch: Sequence[Sequence[int]],
    *,
    layers: Sequence[int],
    pad_token_id: int,
    row_sum_atol: float = 0.01,
) -> list[CapturedRelationalRow]:
    """Replay exact token IDs and retain only selected native-dtype tensors."""
    blocks = _transformer_layers(model)
    selected = tuple(int(layer) for layer in layers)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("layers must be non-empty and unique")
    if min(selected) < 1 or max(selected) > len(blocks):
        raise ValueError(f"layers must be in [1, {len(blocks)}]")
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise ValueError("dense head attention capture requires eager attention")
    parameter = next(model.parameters())
    device = parameter.device
    expected_dtype = parameter.dtype
    expected_heads = int(model.config.num_attention_heads)
    hidden: dict[int, torch.Tensor] = {}
    attention: dict[int, torch.Tensor] = {}
    handles = []

    for layer in selected:
        block = blocks[layer - 1]

        def block_hook(_module, _inputs, output, *, layer=layer):
            tensor = output[0] if isinstance(output, tuple) else output
            hidden[layer] = tensor

        def attention_hook(_module, _inputs, output, *, layer=layer):
            if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                raise ValueError(f"layer {layer} did not return dense attention weights")
            attention[layer] = output[1]

        handles.append(block.register_forward_hook(block_hook))
        handles.append(block.self_attn.register_forward_hook(attention_hook))

    input_ids, attention_mask, position_ids = _right_pad(token_ids_batch, pad_token_id, device)
    try:
        with torch.inference_mode():
            model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    if set(hidden) != set(selected) or set(attention) != set(selected):
        raise ValueError("capture hooks did not observe every selected layer")
    batch_size, padded_length = input_ids.shape
    for layer in selected:
        if hidden[layer].shape[:2] != (batch_size, padded_length):
            raise ValueError(f"layer {layer} residual shape mismatch: {tuple(hidden[layer].shape)}")
        if hidden[layer].dtype != expected_dtype:
            raise ValueError(f"layer {layer} residual dtype changed: {hidden[layer].dtype}")
        expected_attention_shape = (batch_size, expected_heads, padded_length, padded_length)
        if tuple(attention[layer].shape) != expected_attention_shape:
            raise ValueError(
                f"layer {layer} attention shape {tuple(attention[layer].shape)} "
                f"!= {expected_attention_shape}"
            )
        if attention[layer].dtype != expected_dtype:
            raise ValueError(f"layer {layer} attention dtype changed: {attention[layer].dtype}")

    captured: list[CapturedRelationalRow] = []
    for row_index, token_ids in enumerate(token_ids_batch):
        length = len(token_ids)
        row_residuals: dict[int, torch.Tensor] = {}
        row_attentions: dict[int, torch.Tensor] = {}
        for layer in selected:
            residual = hidden[layer][row_index, :length].detach()
            if not torch.isfinite(residual).all().item():
                raise ValueError(f"layer {layer} residual contains non-finite values")
            weights = attention[layer][row_index, :, :length, :length].detach()
            _validate_attention(weights, row_sum_atol=row_sum_atol)
            row_residuals[layer] = residual.to("cpu")
            row_attentions[layer] = pack_causal_lower_triangle(
                weights, n_true_tokens=length
            ).to("cpu")
        captured.append(CapturedRelationalRow(
            token_ids=torch.tensor(token_ids, dtype=torch.int32),
            residuals=row_residuals,
            attentions=row_attentions,
        ))
    return captured


def compare_captured_rows(
    reference: CapturedRelationalRow,
    candidate: CapturedRelationalRow,
    *,
    residual_atol: float,
    attention_atol: float,
) -> dict[str, float]:
    if not torch.equal(reference.token_ids, candidate.token_ids):
        raise ValueError("scalar and batched token IDs differ")
    if set(reference.residuals) != set(candidate.residuals):
        raise ValueError("scalar and batched residual layer sets differ")
    if set(reference.attentions) != set(candidate.attentions):
        raise ValueError("scalar and batched attention layer sets differ")
    if set(reference.residuals) != set(reference.attentions):
        raise ValueError("reference residual and attention layer sets differ")
    metrics: dict[str, float] = {}
    for layer in reference.residuals:
        residual_error = float(
            (reference.residuals[layer].float() - candidate.residuals[layer].float()).abs().max()
        )
        attention_error = float(
            (reference.attentions[layer].float() - candidate.attentions[layer].float()).abs().max()
        )
        if not math.isfinite(residual_error) or not math.isfinite(attention_error):
            raise ValueError(f"layer {layer} scalar/batch parity is non-finite")
        metrics[f"residual_max_abs_L{layer}"] = residual_error
        metrics[f"attention_max_abs_L{layer}"] = attention_error
        if residual_error > residual_atol:
            raise ValueError(f"layer {layer} scalar/batch residual drift {residual_error}")
        if attention_error > attention_atol:
            raise ValueError(f"layer {layer} scalar/batch attention drift {attention_error}")
    return metrics


def validate_relational_parity_entries(
    entries: Sequence[dict[str, Any]],
    *,
    expected_conversation_ids: Sequence[str],
    layers: Sequence[int],
    residual_atol: float,
    attention_atol: float,
) -> None:
    """Validate the exact scalar-versus-batch parity checkpoint schema."""
    selected_layers = tuple(int(layer) for layer in layers)
    if not selected_layers or len(selected_layers) != len(set(selected_layers)):
        raise ValueError("scalar-parity layers must be nonempty and unique")
    tolerances = (float(residual_atol), float(attention_atol))
    if any(not math.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("scalar-parity tolerances must be finite and non-negative")
    expected_ids = tuple(str(value) for value in expected_conversation_ids)
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("scalar-parity expected conversation IDs contain duplicates")
    if len(entries) != len(expected_ids):
        raise ValueError("scalar-parity entry count differs from the expected ID roster")
    metric_limits = {
        **{
            f"residual_max_abs_L{layer}": tolerances[0]
            for layer in selected_layers
        },
        **{
            f"attention_max_abs_L{layer}": tolerances[1]
            for layer in selected_layers
        },
    }
    expected_keys = {"conversation_id", *metric_limits}
    for entry, expected_id in zip(entries, expected_ids):
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            raise ValueError("scalar-parity entry contains missing or foreign fields")
        if entry.get("conversation_id") != expected_id:
            raise ValueError("scalar-parity conversation IDs differ from the frozen roster")
        for key, limit in metric_limits.items():
            raw = entry[key]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"scalar-parity metric {key} is not numeric")
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"scalar-parity metric {key} is not finite and non-negative")
            if value > limit:
                raise ValueError(f"scalar-parity metric {key} exceeds its capture tolerance")


def validate_relational_parity_scope(
    *,
    scope: str,
    selected_count: int,
    verify_scalar_rows: int,
    residual_atol: float,
    attention_atol: float,
) -> None:
    """Apply the stronger parity rule required by development benchmark captures."""
    if scope == "development_smoke" and (
        verify_scalar_rows != selected_count
        or residual_atol != 0.0
        or attention_atol != 0.0
    ):
        raise ValueError(
            "development capture requires scalar parity on every selected row at zero tolerance"
        )


def missing_relational_parity_indices(
    conversation_ids: Sequence[str],
    *,
    required_conversation_ids: Sequence[str],
    completed_entries: Sequence[dict[str, Any]],
) -> tuple[int, ...]:
    """Return batch indices still needing scalar parity, keyed by conversation ID."""
    batch_ids = tuple(str(value) for value in conversation_ids)
    required_ids = tuple(str(value) for value in required_conversation_ids)
    completed_ids = tuple(str(entry.get("conversation_id", "")) for entry in completed_entries)
    for values, label in (
        (batch_ids, "batch"),
        (required_ids, "required"),
        (completed_ids, "completed"),
    ):
        if any(not value for value in values) or len(values) != len(set(values)):
            raise ValueError(f"scalar-parity {label} conversation IDs are empty or duplicated")
    if not set(completed_ids) <= set(required_ids):
        raise ValueError("completed scalar-parity IDs are outside the required roster")
    completed = set(completed_ids)
    required = set(required_ids)
    return tuple(
        index
        for index, conversation_id in enumerate(batch_ids)
        if conversation_id in required and conversation_id not in completed
    )


def _artifact_stem(conversation_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", conversation_id).strip("_")[:80]
    digest = hashlib.sha256(conversation_id.encode()).hexdigest()[:16]
    return f"{readable or 'row'}-{digest}"


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def relational_row_paths(out_dir: Path, conversation_id: str) -> tuple[Path, Path]:
    stem = _artifact_stem(conversation_id)
    tensor_dir = out_dir / "rows"
    return tensor_dir / f"{stem}.safetensors", tensor_dir / f"{stem}.json"


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _source_fields(row: dict[str, Any]) -> dict[str, Any]:
    token_ids = [int(token) for token in row["token_ids"]]
    source = {
        "conversation_id": str(row["conversation_id"]),
        "family": str(row["family"]),
        "scenario_id": str(row["scenario_id"]),
        "orbit_id": str(row["orbit_id"]),
        "source_row_sha256": canonical_json_sha256(row),
        "source_token_ids_sha256": canonical_json_sha256(token_ids),
    }
    if (
        row.get("schema_version") == 2
        and row.get("kind") == "relational_structured_action_row"
    ):
        source["intervention_program"] = str(row["intervention_program"])
    else:
        source["pressure_level"] = str(row["pressure_level"])
    return source


_STRUCTURED_ACTION_ROW_SCHEMA_VERSION = 2
_STRUCTURED_ACTION_ROW_KIND = "relational_structured_action_row"
_TOKEN_ORIGIN_ID_MAPPING = {
    "chat_source": 0,
    "environment": 1,
    "model_sample": 2,
}
_TOKEN_ORIGIN_DETAIL_ID_MAPPING = {
    "chat_source": 0,
    "environment_status_prefix": 1,
    "model_sampled_status_action": 2,
    "environment_caveat_separator": 3,
    "model_sampled_caveat_action": 4,
    "environment_eot": 5,
}
_STRUCTURED_CAPTURE_METADATA_KEYS = {
    "source_schema_version",
    "source_kind",
    "token_origin_id_mapping",
    "token_origin_detail_id_mapping",
    "status_outcome",
    "intervention_program",
    "action_anchors",
    "action_token_spans",
    "assistant_action_turn_records",
    "environment_eot_records",
}


def _capture_schema_version(row: dict[str, Any]) -> int:
    if row.get("schema_version") == _STRUCTURED_ACTION_ROW_SCHEMA_VERSION:
        if row.get("kind") != _STRUCTURED_ACTION_ROW_KIND:
            raise ValueError("unsupported schema-v2 relational source row kind")
        return _STRUCTURED_ACTION_ROW_SCHEMA_VERSION
    return 1


def _canonical_json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _structured_capture_metadata(row: dict[str, Any]) -> dict[str, Any]:
    if _capture_schema_version(row) != _STRUCTURED_ACTION_ROW_SCHEMA_VERSION:
        raise ValueError("structured capture metadata requires a schema-v2 source row")
    required = {
        "token_origins",
        "token_origin_details",
        "status_outcome",
        "intervention_program",
        "action_anchors",
        "action_token_spans",
        "assistant_action_turn_records",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"structured source row lacks required metadata: {missing}")
    token_ids = row["token_ids"]
    origins = row["token_origins"]
    details = row["token_origin_details"]
    if (
        not isinstance(origins, list)
        or not isinstance(details, list)
        or len(origins) != len(token_ids)
        or len(details) != len(token_ids)
        or any(origin not in _TOKEN_ORIGIN_ID_MAPPING for origin in origins)
        or any(detail not in _TOKEN_ORIGIN_DETAIL_ID_MAPPING for detail in details)
    ):
        raise ValueError("structured token-origin annotations do not use the frozen mappings")
    records = row["assistant_action_turn_records"]
    if not isinstance(records, list) or not records:
        raise ValueError("structured source row lacks assistant action turn records")
    eot_records: list[dict[str, Any]] = []
    for turn in records:
        if not isinstance(turn, dict) or not isinstance(turn.get("turn_event_id"), str):
            raise ValueError("structured assistant turn lacks an event ID")
        fields = turn.get("fields")
        if not isinstance(fields, list) or not fields:
            raise ValueError("structured assistant turn lacks field records")
        for field in fields:
            if not isinstance(field, dict) or not all(
                key in field for key in ("field_event_id", "rng_stream_id", "rng_seed")
            ):
                raise ValueError("structured assistant field lacks event or RNG metadata")
        eot = turn.get("environment_eot")
        if not isinstance(eot, dict) or set(eot) != {"token_id", "token_index", "token_origin"}:
            raise ValueError("structured assistant turn lacks environment EOT identity")
        index = eot["token_index"]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(token_ids)
            or token_ids[index] != eot["token_id"]
            or origins[index] != eot["token_origin"]
        ):
            raise ValueError("structured environment EOT does not align with the transcript")
        eot_records.append(eot)
    metadata = {
        "source_schema_version": _STRUCTURED_ACTION_ROW_SCHEMA_VERSION,
        "source_kind": _STRUCTURED_ACTION_ROW_KIND,
        "token_origin_id_mapping": _TOKEN_ORIGIN_ID_MAPPING,
        "token_origin_detail_id_mapping": _TOKEN_ORIGIN_DETAIL_ID_MAPPING,
        "status_outcome": row["status_outcome"],
        "intervention_program": row["intervention_program"],
        "action_anchors": row["action_anchors"],
        "action_token_spans": row["action_token_spans"],
        "assistant_action_turn_records": records,
        "environment_eot_records": eot_records,
    }
    try:
        canonical = json.loads(_canonical_json_text(metadata))
    except (TypeError, ValueError) as error:
        raise ValueError("structured capture metadata is not canonical JSON") from error
    return _validate_structured_capture_metadata(canonical)


def _validate_structured_capture_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict) or set(metadata) != _STRUCTURED_CAPTURE_METADATA_KEYS:
        raise ValueError("structured capture metadata has missing or foreign fields")
    if (
        metadata["source_schema_version"] != _STRUCTURED_ACTION_ROW_SCHEMA_VERSION
        or metadata["source_kind"] != _STRUCTURED_ACTION_ROW_KIND
        or metadata["token_origin_id_mapping"] != _TOKEN_ORIGIN_ID_MAPPING
        or metadata["token_origin_detail_id_mapping"] != _TOKEN_ORIGIN_DETAIL_ID_MAPPING
    ):
        raise ValueError("structured capture metadata mapping or schema mismatch")
    if not isinstance(metadata["status_outcome"], str) or not isinstance(
        metadata["intervention_program"], str
    ):
        raise ValueError("structured capture outcome/intervention metadata is invalid")
    for key in ("action_anchors", "action_token_spans", "assistant_action_turn_records",
                "environment_eot_records"):
        if not isinstance(metadata[key], list):
            raise ValueError(f"structured capture {key} is not a list")
    records = metadata["assistant_action_turn_records"]
    if not records or len(metadata["environment_eot_records"]) != len(records):
        raise ValueError("structured capture assistant-turn EOT metadata is incomplete")
    for record, eot in zip(records, metadata["environment_eot_records"], strict=True):
        if not isinstance(record, dict) or not isinstance(record.get("turn_event_id"), str):
            raise ValueError("structured capture assistant turn event metadata is invalid")
        fields = record.get("fields")
        if not isinstance(fields, list) or not fields or any(
            not isinstance(field, dict)
            or not isinstance(field.get("field_event_id"), str)
            or not isinstance(field.get("rng_stream_id"), str)
            or not isinstance(field.get("rng_seed"), int)
            or isinstance(field.get("rng_seed"), bool)
            for field in fields
        ):
            raise ValueError("structured capture assistant field event/RNG metadata is invalid")
        if record.get("environment_eot") != eot or not isinstance(eot, dict) or set(eot) != {
            "token_id", "token_index", "token_origin"
        }:
            raise ValueError("structured capture environment EOT metadata is invalid")
    return metadata


def _tensor_metadata(
    row: dict[str, Any],
    *,
    capture_contract_sha256: str,
    layers: Sequence[int],
    hidden_size: int,
    n_attention_heads: int,
    residual_dtype: str,
    attention_dtype: str,
) -> dict[str, str]:
    source = _source_fields(row)
    metadata = {
        "schema_version": "1",
        "kind": "relational_capture_row",
        **{key: str(value) for key, value in source.items()},
        "capture_contract_sha256": capture_contract_sha256,
        "layers": ",".join(str(layer) for layer in layers),
        "hidden_size": str(hidden_size),
        "n_attention_heads": str(n_attention_heads),
        "residual_dtype": residual_dtype,
        "attention_dtype": attention_dtype,
    }
    if _capture_schema_version(row) == _STRUCTURED_ACTION_ROW_SCHEMA_VERSION:
        structured = _structured_capture_metadata(row)
        metadata.update({
            "schema_version": "2",
            "source_schema_version": str(structured["source_schema_version"]),
            "source_kind": str(structured["source_kind"]),
            "token_origin_id_mapping_json": _canonical_json_text(
                structured["token_origin_id_mapping"]
            ),
            "token_origin_detail_id_mapping_json": _canonical_json_text(
                structured["token_origin_detail_id_mapping"]
            ),
            "structured_metadata_json": _canonical_json_text(structured),
            "structured_metadata_sha256": canonical_json_sha256(structured),
        })
    return metadata


def _validate_captured_row(
    row: dict[str, Any],
    captured: CapturedRelationalRow,
    *,
    layers: Sequence[int],
    hidden_size: int,
    n_attention_heads: int,
    residual_dtype: str,
    attention_dtype: str,
) -> None:
    token_ids = torch.tensor(row["token_ids"], dtype=torch.int32)
    if not torch.equal(captured.token_ids.cpu(), token_ids):
        raise ValueError(f"{row['conversation_id']}: captured tokens differ from source row")
    expected_layers = tuple(int(layer) for layer in layers)
    if tuple(sorted(captured.residuals)) != tuple(sorted(expected_layers)):
        raise ValueError(f"{row['conversation_id']}: residual layer set differs from contract")
    if tuple(sorted(captured.attentions)) != tuple(sorted(expected_layers)):
        raise ValueError(f"{row['conversation_id']}: attention layer set differs from contract")
    length = len(row["token_ids"])
    edge_count = length * (length + 1) // 2
    for layer in expected_layers:
        residual = captured.residuals[layer]
        attention = captured.attentions[layer]
        if tuple(residual.shape) != (length, hidden_size):
            raise ValueError(f"{row['conversation_id']}: residual L{layer} shape mismatch")
        if tuple(attention.shape) != (n_attention_heads, edge_count):
            raise ValueError(f"{row['conversation_id']}: attention L{layer} shape mismatch")
        if _dtype_name(residual) != residual_dtype:
            raise ValueError(f"{row['conversation_id']}: residual L{layer} dtype mismatch")
        if _dtype_name(attention) != attention_dtype:
            raise ValueError(f"{row['conversation_id']}: attention L{layer} dtype mismatch")


def _row_record(
    row: dict[str, Any],
    tensor_path: Path,
    *,
    capture_contract_sha256: str,
    layers: Sequence[int],
    hidden_size: int,
    n_attention_heads: int,
    residual_dtype: str,
    attention_dtype: str,
) -> dict[str, Any]:
    source = _source_fields(row)
    length = len(row["token_ids"])
    record = {
        "schema_version": 1,
        "kind": "relational_capture_row_record",
        **source,
        "outcome": str(row.get("outcome", "unknown")),
        "token_length": length,
        "stop_token_id": int(row["stop_token_id"]),
        "stop_token_index": int(row["stop_token_index"]),
        "capture_contract_sha256": capture_contract_sha256,
        "tensor_path": str(Path("rows") / tensor_path.name),
        "tensor_bytes": tensor_path.stat().st_size,
        "tensor_sha256": file_sha256(tensor_path),
        "layers": [int(layer) for layer in layers],
        "hidden_size": int(hidden_size),
        "n_attention_heads": int(n_attention_heads),
        "residual_dtype": residual_dtype,
        "attention_dtype": attention_dtype,
    }
    if _capture_schema_version(row) == _STRUCTURED_ACTION_ROW_SCHEMA_VERSION:
        structured = _structured_capture_metadata(row)
        record.update({
            "schema_version": 2,
            "source_schema_version": structured["source_schema_version"],
            "source_kind": structured["source_kind"],
            "structured_metadata": structured,
            "structured_metadata_sha256": canonical_json_sha256(structured),
        })
    return record


def _atomic_record(record_path: Path, record: dict[str, Any]) -> None:
    tmp_record = record_path.with_suffix(".json.tmp")
    tmp_record.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp_record, record_path)


def write_relational_row(
    out_dir: Path,
    row: dict[str, Any],
    captured: CapturedRelationalRow,
    *,
    resume: bool,
    capture_contract_sha256: str,
    layers: Sequence[int],
    hidden_size: int,
    n_attention_heads: int,
    residual_dtype: str,
    attention_dtype: str,
) -> dict[str, Any]:
    conversation_id = str(row["conversation_id"])
    tensor_path, record_path = relational_row_paths(out_dir, conversation_id)
    tensor_dir = tensor_path.parent
    tensor_dir.mkdir(parents=True, exist_ok=True)
    _validate_captured_row(
        row,
        captured,
        layers=layers,
        hidden_size=hidden_size,
        n_attention_heads=n_attention_heads,
        residual_dtype=residual_dtype,
        attention_dtype=attention_dtype,
    )
    if resume and tensor_path.exists() and record_path.exists():
        return validate_relational_row_artifact(
            record_path,
            expected_row=row,
            expected_contract_sha256=capture_contract_sha256,
            expected_layers=layers,
            expected_hidden_size=hidden_size,
            expected_attention_heads=n_attention_heads,
            expected_residual_dtype=residual_dtype,
            expected_attention_dtype=attention_dtype,
        )
    if resume and tensor_path.exists() and not record_path.exists():
        return recover_orphan_relational_row(
            tensor_path,
            row,
            capture_contract_sha256=capture_contract_sha256,
            layers=layers,
            hidden_size=hidden_size,
            n_attention_heads=n_attention_heads,
            residual_dtype=residual_dtype,
            attention_dtype=attention_dtype,
        )
    if tensor_path.exists() or record_path.exists():
        raise FileExistsError(f"partial/existing row artifact for {conversation_id}")

    length = int(captured.token_ids.numel())
    tensors: dict[str, torch.Tensor] = {
        "token_ids": captured.token_ids.contiguous(),
        "position_ids": torch.arange(length, dtype=torch.int32),
    }
    annotation_keys = {
        "token_role_ids": torch.uint8,
        "token_turn_ids": torch.int16,
        "token_message_ids": torch.int16,
        "token_span_flags": torch.int16,
    }
    for key, dtype in annotation_keys.items():
        values = row[key]
        if len(values) != length:
            raise ValueError(f"{conversation_id}: {key} length mismatch")
        tensors[key] = torch.tensor(values, dtype=dtype)
    if _capture_schema_version(row) == _STRUCTURED_ACTION_ROW_SCHEMA_VERSION:
        _structured_capture_metadata(row)
        tensors["token_origin_ids"] = torch.tensor(
            [_TOKEN_ORIGIN_ID_MAPPING[value] for value in row["token_origins"]],
            dtype=torch.uint8,
        )
        tensors["token_origin_detail_ids"] = torch.tensor(
            [_TOKEN_ORIGIN_DETAIL_ID_MAPPING[value] for value in row["token_origin_details"]],
            dtype=torch.uint8,
        )
    for layer, residual in captured.residuals.items():
        tensors[f"residual.L{layer}"] = residual.contiguous()
    for layer, attention in captured.attentions.items():
        tensors[f"attention.L{layer}"] = attention.contiguous()

    tmp_tensor = tensor_path.with_suffix(".safetensors.tmp")
    tmp_tensor.unlink(missing_ok=True)
    metadata = _tensor_metadata(
        row,
        capture_contract_sha256=capture_contract_sha256,
        layers=layers,
        hidden_size=hidden_size,
        n_attention_heads=n_attention_heads,
        residual_dtype=residual_dtype,
        attention_dtype=attention_dtype,
    )
    save_file(tensors, tmp_tensor, metadata=metadata)
    os.replace(tmp_tensor, tensor_path)
    record = _row_record(
        row,
        tensor_path,
        capture_contract_sha256=capture_contract_sha256,
        layers=layers,
        hidden_size=hidden_size,
        n_attention_heads=n_attention_heads,
        residual_dtype=residual_dtype,
        attention_dtype=attention_dtype,
    )
    _atomic_record(record_path, record)
    return record


def validate_relational_row_artifact(
    record_path: Path,
    *,
    expected_row: dict[str, Any] | None = None,
    expected_contract_sha256: str | None = None,
    expected_layers: Sequence[int] | None = None,
    expected_hidden_size: int | None = None,
    expected_attention_heads: int | None = None,
    expected_residual_dtype: str | None = None,
    expected_attention_dtype: str | None = None,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    record = json.loads(record_path.read_text())
    record_schema_version = record.get("schema_version")
    if record_schema_version not in {1, _STRUCTURED_ACTION_ROW_SCHEMA_VERSION} or record.get(
        "kind"
    ) != "relational_capture_row_record":
        raise ValueError(f"invalid row record schema for {record_path}")
    structured_metadata: dict[str, Any] | None = None
    if record_schema_version == _STRUCTURED_ACTION_ROW_SCHEMA_VERSION:
        if (
            record.get("source_schema_version") != _STRUCTURED_ACTION_ROW_SCHEMA_VERSION
            or record.get("source_kind") != _STRUCTURED_ACTION_ROW_KIND
        ):
            raise ValueError(f"structured source schema/kind mismatch for {record_path}")
        structured_metadata = _validate_structured_capture_metadata(
            record.get("structured_metadata")
        )
        if record.get("structured_metadata_sha256") != canonical_json_sha256(structured_metadata):
            raise ValueError(f"structured metadata checksum mismatch for {record_path}")
    conversation_id = str(record.get("conversation_id", ""))
    if record_path.name != relational_row_paths(record_path.parents[1], conversation_id)[1].name:
        raise ValueError(f"record filename identity mismatch for {record_path}")
    if expected_row is not None:
        source = _source_fields(expected_row)
        for key, value in source.items():
            if record.get(key) != value:
                raise ValueError(f"source {key} mismatch for {record_path}")
        for key in ("stop_token_id", "stop_token_index"):
            if int(record.get(key, -1)) != int(expected_row[key]):
                raise ValueError(f"source {key} mismatch for {record_path}")
        if str(record.get("outcome")) != str(expected_row.get("outcome", "unknown")):
            raise ValueError(f"source outcome mismatch for {record_path}")
        if _capture_schema_version(expected_row) != record_schema_version:
            raise ValueError(f"source schema mismatch for {record_path}")
        if structured_metadata is not None and structured_metadata != _structured_capture_metadata(
            expected_row
        ):
            raise ValueError(f"structured source metadata mismatch for {record_path}")
    layers = [int(layer) for layer in record.get("layers", [])]
    if not layers or layers != sorted(set(layers)):
        raise ValueError(f"invalid layer set for {record_path}")
    if expected_layers is not None and layers != [int(layer) for layer in expected_layers]:
        raise ValueError(f"capture layer contract mismatch for {record_path}")
    hidden_size = int(record.get("hidden_size", 0))
    n_attention_heads = int(record.get("n_attention_heads", 0))
    residual_dtype = str(record.get("residual_dtype", ""))
    attention_dtype = str(record.get("attention_dtype", ""))
    if expected_hidden_size is not None and hidden_size != expected_hidden_size:
        raise ValueError(f"hidden-size contract mismatch for {record_path}")
    if expected_attention_heads is not None and n_attention_heads != expected_attention_heads:
        raise ValueError(f"attention-head contract mismatch for {record_path}")
    if expected_residual_dtype is not None and residual_dtype != expected_residual_dtype:
        raise ValueError(f"residual-dtype contract mismatch for {record_path}")
    if expected_attention_dtype is not None and attention_dtype != expected_attention_dtype:
        raise ValueError(f"attention-dtype contract mismatch for {record_path}")
    if expected_contract_sha256 is not None:
        if record.get("capture_contract_sha256") != expected_contract_sha256:
            raise ValueError(f"capture contract mismatch for {record_path}")
    tensor_path = Path(record["tensor_path"])
    if not tensor_path.is_absolute():
        tensor_path = record_path.parents[1] / tensor_path
    if tensor_path.resolve() != record_path.with_suffix(".safetensors").resolve():
        raise ValueError(f"tensor path identity mismatch for {record_path}")
    if not tensor_path.exists():
        raise ValueError(f"missing tensor artifact for {record_path}")
    if verify_checksum and file_sha256(tensor_path) != record["tensor_sha256"]:
        raise ValueError(f"invalid tensor checksum for {record_path}")
    if tensor_path.stat().st_size != int(record.get("tensor_bytes", -1)):
        raise ValueError(f"tensor byte count mismatch for {record_path}")
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        metadata_expected = {
            "schema_version": "1",
            "kind": "relational_capture_row",
            "conversation_id": conversation_id,
            "family": str(record["family"]),
            "scenario_id": str(record["scenario_id"]),
            "orbit_id": str(record["orbit_id"]),
            "source_row_sha256": str(record["source_row_sha256"]),
            "source_token_ids_sha256": str(record["source_token_ids_sha256"]),
            "capture_contract_sha256": str(record["capture_contract_sha256"]),
            "layers": ",".join(str(layer) for layer in layers),
            "hidden_size": str(hidden_size),
            "n_attention_heads": str(n_attention_heads),
            "residual_dtype": residual_dtype,
            "attention_dtype": attention_dtype,
        }
        if structured_metadata is not None:
            metadata_expected.update({
                "schema_version": "2",
                "intervention_program": str(record["intervention_program"]),
                "source_schema_version": str(record["source_schema_version"]),
                "source_kind": str(record["source_kind"]),
                "token_origin_id_mapping_json": _canonical_json_text(
                    structured_metadata["token_origin_id_mapping"]
                ),
                "token_origin_detail_id_mapping_json": _canonical_json_text(
                    structured_metadata["token_origin_detail_id_mapping"]
                ),
                "structured_metadata_json": _canonical_json_text(structured_metadata),
                "structured_metadata_sha256": str(record["structured_metadata_sha256"]),
            })
        else:
            metadata_expected["pressure_level"] = str(record["pressure_level"])
        if metadata != metadata_expected:
            raise ValueError(f"tensor metadata identity mismatch for {record_path}")
        keys = set(handle.keys())
        required = {"token_ids", "position_ids", "token_role_ids", "token_turn_ids",
                    "token_message_ids", "token_span_flags"}
        expected_keys = required | {
            *(f"residual.L{layer}" for layer in layers),
            *(f"attention.L{layer}" for layer in layers),
        }
        if structured_metadata is not None:
            expected_keys |= {"token_origin_ids", "token_origin_detail_ids"}
        if keys != expected_keys:
            raise ValueError(
                f"tensor key set mismatch in {tensor_path}: "
                f"missing={sorted(expected_keys - keys)} extra={sorted(keys - expected_keys)}"
            )
        token_ids = handle.get_tensor("token_ids")
        length = int(record["token_length"])
        if token_ids.dtype != torch.int32 or tuple(token_ids.shape) != (length,):
            raise ValueError(f"token length mismatch for {record_path}")
        if canonical_json_sha256([int(token) for token in token_ids]) != record[
            "source_token_ids_sha256"
        ]:
            raise ValueError(f"source token digest mismatch for {record_path}")
        if expected_row is not None and [int(token) for token in token_ids] != [
            int(token) for token in expected_row["token_ids"]
        ]:
            raise ValueError(f"source token IDs mismatch for {record_path}")
        positions = handle.get_tensor("position_ids")
        if positions.dtype != torch.int32 or not torch.equal(
            positions, torch.arange(length, dtype=torch.int32)
        ):
            raise ValueError(f"position IDs mismatch for {record_path}")
        annotation_dtypes = {
            "token_role_ids": torch.uint8,
            "token_turn_ids": torch.int16,
            "token_message_ids": torch.int16,
            "token_span_flags": torch.int16,
        }
        for key, dtype in annotation_dtypes.items():
            values = handle.get_tensor(key)
            if values.dtype != dtype or tuple(values.shape) != (length,):
                raise ValueError(f"{key} shape/dtype mismatch for {record_path}")
            if expected_row is not None and [int(value) for value in values] != [
                int(value) for value in expected_row[key]
            ]:
                raise ValueError(f"source {key} mismatch for {record_path}")
        if structured_metadata is not None:
            origin_ids = handle.get_tensor("token_origin_ids")
            detail_ids = handle.get_tensor("token_origin_detail_ids")
            if (
                origin_ids.dtype != torch.uint8
                or detail_ids.dtype != torch.uint8
                or tuple(origin_ids.shape) != (length,)
                or tuple(detail_ids.shape) != (length,)
            ):
                raise ValueError(f"structured token-origin array shape/dtype mismatch for {record_path}")
            origin_inverse = {value: key for key, value in _TOKEN_ORIGIN_ID_MAPPING.items()}
            detail_inverse = {value: key for key, value in _TOKEN_ORIGIN_DETAIL_ID_MAPPING.items()}
            try:
                origins = [origin_inverse[int(value)] for value in origin_ids]
                details = [detail_inverse[int(value)] for value in detail_ids]
            except KeyError as error:
                raise ValueError(f"structured token-origin array uses an unknown ID for {record_path}") from error
            if expected_row is not None and (
                origins != expected_row["token_origins"]
                or details != expected_row["token_origin_details"]
            ):
                raise ValueError(f"structured token-origin arrays mismatch source for {record_path}")
            expected_anchors: list[dict[str, Any]] = []
            expected_spans: list[dict[str, Any]] = []
            for turn in structured_metadata["assistant_action_turn_records"]:
                for field in turn["fields"]:
                    sampled = int(field["sampled_token_index"])
                    anchor = int(field["anchor_token_index"])
                    if (
                        sampled < 0
                        or sampled >= length
                        or anchor != sampled - 1
                        or int(token_ids[sampled]) != int(field["raw_token_id"])
                        or origins[sampled] != field["raw_token_origin"]
                    ):
                        raise ValueError(f"structured action field alignment mismatch for {record_path}")
                    field_name = str(field["field_name"])
                    expected_anchors.append({
                        "turn_index": int(turn["turn_index"]),
                        "field_name": field_name,
                        "field_event_id": str(field["field_event_id"]),
                        "anchor_token_index": anchor,
                        "anchor_token_id": int(token_ids[anchor]),
                        "sampled_token_index": sampled,
                        "anchor_semantic_position": (
                            "immediately_before_status_sample"
                            if field_name == "status"
                            else "immediately_before_caveat_sample_status_visible"
                        ),
                    })
                    expected_spans.append({
                        "turn_index": int(turn["turn_index"]),
                        "field_name": field_name,
                        "token_start": sampled,
                        "token_end": sampled + 1,
                        "token_origin": str(field["raw_token_origin"]),
                    })
            if (
                structured_metadata["action_anchors"] != expected_anchors
                or structured_metadata["action_token_spans"] != expected_spans
            ):
                raise ValueError(f"structured action anchor/span metadata mismatch for {record_path}")
            for eot in structured_metadata["environment_eot_records"]:
                eot_index = int(eot["token_index"])
                if (
                    eot_index < 0
                    or eot_index >= length
                    or int(token_ids[eot_index]) != int(eot["token_id"])
                    or origins[eot_index] != eot["token_origin"]
                ):
                    raise ValueError(f"structured environment EOT mismatch for {record_path}")
        dtype_codes = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}
        if residual_dtype not in dtype_codes or attention_dtype not in dtype_codes:
            raise ValueError(f"unsupported recorded dtype for {record_path}")
        edge_count = length * (length + 1) // 2
        for layer in layers:
            residual_slice = handle.get_slice(f"residual.L{layer}")
            attention_slice = handle.get_slice(f"attention.L{layer}")
            if residual_slice.get_shape() != [length, hidden_size]:
                raise ValueError(f"residual L{layer} shape mismatch for {record_path}")
            if residual_slice.get_dtype() != dtype_codes[residual_dtype]:
                raise ValueError(f"residual L{layer} dtype mismatch for {record_path}")
            if attention_slice.get_shape() != [n_attention_heads, edge_count]:
                raise ValueError(f"attention L{layer} shape mismatch for {record_path}")
            if attention_slice.get_dtype() != dtype_codes[attention_dtype]:
                raise ValueError(f"attention L{layer} dtype mismatch for {record_path}")
        if int(record["stop_token_index"]) != length - 1:
            raise ValueError(f"terminal token index mismatch for {record_path}")
        if int(token_ids[-1]) != int(record["stop_token_id"]):
            raise ValueError(f"terminal token mismatch for {record_path}")
    return record


def recover_orphan_relational_row(
    tensor_path: Path,
    row: dict[str, Any],
    *,
    capture_contract_sha256: str,
    layers: Sequence[int],
    hidden_size: int,
    n_attention_heads: int,
    residual_dtype: str,
    attention_dtype: str,
) -> dict[str, Any]:
    expected_tensor, record_path = relational_row_paths(
        tensor_path.parents[1], str(row["conversation_id"])
    )
    if tensor_path != expected_tensor or not tensor_path.is_file() or record_path.exists():
        raise ValueError(f"not a recoverable tensor-only orphan: {tensor_path}")
    record = _row_record(
        row,
        tensor_path,
        capture_contract_sha256=capture_contract_sha256,
        layers=layers,
        hidden_size=hidden_size,
        n_attention_heads=n_attention_heads,
        residual_dtype=residual_dtype,
        attention_dtype=attention_dtype,
    )
    _atomic_record(record_path, record)
    try:
        return validate_relational_row_artifact(
            record_path,
            expected_row=row,
            expected_contract_sha256=capture_contract_sha256,
            expected_layers=layers,
            expected_hidden_size=hidden_size,
            expected_attention_heads=n_attention_heads,
            expected_residual_dtype=residual_dtype,
            expected_attention_dtype=attention_dtype,
            verify_checksum=False,
        )
    except BaseException:
        record_path.unlink(missing_ok=True)
        raise


__all__ = [
    "CapturedRelationalRow", "canonical_json_sha256", "capture_relational_batch",
    "compare_captured_rows", "missing_relational_parity_indices",
    "recover_orphan_relational_row", "relational_row_paths",
    "set_eager_attention", "validate_relational_parity_entries",
    "validate_relational_parity_scope", "validate_relational_row_artifact",
    "write_relational_row",
]
