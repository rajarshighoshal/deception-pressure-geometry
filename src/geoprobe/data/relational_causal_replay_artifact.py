"""Immutable exact-prefix replay-plan artifacts for masked causal vector roots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from safetensors import safe_open

from geoprobe.control.relational_pre_status_vector_bank import (
    CausalVectorBankArrays,
    LEDGER_FILE_NAME as VECTOR_LEDGER_FILE_NAME,
    TENSOR_FILE_NAME as VECTOR_TENSOR_FILE_NAME,
    load_pre_status_causal_vector_bank,
)
from geoprobe.data.relational_causal_replay import (
    RelationalCausalReplayPlanEvent,
    build_relational_causal_replay_inventory,
    join_relational_causal_replay_plan,
)
from geoprobe.geometry.relational_pre_status_rooted_graph import FOLDS
from geoprobe.io import file_sha256
from geoprobe.models.relational_structured_action import int32_token_sha256


SCHEMA_VERSION: Final = 1
LEDGER_NAME: Final = "ledger.json"
EVENTS_NAME: Final = "events.jsonl"
PREFIX_TENSOR_NAME: Final = "prefixes.safetensors"
LEDGER_KIND: Final = "relational_pre_status_causal_replay_plan"
TENSOR_KIND: Final = "relational_pre_status_causal_replay_prefixes"
EVENT_FIELDS: Final = frozenset(
    {
        "replay_event_index",
        "event_id",
        "root_id",
        "family_fold",
        "vector_tensor_row_index",
        "scenario_id",
        "family",
        "turn_index",
        "true_status",
        "desired_status",
        "knowledge_correct",
        "knowledge_status",
        "prefix_token_ids_sha256",
        "prefix_offset",
        "prefix_count",
        "rng_seed",
        "historical_action_provenance",
        "source_conversation_ids",
        "source_row_sha256s",
    }
)


class RelationalCausalReplayArtifactError(ValueError):
    """A replay-plan input or immutable artifact fails its exact identity contract."""


@dataclass(frozen=True, slots=True)
class CausalReplayPrefixArrays:
    """Compact deterministic int32 exact-prefix storage in replay-event order."""

    flat_token_ids: np.ndarray
    offsets: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True, slots=True)
class RelationalCausalReplayArtifactBuild:
    """Validated replay plan ready for immutable persistence."""

    event_rows: tuple[Mapping[str, Any], ...]
    prefix_arrays: CausalReplayPrefixArrays
    ledger_payload: Mapping[str, Any]


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RelationalCausalReplayArtifactError("value is not canonical finite JSON") from error


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RelationalCausalReplayArtifactError(f"{label} must be a lowercase SHA-256")
    return value


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("ledger_sha256", None)
    return sha256(_canonical(payload)).hexdigest()


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(dict(row)) + b"\n" for row in rows)


def _readonly_int32(value: object, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise RelationalCausalReplayArtifactError(f"{label} must be one-dimensional int32")
    limits = np.iinfo(np.int32)
    if raw.size and (raw.min() < limits.min or raw.max() > limits.max):
        raise RelationalCausalReplayArtifactError(f"{label} exceeds int32 capacity")
    array = np.ascontiguousarray(raw, dtype=np.int32)
    result = array.copy()
    result.flags.writeable = False
    return result


def _validate_prefix_arrays(arrays: CausalReplayPrefixArrays, count: int) -> None:
    flat, offsets, counts = arrays.flat_token_ids, arrays.offsets, arrays.counts
    if any(value.dtype != np.int32 or value.ndim != 1 for value in (flat, offsets, counts)):
        raise RelationalCausalReplayArtifactError("prefix tensors must be one-dimensional int32")
    if len(offsets) != count or len(counts) != count or (flat < 0).any() or (offsets < 0).any() or (counts <= 0).any():
        raise RelationalCausalReplayArtifactError("prefix tensor dimensions or values are invalid")
    expected = 0
    for offset, length in zip(offsets.tolist(), counts.tolist(), strict=True):
        if offset != expected or offset + length > len(flat):
            raise RelationalCausalReplayArtifactError("prefix offsets/counts are not contiguous")
        expected += length
    if expected != len(flat):
        raise RelationalCausalReplayArtifactError("prefix tensors have trailing tokens")


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationalCausalReplayArtifactError(f"{label} must be a non-empty string")
    return value


def _unsigned_integer(value: object, label: str, *, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RelationalCausalReplayArtifactError(f"{label} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise RelationalCausalReplayArtifactError(f"{label} exceeds its frozen range")
    return value


def _validate_event_row(
    row: Mapping[str, Any],
    arrays: CausalReplayPrefixArrays,
    index: int,
) -> None:
    if set(row) != EVENT_FIELDS:
        raise RelationalCausalReplayArtifactError("replay-plan event schema is invalid")
    if _unsigned_integer(row.get("replay_event_index"), "event index") != index:
        raise RelationalCausalReplayArtifactError("replay-plan event order is invalid")
    _nonempty_string(row.get("event_id"), "event ID")
    _sha(row.get("root_id"), "event root ID")
    fold = _nonempty_string(row.get("family_fold"), "event family fold")
    if fold not in FOLDS:
        raise RelationalCausalReplayArtifactError("event family fold is not frozen")
    _unsigned_integer(row.get("vector_tensor_row_index"), "vector tensor-row index")
    _nonempty_string(row.get("scenario_id"), "event scenario ID")
    _nonempty_string(row.get("family"), "event family")
    _unsigned_integer(row.get("turn_index"), "event turn index", maximum=3)
    true_status = _nonempty_string(row.get("true_status"), "event true status")
    desired_status = _nonempty_string(row.get("desired_status"), "event desired status")
    knowledge_status = _nonempty_string(row.get("knowledge_status"), "event knowledge status")
    if any(value not in {"PASS", "FAIL"} for value in (true_status, desired_status, knowledge_status)):
        raise RelationalCausalReplayArtifactError("event status semantics are invalid")
    knowledge_correct = row.get("knowledge_correct")
    if not isinstance(knowledge_correct, bool) or (knowledge_status == true_status) != knowledge_correct:
        raise RelationalCausalReplayArtifactError("event knowledge semantics are invalid")
    prefix_sha = _sha(row.get("prefix_token_ids_sha256"), "event prefix hash")
    offset = _unsigned_integer(row.get("prefix_offset"), "event prefix offset")
    count = _unsigned_integer(row.get("prefix_count"), "event prefix count")
    if count == 0 or offset != int(arrays.offsets[index]) or count != int(arrays.counts[index]):
        raise RelationalCausalReplayArtifactError("replay-plan event/prefix mapping is invalid")
    tokens = arrays.flat_token_ids[offset : offset + count].tolist()
    if int32_token_sha256(tokens) != prefix_sha:
        raise RelationalCausalReplayArtifactError("replay-plan prefix differs from its event hash")
    _unsigned_integer(row.get("rng_seed"), "event RNG seed", maximum=2**64 - 1)
    historical = row.get("historical_action_provenance")
    if not isinstance(historical, Mapping) or set(historical) != {
        "historical_raw_token_id",
        "historical_raw_decoded_exact",
        "historical_mapped_action",
    }:
        raise RelationalCausalReplayArtifactError("historical action provenance is invalid")
    _unsigned_integer(historical.get("historical_raw_token_id"), "historical raw token ID")
    _nonempty_string(historical.get("historical_raw_decoded_exact"), "historical decoded token")
    if historical.get("historical_mapped_action") not in {"PASS", "FAIL", "SKIP", "NO_ACTION"}:
        raise RelationalCausalReplayArtifactError("historical mapped action is invalid")
    conversations = row.get("source_conversation_ids")
    source_hashes = row.get("source_row_sha256s")
    if (
        not isinstance(conversations, list)
        or not conversations
        or len(set(conversations)) != len(conversations)
        or any(not isinstance(value, str) or not value for value in conversations)
        or not isinstance(source_hashes, list)
        or not source_hashes
        or len(set(source_hashes)) != len(source_hashes)
    ):
        raise RelationalCausalReplayArtifactError("event source provenance is invalid")
    for source_hash in source_hashes:
        _sha(source_hash, "source row hash")


def _prefix_arrays(plan: Sequence[RelationalCausalReplayPlanEvent]) -> CausalReplayPrefixArrays:
    flat: list[int] = []
    offsets: list[int] = []
    counts: list[int] = []
    for entry in plan:
        tokens = entry.replay.prefix_token_ids
        if len(flat) > np.iinfo(np.int32).max or len(tokens) > np.iinfo(np.int32).max:
            raise RelationalCausalReplayArtifactError("prefix storage exceeds int32 capacity")
        offsets.append(len(flat))
        counts.append(len(tokens))
        flat.extend(tokens)
    arrays = CausalReplayPrefixArrays(
        _readonly_int32(flat, "flat prefix tokens"),
        _readonly_int32(offsets, "prefix offsets"),
        _readonly_int32(counts, "prefix counts"),
    )
    _validate_prefix_arrays(arrays, len(plan))
    return arrays


def _defined_rows(ledger: Mapping[str, Any], arrays: CausalVectorBankArrays) -> tuple[list[Mapping[str, Any]], Mapping[str, int]]:
    rows = ledger.get("rows")
    if not isinstance(rows, list) or len(rows) != arrays.defined.shape[0]:
        raise RelationalCausalReplayArtifactError("validated vector ledger lacks rows matching its tensors")
    selected: list[Mapping[str, Any]] = []
    excluded_roots = 0
    excluded_events = 0
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or raw.get("tensor_row_index") != index:
            raise RelationalCausalReplayArtifactError("vector ledger tensor-row mapping is invalid")
        event_ids = raw.get("event_ids")
        if not isinstance(event_ids, list) or not event_ids:
            raise RelationalCausalReplayArtifactError("vector ledger root has no event IDs")
        if bool(np.all(arrays.defined[index])):
            selected.append(raw)
        else:
            excluded_roots += 1
            excluded_events += len(event_ids)
    return selected, MappingProxyType({"undefined_root_count": excluded_roots, "undefined_event_count": excluded_events})


def load_all_defined_pre_status_causal_vector_rows(
    vector_bank_root: Path,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    """Open every frozen fold bank and retain only H/T/S/G support-defined roots."""
    root = Path(vector_bank_root).resolve()
    selected: list[Mapping[str, Any]] = []
    bindings: dict[str, Any] = {}
    root_ids: set[str] = set()
    for fold in FOLDS:
        fold_root = root / fold
        ledger, arrays = load_pre_status_causal_vector_bank(fold_root)
        if ledger.get("held_out_family_fold") != fold:
            raise RelationalCausalReplayArtifactError("vector ledger fold directory disagrees with its ledger")
        fold_rows, exclusion = _defined_rows(ledger, arrays)
        for row in fold_rows:
            root_id = _sha(row.get("root_id"), "vector root ID")
            if root_id in root_ids:
                raise RelationalCausalReplayArtifactError("selected vector roots repeat across folds")
            root_ids.add(root_id)
        ledger_path, tensor_path = fold_root / VECTOR_LEDGER_FILE_NAME, fold_root / VECTOR_TENSOR_FILE_NAME
        if not ledger_path.is_file() or not tensor_path.is_file():
            raise RelationalCausalReplayArtifactError("validated vector bank has missing physical files")
        bindings[fold] = {
            "ledger": {"path": str(ledger_path), "sha256": file_sha256(ledger_path), "bytes": ledger_path.stat().st_size, "internal_sha256": ledger.get("ledger_sha256")},
            "tensor": {"path": str(tensor_path), "sha256": file_sha256(tensor_path), "bytes": tensor_path.stat().st_size},
            "candidate_root_count": len(ledger["rows"]), "selected_root_count": len(fold_rows), **dict(exclusion),
        }
        selected.extend(fold_rows)
    return tuple(selected), MappingProxyType(bindings)


def build_relational_causal_replay_artifact(
    rows: Sequence[Mapping[str, Any]],
    vector_rows: Sequence[Mapping[str, Any]],
    *,
    rollout_binding: Mapping[str, Any],
    vector_bank_bindings: Mapping[str, Any],
    argv: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
) -> RelationalCausalReplayArtifactBuild:
    """Join support-defined roots to exact, independent status replay streams."""
    inventory = build_relational_causal_replay_inventory(rows)
    plan = join_relational_causal_replay_plan(inventory, vector_rows)
    event_ids = [item.replay.event_id for item in plan]
    streams = [(item.replay.event_id, item.replay.rng_seed) for item in plan]
    if len(set(event_ids)) != len(plan) or len(set(streams)) != len(plan):
        raise RelationalCausalReplayArtifactError("selected roots do not map to distinct event/RNG streams")
    root_ids = [item.root_id for item in plan]
    prefix_arrays = _prefix_arrays(plan)
    event_rows: list[Mapping[str, Any]] = []
    for index, item in enumerate(plan):
        replay = item.replay
        event_rows.append({
            "replay_event_index": index, "event_id": replay.event_id, "root_id": item.root_id,
            "family_fold": item.family_fold, "vector_tensor_row_index": item.vector_tensor_row_index,
            "scenario_id": replay.scenario_id, "family": replay.family, "turn_index": replay.turn_index,
            "true_status": replay.true_status, "desired_status": replay.desired_status,
            "knowledge_correct": replay.knowledge_correct,
            "knowledge_status": replay.knowledge_status,
            "prefix_token_ids_sha256": replay.prefix_token_ids_sha256,
            "prefix_offset": int(prefix_arrays.offsets[index]), "prefix_count": int(prefix_arrays.counts[index]),
            "rng_seed": replay.rng_seed,
            "historical_action_provenance": {
                "historical_raw_token_id": replay.historical_raw_token_id,
                "historical_raw_decoded_exact": replay.historical_raw_decoded_exact,
                "historical_mapped_action": replay.historical_mapped_action,
            },
            "source_conversation_ids": list(replay.source_conversation_ids),
            "source_row_sha256s": list(replay.source_row_sha256s),
        })
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "kind": LEDGER_KIND,
        "selection_did_not_use_historical_action": True,
        "argv": list(argv), "provenance": dict(provenance or {}),
        "rollout_binding": dict(rollout_binding), "vector_bank_bindings": dict(vector_bank_bindings),
        "selected_unique_vector_state_count": len(set(root_ids)), "replay_event_count": len(event_rows),
        "inventory_event_count": len(inventory),
        "excluded_undefined_root_count": sum(int(entry["undefined_root_count"]) for entry in vector_bank_bindings.values()),
        "excluded_undefined_event_count": sum(int(entry["undefined_event_count"]) for entry in vector_bank_bindings.values()),
        "prefix_tensor": {"path": PREFIX_TENSOR_NAME, "event_count": len(event_rows), "flat_token_count": len(prefix_arrays.flat_token_ids)},
        "events": {"path": EVENTS_NAME, "event_count": len(event_rows), "content_sha256": sha256(_jsonl_bytes(event_rows)).hexdigest()},
    }
    return RelationalCausalReplayArtifactBuild(tuple(event_rows), prefix_arrays, MappingProxyType(payload))


def _write_safetensors_new(path: Path, arrays: CausalReplayPrefixArrays, metadata: Mapping[str, str]) -> None:
    payloads = (("flat_token_ids", arrays.flat_token_ids), ("offsets", arrays.offsets), ("counts", arrays.counts))
    offset = 0
    header: dict[str, Any] = {"__metadata__": dict(sorted(metadata.items()))}
    values: list[bytes] = []
    for name, array in payloads:
        data = np.ascontiguousarray(array, dtype="<i4").tobytes(order="C")
        values.append(data)
        header[name] = {"dtype": "I32", "shape": list(array.shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with path.open("xb") as handle:
        handle.write(len(encoded).to_bytes(8, "little"))
        handle.write(encoded)
        for value in values:
            handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_new(path: Path, content: bytes) -> None:
    if path.exists():
        raise RelationalCausalReplayArtifactError("refusing to overwrite replay-plan output")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _validate_file_binding(value: object, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise RelationalCausalReplayArtifactError(f"{label} file binding is invalid")
    path_text, expected_sha, expected_bytes = value.get("path"), value.get("sha256"), value.get("bytes")
    if not isinstance(path_text, str) or not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes < 0:
        raise RelationalCausalReplayArtifactError(f"{label} file binding is invalid")
    path = Path(path_text)
    if not path.is_file() or path.stat().st_size != expected_bytes or file_sha256(path) != _sha(expected_sha, f"{label} SHA-256"):
        raise RelationalCausalReplayArtifactError(f"{label} physical binding is invalid")
    return path


def _validate_input_bindings(ledger: Mapping[str, Any]) -> None:
    rollout = ledger.get("rollout_binding")
    vector_banks = ledger.get("vector_bank_bindings")
    if not isinstance(rollout, Mapping) or not isinstance(vector_banks, Mapping):
        raise RelationalCausalReplayArtifactError("replay-plan input bindings are invalid")
    rows_path = _validate_file_binding(rollout.get("rows"), "rollout rows")
    manifest_path = _validate_file_binding(rollout.get("rollout_manifest"), "rollout manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RelationalCausalReplayArtifactError("bound rollout manifest is invalid") from error
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("kind") != "relational_structured_action_rollout_manifest"
        or manifest.get("status") != "success"
        or manifest.get("output", {}).get("rows_sha256") != file_sha256(rows_path)
    ):
        raise RelationalCausalReplayArtifactError("bound rollout manifest does not bind its rows")
    if set(vector_banks) != set(FOLDS):
        raise RelationalCausalReplayArtifactError("replay-plan does not bind all five vector folds")
    for fold in FOLDS:
        entry = vector_banks[fold]
        if not isinstance(entry, Mapping):
            raise RelationalCausalReplayArtifactError("vector-bank binding is invalid")
        _validate_file_binding(entry.get("ledger"), f"{fold} vector ledger")
        _validate_file_binding(entry.get("tensor"), f"{fold} vector tensor")


def save_relational_causal_replay_artifact(build: RelationalCausalReplayArtifactBuild, out_dir: Path) -> Mapping[str, Any]:
    """Persist the immutable JSONL ledger and deterministic compact prefix tensor."""
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tensor_path, events_path, ledger_path = root / PREFIX_TENSOR_NAME, root / EVENTS_NAME, root / LEDGER_NAME
    if any(path.exists() for path in (tensor_path, events_path, ledger_path)):
        raise RelationalCausalReplayArtifactError("refusing to overwrite replay-plan artifact")
    _validate_prefix_arrays(build.prefix_arrays, len(build.event_rows))
    temporary = Path(tempfile.NamedTemporaryFile(dir=root, suffix=".safetensors.tmp", delete=False).name)
    temporary.unlink()
    try:
        _write_safetensors_new(temporary, build.prefix_arrays, {"schema_version": "1", "kind": TENSOR_KIND, "event_count": str(len(build.event_rows))})
        os.link(temporary, tensor_path)
    finally:
        temporary.unlink(missing_ok=True)
    event_bytes = _jsonl_bytes(build.event_rows)
    _write_new(events_path, event_bytes)
    ledger = dict(build.ledger_payload)
    ledger["prefix_tensor"].update({"sha256": file_sha256(tensor_path), "bytes": tensor_path.stat().st_size, "arrays": {"flat_token_ids": list(build.prefix_arrays.flat_token_ids.shape), "offsets": list(build.prefix_arrays.offsets.shape), "counts": list(build.prefix_arrays.counts.shape)}})
    ledger["events"].update({"sha256": file_sha256(events_path), "bytes": events_path.stat().st_size})
    ledger["ledger_sha256"] = _self_hash(ledger)
    _write_new(ledger_path, json.dumps(ledger, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
    return MappingProxyType(ledger)


def load_relational_causal_replay_artifact(
    root: Path,
    *,
    validate_source_bindings: bool = True,
) -> tuple[
    Mapping[str, Any],
    tuple[Mapping[str, Any], ...],
    CausalReplayPrefixArrays,
]:
    """Load an immutable replay artifact and optionally re-open build-time sources.

    Execution stages are intentionally relocatable: they validate the plan's
    self-bound event/prefix files here, then bind the separately supplied vector
    bank in the runner. Build/audit callers retain the stricter default that
    also re-opens the original rollout and vector-bank paths.
    """
    artifact_root = Path(root).resolve()
    try:
        ledger = json.loads((artifact_root / LEDGER_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RelationalCausalReplayArtifactError("replay-plan ledger is not valid UTF-8 JSON") from error
    if not isinstance(ledger, Mapping) or ledger.get("schema_version") != SCHEMA_VERSION or ledger.get("kind") != LEDGER_KIND or ledger.get("ledger_sha256") != _self_hash(ledger):
        raise RelationalCausalReplayArtifactError("replay-plan ledger schema or self-hash is invalid")
    if ledger.get("selection_did_not_use_historical_action") is not True:
        raise RelationalCausalReplayArtifactError("replay-plan selection provenance is invalid")
    if not isinstance(validate_source_bindings, bool):
        raise RelationalCausalReplayArtifactError(
            "validate_source_bindings must be Boolean"
        )
    if validate_source_bindings:
        _validate_input_bindings(ledger)
    events_meta, tensor_meta = ledger.get("events"), ledger.get("prefix_tensor")
    if not isinstance(events_meta, Mapping) or not isinstance(tensor_meta, Mapping):
        raise RelationalCausalReplayArtifactError("replay-plan physical metadata is invalid")
    events_path, tensor_path = artifact_root / EVENTS_NAME, artifact_root / PREFIX_TENSOR_NAME
    if (
        events_meta.get("path") != EVENTS_NAME
        or tensor_meta.get("path") != PREFIX_TENSOR_NAME
        or not events_path.is_file()
        or not tensor_path.is_file()
        or events_meta.get("bytes") != events_path.stat().st_size
        or tensor_meta.get("bytes") != tensor_path.stat().st_size
        or events_meta.get("sha256") != file_sha256(events_path)
        or tensor_meta.get("sha256") != file_sha256(tensor_path)
    ):
        raise RelationalCausalReplayArtifactError("replay-plan physical bindings are invalid")
    try:
        raw_rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RelationalCausalReplayArtifactError("replay-plan event JSONL is invalid") from error
    if not all(isinstance(row, Mapping) for row in raw_rows) or events_meta.get("event_count") != len(raw_rows) or events_meta.get("content_sha256") != sha256(_jsonl_bytes(raw_rows)).hexdigest():
        raise RelationalCausalReplayArtifactError("replay-plan event ledger binding is invalid")
    with safe_open(tensor_path, framework="numpy") as handle:
        if handle.metadata() != {"schema_version": "1", "kind": TENSOR_KIND, "event_count": str(len(raw_rows))} or set(handle.keys()) != {"flat_token_ids", "offsets", "counts"}:
            raise RelationalCausalReplayArtifactError("replay-plan prefix tensor metadata is invalid")
        arrays = CausalReplayPrefixArrays(*(_readonly_int32(handle.get_tensor(name), name) for name in ("flat_token_ids", "offsets", "counts")))
    _validate_prefix_arrays(arrays, len(raw_rows))
    expected_shapes = {
        "flat_token_ids": list(arrays.flat_token_ids.shape),
        "offsets": list(arrays.offsets.shape),
        "counts": list(arrays.counts.shape),
    }
    if (
        tensor_meta.get("event_count") != len(raw_rows)
        or tensor_meta.get("flat_token_count") != len(arrays.flat_token_ids)
        or tensor_meta.get("arrays") != expected_shapes
    ):
        raise RelationalCausalReplayArtifactError("replay-plan prefix tensor counts are invalid")
    for index, row in enumerate(raw_rows):
        _validate_event_row(row, arrays, index)
    event_ids = [row["event_id"] for row in raw_rows]
    streams = [(row["event_id"], row["rng_seed"]) for row in raw_rows]
    root_ids = {row["root_id"] for row in raw_rows}
    if len(set(event_ids)) != len(event_ids) or len(set(streams)) != len(streams):
        raise RelationalCausalReplayArtifactError("replay-plan event/RNG streams are not unique")
    if (
        ledger.get("replay_event_count") != len(raw_rows)
        or ledger.get("selected_unique_vector_state_count") != len(root_ids)
    ):
        raise RelationalCausalReplayArtifactError("replay-plan ledger counts are invalid")
    return MappingProxyType(dict(ledger)), tuple(MappingProxyType(dict(row)) for row in raw_rows), arrays


__all__ = [
    "CausalReplayPrefixArrays", "EVENTS_NAME", "LEDGER_NAME", "PREFIX_TENSOR_NAME",
    "RelationalCausalReplayArtifactBuild", "RelationalCausalReplayArtifactError",
    "build_relational_causal_replay_artifact", "load_all_defined_pre_status_causal_vector_rows",
    "load_relational_causal_replay_artifact", "save_relational_causal_replay_artifact",
]
