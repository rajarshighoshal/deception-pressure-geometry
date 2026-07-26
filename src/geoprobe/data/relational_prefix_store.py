"""Label-free exact causal-prefix indexing and loading for schema-v2 captures."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
from typing import Any

import numpy as np
from safetensors import safe_open
import torch

from geoprobe.geometry.relational import TypedRelationalObject
from geoprobe.io import file_sha256


_LAYERS = (12, 16, 19, 20)
_MANIFEST_KIND = "relational_state_graph_materialization_manifest"
_GEOMETRY_ROW_KIND = "relational_state_graph_geometry_row"
_GEOMETRY_PROJECTION_KIND = (
    "relational_structured_action_state_graph_geometry_projection"
)
_TOKEN_KEYS = (
    "token_ids",
    "position_ids",
    "token_role_ids",
    "token_turn_ids",
    "token_message_ids",
    "token_span_flags",
    "token_origin_ids",
    "token_origin_detail_ids",
)
_ANNOTATION_KEYS = (
    "token_role_ids",
    "token_turn_ids",
    "token_message_ids",
    "token_span_flags",
    "token_origins",
    "token_origin_details",
)
_FORBIDDEN_INDEX_KEY_PARTS = ("outcome", "target", "label", "cohort")
_ROW_ACTIVATION_HASH_DOMAIN = b"geoprobe.row-activation-prefixes.v1\x00"
_FROZEN_TOKEN_ORIGINS = {
    0: "chat_source",
    1: "environment",
    2: "model_sample",
}
_FROZEN_TOKEN_ORIGIN_DETAILS = {
    0: "chat_source",
    1: "environment_status_prefix",
    2: "model_sampled_status_action",
    3: "environment_caveat_separator",
    4: "model_sampled_caveat_action",
    5: "environment_eot",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _int32_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}i", *values)).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _require_sha256(value: Any, name: str) -> str:
    text = _require_string(value, name)
    if len(text) != 64 or text.lower() != text:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    try:
        bytes.fromhex(text)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256") from error
    return text


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _relative_artifact_path(root: Path, value: Any, name: str) -> Path:
    text = _require_string(value, name)
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"{name} must stay within the artifact directory")
    path = root.joinpath(*pure.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as error:
        raise ValueError(f"{name} must stay within the artifact directory") from error
    return path


def _object_path(value: Any) -> str:
    text = _require_string(value, "source tensor object path")
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or text.endswith("/"):
        raise ValueError("source tensor object path must be a safe relative object key")
    return text


def _assert_file_binding(
    path: Path,
    *,
    expected_sha256: Any,
    expected_bytes: Any,
    name: str,
) -> str:
    digest = _require_sha256(expected_sha256, f"{name} SHA-256")
    size = _require_int(expected_bytes, f"{name} byte count", minimum=1)
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"{name} file or byte-count binding is invalid")
    if file_sha256(path) != digest:
        raise ValueError(f"{name} SHA-256 binding is invalid")
    return digest


def _assert_label_free(value: Any, path: str = "index") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(part in normalized for part in _FORBIDDEN_INDEX_KEY_PARTS):
                raise ValueError(f"forbidden field escaped into {path}: {key}")
            _assert_label_free(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_label_free(item, f"{path}[{index}]")


def _assert_geometry_projection_safe(value: Any, path: str = "geometry") -> None:
    if isinstance(value, Mapping):
        prohibited = {
            "committed_status",
            "committed_caveat",
            "mapped_action",
            "observed_action",
            "raw_token_id",
            "sampled_token_id",
            "true_status",
            "desired_status",
            "knowledge_correct",
            "scientific_cohort",
        }
        for key, item in value.items():
            normalized = str(key).lower()
            if (
                "outcome" in normalized
                or "label" in normalized
                or "deceptive" in normalized
                or normalized in prohibited
            ):
                raise ValueError(f"label-bearing field escaped into {path}: {key}")
            _assert_geometry_projection_safe(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_geometry_projection_safe(item, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class CompactTensorLocators:
    tensor_shard: str
    tensor_shard_sha256: str
    anchor_residual_key: str
    anchor_residual_occurrence_index: int
    incoming_attention_values_key: str
    incoming_attention_offsets_key: str
    incoming_attention_lengths_key: str
    incoming_attention_index: int
    anchor_token_indices_key: str
    layer_ids_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_shard": self.tensor_shard,
            "tensor_shard_sha256": self.tensor_shard_sha256,
            "anchor_residual_key": self.anchor_residual_key,
            "anchor_residual_occurrence_index": self.anchor_residual_occurrence_index,
            "incoming_attention_values_key": self.incoming_attention_values_key,
            "incoming_attention_offsets_key": self.incoming_attention_offsets_key,
            "incoming_attention_lengths_key": self.incoming_attention_lengths_key,
            "incoming_attention_index": self.incoming_attention_index,
            "anchor_token_indices_key": self.anchor_token_indices_key,
            "layer_ids_key": self.layer_ids_key,
        }


@dataclass(frozen=True, slots=True)
class RelationalPrefixReference:
    reference_id: str
    realization_sha256: str
    canonical_realization_id: str
    equality_status: str
    occurrence_id: str
    equivalent_occurrence_ids: tuple[str, ...]
    field_event_id: str
    field_name: str
    turn_index: int
    conversation_id: str
    family: str
    family_fold: str
    split: str
    scenario_id: str
    orbit_id: str
    sample_index: int
    prefix_state_sha256: str
    prefix_token_ids_sha256: str
    prefix_token_stop: int
    packed_attention_prefix_stop: int
    source_row_sha256: str
    source_tensor_object_path: str
    source_tensor_sha256: str
    capture_contract_sha256: str
    layers: tuple[int, ...]
    intervention_history: tuple[str, ...]
    pressure_exposed: bool
    intervention_token_indices: tuple[int, ...]
    intervention_mask_scope: str
    compact: CompactTensorLocators

    def to_dict(self) -> dict[str, Any]:
        value = {
            "reference_id": self.reference_id,
            "realization_sha256": self.realization_sha256,
            "canonical_realization_id": self.canonical_realization_id,
            "equality_status": self.equality_status,
            "occurrence_id": self.occurrence_id,
            "equivalent_occurrence_ids": list(self.equivalent_occurrence_ids),
            "field_event_id": self.field_event_id,
            "field_name": self.field_name,
            "turn_index": self.turn_index,
            "conversation_id": self.conversation_id,
            "family": self.family,
            "family_fold": self.family_fold,
            "split": self.split,
            "scenario_id": self.scenario_id,
            "orbit_id": self.orbit_id,
            "sample_index": self.sample_index,
            "prefix_state_sha256": self.prefix_state_sha256,
            "prefix_token_ids_sha256": self.prefix_token_ids_sha256,
            "prefix_token_stop": self.prefix_token_stop,
            "packed_attention_prefix_stop": self.packed_attention_prefix_stop,
            "source_row_sha256": self.source_row_sha256,
            "source_tensor_object_path": self.source_tensor_object_path,
            "source_tensor_sha256": self.source_tensor_sha256,
            "capture_contract_sha256": self.capture_contract_sha256,
            "layers": list(self.layers),
            "intervention_history": list(self.intervention_history),
            "pressure_exposed": self.pressure_exposed,
            "intervention_token_indices": list(self.intervention_token_indices),
            "intervention_mask_scope": self.intervention_mask_scope,
            "compact": self.compact.to_dict(),
        }
        _assert_label_free(value)
        return value


@dataclass(frozen=True, slots=True)
class RelationalPrefixIndex:
    artifact_root: str
    artifact_manifest_sha256: str
    geometry_projection_sha256: str
    activation_prefix_sha256: str
    family_folds_sha256: str
    deduplicated_exact_realizations: bool
    references: tuple[RelationalPrefixReference, ...]

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": 1,
            "kind": "relational_exact_causal_prefix_index",
            "artifact_root": self.artifact_root,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "geometry_projection_sha256": self.geometry_projection_sha256,
            "activation_prefix_sha256": self.activation_prefix_sha256,
            "family_folds_sha256": self.family_folds_sha256,
            "deduplicated_exact_realizations": self.deduplicated_exact_realizations,
            "reference_count": len(self.references),
            "references": [reference.to_dict() for reference in self.references],
        }
        _assert_label_free(value)
        return value

    def by_reference_id(self, reference_id: str) -> RelationalPrefixReference:
        matches = [value for value in self.references if value.reference_id == reference_id]
        if len(matches) != 1:
            raise KeyError(reference_id)
        return matches[0]


@dataclass(frozen=True, slots=True)
class RawRelationalPrefixCapture:
    """One SHA-bound raw causal prefix with lossless token annotations.

    This is the label-free bridge needed by connection methods.  It exposes the
    exact residual and causal-attention sections together with the structural
    token grammar, while deliberately omitting sampled action values and all
    behavioral outcome fields from its interface.
    """

    object_id: str
    residuals: Mapping[int, np.ndarray]
    attentions: Mapping[int, np.ndarray]
    token_ids: np.ndarray
    position_ids: np.ndarray
    token_role_ids: np.ndarray
    token_turn_ids: np.ndarray
    token_message_ids: np.ndarray
    token_span_flags: np.ndarray
    token_origins: tuple[str, ...]
    token_origin_details: tuple[str, ...]
    anchor_index: int


def _geometry_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("family geometry must be UTF-8 JSONL") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"family geometry row {line_number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"family geometry row {line_number} must be an object")
        geometry = row.get("geometry")
        if geometry is None:
            raise ValueError(f"family geometry row {line_number} lacks its projection")
        _assert_geometry_projection_safe(geometry)
        rows.append(row)
    return rows


def _validate_family_shard(
    path: Path,
    *,
    family: str,
    family_fold: str,
    manifest: Mapping[str, Any],
    expected_key_count: int,
) -> set[str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        expected_metadata = {
            "schema_version": "1",
            "kind": "relational_state_graph_family_activation_shard",
            "family": family,
            "family_fold": family_fold,
            "input_identity_sha256": str(manifest["input_identity_sha256"]),
            "label_payload": "absent",
        }
        if metadata is None or any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(f"family {family} compact shard metadata binding is invalid")
        keys = set(handle.keys())
    if len(keys) != expected_key_count:
        raise ValueError(f"family {family} compact shard key count is invalid")
    return keys


def _prefix_reference(
    *,
    row: Mapping[str, Any],
    realization: Mapping[str, Any],
    family_entry: Mapping[str, Any],
    capture_contract_sha256: str,
    compact_keys: set[str],
) -> RelationalPrefixReference:
    geometry = _require_mapping(row.get("geometry"), "geometry projection")
    if geometry.get("kind") != _GEOMETRY_PROJECTION_KIND:
        raise ValueError("geometry projection kind is invalid")
    conversation_id = _require_string(row.get("conversation_id"), "conversation_id")
    identity_keys = ("conversation_id", "family", "family_fold")
    if any(geometry.get(key) != row.get(key) for key in identity_keys):
        raise ValueError("geometry row/projection identity mismatch")
    occurrence_index = _require_int(realization.get("occurrence_index"), "occurrence_index")
    occurrences = _require_list(geometry.get("occurrences"), "geometry occurrences")
    states = _require_list(geometry.get("causal_prefix_states"), "causal prefix states")
    activations = _require_list(geometry.get("activation_metadata"), "activation metadata")
    if not (len(occurrences) == len(states) == len(activations)) or occurrence_index >= len(occurrences):
        raise ValueError("geometry occurrence inventories are inconsistent")
    occurrence = _require_mapping(occurrences[occurrence_index], "occurrence")
    state = _require_mapping(states[occurrence_index], "causal prefix state")
    activation = _require_mapping(activations[occurrence_index], "activation metadata")
    event_id = _require_string(realization.get("field_event_id"), "field_event_id")
    if occurrence.get("field_event_id") != event_id or activation.get("field_event_id") != event_id:
        raise ValueError("realization field-event identity mismatch")
    occurrence_identity = {
        "conversation_id": conversation_id,
        "family": row.get("family"),
        "family_fold": row.get("family_fold"),
        "split": geometry.get("split"),
        "scenario_id": geometry.get("scenario_id"),
        "orbit_id": geometry.get("orbit_id"),
        "sample_index": geometry.get("sample_index"),
    }
    if any(occurrence.get(key) != value for key, value in occurrence_identity.items()):
        raise ValueError("occurrence identity differs from its geometry row")
    if occurrence.get("prefix_state_sha256") != state.get("prefix_state_sha256") or (
        activation.get("prefix_state_sha256") != state.get("prefix_state_sha256")
    ):
        raise ValueError("realization causal-prefix identity mismatch")
    prefix_stop = _require_int(realization.get("prefix_token_stop"), "prefix token stop", minimum=1)
    packed_stop = _require_int(
        realization.get("packed_attention_prefix_stop"),
        "packed attention prefix stop",
        minimum=1,
    )
    if (
        state.get("prefix_token_count") != prefix_stop
        or activation.get("prefix_token_stop") != prefix_stop
        or packed_stop != prefix_stop * (prefix_stop + 1) // 2
        or activation.get("packed_attention_prefix_stop") != packed_stop
        or realization.get("anchor_token_index") != prefix_stop - 1
        or state.get("anchor_token_index") != prefix_stop - 1
    ):
        raise ValueError("realization prefix cutoffs are inconsistent")
    realization_sha = _require_sha256(
        realization.get("activation_prefix_sha256"), "activation prefix SHA-256"
    )
    canonical_id = _require_sha256(
        realization.get("canonical_realization_id"), "canonical realization ID"
    )
    equality_status = _require_string(realization.get("equality_status"), "equality status")
    if equality_status != "verified_exact_capture" or canonical_id != realization_sha:
        raise ValueError("realization lacks an exact-capture equality certificate")
    layers = tuple(_require_list(activation.get("layers"), "activation layers"))
    if layers != _LAYERS:
        raise ValueError("activation realization does not contain layers 12/16/19/20")
    tensor_keys = _require_mapping(realization.get("tensor_keys"), "tensor keys")
    tensor_offsets = _require_mapping(realization.get("tensor_offsets"), "tensor offsets")
    required_locator_names = {
        "anchor_residual",
        "incoming_attention_values",
        "incoming_attention_offsets",
        "incoming_attention_lengths",
    }
    if set(tensor_keys) != required_locator_names:
        raise ValueError("compact realization tensor locators are incomplete")
    locator_values = {name: _require_string(tensor_keys[name], name) for name in tensor_keys}
    tensor_prefix = _require_string(row.get("tensor_prefix"), "tensor prefix")
    anchor_indices_key = tensor_prefix + "anchor_token_indices"
    layer_ids_key = tensor_prefix + "layer_ids"
    if not set(locator_values.values()) | {anchor_indices_key, layer_ids_key} <= compact_keys:
        raise ValueError("compact realization tensor locator is absent from its bound shard")
    tensor_shard = _require_string(realization.get("tensor_shard"), "tensor shard")
    if tensor_shard != family_entry.get("tensor_path") or row.get("tensor_shard") != tensor_shard:
        raise ValueError("compact realization shard identity mismatch")
    occurrence_id = _require_string(occurrence.get("occurrence_id"), "occurrence ID")
    if tensor_offsets.get("anchor_residual_occurrence_index") != occurrence_index or (
        tensor_offsets.get("incoming_attention_index") != occurrence_index
    ):
        raise ValueError("compact tensor offsets differ from the occurrence index")
    source_tensor_sha = _require_sha256(
        row.get("source_tensor_sha256"), "source tensor SHA-256"
    )
    if activation.get("source_tensor_sha256") != source_tensor_sha:
        raise ValueError("activation source tensor SHA-256 differs from its geometry row")
    reference_id = _canonical_sha256(
        {
            "occurrence_id": occurrence_id,
            "realization_sha256": realization_sha,
            "source_tensor_sha256": row.get("source_tensor_sha256"),
        }
    )
    intervention_history = tuple(
        _require_string(action, "intervention history action")
        for action in _require_list(
            occurrence.get("intervention_history"), "intervention history"
        )
    )
    pressure_exposed = _require_bool(
        occurrence.get("pressure_exposed"), "pressure_exposed"
    )
    intervention_token_indices = tuple(
        _require_int(index, "intervention token index")
        for index in _require_list(
            occurrence.get("intervention_token_indices"),
            "intervention token indices",
        )
    )
    if pressure_exposed != bool(intervention_token_indices):
        raise ValueError(
            "pressure_exposed differs from the intervention token-mask exposure"
        )
    return RelationalPrefixReference(
        reference_id=reference_id,
        realization_sha256=realization_sha,
        canonical_realization_id=canonical_id,
        equality_status=equality_status,
        occurrence_id=occurrence_id,
        equivalent_occurrence_ids=(occurrence_id,),
        field_event_id=event_id,
        field_name=_require_string(occurrence.get("field_name"), "field name"),
        turn_index=_require_int(occurrence.get("turn_index"), "turn index"),
        conversation_id=conversation_id,
        family=_require_string(row.get("family"), "family"),
        family_fold=_require_string(row.get("family_fold"), "family fold"),
        split=_require_string(geometry.get("split"), "split"),
        scenario_id=_require_string(geometry.get("scenario_id"), "scenario_id"),
        orbit_id=_require_string(geometry.get("orbit_id"), "orbit_id"),
        sample_index=_require_int(geometry.get("sample_index"), "sample index"),
        prefix_state_sha256=_require_sha256(
            state.get("prefix_state_sha256"), "prefix-state SHA-256"
        ),
        prefix_token_ids_sha256=_require_sha256(
            state.get("prefix_token_ids_sha256"), "prefix token-ID SHA-256"
        ),
        prefix_token_stop=prefix_stop,
        packed_attention_prefix_stop=packed_stop,
        source_row_sha256=_require_sha256(row.get("source_row_sha256"), "source-row SHA-256"),
        source_tensor_object_path=_object_path(activation.get("source_tensor_path")),
        source_tensor_sha256=source_tensor_sha,
        capture_contract_sha256=capture_contract_sha256,
        layers=_LAYERS,
        intervention_history=intervention_history,
        pressure_exposed=pressure_exposed,
        intervention_token_indices=intervention_token_indices,
        intervention_mask_scope=_require_string(
            occurrence.get("intervention_mask_scope"), "intervention mask scope"
        ),
        compact=CompactTensorLocators(
            tensor_shard=tensor_shard,
            tensor_shard_sha256=_require_sha256(
                family_entry.get("tensor_sha256"), "compact tensor shard SHA-256"
            ),
            anchor_residual_key=locator_values["anchor_residual"],
            anchor_residual_occurrence_index=_require_int(
                tensor_offsets.get("anchor_residual_occurrence_index"),
                "anchor residual occurrence index",
            ),
            incoming_attention_values_key=locator_values["incoming_attention_values"],
            incoming_attention_offsets_key=locator_values["incoming_attention_offsets"],
            incoming_attention_lengths_key=locator_values["incoming_attention_lengths"],
            incoming_attention_index=_require_int(
                tensor_offsets.get("incoming_attention_index"),
                "incoming attention index",
            ),
            anchor_token_indices_key=anchor_indices_key,
            layer_ids_key=layer_ids_key,
        ),
    )


def _row_activation_sha256(realizations: Sequence[Mapping[str, Any]]) -> str:
    hasher = hashlib.sha256(_ROW_ACTIVATION_HASH_DOMAIN)
    for expected_index, realization in enumerate(realizations):
        index = _require_int(realization.get("occurrence_index"), "occurrence index")
        if index != expected_index:
            raise ValueError("activation realizations must be ordered by occurrence")
        event_id = _require_string(realization.get("field_event_id"), "field event ID")
        digest = _require_sha256(
            realization.get("activation_prefix_sha256"), "activation prefix SHA-256"
        )
        encoded = event_id.encode("utf-8")
        hasher.update(struct.pack("<I", index))
        hasher.update(struct.pack("<I", len(encoded)))
        hasher.update(encoded)
        hasher.update(bytes.fromhex(digest))
    return hasher.hexdigest()


def build_relational_prefix_index(
    artifact_root: Path,
    *,
    deduplicate_exact_realizations: bool = False,
) -> RelationalPrefixIndex:
    """Build an immutable label-free index over a completed state-graph artifact."""
    artifact_root = Path(artifact_root)
    manifest_path = artifact_root if artifact_root.is_file() else artifact_root / "manifest.json"
    root = manifest_path.parent
    manifest = _read_json(manifest_path, "state-graph manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != _MANIFEST_KIND
        or manifest.get("status") != "success"
    ):
        raise ValueError("state-graph manifest is not a successful completed artifact")
    manifest_sha = file_sha256(manifest_path)
    capture_contract = _require_mapping(manifest.get("capture_contract"), "capture contract")
    if tuple(_require_list(capture_contract.get("layers"), "capture layers")) != _LAYERS:
        raise ValueError("state-graph capture does not contain layers 12/16/19/20")
    if (
        capture_contract.get("residual_dtype") != "bfloat16"
        or capture_contract.get("attention_dtype") != "bfloat16"
    ):
        raise ValueError("state-graph capture is not native BF16")
    contract_sha = _require_sha256(
        capture_contract.get("capture_contract_sha256"), "capture contract SHA-256"
    )
    folds = _require_mapping(manifest.get("family_folds"), "family folds")
    folds_sha = _require_sha256(manifest.get("family_folds_sha256"), "family folds SHA-256")
    if _canonical_sha256(dict(sorted(folds.items()))) != folds_sha:
        raise ValueError("family-fold binding is invalid")
    families = _require_mapping(manifest.get("families"), "families")
    if set(families) != set(folds):
        raise ValueError("family inventory differs from the bound fold inventory")
    references: list[RelationalPrefixReference] = []
    global_geometry_bindings: list[dict[str, str]] = []
    global_activation_bindings: list[dict[str, str]] = []
    total_rows = 0
    total_occurrences = 0
    for family in sorted(families):
        entry = _require_mapping(families[family], f"family {family}")
        family_fold = _require_string(entry.get("family_fold"), "family fold")
        if folds.get(family) != family_fold or entry.get("finalized") is not True:
            raise ValueError(f"family {family} is not finalized in its bound fold")
        completed = _require_int(entry.get("completed_row_count"), "completed row count", minimum=1)
        if completed != entry.get("expected_row_count"):
            raise ValueError(f"family {family} is incomplete")
        tensor_path = _relative_artifact_path(root, entry.get("tensor_path"), "tensor path")
        _assert_file_binding(
            tensor_path,
            expected_sha256=entry.get("tensor_sha256"),
            expected_bytes=entry.get("tensor_bytes"),
            name=f"family {family} compact tensor shard",
        )
        geometry_path = _relative_artifact_path(
            root, entry.get("geometry_jsonl_path"), "geometry JSONL path"
        )
        _assert_file_binding(
            geometry_path,
            expected_sha256=entry.get("geometry_jsonl_sha256"),
            expected_bytes=entry.get("geometry_jsonl_bytes"),
            name=f"family {family} geometry JSONL",
        )
        compact_keys = _validate_family_shard(
            tensor_path,
            family=family,
            family_fold=family_fold,
            manifest=manifest,
            expected_key_count=_require_int(entry.get("tensor_key_count"), "tensor key count", minimum=1),
        )
        rows = _geometry_lines(geometry_path)
        if len(rows) != completed:
            raise ValueError(f"family {family} geometry row count is invalid")
        family_geometry_bindings: list[dict[str, str]] = []
        family_activation_bindings: list[dict[str, str]] = []
        family_occurrences = 0
        for row in rows:
            if row.get("schema_version") != 1 or row.get("kind") != _GEOMETRY_ROW_KIND:
                raise ValueError(f"family {family} geometry row schema is invalid")
            if row.get("family") != family or row.get("family_fold") != family_fold:
                raise ValueError(f"family {family} geometry identity is invalid")
            realization_rows = _require_list(
                row.get("activation_realizations"), "activation realizations"
            )
            occurrence_count = _require_int(row.get("occurrence_count"), "occurrence count", minimum=1)
            if len(realization_rows) != occurrence_count:
                raise ValueError("activation realization count differs from occurrence count")
            geometry = _require_mapping(row.get("geometry"), "geometry projection")
            if _canonical_sha256(geometry) != row.get("label_free_projection_sha256"):
                raise ValueError("geometry row projection digest is invalid")
            mapped_realizations = [
                _require_mapping(value, "activation realization")
                for value in realization_rows
            ]
            if _row_activation_sha256(mapped_realizations) != row.get(
                "activation_prefix_sha256"
            ):
                raise ValueError("geometry row activation digest is invalid")
            row_references = [
                _prefix_reference(
                    row=row,
                    realization=value,
                    family_entry=entry,
                    capture_contract_sha256=contract_sha,
                    compact_keys=compact_keys,
                )
                for value in mapped_realizations
            ]
            references.extend(row_references)
            family_occurrences += occurrence_count
            conversation_id = _require_string(row.get("conversation_id"), "conversation_id")
            family_geometry_bindings.append(
                {
                    "conversation_id": conversation_id,
                    "sha256": _require_sha256(
                        row.get("label_free_projection_sha256"), "geometry projection SHA-256"
                    ),
                }
            )
            family_activation_bindings.append(
                {
                    "conversation_id": conversation_id,
                    "sha256": _require_sha256(
                        row.get("activation_prefix_sha256"), "row activation SHA-256"
                    ),
                }
            )
        family_geometry_sha = _canonical_sha256(family_geometry_bindings)
        family_activation_sha = _canonical_sha256(family_activation_bindings)
        if family_geometry_sha != entry.get("label_free_projection_sha256"):
            raise ValueError(f"family {family} geometry aggregate binding is invalid")
        if family_activation_sha != entry.get("activation_prefix_sha256"):
            raise ValueError(f"family {family} activation aggregate binding is invalid")
        if family_occurrences != entry.get("occurrence_count"):
            raise ValueError(f"family {family} occurrence count is invalid")
        global_geometry_bindings.append({"family": family, "sha256": family_geometry_sha})
        global_activation_bindings.append({"family": family, "sha256": family_activation_sha})
        total_rows += len(rows)
        total_occurrences += family_occurrences
    geometry_sha = _canonical_sha256(global_geometry_bindings)
    activation_sha = _canonical_sha256(global_activation_bindings)
    if geometry_sha != manifest.get("label_free_projection_sha256"):
        raise ValueError("global geometry aggregate binding is invalid")
    if activation_sha != manifest.get("activation_prefix_sha256"):
        raise ValueError("global activation aggregate binding is invalid")
    if total_rows != manifest.get("row_count") or total_occurrences != manifest.get(
        "occurrence_count"
    ):
        raise ValueError("manifest row or occurrence count is invalid")
    if deduplicate_exact_realizations:
        unique: dict[str, RelationalPrefixReference] = {}
        for reference in references:
            previous = unique.get(reference.canonical_realization_id)
            if previous is None:
                unique[reference.canonical_realization_id] = reference
                continue
            if (
                previous.realization_sha256 != reference.realization_sha256
                or previous.prefix_state_sha256 != reference.prefix_state_sha256
                or previous.prefix_token_stop != reference.prefix_token_stop
                or previous.intervention_history != reference.intervention_history
                or previous.pressure_exposed != reference.pressure_exposed
            ):
                raise ValueError("exact-realization certificate conflicts with prefix identity")
            unique[reference.canonical_realization_id] = replace(
                previous,
                equivalent_occurrence_ids=(
                    *previous.equivalent_occurrence_ids,
                    reference.occurrence_id,
                ),
            )
        references = list(unique.values())
    result = RelationalPrefixIndex(
        artifact_root=str(root.resolve()),
        artifact_manifest_sha256=manifest_sha,
        geometry_projection_sha256=_require_sha256(geometry_sha, "geometry SHA-256"),
        activation_prefix_sha256=_require_sha256(activation_sha, "activation SHA-256"),
        family_folds_sha256=folds_sha,
        deduplicated_exact_realizations=deduplicate_exact_realizations,
        references=tuple(references),
    )
    result.to_dict()
    return result


def _source_prefix_state(source_row: Mapping[str, Any], stop: int) -> tuple[str, str]:
    token_ids = _require_list(source_row.get("token_ids"), "source token IDs")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in token_ids):
        raise ValueError("source token IDs must be integers")
    token_sha = _int32_sha256(token_ids[:stop])
    typed = {
        "prefix_token_ids_sha256": token_sha,
        "prefix_token_count": stop,
        **{
            key: _require_list(source_row.get(key), f"source {key}")[:stop]
            for key in _ANNOTATION_KEYS
        },
    }
    return token_sha, _canonical_sha256(typed)


def _expected_token_tensors(
    source_row: Mapping[str, Any], structured: Mapping[str, Any]
) -> dict[str, list[int]]:
    expected: dict[str, list[int]] = {}
    for key in (
        "token_ids",
        "token_role_ids",
        "token_turn_ids",
        "token_message_ids",
        "token_span_flags",
    ):
        values = _require_list(source_row.get(key), f"source {key}")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError(f"source {key} must contain integers")
        expected[key] = [int(value) for value in values]
    for tensor_key, source_key, mapping_key in (
        ("token_origin_ids", "token_origins", "token_origin_id_mapping"),
        (
            "token_origin_detail_ids",
            "token_origin_details",
            "token_origin_detail_id_mapping",
        ),
    ):
        values = _require_list(source_row.get(source_key), f"source {source_key}")
        mapping = _require_mapping(structured.get(mapping_key), mapping_key)
        try:
            expected[tensor_key] = [int(mapping[value]) for value in values]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"source {source_key} is not exactly mapped") from error
    length = len(expected["token_ids"])
    expected["position_ids"] = list(range(length))
    if any(len(values) != length for values in expected.values()):
        raise ValueError("source token annotations differ in length")
    return expected


def _validate_source_identity(
    reference: RelationalPrefixReference, source_row: Mapping[str, Any]
) -> None:
    if source_row.get("schema_version") != 2 or source_row.get("kind") != (
        "relational_structured_action_row"
    ):
        raise ValueError("frozen source row is not structured-action schema v2")
    if _canonical_sha256(source_row) != reference.source_row_sha256:
        raise ValueError("frozen source-row SHA-256 mismatch")
    for key in ("conversation_id", "family", "scenario_id", "orbit_id", "sample_index"):
        if source_row.get(key) != getattr(reference, key):
            raise ValueError(f"frozen source-row {key} mismatch")
    anchors = _require_list(source_row.get("action_anchors"), "source action anchors")
    matching = [
        anchor
        for anchor in anchors
        if isinstance(anchor, Mapping) and anchor.get("field_event_id") == reference.field_event_id
    ]
    if len(matching) != 1 or matching[0].get("anchor_token_index") != (
        reference.prefix_token_stop - 1
    ):
        raise ValueError("frozen source row does not bind the prefix action anchor")
    token_sha, state_sha = _source_prefix_state(source_row, reference.prefix_token_stop)
    if token_sha != reference.prefix_token_ids_sha256 or state_sha != reference.prefix_state_sha256:
        raise ValueError("frozen source row differs from the typed causal-prefix identity")


def _validate_intervention_mask_identity(
    reference: RelationalPrefixReference, source_row: Mapping[str, Any]
) -> None:
    if reference.intervention_mask_scope != "whole_intervention_bearing_user_message":
        raise ValueError("intervention mask scope is not the conservative whole-message scope")
    payloads = _require_list(source_row.get("intervention_payloads"), "intervention payloads")
    messages = _require_list(source_row.get("messages"), "source messages")
    message_ids = _require_list(source_row.get("token_message_ids"), "token message IDs")
    if len(payloads) != 2:
        raise ValueError("structured source row must contain two intervention payload slots")
    intervention_message_ids: list[int] = []
    for slot_index in range(min(reference.turn_index, 2)):
        payload = _require_list(payloads[slot_index], "intervention payload")
        if payload:
            message_id = 3 + 2 * slot_index
            if message_id >= len(messages):
                raise ValueError("intervention-bearing message is absent from the source row")
            message = _require_mapping(messages[message_id], "intervention-bearing message")
            if message.get("role") != "user":
                raise ValueError("intervention-bearing source message is not a user message")
            intervention_message_ids.append(message_id)
    expected = tuple(
        index
        for index, message_id in enumerate(message_ids[: reference.prefix_token_stop])
        if message_id in intervention_message_ids
    )
    if expected != reference.intervention_token_indices:
        raise ValueError("intervention mask indices differ from the frozen whole-message scope")


def _load_and_validate_full_prefix(
    reference: RelationalPrefixReference,
    fetched_tensor_path: Path,
    source_row: Mapping[str, Any],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, np.ndarray]]:
    _validate_source_identity(reference, source_row)
    if not fetched_tensor_path.is_file() or file_sha256(fetched_tensor_path) != (
        reference.source_tensor_sha256
    ):
        raise ValueError("fetched source tensor object SHA-256 mismatch")
    with safe_open(fetched_tensor_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        if metadata is None:
            raise ValueError("fetched source tensor lacks safetensors metadata")
        required_metadata = {
            "schema_version": "2",
            "kind": "relational_capture_row",
            "conversation_id": reference.conversation_id,
            "family": reference.family,
            "source_row_sha256": reference.source_row_sha256,
            "source_token_ids_sha256": _canonical_sha256(source_row.get("token_ids")),
            "capture_contract_sha256": reference.capture_contract_sha256,
            "scenario_id": reference.scenario_id,
            "orbit_id": reference.orbit_id,
            "layers": ",".join(str(layer) for layer in reference.layers),
            "residual_dtype": "bfloat16",
            "attention_dtype": "bfloat16",
        }
        if any(metadata.get(key) != value for key, value in required_metadata.items()):
            raise ValueError("fetched source tensor metadata identity mismatch")
        try:
            structured = json.loads(metadata["structured_metadata_json"])
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError("fetched tensor structured metadata is invalid") from error
        if not isinstance(structured, dict) or metadata.get("structured_metadata_sha256") != (
            _canonical_sha256(structured)
        ):
            raise ValueError("fetched tensor structured metadata digest mismatch")
        for key in (
            "action_anchors",
            "action_token_spans",
            "assistant_action_turn_records",
            "intervention_program",
        ):
            if structured.get(key) != source_row.get(key):
                raise ValueError(f"fetched tensor structured {key} mismatch")
        expected = _expected_token_tensors(source_row, structured)
        length = len(expected["token_ids"])
        hidden_size = int(metadata.get("hidden_size", "0"))
        n_heads = int(metadata.get("n_attention_heads", "0"))
        if hidden_size < 1 or n_heads < 1:
            raise ValueError("fetched source tensor dimensions are invalid")
        expected_keys = set(_TOKEN_KEYS) | {
            *(f"residual.L{layer}" for layer in reference.layers),
            *(f"attention.L{layer}" for layer in reference.layers),
        }
        if set(handle.keys()) != expected_keys:
            raise ValueError("fetched source tensor key set differs from schema v2")
        token_dtypes = {
            "token_ids": torch.int32,
            "position_ids": torch.int32,
            "token_role_ids": torch.uint8,
            "token_turn_ids": torch.int16,
            "token_message_ids": torch.int16,
            "token_span_flags": torch.int16,
            "token_origin_ids": torch.uint8,
            "token_origin_detail_ids": torch.uint8,
        }
        token_arrays: dict[str, np.ndarray] = {}
        for key in _TOKEN_KEYS:
            tensor = handle.get_tensor(key)
            if tensor.dtype != token_dtypes[key] or tuple(tensor.shape) != (length,):
                raise ValueError(f"fetched source tensor {key} has invalid shape or dtype")
            if tensor.tolist() != expected[key]:
                raise ValueError(f"fetched source tensor {key} differs from frozen source row")
            token_arrays[key] = tensor[: reference.prefix_token_stop].numpy().copy()
        residuals: dict[int, np.ndarray] = {}
        attentions: dict[int, np.ndarray] = {}
        full_edges = length * (length + 1) // 2
        for layer in reference.layers:
            residual = handle.get_tensor(f"residual.L{layer}")
            attention = handle.get_tensor(f"attention.L{layer}")
            if residual.dtype != torch.bfloat16 or tuple(residual.shape) != (length, hidden_size):
                raise ValueError(f"fetched residual L{layer} shape or dtype is invalid")
            if attention.dtype != torch.bfloat16 or tuple(attention.shape) != (
                n_heads,
                full_edges,
            ):
                raise ValueError(f"fetched attention L{layer} shape or dtype is invalid")
            residuals[layer] = (
                residual[: reference.prefix_token_stop].to(torch.float32).numpy().copy()
            )
            attentions[layer] = (
                attention[:, : reference.packed_attention_prefix_stop]
                .to(torch.float32)
                .numpy()
                .copy()
            )
    return residuals, attentions, token_arrays


def _load_and_validate_capture_only_prefix(
    reference: RelationalPrefixReference,
    fetched_tensor_path: Path,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[str, np.ndarray],
    tuple[str, ...],
    tuple[str, ...],
]:
    if not fetched_tensor_path.is_file() or file_sha256(fetched_tensor_path) != (
        reference.source_tensor_sha256
    ):
        raise ValueError("fetched source tensor object SHA-256 mismatch")
    with safe_open(fetched_tensor_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        if metadata is None:
            raise ValueError("fetched source tensor lacks safetensors metadata")
        required_metadata = {
            "schema_version": "2",
            "kind": "relational_capture_row",
            "conversation_id": reference.conversation_id,
            "family": reference.family,
            "source_row_sha256": reference.source_row_sha256,
            "capture_contract_sha256": reference.capture_contract_sha256,
            "scenario_id": reference.scenario_id,
            "orbit_id": reference.orbit_id,
            "layers": ",".join(str(layer) for layer in reference.layers),
            "residual_dtype": "bfloat16",
            "attention_dtype": "bfloat16",
        }
        if any(metadata.get(key) != value for key, value in required_metadata.items()):
            raise ValueError("fetched source tensor metadata identity mismatch")
        structured_json = metadata.get("structured_metadata_json")
        structured_sha = metadata.get("structured_metadata_sha256")
        if (
            not isinstance(structured_json, str)
            or not isinstance(structured_sha, str)
            or hashlib.sha256(structured_json.encode("utf-8")).hexdigest()
            != structured_sha
        ):
            raise ValueError("fetched tensor opaque structured metadata digest mismatch")
        expected_keys = set(_TOKEN_KEYS) | {
            *(f"residual.L{layer}" for layer in reference.layers),
            *(f"attention.L{layer}" for layer in reference.layers),
        }
        if set(handle.keys()) != expected_keys:
            raise ValueError("fetched source tensor key set differs from schema v2")
        prefix_length = reference.prefix_token_stop
        packed_prefix_length = prefix_length * (prefix_length + 1) // 2
        if reference.packed_attention_prefix_stop != packed_prefix_length:
            raise ValueError("causal-prefix attention extent is not lower-triangular")
        token_dtypes = {
            "token_ids": "I32",
            "position_ids": "I32",
            "token_role_ids": "U8",
            "token_turn_ids": "I16",
            "token_message_ids": "I16",
            "token_span_flags": "I16",
            "token_origin_ids": "U8",
            "token_origin_detail_ids": "U8",
        }
        token_arrays: dict[str, np.ndarray] = {}
        prefix_token_values: dict[str, list[int]] = {}
        for key in _TOKEN_KEYS:
            tensor_slice = handle.get_slice(key)
            shape = tensor_slice.get_shape()
            if (
                tensor_slice.get_dtype() != token_dtypes[key]
                or len(shape) != 1
                or shape[0] < prefix_length
            ):
                raise ValueError(f"fetched source tensor {key} has invalid shape or dtype")
            prefix = tensor_slice[:prefix_length]
            values = [int(value) for value in prefix.tolist()]
            prefix_token_values[key] = values
            token_arrays[key] = prefix.numpy().copy()
        if prefix_token_values["position_ids"] != list(range(prefix_length)):
            raise ValueError("fetched source tensor position_ids are not canonical")
        prefix_token_ids = prefix_token_values["token_ids"]
        prefix_token_sha = _int32_sha256(prefix_token_ids)
        if prefix_token_sha != reference.prefix_token_ids_sha256:
            raise ValueError("fetched source tensor differs from the prefix token-ID binding")
        try:
            origins = [
                _FROZEN_TOKEN_ORIGINS[value]
                for value in prefix_token_values["token_origin_ids"]
            ]
            origin_details = [
                _FROZEN_TOKEN_ORIGIN_DETAILS[value]
                for value in prefix_token_values["token_origin_detail_ids"]
            ]
        except KeyError as error:
            raise ValueError(
                "fetched tensor token origins differ from the frozen schema-v2 mapping"
            ) from error
        typed_prefix = {
            "prefix_token_ids_sha256": prefix_token_sha,
            "prefix_token_count": prefix_length,
            "token_role_ids": prefix_token_values["token_role_ids"],
            "token_turn_ids": prefix_token_values["token_turn_ids"],
            "token_message_ids": prefix_token_values["token_message_ids"],
            "token_span_flags": prefix_token_values["token_span_flags"],
            "token_origins": origins,
            "token_origin_details": origin_details,
        }
        if _canonical_sha256(typed_prefix) != reference.prefix_state_sha256:
            raise ValueError("fetched tensor differs from the typed causal-prefix identity")
        hidden_size = int(metadata.get("hidden_size", "0"))
        n_heads = int(metadata.get("n_attention_heads", "0"))
        if hidden_size < 1 or n_heads < 1:
            raise ValueError("fetched source tensor dimensions are invalid")
        residuals: dict[int, np.ndarray] = {}
        attentions: dict[int, np.ndarray] = {}
        for layer in reference.layers:
            residual_slice = handle.get_slice(f"residual.L{layer}")
            attention_slice = handle.get_slice(f"attention.L{layer}")
            residual_shape = residual_slice.get_shape()
            attention_shape = attention_slice.get_shape()
            if (
                residual_slice.get_dtype() != "BF16"
                or len(residual_shape) != 2
                or residual_shape[0] < prefix_length
                or residual_shape[1] != hidden_size
            ):
                raise ValueError(f"fetched residual L{layer} shape or dtype is invalid")
            if (
                attention_slice.get_dtype() != "BF16"
                or len(attention_shape) != 2
                or attention_shape[0] != n_heads
                or attention_shape[1] < packed_prefix_length
            ):
                raise ValueError(f"fetched attention L{layer} shape or dtype is invalid")
            residuals[layer] = (
                residual_slice[:prefix_length, :]
                .to(torch.float32)
                .numpy()
                .copy()
            )
            attentions[layer] = (
                attention_slice[:, :packed_prefix_length]
                .to(torch.float32)
                .numpy()
                .copy()
            )
    return residuals, attentions, token_arrays, tuple(origins), tuple(origin_details)


def load_raw_relational_prefix_capture_only(
    reference: RelationalPrefixReference,
    fetched_tensor_path: Path,
) -> RawRelationalPrefixCapture:
    """Load the exact raw prefix needed for label-free connection analysis.

    The safetensors identity, opaque metadata, and exact causal-prefix tensors
    are checked before arrays are returned. No post-prefix tensor value is
    materialized. The result contains structural origin names but no decoded
    sampled action or outcome value.
    """

    residuals, attentions, tokens, origins, origin_details = (
        _load_and_validate_capture_only_prefix(reference, Path(fetched_tensor_path))
    )
    return RawRelationalPrefixCapture(
        object_id=reference.reference_id,
        residuals=residuals,
        attentions=attentions,
        token_ids=tokens["token_ids"],
        position_ids=tokens["position_ids"],
        token_role_ids=tokens["token_role_ids"],
        token_turn_ids=tokens["token_turn_ids"],
        token_message_ids=tokens["token_message_ids"],
        token_span_flags=tokens["token_span_flags"],
        token_origins=origins,
        token_origin_details=origin_details,
        anchor_index=reference.prefix_token_stop - 1,
    )


def _masked_object_arrays(
    reference: RelationalPrefixReference,
    residuals: Mapping[int, np.ndarray],
    attentions: Mapping[int, np.ndarray],
    token_arrays: Mapping[str, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, np.ndarray]]:
    n_tokens = reference.prefix_token_stop
    mask_indices = np.asarray(reference.intervention_token_indices, dtype=np.int64)
    if mask_indices.size == 0:
        return dict(residuals), dict(attentions), dict(token_arrays)
    if (
        len(np.unique(mask_indices)) != len(mask_indices)
        or np.any(mask_indices < 0)
        or np.any(mask_indices >= n_tokens)
    ):
        raise ValueError("intervention mask indices are duplicated or outside the prefix")
    keep = np.ones(n_tokens, dtype=bool)
    keep[mask_indices] = False
    if int(keep.sum()) < 2:
        raise ValueError("intervention mask would leave fewer than two typed tokens")
    rows, columns = np.tril_indices(n_tokens)
    masked_residuals = {layer: values[keep].copy() for layer, values in residuals.items()}
    masked_attentions: dict[int, np.ndarray] = {}
    for layer, packed in attentions.items():
        dense = np.zeros((packed.shape[0], n_tokens, n_tokens), dtype=np.float32)
        dense[:, rows, columns] = packed
        dense = dense[:, keep][:, :, keep]
        row_sums = dense.sum(axis=-1, keepdims=True, dtype=np.float64)
        if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0):
            raise ValueError("intervention masking leaves an attention row without causal mass")
        masked_attentions[layer] = np.asarray(dense / row_sums, dtype=np.float32)
    return (
        masked_residuals,
        masked_attentions,
        {key: values[keep].copy() for key, values in token_arrays.items()},
    )


def load_typed_relational_prefix(
    reference: RelationalPrefixReference,
    fetched_tensor_path: Path,
    source_row: Mapping[str, Any],
    *,
    mask_intervention_tokens: bool = False,
) -> TypedRelationalObject:
    """Load one exact full causal prefix after file, token, and annotation verification."""
    fetched_tensor_path = Path(fetched_tensor_path)
    if mask_intervention_tokens:
        _validate_intervention_mask_identity(reference, source_row)
    residuals, attentions, tokens = _load_and_validate_full_prefix(
        reference, fetched_tensor_path, source_row
    )
    object_id = reference.reference_id
    if mask_intervention_tokens:
        residuals, attentions, tokens = _masked_object_arrays(
            reference, residuals, attentions, tokens
        )
        object_id += ":intervention-masked"
    positions = tokens["position_ids"].astype(np.float64, copy=False) / max(
        reference.prefix_token_stop - 1, 1
    )
    return TypedRelationalObject(
        object_id=object_id,
        residuals=residuals,
        attentions=attentions,
        role_ids=tokens["token_role_ids"].astype(np.int16, copy=False),
        turn_ids=tokens["token_turn_ids"].astype(np.int16, copy=False),
        span_flags=tokens["token_span_flags"].astype(np.int16, copy=False),
        normalized_positions=positions,
    )


def load_typed_relational_prefix_capture_only(
    reference: RelationalPrefixReference,
    fetched_tensor_path: Path,
    *,
    mask_intervention_tokens: bool = False,
) -> TypedRelationalObject:
    """Load a bound causal prefix without opening the outcome-bearing source row."""

    fetched_tensor_path = Path(fetched_tensor_path)
    residuals, attentions, tokens, _origins, _origin_details = (
        _load_and_validate_capture_only_prefix(reference, fetched_tensor_path)
    )
    object_id = reference.reference_id
    if mask_intervention_tokens:
        if reference.intervention_mask_scope != (
            "whole_intervention_bearing_user_message"
        ):
            raise ValueError("intervention mask scope is not the conservative scope")
        residuals, attentions, tokens = _masked_object_arrays(
            reference, residuals, attentions, tokens
        )
        object_id += ":intervention-masked"
    positions = tokens["position_ids"].astype(np.float64, copy=False) / max(
        reference.prefix_token_stop - 1, 1
    )
    return TypedRelationalObject(
        object_id=object_id,
        residuals=residuals,
        attentions=attentions,
        role_ids=tokens["token_role_ids"].astype(np.int16, copy=False),
        turn_ids=tokens["token_turn_ids"].astype(np.int16, copy=False),
        span_flags=tokens["token_span_flags"].astype(np.int16, copy=False),
        normalized_positions=positions,
    )


__all__ = [
    "CompactTensorLocators",
    "RawRelationalPrefixCapture",
    "RelationalPrefixIndex",
    "RelationalPrefixReference",
    "build_relational_prefix_index",
    "load_raw_relational_prefix_capture_only",
    "load_typed_relational_prefix",
    "load_typed_relational_prefix_capture_only",
]
