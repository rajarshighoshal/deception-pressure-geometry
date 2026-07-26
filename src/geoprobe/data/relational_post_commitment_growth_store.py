"""Immutable, outcome-blind access to sealed post-commitment growth banks."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from safetensors import safe_open

from geoprobe.geometry.relational_post_commitment_growth import BRIDGE_LENGTH, tensor_sha256
from geoprobe.geometry.relational_post_commitment_growth_metric import (
    GrowthMetricInput,
    adapt_growth_edge,
)
from geoprobe.io import file_sha256


_SCHEMA_VERSION = 1
_MANIFEST_KIND = "relational_post_commitment_growth_materialization_manifest"
_ROW_KIND = "relational_post_commitment_growth_materialization_row"
_SHARD_KIND = "relational_post_commitment_growth_family_shard"
_MANIFEST_NAME = "manifest.json"
_METRIC_REALIZATION_DOMAIN = b"geoprobe.post-commitment-growth.metric-realization.v1\x00"
_PHYSICAL_EDGE_ID_DOMAIN = b"geoprobe.post-commitment-growth.physical-edge-id.v1\x00"
_EDGE_PAYLOAD_DOMAIN = b"geoprobe.post-commitment-growth.edge-payload.v1\x00"

_METRIC_TENSOR_KEYS = (
    "residual_bridge_to_base",
    "residual_within_bridge",
    "attention_values",
    "attention_offsets",
    "attention_lengths",
    "attention_query_indices",
    "prefix_position_ids",
    "prefix_normalized_positions",
    "prefix_role_ids",
    "prefix_turn_ids",
    "prefix_message_ids",
    "prefix_span_flags",
    "prefix_origin_ids",
    "prefix_origin_detail_ids",
    "prefix_intervention_mask",
)
_ALL_TENSOR_KEYS = (
    "bridge_token_indices",
    *_METRIC_TENSOR_KEYS[:5],
    "anchor_residuals",
    _METRIC_TENSOR_KEYS[5],
    "prefix_token_ids",
    *_METRIC_TENSOR_KEYS[6:],
)
_ANNOTATION_TENSOR_KEYS = _METRIC_TENSOR_KEYS[6:]
_ANNOTATION_SCHEMA_KEYS = {
    "origin_id_mapping",
    "origin_detail_id_mapping",
    "intervention_mask_scope",
}
_OUTCOME_TERMS = ("outcome", "label", "target", "reward", "success", "correct", "true")
_SEALED_TERMS = ("mapped", "desired", "knowledge")


class RelationalPostCommitmentGrowthStoreError(ValueError):
    """Raised when a sealed growth bank is unsafe or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class GrowthTensorLocators:
    """The immutable physical location of one edge's sealed tensors."""

    family_shard_path: str
    family_shard_sha256: str
    tensor_keys: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class PostCommitmentGrowthReference:
    """Label-free structural provenance and locators for one physical edge."""

    edge_id: str
    edge_sha256: str
    metric_realization_sha256: str
    tensor_locators: GrowthTensorLocators
    metric_tensor_hashes: Mapping[str, str]
    family: str
    family_fold: str
    split: str
    turn_index: int
    scenario_id: str
    orbit_id: str
    intervention_program: str
    intervention_history: tuple[str, ...]
    intervention_token_indices: tuple[int, ...]
    pressure_exposed: bool
    conversation_id: str
    sample_index: int
    status_occurrence_id: str
    caveat_occurrence_id: str
    status_field_event_id: str
    caveat_field_event_id: str
    status_prefix_state_sha256: str
    caveat_prefix_state_sha256: str
    status_activation_realization_id: str | None
    caveat_activation_realization_id: str | None
    status_anchor_index: int
    caveat_anchor_index: int
    status_sampled_token_id: int | None
    source_row_sha256: str
    source_tensor_path: str
    source_tensor_sha256: str
    source_tensor_bytes: int
    capture_contract_sha256: str
    # Defaults preserve synthetic structural references that do not expose anchors.
    anchor_residuals_sha256: str = ""


@dataclass(frozen=True, slots=True)
class RelationalPostCommitmentGrowthIndex:
    """One validated immutable view of a completed materialization bank."""

    artifact_root: Path
    manifest_sha256: str
    input_identity_sha256: str
    family_folds_sha256: str
    references: tuple[PostCommitmentGrowthReference, ...]
    metric_realization_groups: Mapping[str, tuple[PostCommitmentGrowthReference, ...]]
    by_edge_id: Mapping[str, PostCommitmentGrowthReference]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RelationalPostCommitmentGrowthStoreError("record is not canonical JSON") from error


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise RelationalPostCommitmentGrowthStoreError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationalPostCommitmentGrowthStoreError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RelationalPostCommitmentGrowthStoreError(f"{label} must be an integer >= {minimum}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelationalPostCommitmentGrowthStoreError(f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise RelationalPostCommitmentGrowthStoreError(f"{label} contains non-finite JSON constant {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RelationalPostCommitmentGrowthStoreError(f"{label} is not valid UTF-8 JSON") from error
    return _mapping(value, label)


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RelationalPostCommitmentGrowthStoreError(f"{path} is not valid UTF-8 JSONL") from error
    found = False
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        found = True
        try:
            value = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError, json.JSONDecodeError) as error:
            raise RelationalPostCommitmentGrowthStoreError(f"{path} line {number} is not valid JSON") from error
        yield _mapping(value, f"{path} line {number}")
    if not found:
        raise RelationalPostCommitmentGrowthStoreError(f"{path} is empty")


def _safe_child(root: Path, relative: Any, label: str) -> Path:
    text = _string(relative, label)
    if "\\" in text or Path(text).is_absolute():
        raise RelationalPostCommitmentGrowthStoreError(f"{label} must be a safe relative POSIX path")
    candidate = (root / text).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise RelationalPostCommitmentGrowthStoreError(f"{label} escapes artifact root or is absent")
    return candidate


def _reject_outcomes(value: Any, path: str, *, allow_manifest_status: bool = False) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower()
            if name in {"origin_id_mapping", "origin_detail_id_mapping"}:
                continue
            if not (allow_manifest_status and path == "manifest" and name == "status"):
                outcome_name = name in _OUTCOME_TERMS or any(
                    name.startswith(f"{term}_") or name.endswith(f"_{term}")
                    for term in _OUTCOME_TERMS
                )
                if outcome_name or any(term in name for term in _SEALED_TERMS):
                    raise RelationalPostCommitmentGrowthStoreError(f"outcome-bearing field is forbidden at {path}.{key}")
                if "caveat" in name and "sampled" in name:
                    raise RelationalPostCommitmentGrowthStoreError(f"future-caveat sampled field is forbidden at {path}.{key}")
            _reject_outcomes(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_outcomes(child, f"{path}[{index}]")


def _physical_edge_id(identity: Mapping[str, Any]) -> str:
    fields = {
        "conversation_id": _string(identity.get("conversation_id"), "edge conversation_id"),
        "turn_index": _integer(identity.get("turn_index"), "edge turn index"),
        "status_occurrence_id": _string(identity.get("status_occurrence_id"), "status occurrence ID"),
        "caveat_occurrence_id": _string(identity.get("caveat_occurrence_id"), "caveat occurrence ID"),
        "status_field_event_id": _string(identity.get("status_field_event_id"), "status field event ID"),
        "caveat_field_event_id": _string(identity.get("caveat_field_event_id"), "caveat field event ID"),
    }
    return hashlib.sha256(_PHYSICAL_EDGE_ID_DOMAIN + _canonical_bytes(fields)).hexdigest()


def _metric_realization_sha256(
    tensor_hashes: Mapping[str, Any],
    annotation_schema: Mapping[str, Any],
    *,
    status_anchor_index: int,
    caveat_anchor_index: int,
) -> str:
    selected = {name: _sha(tensor_hashes.get(name), f"metric tensor hash {name}") for name in _METRIC_TENSOR_KEYS}
    schema = {
        "origin_id_mapping": _mapping(annotation_schema.get("origin_id_mapping"), "annotation origin mapping"),
        "origin_detail_id_mapping": _mapping(annotation_schema.get("origin_detail_id_mapping"), "annotation origin-detail mapping"),
        "intervention_mask_scope": _string(annotation_schema.get("intervention_mask_scope"), "annotation intervention scope"),
    }
    return hashlib.sha256(
        _METRIC_REALIZATION_DOMAIN
        + _canonical_bytes(
            {
                "tensor_hashes": selected,
                "annotation_schema": schema,
                "adapter": {
                    "status_anchor_index": status_anchor_index,
                    "caveat_anchor_index": caveat_anchor_index,
                    "bridge_length": BRIDGE_LENGTH,
                },
            }
        )
    ).hexdigest()


def _edge_payload_sha256(
    identity: Mapping[str, Any], tensor_hashes: Mapping[str, Any], annotation_schema: Mapping[str, Any]
) -> str:
    schema = {
        "origin_id_mapping": _mapping(annotation_schema.get("origin_id_mapping"), "annotation origin mapping"),
        "origin_detail_id_mapping": _mapping(
            annotation_schema.get("origin_detail_id_mapping"), "annotation origin-detail mapping"
        ),
        "intervention_mask_scope": _string(annotation_schema.get("intervention_mask_scope"), "annotation intervention scope"),
    }
    return hashlib.sha256(
        _EDGE_PAYLOAD_DOMAIN
        + _canonical_bytes({"identity": identity, "tensor_hashes": tensor_hashes, "annotation_schema": schema})
    ).hexdigest()


def _metadata(
    handle: Any,
    *,
    family: str,
    input_identity_sha256: str,
    expected_rows: int | None = None,
    expected_edges: int | None = None,
) -> None:
    metadata = handle.metadata()
    if not isinstance(metadata, Mapping) or metadata != {
        "schema_version": "1",
        "kind": _SHARD_KIND,
        "family": family,
        "row_count": str(metadata.get("row_count", "")),
        "edge_count": str(metadata.get("edge_count", "")),
        "input_identity_sha256": input_identity_sha256,
        "payload_scope": "sealed_geometry",
    }:
        raise RelationalPostCommitmentGrowthStoreError(f"family {family} safetensors metadata is invalid")
    rows = _integer(
        int(metadata["row_count"]) if metadata["row_count"].isdigit() else -1,
        "shard row count",
        minimum=1,
    )
    edges = _integer(
        int(metadata["edge_count"]) if metadata["edge_count"].isdigit() else -1,
        "shard edge count",
        minimum=1,
    )
    if (expected_rows is not None and rows != expected_rows) or (
        expected_edges is not None and edges != expected_edges
    ):
        raise RelationalPostCommitmentGrowthStoreError("safetensors metadata counts disagree with manifest")


def _reference(
    *, family: str, family_fold: str, tensor_path: str, tensor_sha: str, row: Mapping[str, Any], raw_edge: Mapping[str, Any]
) -> PostCommitmentGrowthReference:
    identity = _mapping(raw_edge.get("edge"), "edge identity")
    tensor_keys = _mapping(raw_edge.get("tensor_keys"), "edge tensor keys")
    tensor_hashes = _mapping(raw_edge.get("tensor_hashes"), "edge tensor hashes")
    annotation_schema = _mapping(raw_edge.get("annotation_schema"), "edge annotation schema")
    if set(tensor_keys) != set(_ALL_TENSOR_KEYS) or set(tensor_hashes) != set(_ALL_TENSOR_KEYS):
        raise RelationalPostCommitmentGrowthStoreError("edge tensor key/hash mapping is not exact")
    if set(annotation_schema) != _ANNOTATION_SCHEMA_KEYS:
        raise RelationalPostCommitmentGrowthStoreError("edge annotation schema is not exact")
    locators = MappingProxyType({name: _string(tensor_keys[name], f"tensor key {name}") for name in _ALL_TENSOR_KEYS})
    if len(set(locators.values())) != len(locators):
        raise RelationalPostCommitmentGrowthStoreError("edge tensor locators must be unique")
    for name in _ALL_TENSOR_KEYS:
        _sha(tensor_hashes[name], f"tensor hash {name}")
    edge_id = _sha(identity.get("edge_id"), "edge identifier")
    if _sha(raw_edge.get("edge_sha256"), "edge SHA-256") != _edge_payload_sha256(
        identity, tensor_hashes, annotation_schema
    ):
        raise RelationalPostCommitmentGrowthStoreError(f"edge {edge_id} payload hash is invalid")
    if edge_id != _physical_edge_id(identity):
        raise RelationalPostCommitmentGrowthStoreError(f"edge {edge_id} has invalid physical identity")
    if _string(identity.get("family"), "edge family") != family:
        raise RelationalPostCommitmentGrowthStoreError(f"edge {edge_id} does not bind its family")
    if _string(identity.get("family_fold"), "edge family fold") != family_fold:
        raise RelationalPostCommitmentGrowthStoreError(f"edge {edge_id} does not bind its family fold")
    if _string(identity.get("conversation_id"), "edge conversation") != _string(row.get("conversation_id"), "row conversation"):
        raise RelationalPostCommitmentGrowthStoreError(f"edge {edge_id} does not bind its row conversation")
    if _sha(identity.get("source_row_sha256"), "source row hash") != _sha(row.get("source_row_sha256"), "row source hash"):
        raise RelationalPostCommitmentGrowthStoreError(f"edge {edge_id} does not bind its source row")
    if _sha(identity.get("source_tensor_sha256"), "source tensor hash") != _sha(row.get("source_tensor_sha256"), "row source tensor hash"):
        raise RelationalPostCommitmentGrowthStoreError(f"edge {edge_id} does not bind its source tensor")
    if _integer(identity.get("source_tensor_bytes"), "edge source tensor bytes", minimum=1) != _integer(
        row.get("source_tensor_bytes"), "row source tensor bytes", minimum=1
    ):
        raise RelationalPostCommitmentGrowthStoreError(f"edge {edge_id} does not bind its source tensor bytes")
    status_anchor = _integer(identity.get("status_anchor_index"), "status anchor")
    caveat_anchor = _integer(identity.get("caveat_anchor_index"), "caveat anchor")
    if caveat_anchor - status_anchor != BRIDGE_LENGTH:
        raise RelationalPostCommitmentGrowthStoreError(f"edge {edge_id} violates the six-token bridge")
    history = identity.get("intervention_history")
    if not isinstance(history, list) or any(not isinstance(value, str) or not value for value in history):
        raise RelationalPostCommitmentGrowthStoreError(f"edge {edge_id} has invalid intervention history")
    token_indices = identity.get("intervention_token_indices")
    if (
        not isinstance(token_indices, list)
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in token_indices)
        or token_indices != sorted(set(token_indices))
    ):
        raise RelationalPostCommitmentGrowthStoreError(f"edge {edge_id} has invalid intervention token indices")
    pressure_exposed = identity.get("pressure_exposed")
    if not isinstance(pressure_exposed, bool):
        raise RelationalPostCommitmentGrowthStoreError(f"edge {edge_id} lacks a Boolean pressure indicator")
    status_realization = identity.get("status_activation_realization_id")
    caveat_realization = identity.get("caveat_activation_realization_id")
    for value, label in (
        (status_realization, "status activation realization"),
        (caveat_realization, "caveat activation realization"),
    ):
        if value is not None and (not isinstance(value, str) or not value):
            raise RelationalPostCommitmentGrowthStoreError(f"edge {edge_id} has invalid {label}")
    status_sampled = identity.get("status_sampled_token_id")
    if status_sampled is not None:
        status_sampled = _integer(status_sampled, "status sampled token")
    turn = _integer(identity.get("turn_index"), "turn index")
    if turn not in range(4):
        raise RelationalPostCommitmentGrowthStoreError(f"edge {edge_id} has turn outside 0..3")
    return PostCommitmentGrowthReference(
        edge_id=edge_id,
        edge_sha256=_sha(raw_edge.get("edge_sha256"), "edge SHA-256"),
        metric_realization_sha256=_metric_realization_sha256(
            tensor_hashes,
            annotation_schema,
            status_anchor_index=status_anchor,
            caveat_anchor_index=caveat_anchor,
        ),
        tensor_locators=GrowthTensorLocators(tensor_path, tensor_sha, locators),
        metric_tensor_hashes=MappingProxyType(
            {name: _sha(tensor_hashes[name], f"metric tensor hash {name}") for name in _METRIC_TENSOR_KEYS}
        ),
        family=family,
        family_fold=family_fold,
        split=_string(identity.get("split"), "edge split"),
        turn_index=turn,
        scenario_id=_string(identity.get("scenario_id"), "edge scenario"),
        orbit_id=_string(identity.get("orbit_id"), "edge orbit"),
        intervention_program=_string(identity.get("intervention_program"), "edge intervention"),
        intervention_history=tuple(history),
        intervention_token_indices=tuple(token_indices),
        pressure_exposed=pressure_exposed,
        conversation_id=_string(identity.get("conversation_id"), "edge conversation"),
        sample_index=_integer(identity.get("sample_index"), "edge sample index"),
        status_occurrence_id=_string(identity.get("status_occurrence_id"), "status occurrence ID"),
        caveat_occurrence_id=_string(identity.get("caveat_occurrence_id"), "caveat occurrence ID"),
        status_field_event_id=_string(identity.get("status_field_event_id"), "status field event ID"),
        caveat_field_event_id=_string(identity.get("caveat_field_event_id"), "caveat field event ID"),
        status_prefix_state_sha256=_sha(identity.get("status_prefix_state_sha256"), "status prefix state hash"),
        caveat_prefix_state_sha256=_sha(identity.get("caveat_prefix_state_sha256"), "caveat prefix state hash"),
        status_activation_realization_id=status_realization,
        caveat_activation_realization_id=caveat_realization,
        status_anchor_index=status_anchor,
        caveat_anchor_index=caveat_anchor,
        status_sampled_token_id=status_sampled,
        source_row_sha256=_sha(identity.get("source_row_sha256"), "edge source row hash"),
        source_tensor_path=_string(identity.get("source_tensor_path"), "edge source tensor path"),
        source_tensor_sha256=_sha(identity.get("source_tensor_sha256"), "edge source tensor hash"),
        source_tensor_bytes=_integer(identity.get("source_tensor_bytes"), "edge source tensor bytes", minimum=1),
        capture_contract_sha256=_sha(identity.get("capture_contract_sha256"), "capture contract hash"),
        anchor_residuals_sha256=_sha(tensor_hashes["anchor_residuals"], "anchor residuals hash"),
    )


def build_relational_post_commitment_growth_index(
    root: Path, *, deduplicate_metric_realizations: bool = False
) -> RelationalPostCommitmentGrowthIndex:
    """Validate a successful bank and return only its structural, label-free index."""
    artifact_root = Path(root).resolve()
    manifest_path = artifact_root / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise RelationalPostCommitmentGrowthStoreError("bank manifest.json is missing")
    manifest = _read_json(manifest_path, "bank manifest")
    _reject_outcomes(manifest, "manifest", allow_manifest_status=True)
    if manifest.get("schema_version") != _SCHEMA_VERSION or manifest.get("kind") != _MANIFEST_KIND or manifest.get("status") != "success":
        raise RelationalPostCommitmentGrowthStoreError("bank manifest is not a successful schema-1 growth materialization")
    input_identity = _sha(manifest.get("input_identity_sha256"), "input identity")
    families = _mapping(manifest.get("families"), "bank families")
    if not families:
        raise RelationalPostCommitmentGrowthStoreError("bank has no families")
    references: list[PostCommitmentGrowthReference] = []
    by_edge: dict[str, PostCommitmentGrowthReference] = {}
    family_folds: list[dict[str, str]] = []
    total_rows = total_edges = 0
    for family in sorted(families):
        family_name = _string(family, "family name")
        entry = _mapping(families[family], f"family {family_name}")
        expected_rows = _integer(entry.get("expected_row_count"), "expected row count", minimum=1)
        if entry.get("finalized") is not True or _integer(entry.get("completed_row_count"), "completed row count") != expected_rows:
            raise RelationalPostCommitmentGrowthStoreError(f"family {family_name} is not finalized")
        if _integer(entry.get("edge_count"), "family edge count", minimum=1) != expected_rows * 4:
            raise RelationalPostCommitmentGrowthStoreError(f"family {family_name} has invalid edge count")
        tensor_path = _safe_child(artifact_root, entry.get("tensor_path"), f"family {family_name} tensor path")
        geometry_path = _safe_child(artifact_root, entry.get("geometry_jsonl_path"), f"family {family_name} geometry path")
        tensor_sha = _sha(entry.get("tensor_sha256"), "family tensor hash")
        geometry_sha = _sha(entry.get("geometry_jsonl_sha256"), "family geometry hash")
        if _integer(entry.get("tensor_bytes"), "family tensor bytes", minimum=1) != tensor_path.stat().st_size or tensor_sha != file_sha256(tensor_path):
            raise RelationalPostCommitmentGrowthStoreError(f"family {family_name} tensor file integrity mismatch")
        if _integer(entry.get("geometry_jsonl_bytes"), "family geometry bytes", minimum=1) != geometry_path.stat().st_size or geometry_sha != file_sha256(geometry_path):
            raise RelationalPostCommitmentGrowthStoreError(f"family {family_name} geometry file integrity mismatch")
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            _metadata(
                handle,
                family=family_name,
                input_identity_sha256=input_identity,
                expected_rows=expected_rows,
                expected_edges=expected_rows * 4,
            )
            shard_keys = set(handle.keys())
        if len(shard_keys) != _integer(entry.get("tensor_key_count"), "family tensor key count", minimum=1):
            raise RelationalPostCommitmentGrowthStoreError(f"family {family_name} tensor key count mismatch")
        rows = list(_iter_jsonl(geometry_path))
        if len(rows) != expected_rows:
            raise RelationalPostCommitmentGrowthStoreError(f"family {family_name} geometry row count mismatch")
        folds: set[str] = set()
        family_edges = 0
        row_indices: set[int] = set()
        for row in rows:
            _reject_outcomes(row, f"family[{family_name}]")
            if row.get("schema_version") != _SCHEMA_VERSION or row.get("kind") != _ROW_KIND or row.get("family") != family_name:
                raise RelationalPostCommitmentGrowthStoreError(f"family {family_name} has invalid geometry row schema")
            row_index = _integer(row.get("row_index"), "row index")
            if row_index in row_indices:
                raise RelationalPostCommitmentGrowthStoreError(f"family {family_name} has duplicate row index")
            row_indices.add(row_index)
            family_fold = _string(row.get("family_fold"), "row family fold")
            folds.add(family_fold)
            if _integer(row.get("edge_count"), "row edge count") != 4:
                raise RelationalPostCommitmentGrowthStoreError(f"family {family_name} row does not contain four edges")
            row_keys = row.get("tensor_keys")
            if not isinstance(row_keys, list) or len(row_keys) != len(set(row_keys)) or not set(row_keys).issubset(shard_keys):
                raise RelationalPostCommitmentGrowthStoreError(f"family {family_name} has unresolved row tensor keys")
            raw_edges = row.get("edges")
            if not isinstance(raw_edges, list) or len(raw_edges) != 4:
                raise RelationalPostCommitmentGrowthStoreError(f"family {family_name} row has invalid edge list")
            turns: set[int] = set()
            for raw_edge in raw_edges:
                edge = _mapping(raw_edge, "growth edge")
                reference = _reference(
                    family=family_name,
                    family_fold=family_fold,
                    tensor_path=str(tensor_path.relative_to(artifact_root)),
                    tensor_sha=tensor_sha,
                    row=row,
                    raw_edge=edge,
                )
                if reference.edge_id in by_edge:
                    raise RelationalPostCommitmentGrowthStoreError(f"duplicate physical edge ID {reference.edge_id}")
                if not set(reference.tensor_locators.tensor_keys.values()).issubset(shard_keys | set()):
                    raise RelationalPostCommitmentGrowthStoreError(f"edge {reference.edge_id} has unresolved tensor locator")
                by_edge[reference.edge_id] = reference
                references.append(reference)
                turns.add(reference.turn_index)
                family_edges += 1
            if turns != {0, 1, 2, 3}:
                raise RelationalPostCommitmentGrowthStoreError(f"family {family_name} row does not cover turns 0..3")
        if len(folds) != 1:
            raise RelationalPostCommitmentGrowthStoreError(f"family {family_name} has more than one family_fold")
        family_folds.append({"family": family_name, "family_fold": next(iter(folds))})
        if family_edges != expected_rows * 4:
            raise RelationalPostCommitmentGrowthStoreError(f"family {family_name} edge total is inconsistent")
        total_rows += expected_rows
        total_edges += family_edges
    if _integer(manifest.get("row_count"), "manifest row count") != total_rows or _integer(manifest.get("edge_count"), "manifest edge count") != total_edges:
        raise RelationalPostCommitmentGrowthStoreError("manifest totals do not match finalized families")
    ordered = tuple(sorted(references, key=lambda reference: reference.edge_id))
    groups: dict[str, list[PostCommitmentGrowthReference]] = defaultdict(list)
    for reference in ordered:
        groups[reference.metric_realization_sha256].append(reference)
    exact_groups = MappingProxyType({key: tuple(value) for key, value in sorted(groups.items())})
    selected = (
        tuple(group[0] for _, group in sorted(exact_groups.items()))
        if deduplicate_metric_realizations
        else ordered
    )
    return RelationalPostCommitmentGrowthIndex(
        artifact_root=artifact_root,
        manifest_sha256=file_sha256(manifest_path),
        input_identity_sha256=input_identity,
        family_folds_sha256=hashlib.sha256(_canonical_bytes(sorted(family_folds, key=lambda item: item["family"]))).hexdigest(),
        references=selected,
        metric_realization_groups=exact_groups,
        by_edge_id=MappingProxyType(dict(by_edge)),
    )


def _load_from_handle(handle: Any, reference: PostCommitmentGrowthReference) -> GrowthMetricInput:
    keys = reference.tensor_locators.tensor_keys
    missing = set(_METRIC_TENSOR_KEYS).difference(keys)
    if missing:
        raise RelationalPostCommitmentGrowthStoreError(f"edge {reference.edge_id} lacks metric locators")
    tensors: dict[str, torch.Tensor] = {}
    for name in _METRIC_TENSOR_KEYS:
        key = keys[name]
        if key not in handle.keys():
            raise RelationalPostCommitmentGrowthStoreError(f"edge {reference.edge_id} tensor {name} is absent")
        tensor = handle.get_tensor(key)
        tensors[name] = tensor
    expected_hashes = _metric_hashes(reference)
    for name, tensor in tensors.items():
        if tensor_sha256(name, tensor) != expected_hashes[name]:
            raise RelationalPostCommitmentGrowthStoreError(f"edge {reference.edge_id} metric tensor {name} hash mismatch")
    base_stop = reference.status_anchor_index + 1
    annotations = torch.column_stack([tensors[name][:base_stop] for name in _ANNOTATION_TENSOR_KEYS])
    bridge_annotations = torch.column_stack([tensors[name][base_stop : base_stop + BRIDGE_LENGTH] for name in _ANNOTATION_TENSOR_KEYS])
    return adapt_growth_edge(
        residual_bridge_to_base=tensors["residual_bridge_to_base"],
        residual_within_bridge=tensors["residual_within_bridge"],
        attention_values=tensors["attention_values"],
        attention_offsets=tensors["attention_offsets"],
        attention_lengths=tensors["attention_lengths"],
        attention_query_indices=tensors["attention_query_indices"],
        base_annotations=annotations,
        bridge_annotations=bridge_annotations,
    )


def _metric_hashes(reference: PostCommitmentGrowthReference) -> Mapping[str, str]:
    if set(reference.metric_tensor_hashes) != set(_METRIC_TENSOR_KEYS):
        raise RelationalPostCommitmentGrowthStoreError("reference lacks exact metric tensor hash bindings")
    return reference.metric_tensor_hashes


def _anchor_residuals_hash(reference: PostCommitmentGrowthReference) -> str:
    try:
        return _sha(reference.anchor_residuals_sha256, "anchor residuals hash binding")
    except RelationalPostCommitmentGrowthStoreError as error:
        raise RelationalPostCommitmentGrowthStoreError(
            "reference lacks an anchor residuals hash binding"
        ) from error


def _load_anchor_residuals_from_handle(handle: Any, reference: PostCommitmentGrowthReference) -> torch.Tensor:
    key = reference.tensor_locators.tensor_keys.get("anchor_residuals")
    if not isinstance(key, str) or not key:
        raise RelationalPostCommitmentGrowthStoreError(f"edge {reference.edge_id} lacks an anchor residuals locator")
    if key not in handle.keys():
        raise RelationalPostCommitmentGrowthStoreError(f"edge {reference.edge_id} anchor residuals tensor is absent")
    tensor = handle.get_tensor(key)
    if not isinstance(tensor, torch.Tensor):
        raise RelationalPostCommitmentGrowthStoreError(f"edge {reference.edge_id} anchor residuals tensor is invalid")
    if tensor_sha256("anchor_residuals", tensor) != _anchor_residuals_hash(reference):
        raise RelationalPostCommitmentGrowthStoreError(f"edge {reference.edge_id} anchor residuals hash mismatch")
    if tensor.ndim != 3 or tuple(tensor.shape[:2]) != (4, 2) or tensor.shape[2] <= 0:
        raise RelationalPostCommitmentGrowthStoreError(
            f"edge {reference.edge_id} anchor residuals must have shape [4, 2, hidden] with positive hidden size"
        )
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
        raise RelationalPostCommitmentGrowthStoreError(f"edge {reference.edge_id} anchor residuals must be finite floating values")
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone()


def load_growth_metric_input(
    index: RelationalPostCommitmentGrowthIndex, reference: PostCommitmentGrowthReference
) -> GrowthMetricInput:
    """Load precisely one label-free metric payload; tokens and anchors stay sealed."""
    if index.by_edge_id.get(reference.edge_id) is not reference:
        raise RelationalPostCommitmentGrowthStoreError("reference does not belong to this index")
    path = _safe_child(index.artifact_root, reference.tensor_locators.family_shard_path, "family shard path")
    if file_sha256(path) != reference.tensor_locators.family_shard_sha256:
        raise RelationalPostCommitmentGrowthStoreError("family shard changed after indexing")
    with safe_open(path, framework="pt", device="cpu") as handle:
        _metadata(handle, family=reference.family, input_identity_sha256=index.input_identity_sha256)
        return _load_from_handle(handle, reference)


def load_growth_anchor_residuals(
    index: RelationalPostCommitmentGrowthIndex, reference: PostCommitmentGrowthReference
) -> torch.Tensor:
    """Load one verified sealed anchor tensor as a detached CPU float32 tensor.

    The returned tensor has exact shape ``[4, 2, hidden]``, where ``hidden`` is
    positive, and is independent of the safetensors-backed storage.
    """
    if index.by_edge_id.get(reference.edge_id) is not reference:
        raise RelationalPostCommitmentGrowthStoreError("reference does not belong to this index")
    path = _safe_child(index.artifact_root, reference.tensor_locators.family_shard_path, "family shard path")
    if file_sha256(path) != reference.tensor_locators.family_shard_sha256:
        raise RelationalPostCommitmentGrowthStoreError("family shard changed after indexing")
    with safe_open(path, framework="pt", device="cpu") as handle:
        _metadata(handle, family=reference.family, input_identity_sha256=index.input_identity_sha256)
        return _load_anchor_residuals_from_handle(handle, reference)


def iter_growth_anchor_residuals(
    index: RelationalPostCommitmentGrowthIndex,
    references: Sequence[PostCommitmentGrowthReference] | None = None,
) -> Iterator[tuple[PostCommitmentGrowthReference, torch.Tensor]]:
    """Yield verified sealed anchors, opening and hashing each family shard once."""
    chosen = index.references if references is None else tuple(references)
    grouped: dict[str, list[PostCommitmentGrowthReference]] = defaultdict(list)
    for reference in chosen:
        if index.by_edge_id.get(reference.edge_id) is not reference:
            raise RelationalPostCommitmentGrowthStoreError("reference does not belong to this index")
        grouped[reference.tensor_locators.family_shard_path].append(reference)
    for relative_path in sorted(grouped):
        batch = sorted(grouped[relative_path], key=lambda reference: reference.edge_id)
        shard_sha256 = batch[0].tensor_locators.family_shard_sha256
        if any(reference.tensor_locators.family_shard_sha256 != shard_sha256 for reference in batch):
            raise RelationalPostCommitmentGrowthStoreError("family shard references disagree on the shard hash")
        path = _safe_child(index.artifact_root, relative_path, "family shard path")
        if file_sha256(path) != shard_sha256:
            raise RelationalPostCommitmentGrowthStoreError("family shard changed after indexing")
        with safe_open(path, framework="pt", device="cpu") as handle:
            _metadata(handle, family=batch[0].family, input_identity_sha256=index.input_identity_sha256)
            for reference in batch:
                if reference.family != batch[0].family:
                    raise RelationalPostCommitmentGrowthStoreError("family shard references disagree on family")
                yield reference, _load_anchor_residuals_from_handle(handle, reference)


def iter_growth_metric_inputs(
    index: RelationalPostCommitmentGrowthIndex,
    references: Sequence[PostCommitmentGrowthReference] | None = None,
) -> Iterator[tuple[PostCommitmentGrowthReference, GrowthMetricInput]]:
    """Yield inputs grouped by family shard, reusing one safe_open handle per family."""
    chosen = index.references if references is None else tuple(references)
    grouped: dict[str, list[PostCommitmentGrowthReference]] = defaultdict(list)
    for reference in chosen:
        if index.by_edge_id.get(reference.edge_id) is not reference:
            raise RelationalPostCommitmentGrowthStoreError("reference does not belong to this index")
        grouped[reference.tensor_locators.family_shard_path].append(reference)
    for relative_path in sorted(grouped):
        batch = grouped[relative_path]
        path = _safe_child(index.artifact_root, relative_path, "family shard path")
        if file_sha256(path) != batch[0].tensor_locators.family_shard_sha256:
            raise RelationalPostCommitmentGrowthStoreError("family shard changed after indexing")
        with safe_open(path, framework="pt", device="cpu") as handle:
            _metadata(handle, family=batch[0].family, input_identity_sha256=index.input_identity_sha256)
            for reference in batch:
                yield reference, _load_from_handle(handle, reference)


__all__ = [
    "GrowthTensorLocators",
    "PostCommitmentGrowthReference",
    "RelationalPostCommitmentGrowthIndex",
    "RelationalPostCommitmentGrowthStoreError",
    "build_relational_post_commitment_growth_index",
    "iter_growth_anchor_residuals",
    "iter_growth_metric_inputs",
    "load_growth_anchor_residuals",
    "load_growth_metric_input",
]
