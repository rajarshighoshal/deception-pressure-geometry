"""Outcome-blind extraction of schema-v2 post-commitment growth edges.

The capture stores causal attention as packed lower-triangular rows.  This
module reads a single authoritative capture tensor at a time and emits only
the relations newly introduced between a status anchor and its caveat anchor.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

from safetensors import safe_open
import torch

from geoprobe.io import file_sha256


LAYERS = (12, 16, 19, 20)
BRIDGE_LENGTH = 6
_TYPED_TOKEN_KEYS = (
    "token_ids",
    "position_ids",
    "token_role_ids",
    "token_turn_ids",
    "token_message_ids",
    "token_span_flags",
    "token_origin_ids",
    "token_origin_detail_ids",
)
_TENSOR_HASH_DOMAIN = b"geoprobe.post-commitment-growth.tensor.v1\x00"
_PHYSICAL_EDGE_ID_HASH_DOMAIN = b"geoprobe.post-commitment-growth.physical-edge-id.v1\x00"
_EDGE_PAYLOAD_HASH_DOMAIN = b"geoprobe.post-commitment-growth.edge-payload.v1\x00"
_ROW_HASH_DOMAIN = b"geoprobe.post-commitment-growth.row.v1\x00"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _int32_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}i", *values)).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def tensor_sha256(name: str, tensor: torch.Tensor) -> str:
    """Return a domain-separated, dtype- and shape-bound tensor digest."""
    if not name:
        raise ValueError("tensor hash name must be non-empty")
    hasher = hashlib.sha256(_TENSOR_HASH_DOMAIN)
    encoded_name = name.encode("utf-8")
    encoded_dtype = str(tensor.dtype).removeprefix("torch.").encode("ascii")
    hasher.update(struct.pack("<I", len(encoded_name)))
    hasher.update(encoded_name)
    hasher.update(struct.pack("<I", len(encoded_dtype)))
    hasher.update(encoded_dtype)
    hasher.update(struct.pack("<I", tensor.ndim))
    for dimension in tensor.shape:
        hasher.update(struct.pack("<Q", int(dimension)))
    payload = _tensor_bytes(tensor)
    hasher.update(struct.pack("<Q", len(payload)))
    hasher.update(payload)
    return hasher.hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
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
    digest = _require_string(value, name)
    if len(digest) != 64 or digest.lower() != digest:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    try:
        bytes.fromhex(digest)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256") from error
    return digest


def _read_sidecar(value: Mapping[str, Any] | Path | str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    path = Path(value)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read capture sidecar: {path}") from error
    return _require_mapping(parsed, "capture sidecar")


@dataclass(frozen=True, slots=True)
class PrefixTokenAnnotations:
    """Typed, prefix-only token annotations for one edge endpoint."""

    token_ids: torch.Tensor
    position_ids: torch.Tensor
    normalized_positions: torch.Tensor
    role_ids: torch.Tensor
    turn_ids: torch.Tensor
    message_ids: torch.Tensor
    span_flags: torch.Tensor
    origin_ids: torch.Tensor
    origin_detail_ids: torch.Tensor
    origin_id_mapping: Mapping[str, int]
    origin_detail_id_mapping: Mapping[str, int]
    intervention_mask: torch.Tensor
    intervention_mask_scope: str

    def validate(self, *, prefix_stop: int) -> None:
        values = (
            self.token_ids,
            self.position_ids,
            self.normalized_positions,
            self.role_ids,
            self.turn_ids,
            self.message_ids,
            self.span_flags,
            self.origin_ids,
            self.origin_detail_ids,
            self.intervention_mask,
        )
        if any(tensor.ndim != 1 or tensor.shape[0] != prefix_stop for tensor in values):
            raise ValueError("prefix token annotations have inconsistent prefix length")
        if self.normalized_positions.dtype != torch.float32:
            raise ValueError("normalized positions must be float32")
        if self.intervention_mask.dtype != torch.bool:
            raise ValueError("intervention mask must be bool")
        if not self.origin_id_mapping or not self.origin_detail_id_mapping:
            raise ValueError("origin annotation mappings must be non-empty")
        if not set(self.origin_ids.tolist()).issubset(set(self.origin_id_mapping.values())):
            raise ValueError("prefix origin IDs are outside the declared mapping")
        if not set(self.origin_detail_ids.tolist()).issubset(
            set(self.origin_detail_id_mapping.values())
        ):
            raise ValueError("prefix origin detail IDs are outside the declared mapping")
        if not self.intervention_mask_scope:
            raise ValueError("intervention mask scope must be non-empty")


@dataclass(frozen=True, slots=True)
class GrowthEdgeIdentity:
    """Label-free physical identity and source binding for one growth edge."""

    edge_id: str
    conversation_id: str
    family: str
    family_fold: str
    split: str
    scenario_id: str
    orbit_id: str
    sample_index: int
    intervention_program: str
    turn_index: int
    status_occurrence_id: str
    caveat_occurrence_id: str
    status_field_event_id: str
    caveat_field_event_id: str
    status_activation_realization_id: str | None
    caveat_activation_realization_id: str | None
    intervention_history: tuple[str, ...]
    pressure_exposed: bool
    intervention_token_indices: tuple[int, ...]
    intervention_mask_scope: str
    status_anchor_index: int
    caveat_anchor_index: int
    status_prefix_token_ids_sha256: str
    caveat_prefix_token_ids_sha256: str
    status_prefix_state_sha256: str
    caveat_prefix_state_sha256: str
    status_sampled_token_id: int
    source_row_sha256: str
    source_tensor_path: str
    source_tensor_sha256: str
    source_tensor_bytes: int
    capture_contract_sha256: str

    def hash_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "conversation_id": self.conversation_id,
            "family": self.family,
            "family_fold": self.family_fold,
            "split": self.split,
            "scenario_id": self.scenario_id,
            "orbit_id": self.orbit_id,
            "sample_index": self.sample_index,
            "intervention_program": self.intervention_program,
            "turn_index": self.turn_index,
            "status_occurrence_id": self.status_occurrence_id,
            "caveat_occurrence_id": self.caveat_occurrence_id,
            "status_field_event_id": self.status_field_event_id,
            "caveat_field_event_id": self.caveat_field_event_id,
            "status_activation_realization_id": self.status_activation_realization_id,
            "caveat_activation_realization_id": self.caveat_activation_realization_id,
            "intervention_history": list(self.intervention_history),
            "pressure_exposed": self.pressure_exposed,
            "intervention_token_indices": list(self.intervention_token_indices),
            "intervention_mask_scope": self.intervention_mask_scope,
            "status_anchor_index": self.status_anchor_index,
            "caveat_anchor_index": self.caveat_anchor_index,
            "status_prefix_token_ids_sha256": self.status_prefix_token_ids_sha256,
            "caveat_prefix_token_ids_sha256": self.caveat_prefix_token_ids_sha256,
            "status_prefix_state_sha256": self.status_prefix_state_sha256,
            "caveat_prefix_state_sha256": self.caveat_prefix_state_sha256,
            "status_sampled_token_id": self.status_sampled_token_id,
            "source_row_sha256": self.source_row_sha256,
            "source_tensor_path": self.source_tensor_path,
            "source_tensor_sha256": self.source_tensor_sha256,
            "source_tensor_bytes": self.source_tensor_bytes,
            "capture_contract_sha256": self.capture_contract_sha256,
        }


@dataclass(frozen=True, slots=True)
class PostCommitmentGrowthEdge:
    """One exact status-to-caveat typed relational growth edge."""

    identity: GrowthEdgeIdentity
    bridge_token_indices: torch.Tensor
    attention_query_indices: torch.Tensor
    residual_bridge_to_base: torch.Tensor
    residual_within_bridge: torch.Tensor
    anchor_residuals: torch.Tensor
    attention_values: torch.Tensor
    attention_offsets: torch.Tensor
    attention_lengths: torch.Tensor
    prefix_annotations: PrefixTokenAnnotations
    tensor_hashes: Mapping[str, str]
    edge_sha256: str

    def validate(self) -> None:
        s = self.identity.status_anchor_index
        c = self.identity.caveat_anchor_index
        if c - s != BRIDGE_LENGTH:
            raise ValueError("growth edge must have exact c-s=6 bridge")
        if self.bridge_token_indices.dtype != torch.int32 or self.bridge_token_indices.tolist() != list(range(s + 1, c + 1)):
            raise ValueError("growth edge bridge token indices are invalid")
        if self.attention_query_indices.dtype != torch.int32 or self.attention_query_indices.tolist() != list(range(s, c + 1)):
            raise ValueError("growth edge attention query indices are invalid")
        if self.residual_bridge_to_base.shape != (len(LAYERS), BRIDGE_LENGTH, s + 1):
            raise ValueError("bridge-to-base residual relations have invalid shape")
        if self.residual_within_bridge.shape != (len(LAYERS), BRIDGE_LENGTH, BRIDGE_LENGTH):
            raise ValueError("within-bridge residual relations have invalid shape")
        if self.anchor_residuals.ndim != 3 or self.anchor_residuals.shape[:2] != (len(LAYERS), 2):
            raise ValueError("anchor residual retrieval payload has invalid shape")
        if any(tensor.dtype != torch.float32 for tensor in (self.residual_bridge_to_base, self.residual_within_bridge)):
            raise ValueError("residual relation payloads must be float32")
        if not torch.equal(self.residual_within_bridge, self.residual_within_bridge.transpose(-1, -2)):
            raise ValueError("within-bridge residual relations must be symmetric")
        if self.attention_values.dtype != torch.bfloat16:
            raise ValueError("packed attention values must remain bfloat16")
        expected = (len(LAYERS), self.attention_offsets.shape[1], BRIDGE_LENGTH + 1)
        if self.attention_offsets.shape != expected or self.attention_lengths.shape != expected:
            raise ValueError("packed attention offset/length shape is invalid")
        if self.attention_offsets.dtype != torch.int64 or self.attention_lengths.dtype != torch.int32:
            raise ValueError("packed attention offsets/lengths have invalid dtypes")
        offsets = self.attention_offsets.reshape(-1).tolist()
        lengths = self.attention_lengths.reshape(-1).tolist()
        if any(offset < 0 or length < 1 or offset + length > self.attention_values.numel() for offset, length in zip(offsets, lengths, strict=True)):
            raise ValueError("packed attention rows are outside the payload")
        cursor = 0
        for layer_index in range(len(LAYERS)):
            for query_offset, query_index in enumerate(self.attention_query_indices.tolist()):
                for head_index in range(self.attention_offsets.shape[1]):
                    if (
                        int(self.attention_offsets[layer_index, head_index, query_offset]) != cursor
                        or int(self.attention_lengths[layer_index, head_index, query_offset])
                        != query_index + 1
                    ):
                        raise ValueError("packed attention rows are not complete causal rows")
                    cursor += query_index + 1
        if cursor != self.attention_values.numel():
            raise ValueError("packed attention payload has trailing values")
        if len(set(self.tensor_hashes)) != len(self.tensor_hashes) or not self.edge_sha256:
            raise ValueError("growth edge hashes are invalid")
        self.prefix_annotations.validate(prefix_stop=c + 1)
        tensors = {
            "bridge_token_indices": self.bridge_token_indices,
            "attention_query_indices": self.attention_query_indices,
            "residual_bridge_to_base": self.residual_bridge_to_base,
            "residual_within_bridge": self.residual_within_bridge,
            "anchor_residuals": self.anchor_residuals,
            "attention_values": self.attention_values,
            "attention_offsets": self.attention_offsets,
            "attention_lengths": self.attention_lengths,
            "prefix_token_ids": self.prefix_annotations.token_ids,
            "prefix_position_ids": self.prefix_annotations.position_ids,
            "prefix_normalized_positions": self.prefix_annotations.normalized_positions,
            "prefix_role_ids": self.prefix_annotations.role_ids,
            "prefix_turn_ids": self.prefix_annotations.turn_ids,
            "prefix_message_ids": self.prefix_annotations.message_ids,
            "prefix_span_flags": self.prefix_annotations.span_flags,
            "prefix_origin_ids": self.prefix_annotations.origin_ids,
            "prefix_origin_detail_ids": self.prefix_annotations.origin_detail_ids,
            "prefix_intervention_mask": self.prefix_annotations.intervention_mask,
        }
        expected_hashes = {name: tensor_sha256(name, tensor) for name, tensor in tensors.items()}
        if dict(self.tensor_hashes) != expected_hashes:
            raise ValueError("growth edge tensor hashes do not bind its payload")
        if self.identity.edge_id != _physical_edge_id(self.identity):
            raise ValueError("growth edge physical ID is invalid")
        expected_edge = _edge_payload_sha256(self.identity, expected_hashes, self.prefix_annotations)
        if self.edge_sha256 != expected_edge:
            raise ValueError("growth edge payload hash is invalid")


@dataclass(frozen=True, slots=True)
class PostCommitmentGrowthRow:
    """The four same-turn physical edges emitted from one schema-v2 row."""

    conversation_id: str
    source_tensor_sha256: str
    source_tensor_bytes: int
    edges: tuple[PostCommitmentGrowthEdge, ...]
    row_sha256: str

    def validate(self) -> None:
        if len(self.edges) != 4:
            raise ValueError("schema-v2 post-commitment row must contain four edges")
        if len({edge.identity.turn_index for edge in self.edges}) != 4:
            raise ValueError("growth edges must cover exactly one pair per turn")
        if any(edge.identity.conversation_id != self.conversation_id for edge in self.edges):
            raise ValueError("growth row has inconsistent conversation identity")
        if any(edge.identity.source_tensor_sha256 != self.source_tensor_sha256 for edge in self.edges):
            raise ValueError("growth row has inconsistent tensor identity")
        for edge in self.edges:
            edge.validate()
        payload = {
            "conversation_id": self.conversation_id,
            "source_tensor_sha256": self.source_tensor_sha256,
            "edges": [{"edge_id": edge.identity.edge_id, "edge_sha256": edge.edge_sha256} for edge in self.edges],
        }
        if self.row_sha256 != hashlib.sha256(_ROW_HASH_DOMAIN + _canonical_bytes(payload)).hexdigest():
            raise ValueError("growth row hash is invalid")


def _validate_sidecar(sidecar: Mapping[str, Any], tensor_path: Path) -> tuple[Mapping[str, Any], tuple[int, ...], int, int]:
    if sidecar.get("schema_version") != 2 or sidecar.get("kind") != "relational_capture_row_record":
        raise ValueError("capture sidecar must be a schema-v2 relational capture row record")
    if sidecar.get("residual_dtype") != "bfloat16" or sidecar.get("attention_dtype") != "bfloat16":
        raise ValueError("post-commitment extraction requires BF16 residuals and attention")
    layers = tuple(sidecar.get("layers", ()))
    if layers != LAYERS:
        raise ValueError("post-commitment extraction requires layers 12, 16, 19, and 20")
    hidden_size = _require_int(sidecar.get("hidden_size"), "hidden_size", minimum=1)
    heads = _require_int(sidecar.get("n_attention_heads"), "n_attention_heads", minimum=1)
    expected_bytes = _require_int(sidecar.get("tensor_bytes"), "tensor_bytes", minimum=1)
    expected_sha = _require_sha256(sidecar.get("tensor_sha256"), "tensor_sha256")
    if not tensor_path.is_file() or tensor_path.stat().st_size != expected_bytes:
        raise ValueError("authoritative tensor byte-count binding is invalid")
    if file_sha256(tensor_path) != expected_sha:
        raise ValueError("authoritative tensor SHA-256 binding is invalid")
    structured = _require_mapping(sidecar.get("structured_metadata"), "structured_metadata")
    if sidecar.get("structured_metadata_sha256") != _canonical_sha256(structured):
        raise ValueError("structured metadata SHA-256 binding is invalid")
    return structured, layers, hidden_size, heads


def _anchor_pairs(structured: Mapping[str, Any], token_length: int) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    anchors = structured.get("action_anchors")
    spans = structured.get("action_token_spans")
    if not isinstance(anchors, list) or not isinstance(spans, list):
        raise ValueError("structured metadata must contain action anchors and spans")
    span_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for raw in spans:
        span = _require_mapping(raw, "action token span")
        key = (_require_int(span.get("turn_index"), "span turn_index"), _require_string(span.get("field_name"), "span field_name"))
        if key in span_by_key:
            raise ValueError("action token spans must be unique by turn and field")
        span_by_key[key] = span
    paired: dict[int, dict[str, Mapping[str, Any]]] = {}
    for raw in anchors:
        anchor = _require_mapping(raw, "action anchor")
        turn = _require_int(anchor.get("turn_index"), "anchor turn_index")
        field = _require_string(anchor.get("field_name"), "anchor field_name")
        if field not in {"status", "caveat"}:
            raise ValueError("growth extractor accepts only status and caveat anchors")
        index = _require_int(anchor.get("anchor_token_index"), "anchor token index")
        anchor_token_id = _require_int(anchor.get("anchor_token_id"), "anchor token ID")
        sampled = _require_int(anchor.get("sampled_token_index"), "sampled token index")
        _require_string(anchor.get("field_event_id"), "anchor field event ID")
        if index >= token_length or sampled != index + 1 or sampled >= token_length:
            raise ValueError("action anchor is not strictly causal within the capture tensor")
        if anchor_token_id != 25:
            raise ValueError("action anchor must be the frozen colon token")
        if span_by_key.get((turn, field), {}).get("token_start") != sampled or span_by_key[(turn, field)].get("token_end") != sampled + 1:
            raise ValueError("action anchor/token span binding is invalid")
        if span_by_key[(turn, field)].get("token_origin") != "model_sample":
            raise ValueError("action token span must be a model sample")
        if field in paired.setdefault(turn, {}):
            raise ValueError("duplicate same-turn action anchor")
        paired[turn][field] = anchor
    if set(paired) != {0, 1, 2, 3} or any(set(pair) != {"status", "caveat"} for pair in paired.values()):
        raise ValueError("schema-v2 growth extraction requires one status/caveat pair for all four turns")
    result: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for turn in range(4):
        status, caveat = paired[turn]["status"], paired[turn]["caveat"]
        if int(caveat["anchor_token_index"]) - int(status["anchor_token_index"]) != BRIDGE_LENGTH:
            raise ValueError("same-turn status/caveat anchors must satisfy exact c-s=6")
        if status.get("anchor_semantic_position") != "immediately_before_status_sample":
            raise ValueError("status anchor semantic kind is invalid")
        if caveat.get("anchor_semantic_position") != "immediately_before_caveat_sample_status_visible":
            raise ValueError("caveat anchor semantic kind is invalid")
        result.append((status, caveat))
    return tuple(result)


def _geometry_projection(value: Mapping[str, Any]) -> Mapping[str, Any]:
    geometry = value.get("geometry", value)
    projection = _require_mapping(geometry, "label-free geometry projection")
    if projection.get("schema_version") != 1 or projection.get("kind") not in {
        "relational_state_graph_geometry_row",
        "relational_structured_action_state_graph_geometry_projection",
    }:
        raise ValueError("geometry input must be a label-free state-graph geometry row")
    return projection


def _geometry_occurrence_metadata(
    geometry: Mapping[str, Any], sidecar: Mapping[str, Any], status: Mapping[str, Any], caveat: Mapping[str, Any], token_ids: torch.Tensor
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    for key in ("conversation_id", "family", "scenario_id", "orbit_id"):
        if geometry.get(key) != sidecar.get(key):
            raise ValueError(f"geometry/sidecar identity mismatch: {key}")
    occurrences = geometry.get("occurrences")
    states = geometry.get("causal_prefix_states")
    activations = geometry.get("activation_metadata")
    if not all(isinstance(value, list) for value in (occurrences, states, activations)):
        raise ValueError("geometry projection lacks occurrence, causal-state, or activation metadata")
    occurrence_by_event = {
        item.get("field_event_id"): _require_mapping(item, "geometry occurrence")
        for item in occurrences
        if isinstance(item, Mapping)
    }
    state_by_anchor = {
        item.get("anchor_token_index"): _require_mapping(item, "geometry causal prefix state")
        for item in states
        if isinstance(item, Mapping)
    }
    activation_by_occurrence = {
        item.get("occurrence_id"): _require_mapping(item, "geometry activation metadata")
        for item in activations
        if isinstance(item, Mapping)
    }
    pair_data: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for anchor in (status, caveat):
        event = _require_string(anchor.get("field_event_id"), "anchor field_event_id")
        occurrence = occurrence_by_event.get(event)
        if occurrence is None:
            raise ValueError("same-turn anchor is absent from label-free geometry occurrences")
        for key in ("conversation_id", "family", "scenario_id", "orbit_id", "sample_index"):
            if occurrence.get(key) != geometry.get(key):
                raise ValueError(f"geometry occurrence identity mismatch: {key}")
        if occurrence.get("field_name") != anchor.get("field_name") or occurrence.get("turn_index") != anchor.get("turn_index"):
            raise ValueError("geometry occurrence does not bind the action anchor")
        occurrence_id = _require_string(occurrence.get("occurrence_id"), "geometry occurrence_id")
        activation = activation_by_occurrence.get(occurrence_id)
        if activation is None:
            raise ValueError("geometry occurrence lacks activation realization metadata")
        anchor_index = int(anchor["anchor_token_index"])
        state = state_by_anchor.get(anchor_index)
        if state is None:
            raise ValueError("geometry occurrence lacks its causal prefix state")
        prefix_sha = _require_sha256(state.get("prefix_token_ids_sha256"), "geometry prefix token SHA-256")
        if state.get("prefix_token_count") != anchor_index + 1 or _int32_sha256(token_ids[: anchor_index + 1].tolist()) != prefix_sha:
            raise ValueError("geometry causal prefix token hash does not match capture tokens")
        state_sha = _require_sha256(state.get("prefix_state_sha256"), "geometry prefix state SHA-256")
        if occurrence.get("prefix_state_sha256") != state_sha or activation.get("prefix_state_sha256") != state_sha:
            raise ValueError("geometry occurrence/activation prefix-state binding is invalid")
        if activation.get("prefix_token_stop") != anchor_index + 1:
            raise ValueError("geometry activation prefix stop does not match anchor")
        if activation.get("source_tensor_sha256") != sidecar.get("tensor_sha256") or activation.get("source_tensor_path") != sidecar.get("tensor_path"):
            raise ValueError("geometry activation source tensor binding is invalid")
        pair_data.append((occurrence, activation))
    status_occurrence, status_activation = pair_data[0]
    caveat_occurrence, caveat_activation = pair_data[1]
    shared = (
        "family_fold", "split", "scenario_id", "orbit_id", "sample_index",
        "intervention_history", "pressure_exposed", "intervention_token_indices", "intervention_mask_scope",
    )
    for key in shared:
        if status_occurrence.get(key) != caveat_occurrence.get(key):
            raise ValueError(f"status/caveat occurrence metadata disagrees: {key}")
    indices = status_occurrence.get("intervention_token_indices")
    if not isinstance(indices, list) or any(not isinstance(index, int) or index < 0 for index in indices):
        raise ValueError("geometry intervention token indices are invalid")
    if len(set(indices)) != len(indices) or any(index > int(caveat["anchor_token_index"]) for index in indices):
        raise ValueError("geometry intervention token indices are outside the post-commitment prefix")
    return status_occurrence, caveat_occurrence, status_activation, caveat_activation


def _metadata_matches(handle_metadata: Mapping[str, str] | None, sidecar: Mapping[str, Any], structured: Mapping[str, Any]) -> None:
    if handle_metadata is None:
        raise ValueError("capture tensor lacks safetensors metadata")
    for key, expected in (
        ("schema_version", "2"),
        ("kind", "relational_capture_row"),
        ("conversation_id", str(sidecar.get("conversation_id"))),
        ("family", str(sidecar.get("family"))),
        ("source_row_sha256", str(sidecar.get("source_row_sha256"))),
        ("source_token_ids_sha256", str(sidecar.get("source_token_ids_sha256"))),
        ("capture_contract_sha256", str(sidecar.get("capture_contract_sha256"))),
        ("structured_metadata_sha256", _canonical_sha256(structured)),
    ):
        if handle_metadata.get(key) != expected:
            raise ValueError(f"capture tensor metadata mismatch: {key}")
    if handle_metadata.get("structured_metadata_json") != _canonical_bytes(structured).decode("utf-8"):
        raise ValueError("capture tensor structured metadata payload mismatch")


def _typed_prefix_annotations(
    token_tensors: Mapping[str, torch.Tensor],
    *,
    prefix_stop: int,
    intervention_token_indices: tuple[int, ...],
    intervention_mask_scope: str,
    origin_id_mapping: Mapping[str, int],
    origin_detail_id_mapping: Mapping[str, int],
) -> PrefixTokenAnnotations:
    message_ids = token_tensors["token_message_ids"][:prefix_stop].contiguous()
    mask = torch.zeros(prefix_stop, dtype=torch.bool)
    if intervention_token_indices:
        mask[list(intervention_token_indices)] = True
    denominator = max(prefix_stop - 1, 1)
    return PrefixTokenAnnotations(
        token_ids=token_tensors["token_ids"][:prefix_stop].contiguous(),
        position_ids=token_tensors["position_ids"][:prefix_stop].contiguous(),
        normalized_positions=torch.arange(prefix_stop, dtype=torch.float32) / denominator,
        role_ids=token_tensors["token_role_ids"][:prefix_stop].contiguous(),
        turn_ids=token_tensors["token_turn_ids"][:prefix_stop].contiguous(),
        message_ids=message_ids,
        span_flags=token_tensors["token_span_flags"][:prefix_stop].contiguous(),
        origin_ids=token_tensors["token_origin_ids"][:prefix_stop].contiguous(),
        origin_detail_ids=token_tensors["token_origin_detail_ids"][:prefix_stop].contiguous(),
        origin_id_mapping=dict(origin_id_mapping),
        origin_detail_id_mapping=dict(origin_detail_id_mapping),
        intervention_mask=mask,
        intervention_mask_scope=intervention_mask_scope,
    )


def _physical_edge_id(identity: GrowthEdgeIdentity) -> str:
    physical = {
        "conversation_id": identity.conversation_id,
        "turn_index": identity.turn_index,
        "status_occurrence_id": identity.status_occurrence_id,
        "caveat_occurrence_id": identity.caveat_occurrence_id,
        "status_field_event_id": identity.status_field_event_id,
        "caveat_field_event_id": identity.caveat_field_event_id,
    }
    return hashlib.sha256(
        _PHYSICAL_EDGE_ID_HASH_DOMAIN + _canonical_bytes(physical)
    ).hexdigest()


def _edge_payload_sha256(
    identity: GrowthEdgeIdentity,
    hashes: Mapping[str, str],
    annotations: PrefixTokenAnnotations,
) -> str:
    return hashlib.sha256(_EDGE_PAYLOAD_HASH_DOMAIN + _canonical_bytes({
        "identity": identity.hash_dict(),
        "tensor_hashes": hashes,
        "annotation_schema": {
            "origin_id_mapping": annotations.origin_id_mapping,
            "origin_detail_id_mapping": annotations.origin_detail_id_mapping,
            "intervention_mask_scope": annotations.intervention_mask_scope,
        },
    })).hexdigest()


def _edge_identity(
    sidecar: Mapping[str, Any],
    status: Mapping[str, Any],
    caveat: Mapping[str, Any],
    token_ids: torch.Tensor,
    status_occurrence: Mapping[str, Any],
    caveat_occurrence: Mapping[str, Any],
    status_activation: Mapping[str, Any],
    caveat_activation: Mapping[str, Any],
) -> GrowthEdgeIdentity:
    turn_index = int(status["turn_index"])
    s, c = int(status["anchor_token_index"]), int(caveat["anchor_token_index"])
    status_event, caveat_event = str(status["field_event_id"]), str(caveat["field_event_id"])
    physical = {
        "conversation_id": _require_string(sidecar.get("conversation_id"), "conversation_id"),
        "turn_index": turn_index,
        "status_occurrence_id": _require_string(status_occurrence.get("occurrence_id"), "status occurrence_id"),
        "caveat_occurrence_id": _require_string(caveat_occurrence.get("occurrence_id"), "caveat occurrence_id"),
        "status_field_event_id": status_event,
        "caveat_field_event_id": caveat_event,
    }
    history = status_occurrence.get("intervention_history")
    if not isinstance(history, list) or any(not isinstance(value, str) for value in history):
        raise ValueError("geometry intervention history is invalid")
    status_realization = status_occurrence.get("activation_realization_id")
    caveat_realization = caveat_occurrence.get("activation_realization_id")
    if status_realization is not None and not isinstance(status_realization, str):
        raise ValueError("status activation realization ID is invalid")
    if caveat_realization is not None and not isinstance(caveat_realization, str):
        raise ValueError("caveat activation realization ID is invalid")
    return GrowthEdgeIdentity(
        edge_id=hashlib.sha256(
            _PHYSICAL_EDGE_ID_HASH_DOMAIN + _canonical_bytes(physical)
        ).hexdigest(),
        conversation_id=physical["conversation_id"],
        family=_require_string(sidecar.get("family"), "family"),
        family_fold=_require_string(status_occurrence.get("family_fold"), "family_fold"),
        split=_require_string(status_occurrence.get("split"), "split"),
        scenario_id=_require_string(sidecar.get("scenario_id"), "scenario_id"),
        orbit_id=_require_string(sidecar.get("orbit_id"), "orbit_id"),
        sample_index=_require_int(status_occurrence.get("sample_index"), "sample_index"),
        intervention_program=_require_string(sidecar.get("intervention_program"), "intervention_program"),
        turn_index=turn_index,
        status_occurrence_id=physical["status_occurrence_id"],
        caveat_occurrence_id=physical["caveat_occurrence_id"],
        status_field_event_id=status_event,
        caveat_field_event_id=caveat_event,
        status_activation_realization_id=status_realization,
        caveat_activation_realization_id=caveat_realization,
        intervention_history=tuple(history),
        pressure_exposed=_require_bool(status_occurrence.get("pressure_exposed"), "pressure_exposed"),
        intervention_token_indices=tuple(int(index) for index in status_occurrence["intervention_token_indices"]),
        intervention_mask_scope=_require_string(status_occurrence.get("intervention_mask_scope"), "intervention_mask_scope"),
        status_anchor_index=s,
        caveat_anchor_index=c,
        status_prefix_token_ids_sha256=_int32_sha256(token_ids[: s + 1].tolist()),
        caveat_prefix_token_ids_sha256=_int32_sha256(token_ids[: c + 1].tolist()),
        status_prefix_state_sha256=_require_sha256(status_occurrence.get("prefix_state_sha256"), "status prefix_state_sha256"),
        caveat_prefix_state_sha256=_require_sha256(caveat_occurrence.get("prefix_state_sha256"), "caveat prefix_state_sha256"),
        status_sampled_token_id=int(token_ids[s + 1]),
        source_row_sha256=_require_sha256(sidecar.get("source_row_sha256"), "source_row_sha256"),
        source_tensor_path=_require_string(sidecar.get("tensor_path"), "tensor_path"),
        source_tensor_sha256=_require_sha256(sidecar.get("tensor_sha256"), "tensor_sha256"),
        source_tensor_bytes=_require_int(sidecar.get("tensor_bytes"), "tensor_bytes", minimum=1),
        capture_contract_sha256=_require_sha256(sidecar.get("capture_contract_sha256"), "capture_contract_sha256"),
    )


def extract_post_commitment_growth_row(
    sidecar: Mapping[str, Any] | Path | str,
    tensor_path: Path | str,
    geometry: Mapping[str, Any],
) -> PostCommitmentGrowthRow:
    """Extract four label-free, same-turn growth edges from one capture row.

    The required geometry input is the label-free state-graph projection for
    the same row. The tensor file is verified against the sidecar before it is
    opened. The function touches neither a model nor any outcome projection.
    """
    record = _read_sidecar(sidecar)
    path = Path(tensor_path)
    structured, layers, hidden_size, n_heads = _validate_sidecar(record, path)
    geometry_projection = _geometry_projection(geometry)
    token_length = _require_int(record.get("token_length"), "token_length", minimum=1)
    pairs = _anchor_pairs(structured, token_length)
    origin_id_mapping = _require_mapping(
        structured.get("token_origin_id_mapping"), "token_origin_id_mapping"
    )
    origin_detail_id_mapping = _require_mapping(
        structured.get("token_origin_detail_id_mapping"), "token_origin_detail_id_mapping"
    )
    if (
        any(not isinstance(key, str) or not isinstance(value, int) for key, value in origin_id_mapping.items())
        or any(not isinstance(key, str) or not isinstance(value, int) for key, value in origin_detail_id_mapping.items())
    ):
        raise ValueError("token origin mappings must map strings to integer IDs")
    edge_parts: list[dict[str, Any]] = [
        {"status": status, "caveat": caveat, "residual_base": [], "residual_bridge": [], "anchors": [], "attention": []}
        for status, caveat in pairs
    ]
    geometry_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        _metadata_matches(handle.metadata(), record, structured)
        expected_keys = set(_TYPED_TOKEN_KEYS)
        expected_keys.update(f"residual.L{layer}" for layer in layers)
        expected_keys.update(f"attention.L{layer}" for layer in layers)
        if set(handle.keys()) != expected_keys:
            raise ValueError("capture tensor key set does not match the schema-v2 contract")
        dtypes = {
            "token_ids": torch.int32, "position_ids": torch.int32, "token_role_ids": torch.uint8,
            "token_turn_ids": torch.int16, "token_message_ids": torch.int16, "token_span_flags": torch.int16,
            "token_origin_ids": torch.uint8, "token_origin_detail_ids": torch.uint8,
        }
        token_tensors: dict[str, torch.Tensor] = {}
        for key in _TYPED_TOKEN_KEYS:
            tensor = handle.get_tensor(key)
            if tensor.shape != (token_length,) or tensor.dtype != dtypes[key]:
                raise ValueError(f"capture tensor {key} has invalid shape or dtype")
            token_tensors[key] = tensor
        if _canonical_sha256(token_tensors["token_ids"].tolist()) != _require_sha256(record.get("source_token_ids_sha256"), "source_token_ids_sha256"):
            raise ValueError("capture token IDs do not match source token SHA-256")
        for status, caveat in pairs:
            s, c = int(status["anchor_token_index"]), int(caveat["anchor_token_index"])
            if int(token_tensors["token_ids"][s]) != 25 or int(
                token_tensors["token_ids"][c]
            ) != 25:
                raise ValueError("capture tokens do not contain the frozen action anchors")
            if token_tensors["token_ids"][s + 2 : c + 1].tolist() != [198, 34, 525, 266, 25]:
                raise ValueError("bridge tokens after sampled status are not the frozen Caveat prefix")
            geometry_pairs.append(
                _geometry_occurrence_metadata(
                    geometry_projection, record, status, caveat, token_tensors["token_ids"]
                )
            )
        for layer in layers:
            residual = handle.get_tensor(f"residual.L{layer}")
            attention = handle.get_tensor(f"attention.L{layer}")
            if residual.shape != (token_length, hidden_size) or residual.dtype != torch.bfloat16:
                raise ValueError(f"residual L{layer} has invalid shape or dtype")
            if attention.shape != (n_heads, token_length * (token_length + 1) // 2) or attention.dtype != torch.bfloat16:
                raise ValueError(f"attention L{layer} has invalid shape or dtype")
            for part in edge_parts:
                status, caveat = part["status"], part["caveat"]
                s, c = int(status["anchor_token_index"]), int(caveat["anchor_token_index"])
                bridge = residual[s + 1 : c + 1].float()
                base = residual[: s + 1].float()
                part["residual_base"].append(torch.cdist(bridge, base).to(torch.float32))
                part["residual_bridge"].append(torch.cdist(bridge, bridge).to(torch.float32))
                part["anchors"].append(residual[torch.tensor([s, c])].contiguous())
                rows: list[torch.Tensor] = []
                for query in range(s, c + 1):
                    start, stop = query * (query + 1) // 2, (query + 1) * (query + 2) // 2
                    rows.append(attention[:, start:stop].contiguous())
                part["attention"].append(rows)
    built: list[PostCommitmentGrowthEdge] = []
    for part, geometry_pair in zip(edge_parts, geometry_pairs, strict=True):
        status, caveat = part["status"], part["caveat"]
        s, c = int(status["anchor_token_index"]), int(caveat["anchor_token_index"])
        identity = _edge_identity(
            record, status, caveat, token_tensors["token_ids"], *geometry_pair
        )
        packed: list[torch.Tensor] = []
        offsets = torch.empty((len(layers), n_heads, BRIDGE_LENGTH + 1), dtype=torch.int64)
        lengths = torch.empty((len(layers), n_heads, BRIDGE_LENGTH + 1), dtype=torch.int32)
        cursor = 0
        for layer_index, rows in enumerate(part["attention"]):
            for query_offset, row in enumerate(rows):
                for head in range(n_heads):
                    offsets[layer_index, head, query_offset] = cursor
                    lengths[layer_index, head, query_offset] = row.shape[1]
                    packed.append(row[head])
                    cursor += row.shape[1]
        attention_values = torch.cat(packed).contiguous()
        relations_base = torch.stack(part["residual_base"]).contiguous()
        relations_bridge = torch.stack(part["residual_bridge"]).contiguous()
        anchors = torch.stack(part["anchors"]).contiguous()
        annotations = _typed_prefix_annotations(
            token_tensors,
            prefix_stop=c + 1,
            intervention_token_indices=identity.intervention_token_indices,
            intervention_mask_scope=identity.intervention_mask_scope,
            origin_id_mapping={str(key): int(value) for key, value in origin_id_mapping.items()},
            origin_detail_id_mapping={str(key): int(value) for key, value in origin_detail_id_mapping.items()},
        )
        tensors = {
            "bridge_token_indices": torch.arange(s + 1, c + 1, dtype=torch.int32),
            "attention_query_indices": torch.arange(s, c + 1, dtype=torch.int32),
            "residual_bridge_to_base": relations_base,
            "residual_within_bridge": relations_bridge,
            "anchor_residuals": anchors,
            "attention_values": attention_values,
            "attention_offsets": offsets,
            "attention_lengths": lengths,
            "prefix_token_ids": annotations.token_ids,
            "prefix_position_ids": annotations.position_ids,
            "prefix_normalized_positions": annotations.normalized_positions,
            "prefix_role_ids": annotations.role_ids,
            "prefix_turn_ids": annotations.turn_ids,
            "prefix_message_ids": annotations.message_ids,
            "prefix_span_flags": annotations.span_flags,
            "prefix_origin_ids": annotations.origin_ids,
            "prefix_origin_detail_ids": annotations.origin_detail_ids,
            "prefix_intervention_mask": annotations.intervention_mask,
        }
        hashes = {name: tensor_sha256(name, value) for name, value in tensors.items()}
        edge_sha = _edge_payload_sha256(identity, hashes, annotations)
        edge = PostCommitmentGrowthEdge(
            identity=identity,
            bridge_token_indices=tensors["bridge_token_indices"],
            attention_query_indices=tensors["attention_query_indices"],
            residual_bridge_to_base=relations_base,
            residual_within_bridge=relations_bridge,
            anchor_residuals=anchors,
            attention_values=attention_values,
            attention_offsets=offsets,
            attention_lengths=lengths,
            prefix_annotations=annotations,
            tensor_hashes=hashes,
            edge_sha256=edge_sha,
        )
        edge.validate()
        built.append(edge)
    row_payload = {
        "conversation_id": _require_string(record.get("conversation_id"), "conversation_id"),
        "source_tensor_sha256": _require_sha256(record.get("tensor_sha256"), "tensor_sha256"),
        "edges": [{"edge_id": edge.identity.edge_id, "edge_sha256": edge.edge_sha256} for edge in built],
    }
    row = PostCommitmentGrowthRow(
        conversation_id=row_payload["conversation_id"],
        source_tensor_sha256=row_payload["source_tensor_sha256"],
        source_tensor_bytes=_require_int(record.get("tensor_bytes"), "tensor_bytes", minimum=1),
        edges=tuple(built),
        row_sha256=hashlib.sha256(_ROW_HASH_DOMAIN + _canonical_bytes(row_payload)).hexdigest(),
    )
    row.validate()
    return row


extract_post_commitment_growth_edges = extract_post_commitment_growth_row
