"""Outcome-blind rooted stars at the exact pre-status anchor."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import struct
from typing import Any

import numpy as np
import torch

from geoprobe.data.relational_prefix_store import (
    RawRelationalPrefixCapture,
    RelationalPrefixReference,
)
from geoprobe.geometry.relational_partial_grammar import build_canonical_status_prefix_core


LAYERS = (12, 16, 19, 20)
VIEWS = ("action_free_full_context", "intervention_masked_action_free")
_TENSOR_HASH_DOMAIN = b"geoprobe.pre-status-rooted-star.tensor.v1\x00"
_STAR_HASH_DOMAIN = b"geoprobe.pre-status-rooted-star.payload.v1\x00"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def tensor_sha256(name: str, tensor: torch.Tensor) -> str:
    """Return a domain-separated, dtype- and shape-bound tensor digest."""
    if not name:
        raise ValueError("tensor hash name must be non-empty")
    hasher = hashlib.sha256(_TENSOR_HASH_DOMAIN)
    name_bytes = name.encode()
    dtype_bytes = str(tensor.dtype).removeprefix("torch.").encode()
    hasher.update(struct.pack("<I", len(name_bytes)))
    hasher.update(name_bytes)
    hasher.update(struct.pack("<I", len(dtype_bytes)))
    hasher.update(dtype_bytes)
    hasher.update(struct.pack("<I", tensor.ndim))
    for dimension in tensor.shape:
        hasher.update(struct.pack("<Q", int(dimension)))
    payload = _tensor_bytes(tensor)
    hasher.update(struct.pack("<Q", len(payload)))
    hasher.update(payload)
    return hasher.hexdigest()


def _anchor(reference: RelationalPrefixReference) -> dict[str, Any]:
    return {
        "field_name": reference.field_name,
        "turn_index": reference.turn_index,
        "anchor_token_index": reference.prefix_token_stop - 1,
        "anchor_token_id": 25,
    }


def _message_mask(reference: RelationalPrefixReference, raw: RawRelationalPrefixCapture) -> tuple[int, ...]:
    declared = tuple(reference.intervention_token_indices)
    if not declared:
        return ()
    if reference.intervention_mask_scope != "whole_intervention_bearing_user_message":
        raise ValueError("intervention mask is not the frozen whole-message scope")
    messages = raw.token_message_ids.tolist()
    if len(set(declared)) != len(declared) or any(index < 0 or index >= len(messages) for index in declared):
        raise ValueError("intervention token indices are invalid")
    excluded = tuple(sorted({int(messages[index]) for index in declared}))
    actual = tuple(index for index, message in enumerate(messages) if int(message) in set(excluded))
    if actual != declared:
        raise ValueError("intervention indices do not cover complete frozen messages")
    return excluded


def _core(
    reference: RelationalPrefixReference,
    raw: RawRelationalPrefixCapture,
    *,
    excluded_messages: Sequence[int],
):
    return build_canonical_status_prefix_core(
        object_id=reference.reference_id,
        token_ids=raw.token_ids.tolist(),
        position_ids=raw.position_ids.tolist(),
        token_role_ids=raw.token_role_ids.tolist(),
        token_turn_ids=raw.token_turn_ids.tolist(),
        token_message_ids=raw.token_message_ids.tolist(),
        token_span_flags=raw.token_span_flags.tolist(),
        token_origins=raw.token_origins,
        token_origin_details=raw.token_origin_details,
        anchor=_anchor(reference),
        excluded_message_ids=excluded_messages,
        correspondence_metadata={
            "reference_id": reference.reference_id,
            "prefix_state_sha256": reference.prefix_state_sha256,
        },
    )


def _origin_codes(values: Sequence[str]) -> tuple[torch.Tensor, dict[str, int]]:
    mapping = {value: index for index, value in enumerate(sorted(set(values)))}
    return torch.tensor([mapping[value] for value in values], dtype=torch.uint8), mapping


@dataclass(frozen=True, slots=True)
class RootedStarView:
    """One typed, action-free rooted relational view."""

    name: str
    retained_indices: torch.Tensor
    root_residuals: torch.Tensor
    root_to_context_residual_distances: torch.Tensor
    incoming_attention: torch.Tensor
    removed_attention_mass: torch.Tensor
    annotations: Mapping[str, torch.Tensor]
    origin_id_mapping: Mapping[str, int]
    origin_detail_id_mapping: Mapping[str, int]
    tensor_hashes: Mapping[str, str]
    star_sha256: str

    def tensors(self) -> dict[str, torch.Tensor]:
        result = {
            "retained_token_indices": self.retained_indices,
            "root_residuals": self.root_residuals,
            "root_to_context_residual_distances": self.root_to_context_residual_distances,
            "incoming_attention": self.incoming_attention,
            "removed_attention_mass": self.removed_attention_mass,
        }
        result.update(self.annotations)
        return result

    def validate(self, reference: RelationalPrefixReference) -> None:
        tensors = self.tensors()
        n = self.retained_indices.numel()
        if self.name not in VIEWS or n < 1 or int(self.retained_indices[-1]) != reference.prefix_token_stop - 1:
            raise ValueError("rooted-star retained tokens do not retain the status root")
        if self.retained_indices.dtype != torch.int32 or self.root_residuals.ndim != 2:
            raise ValueError("rooted-star root payload has invalid dtype or rank")
        if self.root_residuals.shape[0] != len(LAYERS) or self.root_residuals.dtype != torch.bfloat16:
            raise ValueError("rooted-star root residuals must be [4, hidden] BF16")
        if self.root_to_context_residual_distances.shape != (len(LAYERS), n) or self.root_to_context_residual_distances.dtype != torch.float32:
            raise ValueError("rooted-star residual relation has invalid shape")
        if self.incoming_attention.ndim != 3 or self.incoming_attention.shape[0] != len(LAYERS) or self.incoming_attention.shape[2] != n or self.incoming_attention.dtype != torch.bfloat16:
            raise ValueError("rooted-star incoming attention has invalid shape")
        if self.removed_attention_mass.shape != self.incoming_attention.shape[:2] or self.removed_attention_mass.dtype != torch.float32:
            raise ValueError("rooted-star removed attention mass has invalid shape")
        if not torch.allclose(self.incoming_attention.float().sum(dim=-1), torch.ones_like(self.removed_attention_mass), atol=0.02, rtol=0.02):
            raise ValueError("rooted-star retained attention is not renormalized")
        if set(tensors) != set(self.tensor_hashes) or {name: tensor_sha256(name, value) for name, value in tensors.items()} != dict(self.tensor_hashes):
            raise ValueError("rooted-star tensor hashes do not bind the payload")
        payload = {"view": self.name, "reference_id": reference.reference_id, "tensor_hashes": dict(self.tensor_hashes), "annotation_schema": {"origin_id_mapping": dict(self.origin_id_mapping), "origin_detail_id_mapping": dict(self.origin_detail_id_mapping)}}
        if self.star_sha256 != hashlib.sha256(_STAR_HASH_DOMAIN + _canonical(payload)).hexdigest():
            raise ValueError("rooted-star payload hash is invalid")


def build_status_rooted_star_views(
    reference: RelationalPrefixReference,
    raw: RawRelationalPrefixCapture,
) -> tuple[RootedStarView, ...]:
    """Build the two frozen action-free status-root views from one raw prefix."""
    if reference.field_name != "status" or raw.anchor_index != reference.prefix_token_stop - 1:
        raise ValueError("rooted stars require an exact status-prefix reference")
    if tuple(reference.layers) != LAYERS or tuple(sorted(raw.residuals)) != LAYERS or tuple(sorted(raw.attentions)) != LAYERS:
        raise ValueError("rooted stars require the frozen four-layer capture")
    full = _core(reference, raw, excluded_messages=())
    masked = _core(reference, raw, excluded_messages=_message_mask(reference, raw))
    result: list[RootedStarView] = []
    for name, core in ((VIEWS[0], full), (VIEWS[1], masked)):
        indices = np.asarray(core.token_indices, dtype=np.int64)
        if indices.size == 0 or int(indices[-1]) != raw.anchor_index:
            raise ValueError("canonical grammar did not retain the fixed status root")
        annotation_values = {
            "token_ids": raw.token_ids[indices],
            "position_ids": raw.position_ids[indices],
            "normalized_positions": raw.position_ids[indices].astype(np.float32) / max(raw.anchor_index, 1),
            "role_ids": raw.token_role_ids[indices],
            "turn_ids": raw.token_turn_ids[indices],
            "message_ids": raw.token_message_ids[indices],
            "span_flags": raw.token_span_flags[indices],
        }
        origin_ids, origin_mapping = _origin_codes([raw.token_origins[index] for index in indices])
        detail_ids, detail_mapping = _origin_codes([raw.token_origin_details[index] for index in indices])
        annotations = {
            "token_ids": torch.from_numpy(annotation_values["token_ids"].astype(np.int32, copy=True)),
            "position_ids": torch.from_numpy(annotation_values["position_ids"].astype(np.int32, copy=True)),
            "normalized_positions": torch.from_numpy(annotation_values["normalized_positions"].copy()),
            "role_ids": torch.from_numpy(annotation_values["role_ids"].astype(np.uint8, copy=True)),
            "turn_ids": torch.from_numpy(annotation_values["turn_ids"].astype(np.int16, copy=True)),
            "message_ids": torch.from_numpy(annotation_values["message_ids"].astype(np.int16, copy=True)),
            "span_flags": torch.from_numpy(annotation_values["span_flags"].astype(np.int16, copy=True)),
            "origin_ids": origin_ids,
            "origin_detail_ids": detail_ids,
        }
        roots, distances, attention_rows, removed = [], [], [], []
        for layer in LAYERS:
            residual = np.asarray(raw.residuals[layer])
            root = residual[raw.anchor_index]
            roots.append(torch.from_numpy(root).to(torch.bfloat16).contiguous())
            distances.append(torch.from_numpy(np.linalg.norm(residual[indices].astype(np.float32) - root.astype(np.float32), axis=1).astype(np.float32)))
            packed = np.asarray(raw.attentions[layer], dtype=np.float32)
            start, stop = raw.anchor_index * (raw.anchor_index + 1) // 2, (raw.anchor_index + 1) * (raw.anchor_index + 2) // 2
            row = packed[:, start:stop]
            retained = row[:, indices]
            mass = retained.sum(axis=1, dtype=np.float64)
            if np.any(~np.isfinite(mass)) or np.any(mass <= 0):
                raise ValueError("action/intervention masking leaves a root attention row without mass")
            removed.append(torch.from_numpy((row.sum(axis=1, dtype=np.float64) - mass).astype(np.float32)))
            attention_rows.append(torch.from_numpy((retained / mass[:, None]).astype(np.float32)).to(torch.bfloat16))
        view = RootedStarView(
            name=name,
            retained_indices=torch.from_numpy(indices.astype(np.int32, copy=True)),
            root_residuals=torch.stack(roots).contiguous(),
            root_to_context_residual_distances=torch.stack(distances).to(torch.float32).contiguous(),
            incoming_attention=torch.stack(attention_rows).contiguous(),
            removed_attention_mass=torch.stack(removed).to(torch.float32).contiguous(),
            annotations=annotations,
            origin_id_mapping=origin_mapping,
            origin_detail_id_mapping=detail_mapping,
            tensor_hashes={},
            star_sha256="",
        )
        hashes = {key: tensor_sha256(key, value) for key, value in view.tensors().items()}
        payload = {"view": name, "reference_id": reference.reference_id, "tensor_hashes": hashes, "annotation_schema": {"origin_id_mapping": origin_mapping, "origin_detail_id_mapping": detail_mapping}}
        view = replace(view, tensor_hashes=hashes, star_sha256=hashlib.sha256(_STAR_HASH_DOMAIN + _canonical(payload)).hexdigest())
        view.validate(reference)
        result.append(view)
    return tuple(result)


__all__ = ["LAYERS", "VIEWS", "RootedStarView", "build_status_rooted_star_views", "tensor_sha256"]
