"""Reusable pure-Numpy causal arm builder for the pre-status relational run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
import json
from typing import Any

import numpy as np


CAUSAL_LAYERS: tuple[int, ...] = (12, 16, 19, 20)
PRIMARY_ACTUATION_LAYER = 12
CAUSAL_ARM_ORDER: tuple[str, ...] = (
    "noop",
    "fixed_global_h",
    "generic_t",
    "specific_s",
    "full_h",
    "generic_minus_s",
    "generic_plus_random_s",
)

_RANDOM_ARM = "generic_plus_random_s"
_RANDOM_DOMAIN = "relational_pre_status_causal_v1_random_r"
_ARRAY_HASH_DOMAIN = b"geoprobe.pre-status-causal-vector.v1\x00"


class RelationalPreStatusCausalError(ValueError):
    """Raised when the causal-arm construction contract is violated."""


def _normalize_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RelationalPreStatusCausalError(f"{name} must be a non-empty string")
    return str(value)


def _as_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {
            str(key): _as_json(value[key]) for key in sorted(value, key=lambda key: str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_as_json(item) for item in value]
    if isinstance(value, set):
        return [_as_json(item) for item in sorted(value, key=lambda item: str(item))]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _as_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f4")
    shape = json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    return sha256(_ARRAY_HASH_DOMAIN + shape + b"\x00" + array.tobytes()).hexdigest()


def _readonly_array(value: Any, name: str, *, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise RelationalPreStatusCausalError(
            f"{name} must be a finite two-dimensional matrix"
        )
    if not np.isfinite(array).all():
        raise RelationalPreStatusCausalError(f"{name} must be finite")
    if expected_shape is not None and array.shape != expected_shape:
        raise RelationalPreStatusCausalError(
            f"{name} has shape {array.shape}, expected {expected_shape}"
        )
    array = np.ascontiguousarray(array.copy())
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise RelationalPreStatusCausalError(f"{name} must have at least one hidden dimension")
    array.setflags(write=False)
    return array


def _build_layer_seed(state_id: str, family_fold: str, layer: int, arm: str) -> tuple[int, str]:
    payload = {
        "kind": "relational_pre_status_causal_random",
        "version": 1,
        "domain": _RANDOM_DOMAIN,
        "state_id": state_id,
        "family_fold": family_fold,
        "layer": layer,
        "arm": arm,
    }
    digest = _canonical_sha256(payload)
    seed = int(digest[:16], 16)
    return seed, digest


def _build_random_row(
    s_row: np.ndarray,
    *,
    state_id: str,
    family_fold: str,
    layer: int,
    arm: str,
) -> tuple[np.ndarray, int, str]:
    target_norm = float(np.linalg.norm(s_row))
    seed, digest = _build_layer_seed(state_id, family_fold, layer, arm)
    rng = np.random.Generator(np.random.PCG64(seed))
    random_row = rng.normal(size=s_row.shape[0]).astype(np.float64)
    random_norm = float(np.linalg.norm(random_row))
    if target_norm == 0.0:
        random_row = np.zeros_like(random_row)
    elif random_norm == 0.0:
        raise RelationalPreStatusCausalError(
            f"failed to draw non-zero random control at layer {layer}"
        )
    else:
        random_row = random_row * (target_norm / random_norm)
    random_row.setflags(write=False)
    return random_row, seed, digest


def _vector_hashes(vectors: Mapping[str, np.ndarray]) -> MappingProxyType[str, str]:
    return MappingProxyType({name: _array_sha256(array) for name, array in vectors.items()})


@dataclass(frozen=True, slots=True)
class PreStatusCausalArmBundle:
    """Immutable container for all causal pre-status arm vectors."""

    state_id: str
    family_fold: str
    h: np.ndarray
    t: np.ndarray
    g: np.ndarray
    s: np.ndarray
    random_s: np.ndarray
    arm_vectors: Mapping[str, np.ndarray]
    per_layer_norms: Mapping[str, tuple[float, ...]]
    source_hashes: Mapping[str, str]
    arm_hashes: Mapping[str, str]
    metadata: Mapping[str, Any]
    bundle_hash: str


def select_primary_actuation_vectors(
    arm_vectors: Mapping[str, np.ndarray],
    *,
    layer: int = PRIMARY_ACTUATION_LAYER,
) -> MappingProxyType[str, np.ndarray]:
    """Select one coherent layer cross-section for causal actuation."""

    if layer not in CAUSAL_LAYERS:
        raise RelationalPreStatusCausalError(
            f"actuation layer must be one of {CAUSAL_LAYERS}, got {layer}"
        )
    if tuple(arm_vectors) != CAUSAL_ARM_ORDER:
        raise RelationalPreStatusCausalError("causal arm order is malformed")
    layer_index = CAUSAL_LAYERS.index(layer)
    selected: dict[str, np.ndarray] = {}
    hidden_size: int | None = None
    for name in CAUSAL_ARM_ORDER:
        matrix = _readonly_array(arm_vectors[name], f"arm {name}")
        if matrix.shape[0] != len(CAUSAL_LAYERS):
            raise RelationalPreStatusCausalError(
                f"arm {name} must contain exactly {len(CAUSAL_LAYERS)} layers"
            )
        if hidden_size is None:
            hidden_size = int(matrix.shape[1])
        elif matrix.shape[1] != hidden_size:
            raise RelationalPreStatusCausalError("causal arm hidden sizes disagree")
        vector = np.ascontiguousarray(matrix[layer_index].copy(), dtype=np.float32)
        vector.setflags(write=False)
        selected[name] = vector
    return MappingProxyType(selected)


def build_pre_status_causal_arm_bundle(
    state_id: str,
    family_fold: str,
    H: np.ndarray,
    T: np.ndarray,
    G: np.ndarray,
) -> PreStatusCausalArmBundle:
    """Build immutable causal-arm vectors on the exact layers used in v1."""

    state_id = _normalize_identifier(state_id, "state_id")
    family_fold = _normalize_identifier(family_fold, "family_fold")

    h = _readonly_array(H, "H")
    if h.shape[0] != len(CAUSAL_LAYERS):
        raise RelationalPreStatusCausalError(
            f"H shape {h.shape} is incompatible with required layer count {len(CAUSAL_LAYERS)}"
        )
    t = _readonly_array(T, "T", expected_shape=h.shape)
    g = _readonly_array(G, "G", expected_shape=h.shape)
    if t.shape != h.shape or g.shape != h.shape:
        raise RelationalPreStatusCausalError("H/T/G must have identical shapes")

    s = _readonly_array(h - t, "S")
    if not np.allclose(h, t + s):
        raise RelationalPreStatusCausalError("H must equal T + S")

    random_rows: list[np.ndarray] = []
    seed_payload: dict[int, dict[str, str | int]] = {}
    for layer, s_row in zip(CAUSAL_LAYERS, s):
        random_row, seed, seed_digest = _build_random_row(
            s_row,
            state_id=state_id,
            family_fold=family_fold,
            layer=layer,
            arm=_RANDOM_ARM,
        )
        random_rows.append(random_row)
        seed_payload[layer] = {"seed": seed, "seed_digest": seed_digest}
    random_s = np.ascontiguousarray(np.stack(random_rows, axis=0), dtype=np.float32)
    random_s.setflags(write=False)

    if not np.allclose(np.linalg.norm(random_s, axis=1), np.linalg.norm(s, axis=1)):
        raise RelationalPreStatusCausalError("random arm norms do not match S")

    zero = np.zeros_like(h)
    arm_vectors: dict[str, np.ndarray] = {
        "noop": zero,
        "fixed_global_h": g,
        "generic_t": t,
        "specific_s": s,
        "full_h": h,
        "generic_minus_s": t - s,
        "generic_plus_random_s": t + random_s,
    }
    for name in CAUSAL_ARM_ORDER:
        if name not in arm_vectors:
            raise RelationalPreStatusCausalError(f"missing required arm {name}")
    arm_vectors = {
        name: _readonly_array(vector, f"arm {name}", expected_shape=h.shape)
        for name, vector in arm_vectors.items()
    }
    arm_vectors = {name: arm_vectors[name] for name in CAUSAL_ARM_ORDER}
    arm_vectors = MappingProxyType(arm_vectors)

    per_layer_norms: dict[str, tuple[float, ...]] = {}
    for name, vector in arm_vectors.items():
        norms = tuple(float(norm) for norm in np.linalg.norm(vector, axis=1))
        if len(norms) != len(CAUSAL_LAYERS):
            raise RelationalPreStatusCausalError(
                f"arm {name} per-layer norms are malformed"
            )
        if any(not np.isfinite(norm) for norm in norms):
            raise RelationalPreStatusCausalError(f"arm {name} has non-finite norms")
        per_layer_norms[name] = norms
    per_layer_norms = MappingProxyType(per_layer_norms)

    source_payload = {
        "H": _array_sha256(h),
        "T": _array_sha256(t),
        "G": _array_sha256(g),
        "S": _array_sha256(s),
        "R": _array_sha256(random_s),
    }
    source_hashes = MappingProxyType(source_payload)

    arm_hashes = _vector_hashes(arm_vectors)
    metadata = MappingProxyType(
        {
            "schema_version": 1,
            "kind": "relational_pre_status_causal_v1",
            "state_id": state_id,
            "family_fold": family_fold,
            "layers": CAUSAL_LAYERS,
            "primary_actuation_layer": PRIMARY_ACTUATION_LAYER,
            "actuation_rule": "single_post_block_residual_cross_section",
            "arm_order": CAUSAL_ARM_ORDER,
            "random_arm": _RANDOM_ARM,
            "random_seed": _RANDOM_DOMAIN,
            "random_bit_generator": "PCG64",
            "random_seed_payload": seed_payload,
            "source_hashes": source_payload,
            "arm_hashes": dict(arm_hashes),
            "per_layer_norms": {name: list(norms) for name, norms in per_layer_norms.items()},
        }
    )

    bundle_hash = _canonical_sha256(metadata)
    metadata = MappingProxyType({**dict(metadata), "bundle_hash": bundle_hash})

    return PreStatusCausalArmBundle(
        state_id=state_id,
        family_fold=family_fold,
        h=h,
        t=t,
        g=g,
        s=s,
        random_s=random_s,
        arm_vectors=arm_vectors,
        per_layer_norms=per_layer_norms,
        source_hashes=source_hashes,
        arm_hashes=arm_hashes,
        metadata=metadata,
        bundle_hash=bundle_hash,
    )


def build_relational_pre_status_causal_arm_bundle(
    state_id: str,
    family_fold: str,
    H: np.ndarray,
    T: np.ndarray,
    G: np.ndarray,
) -> PreStatusCausalArmBundle:
    """Compatibility entrypoint for causal arm construction."""

    return build_pre_status_causal_arm_bundle(state_id, family_fold, H, T, G)


__all__ = [
    "CAUSAL_ARM_ORDER",
    "CAUSAL_LAYERS",
    "PRIMARY_ACTUATION_LAYER",
    "PreStatusCausalArmBundle",
    "RelationalPreStatusCausalError",
    "build_pre_status_causal_arm_bundle",
    "build_relational_pre_status_causal_arm_bundle",
    "select_primary_actuation_vectors",
]
