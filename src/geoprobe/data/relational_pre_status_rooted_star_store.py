"""Immutable, outcome-blind access to materialized pre-status rooted stars."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from geoprobe.data.relational_pre_status_rooted_star import LAYERS, VIEWS, tensor_sha256
from geoprobe.data.relational_prefix_store import CompactTensorLocators, RelationalPrefixReference
from geoprobe.geometry.relational_pre_status_rooted_metric import RootedStarMetricInput
from geoprobe.io import file_sha256


_SCHEMA_VERSION = 1
_MANIFEST_KIND = "relational_pre_status_rooted_star_materialization_manifest"
_ROW_KIND = "relational_pre_status_rooted_star_materialization_row"
_SHARD_KIND = "relational_pre_status_rooted_star_materialization_family_shard"
_STAR_HASH_DOMAIN = b"geoprobe.pre-status-rooted-star.payload.v1\x00"
_GEOMETRY_HASH_DOMAIN = b"geoprobe.pre-status-rooted-star.geometry-realization.v1\x00"
_PHYSICAL_ID_DOMAIN = b"geoprobe.pre-status-rooted-star.physical-reference.v1\x00"
_MANIFEST_NAME = "manifest.json"
_TENSOR_NAMES = (
    "retained_token_indices", "root_residuals", "root_to_context_residual_distances",
    "incoming_attention", "removed_attention_mass", "token_ids", "position_ids",
    "normalized_positions", "role_ids", "turn_ids", "message_ids", "span_flags",
    "origin_ids", "origin_detail_ids",
)
_METRIC_NAMES = (
    "root_to_context_residual_distances", "incoming_attention", "normalized_positions",
    "role_ids", "turn_ids", "message_ids", "span_flags", "origin_ids", "origin_detail_ids",
)
_OUTCOME_TERMS = ("outcome", "mapped", "desired", "knowledge", "decept", "honest", "target", "label")
_CANONICAL_ORIGIN_IDS = {"chat_source": 0, "environment": 1, "model_sample": 2}
_CANONICAL_ORIGIN_DETAIL_IDS = {
    "chat_source": 0,
    "environment_status_prefix": 1,
    "model_sampled_status_action": 2,
    "environment_caveat_separator": 3,
    "model_sampled_caveat_action": 4,
    "environment_eot": 5,
}


class RootedStarStoreError(ValueError):
    """Raised when a rooted-star bank is unsafe or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class RootedStarTensorLocators:
    family_shard_path: str
    family_shard_sha256: str
    tensor_keys: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class RootedStarReference:
    """One physical ``(status occurrence, view)`` record in the sealed bank."""

    rooted_star_id: str
    reference_id: str
    field_event_id: str
    occurrence_id: str
    view: str
    star_sha256: str
    geometry_sha256: str
    source_reference: RelationalPrefixReference
    tensor_locators: RootedStarTensorLocators
    tensor_hashes: Mapping[str, str]
    origin_id_mapping: Mapping[str, int]
    origin_detail_id_mapping: Mapping[str, int]

    @property
    def family(self) -> str:
        return self.source_reference.family


@dataclass(frozen=True, slots=True)
class RootedStarObservationBinding:
    """Outcome-free tensors sufficient to verify one exact live replay anchor."""

    rooted_star_id: str
    reference_id: str
    view: str
    geometry_sha256: str
    prefix_token_count: int
    retained_token_indices: np.ndarray
    root_residuals: np.ndarray
    incoming_attention: np.ndarray

    def __post_init__(self) -> None:
        if (
            not self.rooted_star_id
            or not self.reference_id
            or len(self.geometry_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.geometry_sha256)
            or self.view not in VIEWS
            or self.prefix_token_count < 1
        ):
            raise RootedStarStoreError(
                "observation binding identity, view, or prefix length is invalid"
            )
        retained = np.ascontiguousarray(
            np.asarray(self.retained_token_indices, dtype=np.int64)
        )
        roots = np.ascontiguousarray(np.asarray(self.root_residuals, dtype=np.float32))
        attention = np.ascontiguousarray(
            np.asarray(self.incoming_attention, dtype=np.float32)
        )
        if (
            retained.ndim != 1
            or retained.size < 1
            or int(retained[-1]) != self.prefix_token_count - 1
            or np.any(retained < 0)
            or np.any(retained >= self.prefix_token_count)
            or np.any(np.diff(retained) <= 0)
            or roots.ndim != 2
            or roots.shape[0] != len(LAYERS)
            or attention.ndim != 3
            or attention.shape[0] != len(LAYERS)
            or attention.shape[1] < 1
            or attention.shape[2] != retained.size
            or not np.isfinite(roots).all()
            or not np.isfinite(attention).all()
            or np.any(attention < 0.0)
        ):
            raise RootedStarStoreError("observation binding tensors are invalid")
        if not np.allclose(attention.sum(axis=2), 1.0, atol=0.02, rtol=0.02):
            raise RootedStarStoreError("observation binding attention is not normalized")
        for value in (retained, roots, attention):
            value.flags.writeable = False
        object.__setattr__(self, "retained_token_indices", retained)
        object.__setattr__(self, "root_residuals", roots)
        object.__setattr__(self, "incoming_attention", attention)


@dataclass(frozen=True, slots=True)
class RelationalPreStatusRootedStarIndex:
    artifact_root: Path
    manifest_sha256: str
    input_identity_sha256: str
    references: tuple[RootedStarReference, ...]
    geometry_references: tuple[RootedStarReference, ...]
    exact_view_groups: Mapping[str, tuple[RootedStarReference, ...]]
    by_rooted_star_id: Mapping[str, RootedStarReference]
    by_reference_id: Mapping[str, tuple[RootedStarReference, ...]]
    by_field_event_id: Mapping[str, tuple[RootedStarReference, ...]]
    by_occurrence_id: Mapping[str, tuple[RootedStarReference, ...]]


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RootedStarStoreError("record is not canonical JSON") from error


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RootedStarStoreError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RootedStarStoreError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RootedStarStoreError(f"{label} must be an integer >= {minimum}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RootedStarStoreError(f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise RootedStarStoreError(f"{label} is not valid UTF-8 JSON") from error
    return _mapping(value, label)


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RootedStarStoreError(f"{path} is not valid UTF-8 JSONL") from error
    if not lines:
        raise RootedStarStoreError(f"{path} is empty")
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise RootedStarStoreError(f"{path} has a blank JSONL row")
        try:
            yield _mapping(json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))), f"{path} line {number}")
        except (ValueError, json.JSONDecodeError) as error:
            raise RootedStarStoreError(f"{path} line {number} is not valid JSON") from error


def _safe_child(root: Path, relative: Any, label: str) -> Path:
    text = _string(relative, label)
    if "\\" in text or Path(text).is_absolute():
        raise RootedStarStoreError(f"{label} must be a safe relative POSIX path")
    candidate = (root / text).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise RootedStarStoreError(f"{label} escapes artifact root or is absent")
    return candidate


def _sealed(value: Any, path: str, *, allow_manifest_status: bool = False) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower()
            if not (allow_manifest_status and path == "manifest" and name == "status") and any(term in name for term in _OUTCOME_TERMS):
                raise RootedStarStoreError(f"outcome-bearing field escaped into {path}.{key}")
            _sealed(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _sealed(child, f"{path}[{index}]")


def _metadata(handle: Any, *, family: str, input_identity: str) -> None:
    expected = {
        "schema_version": "1", "kind": _SHARD_KIND, "family": family,
        "input_identity_sha256": input_identity, "payload_scope": "outcome_blind_rooted_stars",
    }
    if handle.metadata() != expected:
        raise RootedStarStoreError(f"family {family} safetensors metadata is invalid")


def _reference_from_dict(raw: Mapping[str, Any]) -> RelationalPrefixReference:
    try:
        compact = _mapping(raw.get("compact"), "compact locators")
        reference = RelationalPrefixReference(
            reference_id=_sha(raw.get("reference_id"), "reference ID"), realization_sha256=_sha(raw.get("realization_sha256"), "realization SHA"),
            canonical_realization_id=_sha(raw.get("canonical_realization_id"), "canonical realization ID"), equality_status=_string(raw.get("equality_status"), "equality status"),
            occurrence_id=_string(raw.get("occurrence_id"), "occurrence ID"), equivalent_occurrence_ids=tuple(_string(value, "equivalent occurrence ID") for value in raw.get("equivalent_occurrence_ids", [])),
            field_event_id=_string(raw.get("field_event_id"), "field event ID"), field_name=_string(raw.get("field_name"), "field name"), turn_index=_integer(raw.get("turn_index"), "turn index"),
            conversation_id=_string(raw.get("conversation_id"), "conversation ID"), family=_string(raw.get("family"), "family"), family_fold=_string(raw.get("family_fold"), "family fold"),
            split=_string(raw.get("split"), "split"), scenario_id=_string(raw.get("scenario_id"), "scenario ID"), orbit_id=_string(raw.get("orbit_id"), "orbit ID"), sample_index=_integer(raw.get("sample_index"), "sample index"),
            prefix_state_sha256=_sha(raw.get("prefix_state_sha256"), "prefix state SHA"), prefix_token_ids_sha256=_sha(raw.get("prefix_token_ids_sha256"), "prefix token ID SHA"),
            prefix_token_stop=_integer(raw.get("prefix_token_stop"), "prefix token stop", minimum=1), packed_attention_prefix_stop=_integer(raw.get("packed_attention_prefix_stop"), "packed attention prefix stop", minimum=1),
            source_row_sha256=_sha(raw.get("source_row_sha256"), "source row SHA"), source_tensor_object_path=_string(raw.get("source_tensor_object_path"), "source tensor object path"), source_tensor_sha256=_sha(raw.get("source_tensor_sha256"), "source tensor SHA"),
            capture_contract_sha256=_sha(raw.get("capture_contract_sha256"), "capture contract SHA"), layers=tuple(_integer(value, "layer") for value in raw.get("layers", [])),
            intervention_history=tuple(_string(value, "intervention history") for value in raw.get("intervention_history", [])), pressure_exposed=raw.get("pressure_exposed"),
            intervention_token_indices=tuple(_integer(value, "intervention index") for value in raw.get("intervention_token_indices", [])), intervention_mask_scope=_string(raw.get("intervention_mask_scope"), "intervention mask scope"),
            compact=CompactTensorLocators(
                _string(compact.get("tensor_shard"), "compact tensor shard"), _sha(compact.get("tensor_shard_sha256"), "compact shard SHA"), _string(compact.get("anchor_residual_key"), "anchor residual key"),
                _integer(compact.get("anchor_residual_occurrence_index"), "anchor residual index"), _string(compact.get("incoming_attention_values_key"), "attention values key"), _string(compact.get("incoming_attention_offsets_key"), "attention offsets key"),
                _string(compact.get("incoming_attention_lengths_key"), "attention lengths key"), _integer(compact.get("incoming_attention_index"), "attention index"), _string(compact.get("anchor_token_indices_key"), "anchor indices key"), _string(compact.get("layer_ids_key"), "layer IDs key"),
            ),
        )
    except (TypeError, KeyError) as error:
        raise RootedStarStoreError("rooted-star source reference is malformed") from error
    if reference.field_name != "status" or reference.turn_index not in range(4) or reference.layers != LAYERS:
        raise RootedStarStoreError("rooted-star source reference is not a frozen status occurrence")
    if not isinstance(reference.pressure_exposed, bool) or reference.pressure_exposed != bool(reference.intervention_token_indices):
        raise RootedStarStoreError("rooted-star source reference has invalid intervention exposure")
    if reference.packed_attention_prefix_stop != reference.prefix_token_stop * (reference.prefix_token_stop + 1) // 2:
        raise RootedStarStoreError("rooted-star source reference has invalid prefix cutoffs")
    return reference


def _star_hash(reference: RelationalPrefixReference, raw: Mapping[str, Any]) -> str:
    hashes = _mapping(raw.get("tensor_hashes"), "view tensor hashes")
    schema = _mapping(raw.get("annotation_schema"), "annotation schema")
    payload = {"view": raw.get("view"), "reference_id": reference.reference_id, "tensor_hashes": hashes, "annotation_schema": schema}
    return hashlib.sha256(_STAR_HASH_DOMAIN + _canonical(payload)).hexdigest()


def _physical_id(reference: RelationalPrefixReference, view: str, star_sha: str) -> str:
    return hashlib.sha256(_PHYSICAL_ID_DOMAIN + _canonical({"reference_id": reference.reference_id, "view": view, "star_sha256": star_sha})).hexdigest()


def _geometry_hash(raw: Mapping[str, Any]) -> str:
    """Reference-free exact payload identity for repeated captured rooted states."""
    return hashlib.sha256(_GEOMETRY_HASH_DOMAIN + _canonical({
        "view": raw.get("view"), "tensor_hashes": raw.get("tensor_hashes"),
        "annotation_schema": raw.get("annotation_schema"),
    })).hexdigest()


def _annotation_mapping(
    value: Any,
    canonical: Mapping[str, int],
    label: str,
) -> Mapping[str, int]:
    raw = _mapping(value, label)
    result: dict[str, int] = {}
    for name, code in raw.items():
        semantic = _string(name, f"{label} semantic name")
        if semantic not in canonical:
            raise RootedStarStoreError(f"{label} has an unsupported semantic value")
        result[semantic] = _integer(code, f"{label} local code")
    if not result or len(set(result.values())) != len(result):
        raise RootedStarStoreError(f"{label} must be a non-empty local bijection")
    return MappingProxyType(result)


def _validate_view(*, family: str, tensor_path: str, tensor_sha: str, row: Mapping[str, Any], reference: RelationalPrefixReference, raw: Mapping[str, Any], shard_keys: set[str]) -> RootedStarReference:
    if reference.family != family or reference.conversation_id != _string(row.get("conversation_id"), "row conversation"):
        raise RootedStarStoreError("view reference does not bind its family row")
    if reference.source_row_sha256 != _sha(row.get("source_row_sha256"), "row source SHA") or reference.source_tensor_sha256 != _sha(row.get("source_tensor_sha256"), "row source tensor SHA"):
        raise RootedStarStoreError("view reference does not bind its source pins")
    view = raw.get("view")
    if view not in VIEWS:
        raise RootedStarStoreError("rooted-star view is not frozen")
    hashes = _mapping(raw.get("tensor_hashes"), "view tensor hashes")
    keys = _mapping(raw.get("tensor_keys"), "view tensor keys")
    schema = _mapping(raw.get("annotation_schema"), "annotation schema")
    if set(hashes) != set(_TENSOR_NAMES) or set(keys) != set(_TENSOR_NAMES) or set(schema) != {"origin_id_mapping", "origin_detail_id_mapping"}:
        raise RootedStarStoreError("view tensor/hash/schema keys are not exact")
    locators = {name: _string(keys[name], f"view tensor key {name}") for name in _TENSOR_NAMES}
    if len(set(locators.values())) != len(locators) or not set(locators.values()).issubset(shard_keys):
        raise RootedStarStoreError("view tensor locators are unresolved or aliased")
    bound_hashes = {name: _sha(hashes[name], f"view tensor hash {name}") for name in _TENSOR_NAMES}
    origin_mapping = _annotation_mapping(
        schema["origin_id_mapping"],
        _CANONICAL_ORIGIN_IDS,
        "origin ID mapping",
    )
    detail_mapping = _annotation_mapping(
        schema["origin_detail_id_mapping"],
        _CANONICAL_ORIGIN_DETAIL_IDS,
        "origin-detail ID mapping",
    )
    if _sha(raw.get("star_sha256"), "star SHA") != _star_hash(reference, raw):
        raise RootedStarStoreError("rooted-star payload hash is invalid")
    return RootedStarReference(
        rooted_star_id=_physical_id(reference, view, raw["star_sha256"]), reference_id=reference.reference_id,
        field_event_id=reference.field_event_id, occurrence_id=reference.occurrence_id, view=view,
        star_sha256=raw["star_sha256"], geometry_sha256=_geometry_hash(raw), source_reference=reference,
        tensor_locators=RootedStarTensorLocators(tensor_path, tensor_sha, MappingProxyType(locators)), tensor_hashes=MappingProxyType(bound_hashes),
        origin_id_mapping=origin_mapping,
        origin_detail_id_mapping=detail_mapping,
    )


def build_relational_pre_status_rooted_star_index(root: Path) -> RelationalPreStatusRootedStarIndex:
    """Validate a completed materialization and expose physical and geometry units.

    ``references`` preserves every physical occurrence/view.  ``geometry_references``
    contains one deterministic representative per exact view payload group (the
    producer's ``star_sha256`` when physical identity is shared, otherwise its
    reference-free activation-equivalent payload), so a repeated rooted state
    contributes exactly one geometry vote.
    """
    artifact_root = Path(root).resolve()
    manifest_path = artifact_root / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise RootedStarStoreError("bank manifest.json is missing")
    manifest = _read_json(manifest_path, "bank manifest")
    _sealed(manifest, "manifest", allow_manifest_status=True)
    if manifest.get("schema_version") != _SCHEMA_VERSION or manifest.get("kind") != _MANIFEST_KIND or manifest.get("status") != "success":
        raise RootedStarStoreError("bank manifest is not a successful schema-1 rooted-star materialization")
    input_identity = _sha(manifest.get("input_identity_sha256"), "input identity")
    source = _mapping(manifest.get("source"), "manifest source pins")
    for key in ("state_graph_manifest_sha256", "state_graph_geometry_sha256", "state_graph_activation_sha256", "capture_manifest_sha256", "capture_inventory_sha256", "action_exclusion_protocol_sha256"):
        _sha(source.get(key), f"source pin {key}")
    if not isinstance(source.get("tensor_source"), Mapping):
        raise RootedStarStoreError("manifest tensor source pin is absent")
    families = _mapping(manifest.get("families"), "bank families")
    if not families:
        raise RootedStarStoreError("bank has no families")
    references: list[RootedStarReference] = []
    total_rows = total_occurrences = 0
    for family in sorted(families):
        name = _string(family, "family name")
        entry = _mapping(families[family], f"family {name}")
        rows_expected = _integer(entry.get("expected_row_count"), "expected row count", minimum=1)
        if entry.get("finalized") is not True or _integer(entry.get("completed_row_count"), "completed row count") != rows_expected or _integer(entry.get("occurrence_count"), "occurrence count") != rows_expected * 4:
            raise RootedStarStoreError(f"family {name} is incomplete")
        tensor = _safe_child(artifact_root, entry.get("tensor_path"), f"family {name} tensor path")
        records = _safe_child(artifact_root, entry.get("stars_jsonl_path"), f"family {name} JSONL path")
        tensor_sha = _sha(entry.get("tensor_sha256"), "family tensor SHA")
        if _integer(entry.get("tensor_bytes"), "family tensor bytes", minimum=1) != tensor.stat().st_size or file_sha256(tensor) != tensor_sha:
            raise RootedStarStoreError(f"family {name} tensor integrity mismatch")
        if _integer(entry.get("stars_jsonl_bytes"), "family JSONL bytes", minimum=1) != records.stat().st_size or file_sha256(records) != _sha(entry.get("stars_jsonl_sha256"), "family JSONL SHA"):
            raise RootedStarStoreError(f"family {name} JSONL integrity mismatch")
        with safe_open(tensor, framework="pt", device="cpu") as handle:
            _metadata(handle, family=name, input_identity=input_identity)
            shard_keys = set(handle.keys())
            if len(shard_keys) != _integer(entry.get("tensor_key_count"), "family tensor key count", minimum=1):
                raise RootedStarStoreError(f"family {name} tensor key count mismatch")
            rows = list(_iter_jsonl(records))
            if len(rows) != rows_expected:
                raise RootedStarStoreError(f"family {name} JSONL row count mismatch")
            row_ids: set[int] = set()
            row_hash_items: list[dict[str, str]] = []
            resolved_family_keys: set[str] = set()
            for row in rows:
                _sealed(row, f"family[{name}]")
                if row.get("schema_version") != _SCHEMA_VERSION or row.get("kind") != _ROW_KIND or row.get("family") != name:
                    raise RootedStarStoreError(f"family {name} has invalid row schema")
                row_index = _integer(row.get("row_index"), "row index")
                if row_index in row_ids:
                    raise RootedStarStoreError(f"family {name} has duplicate row index")
                row_ids.add(row_index)
                row_keys = row.get("tensor_keys")
                if not isinstance(row_keys, list) or len(row_keys) != len(set(row_keys)) or not set(row_keys).issubset(shard_keys):
                    raise RootedStarStoreError(f"family {name} row tensor keys are invalid")
                stars = row.get("stars")
                if not isinstance(stars, list) or len(stars) != 4 or row.get("reference_ids") != [star.get("reference", {}).get("reference_id") for star in stars]:
                    raise RootedStarStoreError(f"family {name} has invalid rooted-star rows")
                expected_row_sha = hashlib.sha256(_canonical({"conversation_id": row.get("conversation_id"), "source_tensor_sha256": row.get("source_tensor_sha256"), "stars": [{"reference_id": star["reference"].get("reference_id"), "view_hashes": [view.get("star_sha256") for view in star.get("views", [])]} for star in stars]})).hexdigest()
                if _sha(row.get("star_row_sha256"), "star row SHA") != expected_row_sha:
                    raise RootedStarStoreError(f"family {name} star row hash is invalid")
                turns: set[int] = set()
                resolved_row_keys: set[str] = set()
                for star in stars:
                    views = star.get("views")
                    if not isinstance(views, list) or len(views) != 2:
                        raise RootedStarStoreError("status occurrence must provide exactly two frozen views")
                    source_reference = _reference_from_dict(_mapping(star.get("reference"), "star reference"))
                    records_for_star = [_validate_view(family=name, tensor_path=str(tensor.relative_to(artifact_root)), tensor_sha=tensor_sha, row=row, reference=source_reference, raw=_mapping(view, "rooted-star view"), shard_keys=shard_keys) for view in views]
                    if {item.view for item in records_for_star} != set(VIEWS):
                        raise RootedStarStoreError("status occurrence has duplicate or missing frozen views")
                    # Hash each independently addressable payload now. This never loads
                    # the whole family bank, while making the returned index immutable.
                    for record in records_for_star:
                        _load_tensors(handle, record, _TENSOR_NAMES)
                        resolved_row_keys.update(record.tensor_locators.tensor_keys.values())
                    turns.add(records_for_star[0].source_reference.turn_index)
                    references.extend(records_for_star)
                if set(row_keys) != resolved_row_keys:
                    raise RootedStarStoreError(f"family {name} row tensor key inventory is not exact")
                resolved_family_keys.update(resolved_row_keys)
                if turns != {0, 1, 2, 3}:
                    raise RootedStarStoreError(f"family {name} row does not cover turns 0..3")
                row_hash_items.append({"conversation_id": _string(row.get("conversation_id"), "row conversation"), "sha256": row["star_row_sha256"]})
            if _sha(entry.get("star_rows_sha256"), "family star rows SHA") != hashlib.sha256(_canonical(sorted(row_hash_items, key=lambda item: item["conversation_id"]))).hexdigest():
                raise RootedStarStoreError(f"family {name} star-row inventory hash is invalid")
            if shard_keys != resolved_family_keys:
                raise RootedStarStoreError(f"family {name} shard tensor inventory is not exact")
        total_rows += rows_expected
        total_occurrences += rows_expected * 4
    if _integer(manifest.get("row_count"), "manifest row count") != total_rows or _integer(manifest.get("occurrence_count"), "manifest occurrence count") != total_occurrences or _integer(manifest.get("view_count"), "manifest view count") != total_occurrences * 2:
        raise RootedStarStoreError("manifest totals do not match finalized families")
    by_physical: dict[str, RootedStarReference] = {}
    groups: dict[str, list[RootedStarReference]] = defaultdict(list)
    mappings: tuple[dict[str, list[RootedStarReference]], ...] = (defaultdict(list), defaultdict(list), defaultdict(list))
    for reference in sorted(references, key=lambda item: item.rooted_star_id):
        if reference.rooted_star_id in by_physical:
            raise RootedStarStoreError("duplicate physical rooted-star ID")
        by_physical[reference.rooted_star_id] = reference
        groups[reference.geometry_sha256].append(reference)
        mappings[0][reference.reference_id].append(reference)
        mappings[1][reference.field_event_id].append(reference)
        mappings[2][reference.occurrence_id].append(reference)
    exact_groups = MappingProxyType({key: tuple(sorted(value, key=lambda item: item.rooted_star_id)) for key, value in sorted(groups.items())})
    return RelationalPreStatusRootedStarIndex(
        artifact_root=artifact_root, manifest_sha256=file_sha256(manifest_path), input_identity_sha256=input_identity,
        references=tuple(sorted(references, key=lambda item: item.rooted_star_id)), geometry_references=tuple(group[0] for _, group in sorted(exact_groups.items())),
        exact_view_groups=exact_groups, by_rooted_star_id=MappingProxyType(by_physical),
        by_reference_id=MappingProxyType({key: tuple(value) for key, value in sorted(mappings[0].items())}), by_field_event_id=MappingProxyType({key: tuple(value) for key, value in sorted(mappings[1].items())}), by_occurrence_id=MappingProxyType({key: tuple(value) for key, value in sorted(mappings[2].items())}),
    )


def _load_tensors(handle: Any, reference: RootedStarReference, names: Sequence[str]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    keys = set(handle.keys())
    for name in names:
        key = reference.tensor_locators.tensor_keys.get(name)
        if key not in keys:
            raise RootedStarStoreError(f"rooted star {reference.rooted_star_id} tensor {name} is absent")
        tensor = handle.get_tensor(key)
        if tensor_sha256(name, tensor) != reference.tensor_hashes[name]:
            raise RootedStarStoreError(f"rooted star {reference.rooted_star_id} tensor {name} hash mismatch")
        tensors[name] = tensor
    return tensors


def _load_metric_from_handle(handle: Any, reference: RootedStarReference) -> RootedStarMetricInput:
    tensors = _load_tensors(handle, reference, _METRIC_NAMES)
    residual = tensors["root_to_context_residual_distances"].detach().cpu().to(torch.float32).numpy()
    attention = tensors["incoming_attention"].detach().cpu().to(torch.float32).numpy()
    origin = _canonical_annotation_codes(
        tensors["origin_ids"],
        reference.origin_id_mapping,
        _CANONICAL_ORIGIN_IDS,
    )
    origin_detail = _canonical_annotation_codes(
        tensors["origin_detail_ids"],
        reference.origin_detail_id_mapping,
        _CANONICAL_ORIGIN_DETAIL_IDS,
    )
    annotations = torch.column_stack(
        [
            tensors[name]
            for name in (
                "normalized_positions",
                "role_ids",
                "turn_ids",
                "message_ids",
                "span_flags",
            )
        ]
        + [origin, origin_detail]
    ).detach().cpu().to(torch.float32).numpy()
    return RootedStarMetricInput(np.array(residual, copy=True), np.array(attention, copy=True), np.array(annotations, copy=True))


def _canonical_annotation_codes(
    tensor: torch.Tensor,
    local_mapping: Mapping[str, int],
    canonical_mapping: Mapping[str, int],
) -> torch.Tensor:
    inverse = {code: name for name, code in local_mapping.items()}
    values = tensor.detach().cpu().to(torch.int64).tolist()
    try:
        canonical = [canonical_mapping[inverse[int(value)]] for value in values]
    except KeyError as error:
        raise RootedStarStoreError(
            "annotation tensor contains a code absent from its semantic mapping"
        ) from error
    return torch.tensor(canonical, dtype=torch.int16)


def _member(index: RelationalPreStatusRootedStarIndex, reference: RootedStarReference) -> None:
    if index.by_rooted_star_id.get(reference.rooted_star_id) is not reference:
        raise RootedStarStoreError("reference does not belong to this rooted-star index")


def _path(index: RelationalPreStatusRootedStarIndex, reference: RootedStarReference) -> Path:
    path = _safe_child(index.artifact_root, reference.tensor_locators.family_shard_path, "family shard path")
    if file_sha256(path) != reference.tensor_locators.family_shard_sha256:
        raise RootedStarStoreError("family shard changed after indexing")
    return path


def load_rooted_star_metric_input(index: RelationalPreStatusRootedStarIndex, reference: RootedStarReference) -> RootedStarMetricInput:
    """Load one metric-only view, intentionally excluding raw token IDs."""
    _member(index, reference)
    with safe_open(_path(index, reference), framework="pt", device="cpu") as handle:
        _metadata(handle, family=reference.family, input_identity=index.input_identity_sha256)
        return _load_metric_from_handle(handle, reference)


def load_rooted_star_root_residuals(index: RelationalPreStatusRootedStarIndex, reference: RootedStarReference) -> torch.Tensor:
    """Load the exact `[4, hidden]` root residual lift target as detached float32."""
    _member(index, reference)
    with safe_open(_path(index, reference), framework="pt", device="cpu") as handle:
        _metadata(handle, family=reference.family, input_identity=index.input_identity_sha256)
        tensor = _load_tensors(handle, reference, ("root_residuals",))["root_residuals"]
    if tensor.ndim != 2 or tuple(tensor.shape[:1]) != (len(LAYERS),) or tensor.shape[1] < 1 or tensor.dtype != torch.bfloat16 or not torch.isfinite(tensor).all().item():
        raise RootedStarStoreError("root residuals must be finite BF16 [4, hidden]")
    return tensor.detach().cpu().to(torch.float32).clone()


def load_rooted_star_observation_binding(
    index: RelationalPreStatusRootedStarIndex,
    reference: RootedStarReference,
) -> RootedStarObservationBinding:
    """Load only outcome-free tensors needed to bind a live pre-status replay."""
    _member(index, reference)
    with safe_open(_path(index, reference), framework="pt", device="cpu") as handle:
        _metadata(handle, family=reference.family, input_identity=index.input_identity_sha256)
        tensors = _load_tensors(
            handle,
            reference,
            ("retained_token_indices", "root_residuals", "incoming_attention"),
        )
    return RootedStarObservationBinding(
        rooted_star_id=reference.rooted_star_id,
        reference_id=reference.reference_id,
        view=reference.view,
        geometry_sha256=reference.geometry_sha256,
        prefix_token_count=reference.source_reference.prefix_token_stop,
        retained_token_indices=tensors["retained_token_indices"]
        .detach()
        .cpu()
        .to(torch.int64)
        .numpy()
        .copy(),
        root_residuals=tensors["root_residuals"]
        .detach()
        .cpu()
        .to(torch.float32)
        .numpy()
        .copy(),
        incoming_attention=tensors["incoming_attention"]
        .detach()
        .cpu()
        .to(torch.float32)
        .numpy()
        .copy(),
    )


__all__ = [
    "RelationalPreStatusRootedStarIndex", "RootedStarObservationBinding", "RootedStarReference", "RootedStarStoreError", "RootedStarTensorLocators",
    "build_relational_pre_status_rooted_star_index", "load_rooted_star_metric_input", "load_rooted_star_observation_binding", "load_rooted_star_root_residuals",
]
