"""Fold-safe immutable causal H/T/S/G vector banks for masked pre-status roots."""

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

from geoprobe.data.relational_pre_status_rooted_star import LAYERS, VIEWS
from geoprobe.data.relational_pre_status_rooted_star_store import (
    RelationalPreStatusRootedStarIndex,
    load_rooted_star_root_residuals,
)
from geoprobe.eval.relational_pre_status_prediction_seal import _root_balanced
from geoprobe.eval.relational_pre_status_supervision import (
    LabelFreePrefixStateQuotient,
    PreStatusQuotientNode,
    StatusEventOutcome,
    _crossings,
    build_label_free_prefix_state_quotient,
)
from geoprobe.geometry.relational_pre_status_honestward import (
    HonestwardCrossingObservation,
    SharedPreStatusHonestwardField,
)
from geoprobe.geometry.relational_pre_status_rooted_graph import (
    FOLDS,
    FoldExactRootedGraph,
)
from geoprobe.io import file_sha256


MASKED_VIEW: Final = VIEWS[1]
VECTOR_NAMES: Final = ("h", "t", "s", "g")
TENSOR_FILE_NAME: Final = "vectors.safetensors"
LEDGER_FILE_NAME: Final = "ledger.json"
LEDGER_KIND: Final = "relational_pre_status_causal_vector_bank"
_TENSOR_KIND: Final = "relational_pre_status_causal_vector_bank_tensor"
_ARRAY_DOMAIN: Final = b"geoprobe.pre-status-causal-vector-bank.array.v1\x00"


class RelationalPreStatusVectorBankError(ValueError):
    """A causal vector bank violates its leakage, geometry, or seal contract."""


@dataclass(frozen=True, slots=True)
class CausalVectorBankArrays:
    """Dense [root, layer, hidden] vectors and per-vector defined flags."""

    h: np.ndarray
    t: np.ndarray
    s: np.ndarray
    g: np.ndarray
    defined: np.ndarray


@dataclass(frozen=True, slots=True)
class CausalVectorBankBuild:
    """One held-out fold build, ready for immutable persistence."""

    held_out_family_fold: str
    arrays: CausalVectorBankArrays
    rows: tuple[Mapping[str, Any], ...]
    ledger_payload: Mapping[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RelationalPreStatusVectorBankError("value is not canonical JSON") from error
    return sha256(payload).hexdigest()


def _array_sha256(name: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    if not name or array.dtype.hasobject:
        raise RelationalPreStatusVectorBankError("array identity is invalid")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise RelationalPreStatusVectorBankError("array contains non-finite values")
    digest = sha256(_ARRAY_DOMAIN)
    digest.update(name.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\x00")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\x00")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _readonly(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy()
    result.flags.writeable = False
    return result


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationalPreStatusVectorBankError(f"{label} must be a non-empty string")
    return value


def _sha(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise RelationalPreStatusVectorBankError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelationalPreStatusVectorBankError(f"{label} must be an object")
    return value


def _endpoint(value: object, label: str) -> Mapping[str, str]:
    row = _mapping(value, label)
    result = {
        "event_id": _string(row.get("event_id"), f"{label} event ID"),
        "family": _string(row.get("family"), f"{label} family"),
        "family_fold": _string(row.get("family_fold"), f"{label} family fold"),
        "prefix_state_sha256": _sha(row.get("prefix_state_sha256"), f"{label} prefix-state SHA-256"),
    }
    if result["family_fold"] not in FOLDS:
        raise RelationalPreStatusVectorBankError(f"{label} has an unsupported family fold")
    return MappingProxyType(result)


def _checked_roster_edges(edges: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    if not edges:
        raise RelationalPreStatusVectorBankError("full roster edge inventory is empty")
    result: list[Mapping[str, Any]] = []
    pair_ids: set[str] = set()
    for raw in edges:
        row = _mapping(raw, "roster edge")
        pair_id = _string(row.get("pair_id"), "roster pair ID")
        if pair_id in pair_ids:
            raise RelationalPreStatusVectorBankError("roster pair IDs are not unique")
        pair_ids.add(pair_id)
        source = _endpoint(row.get("source"), "roster source")
        target = _endpoint(row.get("target"), "roster target")
        if (source["family"], source["family_fold"]) != (target["family"], target["family_fold"]):
            raise RelationalPreStatusVectorBankError("roster edge crosses family or fold")
        result.append(MappingProxyType({
            "pair_id": pair_id,
            "contrast_id": _string(row.get("contrast_id"), "roster contrast ID"),
            "scenario_id": _string(row.get("scenario_id"), "roster scenario ID"),
            "source": source,
            "target": target,
        }))
    return tuple(result)


def _validate_training_outcomes(
    outcomes: Mapping[str, StatusEventOutcome],
    quotient: LabelFreePrefixStateQuotient,
    *,
    held_out_family_fold: str,
) -> None:
    expected = {
        event_id for event_id, view_nodes in quotient.event_to_node_ids.items()
        if next(node for node in quotient.nodes if node.node_id == view_nodes[MASKED_VIEW]).family_fold != held_out_family_fold
    }
    if set(outcomes) != expected:
        raise RelationalPreStatusVectorBankError(
            "training outcomes must exactly cover non-heldout quotient events and no heldout outcomes"
        )
    for event_id, outcome in outcomes.items():
        if not isinstance(outcome, StatusEventOutcome):
            raise RelationalPreStatusVectorBankError("training outcomes must contain StatusEventOutcome values")
        node = next(
            node for node in quotient.nodes
            if node.node_id == quotient.event_to_node_ids[event_id][MASKED_VIEW]
        )
        if (outcome.family, outcome.family_fold, outcome.turn_index, outcome.prefix_state_sha256) != (
            node.family, node.family_fold, node.turn_index, node.prefix_state_sha256,
        ):
            raise RelationalPreStatusVectorBankError("training outcome disagrees with its label-free quotient node")


def _node_token_hash(node: PreStatusQuotientNode) -> str:
    hashes = {reference.source_reference.prefix_token_ids_sha256 for reference in node.representative_references}
    if len(hashes) != 1:
        raise RelationalPreStatusVectorBankError("quotient node has non-unique prefix-token identities")
    return _sha(next(iter(hashes)), "node prefix-token SHA-256")


def _node_residual(
    index: RelationalPreStatusRootedStarIndex,
    node: PreStatusQuotientNode,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    cached = cache.get(node.node_id)
    if cached is not None:
        return cached
    values = [load_rooted_star_root_residuals(index, ref).detach().cpu().numpy() for ref in node.representative_references]
    if not values or any(value.shape != values[0].shape or not np.isfinite(value).all() for value in values):
        raise RelationalPreStatusVectorBankError("node root residuals are inconsistent")
    result = np.asarray(np.mean(np.stack(values, axis=0, dtype=np.float64), axis=0), dtype=np.float32)
    if result.shape[0] != len(LAYERS) or result.shape[1] < 1:
        raise RelationalPreStatusVectorBankError("node root residual has invalid [layer, hidden] shape")
    cache[node.node_id] = result
    return result


def _generic_observations(
    *,
    index: RelationalPreStatusRootedStarIndex,
    nodes_by_id: Mapping[str, PreStatusQuotientNode],
    event_to_nodes: Mapping[str, Mapping[str, str]],
    outcomes: Mapping[str, StatusEventOutcome],
    training_edges: Sequence[Mapping[str, Any]],
) -> tuple[HonestwardCrossingObservation, ...]:
    cache: dict[str, np.ndarray] = {}
    rows: list[HonestwardCrossingObservation] = []
    for edge in training_edges:
        source, target = edge["source"], edge["target"]
        left, right = outcomes[source["event_id"]], outcomes[target["event_id"]]
        if left.true_status != right.true_status:
            continue
        source_node = nodes_by_id[event_to_nodes[source["event_id"]][MASKED_VIEW]]
        target_node = nodes_by_id[event_to_nodes[target["event_id"]][MASKED_VIEW]]
        for origin, destination, origin_node, destination_node in (
            (source, target, source_node, target_node),
            (target, source, target_node, source_node),
        ):
            rows.append(HonestwardCrossingObservation(
                pair_id=f"generic:{edge['pair_id']}:{origin_node.node_id}->{destination_node.node_id}",
                deceptive_root_id=origin_node.node_id,
                honest_root_id=destination_node.node_id,
                family=origin_node.family,
                family_fold=origin_node.family_fold,
                scenario_id=edge["scenario_id"],
                contrast_id=edge["contrast_id"],
                true_status=outcomes[origin["event_id"]].true_status,
                delta=_node_residual(index, destination_node, cache) - _node_residual(index, origin_node, cache),
            ))
    return tuple(rows)


def _fit(
    rows: Sequence[HonestwardCrossingObservation], *, fold: str, graph: FoldExactRootedGraph, label: str,
) -> SharedPreStatusHonestwardField:
    balanced = _root_balanced(rows)
    if not balanced:
        raise RelationalPreStatusVectorBankError(f"{label} has no training observations")
    return SharedPreStatusHonestwardField.fit(
        balanced, held_out_family_fold=fold, training_edges=graph.training_edges,
    )


def _validate_arrays(arrays: CausalVectorBankArrays, *, row_count: int | None = None) -> None:
    values = (arrays.h, arrays.t, arrays.s, arrays.g)
    if any(value.dtype != np.float32 or value.ndim != 3 or value.shape[1] != len(LAYERS) or value.shape[2] < 1 or not np.isfinite(value).all() for value in values):
        raise RelationalPreStatusVectorBankError("vectors must be finite float32 [root, 4, hidden]")
    if len({value.shape for value in values}) != 1:
        raise RelationalPreStatusVectorBankError("H/T/S/G shapes disagree")
    if arrays.defined.dtype != np.bool_ or arrays.defined.shape != (arrays.h.shape[0], len(VECTOR_NAMES)):
        raise RelationalPreStatusVectorBankError("defined flags must be bool [root, 4]")
    if row_count is not None and arrays.h.shape[0] != row_count:
        raise RelationalPreStatusVectorBankError("vector rows do not match ledger rows")
    if not np.allclose(arrays.h, arrays.t + arrays.s, atol=2e-6, rtol=2e-6):
        raise RelationalPreStatusVectorBankError("H != T + S")


def build_pre_status_causal_vector_bank(
    index: RelationalPreStatusRootedStarIndex,
    *,
    held_out_family_fold: str,
    training_outcomes: Mapping[str, StatusEventOutcome],
    roster_edges: Sequence[Mapping[str, Any]],
    graph: FoldExactRootedGraph,
    artifact_bindings: Mapping[str, Any] | None = None,
) -> CausalVectorBankBuild:
    """Build masked H/T/S/G vectors without opening or accepting held-out labels."""
    fold = _string(held_out_family_fold, "held-out family fold")
    if fold not in FOLDS or graph.held_out_family_fold != fold:
        raise RelationalPreStatusVectorBankError("held-out fold does not bind the graph")
    quotient = build_label_free_prefix_state_quotient(index)
    nodes_by_id = MappingProxyType({node.node_id: node for node in quotient.nodes})
    _validate_training_outcomes(training_outcomes, quotient, held_out_family_fold=fold)
    edges = _checked_roster_edges(roster_edges)
    training_edges = tuple(edge for edge in edges if edge["source"]["family_fold"] != fold)
    heldout_edges = tuple(edge for edge in edges if edge["source"]["family_fold"] == fold)
    if not training_edges or not heldout_edges:
        raise RelationalPreStatusVectorBankError("roster must contain training and held-out fold edges")
    if any(edge["target"]["family_fold"] == fold for edge in training_edges) or any(edge["target"]["family_fold"] != fold for edge in heldout_edges):
        raise RelationalPreStatusVectorBankError("roster fold partition is invalid")
    h_raw = _crossings(index, training_outcomes, quotient.event_to_node_ids, nodes_by_id, training_edges)[MASKED_VIEW]
    h_field = _fit(h_raw, fold=fold, graph=graph, label="H")
    t_field = _fit(
        _generic_observations(index=index, nodes_by_id=nodes_by_id, event_to_nodes=quotient.event_to_node_ids, outcomes=training_outcomes, training_edges=training_edges),
        fold=fold, graph=graph, label="T",
    )
    if h_field.shape != t_field.shape or h_field.shape[0] != len(LAYERS):
        raise RelationalPreStatusVectorBankError("H and T fields do not share the four-layer residual shape")
    candidate_ids = sorted({
        quotient.event_to_node_ids[edge[side]["event_id"]][MASKED_VIEW]
        for edge in heldout_edges for side in ("source", "target")
    })
    if not candidate_ids:
        raise RelationalPreStatusVectorBankError("held-out roster has no masked candidate roots")
    rows: list[Mapping[str, Any]] = []
    h_values: list[np.ndarray] = []
    t_values: list[np.ndarray] = []
    s_values: list[np.ndarray] = []
    g_values: list[np.ndarray] = []
    flags: list[tuple[bool, bool, bool, bool]] = []
    for root_id in candidate_ids:
        node = nodes_by_id[root_id]
        if node.view != MASKED_VIEW or node.family_fold != fold:
            raise RelationalPreStatusVectorBankError("candidate root is not a held-out masked node")
        edges_for_root = graph.query_edges.get(root_id)
        if edges_for_root is None:
            raise RelationalPreStatusVectorBankError("candidate root is absent from held-out query graph")
        h_prediction, t_prediction = h_field.predict(root_id, edges_for_root), t_field.predict(root_id, edges_for_root)
        h, t, g = h_prediction.dose_calibrated_local, t_prediction.dose_calibrated_local, h_prediction.global_mean
        s = h - t
        event_ids = tuple(sorted(node.event_ids))
        token_hash = _node_token_hash(node)
        # Every event collapsed into this root must bind the same replay token prefix.
        references = [ref for ref in index.references if ref.field_event_id in set(event_ids) and ref.view == MASKED_VIEW]
        if not references or {ref.source_reference.prefix_token_ids_sha256 for ref in references} != {token_hash}:
            raise RelationalPreStatusVectorBankError("node events do not agree on a replay prefix-token hash")
        row_index = len(rows)
        rows.append(MappingProxyType({
            "tensor_row_index": row_index, "root_id": root_id, "event_ids": list(event_ids),
            "prefix_state_sha256": node.prefix_state_sha256, "prefix_token_ids_sha256": token_hash,
            "family": node.family, "family_fold": node.family_fold, "turn_index": node.turn_index,
            "view": MASKED_VIEW,
            "support": {
                "h": {"root_ids": list(h_prediction.support_root_ids), "pair_ids": list(h_prediction.support_pair_ids), "count": h_prediction.support_count},
                "t": {"root_ids": list(t_prediction.support_root_ids), "pair_ids": list(t_prediction.support_pair_ids), "count": t_prediction.support_count},
            },
            "dose_by_layer": {"h": h_prediction.scalar_dose_by_layer.tolist(), "t": t_prediction.scalar_dose_by_layer.tolist()},
        }))
        h_values.append(h)
        t_values.append(t)
        s_values.append(s)
        g_values.append(g)
        flags.append((h_prediction.defined, t_prediction.defined, h_prediction.defined and t_prediction.defined, True))
    arrays = CausalVectorBankArrays(
        h=_readonly(np.stack(h_values), dtype=np.float32), t=_readonly(np.stack(t_values), dtype=np.float32),
        s=_readonly(np.stack(s_values), dtype=np.float32), g=_readonly(np.stack(g_values), dtype=np.float32),
        defined=_readonly(np.asarray(flags, dtype=np.bool_), dtype=np.bool_),
    )
    _validate_arrays(arrays, row_count=len(rows))
    payload: dict[str, Any] = {
        "schema_version": 1, "kind": LEDGER_KIND, "held_out_family_fold": fold,
        "view": MASKED_VIEW, "layers": list(LAYERS), "vector_order": list(VECTOR_NAMES),
        "training_outcome_event_count": len(training_outcomes), "training_roster_pair_count": len(training_edges),
        "heldout_roster_pair_count": len(heldout_edges), "candidate_root_count": len(rows),
        "candidate_event_count": len({event_id for row in rows for event_id in row["event_ids"]}),
        "contract": {"outcome_access": "caller_supplied_exactly_four_nonheldout_folds", "heldout_action_selection": "not_used", "generic_transport": "bidirectional_same_true_status_training_roster_edges", "root_balancing": "per_source_root_before_fit"},
        "artifact_bindings": _jsonable(artifact_bindings or {}), "rows": [_jsonable(row) for row in rows],
        "dose_by_layer": {
            "h": h_field.scalar_dose_by_layer.tolist(),
            "t": t_field.scalar_dose_by_layer.tolist(),
        },
        "array_semantic_sha256": {name: _array_sha256(name, value) for name, value in zip(VECTOR_NAMES, (arrays.h, arrays.t, arrays.s, arrays.g), strict=True)},
        "defined_semantic_sha256": _array_sha256("defined", arrays.defined.astype(np.uint8)),
    }
    payload["build_payload_sha256"] = _canonical_sha256(payload)
    return CausalVectorBankBuild(fold, arrays, tuple(rows), MappingProxyType(payload))


def _write_safetensors_new(path: Path, arrays: CausalVectorBankArrays, metadata: Mapping[str, str]) -> None:
    payloads = [("defined", np.ascontiguousarray(arrays.defined, dtype=np.uint8)), *[(name, np.ascontiguousarray(value, dtype="<f4")) for name, value in zip(VECTOR_NAMES, (arrays.h, arrays.t, arrays.s, arrays.g), strict=True)]]
    offset = 0
    header: dict[str, Any] = {"__metadata__": dict(sorted(metadata.items()))}
    raw: list[bytes] = []
    for name, array in payloads:
        data = array.tobytes(order="C")
        raw.append(data)
        header[name] = {"dtype": "U8" if name == "defined" else "F32", "shape": list(array.shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with path.open("xb") as handle:
        handle.write(len(encoded).to_bytes(8, "little"))
        handle.write(encoded)
        for data in raw:
            handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _ledger_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("ledger_sha256", None)
    return _canonical_sha256(payload)


def save_pre_status_causal_vector_bank(build: CausalVectorBankBuild, out_dir: Path) -> Mapping[str, Any]:
    """Persist an immutable deterministic vectors.safetensors plus self-hashed ledger."""
    _validate_arrays(build.arrays, row_count=len(build.rows))
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tensor_path, ledger_path = root / TENSOR_FILE_NAME, root / LEDGER_FILE_NAME
    if tensor_path.exists() or ledger_path.exists():
        raise RelationalPreStatusVectorBankError("refusing to overwrite a vector-bank artifact")
    metadata = {"schema_version": "1", "kind": _TENSOR_KIND, "held_out_family_fold": build.held_out_family_fold, "view": MASKED_VIEW, "layers": json.dumps(list(LAYERS), separators=(",", ":")), "root_count": str(len(build.rows))}
    with tempfile.NamedTemporaryFile(dir=root, suffix=".safetensors.tmp", delete=False) as handle:
        temporary = Path(handle.name)
    temporary.unlink()
    try:
        _write_safetensors_new(temporary, build.arrays, metadata)
        os.link(temporary, tensor_path)
    finally:
        temporary.unlink(missing_ok=True)
    ledger = dict(build.ledger_payload)
    ledger["tensor_artifact"] = {"path": TENSOR_FILE_NAME, "bytes": tensor_path.stat().st_size, "sha256": file_sha256(tensor_path), "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "semantic_sha256": _array_sha256(name, value)} for name, value in zip(VECTOR_NAMES, (build.arrays.h, build.arrays.t, build.arrays.s, build.arrays.g), strict=True)}, "defined": {"shape": list(build.arrays.defined.shape), "dtype": "bool", "semantic_sha256": _array_sha256("defined", build.arrays.defined.astype(np.uint8))}}
    ledger["ledger_sha256"] = _ledger_sha256(ledger)
    _write_json_new(ledger_path, ledger)
    return MappingProxyType(ledger)


def load_pre_status_causal_vector_bank(root: Path) -> tuple[Mapping[str, Any], CausalVectorBankArrays]:
    """Load and validate physical, semantic, mapping, and H=T+S bindings."""
    artifact_root = Path(root).resolve()
    try:
        ledger = json.loads((artifact_root / LEDGER_FILE_NAME).read_text(encoding="utf-8"), parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RelationalPreStatusVectorBankError("vector-bank ledger is not finite UTF-8 JSON") from error
    ledger = _mapping(ledger, "vector-bank ledger")
    if ledger.get("schema_version") != 1 or ledger.get("kind") != LEDGER_KIND or ledger.get("ledger_sha256") != _ledger_sha256(ledger):
        raise RelationalPreStatusVectorBankError("vector-bank ledger schema or self-hash is invalid")
    build_payload = dict(ledger)
    build_payload.pop("tensor_artifact", None)
    build_payload.pop("ledger_sha256", None)
    recorded_build_sha = build_payload.pop("build_payload_sha256", None)
    if recorded_build_sha != _canonical_sha256(build_payload):
        raise RelationalPreStatusVectorBankError("vector-bank build-payload hash is invalid")
    if ledger.get("view") != MASKED_VIEW or ledger.get("layers") != list(LAYERS) or ledger.get("vector_order") != list(VECTOR_NAMES):
        raise RelationalPreStatusVectorBankError("vector-bank frozen geometry order is invalid")
    tensor = _mapping(ledger.get("tensor_artifact"), "tensor artifact")
    if tensor.get("path") != TENSOR_FILE_NAME:
        raise RelationalPreStatusVectorBankError("vector-bank tensor filename is invalid")
    path = (artifact_root / TENSOR_FILE_NAME).resolve()
    if not path.is_relative_to(artifact_root) or not path.is_file() or tensor.get("bytes") != path.stat().st_size or tensor.get("sha256") != file_sha256(path):
        raise RelationalPreStatusVectorBankError("vector-bank physical tensor binding is invalid")
    with safe_open(path, framework="numpy") as handle:
        expected = {"schema_version": "1", "kind": _TENSOR_KIND, "held_out_family_fold": ledger.get("held_out_family_fold"), "view": MASKED_VIEW, "layers": json.dumps(list(LAYERS), separators=(",", ":")), "root_count": str(ledger.get("candidate_root_count"))}
        if handle.metadata() != expected or set(handle.keys()) != {"h", "t", "s", "g", "defined"}:
            raise RelationalPreStatusVectorBankError("vector-bank tensor metadata or keys are invalid")
        raw_values = {name: np.asarray(handle.get_tensor(name)) for name in VECTOR_NAMES}
        raw_defined = np.asarray(handle.get_tensor("defined"))
    if any(value.dtype != np.float32 for value in raw_values.values()) or raw_defined.dtype != np.uint8 or not np.isin(raw_defined, (0, 1)).all():
        raise RelationalPreStatusVectorBankError("vector-bank tensor dtypes or flag values are invalid")
    arrays = CausalVectorBankArrays(
        **{name: _readonly(value, dtype=np.float32) for name, value in raw_values.items()},
        defined=_readonly(raw_defined.astype(np.bool_), dtype=np.bool_),
    )
    _validate_arrays(arrays, row_count=ledger.get("candidate_root_count"))
    rows = ledger.get("rows")
    if not isinstance(rows, list) or len(rows) != arrays.h.shape[0] or len({row.get("root_id") for row in rows if isinstance(row, Mapping)}) != len(rows):
        raise RelationalPreStatusVectorBankError("vector-bank root mapping is invalid")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("tensor_row_index") != index or row.get("view") != MASKED_VIEW or not isinstance(row.get("event_ids"), list) or not row["event_ids"] or not all(isinstance(event, str) and event for event in row["event_ids"]):
            raise RelationalPreStatusVectorBankError("vector-bank row mapping is invalid")
        if row["event_ids"] != sorted(set(row["event_ids"])) or not isinstance(row.get("root_id"), str) or not row["root_id"]:
            raise RelationalPreStatusVectorBankError("vector-bank row identity is invalid")
        _sha(row.get("prefix_state_sha256"), "row prefix-state SHA-256")
        _sha(row.get("prefix_token_ids_sha256"), "row prefix-token SHA-256")
        if row.get("family_fold") not in FOLDS or not isinstance(row.get("family"), str) or not row["family"] or not isinstance(row.get("turn_index"), int) or isinstance(row["turn_index"], bool) or row["turn_index"] < 0:
            raise RelationalPreStatusVectorBankError("vector-bank row family/fold/turn binding is invalid")
        support = _mapping(row.get("support"), "row support")
        dose = _mapping(row.get("dose_by_layer"), "row dose")
        for name in ("h", "t"):
            entry = _mapping(support.get(name), f"row {name} support")
            roots, pairs, count = entry.get("root_ids"), entry.get("pair_ids"), entry.get("count")
            if not isinstance(roots, list) or len(set(roots)) != len(roots) or not all(isinstance(item, str) and item for item in roots) or not isinstance(pairs, list) or len(set(pairs)) != len(pairs) or not all(isinstance(item, str) and item for item in pairs) or not isinstance(count, int) or isinstance(count, bool) or count != len(roots):
                raise RelationalPreStatusVectorBankError("vector-bank row support binding is invalid")
            doses = dose.get(name)
            if not isinstance(doses, list) or len(doses) != len(LAYERS) or not all(isinstance(value, (int, float)) and np.isfinite(value) and value >= 0.0 for value in doses):
                raise RelationalPreStatusVectorBankError("vector-bank row dose binding is invalid")
    fold_doses = _mapping(ledger.get("dose_by_layer"), "fold dose")
    for name in ("h", "t"):
        value = fold_doses.get(name)
        if not isinstance(value, list) or len(value) != len(LAYERS) or not all(isinstance(item, (int, float)) and np.isfinite(item) and item >= 0.0 for item in value):
            raise RelationalPreStatusVectorBankError("vector-bank fold dose is invalid")
    arrays_entry = _mapping(tensor.get("arrays"), "tensor arrays")
    for name, value in zip(VECTOR_NAMES, (arrays.h, arrays.t, arrays.s, arrays.g), strict=True):
        entry = _mapping(arrays_entry.get(name), f"{name} tensor entry")
        if entry != {"shape": list(value.shape), "dtype": str(value.dtype), "semantic_sha256": _array_sha256(name, value)}:
            raise RelationalPreStatusVectorBankError("vector-bank tensor semantic binding is invalid")
        if _mapping(ledger.get("array_semantic_sha256"), "ledger array hashes").get(name) != _array_sha256(name, value):
            raise RelationalPreStatusVectorBankError("vector-bank ledger array semantic binding is invalid")
    if _mapping(tensor.get("defined"), "defined tensor entry") != {"shape": list(arrays.defined.shape), "dtype": "bool", "semantic_sha256": _array_sha256("defined", arrays.defined.astype(np.uint8))}:
        raise RelationalPreStatusVectorBankError("vector-bank defined semantic binding is invalid")
    if ledger.get("defined_semantic_sha256") != _array_sha256("defined", arrays.defined.astype(np.uint8)):
        raise RelationalPreStatusVectorBankError("vector-bank ledger defined semantic binding is invalid")
    return MappingProxyType(dict(ledger)), arrays


__all__ = [
    "CausalVectorBankArrays", "CausalVectorBankBuild", "LEDGER_FILE_NAME", "MASKED_VIEW",
    "RelationalPreStatusVectorBankError", "TENSOR_FILE_NAME", "VECTOR_NAMES",
    "build_pre_status_causal_vector_bank", "load_pre_status_causal_vector_bank",
    "save_pre_status_causal_vector_bank",
]
