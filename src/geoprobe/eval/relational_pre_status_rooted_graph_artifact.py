"""Outcome-blind, hash-bound rooted prefix-state graph artifacts.

This module deliberately stops before opening an outcome report.  It turns the
sealed rooted-star bank into one quotient graph per view and outer fold; outcome
joining belongs to a later scoring stage.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from threading import RLock
from typing import Any

import numpy as np

from geoprobe.data.relational_pre_status_rooted_star import VIEWS
from geoprobe.data.relational_pre_status_rooted_star_store import (
    RelationalPreStatusRootedStarIndex,
    RootedStarReference,
    load_rooted_star_metric_input,
)
from geoprobe.eval.relational_pre_status_supervision import (
    LabelFreePrefixStateQuotient,
    build_label_free_prefix_state_quotient,
)
from geoprobe.geometry.relational_pre_status_rooted_graph import (
    ExactGraphEdge,
    FOLDS,
    FoldExactRootedGraph,
    RootedStarNode,
    build_fold_candidate_schedule,
    build_fold_exact_graph,
)
from geoprobe.geometry.relational_pre_status_rooted_metric import (
    RootedStarDistance,
    RootedStarMetricInput,
    RootedStarMetricScaler,
    rooted_star_descriptor,
    rooted_star_distance,
    rooted_star_energy_quotient,
)
from geoprobe.io import file_sha256


_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_ARTIFACT_KIND = "relational_pre_status_rooted_graph_artifact"
_MANIFEST_KIND = "relational_pre_status_rooted_graph_artifact_manifest"
_DISTANCE_CACHE_KIND = "relational_pre_status_rooted_graph_raw_distance_cache"
_OUTCOME_TERMS = ("outcome", "decept", "honest", "desired", "knowledge", "label")


class RelationalPreStatusRootedGraphArtifactError(ValueError):
    """Raised when an outcome-blind graph artifact is invalid or unsafe."""


@dataclass(frozen=True, slots=True)
class LoadedRelationalPreStatusRootedGraphs:
    """Validated persisted graphs, indexed by view then held-out fold."""

    manifest: Mapping[str, Any]
    folds_by_view: Mapping[str, Mapping[str, Mapping[str, Any]]]

    def variant(self, view: str, held_out_family_fold: str, name: str = "joint") -> Mapping[str, Any]:
        """Return one explicitly named top-8 comparator graph for scoring."""
        try:
            return self.folds_by_view[view][held_out_family_fold]["graph"]["variants"][name]
        except KeyError as error:
            raise RelationalPreStatusRootedGraphArtifactError("requested graph variant is absent") from error

    def fold_graph(
        self,
        view: str,
        held_out_family_fold: str,
        name: str = "joint",
    ) -> FoldExactRootedGraph:
        """Rehydrate one validated persisted variant for downstream fields."""
        try:
            artifact = self.folds_by_view[view][held_out_family_fold]
            payload = artifact["graph"]
            variant = payload["variants"][name]
            scaler = payload["metric_scaler"]
        except KeyError as error:
            raise RelationalPreStatusRootedGraphArtifactError(
                "requested graph variant is absent"
            ) from error

        def grouped(field: str) -> Mapping[str, tuple[ExactGraphEdge, ...]]:
            rows: dict[str, list[ExactGraphEdge]] = {}
            for raw in variant[field]:
                edge = ExactGraphEdge(
                    source_id=str(raw["source_id"]),
                    target_id=str(raw["target_id"]),
                    rank=int(raw["rank"]),
                    joint_score=float(raw["joint_score"]),
                    residual_score=float(raw["residual_score"]),
                    attention_score=float(raw["attention_score"]),
                    descriptor_rank=int(raw["descriptor_rank"]),
                )
                rows.setdefault(edge.source_id, []).append(edge)
            return {
                source: tuple(sorted(edges, key=lambda edge: edge.rank))
                for source, edges in rows.items()
            }

        return FoldExactRootedGraph(
            held_out_family_fold=held_out_family_fold,
            graph_width=int(payload["graph_width"]),
            query_edges=grouped("query_to_training"),
            training_edges=grouped("training_to_training"),
            scaler=RootedStarMetricScaler(
                residual_scale=float(scaler["residual_scale"]),
                attention_scale=float(scaler["attention_scale"]),
            ),
            candidate_pair_count=int(payload["candidate_pair_count"]),
            exact_pair_count=int(payload["exact_pair_count"]),
        )

    def evaluation_inventory(
        self,
    ) -> Mapping[str, Mapping[str, Mapping[str, FoldExactRootedGraph]]]:
        """Expose all views and variants with one unambiguous nesting order."""
        variants = ("joint", "residual_only", "attention_only")
        return {
            view: {
                variant: {
                    fold: self.fold_graph(view, fold, variant) for fold in FOLDS
                }
                for variant in variants
            }
            for view in VIEWS
        }


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RelationalPreStatusRootedGraphArtifactError("value is not canonical JSON") from error


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
        raise RelationalPreStatusRootedGraphArtifactError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelationalPreStatusRootedGraphArtifactError(f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RelationalPreStatusRootedGraphArtifactError(f"{label} is not finite UTF-8 JSON") from error
    return _mapping(value, label)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return sha256(_canonical(body)).hexdigest()


def _safe_child(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or Path(relative).is_absolute():
        raise RelationalPreStatusRootedGraphArtifactError(f"{label} is not a safe relative POSIX path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise RelationalPreStatusRootedGraphArtifactError(f"{label} escapes artifact root")
    return path


def _assert_outcome_free(value: object, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower()
            if any(term in name for term in _OUTCOME_TERMS):
                raise RelationalPreStatusRootedGraphArtifactError(f"outcome-bearing field escaped into {path}.{key}")
            _assert_outcome_free(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_outcome_free(child, f"{path}[{index}]")


def _distance_from_row(value: object) -> RootedStarDistance:
    row = _mapping(value, "raw exact distance")
    try:
        return RootedStarDistance(float(row["residual"]), float(row["attention_head_set"]))
    except (KeyError, TypeError, ValueError) as error:
        raise RelationalPreStatusRootedGraphArtifactError("raw exact distance is malformed") from error


class _RawDistanceCache:
    """A small durable cache for raw geometry-to-geometry metric evaluations."""

    def __init__(self, path: Path, binding: Mapping[str, str]) -> None:
        self._connection = sqlite3.connect(
            Path(path).resolve(), timeout=30.0, check_same_thread=False
        )
        self._lock = RLock()
        self._pending_writes = 0
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.execute("PRAGMA journal_mode=DELETE")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._connection.execute("CREATE TABLE IF NOT EXISTS distances (left_id TEXT NOT NULL, right_id TEXT NOT NULL, residual REAL NOT NULL, attention REAL NOT NULL, PRIMARY KEY(left_id, right_id), CHECK(left_id <= right_id))")
        text = _canonical(dict(binding)).decode("utf-8")
        row = self._connection.execute("SELECT value FROM metadata WHERE key = 'binding'").fetchone()
        if row is None:
            with self._connection:
                self._connection.execute("INSERT INTO metadata(key, value) VALUES ('binding', ?)", (text,))
        elif row[0] != text:
            self.close()
            raise RelationalPreStatusRootedGraphArtifactError("raw exact-distance cache has incompatible bank, metric, or view binding")

    def get(self, left: str, right: str) -> RootedStarDistance | None:
        first, second = sorted((left, right))
        with self._lock:
            row = self._connection.execute("SELECT residual, attention FROM distances WHERE left_id = ? AND right_id = ?", (first, second)).fetchone()
        if row is None:
            return None
        return RootedStarDistance(float(row[0]), float(row[1]))

    def put(self, left: str, right: str, value: RootedStarDistance) -> None:
        first, second = sorted((left, right))
        with self._lock:
            self._connection.execute("INSERT OR IGNORE INTO distances VALUES (?, ?, ?, ?)", (first, second, float(value.residual), float(value.attention_head_set)))
            self._pending_writes += 1
            row = self._connection.execute("SELECT residual, attention FROM distances WHERE left_id = ? AND right_id = ?", (first, second)).fetchone()
            if row is None or _distance_from_row({"residual": row[0], "attention_head_set": row[1]}) != value:
                raise RelationalPreStatusRootedGraphArtifactError("raw exact-distance cache write did not round-trip")
            if self._pending_writes >= 256:
                self._connection.commit()
                self._pending_writes = 0

    def close(self) -> None:
        with self._lock:
            self._connection.commit()
            self._pending_writes = 0
            self._connection.close()


def _quotient(index: RelationalPreStatusRootedStarIndex) -> LabelFreePrefixStateQuotient:
    quotient = build_label_free_prefix_state_quotient(index)
    if not getattr(quotient, "nodes", None) or not getattr(quotient, "event_to_node_ids", None):
        raise RelationalPreStatusRootedGraphArtifactError("label-free prefix quotient has no nodes or event mapping")
    return quotient


def _view_nodes(
    quotient: LabelFreePrefixStateQuotient, view: str, metric_loader: Callable[[RelationalPreStatusRootedStarIndex, RootedStarReference], RootedStarMetricInput], index: RelationalPreStatusRootedStarIndex,
) -> tuple[tuple[RootedStarNode, ...], Mapping[str, tuple[RootedStarReference, ...]]]:
    geometry: dict[str, tuple[RootedStarReference, ...]] = {}
    nodes: list[RootedStarNode] = []
    for node in sorted(quotient.nodes, key=lambda item: item.node_id):
        if node.view != view:
            continue
        references = tuple(node.representative_references)
        if not references or len({item.geometry_sha256 for item in references}) != len(references):
            raise RelationalPreStatusRootedGraphArtifactError("quotient node representatives are not unique exact geometry references")
        descriptors = [rooted_star_descriptor(metric_loader(index, reference)) for reference in references]
        if len({tuple(item.shape) for item in descriptors}) != 1:
            raise RelationalPreStatusRootedGraphArtifactError("rooted-star descriptors have inconsistent widths")
        descriptor = np.mean(np.stack(descriptors, axis=0, dtype=np.float64), axis=0)
        nodes.append(RootedStarNode(node.node_id, node.family, node.family_fold, descriptor))
        geometry[node.node_id] = references
    if not nodes:
        raise RelationalPreStatusRootedGraphArtifactError(f"view {view} has no quotient nodes")
    return tuple(nodes), geometry


def _edge_row(row: object) -> Mapping[str, Any]:
    return {"source_id": row.source_id, "target_id": row.target_id, "rank": row.rank, "joint_score": row.joint_score, "residual_score": row.residual_score, "attention_score": row.attention_score, "descriptor_rank": row.descriptor_rank}


def _variant_rows(
    rows: Mapping[str, Sequence[object]], *, distance: Callable[[str, str], RootedStarDistance], component: str, scale: float, width: int,
) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for source in sorted(rows):
        scored = []
        for candidate in rows[source]:
            exact = distance(candidate.source_id, candidate.target_id)
            score = float(getattr(exact, component) / scale)
            scored.append((score, candidate, exact))
        scored.sort(key=lambda item: (item[0], item[1].target_id))
        result.extend({"source_id": candidate.source_id, "target_id": candidate.target_id, "rank": rank, "joint_score": score, "residual_score": exact.residual, "attention_score": exact.attention_head_set, "descriptor_rank": candidate.rank} for rank, (score, candidate, exact) in enumerate(scored[:width], 1))
    return result


def _graph_payload(
    graph: FoldExactRootedGraph,
    schedule: object,
    *,
    distance: Callable[[str, str], RootedStarDistance],
    nodes: Sequence[RootedStarNode],
    audit: Sequence[Mapping[str, Any]],
    dispersion: Mapping[str, Mapping[str, float | int]],
) -> Mapping[str, Any]:
    joint_query = [_edge_row(item) for source in sorted(graph.query_edges) for item in graph.query_edges[source]]
    joint_training = [_edge_row(item) for source in sorted(graph.training_edges) for item in graph.training_edges[source]]
    return {
        "held_out_family_fold": graph.held_out_family_fold,
        "candidate_width": schedule.candidate_width,
        "graph_width": graph.graph_width,
        "descriptor_center": schedule.descriptor_center.tolist(),
        "descriptor_scale": schedule.descriptor_scale.tolist(),
        "metric_scaler": {"residual_scale": graph.scaler.residual_scale, "attention_scale": graph.scaler.attention_scale},
        "node_family_folds": {node.node_id: node.family_fold for node in sorted(nodes, key=lambda item: item.node_id)},
        "query_to_training": joint_query,
        "training_to_training": joint_training,
        "variants": {
            "joint": {"query_to_training": joint_query, "training_to_training": joint_training},
            "residual_only": {
                "query_to_training": _variant_rows(schedule.query_edges, distance=distance, component="residual", scale=graph.scaler.residual_scale, width=graph.graph_width),
                "training_to_training": _variant_rows(schedule.training_edges, distance=distance, component="residual", scale=graph.scaler.residual_scale, width=graph.graph_width),
            },
            "attention_only": {
                "query_to_training": _variant_rows(schedule.query_edges, distance=distance, component="attention_head_set", scale=graph.scaler.attention_scale, width=graph.graph_width),
                "training_to_training": _variant_rows(schedule.training_edges, distance=distance, component="attention_head_set", scale=graph.scaler.attention_scale, width=graph.graph_width),
            },
        },
        "candidate_pair_count": graph.candidate_pair_count,
        "exact_pair_count": graph.exact_pair_count,
        "candidate_recall_audit": list(audit),
        "within_fibre_dispersion": dict(dispersion),
    }


def _audit(
    *, fold: str, nodes: Sequence[RootedStarNode], schedule: object, graph: FoldExactRootedGraph, distance: Callable[[str, str], RootedStarDistance], panel_size: int,
) -> tuple[Mapping[str, Any], ...]:
    eligible = sorted(nodes, key=lambda item: (sha256(f"rooted-graph-audit-v1:{fold}:{item.node_id}".encode()).digest(), item.node_id))[:panel_size]
    result: list[Mapping[str, Any]] = []
    for source in eligible:
        targets = [node for node in nodes if node.family_fold != fold] if source.family_fold == fold else [node for node in nodes if node.family_fold != fold and node.node_id != source.node_id]
        scored = [(graph.scaler.transform(distance(source.node_id, target.node_id)), target.node_id) for target in targets]
        scored.sort(key=lambda item: (item[0], item[1]))
        top = tuple(target for _, target in scored[:graph.graph_width])
        candidates = schedule.query_edges[source.node_id] if source.family_fold == fold else schedule.training_edges[source.node_id]
        candidate_ids = {item.target_id for item in candidates}
        recall = sum(item in candidate_ids for item in top) / len(top) if top else 1.0
        result.append({"source_id": source.node_id, "exact_top8_target_ids": list(top), "candidate_recall_at_64": recall, "exact_top8_count": len(top)})
    return tuple(result)


def _artifact_path(root: Path, view: str, fold: str) -> Path:
    return root / "views" / view / f"{fold}.json"


def _complete_fold(path: Path, binding: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        artifact = _read_json(path, "fold artifact")
        _validate_fold(artifact, binding)
    except RelationalPreStatusRootedGraphArtifactError:
        return None
    return artifact


def _validate_fold(value: Mapping[str, Any], binding: Mapping[str, Any]) -> None:
    _assert_outcome_free(value)
    if value.get("schema_version") != _SCHEMA_VERSION or value.get("kind") != _ARTIFACT_KIND or value.get("status") != "success":
        raise RelationalPreStatusRootedGraphArtifactError("fold artifact schema or status is invalid")
    if value.get("artifact_sha256") != _self_hash(value, "artifact_sha256"):
        raise RelationalPreStatusRootedGraphArtifactError("fold artifact self-hash is invalid")
    if value.get("binding") != dict(binding):
        raise RelationalPreStatusRootedGraphArtifactError("fold artifact binding differs from this request")
    payload = _mapping(value.get("graph"), "fold graph")
    if payload.get("candidate_width") != 64 or payload.get("graph_width") != 8:
        raise RelationalPreStatusRootedGraphArtifactError("fold artifact graph widths are invalid")
    for edge_group in ("query_to_training", "training_to_training"):
        rows = payload.get(edge_group)
        if not isinstance(rows, list):
            raise RelationalPreStatusRootedGraphArtifactError("fold graph edge list is invalid")
        if edge_group == "query_to_training" and any(row.get("source_id") == row.get("target_id") for row in rows if isinstance(row, Mapping)):
            raise RelationalPreStatusRootedGraphArtifactError("fold graph contains a self edge")
    folds = _mapping(payload.get("node_family_folds"), "fold graph node family folds")
    if not folds or any(value not in FOLDS for value in folds.values()):
        raise RelationalPreStatusRootedGraphArtifactError("fold graph node family folds are invalid")
    held_out = value["binding"]["held_out_family_fold"]
    for group, source_must_be_held_out in (("query_to_training", True), ("training_to_training", False)):
        for row in payload[group]:
            if not isinstance(row, Mapping) or row.get("source_id") not in folds or row.get("target_id") not in folds:
                raise RelationalPreStatusRootedGraphArtifactError("fold graph edge has an unknown endpoint")
            if (folds[row["source_id"]] == held_out) != source_must_be_held_out or folds[row["target_id"]] == held_out:
                raise RelationalPreStatusRootedGraphArtifactError("fold graph violates held-out-to-training or training-to-training routing")
    variants = _mapping(payload.get("variants"), "fold graph variants")
    if set(variants) != {"joint", "residual_only", "attention_only"}:
        raise RelationalPreStatusRootedGraphArtifactError("fold graph comparator variants are incomplete")
    for name, variant in variants.items():
        variant_rows = _mapping(variant, f"{name} graph variant")
        for group in ("query_to_training", "training_to_training"):
            if variant_rows.get(group) != payload[group] and name == "joint":
                raise RelationalPreStatusRootedGraphArtifactError("joint graph variant differs from primary graph")
            rows = variant_rows.get(group)
            if not isinstance(rows, list):
                raise RelationalPreStatusRootedGraphArtifactError("graph variant edge list is invalid")
            for row in rows:
                if not isinstance(row, Mapping) or row.get("source_id") not in folds or row.get("target_id") not in folds:
                    raise RelationalPreStatusRootedGraphArtifactError("graph variant has an unknown endpoint")
                source_is_held_out = folds[row["source_id"]] == held_out
                if (group == "query_to_training") != source_is_held_out or folds[row["target_id"]] == held_out:
                    raise RelationalPreStatusRootedGraphArtifactError("graph variant violates fold-safe routing")


def _manifest_binding(*, bank_manifest_sha256: str, protocol_sha256: str, expected_state_count_per_view: int, expected_event_count: int) -> Mapping[str, Any]:
    metric_source = Path(rooted_star_distance.__code__.co_filename).resolve()
    return {"rooted_star_manifest_file_sha256": bank_manifest_sha256, "frozen_protocol_file_sha256": protocol_sha256, "expected_state_count_per_view": expected_state_count_per_view, "expected_event_count": expected_event_count, "metric": "rooted_star_energy_quotient.v1", "metric_source_sha256": file_sha256(metric_source), "views": list(VIEWS), "folds": list(FOLDS)}


def build_relational_pre_status_rooted_graph_artifacts(
    index: RelationalPreStatusRootedStarIndex,
    *,
    out_dir: Path,
    expected_rooted_star_manifest_sha256: str,
    frozen_protocol_path: Path,
    expected_frozen_protocol_sha256: str,
    expected_state_count_per_view: int,
    expected_event_count: int,
    cache_dir: Path | None = None,
    metric_loader: Callable[[RelationalPreStatusRootedStarIndex, RootedStarReference], RootedStarMetricInput] = load_rooted_star_metric_input,
) -> Mapping[str, Any]:
    """Build five fold-safe exact graphs per view without opening outcomes."""
    if not isinstance(expected_state_count_per_view, int) or isinstance(expected_state_count_per_view, bool) or expected_state_count_per_view < 1:
        raise RelationalPreStatusRootedGraphArtifactError("expected per-view state count must be positive")
    if not isinstance(expected_event_count, int) or isinstance(expected_event_count, bool) or expected_event_count < 1:
        raise RelationalPreStatusRootedGraphArtifactError("expected event count must be positive")
    bank_path = Path(index.artifact_root).resolve() / "manifest.json"
    bank_expected = _sha(expected_rooted_star_manifest_sha256, "expected rooted-star manifest SHA-256")
    protocol_expected = _sha(expected_frozen_protocol_sha256, "expected frozen protocol SHA-256")
    if not bank_path.is_file() or file_sha256(bank_path) != bank_expected or index.manifest_sha256 != bank_expected:
        raise RelationalPreStatusRootedGraphArtifactError("rooted-star manifest differs from its expected physical SHA-256")
    protocol_path = Path(frozen_protocol_path).resolve()
    if not protocol_path.is_file() or file_sha256(protocol_path) != protocol_expected:
        raise RelationalPreStatusRootedGraphArtifactError("frozen protocol differs from its expected physical SHA-256")
    quotient = _quotient(index)
    view_counts = {
        view: sum(node.view == view for node in quotient.nodes) for view in VIEWS
    }
    if any(
        count != expected_state_count_per_view for count in view_counts.values()
    ) or len(quotient.nodes) != expected_state_count_per_view * len(VIEWS):
        raise RelationalPreStatusRootedGraphArtifactError(
            "label-free quotient per-view state count is not the caller's expected inventory"
        )
    if len(quotient.event_to_node_ids) != expected_event_count:
        raise RelationalPreStatusRootedGraphArtifactError("label-free quotient count is not the caller's expected inventory")
    if any(set(node_ids) != set(VIEWS) for node_ids in quotient.event_to_node_ids.values()):
        raise RelationalPreStatusRootedGraphArtifactError("each label-free event must map to exactly one node per view")
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    binding = _manifest_binding(bank_manifest_sha256=bank_expected, protocol_sha256=protocol_expected, expected_state_count_per_view=expected_state_count_per_view, expected_event_count=expected_event_count)
    cache_root = (Path(cache_dir).resolve() if cache_dir is not None else root / "raw_distance_cache")
    files: dict[str, dict[str, Mapping[str, str]]] = {}
    for view in VIEWS:
        nodes, fibres = _view_nodes(quotient, view, metric_loader, index)
        inputs: dict[str, RootedStarMetricInput] = {}
        inputs_lock = RLock()
        pair_memo_lock = RLock()
        cache_root.mkdir(parents=True, exist_ok=True)
        cache = _RawDistanceCache(cache_root / f"{view}.sqlite", {"kind": _DISTANCE_CACHE_KIND, "rooted_star_manifest_file_sha256": bank_expected, "metric": "rooted_star_distance.v1", "metric_source_sha256": binding["metric_source_sha256"], "view": view})
        try:
            def raw(left: RootedStarReference, right: RootedStarReference) -> RootedStarDistance:
                if left.geometry_sha256 == right.geometry_sha256:
                    return RootedStarDistance(0.0, 0.0)
                found = cache.get(left.geometry_sha256, right.geometry_sha256)
                if found is not None:
                    return found
                with inputs_lock:
                    for reference in (left, right):
                        if reference.geometry_sha256 not in inputs:
                            inputs[reference.geometry_sha256] = metric_loader(index, reference)
                    left_input = inputs[left.geometry_sha256]
                    right_input = inputs[right.geometry_sha256]
                value = rooted_star_distance(left_input, right_input)
                cache.put(left.geometry_sha256, right.geometry_sha256, value)
                return value

            pair_memo: dict[tuple[str, str], RootedStarDistance] = {}
            dispersion: dict[str, Mapping[str, float | int]] = {}
            for node_id, refs in fibres.items():
                within = [raw(left, right) for number, left in enumerate(refs) for right in refs[number + 1 :]]
                dispersion[node_id] = {"geometry_representative_count": len(refs), "unordered_pair_count": len(within), "mean_residual": float(np.mean([item.residual for item in within])) if within else 0.0, "mean_attention_head_set": float(np.mean([item.attention_head_set for item in within])) if within else 0.0}

            def quotient_distance(left_id: str, right_id: str) -> RootedStarDistance:
                key = tuple(sorted((left_id, right_id)))
                with pair_memo_lock:
                    found = pair_memo.get(key)
                if found is not None:
                    return found
                left_refs, right_refs = fibres[key[0]], fibres[key[1]]
                cross = [raw(left, right) for left in left_refs for right in right_refs]
                left_within = [raw(left, right) for number, left in enumerate(left_refs) for right in left_refs[number + 1 :]]
                right_within = [raw(left, right) for number, left in enumerate(right_refs) for right in right_refs[number + 1 :]]
                value = rooted_star_energy_quotient(cross, left_within, right_within, left_size=len(left_refs), right_size=len(right_refs))
                with pair_memo_lock:
                    value = pair_memo.setdefault(key, value)
                return value

            files[view] = {}
            for fold in FOLDS:
                fold_binding = {**binding, "view": view, "held_out_family_fold": fold}
                path = _artifact_path(root, view, fold)
                artifact = _complete_fold(path, fold_binding)
                if artifact is None:
                    schedule = build_fold_candidate_schedule(nodes, held_out_family_fold=fold, candidate_width=64)
                    graph = build_fold_exact_graph(schedule, pair_distance=quotient_distance, graph_width=8, distance_workers=4)
                    audit = _audit(fold=fold, nodes=nodes, schedule=schedule, graph=graph, distance=quotient_distance, panel_size=min(8, len(nodes)))
                    artifact = {"schema_version": _SCHEMA_VERSION, "kind": _ARTIFACT_KIND, "status": "success", "binding": fold_binding, "graph": _graph_payload(graph, schedule, distance=quotient_distance, nodes=nodes, audit=audit, dispersion=dispersion)}
                    artifact["artifact_sha256"] = _self_hash(artifact, "artifact_sha256")
                    _assert_outcome_free(artifact)
                    _atomic_json(path, artifact)
                files[view][fold] = {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
        finally:
            cache.close()
    manifest: dict[str, Any] = {"schema_version": _SCHEMA_VERSION, "kind": _MANIFEST_KIND, "status": "success", "binding": binding, "fold_artifacts": files}
    manifest["manifest_sha256"] = _self_hash(manifest, "manifest_sha256")
    _assert_outcome_free(manifest)
    _atomic_json(root / _MANIFEST_NAME, manifest)
    return manifest


def load_relational_pre_status_rooted_graph_artifacts(root: Path) -> LoadedRelationalPreStatusRootedGraphs:
    """Load only a complete, self-hashed, physically hash-bound graph bundle."""
    artifact_root = Path(root).resolve()
    manifest = _read_json(artifact_root / _MANIFEST_NAME, "graph artifact manifest")
    _assert_outcome_free(manifest)
    if manifest.get("schema_version") != _SCHEMA_VERSION or manifest.get("kind") != _MANIFEST_KIND or manifest.get("status") != "success" or manifest.get("manifest_sha256") != _self_hash(manifest, "manifest_sha256"):
        raise RelationalPreStatusRootedGraphArtifactError("graph artifact manifest schema, status, or self-hash is invalid")
    binding = _mapping(manifest.get("binding"), "manifest binding")
    files = _mapping(manifest.get("fold_artifacts"), "manifest fold artifacts")
    if set(files) != set(VIEWS):
        raise RelationalPreStatusRootedGraphArtifactError("graph artifact manifest views are incomplete")
    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    for view in VIEWS:
        view_files = _mapping(files.get(view), f"{view} fold artifacts")
        if set(view_files) != set(FOLDS):
            raise RelationalPreStatusRootedGraphArtifactError("graph artifact manifest folds are incomplete")
        result[view] = {}
        for fold in FOLDS:
            row = _mapping(view_files[fold], "fold artifact file binding")
            path = _safe_child(artifact_root, row.get("path"), "fold artifact path")
            expected = _sha(row.get("sha256"), "fold artifact physical SHA-256")
            if not path.is_file() or file_sha256(path) != expected:
                raise RelationalPreStatusRootedGraphArtifactError("fold artifact differs from manifest physical SHA-256")
            artifact = _read_json(path, "fold artifact")
            _validate_fold(artifact, {**binding, "view": view, "held_out_family_fold": fold})
            result[view][fold] = artifact
    return LoadedRelationalPreStatusRootedGraphs(manifest, result)


__all__ = [
    "LoadedRelationalPreStatusRootedGraphs",
    "RelationalPreStatusRootedGraphArtifactError",
    "build_relational_pre_status_rooted_graph_artifacts",
    "load_relational_pre_status_rooted_graph_artifacts",
]
