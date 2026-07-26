"""Durable fold-safe artifacts for the relational gauge controller substrate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from geoprobe.control import relational_horizontal_lift as lift_module
from geoprobe.control import relational_intrinsic_risk_field as field_module
from geoprobe.control.relational_horizontal_lift import (
    HorizontalLift,
    HorizontalLiftPatch,
)
from geoprobe.control.relational_intrinsic_risk_field import (
    GaugeRiskObservation,
    PressureMatchedFieldConfig,
    PressureMatchedRiskField,
)
from geoprobe.data import relational_pre_status_rooted_star_store as store_module
from geoprobe.data.relational_pre_status_rooted_star_store import (
    RelationalPreStatusRootedStarIndex,
)
from geoprobe.eval import relational_gauge_bundle as bundle_module
from geoprobe.eval import relational_gauge_training_supervision as training_module
from geoprobe.eval import relational_pre_status_rooted_graph_artifact as graph_module
from geoprobe.eval.relational_gauge_bundle import (
    FoldGaugeBundleDiagnostics,
    FoldGaugeControllerBundle,
    RelationalGaugeBundleConfig,
    build_fold_gauge_controller_bundle,
    build_training_gauge_atlas,
    load_training_node_fibers,
)
from geoprobe.eval.relational_pre_status_rooted_graph_artifact import (
    LoadedRelationalPreStatusRootedGraphs,
)
from geoprobe.eval.relational_gauge_training_supervision import (
    build_fold_gauge_training_supervision,
)
from geoprobe.geometry import relational_gauge_atlas as atlas_module
from geoprobe.geometry.relational_gauge_atlas import (
    GaugeChart,
    GaugeConnection,
    GaugeQueryState,
    RelationalGaugeAtlas,
    RelationalGaugeAtlasConfig,
)
from geoprobe.geometry.relational_pre_status_rooted_graph import FOLDS, FoldExactRootedGraph
from geoprobe.io import file_sha256
from geoprobe.provenance import git_provenance


# Schema v2 (2026-07-23): the atlas is PERSISTED (arrays in controller.safetensors,
# structure in bundle.json) and rehydrated at load. v1 rebuilt the atlas from the
# graph at load and required bit-identical eigendecompositions, which cannot hold
# across platforms (Mac build vs Linux pod: BLAS differences) — the pod failed with
# "rebuilt atlas differs from persisted fingerprint". Rebuild-determinism is still
# asserted, but at BUILD time on the build machine, where it is meaningful.
SCHEMA_VERSION = 2
MANIFEST_KIND = "relational_gauge_controller_artifact_manifest"
FOLD_KIND = "relational_gauge_controller_fold_artifact"
REPORT_KIND = "relational_gauge_controller_build_report"

_LIFT_TENSOR_NAMES = frozenset(
    {
        "lift_matrix",
        "tangent_metric",
        "layer_fiber_norm_caps",
        "intrinsic_trust_radius",
        "weighted_relative_fit_error",
        "weighted_relative_roundtrip_error",
        "condition_number",
    }
)


class RelationalGaugeControllerArtifactError(ValueError):
    """A persisted controller bundle violates its binding or tensor contract."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RelationalGaugeControllerArtifactError(
            "artifact value is not canonical JSON"
        ) from error


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return sha256(_canonical(payload)).hexdigest()


def _sha(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RelationalGaugeControllerArtifactError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _binding(path: Path, *, expected_sha256: str | None = None) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise RelationalGaugeControllerArtifactError(f"input is absent: {resolved}")
    actual = file_sha256(resolved)
    if expected_sha256 is not None and actual != _sha(
        expected_sha256, name="expected input SHA-256"
    ):
        raise RelationalGaugeControllerArtifactError(
            f"input differs from expected SHA-256: {resolved}"
        )
    return {"path": str(resolved), "sha256": actual}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()},
        temporary,
        metadata={
            "schema_version": str(SCHEMA_VERSION),
            "kind": FOLD_KIND,
        },
    )
    os.replace(temporary, path)


def _array_sha(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\x00")
    digest.update(_canonical(list(array.shape)))
    digest.update(b"\x00")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _atlas_fingerprint(bundle: FoldGaugeControllerBundle) -> dict[str, Any]:
    return _atlas_content_fingerprint(bundle.atlas)


def _atlas_content_fingerprint(atlas: RelationalGaugeAtlas) -> dict[str, Any]:
    charts = []
    for chart_id in atlas.chart_order:
        if chart_id not in atlas.charts:
            continue
        chart = atlas.get_chart(chart_id)
        charts.append(
            {
                "chart_id": chart_id,
                "center_node_id": str(chart.center_node_id),
                "support_ids": [str(value) for value in chart.support_ids],
                "support_distances_sha256": _array_sha(chart.support_distances),
                "coordinates_sha256": _array_sha(chart.coordinates),
                "eigenvalues_sha256": _array_sha(chart.eigenvalues),
                "stress": float(chart.stress),
                "support_radius": float(chart.support_radius),
            }
        )
    connections = [
        {
            "source_chart_id": source,
            "target_chart_id": target,
            "overlap_node_ids": [str(value) for value in connection.overlap_node_ids],
            "transport_sha256": _array_sha(connection.transport),
            "residual": float(connection.residual),
        }
        for (source, target), connection in sorted(atlas.connections.items())
    ]
    payload = {
        "node_ids": [str(value) for value in atlas.node_ids],
        "distances_sha256": _array_sha(atlas.distances),
        "charts": charts,
        "connections": connections,
    }
    return {
        "node_count": len(atlas.node_ids),
        "chart_count": len(charts),
        "connection_count": len(connections),
        "content_sha256": sha256(_canonical(payload)).hexdigest(),
    }


def _atlas_persist_payload(
    atlas: RelationalGaugeAtlas,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Serialize the complete atlas: arrays as tensors, structure as JSON."""
    tensors: dict[str, torch.Tensor] = {
        "atlas.distances": torch.from_numpy(
            np.ascontiguousarray(np.asarray(atlas.distances, dtype=np.float64))
        ),
    }
    charts_meta: list[dict[str, Any]] = []
    chart_index = 0
    for chart_id in atlas.chart_order:
        if chart_id not in atlas.charts:
            continue
        chart = atlas.get_chart(chart_id)
        tensors[f"atlas.chart.{chart_index}.coordinates"] = torch.from_numpy(
            np.ascontiguousarray(np.asarray(chart.coordinates, dtype=np.float64))
        )
        tensors[f"atlas.chart.{chart_index}.eigenvalues"] = torch.from_numpy(
            np.ascontiguousarray(np.asarray(chart.eigenvalues, dtype=np.float64))
        )
        tensors[f"atlas.chart.{chart_index}.support_distances"] = torch.from_numpy(
            np.ascontiguousarray(np.asarray(chart.support_distances, dtype=np.float64))
        )
        charts_meta.append(
            {
                "chart_id": chart.chart_id,
                "center_node_id": str(chart.center_node_id),
                "support_ids": [str(value) for value in chart.support_ids],
                "stress": float(chart.stress),
                "support_radius": float(chart.support_radius),
            }
        )
        chart_index += 1
    connections_meta: list[dict[str, Any]] = []
    for index, ((source, target), connection) in enumerate(
        sorted(atlas.connections.items())
    ):
        tensors[f"atlas.connection.{index}.transport"] = torch.from_numpy(
            np.ascontiguousarray(np.asarray(connection.transport, dtype=np.float64))
        )
        connections_meta.append(
            {
                "source_chart_id": source,
                "target_chart_id": target,
                "source_support_ids": [str(v) for v in connection.source_support_ids],
                "target_support_ids": [str(v) for v in connection.target_support_ids],
                "overlap_node_ids": [str(v) for v in connection.overlap_node_ids],
                "residual": float(connection.residual),
            }
        )
    config = atlas.config
    content = {
        "config": {
            "fixed_rank": int(config.fixed_rank),
            "support_radius": float(config.support_radius),
            "min_chart_support": int(config.min_chart_support),
            "max_chart_support": int(config.max_chart_support),
            "min_overlap": int(config.min_overlap),
        },
        "node_ids": [str(value) for value in atlas.node_ids],
        "chart_order": [str(value) for value in atlas.chart_order],
        "charts": charts_meta,
        "connections": connections_meta,
    }
    return tensors, content


def _rehydrate_atlas(
    content: Mapping[str, Any],
    tensors: Mapping[str, np.ndarray],
) -> RelationalGaugeAtlas:
    """Reconstruct the atlas from persisted arrays without any recomputation.

    Bypasses ``RelationalGaugeAtlas.__init__`` on purpose: the constructor would
    recompute shortest paths, classical MDS, and overlap transports, and the
    eigendecomposition results are not bit-reproducible across platforms. The
    chart/connection dataclass validators still run, and the caller verifies the
    persisted atlas fingerprint over these exact bytes afterwards.
    """
    config_raw = content.get("config")
    if not isinstance(config_raw, Mapping):
        raise RelationalGaugeControllerArtifactError("persisted atlas config is absent")
    config = RelationalGaugeAtlasConfig(
        fixed_rank=int(config_raw["fixed_rank"]),
        support_radius=float(config_raw["support_radius"]),
        min_chart_support=int(config_raw["min_chart_support"]),
        max_chart_support=int(config_raw["max_chart_support"]),
        min_overlap=int(config_raw["min_overlap"]),
    )
    node_ids = tuple(str(value) for value in content["node_ids"])
    if not node_ids:
        raise RelationalGaugeControllerArtifactError("persisted atlas has no nodes")
    distances = np.asarray(tensors["atlas.distances"], dtype=np.float64)
    if distances.shape != (len(node_ids), len(node_ids)):
        raise RelationalGaugeControllerArtifactError(
            "persisted atlas distances shape mismatch"
        )
    charts: dict[str, GaugeChart] = {}
    for index, row in enumerate(content["charts"]):
        chart = GaugeChart(
            chart_id=str(row["chart_id"]),
            center_node_id=str(row["center_node_id"]),
            support_ids=tuple(str(value) for value in row["support_ids"]),
            support_distances=tuple(
                tuple(float(v) for v in line)
                for line in np.asarray(
                    tensors[f"atlas.chart.{index}.support_distances"], dtype=np.float64
                )
            ),
            coordinates=np.asarray(
                tensors[f"atlas.chart.{index}.coordinates"], dtype=np.float64
            ),
            eigenvalues=np.asarray(
                tensors[f"atlas.chart.{index}.eigenvalues"], dtype=np.float64
            ),
            stress=float(row["stress"]),
            support_radius=float(row["support_radius"]),
        )
        charts[chart.chart_id] = chart
    connections: dict[tuple[str, str], GaugeConnection] = {}
    for index, row in enumerate(content["connections"]):
        connection = GaugeConnection(
            source_chart_id=str(row["source_chart_id"]),
            target_chart_id=str(row["target_chart_id"]),
            source_support_ids=tuple(str(v) for v in row["source_support_ids"]),
            target_support_ids=tuple(str(v) for v in row["target_support_ids"]),
            overlap_node_ids=tuple(str(v) for v in row["overlap_node_ids"]),
            transport=np.asarray(
                tensors[f"atlas.connection.{index}.transport"], dtype=np.float64
            ),
            residual=float(row["residual"]),
        )
        connections[(connection.source_chart_id, connection.target_chart_id)] = (
            connection
        )
    atlas = object.__new__(RelationalGaugeAtlas)
    atlas.config = config
    # Edges exist only to derive the persisted state; the rehydrated atlas serves
    # charts/connections/distances and never re-derives anything from edges.
    atlas.edges = ()
    atlas._edge_types = ()
    atlas.node_ids = node_ids
    atlas.node_to_index = {node: index for index, node in enumerate(node_ids)}
    atlas.distances = distances
    atlas.chart_order = tuple(str(value) for value in content["chart_order"])
    atlas.charts = charts
    atlas.connections = connections
    if not atlas.charts:
        raise RelationalGaugeControllerArtifactError("persisted atlas has no charts")
    return atlas


def _expected_tensor_names(content: Mapping[str, Any]) -> set[str]:
    names = set(_LIFT_TENSOR_NAMES)
    names.add("atlas.distances")
    for index in range(len(content["charts"])):
        names.add(f"atlas.chart.{index}.coordinates")
        names.add(f"atlas.chart.{index}.eigenvalues")
        names.add(f"atlas.chart.{index}.support_distances")
    for index in range(len(content["connections"])):
        names.add(f"atlas.connection.{index}.transport")
    return names


def _config_payload(config: RelationalGaugeBundleConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["field_config"] = asdict(config.field_config)
    return payload


def _config_from_payload(value: Mapping[str, Any]) -> RelationalGaugeBundleConfig:
    payload = dict(value)
    field = payload.pop("field_config", None)
    if not isinstance(field, Mapping):
        raise RelationalGaugeControllerArtifactError("field config is absent")
    return RelationalGaugeBundleConfig(
        **payload,
        field_config=PressureMatchedFieldConfig(**dict(field)),
    )


def _risk_payload(field: PressureMatchedRiskField) -> list[dict[str, Any]]:
    return [
        {
            "observation_id": row.observation_id,
            "node_id": str(row.node_id),
            "family_fold": row.family_fold,
            "nuisance_key": list(row.nuisance_key),
            "outcome_class": row.outcome_class,
            "weight": float(row.weight),
        }
        for row in field.observations
    ]


def _query_payload(query: GaugeQueryState) -> dict[str, Any]:
    return {
        "chart_id": query.chart_id,
        "query_coordinates": query.query_coordinates.tolist(),
        "nearest_node_id": str(query.nearest_node_id),
        "nearest_node_distance": float(query.nearest_node_distance),
        "stress": float(query.stress),
        "support_status": bool(query.support_status),
        "support_reason": query.support_reason,
    }


def _lift_payload(
    lift: HorizontalLift,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    patch_ids = tuple(sorted(lift.patches))
    patches = [lift.patches[patch_id] for patch_id in patch_ids]
    tensors = {
        "lift_matrix": torch.from_numpy(
            np.stack([patch.matrix for patch in patches], axis=0).astype(
                np.float64, copy=False
            )
        ),
        "tangent_metric": torch.from_numpy(
            np.stack([patch.tangent_metric for patch in patches], axis=0).astype(
                np.float64, copy=False
            )
        ),
        "layer_fiber_norm_caps": torch.from_numpy(
            np.stack([patch.layer_fiber_norm_caps for patch in patches], axis=0).astype(
                np.float64, copy=False
            )
        ),
        "intrinsic_trust_radius": torch.tensor(
            [patch.intrinsic_trust_radius for patch in patches], dtype=torch.float64
        ),
        "weighted_relative_fit_error": torch.tensor(
            [patch.weighted_relative_fit_error for patch in patches],
            dtype=torch.float64,
        ),
        "weighted_relative_roundtrip_error": torch.tensor(
            [patch.weighted_relative_roundtrip_error for patch in patches],
            dtype=torch.float64,
        ),
        "condition_number": torch.tensor(
            [patch.condition_number for patch in patches], dtype=torch.float64
        ),
    }
    metadata = [
        {
            "chart_id": patch.chart_id,
            "support_sample_ids": list(patch.support_sample_ids),
            "support_family_folds": list(patch.support_family_folds),
        }
        for patch in patches
    ]
    return tensors, metadata


def _fold_binding(
    *,
    common_binding: Mapping[str, Any],
    fold: str,
    config: RelationalGaugeBundleConfig,
    opened_training_shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        **dict(common_binding),
        "held_out_family_fold": fold,
        "view": config.view,
        "graph_variant": "joint",
        "config": _config_payload(config),
        "opened_training_outcome_shards": [
            dict(value) for value in opened_training_shards
        ],
    }


def _validate_fold_payload(
    value: Mapping[str, Any],
    *,
    expected_binding: Mapping[str, Any] | None = None,
) -> None:
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != FOLD_KIND
        or value.get("status") != "success"
        or value.get("artifact_sha256") != _self_hash(value, "artifact_sha256")
    ):
        raise RelationalGaugeControllerArtifactError(
            "fold artifact schema, status, kind, or self-hash is invalid"
        )
    if expected_binding is not None and value.get("binding") != dict(expected_binding):
        raise RelationalGaugeControllerArtifactError("fold artifact binding changed")
    tensor = value.get("tensor_artifact")
    if not isinstance(tensor, Mapping):
        raise RelationalGaugeControllerArtifactError("tensor binding is absent")
    if not isinstance(value.get("atlas_content"), Mapping):
        raise RelationalGaugeControllerArtifactError(
            "persisted atlas content is absent (schema v1 artifact? rebuild the "
            "controller substrate with the current code)"
        )


def _complete_fold(
    path: Path,
    *,
    expected_binding: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            return None
        _validate_fold_payload(value, expected_binding=expected_binding)
        tensor = value["tensor_artifact"]
        tensor_path = path.parent / str(tensor["path"])
        if not tensor_path.is_file() or file_sha256(tensor_path) != tensor["sha256"]:
            return None
        return value
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
        return None


def _write_fold(
    artifact_root: Path,
    bundle: FoldGaugeControllerBundle,
    *,
    binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    if bundle.horizontal_lift is None:
        raise RelationalGaugeControllerArtifactError(
            "fold has no horizontal lift and cannot be persisted for control"
        )
    fold_root = artifact_root / bundle.held_out_family_fold
    tensor_path = fold_root / "controller.safetensors"
    tensors, lift_metadata = _lift_payload(bundle.horizontal_lift)
    atlas_tensors, atlas_content = _atlas_persist_payload(bundle.atlas)
    overlap = set(tensors) & set(atlas_tensors)
    if overlap:
        raise RelationalGaugeControllerArtifactError(
            f"atlas tensor names collide with lift tensors: {sorted(overlap)}"
        )
    tensors = {**tensors, **atlas_tensors}
    _atomic_safetensors(tensor_path, tensors)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": FOLD_KIND,
        "status": "success",
        "binding": dict(binding),
        "atlas_fingerprint": _atlas_fingerprint(bundle),
        "atlas_content": atlas_content,
        "risk_observations": _risk_payload(bundle.risk_field),
        "held_out_queries": {
            root_id: _query_payload(query)
            for root_id, query in sorted(bundle.held_out_queries.items())
        },
        "lift_patches": lift_metadata,
        "tensor_artifact": {
            "path": tensor_path.name,
            "sha256": file_sha256(tensor_path),
            "byte_count": tensor_path.stat().st_size,
            "tensor_shapes": {
                name: list(tensor.shape) for name, tensor in sorted(tensors.items())
            },
            "tensor_dtypes": {
                name: str(tensor.dtype) for name, tensor in sorted(tensors.items())
            },
        },
        "diagnostics": asdict(bundle.diagnostics),
    }
    payload["artifact_sha256"] = _self_hash(payload, "artifact_sha256")
    _validate_fold_payload(payload, expected_binding=binding)
    _atomic_json(fold_root / "bundle.json", payload)
    return payload


def _source_files(extra_paths: Sequence[Path]) -> dict[str, dict[str, str]]:
    paths = {
        "artifact": Path(__file__).resolve(),
        "bundle": Path(bundle_module.__file__).resolve(),
        "atlas": Path(atlas_module.__file__).resolve(),
        "horizontal_lift": Path(lift_module.__file__).resolve(),
        "risk_field": Path(field_module.__file__).resolve(),
        "rooted_graph_artifact": Path(graph_module.__file__).resolve(),
        "training_supervision": Path(training_module.__file__).resolve(),
        "rooted_star_store": Path(store_module.__file__).resolve(),
    }
    for path in extra_paths:
        resolved = Path(path).resolve()
        paths.setdefault(resolved.stem, resolved)
    return {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in sorted(paths.items())
    }


def build_relational_gauge_controller_artifact(
    index: RelationalPreStatusRootedStarIndex,
    graphs: LoadedRelationalPreStatusRootedGraphs,
    *,
    artifact_root: Path,
    expected_rooted_star_manifest_sha256: str,
    rooted_graph_artifact_root: Path,
    expected_rooted_graph_manifest_file_sha256: str,
    outcome_shard_root: Path,
    expected_outcome_shard_manifest_file_sha256: str,
    expected_source_report_file_sha256: str,
    config: RelationalGaugeBundleConfig | None = None,
    argv: Sequence[str] = (),
    extra_source_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Build or resume all five fold bundles without loading a model or GPU."""
    settings = config or RelationalGaugeBundleConfig()
    root = Path(artifact_root).resolve()
    bank_manifest = Path(index.artifact_root).resolve() / "manifest.json"
    expected_bank = _sha(
        expected_rooted_star_manifest_sha256,
        name="expected rooted-star manifest SHA-256",
    )
    if index.manifest_sha256 != expected_bank:
        raise RelationalGaugeControllerArtifactError(
            "rooted-star index does not bind the expected manifest"
        )
    graph_manifest = Path(rooted_graph_artifact_root).resolve() / "manifest.json"
    graph_binding = graphs.manifest.get("binding")
    if not isinstance(graph_binding, Mapping):
        raise RelationalGaugeControllerArtifactError("graph manifest has no binding")
    if graph_binding.get("rooted_star_manifest_file_sha256") != expected_bank:
        raise RelationalGaugeControllerArtifactError(
            "graph artifact is not bound to the requested rooted-star bank"
        )
    shard_manifest_path = Path(outcome_shard_root).resolve() / "manifest.json"
    source_report_sha = _sha(
        expected_source_report_file_sha256,
        name="expected source outcome-report SHA-256",
    )
    common_binding = {
        "rooted_star_manifest": _binding(
            bank_manifest, expected_sha256=expected_bank
        ),
        "rooted_graph_manifest": {
            **_binding(
                graph_manifest,
                expected_sha256=expected_rooted_graph_manifest_file_sha256,
            ),
            "content_sha256": _sha(
                graphs.manifest.get("manifest_sha256"),
                name="graph manifest content SHA-256",
            ),
        },
        "outcome_shard_manifest": _binding(
            shard_manifest_path,
            expected_sha256=expected_outcome_shard_manifest_file_sha256,
        ),
        "source_outcome_report_file_sha256": source_report_sha,
    }
    fold_files: dict[str, dict[str, str]] = {}
    fold_summaries: dict[str, Any] = {}
    for fold in FOLDS:
        supervision = build_fold_gauge_training_supervision(
            index,
            held_out_family_fold=fold,
            outcome_shard_root=outcome_shard_root,
            expected_outcome_shard_manifest_file_sha256=(
                expected_outcome_shard_manifest_file_sha256
            ),
            expected_source_report_file_sha256=source_report_sha,
        )
        binding = _fold_binding(
            common_binding=common_binding,
            fold=fold,
            config=settings,
            opened_training_shards=supervision.opened_training_shards,
        )
        fold_path = root / fold / "bundle.json"
        payload = _complete_fold(fold_path, expected_binding=binding)
        if payload is None:
            graph = graphs.fold_graph(settings.view, fold, "joint")
            atlas, _, _ = build_training_gauge_atlas(graph, config=settings)
            # Rebuild-determinism is asserted HERE, on the build machine, where it
            # is meaningful; the load path serves persisted bytes (schema v2).
            atlas_again, _, _ = build_training_gauge_atlas(graph, config=settings)
            if _atlas_content_fingerprint(atlas) != _atlas_content_fingerprint(
                atlas_again
            ):
                raise RelationalGaugeControllerArtifactError(
                    "atlas rebuild is nondeterministic on the build machine"
                )
            node_fibers = load_training_node_fibers(
                index,
                supervision,
                atlas,
                view=settings.view,
                held_out_family_fold=fold,
            )
            bundle = build_fold_gauge_controller_bundle(
                graph,
                supervision,
                node_fibers=node_fibers,
                config=settings,
            )
            payload = _write_fold(root, bundle, binding=binding)
        fold_files[fold] = {
            "path": str(fold_path.relative_to(root)),
            "sha256": file_sha256(fold_path),
            "tensor_path": str((fold_path.parent / "controller.safetensors").relative_to(root)),
            "tensor_sha256": str(payload["tensor_artifact"]["sha256"]),
        }
        fold_summaries[fold] = dict(payload["diagnostics"])

    source_files = _source_files(extra_source_paths)
    roundtrip_validation: dict[str, Any] = {}
    for fold in FOLDS:
        restored = load_fold_gauge_controller_artifact(
            root / fold,
            graphs.fold_graph(settings.view, fold, "joint"),
        )
        if restored.horizontal_lift is None:
            raise RelationalGaugeControllerArtifactError(
                "round-trip bundle lost its horizontal lift"
            )
        roundtrip_validation[fold] = {
            "status": "pass",
            "atlas_content_sha256": _atlas_fingerprint(restored)[
                "content_sha256"
            ],
            "lift_patch_count": len(restored.horizontal_lift.patches),
            "held_out_query_count": len(restored.held_out_queries),
        }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "status": "success",
        "argv": [str(value) for value in argv],
        "scope": {
            "evidence_label": "offline gauge-controller substrate",
            "causal_controller_claim": False,
            "universal_controller_claim": False,
            "statement": (
                "This artifact constructs the fold-safe gauge atlas, pressure-matched "
                "field, natural-transition horizontal lift, and held-out query attachment. "
                "It does not by itself establish behavioral control or universality."
            ),
        },
        "binding": common_binding,
        "config": _config_payload(settings),
        "fold_artifacts": fold_files,
        "fold_summaries": fold_summaries,
        "roundtrip_validation": roundtrip_validation,
        "source_files": source_files,
        "provenance": git_provenance(
            [Path(row["path"]) for row in source_files.values()]
            + [bank_manifest, graph_manifest, shard_manifest_path]
        ),
    }
    manifest["manifest_sha256"] = _self_hash(manifest, "manifest_sha256")
    _atomic_json(root / "manifest.json", manifest)
    return manifest


def validate_relational_gauge_controller_manifest(
    manifest: Mapping[str, Any],
) -> None:
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != MANIFEST_KIND
        or manifest.get("status") != "success"
        or manifest.get("manifest_sha256")
        != _self_hash(manifest, "manifest_sha256")
    ):
        raise RelationalGaugeControllerArtifactError(
            "manifest schema, status, kind, or self-hash is invalid"
        )
    folds = manifest.get("fold_artifacts")
    if not isinstance(folds, Mapping) or set(folds) != set(FOLDS):
        raise RelationalGaugeControllerArtifactError("manifest fold inventory is incomplete")


def _load_tensors(path: Path) -> dict[str, np.ndarray]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        if (
            metadata.get("schema_version") != str(SCHEMA_VERSION)
            or metadata.get("kind") != FOLD_KIND
        ):
            raise RelationalGaugeControllerArtifactError(
                "controller tensor metadata is invalid"
            )
        return {
            name: handle.get_tensor(name).detach().cpu().numpy().copy()
            for name in handle.keys()
        }


def load_fold_gauge_controller_artifact(
    fold_root: Path,
    graph: FoldExactRootedGraph,
) -> FoldGaugeControllerBundle:
    """Rehydrate one serving bundle from persisted content (no atlas recomputation)."""
    root = Path(fold_root).resolve()
    payload = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RelationalGaugeControllerArtifactError("fold artifact is not an object")
    _validate_fold_payload(payload)
    binding = payload.get("binding")
    if not isinstance(binding, Mapping):
        raise RelationalGaugeControllerArtifactError("fold binding is absent")
    if binding.get("held_out_family_fold") != graph.held_out_family_fold:
        raise RelationalGaugeControllerArtifactError("graph fold and artifact disagree")
    config_raw = binding.get("config")
    if not isinstance(config_raw, Mapping):
        raise RelationalGaugeControllerArtifactError("fold config is absent")
    config = _config_from_payload(config_raw)

    risk_rows = payload.get("risk_observations")
    if not isinstance(risk_rows, list) or not risk_rows:
        raise RelationalGaugeControllerArtifactError("risk observations are absent")
    risk_field = PressureMatchedRiskField(
        tuple(
            GaugeRiskObservation(
                observation_id=str(row["observation_id"]),
                node_id=str(row["node_id"]),
                family_fold=str(row["family_fold"]),
                nuisance_key=tuple(str(value) for value in row["nuisance_key"]),
                outcome_class=str(row["outcome_class"]),
                weight=float(row["weight"]),
            )
            for row in risk_rows
        ),
        config=config.field_config,
        held_out_family_fold=graph.held_out_family_fold,
    )

    tensor_binding = payload["tensor_artifact"]
    tensor_path = root / str(tensor_binding["path"])
    if file_sha256(tensor_path) != tensor_binding["sha256"]:
        raise RelationalGaugeControllerArtifactError("controller tensor file changed")
    tensors = _load_tensors(tensor_path)
    patch_metadata = payload.get("lift_patches")
    if not isinstance(patch_metadata, list) or not patch_metadata:
        raise RelationalGaugeControllerArtifactError("lift patch metadata is absent")
    atlas_content = payload["atlas_content"]
    if set(tensors) != _expected_tensor_names(atlas_content):
        raise RelationalGaugeControllerArtifactError("controller tensor inventory changed")
    atlas = _rehydrate_atlas(atlas_content, tensors)
    count = len(patch_metadata)
    if any(tensors[name].shape[0] != count for name in _LIFT_TENSOR_NAMES):
        raise RelationalGaugeControllerArtifactError("lift tensors disagree on patch count")
    patches = {}
    for index, row in enumerate(patch_metadata):
        patch = HorizontalLiftPatch(
            chart_id=str(row["chart_id"]),
            matrix=tensors["lift_matrix"][index],
            tangent_metric=tensors["tangent_metric"][index],
            intrinsic_trust_radius=float(tensors["intrinsic_trust_radius"][index]),
            layer_fiber_norm_caps=tensors["layer_fiber_norm_caps"][index],
            support_sample_ids=tuple(str(value) for value in row["support_sample_ids"]),
            support_family_folds=tuple(
                str(value) for value in row["support_family_folds"]
            ),
            weighted_relative_fit_error=float(
                tensors["weighted_relative_fit_error"][index]
            ),
            weighted_relative_roundtrip_error=float(
                tensors["weighted_relative_roundtrip_error"][index]
            ),
            condition_number=float(tensors["condition_number"][index]),
        )
        patches[patch.chart_id] = patch
    lift = HorizontalLift(patches)

    queries_raw = payload.get("held_out_queries")
    if not isinstance(queries_raw, Mapping):
        raise RelationalGaugeControllerArtifactError("held-out query inventory is absent")
    queries = {
        str(root_id): GaugeQueryState(
            chart_id=str(row["chart_id"]),
            query_coordinates=np.asarray(row["query_coordinates"], dtype=np.float64),
            nearest_node_id=str(row["nearest_node_id"]),
            nearest_node_distance=float(row["nearest_node_distance"]),
            stress=float(row["stress"]),
            support_status=bool(row["support_status"]),
            support_reason=str(row["support_reason"]),
        )
        for root_id, row in queries_raw.items()
    }
    diagnostics_payload = dict(payload["diagnostics"])
    diagnostics_payload["chart_stress_quantiles"] = tuple(
        float(value) for value in diagnostics_payload["chart_stress_quantiles"]
    )
    diagnostics = FoldGaugeBundleDiagnostics(**diagnostics_payload)
    bundle = FoldGaugeControllerBundle(
        held_out_family_fold=graph.held_out_family_fold,
        view=config.view,
        atlas=atlas,
        risk_field=risk_field,
        horizontal_lift=lift,
        held_out_queries=queries,
        diagnostics=diagnostics,
    )
    if _atlas_fingerprint(bundle) != payload.get("atlas_fingerprint"):
        raise RelationalGaugeControllerArtifactError(
            "rehydrated atlas differs from persisted fingerprint"
        )
    return bundle


def build_relational_gauge_controller_report(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    validate_relational_gauge_controller_manifest(manifest)
    folds = manifest["fold_summaries"]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "status": "success",
        "scope": dict(manifest["scope"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "folds": folds,
        "aggregate": {
            "fold_count": len(folds),
            "training_node_count_range": [
                min(int(row["training_node_count"]) for row in folds.values()),
                max(int(row["training_node_count"]) for row in folds.values()),
            ],
            "mean_lift_patch_coverage": float(
                np.mean([float(row["lift_patch_coverage"]) for row in folds.values()])
            ),
            "held_out_query_count": sum(
                int(row["held_out_query_count"]) for row in folds.values()
            ),
            "held_out_query_in_support_count": sum(
                int(row["held_out_query_in_support_count"])
                for row in folds.values()
            ),
            "held_out_query_field_defined_count": sum(
                int(row["held_out_query_field_defined_count"])
                for row in folds.values()
            ),
            "held_out_query_field_evaluated_count": sum(
                int(row["held_out_query_field_evaluated_count"])
                for row in folds.values()
            ),
            "held_out_query_lift_defined_count": sum(
                int(row["held_out_query_lift_defined_count"])
                for row in folds.values()
            ),
        },
    }
    report["report_sha256"] = _self_hash(report, "report_sha256")
    return report


def render_relational_gauge_controller_markdown(report: Mapping[str, Any]) -> str:
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("kind") != REPORT_KIND
        or report.get("status") != "success"
        or report.get("report_sha256") != _self_hash(report, "report_sha256")
    ):
        raise RelationalGaugeControllerArtifactError("build report is invalid")
    lines = [
        "# Relational gauge-controller substrate",
        "",
        "**Scope:** Offline fold-safe construction only. This does not establish causal or universal control.",
        "",
        f"Report SHA-256: `{report['report_sha256']}`",
        "",
        "| Fold | Nodes | Charts | Connections | Lift patches | Lift coverage | Queries | In support | Field evaluated | Field defined |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold, row in sorted(report["folds"].items()):
        lines.append(
            f"| `{fold}` | {row['training_node_count']} | {row['chart_count']} | "
            f"{row['connection_count']} | {row['lift_patch_count']} | "
            f"{float(row['lift_patch_coverage']):.3f} | {row['held_out_query_count']} | "
            f"{row['held_out_query_in_support_count']} | "
            f"{row['held_out_query_field_evaluated_count']} | "
            f"{row['held_out_query_field_defined_count']} |"
        )
    aggregate = report["aggregate"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Across {aggregate['fold_count']} folds, the artifact attaches "
            f"{aggregate['held_out_query_in_support_count']}/"
            f"{aggregate['held_out_query_count']} held-out queries within local chart support. "
            f"The pressure-matched field was intentionally not evaluated against sealed "
            f"held-out nuisance/outcome rows during construction "
            f"({aggregate['held_out_query_field_evaluated_count']} evaluated; "
            f"{aggregate['held_out_query_field_defined_count']} defined), and the "
            f"horizontal lift is available for {aggregate['held_out_query_lift_defined_count']}.",
            "",
            "These are construction/support diagnostics. Behavioral efficacy requires the separately reviewed closed-loop causal comparison.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "FOLD_KIND",
    "MANIFEST_KIND",
    "REPORT_KIND",
    "RelationalGaugeControllerArtifactError",
    "build_relational_gauge_controller_artifact",
    "build_relational_gauge_controller_report",
    "load_fold_gauge_controller_artifact",
    "render_relational_gauge_controller_markdown",
    "validate_relational_gauge_controller_manifest",
]
