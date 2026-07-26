from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from geoprobe.control.relational_intrinsic_risk_field import PressureMatchedFieldConfig
from geoprobe.eval import relational_gauge_controller_artifact as artifact
from geoprobe.eval.relational_gauge_bundle import (
    RelationalGaugeBundleConfig,
    build_fold_gauge_controller_bundle,
)
from geoprobe.eval.relational_pre_status_risk_field import PreStatusRiskEvent
from geoprobe.geometry.relational_pre_status_rooted_graph import (
    ExactGraphEdge,
    FoldExactRootedGraph,
)
from geoprobe.geometry.relational_pre_status_rooted_metric import RootedStarMetricScaler


VIEW = "intervention_masked_action_free"


def _edge(source: str, target: str, score: float, rank: int) -> ExactGraphEdge:
    return ExactGraphEdge(source, target, rank, score, score, score, rank)


def _graph() -> FoldExactRootedGraph:
    points = np.column_stack(
        [
            np.cos(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)),
            np.sin(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)),
        ]
    )
    node_ids = tuple(f"n{index}" for index in range(8))
    training = {}
    for index, source in enumerate(node_ids):
        scored = sorted(
            (
                (float(np.linalg.norm(points[index] - points[target])), node_ids[target])
                for target in range(8)
                if target != index
            )
        )
        training[source] = tuple(
            _edge(source, target, score, rank)
            for rank, (score, target) in enumerate(scored, 1)
        )
    queries = {
        "q0": tuple(
            _edge("q0", target, score, rank)
            for rank, (score, target) in enumerate(
                sorted(
                    (
                        (float(np.linalg.norm(np.asarray([0.9, 0.1]) - points[index])), node_ids[index])
                        for index in range(8)
                    )
                )[:4],
                1,
            )
        )
    }
    return FoldExactRootedGraph(
        held_out_family_fold="outer_1",
        graph_width=7,
        query_edges=queries,
        training_edges=training,
        scaler=RootedStarMetricScaler(1.0, 1.0),
        candidate_pair_count=1,
        exact_pair_count=1,
    )


def _supervision() -> SimpleNamespace:
    outcomes = ("HONEST", "DECEPTIVE", "SKIP", "NO_ACTION")
    rows = tuple(
        PreStatusRiskEvent(
            event_id=f"e{index}",
            root_id=f"n{index % 8}",
            family=f"family{index % 4}",
            family_fold=f"outer_{2 + index % 4}",
            outcome_class=outcomes[index % len(outcomes)],
            nuisance_key=("program:A", "dose:3"),
        )
        for index in range(32)
    )
    heldout = PreStatusRiskEvent(
        event_id="held",
        root_id="q0",
        family="held",
        family_fold="outer_1",
        outcome_class="DECEPTIVE",
        nuisance_key=("program:A", "dose:3"),
    )
    return SimpleNamespace(
        risk_events_by_view={VIEW: (*rows, heldout)},
        honestward_observations_by_view={VIEW: tuple()},
        nodes_by_id={
            f"n{index}": SimpleNamespace(family_fold=f"outer_{2 + index % 4}")
            for index in range(8)
        },
    )


def _config() -> RelationalGaugeBundleConfig:
    return RelationalGaugeBundleConfig(
        fixed_rank=2,
        support_radius_quantile=0.9,
        support_radius_multiplier=2.0,
        minimum_chart_support=4,
        maximum_chart_support=8,
        minimum_overlap=3,
        minimum_lift_samples=4,
        field_config=PressureMatchedFieldConfig(
            minimum_support_nodes=3,
            minimum_effective_observations=2.0,
        ),
    )


def _fibers() -> dict[str, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    maps = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]],
            [[0.5, 0.5], [1.0, -0.5], [-0.25, 1.0]],
        ]
    )
    return {
        f"n{index}": np.einsum("lhr,r->lh", maps, point).astype(np.float32)
        for index, point in enumerate(points)
    }


def test_fold_artifact_roundtrip_is_exact_and_detects_corruption(tmp_path: Path) -> None:
    graph = _graph()
    config = _config()
    bundle = build_fold_gauge_controller_bundle(
        graph,
        _supervision(),
        node_fibers=_fibers(),
        config=config,
    )
    config_payload = asdict(config)
    config_payload["field_config"] = asdict(config.field_config)
    binding = {
        "held_out_family_fold": "outer_1",
        "view": VIEW,
        "graph_variant": "joint",
        "config": config_payload,
    }
    artifact._write_fold(tmp_path, bundle, binding=binding)
    loaded = artifact.load_fold_gauge_controller_artifact(tmp_path / "outer_1", graph)
    assert loaded.diagnostics == bundle.diagnostics
    assert artifact._atlas_fingerprint(loaded) == artifact._atlas_fingerprint(bundle)
    assert loaded.horizontal_lift is not None
    assert bundle.horizontal_lift is not None
    for chart_id, patch in bundle.horizontal_lift.patches.items():
        restored = loaded.horizontal_lift.patch(chart_id)
        assert np.array_equal(restored.matrix, patch.matrix)
        assert np.array_equal(restored.tangent_metric, patch.tangent_metric)

    tensor_path = tmp_path / "outer_1" / "controller.safetensors"
    data = bytearray(tensor_path.read_bytes())
    data[-1] ^= 1
    tensor_path.write_bytes(data)
    with pytest.raises(artifact.RelationalGaugeControllerArtifactError, match="changed"):
        artifact.load_fold_gauge_controller_artifact(tmp_path / "outer_1", graph)


def _write_test_fold(tmp_path: Path):
    graph = _graph()
    config = _config()
    bundle = build_fold_gauge_controller_bundle(
        graph,
        _supervision(),
        node_fibers=_fibers(),
        config=config,
    )
    config_payload = asdict(config)
    config_payload["field_config"] = asdict(config.field_config)
    binding = {
        "held_out_family_fold": "outer_1",
        "view": VIEW,
        "graph_variant": "joint",
        "config": config_payload,
    }
    artifact._write_fold(tmp_path, bundle, binding=binding)
    return graph


def test_load_never_recomputes_the_atlas(tmp_path: Path, monkeypatch) -> None:
    # Schema v2 regression for the pod failure: load must serve persisted atlas
    # bytes and never call the platform-dependent rebuild.
    graph = _write_test_fold(tmp_path)

    def _forbidden(*args, **kwargs):
        raise AssertionError("load must not rebuild the atlas")

    monkeypatch.setattr(artifact, "build_training_gauge_atlas", _forbidden)
    loaded = artifact.load_fold_gauge_controller_artifact(tmp_path / "outer_1", graph)
    assert loaded.atlas.charts
    assert loaded.atlas.get_chart(next(iter(loaded.atlas.charts)))


def test_load_rejects_payload_without_persisted_atlas(tmp_path: Path) -> None:
    import json as json_module

    graph = _write_test_fold(tmp_path)
    bundle_path = tmp_path / "outer_1" / "bundle.json"
    payload = json_module.loads(bundle_path.read_text(encoding="utf-8"))
    payload.pop("atlas_content")
    payload["artifact_sha256"] = artifact._self_hash(payload, "artifact_sha256")
    bundle_path.write_text(json_module.dumps(payload), encoding="utf-8")
    with pytest.raises(
        artifact.RelationalGaugeControllerArtifactError, match="persisted atlas content"
    ):
        artifact.load_fold_gauge_controller_artifact(tmp_path / "outer_1", graph)


def test_load_rejects_tampered_fingerprint(tmp_path: Path) -> None:
    import json as json_module

    graph = _write_test_fold(tmp_path)
    bundle_path = tmp_path / "outer_1" / "bundle.json"
    payload = json_module.loads(bundle_path.read_text(encoding="utf-8"))
    payload["atlas_fingerprint"]["content_sha256"] = "0" * 64
    payload["artifact_sha256"] = artifact._self_hash(payload, "artifact_sha256")
    bundle_path.write_text(json_module.dumps(payload), encoding="utf-8")
    with pytest.raises(
        artifact.RelationalGaugeControllerArtifactError,
        match="differs from persisted fingerprint",
    ):
        artifact.load_fold_gauge_controller_artifact(tmp_path / "outer_1", graph)
