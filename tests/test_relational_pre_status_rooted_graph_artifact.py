"""Synthetic contracts for sealed rooted pre-status quotient graph artifacts."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from geoprobe.data.relational_pre_status_rooted_star import VIEWS
from geoprobe.geometry.relational_pre_status_rooted_graph import (
    CandidateEdge,
    ExactGraphEdge,
    FoldExactRootedGraph,
)
from geoprobe.geometry.relational_pre_status_rooted_metric import (
    RootedStarDistance,
    RootedStarMetricInput,
    RootedStarMetricScaler,
    rooted_star_energy_quotient,
)
from geoprobe.io import file_sha256


@dataclass(frozen=True)
class _Reference:
    geometry_sha256: str
    value: float


def _metric(_: object, reference: _Reference) -> RootedStarMetricInput:
    return RootedStarMetricInput(
        np.asarray([[reference.value]], dtype=float),
        np.asarray([[[1.0]]], dtype=float),
        np.asarray([[0.0, 0.0]], dtype=float),
    )


def _quotient() -> SimpleNamespace:
    nodes = []
    mapping: dict[str, dict[str, str]] = {}
    for index in range(1, 6):
        event = f"event-{index}"
        mapping[event] = {}
        for view_index, view in enumerate(VIEWS):
            node_id = f"{view}-{index}"
            mapping[event][view] = node_id
            nodes.append(SimpleNamespace(
                node_id=node_id, view=view, family=f"family-{index}",
                family_fold=f"outer_{index}",
                representative_references=(
                    _Reference(f"{node_id}-a", float(index + view_index)),
                    _Reference(f"{node_id}-b", float(index + view_index) + 0.1),
                ),
            ))
    return SimpleNamespace(nodes=tuple(nodes), event_to_node_ids=mapping)


def _schedule(nodes: tuple[object, ...], *, held_out_family_fold: str, **_: object) -> SimpleNamespace:
    training = [node for node in nodes if node.family_fold != held_out_family_fold]
    queries = [node for node in nodes if node.family_fold == held_out_family_fold]

    def edge(source: object, target: object) -> CandidateEdge:
        return CandidateEdge(source.node_id, target.node_id, 1, 0.0)

    return SimpleNamespace(
        held_out_family_fold=held_out_family_fold,
        candidate_width=64,
        descriptor_center=np.zeros_like(nodes[0].descriptor),
        descriptor_scale=np.ones_like(nodes[0].descriptor),
        query_edges={node.node_id: (edge(node, training[0]),) for node in queries},
        training_edges={node.node_id: (edge(node, next(target for target in training if target.node_id != node.node_id)),) for node in training},
    )


def _exact(schedule: SimpleNamespace, *, pair_distance: object, **_: object) -> FoldExactRootedGraph:
    assert callable(pair_distance)

    def rows(values: dict[str, tuple[CandidateEdge, ...]]) -> dict[str, tuple[ExactGraphEdge, ...]]:
        return {
            source: tuple(
                ExactGraphEdge(edge.source_id, edge.target_id, 1, 1.0, 1.0, 1.0, edge.rank)
                for edge in edges
            )
            for source, edges in values.items()
        }

    return FoldExactRootedGraph(
        schedule.held_out_family_fold, 8, rows(schedule.query_edges), rows(schedule.training_edges),
        RootedStarMetricScaler(1.0, 1.0), 1, 1,
    )


def test_build_is_outcome_free_uses_energy_and_recovers_a_corrupt_fold(tmp_path: Path, monkeypatch) -> None:
    from geoprobe.eval import relational_pre_status_rooted_graph_artifact as artifact

    bank = tmp_path / "bank"
    bank.mkdir()
    bank_manifest = bank / "manifest.json"
    bank_manifest.write_text("{}\n", encoding="utf-8")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen label-free protocol\n", encoding="utf-8")
    index = SimpleNamespace(artifact_root=bank, manifest_sha256=file_sha256(bank_manifest))
    monkeypatch.setattr(artifact, "build_label_free_prefix_state_quotient", lambda _: _quotient())
    monkeypatch.setattr(artifact, "build_fold_candidate_schedule", _schedule)
    calls = {"exact": 0, "energy": 0}

    def exact(*args, **kwargs):
        calls["exact"] += 1
        return _exact(*args, **kwargs)

    def raw(left: RootedStarMetricInput, right: RootedStarMetricInput) -> RootedStarDistance:
        return RootedStarDistance(abs(float(left.residual_root_distances[0, 0] - right.residual_root_distances[0, 0])), 0.2)

    def energy(*args, **kwargs):
        calls["energy"] += 1
        return rooted_star_energy_quotient(*args, **kwargs)

    monkeypatch.setattr(artifact, "build_fold_exact_graph", exact)
    monkeypatch.setattr(artifact, "rooted_star_distance", raw)
    monkeypatch.setattr(artifact, "rooted_star_energy_quotient", energy)
    out = tmp_path / "graphs"
    kwargs = dict(
        out_dir=out,
        expected_rooted_star_manifest_sha256=file_sha256(bank_manifest),
        frozen_protocol_path=protocol,
        expected_frozen_protocol_sha256=file_sha256(protocol),
        expected_state_count_per_view=5,
        expected_event_count=5,
        metric_loader=_metric,
    )
    artifact.build_relational_pre_status_rooted_graph_artifacts(index, **kwargs)
    assert calls["energy"] > 0
    loaded = artifact.load_relational_pre_status_rooted_graph_artifacts(out)
    for view in VIEWS:
        for fold in (f"outer_{item}" for item in range(1, 6)):
            primary = loaded.variant(view, fold)
            assert primary == loaded.folds_by_view[view][fold]["graph"]["variants"]["joint"]
            assert set(loaded.folds_by_view[view][fold]["graph"]["variants"]) == {"joint", "residual_only", "attention_only"}
            rehydrated = loaded.fold_graph(view, fold)
            assert rehydrated.held_out_family_fold == fold
            assert sum(map(len, rehydrated.query_edges.values())) == len(primary["query_to_training"])
            held = fold
            node_folds = loaded.folds_by_view[view][fold]["graph"]["node_family_folds"]
            assert all(node_folds[row["target_id"]] != held for row in primary["query_to_training"])
    assert set(loaded.evaluation_inventory()[VIEWS[0]]) == {
        "joint",
        "residual_only",
        "attention_only",
    }
    text = (out / "manifest.json").read_text(encoding="utf-8")
    assert "HONEST" not in text and "DECEPTIVE" not in text
    corrupt = out / "views" / VIEWS[0] / "outer_1.json"
    corrupt.write_text("{not json", encoding="utf-8")
    calls["exact"] = 0
    artifact.build_relational_pre_status_rooted_graph_artifacts(index, **kwargs)
    assert calls["exact"] == 1
    assert artifact.load_relational_pre_status_rooted_graph_artifacts(out).manifest["status"] == "success"


def test_raw_distance_cache_batches_threaded_writes_durably(tmp_path: Path) -> None:
    from geoprobe.eval import relational_pre_status_rooted_graph_artifact as artifact

    path = tmp_path / "distances.sqlite"
    binding = {"bank": "sealed", "view": "masked"}
    cache = artifact._RawDistanceCache(path, binding)

    def write(index: int) -> None:
        cache.put(
            f"left-{index:03d}",
            f"right-{index:03d}",
            RootedStarDistance(float(index + 1), float(index + 2)),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        tuple(executor.map(write, range(300)))
    cache.close()
    reopened = artifact._RawDistanceCache(path, binding)
    assert reopened.get("left-299", "right-299") == RootedStarDistance(300.0, 301.0)
    reopened.close()
