from __future__ import annotations

import numpy as np

from geoprobe.geometry.relational_pre_status_rooted_graph import (
    RootedStarNode,
    build_fold_candidate_schedule,
    build_fold_exact_graph,
)
from geoprobe.geometry.relational_pre_status_rooted_metric import (
    RootedStarDistance,
    RootedStarMetricInput,
)


def _metric(value: float) -> RootedStarMetricInput:
    residual = np.asarray([[0.0, 1.0 + value, 2.0 + value]], dtype=float)
    raw = np.asarray([[[1.0 + value, 2.0, 3.0]]], dtype=float)
    attention = raw / raw.sum(axis=2, keepdims=True)
    annotations = np.asarray([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
    return RootedStarMetricInput(residual, attention, annotations)


def _nodes() -> tuple[RootedStarNode, ...]:
    rows = []
    for fold_index in range(1, 6):
        for item in range(4):
            value = fold_index + item / 10.0
            rows.append(
                RootedStarNode(
                    node_id=f"n{fold_index}-{item}",
                    family=f"f{fold_index}",
                    family_fold=f"outer_{fold_index}",
                    descriptor=np.asarray([value, value * value]),
                )
            )
    return tuple(rows)


def test_candidate_schedule_has_no_heldout_targets() -> None:
    schedule = build_fold_candidate_schedule(
        _nodes(), held_out_family_fold="outer_1", candidate_width=8, calibration_pair_count=8
    )
    assert set(schedule.query_edges) == {f"n1-{item}" for item in range(4)}
    for edges in schedule.query_edges.values():
        assert all(not edge.target_id.startswith("n1-") for edge in edges)
    for source, edges in schedule.training_edges.items():
        assert not source.startswith("n1-")
        assert all(not edge.target_id.startswith("n1-") for edge in edges)


def test_schedule_is_deterministic() -> None:
    first = build_fold_candidate_schedule(
        _nodes(), held_out_family_fold="outer_2", candidate_width=8, calibration_pair_count=8
    )
    second = build_fold_candidate_schedule(
        tuple(reversed(_nodes())),
        held_out_family_fold="outer_2",
        candidate_width=8,
        calibration_pair_count=8,
    )
    assert first.query_edges == second.query_edges
    assert first.training_edges == second.training_edges
    assert first.calibration_pairs == second.calibration_pairs


def test_exact_graph_uses_training_scale_and_top_width() -> None:
    nodes = _nodes()
    metrics = {
        node.node_id: _metric(float(node.descriptor[0]) / 100.0) for node in nodes
    }
    schedule = build_fold_candidate_schedule(
        nodes, held_out_family_fold="outer_3", candidate_width=8, calibration_pair_count=12
    )
    graph = build_fold_exact_graph(
        schedule, load_metric_input=metrics.__getitem__, graph_width=3
    )
    assert graph.graph_width == 3
    assert all(len(edges) == 3 for edges in graph.query_edges.values())
    assert all(len(edges) == 3 for edges in graph.training_edges.values())
    assert graph.scaler.residual_scale > 0.0
    assert graph.scaler.attention_scale > 0.0


def test_exact_graph_accepts_quotient_pair_distance() -> None:
    nodes = _nodes()
    schedule = build_fold_candidate_schedule(
        nodes,
        held_out_family_fold="outer_1",
        candidate_width=8,
        calibration_pair_count=8,
    )
    vectors = {node.node_id: node.descriptor for node in nodes}

    def distance(left: str, right: str) -> RootedStarDistance:
        value = float(np.linalg.norm(vectors[left] - vectors[right])) + 0.1
        return RootedStarDistance(
            residual=value,
            attention_head_set=value / 2.0,
        )

    graph = build_fold_exact_graph(
        schedule,
        pair_distance=distance,
        graph_width=3,
    )
    assert graph.graph_width == 3
    assert all(len(edges) == 3 for edges in graph.query_edges.values())
    parallel = build_fold_exact_graph(
        schedule,
        pair_distance=distance,
        graph_width=3,
        distance_workers=4,
    )
    assert parallel.query_edges == graph.query_edges
    assert parallel.training_edges == graph.training_edges
    assert parallel.scaler == graph.scaler
