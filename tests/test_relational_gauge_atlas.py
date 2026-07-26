"""Tests for relational discrete intrinsic gauge atlas and holonomy helpers."""

from __future__ import annotations

import math
import numpy as np
import pytest

from geoprobe.geometry.relational_gauge_atlas import (
    RelationalGaugeAtlas,
    RelationalGaugeAtlasConfig,
    RelationalGaugeAtlasError,
    WeightedGraphEdge,
    GraphEdgeType,
    gauge_transform_chart,
    gauge_transform_connection,
    conjugacy_invariant_connection_spectrum,
)


def _build_complete_graph_edges(points: np.ndarray) -> tuple[WeightedGraphEdge, ...]:
    edge_type = GraphEdgeType("intrinsic")
    edges: list[WeightedGraphEdge] = []
    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            weight = float(np.linalg.norm(points[i] - points[j]))
            edges.append(WeightedGraphEdge(i, j, weight, edge_type=edge_type))
    return tuple(edges)


def _pairwise(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _query_distances(points: np.ndarray, query: np.ndarray) -> dict[int, float]:
    return {index: float(np.linalg.norm(point - query)) for index, point in enumerate(points)}


def test_flat_recovery_from_complete_square_graph() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    atlas = RelationalGaugeAtlas(
        edges=_build_complete_graph_edges(points),
        config=RelationalGaugeAtlasConfig(
            fixed_rank=2,
            support_radius=3.0,
            min_chart_support=3,
            max_chart_support=8,
            min_overlap=2,
        ),
    )

    chart = atlas.get_chart("chart:0")
    support = chart.support_ids
    support_indices = np.array([atlas.node_to_index[node] for node in support], dtype=int)
    reconstructed = _pairwise(chart.coordinates)
    expected = _pairwise(points[support_indices])

    np.testing.assert_allclose(chart.eigenvalues, np.array([1.0, 1.0]), atol=1e-8)
    assert chart.stress <= 1e-8
    np.testing.assert_allclose(reconstructed, expected, atol=1e-7)
    assert len(chart.support_ids) == 4
    assert chart.center_node_id == 0
    assert chart.support_diameter == pytest.approx(np.sqrt(2.0), rel=0, abs=1e-12)
    assert chart.support_radius == pytest.approx(3.0)
    np.testing.assert_allclose(np.asarray(chart.support_distances, dtype=np.float64), _pairwise(points[support_indices]))
    assert chart.trust_radius == pytest.approx(chart.support_diameter)


def test_deterministic_chart_build_is_reproducible() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    config = RelationalGaugeAtlasConfig(
        fixed_rank=2,
        support_radius=3.0,
        min_chart_support=3,
        max_chart_support=8,
        min_overlap=2,
    )

    first = RelationalGaugeAtlas(_build_complete_graph_edges(points), config=config)
    second = RelationalGaugeAtlas(_build_complete_graph_edges(points), config=config)

    assert first.chart_order == second.chart_order
    for chart_id in first.chart_order:
        left = first.get_chart(chart_id)
        right = second.get_chart(chart_id)
        assert left.support_ids == right.support_ids
        np.testing.assert_allclose(left.coordinates, right.coordinates)
        np.testing.assert_allclose(left.eigenvalues, right.eigenvalues)
        assert left.stress == pytest.approx(right.stress)


def test_disconnected_charts_fail_closed() -> None:
    edge_type = GraphEdgeType("intrinsic")
    edges = (
        WeightedGraphEdge(0, 1, 1.0, edge_type=edge_type),
        WeightedGraphEdge(2, 3, 1.0, edge_type=edge_type),
    )
    atlas = RelationalGaugeAtlas(
        edges=edges,
        config=RelationalGaugeAtlasConfig(
            fixed_rank=1,
            support_radius=2.0,
            min_chart_support=2,
            max_chart_support=4,
            min_overlap=2,
        ),
    )

    assert atlas.chart_for_node(0) != atlas.chart_for_node(2)
    with pytest.raises(RelationalGaugeAtlasError):
        atlas.get_connection(atlas.chart_for_node(0), atlas.chart_for_node(2))


def test_reverse_connection_is_transpose_and_path_composition_is_identity() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    atlas = RelationalGaugeAtlas(
        edges=_build_complete_graph_edges(points),
        config=RelationalGaugeAtlasConfig(
            fixed_rank=2,
            support_radius=3.0,
            min_chart_support=3,
            max_chart_support=6,
            min_overlap=2,
        ),
    )

    source = atlas.chart_for_node(0)
    target = atlas.chart_for_node(1)

    forward = atlas.get_connection(source, target)
    reverse = atlas.get_connection(target, source)

    np.testing.assert_allclose(reverse.transport, forward.transport.T, atol=1e-8)

    back_and_forth = RelationalGaugeAtlas.compose_connection_path((forward, reverse))
    np.testing.assert_allclose(back_and_forth.transport, np.eye(atlas.fixed_rank), atol=1e-8)


def test_nontrivial_curved_path_composition_matches_direct_transport() -> None:
    angles = np.linspace(0.0, math.pi / 2.0, 5)
    points = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float64)

    atlas = RelationalGaugeAtlas(
        edges=_build_complete_graph_edges(points),
        config=RelationalGaugeAtlasConfig(
            fixed_rank=2,
            support_radius=0.90,
            min_chart_support=3,
            max_chart_support=4,
            min_overlap=2,
        ),
    )

    c0 = atlas.chart_for_node(0)
    c1 = atlas.chart_for_node(1)
    c2 = atlas.chart_for_node(2)

    path = RelationalGaugeAtlas.compose_connection_path((atlas.get_connection(c0, c1), atlas.get_connection(c1, c2)))
    direct = atlas.get_connection(c0, c2)

    direct_spectrum = conjugacy_invariant_connection_spectrum(direct.transport)
    path_spectrum = conjugacy_invariant_connection_spectrum(path.transport)
    np.testing.assert_allclose(np.array(path_spectrum), np.array(direct_spectrum), atol=1e-8)
    assert np.linalg.norm(path.transport - np.eye(atlas.fixed_rank)) > 1e-3


def test_gauge_transform_covariance_of_chart_and_connection() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    atlas = RelationalGaugeAtlas(
        edges=_build_complete_graph_edges(points),
        config=RelationalGaugeAtlasConfig(
            fixed_rank=2,
            support_radius=3.0,
            min_chart_support=3,
            max_chart_support=6,
            min_overlap=2,
        ),
    )
    source_id = atlas.chart_for_node(0)
    target_id = atlas.chart_for_node(1)
    connection = atlas.get_connection(source_id, target_id)

    rng = np.random.default_rng(0)
    q_source, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    q_target, _ = np.linalg.qr(rng.normal(size=(2, 2)))

    source = atlas.get_chart(source_id)
    target = atlas.get_chart(target_id)
    source_prime = gauge_transform_chart(source, q_source)
    target_prime = gauge_transform_chart(target, q_target)
    connection_prime = gauge_transform_connection(
        connection,
        source_basis=q_source,
        target_basis=q_target,
    )

    overlap = tuple(node for node in source.support_ids if node in target.node_to_index)
    anchor_one, anchor_two = overlap[0], overlap[1]
    source_delta = source.coordinate_for(anchor_two) - source.coordinate_for(anchor_one)
    target_delta = target.coordinate_for(anchor_two) - target.coordinate_for(anchor_one)

    source_delta_prime = source_prime.coordinate_for(anchor_two) - source_prime.coordinate_for(anchor_one)
    target_delta_prime = target_prime.coordinate_for(anchor_two) - target_prime.coordinate_for(anchor_one)
    np.testing.assert_allclose(source_delta_prime, source_delta @ q_source.T, atol=1e-8)
    np.testing.assert_allclose(target_delta_prime, target_delta @ q_target.T, atol=1e-8)

    left = q_target @ target_delta[:, None]
    right = connection_prime.transport @ (q_source @ source_delta[:, None])
    np.testing.assert_allclose(left, right, atol=1e-8)


def test_holonomy_spectrum_is_conjugacy_invariant() -> None:
    angles = np.linspace(0.0, math.pi / 2.0, 5)
    points = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float64)

    atlas = RelationalGaugeAtlas(
        edges=_build_complete_graph_edges(points),
        config=RelationalGaugeAtlasConfig(
            fixed_rank=2,
            support_radius=0.90,
            min_chart_support=3,
            max_chart_support=4,
            min_overlap=2,
        ),
    )

    c0 = atlas.chart_for_node(0)
    c1 = atlas.chart_for_node(1)
    c2 = atlas.chart_for_node(2)

    path = (
        atlas.get_connection(c0, c1),
        atlas.get_connection(c1, c2),
        atlas.get_connection(c2, c0),
    )
    baseline = atlas.holonomy_spectrum(path)

    rng = np.random.default_rng(7)
    q0, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    q1, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    q2, _ = np.linalg.qr(rng.normal(size=(2, 2)))

    q = {c0: q0, c1: q1, c2: q2}
    transformed = tuple(
        gauge_transform_connection(
            c,
            source_basis=q[c.source_chart_id],
            target_basis=q[c.target_chart_id],
        )
        for c in path
    )
    changed = atlas.holonomy_spectrum(transformed)

    np.testing.assert_allclose(np.array(changed), np.array(baseline), atol=1e-8)


def test_chart_coordinate_off_support_and_trust_radius_metadata() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    atlas = RelationalGaugeAtlas(
        edges=_build_complete_graph_edges(points),
        config=RelationalGaugeAtlasConfig(
            fixed_rank=2,
            support_radius=3.0,
            min_chart_support=3,
            max_chart_support=8,
            min_overlap=2,
        ),
    )
    chart = atlas.get_chart(atlas.chart_for_node(0))

    with pytest.raises(RelationalGaugeAtlasError):
        chart.coordinate_for(99)

    support_indices = np.array([atlas.node_to_index[node] for node in chart.support_ids], dtype=int)
    full = atlas.distances[np.ix_(support_indices, support_indices)]
    assert chart.support_diameter == pytest.approx(np.max(full), rel=0, abs=1e-12)
    assert chart.support_radius == pytest.approx(3.0)


def test_gauge_query_state_recovers_out_of_sample_point() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    atlas = RelationalGaugeAtlas(
        edges=_build_complete_graph_edges(points),
        config=RelationalGaugeAtlasConfig(
            fixed_rank=2,
            support_radius=3.0,
            min_chart_support=3,
            max_chart_support=8,
            min_overlap=2,
        ),
    )

    query = np.array([0.2, 0.3], dtype=np.float64)
    distances = _query_distances(points, query)
    state = atlas.locate_query(distances)
    chart = atlas.get_chart(state.chart_id)
    reconstructed = np.linalg.norm(chart.coordinates - state.query_coordinates, axis=1)
    expected = np.array([distances[node] for node in chart.support_ids], dtype=np.float64)

    np.testing.assert_allclose(reconstructed, expected, atol=1e-6)
    assert state.support_status is True
    assert state.nearest_node_id == chart.center_node_id
    assert state.stress <= 1e-6


def test_locate_query_prefers_deterministic_nearest_training_chart() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    atlas = RelationalGaugeAtlas(
        edges=_build_complete_graph_edges(points),
        config=RelationalGaugeAtlasConfig(
            fixed_rank=2,
            support_radius=3.0,
            min_chart_support=3,
            max_chart_support=8,
            min_overlap=2,
        ),
    )

    query = np.array([0.5, 0.0], dtype=np.float64)
    state = atlas.locate_query(_query_distances(points, query))
    assert state.chart_id == atlas.chart_for_node(0)
    assert state.nearest_node_id == 0


def test_locate_query_fails_on_unknown_chart_or_disconnected_distances() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    atlas = RelationalGaugeAtlas(
        edges=_build_complete_graph_edges(points),
        config=RelationalGaugeAtlasConfig(
            fixed_rank=2,
            support_radius=3.0,
            min_chart_support=3,
            max_chart_support=8,
            min_overlap=2,
        ),
    )

    query = np.array([0.2, 0.3], dtype=np.float64)
    with pytest.raises(RelationalGaugeAtlasError):
        atlas.locate_query(_query_distances(points, query), preferred_chart_id="chart:does-not-exist")


def test_locate_query_reports_off_support() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    atlas = RelationalGaugeAtlas(
        edges=_build_complete_graph_edges(points),
        config=RelationalGaugeAtlasConfig(
            fixed_rank=2,
            support_radius=1.0,
            min_chart_support=3,
            max_chart_support=8,
            min_overlap=2,
        ),
    )

    query = np.array([5.0, 5.0], dtype=np.float64)
    state = atlas.locate_query(_query_distances(points, query), preferred_chart_id=atlas.chart_for_node(0))

    assert state.support_status is False
    assert "exceeds" in state.support_reason
    assert state.chart_id == atlas.chart_for_node(0)
