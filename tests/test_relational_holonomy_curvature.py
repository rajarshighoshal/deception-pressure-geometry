"""Tests for rung-4 holonomy/curvature primitives (torch-free)."""

from __future__ import annotations

import numpy as np
import pytest

from geoprobe.geometry.relational_gauge_atlas import (
    GraphEdgeType,
    RelationalGaugeAtlas,
    RelationalGaugeAtlasConfig,
    WeightedGraphEdge,
)
from geoprobe.geometry.relational_holonomy_curvature import (
    RelationalHolonomyCurvatureError,
    atlas_transport_map,
    build_chart_overlap_graph,
    canonical_cycle,
    edge_residual_angles,
    edge_shuffle_null,
    enumerate_quads,
    enumerate_triangles,
    haar_so_null,
    holonomy_group_diagnostics,
    loop_angles_over_transports,
    path_discrepancy,
    plaquette_curvature,
    residual_matched_null,
    rotation_angle,
    rotation_from_axis_angle,
    rotation_log,
    spanning_tree_gauge_fix,
)


def _complete_graph_atlas(points: np.ndarray, distances: np.ndarray, **config: object) -> RelationalGaugeAtlas:
    edge_type = GraphEdgeType("intrinsic")
    edges = []
    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            edges.append(WeightedGraphEdge(i, j, float(distances[i, j]), edge_type=edge_type))
    defaults: dict[str, object] = {
        "fixed_rank": 3,
        "support_radius": float(np.median(distances) * 1.2),
        "min_chart_support": 5,
        "max_chart_support": 24,
        "min_overlap": 4,
    }
    defaults.update(config)
    return RelationalGaugeAtlas(tuple(edges), config=RelationalGaugeAtlasConfig(**defaults))


def _flat_atlas(seed: int = 3) -> RelationalGaugeAtlas:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(26, 3)) * 2.0
    diff = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1))
    return _complete_graph_atlas(points, distances)


def _sphere_atlas(seed: int = 5, n: int = 42) -> RelationalGaugeAtlas:
    indices = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n)
    golden = np.pi * (1.0 + np.sqrt(5.0))
    theta = golden * indices
    points = np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1
    )
    dots = np.clip(points @ points.T, -1.0, 1.0)
    geodesic = np.arccos(dots)
    np.fill_diagonal(geodesic, 0.0)
    return _complete_graph_atlas(points, geodesic, support_radius=1.1, max_chart_support=16)


def _triangles_of(atlas: RelationalGaugeAtlas) -> tuple[tuple[str, str, str], ...]:
    return enumerate_triangles(build_chart_overlap_graph(atlas.connections))


def test_flat_atlas_parity_consistent_plaquettes_vanish() -> None:
    # MDS charts carry arbitrary orientation PARITY; SO(3)-forced transports
    # between parity-mismatched charts are high-residual by construction. The
    # clean flat property holds on loops whose connections all fit exactly:
    # zero-residual loops must compose to the identity.
    atlas = _flat_atlas()
    triangles = _triangles_of(atlas)
    assert len(triangles) >= 20
    clean_records = []
    for tri in triangles:
        record = plaquette_curvature(atlas, tri)
        if record.loop_residual_sum < 1e-8:
            clean_records.append(record)
    assert len(clean_records) >= 5
    assert max(r.angle for r in clean_records) < 1e-7
    assert max(r.frobenius_defect for r in clean_records) < 1e-7
    assert max(r.spectral_defect for r in clean_records) < 1e-7


def test_sphere_atlas_shows_positive_angles_scaling_with_size() -> None:
    atlas = _sphere_atlas()
    triangles = _triangles_of(atlas)
    assert len(triangles) >= 30
    records = [plaquette_curvature(atlas, tri) for tri in triangles]
    angles = np.array([r.angle for r in records])
    assert np.median(angles) > 1e-3  # genuinely curved, far above flat numerics
    # size proxy: perimeter in shortest-path distance between chart centers
    def perimeter(cycle: tuple[str, ...]) -> float:
        nodes = [atlas.get_chart(c).center_node_id for c in cycle]
        idx = [atlas.node_to_index[n] for n in nodes]
        return float(
            atlas.distances[idx[0], idx[1]]
            + atlas.distances[idx[1], idx[2]]
            + atlas.distances[idx[2], idx[0]]
        )

    perimeters = np.array([perimeter(r.cycle) for r in records])
    small = angles[perimeters <= np.quantile(perimeters, 0.3)]
    large = angles[perimeters >= np.quantile(perimeters, 0.7)]
    assert large.mean() > small.mean()


def test_gauge_invariance_of_angles_and_span() -> None:
    atlas = _sphere_atlas()
    transports = atlas_transport_map(atlas)
    triangles = _triangles_of(atlas)[:40]
    loops = [canonical_cycle(t) for t in triangles]
    base_angles = loop_angles_over_transports(transports, loops)

    rng = np.random.default_rng(11)
    gauges: dict[str, np.ndarray] = {}
    for chart in build_chart_overlap_graph(atlas.connections):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        gauges[chart] = q * np.sign(np.diag(r))[None, :]
    transformed = {
        (u, v): gauges[v] @ matrix @ gauges[u].T for (u, v), matrix in transports.items()
    }
    new_angles = loop_angles_over_transports(transformed, loops)
    assert np.allclose(base_angles, new_angles, atol=1e-10)

    fix = spanning_tree_gauge_fix(atlas, max(build_chart_overlap_graph(atlas.connections)))
    generators = list(fix.generators.values())[:30]
    diag = holonomy_group_diagnostics(
        generators, closure_epsilon=0.5, rng=np.random.default_rng(0)
    )
    shared = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))[0]
    rotated = [shared @ g @ shared.T for g in generators]
    diag_rotated = holonomy_group_diagnostics(
        rotated, closure_epsilon=0.5, rng=np.random.default_rng(0)
    )
    assert diag.span_dimension == diag_rotated.span_dimension
    assert np.allclose(diag.singular_values, diag_rotated.singular_values, atol=1e-8)


def test_reversed_cycle_gives_equal_angle_and_transposed_holonomy() -> None:
    atlas = _sphere_atlas()
    triangle = _triangles_of(atlas)[0]
    forward = plaquette_curvature(atlas, triangle)
    backward = plaquette_curvature(atlas, tuple(reversed(triangle)))
    assert forward.angle == pytest.approx(backward.angle, abs=1e-10)
    # canonical ordering fixes orientation, so both calls resolve identically OR
    # to the transposed loop; verify at the raw transport level instead.
    transports = atlas_transport_map(atlas)
    a, b, c = triangle
    hol = transports[(c, a)] @ transports[(b, c)] @ transports[(a, b)]
    rev = transports[(b, a)] @ transports[(c, b)] @ transports[(a, c)]
    assert np.allclose(rev, hol.T, atol=1e-12)
    assert rotation_angle(rev) == pytest.approx(rotation_angle(hol), abs=1e-12)


def test_path_discrepancy_matches_triangle_angle() -> None:
    atlas = _sphere_atlas()
    a, b, c = _triangles_of(atlas)[0]
    angle = path_discrepancy(atlas, a, c, b)
    triangle_angle = plaquette_curvature(atlas, (a, b, c)).angle
    assert angle == pytest.approx(triangle_angle, abs=1e-9)


def test_spanning_tree_gauge_fix_makes_tree_identity() -> None:
    atlas = _flat_atlas()
    adjacency = build_chart_overlap_graph(atlas.connections)
    base = sorted(adjacency)[0]
    fix = spanning_tree_gauge_fix(atlas, base)
    assert fix.base_chart == base
    assert len(fix.component) >= 3
    for parent, child in fix.tree_edges[:10]:
        transport = atlas.get_connection(parent, child).transport
        transformed = fix.gauges[child] @ transport @ fix.gauges[parent].T
        assert np.allclose(transformed, np.eye(3), atol=1e-9)
    for generator in fix.generators.values():
        assert np.allclose(generator @ generator.T, np.eye(3), atol=1e-9)


def test_null_determinism_and_orthogonality() -> None:
    atlas = _flat_atlas()
    transports = atlas_transport_map(atlas)
    first = haar_so_null(transports, np.random.default_rng(7))
    second = haar_so_null(transports, np.random.default_rng(7))
    for key in first:
        assert np.allclose(first[key], second[key])
        assert np.allclose(first[key] @ first[key].T, np.eye(3), atol=1e-9)
        assert np.linalg.det(first[key]) == pytest.approx(1.0, abs=1e-9)
    shuffled = edge_shuffle_null(transports, np.random.default_rng(9))
    measured_pool = sorted(
        rotation_angle(transports[key]) for key in transports if key[0] < key[1]
    )
    shuffled_pool = sorted(
        rotation_angle(shuffled[key]) for key in shuffled if key[0] < key[1]
    )
    assert np.allclose(measured_pool, shuffled_pool, atol=1e-10)


def test_residual_matched_null_tracks_residual_scale() -> None:
    atlas = _sphere_atlas()
    transports = atlas_transport_map(atlas)
    phis = edge_residual_angles(atlas)
    null = residual_matched_null(transports, phis, np.random.default_rng(3))
    for edge, phi in list(phis.items())[:25]:
        assert rotation_angle(null[edge]) == pytest.approx(phi, abs=1e-9)


def test_span_dimensions_one_three_zero_and_closure() -> None:
    rng = np.random.default_rng(1)
    axis = np.array([0.0, 0.0, 1.0])
    about_z = [rotation_from_axis_angle(axis, a) for a in (0.3, 0.7, 1.1, 0.2, 0.9)]
    diag_z = holonomy_group_diagnostics(about_z, closure_epsilon=0.05, rng=rng)
    assert diag_z.span_dimension == 1
    assert diag_z.closure_fraction == pytest.approx(1.0)

    generic = [
        rotation_from_axis_angle(np.random.default_rng(k).normal(size=3), 0.4 + 0.1 * k)
        for k in range(6)
    ]
    diag_generic = holonomy_group_diagnostics(
        generic, closure_epsilon=0.05, rng=np.random.default_rng(2)
    )
    assert diag_generic.span_dimension == 3
    assert diag_generic.closure_fraction == pytest.approx(1.0)

    identities = [np.eye(3) for _ in range(4)]
    diag_id = holonomy_group_diagnostics(
        identities, closure_epsilon=0.05, rng=np.random.default_rng(3)
    )
    assert diag_id.span_dimension == 0
    assert diag_id.closure_fraction == pytest.approx(1.0)


def test_near_pi_exclusion_counted_and_log_none() -> None:
    near_pi = rotation_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi - 0.01)
    assert rotation_log(near_pi) is None
    mixed = [near_pi, rotation_from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.5)]
    diag = holonomy_group_diagnostics(
        mixed, closure_epsilon=0.1, rng=np.random.default_rng(0)
    )
    assert diag.n_excluded_near_pi == 1
    assert diag.n_holonomies == 2


def test_error_paths() -> None:
    atlas = _flat_atlas()
    with pytest.raises(RelationalHolonomyCurvatureError):
        canonical_cycle(("a", "a", "b"))
    with pytest.raises(RelationalHolonomyCurvatureError):
        spanning_tree_gauge_fix(atlas, "chart:'missing'")
    adjacency = build_chart_overlap_graph(atlas.connections)
    charts = sorted(adjacency)
    disconnected = None
    for a in charts:
        for b in charts:
            if a < b and b not in adjacency[a]:
                disconnected = (a, b)
                break
        if disconnected:
            break
    if disconnected is not None:
        third = next(c for c in charts if c not in disconnected)
        with pytest.raises(RelationalHolonomyCurvatureError):
            plaquette_curvature(atlas, (disconnected[0], disconnected[1], third))


def test_enumerations_are_deterministic_and_capped() -> None:
    atlas = _sphere_atlas()
    adjacency = build_chart_overlap_graph(atlas.connections)
    triangles_a = enumerate_triangles(adjacency)
    triangles_b = enumerate_triangles(adjacency)
    assert triangles_a == triangles_b
    quads_full = enumerate_quads(adjacency, cap=None)
    if len(quads_full) > 5:
        capped = enumerate_quads(adjacency, cap=5, rng=np.random.default_rng(20260723))
        again = enumerate_quads(adjacency, cap=5, rng=np.random.default_rng(20260723))
        assert capped == again
        assert len(capped) == 5
        assert set(capped) <= set(quads_full)
