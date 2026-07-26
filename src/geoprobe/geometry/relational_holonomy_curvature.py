"""Rung-4 holonomy/curvature measurement primitives for the discrete gauge atlas.

Registered instrument (privately retained results ledger of the program, stage-2 rung registration,
2026-07-23):
loop holonomies of the persisted SO(3) overlap transports, plaquette rotation
angles, log-holonomy span diagnostics, and the three registered null models
(Haar, edge-shuffle, residual-matched flat). Pure NumPy, torch-free, validation
style of :mod:`geoprobe.geometry.relational_gauge_atlas`.

Vocabulary note (binding): outputs speak of "log-holonomy span" and "matrix
commutators of measured transports" only — the analysis contract's prohibited
interpretations are untouched by this module.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from geoprobe.geometry.relational_gauge_atlas import (
    GaugeConnection,
    RelationalGaugeAtlas,
    RelationalGaugeAtlasError,
    gauge_transform_connection,
)


class RelationalHolonomyCurvatureError(ValueError):
    """Raised when a holonomy/curvature computation violates a validity contract."""


NEAR_PI_EXCLUSION_RAD = 0.05  # registered: loops with theta within this of pi are excluded from logs


# --------------------------------------------------------------------------------------
# Overlap graph and loop enumeration
# --------------------------------------------------------------------------------------

def build_chart_overlap_graph(
    connections: Mapping[tuple[str, str], GaugeConnection],
) -> dict[str, tuple[str, ...]]:
    """Undirected chart adjacency from the (bidirectionally stored) connections."""
    if not isinstance(connections, Mapping) or not connections:
        raise RelationalHolonomyCurvatureError("connections must be a non-empty mapping")
    neighbors: dict[str, set[str]] = {}
    for key, connection in connections.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(connection, GaugeConnection)
        ):
            raise RelationalHolonomyCurvatureError("connections mapping is malformed")
        source, target = key
        if source == target:
            raise RelationalHolonomyCurvatureError("self-connections are not supported")
        neighbors.setdefault(source, set()).add(target)
        neighbors.setdefault(target, set()).add(source)
    return {chart: tuple(sorted(peers)) for chart, peers in sorted(neighbors.items())}


def enumerate_triangles(adjacency: Mapping[str, Sequence[str]]) -> tuple[tuple[str, str, str], ...]:
    """All triangles (a<b<c lexicographically) via sorted-adjacency intersection."""
    sets = {chart: set(peers) for chart, peers in adjacency.items()}
    triangles: list[tuple[str, str, str]] = []
    for a in sorted(sets):
        higher_a = [b for b in adjacency[a] if b > a]
        for i, b in enumerate(higher_a):
            common = sets[a] & sets.get(b, set())
            for c in higher_a[i + 1 :]:
                if c in common:
                    triangles.append((a, b, c))
    return tuple(triangles)


def enumerate_quads(
    adjacency: Mapping[str, Sequence[str]],
    *,
    chordless: bool = True,
    cap: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[str, str, str, str], ...]:
    """4-cycles a-b-c-d-a with a the lexicographic minimum; optionally chordless.

    When ``cap`` is set and the enumeration exceeds it, a seeded uniform
    subsample of exactly ``cap`` quads is returned (rng required).
    """
    sets = {chart: set(peers) for chart, peers in adjacency.items()}
    quads: list[tuple[str, str, str, str]] = []
    for a in sorted(sets):
        higher = [x for x in adjacency[a] if x > a]
        for b in higher:
            for d in higher:
                if d <= b:
                    continue
                # common neighbors c of b and d, c > a, forming a-b-c-d-a
                for c in sorted(sets[b] & sets[d]):
                    if c <= a or c == b or c == d:
                        continue
                    if chordless and (c in sets[a] or d in sets[b]):
                        continue
                    quads.append((a, b, c, d))
    if cap is not None and len(quads) > cap:
        if rng is None:
            raise RelationalHolonomyCurvatureError("quad subsampling requires an rng")
        indices = rng.choice(len(quads), size=cap, replace=False)
        quads = [quads[i] for i in sorted(indices)]
    return tuple(quads)


def canonical_cycle(cycle: Sequence[str]) -> tuple[str, ...]:
    """Rotate/orient so the lexicographic minimum leads and its smaller neighbor is second."""
    charts = tuple(cycle)
    if len(charts) < 3 or len(set(charts)) != len(charts):
        raise RelationalHolonomyCurvatureError("cycle must contain at least 3 distinct charts")
    start = min(range(len(charts)), key=lambda i: charts[i])
    rotated = charts[start:] + charts[:start]
    forward_second = rotated[1]
    backward = (rotated[0],) + tuple(reversed(rotated[1:]))
    if backward[1] < forward_second:
        return backward
    return rotated


# --------------------------------------------------------------------------------------
# SO(3) helpers
# --------------------------------------------------------------------------------------

def rotation_angle(matrix: np.ndarray) -> float:
    """Gauge-invariant rotation angle of one SO(3) matrix."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (3, 3):
        raise RelationalHolonomyCurvatureError("rotation angle requires a 3x3 matrix")
    return float(np.arccos(np.clip((np.trace(values) - 1.0) / 2.0, -1.0, 1.0)))


def rotation_angles(stack: np.ndarray) -> np.ndarray:
    """Vectorized rotation angles for a [n, 3, 3] stack."""
    values = np.asarray(stack, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise RelationalHolonomyCurvatureError("stack must have shape [n, 3, 3]")
    traces = np.trace(values, axis1=1, axis2=2)
    return np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))


def rotation_log(matrix: np.ndarray, *, near_pi_exclusion: float = NEAR_PI_EXCLUSION_RAD) -> np.ndarray | None:
    """so(3) rotation vector of one SO(3) matrix; None when theta is near pi (excluded)."""
    values = np.asarray(matrix, dtype=np.float64)
    theta = rotation_angle(values)
    if theta >= np.pi - near_pi_exclusion:
        return None
    skew = 0.5 * (values - values.T)
    vee = np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float64)
    if theta < 1e-8:
        return vee  # first-order limit: log(R) ~ skew part
    return vee * (theta / np.sin(theta))


def rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation; axis must be a unit 3-vector."""
    unit = np.asarray(axis, dtype=np.float64)
    if unit.shape != (3,) or not np.isfinite(unit).all():
        raise RelationalHolonomyCurvatureError("axis must be a finite 3-vector")
    norm = float(np.linalg.norm(unit))
    if norm <= 0:
        raise RelationalHolonomyCurvatureError("axis must be nonzero")
    unit = unit / norm
    k = np.array(
        [
            [0.0, -unit[2], unit[1]],
            [unit[2], 0.0, -unit[0]],
            [-unit[1], unit[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


# --------------------------------------------------------------------------------------
# Plaquette / loop holonomy on an atlas
# --------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PlaquetteRecord:
    """One measured loop holonomy (gauge-invariant scalars + covariates)."""

    cycle: tuple[str, ...]
    holonomy: np.ndarray
    angle: float
    frobenius_defect: float
    spectral_defect: float
    axis: tuple[float, float, float] | None  # NOT gauge-invariant; recorded for diagnostics only
    loop_residual_sum: float
    min_overlap: int


def _loop_connections(atlas: RelationalGaugeAtlas, cycle: Sequence[str]) -> list[GaugeConnection]:
    ordered = canonical_cycle(cycle)
    hops = list(zip(ordered, ordered[1:] + (ordered[0],)))
    try:
        return [atlas.get_connection(source, target) for source, target in hops]
    except RelationalGaugeAtlasError as error:
        raise RelationalHolonomyCurvatureError(f"cycle is not connection-covered: {error}") from error


def plaquette_curvature(atlas: RelationalGaugeAtlas, cycle: Sequence[str]) -> PlaquetteRecord:
    """Holonomy of the ordered transport product around a minimal cycle."""
    connections = _loop_connections(atlas, cycle)
    composed = RelationalGaugeAtlas.compose_connection_path(connections)
    if composed.source_chart_id != composed.target_chart_id:
        raise RelationalHolonomyCurvatureError("composed loop must return to its base chart")
    holonomy = np.asarray(composed.transport, dtype=np.float64)
    if holonomy.shape != (3, 3):
        raise RelationalHolonomyCurvatureError(
            "registered rung-4 observables are defined for rank-3 transports"
        )
    angle = rotation_angle(holonomy)
    defect = holonomy - np.eye(3)
    log_vector = rotation_log(holonomy)
    axis: tuple[float, float, float] | None = None
    if log_vector is not None and np.linalg.norm(log_vector) > 1e-12:
        unit = log_vector / np.linalg.norm(log_vector)
        axis = (float(unit[0]), float(unit[1]), float(unit[2]))
    return PlaquetteRecord(
        cycle=canonical_cycle(cycle),
        holonomy=holonomy,
        angle=angle,
        frobenius_defect=float(np.linalg.norm(defect)),
        spectral_defect=float(np.linalg.norm(defect, ord=2)),
        axis=axis,
        loop_residual_sum=float(sum(connection.residual for connection in connections)),
        min_overlap=int(min(len(connection.overlap_node_ids) for connection in connections)),
    )


def loop_holonomy_defect(atlas: RelationalGaugeAtlas, cycle: Sequence[str]) -> PlaquetteRecord:
    """Holonomy record for an arbitrary-length enumerated cycle (>= 3 charts)."""
    return plaquette_curvature(atlas, cycle)


def path_discrepancy(atlas: RelationalGaugeAtlas, chart_a: str, chart_b: str, via: str) -> float:
    """Conjugacy-invariant angle between the direct and the 2-hop transport a->via->b.

    Equal, by construction, to the triangle plaquette angle of (a, via, b).
    """
    direct = atlas.get_connection(chart_a, chart_b).transport
    two_hop = (
        atlas.get_connection(via, chart_b).transport
        @ atlas.get_connection(chart_a, via).transport
    )
    return rotation_angle(two_hop @ direct.T)


# --------------------------------------------------------------------------------------
# Vectorized loop angles over an arbitrary transport map (measured or null)
# --------------------------------------------------------------------------------------

def loop_angles_over_transports(
    transports: Mapping[tuple[str, str], np.ndarray],
    loops: Sequence[Sequence[str]],
) -> np.ndarray:
    """Vectorized loop rotation angles for many cycles over one transport mapping.

    ``transports`` maps DIRECTED chart pairs to 3x3 matrices (both directions
    present). Loops are grouped by length internally; returns angles aligned to
    the input loop order.
    """
    angles = np.empty(len(loops), dtype=np.float64)
    by_length: dict[int, list[int]] = {}
    for index, loop in enumerate(loops):
        by_length.setdefault(len(loop), []).append(index)
    for length, indices in by_length.items():
        product = None
        for hop in range(length):
            matrices = np.stack(
                [
                    np.asarray(
                        transports[(loops[i][hop], loops[i][(hop + 1) % length])],
                        dtype=np.float64,
                    )
                    for i in indices
                ]
            )
            product = matrices if product is None else matrices @ product
        angles[indices] = rotation_angles(product)
    return angles


def atlas_transport_map(atlas: RelationalGaugeAtlas) -> dict[tuple[str, str], np.ndarray]:
    """Directed transport lookup extracted from the atlas connections."""
    return {
        key: np.asarray(connection.transport, dtype=np.float64)
        for key, connection in atlas.connections.items()
    }


# --------------------------------------------------------------------------------------
# Spanning-tree gauge fix and holonomy-group diagnostics
# --------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SpanningTreeGaugeFix:
    """BFS gauge fix: tree transports become identity; non-tree edges give generators."""

    base_chart: str
    component: tuple[str, ...]
    gauges: Mapping[str, np.ndarray]
    tree_edges: tuple[tuple[str, str], ...]
    generators: Mapping[tuple[str, str], np.ndarray]


def spanning_tree_gauge_fix(atlas: RelationalGaugeAtlas, base_chart: str) -> SpanningTreeGaugeFix:
    adjacency = build_chart_overlap_graph(atlas.connections)
    if base_chart not in adjacency:
        raise RelationalHolonomyCurvatureError(f"unknown base chart {base_chart!r}")
    gauges: dict[str, np.ndarray] = {base_chart: np.eye(3, dtype=np.float64)}
    parent: dict[str, str] = {}
    order: list[str] = [base_chart]
    frontier = [base_chart]
    while frontier:
        next_frontier: list[str] = []
        for chart in frontier:
            for peer in adjacency[chart]:
                if peer in gauges:
                    continue
                transport = np.asarray(
                    atlas.get_connection(chart, peer).transport, dtype=np.float64
                )
                if transport.shape != (3, 3):
                    raise RelationalHolonomyCurvatureError(
                        "registered rung-4 gauge fix is defined for rank-3 transports"
                    )
                # Q_peer = Q_chart @ T_{chart->peer}^T makes the transformed tree
                # transport Q_peer T Q_chart^T equal the identity.
                gauges[peer] = gauges[chart] @ transport.T
                parent[peer] = chart
                order.append(peer)
                next_frontier.append(peer)
        frontier = next_frontier

    tree_edges = tuple(sorted((parent[c], c) for c in parent))
    tree_set = {frozenset(edge) for edge in tree_edges}
    generators: dict[tuple[str, str], np.ndarray] = {}
    for chart in order:
        for peer in adjacency[chart]:
            if peer <= chart or peer not in gauges:
                continue
            if frozenset((chart, peer)) in tree_set:
                continue
            connection = atlas.get_connection(chart, peer)
            transformed = gauge_transform_connection(
                connection,
                source_basis=gauges[chart],
                target_basis=gauges[peer],
            )
            generators[(chart, peer)] = np.asarray(transformed.transport, dtype=np.float64)
    return SpanningTreeGaugeFix(
        base_chart=base_chart,
        component=tuple(order),
        gauges=gauges,
        tree_edges=tree_edges,
        generators=generators,
    )


@dataclass(frozen=True, slots=True)
class HolonomyGroupDiagnostics:
    """Log-holonomy span and product-closure diagnostics (neutral vocabulary)."""

    n_holonomies: int
    n_excluded_near_pi: int
    singular_values: tuple[float, ...]
    span_dimension: int
    bootstrap_modal_fraction: float
    closure_fraction: float
    closure_epsilon: float


def holonomy_group_diagnostics(
    holonomies: Sequence[np.ndarray],
    *,
    closure_epsilon: float,
    rng: np.random.Generator,
    svd_retain_ratio: float = 0.1,
    n_bootstrap: int = 200,
    n_closure_samples: int = 2000,
    near_pi_exclusion: float = NEAR_PI_EXCLUSION_RAD,
) -> HolonomyGroupDiagnostics:
    if closure_epsilon <= 0 or not np.isfinite(closure_epsilon):
        raise RelationalHolonomyCurvatureError("closure_epsilon must be positive and finite")
    matrices = [np.asarray(h, dtype=np.float64) for h in holonomies]
    if not matrices:
        raise RelationalHolonomyCurvatureError("at least one holonomy is required")
    logs: list[np.ndarray] = []
    excluded = 0
    for matrix in matrices:
        vector = rotation_log(matrix, near_pi_exclusion=near_pi_exclusion)
        if vector is None:
            excluded += 1
        else:
            logs.append(vector)
    if not logs:
        raise RelationalHolonomyCurvatureError("all holonomies were excluded near pi")
    stack = np.stack(logs)

    def span_dim(rows: np.ndarray) -> tuple[int, np.ndarray]:
        singular = np.linalg.svd(rows, compute_uv=False)
        if singular.size == 0 or singular[0] < 1e-12:
            return 0, singular
        return int(np.sum(singular >= svd_retain_ratio * singular[0])), singular

    dimension, singular_values = span_dim(stack)
    draws = np.empty(n_bootstrap, dtype=np.int64)
    for b in range(n_bootstrap):
        sample = stack[rng.integers(0, stack.shape[0], size=stack.shape[0])]
        draws[b], _ = span_dim(sample)
    modal_fraction = float(np.mean(draws == dimension))

    if dimension >= 3:
        closure_fraction = 1.0  # full so(3) span: products cannot leave the span
    else:
        _, _, vt = np.linalg.svd(stack, full_matrices=True)
        basis = vt[:dimension] if dimension > 0 else np.zeros((0, 3))
        hits = 0
        total = 0
        for _ in range(n_closure_samples):
            i, j = rng.integers(0, len(matrices), size=2)
            product = matrices[i] @ matrices[j]
            vector = rotation_log(product, near_pi_exclusion=near_pi_exclusion)
            if vector is None:
                continue
            total += 1
            norm = float(np.linalg.norm(vector))
            if norm < 1e-12:
                hits += 1
                continue
            if dimension == 0:
                # subgroup implied by an empty span is the identity: residual is the full angle
                residual_angle = norm
            else:
                projected = basis.T @ (basis @ vector)
                residual_angle = float(np.linalg.norm(vector - projected))
            if residual_angle <= closure_epsilon:
                hits += 1
        closure_fraction = float(hits / total) if total else float("nan")

    return HolonomyGroupDiagnostics(
        n_holonomies=len(matrices),
        n_excluded_near_pi=excluded,
        singular_values=tuple(float(s) for s in singular_values),
        span_dimension=dimension,
        bootstrap_modal_fraction=modal_fraction,
        closure_fraction=closure_fraction,
        closure_epsilon=float(closure_epsilon),
    )


# --------------------------------------------------------------------------------------
# Registered null models
# --------------------------------------------------------------------------------------

def _haar_so3(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q = q * np.sign(np.diag(r))[None, :]
    if np.linalg.det(q) < 0:
        q[:, [0, 1]] = q[:, [1, 0]]
    return q


def _undirected_edges(
    transports: Mapping[tuple[str, str], np.ndarray] | Mapping[tuple[str, str], GaugeConnection],
) -> list[tuple[str, str]]:
    edges = {tuple(sorted(key)) for key in transports}
    return sorted(edges)  # type: ignore[arg-type]


def haar_so_null(
    transports: Mapping[tuple[str, str], np.ndarray],
    rng: np.random.Generator,
) -> dict[tuple[str, str], np.ndarray]:
    """N1: Haar-random SO(3) transports on the true overlap graph."""
    null: dict[tuple[str, str], np.ndarray] = {}
    for u, v in _undirected_edges(transports):
        sample = _haar_so3(rng)
        null[(u, v)] = sample
        null[(v, u)] = sample.T
    return null


def edge_shuffle_null(
    transports: Mapping[tuple[str, str], np.ndarray],
    rng: np.random.Generator,
) -> dict[tuple[str, str], np.ndarray]:
    """N2: true transports shuffled across edges (marginals kept, coherence destroyed)."""
    edges = _undirected_edges(transports)
    pool = [np.asarray(transports[edge], dtype=np.float64) for edge in edges]
    permutation = rng.permutation(len(pool))
    null: dict[tuple[str, str], np.ndarray] = {}
    for edge, source_index in zip(edges, permutation):
        u, v = edge
        sample = pool[source_index]
        null[(u, v)] = sample
        null[(v, u)] = sample.T
    return null


def edge_residual_angles(atlas: RelationalGaugeAtlas) -> dict[tuple[str, str], float]:
    """Per-undirected-edge rotation scale calibrated to the Procrustes residual.

    A rotation by phi displaces a centered overlap cloud of Frobenius norm rho by
    ~ phi * rho, so phi = residual / rho reproduces the measured misfit scale.
    """
    angles: dict[tuple[str, str], float] = {}
    for key, connection in atlas.connections.items():
        edge = tuple(sorted(key))
        if edge in angles:
            continue
        source_chart = atlas.get_chart(connection.source_chart_id)
        overlap = connection.overlap_node_ids
        if not overlap:
            angles[edge] = 0.0
            continue
        points = source_chart.coordinates_for(overlap)
        centered = points - points.mean(axis=0, keepdims=True)
        rho = float(np.linalg.norm(centered))
        phi = 0.0 if rho <= 1e-12 else float(connection.residual) / rho
        angles[edge] = float(min(phi, np.pi / 2))
    return angles


def residual_matched_null(
    transports: Mapping[tuple[str, str], np.ndarray],
    residual_angles: Mapping[tuple[str, str], float],
    rng: np.random.Generator,
) -> dict[tuple[str, str], np.ndarray]:
    """N3: flat connection plus per-edge rotation noise at the measured residual scale."""
    null: dict[tuple[str, str], np.ndarray] = {}
    for edge in _undirected_edges(transports):
        u, v = edge
        if edge not in residual_angles:
            raise RelationalHolonomyCurvatureError(f"missing residual angle for edge {edge!r}")
        axis = rng.normal(size=3)
        while np.linalg.norm(axis) < 1e-12:
            axis = rng.normal(size=3)
        sample = rotation_from_axis_angle(axis, float(residual_angles[edge]))
        null[(u, v)] = sample
        null[(v, u)] = sample.T
    return null


__all__ = [
    "NEAR_PI_EXCLUSION_RAD",
    "HolonomyGroupDiagnostics",
    "PlaquetteRecord",
    "RelationalHolonomyCurvatureError",
    "SpanningTreeGaugeFix",
    "atlas_transport_map",
    "build_chart_overlap_graph",
    "canonical_cycle",
    "edge_residual_angles",
    "edge_shuffle_null",
    "enumerate_quads",
    "enumerate_triangles",
    "haar_so_null",
    "holonomy_group_diagnostics",
    "loop_angles_over_transports",
    "loop_holonomy_defect",
    "path_discrepancy",
    "plaquette_curvature",
    "residual_matched_null",
    "rotation_angle",
    "rotation_angles",
    "rotation_from_axis_angle",
    "rotation_log",
    "spanning_tree_gauge_fix",
]
