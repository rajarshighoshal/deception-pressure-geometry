"""Discrete intrinsic chart atlas and gauge-connection primitives for graphs.

The implementation is intentionally self-contained: pure NumPy/SciPy geometry with
strict validation, deterministic local MDS charts, overlap-driven orthogonal
connections, and gauge-conjugacy helpers.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class RelationalGaugeAtlasError(ValueError):
    """Raised when atlas/connection construction violates a validity contract."""


def _ensure_hashable(value: Hashable, name: str) -> Hashable:
    if value is None:
        raise RelationalGaugeAtlasError(f"{name} must be hashable and not None")
    try:
        hash(value)
    except TypeError as error:
        raise RelationalGaugeAtlasError(f"{name} must be hashable") from error
    return value


def _ensure_finite_float(value: object, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise RelationalGaugeAtlasError(f"{name} must be a finite float") from error
    if not np.isfinite(parsed):
        raise RelationalGaugeAtlasError(f"{name} must be finite")
    return parsed


def _as_tuple(values: Sequence[Hashable], *, name: str, non_empty: bool = False, unique: bool = False) -> tuple[Hashable, ...]:
    try:
        result = tuple(values)
    except TypeError as error:
        raise RelationalGaugeAtlasError(f"{name} must be a finite sequence") from error
    if non_empty and not result:
        raise RelationalGaugeAtlasError(f"{name} must be non-empty")
    for value in result:
        _ensure_hashable(value, f"element in {name}")
    if unique and len(set(result)) != len(result):
        raise RelationalGaugeAtlasError(f"{name} must contain unique values")
    return result


def _matrix_is_orthogonal(matrix: np.ndarray, *, tol: float = 1e-8) -> bool:
    identity = np.eye(matrix.shape[0], dtype=np.float64)
    return np.allclose(matrix.T @ matrix, identity, atol=tol) and np.allclose(matrix @ matrix.T, identity, atol=tol)


def _pairwise_euclidean(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2:
        raise RelationalGaugeAtlasError("points must be a 2D array")
    diff = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _sort_key(value: Hashable) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _classic_mds(distance_matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, float]:
    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise RelationalGaugeAtlasError("distance matrix must be square")
    n = distances.shape[0]
    if n == 0:
        raise RelationalGaugeAtlasError("distance matrix must be non-empty")
    if not np.isfinite(distances).all():
        raise RelationalGaugeAtlasError("distance matrix must be finite")
    if np.any(distances < 0):
        raise RelationalGaugeAtlasError("distance matrix must be non-negative")

    max_off_diag = np.max(np.abs(distances[~np.eye(n, dtype=bool)]))
    if max_off_diag <= 0:
        raise RelationalGaugeAtlasError("distance matrix must contain non-zero off-diagonal entries")
    if not np.allclose(distances, distances.T, atol=1e-10):
        raise RelationalGaugeAtlasError("distance matrix must be symmetric")

    squared = distances * distances
    centering = np.eye(n, dtype=np.float64)
    centering -= np.ones((n, n), dtype=np.float64) / n
    gram = -0.5 * centering @ squared @ centering

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    use = min(rank, n - 1)
    selected = eigenvalues[:use]
    selected_vectors = eigenvectors[:, :use].copy()
    for idx in range(use):
        axis = np.argmax(np.abs(selected_vectors[:, idx]))
        if selected_vectors[axis, idx] < 0:
            selected_vectors[:, idx] *= -1.0

    scales = np.maximum(selected, 0.0)
    coordinates = selected_vectors * np.sqrt(scales)[None, :]
    reconstructed = _pairwise_euclidean(coordinates)
    with np.errstate(divide="ignore", invalid="ignore"):
        stress = float(np.linalg.norm(reconstructed - distances) / max(np.linalg.norm(distances), 1e-12))

    return coordinates, scales, stress


def _solve_overlap_transport(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape:
        raise RelationalGaugeAtlasError("source/target overlap matrices must share the same shape")
    if source.ndim != 2:
        raise RelationalGaugeAtlasError("overlap matrices must be 2D")
    if source.shape[0] < 2:
        raise RelationalGaugeAtlasError("overlap size must be at least two")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise RelationalGaugeAtlasError("overlap matrices must be finite")

    source_centered = source - source.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)

    correlation = source_centered.T @ target_centered
    left, _, right = np.linalg.svd(correlation)
    row_mapping = left @ right
    if np.linalg.det(row_mapping) < 0:
        left[:, -1] *= -1.0
        row_mapping = left @ right

    transport = row_mapping.T
    aligned = source_centered @ row_mapping
    residual = float(np.linalg.norm(aligned - target_centered))
    return transport, residual


def _canonicalize_eigenvalues(values: np.ndarray) -> tuple[complex, ...]:
    ordered = np.asarray(values, dtype=np.complex128)
    ordered = np.sort_complex(ordered)
    return tuple(complex(value) for value in ordered)


def _distances_to_vector(
    node_ids: tuple[Hashable, ...],
    distances_to_training: Mapping[Hashable, float] | Sequence[float],
) -> np.ndarray:
    if isinstance(distances_to_training, Mapping):
        missing = [node for node in node_ids if node not in distances_to_training]
        if missing:
            raise RelationalGaugeAtlasError(f"missing distances for nodes: {missing!r}")
        values: list[float] = []
        for node in node_ids:
            values.append(float(distances_to_training[node]))
        return np.asarray(values, dtype=np.float64)
    values = np.asarray(distances_to_training, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != len(node_ids):
        raise RelationalGaugeAtlasError("distances_to_training must align to all training nodes")
    return values


def _project_query_into_chart(
    chart: GaugeChart,
    distances_to_support: np.ndarray,
) -> tuple[np.ndarray, float]:
    support_distances = np.asarray(chart.support_distances, dtype=np.float64)
    if distances_to_support.shape != (support_distances.shape[0],):
        raise RelationalGaugeAtlasError("distance support length mismatch")
    if not np.isfinite(distances_to_support).all():
        raise RelationalGaugeAtlasError("all support distances must be finite")

    support_sq = support_distances * support_distances
    with np.errstate(divide="ignore", invalid="ignore"):
        support_mean = np.mean(support_sq, axis=0)
        support_norm = np.mean(support_sq)
    query_sq = distances_to_support * distances_to_support
    query_mean = np.mean(query_sq)
    inner = -0.5 * (query_sq - query_mean - support_mean + support_norm)

    pseudo = np.linalg.pinv(np.asarray(chart.coordinates, dtype=np.float64))
    query_coordinates = pseudo @ inner
    joined = np.vstack([chart.coordinates, query_coordinates[None, :]])
    full_distances = _pairwise_euclidean(joined)
    reconstruction = full_distances[-1, :-1]
    stress = float(np.linalg.norm(reconstruction - distances_to_support) / max(np.linalg.norm(distances_to_support), 1e-12))
    return query_coordinates.astype(np.float64), stress


@dataclass(frozen=True, slots=True)
class GraphEdgeType:
    """Immutable edge-type marker used for validation and provenance."""

    name: str
    directed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise RelationalGaugeAtlasError("edge type name must be a non-empty string")


@dataclass(frozen=True, slots=True)
class WeightedGraphEdge:
    """One intrinsic edge in the relational graph."""

    source: Hashable
    target: Hashable
    weight: float
    edge_type: GraphEdgeType = GraphEdgeType("intrinsic")

    def __post_init__(self) -> None:
        _ensure_hashable(self.source, "source")
        _ensure_hashable(self.target, "target")
        if self.source == self.target:
            raise RelationalGaugeAtlasError("self-loop edges are not supported")
        if not isinstance(self.edge_type, GraphEdgeType):
            raise RelationalGaugeAtlasError("edge_type must be a GraphEdgeType")
        parsed = _ensure_finite_float(self.weight, "weight")
        if parsed <= 0:
            raise RelationalGaugeAtlasError("edge weight must be positive")
        object.__setattr__(self, "weight", parsed)


@dataclass(frozen=True, slots=True)
class RelationalGaugeAtlasConfig:
    """Parameters for local-chart atlas construction."""

    fixed_rank: int
    support_radius: float | None = None
    min_chart_support: int = 3
    max_chart_support: int = 64
    min_overlap: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.fixed_rank, Integral) or isinstance(self.fixed_rank, bool):
            raise RelationalGaugeAtlasError("fixed_rank must be an integer")
        rank = int(self.fixed_rank)
        if rank < 1:
            raise RelationalGaugeAtlasError("fixed_rank must be at least 1")
        if self.support_radius is None:
            raise RelationalGaugeAtlasError("support_radius must be provided")
        radius = _ensure_finite_float(self.support_radius, "support_radius")
        if radius <= 0:
            raise RelationalGaugeAtlasError("support_radius must be positive")
        if not isinstance(self.min_chart_support, Integral) or self.min_chart_support < 2:
            raise RelationalGaugeAtlasError("min_chart_support must be at least 2")
        if not isinstance(self.max_chart_support, Integral) or self.max_chart_support < self.min_chart_support:
            raise RelationalGaugeAtlasError("max_chart_support must be >= min_chart_support")
        if not isinstance(self.min_overlap, Integral) or self.min_overlap < 1:
            raise RelationalGaugeAtlasError("min_overlap must be >= 1")
        object.__setattr__(self, "fixed_rank", rank)
        object.__setattr__(self, "support_radius", float(radius))
        object.__setattr__(self, "min_chart_support", int(self.min_chart_support))
        object.__setattr__(self, "max_chart_support", int(self.max_chart_support))
        object.__setattr__(self, "min_overlap", int(self.min_overlap))


@dataclass(frozen=True, slots=True)
class GaugeChart:
    """One local coordinate chart from shortest-path distances."""

    chart_id: str
    center_node_id: Hashable
    support_ids: tuple[Hashable, ...]
    support_distances: tuple[tuple[float, ...], ...]
    coordinates: np.ndarray
    eigenvalues: np.ndarray
    stress: float
    support_radius: float

    def __post_init__(self) -> None:
        if not isinstance(self.chart_id, str) or not self.chart_id:
            raise RelationalGaugeAtlasError("chart_id must be a non-empty string")
        _ensure_hashable(self.center_node_id, "center_node_id")
        support = _as_tuple(self.support_ids, name="support_ids", non_empty=True, unique=True)
        if len(support) != len(self.support_ids):
            object.__setattr__(self, "support_ids", support)
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        if coordinates.ndim != 2:
            raise RelationalGaugeAtlasError("chart coordinates must be a 2D array")
        if coordinates.shape[0] != len(support):
            raise RelationalGaugeAtlasError("row count must match support size")
        if not np.isfinite(coordinates).all():
            raise RelationalGaugeAtlasError("chart coordinates must be finite")
        eigenvalues = np.asarray(self.eigenvalues, dtype=np.float64)
        if eigenvalues.ndim != 1:
            raise RelationalGaugeAtlasError("eigenvalues must be 1D")
        if eigenvalues.shape[0] != coordinates.shape[1]:
            raise RelationalGaugeAtlasError("eigenvalues and embedding rank must match")
        if np.any(eigenvalues < 0):
            raise RelationalGaugeAtlasError("eigenvalues must be non-negative")
        support_distances = np.asarray(self.support_distances, dtype=np.float64)
        if support_distances.ndim != 2:
            raise RelationalGaugeAtlasError("support_distances must be a 2D array")
        if support_distances.shape != (len(support), len(support)):
            raise RelationalGaugeAtlasError("support_distances must match support size")
        if not np.isfinite(support_distances).all():
            raise RelationalGaugeAtlasError("support_distances must be finite")
        if not np.allclose(support_distances, support_distances.T, atol=1e-10):
            raise RelationalGaugeAtlasError("support_distances must be symmetric")
        if not np.allclose(np.diag(support_distances), 0.0, atol=1e-10):
            raise RelationalGaugeAtlasError("support_distances diagonal must be zero")
        stress = _ensure_finite_float(self.stress, "chart stress")
        if stress < 0:
            raise RelationalGaugeAtlasError("chart stress must be non-negative")
        support_radius = _ensure_finite_float(self.support_radius, "support_radius")
        if support_radius <= 0:
            raise RelationalGaugeAtlasError("support_radius must be positive")
        object.__setattr__(self, "support_ids", support)
        object.__setattr__(self, "support_distances", tuple(tuple(row) for row in support_distances))
        object.__setattr__(self, "coordinates", coordinates.copy())
        object.__setattr__(self, "eigenvalues", eigenvalues.copy())
        object.__setattr__(self, "stress", float(stress))
        object.__setattr__(self, "support_radius", float(support_radius))

    @property
    def support_diameter(self) -> float:
        return float(np.max(np.asarray(self.support_distances, dtype=np.float64)))

    @property
    def trust_radius(self) -> float:
        return self.support_diameter

    @property
    def rank(self) -> int:
        return int(self.coordinates.shape[1])

    @property
    def node_to_index(self) -> dict[Hashable, int]:
        return {node: idx for idx, node in enumerate(self.support_ids)}

    def support_contains(self, node_id: Hashable) -> bool:
        return node_id in self.node_to_index

    def coordinate_for(self, node_id: Hashable) -> np.ndarray:
        node = _ensure_hashable(node_id, "node_id")
        if node not in self.node_to_index:
            raise RelationalGaugeAtlasError(f"node {node!r} is off-support")
        return self.coordinates[self.node_to_index[node]].copy()

    def coordinates_for(self, nodes: Sequence[Hashable]) -> np.ndarray:
        indices = []
        for node in nodes:
            if node not in self.node_to_index:
                raise RelationalGaugeAtlasError(f"node {node!r} is off-support")
            indices.append(self.node_to_index[node])
        return self.coordinates[indices].copy()


@dataclass(frozen=True, slots=True)
class GaugeConnection:
    """Orthogonal transport from source to target coordinates on the overlap."""

    source_chart_id: str
    target_chart_id: str
    source_support_ids: tuple[Hashable, ...]
    target_support_ids: tuple[Hashable, ...]
    overlap_node_ids: tuple[Hashable, ...]
    transport: np.ndarray
    residual: float

    def __post_init__(self) -> None:
        if not isinstance(self.source_chart_id, str) or not self.source_chart_id:
            raise RelationalGaugeAtlasError("source_chart_id must be a non-empty string")
        if not isinstance(self.target_chart_id, str) or not self.target_chart_id:
            raise RelationalGaugeAtlasError("target_chart_id must be a non-empty string")
        sources = _as_tuple(self.source_support_ids, name="source_support_ids", non_empty=True, unique=True)
        targets = _as_tuple(self.target_support_ids, name="target_support_ids", non_empty=True, unique=True)
        overlap = _as_tuple(self.overlap_node_ids, name="overlap_node_ids", non_empty=False, unique=True)
        transport = np.asarray(self.transport, dtype=np.float64)
        if transport.ndim != 2 or transport.shape[0] != transport.shape[1]:
            raise RelationalGaugeAtlasError("transport must be a square matrix")
        if not np.isfinite(transport).all():
            raise RelationalGaugeAtlasError("transport must be finite")
        if not _matrix_is_orthogonal(transport):
            raise RelationalGaugeAtlasError("transport must be orthogonal")
        residual = _ensure_finite_float(self.residual, "connection residual")
        if residual < 0:
            raise RelationalGaugeAtlasError("connection residual must be non-negative")

        if any(node not in sources for node in overlap) or any(node not in targets for node in overlap):
            raise RelationalGaugeAtlasError("overlap nodes must be supported by both charts")

        object.__setattr__(self, "source_support_ids", sources)
        object.__setattr__(self, "target_support_ids", targets)
        object.__setattr__(self, "overlap_node_ids", overlap)
        object.__setattr__(self, "transport", transport)
        object.__setattr__(self, "residual", float(residual))

    @property
    def rank(self) -> int:
        return int(self.transport.shape[0])

    @property
    def source_support_set(self) -> frozenset[Hashable]:
        return frozenset(self.source_support_ids)

    @property
    def target_support_set(self) -> frozenset[Hashable]:
        return frozenset(self.target_support_ids)

    @property
    def overlap_set(self) -> frozenset[Hashable]:
        return frozenset(self.overlap_node_ids)

    @property
    def reverse(self) -> "GaugeConnection":
        return GaugeConnection(
            source_chart_id=self.target_chart_id,
            target_chart_id=self.source_chart_id,
            source_support_ids=self.target_support_ids,
            target_support_ids=self.source_support_ids,
            overlap_node_ids=self.overlap_node_ids,
            transport=self.transport.T,
            residual=self.residual,
        )


@runtime_checkable
class GaugeChartInterface(Protocol):
    chart_id: str
    center_node_id: Hashable
    support_ids: tuple[Hashable, ...]
    support_distances: tuple[tuple[float, ...], ...]
    coordinates: np.ndarray
    eigenvalues: np.ndarray
    stress: float
    support_radius: float
    support_diameter: float


@dataclass(frozen=True, slots=True)
class GaugeQueryState:
    """Out-of-sample projection state for a query point."""

    chart_id: str
    query_coordinates: np.ndarray
    nearest_node_id: Hashable
    nearest_node_distance: float
    stress: float
    support_status: bool
    support_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.chart_id, str) or not self.chart_id:
            raise RelationalGaugeAtlasError("chart_id must be a non-empty string")
        _ensure_hashable(self.nearest_node_id, "nearest_node_id")
        if not isinstance(self.support_status, bool):
            raise RelationalGaugeAtlasError("support_status must be a bool")
        if not isinstance(self.support_reason, str):
            raise RelationalGaugeAtlasError("support_reason must be a string")
        query_coordinates = np.asarray(self.query_coordinates, dtype=np.float64)
        if query_coordinates.ndim != 1:
            raise RelationalGaugeAtlasError("query_coordinates must be a 1D array")
        if not np.isfinite(query_coordinates).all():
            raise RelationalGaugeAtlasError("query_coordinates must be finite")
        nearest_distance = _ensure_finite_float(self.nearest_node_distance, "nearest_node_distance")
        if nearest_distance < 0:
            raise RelationalGaugeAtlasError("nearest_node_distance must be non-negative")
        stress = _ensure_finite_float(self.stress, "query stress")
        if stress < 0:
            raise RelationalGaugeAtlasError("query stress must be non-negative")
        object.__setattr__(self, "query_coordinates", query_coordinates)
        object.__setattr__(self, "nearest_node_distance", float(nearest_distance))
        object.__setattr__(self, "stress", float(stress))


@runtime_checkable
class GaugeConnectionInterface(Protocol):
    source_chart_id: str
    target_chart_id: str
    transport: np.ndarray


@runtime_checkable
class GaugeAtlasInterface(Protocol):
    fixed_rank: int
    charts: Mapping[str, GaugeChart]
    def get_chart(self, chart_id: str) -> GaugeChart:
        ...

    def get_connection(self, source_chart_id: str, target_chart_id: str) -> GaugeConnection:
        ...


class RelationalGaugeAtlas(GaugeAtlasInterface):
    """Local classical-MDS atlas with overlap-based orthogonal transport."""

    def __init__(
        self,
        edges: Sequence[WeightedGraphEdge],
        *,
        config: RelationalGaugeAtlasConfig,
    ) -> None:
        if not isinstance(config, RelationalGaugeAtlasConfig):
            raise RelationalGaugeAtlasError("config must be RelationalGaugeAtlasConfig")
        self.config = config
        self.edges = tuple(_as_tuple(edges, name="edges", non_empty=True, unique=False))
        if not self.edges:
            raise RelationalGaugeAtlasError("edges must be non-empty")

        self._edge_types = self._validate_edge_types(self.edges)
        self.node_ids = self._build_node_ids(self.edges)
        self.node_to_index = {node: index for index, node in enumerate(self.node_ids)}

        self.distances = self._build_shortest_path_matrix(self.edges)
        self.chart_order = tuple(self._build_chart_order())
        self.charts = self._build_charts()
        self.connections = self._build_connections()

        if not self.charts:
            raise RelationalGaugeAtlasError("no valid local chart was built")

    @property
    def fixed_rank(self) -> int:
        return self.config.fixed_rank

    def _validate_edge_types(self, edges: tuple[WeightedGraphEdge, ...]) -> tuple[GraphEdgeType, ...]:
        types: list[GraphEdgeType] = []
        for edge in edges:
            if edge.edge_type not in types:
                types.append(edge.edge_type)
        return tuple(types)

    def _build_node_ids(self, edges: tuple[WeightedGraphEdge, ...]) -> tuple[Hashable, ...]:
        nodes: set[Hashable] = set()
        for edge in edges:
            nodes.add(_ensure_hashable(edge.source, "edge source"))
            nodes.add(_ensure_hashable(edge.target, "edge target"))
        if not nodes:
            raise RelationalGaugeAtlasError("at least one node is required")
        return tuple(sorted(nodes, key=_sort_key))

    def _build_shortest_path_matrix(self, edges: tuple[WeightedGraphEdge, ...]) -> np.ndarray:
        n = len(self.node_ids)
        adjacency = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(adjacency, 0.0)

        for edge in edges:
            i = self.node_to_index[edge.source]
            j = self.node_to_index[edge.target]
            if edge.weight < adjacency[i, j]:
                adjacency[i, j] = edge.weight
            if not edge.edge_type.directed and edge.weight < adjacency[j, i]:
                adjacency[j, i] = edge.weight

        matrix = shortest_path(csr_matrix(adjacency), directed=False, unweighted=False)
        return np.asarray(matrix, dtype=np.float64)

    def _build_chart_order(self) -> list[str]:
        return [
            f"chart:{repr(node)}"
            for node in self.node_ids
        ]

    def _chart_id_for_node(self, node_id: Hashable) -> str:
        node = _ensure_hashable(node_id, "node_id")
        return f"chart:{repr(node)}"

    def chart_for_node(self, node_id: Hashable) -> str:
        chart_id = self._chart_id_for_node(node_id)
        if chart_id not in self.charts:
            raise RelationalGaugeAtlasError(f"no chart for node {node_id!r}")
        return chart_id

    def get_chart(self, chart_id: str) -> GaugeChart:
        if not isinstance(chart_id, str) or not chart_id:
            raise RelationalGaugeAtlasError("chart_id must be a non-empty string")
        try:
            return self.charts[chart_id]
        except KeyError as error:
            raise RelationalGaugeAtlasError(f"unknown chart {chart_id!r}") from error

    def _chart_support(self, center_index: int) -> tuple[int, ...]:
        distances = self.distances[center_index]
        candidates = []
        for index, dist in enumerate(distances):
            if index == center_index or np.isfinite(dist):
                if dist <= self.config.support_radius:
                    candidates.append((dist, _sort_key(self.node_ids[index]), index))

        candidates.sort(key=lambda item: (item[0], item[1]))
        selected = tuple(index for _, _, index in candidates)
        minimum = max(self.config.min_chart_support, self.fixed_rank + 1)
        if len(selected) < minimum:
            raise RelationalGaugeAtlasError("insufficient support for a local chart")
        if len(selected) > self.config.max_chart_support:
            selected = selected[: self.config.max_chart_support]
        return selected

    def _build_chart(self, chart_id: str, center_index: int) -> GaugeChart:
        support_indices = self._chart_support(center_index)
        support_ids = tuple(self.node_ids[index] for index in support_indices)
        submatrix = self.distances[np.ix_(support_indices, support_indices)]
        coordinates, eigenvalues, stress = _classic_mds(submatrix, self.fixed_rank)
        support_distances = tuple(tuple(row) for row in submatrix)
        effective_rank = coordinates.shape[1]
        if effective_rank != self.fixed_rank:
            raise RelationalGaugeAtlasError("chart rank is inconsistent with fixed rank")
        return GaugeChart(
            chart_id=chart_id,
            center_node_id=self.node_ids[center_index],
            support_ids=support_ids,
            support_distances=support_distances,
            coordinates=coordinates,
            eigenvalues=eigenvalues,
            stress=stress,
            support_radius=self.config.support_radius,
        )

    def _build_charts(self) -> dict[str, GaugeChart]:
        charts: dict[str, GaugeChart] = {}
        for index, node in enumerate(self.node_ids):
            chart_id = f"chart:{repr(node)}"
            try:
                chart = self._build_chart(chart_id, index)
            except RelationalGaugeAtlasError:
                continue
            charts[chart_id] = chart
        return charts

    def _build_connection(self, source: GaugeChart, target: GaugeChart) -> GaugeConnection:
        overlap = tuple(node for node in source.support_ids if node in target.node_to_index)
        if len(overlap) < self.config.min_overlap:
            raise RelationalGaugeAtlasError("insufficient chart overlap")

        source_points = source.coordinates_for(overlap)
        target_points = target.coordinates_for(overlap)
        transport, residual = _solve_overlap_transport(source_points, target_points)

        return GaugeConnection(
            source_chart_id=source.chart_id,
            target_chart_id=target.chart_id,
            source_support_ids=source.support_ids,
            target_support_ids=target.support_ids,
            overlap_node_ids=overlap,
            transport=transport,
            residual=residual,
        )

    def _build_connections(self) -> dict[tuple[str, str], GaugeConnection]:
        output: dict[tuple[str, str], GaugeConnection] = {}
        chart_ids = list(self.chart_order)
        for i, first_id in enumerate(chart_ids):
            for second_id in chart_ids[i + 1 :]:
                if first_id not in self.charts or second_id not in self.charts:
                    continue
                source = self.get_chart(first_id)
                target = self.get_chart(second_id)
                try:
                    connection = self._build_connection(source, target)
                except RelationalGaugeAtlasError:
                    continue
                output[(first_id, second_id)] = connection
                output[(second_id, first_id)] = connection.reverse
        return output

    def get_connection(self, source_chart_id: str, target_chart_id: str) -> GaugeConnection:
        key = (self._validated_chart_id(source_chart_id), self._validated_chart_id(target_chart_id))
        try:
            return self.connections[key]
        except KeyError as error:
            raise RelationalGaugeAtlasError(
                "no overlap/connection between charts"
            ) from error

    def locate_query(
        self,
        distances_to_training: Mapping[Hashable, float] | Sequence[float],
        *,
        preferred_chart_id: str | None = None,
    ) -> GaugeQueryState:
        distances = _distances_to_vector(self.node_ids, distances_to_training)
        finite = np.isfinite(distances)
        if not finite.any():
            raise RelationalGaugeAtlasError("all query-to-training distances are non-finite")

        ranked = [
            (distances[index], _sort_key(self.node_ids[index]), index)
            for index in np.where(finite)[0]
        ]
        ranked.sort(key=lambda item: (item[0], item[1]))

        nearest_distance, _, nearest_node_index = ranked[0]
        nearest_node = self.node_ids[nearest_node_index]

        if preferred_chart_id is not None:
            chart_id = self._validated_chart_id(preferred_chart_id)
        else:
            chart_id = None
            for _, _, index in ranked:
                candidate = self._chart_id_for_node(self.node_ids[index])
                if candidate in self.charts:
                    chart_id = candidate
                    break
            if chart_id is None:
                raise RelationalGaugeAtlasError("query is disconnected from all local charts")

        chart = self.get_chart(chart_id)
        support_distances = np.asarray(
            [
                distances[self.node_to_index[node]]
                for node in chart.support_ids
            ],
            dtype=np.float64,
        )
        if not np.isfinite(support_distances).all():
            raise RelationalGaugeAtlasError("query must have finite distances for chart support")

        query_coordinates, stress = _project_query_into_chart(chart, support_distances)
        center_distance = float(support_distances[chart.node_to_index[chart.center_node_id]])
        in_support = center_distance <= chart.support_radius + 1e-12
        if in_support:
            reason = "query is within chart support radius"
        else:
            reason = (
                f"query center distance {center_distance:.6g} exceeds "
                f"support radius {chart.support_radius:.6g}"
            )

        return GaugeQueryState(
            chart_id=chart_id,
            query_coordinates=query_coordinates,
            nearest_node_id=nearest_node,
            nearest_node_distance=float(nearest_distance),
            stress=stress,
            support_status=in_support,
            support_reason=reason,
        )

    def _validated_chart_id(self, chart_id: str) -> str:
        if not isinstance(chart_id, str) or not chart_id:
            raise RelationalGaugeAtlasError("chart_id must be a non-empty string")
        if chart_id not in self.charts:
            raise RelationalGaugeAtlasError(f"unknown chart {chart_id!r}")
        return chart_id

    @staticmethod
    def compose_connection_path(path: Sequence[GaugeConnection]) -> GaugeConnection:
        if len(path) < 1:
            raise RelationalGaugeAtlasError("path must contain at least one connection")

        if len(path) == 1:
            connection = path[0]
            return GaugeConnection(
                source_chart_id=connection.source_chart_id,
                target_chart_id=connection.target_chart_id,
                source_support_ids=connection.source_support_ids,
                target_support_ids=connection.target_support_ids,
                overlap_node_ids=connection.overlap_node_ids,
                transport=connection.transport.copy(),
                residual=connection.residual,
            )

        transport = np.eye(path[0].rank, dtype=np.float64)
        for step, connection in enumerate(path):
            if step == 0:
                if connection.rank != path[0].rank:
                    raise RelationalGaugeAtlasError("incompatible connection ranks in path")
                transport = connection.transport @ transport
                continue
            previous = path[step - 1]
            if previous.target_chart_id != connection.source_chart_id:
                raise RelationalGaugeAtlasError("connection path is not chain-compatible")
            if connection.rank != path[0].rank:
                raise RelationalGaugeAtlasError("incompatible connection ranks in path")
            transport = connection.transport @ transport
        if transport.shape != (path[0].rank, path[0].rank):
            raise RelationalGaugeAtlasError("incompatible connection ranks in path")

        return GaugeConnection(
            source_chart_id=path[0].source_chart_id,
            target_chart_id=path[-1].target_chart_id,
            source_support_ids=path[0].source_support_ids,
            target_support_ids=path[-1].target_support_ids,
            overlap_node_ids=tuple(),
            transport=transport,
            residual=0.0,
        )

    @staticmethod
    def holonomy_spectrum(path: Sequence[GaugeConnection]) -> tuple[complex, ...]:
        if len(path) < 1:
            raise RelationalGaugeAtlasError("path must contain at least one connection")
        transport = RelationalGaugeAtlas.compose_connection_path(path).transport
        values = np.linalg.eigvals(transport)
        return _canonicalize_eigenvalues(values)


def gauge_transform_chart(chart: GaugeChartInterface, basis: np.ndarray) -> GaugeChart:
    if not isinstance(chart, GaugeChart):
        raise RelationalGaugeAtlasError("chart must be a GaugeChart")
    basis_matrix = np.asarray(basis, dtype=np.float64)
    if basis_matrix.ndim != 2:
        raise RelationalGaugeAtlasError("basis must be 2D")
    if basis_matrix.shape[0] != chart.coordinates.shape[1] or basis_matrix.shape[1] != chart.coordinates.shape[1]:
        raise RelationalGaugeAtlasError("basis shape must match chart rank")
    if not np.isfinite(basis_matrix).all():
        raise RelationalGaugeAtlasError("basis must be finite")
    if not _matrix_is_orthogonal(basis_matrix):
        raise RelationalGaugeAtlasError("basis must be orthogonal")

    transformed = np.asarray(chart.coordinates, dtype=np.float64) @ basis_matrix.T
    return GaugeChart(
        chart_id=chart.chart_id,
        center_node_id=chart.center_node_id,
        support_ids=chart.support_ids,
        support_distances=chart.support_distances,
        coordinates=transformed,
        eigenvalues=np.asarray(chart.eigenvalues, dtype=np.float64).copy(),
        stress=float(chart.stress),
        support_radius=float(chart.support_radius),
    )


def gauge_transform_connection(
    connection: GaugeConnectionInterface,
    *,
    source_basis: np.ndarray,
    target_basis: np.ndarray,
) -> GaugeConnection:
    if not isinstance(connection, GaugeConnection):
        raise RelationalGaugeAtlasError("connection must be a GaugeConnection")
    source = np.asarray(source_basis, dtype=np.float64)
    target = np.asarray(target_basis, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2:
        raise RelationalGaugeAtlasError("source_basis and target_basis must be 2D")
    if source.shape != target.shape:
        raise RelationalGaugeAtlasError("source_basis and target_basis must share the same shape")
    if source.shape[0] != source.shape[1]:
        raise RelationalGaugeAtlasError("basis matrices must be square")
    if source.shape[0] != connection.transport.shape[0]:
        raise RelationalGaugeAtlasError("basis size must match connection rank")
    if not _matrix_is_orthogonal(source) or not _matrix_is_orthogonal(target):
        raise RelationalGaugeAtlasError("basis matrices must be orthogonal")

    transformed = target @ np.asarray(connection.transport, dtype=np.float64) @ source.T
    if not _matrix_is_orthogonal(transformed):
        raise RelationalGaugeAtlasError("transformed transport must remain orthogonal")

    return GaugeConnection(
        source_chart_id=connection.source_chart_id,
        target_chart_id=connection.target_chart_id,
        source_support_ids=tuple(connection.source_support_ids),
        target_support_ids=tuple(connection.target_support_ids),
        overlap_node_ids=tuple(connection.overlap_node_ids),
        transport=transformed,
        residual=float(connection.residual),
    )


def conjugacy_invariant_connection_spectrum(matrix: np.ndarray) -> tuple[complex, ...]:
    values = np.asarray(matrix, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise RelationalGaugeAtlasError("matrix must be square")
    if not np.isfinite(values).all():
        raise RelationalGaugeAtlasError("matrix must be finite")
    return _canonicalize_eigenvalues(np.linalg.eigvals(values))


__all__ = [
    "GraphEdgeType",
    "GaugeAtlasInterface",
    "GaugeChart",
    "GaugeChartInterface",
    "GaugeConnection",
    "GaugeConnectionInterface",
    "RelationalGaugeAtlas",
    "RelationalGaugeAtlasConfig",
    "RelationalGaugeAtlasError",
    "GaugeQueryState",
    "WeightedGraphEdge",
    "conjugacy_invariant_connection_spectrum",
    "gauge_transform_chart",
    "gauge_transform_connection",
]
