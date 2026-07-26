"""Fold-safe candidate retrieval and exact graphs for rooted pre-status stars."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import math
from threading import Lock

import numpy as np

from geoprobe.geometry.relational_pre_status_rooted_metric import (
    RootedStarDistance,
    RootedStarMetricInput,
    RootedStarMetricScaler,
    rooted_star_distance,
)


FOLDS = ("outer_1", "outer_2", "outer_3", "outer_4", "outer_5")
CANDIDATE_WIDTH = 64
GRAPH_WIDTH = 8
CALIBRATION_PAIR_COUNT = 512


class RelationalPreStatusRootedGraphError(ValueError):
    """A fold graph would violate its label-free construction contract."""


@dataclass(frozen=True, slots=True)
class RootedStarNode:
    node_id: str
    family: str
    family_fold: str
    descriptor: np.ndarray

    def __post_init__(self) -> None:
        for name in ("node_id", "family"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise RelationalPreStatusRootedGraphError(f"{name} is invalid")
        if self.family_fold not in FOLDS:
            raise RelationalPreStatusRootedGraphError("family_fold is invalid")
        descriptor = np.ascontiguousarray(np.asarray(self.descriptor, dtype=np.float64))
        if descriptor.ndim != 1 or not descriptor.size or not np.isfinite(descriptor).all():
            raise RelationalPreStatusRootedGraphError(
                "node descriptor must be a finite non-empty vector"
            )
        descriptor.flags.writeable = False
        object.__setattr__(self, "descriptor", descriptor)


@dataclass(frozen=True, slots=True)
class CandidateEdge:
    source_id: str
    target_id: str
    rank: int
    descriptor_distance: float


@dataclass(frozen=True, slots=True)
class ExactGraphEdge:
    source_id: str
    target_id: str
    rank: int
    joint_score: float
    residual_score: float
    attention_score: float
    descriptor_rank: int


@dataclass(frozen=True, slots=True)
class FoldCandidateSchedule:
    held_out_family_fold: str
    candidate_width: int
    query_edges: Mapping[str, tuple[CandidateEdge, ...]]
    training_edges: Mapping[str, tuple[CandidateEdge, ...]]
    calibration_pairs: tuple[tuple[str, str], ...]
    descriptor_center: np.ndarray
    descriptor_scale: np.ndarray


@dataclass(frozen=True, slots=True)
class FoldExactRootedGraph:
    held_out_family_fold: str
    graph_width: int
    query_edges: Mapping[str, tuple[ExactGraphEdge, ...]]
    training_edges: Mapping[str, tuple[ExactGraphEdge, ...]]
    scaler: RootedStarMetricScaler
    candidate_pair_count: int
    exact_pair_count: int


def _digest(*parts: str) -> bytes:
    return sha256("\x00".join(parts).encode("utf-8")).digest()


def _robust_scale(training: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(training, axis=0)
    q25, q75 = np.quantile(training, (0.25, 0.75), axis=0)
    scale = q75 - q25
    standard_deviation = training.std(axis=0)
    scale = np.where(scale > 1e-12, scale, standard_deviation)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return np.asarray(center, dtype=np.float64), np.asarray(scale, dtype=np.float64)


def _rank_candidates(
    source: RootedStarNode,
    targets: Sequence[RootedStarNode],
    *,
    center: np.ndarray,
    scale: np.ndarray,
    width: int,
) -> tuple[CandidateEdge, ...]:
    source_descriptor = (source.descriptor - center) / scale
    rows: list[tuple[float, str]] = []
    for target in targets:
        if target.node_id == source.node_id:
            continue
        distance = float(
            np.linalg.norm(source_descriptor - (target.descriptor - center) / scale)
            / math.sqrt(source_descriptor.size)
        )
        rows.append((distance, target.node_id))
    rows.sort(key=lambda item: (item[0], item[1]))
    return tuple(
        CandidateEdge(source.node_id, target_id, rank, distance)
        for rank, (distance, target_id) in enumerate(rows[:width], 1)
    )


def _calibration_pairs(
    training: Sequence[RootedStarNode], *, count: int
) -> tuple[tuple[str, str], ...]:
    candidates: list[tuple[bytes, str, str]] = []
    for left_index, left in enumerate(training):
        for right in training[left_index + 1 :]:
            first, second = sorted((left.node_id, right.node_id))
            candidates.append(
                (_digest("pre-status-rooted-calibration-v1", first, second), first, second)
            )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple((first, second) for _, first, second in candidates[:count])


def build_fold_candidate_schedule(
    nodes: Sequence[RootedStarNode],
    *,
    held_out_family_fold: str,
    candidate_width: int = CANDIDATE_WIDTH,
    calibration_pair_count: int = CALIBRATION_PAIR_COUNT,
) -> FoldCandidateSchedule:
    """Create train-only descriptor scaling and legal candidate neighborhoods."""
    if held_out_family_fold not in FOLDS:
        raise RelationalPreStatusRootedGraphError("held-out fold is invalid")
    if candidate_width < GRAPH_WIDTH or calibration_pair_count < 1:
        raise RelationalPreStatusRootedGraphError(
            "candidate width or calibration pair count is invalid"
        )
    rows = tuple(nodes)
    if not rows or len({node.node_id for node in rows}) != len(rows):
        raise RelationalPreStatusRootedGraphError("nodes must be non-empty and unique")
    widths = {node.descriptor.shape for node in rows}
    if len(widths) != 1:
        raise RelationalPreStatusRootedGraphError("descriptors have inconsistent widths")
    training = tuple(node for node in rows if node.family_fold != held_out_family_fold)
    queries = tuple(node for node in rows if node.family_fold == held_out_family_fold)
    if len(training) <= candidate_width or not queries:
        raise RelationalPreStatusRootedGraphError(
            "fold has insufficient training or held-out nodes"
        )
    matrix = np.stack([node.descriptor for node in training], axis=0)
    center, scale = _robust_scale(matrix)
    query_edges = {
        node.node_id: _rank_candidates(
            node, training, center=center, scale=scale, width=candidate_width
        )
        for node in sorted(queries, key=lambda item: item.node_id)
    }
    training_edges = {
        node.node_id: _rank_candidates(
            node, training, center=center, scale=scale, width=candidate_width
        )
        for node in sorted(training, key=lambda item: item.node_id)
    }
    center.flags.writeable = False
    scale.flags.writeable = False
    return FoldCandidateSchedule(
        held_out_family_fold=held_out_family_fold,
        candidate_width=candidate_width,
        query_edges=query_edges,
        training_edges=training_edges,
        calibration_pairs=_calibration_pairs(
            training, count=min(calibration_pair_count, len(training) * (len(training) - 1) // 2)
        ),
        descriptor_center=center,
        descriptor_scale=scale,
    )


def build_fold_exact_graph(
    schedule: FoldCandidateSchedule,
    *,
    load_metric_input: Callable[[str], RootedStarMetricInput] | None = None,
    pair_distance: Callable[[str, str], RootedStarDistance] | None = None,
    graph_width: int = GRAPH_WIDTH,
    distance_workers: int = 1,
) -> FoldExactRootedGraph:
    """Exact-rerank one fold's schedule after training-only component scaling."""
    if graph_width < 1 or graph_width > schedule.candidate_width:
        raise RelationalPreStatusRootedGraphError("graph width is invalid")
    if (load_metric_input is None) == (pair_distance is None):
        raise RelationalPreStatusRootedGraphError(
            "provide exactly one metric-input loader or quotient pair distance"
        )
    if (
        not isinstance(distance_workers, int)
        or isinstance(distance_workers, bool)
        or distance_workers not in range(1, 5)
    ):
        raise RelationalPreStatusRootedGraphError(
            "distance_workers must be an integer from one through four"
        )
    cache: dict[tuple[str, str], RootedStarDistance] = {}
    cache_lock = Lock()

    def exact(left_id: str, right_id: str) -> RootedStarDistance:
        key = tuple(sorted((left_id, right_id)))
        with cache_lock:
            value = cache.get(key)
        if value is None:
            if pair_distance is not None:
                value = pair_distance(key[0], key[1])
            else:
                assert load_metric_input is not None
                value = rooted_star_distance(
                    load_metric_input(key[0]), load_metric_input(key[1])
                )
            with cache_lock:
                value = cache.setdefault(key, value)
        return value

    required_pairs = {
        tuple(sorted(pair)) for pair in schedule.calibration_pairs
    }
    required_pairs.update(
        tuple(sorted((source_id, edge.target_id)))
        for rows in (schedule.query_edges, schedule.training_edges)
        for source_id, edges in rows.items()
        for edge in edges
    )
    if distance_workers > 1:
        with ThreadPoolExecutor(
            max_workers=distance_workers,
            thread_name_prefix="rooted-distance",
        ) as executor:
            tuple(executor.map(lambda pair: exact(*pair), sorted(required_pairs)))

    calibration = tuple(exact(left, right) for left, right in schedule.calibration_pairs)
    scaler = RootedStarMetricScaler.fit(calibration)

    def rerank(rows: Mapping[str, tuple[CandidateEdge, ...]]) -> dict[str, tuple[ExactGraphEdge, ...]]:
        result: dict[str, tuple[ExactGraphEdge, ...]] = {}
        for source_id, candidates in rows.items():
            scored: list[tuple[float, CandidateEdge, RootedStarDistance]] = []
            for candidate in candidates:
                distance = exact(source_id, candidate.target_id)
                scored.append((scaler.transform(distance), candidate, distance))
            scored.sort(key=lambda item: (item[0], item[1].target_id))
            result[source_id] = tuple(
                ExactGraphEdge(
                    source_id=source_id,
                    target_id=candidate.target_id,
                    rank=rank,
                    joint_score=float(score),
                    residual_score=float(distance.residual),
                    attention_score=float(distance.attention_head_set),
                    descriptor_rank=candidate.rank,
                )
                for rank, (score, candidate, distance) in enumerate(
                    scored[:graph_width], 1
                )
            )
        return result

    query_edges = rerank(schedule.query_edges)
    training_edges = rerank(schedule.training_edges)
    candidate_pair_count = sum(
        len(rows) for rows in (*schedule.query_edges.values(), *schedule.training_edges.values())
    )
    return FoldExactRootedGraph(
        held_out_family_fold=schedule.held_out_family_fold,
        graph_width=graph_width,
        query_edges=query_edges,
        training_edges=training_edges,
        scaler=scaler,
        candidate_pair_count=candidate_pair_count,
        exact_pair_count=len(cache),
    )


__all__ = [
    "CALIBRATION_PAIR_COUNT",
    "CANDIDATE_WIDTH",
    "FOLDS",
    "GRAPH_WIDTH",
    "CandidateEdge",
    "ExactGraphEdge",
    "FoldCandidateSchedule",
    "FoldExactRootedGraph",
    "RelationalPreStatusRootedGraphError",
    "RootedStarNode",
    "build_fold_candidate_schedule",
    "build_fold_exact_graph",
]
