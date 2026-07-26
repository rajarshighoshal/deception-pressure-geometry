"""One shared pre-status honestward lift over a frozen rooted-star graph."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

from geoprobe.geometry.relational_pre_status_rooted_graph import _digest


class RelationalPreStatusHonestwardError(ValueError):
    """A shared honestward fit or query violates its fold-safe contract."""


@dataclass(frozen=True, slots=True)
class HonestwardCrossingObservation:
    pair_id: str
    deceptive_root_id: str
    honest_root_id: str
    family: str
    family_fold: str
    scenario_id: str
    contrast_id: str
    true_status: str
    delta: np.ndarray
    contrast_ids: tuple[str, ...] = ()
    source_pair_ids: tuple[str, ...] = ()
    honest_root_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "pair_id",
            "deceptive_root_id",
            "honest_root_id",
            "family",
            "family_fold",
            "scenario_id",
            "contrast_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise RelationalPreStatusHonestwardError(f"{name} is invalid")
        if self.true_status not in {"PASS", "FAIL"}:
            raise RelationalPreStatusHonestwardError("true_status is invalid")
        contrast_ids = self.contrast_ids or (self.contrast_id,)
        source_pair_ids = self.source_pair_ids or (self.pair_id,)
        honest_root_ids = self.honest_root_ids or (self.honest_root_id,)
        for name, values in (
            ("contrast_ids", contrast_ids),
            ("source_pair_ids", source_pair_ids),
            ("honest_root_ids", honest_root_ids),
        ):
            if not values or any(not isinstance(value, str) or not value for value in values):
                raise RelationalPreStatusHonestwardError(f"{name} is invalid")
            object.__setattr__(self, name, tuple(sorted(set(values))))
        delta = np.ascontiguousarray(np.asarray(self.delta, dtype=np.float32))
        if delta.ndim != 2 or not delta.size or not np.isfinite(delta).all():
            raise RelationalPreStatusHonestwardError(
                "honestward delta must be finite [layer, hidden]"
            )
        delta.flags.writeable = False
        object.__setattr__(self, "delta", delta)


@dataclass(frozen=True, slots=True)
class HonestwardSupportEdge:
    target_id: str
    rank: int
    joint_score: float


@dataclass(frozen=True, slots=True)
class HonestwardPrediction:
    query_root_id: str
    defined: bool
    local: np.ndarray
    dose_calibrated_local: np.ndarray
    global_mean: np.ndarray
    nearest: np.ndarray
    shuffled: np.ndarray
    sign_flipped: np.ndarray
    random_local_span: np.ndarray
    support_root_ids: tuple[str, ...]
    support_pair_ids: tuple[str, ...]
    support_count: int
    scalar_dose_by_layer: np.ndarray


@dataclass(frozen=True, slots=True)
class _RootTarget:
    root_id: str
    deltas: tuple[np.ndarray, ...]
    pair_ids: tuple[str, ...]
    contrast_ids: tuple[str, ...]
    true_statuses: tuple[str, ...]

    @property
    def mean_delta(self) -> np.ndarray:
        return np.asarray(
            np.mean(np.stack(self.deltas, axis=0, dtype=np.float64), axis=0),
            dtype=np.float32,
        )


def _edge(edge: object) -> tuple[str, int, float]:
    try:
        target = str(getattr(edge, "target_id"))
        rank = getattr(edge, "rank")
        score = float(getattr(edge, "joint_score"))
    except (AttributeError, TypeError, ValueError) as error:
        raise RelationalPreStatusHonestwardError(
            "support edge lacks target_id, rank, or joint_score"
        ) from error
    if not target or not isinstance(rank, int) or rank < 1 or not math.isfinite(score):
        raise RelationalPreStatusHonestwardError("support edge is invalid")
    return target, rank, score


def _root_targets(
    observations: Sequence[HonestwardCrossingObservation],
    *,
    excluded_contrast: str | None = None,
    required_truth: str | None = None,
) -> Mapping[str, _RootTarget]:
    grouped: dict[str, list[HonestwardCrossingObservation]] = defaultdict(list)
    for row in observations:
        if excluded_contrast is not None and excluded_contrast in row.contrast_ids:
            continue
        if required_truth is not None and row.true_status != required_truth:
            continue
        grouped[row.deceptive_root_id].append(row)
    result: dict[str, _RootTarget] = {}
    for root_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: item.pair_id)
        result[root_id] = _RootTarget(
            root_id=root_id,
            deltas=tuple(row.delta for row in ordered),
            pair_ids=tuple(row.pair_id for row in ordered),
            contrast_ids=tuple(row.contrast_id for row in ordered),
            true_statuses=tuple(row.true_status for row in ordered),
        )
    return MappingProxyType(result)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=np.float32)).copy()
    result.flags.writeable = False
    return result


def _mean(values: Sequence[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if not values:
        return np.zeros(shape, dtype=np.float32)
    return np.asarray(
        np.mean(np.stack(values, axis=0, dtype=np.float64), axis=0),
        dtype=np.float32,
    )


class SharedPreStatusHonestwardField:
    """A root-balanced local field with no contrast/program/family cells."""

    def __init__(
        self,
        *,
        held_out_family_fold: str,
        observations: tuple[HonestwardCrossingObservation, ...],
        training_edges: Mapping[str, Sequence[object]],
        scalar_dose_by_layer: np.ndarray,
        shape: tuple[int, int],
    ) -> None:
        self.held_out_family_fold = held_out_family_fold
        self._observations = observations
        self._training_edges = MappingProxyType(
            {root: tuple(edges) for root, edges in training_edges.items()}
        )
        self.scalar_dose_by_layer = _readonly(scalar_dose_by_layer)
        self.shape = shape

    @classmethod
    def fit(
        cls,
        observations: Sequence[HonestwardCrossingObservation],
        *,
        held_out_family_fold: str,
        training_edges: Mapping[str, Sequence[object]],
    ) -> SharedPreStatusHonestwardField:
        rows = tuple(observations)
        if not rows:
            raise RelationalPreStatusHonestwardError("fit requires crossing observations")
        if any(row.family_fold == held_out_family_fold for row in rows):
            raise RelationalPreStatusHonestwardError(
                "held-out crossing entered honestward training"
            )
        if len({row.pair_id for row in rows}) != len(rows):
            raise RelationalPreStatusHonestwardError("crossing pair IDs are not unique")
        shape = rows[0].delta.shape
        if any(row.delta.shape != shape for row in rows):
            raise RelationalPreStatusHonestwardError("crossing deltas have mixed shapes")
        temporary = cls(
            held_out_family_fold=held_out_family_fold,
            observations=rows,
            training_edges=training_edges,
            scalar_dose_by_layer=np.ones(shape[0], dtype=np.float32),
            shape=shape,
        )
        targets = _root_targets(rows)
        numerator = np.zeros(shape[0], dtype=np.float64)
        denominator = np.zeros(shape[0], dtype=np.float64)
        for root_id, target in targets.items():
            predicted, _, _ = temporary._local(
                training_edges.get(root_id, ()), targets, excluded_root=root_id
            )
            if predicted is None:
                continue
            actual = target.mean_delta.astype(np.float64)
            estimate = predicted.astype(np.float64)
            numerator += np.sum(actual * estimate, axis=1)
            denominator += np.sum(estimate * estimate, axis=1)
        dose = np.ones(shape[0], dtype=np.float32)
        valid = denominator > 1e-12
        dose[valid] = np.maximum(numerator[valid] / denominator[valid], 0.0).astype(
            np.float32
        )
        return cls(
            held_out_family_fold=held_out_family_fold,
            observations=rows,
            training_edges=training_edges,
            scalar_dose_by_layer=dose,
            shape=shape,
        )

    def _local(
        self,
        edges: Sequence[object],
        targets: Mapping[str, _RootTarget],
        *,
        excluded_root: str | None = None,
    ) -> tuple[np.ndarray | None, tuple[str, ...], tuple[str, ...]]:
        ordered = sorted((_edge(edge) for edge in edges), key=lambda item: (item[1], item[2], item[0]))
        roots: list[str] = []
        deltas: list[np.ndarray] = []
        pair_ids: list[str] = []
        for target_id, _, _ in ordered:
            if target_id == excluded_root or target_id in roots:
                continue
            target = targets.get(target_id)
            if target is None:
                continue
            roots.append(target_id)
            deltas.append(target.mean_delta)
            pair_ids.extend(target.pair_ids)
        if not deltas:
            return None, (), ()
        return _mean(deltas, self.shape), tuple(roots), tuple(pair_ids)

    def predict(
        self,
        query_root_id: str,
        edges: Sequence[object],
        *,
        exclude_contrast: str | None = None,
        opposite_of_true_status: str | None = None,
    ) -> HonestwardPrediction:
        if not isinstance(query_root_id, str) or not query_root_id:
            raise RelationalPreStatusHonestwardError("query_root_id is invalid")
        required_truth = None
        if opposite_of_true_status is not None:
            if opposite_of_true_status not in {"PASS", "FAIL"}:
                raise RelationalPreStatusHonestwardError(
                    "opposite_of_true_status is invalid"
                )
            required_truth = "FAIL" if opposite_of_true_status == "PASS" else "PASS"
        targets = _root_targets(
            self._observations,
            excluded_contrast=exclude_contrast,
            required_truth=required_truth,
        )
        local, roots, pairs = self._local(edges, targets)
        all_deltas = [target.mean_delta for target in targets.values()]
        global_mean = _mean(all_deltas, self.shape)
        zero = np.zeros(self.shape, dtype=np.float32)
        if local is None:
            arrays = tuple(_readonly(value) for value in (zero, zero, global_mean, zero, zero, zero, zero))
            return HonestwardPrediction(
                query_root_id,
                False,
                *arrays,
                (),
                (),
                0,
                self.scalar_dose_by_layer,
            )
        support_deltas = [targets[root].mean_delta for root in roots]
        nearest = support_deltas[0]
        shuffled_roots = sorted(
            targets,
            key=lambda root: (_digest("pre-status-honestward-shuffle-v1", root), root),
        )
        shifted = shuffled_roots[1:] + shuffled_roots[:1]
        shuffle_map = dict(zip(shuffled_roots, shifted, strict=True))
        shuffled = _mean([targets[shuffle_map[root]].mean_delta for root in roots], self.shape)
        random = np.zeros(self.shape, dtype=np.float64)
        for root, delta in zip(roots, support_deltas, strict=True):
            sign = 1.0 if _digest("pre-status-honestward-random-v1", query_root_id, root)[0] & 1 else -1.0
            random += sign * delta
        local_norm = np.linalg.norm(local, axis=1)
        random_norm = np.linalg.norm(random, axis=1)
        for layer in range(self.shape[0]):
            if random_norm[layer] > 1e-12:
                random[layer] *= local_norm[layer] / random_norm[layer]
            else:
                random[layer] = 0.0
        calibrated = local * self.scalar_dose_by_layer[:, None]
        arrays = tuple(
            _readonly(value)
            for value in (
                local,
                calibrated,
                global_mean,
                nearest,
                shuffled,
                -local,
                random,
            )
        )
        return HonestwardPrediction(
            query_root_id=query_root_id,
            defined=True,
            local=arrays[0],
            dose_calibrated_local=arrays[1],
            global_mean=arrays[2],
            nearest=arrays[3],
            shuffled=arrays[4],
            sign_flipped=arrays[5],
            random_local_span=arrays[6],
            support_root_ids=roots,
            support_pair_ids=pairs,
            support_count=len(roots),
            scalar_dose_by_layer=self.scalar_dose_by_layer,
        )


__all__ = [
    "HonestwardCrossingObservation",
    "HonestwardPrediction",
    "HonestwardSupportEdge",
    "RelationalPreStatusHonestwardError",
    "SharedPreStatusHonestwardField",
]
