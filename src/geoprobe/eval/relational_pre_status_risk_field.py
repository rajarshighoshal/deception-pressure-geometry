"""Fold-safe multinomial risk field on the action-free pre-status graph."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

from geoprobe.eval.relational_outcome_events import OUTCOME_CLASSES


SMOOTHING = 0.5


class RelationalPreStatusRiskFieldError(ValueError):
    """A pre-status risk fit or query violates event/fold boundaries."""


@dataclass(frozen=True, slots=True)
class PreStatusRiskEvent:
    event_id: str
    root_id: str
    family: str
    family_fold: str
    outcome_class: str
    nuisance_key: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("event_id", "root_id", "family", "family_fold"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise RelationalPreStatusRiskFieldError(f"{name} is invalid")
        if self.outcome_class not in OUTCOME_CLASSES:
            raise RelationalPreStatusRiskFieldError("outcome_class is invalid")
        if not self.nuisance_key or any(
            not isinstance(value, str) or not value for value in self.nuisance_key
        ):
            raise RelationalPreStatusRiskFieldError("nuisance_key is invalid")


@dataclass(frozen=True, slots=True)
class PreStatusRiskPrediction:
    event_id: str
    root_id: str
    local_probabilities: Mapping[str, float]
    nuisance_probabilities: Mapping[str, float]
    base_probabilities: Mapping[str, float]
    support_event_ids: tuple[str, ...]
    support_root_ids: tuple[str, ...]
    support_count: int


def _edge(edge: object) -> tuple[str, int, float]:
    try:
        target = str(getattr(edge, "target_id"))
        rank = getattr(edge, "rank")
        score = float(getattr(edge, "joint_score"))
    except (AttributeError, TypeError, ValueError) as error:
        raise RelationalPreStatusRiskFieldError("risk support edge is invalid") from error
    if not target or not isinstance(rank, int) or rank < 1 or not math.isfinite(score):
        raise RelationalPreStatusRiskFieldError("risk support edge is invalid")
    return target, rank, score


def _probabilities(
    outcomes: Sequence[str], *, fallback: Mapping[str, float] | None = None
) -> Mapping[str, float]:
    if not outcomes:
        if fallback is None:
            raise RelationalPreStatusRiskFieldError("empty risk support has no fallback")
        return MappingProxyType(dict(fallback))
    counts = Counter(outcomes)
    denominator = len(outcomes) + SMOOTHING * len(OUTCOME_CLASSES)
    return MappingProxyType(
        {
            label: float((counts[label] + SMOOTHING) / denominator)
            for label in OUTCOME_CLASSES
        }
    )


class FoldSafePreStatusRiskField:
    """Local event-count field with exact nuisance and unconditional controls."""

    def __init__(
        self,
        *,
        held_out_family_fold: str,
        events: tuple[PreStatusRiskEvent, ...],
    ) -> None:
        self.held_out_family_fold = held_out_family_fold
        by_root: dict[str, list[PreStatusRiskEvent]] = defaultdict(list)
        by_nuisance: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for event in events:
            by_root[event.root_id].append(event)
            by_nuisance[event.nuisance_key].append(event.outcome_class)
        self._by_root = MappingProxyType(
            {
                root: tuple(sorted(rows, key=lambda row: row.event_id))
                for root, rows in by_root.items()
            }
        )
        self._base = _probabilities([event.outcome_class for event in events])
        self._nuisance = MappingProxyType(
            {key: _probabilities(values) for key, values in by_nuisance.items()}
        )

    @classmethod
    def fit(
        cls,
        events: Sequence[PreStatusRiskEvent],
        *,
        held_out_family_fold: str,
    ) -> FoldSafePreStatusRiskField:
        rows = tuple(events)
        if not rows:
            raise RelationalPreStatusRiskFieldError("risk fit requires events")
        if len({row.event_id for row in rows}) != len(rows):
            raise RelationalPreStatusRiskFieldError("risk event IDs are not unique")
        if any(row.family_fold == held_out_family_fold for row in rows):
            raise RelationalPreStatusRiskFieldError(
                "held-out event entered risk training"
            )
        return cls(held_out_family_fold=held_out_family_fold, events=rows)

    def predict(
        self,
        *,
        event_id: str,
        root_id: str,
        nuisance_key: tuple[str, ...],
        edges: Sequence[object],
    ) -> PreStatusRiskPrediction:
        if not event_id or not root_id:
            raise RelationalPreStatusRiskFieldError("risk query identity is invalid")
        ordered = sorted((_edge(edge) for edge in edges), key=lambda row: (row[1], row[2], row[0]))
        support_roots: list[str] = []
        support_events: list[PreStatusRiskEvent] = []
        seen_events: set[str] = set()
        for target_id, _, _ in ordered:
            if target_id in support_roots:
                continue
            rows = self._by_root.get(target_id)
            if rows is None:
                continue
            support_roots.append(target_id)
            for row in rows:
                if row.event_id not in seen_events:
                    support_events.append(row)
                    seen_events.add(row.event_id)
        local = _probabilities(
            [row.outcome_class for row in support_events], fallback=self._base
        )
        nuisance = self._nuisance.get(nuisance_key, self._base)
        return PreStatusRiskPrediction(
            event_id=event_id,
            root_id=root_id,
            local_probabilities=local,
            nuisance_probabilities=nuisance,
            base_probabilities=self._base,
            support_event_ids=tuple(row.event_id for row in support_events),
            support_root_ids=tuple(support_roots),
            support_count=len(support_events),
        )


def multiclass_log_loss(
    outcome_class: str, probabilities: Mapping[str, float]
) -> float:
    if outcome_class not in OUTCOME_CLASSES or set(probabilities) != set(OUTCOME_CLASSES):
        raise RelationalPreStatusRiskFieldError("log-loss inputs are invalid")
    value = float(probabilities[outcome_class])
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise RelationalPreStatusRiskFieldError("outcome probability is invalid")
    return float(-math.log(value))


def multiclass_brier(
    outcome_class: str, probabilities: Mapping[str, float]
) -> float:
    if outcome_class not in OUTCOME_CLASSES or set(probabilities) != set(OUTCOME_CLASSES):
        raise RelationalPreStatusRiskFieldError("Brier inputs are invalid")
    values = np.asarray([probabilities[label] for label in OUTCOME_CLASSES], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0.0) or not np.isclose(values.sum(), 1.0):
        raise RelationalPreStatusRiskFieldError("probability vector is invalid")
    target = np.zeros(len(OUTCOME_CLASSES), dtype=float)
    target[OUTCOME_CLASSES.index(outcome_class)] = 1.0
    return float(np.sum((values - target) ** 2))


__all__ = [
    "SMOOTHING",
    "FoldSafePreStatusRiskField",
    "PreStatusRiskEvent",
    "PreStatusRiskPrediction",
    "RelationalPreStatusRiskFieldError",
    "multiclass_brier",
    "multiclass_log_loss",
]
