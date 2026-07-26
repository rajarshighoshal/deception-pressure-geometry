"""Helpers for canonical outcome-event handling in relational analyses."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final


class RelationalOutcomeEventsError(ValueError):
    """Raised when immutable outcome-event records are inconsistent."""


OUTCOME_CLASSES: Final[tuple[str, ...]] = (
    "HONEST",
    "DECEPTIVE",
    "SKIP",
    "NO_ACTION",
    "WRONG_WITHOUT_BASELINE_KNOWLEDGE",
)

_SCI_COHORT_TO_OUTCOME: Final[MappingProxyType[str, str]] = MappingProxyType(
    {
        "HONEST": "HONEST",
        "DECEPTIVE_WITH_KNOWLEDGE": "DECEPTIVE",
        "SKIP": "SKIP",
        "NO_ACTION": "NO_ACTION",
        "WRONG_WITHOUT_BASELINE_KNOWLEDGE": "WRONG_WITHOUT_BASELINE_KNOWLEDGE",
    }
)


def outcome_class_from_scientific_cohort(scientific_cohort: Any) -> str:
    value = _string(scientific_cohort, "scientific_cohort")
    try:
        return _SCI_COHORT_TO_OUTCOME[value]
    except KeyError as error:
        raise RelationalOutcomeEventsError("unsupported scientific_cohort") from error


def build_zero_filled_outcome_counts(classes: Iterable[str]) -> "OutcomeCountVector":
    normalized = tuple(_validate_class(value, "outcome class") for value in classes)
    mapped_counts = Counter(normalized)
    total = len(normalized)
    class_counts = MappingProxyType(
        {label: mapped_counts[label] for label in OUTCOME_CLASSES}
    )
    class_rates = MappingProxyType(
        {
            label: (mapped_counts[label] / total if total else 0.0)
            for label in OUTCOME_CLASSES
        }
    )
    return OutcomeCountVector(total, class_counts, class_rates)


@dataclass(frozen=True, slots=True)
class OutcomeCountVector:
    unique_field_event_count: int
    class_counts: Mapping[str, int]
    class_rates: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class OutcomeEventUnit:
    field_event_id: str
    field_name: str
    turn_index: int
    scientific_cohort: str
    outcome_class: str
    occurrence_ids: tuple[str, ...]

    @property
    def multiplicity(self) -> int:
        return len(self.occurrence_ids)


def collapse_outcome_occurrences(
    records: Sequence[Mapping[str, Any]]
) -> tuple[OutcomeEventUnit, ...]:
    state: dict[str, dict[str, Any]] = {}
    occurrence_owners: dict[str, str] = {}
    for record in records:
        event = _mapping(record, "outcome record")
        field_event_id = _string(event.get("field_event_id"), "field_event_id")
        field_name = _string(event.get("field_name"), "field_name")
        turn_index = _integer(event.get("turn_index"), "turn_index")
        occurrence_id = _string(event.get("occurrence_id"), "occurrence_id")
        if occurrence_id in occurrence_owners:
            raise RelationalOutcomeEventsError(
                f"duplicate occurrence_id={occurrence_id!r}"
            )
        occurrence_owners[occurrence_id] = field_event_id
        scientific_cohort = _string(event.get("scientific_cohort"), "scientific_cohort")
        outcome_class = outcome_class_from_scientific_cohort(scientific_cohort)

        current = state.get(field_event_id)
        if current is None:
            state[field_event_id] = {
                "field_name": field_name,
                "turn_index": turn_index,
                "scientific_cohort": scientific_cohort,
                "outcome_class": outcome_class,
                "occurrence_ids": [occurrence_id],
            }
            continue

        if current["field_name"] != field_name or current["turn_index"] != turn_index:
            raise RelationalOutcomeEventsError(
                f"inconsistent identities for field_event_id={field_event_id!r}"
            )
        if current["scientific_cohort"] != scientific_cohort:
            raise RelationalOutcomeEventsError(
                f"inconsistent outcomes for field_event_id={field_event_id!r}"
            )
        current["occurrence_ids"].append(occurrence_id)

    groups: list[OutcomeEventUnit] = []
    for field_event_id in sorted(state):
        current = state[field_event_id]
        groups.append(
            OutcomeEventUnit(
                field_event_id=field_event_id,
                field_name=current["field_name"],
                turn_index=current["turn_index"],
                scientific_cohort=current["scientific_cohort"],
                outcome_class=current["outcome_class"],
                occurrence_ids=tuple(sorted(current["occurrence_ids"])),
            )
        )
    return tuple(groups)


def _validate_class(value: Any, name: str) -> str:
    outcome_class = _string(value, name)
    if outcome_class not in OUTCOME_CLASSES:
        raise RelationalOutcomeEventsError(f"{name} is unsupported: {outcome_class}")
    return outcome_class


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelationalOutcomeEventsError(f"{name} must be an object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RelationalOutcomeEventsError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise RelationalOutcomeEventsError(f"{name} must be an integer >= {minimum}")
    return value


__all__ = [
    "OutcomeCountVector",
    "OutcomeEventUnit",
    "OUTCOME_CLASSES",
    "RelationalOutcomeEventsError",
    "build_zero_filled_outcome_counts",
    "collapse_outcome_occurrences",
    "outcome_class_from_scientific_cohort",
]
