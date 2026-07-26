from __future__ import annotations

import pytest

from geoprobe.eval import relational_outcome_events as outcome_events


def _row(
    *,
    field_event_id: str,
    scientific_cohort: str,
    occurrence_id: str,
    field_name: str = "status",
    turn_index: int = 0,
) -> dict[str, str]:
    return {
        "field_event_id": field_event_id,
        "field_name": field_name,
        "turn_index": turn_index,
        "scientific_cohort": scientific_cohort,
        "occurrence_id": occurrence_id,
    }


def test_zero_fill_builds_five_class_vector() -> None:
    counts = outcome_events.build_zero_filled_outcome_counts(
        [
            "HONEST",
            "DECEPTIVE",
            "DECEPTIVE",
            "NO_ACTION",
        ]
    )
    assert counts.unique_field_event_count == 4
    assert counts.class_counts == {
        "HONEST": 1,
        "DECEPTIVE": 2,
        "SKIP": 0,
        "NO_ACTION": 1,
        "WRONG_WITHOUT_BASELINE_KNOWLEDGE": 0,
    }
    assert counts.class_rates["DECEPTIVE"] == 0.5
    assert counts.class_rates["WRONG_WITHOUT_BASELINE_KNOWLEDGE"] == 0.0


def test_zero_fill_accepts_single_pass_iterables() -> None:
    counts = outcome_events.build_zero_filled_outcome_counts(
        label for label in ("HONEST", "SKIP")
    )
    assert counts.unique_field_event_count == 2
    assert counts.class_counts["HONEST"] == 1
    assert counts.class_counts["SKIP"] == 1


def test_clone_occurrence_records_are_collapsed_with_multiplicity() -> None:
    grouped = outcome_events.collapse_outcome_occurrences(
        [
            _row(
                field_event_id="event-status-0",
                scientific_cohort="HONEST",
                occurrence_id="occ-a",
            ),
            _row(
                field_event_id="event-status-0",
                scientific_cohort="HONEST",
                occurrence_id="occ-c",
            ),
            _row(
                field_event_id="event-status-1",
                scientific_cohort="DECEPTIVE_WITH_KNOWLEDGE",
                occurrence_id="occ-b",
            ),
        ]
    )
    assert len(grouped) == 2
    assert grouped[0].field_event_id == "event-status-0"
    assert grouped[0].occurrence_ids == ("occ-a", "occ-c")
    assert grouped[0].multiplicity == 2
    assert grouped[0].outcome_class == "HONEST"


def test_divergent_same_prefix_events_are_retained_as_distinct() -> None:
    grouped = outcome_events.collapse_outcome_occurrences(
        [
            _row(
                field_event_id="turn-0:status-0",
                scientific_cohort="HONEST",
                occurrence_id="occ-0a",
            ),
            _row(
                field_event_id="turn-0:status-1",
                scientific_cohort="SKIP",
                occurrence_id="occ-1a",
            ),
        ]
    )
    assert [item.field_event_id for item in grouped] == [
        "turn-0:status-0",
        "turn-0:status-1",
    ]
    assert grouped[0].field_name == "status"
    assert grouped[1].field_name == "status"


@pytest.mark.parametrize(
    "rows",
    [
        [
            _row(
                field_event_id="event-a",
                scientific_cohort="HONEST",
                occurrence_id="occ-a",
            ),
            _row(
                field_event_id="event-a",
                scientific_cohort="SKIP",
                occurrence_id="occ-b",
            ),
        ],
        [
            _row(
                field_event_id="event-a",
                scientific_cohort="HONEST",
                occurrence_id="occ-a",
            ),
            _row(
                field_event_id="event-a",
                scientific_cohort="HONEST",
                occurrence_id="occ-b",
                field_name="caveat",
            ),
        ],
    ],
)
def test_inconsistent_event_records_are_rejected(rows: list[dict[str, str]]) -> None:
    with pytest.raises(outcome_events.RelationalOutcomeEventsError, match="inconsistent"):
        outcome_events.collapse_outcome_occurrences(rows)


def test_deterministic_output_is_order_and_counts_stable() -> None:
    rows = [
        _row(
            field_event_id="event-b",
            scientific_cohort="NO_ACTION",
            occurrence_id="occ-3",
        ),
        _row(
            field_event_id="event-a",
            scientific_cohort="HONEST",
            occurrence_id="occ-2",
            turn_index=1,
        ),
        _row(
            field_event_id="event-a",
            scientific_cohort="HONEST",
            occurrence_id="occ-1",
            turn_index=1,
        ),
        _row(
            field_event_id="event-c",
            scientific_cohort="DECEPTIVE_WITH_KNOWLEDGE",
            occurrence_id="occ-4",
        ),
    ]
    first = outcome_events.collapse_outcome_occurrences(rows)
    second = outcome_events.collapse_outcome_occurrences(list(reversed(rows)))
    assert first == second
    assert [item.field_event_id for item in first] == ["event-a", "event-b", "event-c"]
    assert first[0].occurrence_ids == ("occ-1", "occ-2")
    vector = outcome_events.build_zero_filled_outcome_counts(
        [item.outcome_class for item in first]
    )
    assert vector.class_counts["NO_ACTION"] == 1
    assert vector.class_counts["DECEPTIVE"] == 1
    assert vector.class_counts["HONEST"] == 1


def test_duplicate_occurrence_id_is_rejected() -> None:
    rows = [
        _row(
            field_event_id="event-a",
            scientific_cohort="HONEST",
            occurrence_id="occ-shared",
        ),
        _row(
            field_event_id="event-b",
            scientific_cohort="SKIP",
            occurrence_id="occ-shared",
        ),
    ]
    with pytest.raises(
        outcome_events.RelationalOutcomeEventsError,
        match="duplicate occurrence_id",
    ):
        outcome_events.collapse_outcome_occurrences(rows)
