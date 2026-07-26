from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from geoprobe.data.relational_pre_status_rooted_star_store import (
    RootedStarObservationBinding,
)
from geoprobe.runtime import relational_gauge_source_replay as source_replay


def _binding() -> RootedStarObservationBinding:
    return RootedStarObservationBinding(
        rooted_star_id="rooted",
        reference_id="reference",
        view="intervention_masked_action_free",
        geometry_sha256="a" * 64,
        prefix_token_count=3,
        retained_token_indices=np.asarray([0, 2]),
        root_residuals=np.ones((4, 3), dtype=np.float32),
        incoming_attention=np.full((4, 2, 2), 0.5, dtype=np.float32),
    )


def _inputs() -> tuple[object, object, object]:
    source = SimpleNamespace(
        prefix_token_ids_sha256="prefix-sha",
        prefix_state_sha256="state-sha",
        turn_index=1,
        intervention_history=("A",),
        pressure_exposed=True,
    )
    reference = SimpleNamespace(
        view="intervention_masked_action_free",
        source_reference=source,
    )
    node = SimpleNamespace(
        node_id="root",
        view="intervention_masked_action_free",
        family_fold="outer_1",
        prefix_state_sha256="state-sha",
        representative_references=(reference,),
    )
    event = {
        "event_id": "event",
        "turn_index": 1,
        "true_status": "PASS",
        "desired_status": "FAIL",
    }
    work = SimpleNamespace(
        root_id="root",
        family_fold="outer_1",
        prefix_token_ids=(1, 2583, 25),
        prefix_token_ids_sha256="prefix-sha",
        events=(event,),
    )
    index = SimpleNamespace(by_field_event_id={"event": (reference,)})
    quotient = SimpleNamespace(nodes=(node,))
    return index, work, quotient


def test_build_rows_joins_exact_prefix_query_and_nuisance(monkeypatch: pytest.MonkeyPatch) -> None:
    index, work, quotient = _inputs()
    monkeypatch.setattr(
        source_replay, "build_label_free_prefix_state_quotient", lambda _index: quotient
    )
    monkeypatch.setattr(
        source_replay,
        "load_rooted_star_observation_binding",
        lambda _index, _reference: _binding(),
    )
    rows = source_replay.build_frozen_pre_status_gauge_replay_rows(
        index, (work,)
    )
    row = rows["outer_1"][0]
    assert row.row_id == "root"
    assert row.root_id == "root"
    assert row.expected_token_ids == (1, 2583, 25)
    assert row.nuisance_key == ("1", '["A"]', "pressure", "PASS", "FAIL")
    assert len(row.rooted_stars) == 1


def test_events_sharing_a_root_must_share_nuisance(monkeypatch: pytest.MonkeyPatch) -> None:
    index, work, quotient = _inputs()
    source_two = SimpleNamespace(
        prefix_token_ids_sha256="prefix-sha",
        prefix_state_sha256="state-sha",
        turn_index=1,
        intervention_history=("B",),
        pressure_exposed=True,
    )
    reference_two = SimpleNamespace(
        view="intervention_masked_action_free", source_reference=source_two
    )
    event_two = {
        "event_id": "event-two",
        "turn_index": 1,
        "true_status": "PASS",
        "desired_status": "FAIL",
    }
    index.by_field_event_id["event-two"] = (reference_two,)
    work.events = (*work.events, event_two)
    monkeypatch.setattr(
        source_replay, "build_label_free_prefix_state_quotient", lambda _index: quotient
    )
    monkeypatch.setattr(
        source_replay,
        "load_rooted_star_observation_binding",
        lambda _index, _reference: _binding(),
    )
    with pytest.raises(source_replay.RelationalGaugeSourceReplayError, match="nuisance"):
        source_replay.build_frozen_pre_status_gauge_replay_rows(index, (work,))
