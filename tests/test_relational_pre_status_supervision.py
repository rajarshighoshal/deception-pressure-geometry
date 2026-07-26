from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from geoprobe.data.relational_pre_status_rooted_star import VIEWS
from geoprobe.eval import relational_pre_status_supervision as supervision
from geoprobe.io import file_sha256


def _sha(seed: str) -> str:
    return sha256(seed.encode()).hexdigest()


def _reference(*, event_id: str, prefix: str, geometry: str, reference_id: str, value: float, family: str = "family", fold: str = "fold", turn: int = 1) -> SimpleNamespace:
    source = SimpleNamespace(
        prefix_state_sha256=prefix, family_fold=fold, turn_index=turn,
        occurrence_id=f"occ:{reference_id}", conversation_id=f"conversation:{reference_id}",
        realization_sha256=_sha(f"realization:{reference_id}"), canonical_realization_id=_sha(f"canonical:{reference_id}"),
        source_row_sha256=_sha(f"row:{reference_id}"), source_tensor_sha256=_sha(f"tensor:{reference_id}"),
    )
    return SimpleNamespace(
        rooted_star_id=_sha(f"rooted:{reference_id}"), reference_id=reference_id,
        field_event_id=event_id, view=None, geometry_sha256=geometry, source_reference=source,
        family=family, value=value,
    )


def _index(items: list[SimpleNamespace]) -> SimpleNamespace:
    references: list[SimpleNamespace] = []
    for item in items:
        for view in VIEWS:
            references.append(SimpleNamespace(**{**item.__dict__, "view": view}))
    by_reference: dict[str, list[SimpleNamespace]] = {}
    for reference in references:
        by_reference.setdefault(reference.reference_id, []).append(reference)
    return SimpleNamespace(references=tuple(references), by_reference_id={key: tuple(value) for key, value in by_reference.items()})


def _outcome(event_id: str, *, prefix: str, outcome: str, knowledge: bool = True, family: str = "family", fold: str = "fold", turn: int = 1) -> dict[str, object]:
    cohort = {
        "HONEST": "HONEST",
        "DECEPTIVE": "DECEPTIVE_WITH_KNOWLEDGE",
        "NO_ACTION": "NO_ACTION",
        "SKIP": "SKIP",
        "WRONG_WITHOUT_BASELINE_KNOWLEDGE": "WRONG_WITHOUT_BASELINE_KNOWLEDGE",
    }[outcome]
    return {
        "field_event_id": event_id, "outcome_class": outcome, "scientific_cohort": cohort,
        "mapped_action": "PASS", "knowledge_correct": knowledge,
        "family": family, "family_fold": fold, "scenario_id": "scenario", "orbit_id": "orbit",
        "turn_index": turn, "intervention_history": ["A"], "pressure_exposed": True,
        "true_status": "PASS", "desired_status": "FAIL", "prefix_state_sha256": prefix,
    }


def _member(reference: SimpleNamespace) -> dict[str, object]:
    source = reference.source_reference
    return {
        "reference_id": reference.reference_id, "field_event_id": reference.field_event_id,
        "occurrence_id": source.occurrence_id, "conversation_id": source.conversation_id,
        "realization_sha256": source.realization_sha256, "canonical_realization_id": source.canonical_realization_id,
        "source_row_sha256": source.source_row_sha256, "source_tensor_sha256": source.source_tensor_sha256,
    }


def _endpoint(reference: SimpleNamespace | list[SimpleNamespace]) -> dict[str, object]:
    references = reference if isinstance(reference, list) else [reference]
    first = references[0]
    source = first.source_reference
    return {
        "field_name": "status", "prefix_state_sha256": source.prefix_state_sha256,
        "family": first.family, "family_fold": source.family_fold, "turn_index": source.turn_index,
        "members": [_member(item) for item in references], "reference_ids": [item.reference_id for item in references],
    }


def _write(path: Path, value: dict[str, object], *, hash_key: str | None = None) -> Path:
    if hash_key is not None:
        value[hash_key] = sha256(json.dumps({key: item for key, item in value.items() if key != hash_key}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def test_label_free_quotient_keeps_mixed_outcomes_outside_node_construction() -> None:
    prefix = _sha("shared-prefix")
    index = _index([
        _reference(event_id="honest", prefix=prefix, geometry=_sha("geometry"), reference_id=_sha("honest-ref"), value=1.0),
        _reference(event_id="deceptive", prefix=prefix, geometry=_sha("geometry"), reference_id=_sha("deceptive-ref"), value=1.0),
    ])

    quotient = supervision.build_label_free_prefix_state_quotient(index)

    assert len(quotient.nodes) == len(VIEWS)
    for view in VIEWS:
        node_id = quotient.event_to_node_ids["honest"][view]
        assert node_id == quotient.event_to_node_ids["deceptive"][view]
        node = next(item for item in quotient.nodes if item.node_id == node_id)
        assert node.event_ids == ("deceptive", "honest")
        assert len(node.representative_references) == 1


def test_opening_averages_unique_replays_and_orients_deceptive_to_honest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deceptive = _reference(event_id="deceptive", prefix=_sha("d-prefix"), geometry=_sha("d-one"), reference_id=_sha("d-one-ref"), value=1.0)
    deceptive_replay = _reference(event_id="deceptive", prefix=deceptive.source_reference.prefix_state_sha256, geometry=_sha("d-two"), reference_id=_sha("d-two-ref"), value=3.0)
    honest = _reference(event_id="honest", prefix=_sha("h-prefix"), geometry=_sha("h-one"), reference_id=_sha("h-one-ref"), value=8.0)
    index = _index([deceptive, deceptive_replay, honest])
    outcomes = {"schema_version": 1, "status": "success", "scored_events": [
        _outcome("deceptive", prefix=deceptive.source_reference.prefix_state_sha256, outcome="DECEPTIVE"),
        _outcome("honest", prefix=honest.source_reference.prefix_state_sha256, outcome="HONEST"),
    ]}
    outcome_path = _write(tmp_path / "outcomes.json", outcomes, hash_key="report_sha256")
    edge = {
        "direction": "forward", "edge_pair_id": _sha("pair"), "contrast_id": "contrast", "scenario_id": "scenario",
        "source": _endpoint(honest), "target": _endpoint([deceptive, deceptive_replay]),
    }
    roster = {"schema_version": 1, "status": "frozen_label_free", "edges": [edge]}
    roster["edge_roster_sha256"] = sha256(json.dumps(roster["edges"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    roster_path = _write(tmp_path / "roster.json", roster)

    def load(_index: object, reference: SimpleNamespace) -> torch.Tensor:
        return torch.full((4, 2), reference.value, dtype=torch.float32)

    monkeypatch.setattr(supervision, "load_rooted_star_root_residuals", load)
    result = supervision.build_relational_pre_status_supervision(
        index, outcome_report_path=outcome_path, expected_outcome_report_sha256=file_sha256(outcome_path),
        roster_path=roster_path, expected_roster_sha256=file_sha256(roster_path),
        expected_outcome_event_count=2, expected_forward_edge_count=1,
    )

    for view in VIEWS:
        crossing = result.honestward_observations_by_view[view][0]
        assert crossing.deceptive_root_id == result.event_to_node_ids["deceptive"][view]
        assert crossing.honest_root_id == result.event_to_node_ids["honest"][view]
        assert crossing.delta.tolist() == [[6.0, 6.0]] * 4
    assert result.edge_outcome_transition_counts == {"DECEPTIVE <-> HONEST": 1}


def test_opening_rejects_corrupted_physical_input(tmp_path: Path) -> None:
    reference = _reference(event_id="event", prefix=_sha("prefix"), geometry=_sha("geometry"), reference_id=_sha("reference"), value=1.0)
    index = _index([reference])
    report = {"schema_version": 1, "status": "success", "scored_events": [_outcome("event", prefix=reference.source_reference.prefix_state_sha256, outcome="HONEST")]}
    outcome_path = _write(tmp_path / "outcomes.json", report, hash_key="report_sha256")
    roster = {"schema_version": 1, "status": "frozen_label_free", "edges": [], "edge_roster_sha256": sha256(b"[]").hexdigest()}
    roster_path = _write(tmp_path / "roster.json", roster)

    with pytest.raises(supervision.RelationalPreStatusSupervisionError, match="expected physical SHA"):
        supervision.build_relational_pre_status_supervision(
            index, outcome_report_path=outcome_path, expected_outcome_report_sha256=_sha("wrong"),
            roster_path=roster_path, expected_roster_sha256=file_sha256(roster_path), expected_outcome_event_count=1,
        )
