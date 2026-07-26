from __future__ import annotations

from hashlib import sha256
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from geoprobe.control import relational_pre_status_vector_bank as bank
from geoprobe.data.relational_pre_status_rooted_star import VIEWS
from geoprobe.eval.relational_pre_status_supervision import StatusEventOutcome
from geoprobe.geometry.relational_pre_status_honestward import (
    HonestwardCrossingObservation,
)
from geoprobe.geometry.relational_pre_status_rooted_graph import (
    ExactGraphEdge,
    FoldExactRootedGraph,
)
from geoprobe.geometry.relational_pre_status_rooted_metric import RootedStarMetricScaler


def _sha(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _outcome(event: str, fold: str, truth: str = "PASS") -> StatusEventOutcome:
    return StatusEventOutcome(
        event_id=event,
        outcome_class="DECEPTIVE",
        knowledge_correct=True,
        family=f"family-{fold}",
        family_fold=fold,
        scenario_id=f"scenario-{event}",
        orbit_id=f"orbit-{event}",
        turn_index=1,
        intervention_history=(),
        pressure_exposed=False,
        true_status=truth,
        desired_status="FAIL" if truth == "PASS" else "PASS",
        prefix_state_sha256=_sha(event),
    )


def _endpoint(event: str, fold: str) -> MappingProxyType[str, str]:
    return MappingProxyType(
        {
            "event_id": event,
            "family": f"family-{fold}",
            "family_fold": fold,
            "prefix_state_sha256": _sha(event),
        }
    )


def _edge(pair: str, source: str, target: str, fold: str) -> MappingProxyType[str, Any]:
    return MappingProxyType(
        {
            "pair_id": pair,
            "contrast_id": f"contrast-{pair}",
            "scenario_id": f"scenario-{pair}",
            "source": _endpoint(source, fold),
            "target": _endpoint(target, fold),
        }
    )


def _node(event: str, fold: str) -> Any:
    token_hash = _sha(f"tokens:{event}")
    reference = SimpleNamespace(
        field_event_id=event,
        view=bank.MASKED_VIEW,
        source_reference=SimpleNamespace(prefix_token_ids_sha256=token_hash),
    )
    return SimpleNamespace(
        node_id=f"node:{event}",
        view=bank.MASKED_VIEW,
        prefix_state_sha256=_sha(event),
        family=f"family-{fold}",
        family_fold=fold,
        turn_index=1,
        event_ids=(event,),
        representative_references=(reference,),
    )


def _observation(root: str, fold: str, value: float) -> HonestwardCrossingObservation:
    return HonestwardCrossingObservation(
        pair_id=f"h:{root}",
        deceptive_root_id=root,
        honest_root_id=f"honest:{root}",
        family=f"family-{fold}",
        family_fold=fold,
        scenario_id=f"scenario:{root}",
        contrast_id=f"contrast:{root}",
        true_status="PASS",
        delta=np.full((4, 2), value, dtype=np.float32),
    )


def test_generic_observations_are_bidirectional_and_same_truth_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = {event: _node(event, "outer_1") for event in ("a", "b", "c")}
    event_nodes = {event: MappingProxyType({bank.MASKED_VIEW: node.node_id}) for event, node in nodes.items()}
    node_by_id = MappingProxyType({node.node_id: node for node in nodes.values()})
    outcomes = {"a": _outcome("a", "outer_1", "PASS"), "b": _outcome("b", "outer_1", "PASS"), "c": _outcome("c", "outer_1", "FAIL")}
    values = {node.node_id: np.full((4, 2), index, dtype=np.float32) for index, node in enumerate(nodes.values())}
    monkeypatch.setattr(bank, "_node_residual", lambda _index, node, _cache: values[node.node_id])
    rows = bank._generic_observations(
        index=SimpleNamespace(), nodes_by_id=node_by_id, event_to_nodes=event_nodes,
        outcomes=outcomes, training_edges=(_edge("same", "a", "b", "outer_1"), _edge("mixed", "a", "c", "outer_1")),
    )
    assert len(rows) == 2
    assert {row.deceptive_root_id for row in rows} == {"node:a", "node:b"}
    np.testing.assert_allclose(rows[0].delta, -rows[1].delta)


def _build_fixture(monkeypatch: pytest.MonkeyPatch) -> bank.CausalVectorBankBuild:
    heldout = "outer_5"
    training_events = [(f"train-{index}", f"outer_{index}") for index in range(1, 5)]
    heldout_events = [("query-a", heldout), ("query-b", heldout)]
    all_nodes = [_node(event, fold) for event, fold in (*training_events, *heldout_events)]
    event_to_nodes = {
        node.event_ids[0]: MappingProxyType({bank.MASKED_VIEW: node.node_id, VIEWS[0]: f"full:{node.node_id}"})
        for node in all_nodes
    }
    quotient = SimpleNamespace(nodes=tuple(all_nodes), event_to_node_ids=MappingProxyType(event_to_nodes))
    index = SimpleNamespace(references=tuple(ref for node in all_nodes for ref in node.representative_references))
    node_by_event = {node.event_ids[0]: node for node in all_nodes}
    training_roots = [node_by_event[event].node_id for event, _ in training_events]

    def edges(source: str, targets: list[str]) -> tuple[ExactGraphEdge, ...]:
        return tuple(ExactGraphEdge(source, target, rank + 1, float(rank), 1.0, 1.0, rank + 1) for rank, target in enumerate(targets))

    graph = FoldExactRootedGraph(
        held_out_family_fold=heldout,
        graph_width=4,
        query_edges={node_by_event[event].node_id: edges(node_by_event[event].node_id, training_roots) for event, _ in heldout_events},
        training_edges={root: edges(root, [other for other in training_roots if other != root]) for root in training_roots},
        scaler=RootedStarMetricScaler(1.0, 1.0), candidate_pair_count=1, exact_pair_count=1,
    )
    h_rows = tuple(_observation(root, fold, float(index + 1)) for index, (root, (_, fold)) in enumerate(zip(training_roots, training_events, strict=True)))
    t_rows = tuple(_observation(root, fold, float(index + 2)) for index, (root, (_, fold)) in enumerate(zip(training_roots, training_events, strict=True)))
    monkeypatch.setattr(bank, "build_label_free_prefix_state_quotient", lambda _index: quotient)
    monkeypatch.setattr(bank, "_crossings", lambda *_args: MappingProxyType({bank.MASKED_VIEW: h_rows}))
    monkeypatch.setattr(bank, "_generic_observations", lambda **_kwargs: t_rows)
    return bank.build_pre_status_causal_vector_bank(
        index, held_out_family_fold=heldout,
        training_outcomes={event: _outcome(event, fold) for event, fold in training_events},
        roster_edges=(*(_edge(f"train-{fold}", event, event, fold) for event, fold in training_events), _edge("heldout-one", "query-a", "query-b", heldout), _edge("heldout-two", "query-a", "query-b", heldout)),
        graph=graph, artifact_bindings={"rooted_star_manifest_file_sha256": "a" * 64},
    )


def test_build_deduplicates_masked_candidates_and_preserves_algebra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = _build_fixture(monkeypatch)
    assert [row["root_id"] for row in build.rows] == ["node:query-a", "node:query-b"]
    assert all(row["prefix_state_sha256"] != row["prefix_token_ids_sha256"] for row in build.rows)
    np.testing.assert_allclose(build.arrays.h, build.arrays.t + build.arrays.s)
    assert build.arrays.defined.shape == (2, 4)


def test_persistence_roundtrip_rejects_tensor_corruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    build = _build_fixture(monkeypatch)
    ledger = bank.save_pre_status_causal_vector_bank(build, tmp_path)
    loaded_ledger, loaded = bank.load_pre_status_causal_vector_bank(tmp_path)
    assert loaded_ledger["ledger_sha256"] == ledger["ledger_sha256"]
    np.testing.assert_array_equal(loaded.g, build.arrays.g)
    path = tmp_path / bank.TENSOR_FILE_NAME
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(bank.RelationalPreStatusVectorBankError, match="physical tensor"):
        bank.load_pre_status_causal_vector_bank(tmp_path)


def test_build_rejects_heldout_outcome_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = _build_fixture(monkeypatch)
    del build
    # The same fixture's strict exact-coverage check is reached before fitting.
    # Recreate only enough state through the already-installed fixture patch.
    with pytest.raises(bank.RelationalPreStatusVectorBankError, match="exactly cover"):
        bank.build_pre_status_causal_vector_bank(
            SimpleNamespace(references=()), held_out_family_fold="outer_5",
            training_outcomes={"query-a": _outcome("query-a", "outer_5")},
            roster_edges=(),
            graph=SimpleNamespace(held_out_family_fold="outer_5"),
        )
