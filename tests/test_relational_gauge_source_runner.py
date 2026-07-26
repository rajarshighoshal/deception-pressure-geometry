from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geoprobe.control.relational_gauge_controller import (
    GaugeControlProposal,
    GaugeControllerObservation,
    GaugeControllerState,
)
from geoprobe.control.relational_intrinsic_risk_field import (
    OUTCOME_CLASSES,
    PressureMatchedFieldEvaluation,
)
from geoprobe.data.relational_pre_status_rooted_star_store import (
    RootedStarObservationBinding,
)
from geoprobe.geometry.relational_gauge_atlas import GaugeChart, GaugeQueryState
from geoprobe.models.relational_gauge_replay import (
    GaugeRootArmForward,
    SOURCE_GAUGE_ARM_ORDER,
)
from geoprobe.models.relational_structured_action import int32_token_sha256
from geoprobe.runtime.relational_gauge_replay_observation import (
    FrozenPreStatusGaugeReplayRow,
    GaugeReplayObservationEvidence,
)
from geoprobe.runtime.relational_gauge_source_runner import (
    REQUIRED_EXECUTION_SOURCE_NAMES,
    RelationalGaugeSourceRunnerError,
    join_gauge_source_root_works,
    run_exact_prefix_gauge_source_replay,
)
from geoprobe.runtime.relational_pre_status_causal_runner import CausalRootWork


def _sha(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _event(identifier: str = "event") -> dict[str, object]:
    return {
        "event_id": identifier,
        "family": "family",
        "turn_index": 1,
        "scenario_id": "scenario",
        "true_status": "PASS",
        "desired_status": "FAIL",
        "knowledge_status": "PASS",
        "knowledge_correct": True,
        "rng_seed": 7,
    }


def _causal(root: str = "root") -> CausalRootWork:
    prefix = (1, 2583, 25)
    return CausalRootWork(
        root_id=root,
        family_fold="outer_1",
        vector_tensor_row_index=0,
        prefix_token_ids=prefix,
        prefix_token_ids_sha256=int32_token_sha256(prefix),
        events=(_event(),),
        bundle=SimpleNamespace(),
    )


def _replay(root: str = "root") -> FrozenPreStatusGaugeReplayRow:
    binding = RootedStarObservationBinding(
        rooted_star_id="star",
        reference_id="reference",
        view="intervention_masked_action_free",
        geometry_sha256=_sha("geometry"),
        prefix_token_count=3,
        retained_token_indices=np.asarray([0, 2]),
        root_residuals=np.ones((4, 3), dtype=np.float32),
        incoming_attention=np.full((4, 2, 2), 0.5, dtype=np.float32),
    )
    return FrozenPreStatusGaugeReplayRow(
        row_id=root,
        held_out_family_fold="outer_1",
        root_id=root,
        expected_token_ids=(1, 2583, 25),
        nuisance_key=("1", "[]", "pressure", "PASS", "FAIL"),
        rooted_stars=(binding,),
    )


def _observation() -> GaugeControllerObservation:
    chart = GaugeChart(
        chart_id="chart",
        center_node_id="a",
        support_ids=("a", "b"),
        support_distances=((0.0, 1.0), (1.0, 0.0)),
        coordinates=np.asarray([[0.0], [1.0]]),
        eigenvalues=np.asarray([1.0]),
        stress=0.0,
        support_radius=1.0,
    )
    return GaugeControllerObservation(
        observation_id="gauge-replay:" + _sha("evidence"),
        chart=chart,
        query=GaugeQueryState(
            chart_id="chart",
            query_coordinates=np.asarray([0.25]),
            nearest_node_id="a",
            nearest_node_distance=0.25,
            stress=0.01,
            support_status=True,
            support_reason="test",
        ),
        nuisance_key=("1", "[]", "pressure", "PASS", "FAIL"),
        randomization_key="sealed-root",
    )


def _field() -> PressureMatchedFieldEvaluation:
    probabilities = np.full(len(OUTCOME_CLASSES), 1.0 / len(OUTCOME_CLASSES))
    return PressureMatchedFieldEvaluation(
        chart_id="chart",
        nuisance_key=("1", "[]", "pressure", "PASS", "FAIL"),
        defined=True,
        reason="defined",
        outcome_probabilities=probabilities,
        nuisance_probabilities=probabilities,
        absolute_deception_log_odds=1.0,
        pressure_residual_deception_log_odds=0.5,
        absolute_gradient=np.asarray([1.0]),
        pressure_residual_gradient=np.asarray([0.5]),
        support_node_ids=("a", "b"),
        support_observation_count=4,
        effective_observation_count=4.0,
        bandwidth=1.0,
    )


def _proposal(arm: str) -> GaugeControlProposal:
    active = arm != "no_intervention"
    fiber = (
        np.ones((4, 3), dtype=np.float32)
        if active
        else np.zeros((4, 3), dtype=np.float32)
    )
    return GaugeControlProposal(
        observation_id=_observation().observation_id,
        arm=arm,
        field=_field(),
        status="active" if active else "no_intervention",
        stop=not active,
        intrinsic_direction=np.asarray([1.0 if active else 0.0]),
        intrinsic_step=np.asarray([0.1 if active else 0.0]),
        intrinsic_step_norm=0.1 if active else 0.0,
        distance_to_boundary=1.0 if active else 0.0,
        remaining_path_budget=1.0 if active else 0.0,
        fiber_step=fiber,
        fiber_scale=1.0 if active else 0.0,
        layer_fiber_norms=np.linalg.norm(fiber, axis=1),
        layer_fiber_norm_caps=np.full(4, 4.0),
        next_state=(
            GaugeControllerState(step_count=1, cumulative_intrinsic_length=0.1)
            if active
            else GaugeControllerState(stopped=True, stop_reason="no_intervention")
        ),
    )


class _Builder:
    def evidence(self, observation_id: str) -> GaugeReplayObservationEvidence:
        return GaugeReplayObservationEvidence(
            observation_id=observation_id,
            row_id="root",
            root_id="root",
            held_out_family_fold="outer_1",
            chart_id="chart",
            token_ids_sha256=_sha("tokens"),
            rooted_geometry_sha256=_sha("geometry"),
            live_root_residuals_sha256=_sha("roots"),
            live_incoming_attention_sha256=_sha("attention"),
            minimum_residual_cosine=1.0,
            maximum_residual_norm_ratio_deviation=0.0,
            minimum_attention_correlation=1.0,
            maximum_residual_error=0.0,
            maximum_attention_error=0.0,
            evidence_sha256=_sha("evidence"),
        )


class _Tokenizer:
    @staticmethod
    def decode(
        tokens: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return str(tokens[0])


def _forward(
    _model: object,
    work: object,
    _builder: object,
    _controllers: object,
) -> GaugeRootArmForward:
    logits = torch.full((4, 83000), -20.0)
    logits[:, 51935] = 20.0
    return GaugeRootArmForward(
        row_id=work.causal.root_id,
        observation=_observation(),
        proposals={arm: _proposal(arm) for arm in SOURCE_GAUGE_ARM_ORDER},
        logits=logits,
    )


def _run(tmp_path: Path, *, resume: bool = False) -> dict[str, object]:
    works = join_gauge_source_root_works(
        (_causal(),), {"outer_1": (_replay(),)}
    )
    manifest = run_exact_prefix_gauge_source_replay(
        works,
        model=object(),
        tokenizer=_Tokenizer(),
        observation_builders={"outer_1": _Builder()},
        controllers_by_fold={
            "outer_1": {arm: object() for arm in SOURCE_GAUGE_ARM_ORDER}
        },
        out=tmp_path / "rows.jsonl",
        manifest_out=tmp_path / "manifest.json",
        status_out=tmp_path / "STATUS",
        controller_artifact_sha256=_sha("controller"),
        expected_root_count=1,
        expected_event_count=1,
        input_hashes={"plan": _sha("plan")},
        model_provenance={"model": "test"},
        runtime_provenance={"runtime": "test"},
        execution_source_sha256={
            name: _sha(name) for name in REQUIRED_EXECUTION_SOURCE_NAMES
        },
        resume=resume,
        forward_fn=_forward,
    )
    return dict(manifest)


def test_join_requires_exact_complete_root_inventory() -> None:
    works = join_gauge_source_root_works(
        (_causal(),), {"outer_1": (_replay(),)}
    )
    assert len(works) == 1
    assert works[0].causal.root_id == works[0].replay.root_id
    with pytest.raises(RelationalGaugeSourceRunnerError, match="inventories"):
        join_gauge_source_root_works(
            (_causal(),), {"outer_1": (_replay("different"),)}
        )


def test_runner_retains_all_four_arms_and_binds_joint_observation(tmp_path: Path) -> None:
    manifest = _run(tmp_path)
    assert manifest["status"] == "success"
    assert manifest["completed_rows"] == 4
    rows = [
        json.loads(line)
        for line in (tmp_path / "rows.jsonl").read_text().splitlines()
    ]
    assert [row["arm"] for row in rows] == list(SOURCE_GAUGE_ARM_ORDER)
    assert rows[1]["actuation_layers"] == [12, 16, 19, 20]
    assert rows[0]["actuation_layers"] == []
    assert rows[1]["observation"]["live_root_residuals_sha256"] == _sha("roots")
    assert rows[1]["observation"]["live_incoming_attention_sha256"] == _sha(
        "attention"
    )
    assert rows[1]["proposal"]["fiber_shape"] == [4, 3]
    assert set(rows[1]["proposal"]["per_layer_fiber_sha256"]) == {
        "12",
        "16",
        "19",
        "20",
    }
    assert len(rows[1]["proposal"]["proposal_sha256"]) == 64
    assert rows[1]["sampling"] == manifest["contract"]["sampling"]
    assert manifest["contract"]["sampling"]["full_vocabulary"] is True
    assert set(manifest["contract"]["execution_source_sha256"]) == set(
        REQUIRED_EXECUTION_SOURCE_NAMES
    )


def test_complete_resume_is_idempotent_and_mutation_fails(tmp_path: Path) -> None:
    first = _run(tmp_path)
    second = _run(tmp_path, resume=True)
    assert second["rows_content_sha256"] == first["rows_content_sha256"]
    rows_path = tmp_path / "rows.jsonl"
    rows = rows_path.read_text().splitlines()
    payload = json.loads(rows[0])
    payload["proposal"]["status"] = "boundary_exit"
    rows[0] = json.dumps(payload, sort_keys=True)
    rows_path.write_text("\n".join(rows) + "\n")
    with pytest.raises(RelationalGaugeSourceRunnerError, match="manifest"):
        _run(tmp_path, resume=True)
