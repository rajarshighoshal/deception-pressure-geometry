from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geoprobe.control.relational_gauge_controller import (
    GaugeControlProposal,
    GaugeControllerObservation,
    GaugeControllerState,
)
from geoprobe.geometry.relational_gauge_atlas import GaugeChart, GaugeQueryState
from geoprobe.models import relational_gauge_replay as replay


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
        observation_id="observation",
        chart=chart,
        query=GaugeQueryState(
            chart_id="chart",
            query_coordinates=np.asarray([0.25]),
            nearest_node_id="a",
            nearest_node_distance=0.25,
            stress=0.0,
            support_status=True,
            support_reason="test",
        ),
        nuisance_key=("test",),
    )


def _proposal(arm: str, *, active: bool) -> GaugeControlProposal:
    fiber = (
        np.arange(12, dtype=np.float32).reshape(4, 3) + 1.0
        if active
        else np.zeros((4, 3), dtype=np.float32)
    )
    return GaugeControlProposal(
        observation_id="observation",
        arm=arm,
        field=object(),
        status="active" if active else "no_intervention",
        stop=not active,
        intrinsic_direction=np.asarray([1.0]),
        intrinsic_step=np.asarray([0.25 if active else 0.0]),
        intrinsic_step_norm=0.25 if active else 0.0,
        distance_to_boundary=1.0,
        remaining_path_budget=1.0,
        fiber_step=fiber,
        fiber_scale=1.0 if active else 0.0,
        layer_fiber_norms=np.linalg.norm(fiber, axis=1),
        layer_fiber_norm_caps=np.full(4, 100.0),
        next_state=(
            GaugeControllerState(step_count=1, cumulative_intrinsic_length=0.25)
            if active
            else GaugeControllerState(stopped=True, stop_reason="no_intervention")
        ),
    )


def test_steering_batch_uses_all_four_layers_and_preserves_arm_order() -> None:
    proposals = {
        arm: _proposal(arm, active=arm != "no_intervention")
        for arm in replay.SOURCE_GAUGE_ARM_ORDER
    }
    steering = replay.build_gauge_steering_batch(proposals)
    assert steering[0] is None
    assert all(tuple(spec.layer for spec in arm) == (12, 16, 19, 20) for arm in steering[1:])


def test_steering_batch_rejects_stale_or_zero_active_proposals() -> None:
    proposals = {
        arm: _proposal(arm, active=arm != "no_intervention")
        for arm in replay.SOURCE_GAUGE_ARM_ORDER
    }
    proposals["sign_flipped"] = replace(
        proposals["sign_flipped"], observation_id="stale"
    )
    with pytest.raises(replay.RelationalGaugeReplayError, match="shared observation"):
        replay.build_gauge_steering_batch(proposals)

    proposals["sign_flipped"] = _proposal("sign_flipped", active=True)
    proposals["gauge_geodesic"] = replace(
        proposals["gauge_geodesic"],
        fiber_step=np.zeros((4, 3), dtype=np.float32),
        layer_fiber_norms=np.zeros(4),
    )
    with pytest.raises(replay.RelationalGaugeReplayError, match="no fiber"):
        replay.build_gauge_steering_batch(proposals)


def test_source_controller_factory_rejects_multistep_domain() -> None:
    bundle = SimpleNamespace(risk_field=object(), horizontal_lift=object())
    controllers = replay.build_source_gauge_controllers(bundle)
    assert tuple(controllers) == replay.SOURCE_GAUGE_ARM_ORDER
    assert {controller.config.maximum_steps for controller in controllers.values()} == {1}
    with pytest.raises(replay.RelationalGaugeReplayError, match="exactly one"):
        replay.build_source_gauge_controllers(
            bundle,
            base_config=SimpleNamespace(maximum_steps=2),
        )


def test_forward_captures_then_scores_all_arms_from_one_repeated_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Cache:
        def __init__(self) -> None:
            self.repeated = 1

        def batch_repeat_interleave(self, count: int) -> None:
            events.append(f"repeat:{count}")
            self.repeated = count

    prefix_cache = SimpleNamespace(
        past_key_values=Cache(), batch_size=1, context_length=2, last_token_id=25
    )
    monkeypatch.setattr(
        replay,
        "build_prefix_cache",
        lambda _model, _tokens, batch_size: (
            events.append(f"prefill:{batch_size}") or prefix_cache
        ),
    )
    monkeypatch.setattr(
        replay,
        "capture_relational_step_with_cache",
        lambda *_args, **_kwargs: (
            events.append("capture:rollback")
            or SimpleNamespace(capture=object())
        ),
    )

    def forward(_model: object, cache: object, *, steering_batch: object) -> torch.Tensor:
        events.append(f"forward:{len(steering_batch)}:{cache.batch_size}")
        return torch.ones((4, 1, 7))

    monkeypatch.setattr(replay, "forward_logits_batch_with_steering_cached", forward)

    class Builder:
        def validate_backend_row(self, row_id: str, token_ids: object) -> None:
            assert row_id == "root"
            assert tuple(token_ids) == (1, 2583, 25)
            events.append("validate")

        def __call__(self, row_id: str, token_index: int, capture: object) -> object:
            assert (row_id, token_index) == ("root", 0)
            assert capture is not None
            events.append("observe")
            return _observation()

    class Controller:
        def __init__(self, arm: str) -> None:
            self.arm = arm

        @staticmethod
        def initial_state() -> GaugeControllerState:
            return GaugeControllerState()

        def propose(self, _observation: object, _state: object) -> GaugeControlProposal:
            return _proposal(self.arm, active=self.arm != "no_intervention")

    controllers = {
        arm: Controller(arm) for arm in replay.SOURCE_GAUGE_ARM_ORDER
    }
    model = torch.nn.Linear(3, 3, bias=False)
    result = replay.forward_source_gauge_arms_hf(
        model,
        (1, 2583, 25),
        row_id="root",
        observation_builder=Builder(),
        controllers=controllers,
    )
    assert result.logits.shape == (4, 7)
    assert events == [
        "validate",
        "prefill:1",
        "capture:rollback",
        "observe",
        "repeat:4",
        "forward:4:4",
    ]


def test_forward_rejects_non_status_prefix_before_model_work() -> None:
    model = torch.nn.Linear(3, 3, bias=False)
    with pytest.raises(replay.RelationalGaugeReplayError, match="Status anchor"):
        replay.forward_source_gauge_arms_hf(
            model,
            (1, 2, 25),
            row_id="root",
            observation_builder=object(),
            controllers={arm: object() for arm in replay.SOURCE_GAUGE_ARM_ORDER},
        )
