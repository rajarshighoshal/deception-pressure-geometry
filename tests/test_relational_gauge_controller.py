from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from geoprobe.control.relational_gauge_controller import (
    GaugeControllerConfig,
    GaugeControllerObservation,
    RelationalGaugeController,
)
from geoprobe.control.relational_horizontal_lift import (
    GaugeLiftSample,
    HorizontalLift,
    fit_horizontal_lift_patch,
)
from geoprobe.control.relational_intrinsic_risk_field import (
    GaugeRiskObservation,
    PressureMatchedFieldConfig,
    PressureMatchedRiskField,
)
from geoprobe.geometry.relational_gauge_atlas import (
    GaugeChart,
    GaugeQueryState,
    gauge_transform_chart,
)


def _chart() -> GaugeChart:
    coordinates = np.asarray(
        [[-2.0, -0.2], [-1.0, 0.2], [1.0, -0.2], [2.0, 0.2], [0.0, 1.0]]
    )
    distances = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=2
    )
    return GaugeChart(
        chart_id="c",
        center_node_id="e",
        support_ids=("a", "b", "c", "d", "e"),
        support_distances=tuple(tuple(row) for row in distances),
        coordinates=coordinates,
        eigenvalues=np.asarray([2.0, 1.0]),
        stress=0.0,
        support_radius=5.0,
    )


def _risk_field() -> PressureMatchedRiskField:
    outcomes = {
        "a": ("HONEST", "HONEST", "SKIP"),
        "b": ("HONEST", "HONEST", "DECEPTIVE"),
        "c": ("DECEPTIVE", "DECEPTIVE", "HONEST"),
        "d": ("DECEPTIVE", "DECEPTIVE", "NO_ACTION"),
        "e": ("HONEST", "DECEPTIVE", "SKIP"),
    }
    rows = []
    index = 0
    for node, values in outcomes.items():
        for outcome in values:
            rows.append(
                GaugeRiskObservation(
                    observation_id=f"r{index}",
                    node_id=node,
                    family_fold=f"f{index % 3}",
                    nuisance_key=("p3", "A"),
                    outcome_class=outcome,
                )
            )
            index += 1
    return PressureMatchedRiskField(
        rows,
        config=PressureMatchedFieldConfig(
            minimum_support_nodes=4,
            minimum_effective_observations=2.0,
            ridge=1e-4,
        ),
    )


def _lift(*, basis: np.ndarray | None = None) -> HorizontalLift:
    matrix = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]],
            [[0.5, 0.5], [1.0, -0.5], [-0.25, 1.0]],
        ]
    )
    tangents = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5], [0.5, -1.0], [1.5, -0.5]]
    )
    rows = []
    for index, tangent in enumerate(tangents):
        observed = tangent if basis is None else basis @ tangent
        rows.append(
            GaugeLiftSample(
                sample_id=f"l{index}",
                chart_id="c",
                family_fold=f"f{index % 3}",
                tangent=observed,
                fiber_delta=np.einsum("lhr,r->lh", matrix, tangent),
            )
        )
    patch = fit_horizontal_lift_patch(rows, chart_id="c", ridge=1e-10)
    return HorizontalLift({"c": patch})


def _observation(
    coordinate: np.ndarray,
    *,
    chart: GaugeChart | None = None,
    in_support: bool = True,
    observation_id: str = "o",
    randomization_key: str | None = None,
) -> GaugeControllerObservation:
    active_chart = chart or _chart()
    return GaugeControllerObservation(
        observation_id=observation_id,
        chart=active_chart,
        query=GaugeQueryState(
            chart_id="c",
            query_coordinates=coordinate,
            nearest_node_id="e",
            nearest_node_distance=0.1,
            stress=0.01,
            support_status=in_support,
            support_reason="test",
        ),
        nuisance_key=("p3", "A"),
        randomization_key=randomization_key,
    )


def _controller(arm: str = "gauge_geodesic") -> RelationalGaugeController:
    return RelationalGaugeController(
        risk_field=_risk_field(),
        horizontal_lift=_lift(),
        config=GaugeControllerConfig(
            field_kind="absolute_deception",
            control_arm=arm,
            metric_regularization=0.1,
            trust_fraction=0.5,
            maximum_query_stress=0.2,
        ),
    )


def test_geodesic_proposal_descends_and_is_multilayer_capped() -> None:
    proposal = _controller().propose(
        _observation(np.asarray([2.0, 0.0])),
        RelationalGaugeController.initial_state(),
    )
    assert proposal.status == "active"
    assert proposal.intrinsic_step[0] < 0.0
    assert proposal.fiber_step.shape == (2, 3)
    assert np.all(proposal.layer_fiber_norms <= proposal.layer_fiber_norm_caps + 1e-9)
    assert proposal.next_state.step_count == 1


def test_off_support_and_boundary_exit_fail_closed() -> None:
    controller = _controller()
    off = controller.propose(
        _observation(np.asarray([1.0, 0.0]), in_support=False),
        RelationalGaugeController.initial_state(),
    )
    assert off.stop and off.status == "off_support"
    boundary = controller.propose(
        _observation(np.asarray([-2.0, 0.0])),
        RelationalGaugeController.initial_state(),
    )
    assert boundary.stop and boundary.status == "boundary_exit"


def test_sign_and_random_controls_match_intrinsic_step_norm() -> None:
    state = RelationalGaugeController.initial_state()
    observation = _observation(np.asarray([2.0, 0.0]))
    gauge = _controller("gauge_geodesic").propose(observation, state)
    sign = _controller("sign_flipped").propose(observation, state)
    random = _controller("random_tangent").propose(observation, state)
    assert sign.intrinsic_step_norm == pytest.approx(gauge.intrinsic_step_norm)
    assert random.intrinsic_step_norm == pytest.approx(gauge.intrinsic_step_norm)
    np.testing.assert_allclose(sign.intrinsic_step, -gauge.intrinsic_step)


def test_random_control_is_stable_across_capture_evidence_ids() -> None:
    controller = _controller("random_tangent")
    first = controller.propose(
        _observation(
            np.asarray([2.0, 0.0]),
            observation_id="live-evidence-a",
            randomization_key="sealed-root",
        ),
        RelationalGaugeController.initial_state(),
    )
    second = controller.propose(
        _observation(
            np.asarray([2.0, 0.0]),
            observation_id="live-evidence-b",
            randomization_key="sealed-root",
        ),
        RelationalGaugeController.initial_state(),
    )
    np.testing.assert_array_equal(first.intrinsic_step, second.intrinsic_step)


def test_controller_is_gauge_covariant() -> None:
    theta = 0.37
    q = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    chart = _chart()
    coordinate = np.asarray([2.0, 0.0])
    baseline = _controller().propose(
        _observation(coordinate, chart=chart),
        RelationalGaugeController.initial_state(),
    )
    transformed_lift = _lift()
    transformed_lift = HorizontalLift(
        {"c": transformed_lift.patch("c").transformed_gauge(q)}
    )
    transformed_controller = RelationalGaugeController(
        risk_field=_risk_field(),
        horizontal_lift=transformed_lift,
        config=_controller().config,
    )
    changed = transformed_controller.propose(
        _observation(q @ coordinate, chart=gauge_transform_chart(chart, q)),
        RelationalGaugeController.initial_state(),
    )
    np.testing.assert_allclose(changed.intrinsic_step, q @ baseline.intrinsic_step, atol=1e-8)
    np.testing.assert_allclose(changed.fiber_step, baseline.fiber_step, atol=1e-7)


def test_max_steps_stops_closed_loop_state() -> None:
    controller = _controller()
    state = replace(
        RelationalGaugeController.initial_state(),
        step_count=controller.config.maximum_steps,
    )
    proposal = controller.propose(_observation(np.asarray([2.0, 0.0])), state)
    assert proposal.stop and proposal.status == "max_steps"
