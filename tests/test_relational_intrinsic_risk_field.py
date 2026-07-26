from __future__ import annotations

import numpy as np
import pytest

from geoprobe.control.relational_intrinsic_risk_field import (
    GaugeRiskObservation,
    PressureMatchedFieldConfig,
    PressureMatchedRiskField,
    RelationalIntrinsicRiskFieldError,
)
from geoprobe.geometry.relational_gauge_atlas import GaugeChart, gauge_transform_chart


def _chart() -> GaugeChart:
    coordinates = np.asarray(
        [
            [-2.0, -0.2],
            [-1.0, 0.2],
            [1.0, -0.2],
            [2.0, 0.2],
            [0.0, 1.0],
        ]
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
        support_radius=float(np.max(distances)),
    )


def _rows() -> tuple[GaugeRiskObservation, ...]:
    result = []
    outcomes = {
        "a": ("HONEST", "HONEST", "SKIP"),
        "b": ("HONEST", "HONEST", "DECEPTIVE"),
        "c": ("DECEPTIVE", "DECEPTIVE", "HONEST"),
        "d": ("DECEPTIVE", "DECEPTIVE", "NO_ACTION"),
        "e": ("HONEST", "DECEPTIVE", "SKIP"),
    }
    index = 0
    for node, values in outcomes.items():
        for outcome in values:
            result.append(
                GaugeRiskObservation(
                    observation_id=f"o{index}",
                    node_id=node,
                    family_fold=f"f{index % 3}",
                    nuisance_key=("program:A", "dose:3"),
                    outcome_class=outcome,
                )
            )
            index += 1
    return tuple(result)


def _field() -> PressureMatchedRiskField:
    return PressureMatchedRiskField(
        _rows(),
        config=PressureMatchedFieldConfig(
            minimum_support_nodes=4,
            minimum_effective_observations=2.0,
            ridge=1e-4,
        ),
    )


def test_pressure_matched_field_retains_multinomial_outcomes_and_gradient() -> None:
    chart = _chart()
    field = _field()
    evaluation = field.evaluate(
        chart,
        np.zeros(2),
        nuisance_key=("program:A", "dose:3"),
    )
    assert evaluation.defined
    assert evaluation.outcome_probabilities.sum() == pytest.approx(1.0)
    assert evaluation.support_observation_count == 15
    assert evaluation.absolute_gradient[0] > 0.0
    assert evaluation.gradient("pressure_residual_deception").shape == (2,)


def test_field_never_falls_back_across_pressure_cells() -> None:
    evaluation = _field().evaluate(
        _chart(),
        np.zeros(2),
        nuisance_key=("program:B", "dose:3"),
    )
    assert not evaluation.defined
    assert evaluation.reason == "unseen_nuisance_cell"


def test_field_gradient_is_gauge_covariant() -> None:
    chart = _chart()
    theta = 0.41
    q = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    baseline = _field().evaluate(
        chart, np.asarray([0.1, -0.1]), nuisance_key=("program:A", "dose:3")
    )
    transformed = _field().evaluate(
        gauge_transform_chart(chart, q),
        q @ np.asarray([0.1, -0.1]),
        nuisance_key=("program:A", "dose:3"),
    )
    assert transformed.absolute_deception_log_odds == pytest.approx(
        baseline.absolute_deception_log_odds, abs=1e-9
    )
    np.testing.assert_allclose(
        transformed.absolute_gradient, q @ baseline.absolute_gradient, atol=1e-8
    )
    np.testing.assert_allclose(
        transformed.pressure_residual_gradient,
        q @ baseline.pressure_residual_gradient,
        atol=1e-8,
    )


def test_held_out_fold_is_rejected_from_training() -> None:
    with pytest.raises(RelationalIntrinsicRiskFieldError, match="held-out"):
        PressureMatchedRiskField(_rows(), held_out_family_fold="f0")
