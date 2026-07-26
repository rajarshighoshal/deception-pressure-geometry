from __future__ import annotations

import numpy as np
import pytest

from geoprobe.control.relational_horizontal_lift import (
    GaugeLiftSample,
    HorizontalLift,
    RelationalHorizontalLiftError,
    fit_horizontal_lift_patch,
)


def _samples(
    matrix: np.ndarray,
    *,
    basis: np.ndarray | None = None,
) -> tuple[GaugeLiftSample, ...]:
    tangents = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.5],
            [0.5, -1.0],
            [1.5, -0.5],
        ],
        dtype=np.float64,
    )
    result = []
    for index, tangent in enumerate(tangents):
        physical = np.einsum("lhr,r->lh", matrix, tangent)
        observed = tangent if basis is None else basis @ tangent
        result.append(
            GaugeLiftSample(
                sample_id=f"s{index}",
                chart_id="c",
                family_fold=f"f{index % 3}",
                tangent=observed,
                fiber_delta=physical,
            )
        )
    return tuple(result)


def test_exact_multilayer_lift_and_roundtrip() -> None:
    matrix = np.asarray(
        [
            [[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]],
            [[0.5, 1.0], [1.0, 0.5], [-1.0, 2.0]],
        ],
        dtype=np.float64,
    )
    patch = fit_horizontal_lift_patch(_samples(matrix), chart_id="c", ridge=1e-10)
    tangent = np.asarray([0.25, -0.4])
    expected = np.einsum("lhr,r->lh", matrix, tangent)
    np.testing.assert_allclose(
        patch.lift(tangent, require_in_support=False), expected, atol=1e-8
    )
    np.testing.assert_allclose(patch.infer_tangent(expected), tangent, atol=1e-8)
    np.testing.assert_allclose(patch.project_horizontal(expected), expected, atol=1e-8)
    assert patch.weighted_relative_fit_error < 1e-8
    assert patch.weighted_relative_roundtrip_error < 1e-8


def test_lift_is_gauge_covariant() -> None:
    matrix = np.asarray(
        [
            [[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]],
            [[0.5, 1.0], [1.0, 0.5], [-1.0, 2.0]],
        ]
    )
    theta = 0.7
    q = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    original = fit_horizontal_lift_patch(_samples(matrix), chart_id="c", ridge=1e-10)
    refit = fit_horizontal_lift_patch(
        _samples(matrix, basis=q), chart_id="c", ridge=1e-10
    )
    transformed = original.transformed_gauge(q)
    tangent = np.asarray([0.3, -0.2])
    tangent_prime = q @ tangent
    np.testing.assert_allclose(
        transformed.lift(tangent_prime, require_in_support=False),
        original.lift(tangent, require_in_support=False),
        atol=1e-8,
    )
    np.testing.assert_allclose(refit.matrix, transformed.matrix, atol=1e-8)
    np.testing.assert_allclose(
        refit.tangent_metric, transformed.tangent_metric, atol=1e-8
    )


def test_trust_region_refuses_large_tangent() -> None:
    matrix = np.ones((2, 3, 2), dtype=np.float64)
    patch = fit_horizontal_lift_patch(_samples(matrix), chart_id="c", ridge=1e-6)
    assert patch.in_support(np.zeros(2))
    with pytest.raises(RelationalHorizontalLiftError, match="trust region"):
        patch.lift(np.asarray([100.0, 100.0]))


def test_rank_deficient_fit_fails_closed() -> None:
    matrix = np.ones((2, 3, 2), dtype=np.float64)
    rows = tuple(
        GaugeLiftSample(
            sample_id=f"s{i}",
            chart_id="c",
            family_fold="f",
            tangent=np.asarray([float(i + 1), 0.0]),
            fiber_delta=np.einsum("lhr,r->lh", matrix, [float(i + 1), 0.0]),
        )
        for i in range(4)
    )
    with pytest.raises(RelationalHorizontalLiftError, match="do not span"):
        fit_horizontal_lift_patch(rows, chart_id="c")


def test_horizontal_lift_bank_is_shape_checked() -> None:
    matrix = np.asarray(
        [
            [[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]],
            [[0.5, 1.0], [1.0, 0.5], [-1.0, 2.0]],
        ]
    )
    patch = fit_horizontal_lift_patch(_samples(matrix), chart_id="c", ridge=1e-10)
    bank = HorizontalLift({"c": patch})
    np.testing.assert_allclose(
        bank.lift("c", np.asarray([0.1, 0.2]), require_in_support=False),
        patch.lift(np.asarray([0.1, 0.2]), require_in_support=False),
    )
    with pytest.raises(RelationalHorizontalLiftError, match="no horizontal lift"):
        bank.patch("missing")
