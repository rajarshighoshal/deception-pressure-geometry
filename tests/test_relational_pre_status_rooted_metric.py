from __future__ import annotations

import numpy as np
import pytest

from geoprobe.geometry.relational_pre_status_rooted_metric import (
    RelationalPreStatusRootedMetricError,
    RootedStarDistance,
    RootedStarMetricInput,
    RootedStarMetricScaler,
    rooted_star_descriptor,
    rooted_star_distance,
    rooted_star_energy_quotient,
)


def _star(*, token_count: int = 5, heads: int = 3, offset: float = 0.0) -> RootedStarMetricInput:
    position = np.linspace(0.0, 1.0, token_count)
    annotations = np.column_stack(
        [position, np.arange(token_count) % 2, np.arange(token_count) % 3]
    )
    residual = np.stack(
        [np.abs(position - 1.0) * (layer + 1.0) + offset for layer in range(2)]
    )
    raw_attention = np.stack(
        [
            np.stack(
                [np.arange(1, token_count + 1, dtype=float) + head + layer + offset for head in range(heads)]
            )
            for layer in range(2)
        ]
    )
    attention = raw_attention / raw_attention.sum(axis=2, keepdims=True)
    return RootedStarMetricInput(residual, attention, annotations)


def test_identical_star_has_zero_distance() -> None:
    star = _star()
    distance = rooted_star_distance(star, star)
    assert distance.residual == pytest.approx(0.0, abs=1e-12)
    assert distance.attention_head_set == pytest.approx(0.0, abs=1e-12)


def test_attention_distance_is_invariant_to_independent_head_permutation() -> None:
    left = _star()
    right = RootedStarMetricInput(
        left.residual_root_distances,
        left.incoming_attention[:, [2, 0, 1], :],
        left.typed_annotations,
    )
    distance = rooted_star_distance(left, right)
    assert distance.residual == pytest.approx(0.0, abs=1e-12)
    assert distance.attention_head_set == pytest.approx(0.0, abs=1e-12)


def test_metric_supports_unequal_token_and_head_counts() -> None:
    distance = rooted_star_distance(_star(token_count=4, heads=2), _star(token_count=6, heads=4))
    assert distance.residual >= 0.0
    assert distance.attention_head_set >= 0.0


def test_scaler_fuses_two_training_scaled_views() -> None:
    one = rooted_star_distance(_star(), _star(offset=0.2))
    two = rooted_star_distance(_star(), _star(offset=0.4))
    scaler = RootedStarMetricScaler.fit((one, two))
    assert scaler.transform(one) > 0.0
    assert scaler.transform(two) > scaler.transform(one)


def test_descriptor_is_fixed_width_for_token_count_changes() -> None:
    assert rooted_star_descriptor(_star(token_count=4)).shape == rooted_star_descriptor(
        _star(token_count=7)
    ).shape


def test_input_rejects_unnormalized_attention() -> None:
    star = _star()
    with pytest.raises(RelationalPreStatusRootedMetricError, match="normalized"):
        RootedStarMetricInput(
            star.residual_root_distances,
            star.incoming_attention * 0.5,
            star.typed_annotations,
        )


def test_energy_quotient_reduces_to_singleton_and_subtracts_replay_noise() -> None:
    singleton = RootedStarDistance(residual=3.0, attention_head_set=2.0)
    assert rooted_star_energy_quotient(
        (singleton,), (), (), left_size=1, right_size=1
    ) == singleton

    quotient = rooted_star_energy_quotient(
        (
            RootedStarDistance(3.0, 3.0),
            RootedStarDistance(5.0, 5.0),
            RootedStarDistance(5.0, 5.0),
            RootedStarDistance(3.0, 3.0),
        ),
        (RootedStarDistance(4.0, 4.0),),
        (RootedStarDistance(4.0, 4.0),),
        left_size=2,
        right_size=2,
    )
    assert quotient.residual == pytest.approx(3.0)
    assert quotient.attention_head_set == pytest.approx(3.0)
