from __future__ import annotations

import numpy as np

from geoprobe.control import geometry_map as gm
from geoprobe.control.local_control_flow import (
    BoundaryDoseConfig,
    CausalDoseConfig,
    CausalGainDoseEstimator,
    LinearBoundaryDoseEstimator,
    LocalControlFlowConfig,
    LocalControlFlowEstimator,
)
from geoprobe.control.local_deformation_field import SteeringSpecLookup
from tests.test_action_fiber_models import make_row
from tests.test_local_deformation_field import toy_rows_states_vectors, write_spec_bank


def cfg(**overrides: object) -> LocalControlFlowConfig:
    values = {
        "top_k": 2,
        "neighbor_pool_multiplier": 2,
        "geometry_dim": 2,
        "objective": "reward",
        "target_route_only": True,
        "use_local_neighbors": True,
        "seed": 7,
    }
    values.update(overrides)
    return LocalControlFlowConfig(**values)


def test_local_control_flow_scores_local_good_direction(tmp_path) -> None:
    rows, states, vectors = toy_rows_states_vectors()
    lookup = SteeringSpecLookup(write_spec_bank(tmp_path, rows, vectors))
    train = [row for row in rows if not gm.state_id(row).startswith("held")]
    test = [row for row in rows if gm.state_id(row) == "held_left"]
    estimator = LocalControlFlowEstimator(lookup, cfg()).fit(train, states)
    scores = estimator.score(test, states, exclude_same_state=False)
    choice = gm.choose_by_scores(test, scores, threshold=-1e9)
    assert choice["method"] == "global_mean_gated"
    assert choice["fixes_error"] is True
    pred = estimator.predict(choice, states, exclude_same_state=False)
    assert pred.confidence > 0.1
    np.testing.assert_allclose(pred.vector / np.linalg.norm(pred.vector), np.asarray([1.0, 0.0, 0.0, 0.0]), atol=1e-5)


def test_local_control_flow_context_scores_ignore_mutated_query_response_fields(tmp_path) -> None:
    rows, states, vectors = toy_rows_states_vectors()
    lookup = SteeringSpecLookup(write_spec_bank(tmp_path, rows, vectors))
    train = [row for row in rows if not gm.state_id(row).startswith("held")]
    group = [row for row in rows if gm.state_id(row) == "held_right"]
    estimator = LocalControlFlowEstimator(lookup, cfg()).fit(train, states)
    mutated = []
    for row in group:
        copy = dict(row)
        copy.update({
            "final_margin": 999.0,
            "decision_margin": 999.0,
            "delta_margin": 999.0,
            "reward": -123.0,
            "strict_reward": -123.0,
            "correct_after": not bool(row.get("correct_after")),
            "fixes_error": not bool(row.get("fixes_error")),
            "fixes_strict": not bool(row.get("fixes_strict")),
            "harms_honest": not bool(row.get("harms_honest")),
            "harms_honest_strict": not bool(row.get("harms_honest_strict")),
        })
        mutated.append(copy)
    before = estimator.score(group, states, exclude_same_state=False)
    after = estimator.score(mutated, states, exclude_same_state=False)
    np.testing.assert_allclose(before, after, rtol=0.0, atol=1e-7)


def test_global_flow_removes_local_state_dependence(tmp_path) -> None:
    rows, states, vectors = toy_rows_states_vectors()
    lookup = SteeringSpecLookup(write_spec_bank(tmp_path, rows, vectors))
    train = [row for row in rows if not gm.state_id(row).startswith("held")]
    left = [row for row in rows if gm.state_id(row) == "held_left"]
    right = [row for row in rows if gm.state_id(row) == "held_right"]
    estimator = LocalControlFlowEstimator(lookup, cfg(use_local_neighbors=False)).fit(train, states)
    left_scores = estimator.score(left, states, exclude_same_state=False)
    right_scores = estimator.score(right, states, exclude_same_state=False)
    np.testing.assert_allclose(left_scores, right_scores, rtol=0.0, atol=1e-7)


def test_linear_boundary_dose_estimates_distance_to_detector_boundary() -> None:
    train_left = make_row("train_left", "toy", "left", method="global_mean_gated", target_status="PASS", layer=16)
    train_right = make_row("train_right", "toy", "right", method="global_mean_gated", target_status="PASS", layer=16)
    held = make_row("held", "toy", "held", method="global_mean_gated", target_status="PASS", layer=16)
    train_left["base_margin"] = -2.0
    train_right["base_margin"] = 2.0
    held["base_margin"] = -1.0
    states = {
        ("train_left", 16): np.asarray([-2.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ("train_right", 16): np.asarray([2.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ("held", 16): np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }
    estimator = LinearBoundaryDoseEstimator(
        BoundaryDoseConfig(target_margin=0.0, max_alpha=10.0, ridge_alpha=1e-9)
    ).fit([train_left, train_right], states)

    pred = estimator.alpha_for(held, states, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert pred.reason == "boundary_crossing"
    assert pred.alpha is not None
    assert abs(pred.alpha - 1.0) < 1e-5
    assert pred.directional_slope > 0.0

    safe = dict(held, base_margin=1.0)
    already = estimator.alpha_for(safe, states, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert already.reason == "already_beyond_boundary"
    assert already.alpha == 0.0

    wrong_way = estimator.alpha_for(held, states, np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert wrong_way.reason == "non_positive_boundary_slope"
    assert wrong_way.alpha is None


def test_linear_boundary_dose_handles_fail_target_sign() -> None:
    train_left = make_row(
        "train_left",
        "toy",
        "left",
        method="global_mean_gated",
        target_status="FAIL",
        layer=16,
        status_class="false_PASS",
    )
    train_right = make_row(
        "train_right",
        "toy",
        "right",
        method="global_mean_gated",
        target_status="FAIL",
        layer=16,
        status_class="false_PASS",
    )
    held = make_row(
        "held",
        "toy",
        "held",
        method="global_mean_gated",
        target_status="FAIL",
        layer=16,
        status_class="false_PASS",
    )
    train_left["base_margin"] = -2.0
    train_right["base_margin"] = 2.0
    held["base_margin"] = 1.0
    states = {
        ("train_left", 16): np.asarray([-2.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ("train_right", 16): np.asarray([2.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ("held", 16): np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }
    estimator = LinearBoundaryDoseEstimator(
        BoundaryDoseConfig(target_margin=0.0, max_alpha=10.0, ridge_alpha=1e-9)
    ).fit([train_left, train_right], states)

    pred = estimator.alpha_for(held, states, np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert pred.reason == "boundary_crossing"
    assert pred.alpha is not None
    assert abs(pred.alpha - 1.0) < 1e-5
    assert pred.directional_slope > 0.0

    safe = dict(held, base_margin=-1.0)
    already = estimator.alpha_for(safe, states, np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert already.reason == "already_beyond_boundary"
    assert already.alpha == 0.0

    wrong_way = estimator.alpha_for(held, states, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert wrong_way.reason == "non_positive_boundary_slope"
    assert wrong_way.alpha is None


def _causal_score_row(cid: str, *, target: str, base: float, alpha: float, final: float) -> dict:
    row = make_row(
        cid,
        "toy",
        cid,
        method="local_control_flow",
        target_status=target,
        layer=16,
        status_class="false_FAIL" if target == "PASS" else "false_PASS",
    )
    row.update({
        "base_margin": base,
        "alpha": alpha,
        "final_margin": final,
        "pc_predicted_flow_norm": 72.0,
        "pc_n_train": 12.0,
        "pc_neighbor_distance_mean": 0.25,
        "pc_neighbor_distance_min": 0.1,
        "pc_neighbor_distance_max": 0.4,
        "pc_boundary_directional_slope": 0.01,
        "pc_boundary_alpha": 8.0,
        "pc_boundary_alpha_raw": 8.0,
    })
    return row


def test_causal_gain_dose_learns_alpha_from_fixed_crossings() -> None:
    rows = [
        _causal_score_row("train_a", target="PASS", base=-2.0, alpha=48.0, final=0.4),
        _causal_score_row("train_a", target="PASS", base=-2.0, alpha=96.0, final=2.0),
        _causal_score_row("train_b", target="PASS", base=-4.0, alpha=48.0, final=-1.0),
        _causal_score_row("train_b", target="PASS", base=-4.0, alpha=96.0, final=0.2),
    ]
    estimator = CausalGainDoseEstimator(
        CausalDoseConfig(target_margin=0.0, max_alpha=128.0, ridge_alpha=1e-6)
    ).fit(rows)
    held = _causal_score_row("held", target="PASS", base=-3.0, alpha=48.0, final=-99.0)
    pred = estimator.alpha_for(held)
    assert pred.reason == "causal_gain_crossing"
    assert pred.alpha is not None
    assert 40.0 < pred.alpha < 128.0
    assert pred.predicted_gain is not None and pred.predicted_gain > 0.0


def test_causal_gain_dose_context_score_ignores_query_response_fields() -> None:
    rows = [
        _causal_score_row("train_a", target="FAIL", base=2.0, alpha=48.0, final=-0.4),
        _causal_score_row("train_a", target="FAIL", base=2.0, alpha=96.0, final=-2.0),
        _causal_score_row("train_b", target="FAIL", base=4.0, alpha=48.0, final=1.0),
        _causal_score_row("train_b", target="FAIL", base=4.0, alpha=96.0, final=-0.2),
    ]
    estimator = CausalGainDoseEstimator(
        CausalDoseConfig(target_margin=0.0, max_alpha=128.0, ridge_alpha=1e-6)
    ).fit(rows)
    held = _causal_score_row("held", target="FAIL", base=3.0, alpha=48.0, final=99.0)
    mutated = dict(
        held,
        final_margin=-999.0,
        decision_margin=-999.0,
        delta_margin=-999.0,
        reward=-999.0,
        fixes_error=True,
        harms_honest=True,
    )
    before = estimator.alpha_for(held)
    after = estimator.alpha_for(mutated)
    assert before.alpha == after.alpha
    assert before.predicted_gain == after.predicted_gain
