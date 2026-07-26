from __future__ import annotations

import numpy as np

from geoprobe.control.dense_dose_response import (
    DenseDoseConfig,
    DenseDoseResponseModel,
    alpha_anchor,
    extract_dose_curves,
    feature_vector,
    first_crossing_alpha,
    stable_crossing_alpha,
    transform_target_alpha,
)


def _row(
    cid: str,
    status: str,
    target: str,
    alpha: float,
    final_margin: float,
    *,
    family: str = "f0",
    boundary_alpha: float = 4.0,
    causal_alpha: float = 32.0,
) -> dict:
    return {
        "conversation_id": cid,
        "family": family,
        "scenario_id": "s0",
        "status_class": status,
        "route_action": "steer_to_PASS" if target == "PASS" else "steer_to_FAIL",
        "reported_status_before": "FAIL" if target == "PASS" else "PASS",
        "target_status": target,
        "method": "local_control_flow",
        "layer": 16,
        "alpha": alpha,
        "base_margin": -2.0 if target == "PASS" else 2.0,
        "final_margin": final_margin,
        "reward": 1.0 if (final_margin >= 0.0 if target == "PASS" else final_margin <= 0.0) else 0.0,
        "fixes_error": status.startswith("false") and (final_margin >= 0.0 if target == "PASS" else final_margin <= 0.0),
        "harms_honest": False,
        "gate_score_PASS_minus_FAIL": -2.0 if target == "PASS" else 2.0,
        "gate_proba_PASS": 0.1 if target == "PASS" else 0.9,
        "pc_predicted_flow_norm": 40.0,
        "pc_n_train": 12,
        "pc_neighbor_distance_mean": 2.0,
        "pc_neighbor_distance_min": 1.0,
        "pc_neighbor_distance_max": 3.0,
        "pc_boundary_directional_slope": 0.5,
        "pc_boundary_alpha": boundary_alpha,
        "pc_boundary_alpha_raw": boundary_alpha,
        "pc_causal_alpha": causal_alpha,
        "pc_causal_alpha_raw": causal_alpha,
        "pc_causal_predicted_gain": 0.1,
        "pc_causal_required_margin_gain": 2.0,
    }


def test_stable_crossing_differs_from_first_crossing() -> None:
    rows = [
        _row("a", "false_FAIL", "PASS", 8.0, -1.0),
        _row("a", "false_FAIL", "PASS", 16.0, 0.2),
        _row("a", "false_FAIL", "PASS", 24.0, -0.1),
        _row("a", "false_FAIL", "PASS", 32.0, 0.3),
        _row("a", "false_FAIL", "PASS", 40.0, 0.4),
    ]
    curve = extract_dose_curves(rows)[0]

    assert first_crossing_alpha(curve) == 16.0
    assert stable_crossing_alpha(curve) == 32.0


def test_dense_dose_features_ignore_query_response_fields() -> None:
    rows = [
        _row("a", "false_FAIL", "PASS", 8.0, -1.0),
        _row("a", "false_FAIL", "PASS", 16.0, 0.2),
    ]
    curve = extract_dose_curves(rows)[0]
    before = feature_vector(curve)
    for row in curve.rows_by_alpha.values():
        row["final_margin"] = 999.0
        row["reward"] = -999.0
        row["fixes_error"] = True
        row["harms_honest"] = True
        row["delta_margin"] = 999.0

    np.testing.assert_allclose(before, feature_vector(curve))


def test_dense_dose_model_fits_and_predicts_grid_alpha() -> None:
    rows = []
    for idx in range(8):
        family = f"f{idx % 2}"
        rows.extend([
            _row(f"a{idx}", "false_FAIL", "PASS", 8.0, -1.0, family=family),
            _row(f"a{idx}", "false_FAIL", "PASS", 16.0, -0.2, family=family),
            _row(f"a{idx}", "false_FAIL", "PASS", 32.0, 0.2, family=family),
        ])
    curves = extract_dose_curves(rows)
    model = DenseDoseResponseModel(DenseDoseConfig(safety_quantile=0.5)).fit(curves[:6])
    pred = model.predict(curves[6])

    assert pred.alpha in {8.0, 16.0, 32.0}
    assert pred.alpha >= 16.0


def test_dense_dose_model_can_calibrate_on_train_family_blocks() -> None:
    rows = []
    for idx in range(12):
        family = f"f{idx % 3}"
        base = -2.0 - idx * 0.1
        rows.extend([
            {**_row(f"a{idx}", "false_FAIL", "PASS", 8.0, base + 0.5, family=family), "base_margin": base},
            {**_row(f"a{idx}", "false_FAIL", "PASS", 16.0, base + 1.0, family=family), "base_margin": base},
            {**_row(f"a{idx}", "false_FAIL", "PASS", 32.0, base + 3.0, family=family), "base_margin": base},
        ])
    curves = extract_dose_curves(rows)
    model = DenseDoseResponseModel(DenseDoseConfig(safety_quantile=1.0, calibration_folds=3)).fit(curves)

    assert model.calibration_mode_ == "family_block_oof"
    assert model.n_calibration_ > 0
    assert model.calibration_residual_summary_["count"] == model.n_calibration_


def test_dense_dose_boundary_ratio_learns_anchor_multiplier() -> None:
    rows = []
    for idx in range(8):
        family = f"f{idx % 2}"
        rows.extend([
            _row(f"a{idx}", "false_FAIL", "PASS", 8.0, -1.0, family=family, boundary_alpha=16.0),
            _row(f"a{idx}", "false_FAIL", "PASS", 16.0, -0.2, family=family, boundary_alpha=16.0),
            _row(f"a{idx}", "false_FAIL", "PASS", 32.0, 0.2, family=family, boundary_alpha=16.0),
        ])
    curves = extract_dose_curves(rows)
    target = stable_crossing_alpha(curves[0])

    assert alpha_anchor(curves[0], "boundary_ratio") == 16.0
    assert target == 32.0
    assert np.isclose(transform_target_alpha(curves[0], target, "boundary_ratio"), np.log(2.0))

    model = DenseDoseResponseModel(DenseDoseConfig(prediction_mode="boundary_ratio", safety_quantile=0.5)).fit(curves[:6])
    pred = model.predict(curves[6])

    assert pred.alpha == 32.0
    assert np.isclose(pred.anchor_alpha, 16.0)
    assert pred.multiplier is not None
    assert 1.5 < pred.multiplier < 2.5
