from __future__ import annotations

import numpy as np

from experiments.powered150_guarded_neural_residual import (
    FeatureOverrideGate,
    feature_gate_choice,
    gate_features,
    override_label,
)


def row(
    cid: str,
    *,
    method: str,
    target: str | None,
    layer: int | None,
    alpha: float | None,
    reward: float,
    fixes: bool,
) -> dict:
    return {
        "conversation_id": cid,
        "scenario_id": f"scenario_{cid}",
        "family": "fam",
        "status_class": "false_PASS",
        "route_action": "steer_to_FAIL",
        "reported_status_before": "PASS",
        "method": method,
        "target_status": target,
        "layer": layer,
        "alpha": alpha,
        "reward": reward,
        "strict_reward": reward,
        "fixes_error": fixes,
        "harms_honest": False,
        "base_margin": 2.0,
        "final_margin": -2.0 if fixes else 2.0,
        "delta_margin": -4.0 if fixes else 0.0,
        "correct_after": fixes,
        "gate_score_PASS_minus_FAIL": 2.0,
        "gate_proba_PASS": 0.9,
    }


def scored_group(cid: str, *, neural_wins: bool, neural_score: float, chart_support: float) -> dict:
    chart_reward = 0.0 if neural_wins else 1.0
    neural_reward = 1.0 if neural_wins else 0.0
    candidates = [
        row(cid, method="baseline", target=None, layer=None, alpha=None, reward=0.0, fixes=False),
        row(cid, method="global_mean_gated", target="FAIL", layer=8, alpha=48.0, reward=chart_reward, fixes=bool(chart_reward)),
        row(cid, method="bidir_linear", target="FAIL", layer=16, alpha=96.0, reward=neural_reward, fixes=bool(neural_reward)),
    ]
    return {
        "candidates": candidates,
        "chart_scores": np.asarray([-1e9, 0.8, 0.2], dtype=np.float64),
        "neural_scores": np.asarray([-1e9, 0.1, neural_score], dtype=np.float64),
        "chart_metas": [
            {},
            {
                "chart_prediction_std": 0.02,
                "chart_top": [{"weight": 1.0, "support": chart_support, "action_support": chart_support, "action_count": 8, "prediction": 0.8}],
            },
            {
                "chart_prediction_std": 0.2,
                "chart_top": [{"weight": 1.0, "support": 4.0, "action_support": 2.0, "action_count": 2, "prediction": 0.2}],
            },
        ],
        "baseline_idx": 0,
        "chart_idx": 1,
        "neural_idx": 2,
        "neural_margin_over_chart": neural_score - 0.1,
    }


def test_gate_features_exclude_measured_candidate_response_fields():
    scored = scored_group("cid", neural_wins=True, neural_score=2.0, chart_support=4.0)
    features = gate_features(scored)
    forbidden = {
        "reward",
        "strict_reward",
        "fixes_error",
        "harms_honest",
        "final_margin",
        "delta_margin",
        "correct_after",
        "decision_margin",
    }
    assert not (forbidden & set(features))
    assert features["target_agree"] == 1.0
    assert features["method_agree"] == 0.0
    assert features["layer_abs_diff"] == 8.0


def test_feature_gate_learns_conservative_override_boundary():
    train = [
        scored_group("pos1", neural_wins=True, neural_score=2.2, chart_support=3.0),
        scored_group("pos2", neural_wins=True, neural_score=2.0, chart_support=4.0),
        scored_group("neg1", neural_wins=False, neural_score=-0.2, chart_support=20.0),
        scored_group("neg2", neural_wins=False, neural_score=0.0, chart_support=18.0),
    ]
    gate = FeatureOverrideGate(c=0.5, max_iter=200).fit(
        [gate_features(group) for group in train],
        [override_label(group) for group in train],
    )
    scores = gate.predict_proba([gate_features(group) for group in train])
    assert float(scores[:2].mean()) > float(scores[2:].mean())
    tau = float(np.median(scores))
    choices = [
        feature_gate_choice(group, gate_score=float(score), tau=tau)
        for group, score in zip(train, scores)
    ]
    assert any(choice["policy_source"] == "neural_feature_gate" for choice in choices[:2])
    assert all(choice["policy_source"] == "chart_feature_gate" for choice in choices[2:])
