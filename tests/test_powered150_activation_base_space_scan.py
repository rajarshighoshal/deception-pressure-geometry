from __future__ import annotations

from experiments.powered150_activation_base_space_scan import render_markdown, state_records_from_action_response


def _row(cid: str, *, method: str, target: str | None, reward: float, fixes: bool) -> dict:
    return {
        "conversation_id": cid,
        "scenario_id": "s1",
        "family": "fam",
        "status_class": "false_FAIL",
        "route_action": "steer_to_PASS",
        "method": method,
        "target_status": target,
        "layer": None if method == "abstain" else 16,
        "alpha": 0.0 if method == "abstain" else 24.0,
        "base_margin": -1.0,
        "final_margin": 1.0 if fixes else -1.0,
        "desired_margin_sign": 1.0,
        "reward": reward,
        "fixes_error": fixes,
        "harms_honest": False,
    }


def test_state_records_extract_best_action_and_response_density() -> None:
    rows = [
        _row("c1", method="abstain", target=None, reward=0.0, fixes=False),
        _row("c1", method="global_mean", target="PASS", reward=1.0, fixes=True),
        _row("c1", method="bidir_linear", target="PASS", reward=0.0, fixes=False),
    ]
    rec = state_records_from_action_response(rows)["c1"]
    assert rec["best_method"] == "global_mean"
    assert rec["best_target_status"] == "PASS"
    assert rec["response_density"] == 0.5


def test_render_separates_controller_available_pre_response() -> None:
    def item(phase: str, score: float) -> dict:
        return {
            "layer": 16,
            "turn": 3,
            "phase": phase,
            "pca_dim": 16,
            "metric": "euclidean",
            "metrics": {
                "base_space_smoothness_score": score,
                "status_acc": score,
                "route_acc": score,
                "best_method_acc": score,
                "best_target_acc": score,
                "response_density_mae": 1.0 - score,
                "local_dim_participation_ratio": {"median_mean": 4.0},
                "pca_explained_variance_mean": 0.5,
            },
        }

    payload = {
        "grid": {
            "layers": [16],
            "turns": [3],
            "phases": ["pre_response", "post_response"],
            "pca_dims": [16],
            "metrics": ["euclidean"],
            "k": 15,
            "max_cids": 0,
        },
        "results": [item("post_response", 0.9), item("pre_response", 0.8)],
    }
    text = render_markdown(payload, top_n=5)
    assert "Top Controller-Available Base Spaces" in text
    assert "`post_response` states are diagnostic only" in text
