from __future__ import annotations

from experiments.powered150_action_response_adapter import adapt_row, validate_adapted


def test_adapter_maps_abstain_to_baseline_and_margin_fields():
    row = {
        "conversation_id": "c1",
        "family": "fam",
        "scenario_id": "s",
        "method": "abstain",
        "status_class": "false_FAIL",
        "target_status": None,
        "route_action": "steer_to_PASS",
        "base_margin": -2.0,
        "final_margin": -2.0,
        "correct_after": False,
        "fixes_error": False,
        "harms_honest": False,
        "reward": 0.0,
    }
    out = adapt_row(row)
    assert out["method"] == "baseline"
    assert out["original_method"] == "abstain"
    assert out["decision_margin"] == -2.0
    assert out["decision_forced_status"] == "FAIL"
    assert out["strict_reward"] == 0.0
    assert out["parse_success"] is True


def test_adapter_validation_requires_baseline_per_conversation():
    rows = [
        adapt_row(
            {
                "conversation_id": "c1",
                "family": "fam",
                "method": "abstain",
                "status_class": "false_PASS",
                "target_status": None,
                "route_action": "steer_to_FAIL",
                "final_margin": 2.0,
                "base_margin": 2.0,
                "correct_after": False,
                "fixes_error": False,
                "harms_honest": False,
                "reward": 0.0,
            }
        ),
        adapt_row(
            {
                "conversation_id": "c1",
                "family": "fam",
                "method": "bidir_linear",
                "status_class": "false_PASS",
                "target_status": "FAIL",
                "route_action": "steer_to_FAIL",
                "layer": 8,
                "alpha": 24.0,
                "final_margin": -1.0,
                "base_margin": 2.0,
                "correct_after": True,
                "fixes_error": True,
                "harms_honest": False,
                "reward": 1.0,
            }
        ),
    ]
    summary = validate_adapted(rows)
    assert summary["n_conversations"] == 1
    assert summary["method_counts"]["baseline"] == 1
