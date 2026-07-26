from __future__ import annotations

from experiments.report_powered150_selector_eval import render_markdown


def test_report_marks_next_candidates_as_not_run():
    payload = {
        "action_response": "ar.json",
        "action_response_sha256": "abc",
        "activations": "turns.pt",
        "typed_graph": None,
        "n_rows": 2,
        "n_conversations": 1,
        "adapted_summary": {"n_rows": 2},
        "structural_families": ["chart"],
        "status_class_balance": {"false_FAIL": 1},
        "comparison_references": ["always_abstain"],
        "policy_types": {"always_abstain": "baseline"},
        "next_structural_candidates_not_run": [
            {"name": "candidate_x", "status": "not_run", "reason": "future diagnostic"}
        ],
        "policies": {
            "always_abstain": {
                "summary": {
                    "fixes_error": 0,
                    "deceptive_n": 1,
                    "honest_harms": 0,
                    "honest_n": 1,
                    "mean_reward": 0.0,
                    "mean_aligned_margin": 0.0,
                    "chosen_methods": {"abstain": 2},
                }
            }
        },
    }
    text = render_markdown(payload, top_n=10)
    assert "candidate_x" in text
    assert "not_run" in text
    assert "not audited generation strictness" in text
