from __future__ import annotations

import json
from pathlib import Path

from experiments.report_powered150_matched_control_audit import (
    build_parser,
    build_payload,
    build_public_receipt,
    render_markdown,
)


def _selector_row(
    cid: str,
    family: str,
    *,
    method: str,
    target_status: str,
    route_action: str,
    status_class: str,
    reward: float,
) -> dict:
    return {
        "conversation_id": cid,
        "scenario_id": f"scenario_{cid}",
        "family": family,
        "method": method,
        "target_status": target_status,
        "status_class": status_class,
        "route_action": route_action,
        "layer": 16,
        "alpha": 96.0,
        "base_margin": 1.0,
        "final_margin": 2.0,
        "delta_margin": 1.0,
        "reward": reward,
        "fixes_error": reward > 0,
        "harms_honest": False,
        "correct_after": reward > 0,
    }


def _compact_selector_row(
    cid: str,
    family: str,
    *,
    method: str,
    target_status: str | None,
    route_action: str,
    status_class: str,
    reward: float,
) -> dict:
    return {
        "conversation_id": cid,
        "scenario_id": f"scenario_{cid}",
        "family": family,
        "method": method,
        "target_status": target_status,
        "status_class": status_class,
        "route_action": route_action,
        "layer": 16,
        "alpha": 96.0,
        "reward": reward,
        "fixes_error": reward > 0,
        "harms_honest": False,
        "correct_after": reward > 0,
    }


def _action_row(
    cid: str,
    family: str,
    *,
    method: str,
    route_action: str,
    target_status: str | None,
    layer: int | None,
    alpha: float,
    status_class: str,
    reward: float,
    base_margin: float,
) -> dict:
    true_status = "FAIL" if status_class in {"false_PASS", "honest_FAIL"} else "PASS"
    return {
        "conversation_id": cid,
        "scenario_id": f"scenario_{cid}",
        "family": family,
        "arm": "p5",
        "method": method,
        "route_action": route_action,
        "target_status": target_status,
        "status_class": status_class,
        "layer": layer,
        "alpha": alpha,
        "base_margin": base_margin,
        "final_margin": (base_margin + 4.0) if method == "bidir_linear" else base_margin,
        "delta_margin": (4.0 if method == "bidir_linear" else 0.0),
        "reward": reward,
        "fixes_error": reward > 0 and status_class.startswith("false_"),
        "harms_honest": False,
        "correct_after": reward > 0,
        "reported_status_before": "PASS" if base_margin > 0 else "FAIL",
        "true_status": true_status,
        "desired_status": true_status,
        "desired_margin_sign": 1 if true_status == "PASS" else -1,
        "target_margin_sign": (1 if target_status == "PASS" else -1) if target_status else 0,
        "projection_fraction": 0.5,
        "cos_to_raw": 0.1,
        "neighbor_distance_mean": 0.2,
        "neighbor_distance_max": 0.4,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_cli_requires_explicit_sources_and_has_provider_neutral_help() -> None:
    parser = build_parser()
    required = {action.dest for action in parser._actions if action.required}
    assert {"selector", "cng", "action_response"} <= required

    help_text = parser.format_help().lower()
    assert "runpod" not in help_text
    assert "runpod_results" not in help_text
    assert "/users/" not in help_text


def test_matched_control_payload_keeps_context_floor_and_fixed_route_roles(tmp_path: Path) -> None:
    cng_payload = {
        "policies": {
            "chart_feature_gate_equivariant_neural_context": {
                "summary": {
                    "n": 4,
                    "deceptive_n": 2,
                    "honest_n": 2,
                    "fixes_error": 2,
                    "honest_harms": 0,
                    "mean_reward": 0.25,
                    "mean_aligned_margin": 0.1,
                    "chosen_methods": {"abstain": 2, "bidir_linear": 2},
                },
                "choices": [
                    _selector_row("t1", "famA", method="abstain", target_status=None, route_action="steer_to_PASS", status_class="false_FAIL", reward=0.0),
                    _selector_row("t2", "famA", method="bidir_linear", target_status="PASS", route_action="steer_to_PASS", status_class="false_FAIL", reward=1.0),
                    _selector_row("t3", "famA", method="abstain", target_status=None, route_action="steer_to_FAIL", status_class="false_PASS", reward=0.0),
                    _selector_row("t4", "famA", method="bidir_linear", target_status="FAIL", route_action="steer_to_FAIL", status_class="false_PASS", reward=1.0),
                ],
            }
        }
    }

    selector_payload = {
        "policies": {
            "train_best_route_full_reward": {
                "summary": {
                    "n": 4,
                    "deceptive_n": 2,
                    "honest_n": 2,
                    "fixes_error": 3,
                    "honest_harms": 1,
                    "mean_reward": 0.12,
                    "mean_aligned_margin": 0.05,
                    "chosen_methods": {"bidir_linear": 4},
                },
                "choices": [
                    _selector_row("t1", "famA", method="bidir_linear", target_status="PASS", route_action="steer_to_PASS", status_class="false_FAIL", reward=1.0),
                    _selector_row("t2", "famA", method="bidir_linear", target_status="PASS", route_action="steer_to_PASS", status_class="false_FAIL", reward=1.0),
                    _selector_row("t3", "famA", method="bidir_linear", target_status="FAIL", route_action="steer_to_FAIL", status_class="false_PASS", reward=1.0),
                    _selector_row("t4", "famA", method="bidir_linear", target_status="FAIL", route_action="steer_to_FAIL", status_class="false_PASS", reward=1.0),
                ],
            },
            "learned_context_ridge_reward": {
                "summary": {
                    "n": 4,
                    "deceptive_n": 2,
                    "honest_n": 2,
                    "fixes_error": 1,
                    "honest_harms": 1,
                    "mean_reward": 0.0,
                    "mean_aligned_margin": 0.01,
                    "chosen_methods": {"abstain": 2, "global_mean": 2},
                },
                "choices": [
                    _selector_row("t1", "famA", method="global_mean", target_status="PASS", route_action="steer_to_PASS", status_class="false_FAIL", reward=1.0),
                    _selector_row("t2", "famA", method="abstain", target_status=None, route_action="steer_to_PASS", status_class="false_FAIL", reward=0.0),
                    _selector_row("t3", "famA", method="global_mean", target_status="FAIL", route_action="steer_to_FAIL", status_class="false_PASS", reward=0.0),
                    _selector_row("t4", "famA", method="abstain", target_status=None, route_action="steer_to_FAIL", status_class="false_PASS", reward=0.0),
                ],
            },
        }
    }

    rows = []
    for family in ("famA", "famB"):
        for route, target, status, reward in (
            ("steer_to_PASS", "PASS", "false_FAIL", 1.0),
            ("steer_to_FAIL", "FAIL", "false_PASS", 1.0),
        ):
            cid = f"{family}_{target.lower()}"
            rows.append(
                _action_row(
                    cid, family, method="abstain", route_action=route, target_status=None,
                    layer=None, alpha=0.0, status_class=status, reward=0.0, base_margin=2.0 if target == "FAIL" else -2.0,
                )
            )
            rows.append(
                _action_row(
                    cid, family, method="bidir_linear", route_action=route, target_status=target,
                    layer=16, alpha=96.0, status_class=status, reward=reward,
                    base_margin=2.0 if target == "FAIL" else -2.0,
                )
            )
            rows.append(
                _action_row(
                    cid, family, method="global_mean", route_action=route, target_status=target,
                    layer=8, alpha=48.0, status_class=status, reward=0.2,
                    base_margin=2.0 if target == "FAIL" else -2.0,
                )
            )
    action_response_payload = {"action_response": {"rows": rows}}

    cng_path = tmp_path / "cng.json"
    selector_path = tmp_path / "selector.json"
    ar_path = tmp_path / "action_response.json"
    _write_json(cng_path, cng_payload)
    _write_json(selector_path, selector_payload)
    _write_json(ar_path, action_response_payload)

    payload = build_payload(
        cng_path=cng_path,
        selector_path=selector_path,
        action_response_path=ar_path,
        context_policy="chart_feature_gate_equivariant_neural_context",
        route_floor_policy="train_best_route_full_reward",
        route_matched_methods={"bidir_linear"},
        out_of_domain_route_floor_fallback=False,
        objective="reward",
        folds_count=2,
        bootstrap=0,
        seed=0,
    )

    assert payload["policy_order"] == [
        "context_chart_feature_gate_equivariant_neural_context",
        "historical_route_floor",
        "learned_context_ridge_reward",
        "fixed_route_bidir_linear_L16_a96",
        "route_matched_fixed_coordinate",
    ]
    assert payload["policies"]["fixed_route_bidir_linear_L16_a96"]["type"] == "posthoc_fixed_route_whole_grid"
    assert payload["policies"]["route_matched_fixed_coordinate"]["methods_restricted_to"] == ["bidir_linear"]
    for fold in payload["policies"]["route_matched_fixed_coordinate"]["folds"].values():
        assert fold["coordinate"] == ["bidir_linear", 16, 96.0]

    receipt = build_public_receipt(payload)
    assert receipt["chronology"]["prospective_fresh_generation_controller"] == (
        "not established by this receipt"
    )
    assert receipt["policies"]["learned_context_ridge_reward"]["type"] == (
        "learned_oracle_route_feature_candidate_ranker"
    )
    assert receipt["chronology"]["context_only_selector"] == (
        "heldout-family ridge candidate ranker; receives the oracle route as a feature, "
        "scores both target signs, and never inspects heldout candidate outcomes"
    )
    assert (
        receipt["information_audit"]["policy_information"]["learned_context_ridge_reward"][
            "selected_target_route_mismatches"
        ]
        == 0
    )
    assert receipt["policies"]["route_matched_fixed_coordinate"]["folds"]
    encoded = json.dumps(receipt).lower()
    assert "\"choices\"" not in encoded
    assert "runpod" not in encoded
    assert str(tmp_path).lower() not in encoded


def test_markdown_flags_fixed_route_ceiling_and_matched_route_basis(tmp_path: Path) -> None:
    cng_payload = {
        "policies": {
            "chart_feature_gate_equivariant_neural_context": {
                "summary": {
                    "n": 2,
                    "deceptive_n": 1,
                    "honest_n": 1,
                    "fixes_error": 1,
                    "honest_harms": 0,
                    "mean_reward": 0.0,
                    "mean_aligned_margin": 0.05,
                    "chosen_methods": {"abstain": 1, "bidir_linear": 1},
                },
            }
        }
    }

    selector_payload = {"policies": {}}
    cng_path = tmp_path / "cng.json"
    selector_path = tmp_path / "selector.json"
    _write_json(cng_path, cng_payload)
    _write_json(selector_path, selector_payload)

    rows = [
        _action_row(
            "a", "famA", method="abstain", route_action="steer_to_PASS", target_status=None,
            layer=None, alpha=0.0, status_class="false_FAIL", reward=0.0, base_margin=-2.0,
        ),
        _action_row(
            "a", "famA", method="bidir_linear", route_action="steer_to_PASS", target_status="PASS",
            layer=16, alpha=96.0, status_class="false_FAIL", reward=1.0, base_margin=-2.0,
        ),
        _action_row(
            "b", "famB", method="abstain", route_action="steer_to_FAIL", target_status=None,
            layer=None, alpha=0.0, status_class="false_PASS", reward=0.0, base_margin=2.0,
        ),
        _action_row(
            "b", "famB", method="bidir_linear", route_action="steer_to_FAIL", target_status="FAIL",
            layer=16, alpha=96.0, status_class="false_PASS", reward=1.0, base_margin=2.0,
        ),
    ]
    action_response_payload = {"action_response": {"rows": rows}}
    ar_path = tmp_path / "action_response.json"
    _write_json(ar_path, action_response_payload)

    payload = build_payload(
        cng_path=cng_path,
        selector_path=selector_path,
        action_response_path=ar_path,
        context_policy="chart_feature_gate_equivariant_neural_context",
        route_floor_policy="train_best_route_full_reward",
        route_matched_methods={"bidir_linear"},
        out_of_domain_route_floor_fallback=True,
        objective="reward",
        folds_count=2,
        bootstrap=10,
        seed=0,
    )

    text = render_markdown(payload)
    assert "post-hoc whole-grid fixed-route ceiling (`fixed_route_bidir_linear_L16_a96`)" in text
    assert "heldout-family route-matched fixed-coordinate reconstruction" in text
    assert "`fixed_route_bidir_linear_L16_a96` is a post-hoc whole-grid feasibility ceiling" in text


def test_c1_public_receipt_records_learned_context_route_mismatch_contract() -> None:
    receipt = json.loads(
        Path("paper_artifacts/c1_matched_control_audit.json").read_text()
    )
    assert (
        receipt["information_audit"]["policy_information"]["learned_context_ridge_reward"][
            "selected_target_route_mismatches"
        ]
        == 233
    )


def test_regression_no_aligned_margin_metrics_for_compact_cng_payload(tmp_path: Path) -> None:
    cng_payload = {
        "policies": {
            "chart_feature_gate_equivariant_neural_context": {
                "summary": {
                    "n": 2,
                    "deceptive_n": 1,
                    "honest_n": 1,
                    "fixes_error": 1,
                    "honest_harms": 0,
                    "mean_reward": 0.15,
                    "mean_aligned_margin": 0.8,
                    "chosen_methods": {"abstain": 1, "bidir_linear": 1},
                },
                "choices": [
                    _compact_selector_row(
                        "a", "famA", method="abstain", target_status=None, route_action="steer_to_PASS", status_class="false_FAIL", reward=0.0,
                    ),
                    _compact_selector_row(
                        "b", "famB", method="abstain", target_status=None, route_action="steer_to_FAIL", status_class="false_PASS", reward=1.0,
                    ),
                ],
            },
        }
    }

    selector_payload = {
        "policies": {
            "train_best_route_full_reward": {
                "summary": {
                    "n": 2,
                    "deceptive_n": 1,
                    "honest_n": 1,
                    "fixes_error": 0,
                    "honest_harms": 0,
                    "mean_reward": 0.0,
                    "mean_aligned_margin": 0.3,
                    "chosen_methods": {"bidir_linear": 2},
                },
                "choices": [
                    _selector_row("a", "famA", method="bidir_linear", target_status="PASS", route_action="steer_to_PASS", status_class="false_FAIL", reward=1.0),
                    _selector_row("b", "famB", method="bidir_linear", target_status="FAIL", route_action="steer_to_FAIL", status_class="false_PASS", reward=0.0),
                ],
            }
        }
    }

    rows = [
        _action_row(
            "a", "famA", method="abstain", route_action="steer_to_PASS", target_status=None,
            layer=None, alpha=0.0, status_class="false_FAIL", reward=0.0, base_margin=-2.0,
        ),
        _action_row(
            "a", "famA", method="bidir_linear", route_action="steer_to_PASS", target_status="PASS",
            layer=16, alpha=96.0, status_class="false_FAIL", reward=1.0, base_margin=-2.0,
        ),
        _action_row(
            "b", "famB", method="abstain", route_action="steer_to_FAIL", target_status=None,
            layer=None, alpha=0.0, status_class="false_PASS", reward=0.0, base_margin=2.0,
        ),
        _action_row(
            "b", "famB", method="bidir_linear", route_action="steer_to_FAIL", target_status="FAIL",
            layer=16, alpha=96.0, status_class="false_PASS", reward=1.0, base_margin=2.0,
        ),
    ]
    action_response_payload = {"action_response": {"rows": rows}}
    cng_path = tmp_path / "cng.json"
    selector_path = tmp_path / "selector.json"
    ar_path = tmp_path / "action_response.json"
    _write_json(cng_path, cng_payload)
    _write_json(selector_path, selector_payload)
    _write_json(ar_path, action_response_payload)

    payload = build_payload(
        cng_path=cng_path,
        selector_path=selector_path,
        action_response_path=ar_path,
        context_policy="chart_feature_gate_equivariant_neural_context",
        route_floor_policy="train_best_route_full_reward",
        route_matched_methods={"bidir_linear"},
        out_of_domain_route_floor_fallback=False,
        objective="reward",
        folds_count=2,
        bootstrap=10,
        seed=0,
    )

    context_key = "context_chart_feature_gate_equivariant_neural_context"
    for section_name in ("paired_gaps", "family_clustered_gaps"):
        section = payload[section_name]
        assert context_key in section
        for pair in section[context_key].values():
            assert set(pair.keys()) == {"fixes_error", "honest_harm", "reward"}

    audit = payload["information_audit"]
    assert audit["total_conversations"] == 2
    policy_audit = audit["policy_information"]
    assert policy_audit[context_key]["route_truth_mismatches"] == 0
    assert policy_audit[context_key]["selected_target_route_mismatches"] == 0
    assert audit["raw_nonbaseline_candidate_counts"] == {
        "route_matched": 2,
        "counter_target": 0,
        "n_nonbaseline": 2,
    }
