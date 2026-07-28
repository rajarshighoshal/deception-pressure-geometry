from __future__ import annotations

import json
from pathlib import Path

from experiments.report_powered150_matched_control_audit import (
    build_parser,
    build_payload,
    build_public_receipt,
    extract_descriptive_structural_evidence,
    render_markdown,
)


def _selector_row(
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


# ---------------------------------------------------------------------------
# descriptive structural evidence fixture data
# ---------------------------------------------------------------------------

def _evidence_learned_geometry_map() -> dict:
    return {
        "policies": {
            "train_best_route_full_reward": {
                "summary": {"fixes_error": 539, "deceptive_n": 600, "honest_n": 600, "honest_harms": 0},
            },
            "chart_mean_context_cauto_d12_strict": {
                "summary": {"fixes_error": 596, "deceptive_n": 600, "honest_n": 600, "honest_harms": 1},
            },
            "graph_mean_context_cauto_gauto_d12_strict": {
                "summary": {"fixes_error": 598, "deceptive_n": 600, "honest_n": 600, "honest_harms": 1},
            },
            "chart_distilled_context_rf": {
                "summary": {"fixes_error": 584, "deceptive_n": 600, "honest_n": 600, "honest_harms": 2},
            },
            "product_z2_context_rf": {
                "summary": {"fixes_error": 581, "deceptive_n": 600, "honest_n": 600, "honest_harms": 2},
            },
            "typed_graph_context_reward": {
                "summary": {"fixes_error": 572, "deceptive_n": 600, "honest_n": 600, "honest_harms": 3},
            },
        }
    }


def _evidence_local_control_flow() -> dict:
    return {
        "policies": {
            "local_control_flow_context": {
                "summary": {"fixes_error": 275, "deceptive_n": 600, "honest_n": 600, "honest_harms": 11},
            },
            "global_control_flow_context": {
                "summary": {"fixes_error": 52, "deceptive_n": 600, "honest_n": 600, "honest_harms": 7},
            },
        },
        "paired_gaps": {
            "local_control_flow_context": {
                "global_control_flow_context": {
                    "fixes_error": {
                        "n": 600, "n_clusters": 20,
                        "point": 0.37166666666666665,
                        "ci95": [0.30640432909530313, 0.4436907954308003],
                    },
                },
            },
            "global_control_flow_context": {
                "local_control_flow_context": {
                    "fixes_error": {
                        "n": 600, "n_clusters": 20,
                        "point": -0.37166666666666665,
                        "ci95": [-0.4436907954308003, -0.3064043290953032],
                    },
                },
            },
        },
    }


def _evidence_gate_l20() -> dict:
    return {
        "summary": {
            "n": 1680,
            "routing_correct_rate": 1.0,
            "target_status_accuracy": 1.0,
        },
    }


def _evidence_fresh_equivariant() -> dict:
    return {
        "policies": {
            "route_hybrid_mean_probe": {
                "summary": {"deceptive_strict_fixes": 64, "deceptive_n": 100},
            },
            "atlas_context_local_k21_strict": {
                "summary": {"deceptive_strict_fixes": 71, "deceptive_n": 100},
            },
            "margin_argmax_all": {
                "summary": {"deceptive_strict_fixes": 79, "deceptive_n": 100},
            },
            "atlas_response_local_k5_strict": {
                "summary": {"deceptive_strict_fixes": 79, "deceptive_n": 100},
            },
        },
        "paired_gaps": {
            "atlas_context_local_k21_strict": {
                "route_hybrid_mean_probe": {
                    "strict_fix": {
                        "point": 0.07,
                        "ci95": [0.0, 0.1485148514851485],
                        "n": 100,
                        "n_clusters": 68,
                    },
                },
            },
            "atlas_response_local_k5_strict": {
                "margin_argmax_all": {
                    "strict_fix": {
                        "point": 0.0,
                        "ci95": [0.0, 0.0],
                        "n": 100,
                        "n_clusters": 68,
                    },
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# descriptive structural evidence tests
# ---------------------------------------------------------------------------

def test_descriptive_structural_evidence_extraction_from_fixtures(tmp_path: Path) -> None:
    lgm_path = tmp_path / "lgm.json"
    lcf_path = tmp_path / "lcf.json"
    gate_path = tmp_path / "gate.json"
    fea_path = tmp_path / "fea.json"
    _write_json(lgm_path, _evidence_learned_geometry_map())
    _write_json(lcf_path, _evidence_local_control_flow())
    _write_json(gate_path, _evidence_gate_l20())
    _write_json(fea_path, _evidence_fresh_equivariant())

    evidence = extract_descriptive_structural_evidence(
        learned_geometry_map_path=lgm_path,
        local_control_flow_path=lcf_path,
        gate_l20_path=gate_path,
        fresh_equivariant_path=fea_path,
    )

    # registration labels
    assert evidence["registration_status"] == "unregistered_descriptive"
    assert evidence["confirmatory"] is False
    assert "boundaries" in evidence
    assert "saved_field_response_free_selectors" in evidence["boundaries"]
    assert "local_control_flow" in evidence["boundaries"]
    assert "gate_l20" in evidence["boundaries"]
    assert "fresh_equivariant_atlas" in evidence["boundaries"]

    # saved-field
    sf = evidence["saved_field"]
    assert sf["train_best_route_full_reward"]["fixes_error"] == 539
    assert sf["chart_mean_context_cauto_d12_strict"]["fixes_error"] == 596
    assert sf["graph_mean_context_cauto_gauto_d12_strict"]["fixes_error"] == 598
    assert sf["chart_distilled_context_rf"]["fixes_error"] == 584
    assert sf["product_z2_context_rf"]["fixes_error"] == 581
    assert sf["typed_graph_context_reward"]["fixes_error"] == 572

    # locality proxy
    lp = evidence["locality_proxy"]
    assert lp["policy_summaries"]["local_control_flow_context"]["fixes_error"] == 275
    assert lp["policy_summaries"]["global_control_flow_context"]["fixes_error"] == 52
    paired = lp["paired_gaps"]["local_control_flow_context"]["global_control_flow_context"]["fixes_error"]
    assert abs(paired["point"] - 0.37166666666666665) < 1e-9
    assert paired["n"] == 600

    # gate L20
    gate = evidence["gate_l20_routing_diagnostic"]
    assert gate["n"] == 1680
    assert gate["routing_correct_rate"] == 1.0
    assert gate["target_status_accuracy"] == 1.0

    # fresh equivariant atlas
    fea = evidence["fresh_equivariant_atlas"]
    assert fea["policy_summaries"]["route_hybrid_mean_probe"]["deceptive_strict_fixes"] == 64
    assert fea["policy_summaries"]["atlas_context_local_k21_strict"]["deceptive_strict_fixes"] == 71
    assert fea["policy_summaries"]["margin_argmax_all"]["deceptive_strict_fixes"] == 79
    assert fea["policy_summaries"]["atlas_response_local_k5_strict"]["deceptive_strict_fixes"] == 79
    # paired gaps
    pg = fea["paired_gaps"]
    assert abs(pg["context_k21_vs_route_floor"]["point"] - 0.07) < 1e-9
    assert pg["context_k21_vs_route_floor"]["n"] == 100
    assert pg["context_k21_vs_route_floor"]["n_clusters"] == 68
    assert pg["response_k5_vs_margin_argmax"]["point"] == 0.0
    assert pg["response_k5_vs_margin_argmax"]["n"] == 100
    assert pg["response_k5_vs_margin_argmax"]["n_clusters"] == 68

    # source metadata
    for key in ("learned_geometry_map", "local_control_flow", "gate_l20", "fresh_equivariant_atlas"):
        src = evidence["sources"][key]
        assert "sha256" in src
        assert "byte_size" in src
        assert "path" in src


def test_evidence_section_absent_when_no_optional_args(tmp_path: Path) -> None:
    cng_payload = {
        "policies": {
            "chart_feature_gate_equivariant_neural_context": {
                "summary": {"deceptive_n": 1, "honest_n": 1, "fixes_error": 0, "honest_harms": 0,
                            "mean_reward": 0.0, "mean_aligned_margin": 0.0, "chosen_methods": {},
                            "n": 2},
                "choices": [],
            }
        }
    }
    selector_payload = {"policies": {}}
    cng_path = tmp_path / "cng.json"
    selector_path = tmp_path / "selector.json"
    _write_json(cng_path, cng_payload)
    _write_json(selector_path, selector_payload)
    rows = [
        _action_row("a", "famA", method="abstain", route_action="steer_to_PASS",
                    target_status=None, layer=None, alpha=0.0, status_class="false_FAIL",
                    reward=0.0, base_margin=-2.0),
        _action_row("a", "famA", method="bidir_linear", route_action="steer_to_PASS",
                    target_status="PASS", layer=16, alpha=96.0, status_class="false_FAIL",
                    reward=1.0, base_margin=-2.0),
    ]
    ar_path = tmp_path / "action_response.json"
    _write_json(ar_path, {"action_response": {"rows": rows}})

    payload = build_payload(
        cng_path=cng_path, selector_path=selector_path, action_response_path=ar_path,
        context_policy="chart_feature_gate_equivariant_neural_context",
        route_floor_policy="train_best_route_full_reward",
        route_matched_methods={"bidir_linear"}, out_of_domain_route_floor_fallback=True,
        objective="reward", folds_count=2, bootstrap=10, seed=0,
    )
    assert "descriptive_structural_evidence" not in payload

    receipt = build_public_receipt(payload)
    assert "descriptive_structural_evidence" not in receipt


def test_evidence_in_receipt_when_args_supplied(tmp_path: Path) -> None:
    """End-to-end: descriptive structural evidence flows through to the public receipt."""
    cng_payload = {
        "policies": {
            "chart_feature_gate_equivariant_neural_context": {
                "summary": {"deceptive_n": 1, "honest_n": 1, "fixes_error": 0, "honest_harms": 0,
                            "mean_reward": 0.0, "mean_aligned_margin": 0.0, "chosen_methods": {},
                            "n": 2},
                "choices": [],
            }
        }
    }
    selector_payload = {"policies": {}}
    cng_path = tmp_path / "cng.json"
    selector_path = tmp_path / "selector.json"
    gate_path = tmp_path / "gate.json"
    _write_json(cng_path, cng_payload)
    _write_json(selector_path, selector_payload)
    _write_json(gate_path, _evidence_gate_l20())
    rows = [
        _action_row("a", "famA", method="abstain", route_action="steer_to_PASS",
                    target_status=None, layer=None, alpha=0.0, status_class="false_FAIL",
                    reward=0.0, base_margin=-2.0),
        _action_row("a", "famA", method="bidir_linear", route_action="steer_to_PASS",
                    target_status="PASS", layer=16, alpha=96.0, status_class="false_FAIL",
                    reward=1.0, base_margin=-2.0),
    ]
    ar_path = tmp_path / "action_response.json"
    _write_json(ar_path, {"action_response": {"rows": rows}})

    payload = build_payload(
        cng_path=cng_path, selector_path=selector_path, action_response_path=ar_path,
        context_policy="chart_feature_gate_equivariant_neural_context",
        route_floor_policy="train_best_route_full_reward",
        route_matched_methods={"bidir_linear"}, out_of_domain_route_floor_fallback=True,
        objective="reward", folds_count=2, bootstrap=10, seed=0,
        gate_l20_path=gate_path,
    )
    assert "descriptive_structural_evidence" in payload
    assert payload["descriptive_structural_evidence"]["gate_l20_routing_diagnostic"]["n"] == 1680

    receipt = build_public_receipt(payload)
    assert "descriptive_structural_evidence" in receipt
    dse = receipt["descriptive_structural_evidence"]
    assert dse["registration_status"] == "unregistered_descriptive"
    assert dse["confirmatory"] is False
    assert dse["gate_l20_routing_diagnostic"]["n"] == 1680


def test_evidence_extraction_missing_policy_key_fails_loudly(tmp_path: Path) -> None:
    """Missing expected policy key raises KeyError."""
    bad_lgm = {"policies": {"some_other_key": {"summary": {}}}}
    lgm_path = tmp_path / "bad_lgm.json"
    _write_json(lgm_path, bad_lgm)
    with __import__("pytest").raises(KeyError, match="train_best_route_full_reward"):
        extract_descriptive_structural_evidence(
            learned_geometry_map_path=lgm_path,
            local_control_flow_path=None,
            gate_l20_path=None,
            fresh_equivariant_path=None,
        )


def test_gate_l20_count_is_1680_not_4658() -> None:
    """The public receipt preserves the corrected clean-clone count."""
    receipt = json.loads(
        Path("paper_artifacts/c1_matched_control_audit.json").read_text()
    )
    evidence = receipt["descriptive_structural_evidence"]
    gate = evidence["gate_l20_routing_diagnostic"]
    assert gate["n"] == 1680, f"gate_l20 n is {gate['n']}, expected 1680 (NOT 4658)"


def test_evidence_boundaries_contain_nonconfirmatory_chronology_labels() -> None:
    """The clean-clone receipt labels descriptive evidence nonconfirmatory."""
    receipt = json.loads(
        Path("paper_artifacts/c1_matched_control_audit.json").read_text()
    )
    evidence = receipt["descriptive_structural_evidence"]
    assert evidence["section"] == "descriptive_structural_evidence"
    assert evidence["registration_status"] == "unregistered_descriptive"
    assert evidence["confirmatory"] is False
    boundaries = evidence["boundaries"]
    assert "retrospective" in boundaries["saved_field_response_free_selectors"].lower()
    assert "not true ood" in boundaries["gate_l20"].lower()
    assert "separate fresh-family" in boundaries["fresh_equivariant_atlas"].lower()


# ---------------------------------------------------------------------------
# Apollo transfer boundary fixture data
# ---------------------------------------------------------------------------

def _evidence_apollo_residual_transfer() -> dict:
    return {
        "schema_version": 1,
        "kind": "apollo_residual_transfer_report",
        "layers": [
            {
                "layer": 16,
                "detectors": [
                    {
                        "name": "powered150_linear_pca64_zero_shot",
                        "metrics": {
                            "n": 484,
                            "auroc": 0.6286916208791209,
                            "orientation_corrected_auroc": 0.6286916208791209,
                        },
                    },
                    {
                        "name": "apollo_native_linear_pca64_train_eval",
                        "metrics": {
                            "n": 484,
                            "auroc": 0.8164835164835165,
                            "orientation_corrected_auroc": 0.8164835164835165,
                        },
                    },
                ],
            },
        ],
    }


def _evidence_apollo_path_transfer() -> dict:
    return {
        "schema_version": 1,
        "kind": "apollo_residual_path_transfer_report",
        "detectors": [
            {
                "name": "powered150_linear_path_zero_shot",
                "metrics": {
                    "n": 484,
                    "auroc": 0.5136160714285715,
                    "orientation_corrected_auroc": 0.5136160714285715,
                },
            },
            {
                "name": "apollo_native_linear_path_train_eval",
                "metrics": {
                    "n": 484,
                    "auroc": 0.7819024725274725,
                    "orientation_corrected_auroc": 0.7819024725274725,
                },
            },
            {
                "name": "powered150_grid_gcn_zero_shot",
                "metrics": {
                    "n": 484,
                    "auroc": 0.5710164835164835,
                    "orientation_corrected_auroc": 0.5710164835164835,
                },
            },
            {
                "name": "apollo_native_grid_gcn_train_eval",
                "metrics": {
                    "n": 484,
                    "auroc": 0.8649896978021978,
                    "orientation_corrected_auroc": 0.8649896978021978,
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# Apollo transfer boundary tests
# ---------------------------------------------------------------------------

def test_transfer_boundary_exact_extraction(tmp_path: Path) -> None:
    art_path = tmp_path / "art.json"
    apt_path = tmp_path / "apt.json"
    _write_json(art_path, _evidence_apollo_residual_transfer())
    _write_json(apt_path, _evidence_apollo_path_transfer())

    evidence = extract_descriptive_structural_evidence(
        learned_geometry_map_path=None,
        local_control_flow_path=None,
        gate_l20_path=None,
        fresh_equivariant_path=None,
        apollo_residual_transfer_path=art_path,
        apollo_path_transfer_path=apt_path,
    )

    tb = evidence["transfer_boundary"]
    assert tb["section"] == "descriptive_transfer_boundary"
    assert tb["registration_status"] == "unregistered_descriptive"
    assert tb["confirmatory"] is False
    assert tb["boundaries"]["task_instrument"] == "different task/instrument"
    assert tb["boundaries"]["control_target"] == (
        "target bank lacks machine-checkable control target"
    )
    assert tb["boundaries"]["representation"] == "representation transfer only"
    assert tb["boundaries"]["controller_status"] == "not a controller result"

    # residual metrics
    res = tb["residual"]
    assert res["powered150_linear_pca64_zero_shot"]["auroc"] == 0.6286916208791209
    assert res["powered150_linear_pca64_zero_shot"]["orientation_corrected_auroc"] == 0.6286916208791209
    assert res["powered150_linear_pca64_zero_shot"]["n"] == 484
    assert res["apollo_native_linear_pca64_train_eval"]["auroc"] == 0.8164835164835165
    assert res["apollo_native_linear_pca64_train_eval"]["orientation_corrected_auroc"] == 0.8164835164835165
    assert res["apollo_native_linear_pca64_train_eval"]["n"] == 484

    # path metrics
    path = tb["path"]
    assert path["powered150_linear_path_zero_shot"]["auroc"] == 0.5136160714285715
    assert path["powered150_linear_path_zero_shot"]["orientation_corrected_auroc"] == 0.5136160714285715
    assert path["powered150_linear_path_zero_shot"]["n"] == 484
    assert path["apollo_native_linear_path_train_eval"]["auroc"] == 0.7819024725274725
    assert path["apollo_native_linear_path_train_eval"]["orientation_corrected_auroc"] == 0.7819024725274725
    assert path["apollo_native_linear_path_train_eval"]["n"] == 484
    assert path["powered150_grid_gcn_zero_shot"]["auroc"] == 0.5710164835164835
    assert path["powered150_grid_gcn_zero_shot"]["orientation_corrected_auroc"] == 0.5710164835164835
    assert path["powered150_grid_gcn_zero_shot"]["n"] == 484
    assert path["apollo_native_grid_gcn_train_eval"]["auroc"] == 0.8649896978021978
    assert path["apollo_native_grid_gcn_train_eval"]["orientation_corrected_auroc"] == 0.8649896978021978
    assert path["apollo_native_grid_gcn_train_eval"]["n"] == 484

    # source metadata
    assert "apollo_residual_transfer" in tb["sources"]
    assert "apollo_path_transfer" in tb["sources"]
    for key in ("apollo_residual_transfer", "apollo_path_transfer"):
        src = tb["sources"][key]
        assert "sha256" in src
        assert "byte_size" in src
        assert "path" in src

    # evidence top-level still has other fields
    assert "sources" in evidence


def test_transfer_boundary_missing_detector_fails_loudly(tmp_path: Path) -> None:
    """Missing expected detector name raises KeyError."""
    bad_art = {
        "schema_version": 1,
        "kind": "apollo_residual_transfer_report",
        "layers": [{"layer": 16, "detectors": [{"name": "unexpected_detector", "metrics": {"n": 1, "auroc": 0.5, "orientation_corrected_auroc": 0.5}}]}],
    }
    art_path = tmp_path / "bad_art.json"
    _write_json(art_path, bad_art)
    with __import__("pytest").raises(KeyError, match="powered150_linear_pca64_zero_shot"):
        extract_descriptive_structural_evidence(
            learned_geometry_map_path=None,
            local_control_flow_path=None,
            gate_l20_path=None,
            fresh_equivariant_path=None,
            apollo_residual_transfer_path=art_path,
            apollo_path_transfer_path=None,
        )


def test_transfer_boundary_missing_layer_16_fails(tmp_path: Path) -> None:
    """Missing layer 16 raises ValueError."""
    bad_art = {
        "schema_version": 1,
        "kind": "apollo_residual_transfer_report",
        "layers": [{"layer": 0, "detectors": []}],
    }
    art_path = tmp_path / "bad_layer.json"
    _write_json(art_path, bad_art)
    with __import__("pytest").raises(ValueError, match="layer 16 not found"):
        extract_descriptive_structural_evidence(
            learned_geometry_map_path=None,
            local_control_flow_path=None,
            gate_l20_path=None,
            fresh_equivariant_path=None,
            apollo_residual_transfer_path=art_path,
            apollo_path_transfer_path=None,
        )


def test_transfer_boundary_absent_when_no_args(tmp_path: Path) -> None:
    """When neither Apollo arg is provided, transfer_boundary is absent."""
    evidence = extract_descriptive_structural_evidence(
        learned_geometry_map_path=None,
        local_control_flow_path=None,
        gate_l20_path=None,
        fresh_equivariant_path=None,
        apollo_residual_transfer_path=None,
        apollo_path_transfer_path=None,
    )
    assert "transfer_boundary" not in evidence


def test_transfer_boundary_flows_to_receipt(tmp_path: Path) -> None:
    """End-to-end: transfer boundary flows through to the public receipt."""
    cng_payload = {
        "policies": {
            "chart_feature_gate_equivariant_neural_context": {
                "summary": {"deceptive_n": 1, "honest_n": 1, "fixes_error": 0, "honest_harms": 0,
                            "mean_reward": 0.0, "mean_aligned_margin": 0.0, "chosen_methods": {},
                            "n": 2},
                "choices": [],
            }
        }
    }
    selector_payload = {"policies": {}}
    cng_path = tmp_path / "cng.json"
    selector_path = tmp_path / "selector.json"
    art_path = tmp_path / "art.json"
    _write_json(cng_path, cng_payload)
    _write_json(selector_path, selector_payload)
    _write_json(art_path, _evidence_apollo_residual_transfer())
    rows = [
        _action_row("a", "famA", method="abstain", route_action="steer_to_PASS",
                    target_status=None, layer=None, alpha=0.0, status_class="false_FAIL",
                    reward=0.0, base_margin=-2.0),
        _action_row("a", "famA", method="bidir_linear", route_action="steer_to_PASS",
                    target_status="PASS", layer=16, alpha=96.0, status_class="false_FAIL",
                    reward=1.0, base_margin=-2.0),
    ]
    ar_path = tmp_path / "action_response.json"
    _write_json(ar_path, {"action_response": {"rows": rows}})

    payload = build_payload(
        cng_path=cng_path, selector_path=selector_path, action_response_path=ar_path,
        context_policy="chart_feature_gate_equivariant_neural_context",
        route_floor_policy="train_best_route_full_reward",
        route_matched_methods={"bidir_linear"}, out_of_domain_route_floor_fallback=True,
        objective="reward", folds_count=2, bootstrap=10, seed=0,
        apollo_residual_transfer_path=art_path,
    )
    assert "descriptive_structural_evidence" in payload
    assert "transfer_boundary" in payload["descriptive_structural_evidence"]
    tb = payload["descriptive_structural_evidence"]["transfer_boundary"]
    assert tb["confirmatory"] is False
    assert tb["residual"]["powered150_linear_pca64_zero_shot"]["auroc"] == 0.6286916208791209

    receipt = build_public_receipt(payload)
    assert "descriptive_structural_evidence" in receipt
    assert "transfer_boundary" in receipt["descriptive_structural_evidence"]
    assert receipt["descriptive_structural_evidence"]["transfer_boundary"]["confirmatory"] is False


def test_transfer_boundary_path_report_missing_detector_fails(tmp_path: Path) -> None:
    """Missing expected detector in path report raises KeyError."""
    bad_apt = {
        "schema_version": 1,
        "kind": "apollo_residual_path_transfer_report",
        "detectors": [],
    }
    apt_path = tmp_path / "bad_apt.json"
    _write_json(apt_path, bad_apt)
    with __import__("pytest").raises(KeyError, match="powered150_linear_path_zero_shot"):
        extract_descriptive_structural_evidence(
            learned_geometry_map_path=None,
            local_control_flow_path=None,
            gate_l20_path=None,
            fresh_equivariant_path=None,
            apollo_residual_transfer_path=None,
            apollo_path_transfer_path=apt_path,
        )
