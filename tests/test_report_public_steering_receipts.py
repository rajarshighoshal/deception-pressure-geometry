from __future__ import annotations

import json
from pathlib import Path

from experiments.report_public_steering_receipts import (
    C12_PRIMARY_POLICIES,
    build_parser,
    build_c2_receipt,
    build_c12_receipt,
)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def test_cli_requires_explicit_sources_and_has_provider_neutral_help() -> None:
    parser = build_parser()
    required = {action.dest for action in parser._actions if action.required}
    assert {"c2_source", "c12_primary", "c12_followup"} <= required

    help_text = parser.format_help().lower()
    assert "runpod" not in help_text
    assert "runpod_results" not in help_text
    assert "/users/" not in help_text


def _generated_summary(*, fixes: int, harms: int, strict_fixes: int = 4) -> dict:
    return {
        "n": 12,
        "deceptive_n": 6,
        "honest_n": 6,
        "deceptive_status_fixes": fixes,
        "deceptive_strict_fixes": strict_fixes,
        "honest_status_harms": harms,
        "honest_strict_harms": harms,
        "mean_status_reward": (fixes - harms) / 12,
        "mean_strict_reward": (strict_fixes - harms) / 12,
    }


def test_c2_receipt_separates_fixed_and_learned_harm(tmp_path: Path) -> None:
    fixed = _generated_summary(fixes=6, harms=0)
    dense = _generated_summary(fixes=6, harms=2)
    source = {
        "policies": {
            "local_control_flow_fixed_88": {"summary": fixed, "sha256": "a" * 64},
            "local_control_flow_fixed_96": {"summary": fixed, "sha256": "b" * 64},
            "local_control_flow_fixed_128": {"summary": fixed, "sha256": "c" * 64},
            "local_control_flow_dense_alpha": {"summary": dense, "sha256": "d" * 64},
        },
        "provenance": {"git_hash": "abc1234", "git_dirty": True},
    }
    path = tmp_path / "c2.json"
    _write(path, source)

    receipt = build_c2_receipt(path)

    assert receipt["checks"]["fixed_88_96_128_summaries_identical"] is True
    assert receipt["checks"]["dense_minus_fixed_deceptive_status_fixes"] == 0
    assert receipt["checks"]["dense_minus_fixed_honest_status_harms"] == 2
    assert str(tmp_path).lower() not in json.dumps(receipt).lower()


def test_c12_receipt_pins_population_and_retrospective_bootstrap(tmp_path: Path) -> None:
    primary = {
        "summary": {
            name: {
                "n": 160,
                "deceptive_status_fixes": 48 if name == "bidir_tangent" else 37,
                "honest_status_harms": 2,
            }
            for name in C12_PRIMARY_POLICIES
        },
        "results_sha256": "e" * 64,
    }
    differences = [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, -1, 1, 1, 0, 1, 0]
    rows = []
    for index, difference in enumerate(differences):
        tangent = difference >= 0 and difference == 1
        off_tangent = difference == -1
        rows.append(
            {
                "conversation_id": f"d{index}",
                "status_class_before": "false_FAIL",
                "bidir_tangent_status_correct": tangent,
                "bidir_off_tangent_status_correct": off_tangent,
            }
        )
    rows.extend(
        {
            "conversation_id": f"h{index}",
            "status_class_before": "honest_PASS",
            "bidir_tangent_status_correct": True,
            "bidir_off_tangent_status_correct": True,
        }
        for index in range(16)
    )
    followup = {
        "summary": {"bidir_tangent": {}, "bidir_off_tangent": {}},
        "comparison_rows": rows,
        "results_sha256": "f" * 64,
    }
    primary_path = tmp_path / "primary.json"
    followup_path = tmp_path / "followup.json"
    _write(primary_path, primary)
    _write(followup_path, followup)

    receipt = build_c12_receipt(
        primary_path,
        followup_path,
        bootstrap_seed=20260724,
        bootstrap_resamples=10_000,
    )

    follow = receipt["off_tangent_followup"]
    assert follow["population"] == {"total_comparison_rows": 32, "deceptive_paired_rows": 16}
    assert follow["tangent_fixes"] == 9
    assert follow["off_tangent_fixes"] == 1
    assert follow["paired_difference"] == 0.5
    assert follow["paired_bootstrap"]["ci95"] == [0.1875, 0.8125]
    assert follow["provenance_tier"] == "retrospective_recalculation"


def test_c12_receipt_documents_shared_route_gating_contract(tmp_path: Path) -> None:
    primary = {
        "summary": {
            name: {
                "n": 160,
                "deceptive_status_fixes": 48 if name == "bidir_tangent" else 37,
                "honest_status_harms": 2,
            }
            for name in C12_PRIMARY_POLICIES
        },
        "results_sha256": "e" * 64,
    }
    followup = {
        "summary": {"bidir_tangent": {}, "bidir_off_tangent": {}},
        "comparison_rows": [
            {
                "conversation_id": "d0",
                "status_class_before": "false_FAIL",
                "bidir_tangent_status_correct": True,
                "bidir_off_tangent_status_correct": False,
            },
            {
                "conversation_id": "d1",
                "status_class_before": "false_PASS",
                "bidir_tangent_status_correct": False,
                "bidir_off_tangent_status_correct": False,
            },
        ],
        "results_sha256": "f" * 64,
    }
    primary_path = tmp_path / "primary.json"
    followup_path = tmp_path / "followup.json"
    _write(primary_path, primary)
    _write(followup_path, followup)

    receipt = build_c12_receipt(
        primary_path,
        followup_path,
        bootstrap_seed=20260724,
        bootstrap_resamples=10_000,
    )

    contract = receipt["primary_six_arm_evaluation"]["routing_contract"]
    assert "all nonbaseline bidirectional" in contract["shared_gate_scope"]
    assert contract["gate_action"] == "steer toward the gate-predicted true status, or abstain"
    assert contract["bidir_linear_projection"] == "raw_unprojected"
    assert contract["bidir_linear_route_policy"] == (
        "same gate route and abstention as bidir_tangent"
    )
    assert contract["baseline_intervention"] == "none"
