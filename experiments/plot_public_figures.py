"""Render the five public-paper figures from registry metadata and receipts.

The figures are generated only from ``docs/results_registry.yaml`` and the
tracked ``paper_artifacts/*.json`` receipts, with no dependence on
large local result trees.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

matplotlib.use("Agg", force=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "docs" / "results_registry.yaml"
DEFAULT_FIG_DIR = REPO_ROOT / "docs" / "figures"

FIGURE_NAMES = [
    "pressure_behavior_and_hazard.png",
    "decodability_timing_gap.png",
    "structured_action_control_audit.png",
    "natural_prose_control_failure.png",
    "gauge_control_null.png",
]

RECEIPT_SPECS: dict[str, dict[str, str]] = {
    "C1": {
        "path": "paper_artifacts/c1_matched_control_audit.json",
        "kind": "powered150_matched_control_public_receipt",
    },
    "C2": {
        "path": "paper_artifacts/c2_dose_control_receipt.json",
        "kind": "c2_dose_control_public_receipt",
    },
    "C5": {
        "path": "paper_artifacts/c5_natural_prose_control_receipt.json",
        "kind": "c5_natural_prose_control_public_receipt",
    },
    "C9": {
        "path": "paper_artifacts/c9_pressure_commitment_receipt.json",
        "kind": "c9_pressure_commitment_public_receipt",
    },
    "C10": {
        "path": "paper_artifacts/c10_postcommitment_detection_receipt.json",
        "kind": "c10_postcommitment_detection_public_receipt",
    },
    "C11": {
        "path": "paper_artifacts/c11_precommitment_warning_receipt.json",
        "kind": "c11_precommitment_warning_public_receipt",
    },
    "C12": {
        "path": "paper_artifacts/c12_steering_decomposition_receipt.json",
        "kind": "c12_steering_decomposition_public_receipt",
    },
    "C13": {
        "path": "paper_artifacts/c13_gauge_control_receipt.json",
        "kind": "c13_gauge_control_public_receipt",
    },
}

DPI = 200
FIG_W = 7.2
BAR_COLOR = "#2a78d6"
GRID_COLOR = "#e4e4e0"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
CONTEXT = "#8a8a85"
THRESHOLD = "#e34948"

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9.0,
    "axes.labelsize": 9.0,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelcolor": INK_SOFT,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID_COLOR,
    "axes.grid": True,
    "axes.grid.which": "major",
    "axes.axisbelow": True,
    "grid.color": GRID_COLOR,
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "savefig.dpi": DPI,
    "savefig.facecolor": SURFACE,
    "figure.figsize": (FIG_W, 4.6),
    "figure.autolayout": False,
}


def die(msg: str) -> None:
    raise SystemExit(f"plot_public_figures: ERROR: {msg}")


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        die(f"required source missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        die(f"invalid YAML payload in {path}")
    return payload


def _human_scope(scope: str) -> str:
    if not scope:
        return "not specified"
    if scope == "development_bank_no_ood_claims":
        return "development bank (no OOD claims)"
    if scope.startswith("layer-"):
        return scope.replace("_", " ")
    return scope.replace("_", " ")


def _humanize_status(value: str) -> str:
    if not isinstance(value, str):
        return "not reported"
    return value.replace("_", " ")


def _short_scope(scope: str) -> str:
    if not scope:
        return "not specified"
    normalized = _human_scope(scope)
    if normalized.startswith("layer 16"):
        return "L16 residual"
    return normalized


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _ci_text(row: dict[str, Any], label: str) -> str:
    lo, hi = _interval(row, label)
    return f"[{lo:.3f}, {hi:.3f}]"


def _format_count_rate(label: str, numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return f"{label}: {numerator}/{denominator}"
    return f"{label}: {numerator}/{denominator} ({_pct(numerator / denominator)})"


def _safe_pct_point(value: float) -> str:
    return f"{value * 100:.1f}%"


def _signed_point(value: float) -> str:
    return f"{value:+.4f}"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        die(f"required source missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        die(f"invalid JSON payload in {path}: {exc}")
    if not isinstance(payload, dict):
        die(f"invalid JSON payload in {path}")
    return payload


def assert_condition(condition: bool, msg: str) -> None:
    if not condition:
        die(msg)


def _interval(row: dict[str, Any], label: str) -> tuple[float, float]:
    if "ci95" in row:
        lo_hi = row["ci95"]
    elif "ci" in row:
        lo_hi = row["ci"]
    else:
        lo_hi = [row["lo"], row["hi"]]
    if isinstance(lo_hi, dict):
        lo, hi = lo_hi.get("lo"), lo_hi.get("hi")
    else:
        if not isinstance(lo_hi, list):
            die(f"{label} must contain a two-value interval")
        if len(lo_hi) != 2:
            die(f"{label} must contain a two-value interval")
        lo, hi = lo_hi[0], lo_hi[1]
    assert_condition(isinstance(lo, (int, float)), f"{label} lo must be numeric")
    assert_condition(isinstance(hi, (int, float)), f"{label} hi must be numeric")
    return float(lo), float(hi)


def _point(row: dict[str, Any], label: str) -> float:
    value = row.get("point")
    assert_condition(isinstance(value, (int, float)), f"{label} point missing")
    return float(value)


def _ratio(n: float, d: float, label: str) -> float:
    assert_condition(d > 0, f"{label} denominator must be positive")
    return float(n) / float(d)


def _safe_rate(n: float, d: float) -> float:
    return 0.0 if d <= 0 else float(n) / float(d)


def _to_int(payload: dict[str, Any], field: str, label: str, *, default: int | None = None) -> int | None:
    if field not in payload:
        return default
    value = payload[field]
    assert_condition(isinstance(value, int), f"{label} {field} must be integer")
    return int(value)


def validate_claim(payload: dict[str, Any], claim_id: str) -> dict[str, Any]:
    for field in ("id", "statement", "status", "registration_tier", "boundary"):
        if field not in payload:
            die(f"claim {claim_id} missing required field {field}")
    if payload["id"] != claim_id:
        die(f"claim id mismatch: expected {claim_id}, got {payload['id']}")
    return payload


def validate_receipt(payload: dict[str, Any], claim_id: str, kind: str) -> dict[str, Any]:
    assert_condition(payload.get("schema_version") == 1, f"{claim_id} receipt schema_version must be 1")
    assert_condition(payload.get("claim_id") == claim_id, f"{claim_id} receipt claim_id mismatch")
    assert_condition(payload.get("kind") == kind, f"{claim_id} receipt kind mismatch")
    assert_condition(isinstance(payload.get("producer"), str), f"{claim_id} receipt missing producer")
    assert_condition(isinstance(payload.get("producer_sha256"), str), f"{claim_id} receipt missing producer_sha256")
    return payload


def claim_meta() -> dict[str, dict[str, Any]]:
    registry = read_yaml(REGISTRY_PATH)
    claims = registry.get("claims")
    if not isinstance(claims, list):
        die("registry missing claims list")
    out: dict[str, dict[str, Any]] = {}
    for claim in claims:
        if not isinstance(claim, dict):
            die("registry claim entries must be mappings")
        claim_id = claim.get("id")
        if not isinstance(claim_id, str):
            die("registry claim ids must be strings")
        out[claim_id] = validate_claim(claim, claim_id)
    for claim_id in RECEIPT_SPECS:
        if claim_id not in out:
            die(f"registry missing required claim {claim_id}")
    return out


def load_receipts() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for claim_id, spec in RECEIPT_SPECS.items():
        payload = read_json(REPO_ROOT / spec["path"])
        out[claim_id] = validate_receipt(payload, claim_id, spec["kind"])
    return out


def parse_c1(payload: dict[str, Any]) -> dict[str, Any]:
    policies = payload.get("policies")
    assert_condition(isinstance(policies, dict), "C1 policies missing")
    info = payload.get("information_audit", {}).get("policy_information")
    assert_condition(isinstance(info, dict), "C1 policy information missing")

    def _extract(policy_name: str) -> dict[str, Any]:
        policy = policies.get(policy_name)
        assert_condition(isinstance(policy, dict), f"C1 missing policy {policy_name}")
        summary = policy.get("summary")
        assert_condition(isinstance(summary, dict), f"C1 {policy_name} summary missing")
        assert_condition(
            summary.get("deceptive_n", 0) > 0 and summary.get("honest_n", 0) > 0,
            f"C1 {policy_name} denominators invalid",
        )
        return summary

    cng_oracle = _extract("context_chart_feature_gate_equivariant_neural_context")
    learned_ridge_route_feature = _extract("learned_context_ridge_reward")
    route_matched = _extract("route_matched_fixed_coordinate")

    c1 = {
        "policies": {
            "cng_oracle_route": {
                "fix_rate": _ratio(
                    cng_oracle["fixes_error"], cng_oracle["deceptive_n"], "C1 CNG oracle-route fix rate"
                ),
                "harm_rate": _ratio(
                    cng_oracle["honest_harms"], cng_oracle["honest_n"], "C1 CNG oracle-route harm rate"
                ),
                "fixes": cng_oracle["fixes_error"],
                "harms": cng_oracle["honest_harms"],
                "deceptive_n": cng_oracle["deceptive_n"],
                "honest_n": cng_oracle["honest_n"],
            },
            "learned_ridge_route_feature": {
                "fix_rate": _ratio(
                    learned_ridge_route_feature["fixes_error"],
                    learned_ridge_route_feature["deceptive_n"],
                    "C1 learned ranker (route feature) fix rate",
                ),
                "harm_rate": _ratio(
                    learned_ridge_route_feature["honest_harms"],
                    learned_ridge_route_feature["honest_n"],
                    "C1 learned ranker (route feature) harm rate",
                ),
                "fixes": learned_ridge_route_feature["fixes_error"],
                "harms": learned_ridge_route_feature["honest_harms"],
                "deceptive_n": learned_ridge_route_feature["deceptive_n"],
                "honest_n": learned_ridge_route_feature["honest_n"],
            },
            "route_matched": {
                "fix_rate": _ratio(
                    route_matched["fixes_error"],
                    route_matched["deceptive_n"],
                    "C1 route-matched fixed-coordinate fix rate",
                ),
                "harm_rate": _ratio(
                    route_matched["honest_harms"],
                    route_matched["honest_n"],
                    "C1 route-matched fixed-coordinate harm rate",
                ),
                "fixes": route_matched["fixes_error"],
                "harms": route_matched["honest_harms"],
                "deceptive_n": route_matched["deceptive_n"],
                "honest_n": route_matched["honest_n"],
            },
        }
    }
    c1["route_audit"] = {
        "cng_truth_mismatches": int(
            info.get("context_chart_feature_gate_equivariant_neural_context", {}).get("route_truth_mismatches", 0)
        ),
        "cng_selected_mismatches": int(
            info.get("context_chart_feature_gate_equivariant_neural_context", {}).get("selected_target_route_mismatches", 0)
        ),
        "learned_ridge_route_feature_truth_mismatches": int(
            info.get("learned_context_ridge_reward", {}).get("route_truth_mismatches", 0)
        ),
        "learned_ridge_route_feature_selected_mismatches": int(
            info.get("learned_context_ridge_reward", {}).get("selected_target_route_mismatches", 0)
        ),
        "route_matched_truth_mismatches": int(
            info.get("route_matched_fixed_coordinate", {}).get("route_truth_mismatches", 0)
        ),
        "route_matched_selected_mismatches": int(
            info.get("route_matched_fixed_coordinate", {}).get("selected_target_route_mismatches", 0)
        ),
    }
    return c1


def parse_c2(payload: dict[str, Any]) -> dict[str, Any]:
    policies = payload.get("policies")
    assert_condition(isinstance(policies, dict), "C2 policies missing")
    checks = payload.get("checks")
    assert_condition(isinstance(checks, dict), "C2 checks missing")
    for key in (
        "fixed_88_96_128_summaries_identical",
        "dense_minus_fixed_deceptive_status_fixes",
        "dense_minus_fixed_honest_status_harms",
    ):
        assert_condition(key in checks, f"C2 missing check {key}")
    dense = policies["local_control_flow_dense_alpha"]["summary"]
    fixed = policies["local_control_flow_fixed_88"]["summary"]
    assert_condition(dense["deceptive_n"] > 0 and dense["honest_n"] > 0, "C2 rates require positive denominators")
    return {
        "checks": checks,
        "rates": {
            "fixed_fix_rate": _ratio(fixed["deceptive_status_fixes"], fixed["deceptive_n"], "C2 fixed fix rate"),
            "dense_fix_rate": _ratio(dense["deceptive_status_fixes"], dense["deceptive_n"], "C2 dense fix rate"),
            "fixed_harm_rate": _ratio(fixed["honest_status_harms"], fixed["honest_n"], "C2 fixed harm rate"),
            "dense_harm_rate": _ratio(dense["honest_status_harms"], dense["honest_n"], "C2 dense harm rate"),
            "fixed_deceptive_n": fixed["deceptive_n"],
            "fixed_honest_n": fixed["honest_n"],
            "dense_deceptive_n": dense["deceptive_n"],
            "dense_honest_n": dense["honest_n"],
        },
    }


def parse_c5(payload: dict[str, Any]) -> dict[str, Any]:
    eval_payload = payload.get("evaluation", {})
    assert_condition(isinstance(eval_payload, dict), "C5 evaluation missing")
    arms = eval_payload.get("arms")
    assert_condition(isinstance(arms, dict), "C5 arms missing")
    result = {}
    for arm in ("native_gated", "frequent_early_window", "family_matched_linear"):
        arm_payload = arms.get(arm, {})
        machine_status = arm_payload.get("machine_status", {})
        status = machine_status.get("deceptive_delta_vs_base", {})
        family_status = arm_payload.get("machine_status_by_family")
        assert_condition(isinstance(family_status, dict), f"C5 machine_status_by_family missing for {arm}")
        assert_condition(isinstance(machine_status, dict), f"C5 machine_status missing for {arm}")
        assert_condition(isinstance(status, dict), f"C5 missing machine-status delta for {arm}")
        assert_condition("point" in status, f"C5 missing point for {arm}")
        population = arm_payload.get("population", {})
        assert_condition(isinstance(population, dict), f"C5 population missing for {arm}")
        telemetry = arm_payload.get("intervention_telemetry", {})
        intervention = telemetry if isinstance(telemetry, dict) else {}
        deceptive_telemetry = intervention.get("deceptive", {})
        honest_telemetry = intervention.get("honest", {})
        if not isinstance(deceptive_telemetry, dict):
            deceptive_telemetry = {}
        if not isinstance(honest_telemetry, dict):
            honest_telemetry = {}

        result[arm] = {
            **status,
            "changed_vs_base": _to_int(arm_payload, "changed_vs_base", f"C5 {arm}", default=0) or 0,
            "deceptive_fixes": _to_int(machine_status, "deceptive_fixes", f"C5 {arm}", default=0) or 0,
            "deceptive_harms": _to_int(machine_status, "deceptive_harms", f"C5 {arm}", default=0) or 0,
            "honest_fixes": _to_int(machine_status, "honest_fixes", f"C5 {arm}", default=0) or 0,
            "honest_harms": _to_int(machine_status, "honest_harms", f"C5 {arm}", default=0) or 0,
            "deceptive_population": _to_int(population, "deceptive", f"C5 {arm}", default=0) or 0,
            "honest_population": _to_int(population, "honest", f"C5 {arm}", default=0) or 0,
            "deceptive_eligible": _to_int(deceptive_telemetry, "eligible", f"C5 {arm} deceptive intervention", default=0) or 0,
            "deceptive_fired": _to_int(deceptive_telemetry, "fired", f"C5 {arm} deceptive intervention", default=0) or 0,
            "honest_eligible": _to_int(honest_telemetry, "eligible", f"C5 {arm} honest intervention", default=0) or 0,
            "honest_fired": _to_int(honest_telemetry, "fired", f"C5 {arm} honest intervention", default=0) or 0,
            "machine_status_by_family": [
                {
                    "family": family,
                    "deceptive_delta": _point(
                        family_payload.get("deceptive_delta_vs_base", {}),
                        f"C5 {arm} {family} deceptive delta",
                    ),
                }
                for family, family_payload in sorted(family_status.items())
                if isinstance(family_payload, dict)
                and isinstance(family_payload.get("deceptive_delta_vs_base"), dict)
            ],
        }
        assert_condition(
            len(result[arm]["machine_status_by_family"]) == 4,
            f"C5 {arm} should expose four family deceptive-delta effects",
        )
    checks = payload.get("checks")
    assert_condition(isinstance(checks, dict), "C5 checks missing")
    heldout = eval_payload.get("heldout_families", [])
    assert_condition(isinstance(heldout, list), "C5 heldout families missing")
    native_scope = eval_payload.get("activation_scope", "")
    native_rows = arms["native_gated"]["population"]["deceptive"]
    return {
        "arms": result,
        "meta": {
            "activation_scope": native_scope,
            "heldout_family_count": len(heldout),
            "native_gated": {
                "deceptive_delta_ci95": arms["native_gated"]["machine_status"]["deceptive_delta_vs_base"]["ci95"],
                "deceptive_population": native_rows,
            },
        },
        "checks": checks,
    }


def parse_c9(payload: dict[str, Any]) -> dict[str, Any]:
    outcomes = payload.get("outcomes", {})
    assert_condition(isinstance(outcomes, dict), "C9 outcomes missing")
    hazard = payload.get("hazard", {})
    assert_condition(isinstance(hazard, dict), "C9 hazard missing")

    adaptive = outcomes.get("adaptive", {})
    scripted = outcomes.get("scripted", {})
    assert_condition(isinstance(adaptive, dict), "C9 adaptive outcomes missing")
    assert_condition(isinstance(scripted, dict), "C9 scripted outcomes missing")
    adaptive_arm = adaptive.get("arm_summary", {})
    scripted_arm = scripted.get("arm_summary", {})
    assert_condition(isinstance(adaptive_arm, dict), "C9 adaptive arm_summary missing")
    assert_condition(isinstance(scripted_arm, dict), "C9 scripted arm_summary missing")
    adaptive_contrasts = adaptive.get("contrasts", {})
    scripted_contrasts = scripted.get("contrasts", {})
    assert_condition(isinstance(adaptive_contrasts, dict), "C9 adaptive contrasts missing")
    assert_condition(isinstance(scripted_contrasts, dict), "C9 scripted contrasts missing")

    adaptive_coeff = hazard["adaptive_bank"]["adaptive_coefficients"]
    diss_coeff = hazard["dissociation_bank"]["coefficients"]
    assert_condition(isinstance(adaptive_coeff, dict) and isinstance(diss_coeff, dict), "C9 coefficient blocks missing")
    p3 = payload.get("p3", {}).get("primary", {})
    assert_condition(isinstance(p3, dict), "C9 p3 primary missing")
    safety_scope = payload.get("scope")
    assert_condition(isinstance(safety_scope, str), "C9 scope missing")
    sanity = payload.get("sanity", {})
    assert_condition(isinstance(sanity, dict), "C9 sanity missing")
    registered_p2a = []
    for bank_name, contrasts in (
        ("Scripted", scripted_contrasts),
        ("Adaptive", adaptive_contrasts),
    ):
        p2a = contrasts.get("P2a", {})
        if not isinstance(p2a, dict) or p2a.get("status") != "registered":
            continue
        contrast = p2a.get("contrast")
        if isinstance(contrast, dict) and "point" in contrast:
            registered_p2a.append({"name": f"{bank_name} P2a", "row": contrast})

    return {
        "scope": {"token": safety_scope, "label": _human_scope(safety_scope)},
        "sanity": sanity,
        "arms": {
            "scripted_smooth": scripted_arm["smooth"]["p1b_deceptive_commitment"],
            "scripted_late": scripted_arm["latedump"]["p1b_deceptive_commitment"],
            "adaptive_smooth": adaptive_arm["smooth"]["p1b_deceptive_commitment"],
            "adaptive_late": adaptive_arm["latedump"]["p1b_deceptive_commitment"],
        },
        "contrasts": {
            "p2a": registered_p2a,
        },
        "hazard": {
            "adaptive_alpha": adaptive_coeff["alpha"],
            "adaptive_gamma": adaptive_coeff["gamma"],
            "dissociation_alpha": diss_coeff["alpha"],
            "dissociation_gamma": diss_coeff["gamma"],
        },
        "p3": {
            "edf": p3["edf"],
            "fit": {"mean_auc": p3["fit"]["mean_auc"], "adequate": p3["fit"]["adequate"]},
        },
    }


def parse_c10(payload: dict[str, Any]) -> dict[str, Any]:
    primary = payload.get("primary", {})
    assert_condition(isinstance(primary, dict), "C10 primary missing")
    models = primary.get("models", {})
    assert_condition(isinstance(models, dict), "C10 models missing")
    gain = primary.get("exact_nuisance_gain", {})
    assert_condition(isinstance(gain, dict), "C10 gain block missing")
    perm = gain.get("nuisance_preserving_permutation", {})
    assert_condition(isinstance(perm, dict), "C10 permutation block missing")
    exact = models.get("exact_nuisance_family_balanced", {})
    graph = models.get("local_joint_top8", {})
    assert_condition(isinstance(exact, dict) and isinstance(graph, dict), "C10 model baselines missing")
    checks = payload.get("checks")
    assert_condition(isinstance(checks, dict), "C10 checks missing")

    return {
        "family": {
            "exact_prior_brier": exact["family_macro_brier"],
            "graph_brier": graph["family_macro_brier"],
        },
        "auroc": {
            "exact_prior": exact["event_pooled_auroc"],
            "graph": graph["event_pooled_auroc"],
            "probe": payload["linear_probe_comparator"]["secondary_auroc"]["registered_probe"],
        },
        "probe": {
            "brier": payload["linear_probe_comparator"]["family_macro_brier"]["registered_probe"],
        },
        "checks": checks,
        "null": {
            "mean": perm["null_summary"]["mean"],
            "observed_gain": perm["observed_family_macro_brier_gain"],
            "excess": perm["observed_excess_over_null_mean"],
            "min": perm["null_summary"]["min"],
            "max": perm["null_summary"]["max"],
            "pair_inventory": payload["exact_prefix_pairs"]["pair_inventory_count"],
            "event_count": primary["population"]["event_count"],
            "per_family_positive": gain["per_family_positive_gain_count"],
            "per_family_count": gain["per_family_count"],
        },
    }


def parse_c11(payload: dict[str, Any]) -> dict[str, Any]:
    checks = payload.get("checks")
    assert_condition(isinstance(checks, dict), "C11 checks missing")
    risk = payload.get("risk_gate_repair", {})
    assert_condition(isinstance(risk, dict), "C11 risk_gate_repair missing")
    risk_gain = risk["interpretation"]["primary_geometry_only_log_loss_gain_over_nuisance"]
    risk_ci = risk["interpretation"]["primary_geometry_only_log_loss_gain_ci"]["interval"]
    secondary = risk["secondary_comparisons"]["sealed_local_over_nuisance_prior"]
    secondary_ci = secondary["scenario_cluster_ci"]["interval"]
    return {
        "checks": checks,
        "spectral_auroc": payload["spectral_field"]["equal_view"]["auroc"],
        "connection_auroc": payload["connection_path_field"]["auroc"],
        "risk_gain": risk_gain,
        "risk_gain_ci": risk_ci,
        "sealed_local_gain": secondary["mean_log_loss_gain"],
        "sealed_local_ci": secondary_ci,
        "risk_n": payload["risk_gate_repair"]["model_scores"]["geometry_only_logistic"]["event_count"],
        "status": {
            "spectral": payload["spectral_field"]["equal_view"].get("status", "n/a")
            if isinstance(payload["spectral_field"]["equal_view"], dict)
            else "n/a",
            "connection": payload["connection_path_field"].get("status", "n/a"),
            "risk": payload["risk_gate_repair"]["conclusion"],
        },
    }


def parse_c12(payload: dict[str, Any]) -> dict[str, Any]:
    primary = payload.get("primary_six_arm_evaluation", {})
    assert_condition(isinstance(primary, dict), "C12 primary evaluation missing")
    populations = primary.get("population", {})
    assert_condition(isinstance(populations, dict), "C12 population missing")
    deceptive_n = populations["deceptive_rows"]
    honest_n = populations["honest_rows"]
    assert_condition(isinstance(deceptive_n, int) and deceptive_n > 0, "C12 deceptive denominator must be > 0")
    assert_condition(isinstance(honest_n, int) and honest_n > 0, "C12 honest denominator must be > 0")

    def _policy_rate(policy: str) -> dict[str, float]:
        p = primary["policies"][policy]
        return {
            "fix_rate": _ratio(p["deceptive_status_fixes"], deceptive_n, f"C12 {policy} fix rate"),
            "harm_rate": _ratio(p["honest_status_harms"], honest_n, f"C12 {policy} harm rate"),
            "strict_fix_rate": _ratio(p["deceptive_strict_fixes"], deceptive_n, f"C12 {policy} strict fix rate"),
            "strict_harm_rate": _ratio(p["honest_strict_harms"], honest_n, f"C12 {policy} strict harm rate"),
            "fixes": p["deceptive_status_fixes"],
            "strict_fixes": p["deceptive_strict_fixes"],
            "harms": p["honest_status_harms"],
            "strict_harms": p["honest_strict_harms"],
            "honest_n": honest_n,
            "deceptive_n": deceptive_n,
        }

    policies = primary.get("policies", {})
    assert_condition(isinstance(policies, dict), "C12 policies missing")

    followup = payload.get("off_tangent_followup", {})
    paired = followup.get("paired_bootstrap", {})
    return {
        "policies": {name: _policy_rate(name) for name in policies},
        "scope": payload.get("scope", ""),
        "followup": {
            "paired_difference": followup["paired_difference"],
            "paired_ci": paired["ci95"],
            "paired_bootseed": paired["seed"],
            "paired_pop": followup["population"],
        },
    }


def parse_c13(payload: dict[str, Any]) -> dict[str, Any]:
    causal = payload.get("causal_replay", {})
    assert_condition(isinstance(causal, dict), "C13 causal replay missing")
    transport = payload.get("transport_decomposition", {})
    assert_condition(isinstance(transport, dict), "C13 transport_decomposition missing")
    checks = payload.get("checks")
    assert_condition(isinstance(checks, dict), "C13 checks missing")
    holonomy = payload.get("holonomy_instrument", {})
    assert_condition(isinstance(holonomy, dict), "C13 holonomy instrument missing")
    return {
        "all_roots": causal["contrasts"]["all_roots"],
        "active_roots": causal["contrasts"]["active_roots"],
        "controller_scope": causal["controller_scope"],
        "proposal_status_counts": causal["proposal_status_counts"],
        "crossed": payload["transport_decomposition"]["crossed_committed_roots"],
        "sample_pool": payload["transport_decomposition"]["sample_pool"],
        "transport": payload["transport_decomposition"]["truthful_push"],
        "holonomy": holonomy,
        "checks": checks,
    }


def parse_data() -> dict[str, Any]:
    claims = claim_meta()
    receipts = load_receipts()
    return {
        "claims": claims,
        "c1": parse_c1(receipts["C1"]),
        "c2": parse_c2(receipts["C2"]),
        "c5": parse_c5(receipts["C5"]),
        "c9": parse_c9(receipts["C9"]),
        "c10": parse_c10(receipts["C10"]),
        "c11": parse_c11(receipts["C11"]),
        "c12": parse_c12(receipts["C12"]),
        "c13": parse_c13(receipts["C13"]),
    }


def _errbar(
    row: dict[str, Any],
    *,
    labels: list[str] | None = None,
    point: float | None = None,
) -> np.ndarray:
    lo, hi = _interval(row, labels[0] if labels else "row")
    if point is None:
        point = _point(row, "row")
    else:
        assert_condition(isinstance(point, (int, float)), "row point override missing or invalid")
    return np.array([point - lo, hi - point], dtype=float)


def fig_pressure_behavior_and_hazard(data: dict[str, Any], out_dir: Path) -> tuple[Path, int]:
    c9 = data["c9"]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 7.2), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)

    top = axes[0]
    labels = [
        "Scripted smooth",
        "Scripted late-compressed",
        "Adaptive smooth",
        "Adaptive late-compressed",
    ]
    rows = [
        c9["arms"]["scripted_smooth"],
        c9["arms"]["scripted_late"],
        c9["arms"]["adaptive_smooth"],
        c9["arms"]["adaptive_late"],
    ]
    colors = [BAR_COLOR, THRESHOLD if c9["arms"]["scripted_late"]["point"] < 0 else CONTEXT, BAR_COLOR, BAR_COLOR]
    points = np.array([_point(row, f"C9 {name}") for row, name in zip(rows, labels)], dtype=float)
    err = np.array(
        [_errbar(row, labels=[f"C9 {name}"]) for row, name in zip(rows, labels)],
        dtype=float,
    ).T
    x = np.arange(len(labels))
    top.plot(x[:2], points[:2], color=CONTEXT, lw=1.0, ls="--", alpha=0.8)
    top.plot(x[2:], points[2:], color=BAR_COLOR, lw=1.0, ls="--", alpha=0.8)
    top.errorbar(
        x,
        points,
        yerr=err,
        fmt="o",
        markersize=6,
        markerfacecolor=BAR_COLOR,
        markeredgecolor=INK,
        ecolor=INK,
        capsize=3,
    )
    top.scatter(x, points, s=68, c=colors, zorder=3)
    for idx, (value, name) in enumerate(zip(points, labels)):
        top.text(idx, value + 0.03, f"{value:.3f}", ha="center", va="bottom", fontsize=7.2)
    top.set_title("Pressure behavior", loc="left")
    top.set_xticks(x)
    top.set_xticklabels(labels, rotation=10, ha="right")
    top.set_ylabel("P1b deceptive commitment rate")
    top.set_ylim(-0.05, 1.0)
    top.axhline(0.0, color=INK_SOFT, lw=1)
    top.set_ylim(0.0, 1.0)
    p2a_lines: list[str] = []
    for row in c9["contrasts"]["p2a"]:
        lo, hi = _interval(row["row"], f"C9 {row['name']}")
        p2a_label = f"C9 {row['name']}"
        p2a_lines.append(
            f"{row['name']}: {_signed_point(_point(row['row'], p2a_label))} "
            f"[{lo:.3f}, {hi:.3f}]"
        )
    if not p2a_lines:
        p2a_lines.append("no registered P2a contrasts available")
    top.text(
        0.0,
        -0.36,
        (
            f"Development banks only; adaptive n={c9['sanity']['adaptive_population']}; "
            f"dissociation analysis n={c9['sanity']['dissociation_analyzed_population']}.\n"
            "Registered smooth−late-compressed P2a (Newcombe 95%):\n"
            f"{'; '.join(p2a_lines)}"
        ),
        transform=top.transAxes,
        fontsize=7.2,
        color=INK_SOFT,
        ha="left",
    )

    bottom = axes[1]
    hazard_labels = ["Adaptive α", "Adaptive γ", "Dissociation α", "Dissociation γ"]
    hazard_rows = [
        c9["hazard"]["adaptive_alpha"],
        c9["hazard"]["adaptive_gamma"],
        c9["hazard"]["dissociation_alpha"],
        c9["hazard"]["dissociation_gamma"],
    ]
    hazard_points = [_point(row, f"C9 {name}") for row, name in zip(hazard_rows, hazard_labels)]
    hazard_err = np.array(
        [_errbar(row, labels=[f"C9 {name}"]) for row, name in zip(hazard_rows, hazard_labels)],
        dtype=float,
    ).T
    hazard_x = np.arange(len(hazard_labels))
    bottom.errorbar(
        hazard_x,
        hazard_points,
        yerr=hazard_err,
        fmt="o",
        markersize=5,
        markerfacecolor=BAR_COLOR,
        markeredgecolor=INK,
        ecolor=INK,
        capsize=3,
    )
    bottom.set_title("Pressure hazard-law coefficients", loc="left")
    bottom.set_xticks(hazard_x)
    bottom.set_xticklabels(hazard_labels, rotation=10, ha="right")
    bottom.set_ylabel("Coefficient")
    bottom.axhline(0.0, color=INK_SOFT, lw=1)
    for axis in axes:
        axis.set_axisbelow(True)
        axis.grid(color=GRID_COLOR)
        axis.tick_params(color=INK_SOFT)

    path = out_dir / FIGURE_NAMES[0]
    fig.text(
        0.12,
        0.012,
        (
            f"Adaptive bank n={c9['sanity']['adaptive_population']}; dissociation analysis "
            f"n={c9['sanity']['dissociation_analyzed_population']} of "
            f"{c9['sanity']['dissociation_source_population']} conversations."
        ),
        fontsize=6.9,
        color=INK_SOFT,
        ha="left",
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.10, hspace=0.73)
    return path, _save(fig, path)


def fig_decodability_timing_gap(data: dict[str, Any], out_dir: Path) -> tuple[Path, int]:
    c10 = data["c10"]
    c11 = data["c11"]

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 6.6), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(
        "Post-commitment readout and pre-action warning",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )

    dec_ax, null_ax, auroc_ax, gain_ax = axes.flat

    dec_labels = ["Exact nuisance prior", "Relational graph", "Registered probe"]
    dec_points = [
        c10["family"]["exact_prior_brier"],
        c10["family"]["graph_brier"],
        c10["probe"]["brier"],
    ]
    dec_x = np.arange(len(dec_labels))
    for idx, value in enumerate(dec_points):
        dec_ax.scatter([idx], [value], s=44, color=BAR_COLOR, zorder=3)
        dec_ax.text(idx, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=7.2)
    dec_ax.set_xticks(dec_x)
    dec_ax.set_xticklabels(["Exact nuisance", "Relational graph", "Linear probe"], rotation=12, ha="right")
    dec_ax.set_ylabel("Family-macro Brier ↓")
    dec_ax.set_title("(a) Post-commitment prediction", loc="left")

    null_labels = ["Observed\ngraph gain", "Permutation\nnull (mean)", "Excess over\nnull"]
    null_observed = float(c10["null"]["observed_gain"])
    null_null_mean = float(c10["null"]["mean"])
    null_excess = float(c10["null"]["excess"])
    null_range_min = float(c10["null"]["min"])
    null_range_max = float(c10["null"]["max"])
    null_points = [null_observed, null_null_mean, null_excess]
    null_x = np.arange(len(null_labels))

    null_ax.errorbar(
        null_x,
        null_points,
        yerr=np.zeros((2, len(null_x))),
        fmt=" ",
        ecolor=INK,
        capsize=3,
        zorder=3,
    )

    null_colors = [BAR_COLOR, CONTEXT, THRESHOLD if null_excess < 0 else BAR_COLOR]
    null_ax.scatter(
        null_x,
        null_points,
        s=56,
        c=null_colors,
        marker="o",
        edgecolors=INK,
        linewidths=0.9,
        zorder=4,
    )

    null_ax.vlines(1, null_range_min, null_range_max, color=CONTEXT, lw=2.2, alpha=0.85)
    null_ax.scatter([1, 1], [null_range_min, null_range_max], color=CONTEXT, marker="_")

    for idx, (value, color) in enumerate(zip(null_points, null_colors)):
        null_ax.text(
            idx,
            value + (0.0025 if value >= 0 else -0.0025),
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=7.0,
            color=color,
        )
    null_ax.text(
        1.0,
        null_range_max + 0.004,
        (
            "Permutation null range: "
            f"[{null_range_min:.3f}, {null_range_max:.3f}]\n"
            f"mean={null_null_mean:+.3f}"
        ),
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=INK_SOFT,
    )
    null_ax.set_xticks(null_x)
    null_ax.set_xticklabels(null_labels)
    null_ax.set_ylabel("Brier gain over nuisance ↑")
    null_ax.set_title("(b) Nuisance-preserving null", loc="left")
    null_ax.axhline(0.0, color=INK_SOFT, lw=1)

    warn_auroc_labels = ["Spectral", "Connection"]
    warn_auroc_points = [c11["spectral_auroc"], c11["connection_auroc"]]
    warn_auroc_x = np.arange(len(warn_auroc_labels))
    auroc_ax.scatter(
        warn_auroc_x,
        warn_auroc_points,
        s=57,
        c=[BAR_COLOR, CONTEXT],
        marker="o",
        edgecolors=INK,
        linewidths=0.9,
        zorder=3,
    )
    auroc_ax.set_xticks(warn_auroc_x)
    auroc_ax.set_xticklabels(["Spectral field", "Path connection"])
    auroc_ax.set_title("(c) Pre-action warning channels", loc="left")
    auroc_ax.set_ylabel("AUROC ↑")
    auroc_ax.set_ylim(0.28, 0.96)
    auroc_ax.axhline(0.5, color=INK_SOFT, ls="--", lw=1.1, zorder=0)
    auroc_ax.text(
        1.00,
        0.5 + 0.014,
        "chance = 0.500",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=INK_SOFT,
    )
    for x_i, value in zip(warn_auroc_x, warn_auroc_points):
        auroc_ax.text(x_i, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=7.0)

    warn_gain_labels = ["Geometry-only", "Sealed-local"]
    warn_gain_points = [c11["risk_gain"], c11["sealed_local_gain"]]
    warn_gain_x = np.arange(len(warn_gain_labels))
    warn_gain_err = np.array(
        [
            _errbar({"ci95": c11["risk_gain_ci"]}, labels=["C11 geometry-only log-loss gain"], point=c11["risk_gain"]),
            _errbar(
                {"ci95": c11["sealed_local_ci"]},
                labels=["C11 sealed-local log-loss gain"],
                point=c11["sealed_local_gain"],
            ),
        ],
        dtype=float,
    ).T
    gain_ax.bar(warn_gain_x, warn_gain_points, color=[THRESHOLD, CONTEXT], alpha=0.9)
    gain_colors = [THRESHOLD if point < 0 else BAR_COLOR for point in warn_gain_points]
    gain_ax.scatter(warn_gain_x, warn_gain_points, s=56, c=gain_colors, zorder=3)
    gain_ax.errorbar(
        warn_gain_x,
        warn_gain_points,
        yerr=warn_gain_err,
        fmt="none",
        color=INK,
        ecolor=INK,
        capsize=3,
    )
    for x_i, value, label in zip(warn_gain_x, warn_gain_points, warn_gain_labels):
        gain_ax.text(x_i, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=7.0)
    gain_ax.set_xticks(warn_gain_x)
    gain_ax.set_xticklabels(warn_gain_labels, rotation=10, ha="right")
    gain_ax.set_title("(d) Pre-action risk prediction", loc="left")
    gain_ax.set_ylabel("Log-loss gain over nuisance ↑")
    gain_ax.axhline(0.0, color=INK_SOFT, ls="--", lw=1)

    for axis in axes.flat:
        axis.set_axisbelow(True)
        axis.grid(color=GRID_COLOR)
        axis.tick_params(color=INK_SOFT)

    path = out_dir / FIGURE_NAMES[1]
    fig.text(
        0.5,
        0.018,
        (
            f"C10: {c10['null']['event_count']} scored honest/deceptive events; "
            f"{c10['null']['pair_inventory']} exact-prefix pairs. "
            f"C11: {c11['risk_n']} risk-model events.\n"
            "Different instruments and populations; this is not a matched temporal ablation."
        ),
        ha="center",
        fontsize=7.1,
        color=INK_SOFT,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.15, wspace=0.30, hspace=0.42)
    return path, _save(fig, path)


def fig_structured_action_control_audit(data: dict[str, Any], out_dir: Path) -> tuple[Path, int]:
    c1 = data["c1"]
    c2 = data["c2"]
    c12 = data["c12"]
    c12_deceptive_n = [c12["policies"][name]["deceptive_n"] for name in c12["policies"]]
    c12_honest_n = [c12["policies"][name]["honest_n"] for name in c12["policies"]]
    c12_denoms = (
        f"deceptive={min(c12_deceptive_n)}..{max(c12_deceptive_n)}, "
        f"honest={min(c12_honest_n)}..{max(c12_honest_n)}"
        if len(set(c12_deceptive_n)) > 1 or len(set(c12_honest_n)) > 1
        else f"deceptive={c12_deceptive_n[0]}, honest={c12_honest_n[0]}"
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("Structured-action control audit", fontsize=13, fontweight="bold", y=0.98)

    c1_ax, c2_ax, c12_fix_ax, c12_strict_ax = axes.flat
    policy_labels = [
        "None",
        "Raw\nlinear",
        "Tangent",
        "Global\nmean",
        "Global\nprobe",
        "Random",
    ]

    c1_order = ("learned_ridge_route_feature", "cng_oracle_route", "route_matched")
    c1_labels = [
        "Ridge ranker\n(route feature)",
        "CNG\n(hard route)",
        "Fixed L16\n(hard route)",
    ]
    c1_x = np.arange(len(c1_order))
    c1_fix = [c1["policies"][name]["fix_rate"] for name in c1_order]
    c1_harm = [c1["policies"][name]["harm_rate"] for name in c1_order]
    c1_bar_w = 0.34
    c1_ax.bar(c1_x - 0.17, c1_fix, width=c1_bar_w, color=BAR_COLOR, label="deceptive fix")
    c1_ax.bar(c1_x + 0.17, c1_harm, width=c1_bar_w, color=THRESHOLD, label="honest harm")
    c1_ax.set_title("(a) C1 status endpoint by policy", loc="left", fontsize=8.8)
    c1_ax.set_xticks(c1_x)
    c1_ax.set_xticklabels(c1_labels, fontsize=7.2)
    c1_ax.set_ylim(0.0, 1.0)
    c1_ax.set_ylabel("rate")
    for i, policy in enumerate(c1_order):
        c1_ax.text(
            i - 0.17,
            max(0.0, c1_fix[i] - 0.06),
            f"{c1['policies'][policy]['fixes']}/{c1['policies'][policy]['deceptive_n']}",
            ha="center",
            fontsize=6.5,
        )
        c1_ax.text(
            i + 0.17,
            max(0.0, c1_harm[i] - 0.06),
            f"{c1['policies'][policy]['harms']}/{c1['policies'][policy]['honest_n']}",
            ha="center",
            fontsize=6.5,
        )
    c1_ax.text(
        0.0,
        -0.31,
        (
            "Selected counter-route candidates: ridge "
            f"{c1['route_audit']['learned_ridge_route_feature_selected_mismatches']}/1,200; "
            "hard-route CNG/fixed: 0."
        ),
        transform=c1_ax.transAxes,
        fontsize=6.2,
        color=INK_SOFT,
        ha="left",
    )

    c2_x = np.arange(2)
    c2_fix = [c2["rates"]["fixed_fix_rate"], c2["rates"]["dense_fix_rate"]]
    c2_harm = [c2["rates"]["fixed_harm_rate"], c2["rates"]["dense_harm_rate"]]
    c2_bar_w = 0.34
    c2_ax.bar(c2_x - 0.17, c2_fix, width=c2_bar_w, color=BAR_COLOR, label="deceptive fix")
    c2_ax.bar(c2_x + 0.17, c2_harm, width=c2_bar_w, color=THRESHOLD, label="honest harm")
    c2_ax.set_title("(b) C2 dose transfer design", loc="left", fontsize=9.0)
    c2_ax.set_xticks(c2_x)
    c2_ax.set_xticklabels(["Fixed", "Dense"])
    c2_ax.set_ylim(0.0, 1.0)
    c2_ax.set_ylabel("rate")
    c2_ax.text(
        0.03,
        0.78,
        f"fixed: {c2['rates']['fixed_fix_rate']:.1%} fix · {c2['rates']['fixed_harm_rate']:.1%} harm",
        transform=c2_ax.transAxes,
        fontsize=6.7,
    )
    c2_ax.text(
        0.03,
        0.63,
        (
            "dense: "
            f"{c2['rates']['dense_fix_rate']:.1%} fix · "
            f"{c2['rates']['dense_harm_rate']:.1%} harm"
        ),
        transform=c2_ax.transAxes,
        fontsize=6.7,
    )

    c12_order = (
        "baseline",
        "bidir_linear",
        "bidir_tangent",
        "global_mean_gated",
        "global_probe_gated",
        "random_gated",
    )
    c12_x = np.arange(len(c12_order))
    c12_fix = [c12["policies"][name]["fix_rate"] for name in c12_order]
    c12_harm = [c12["policies"][name]["harm_rate"] for name in c12_order]
    c12_strict_fix = [c12["policies"][name]["strict_fix_rate"] for name in c12_order]
    c12_strict_harm = [c12["policies"][name]["strict_harm_rate"] for name in c12_order]
    c12_fix_ax.bar(c12_x - 0.18, c12_fix, width=0.35, color=BAR_COLOR, label="deceptive fix")
    c12_fix_ax.bar(
        c12_x + 0.18,
        c12_strict_fix,
        width=0.35,
        color="#1b6ec2",
        label="deceptive strict fix",
    )
    c12_fix_ax.set_title("(c) C12 status endpoint", loc="left", fontsize=8.6)
    c12_fix_ax.set_xticks(c12_x)
    c12_fix_ax.set_xticklabels(policy_labels, fontsize=6.6)
    c12_fix_ax.set_ylim(0.0, 1.0)
    c12_fix_ax.set_ylabel("rate")
    for idx, policy in enumerate(c12_order):
        c12_fix_ax.text(
            idx - 0.18,
            max(0.0, c12_fix[idx] - 0.06),
            f"{c12['policies'][policy]['fixes']}/{c12['policies'][policy]['deceptive_n']}",
            ha="center",
            fontsize=5.9,
        )
        c12_fix_ax.text(
            idx + 0.18,
            max(0.0, c12_strict_fix[idx] - 0.06),
            f"{c12['policies'][policy]['strict_fixes']}/{c12['policies'][policy]['deceptive_n']}",
            ha="center",
            fontsize=5.9,
        )

    c12_strict_ax.bar(c12_x - 0.18, c12_harm, width=0.35, color=THRESHOLD, label="honest harm")
    c12_strict_ax.bar(
        c12_x + 0.18,
        c12_strict_harm,
        width=0.35,
        color="#b03a36",
        label="honest strict harm",
    )
    c12_strict_ax.set_title("(d) C12 strict endpoint", loc="left", fontsize=8.6)
    c12_strict_ax.set_xticks(c12_x)
    c12_strict_ax.set_xticklabels(policy_labels, fontsize=6.6)
    c12_strict_ax.set_ylim(0.0, 1.0)
    c12_strict_ax.set_ylabel("rate")
    for idx, policy in enumerate(c12_order):
        c12_strict_ax.text(
            idx - 0.18,
            max(0.0, c12_harm[idx] - 0.06),
            f"{c12['policies'][policy]['harms']}/{c12['policies'][policy]['honest_n']}",
            ha="center",
            fontsize=5.9,
        )
        c12_strict_ax.text(
            idx + 0.18,
            max(0.0, c12_strict_harm[idx] - 0.06),
            f"{c12['policies'][policy]['strict_harms']}/{c12['policies'][policy]['honest_n']}",
            ha="center",
            fontsize=5.9,
        )
    c12_strict_ax.text(
        0.01,
        0.76,
        f"Arm denominators: {c12_denoms}.\n"
        "Strict = status and caveat criteria both satisfied.",
        transform=c12_strict_ax.transAxes,
        fontsize=6.2,
        color=INK_SOFT,
        ha="left",
    )
    c12_fix_ax.legend(loc="upper right", fontsize=6.6, ncol=1, frameon=False)
    c12_strict_ax.legend(loc="upper right", fontsize=6.6, ncol=1, frameon=False)

    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(color=GRID_COLOR)
        ax.tick_params(color=INK_SOFT)
        for spine_key in ("top", "right"):
            ax.spines[spine_key].set_visible(False)

    c1_ax.legend(loc="upper right", fontsize=6.6, frameon=False)
    c2_ax.legend(loc="upper right", fontsize=6.6, frameon=False)
    path = out_dir / FIGURE_NAMES[2]
    fig.text(
        0.5,
        0.02,
        (
            "C1/C2 use 600 deceptive and 600 honest rows. "
            "C12 is a pilot; all nonbaseline arms are gate-routed and raw linear is unprojected.\n"
            f"Its off-tangent follow-up uses 16 deceptive pairs (9 vs 1): difference "
            f"{c12['followup']['paired_difference']:.3f} "
            f"[{c12['followup']['paired_ci'][0]:.3f}, {c12['followup']['paired_ci'][1]:.3f}]."
        ),
        ha="center",
        fontsize=6.9,
        color=INK_SOFT,
    )
    fig.subplots_adjust(
        left=0.10, right=0.99, top=0.90, bottom=0.15, hspace=0.70, wspace=0.30
    )
    return path, _save(fig, path)


def fig_natural_prose_control_failure(data: dict[str, Any], out_dir: Path) -> tuple[Path, int]:
    c5 = data["c5"]
    claim5 = data["claims"]["C5"]
    fig, (effect_ax, transition_ax) = plt.subplots(2, 1, figsize=(FIG_W, 6.6), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)

    labels = ["native", "early\nwindow", "linear\ncomparator"]
    arm_order = [
        c5["arms"]["native_gated"],
        c5["arms"]["frequent_early_window"],
        c5["arms"]["family_matched_linear"],
    ]
    x = np.arange(len(labels))
    points = [_point(arm, name) for arm, name in zip(arm_order, labels)]
    intervals = [_interval(arm, name) for arm, name in zip(arm_order, labels)]
    family_effects = [arm.get("machine_status_by_family", []) for arm in arm_order]
    errs = np.array(
        [
            np.array(points) - np.array([lo for lo, _hi in intervals]),
            np.array([hi for _lo, hi in intervals]) - np.array(points),
        ]
    )

    effect_ax.errorbar(x, points, yerr=errs, fmt="o", color=INK, ecolor=INK, capsize=3.5)
    effect_ax.scatter(x, points, s=58, color=BAR_COLOR, zorder=3)
    for idx, (pt, arm, family_deltas) in enumerate(zip(points, arm_order, family_effects)):
        if family_deltas:
            family_x = np.full(len(family_deltas), x[idx], dtype=float) + np.linspace(
                -0.20, 0.20, len(family_deltas)
            )
            family_y = np.array([entry["deceptive_delta"] for entry in family_deltas], dtype=float)
            effect_ax.scatter(
                family_x,
                family_y,
                s=20,
                color=INK_SOFT,
                alpha=0.3,
                zorder=2,
            )
        effect_ax.text(idx, pt + 0.03, f"{pt:.3f}", ha="center", va="bottom", fontsize=6.8)
        net = arm.get("deceptive_fixes", 0) - arm.get("deceptive_harms", 0)
        effect_ax.text(idx, -0.265, f"net {net:+d}/48", ha="center", fontsize=6.3)
    effect_ax.set_title(
        f"C5 natural-prose development controller: {claim5['status']}",
        loc="left",
    )
    effect_ax.set_xticks(x)
    effect_ax.set_xticklabels(labels, fontsize=7.2)
    effect_ax.set_ylabel("Δ deceptive status correction\n(machine score)")
    effect_ax.axhline(0.0, color=INK_SOFT, ls="--", lw=1)
    effect_ax.set_xlim(-0.35, len(labels) - 0.65)
    effect_ax.set_ylim(-0.30, 0.55)
    effect_ax.grid(color=GRID_COLOR)
    effect_ax.set_axisbelow(True)

    deceptive_fix = [
        _safe_rate(arm.get("deceptive_fixes", 0), arm.get("deceptive_population", 0))
        for arm in arm_order
    ]
    deceptive_harm = [
        _safe_rate(arm.get("deceptive_harms", 0), arm.get("deceptive_population", 0))
        for arm in arm_order
    ]
    honest_fix = [
        _safe_rate(arm.get("honest_fixes", 0), arm.get("honest_population", 0))
        for arm in arm_order
    ]
    honest_harm = [
        _safe_rate(arm.get("honest_harms", 0), arm.get("honest_population", 0))
        for arm in arm_order
    ]

    bar_w = 0.20
    transition_ax.bar(x - 0.3, deceptive_fix, width=bar_w, color=BAR_COLOR, label="deceptive fixed")
    transition_ax.bar(x - 0.1, deceptive_harm, width=bar_w, color="#a13f2e", label="deceptive harmed")
    transition_ax.bar(x + 0.1, honest_fix, width=bar_w, color="#2f7f48", label="honest fixed")
    transition_ax.bar(x + 0.3, honest_harm, width=bar_w, color=THRESHOLD, label="honest harmed")
    for idx, arm in enumerate(arm_order):
        transition_ax.text(
            idx - 0.30,
            max(0.0, deceptive_fix[idx] - 0.06),
            f"{arm.get('deceptive_fixes', 0)}/{arm.get('deceptive_population', 0)}",
            ha="center",
            fontsize=5.8,
        )
        transition_ax.text(
            idx - 0.1,
            max(0.0, deceptive_harm[idx] - 0.06),
            f"{arm.get('deceptive_harms', 0)}/{arm.get('deceptive_population', 0)}",
            ha="center",
            fontsize=5.8,
        )
        transition_ax.text(
            idx + 0.1,
            max(0.0, honest_fix[idx] - 0.06),
            f"{arm.get('honest_fixes', 0)}/{arm.get('honest_population', 0)}",
            ha="center",
            fontsize=5.8,
        )
        transition_ax.text(
            idx + 0.3,
            max(0.0, honest_harm[idx] - 0.06),
            f"{arm.get('honest_harms', 0)}/{arm.get('honest_population', 0)}",
            ha="center",
            fontsize=5.8,
        )
    transition_ax.set_title("(b) Transition outcomes per arm", loc="left")
    transition_ax.set_xticks(x)
    transition_ax.set_xticklabels(labels, fontsize=7.2)
    transition_ax.set_ylabel("rate")
    transition_ax.set_ylim(0.0, 1.0)
    transition_ax.legend(loc="upper right", ncol=2, fontsize=6.2, frameon=False)
    transition_ax.set_xlim(-0.35, len(labels) - 0.65)
    transition_ax.grid(color=GRID_COLOR)
    transition_ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.012,
        (
            f"{_short_scope(c5['meta']['activation_scope'])}; "
            f"{c5['meta']['native_gated']['deceptive_population']} deceptive rows in native arm; "
            f"{c5['meta']['heldout_family_count']} held-out families; prospectively specified, "
            "nonconfirmatory.\nFour family effects shown; aggregate family-resampling interval "
            "is descriptive. Native fires on 2/48 D; early window on 47/48 D and 37/48 H."
        ),
        transform=fig.transFigure,
        fontsize=6.4,
        color=INK_SOFT,
        ha="center",
    )

    path = out_dir / FIGURE_NAMES[3]
    fig.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.17, hspace=0.50)
    return path, _save(fig, path)


def fig_gauge_control_null(data: dict[str, Any], out_dir: Path) -> tuple[Path, int]:
    c13 = data["c13"]
    fig, (null_ax, trans_ax) = plt.subplots(2, 1, figsize=(FIG_W, 7.2), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("One-step four-layer gauge replay was behaviorally null", fontsize=12, fontweight="bold")

    null = c13["all_roots"]
    all_labels = [
        "vs no control",
        "vs sign flip",
        "vs random tangent",
    ]
    all_points = [
        null["gauge_geodesic_minus_no_intervention"]["point"],
        null["gauge_geodesic_minus_sign_flipped"]["point"],
        null["gauge_geodesic_minus_random_tangent"]["point"],
    ]
    all_ci = [
        null["gauge_geodesic_minus_no_intervention"]["ci95"],
        null["gauge_geodesic_minus_sign_flipped"]["ci95"],
        null["gauge_geodesic_minus_random_tangent"]["ci95"],
    ]
    all_x = np.arange(len(all_labels))
    null_err = np.array(
        [
            _errbar({"ci95": ci}, labels=[name], point=point)
            for ci, point, name in zip(all_ci, all_points, all_labels)
        ],
        dtype=float,
    ).T

    null_ax.set_title("(a) Gauge intervention contrasts", loc="left")
    null_ax.bar(all_x, all_points, color=BAR_COLOR, alpha=0.85)
    null_ax.errorbar(all_x, all_points, yerr=null_err, fmt="none", color=INK, capsize=3.5)
    null_ax.set_xticks(all_x)
    null_ax.set_xticklabels(all_labels, rotation=12, ha="right")
    null_ax.axhline(0.0, color=INK_SOFT, lw=1.0)
    null_ax.set_ylabel("deceptive-probability difference")
    null_ax.text(
        0.0,
        -0.46,
        (
            "L12/L16/L19/L20 structured-action replay: "
            f"{c13['proposal_status_counts']['active']}/402 roots active; "
            f"{c13['proposal_status_counts']['boundary_exit']} boundary exits, "
            f"{c13['proposal_status_counts']['field_undefined']} undefined, "
            f"{c13['proposal_status_counts']['off_support']} off-support, "
            f"{c13['proposal_status_counts']['zero_direction']} zero-direction.\n"
            "Gauge−no-control was zero overall and on active roots. "
            f"Holonomy: {c13['holonomy']['adequate_folds']}/{c13['holonomy']['folds']} folds adequate."
        ),
        transform=null_ax.transAxes,
        fontsize=6.9,
        color=INK_SOFT,
        ha="left",
    )

    trans_rows = c13["transport"]
    generic_point = trans_rows["generic_reach"]["point"]
    generic_ci = trans_rows["generic_reach"]["ci95"]
    specific_point = trans_rows["specific_after_generic"]["remainder"]
    specific_ci = trans_rows["specific_after_generic"]["remainder_ci"]
    full_point = trans_rows["full_reach"]["point"]
    full_ci = trans_rows["full_reach"]["ci95"]

    full_err = _errbar({"ci95": full_ci}, labels=["C13 full truthful push"], point=full_point)
    generic_err = _errbar({"ci95": generic_ci}, labels=["C13 generic reach"], point=generic_point)
    specific_err = _errbar({"ci95": specific_ci}, labels=["C13 specific remainder"], point=specific_point)

    generic_color = "#2f7f95"
    specific_color = "#7f4da4"
    trans_ax.set_title("(b) Transport decomposes into generic + specific remainder", loc="left")
    trans_ax.set_ylabel("deceptive-probability difference")
    trans_ax.set_xticks([0, 0.2])
    trans_ax.set_xticklabels(["generic+specific decomposition", "observed full"], ha="center")

    trans_ax.bar(0, generic_point, width=0.32, color=generic_color, label="generic reach")
    trans_ax.bar(0, specific_point, width=0.32, bottom=generic_point, color=specific_color, label="specific remainder")
    trans_ax.errorbar(0, generic_point, yerr=generic_err.reshape(2, 1), fmt="none", color=INK, capsize=3.5, lw=1)
    trans_ax.errorbar(
        0.2,
        generic_point + specific_point,
        yerr=specific_err.reshape(2, 1),
        fmt="none",
        color=INK,
        capsize=3.5,
        lw=1,
    )
    trans_ax.scatter([0.2], [full_point], s=60, color=INK, zorder=4, label="observed full reach")
    trans_ax.errorbar(
        [0.2],
        [full_point],
        yerr=full_err.reshape(2, 1),
        fmt="none",
        color=INK,
        capsize=3.5,
        lw=1.1,
    )
    trans_ax.axhline(0.0, color=INK_SOFT, lw=1.0)
    trans_ax.text(
        0.02,
        0.96,
        "Stacked decomposition: full = generic reach + specific remainder",
        transform=trans_ax.transAxes,
        fontsize=7.0,
        color=INK_SOFT,
        ha="left",
    )
    trans_ax.legend(loc="upper right", fontsize=6.6, frameon=False)
    trans_ax.text(
        0.02,
        -0.35,
        (
            f"generic={generic_point:.3f}, specific={specific_point:.3f} => full target {generic_point + specific_point:.3f}; "
            f"observed full {full_point:.3f}\n"
            f"crossed committed roots: {c13['crossed']['crossed']} / {c13['crossed']['committed']} · "
            f"sample pool: {c13['sample_pool']['flip_count']}/{c13['sample_pool']['flip_denominator']} "
            f"(defined roots {c13['sample_pool']['defined_root_count']})"
        ),
        transform=trans_ax.transAxes,
        fontsize=6.6,
        color=INK_SOFT,
        ha="left",
    )
    trans_ax.set_ylim(
        min(-0.08, generic_point + specific_point - 0.08),
        max(0.08, generic_point + specific_point + 0.08, full_point + 0.08),
    )
    trans_ax.set_xlim(-0.52, 0.62)

    for axis in (null_ax, trans_ax):
        axis.set_axisbelow(True)
        axis.grid(color=GRID_COLOR)
        axis.tick_params(color=INK_SOFT)

    path = out_dir / FIGURE_NAMES[4]
    fig.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.09, hspace=0.62)
    return path, _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, metadata={"Software": None, "Creation Date": None})
    plt.close(fig)
    return path.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help="where to write the five public figures",
    )
    args = parser.parse_args(argv)

    plt.rcParams.update(RC)
    data = parse_data()

    renderers = [
        fig_pressure_behavior_and_hazard,
        fig_decodability_timing_gap,
        fig_structured_action_control_audit,
        fig_natural_prose_control_failure,
        fig_gauge_control_null,
    ]

    outputs = [renderer(data, args.out_dir) for renderer in renderers]
    names = [path.name for path, _ in outputs]
    if sorted(names) != sorted(FIGURE_NAMES):
        die(f"figure name drift: expected {FIGURE_NAMES}, wrote {names}")

    for path, size in sorted(outputs):
        print(f"wrote {path} ({size:,} bytes)")
    print(f"all {len(outputs)} figures written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
