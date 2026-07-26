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
    learned_context = _extract("learned_context_ridge_reward")
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
            "learned_context": {
                "fix_rate": _ratio(
                    learned_context["fixes_error"],
                    learned_context["deceptive_n"],
                    "C1 learned context-only fix rate",
                ),
                "harm_rate": _ratio(
                    learned_context["honest_harms"],
                    learned_context["honest_n"],
                    "C1 learned context-only harm rate",
                ),
                "fixes": learned_context["fixes_error"],
                "harms": learned_context["honest_harms"],
                "deceptive_n": learned_context["deceptive_n"],
                "honest_n": learned_context["honest_n"],
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
        status = arm_payload.get("machine_status", {}).get("deceptive_delta_vs_base", {})
        assert_condition(isinstance(status, dict), f"C5 missing machine-status delta for {arm}")
        assert_condition("point" in status, f"C5 missing point for {arm}")
        result[arm] = status
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
    points = [_point(row, f"C9 {name}") for row, name in zip(rows, labels)]
    err = np.array(
        [_errbar(row, labels=[f"C9 {name}"]) for row, name in zip(rows, labels)],
        dtype=float,
    ).T
    x = np.arange(len(labels))
    top.bar(x, points, color=BAR_COLOR, alpha=0.85)
    top.errorbar(x, points, yerr=err, fmt="none", color=INK, capsize=3)
    top.set_title("Pressure behavior", loc="left")
    top.set_xticks(x)
    top.set_xticklabels(labels, rotation=10, ha="right")
    top.set_ylabel("P1b deceptive commitment rate")
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
    bottom.bar(hazard_x, hazard_points, color=CONTEXT, alpha=0.85)
    bottom.errorbar(hazard_x, hazard_points, yerr=hazard_err, fmt="none", color=INK, capsize=3)
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
    dec_ax.bar(dec_x, dec_points, color=BAR_COLOR, alpha=0.9)
    dec_ax.set_xticks(dec_x)
    dec_ax.set_xticklabels(["Exact nuisance", "Relational graph", "Linear probe"], rotation=12, ha="right")
    dec_ax.set_ylabel("Family-macro Brier ↓")
    dec_ax.set_title("(a) Post-commitment prediction", loc="left")
    for x_i, value in zip(dec_x, dec_points):
        dec_ax.text(x_i, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=7.2)

    null_labels = ["Observed\ngraph gain", "Permutation\nnull mean", "Excess over\nnull mean"]
    null_points = [c10["null"]["observed_gain"], c10["null"]["mean"], c10["null"]["excess"]]
    null_x = np.arange(len(null_labels))
    null_ax.bar(null_x, null_points, color=[BAR_COLOR, CONTEXT, THRESHOLD], alpha=0.88)
    null_ax.set_xticks(null_x)
    null_ax.set_xticklabels(null_labels)
    null_ax.set_ylabel("Brier gain over nuisance ↑")
    null_ax.set_title("(b) Nuisance-preserving null", loc="left")
    null_ax.axhline(0.0, color=INK_SOFT, lw=1)
    for x_i, value in zip(null_x, null_points):
        null_ax.text(x_i, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=7.2)

    warn_auroc_labels = ["Spectral AUROC", "Connection AUROC"]
    warn_auroc_points = [c11["spectral_auroc"], c11["connection_auroc"]]
    warn_auroc_x = np.arange(len(warn_auroc_labels) + 1)
    warn_auroc_points = np.append(warn_auroc_points, 0.5)
    auroc_ax.bar(
        warn_auroc_x,
        warn_auroc_points,
        color=[BAR_COLOR, CONTEXT, INK_SOFT],
        alpha=0.9,
    )
    auroc_ax.set_xticks(warn_auroc_x)
    auroc_ax.set_xticklabels(["Spectral field", "Path connection", "Chance"])
    auroc_ax.set_title("(c) Pre-action warning channels", loc="left")
    auroc_ax.set_ylabel("AUROC ↑")
    auroc_ax.set_ylim(0.25, 0.62)
    auroc_ax.axhline(0.5, color=INK_SOFT, ls="--", lw=1)
    for x_i, value in zip(warn_auroc_x, warn_auroc_points):
        auroc_ax.text(x_i, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=7.2)

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
    gain_ax.errorbar(warn_gain_x, warn_gain_points, yerr=warn_gain_err, fmt="none", color=INK, capsize=3)
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
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("Structured-action control audit", fontsize=13, fontweight="bold", y=0.98)

    c1_ax = axes[0]
    c1_order = ("learned_context", "cng_oracle_route", "route_matched")
    c1_labels = [
        "Learned\ncontext-only",
        "CNG\noracle route",
        "Fixed L16\noracle route",
    ]
    c1_x = np.arange(len(c1_order))
    c1_fix = [c1["policies"][name]["fix_rate"] for name in c1_order]
    c1_harm = [c1["policies"][name]["harm_rate"] for name in c1_order]
    c1_ax.bar(c1_x - 0.16, c1_fix, width=0.32, color=BAR_COLOR, label="deceptive fix")
    c1_ax.bar(c1_x + 0.16, c1_harm, width=0.32, color=CONTEXT, label="honest harm")
    c1_ax.set_title("(a) Joint target/action selection unresolved", fontsize=8.6, loc="left")
    c1_ax.set_xticks(c1_x)
    c1_ax.set_xticklabels(c1_labels)
    c1_ax.set_ylim(0.0, 1.0)
    c1_ax.set_ylabel("rate")
    c1_ax.legend(loc="upper right", fontsize=7.0)
    c1_ax.text(0.02, 0.77, "170/600 fixes\n11/600 harms", transform=c1_ax.transAxes, fontsize=7.0)
    c1_ax.text(0.51, 0.77, "599/600 fixes\n1/600 harms", transform=c1_ax.transAxes, fontsize=7.0)
    c1_ax.text(0.74, 0.67, "600/600 fixes\n0/600 harms", transform=c1_ax.transAxes, fontsize=7.0)

    c2_ax = axes[1]
    c2_x = np.arange(2)
    c2_fix = [c2["rates"]["fixed_fix_rate"], c2["rates"]["dense_fix_rate"]]
    c2_harm = [c2["rates"]["fixed_harm_rate"], c2["rates"]["dense_harm_rate"]]
    c2_ax.bar(c2_x - 0.16, c2_fix, width=0.32, color=BAR_COLOR, label="deceptive fix")
    c2_ax.bar(c2_x + 0.16, c2_harm, width=0.32, color=CONTEXT, label="honest harm")
    c2_ax.set_title("(b) Dense dose adds no benefit", fontsize=9.0, loc="left")
    c2_ax.set_xticks(c2_x)
    c2_ax.set_xticklabels(["Fixed", "Dense"])
    c2_ax.set_ylim(0.0, 1.0)
    c2_ax.set_ylabel("rate")

    c12_ax = axes[2]
    c12_order = ("baseline", "bidir_linear", "bidir_tangent", "global_mean_gated", "global_probe_gated", "random_gated")
    c12_fix = [c12["policies"][name]["fix_rate"] for name in c12_order]
    c12_x = np.arange(len(c12_order))
    c12_ax.bar(c12_x, c12_fix, color=BAR_COLOR, alpha=0.9)
    c12_ax.set_title("(c) Pilot: route and dose carry the effect", fontsize=9.0, loc="left")
    c12_ax.set_xticks(c12_x)
    c12_ax.set_xticklabels(
        ["No control", "Fixed linear", "Tangent", "Global mean", "Global probe", "Random"],
        rotation=32,
        ha="right",
    )
    c12_ax.set_ylim(0.0, 1.0)
    c12_ax.set_ylabel("rate")
    c12_ax.text(
        0.02,
        0.89,
        "Honest status harms: 2/80 all arms\nHonest strict harms: 10/80 all arms",
        transform=c12_ax.transAxes,
        fontsize=6.8,
    )

    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(color=GRID_COLOR)
        ax.tick_params(color=INK_SOFT)

    path = out_dir / FIGURE_NAMES[2]
    handles, labels = c1_ax.get_legend_handles_labels()
    c1_ax.get_legend().remove()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False)
    fig.text(
        0.5,
        0.02,
        (
            "C1/C2 use 600 deceptive and 600 honest rows. C12 is an 80+80-row mixed-precision pilot.\n"
            f"Its off-tangent follow-up uses 16 deceptive pairs (9 vs 1): difference "
            f"{c12['followup']['paired_difference']:.3f} "
            f"[{c12['followup']['paired_ci'][0]:.3f}, {c12['followup']['paired_ci'][1]:.3f}]."
        ),
        ha="center",
        fontsize=7.1,
        color=INK_SOFT,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.31, wspace=0.32)
    return path, _save(fig, path)


def fig_natural_prose_control_failure(data: dict[str, Any], out_dir: Path) -> tuple[Path, int]:
    c5 = data["c5"]
    claim5 = data["claims"]["C5"]
    fig, ax = plt.subplots(figsize=(FIG_W, 4.8), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)

    labels = ["native gated", "frequent early window", "family-matched linear"]
    points = [_point(c5["arms"]["native_gated"], "C5 native"), _point(c5["arms"]["frequent_early_window"], "C5 frequent"), _point(c5["arms"]["family_matched_linear"], "C5 linear")]
    ci = [_interval(c5["arms"]["native_gated"], "C5 native"), _interval(c5["arms"]["frequent_early_window"], "C5 frequent"), _interval(c5["arms"]["family_matched_linear"], "C5 linear")]
    los = [i[0] for i in ci]
    his = [i[1] for i in ci]
    errs = np.array([np.array(points) - np.array(los), np.array(his) - np.array(points)])
    x = np.arange(len(labels))

    ax.bar(x, points, color=BAR_COLOR, alpha=0.85)
    ax.errorbar(x, points, yerr=errs, fmt="none", color=INK, capsize=3.5)
    ax.axhline(0.0, color=INK_SOFT, lw=1.0, ls="--")
    ax.set_title(f"C5 status: {claim5['status']} · {claim5['registration_tier']}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Δ deceptive status correction (machine score)")
    ax.set_title("C5: natural-prose prospective controller failure")
    ax.text(
        0.0,
        -0.34,
        (
            f"{_short_scope(c5['meta']['activation_scope'])}; "
            f"{c5['meta']['native_gated']['deceptive_population']} deceptive rows; "
            f"{c5['meta']['heldout_family_count']} held-out families; "
            "95% family-cluster CI"
        ),
        transform=ax.transAxes,
        fontsize=7.0,
        color=INK_SOFT,
        ha="left",
    )

    ax.grid(color=GRID_COLOR)
    ax.set_axisbelow(True)
    path = out_dir / FIGURE_NAMES[3]
    fig.tight_layout()
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
    trans_labels = ["Full truthful push", "Generic reach", "Specific remainder"]
    trans_points = [
        trans_rows["full_reach"]["point"],
        trans_rows["generic_reach"]["point"],
        trans_rows["specific_after_generic"]["remainder"],
    ]
    trans_ci = [
        trans_rows["full_reach"]["ci95"],
        trans_rows["generic_reach"]["ci95"],
        trans_rows["specific_after_generic"]["remainder_ci"],
    ]
    trans_x = np.arange(len(trans_labels))
    trans_err = np.array(
        [
            _errbar({"ci95": ci}, labels=[name], point=point)
            for ci, point, name in zip(trans_ci, trans_points, trans_labels)
        ],
        dtype=float,
    ).T
    trans_ax.set_title("(b) Generic reach explains the transport", loc="left")
    trans_ax.bar(trans_x, trans_points, color=CONTEXT, alpha=0.9)
    trans_ax.errorbar(trans_x, trans_points, yerr=trans_err, fmt="none", color=INK, capsize=3.5)
    trans_ax.set_xticks(trans_x)
    trans_ax.set_xticklabels(trans_labels, rotation=12, ha="right")
    trans_ax.set_ylabel("push / remainder")
    trans_ax.axhline(0.0, color=INK_SOFT, lw=1.0)
    trans_ax.text(
        0.0,
        -0.35,
        (
            f"crossed committed roots: {c13['crossed']['crossed']} / {c13['crossed']['committed']} · "
            f"sample pool: {c13['sample_pool']['flip_count']}/{c13['sample_pool']['flip_denominator']} "
            f"(defined roots {c13['sample_pool']['defined_root_count']})"
        ),
        transform=trans_ax.transAxes,
        fontsize=6.8,
        color=INK_SOFT,
        ha="left",
    )

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
