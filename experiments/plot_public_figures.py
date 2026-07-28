"""Render the five public-paper figures from registry metadata and receipts.

Publication-quality, visually catchy, direct-label, uncluttered two-panel figures
that tell one coherent thesis: local geometry is real and useful under the right
information budget, while universal/pre-imposed structure fails.

Color semantics:
  geometric/structural — blue or purple
  linear/margin — charcoal
  positive outcome — green
  harmful/negative — red
  uncertainty — gray
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

# =============================================================================
# COLOR PALETTE — consistent semantics
# =============================================================================
BLUE = "#4C78A8"           # geometric / structural
PURPLE = "#9D68A8"         # geometric / structural (alternative)
CHARCOAL = "#333333"       # linear / margin
GREEN = "#54A24B"          # positive outcome
RED = "#E45756"            # harmful / negative
GRAY = "#888888"           # uncertainty
LIGHT_GRAY = "#CCCCCC"     # grid / context
INK = "#2D2D2D"            # primary text
INK_SOFT = "#666666"       # secondary text

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "medium",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "axes.edgecolor": LIGHT_GRAY,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})


def die(msg: str) -> None:
    raise SystemExit(f"plot_public_figures: ERROR: {msg}")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        die(f"registry not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        die(f"receipt not found: {path}")
    with open(path) as f:
        return json.load(f)


# =============================================================================
# DATA LOADING — expose semantic records from registry and receipts
# =============================================================================

def _extract_c10_truth_aware_from_receipt(c10: dict[str, Any]) -> dict[str, Any]:
    """Extract truth-aware C10 values from the receipt JSON truth_aware_rescore section.

    Returns a dict with:
      nuisance_prior_brier, graph_brier, linear_brier,
      graph_gain_over_truth_aware, linear_gain_over_truth_aware,
      graph_gain_ci_crosses_zero, verdict, families_positive
    """
    ta = c10.get("truth_aware_rescore")
    if ta is None:
        die("C10 receipt missing truth_aware_rescore section")

    prior_brier = ta["models"]["truth_aware_prior"]["family_macro_brier"]
    graph_brier = ta["models"]["graph_local_joint_top8"]["family_macro_brier"]
    linear_brier = ta["models"]["linear_probe_registered"]["family_macro_brier"]
    graph_gain = ta["graph_gain"]["family_macro_brier_gain"]
    linear_gain = ta["models"]["linear_probe_registered"]["gain_over_truth_aware_prior"]

    ci = ta["graph_gain"]["family_cluster_bootstrap"]["percentile_95_interval"]
    ci_crosses_zero = (ci[0] < 0 and ci[1] > 0) or ci[0] == 0 or ci[1] == 0

    verdict = ta["decision"]["verdict"]
    positive = ta["graph_gain"]["families_with_positive_gain"]
    total = ta["population"]["family_count"]

    return {
        "nuisance_prior_brier": prior_brier,
        "graph_brier": graph_brier,
        "linear_brier": linear_brier,
        "graph_gain_over_truth_aware": graph_gain,
        "linear_gain_over_truth_aware": linear_gain,
        "graph_gain_ci_crosses_zero": ci_crosses_zero,
        "verdict": f"{verdict} (truth-aware)",
        "families_positive": (positive, total),
    }


def parse_data() -> dict[str, Any]:
    """Load all receipts and registry into a unified data dict.

    Returns a dict with keys: c1..c13 (receipt JSON), registry (parsed YAML),
    truth_aware_c10 (parsed strictly from registry claim boundary), and
    convenience semantic accessors.
    """
    data: dict[str, Any] = {}

    # --- receipts ---
    for claim_id, spec in RECEIPT_SPECS.items():
        path = REPO_ROOT / spec["path"]
        receipt = _read_json(path)
        if receipt.get("kind") != spec["kind"]:
            die(f"{claim_id}: kind mismatch")
        data[claim_id.lower()] = receipt

    # --- registry ---
    registry = _read_yaml(REGISTRY_PATH)
    data["registry"] = registry

    # --- extract truth-aware C10 values from C10 receipt ---
    data["truth_aware_c10"] = _extract_c10_truth_aware_from_receipt(data["c10"])

    return data


# =============================================================================
# FIGURE 1: PRESSURE → BEHAVIOR + HAZARD
# =============================================================================
def fig_pressure_behavior_and_hazard(data: dict[str, Any], out_dir: Path) -> tuple[Path, int]:
    """Pressure creates behavior and a flow-like state organization."""
    c9 = data["c9"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- Panel A: Deceptive commitment rates with Wilson CIs ----
    conditions = ["Smooth", "Late-compressed"]
    scr = c9["outcomes"]["scripted"]["arm_summary"]
    adp = c9["outcomes"]["adaptive"]["arm_summary"]

    scripted_pts = [
        scr["smooth"]["p1b_deceptive_commitment"]["point"],
        scr["latedump"]["p1b_deceptive_commitment"]["point"],
    ]
    scripted_cis = [
        [scr["smooth"]["p1b_deceptive_commitment"]["point"]
         - scr["smooth"]["p1b_deceptive_commitment"]["ci"][0],
         scr["smooth"]["p1b_deceptive_commitment"]["ci"][1]
         - scr["smooth"]["p1b_deceptive_commitment"]["point"]],
        [scr["latedump"]["p1b_deceptive_commitment"]["point"]
         - scr["latedump"]["p1b_deceptive_commitment"]["ci"][0],
         scr["latedump"]["p1b_deceptive_commitment"]["ci"][1]
         - scr["latedump"]["p1b_deceptive_commitment"]["point"]],
    ]
    adaptive_pts = [
        adp["smooth"]["p1b_deceptive_commitment"]["point"],
        adp["latedump"]["p1b_deceptive_commitment"]["point"],
    ]
    adaptive_cis = [
        [adp["smooth"]["p1b_deceptive_commitment"]["point"]
         - adp["smooth"]["p1b_deceptive_commitment"]["ci"][0],
         adp["smooth"]["p1b_deceptive_commitment"]["ci"][1]
         - adp["smooth"]["p1b_deceptive_commitment"]["point"]],
        [adp["latedump"]["p1b_deceptive_commitment"]["point"]
         - adp["latedump"]["p1b_deceptive_commitment"]["ci"][0],
         adp["latedump"]["p1b_deceptive_commitment"]["ci"][1]
         - adp["latedump"]["p1b_deceptive_commitment"]["point"]],
    ]

    x = np.arange(len(conditions))
    width = 0.32

    bars1 = ax1.bar(x - width / 2, scripted_pts, width,
                    label="Scripted", color=GRAY, alpha=0.6, edgecolor=INK_SOFT, linewidth=0.5)
    bars2 = ax1.bar(x + width / 2, adaptive_pts, width,
                    label="Adaptive", color=BLUE, alpha=0.7, edgecolor=INK_SOFT, linewidth=0.5)

    # Direct percent labels positioned above CI whiskers
    for i, (bar, pt, cis_arr) in enumerate(zip(bars1, scripted_pts, scripted_cis)):
        y_top = pt + cis_arr[1] + 0.04
        ax1.text(bar.get_x() + bar.get_width() / 2, y_top,
                 f"{pt:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for i, (bar, pt, cis_arr) in enumerate(zip(bars2, adaptive_pts, adaptive_cis)):
        y_top = pt + cis_arr[1] + 0.04
        ax1.text(bar.get_x() + bar.get_width() / 2, y_top,
                 f"{pt:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax1.set_ylabel("Deceptive commitment rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions)
    ax1.set_ylim(0, 1.18)
    ax1.legend(loc="upper left", frameon=False, fontsize=9)
    ax1.set_title("(a) Smooth pressure → more deceptive commitment",
                  loc="left", pad=8, fontweight="bold")

    # ---- Panel B: Hazard coefficients ----
    coef = c9["hazard"]["adaptive_bank"]["adaptive_coefficients"]
    coef_names = ["Current\npressure (α)", "Accumulated\nhistory (γ)"]
    alpha_pt = coef["alpha"]["point"]
    alpha_lo = coef["alpha"]["lo"]
    alpha_hi = coef["alpha"]["hi"]
    gamma_pt = coef["gamma"]["point"]
    gamma_lo = coef["gamma"]["lo"]
    gamma_hi = coef["gamma"]["hi"]

    pts = [alpha_pt, gamma_pt]
    errs = [[alpha_pt - alpha_lo, alpha_hi - alpha_pt],
            [gamma_pt - gamma_lo, gamma_hi - gamma_pt]]

    x2 = np.arange(len(coef_names))
    colors_b = [BLUE, GRAY]

    ax2.bar(x2, pts, color=colors_b, alpha=0.8, width=0.45,
            edgecolor=INK_SOFT, linewidth=0.5)
    ax2.errorbar(x2, pts, yerr=np.array(errs).T,
                 fmt="none", color=INK, capsize=5, capthick=1.2, elinewidth=1.2)
    ax2.axhline(0, color=INK, linewidth=0.8, linestyle="--", alpha=0.4)
    ax2.set_ylabel("Hazard coefficient")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(coef_names)
    ax2.set_ylim(-0.12, 0.92)
    ax2.set_title("(b) Current pressure tracks commitment hazard",
                  loc="left", pad=8, fontweight="bold")

    # Direct labels above CI whiskers — NO p-value, NO n.s.
    ax2.text(0, alpha_hi + 0.055, f"{alpha_pt:.3f}",
             ha="center", va="bottom", fontsize=11, fontweight="bold", color=BLUE)
    ax2.text(1, gamma_hi + 0.055, f"{gamma_pt:.3f}",
             ha="center", va="bottom", fontsize=11, fontweight="bold", color=GRAY)

    # Compact note about gamma LL regression
    ll_info = c9["hazard"]["adaptive_bank"]["ll_regression"]
    ll_note = (
        f"ΔLL/event = {ll_info['mean_delta_ll_per_event']:.3f}  "
        f"CI [{ll_info['ci']['lo']:.3f}, {ll_info['ci']['hi']:.3f}]  "
        f"— not found under this instrument"
    )
    ax2.text(0.5, -0.20, ll_note, transform=ax2.transAxes,
             ha="center", va="top", fontsize=7.5, color=INK_SOFT, style="italic")

    # Descriptive flow subtitle
    flow = c9["descriptive_pressure_flow"]
    flow_note = (
        f"{flow['n_pseudo_orbits']:,} pseudo-orbits,  "
        f"median depth Spearman {flow['monotonicity']['probe_depth']['median_spearman']:.3f},  "
        f"cross-family field cosine {flow['cross_family_field_cosine']['median']:.3f}  "
        f"(descriptive, in-sample)"
    )
    fig.text(0.5, 0.01, flow_note, ha="center", va="bottom",
             fontsize=7.5, color=INK_SOFT, style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    path = out_dir / FIGURE_NAMES[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path, path.stat().st_size


# =============================================================================
# FIGURE 2: DECODABILITY TIMING GAP
# =============================================================================
def fig_decodability_timing_gap(data: dict[str, Any], out_dir: Path) -> tuple[Path, int]:
    """After commitment is linear; before commitment remains unresolved."""
    c10 = data["c10"]
    c11 = data["c11"]
    truth_aware = data["truth_aware_c10"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- Panel A: Post-commitment Brier scores (lower = better) ----
    methods = ["Truth-aware\nnuisance prior", "Relational\ngraph", "Linear\nprobe"]
    briers = [
        truth_aware["nuisance_prior_brier"],
        c10["primary"]["models"]["local_joint_top8"]["family_macro_brier"],
        c10["linear_probe_comparator"]["family_macro_brier"]["registered_probe"],
    ]

    # Color: truth-aware prior = charcoal (linear/margin),
    #        graph = blue (geometric), linear probe = green (positive outcome)
    colors_a = [CHARCOAL, BLUE, GREEN]
    bars = ax1.bar(methods, briers, color=colors_a, alpha=0.8, width=0.55,
                   edgecolor=INK_SOFT, linewidth=0.5)

    # Direct value labels
    for bar, val in zip(bars, briers):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                 f"{val:.5f}", ha="center", va="bottom", fontsize=10)

    ax1.set_ylabel("Brier score (lower = better)")
    ax1.set_ylim(0, 0.040)
    ax1.set_title("(a) After commitment: the retained signal is linear",
                  loc="left", pad=8, fontweight="bold")

    # Note about graph gain CI crossing zero (truth-aware context)
    ax1.text(0.5, -0.18, "Graph gain over truth-aware prior CI crosses zero\n"
             "Linear probe retains gain",
             transform=ax1.transAxes, ha="center", va="top",
             fontsize=7.5, color=INK_SOFT, style="italic")

    # ---- Panel B: Pre-action geometry-only log-loss gain ----
    risk = c11["risk_gate_repair"]
    geo_gain = risk["interpretation"]["primary_geometry_only_log_loss_gain_over_nuisance"]
    geo_ci = risk["interpretation"]["primary_geometry_only_log_loss_gain_ci"]
    ci_lo, ci_hi = geo_ci["interval"][0], geo_ci["interval"][1]

    ax2.bar(["Geometry-only\nlog-loss gain\nvs sealed nuisance"],
            [geo_gain], color=GRAY, alpha=0.7, width=0.4,
            edgecolor=INK_SOFT, linewidth=0.5)
    ax2.errorbar([0], [geo_gain],
                 yerr=[[geo_gain - ci_lo], [ci_hi - geo_gain]],
                 fmt="none", color=INK, capsize=6, capthick=1.2, elinewidth=1.2)
    ax2.axhline(0, color=INK, linewidth=1.0, linestyle="-", alpha=0.7)

    # Direct label above CI
    ax2.text(0, ci_hi + 0.004, f"{geo_gain:.5f}",
             ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.text(0, ci_hi + 0.012,
             f"CI [{ci_lo:.5f}, {ci_hi:.5f}]",
             ha="center", va="bottom", fontsize=8, color=INK_SOFT)

    ax2.set_ylabel("Log-loss gain")
    ax2.set_ylim(-0.09, 0.04)
    ax2.set_title("(b) Pre-action: unresolved",
                  loc="left", pad=8, fontweight="bold")

    # Three-way language
    ax2.text(0.5, -0.18, "not found under this instrument",
             transform=ax2.transAxes, ha="center", va="top",
             fontsize=8, color=INK_SOFT, style="italic")

    fig.tight_layout()

    path = out_dir / FIGURE_NAMES[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path, path.stat().st_size


# =============================================================================
# FIGURE 3: STRUCTURED ACTION CONTROL AUDIT
# =============================================================================
def fig_structured_action_control_audit(data: dict[str, Any], out_dir: Path) -> tuple[Path, int]:
    """Information budget changes what works."""
    c1 = data["c1"]
    dse = c1["descriptive_structural_evidence"]
    total = 600

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Panel A: Horizontal bar chart sorted by fixes ----
    labels_map = {
        "Route-feature ridge": c1["policies"]["learned_context_ridge_reward"]["summary"]["fixes_error"],
        "Historical route floor": c1["policies"]["historical_route_floor"]["summary"]["fixes_error"],
        "Chart-distilled RF": dse["saved_field"]["chart_distilled_context_rf"]["fixes_error"],
        "Product-Z2 RF": dse["saved_field"]["product_z2_context_rf"]["fixes_error"],
        "Graph mean": dse["saved_field"]["graph_mean_context_cauto_gauto_d12_strict"]["fixes_error"],
        "CNG": c1["policies"]["context_chart_feature_gate_equivariant_neural_context"]["summary"]["fixes_error"],
    }
    # Sort by fixes ascending (170 → 599 top-to-bottom reading order)
    sorted_items = sorted(labels_map.items(), key=lambda kv: kv[1])
    names_a = [item[0] for item in sorted_items]
    fixes_a = [item[1] for item in sorted_items]

    _color_map = {
        "Route-feature ridge": CHARCOAL,
        "Historical route floor": CHARCOAL,
        "Chart-distilled RF": BLUE,
        "Product-Z2 RF": PURPLE,
        "Graph mean": BLUE,
        "CNG": BLUE,
    }
    colors_a = [_color_map[name] for name in names_a]

    ys = list(reversed(range(len(names_a))))  # smallest bar at top
    fractions = [f / total for f in fixes_a]
    bars = ax1.barh(ys, fractions, color=colors_a, alpha=0.8,
                    height=0.65, edgecolor=INK_SOFT, linewidth=0.5)

    # Value labels just to the right of bars
    for bar, fix in zip(bars, fixes_a):
        ax1.text(bar.get_width() + 0.018, bar.get_y() + bar.get_height() / 2,
                 f"{fix}/{total}", ha="left", va="center",
                 fontsize=9, fontweight="bold")

    ax1.set_xlabel("Fraction deceptive rows fixed")
    ax1.set_yticks(ys)
    ax1.set_yticklabels(names_a, fontsize=8.5)
    ax1.set_xlim(0, 1.18)
    ax1.set_title("(a) Oracle-route-conditioned saved candidates\n(retrospective)",
                  loc="left", pad=8, fontweight="bold")

    # ---- Panel B: Horizontal paired-dot/slope comparisons ----
    # Extract values from receipt (not hardcoded)
    lp = dse["locality_proxy"]["policy_summaries"]
    local_flow = lp["local_control_flow_context"]["fixes_error"]   # 275
    global_flow = lp["global_control_flow_context"]["fixes_error"] # 52

    fa = dse["fresh_equivariant_atlas"]["policy_summaries"]
    atlas_context = fa["atlas_context_local_k21_strict"]["deceptive_strict_fixes"]  # 71
    route_floor = fa["route_hybrid_mean_probe"]["deceptive_strict_fixes"]           # 64
    atlas_response = fa["atlas_response_local_k5_strict"]["deceptive_strict_fixes"] # 79
    margin_argmax = fa["margin_argmax_all"]["deceptive_strict_fixes"]               # 79

    comparisons = [
        # (y_label, comparator_val, struct_val, denom,
        #  comp_text_label, struct_text_label, note)
        ("Locality proxy\n(global vs local flow)",
         global_flow, local_flow, 600,
         "Global flow", "Local flow", None),
        ("Fresh response-free\n(route floor vs atlas)",
         route_floor, atlas_context, 100,
         "Route floor", "Atlas", "CI touches zero"),
        ("Response-aware\n(margin argmax vs atlas)",
         margin_argmax, atlas_response, 100,
         "Margin argmax", "Atlas",          None),
    ]

    n_rows = len(comparisons)
    y_positions = list(reversed(range(n_rows)))  # [2, 1, 0] so top-to-bottom

    # Custom legend handles
    legend_elements = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=CHARCOAL,
                   markersize=8, label="Linear / margin"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE,
                   markersize=8, label="Geometric / structural"),
    ]

    for y, (y_label, comp_val, struct_val, denom,
            comp_label, struct_label, note) in zip(y_positions, comparisons):
        comp_frac = comp_val / denom
        struct_frac = struct_val / denom

        # Connecting slope line
        ax2.plot([comp_frac, struct_frac], [y, y],
                 color=LIGHT_GRAY, linewidth=2.5, zorder=1, solid_capstyle="round")

        # Gray square marker (comparator / linear)
        ax2.scatter(comp_frac, y,
                    color=CHARCOAL, s=100, marker="s", zorder=3,
                    edgecolors="white", linewidth=0.5)

        # Blue circle marker (structural), offset if tie to avoid overlap
        is_tie = abs(comp_frac - struct_frac) < 1e-9
        offset = 0.18 if is_tie else 0.0
        ax2.scatter(struct_frac, y + offset,
                    color=BLUE, s=100, marker="o", zorder=3,
                    edgecolors="white", linewidth=0.5)

        # Denominator labels beside points — avoid collision
        # Comparator label left of marker, structural label right of marker
        ax2.text(comp_frac - 0.015, y - 0.22,
                 f"{comp_label}\n{comp_val}/{denom}",
                 ha="right", va="top", fontsize=7.5, color=INK_SOFT,
                 linespacing=1.2)
        ax2.text(struct_frac + 0.015, y + offset + 0.22,
                 f"{struct_label}\n{struct_val}/{denom}",
                 ha="left", va="bottom", fontsize=7.5,
                 fontweight="bold", color=BLUE, linespacing=1.2)

        # Note annotation beside the structural marker
        if note:
            ax2.text(struct_frac + 0.015, y + offset - 0.12,
                     note, ha="left", va="top",
                     fontsize=7, color=INK_SOFT, style="italic")

    # Y-axis labels
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([c[0] for c in comparisons], fontsize=8.5, linespacing=1.3)

    # X-axis
    ax2.set_xlabel("Fraction fixed")
    ax2.set_xlim(0, 0.92)
    ax2.set_ylim(-0.6, n_rows - 0.4)

    # Legend inside upper-right whitespace
    ax2.legend(handles=legend_elements, loc="upper right",
               frameon=True, framealpha=0.85, edgecolor=LIGHT_GRAY,
               fontsize=7, borderaxespad=0.3, handletextpad=0.5)

    # Route diagnostic as shaded callout inside lower-left whitespace
    gate_diag = dse["gate_l20_routing_diagnostic"]
    badge = (
        f"Route gate: {gate_diag['n']:,}/{gate_diag['n']:,}\n"
        f"held-out-family predictions\n"
        f"(routing only; same-distribution CV)"
    )
    ax2.set_title("(b) Paired comparisons", loc="left", pad=8, fontweight="bold")
    ax2.text(0.02, 0.03, badge,
             transform=ax2.transAxes,
             ha="left", va="bottom",
             fontsize=6.5, color=INK,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0",
                       edgecolor=LIGHT_GRAY, alpha=0.9))

    fig.tight_layout()

    path = out_dir / FIGURE_NAMES[2]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path, path.stat().st_size


# =============================================================================
# FIGURE 4: NATURAL PROSE CONTROL FAILURE
# =============================================================================
def fig_natural_prose_control_failure(data: dict[str, Any], out_dir: Path) -> tuple[Path, int]:
    """Prospective geometric controller failed; linear moved both populations."""
    c5 = data["c5"]
    arms_data = c5["evaluation"]["arms"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Panel A: Net deceptive status delta with CIs ----
    arm_keys = ["native_gated", "frequent_early_window", "family_matched_linear"]
    arm_labels = ["Native\ngated", "Frequent\nearly-window", "Family-matched\nlinear"]

    effects = []
    cis_a = []
    for key in arm_keys:
        ms = arms_data[key]["machine_status"]
        pt = ms["deceptive_delta_vs_base"]["point"]
        lo, hi = ms["deceptive_delta_vs_base"]["ci95"]
        effects.append(pt)
        cis_a.append([pt - lo, hi - pt])

    colors_a = [RED, RED, GREEN]
    x_a = np.arange(len(arm_keys))

    bars = ax1.bar(x_a, effects, color=colors_a, alpha=0.8, width=0.55,
                   edgecolor=INK_SOFT, linewidth=0.5)
    ax1.errorbar(x_a, effects, yerr=np.array(cis_a).T,
                 fmt="none", color=INK, capsize=6, capthick=1.2, elinewidth=1.2)
    ax1.axhline(0, color=INK, linewidth=1.0, linestyle="--", alpha=0.5)

    # Direct labels above CI
    for i, (bar, val, ci_arr) in enumerate(zip(bars, effects, cis_a)):
        y_pos = val + ci_arr[1] + 0.04
        ax1.text(bar.get_x() + bar.get_width() / 2, y_pos,
                 f"{val:+.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    ax1.set_ylabel("Δ deceptive status correction")
    ax1.set_xticks(x_a)
    ax1.set_xticklabels(arm_labels)
    ax1.set_ylim(-0.35, 0.65)
    ax1.set_title("(a) Prospective geometric controller FAILED",
                  loc="left", pad=8, fontweight="bold")

    # ---- Panel B: Transition counts ----
    nat = arms_data["native_gated"]["machine_status"]
    freq = arms_data["frequent_early_window"]["machine_status"]
    lin = arms_data["family_matched_linear"]["machine_status"]

    trans_data = {
        "Native gated": {
            "Fixes (dec.)": nat["deceptive_fixes"],
            "Harms (dec.)": nat["deceptive_harms"],
            "Fixes (hon.)": nat["honest_fixes"],
            "Harms (hon.)": nat["honest_harms"],
        },
        "Frequent EW": {
            "Fixes (dec.)": freq["deceptive_fixes"],
            "Harms (dec.)": freq["deceptive_harms"],
            "Fixes (hon.)": freq["honest_fixes"],
            "Harms (hon.)": freq["honest_harms"],
        },
        "Linear": {
            "Fixes (dec.)": lin["deceptive_fixes"],
            "Harms (dec.)": lin["deceptive_harms"],
            "Fixes (hon.)": lin["honest_fixes"],
            "Harms (hon.)": lin["honest_harms"],
        },
    }

    categories = list(trans_data.keys())
    subcats = ["Fixes (dec.)", "Harms (dec.)", "Fixes (hon.)", "Harms (hon.)"]
    subcat_colors = [GREEN, RED, GREEN, RED]
    subcat_alphas = [0.9, 0.8, 0.6, 0.5]

    x_b = np.arange(len(categories))
    n_sub = len(subcats)
    bar_width = 0.18

    for j, (sub, col, al) in enumerate(zip(subcats, subcat_colors, subcat_alphas)):
        vals = [trans_data[cat][sub] for cat in categories]
        offset = (j - (n_sub - 1) / 2) * bar_width
        bars_b = ax2.bar(x_b + offset, vals, bar_width,
                         label=sub, color=col, alpha=al,
                         edgecolor=INK_SOFT, linewidth=0.3)
        for bar, val in zip(bars_b, vals):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.4,
                         str(val), ha="center", va="bottom",
                         fontsize=7.5, fontweight="bold")

    ax2.set_ylabel("Transition count")
    ax2.set_xticks(x_b)
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 30)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
               frameon=False, fontsize=7)
    ax2.set_title("(b) Linear: 21 fixes / 5 harms (dec.), 26 / 6 (hon.)\n"
                  "changed both populations",
                  loc="left", pad=8, fontweight="bold")

    fig.tight_layout()

    path = out_dir / FIGURE_NAMES[3]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path, path.stat().st_size


# =============================================================================
# FIGURE 5: GAUGE CONTROL NULL
# =============================================================================
def fig_gauge_control_null(data: dict[str, Any], out_dir: Path) -> tuple[Path, int]:
    """A tested geometric prior was behaviorally null."""
    c13 = data["c13"]
    replay = c13["causal_replay"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- Panel A: Gauge replay contrasts ----
    all_roots = replay["contrasts"]["all_roots"]
    contrasts_info = [
        ("vs No\ncontrol", "gauge_geodesic_minus_no_intervention"),
        ("vs Sign\nflip", "gauge_geodesic_minus_sign_flipped"),
        ("vs Random\ntangent", "gauge_geodesic_minus_random_tangent"),
    ]

    contrast_labels = [c[0] for c in contrasts_info]
    vals = []
    cis_arr = []
    for _, key in contrasts_info:
        entry = all_roots[key]
        pt = entry["point"]
        lo, hi = entry["ci95"]
        vals.append(pt)
        cis_arr.append([pt - lo, hi - pt])

    x_a = np.arange(len(contrast_labels))
    ax1.bar(x_a, vals, color=GRAY, alpha=0.6, width=0.5,
            edgecolor=INK_SOFT, linewidth=0.5)
    ax1.errorbar(x_a, vals, yerr=np.array(cis_arr).T,
                 fmt="none", color=INK, capsize=5, capthick=1.2, elinewidth=1.2)
    ax1.axhline(0, color=INK, linewidth=1.2, linestyle="-")

    # Explicit zero marker for exact zero
    for i, val in enumerate(vals):
        if val == 0.0:
            ax1.plot(i, 0, "o", color=INK, markersize=8, zorder=5)
        y_pos = val + cis_arr[i][1] + 0.0006
        ax1.text(i, y_pos, f"{val:.4f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_ylabel("Δ deceptive probability")
    ax1.set_xticks(x_a)
    ax1.set_xticklabels(contrast_labels)
    ax1.set_ylim(-0.006, 0.006)
    ax1.set_title("(a) Gauge replay: no detectable difference\n"
                  "(all CIs cross zero)",
                  loc="left", pad=8, fontweight="bold")

    # ---- Panel B: Proposal support counts ----
    status = replay["proposal_status_counts"]
    cats_b = ["Active", "Boundary\nexit", "Undefined", "Off-support", "Zero\ndirection"]
    counts_b = [
        status["active"],
        status["boundary_exit"],
        status["field_undefined"],
        status["off_support"],
        status["zero_direction"],
    ]

    colors_b = [GREEN, GRAY, GRAY, GRAY, GRAY]
    x_b = np.arange(len(cats_b))
    bars_b = ax2.bar(x_b, counts_b, color=colors_b, alpha=0.8, width=0.6,
                     edgecolor=INK_SOFT, linewidth=0.5)

    for bar, count in zip(bars_b, counts_b):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 str(count), ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    ax2.set_ylabel("Number of roots")
    ax2.set_xticks(x_b)
    ax2.set_xticklabels(cats_b, fontsize=8)
    ax2.set_ylim(0, 370)
    ax2.set_title("(b) 21 / 402 active roots",
                  loc="left", pad=8, fontweight="bold")

    fig.tight_layout()

    path = out_dir / FIGURE_NAMES[4]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path, path.stat().st_size


# =============================================================================
# MAIN
# =============================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIG_DIR)
    args = parser.parse_args(argv)

    data = parse_data()

    renderers = [
        fig_pressure_behavior_and_hazard,
        fig_decodability_timing_gap,
        fig_structured_action_control_audit,
        fig_natural_prose_control_failure,
        fig_gauge_control_null,
    ]

    outputs = [renderer(data, args.out_dir) for renderer in renderers]

    for path, size in sorted(outputs):
        print(f"wrote {path} ({size:,} bytes)")
    print(f"all {len(outputs)} figures written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
