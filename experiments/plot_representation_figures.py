"""Render the three representation figures from the committed public receipt.

Ledger theme: serif typography matched to the manuscript body, an ultramarine/ember
duotone (CVD-validated), warm grays for baselines and nulls, panel chips, and
chronology stamp chips. Figures are designed at final print width (6.5 in) and
emitted as vector PDF for the manuscript plus PNG as a web byproduct.

  representation_reconstruction.{pdf,png} — held-out cosines + paired differences
  representation_structure.{pdf,png}      — specificity + output compression
  representation_factorization.{pdf,png}  — source-plus-action waterfall
  *_mobile.png                           — stacked web variants for narrow screens
  addressability_social_card.png         — 1200 x 630 article share card

Every scientific value in the charts and social card is parsed from
paper_artifacts/c14_representation_receipt.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Deterministic vector output: matplotlib stamps PDF CreationDate from this.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

matplotlib.use("Agg", force=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = REPO_ROOT / "paper_artifacts" / "c14_representation_receipt.json"
DEFAULT_FIG_DIR = REPO_ROOT / "docs" / "figures"
EXPECTED_KIND = "c14_representation_structure_public_receipt"

FIGURE_STEMS = [
    "representation_reconstruction",
    "representation_structure",
    "representation_factorization",
]
MOBILE_FIGURE_STEMS = [f"{stem}_mobile" for stem in FIGURE_STEMS]
SOCIAL_CARD_NAME = "addressability_social_card.png"
COMBINED_STEM = "representation_structure_factorization"
PRESSURE_STEM = "representation_pressure_behavior"
PRESSURE_RECEIPT_PATH = REPO_ROOT / "paper_artifacts" / "pressure_behavior_receipt.json"
ACTIVE_FORMATS: tuple[str, ...] = ("pdf", "png")
# Blog style (opt-in via --blog): de-framed fields and lighter labels for the
# public article; the canonical manuscript outputs stay byte-stable.
VALUE_WEIGHT = "bold"
OUTPUT_SUFFIX = ""
FIGURE_NAMES = (
    [f"{stem}.{ext}" for stem in FIGURE_STEMS for ext in ("pdf", "png")]
    + [f"{COMBINED_STEM}.{ext}" for ext in ("pdf", "png")]
    + [f"{PRESSURE_STEM}.{ext}" for ext in ("pdf", "png")]
    + [f"{stem}.png" for stem in MOBILE_FIGURE_STEMS]
    + [SOCIAL_CARD_NAME]
)

# ---------------------------------------------------------------------------
# Themes. PRINT is the manuscript identity ("Archive Indigo"): white page,
# near-black indigo protagonist, bronze comparators, mauve metadata baseline,
# monochrome archival chronology stamps. WEB is the blog rendering (warm
# surround); the PDF export must never inherit the web canvas.
# ---------------------------------------------------------------------------
THEMES = {
    "print": {
        "INK": "#33312C", "INK_SOFT": "#6E685C", "HAIR": "#E3DED3",
        "PAPER": "#FFFFFF", "WEB_PAPER": "#FFFFFF",
        "BLUE": "#5C6E6C",       # protagonist estimator (balsam green)
        "EMBER": "#D2A96A",      # retrieval/alternative, primary (artemis gold)
        "EMBER_LT": "#E8D5B0",   # retrieval/alternative, light (web only)
        "MAUVE": "#D39D87",      # truth-aware metadata baseline (dusty coral)
        "CHARCOAL": "#BB7154",   # global-mean baseline (warm copper)
        "GRAY": "#A6B7AA",       # nulls and shuffles (aquatone sage)
        "RED": "#A34A38",        # reserved: refuted only, never a series color
        "BLUE_WASH": "#EDF1EF",  # hollow-variant fill, protagonist family
        "EMBER_WASH": "#F7EDDA", # hollow-variant fill, retrieval family
        "STAMP_FC": "#F4F2EC", "STAMP_EC": "#DCD6C9", "STAMP_TC": "#6E685C",
    },
    "web": {  # the blog's Ink & Paper palette (kept in sync with the article)
        "INK": "#171814", "INK_SOFT": "#5B554C", "HAIR": "#C4B7A5",
        "PAPER": "#FFFDF8", "WEB_PAPER": "#DCD4C3",
        "BLUE": "#1F5E8C", "EMBER": "#B0722A", "EMBER_LT": "#E69A54",
        "MAUVE": "#654d82", "CHARCOAL": "#332F29", "GRAY": "#827668",
        "RED": "#B3403A",
        "BLUE_WASH": "#E4EBF3", "EMBER_WASH": "#F6E3CE",
        "STAMP_FC": "#F1EFEC", "STAMP_EC": "#C4B7A5", "STAMP_TC": "#5B554C",
        "TIER_STYLE": {
            "U": {"label": "RETROSPECTIVE UNREGISTERED DESCRIPTIVE",
                  "fc": "#F1EFEC", "ec": "#C4B7A5", "tc": "#5B554C"},
            "D": {"label": "POST-EVIDENCE REGISTERED DESCRIPTIVE",
                  "fc": "#F8E8DC", "ec": "#E5C4A8", "tc": "#8A4415"},
            "R": {"label": "REGISTERED ENDPOINT",
                  "fc": "#E4E9F8", "ec": "#BCC8EE", "tc": "#41598C"},
        },
    },
}
ACTIVE_THEME = "print"
INK = INK_SOFT = HAIR = PAPER = WEB_PAPER = ""
BLUE = EMBER = EMBER_LT = MAUVE = CHARCOAL = GRAY = RED = ""
BLUE_WASH = EMBER_WASH = ""
TIER_STYLE: dict[str, dict[str, str]] = {}


def apply_theme(name: str) -> None:
    """Bind the named theme's colors to the module globals and rcParams."""
    global INK, INK_SOFT, HAIR, PAPER, WEB_PAPER, BLUE, EMBER, EMBER_LT
    global MAUVE, CHARCOAL, GRAY, RED, TIER_STYLE, ACTIVE_THEME
    global BLUE_WASH, EMBER_WASH
    th = THEMES[name]
    ACTIVE_THEME = name
    INK, INK_SOFT, HAIR = th["INK"], th["INK_SOFT"], th["HAIR"]
    PAPER, WEB_PAPER = th["PAPER"], th["WEB_PAPER"]
    BLUE, EMBER, EMBER_LT = th["BLUE"], th["EMBER"], th["EMBER_LT"]
    MAUVE, CHARCOAL, GRAY, RED = th["MAUVE"], th["CHARCOAL"], th["GRAY"], th["RED"]
    BLUE_WASH, EMBER_WASH = th["BLUE_WASH"], th["EMBER_WASH"]
    TIER_STYLE = th.get("TIER_STYLE") or {
        key: {"label": label, "fc": th["STAMP_FC"], "ec": th["STAMP_EC"],
              "tc": th["STAMP_TC"]}
        for key, label in (
            ("U", "RETROSPECTIVE UNREGISTERED DESCRIPTIVE"),
            ("D", "POST-EVIDENCE REGISTERED DESCRIPTIVE"),
            ("R", "REGISTERED ENDPOINT"),
        )
    }
    plt.rcParams.update({
        "text.color": INK, "axes.labelcolor": INK_SOFT, "xtick.color": INK_SOFT,
        "ytick.color": INK, "axes.edgecolor": HAIR, "figure.facecolor": PAPER,
        "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    })


apply_theme("print")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.linewidth": 0.7,
    "text.color": INK,
    "axes.labelcolor": INK_SOFT,
    "xtick.color": INK_SOFT,
    "ytick.color": INK,
    "axes.edgecolor": HAIR,
    "xtick.direction": "out",
    "ytick.major.size": 0,
    "xtick.major.size": 3,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})


def die(msg: str) -> None:
    raise SystemExit(f"plot_representation_figures: ERROR: {msg}")


def load_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        die(f"receipt not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("kind") != EXPECTED_KIND:
        die(f"unexpected receipt kind: {data.get('kind')!r}")
    return data


def parse_data(receipt: dict[str, Any]) -> dict[str, Any]:
    """Extract exactly the figure-bearing values from the receipt."""
    hf = receipt["honestward_field"]
    models = hf["primary_view_models"]
    comps = hf["primary_view_comparisons"]
    sec_comps = hf["secondary_view_comparisons"]
    spec = receipt["specificity_controls"]["comparisons"]
    comp = receipt["compression_frontier"]
    cmodels = comp["models"]
    add = receipt["additive_compositional_transport"]
    end = receipt["endpoint_prototype_diagnostic"]

    sa = receipt["simple_address_baselines"]
    sa_models = sa["models"]["intervention_masked_action_free"]
    sa_comps = sa["comparisons"]["intervention_masked_action_free"]

    return {
        "bars": [
            ("Raw-activation k=8", sa_models["raw_k8"]["cosine_mean"],
             sa_models["raw_k8"]["defined_count"], sa_models["raw_k8"]["total_count"],
             EMBER),
            ("Raw-activation nearest", sa_models["raw_nn"]["cosine_mean"],
             sa_models["raw_nn"]["defined_count"], sa_models["raw_nn"]["total_count"],
             "HOLLOW_ALT"),
            ("Typed-graph local", models["local"]["cosine_mean"],
             models["local"]["defined_count"], models["local"]["total_count"], BLUE),
            ("Truth-aware design-cell mean", sa_models["design_cell_mean"]["cosine_mean"],
             sa_models["design_cell_mean"]["defined_count"],
             sa_models["design_cell_mean"]["total_count"], MAUVE),
            ("Typed-graph nearest", models["nearest"]["cosine_mean"],
             models["nearest"]["defined_count"], models["nearest"]["total_count"],
             "HOLLOW"),
            ("Global train mean", models["global_mean"]["cosine_mean"],
             models["global_mean"]["defined_count"], models["global_mean"]["total_count"],
             CHARCOAL),
            ("Cyclic target shuffle", models["shuffled"]["cosine_mean"],
             models["shuffled"]["defined_count"], models["shuffled"]["total_count"], GRAY),
        ],
        "paired": [
            ("local $-$ shuffle", comps["shuffled"]["mean_cosine_difference"],
             comps["shuffled"]["scenario_cluster_ci"], GRAY, False),
            ("local $-$ global", comps["global_mean"]["mean_cosine_difference"],
             comps["global_mean"]["scenario_cluster_ci"], CHARCOAL, False),
            ("local $-$ per-contrast oracle",
             comps["contrast_global_oracle"]["mean_cosine_difference"],
             comps["contrast_global_oracle"]["scenario_cluster_ci"], CHARCOAL, False),
            ("local $-$ nearest", comps["nearest"]["mean_cosine_difference"],
             comps["nearest"]["scenario_cluster_ci"], EMBER, False),
            ("local $-$ nearest, secondary view",
             sec_comps["nearest"]["mean_cosine_difference"],
             sec_comps["nearest"]["scenario_cluster_ci"], BLUE, True),
            ("graph local $-$ raw k=8",
             sa_comps["graph_local_minus_raw_k8"]["mean_cosine_difference"],
             sa_comps["graph_local_minus_raw_k8"]["scenario_cluster_ci"], EMBER, False),
            ("graph local $-$ design-cell mean",
             sa_comps["graph_local_minus_design_cell_mean"]["mean_cosine_difference"],
             sa_comps["graph_local_minus_design_cell_mean"]["scenario_cluster_ci"],
             MAUVE, False),
        ],
        "specificity": [
            ("honestward $-$ generic",
             spec["honestward_minus_generic"]["mean_cosine_difference"],
             spec["honestward_minus_generic"]["cosine_scenario_cluster_ci"],
             spec["honestward_minus_generic"]["mean_nse_improvement"],
             spec["honestward_minus_generic"]["nse_scenario_cluster_ci"]),
            ("honestward $-$ shuffle",
             spec["honestward_minus_nuisance_shuffle"]["mean_cosine_difference"],
             spec["honestward_minus_nuisance_shuffle"]["cosine_scenario_cluster_ci"],
             spec["honestward_minus_nuisance_shuffle"]["mean_nse_improvement"],
             spec["honestward_minus_nuisance_shuffle"]["nse_scenario_cluster_ci"]),
            ("generic $-$ shuffle",
             spec["generic_minus_nuisance_shuffle"]["mean_cosine_difference"],
             spec["generic_minus_nuisance_shuffle"]["cosine_scenario_cluster_ci"],
             spec["generic_minus_nuisance_shuffle"]["mean_nse_improvement"],
             spec["generic_minus_nuisance_shuffle"]["nse_scenario_cluster_ci"]),
        ],
        "compression": [
            ("Full local estimator", cmodels["full_exemplar_local"]["cosine_mean"],
             cmodels["full_exemplar_local"]["defined_count"],
             cmodels["full_exemplar_local"]["total_count"], BLUE),
            ("Rank-32 projection", cmodels["low_rank_projected_full"]["cosine_mean"],
             cmodels["low_rank_projected_full"]["defined_count"],
             cmodels["low_rank_projected_full"]["total_count"], "HOLLOW"),
            ("256-landmark fallback", cmodels["landmark_local"]["cosine_mean"],
             cmodels["landmark_local"]["defined_count"],
             cmodels["landmark_local"]["total_count"], EMBER),
            ("Global mean", cmodels["global_mean"]["cosine_mean"],
             cmodels["global_mean"]["defined_count"],
             cmodels["global_mean"]["total_count"], CHARCOAL),
        ],
        "rank_var_min": min(f["rank_variance_explained"] for f in comp["fold_selections"]),
        "rank_var_max": max(f["rank_variance_explained"] for f in comp["fold_selections"]),
        "action_cos": add["action_family_macro_cosine"],
        "additive_cos": add["additive_family_macro_cosine"],
        "constrained_cos": end["constrained_family_macro_cosine"],
        "endpoint_ratio": end["ratio_explained_by_endpoint"],
        "gap_free_minus_constrained": end["gap_free_minus_constrained"],
        "action_folds": [f["action_family_macro"] for f in add["per_fold"]],
        "additive_folds": [f["additive_family_macro"] for f in add["per_fold"]],
        "constrained_folds": [f["constrained_family_macro"] for f in end["per_fold"]],
    }


# ---------------------------------------------------------------------------
# Theme components
# ---------------------------------------------------------------------------

def _chip(ax: plt.Axes, letter: str, title: str, y: float = 1.0875,
          box_h: float = 0.075) -> None:
    """Plain panel label: bold serif, no box."""
    label = f"{letter}.  {title}" if letter else title
    ax.text(0.0, y, label, transform=ax.transAxes, ha="left",
            va="center", fontsize=10, color=INK, fontweight=VALUE_WEIGHT, clip_on=False)


def _tier_fig(fig: plt.Figure, tier: str, x: float = 0.995, y: float = 0.99) -> None:
    """One chronology stamp per figure, top-right of the canvas."""
    s = TIER_STYLE[tier]
    fig.text(x, y, s["label"], ha="right", va="top", fontsize=6.0, color=s["tc"],
             bbox=dict(boxstyle="square,pad=0.45", fc=s["fc"],
                       ec=s["ec"], lw=0.45))


def _tier(ax: plt.Axes, tier: str, x: float = 1.0, y: float = 1.0875) -> None:
    """Chronology stamp chip, top-right, axes-coords."""
    s = TIER_STYLE[tier]
    ax.text(x, y, s["label"], transform=ax.transAxes, ha="right", va="center",
            fontsize=6.0, color=s["tc"], clip_on=False,
            bbox=dict(boxstyle="square,pad=0.45", fc=s["fc"],
                      ec=s["ec"], lw=0.45))


def _coverage_chip(ax: plt.Axes, x: float, y: float, defined: int, total: int) -> None:
    ax.text(x, y, f"{defined}/{total}", va="center", ha="left", fontsize=7.2,
            color=PAPER, style="italic")




def _set_figure_background(fig: plt.Figure) -> None:
    """Use a warm editorial canvas while keeping each plotting field white."""
    fig.patch.set_facecolor(WEB_PAPER)
    for ax in fig.axes:
        ax.set_facecolor(PAPER)


def _save_standard(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    _set_figure_background(fig)
    for ext in ACTIVE_FORMATS:
        fig.savefig(out_dir / f"{stem}{OUTPUT_SUFFIX}.{ext}", facecolor=WEB_PAPER)


def _plot_interval_row(
    ax: plt.Axes, yi: float, point: float, ci: list[float] | tuple[float, float],
    color: str, *, hollow: bool = False, marker_size: float = 7.5,
) -> None:
    ax.plot([ci[0], ci[1]], [yi, yi], "-", color=color, lw=2.8,
            solid_capstyle="butt", zorder=3)
    for cap in ci:
        ax.plot([cap, cap], [yi - 0.13, yi + 0.13], "-", color=color, lw=1.3, zorder=3)
    ax.plot(point, yi, "o", ms=marker_size, mfc=PAPER if hollow else color,
            mec=color, mew=1.5, zorder=4)


def figure_reconstruction_mobile(data: dict[str, Any], out_dir: Path) -> None:
    """Stack the reconstruction evidence with phone-sized labels."""
    fig = plt.figure(figsize=(4.4, 10.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.18, 1.0], hspace=0.46,
                          top=0.94, bottom=0.07, left=0.34, right=0.94)
    ax = fig.add_subplot(gs[0])
    bars = data["bars"]
    y = np.arange(len(bars))[::-1].astype(float)
    y[5:] -= 0.7  # visual gap: local addresses above, baselines below
    mfills = [BLUE_WASH if b[4] == "HOLLOW" else EMBER_WASH if b[4] == "HOLLOW_ALT"
              else b[4] for b in bars]
    medges = [BLUE if b[4] == "HOLLOW" else EMBER if b[4] == "HOLLOW_ALT" else "none"
              for b in bars]
    ax.barh(y, [b[1] for b in bars], height=0.58, color=mfills, zorder=3,
            edgecolor=medges, linewidth=1.1)
    for yi, (_, val, defined, total, _) in zip(y, bars):
        ax.text(min(val + 0.018, 1.005), yi, f"{val:.3f}", va="center", ha="left",
                fontsize=13.5, color=INK, fontweight=VALUE_WEIGHT)
        ax.text(0.018, yi, f"{defined}/{total}", va="center", ha="left", fontsize=9.8,
                color=PAPER, style="italic")
    ax.set_yticks(y, [b[0] for b in bars], fontsize=11.5)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(axis="x", labelsize=10.5)
    ax.set_xlabel("held-out reconstruction cosine", fontsize=11.5)
    ax.grid(axis="x", color=HAIR, lw=0.7, zorder=0)
    _chip(ax, "A", "Local addresses reconstruct held-out displacement", y=1.075)

    ax2 = fig.add_subplot(gs[1])
    paired = data["paired"]
    y2 = np.arange(len(paired))[::-1]
    for yi, (name, point, ci, color, hollow) in zip(y2, paired):
        _plot_interval_row(ax2, yi, point, ci, color, hollow=hollow)
        label = f"+{point:.3f}" if point >= 0 else f"−{abs(point):.3f}"
        ax2.text(ci[1] + 0.017, yi, label, va="center", fontsize=11.5, color=INK)
    ax2.axvline(0.0, color=INK, lw=1.0, zorder=2)
    ax2.set_yticks(y2, [p[0] for p in paired], fontsize=10.7)
    ax2.set_xlim(-0.08, 0.69)
    ax2.set_xticks([0, 0.25, 0.5])
    ax2.tick_params(axis="x", labelsize=10.5)
    ax2.set_xlabel("paired cosine difference · 95% interval", fontsize=11.5)
    ax2.grid(axis="x", color=HAIR, lw=0.7, zorder=0)
    _chip(ax2, "B", "Retrieval, not graph machinery, carries the gain", y=1.075)
    _set_figure_background(fig)
    fig.savefig(out_dir / f"{MOBILE_FIGURE_STEMS[0]}{OUTPUT_SUFFIX}.png", dpi=220, facecolor=WEB_PAPER)
    plt.close(fig)


def figure_structure_mobile(data: dict[str, Any], out_dir: Path) -> None:
    """Stack specificity and output compression for narrow screens."""
    fig = plt.figure(figsize=(4.4, 9.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 0.9], hspace=0.48,
                          top=0.93, bottom=0.075, left=0.37, right=0.94)
    _tier_fig(fig, "U", x=0.97, y=0.985)
    ax = fig.add_subplot(gs[0])
    rows = []
    for name, cos_pt, cos_ci, nse_pt, nse_ci in data["specificity"]:
        rows.append((f"{name} (cos)", cos_pt, cos_ci, BLUE))
        rows.append((f"{name} (NSE)", nse_pt, nse_ci, GRAY))
    spacing = np.array([0.0, 1.0, 2.6, 3.6, 5.2, 6.2])
    y = spacing.max() - spacing
    for yi, (_, pt, ci, color) in zip(y, rows):
        _plot_interval_row(ax, yi, pt, ci, color)
    ax.axvline(0.0, color=INK, lw=1.0, zorder=2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=10.6)
    ax.tick_params(axis="x", labelsize=10.5)
    ax.set_xlabel("paired difference · 95% interval", fontsize=11.5)
    ax.grid(axis="x", color=HAIR, lw=0.7, zorder=0)
    _chip(ax, "A", "Specificity is small and metric-dependent", y=1.08)

    ax2 = fig.add_subplot(gs[1])
    comp = data["compression"]
    y2 = np.arange(len(comp))[::-1]
    mcfills = [BLUE_WASH if c[4] == "HOLLOW" else c[4] for c in comp]
    mcedges = [BLUE if c[4] == "HOLLOW" else "none" for c in comp]
    ax2.barh(y2, [c[1] for c in comp], height=0.58, color=mcfills, zorder=3,
             edgecolor=mcedges, linewidth=1.1)
    for yi, (_, val, defined, total, _) in zip(y2, comp):
        ax2.text(val + 0.017, yi, f"{val:.3f}", va="center", fontsize=13,
                 color=INK, fontweight=VALUE_WEIGHT)
        if defined != total:
            ax2.text(0.018, yi, f"{defined}/{total}", va="center", fontsize=9.5,
                     color=PAPER, style="italic")
    ax2.set_yticks(y2, [c[0] for c in comp], fontsize=11.2)
    ax2.set_xlim(0, 1.08)
    ax2.set_xticks([0, 0.5, 1.0])
    ax2.tick_params(axis="x", labelsize=10.5)
    ax2.set_xlabel("generic reconstruction cosine · 849 roots", fontsize=11.5)
    ax2.grid(axis="x", color=HAIR, lw=0.7, zorder=0)
    _chip(ax2, "B", "Rank-32 projection preserves the output vocabulary", y=1.08)
    _set_figure_background(fig)
    fig.savefig(out_dir / f"{MOBILE_FIGURE_STEMS[1]}{OUTPUT_SUFFIX}.png", dpi=220, facecolor=WEB_PAPER)
    plt.close(fig)


def figure_factorization_mobile(data: dict[str, Any], out_dir: Path) -> None:
    """Render the factorization as three horizontal phone-readable stages."""
    fig, ax = plt.subplots(figsize=(4.4, 6.4))
    fig.subplots_adjust(left=0.08, right=0.94, top=0.88, bottom=0.25)
    _tier_fig(fig, "D", x=0.97, y=0.98)
    action = data["action_cos"]
    constrained = data["constrained_cos"]
    additive = data["additive_cos"]
    rows = [
        ("Action only", action, CHARCOAL),
        ("+ endpoint subtraction", constrained, "#B98258"),
        ("+ learned source coupling", additive, BLUE),
    ]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], height=0.52, color=[r[2] for r in rows], zorder=3)
    for yi, (label, value, _) in zip(y, rows):
        ax.text(0.025, yi, label, va="center", ha="left", color=PAPER, fontsize=11.5,
                fontweight=VALUE_WEIGHT)
        ax.text(value + 0.018, yi, f"{value:.4f}", va="center", fontsize=14,
                color=INK, fontweight=VALUE_WEIGHT)
    ax.text(0.03, -0.23,
            f"Endpoint subtraction explains {data['endpoint_ratio']*100:.1f}% of the observed lift",
            transform=ax.transAxes, fontsize=10.5, color=INK_SOFT, clip_on=False)
    ax.text(0.03, -0.30,
            f"Learned source coupling adds +{data['gap_free_minus_constrained']:.4f}",
            transform=ax.transAxes, fontsize=10.5, color=BLUE, clip_on=False)
    ax.set_xlim(0, 1.04)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(axis="x", labelsize=10.5)
    ax.set_xlabel("held-out family-macro cosine", fontsize=11.5)
    ax.grid(axis="x", color=HAIR, lw=0.7, zorder=0)
    _chip(ax, "", "A destination-conditioned rule predicts displacement", y=1.08)
    _set_figure_background(fig)
    fig.savefig(out_dir / f"{MOBILE_FIGURE_STEMS[2]}{OUTPUT_SUFFIX}.png", dpi=220, facecolor=WEB_PAPER)
    plt.close(fig)


def figure_social_card(data: dict[str, Any], out_dir: Path) -> None:
    """Generate a restrained 1200 x 630 share card from receipt-derived values."""
    local = data["bars"][0][1]
    global_mean = next(row[1] for row in data["bars"] if row[0] == "Global train mean")
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=WEB_PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.3)
    ax.axis("off")
    for radius, color, alpha, angle in ((2.4, BLUE, 0.42, -12), (1.6, EMBER, 0.44, 16)):
        ellipse = matplotlib.patches.Ellipse((10.2, 3.25), radius * 2, radius * 0.95,
                                             angle=angle, fill=False, lw=1.6,
                                             edgecolor=color, alpha=alpha)
        ax.add_patch(ellipse)
    ax.scatter([8.75, 10.05, 11.05], [3.1, 4.25, 2.35], s=[85, 55, 70],
               color=[BLUE, INK, EMBER], alpha=0.8)
    ax.annotate("", xy=(10.9, 3.15), xytext=(8.9, 3.15),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=BLUE))
    ax.text(0.75, 5.55, "DECEPTION PRESSURE GEOMETRY", fontsize=13, color=BLUE,
            fontweight=VALUE_WEIGHT, family="sans-serif")
    ax.text(0.75, 4.72, "The state before the model lies", fontsize=34, color=INK,
            fontweight=VALUE_WEIGHT, family="sans-serif")
    ax.text(0.75, 4.23, "is an address", fontsize=34, color=INK,
            fontweight=VALUE_WEIGHT, family="sans-serif")
    ax.text(0.75, 3.25, "Local activation retrieval reconstructs held-out displacement",
            fontsize=15.5, color=INK_SOFT, family="sans-serif")
    ax.text(0.75, 2.14, f"{local:.2f}", fontsize=43, color=BLUE, fontweight=VALUE_WEIGHT,
            family="sans-serif")
    ax.text(2.32, 2.32, "local retrieval", fontsize=14, color=INK_SOFT, family="sans-serif")
    ax.text(4.15, 2.14, f"{global_mean:.2f}", fontsize=43, color=CHARCOAL,
            fontweight=VALUE_WEIGHT, family="sans-serif")
    ax.text(5.73, 2.32, "one global direction", fontsize=14, color=INK_SOFT,
            family="sans-serif")
    ax.text(0.75, 0.74,
            "Offline reconstruction · held-out families · causal injection remains open",
            fontsize=13, color=INK_SOFT, family="sans-serif")
    fig.savefig(out_dir / SOCIAL_CARD_NAME, dpi=100, facecolor=WEB_PAPER,
                bbox_inches=None, pad_inches=0)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: reconstruction
# ---------------------------------------------------------------------------

def figure_reconstruction(data: dict[str, Any], out_dir: Path) -> None:
    fig = plt.figure(figsize=(6.5, 4.45))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.46, 1.0], hspace=0.58,
                          top=0.925, bottom=0.10)
    # Mixed chronology (sealed models vs. registered baselines): stated in the
    # caption rather than stamped, to keep the header clean.

    # -- A: one axis, the chasm and the clump --------------------------------
    ax = fig.add_subplot(gs[0])
    bars = data["bars"]
    def _rescolor(c):
        return BLUE if c == "HOLLOW" else EMBER if c == "HOLLOW_ALT" else c

    tied = [(n, v, _rescolor(c), str(c).startswith("HOLLOW"))
            for n, v, d, tot, c in bars if v > 0.9]
    others = [(n, v, d, tot, c) for n, v, d, tot, c in bars if v <= 0.9]
    cover = {n: (d, tot) for n, v, d, tot, c in bars}
    band_lo = min(v for _, v, _, _ in tied) - 0.004
    band_hi = max(v for _, v, _, _ in tied) + 0.004
    spread = max(v for _, v, _, _ in tied) - min(v for _, v, _, _ in tied)

    ax.set_xlim(0.38, 0.97)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    # the bottom spine IS the number line: dots sit on it, ticks hang off it
    ax.axvspan(band_lo, band_hi, ymin=0.0, ymax=0.055,
               color="#E7E2D6", zorder=0,
               transform=ax.get_xaxis_transform())

    def _dot(x, y, color, hollow, ms=9.0):
        ax.plot(x, y, "o", ms=ms, mfc=(BLUE_WASH if color == BLUE else EMBER_WASH)
                if hollow else color, mec=color if hollow else INK,
                mew=1.2 if hollow else 0.5, zorder=4, clip_on=False)

    for name, v, c, hollow in sorted(tied, key=lambda r: r[1]):
        _dot(v, 0.0, c, hollow)
    for name, v, d, tot, c in others:
        _dot(v, 0.0, c, False)
        left = v == min(o[1] for o in others)
        ax.annotate(f"{name.lower()}\n{v:.4f} $\\cdot$ {d}/{tot}",
                    xy=(v, 0.05),
                    xytext=(v - 0.014 if left else v + 0.014, 0.155),
                    ha="right" if left else "left", va="bottom", fontsize=7.8,
                    color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK_SOFT, lw=0.6))
    ax.annotate("", xy=(band_lo - 0.006, 0.30), xytext=(0.492, 0.30),
                arrowprops=dict(arrowstyle="<->", color=INK_SOFT, lw=0.9,
                                shrinkA=0, shrinkB=0))
    gap = next(row for row in data["paired"] if row[0] == "local $-$ global")
    ax.text(0.695, 0.335,
            "no address-free summary lives in this gap:\n"
            f"local over global $+{gap[1]:.4f}$ "
            f"$[+{gap[2][0]:.4f},+{gap[2][1]:.4f}]$, paired",
            ha="center", va="bottom", fontsize=8.0, color=INK_SOFT, style="italic")

    # the lens: five addresses magnified, large enough to read at print scale
    axz = ax.inset_axes([0.355, 0.585, 0.63, 0.395])
    axz.set_facecolor("#F1EEE6")
    zlo, zhi = 0.9095, 0.9425
    ys = np.linspace(0.84, 0.16, len(tied))
    for (name, v, c, hollow), yy in zip(sorted(tied, key=lambda r: -r[1]), ys):
        axz.plot(v, yy, "o", ms=7.5,
                 mfc=(BLUE_WASH if c == BLUE else EMBER_WASH) if hollow else c,
                 mec=c if hollow else INK, mew=1.1 if hollow else 0.5, zorder=4)
        d, tot = cover[name]
        label = f"{name}  {v:.4f} $\\cdot$ {d}/{tot}"
        if v > zlo + 0.35 * (zhi - zlo):
            axz.text(v - 0.0012, yy, label + "  ", va="center", ha="right",
                     fontsize=7.3, color=INK)
        else:
            axz.text(v + 0.0012, yy, "  " + label, va="center", ha="left",
                     fontsize=7.3, color=INK)
    axz.set_xlim(zlo, zhi)
    axz.set_ylim(0, 1)
    axz.set_yticks([])
    axz.set_xticks([0.915, 0.925, 0.935])
    axz.tick_params(labelsize=6.6, colors=INK_SOFT, length=2, pad=1.5)
    for sp in ("top", "right", "left"):
        axz.spines[sp].set_visible(False)
    axz.spines["bottom"].set_color(HAIR)
    for spine in axz.spines.values():
        spine.set_linewidth(0.6)
    axz.set_title(f"the five local addresses, magnified — spread {spread:.4f}",
                  fontsize=7.6, color=INK_SOFT, style="italic", pad=3)
    from matplotlib.patches import ConnectionPatch
    # one leader only: a left leader would cross the gap annotation
    fig.add_artist(ConnectionPatch(
        xyA=(band_hi, 0.055), coordsA=ax.get_xaxis_transform(),
        xyB=(zhi, 0.0), coordsB=axz.transData,
        color=HAIR, lw=0.8, zorder=1))

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(HAIR)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.tick_params(labelsize=8, colors=INK_SOFT, pad=7)
    ax.set_xlabel("held-out reconstruction cosine  $\\cdot$  200 deceptive source roots")
    _chip(ax, "A", "Every local address lands together; nothing global comes close",
          y=1.10)

    # -- B: paired differences ------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    paired = data["paired"]
    y2 = np.arange(len(paired))[::-1]
    for yi, (name, point, ci, color, hollow) in zip(y2, paired):
        ax2.plot([ci[0], ci[1]], [yi, yi], "-", color=color, lw=1.35,
                 solid_capstyle="butt", zorder=3)
        for cap in ci:
            ax2.plot([cap, cap], [yi - 0.14, yi + 0.14], "-", color=color, lw=0.75,
                     zorder=3)
        ax2.plot(point, yi, "o", ms=5.2, mfc=PAPER if hollow else color,
                 mec=color, mew=1.0, zorder=4)
        label = f"$+{point:.4f}$" if point >= 0 else f"$-{abs(point):.4f}$"
        ax2.text(ci[1] + 0.013, yi, label, va="center", ha="left",
                 fontsize=8.4, color=INK)
    ax2.axvline(0.0, color=INK, lw=0.9, zorder=2)
    ax2.set_yticks(y2, [p[0] for p in paired], fontsize=8)
    ax2.set_xlim(-0.075, 0.66)
    ax2.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax2.set_xlabel("paired cosine difference  $\\cdot$  scenario-cluster 95% interval")
    ax2.grid(axis="x", color=HAIR, lw=0.35, zorder=0)
    ax2.set_axisbelow(True)
    _chip(ax2, "B", "Retrieval carries the gain, under every address tested", y=1.14)

    _save_standard(fig, out_dir, FIGURE_STEMS[0])
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: specificity + compression
# ---------------------------------------------------------------------------

def figure_structure(data: dict[str, Any], out_dir: Path) -> None:
    fig = plt.figure(figsize=(6.5, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.14, 1.0], wspace=0.45,
                          top=0.84, bottom=0.17)
    _tier_fig(fig, "U")

    # -- A: specificity, both metrics ----------------------------------------
    ax = fig.add_subplot(gs[0])
    rows = []
    for name, cos_pt, cos_ci, nse_pt, nse_ci in data["specificity"]:
        rows.append((f"{name} (cos)", cos_pt, cos_ci, BLUE))
        rows.append((f"{name} (NSE)", nse_pt, nse_ci, GRAY))
    spacing = np.array([0.0, 1.0, 2.6, 3.6, 5.2, 6.2])
    y = spacing.max() - spacing
    for yi, (name, pt, ci, color) in zip(y, rows):
        ax.plot([ci[0], ci[1]], [yi, yi], "-", color=color, lw=1.35,
                solid_capstyle="butt", zorder=3)
        for cap in ci:
            ax.plot([cap, cap], [yi - 0.17, yi + 0.17], "-", color=color, lw=0.75,
                    zorder=3)
        ax.plot(pt, yi, "o", ms=5.2, mfc=color, mec=color, zorder=4)
    ax.axvline(0.0, color=INK, lw=0.9, zorder=2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set_xlabel("paired difference  $\\cdot$  scenario-cluster 95% interval")
    ax.grid(axis="x", color=HAIR, lw=0.35, zorder=0)
    ax.set_axisbelow(True)
    _chip(ax, "A", "Specificity, both metrics")

    # -- B: compression -------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    comp = data["compression"]
    y2 = np.arange(len(comp))[::-1]
    cfills = [BLUE_WASH if c[4] == "HOLLOW" else c[4] for c in comp]
    cedges = [BLUE if c[4] == "HOLLOW" else INK for c in comp]
    cwidths = [0.9 if c[4] == "HOLLOW" else 0.35 for c in comp]
    ax2.barh(y2, [c[1] for c in comp], height=0.48, color=cfills, zorder=3,
             edgecolor=cedges, linewidth=cwidths)
    for yi, (name, val, defined, total, color) in zip(y2, comp):
        ax2.text(val + 0.014, yi, f"{val:.4f}", va="center", ha="left", fontsize=9.25,
                 color=INK, fontweight=VALUE_WEIGHT)
    landmark = next(c for c in comp if c[2] != c[3])
    ax2.set_yticks(y2, [c[0] for c in comp], fontsize=8)
    ax2.set_xlim(0, 1.05)
    ax2.set_xticks([0, 0.5, 1.0])
    ax2.set_xlabel("generic cosine  $\\cdot$  849 roots")
    ax2.grid(axis="x", color=HAIR, lw=0.35, zorder=0)
    ax2.set_axisbelow(True)
    ax2.text(0.99, 0.02, f"landmark coverage {landmark[2]}/{landmark[3]}\n"
             "rank 32 selected in every fold\n"
             f"({data['rank_var_min']*100:.2f}–{data['rank_var_max']*100:.2f}% "
             "train variance)", transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=7.2, color=INK_SOFT, style="italic")
    _chip(ax2, "B", "Output compression")

    _save_standard(fig, out_dir, FIGURE_STEMS[1])
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: factorization waterfall
# ---------------------------------------------------------------------------

def figure_factorization(data: dict[str, Any], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 2.95))
    fig.subplots_adjust(top=0.82)
    _tier_fig(fig, "D")
    action = data["action_cos"]
    constrained = data["constrained_cos"]
    additive = data["additive_cos"]

    xs = np.array([0.0, 1.0, 2.0])
    w = 0.56
    ax.bar(xs[0], action, w, color=CHARCOAL, zorder=3,
           edgecolor=INK, linewidth=0.45)
    for x0, segs in ((xs[1], [(0.0, action, CHARCOAL), (action, constrained, EMBER)]),
                     (xs[2], [(0.0, action, CHARCOAL), (action, constrained, EMBER),
                              (constrained, additive, BLUE)])):
        for lo, hi, color in segs:
            ax.bar(x0, hi - lo, w, bottom=lo, color=color, zorder=3,
                   edgecolor=PAPER, linewidth=0.4)
        top = segs[-1][1]
        ax.bar(x0, top, w, fill=False, zorder=4, edgecolor=INK, linewidth=0.45)

    for x0, level in ((xs[0], action), (xs[1], constrained)):
        ax.plot([x0 + w / 2, x0 + 1 - w / 2], [level, level], ls=(0, (2, 2)),
                color=INK_SOFT, lw=0.9, zorder=4)

    off = np.linspace(-0.16, 0.16, len(data["action_folds"]))
    for x0, folds in ((xs[0], data["action_folds"]),
                      (xs[1], data["constrained_folds"]),
                      (xs[2], data["additive_folds"])):
        ax.plot(x0 + off, folds, "o", ms=3.6, mfc=PAPER, mec=INK, mew=0.9, zorder=5)

    for x0, v in ((xs[0], action), (xs[1], constrained), (xs[2], additive)):
        ax.text(x0, v + 0.045, f"{v:.4f}", ha="center", fontsize=10, color=INK,
                fontweight=VALUE_WEIGHT)

    ax.annotate(f"$+{constrained - action:.4f}$  endpoint subtraction\n"
                f"({data['endpoint_ratio']*100:.1f}% of the observed\n"
                "cosine improvement)",
                xy=(xs[1] - w / 2, (action + constrained) / 2),
                xytext=(-0.50, 0.985),
                fontsize=8, color=INK, va="top", ha="left",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7,
                                shrinkB=4))
    ax.annotate(f"$+{data['gap_free_minus_constrained']:.4f}$  learned source coupling",
                xy=(xs[2] + w / 2, (constrained + additive) / 2),
                xytext=(xs[2] + 0.42, (constrained + additive) / 2 + 0.06),
                fontsize=8, color=BLUE, va="center",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7))

    ax.set_xticks(xs, ["action only", "+ endpoint subtraction\n($B=-I$)",
                       "+ learned source\ncoupling (free $B$)"], fontsize=8.5)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("family-macro cosine")
    ax.grid(axis="y", color=HAIR, lw=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.0, -0.315, "open points: five held-out family folds "
            "(fold consistency, not a confidence interval)",
            transform=ax.transAxes, fontsize=7.2, color=INK_SOFT, style="italic")
    _chip(ax, "", "A linear source-plus-action model predicts displacement")

    _save_standard(fig, out_dir, FIGURE_STEMS[2])
    plt.close(fig)


def figure_structure_factorization(data: dict[str, Any], out_dir: Path) -> None:
    """Hero waterfall on top, compression beneath: one top-to-bottom reading path."""
    fig = plt.figure(figsize=(6.5, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.52,
                          top=0.90, bottom=0.10, left=0.08, right=0.975)

    # -- A (hero): factorization waterfall, annotated -------------------------
    ax3 = fig.add_subplot(gs[0])
    action = data["action_cos"]
    constrained = data["constrained_cos"]
    additive = data["additive_cos"]
    xs = np.array([0.0, 1.0, 2.0])
    w = 0.5
    ax3.bar(xs[0], action, w, color=CHARCOAL, zorder=3, edgecolor=INK, linewidth=0.45)
    for x0, segs in ((xs[1], [(0.0, action, CHARCOAL), (action, constrained, EMBER)]),
                     (xs[2], [(0.0, action, CHARCOAL), (action, constrained, EMBER),
                              (constrained, additive, BLUE)])):
        for lo, hi, color in segs:
            ax3.bar(x0, hi - lo, w, bottom=lo, color=color, zorder=3,
                    edgecolor=PAPER, linewidth=0.4)
        ax3.bar(x0, segs[-1][1], w, fill=False, zorder=4, edgecolor=INK,
                linewidth=0.45)
    for x0, level in ((xs[0], action), (xs[1], constrained)):
        ax3.plot([x0 + w / 2, x0 + 1 - w / 2], [level, level], ls=(0, (2, 2)),
                 color=INK_SOFT, lw=0.8, zorder=4)
    off = np.linspace(-0.13, 0.13, len(data["action_folds"]))
    for x0, folds in ((xs[0], data["action_folds"]),
                      (xs[1], data["constrained_folds"]),
                      (xs[2], data["additive_folds"])):
        ax3.plot(x0 + off, folds, "o", ms=2.6, mfc=PAPER, mec=INK, mew=0.7, zorder=5)
    for x0, v in ((xs[0], action), (xs[1], constrained), (xs[2], additive)):
        ax3.text(x0, v + 0.05, f"{v:.4f}", ha="center", fontsize=9, color=INK,
                 fontweight="bold")
    ax3.annotate(f"$+{constrained - action:.4f}$  endpoint subtraction\n"
                 f"({data['endpoint_ratio']*100:.1f}% of the observed\n"
                 "cosine improvement)",
                 xy=(xs[1] - w / 2, (action + constrained) / 2),
                 xytext=(-0.58, 0.99), fontsize=7.6, color=INK,
                 va="top", ha="left",
                 arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkB=4))
    ax3.annotate(f"$+{data['gap_free_minus_constrained']:.4f}$  learned\n"
                 "source coupling",
                 xy=(xs[2] + w / 2, (constrained + additive) / 2),
                 xytext=(2.45, 0.62), fontsize=7.6, color=BLUE, va="center",
                 arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7, shrinkB=2))
    ax3.set_xticks(xs, ["action only", "$+$ endpoint ($B=-I$)",
                        "$+$ learned source (free $B$)"], fontsize=8)
    ax3.set_xlim(-0.72, 3.05)
    ax3.set_ylim(0, 1.02)
    ax3.set_yticks([0, 0.5, 1.0])
    ax3.tick_params(labelsize=7.5)
    ax3.set_ylabel("family-macro cosine", fontsize=8)
    ax3.grid(axis="y", color=HAIR, lw=0.35, zorder=0)
    ax3.set_axisbelow(True)
    _chip(ax3, "A", "A linear source-plus-action rule predicts displacement",
          y=1.075)

    # -- B: compression -------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    comp = data["compression"]
    y2 = np.arange(len(comp))[::-1]
    cfills = [BLUE_WASH if c[4] == "HOLLOW" else c[4] for c in comp]
    cedges = [BLUE if c[4] == "HOLLOW" else INK for c in comp]
    cwidths = [0.9 if c[4] == "HOLLOW" else 0.35 for c in comp]
    ax2.barh(y2, [c[1] for c in comp], height=0.6, color=cfills, zorder=3,
             edgecolor=cedges, linewidth=cwidths)
    full_cos = comp[0][1]
    for yi, (name, val, defined, total, color) in zip(y2, comp):
        dark_fill = color in (BLUE, CHARCOAL)
        inside = val > 0.4
        shown = name
        if color == "HOLLOW":
            shown = f"{name}  $\\cdot$  {val / full_cos * 100:.1f}% of full"
        ax2.text(0.015 if inside else val + 0.015, yi, shown, va="center", ha="left",
                 fontsize=8, color=PAPER if (dark_fill and inside) else INK,
                 zorder=5)
        ax2.text(val + 0.015 if inside else val + 0.36, yi, f"{val:.4f}",
                 va="center", ha="left", fontsize=8.4, color=INK,
                 fontweight="bold")
    ax2.set_yticks([])
    ax2.set_xlim(0, 1.12)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.tick_params(labelsize=7.5)
    ax2.set_xlabel("generic reconstruction cosine  $\\cdot$  849 roots", fontsize=8)
    ax2.grid(axis="x", color=HAIR, lw=0.35, zorder=0)
    ax2.set_axisbelow(True)
    _chip(ax2, "B", "Rank 32 keeps the whole vocabulary", y=1.14)

    _save_standard(fig, out_dir, COMBINED_STEM)
    plt.close(fig)


def figure_pressure_behavior(out_dir: Path) -> None:
    """Deceptive-commitment rate per two-slot pressure program (own receipt)."""
    receipt = json.loads(PRESSURE_RECEIPT_PATH.read_text(encoding="utf-8"))
    if receipt.get("kind") != "pressure_behavior_public_receipt":
        die("unexpected pressure receipt kind")
    programs = receipt["programs"]
    order = ["NN", "AN", "D2N", "AA", "AB", "BA"]
    gloss = {
        "NN": "no pressure",
        "AN": "one pressure sentence",
        "D2N": "doubled sentence, one slot",
        "AA": "pressure at both slots",
        "AB": "pressure, then caveat-suppression",
        "BA": "caveat-suppression, then pressure",
    }
    fig, ax = plt.subplots(figsize=(6.5, 1.92))
    fig.subplots_adjust(top=0.80, bottom=0.235, left=0.30, right=0.955)
    y = np.arange(len(order))[::-1]
    rates = [programs[k]["deceptive_rate"] for k in order]
    colors = [GRAY if k == "NN" else CHARCOAL for k in order]
    ax.barh(y, rates, height=0.58, color=colors, zorder=3,
            edgecolor=INK, linewidth=0.35)
    for yi, k, rate in zip(y, order, rates):
        n = programs[k]["n"]
        if rate >= 0.5:
            ax.text(rate - 0.012, yi, f"{rate * 100:.1f}%", va="center", ha="right",
                    fontsize=8.4, color=PAPER, fontweight="bold")
        else:
            ax.text(rate + 0.012, yi, f"{rate * 100:.1f}%", va="center", ha="left",
                    fontsize=8.4, color=INK, fontweight="bold")
        ax.text(1.115, yi, f"{programs[k]['deceptive']}/{n}", va="center",
                ha="right", fontsize=7.0, color=INK_SOFT)
    ax.set_yticks(y, [gloss[k] for k in order], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=7.5)
    ax.set_xlabel("conversations committing a false status  $\\cdot$  600 presented",
                  fontsize=8)
    ax.grid(axis="x", color=HAIR, lw=0.35, zorder=0)
    ax.set_axisbelow(True)
    _chip(ax, "", "One pressure sentence flips most commitments", y=1.13)
    _save_standard(fig, out_dir, PRESSURE_STEM)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=RECEIPT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--formats", default="pdf,png",
                        help="comma-separated subset of pdf,png")
    parser.add_argument("--theme", default="print", choices=sorted(THEMES),
                        help="print = manuscript (Archive Indigo); web = blog")
    parser.add_argument("--blog", action="store_true",
                        help="emit de-framed, lighter *_blog.png web variants only")
    args = parser.parse_args(argv)
    global ACTIVE_FORMATS, VALUE_WEIGHT, OUTPUT_SUFFIX, PAPER, WEB_PAPER
    if args.blog:
        args.theme = "web"
        VALUE_WEIGHT = "normal"
        OUTPUT_SUFFIX = "_blog"
        args.formats = "png"
    apply_theme(args.theme)
    if args.blog:
        # blog variants sit on the article page surface, not the web canvas
        PAPER = WEB_PAPER = "#FFFDF8"
        plt.rcParams.update({"figure.facecolor": PAPER, "axes.facecolor": PAPER,
                             "savefig.facecolor": PAPER})
    ACTIVE_FORMATS = tuple(f for f in args.formats.split(",") if f in ("pdf", "png"))
    if not ACTIVE_FORMATS:
        die("--formats must include pdf and/or png")

    data = parse_data(load_receipt(args.receipt))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_reconstruction(data, args.out_dir)
    figure_structure(data, args.out_dir)
    figure_factorization(data, args.out_dir)
    if args.blog:
        figure_reconstruction_mobile(data, args.out_dir)
        figure_structure_mobile(data, args.out_dir)
        figure_factorization_mobile(data, args.out_dir)
        return 0
    figure_structure_factorization(data, args.out_dir)
    figure_pressure_behavior(args.out_dir)
    apply_theme("web")
    web_data = parse_data(load_receipt(args.receipt))
    figure_reconstruction_mobile(web_data, args.out_dir)
    figure_structure_mobile(web_data, args.out_dir)
    figure_factorization_mobile(web_data, args.out_dir)
    figure_social_card(web_data, args.out_dir)
    apply_theme(args.theme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
