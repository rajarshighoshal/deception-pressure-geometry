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
ACTIVE_FORMATS: tuple[str, ...] = ("pdf", "png")
# Blog style (opt-in via --blog): de-framed fields and lighter labels for the
# public article; the canonical manuscript outputs stay byte-stable.
VALUE_WEIGHT = "bold"
OUTPUT_SUFFIX = ""
FIGURE_NAMES = (
    [f"{stem}{OUTPUT_SUFFIX}.{ext}" for stem in FIGURE_STEMS for ext in ("pdf", "png")]
    + [f"{stem}.png" for stem in MOBILE_FIGURE_STEMS]
    + [SOCIAL_CARD_NAME]
)

# ---------------------------------------------------------------------------
# Ledger theme
# ---------------------------------------------------------------------------
INK = "#171814"          # warm near-black: text, primary marks
INK_SOFT = "#5B554C"     # secondary text
HAIR = "#C4B7A5"         # hairlines, grid
PAPER = "#FFFDF8"
WEB_PAPER = "#DCD4C3"
BLUE = "#1F5E8C"         # muted steel blue: the local estimator / this paper's method
EMBER = "#B0722A"        # soft amber: retrieval/alternative family (nearest, landmark)
CHARCOAL = "#332F29"     # global / linear baselines
GRAY = "#827668"         # nulls and shuffles
RED = "#B3403A"          # reserved status color: refuted (never a series color)

TIER_STYLE = {
    "U": {"label": "RETROSPECTIVE UNREGISTERED DESCRIPTIVE", "fc": "#F1EFEC", "ec": HAIR,
          "tc": INK_SOFT},
    "D": {"label": "POST-EVIDENCE REGISTERED DESCRIPTIVE", "fc": "#F8E8DC", "ec": "#E5C4A8",
          "tc": "#8A4415"},
    "R": {"label": "REGISTERED ENDPOINT", "fc": "#E4E9F8", "ec": "#BCC8EE",
          "tc": "#41598C"},
}

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
             "#E69A54"),
            ("Typed-graph local", models["local"]["cosine_mean"],
             models["local"]["defined_count"], models["local"]["total_count"], BLUE),
            ("Truth-aware design-cell mean", sa_models["design_cell_mean"]["cosine_mean"],
             sa_models["design_cell_mean"]["defined_count"],
             sa_models["design_cell_mean"]["total_count"], "#654d82"),
            ("Typed-graph nearest", models["nearest"]["cosine_mean"],
             models["nearest"]["defined_count"], models["nearest"]["total_count"],
             "#6E91C5"),
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
             sec_comps["nearest"]["scenario_cluster_ci"], "#6E91C5", True),
            ("graph local $-$ raw k=8",
             sa_comps["graph_local_minus_raw_k8"]["mean_cosine_difference"],
             sa_comps["graph_local_minus_raw_k8"]["scenario_cluster_ci"], EMBER, False),
            ("graph local $-$ design-cell mean",
             sa_comps["graph_local_minus_design_cell_mean"]["mean_cosine_difference"],
             sa_comps["graph_local_minus_design_cell_mean"]["scenario_cluster_ci"],
             "#654d82", False),
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
             cmodels["low_rank_projected_full"]["total_count"], "#75A0D1"),
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
             bbox=dict(boxstyle="round,pad=0.42,rounding_size=0.35", fc=s["fc"],
                       ec=s["ec"], lw=0.7))


def _tier(ax: plt.Axes, tier: str, x: float = 1.0, y: float = 1.0875) -> None:
    """Chronology stamp chip, top-right, axes-coords."""
    s = TIER_STYLE[tier]
    ax.text(x, y, s["label"], transform=ax.transAxes, ha="right", va="center",
            fontsize=6.0, color=s["tc"], clip_on=False,
            bbox=dict(boxstyle="round,pad=0.42,rounding_size=0.35", fc=s["fc"],
                      ec=s["ec"], lw=0.7))


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
    ax.barh(y, [b[1] for b in bars], height=0.58, color=[b[4] for b in bars], zorder=3)
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
    ax2.barh(y2, [c[1] for c in comp], height=0.58, color=[c[4] for c in comp], zorder=3)
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
    fig = plt.figure(figsize=(6.5, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.58,
                          top=0.90, bottom=0.085)
    # Mixed chronology (sealed models vs. registered baselines): stated in the
    # caption rather than stamped, to keep the header clean.

    # -- A: model cosines -----------------------------------------------------
    ax = fig.add_subplot(gs[0])
    bars = data["bars"]
    y = np.arange(len(bars))[::-1].astype(float)
    y[5:] -= 0.7  # visual gap: local addresses above, baselines below
    ax.barh(y, [b[1] for b in bars], height=0.62, color=[b[4] for b in bars], zorder=3)
    for yi, (name, val, defined, total, color) in zip(y, bars):
        ax.text(val + 0.012, yi, f"{val:.4f}", va="center", ha="left",
                fontsize=12, color=INK, fontweight=VALUE_WEIGHT)
        _coverage_chip(ax, 0.012, yi, defined, total)
    ax.set_yticks(y, [b[0] for b in bars])
    ax.set_xlim(0, 1.06)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("held-out reconstruction cosine  $\\cdot$  200 deceptive source roots")
    ax.grid(axis="x", color=HAIR, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    _chip(ax, "A", "Simple local addresses predict held-out displacement")

    # -- B: paired differences ------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    paired = data["paired"]
    y2 = np.arange(len(paired))[::-1]
    for yi, (name, point, ci, color, hollow) in zip(y2, paired):
        ax2.plot([ci[0], ci[1]], [yi, yi], "-", color=color, lw=2.4,
                 solid_capstyle="butt", zorder=3)
        for cap in ci:
            ax2.plot([cap, cap], [yi - 0.14, yi + 0.14], "-", color=color, lw=1.1,
                     zorder=3)
        ax2.plot(point, yi, "o", ms=6.5, mfc=PAPER if hollow else color,
                 mec=color, mew=1.4, zorder=4)
        label = f"$+{point:.4f}$" if point >= 0 else f"$-{abs(point):.4f}$"
        ax2.text(ci[1] + 0.013, yi, label, va="center", ha="left",
                 fontsize=9.5, color=INK)
    ax2.axvline(0.0, color=INK, lw=0.9, zorder=2)
    ax2.set_yticks(y2, [p[0] for p in paired])
    ax2.set_xlim(-0.075, 0.66)
    ax2.set_xlabel("paired cosine difference  $\\cdot$  scenario-cluster 95% interval")
    ax2.grid(axis="x", color=HAIR, lw=0.6, zorder=0)
    ax2.set_axisbelow(True)
    _chip(ax2, "B", "Retrieval carries the gain, under every address tested")

    _save_standard(fig, out_dir, FIGURE_STEMS[0])
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: specificity + compression
# ---------------------------------------------------------------------------

def figure_structure(data: dict[str, Any], out_dir: Path) -> None:
    fig = plt.figure(figsize=(6.5, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.14, 1.0], wspace=0.55,
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
        ax.plot([ci[0], ci[1]], [yi, yi], "-", color=color, lw=2.4,
                solid_capstyle="butt", zorder=3)
        for cap in ci:
            ax.plot([cap, cap], [yi - 0.17, yi + 0.17], "-", color=color, lw=1.1,
                    zorder=3)
        ax.plot(pt, yi, "o", ms=6, mfc=color, mec=color, zorder=4)
    ax.axvline(0.0, color=INK, lw=0.9, zorder=2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set_xlabel("paired difference  $\\cdot$  scenario-cluster 95% interval")
    ax.grid(axis="x", color=HAIR, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    _chip(ax, "A", "Specificity, both metrics")

    # -- B: compression -------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    comp = data["compression"]
    y2 = np.arange(len(comp))[::-1]
    ax2.barh(y2, [c[1] for c in comp], height=0.6, color=[c[4] for c in comp], zorder=3)
    for yi, (name, val, defined, total, color) in zip(y2, comp):
        ax2.text(val + 0.014, yi, f"{val:.4f}", va="center", ha="left", fontsize=10.5,
                 color=INK, fontweight=VALUE_WEIGHT)
        if defined != total:
            _coverage_chip(ax2, 0.014, yi, defined, total)
    ax2.set_yticks(y2, [c[0] for c in comp], fontsize=8)
    ax2.set_xlim(0, 1.05)
    ax2.set_xticks([0, 0.5, 1.0])
    ax2.set_xlabel("generic cosine  $\\cdot$  849 roots")
    ax2.grid(axis="x", color=HAIR, lw=0.6, zorder=0)
    ax2.set_axisbelow(True)
    ax2.text(0.99, 0.02, "rank 32 selected in every fold\n"
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
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    fig.subplots_adjust(top=0.84)
    _tier_fig(fig, "D")
    action = data["action_cos"]
    constrained = data["constrained_cos"]
    additive = data["additive_cos"]

    xs = np.array([0.0, 1.0, 2.0])
    w = 0.56
    ax.bar(xs[0], action, w, color=CHARCOAL, zorder=3)
    ax.bar(xs[1], constrained - action, w, bottom=action, color="#B98258", zorder=3)
    ax.bar(xs[1], action, w, color=CHARCOAL, alpha=0.18, zorder=2)
    ax.bar(xs[2], additive - constrained, w, bottom=constrained, color=BLUE, zorder=3)
    ax.bar(xs[2], constrained - action, w, bottom=action, color="#B98258",
           alpha=0.30, zorder=2)
    ax.bar(xs[2], action, w, color=CHARCOAL, alpha=0.18, zorder=2)

    for x0, level in ((xs[0], action), (xs[1], constrained)):
        ax.plot([x0 + w / 2, x0 + 1 - w / 2], [level, level], ls=(0, (2, 2)),
                color=INK_SOFT, lw=0.9, zorder=4)

    off = np.linspace(-0.16, 0.16, len(data["action_folds"]))
    for x0, folds in ((xs[0], data["action_folds"]),
                      (xs[1], data["constrained_folds"]),
                      (xs[2], data["additive_folds"])):
        ax.plot(x0 + off, folds, "o", ms=3.6, mfc=PAPER, mec=INK, mew=0.9, zorder=5)

    for x0, v in ((xs[0], action), (xs[1], constrained), (xs[2], additive)):
        ax.text(x0, v + 0.045, f"{v:.4f}", ha="center", fontsize=12, color=INK,
                fontweight=VALUE_WEIGHT)

    ax.annotate(f"$+{constrained - action:.4f}$  endpoint subtraction\n"
                f"({data['endpoint_ratio']*100:.1f}% of the observed cosine improvement)",
                xy=(xs[1] + w / 2, (action + constrained) / 2),
                xytext=(xs[1] + 0.55, (action + constrained) / 2 - 0.13),
                fontsize=8, color=INK_SOFT, va="center",
                arrowprops=dict(arrowstyle="-", color=INK_SOFT, lw=0.7))
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
    ax.grid(axis="y", color=HAIR, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.0, -0.315, "open points: five held-out family folds "
            "(fold consistency, not a confidence interval)",
            transform=ax.transAxes, fontsize=7.2, color=INK_SOFT, style="italic")
    _chip(ax, "", "A linear source-plus-action model predicts displacement")

    _save_standard(fig, out_dir, FIGURE_STEMS[2])
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=RECEIPT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--formats", default="pdf,png",
                        help="comma-separated subset of pdf,png")
    parser.add_argument("--blog", action="store_true",
                        help="emit de-framed, lighter *_blog.png web variants only")
    args = parser.parse_args(argv)
    global ACTIVE_FORMATS, VALUE_WEIGHT, OUTPUT_SUFFIX, PAPER, WEB_PAPER
    if args.blog:
        VALUE_WEIGHT = "normal"
        OUTPUT_SUFFIX = "_blog"
        PAPER = WEB_PAPER = "#FFFDF8"
        args.formats = "png"
    ACTIVE_FORMATS = tuple(f for f in args.formats.split(",") if f in ("pdf", "png"))
    if not ACTIVE_FORMATS:
        die("--formats must include pdf and/or png")

    data = parse_data(load_receipt(args.receipt))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_reconstruction(data, args.out_dir)
    figure_structure(data, args.out_dir)
    figure_factorization(data, args.out_dir)
    figure_reconstruction_mobile(data, args.out_dir)
    figure_structure_mobile(data, args.out_dir)
    figure_factorization_mobile(data, args.out_dir)
    if not args.blog:
        figure_social_card(data, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
