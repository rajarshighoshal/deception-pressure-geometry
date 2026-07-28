"""Unified plotting theme for lie-geometry-probes figures.

A clean, publication-ready style inspired by Tufte principles and modern
data visualization best practices. Designed for both paper (PDF) and web (PNG).

Usage:
    from geoprobe.plotting.theme import apply_theme, COLORS, save_figure

    apply_theme()  # Call once at script start
    fig, ax = plt.subplots()
    # ... your plotting code ...
    save_figure(fig, "my_figure")  # Saves both PDF and PNG
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Force non-interactive backend for reproducible output
matplotlib.use("Agg", force=True)

# =============================================================================
# COLOR PALETTE
# =============================================================================
# Clean, colorblind-friendly palette based on Tableau 10 / D3 category10

COLORS = {
    # Primary semantic colors
    "primary": "#4C78A8",      # Steel blue - main data, positive results
    "secondary": "#F58518",    # Orange - secondary series, alternatives
    "success": "#54A24B",      # Green - positive outcomes, fixes
    "danger": "#E45756",       # Coral red - negative outcomes, harms, failures
    "neutral": "#72B7B2",      # Teal - neutral/baseline comparisons

    # Grayscale for structure
    "ink": "#2D2D2D",          # Near-black for primary text
    "ink_soft": "#666666",     # Medium gray for secondary text
    "context": "#999999",      # Light gray for context/annotations
    "grid": "#E5E5E5",         # Very light gray for gridlines
    "surface": "#FFFFFF",      # Pure white background

    # Extended palette for multi-series
    "blue": "#4C78A8",
    "orange": "#F58518",
    "red": "#E45756",
    "green": "#54A24B",
    "teal": "#72B7B2",
    "purple": "#9D68A8",
    "pink": "#ED97CA",
    "brown": "#A57706",
    "gray": "#AAAAAA",
}

# Categorical palette for bar charts, scatter, etc.
PALETTE_CATEGORICAL = [
    COLORS["blue"],
    COLORS["orange"],
    COLORS["green"],
    COLORS["red"],
    COLORS["purple"],
    COLORS["teal"],
]

# Diverging palette for difference plots
PALETTE_DIVERGING = [COLORS["danger"], COLORS["neutral"], COLORS["success"]]


# =============================================================================
# RCPARAMS - Global matplotlib configuration
# =============================================================================

RCPARAMS = {
    # Typography - clean sans-serif with good Unicode support
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "medium",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,

    # Colors
    "text.color": COLORS["ink"],
    "axes.labelcolor": COLORS["ink"],
    "xtick.color": COLORS["ink_soft"],
    "ytick.color": COLORS["ink_soft"],
    "axes.edgecolor": COLORS["grid"],
    "figure.facecolor": COLORS["surface"],
    "axes.facecolor": COLORS["surface"],
    "savefig.facecolor": COLORS["surface"],

    # Spines - minimal, Tufte-style
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.linewidth": 0.8,

    # Grid - subtle, y-axis only by default
    "axes.grid": False,  # Enable selectively per-plot
    "axes.axisbelow": True,
    "grid.color": COLORS["grid"],
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,

    # Lines and markers
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "patch.linewidth": 0.5,
    "patch.edgecolor": COLORS["ink"],

    # Legend
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": COLORS["grid"],
    "legend.fancybox": False,

    # Figure
    "figure.figsize": (7, 4.5),
    "figure.dpi": 100,
    "figure.autolayout": False,

    # Saving
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "pdf.fonttype": 42,  # TrueType fonts in PDF for editability
    "ps.fonttype": 42,
}


def apply_theme() -> None:
    """Apply the unified theme to matplotlib.

    Call this once at the start of your plotting script.
    """
    plt.rcParams.update(RCPARAMS)


def reset_theme() -> None:
    """Reset matplotlib to default settings."""
    plt.rcdefaults()


# =============================================================================
# FIGURE HELPERS
# =============================================================================

def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    *,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    width_ratios: list[float] | None = None,
    height_ratios: list[float] | None = None,
) -> tuple[plt.Figure, Any]:
    """Create a figure with the theme applied.

    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size (width, height) in inches. If None, auto-computed.
        sharex: Share x-axis across subplots
        sharey: Share y-axis across subplots
        squeeze: If True, return single Axes for 1x1, array otherwise
        width_ratios: Relative widths of columns
        height_ratios: Relative heights of rows

    Returns:
        (fig, axes) tuple
    """
    if figsize is None:
        # Auto-compute based on subplot count
        base_w, base_h = 3.5, 3.0
        figsize = (base_w * ncols + 0.5, base_h * nrows + 0.5)

    gridspec_kw = {}
    if width_ratios is not None:
        gridspec_kw["width_ratios"] = width_ratios
    if height_ratios is not None:
        gridspec_kw["height_ratios"] = height_ratios

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        gridspec_kw=gridspec_kw if gridspec_kw else None,
    )
    return fig, axes


def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: Path | str = ".",
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int | None = None,
    close: bool = True,
) -> list[Path]:
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib figure
        name: Base filename (without extension)
        output_dir: Directory to save to
        formats: Tuple of formats to save ("png", "pdf", "svg")
        dpi: Override DPI for raster formats
        close: Close figure after saving

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        save_kw: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": 0.1,
            "facecolor": COLORS["surface"],
            "edgecolor": "none",
        }
        if fmt in ("png", "jpg", "jpeg"):
            save_kw["dpi"] = dpi or 150
        if fmt == "pdf":
            save_kw["dpi"] = dpi or 300
            # Strip metadata for reproducibility
            save_kw["metadata"] = {"Creator": None, "Producer": None}

        fig.savefig(path, format=fmt, **save_kw)
        saved.append(path)

    if close:
        plt.close(fig)

    return saved


# =============================================================================
# PLOT STYLING HELPERS
# =============================================================================

def style_axis(
    ax: plt.Axes,
    *,
    grid: str | bool = "y",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> plt.Axes:
    """Apply consistent styling to an axis.

    Args:
        ax: Matplotlib axes
        grid: "x", "y", "both", True, or False
        title: Axis title
        xlabel: X-axis label
        ylabel: Y-axis label

    Returns:
        The styled axes
    """
    # Grid
    if grid is True or grid == "both":
        ax.grid(True, axis="both", alpha=0.5, linewidth=0.5)
    elif grid == "y":
        ax.grid(True, axis="y", alpha=0.5, linewidth=0.5)
    elif grid == "x":
        ax.grid(True, axis="x", alpha=0.5, linewidth=0.5)
    else:
        ax.grid(False)

    ax.set_axisbelow(True)

    # Labels
    if title:
        ax.set_title(title, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    return ax


def add_zero_line(ax: plt.Axes, axis: str = "y", **kwargs: Any) -> None:
    """Add a subtle zero reference line."""
    defaults = {"color": COLORS["ink_soft"], "linewidth": 0.8, "linestyle": "--", "alpha": 0.6}
    defaults.update(kwargs)
    if axis == "y":
        ax.axhline(0, **defaults)
    else:
        ax.axvline(0, **defaults)


def annotate_bar(
    ax: plt.Axes,
    bars: Any,
    fmt: str = "{:.2f}",
    offset: float = 0.02,
    fontsize: int = 8,
    **kwargs: Any,
) -> None:
    """Add value labels above/below bars."""
    for bar in bars:
        height = bar.get_height()
        y_pos = height + offset if height >= 0 else height - offset
        va = "bottom" if height >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=fontsize,
            **kwargs,
        )


def format_percent_axis(ax: plt.Axes, axis: str = "y", decimals: int = 0) -> None:
    """Format axis ticks as percentages."""
    from matplotlib.ticker import PercentFormatter
    formatter = PercentFormatter(1.0, decimals=decimals)
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


# =============================================================================
# ERROR BAR HELPERS
# =============================================================================

def plot_with_ci(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
    *,
    color: str | None = None,
    label: str | None = None,
    marker: str = "o",
    fill_alpha: float = 0.2,
) -> None:
    """Plot line with confidence interval band."""
    color = color or COLORS["primary"]
    ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=fill_alpha, linewidth=0)
    ax.plot(x, y, color=color, marker=marker, label=label, linewidth=1.5, markersize=5)


def errorbar_styled(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | tuple[np.ndarray, np.ndarray],
    *,
    color: str | None = None,
    label: str | None = None,
    marker: str = "o",
    capsize: float = 4,
) -> Any:
    """Styled error bar plot."""
    color = color or COLORS["primary"]
    return ax.errorbar(
        x, y, yerr=yerr,
        fmt=marker,
        color=color,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=1.2,
        markersize=7,
        ecolor=COLORS["ink_soft"],
        elinewidth=1.2,
        capsize=capsize,
        capthick=1.2,
        label=label,
    )


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    "COLORS",
    "PALETTE_CATEGORICAL",
    "PALETTE_DIVERGING",
    "RCPARAMS",
    "apply_theme",
    "reset_theme",
    "create_figure",
    "save_figure",
    "style_axis",
    "add_zero_line",
    "annotate_bar",
    "format_percent_axis",
    "plot_with_ci",
    "errorbar_styled",
]
