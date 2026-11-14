"""Shared matplotlib style: one palette, recessive axes, thin marks."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots (validated colour-blind-safe order), plus neutrals.
BLUE = "#2a78d6"     # GLM / observed series
ORANGE = "#eb6834"   # GBM / banded series
AQUA = "#1baf7a"     # actual / third series
GREY = "#a3a29c"     # exposure bars, reference lines
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"


def set_style() -> None:
    """Apply the project-wide rcParams."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GREY,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e5e0",
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "font.size": 9,
        }
    )


def save(fig: plt.Figure, path) -> None:
    """Save a figure at 150 dpi and close it."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
