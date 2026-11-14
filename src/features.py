"""Rating-factor banding for the GLM.

Band limits come from the one-way analyses (observed claim frequency per unit
of exposure by raw value, see ``src/oneway.py`` and the README): adjacent
values with a similar observed frequency and/or thin exposure are merged, and
breaks are placed where the frequency changes level. The GBM uses the raw
(capped) variables instead.
"""

from __future__ import annotations

import math

import pandas as pd

# (upper bounds inclusive, labels). Values are already capped in cleaning.
BANDS: dict[str, tuple[list[float], list[str]]] = {
    "DrivAge": (
        [17, 20, 22, 25, 30, 40, 50, 60, 70, 90],
        ["18-20", "21-22", "23-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71-90"],
    ),
    "VehAge": (
        [-1, 3, 11, 15, 20],
        ["0-3", "4-11", "12-15", "16-20"],
    ),
    "BonusMalus": (
        [49, 50, 59, 69, 89, 99, 100, 150],
        ["50", "51-59", "60-69", "70-89", "90-99", "100", "101-150"],
    ),
    "VehPower": (
        [3, 4, 6, 8, 15],
        ["4", "5-6", "7-8", "9-15"],
    ),
}


def band(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    """Cut a numeric series into right-closed intervals ``(edges[i], edges[i+1]]``.

    Raises ``ValueError`` if any value falls outside the bands, so that a new
    value outside the tariff structure is never silently dropped.
    """
    out = pd.cut(values, bins=edges, labels=labels, right=True)
    if out.isna().any():
        bad = values[out.isna()].unique()[:5]
        raise ValueError(f"{values.name}: values outside band limits, e.g. {bad}")
    return out.astype(str)


def claim_free_ladder(start: float = 1.00, floor: float = 0.50, step: float = 0.95) -> list[int]:
    """Bonus-malus values reachable from ``start`` with claim-free years only (x 100).

    French bonus-malus rule (CRM): each claim-free year multiplies the
    coefficient by 0.95, truncated to two decimals, with a floor at 0.50.
    From the entry level 1.00 this gives 100, 95, 90, 85, 80, 76, 72, 68, 64,
    60, 57, 54, 51, 50. A value below 100 that is *not* on this path can only
    be reached after a malus (+25% per at-fault claim), i.e. it signals a
    recent claim.
    """
    values = [start]
    while values[-1] > floor:
        nxt = math.floor(values[-1] * step * 100 + 1e-9) / 100
        values.append(max(floor, nxt))
    return [int(round(100 * v)) for v in values]


def add_bm_history_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``BMOffLadder`` = 1 if BonusMalus < 100 and not on the claim-free ladder.

    This is a deterministic function of BonusMalus (no new data): it splits
    each bonus-malus band into drivers who reached their level claim-free and
    drivers who carry a recent malus. The LightGBM benchmark found this
    structure on its own (double-lift and SHAP, see README).
    """
    out = df.copy()
    ladder = set(claim_free_ladder())
    out["BMOffLadder"] = ((out["BonusMalus"] < 100) & ~out["BonusMalus"].isin(ladder)).astype(int)
    return out


def add_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``<var>Band`` columns for every banded rating factor and the BM history flag."""
    out = df.copy()
    for var, (edges, labels) in BANDS.items():
        out[f"{var}Band"] = band(out[var], edges, labels)
    return add_bm_history_flag(out)


def base_level(levels: pd.Series, exposure: pd.Series) -> str:
    """Reference level for a factor = level with the largest exposure (actuarial convention)."""
    return str(exposure.groupby(levels).sum().idxmax())


def ordered_levels(var: str) -> list[str]:
    """Band labels in natural order, for plotting."""
    return BANDS[var][1]

