"""One-way analyses on the training set: observed frequency and exposure by level.

These plots are the evidence for the band limits in ``src/features.py``: the
dots show observed frequency per raw value, the orange steps the frequency of
the chosen band, and the lower panel the exposure behind each value.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, features, plotting


def oneway_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Exposure, claim count and observed frequency (claims / exposure) by level of ``by``."""
    out = (
        df.groupby(by, observed=True)
        .agg(exposure=("Exposure", "sum"), claims=("ClaimNb", "sum"), policies=("ClaimNb", "size"))
        .reset_index()
    )
    out["frequency"] = out["claims"] / out["exposure"]
    return out


def _plot_banded(ax_f: plt.Axes, ax_e: plt.Axes, df: pd.DataFrame, var: str) -> None:
    raw = oneway_table(df, var)
    banded = oneway_table(df, f"{var}Band").set_index(f"{var}Band")
    edges, labels = features.BANDS[var]
    overall = df["ClaimNb"].sum() / df["Exposure"].sum()

    min_exposure = 500 if var == "BonusMalus" else 200  # hide values with thin exposure
    reliable = raw["exposure"] >= min_exposure
    ax_f.plot(raw.loc[reliable, var], raw.loc[reliable, "frequency"], "o", ms=3.5,
              color=plotting.BLUE, label="observed, raw value")
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        x0 = max(lo + 0.5, raw[var].min() - 0.5)
        ax_f.hlines(banded.loc[lab, "frequency"], x0, hi + 0.5, color=plotting.ORANGE, lw=2)
    ax_f.plot([], [], color=plotting.ORANGE, lw=2, label="band average")
    ax_f.axhline(overall, color=plotting.GREY, lw=1, ls="--")
    ax_f.set_title(var)
    ax_f.set_ylabel("claims / exposure-year")
    ax_e.bar(raw[var], raw["exposure"] / 1000, width=0.8, color=plotting.GREY)
    ax_e.set_ylabel("exposure ('000 yrs)")
    ax_e.set_xlabel(f"{var} (capped)")
    if var == "BonusMalus":
        ax_e.set_yscale("log")
        ax_e.set_ylabel("exposure ('000 yrs, log)")


def _plot_categorical(ax_f: plt.Axes, ax_e: plt.Axes, df: pd.DataFrame, var: str, order: list[str]) -> None:
    tab = oneway_table(df, var).set_index(var).loc[order]
    x = np.arange(len(order))
    overall = df["ClaimNb"].sum() / df["Exposure"].sum()
    ax_f.plot(x, tab["frequency"], "o-", ms=4, color=plotting.BLUE)
    ax_f.axhline(overall, color=plotting.GREY, lw=1, ls="--")
    ax_f.set_title(var)
    ax_f.set_ylabel("claims / exposure-year")
    ax_e.bar(x, tab["exposure"] / 1000, width=0.8, color=plotting.GREY)
    ax_e.set_xticks(x, order, rotation=90 if len(order) > 6 else 0)
    ax_e.set_ylabel("exposure ('000 yrs)")


def plot_oneways(train: pd.DataFrame) -> None:
    """Write the two one-way figures and the one-way tables."""
    plotting.set_style()
    banded_vars = ["DrivAge", "VehAge", "BonusMalus", "VehPower"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex="col",
                             gridspec_kw={"height_ratios": [2.2, 1]})
    for j, var in enumerate(banded_vars):
        _plot_banded(axes[0, j], axes[1, j], train, var)
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("One-way analysis (train): observed frequency by raw value and by chosen band",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plotting.save(fig, config.FIG_DIR / "oneway_banded.png")

    tmp = train.assign(LogDensityBin=(np.floor(train["LogDensity"] * 2) / 2).clip(0.5, 10).map("{:.1f}".format))
    dens_order = sorted(tmp["LogDensityBin"].unique(), key=float)
    brand_order = sorted(train["VehBrand"].unique(), key=lambda s: int(s[1:]))
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex="col",
                             gridspec_kw={"height_ratios": [2.2, 1]})
    _plot_categorical(axes[0, 0], axes[1, 0], tmp, "LogDensityBin", dens_order)
    axes[0, 0].set_title("log(Density), bins of 0.5")
    _plot_categorical(axes[0, 1], axes[1, 1], train, "Area", list("ABCDEF"))
    _plot_categorical(axes[0, 2], axes[1, 2], train, "VehGas", ["Diesel", "Regular"])
    _plot_categorical(axes[0, 3], axes[1, 3], train, "VehBrand", brand_order)
    fig.suptitle("One-way analysis (train): density, area, fuel and brand",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plotting.save(fig, config.FIG_DIR / "oneway_other.png")

    tables = [oneway_table(train, f"{v}Band").rename(columns={f"{v}Band": "level"}).assign(variable=v)
              for v in banded_vars]
    tables += [oneway_table(train, v).rename(columns={v: "level"}).assign(variable=v)
               for v in ["Area", "VehGas", "VehBrand", "Region"]]
    pd.concat(tables)[["variable", "level", "policies", "exposure", "claims", "frequency"]].round(4).to_csv(
        config.TABLE_DIR / "oneway_tables.csv", index=False)


def main() -> None:
    """Entry point: one-way analyses on the processed training set."""
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(config.PROCESSED_DIR / "train.csv", dtype={"VehPowerBand": str, "BonusMalusBand": str})
    plot_oneways(train)


if __name__ == "__main__":
    main()
