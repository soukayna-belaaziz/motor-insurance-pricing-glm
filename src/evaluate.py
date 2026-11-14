"""Stage 4: compare GLM and GBM on the test set; pure-premium balance.

Outputs
-------
reports/tables/test_metrics.csv          deviance, Gini, balance per model
reports/tables/double_lift.csv           GBM vs GLM double-lift table
reports/tables/pure_premium_balance.csv  predicted vs actual claim cost (test)
reports/tables/severity_test_check.csv   severity GLM on test claims
reports/figures/lorenz_curves.png, double_lift.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, metrics, plotting


def load_test_predictions() -> pd.DataFrame:
    """Test policies joined with GLM and GBM predictions (by IDpol)."""
    test = pd.read_csv(config.PROCESSED_DIR / "test.csv")
    glm = pd.read_csv(config.PROCESSED_DIR / "glm_test_predictions.csv")
    gbm = pd.read_csv(config.PROCESSED_DIR / "gbm_test_predictions.csv")
    out = test.merge(glm, on="IDpol", validate="one_to_one").merge(gbm, on="IDpol", validate="one_to_one")
    if len(out) != len(test):
        raise ValueError("prediction files do not cover the test set")
    return out


def frequency_metrics(df: pd.DataFrame, train_freq: float) -> pd.DataFrame:
    """Deviance, D^2, Gini and balance for the null model, the GLM and the GBM."""
    y, e = df["ClaimNb"].to_numpy(), df["Exposure"].to_numpy()
    preds = {
        "Null (train mean frequency)": np.full(len(df), train_freq),
        "Poisson GLM (R)": df["freq_glm"].to_numpy(),
        "LightGBM Poisson (Python)": df["freq_gbm"].to_numpy(),
    }
    dev_null = metrics.poisson_deviance(y, preds["Null (train mean frequency)"] * e)
    rows = []
    for name, f in preds.items():
        dev = metrics.poisson_deviance(y, f * e)
        rows.append({
            "model": name,
            "mean_poisson_deviance": dev / len(y),
            "deviance_explained_vs_null_pct": 100 * (1 - dev / dev_null),
            "gini": metrics.gini(y, f, e) if name != "Null (train mean frequency)" else 0.0,
            "predicted_claims": float((f * e).sum()),
            "observed_claims": int(y.sum()),
            "predicted_over_observed": float((f * e).sum() / y.sum()),
        })
    return pd.DataFrame(rows)


def plot_lorenz(df: pd.DataFrame) -> None:
    """Ordered Lorenz curves of the two models on the test set."""
    fig, ax = plotting.plt.subplots(figsize=(5.5, 5))
    y, e = df["ClaimNb"].to_numpy(), df["Exposure"].to_numpy()
    for col, name, color in [("freq_glm", "Poisson GLM", plotting.BLUE), ("freq_gbm", "LightGBM", plotting.ORANGE)]:
        x, c = metrics.lorenz_curve(y, df[col].to_numpy(), e)
        idx = np.linspace(0, len(x) - 1, 2000).astype(int)
        ax.plot(x[idx], c[idx], color=color, label=f"{name} (Gini {metrics.gini(y, df[col].to_numpy(), e):.3f})")
    ax.plot([0, 1], [0, 1], color=plotting.GREY, lw=1, ls="--", label="random ranking")
    ax.set_xlabel("cumulative share of exposure (sorted by predicted frequency, low to high)")
    ax.set_ylabel("cumulative share of observed claims")
    ax.set_title("Ordered Lorenz curves, test set", loc="left")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plotting.save(fig, config.FIG_DIR / "lorenz_curves.png")


def plot_double_lift(tab: pd.DataFrame) -> None:
    """Double-lift chart: buckets of GBM/GLM ratio, observed vs both models."""
    fig, ax = plotting.plt.subplots(figsize=(7.5, 4.5))
    x = tab["bin"]
    ax.plot(x, tab["observed_index"], "o-", color=plotting.INK, ms=5, label="observed")
    ax.plot(x, tab["pred_b_index"], "o-", color=plotting.BLUE, ms=5, label="Poisson GLM")
    ax.plot(x, tab["pred_a_index"], "o-", color=plotting.ORANGE, ms=5, label="LightGBM")
    ax.axhline(1.0, color=plotting.GREY, lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xlabel("decile of LightGBM / GLM predicted ratio (equal exposure)")
    ax.set_ylabel("frequency / overall observed frequency")
    ax.set_title("Double-lift chart, test set: where the two models disagree, who is right?", loc="left")
    ax.legend(loc="upper left")
    plotting.save(fig, config.FIG_DIR / "double_lift.png")


def pure_premium_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Total predicted GLM pure premium vs actual claim cost on the test set."""
    pred = float((df["pure_premium_glm"] * df["Exposure"]).sum())
    capped = float(df["ClaimAmountCappedTotal"].sum())
    incurred = float(df["ClaimAmountTotal"].sum())
    ll = pd.read_csv(config.TABLE_DIR / "large_loss.csv")["large_loss_loading"].iloc[0]
    return pd.DataFrame([
        {"quantity": "predicted pure premium x exposure (GLM, incl. large-loss loading)", "eur": pred},
        {"quantity": "actual incurred claims (uncapped)", "eur": incurred},
        {"quantity": "actual capped claims", "eur": capped},
        {"quantity": "actual capped claims x large-loss loading", "eur": capped * ll},
        {"quantity": "test incurred / capped (test-implied loading)", "eur": incurred / capped},
        {"quantity": "predicted / actual incurred", "eur": pred / incurred},
        {"quantity": "predicted / (actual capped x loading)", "eur": pred / (capped * ll)},
    ])


def severity_check(df: pd.DataFrame) -> pd.DataFrame:
    """Claim-weighted mean predicted vs observed capped severity on test claims."""
    s = df[df["NbClaimsSev"] > 0]
    obs = s["ClaimAmountCappedTotal"].sum() / s["NbClaimsSev"].sum()
    pred = float(np.average(s["sev_glm"], weights=s["NbClaimsSev"]))
    return pd.DataFrame([{"test_claims": int(s["NbClaimsSev"].sum()), "observed_mean_capped_severity": obs,
                          "predicted_mean_capped_severity": pred, "ratio": pred / obs}])


def twin_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Metrics split by whether a test policy has an identical rating profile in train.

    freMTPL2 contains many policies with identical rating factors (often the
    same risk over several contract periods). A policy-level split can put
    such "twins" on both sides; if the models only won on twins, the test
    score would be optimistic. Reported separately to check this.
    """
    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    keys = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"]
    train_keys = set(map(tuple, train[keys].astype(str).to_numpy()))
    has_twin = np.array([tuple(r) in train_keys for r in df[keys].astype(str).to_numpy()])
    rows = []
    for label, mask in [("test policy has a twin in train", has_twin), ("no twin in train", ~has_twin)]:
        s = df[mask]
        y, e = s["ClaimNb"].to_numpy(), s["Exposure"].to_numpy()
        row = {"subset": label, "policies": int(mask.sum()), "observed_frequency": y.sum() / e.sum()}
        for col, name in [("freq_glm", "glm"), ("freq_gbm", "gbm")]:
            row[f"mean_deviance_{name}"] = metrics.mean_poisson_deviance(y, s[col].to_numpy() * e)
            row[f"gini_{name}"] = metrics.gini(y, s[col].to_numpy(), e)
        rows.append(row)
    return pd.DataFrame(rows)


def top_bucket_profile(df: pd.DataFrame, dl: pd.DataFrame) -> pd.DataFrame:
    """Most frequent BonusMalus values in the top double-lift bucket (GBM >> GLM) vs the rest."""
    ratio = df["freq_gbm"] / df["freq_glm"]
    top = ratio >= dl["ratio_min"].iloc[-1]
    rows = []
    for value in df.loc[top, "BonusMalus"].value_counts().index[:5]:
        rows.append({"BonusMalus": int(value),
                     "share_top_bucket_pct": 100 * (df.loc[top, "BonusMalus"] == value).mean(),
                     "share_other_buckets_pct": 100 * (df.loc[~top, "BonusMalus"] == value).mean()})
    return pd.DataFrame(rows)


def main() -> None:
    """Entry point for the evaluation stage."""
    plotting.set_style()
    df = load_test_predictions()
    split = pd.read_csv(config.TABLE_DIR / "split_summary.csv").set_index("set")
    train_freq = split.loc["train", "claims"] / split.loc["train", "exposure_years"]

    fm = frequency_metrics(df, train_freq)
    fm.round(5).to_csv(config.TABLE_DIR / "test_metrics.csv", index=False)
    print(fm.round(4).to_string(index=False))

    dl = metrics.double_lift(df["ClaimNb"].to_numpy(), df["freq_gbm"].to_numpy(), df["freq_glm"].to_numpy(),
                             df["Exposure"].to_numpy())
    plot_double_lift(dl)
    dl.rename(columns=lambda c: c.replace("pred_a", "pred_gbm").replace("pred_b", "pred_glm")).round(4).to_csv(
        config.TABLE_DIR / "double_lift.csv", index=False)
    plot_lorenz(df)
    top_bucket_profile(df, dl).round(2).to_csv(config.TABLE_DIR / "double_lift_top_bucket_bm.csv", index=False)

    tw = twin_robustness(df)
    tw.round(4).to_csv(config.TABLE_DIR / "twin_robustness.csv", index=False)
    print(tw.round(4).to_string(index=False))

    pb = pure_premium_balance(df)
    pb.round(4).to_csv(config.TABLE_DIR / "pure_premium_balance.csv", index=False)
    print(pb.round(3).to_string(index=False))
    sc = severity_check(df)
    sc.round(3).to_csv(config.TABLE_DIR / "severity_test_check.csv", index=False)
    print(sc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
