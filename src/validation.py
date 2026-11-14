"""Stage 5: how stable is the GLM-GBM comparison?

1. 5-fold cross-validation on the training set. Folds are assigned by policy
   in ``src/prepare.py``. The GLM out-of-fold predictions come from
   ``R/fit_freq_extensions.R`` (same terms, refitted on each set of 4 folds);
   here LightGBM is refitted on each set of 4 folds with the hyper-parameters
   and number of rounds selected in ``src/gbm.py``.
2. Paired bootstrap of the test set: policies are resampled with replacement
   and the Gini index of every model is recomputed on the *same* resample, so
   the confidence interval is for the Gini gap between models.

Outputs
-------
reports/tables/cv_folds.csv            deviance and Gini per fold and model
reports/tables/cv_summary.csv          mean and sd across folds, paired gaps
reports/tables/bootstrap_gini_gap.csv  test Gini gaps with 95% percentile intervals
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config, evaluate, gbm, metrics

MODELS = {
    "freq_glm": "Poisson GLM v1",
    "freq_glm_refined": "Poisson GLM refined",
    "freq_gbm": "LightGBM",
}


def gbm_oof_predictions(train: pd.DataFrame, best: pd.Series) -> pd.Series:
    """Out-of-fold LightGBM frequency predictions (hyper-parameters fixed, one fit per fold)."""
    params = {**gbm.BASE_PARAMS, **{k: int(best[k]) for k in gbm.GRID}}
    oof = pd.Series(np.nan, index=train.index)
    for k in sorted(train["CVFold"].unique()):
        t0 = time.time()
        fit, hold = train[train["CVFold"] != k], train[train["CVFold"] == k]
        X_fit, cats = gbm.make_features(fit)
        X_hold, _ = gbm.make_features(hold, cats)
        booster = lgb.train(params, gbm.make_dataset(X_fit, fit["ClaimNb"], fit["Exposure"]),
                            num_boost_round=int(best["best_iteration"]))
        oof.loc[hold.index] = booster.predict(X_hold)
        print(f"  GBM fold {k}: {time.time() - t0:.1f} s", flush=True)
    return oof


def fold_metrics(df: pd.DataFrame, fold_col: str = "CVFold") -> pd.DataFrame:
    """Mean Poisson deviance and Gini of every model in ``MODELS`` on each fold."""
    rows = []
    for k, s in df.groupby(fold_col):
        y, e = s["ClaimNb"].to_numpy(), s["Exposure"].to_numpy()
        for col, name in MODELS.items():
            rows.append({"fold": int(k), "model": name, "policies": len(s),
                         "mean_poisson_deviance": metrics.mean_poisson_deviance(y, s[col].to_numpy() * e),
                         "gini": metrics.gini(y, s[col].to_numpy(), e)})
    return pd.DataFrame(rows)


def summarise_folds(folds: pd.DataFrame, reference: str = "LightGBM") -> pd.DataFrame:
    """Mean and sd across folds per model, plus the paired gap to ``reference`` (same folds)."""
    rows = []
    ref = folds[folds["model"] == reference].set_index("fold")
    for name, s in folds.groupby("model", sort=False):
        s = s.set_index("fold")
        row = {"model": name}
        for m in ["mean_poisson_deviance", "gini"]:
            row[f"{m}_mean"] = s[m].mean()
            row[f"{m}_sd"] = s[m].std(ddof=1)
            if name != reference:
                gap = ref[m] - s[m]
                row[f"{m}_gap_{reference}_minus_model_mean"] = gap.mean()
                row[f"{m}_gap_{reference}_minus_model_sd"] = gap.std(ddof=1)
                row[f"folds_where_{reference}_better_{m}"] = int(
                    (gap < 0).sum() if m == "mean_poisson_deviance" else (gap > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_gini(y: np.ndarray, exposure: np.ndarray, preds: dict[str, np.ndarray],
                   n_boot: int = config.N_BOOTSTRAP, seed: int = config.SEED) -> pd.DataFrame:
    """Gini of each model on ``n_boot`` paired bootstrap resamples of the rows.

    A resample is represented by integer counts (how many times each row is
    drawn); the Gini of a resample is then the weighted Gini with those
    counts, so each model's sort order is computed only once.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    orders = {k: np.argsort(np.asarray(p), kind="mergesort") for k, p in preds.items()}
    y = np.asarray(y, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    out = {k: np.empty(n_boot) for k in preds}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        for k, o in orders.items():
            ew, yw = (exposure * w)[o], (y * w)[o]
            x = np.concatenate([[0.0], np.cumsum(ew) / ew.sum()])
            c = np.concatenate([[0.0], np.cumsum(yw) / yw.sum()])
            out[k][b] = 1.0 - 2.0 * np.trapezoid(c, x)
    return pd.DataFrame(out)


def gap_intervals(y: np.ndarray, exposure: np.ndarray, preds: dict[str, np.ndarray],
                  boot: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Point estimate and 95% percentile interval of Gini(a) - Gini(b) for each pair."""
    rows = []
    for a, b in pairs:
        gap = boot[a] - boot[b]
        rows.append({"comparison": f"{MODELS[a]} minus {MODELS[b]}",
                     "test_gini_gap": metrics.gini(y, preds[a], exposure) - metrics.gini(y, preds[b], exposure),
                     "bootstrap_sd": gap.std(ddof=1),
                     "ci95_lower": gap.quantile(0.025), "ci95_upper": gap.quantile(0.975),
                     "share_of_resamples_gap_positive": (gap > 0).mean(),
                     "replicates": len(gap)})
    return pd.DataFrame(rows)


def main() -> None:
    """Entry point: GBM out-of-fold fits, CV tables, test bootstrap."""
    t0 = time.time()
    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    best = pd.read_csv(config.TABLE_DIR / "gbm_tuning.csv").iloc[0]
    train["freq_gbm"] = gbm_oof_predictions(train, best)
    glm_oof = pd.read_csv(config.PROCESSED_DIR / "glm_cv_predictions.csv")
    cv = train.merge(glm_oof.drop(columns="CVFold"), on="IDpol", validate="one_to_one")
    if len(cv) != len(train):
        raise ValueError("GLM out-of-fold predictions do not cover the training set")

    folds = fold_metrics(cv)
    folds.round(5).to_csv(config.TABLE_DIR / "cv_folds.csv", index=False)
    summary = summarise_folds(folds)
    summary.round(5).to_csv(config.TABLE_DIR / "cv_summary.csv", index=False)
    print(summary.round(4).T.to_string())

    test = evaluate.load_test_predictions()
    y, e = test["ClaimNb"].to_numpy(), test["Exposure"].to_numpy()
    preds = {k: test[k].to_numpy() for k in MODELS}
    boot = bootstrap_gini(y, e, preds)
    gaps = gap_intervals(y, e, preds, boot, [("freq_gbm", "freq_glm"), ("freq_gbm", "freq_glm_refined"),
                                             ("freq_glm_refined", "freq_glm")])
    gaps.round(5).to_csv(config.TABLE_DIR / "bootstrap_gini_gap.csv", index=False)
    print(gaps.round(4).to_string(index=False))
    print(f"Validation stage runtime: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
