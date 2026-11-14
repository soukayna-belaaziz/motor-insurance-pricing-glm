"""Stage 3: LightGBM Poisson frequency benchmark.

Exposure is handled with ``init_score = log(Exposure)``: the booster learns
log(annual frequency) and the offset turns it into an expected count, exactly
like the GLM offset. ``Booster.predict`` excludes the init score, so it
returns the annual frequency directly.

Tuning: a small grid evaluated on a policy-level 80/20 split of the training
set with early stopping; the final model is refitted on the whole training
set with the selected number of rounds. The test set is used once, at the end.
"""

from __future__ import annotations

import itertools
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import cleaning, config, metrics

BASE_PARAMS: dict = {
    "objective": "poisson",
    "learning_rate": 0.1,
    "max_bin": 63,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 1,  # 2 threads were slower on the shared 2-CPU machine (oversubscription)
    "seed": config.SEED,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}
GRID: dict[str, list] = {"num_leaves": [15, 31, 63], "min_data_in_leaf": [200, 1000]}
MAX_ROUNDS = 1000
EARLY_STOPPING = 50


def make_features(df: pd.DataFrame, categories: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, dict]:
    """Return the GBM design matrix with pandas categoricals (train categories reused on test)."""
    X = df[config.GBM_FEATURES].copy()
    if categories is None:
        categories = {c: sorted(X[c].astype(str).unique()) for c in config.CATEGORICAL}
    for c in config.CATEGORICAL:
        X[c] = pd.Categorical(X[c].astype(str), categories=categories[c])
    return X, categories


def make_dataset(X: pd.DataFrame, y: pd.Series, exposure: pd.Series, reference: lgb.Dataset | None = None) -> lgb.Dataset:
    """LightGBM dataset with log-exposure offset as init score."""
    return lgb.Dataset(X, label=y, init_score=np.log(exposure.to_numpy()), reference=reference,
                       categorical_feature=config.CATEGORICAL, free_raw_data=False)


def tune(train: pd.DataFrame) -> pd.DataFrame:
    """Grid search with early stopping on an internal policy-level validation split."""
    is_val = cleaning.split_policies(train["IDpol"], test_share=0.2, seed=config.SEED + 1)
    tr, va = train[~is_val], train[is_val]
    X_tr, cats = make_features(tr)
    X_va, _ = make_features(va, cats)
    d_tr = make_dataset(X_tr, tr["ClaimNb"], tr["Exposure"])
    d_va = make_dataset(X_va, va["ClaimNb"], va["Exposure"], reference=d_tr)
    rows = []
    for values in itertools.product(*GRID.values()):
        params = {**BASE_PARAMS, **dict(zip(GRID.keys(), values))}
        t0 = time.time()
        booster = lgb.train(params, d_tr, num_boost_round=MAX_ROUNDS, valid_sets=[d_va],
                            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
        mu_va = booster.predict(X_va, num_iteration=booster.best_iteration) * va["Exposure"].to_numpy()
        rows.append({**dict(zip(GRID.keys(), values)), "best_iteration": booster.best_iteration,
                     "valid_mean_poisson_deviance": metrics.mean_poisson_deviance(va["ClaimNb"].to_numpy(), mu_va),
                     "seconds": round(time.time() - t0, 1)})
        print(rows[-1])
    return pd.DataFrame(rows).sort_values("valid_mean_poisson_deviance").reset_index(drop=True)


def fit_final(train: pd.DataFrame, best: pd.Series) -> tuple[lgb.Booster, dict]:
    """Refit on the full training set with the selected hyper-parameters and rounds."""
    X, cats = make_features(train)
    params = {**BASE_PARAMS, **{k: int(best[k]) for k in GRID}}
    booster = lgb.train(params, make_dataset(X, train["ClaimNb"], train["Exposure"]),
                        num_boost_round=int(best["best_iteration"]))
    return booster, cats


def shap_summary(booster: lgb.Booster, X: pd.DataFrame, n: int = 5000) -> None:
    """SHAP beeswarm summary (log-frequency scale) on a random test subsample."""
    import shap

    from src import plotting

    plotting.set_style()
    sample = X.sample(n=min(n, len(X)), random_state=config.SEED)
    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(sample)
    plt = plotting.plt
    shap.summary_plot(values, sample, show=False, max_display=len(config.GBM_FEATURES))
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    plt.title("LightGBM frequency model: SHAP values (log-frequency scale), 5,000 test policies",
              fontsize=10, loc="left")
    plotting.save(fig, config.FIG_DIR / "gbm_shap_summary.png")
    imp = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
    imp.sort_values("mean_abs_shap", ascending=False).round(4).to_csv(
        config.TABLE_DIR / "gbm_shap_importance.csv", index=False)


def main() -> None:
    """Entry point: tune, refit, predict on test, SHAP."""
    t0 = time.time()
    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    test = pd.read_csv(config.PROCESSED_DIR / "test.csv")
    results = tune(train)
    results.to_csv(config.TABLE_DIR / "gbm_tuning.csv", index=False)
    booster, cats = fit_final(train, results.iloc[0])

    X_train, _ = make_features(train, cats)
    X_test, _ = make_features(test, cats)
    train_mu = booster.predict(X_train) * train["Exposure"].to_numpy()
    pd.DataFrame({"IDpol": test["IDpol"], "freq_gbm": booster.predict(X_test)}).to_csv(
        config.PROCESSED_DIR / "gbm_test_predictions.csv", index=False)
    pd.DataFrame([{"train_sum_observed_claims": int(train["ClaimNb"].sum()),
                   "train_sum_fitted_claims": round(float(train_mu.sum()), 1)}]).to_csv(
        config.TABLE_DIR / "gbm_train_balance.csv", index=False)
    shap_summary(booster, X_test)
    print(f"GBM stage runtime: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
