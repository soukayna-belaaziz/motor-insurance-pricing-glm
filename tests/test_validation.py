"""Tests for the cross-validation summaries and the paired bootstrap of the Gini gap."""

import numpy as np
import pandas as pd
import pytest

from src import metrics, validation


@pytest.fixture
def small_portfolio() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.normal(size=n)
    e = rng.uniform(0.2, 1.0, n)
    true = 0.1 * np.exp(0.8 * x)
    return pd.DataFrame({
        "ClaimNb": rng.poisson(true * e), "Exposure": e, "CVFold": np.arange(n) % 3 + 1,
        # GLMs see the risk driver through noise (different ranking), the "GBM" sees it exactly.
        "freq_glm": 0.1 * np.exp(0.8 * (x + rng.normal(scale=1.5, size=n))),
        "freq_glm_refined": 0.1 * np.exp(0.8 * (x + rng.normal(scale=0.7, size=n))),
        "freq_gbm": true,                           # true model
    })


def test_weighted_gini_equals_gini_on_repeated_rows():
    rng = np.random.default_rng(0)
    y, e, p = rng.poisson(0.3, 200), rng.uniform(0.1, 1, 200), rng.uniform(size=200)
    w = rng.integers(0, 4, 200)
    rep = np.repeat(np.arange(200), w)
    assert metrics.gini(y, p, e, sample_weight=w) == pytest.approx(metrics.gini(y[rep], p[rep], e[rep]), abs=1e-12)


def test_fold_metrics_and_summary(small_portfolio):
    folds = validation.fold_metrics(small_portfolio)
    assert len(folds) == 3 * len(validation.MODELS)
    summary = validation.summarise_folds(folds).set_index("model")
    gap = summary.loc["Poisson GLM v1", "gini_gap_LightGBM_minus_model_mean"]
    assert gap > 0
    assert summary.loc["Poisson GLM v1", "folds_where_LightGBM_better_gini"] == 3
    assert np.isnan(summary.loc["LightGBM"].get("gini_gap_LightGBM_minus_model_mean", np.nan))


def test_bootstrap_gini_is_reproducible_and_centred(small_portfolio):
    df = small_portfolio
    y, e = df["ClaimNb"].to_numpy(), df["Exposure"].to_numpy()
    preds = {k: df[k].to_numpy() for k in validation.MODELS}
    b1 = validation.bootstrap_gini(y, e, preds, n_boot=200, seed=3)
    b2 = validation.bootstrap_gini(y, e, preds, n_boot=200, seed=3)
    pd.testing.assert_frame_equal(b1, b2)
    full = metrics.gini(y, preds["freq_gbm"], e)
    assert abs(b1["freq_gbm"].mean() - full) < 3 * b1["freq_gbm"].std()
    gaps = validation.gap_intervals(y, e, preds, b1, [("freq_gbm", "freq_glm")])
    row = gaps.iloc[0]
    assert row["ci95_lower"] < row["test_gini_gap"] < row["ci95_upper"]
    assert row["ci95_lower"] > 0  # the true model ranks clearly better than the weak one
