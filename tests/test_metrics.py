"""Tests for the evaluation metrics and the Poisson-GLM balance property."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance as sk_mean_poisson_deviance

from src import metrics


def test_poisson_deviance_matches_sklearn():
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.01, 0.5, 1000)
    y = rng.poisson(mu)
    assert metrics.mean_poisson_deviance(y, mu) == pytest.approx(sk_mean_poisson_deviance(y, mu), rel=1e-10)


def test_poisson_deviance_zero_for_perfect_fit_and_positive_otherwise():
    y = np.array([0, 1, 2, 5])
    assert metrics.poisson_deviance(y, np.maximum(y, 1e-12)) == pytest.approx(0.0, abs=1e-9)
    assert metrics.poisson_deviance(y, np.full(4, 2.0)) > 0


def test_poisson_deviance_rejects_non_positive_mu():
    with pytest.raises(ValueError):
        metrics.poisson_deviance(np.array([1, 0]), np.array([1.0, 0.0]))


def test_gini_perfect_and_random_ordering():
    rng = np.random.default_rng(1)
    n = 200_000
    true_freq = rng.gamma(2.0, 0.05, n)
    e = rng.uniform(0.1, 1.0, n)
    y = rng.poisson(true_freq * e)
    g_true = metrics.gini(y, true_freq, e)
    g_random = metrics.gini(y, rng.uniform(size=n), e)
    g_inverse = metrics.gini(y, -true_freq, e)
    assert g_true > 0.2
    assert abs(g_random) < 0.02
    assert g_inverse == pytest.approx(-g_true, abs=0.02)


def test_lorenz_curve_endpoints():
    x, c = metrics.lorenz_curve(np.array([0, 1, 3]), np.array([0.1, 0.2, 0.3]), np.array([1.0, 1.0, 2.0]))
    assert x[0] == 0 and c[0] == 0
    assert x[-1] == pytest.approx(1.0) and c[-1] == pytest.approx(1.0)
    assert np.all(np.diff(x) >= 0) and np.all(np.diff(c) >= 0)


def test_double_lift_bins_and_indices():
    rng = np.random.default_rng(2)
    n = 50_000
    e = rng.uniform(0.1, 1, n)
    fa, fb = rng.uniform(0.05, 0.1, n), rng.uniform(0.05, 0.1, n)
    y = rng.poisson(fa * e)
    tab = metrics.double_lift(y, fa, fb, e, n_bins=10)
    assert tab["bin"].tolist() == list(range(1, 11))
    assert tab["exposure"].sum() == pytest.approx(e.sum())
    assert np.allclose(tab["exposure"], e.sum() / 10, rtol=0.01)
    # Observed index is a weighted average of 1 across bins.
    assert np.average(tab["observed_index"], weights=tab["exposure"]) == pytest.approx(1.0)
    assert tab["ratio_min"].is_monotonic_increasing


def test_poisson_glm_canonical_link_is_balanced():
    """Unpenalised Poisson GLM with log link and intercept: sum(fitted) = sum(observed)."""
    rng = np.random.default_rng(3)
    n = 20_000
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.integers(0, 2, n)})
    e = rng.uniform(0.1, 1.0, n)
    y = rng.poisson(e * np.exp(-2.5 + 0.3 * X["x1"] + 0.4 * X["x2"]))
    # Frequency target with exposure weights is equivalent to a log-exposure offset.
    model = PoissonRegressor(alpha=0.0, max_iter=1000, tol=1e-10).fit(X, y / e, sample_weight=e)
    fitted_counts = model.predict(X) * e
    assert fitted_counts.sum() == pytest.approx(y.sum(), rel=1e-6)
