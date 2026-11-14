"""Evaluation metrics for claim-frequency models.

Conventions: ``y`` is the observed claim count, ``mu`` the *expected count*
(predicted annual frequency x exposure). Ranking metrics use the predicted
frequency and weight by exposure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Total Poisson deviance 2 * sum(y log(y / mu) - (y - mu)), with 0 log 0 = 0."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if y.shape != mu.shape:
        raise ValueError("y and mu must have the same shape")
    if np.any(mu <= 0):
        raise ValueError("mu must be strictly positive")
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2.0 * np.sum(term - (y - mu)))


def mean_poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Average Poisson deviance per policy."""
    return poisson_deviance(y, mu) / len(y)


def lorenz_curve(
    y: np.ndarray, pred_freq: np.ndarray, exposure: np.ndarray, sample_weight: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Ordered Lorenz (concentration) curve.

    Policies are sorted from the lowest to the highest predicted frequency;
    returns the cumulative share of exposure (x) and of observed claims (y),
    both starting at 0. ``sample_weight`` (e.g. bootstrap counts) multiplies
    both exposure and claims, which is the same as repeating each row.
    """
    order = np.argsort(np.asarray(pred_freq), kind="mergesort")
    w = np.ones(len(order)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    exp_sorted = (np.asarray(exposure, dtype=float) * w)[order]
    y_sorted = (np.asarray(y, dtype=float) * w)[order]
    x = np.concatenate([[0.0], np.cumsum(exp_sorted) / exp_sorted.sum()])
    c = np.concatenate([[0.0], np.cumsum(y_sorted) / y_sorted.sum()])
    return x, c


def gini(
    y: np.ndarray, pred_freq: np.ndarray, exposure: np.ndarray, sample_weight: np.ndarray | None = None
) -> float:
    """Gini index = 1 - 2 * area under the ordered Lorenz curve.

    0 for a model that does not rank risks better than random; larger is
    better. Ties in the prediction are broken by input order.
    """
    x, c = lorenz_curve(y, pred_freq, exposure, sample_weight)
    area = np.trapezoid(c, x)
    return float(1.0 - 2.0 * area)


def double_lift(
    y: np.ndarray,
    freq_a: np.ndarray,
    freq_b: np.ndarray,
    exposure: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Double-lift table comparing model A and model B.

    Policies are sorted by the ratio freq_a / freq_b and cut into ``n_bins``
    buckets of equal exposure. For each bucket the observed frequency and the
    two predicted frequencies are reported, each divided by the overall
    observed frequency. The better model tracks the observed line, especially
    in the outer buckets where the two models disagree most.
    """
    df = pd.DataFrame(
        {"y": y, "a": np.asarray(freq_a) * exposure, "b": np.asarray(freq_b) * exposure, "e": exposure,
         "ratio": np.asarray(freq_a) / np.asarray(freq_b)}
    ).sort_values("ratio", kind="mergesort")
    cum = df["e"].cumsum() / df["e"].sum()
    df["bin"] = np.minimum((cum * n_bins).apply(np.ceil).astype(int), n_bins).clip(lower=1)
    overall = df["y"].sum() / df["e"].sum()
    out = df.groupby("bin").agg(exposure=("e", "sum"), observed=("y", "sum"), pred_a=("a", "sum"),
                                pred_b=("b", "sum"), ratio_min=("ratio", "min"), ratio_max=("ratio", "max"))
    for col in ["observed", "pred_a", "pred_b"]:
        out[f"{col}_index"] = out[col] / out["exposure"] / overall
    return out.reset_index()
