"""Data cleaning for freMTPL2: caps, transformations, severity join, split.

Every function is pure (returns a new DataFrame) so it can be unit-tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def load_raw(freq_path=config.FREQ_CSV, sev_path=config.SEV_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the raw frequency (one row per policy) and severity (one row per claim) tables."""
    freq = pd.read_csv(freq_path)
    sev = pd.read_csv(sev_path)
    return freq, sev


def clean_frequency(freq: pd.DataFrame) -> pd.DataFrame:
    """Apply the a-priori caps and transformations to the policy table.

    - ``ClaimNb`` capped at ``CAP_CLAIM_NB`` (original kept as ``ClaimNbRaw``)
    - ``Exposure`` capped at one year
    - ``VehAge``, ``DrivAge``, ``BonusMalus`` capped at their thresholds
    - ``LogDensity = log(Density)`` (Density spans 1 to 27,000 inhabitants/km2)
    - categorical columns cast to ``str``
    """
    out = freq.copy()
    out["ClaimNbRaw"] = out["ClaimNb"].astype(int)
    out["ClaimNb"] = out["ClaimNb"].clip(upper=config.CAP_CLAIM_NB).astype(int)
    out["Exposure"] = out["Exposure"].clip(upper=config.CAP_EXPOSURE)
    out["VehAge"] = out["VehAge"].clip(upper=config.CAP_VEH_AGE)
    out["DrivAge"] = out["DrivAge"].clip(upper=config.CAP_DRIV_AGE)
    out["BonusMalus"] = out["BonusMalus"].clip(upper=config.CAP_BONUS_MALUS)
    out["LogDensity"] = np.log(out["Density"].astype(float))
    for col in config.CATEGORICAL:
        out[col] = out[col].astype(str)
    return out


def large_claim_threshold(claim_amounts: pd.Series, quantile: float = config.LARGE_CLAIM_QUANTILE) -> float:
    """Empirical quantile of individual claim amounts used as the large-claim cap."""
    return float(np.quantile(claim_amounts.to_numpy(), quantile))


def cap_claims(sev: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add ``ClaimAmountCapped = min(ClaimAmount, threshold)`` to the claim table."""
    out = sev.copy()
    out["ClaimAmountCapped"] = out["ClaimAmount"].clip(upper=threshold)
    return out


def large_loss_loading(sev_capped: pd.DataFrame) -> float:
    """Ratio of total incurred to total capped claim cost.

    Multiplying the capped pure premium by this factor redistributes the
    excess over the cap across all policies (flat large-loss loading).
    """
    return float(sev_capped["ClaimAmount"].sum() / sev_capped["ClaimAmountCapped"].sum())


def aggregate_severity(sev_capped: pd.DataFrame) -> pd.DataFrame:
    """Aggregate claims to policy level: number of claims, incurred and capped totals."""
    return (
        sev_capped.groupby("IDpol", as_index=False)
        .agg(
            NbClaimsSev=("ClaimAmount", "size"),
            ClaimAmountTotal=("ClaimAmount", "sum"),
            ClaimAmountCappedTotal=("ClaimAmountCapped", "sum"),
        )
    )


def join_severity(freq: pd.DataFrame, sev_policy: pd.DataFrame) -> pd.DataFrame:
    """Left-join policy-level claim amounts onto the policy table (zeros for no claim)."""
    out = freq.merge(sev_policy, on="IDpol", how="left", validate="one_to_one")
    for col in ["NbClaimsSev", "ClaimAmountTotal", "ClaimAmountCappedTotal"]:
        out[col] = out[col].fillna(0)
    out["NbClaimsSev"] = out["NbClaimsSev"].astype(int)
    return out


def split_policies(ids: pd.Series, test_share: float = config.TEST_SHARE, seed: int = config.SEED) -> pd.Series:
    """Return a boolean mask (True = test) assigning whole policies to train or test.

    The assignment is a random permutation of the unique policy ids with a
    fixed seed, so all claims of a policy end up on the same side.
    """
    unique_ids = np.sort(ids.unique())
    rng = np.random.default_rng(seed)
    n_test = int(round(test_share * len(unique_ids)))
    test_ids = set(rng.permutation(unique_ids)[:n_test].tolist())
    return ids.isin(test_ids)
