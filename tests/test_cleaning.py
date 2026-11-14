"""Tests for the cleaning and banding functions."""

import numpy as np
import pandas as pd
import pytest

from src import cleaning, config, features


@pytest.fixture
def raw_policies() -> pd.DataFrame:
    return pd.DataFrame({
        "IDpol": [1, 2, 3, 4],
        "ClaimNb": [0, 1, 11, 4],
        "Exposure": [0.5, 1.0, 1.7, 0.01],
        "VehPower": [4, 6, 15, 9],
        "VehAge": [0, 25, 3, 100],
        "DrivAge": [18, 45, 99, 90],
        "BonusMalus": [50, 230, 100, 150],
        "VehBrand": ["B1", "B12", "B2", "B3"],
        "VehGas": ["Diesel", "Regular", "Regular", "Diesel"],
        "Area": ["A", "C", "F", "D"],
        "Density": [1, 1000, 27000, 54],
        "Region": ["Centre", "Bretagne", "Ile-de-France", "Corse"],
    })


def test_clean_frequency_caps(raw_policies):
    out = cleaning.clean_frequency(raw_policies)
    assert out["ClaimNb"].max() == config.CAP_CLAIM_NB
    assert out["ClaimNbRaw"].tolist() == [0, 1, 11, 4]
    assert out["Exposure"].max() == pytest.approx(1.0)
    assert out["Exposure"].tolist()[0] == pytest.approx(0.5)  # untouched below the cap
    assert out["VehAge"].max() == config.CAP_VEH_AGE
    assert out["DrivAge"].max() == config.CAP_DRIV_AGE
    assert out["BonusMalus"].tolist() == [50, 150, 100, 150]
    assert out["LogDensity"].iloc[0] == pytest.approx(0.0)
    assert out["LogDensity"].iloc[2] == pytest.approx(np.log(27000))


def test_clean_frequency_does_not_mutate_input(raw_policies):
    before = raw_policies.copy()
    cleaning.clean_frequency(raw_policies)
    pd.testing.assert_frame_equal(raw_policies, before)


def test_large_claim_capping_and_loading():
    sev = pd.DataFrame({"IDpol": [1, 1, 2, 3], "ClaimAmount": [100.0, 200.0, 300.0, 10_000.0]})
    capped = cleaning.cap_claims(sev, threshold=1000.0)
    assert capped["ClaimAmountCapped"].tolist() == [100.0, 200.0, 300.0, 1000.0]
    loading = cleaning.large_loss_loading(capped)
    assert loading == pytest.approx(10_600 / 1_600)
    # Loading x capped total recovers the incurred total exactly.
    assert loading * capped["ClaimAmountCapped"].sum() == pytest.approx(sev["ClaimAmount"].sum())


def test_large_claim_threshold_is_quantile():
    s = pd.Series(np.arange(1, 1001, dtype=float))
    assert cleaning.large_claim_threshold(s, 0.5) == pytest.approx(500.5)


def test_aggregate_and_join_severity():
    freq = pd.DataFrame({"IDpol": [1, 2, 3], "ClaimNb": [2, 0, 1]})
    sev = cleaning.cap_claims(pd.DataFrame({"IDpol": [1, 1, 3], "ClaimAmount": [10.0, 5000.0, 7.0]}), 1000.0)
    out = cleaning.join_severity(freq, cleaning.aggregate_severity(sev))
    assert out["NbClaimsSev"].tolist() == [2, 0, 1]
    assert out["ClaimAmountTotal"].tolist() == [5010.0, 0.0, 7.0]
    assert out["ClaimAmountCappedTotal"].tolist() == [1010.0, 0.0, 7.0]


def test_split_is_deterministic_disjoint_and_sized():
    ids = pd.Series(np.arange(10_000))
    m1 = cleaning.split_policies(ids, test_share=0.2, seed=42)
    m2 = cleaning.split_policies(ids, test_share=0.2, seed=42)
    m3 = cleaning.split_policies(ids, test_share=0.2, seed=7)
    assert m1.equals(m2)
    assert not m1.equals(m3)
    assert m1.sum() == 2000


def test_split_keeps_all_rows_of_a_policy_together():
    ids = pd.Series([1, 1, 2, 3, 3, 3, 4, 5, 6, 7])
    mask = cleaning.split_policies(ids, test_share=0.3, seed=0)
    assert mask.groupby(ids).nunique().max() == 1


def test_bands_cover_all_capped_values(raw_policies):
    out = features.add_bands(cleaning.clean_frequency(raw_policies))
    assert out["DrivAgeBand"].tolist() == ["18-20", "41-50", "71-90", "71-90"]
    assert out["VehAgeBand"].tolist() == ["0-3", "16-20", "0-3", "16-20"]
    assert out["BonusMalusBand"].tolist() == ["50", "101-150", "100", "101-150"]
    assert out["VehPowerBand"].tolist() == ["4", "5-6", "9-15", "9-15"]


def test_band_raises_outside_limits():
    edges, labels = features.BANDS["DrivAge"]
    with pytest.raises(ValueError):
        features.band(pd.Series([17], name="DrivAge"), edges, labels)


def test_base_level_is_highest_exposure():
    levels = pd.Series(["a", "b", "b", "c"])
    assert features.base_level(levels, pd.Series([5.0, 1.0, 1.0, 1.0])) == "a"
    assert features.base_level(levels, pd.Series([1.0, 3.0, 3.0, 1.0])) == "b"
