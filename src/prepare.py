"""Stage 1: clean raw data, treat large claims, split by policy, band, write CSVs.

Outputs
-------
data/processed/train.csv, data/processed/test.csv  (inputs to R and the GBM)
reports/tables/cleaning_summary.csv                 (rows affected by each rule)
reports/tables/large_loss.csv                       (cap and loading, train only)
"""

from __future__ import annotations

import pandas as pd

from src import cleaning, config, features


def cleaning_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Count the policies touched by each cleaning rule (on the full raw table)."""
    rules = {
        f"ClaimNb > {config.CAP_CLAIM_NB}": raw["ClaimNb"] > config.CAP_CLAIM_NB,
        f"Exposure > {config.CAP_EXPOSURE}": raw["Exposure"] > config.CAP_EXPOSURE,
        f"VehAge > {config.CAP_VEH_AGE}": raw["VehAge"] > config.CAP_VEH_AGE,
        f"DrivAge > {config.CAP_DRIV_AGE}": raw["DrivAge"] > config.CAP_DRIV_AGE,
        f"BonusMalus > {config.CAP_BONUS_MALUS}": raw["BonusMalus"] > config.CAP_BONUS_MALUS,
    }
    rows = [
        {"rule": name, "policies_capped": int(mask.sum()), "share_pct": round(100 * mask.mean(), 3)}
        for name, mask in rules.items()
    ]
    return pd.DataFrame(rows)


def build_datasets() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run the full preparation and return (train, test, large-loss info)."""
    freq_raw, sev_raw = cleaning.load_raw()
    freq = features.add_bands(cleaning.clean_frequency(freq_raw))

    is_test = cleaning.split_policies(freq["IDpol"])
    test_ids = set(freq.loc[is_test, "IDpol"])
    sev_train = sev_raw[~sev_raw["IDpol"].isin(test_ids)]

    # Threshold and loading are estimated on training claims only (no leakage).
    threshold = cleaning.large_claim_threshold(sev_train["ClaimAmount"])
    sev_capped = cleaning.cap_claims(sev_raw, threshold)
    loading = cleaning.large_loss_loading(sev_capped[~sev_capped["IDpol"].isin(test_ids)])

    full = cleaning.join_severity(freq, cleaning.aggregate_severity(sev_capped))
    train, test = full[~is_test].reset_index(drop=True), full[is_test].reset_index(drop=True)
    # Cross-validation folds on the training set only (the test set stays untouched).
    train["CVFold"] = cleaning.assign_folds(train["IDpol"], seed=config.SEED + 2)

    n_large = int((sev_train["ClaimAmount"] > threshold).sum())
    info = {
        "quantile": config.LARGE_CLAIM_QUANTILE,
        "threshold_eur": round(threshold, 2),
        "train_claims": int(len(sev_train)),
        "train_claims_above_threshold": n_large,
        "train_incurred_eur": round(float(sev_train["ClaimAmount"].sum()), 0),
        "train_capped_eur": round(float(sev_capped.loc[~sev_capped["IDpol"].isin(test_ids), "ClaimAmountCapped"].sum()), 0),
        "large_loss_loading": round(loading, 4),
    }
    return train, test, info


def main() -> None:
    """Entry point for the preparation stage."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)

    freq_raw, _ = cleaning.load_raw()
    cleaning_summary(freq_raw).to_csv(config.TABLE_DIR / "cleaning_summary.csv", index=False)

    train, test, info = build_datasets()
    train.to_csv(config.PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(config.PROCESSED_DIR / "test.csv", index=False)
    pd.DataFrame([info]).to_csv(config.TABLE_DIR / "large_loss.csv", index=False)

    split = pd.DataFrame(
        {
            "set": ["train", "test"],
            "policies": [len(train), len(test)],
            "exposure_years": [round(train["Exposure"].sum(), 1), round(test["Exposure"].sum(), 1)],
            "claims": [int(train["ClaimNb"].sum()), int(test["ClaimNb"].sum())],
            "frequency": [
                round(train["ClaimNb"].sum() / train["Exposure"].sum(), 4),
                round(test["ClaimNb"].sum() / test["Exposure"].sum(), 4),
            ],
        }
    )
    split.to_csv(config.TABLE_DIR / "split_summary.csv", index=False)

    full = pd.concat([train, test])
    checks = pd.DataFrame([{
        "policies_with_claims": int((full["ClaimNbRaw"] > 0).sum()),
        "claim_records": int(full["NbClaimsSev"].sum()),
        "policies_where_ClaimNb_differs_from_claim_records": int((full["ClaimNbRaw"] != full["NbClaimsSev"]).sum()),
    }])
    checks.to_csv(config.TABLE_DIR / "data_checks.csv", index=False)
    print(checks.to_string(index=False))
    print(split.to_string(index=False))
    print(info)


if __name__ == "__main__":
    main()
