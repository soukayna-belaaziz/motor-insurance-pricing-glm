"""Project-wide constants: paths, random seed and cleaning thresholds.

All paths are relative to the project root, so every entry point must be run
from there (``python -m src.pipeline`` or ``make all``).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(".")
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

FREQ_CSV = RAW_DIR / "freMTPL2freq.csv"
SEV_CSV = RAW_DIR / "freMTPL2sev.csv"

SEED = 42
TEST_SHARE = 0.20
CV_FOLDS = 5              # K-fold cross-validation on the training set
N_BOOTSTRAP = 1000        # paired bootstrap replicates for the test-set Gini gap

# Cleaning caps. These are business rules fixed a priori from the one-way
# tables (see README), not parameters fitted on the data.
CAP_CLAIM_NB = 4          # > 4 claims in < 1 year: data error or fleet, 8 policies
CAP_EXPOSURE = 1.0        # policies are annual; exposure > 1 year is a data error
CAP_VEH_AGE = 20          # thin, noisy tail above 20 years
CAP_DRIV_AGE = 90         # thin tail above 90
CAP_BONUS_MALUS = 150     # 209 policies above 150

# Large claims: individual claims above this quantile of the *training* claim
# distribution are capped; the excess is spread back with a flat loading.
LARGE_CLAIM_QUANTILE = 0.995

# Column groups
CATEGORICAL = ["VehBrand", "VehGas", "Area", "Region"]
GBM_FEATURES = [
    "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand",
    "VehGas", "Area", "LogDensity", "Region",
]
