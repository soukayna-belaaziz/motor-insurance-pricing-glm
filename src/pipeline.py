"""Single entry point: ``python -m src.pipeline`` (run from the project root).

Stages: download -> prepare (Python) -> one-way analyses (Python)
-> GLMs (R) -> frequency GLM extensions: refinement, negative binomial,
cross-validation fits (R) -> LightGBM (Python) -> test evaluation (Python)
-> cross-validation and bootstrap (Python).
"""

from __future__ import annotations

import subprocess
import time

import pandas as pd

from src import config, evaluate, gbm, oneway, prepare, validation


def run_r(script: str) -> None:
    """Run an R script with Rscript and fail loudly on error."""
    subprocess.run(["Rscript", script], check=True)


def main() -> None:
    """Run every stage in order and record wall-clock times."""
    stages = [
        ("download", lambda: None if (config.FREQ_CSV.exists() and config.SEV_CSV.exists())
         else run_r("data/download_data.R")),
        ("prepare", prepare.main),
        ("oneway", oneway.main),
        ("glm_r", lambda: run_r("R/fit_glm.R")),
        ("glm_extensions_r", lambda: run_r("R/fit_freq_extensions.R")),
        ("gbm", gbm.main),
        ("evaluate", evaluate.main),
        ("validation", validation.main),
    ]
    times = []
    for name, fn in stages:
        t0 = time.time()
        print(f"=== {name} ===", flush=True)
        fn()
        times.append({"stage": name, "seconds": round(time.time() - t0, 1)})
    times.append({"stage": "total", "seconds": round(sum(t["seconds"] for t in times), 1)})
    pd.DataFrame(times).to_csv(config.TABLE_DIR / "runtime.csv", index=False)
    print(pd.DataFrame(times).to_string(index=False))


if __name__ == "__main__":
    main()
