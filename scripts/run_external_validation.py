#!/usr/bin/env python3
"""External validation: develop on UCI, validate on PhysioNet Zigong (plan option A).

Outputs aggregate metrics only (no patient rows) to
`reports/external_validation/external_validation.csv` (git-ignored).

Run:
    python scripts/run_external_validation.py
    python scripts/run_external_validation.py --synthetic --n-boot 50 --models logistic_regression
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src import external_validation as ev  # noqa: E402
from src import harmonization as hz  # noqa: E402
from src.data_loader import load_raw_data, make_synthetic_dataset  # noqa: E402

SUMMARY_COLUMNS = [
    "uci_like", "horizon_days", "feature_set", "model", "zigong_n", "zigong_deaths",
    "internal_oof_auc", "external_auc", "auc_ci95_low", "auc_ci95_high", "external_pr_auc",
    "zigong_rate", "mean_predicted_risk", "calibration_in_the_large", "calibration_slope",
    "brier", "brier_after_intercept_update", "brier_null",
    "sensitivity_at_uci_threshold", "specificity_at_uci_threshold",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic UCI + Zigong data (no restricted data needed).")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap replicates for AUC CI.")
    parser.add_argument("--models", nargs="+", default=ev.DEFAULT_EXTERNAL_MODELS,
                        help="Models to develop on UCI.")
    parser.add_argument("--output", default=str(config.REPORTS_DIR / "external_validation"))
    args = parser.parse_args(argv)
    pd.set_option("display.width", 250)

    if args.synthetic:
        print("[WARNING] SYNTHETIC DATA - do not use for scientific reporting.")
        uci_raw, zigong_raw = make_synthetic_dataset(n_rows=400), hz.make_synthetic_zigong(1500)
    else:
        uci_raw, zigong_raw = load_raw_data(), hz.load_zigong_raw()

    table = ev.run_external_validation(
        uci_raw, zigong_raw, model_names=tuple(args.models), n_boot=args.n_boot
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "external_validation.csv", index=False)

    print("\n================ PRIMARY: 90-day death, all Zigong admissions ================")
    print(table.loc[table["primary"], SUMMARY_COLUMNS].round(3).to_string(index=False))
    print("\n================ SENSITIVITY ANALYSES ================")
    print(table.loc[~table["primary"], SUMMARY_COLUMNS].round(3).to_string(index=False))
    print(f"\nResults written to: {out / 'external_validation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
