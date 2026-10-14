#!/usr/bin/env python3
"""Pool UCI + Zigong and compare ways of handling the missing LVEF.

Scenarios (see `src/pooled_missing.py`): complete case (drop admissions without
LVEF), drop the LVEF feature, and median / MICE / KNN imputation with or
without missing indicators; each with and without a cohort indicator.
Aggregate metrics only, written to `reports/pooled_missing/` (git-ignored).

Run:
    python scripts/run_pooled_missing.py
    python scripts/run_pooled_missing.py --synthetic --n-repeats 1 --models logistic_regression
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src import harmonization as hz  # noqa: E402
from src import pooled_missing as pm  # noqa: E402
from src.data_loader import load_raw_data, make_synthetic_dataset  # noqa: E402

SUMMARY = ["scenario", "model", "n_rows", "n_features", "pooled_auc", "uci_auc", "zigong_auc",
           "zigong_lvef_auc", "common_auc", "zigong_pr_auc", "zigong_mean_pred", "zigong_rate",
           "zigong_citl", "zigong_auc_sd_repeats"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="Synthetic data (no restricted data).")
    parser.add_argument("--n-repeats", type=int, default=3, help="Repeats of stratified 5-fold CV.")
    parser.add_argument("--models", nargs="+", default=pm.DEFAULT_POOLED_MODELS)
    parser.add_argument("--horizon", type=int, default=90, choices=sorted(hz.ZIGONG_DEATH_COLUMNS))
    parser.add_argument("--output", default=str(config.REPORTS_DIR / "pooled_missing"))
    args = parser.parse_args(argv)
    pd.set_option("display.width", 250)

    if args.synthetic:
        print("[WARNING] SYNTHETIC DATA - do not use for scientific reporting.")
        uci_raw, zg_raw = make_synthetic_dataset(n_rows=400), hz.make_synthetic_zigong(1200, death_rate=0.06)
    else:
        uci_raw, zg_raw = load_raw_data(), hz.load_zigong_raw()
    pooled = hz.harmonize_both(uci_raw, zg_raw, horizon_days=args.horizon)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    missing = pm.missingness_summary(pooled, args.horizon)
    accuracy = pm.imputation_accuracy(pooled)
    table = pm.run_pooled_scenarios(pooled, model_names=tuple(args.models),
                                    n_repeats=args.n_repeats, horizon_days=args.horizon)
    missing.to_csv(out / "lvef_missingness.csv", index=False)
    accuracy.to_csv(out / "lvef_imputation_accuracy.csv", index=False)
    table.to_csv(out / "pooled_scenarios.csv", index=False)

    print("\n==== Zigong: admissions with vs without measured LVEF ====")
    print(missing.round(3).to_string(index=False))
    print("\n==== Imputing OBSERVED Zigong LVEF values (masked, held-out folds) ====")
    print(accuracy.round(2).to_string(index=False))
    for src in (False, True):
        print(f"\n==== Pooled CV, {args.horizon}-day death, cohort indicator = {src} ====")
        part = table[table["source_feature"] == src]
        print(part[SUMMARY].round(3).to_string(index=False))
    print(f"\nResults written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
