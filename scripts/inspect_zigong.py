#!/usr/bin/env python3
"""Aggregate-only profile of the restricted Zigong data + harmonization check.

Prints counts, missingness, value ranges and per-horizon event counts — never
individual patient rows — so the output is safe to share within the project.

Run:
    python scripts/inspect_zigong.py
    python scripts/inspect_zigong.py --synthetic   # without the restricted data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import harmonization as hz  # noqa: E402
from src.data_loader import load_raw_data  # noqa: E402

RAW_COLUMNS_USED = [
    "ageCat", "gender", "creatine.kinase", "LVEF", "platelet",
    "creatinine.enzymatic.method", "sodium", "hemoglobin", "diabetes",
    "NYHA.cardiac.function.classification", *hz.ZIGONG_DEATH_COLUMNS.values(),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic Zigong data.")
    args = parser.parse_args(argv)
    pd.set_option("display.width", 200)

    zigong = hz.make_synthetic_zigong() if args.synthetic else hz.load_zigong_raw()
    uci = load_raw_data()
    print(f"Zigong raw: {zigong.shape[0]:,} rows x {zigong.shape[1]} columns")

    print("\n--- Raw columns used for harmonization (missingness, range) ---")
    present = [c for c in RAW_COLUMNS_USED if c in zigong.columns]
    numeric = zigong[present].select_dtypes("number")
    print(pd.DataFrame({
        "pct_missing": zigong[present].isna().mean().round(3),
        "min": numeric.min(), "median": numeric.median(), "max": numeric.max(),
    }).to_string())

    if "visit.times" in zigong.columns:
        print("\nvisit.times:", zigong["visit.times"].value_counts().sort_index().to_dict())

    print("\n--- Events per horizon ---")
    rows = []
    for horizon in hz.ZIGONG_DEATH_COLUMNS:
        u = hz.harmonize_uci(uci, horizon_days=horizon)
        z = hz.harmonize_zigong(zigong, horizon_days=horizon)
        label = hz.outcome_column(horizon)
        rows.append({
            "horizon_days": horizon,
            "uci_n": len(u), "uci_deaths": int(u[label].sum()),
            "zigong_n": len(z), "zigong_deaths": int(z[label].sum()),
            "zigong_rate": round(float(z[label].mean()), 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    for uci_like in (False, True):
        both = hz.harmonize_both(uci, zigong, horizon_days=90, uci_like=uci_like)
        print(f"\n--- Harmonized cohorts, 90-day death (uci_like={uci_like}) ---")
        print(hz.cohort_summary(both, 90).round(3).T.to_string(header=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
