#!/usr/bin/env python3
"""Download the Heart Failure Clinical Records dataset into `data/raw/`.

Source: UCI Machine Learning Repository, dataset #519 (CC BY 4.0),
originally published by Ahmad et al. (2017) and re-analysed by
Chicco & Jurman (2020). The CSV is already bundled in the repo; this script
only matters if it was deleted.

Run:
    python scripts/download_data.py
    python scripts/download_data.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip"
CSV_NAME = config.RAW_CSV_PATH.name
EXPECTED_SHA256 = "9c73cea7468ff5d517801ec050fe9993da5912fce4b56f296f8df3b38dd75912"

MANUAL_INSTRUCTIONS = f"""
Automatic download failed. Please download manually in 3 steps:

  1. Open: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
     (mirror: https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)
  2. Click "Download" and unzip.
  3. Place `{CSV_NAME}` into:
         {config.RAW_DATA_DIR}
"""


def _try_uci() -> bytes | None:
    try:
        with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as response:  # noqa: S310
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            return archive.read(CSV_NAME)
    except Exception as exc:  # pragma: no cover - network-dependent
        print(f"  UCI download failed: {exc}")
        return None


def verify() -> bool:
    """Check that the file exists and matches the expected size/quality/checksum."""
    from src.data_loader import load_raw_data, profile_dataset

    if not config.RAW_CSV_PATH.exists():
        print(f"[MISSING] {config.RAW_CSV_PATH}")
        return False

    profile = profile_dataset(load_raw_data())
    digest = hashlib.sha256(config.RAW_CSV_PATH.read_bytes()).hexdigest()
    print(f"[OK] {config.RAW_CSV_PATH}")
    print(f"     {profile}")

    checks = {
        "n_rows = 299": profile.n_rows == config.EXPECTED_N_ROWS,
        "n_positive (DEATH_EVENT=1) = 96": profile.n_positive == config.EXPECTED_N_POSITIVE,
        "n_missing = 0": profile.n_missing == config.EXPECTED_N_MISSING,
        "sha256 matches the UCI release": digest == EXPECTED_SHA256,
    }
    for label, passed in checks.items():
        print(f"     [{'OK ' if passed else 'FAIL'}] {label}")
    return all(checks.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Only verify, do not download.")
    args = parser.parse_args(argv)

    if args.verify_only:
        return 0 if verify() else 1

    if config.RAW_CSV_PATH.exists():
        print("Data already exists, skipping download.")
        return 0 if verify() else 1

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Trying to download from the UCI repository ...")
    content = _try_uci()
    if content is None:
        print(MANUAL_INSTRUCTIONS)
        return 1

    config.RAW_CSV_PATH.write_bytes(content)
    print(f"Saved: {config.RAW_CSV_PATH}")
    return 0 if verify() else 1


if __name__ == "__main__":
    raise SystemExit(main())
