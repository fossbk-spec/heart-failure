#!/usr/bin/env python3
"""Download the restricted PhysioNet Zigong data into `data/raw/zigong/` (never committed).

Source: Heart failure clinical records from Zigong Fourth People's Hospital,
PhysioNet v1.3 (Zhang et al. 2021). Access requires a credentialed PhysioNet
account that has signed the dataset's DUA, so every machine downloads its own
copy with its own account: the data must never be pushed to git, CI or a
public notebook.

Run (the password is prompted, or read from $PHYSIONET_PASSWORD; never stored):
    python scripts/download_zigong.py --user <physionet-username>
    python scripts/download_zigong.py --verify-only
    python scripts/download_zigong.py --user <name> --cafile "C:/ProgramData/Avast Software/Avast/wscert.pem"
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import os
import ssl
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import harmonization as hz  # noqa: E402

PHYSIONET_BASE_URL = "https://physionet.org/files/heart-failure-zigong/1.3"
FILES = ("dat.csv",)
EXPECTED_N_ROWS = 2008
EXPECTED_DEATHS_90D = 42
EXPECTED_SHA256 = "f2be23eeaf2e6f457512ad7653f6ce98a87ac23376596eef6d3bf46e4f311ae8"

MANUAL_INSTRUCTIONS = f"""
Automatic download failed. Download manually with your credentialed account:

  1. Log in at https://physionet.org and open
     https://physionet.org/content/heart-failure-zigong/1.3/
  2. Download `dat.csv` (Files section, or "Download the ZIP file").
  3. Place it into:
         {hz.ZIGONG_DIR}
  4. Check it: python scripts/download_zigong.py --verify-only
"""


def _download(name: str, user: str, password: str, cafile: str | None) -> bytes:
    request = urllib.request.Request(f"{PHYSIONET_BASE_URL}/{name}")
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    request.add_header("Authorization", f"Basic {token}")
    context = ssl.create_default_context(cafile=cafile)
    with urllib.request.urlopen(request, timeout=120, context=context) as response:  # noqa: S310
        return response.read()


def verify(csv_path: Path = hz.ZIGONG_CSV_PATH) -> bool:
    """Aggregate-only checks: row count, 90-day deaths, and the sha256 to compare across machines."""
    if not csv_path.exists():
        print(f"[MISSING] {csv_path}")
        return False
    df = hz.load_zigong_raw(csv_path)
    deaths = int(df[hz.ZIGONG_DEATH_COLUMNS[90]].sum())
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    checks = {
        f"n_rows = {EXPECTED_N_ROWS}": len(df) == EXPECTED_N_ROWS,
        f"deaths within 90 days = {EXPECTED_DEATHS_90D}": deaths == EXPECTED_DEATHS_90D,
        "sha256 matches the copy used for the reported results": digest == EXPECTED_SHA256,
    }
    print(f"[OK] {csv_path}")
    print(f"     sha256 = {digest}")
    for label, passed in checks.items():
        print(f"     [{'OK ' if passed else 'FAIL'}] {label}")
    return all(checks.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", help="Credentialed PhysioNet username.")
    parser.add_argument("--cafile", help="Extra CA bundle (e.g. antivirus TLS interception).")
    parser.add_argument("--verify-only", action="store_true", help="Only verify, do not download.")
    args = parser.parse_args(argv)

    if args.verify_only or hz.ZIGONG_CSV_PATH.exists():
        if not args.verify_only:
            print("Zigong data already exists, skipping download.")
        return 0 if verify() else 1
    if not args.user:
        parser.error("--user is required to download (or use --verify-only).")

    password = os.environ.get("PHYSIONET_PASSWORD") or getpass.getpass("PhysioNet password: ")
    hz.ZIGONG_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        print(f"Downloading {name} from PhysioNet ...")
        try:
            content = _download(name, args.user, password, args.cafile)
        except Exception as exc:  # pragma: no cover - network-dependent
            print(f"  Download failed: {exc}")
            print(MANUAL_INSTRUCTIONS)
            return 1
        (hz.ZIGONG_DIR / name).write_bytes(content)
        print(f"Saved: {hz.ZIGONG_DIR / name}")
    return 0 if verify() else 1


if __name__ == "__main__":
    raise SystemExit(main())
