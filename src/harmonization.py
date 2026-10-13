"""Harmonize UCI (Faisalabad 2015) and PhysioNet Zigong (2016-2019) heart failure data.

Both cohorts are mapped onto ONE common schema so that a model developed on
one can be validated on the other. Decisions (see docs/ACTION_PLAN_FOR_CLAUDE.md):

- **Outcome**: death within a fixed horizon (default 90 days). Zigong records
  `death.within.28.days / 3.months / 6.months` directly. UCI only records
  "death during follow-up" with variable follow-up, so a UCI patient alive but
  censored before the horizon has an UNKNOWN label and is dropped.
- **Common features**: age (Zigong only releases decade bands, so UCI age is
  binned identically), sex, diabetes, anaemia, ejection fraction, serum
  creatinine, serum sodium, platelets, creatine (phospho)kinase.
  `high_blood_pressure` and `smoking` do not exist in Zigong and are dropped.
- **Units** (inferred from Zigong value ranges, to confirm against the PhysioNet
  data dictionary): creatinine umol/L -> mg/dL (/88.4); hemoglobin g/L;
  platelets 10^9/L (UCI's ~263,358 per uL -> /1000); CK assumed U/L in both.
- **Identifiers**: Zigong has no patient identifier, only one `inpatient.number`
  per admission, although 7% of rows report >= 2 visits. Admissions of the same
  patient therefore cannot be grouped - a documented limitation.

Zigong is restricted-access data (PhysioNet Restricted Health Data License
1.5.0): it lives in `data/raw/zigong/`, is git-ignored, and must never be
committed. Tests use `make_synthetic_zigong` instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config

ZIGONG_DIR = config.RAW_DATA_DIR / "zigong"
ZIGONG_CSV_PATH = ZIGONG_DIR / "dat.csv"

CREATININE_UMOL_PER_MG_DL = 88.4
# WHO anaemia thresholds, hemoglobin in g/L.
ANAEMIA_HB_THRESHOLD_G_L = {1: 130.0, 0: 120.0}  # sex: 1 = male, 0 = female

# Zigong releases age only as these right-closed bands; each band is represented
# by its decade start + 5 so that UCI ages map onto exactly the same values.
AGE_BANDS: list[tuple[str, float, float, float]] = [
    ("(21,29]", 21, 29, 25.0),
    ("(29,39]", 29, 39, 35.0),
    ("(39,49]", 39, 49, 45.0),
    ("(49,59]", 49, 59, 55.0),
    ("(59,69]", 59, 69, 65.0),
    ("(69,79]", 69, 79, 75.0),
    ("(79,89]", 79, 89, 85.0),
    ("(89,110]", 89, 110, 95.0),
]

ZIGONG_DEATH_COLUMNS = {
    28: "death.within.28.days",
    90: "death.within.3.months",
    180: "death.within.6.months",
}

ID_COLUMN = "patient_id"
SOURCE_COLUMN = "source"
HARMONIZED_NUMERIC = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
]
HARMONIZED_BINARY = ["anaemia", "diabetes", "sex"]
HARMONIZED_FEATURES = HARMONIZED_NUMERIC + HARMONIZED_BINARY
DROPPED_UCI_FEATURES = ["high_blood_pressure", "smoking"]


def outcome_column(horizon_days: int) -> str:
    """Name of the harmonized label column, e.g. `death_90d`."""
    return f"death_{horizon_days}d"


def harmonized_columns(horizon_days: int) -> list[str]:
    return [ID_COLUMN, SOURCE_COLUMN] + HARMONIZED_FEATURES + [outcome_column(horizon_days)]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def age_to_band_value(age: pd.Series) -> pd.Series:
    """Map exact ages onto the Zigong band representatives (e.g. 63.5 -> 65.0)."""
    edges = [AGE_BANDS[0][1]] + [hi for _label, _lo, hi, _value in AGE_BANDS]
    values = [value for *_rest, value in AGE_BANDS]
    banded = pd.cut(age, bins=edges, labels=values, right=True)
    return banded.astype(float)


def age_band_label_to_value(age_cat: pd.Series) -> pd.Series:
    mapping = {label: value for label, _lo, _hi, value in AGE_BANDS}
    unknown = set(age_cat.dropna().unique()) - set(mapping)
    if unknown:
        raise ValueError(f"Unknown Zigong age bands: {sorted(unknown)}")
    return age_cat.map(mapping).astype(float)


def anaemia_from_hemoglobin(hemoglobin_g_l: pd.Series, sex: pd.Series) -> pd.Series:
    """WHO definition: Hb < 130 g/L (men) or < 120 g/L (women); NaN if Hb is missing."""
    threshold = sex.map(ANAEMIA_HB_THRESHOLD_G_L)
    out = (hemoglobin_g_l < threshold).astype(float)
    out[hemoglobin_g_l.isna() | threshold.isna()] = np.nan
    return out


def uci_fixed_horizon_label(df: pd.DataFrame, horizon_days: int) -> pd.Series:
    """Death within `horizon_days` for UCI rows; NaN when the label is unknown.

    1   : died with follow-up time <= horizon
    0   : known to be alive at the horizon (follow-up >= horizon, or died later)
    NaN : alive but censored before the horizon -> outcome unknown, drop.
    """
    died = df[config.TARGET_COLUMN].astype(int) == 1
    time = df["time"]
    label = pd.Series(np.nan, index=df.index, dtype=float)
    label[died & (time <= horizon_days)] = 1.0
    label[(~died & (time >= horizon_days)) | (died & (time > horizon_days))] = 0.0
    return label


# --------------------------------------------------------------------------- #
# UCI
# --------------------------------------------------------------------------- #
def harmonize_uci(df: pd.DataFrame, *, horizon_days: int = 90) -> pd.DataFrame:
    """Map a UCI frame (as returned by `load_raw_data`) onto the common schema.

    Patients censored alive before `horizon_days` are dropped (unknown label).
    """
    label = uci_fixed_horizon_label(df, horizon_days)
    out = pd.DataFrame(
        {
            ID_COLUMN: "uci-" + df[config.ID_COLUMN].astype(str),
            SOURCE_COLUMN: "uci",
            "age": age_to_band_value(df["age"]),
            "creatinine_phosphokinase": df["creatinine_phosphokinase"].astype(float),
            "ejection_fraction": df["ejection_fraction"].astype(float),
            "platelets": df["platelets"].astype(float) / 1000.0,
            "serum_creatinine": df["serum_creatinine"].astype(float),
            "serum_sodium": df["serum_sodium"].astype(float),
            "anaemia": df["anaemia"].astype(float),
            "diabetes": df["diabetes"].astype(float),
            "sex": df["sex"].astype(float),
            outcome_column(horizon_days): label,
        },
        index=df.index,
    )
    return out[label.notna()].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Zigong
# --------------------------------------------------------------------------- #
def load_zigong_raw(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the restricted-access Zigong `dat.csv` (never committed to git)."""
    path = Path(csv_path) if csv_path is not None else ZIGONG_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Zigong data not found at: {path}\n"
            "It is restricted-access: download PhysioNet 'heart-failure-zigong' v1.3 with a "
            "credentialed account and a signed DUA, then place dat.csv in data/raw/zigong/."
        )
    return pd.read_csv(path)


def harmonize_zigong(
    df: pd.DataFrame, *, horizon_days: int = 90, uci_like: bool = False
) -> pd.DataFrame:
    """Map a raw Zigong frame onto the common schema.

    Args:
        horizon_days: 28, 90 or 180 (the follow-up horizons recorded in Zigong).
        uci_like: keep only admissions matching the UCI inclusion criteria that
            Zigong can express (age >= 40, NYHA class III-IV).
    """
    if horizon_days not in ZIGONG_DEATH_COLUMNS:
        raise ValueError(
            f"Zigong records death at {sorted(ZIGONG_DEATH_COLUMNS)} days only, got {horizon_days}."
        )
    sex = df["gender"].map({"Male": 1.0, "Female": 0.0})
    out = pd.DataFrame(
        {
            ID_COLUMN: "zg-" + df["inpatient.number"].astype(str),
            SOURCE_COLUMN: "zigong",
            "age": age_band_label_to_value(df["ageCat"]),
            "creatinine_phosphokinase": df["creatine.kinase"].astype(float),
            "ejection_fraction": df["LVEF"].astype(float),
            "platelets": df["platelet"].astype(float),
            "serum_creatinine": df["creatinine.enzymatic.method"].astype(float)
            / CREATININE_UMOL_PER_MG_DL,
            "serum_sodium": df["sodium"].astype(float),
            "anaemia": anaemia_from_hemoglobin(df["hemoglobin"].astype(float), sex),
            "diabetes": df["diabetes"].astype(float),
            "sex": sex,
            outcome_column(horizon_days): df[ZIGONG_DEATH_COLUMNS[horizon_days]].astype(float),
        },
        index=df.index,
    )
    if uci_like:
        keep = (out["age"] >= 45.0) & df["NYHA.cardiac.function.classification"].isin(["III", "IV"])
        out = out[keep]
    return out.reset_index(drop=True)


def harmonize_both(
    uci: pd.DataFrame, zigong: pd.DataFrame, *, horizon_days: int = 90, uci_like: bool = False
) -> pd.DataFrame:
    """Stack both harmonized cohorts (the `source` column tells them apart)."""
    return pd.concat(
        [
            harmonize_uci(uci, horizon_days=horizon_days),
            harmonize_zigong(zigong, horizon_days=horizon_days, uci_like=uci_like),
        ],
        ignore_index=True,
    )


def cohort_summary(harmonized: pd.DataFrame, horizon_days: int = 90) -> pd.DataFrame:
    """Aggregate-only description per source (safe to print for restricted data)."""
    label = outcome_column(horizon_days)
    rows = []
    for source, part in harmonized.groupby(SOURCE_COLUMN):
        row: dict[str, float | str | int] = {
            "source": source,
            "n": len(part),
            "n_death": int(part[label].sum()),
            "death_rate": float(part[label].mean()),
            "female_share": float((part["sex"] == 0).mean()),
            "age_median": float(part["age"].median()),
        }
        for col in HARMONIZED_FEATURES:
            row[f"missing_{col}"] = float(part[col].isna().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def make_synthetic_zigong(
    n_rows: int = 400, *, death_rate: float = 0.03, random_state: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """Synthetic frame with the raw Zigong columns used here (tests / CI only)."""
    rng = np.random.default_rng(random_state)
    death_6m = rng.binomial(1, death_rate * 1.4, n_rows)
    death_3m = death_6m * rng.binomial(1, 0.75, n_rows)
    death_28d = death_3m * rng.binomial(1, 0.85, n_rows)
    labels = [label for label, *_rest in AGE_BANDS]
    lvef = rng.normal(50, 13, n_rows).clip(5, 82).round(0)
    lvef[rng.random(n_rows) < 0.68] = np.nan
    ck = np.exp(rng.normal(4.5, 0.8, n_rows)).round(0)
    ck[rng.random(n_rows) < 0.12] = np.nan
    return pd.DataFrame(
        {
            "inpatient.number": rng.permutation(np.arange(700_000, 700_000 + n_rows)),
            "gender": rng.choice(["Female", "Male"], n_rows, p=[0.58, 0.42]),
            "ageCat": rng.choice(labels, n_rows, p=[0.002, 0.006, 0.028, 0.053, 0.183, 0.356,
                                                   0.322, 0.050]),
            "NYHA.cardiac.function.classification": rng.choice(
                ["II", "III", "IV"], n_rows, p=[0.18, 0.52, 0.30]
            ),
            "creatine.kinase": ck,
            "LVEF": lvef,
            "platelet": rng.normal(145, 65, n_rows).clip(5, 646).round(0),
            "creatinine.enzymatic.method": np.exp(rng.normal(np.log(87), 0.45, n_rows)).round(1),
            "sodium": rng.normal(138, 4.9, n_rows).round(1),
            "hemoglobin": rng.normal(115, 24.5, n_rows).clip(29, 200).round(0),
            "diabetes": rng.binomial(1, 0.23, n_rows),
            "death.within.28.days": death_28d,
            "death.within.3.months": death_3m,
            "death.within.6.months": death_6m,
        }
    )
