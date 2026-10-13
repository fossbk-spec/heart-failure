"""Data loading and schema validation.

Goal: every notebook / script uses a SINGLE data entry point, with integrity
checks that catch structural or dtype errors early.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


class SchemaValidationError(ValueError):
    """The dataset does not match the expected schema."""


@dataclass
class DatasetProfile:
    """Quick descriptive summary of a loaded dataset."""

    n_rows: int
    n_cols: int
    n_positive: int
    n_negative: int
    positive_rate: float
    n_missing: int
    n_duplicate_ids: int
    n_duplicate_records: int

    def as_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"Rows={self.n_rows:,} | Cols={self.n_cols} | "
            f"Death=1: {self.n_positive:,} ({self.positive_rate:.2%}) | "
            f"Death=0: {self.n_negative:,} | "
            f"Missing cells={self.n_missing:,} | "
            f"Duplicate IDs={self.n_duplicate_ids} | "
            f"Duplicate records={self.n_duplicate_records}"
        )


def load_raw_data(
    csv_path: str | Path | None = None, *, validate: bool = True
) -> pd.DataFrame:
    """Load the raw Heart Failure Clinical Records CSV.

    The raw file has no identifier column; a `patient_id` (1..n, in file
    order) is inserted as the first column so every downstream step can
    split and audit at the patient level.

    Args:
        csv_path: optional path. Defaults to `config.RAW_CSV_PATH`.
        validate: whether to run `validate_schema` right after loading.

    Raises:
        FileNotFoundError: when the raw data has not been placed in `data/raw/`.
    """
    path = Path(csv_path) if csv_path is not None else config.RAW_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at: {path}\n"
            "Download it first:  python scripts/download_data.py\n"
            "(or manually from UCI: https://archive.ics.uci.edu/dataset/519)"
        )

    df = pd.read_csv(path)
    df = add_patient_id(df)
    df = _coerce_dtypes(df)
    if validate:
        validate_schema(df)
    return df


def add_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    """Insert a 1-based `patient_id` column if the data does not already have one."""
    if config.ID_COLUMN in df.columns:
        return df
    df = df.copy()
    df.insert(0, config.ID_COLUMN, np.arange(1, len(df) + 1))
    return df


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes: continuous -> float, binary flags / target -> nullable Int64."""
    df = df.copy()
    for col in config.NUMERIC_FEATURES + config.LEAKAGE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in config.BINARY_FEATURES + [config.TARGET_COLUMN]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Validate the required schema. Raises `SchemaValidationError` on mismatch."""
    missing = [c for c in config.EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"Missing required columns: {missing}")

    if df.empty:
        raise SchemaValidationError("Dataset is empty.")

    target = df[config.TARGET_COLUMN].dropna().unique()
    invalid_target = sorted(set(np.asarray(target).tolist()) - {0, 1})
    if invalid_target:
        raise SchemaValidationError(
            f"Target column '{config.TARGET_COLUMN}' contains invalid values: {invalid_target}"
        )

    if df[config.ID_COLUMN].isna().any():
        raise SchemaValidationError(f"Column '{config.ID_COLUMN}' must not contain nulls.")

    for col in config.BINARY_FEATURES:
        extra = sorted(set(df[col].dropna().unique().tolist()) - {0, 1})
        if extra:
            raise SchemaValidationError(f"Column '{col}' must be 0/1 only, found: {extra}")

    ef = df["ejection_fraction"].dropna()
    if ((ef < 0) | (ef > 100)).any():
        raise SchemaValidationError("Column 'ejection_fraction' must be a percentage in [0, 100].")


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Compute quick descriptive statistics used in EDA and reporting."""
    n_rows = len(df)
    n_positive = int((df[config.TARGET_COLUMN] == 1).sum())
    record_cols = [c for c in df.columns if c != config.ID_COLUMN]
    return DatasetProfile(
        n_rows=n_rows,
        n_cols=df.shape[1],
        n_positive=n_positive,
        n_negative=n_rows - n_positive,
        positive_rate=n_positive / n_rows if n_rows else 0.0,
        n_missing=int(df.isna().sum().sum()),
        n_duplicate_ids=int(df[config.ID_COLUMN].duplicated().sum()),
        n_duplicate_records=int(df[record_cols].duplicated().sum()),
    )


def split_features_target(
    df: pd.DataFrame, *, keep_id: bool = False, feature_set: str = "all"
) -> tuple[pd.DataFrame, pd.Series]:
    """Split into the feature matrix X and the label vector y.

    Excluded from X by default:
    - `patient_id`: an identifier with no clinical meaning (identifier leakage).
    - `time` (follow-up days): known only AFTER the outcome; death truncates
      follow-up, so it encodes the label (temporal / target leakage).

    `feature_set` selects the columns of X (see `config.FEATURE_SETS`).
    """
    numeric, binary = config.get_feature_set(feature_set)
    cols = numeric + binary
    if keep_id:
        cols = [config.ID_COLUMN] + cols
    X = df[cols].copy()
    y = df[config.TARGET_COLUMN].astype(int).copy()
    return X, y


def make_synthetic_dataset(
    n_rows: int = 300,
    *,
    positive_rate: float = 0.321,
    missing_rate: float = 0.03,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a synthetic dataset matching the Heart Failure Clinical Records schema.

    Used for unit tests and CI (environments without the original CSV). This
    data MUST NOT be used to report scientific results.

    Unlike the real file (no missing values), `missing_rate` blanks out a
    fraction of `serum_creatinine` / `ejection_fraction` so the imputation
    path - needed for real-world inference - is exercised by tests.
    """
    rng = np.random.default_rng(random_state)
    n_pos = max(2, int(round(n_rows * positive_rate)))
    y = np.zeros(n_rows, dtype=int)
    y[rng.choice(n_rows, size=n_pos, replace=False)] = 1
    dead = y == 1

    # Deceased patients tend to be older, with lower EF and worse renal function.
    age = np.clip(rng.normal(np.where(dead, 65, 58.8), 12), 40, 95).round(0)
    ejection_fraction = np.clip(rng.normal(np.where(dead, 33.5, 40.3), 11), 14, 80).round(0)
    serum_creatinine = np.clip(
        np.exp(rng.normal(np.log(np.where(dead, 1.35, 1.05)), 0.4)), 0.5, 9.4
    ).round(2)
    serum_sodium = np.clip(rng.normal(np.where(dead, 135.4, 137.2), 4.4), 113, 148).round(0)
    cpk = np.clip(np.exp(rng.normal(5.9, 1.0, n_rows)), 23, 7861).round(0)
    platelets = np.clip(np.exp(rng.normal(np.log(255_000), 0.35, n_rows)), 25_100, 850_000).round(0)
    # Follow-up is truncated by death -> deliberately leaks the label.
    time = np.clip(rng.normal(np.where(dead, 70, 158), 62), 4, 285).round(0)

    df = pd.DataFrame(
        {
            config.ID_COLUMN: rng.permutation(np.arange(10_000, 10_000 + n_rows)),
            "age": age,
            "anaemia": rng.binomial(1, np.where(dead, 0.48, 0.41)),
            "creatinine_phosphokinase": cpk,
            "diabetes": rng.binomial(1, 0.42, n_rows),
            "ejection_fraction": ejection_fraction,
            "high_blood_pressure": rng.binomial(1, np.where(dead, 0.41, 0.33)),
            "platelets": platelets,
            "serum_creatinine": serum_creatinine,
            "serum_sodium": serum_sodium,
            "sex": rng.binomial(1, 0.65, n_rows),
            "smoking": rng.binomial(1, 0.32, n_rows),
            "time": time,
            config.TARGET_COLUMN: y,
        }
    )

    if missing_rate > 0:
        n_missing = max(1, int(n_rows * missing_rate))
        for col in ("serum_creatinine", "ejection_fraction"):
            df.loc[rng.choice(n_rows, size=n_missing, replace=False), col] = np.nan
    return df
