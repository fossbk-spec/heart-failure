"""Central configuration (single source of truth for project parameters)."""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PDF_REFERENCES_DIR = DATA_DIR / "SSoT_pdf_references"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CSV_PATH = RAW_DATA_DIR / "heart_failure_clinical_records_dataset.csv"

# --------------------------------------------------------------------------- #
# Heart Failure Clinical Records schema (Ahmad et al. 2017; Chicco & Jurman 2020)
# 299 x 13 in the raw CSV, + 1 synthetic `patient_id` added at load time.
# --------------------------------------------------------------------------- #
# The raw CSV has NO identifier column. Each row is one patient, so
# `data_loader` adds `patient_id` = 1..n (row order) to make the patient-level
# split explicit and auditable, exactly as it would be on real EHR data.
ID_COLUMN = "patient_id"
TARGET_COLUMN = "DEATH_EVENT"

# Follow-up period (days). It is NOT a baseline feature: patients who die
# have short follow-up by construction (death ends follow-up), so using it as
# a predictor is TEMPORAL / TARGET LEAKAGE. Kept in the data for EDA only.
LEAKAGE_COLUMNS = ["time"]

NUMERIC_FEATURES = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
]
# Heavily right-skewed lab values -> log1p before scaling.
SKEWED_FEATURES = ["creatinine_phosphokinase", "serum_creatinine", "platelets"]
BINARY_FEATURES = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]

FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES

# Named feature sets: (numeric, binary). "minimal_two" tests the Chicco &
# Jurman (2020) claim that serum creatinine + ejection fraction suffice.
FEATURE_SETS: dict[str, tuple[list[str], list[str]]] = {
    "all": (NUMERIC_FEATURES, BINARY_FEATURES),
    "minimal_two": (["ejection_fraction", "serum_creatinine"], []),
}


def get_feature_set(name: str = "all") -> tuple[list[str], list[str]]:
    """Return copies of the (numeric, binary) columns of a named feature set."""
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{name}'. Valid options: {sorted(FEATURE_SETS)}")
    numeric, binary = FEATURE_SETS[name]
    return list(numeric), list(binary)


# Column order EXACTLY as in the original CSV (used for schema validation).
RAW_COLUMNS = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time",
    TARGET_COLUMN,
]
EXPECTED_COLUMNS = [ID_COLUMN] + RAW_COLUMNS

# Expected raw-data size / quality (used for data validation)
EXPECTED_N_ROWS = 299
EXPECTED_N_POSITIVE = 96
EXPECTED_N_MISSING = 0

# --------------------------------------------------------------------------- #
# Experiment parameters
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
N_SPLITS = 5          # Number of folds for StratifiedGroupKFold
TEST_SIZE = 0.2       # Test set fraction (patient-level hold-out)
VAL_SIZE = 0.2        # Validation set fraction (carved out of the remainder)

# Metric priority for the clinical problem: FN >> FP  =>  beta = 2
F_BETA = 2.0

# Default probability threshold, before F-beta-based tuning
DEFAULT_THRESHOLD = 0.5
