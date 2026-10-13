"""Unit tests for data loading and schema validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import (
    SchemaValidationError,
    load_raw_data,
    make_synthetic_dataset,
    profile_dataset,
    split_features_target,
    validate_schema,
)


class TestSchemaValidation:
    def test_valid_dataset_passes(self, synthetic_df):
        validate_schema(synthetic_df)  # must not raise

    def test_missing_column_is_rejected(self, synthetic_df):
        with pytest.raises(SchemaValidationError, match="Missing required"):
            validate_schema(synthetic_df.drop(columns=["serum_creatinine"]))

    def test_invalid_target_value_is_rejected(self, synthetic_df):
        broken = synthetic_df.copy()
        broken.loc[0, config.TARGET_COLUMN] = 2
        with pytest.raises(SchemaValidationError, match="invalid values"):
            validate_schema(broken)

    def test_empty_dataset_is_rejected(self, synthetic_df):
        with pytest.raises(SchemaValidationError, match="empty"):
            validate_schema(synthetic_df.iloc[0:0])

    def test_null_id_is_rejected(self, synthetic_df):
        broken = synthetic_df.copy().astype({config.ID_COLUMN: "float"})
        broken.loc[0, config.ID_COLUMN] = np.nan
        with pytest.raises(SchemaValidationError, match="must not contain nulls"):
            validate_schema(broken)

    def test_non_binary_flag_is_rejected(self, synthetic_df):
        broken = synthetic_df.copy()
        broken.loc[0, "anaemia"] = 3
        with pytest.raises(SchemaValidationError, match="0/1 only"):
            validate_schema(broken)

    def test_ejection_fraction_out_of_range_is_rejected(self, synthetic_df):
        broken = synthetic_df.copy()
        broken.loc[0, "ejection_fraction"] = 140
        with pytest.raises(SchemaValidationError, match="ejection_fraction"):
            validate_schema(broken)


class TestLoadRawData:
    def test_missing_file_gives_actionable_message(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="download_data.py"):
            load_raw_data(tmp_path / "does-not-exist.csv")

    def test_patient_id_is_added_to_raw_csv(self, tmp_path, synthetic_df):
        """The raw CSV has no id column -> `patient_id` 1..n is inserted first."""
        path = tmp_path / "raw.csv"
        synthetic_df.drop(columns=[config.ID_COLUMN]).to_csv(path, index=False)

        loaded = load_raw_data(path)
        assert loaded.columns[0] == config.ID_COLUMN
        assert list(loaded[config.ID_COLUMN]) == list(range(1, len(synthetic_df) + 1))
        assert list(loaded.columns) == config.EXPECTED_COLUMNS

    def test_non_numeric_lab_value_is_coerced_to_nan(self, tmp_path, synthetic_df):
        df = synthetic_df.drop(columns=[config.ID_COLUMN])
        df["serum_sodium"] = df["serum_sodium"].astype(object)
        df.loc[0, "serum_sodium"] = "N/A"
        path = tmp_path / "raw.csv"
        df.to_csv(path, index=False)

        loaded = load_raw_data(path)
        assert pd.isna(loaded.loc[0, "serum_sodium"])
        assert pd.api.types.is_numeric_dtype(loaded["serum_sodium"])


class TestProfile:
    def test_profile_counts_are_consistent(self, synthetic_df):
        p = profile_dataset(synthetic_df)
        assert p.n_positive + p.n_negative == p.n_rows
        assert p.positive_rate == pytest.approx(p.n_positive / p.n_rows)
        assert p.n_duplicate_ids == 0

    def test_synthetic_generator_respects_positive_rate(self):
        df = make_synthetic_dataset(n_rows=1000, positive_rate=0.321, random_state=11)
        assert (df[config.TARGET_COLUMN] == 1).sum() == 321
        assert df["serum_creatinine"].isna().sum() > 0
        assert list(df.columns) == config.EXPECTED_COLUMNS

    def test_synthetic_generator_can_disable_missing_values(self):
        df = make_synthetic_dataset(n_rows=200, missing_rate=0.0)
        assert df.isna().sum().sum() == 0


class TestFeatureSelection:
    def test_features_are_exactly_the_baseline_columns(self, synthetic_df):
        X, y = split_features_target(synthetic_df)
        assert list(X.columns) == config.FEATURE_COLUMNS
        assert len(y) == len(X)

    def test_keep_id_prepends_identifier(self, synthetic_df):
        X, _ = split_features_target(synthetic_df, keep_id=True)
        assert X.columns[0] == config.ID_COLUMN
