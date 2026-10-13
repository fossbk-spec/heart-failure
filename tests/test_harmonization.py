"""Unit tests for UCI <-> Zigong harmonization (synthetic data only - Zigong is restricted)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src import harmonization as hz


@pytest.fixture(scope="module")
def zigong_df():
    return hz.make_synthetic_zigong(n_rows=500, random_state=3)


class TestUciLabels:
    def test_fixed_horizon_label_rules(self):
        df = pd.DataFrame(
            {
                config.TARGET_COLUMN: [1, 1, 0, 0, 1],
                "time": [30, 120, 200, 60, 90],
            }
        )
        label = hz.uci_fixed_horizon_label(df, 90)
        # died early -> 1; died after horizon -> 0; alive past horizon -> 0;
        # alive but censored before horizon -> unknown; died exactly at horizon -> 1.
        assert label.tolist()[:3] == [1.0, 0.0, 0.0]
        assert np.isnan(label.iloc[3])
        assert label.iloc[4] == 1.0

    def test_harmonize_uci_drops_unknown_labels(self, synthetic_df):
        out = hz.harmonize_uci(synthetic_df, horizon_days=90)
        expected = hz.uci_fixed_horizon_label(synthetic_df, 90).notna().sum()
        assert len(out) == expected
        assert out[hz.outcome_column(90)].isin([0.0, 1.0]).all()


class TestUnitsAndDerivedVariables:
    def test_age_band_mapping_matches_zigong_bands(self):
        ages = pd.Series([40, 49, 49.5, 60.667, 89, 95])
        assert hz.age_to_band_value(ages).tolist() == [45.0, 45.0, 55.0, 65.0, 85.0, 95.0]
        labels = pd.Series(["(39,49]", "(89,110]"])
        assert hz.age_band_label_to_value(labels).tolist() == [45.0, 95.0]

    def test_unknown_age_band_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown Zigong age bands"):
            hz.age_band_label_to_value(pd.Series(["(110,120]"]))

    def test_anaemia_uses_sex_specific_who_thresholds(self):
        hb = pd.Series([125.0, 125.0, 135.0, np.nan])
        sex = pd.Series([1.0, 0.0, 1.0, 0.0])
        out = hz.anaemia_from_hemoglobin(hb, sex)
        assert out.tolist()[:3] == [1.0, 0.0, 0.0]
        assert np.isnan(out.iloc[3])

    def test_zigong_unit_conversions(self, zigong_df):
        out = hz.harmonize_zigong(zigong_df, horizon_days=90)
        np.testing.assert_allclose(
            out["serum_creatinine"], zigong_df["creatinine.enzymatic.method"] / 88.4
        )
        np.testing.assert_allclose(out["platelets"], zigong_df["platelet"])

    def test_uci_platelets_converted_to_1e9_per_l(self, synthetic_df):
        out = hz.harmonize_uci(synthetic_df, horizon_days=28)
        assert out["platelets"].median() == pytest.approx(
            synthetic_df["platelets"].median() / 1000, rel=0.2
        )

    def test_creatinine_scales_agree_after_conversion(self, synthetic_df, zigong_df):
        """Both cohorts must land on the same mg/dL scale (medians ~1 mg/dL)."""
        u = hz.harmonize_uci(synthetic_df, horizon_days=28)["serum_creatinine"].median()
        z = hz.harmonize_zigong(zigong_df, horizon_days=28)["serum_creatinine"].median()
        assert 0.6 < u < 1.8 and 0.6 < z < 1.8


class TestSchema:
    @pytest.mark.parametrize("horizon", [28, 90, 180])
    def test_both_cohorts_share_one_schema(self, synthetic_df, zigong_df, horizon):
        u = hz.harmonize_uci(synthetic_df, horizon_days=horizon)
        z = hz.harmonize_zigong(zigong_df, horizon_days=horizon)
        assert list(u.columns) == list(z.columns) == hz.harmonized_columns(horizon)

    def test_zigong_label_comes_from_matching_column(self, zigong_df):
        for horizon, column in hz.ZIGONG_DEATH_COLUMNS.items():
            out = hz.harmonize_zigong(zigong_df, horizon_days=horizon)
            assert out[hz.outcome_column(horizon)].sum() == zigong_df[column].sum()

    def test_unsupported_zigong_horizon_is_rejected(self, zigong_df):
        with pytest.raises(ValueError, match="28, 90, 180"):
            hz.harmonize_zigong(zigong_df, horizon_days=365)

    def test_ids_are_unique_and_prefixed_by_source(self, synthetic_df, zigong_df):
        both = hz.harmonize_both(synthetic_df, zigong_df, horizon_days=90)
        assert both[hz.ID_COLUMN].is_unique
        assert set(both[hz.SOURCE_COLUMN]) == {"uci", "zigong"}
        for source, prefix in (("uci", "uci-"), ("zigong", "zg-")):
            ids = both.loc[both[hz.SOURCE_COLUMN] == source, hz.ID_COLUMN]
            assert ids.str.startswith(prefix).all()

    def test_no_leakage_or_dropped_columns_in_features(self):
        forbidden = {"time", "high_blood_pressure", "smoking", hz.ID_COLUMN, hz.SOURCE_COLUMN}
        assert not forbidden & set(hz.HARMONIZED_FEATURES)

    def test_uci_like_filter(self, zigong_df):
        out = hz.harmonize_zigong(zigong_df, horizon_days=90, uci_like=True)
        nyha = zigong_df["NYHA.cardiac.function.classification"].isin(["III", "IV"])
        old = hz.age_band_label_to_value(zigong_df["ageCat"]) >= 45
        assert len(out) == int((nyha & old).sum())
        assert (out["age"] >= 45).all()

    def test_harmonization_does_not_mutate_inputs(self, synthetic_df, zigong_df):
        u_before, z_before = synthetic_df.copy(), zigong_df.copy()
        hz.harmonize_both(synthetic_df, zigong_df)
        pd.testing.assert_frame_equal(synthetic_df, u_before)
        pd.testing.assert_frame_equal(zigong_df, z_before)


class TestLoaderAndSummary:
    def test_missing_zigong_file_explains_access(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="DUA"):
            hz.load_zigong_raw(tmp_path / "dat.csv")

    def test_cohort_summary_is_aggregate_only(self, synthetic_df, zigong_df):
        summary = hz.cohort_summary(hz.harmonize_both(synthetic_df, zigong_df), 90)
        assert set(summary["source"]) == {"uci", "zigong"}
        assert hz.ID_COLUMN not in summary.columns
        assert summary["death_rate"].between(0, 1).all()

    def test_inspect_script_runs_on_synthetic_data(self, capsys):
        import importlib.util

        path = config.PROJECT_ROOT / "scripts" / "inspect_zigong.py"
        spec = importlib.util.spec_from_file_location("inspect_zigong", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.main(["--synthetic"]) == 0
        assert "Events per horizon" in capsys.readouterr().out
