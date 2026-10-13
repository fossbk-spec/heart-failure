"""Unit tests for the EDA module: every function must run end-to-end without errors.

The assignment explicitly requires: "EDA source code must be packaged into
reusable functions, executable end-to-end without errors" -> this test suite
enforces exactly that.
"""

from __future__ import annotations

import pandas as pd

from src import config, eda


class TestEdaTables:
    def test_missing_value_report_flags_missing_labs(self, synthetic_df):
        report = eda.missing_value_report(synthetic_df).set_index("feature")
        assert report.loc["serum_creatinine", "n_missing"] > 0
        assert report["top_value_share"].between(0, 1).all()

    def test_suspected_imputation_detects_mean_fill(self, synthetic_df):
        """A non-integer value repeated many times (a mean fill-in) must be flagged."""
        df = synthetic_df.copy()
        df.loc[df.index[:30], "platelets"] = 263358.03  # the value seen in the real file
        report = eda.suspected_imputation_report(df).set_index("feature")
        assert report.loc["platelets", "top_count"] == 30
        assert bool(report.loc["platelets", "non_integer_repeated"])

    def test_class_balance_reports_both_levels(self, synthetic_df):
        report = eda.class_balance_report(synthetic_df)
        assert set(report["level"]) == {"sample-level", "patient-level"}
        assert (report["positive_rate"].between(0.2, 0.45)).all()

    def test_patient_level_differs_when_records_are_duplicated(self, multi_record_df):
        """With multiple records per patient, the two levels must give different counts."""
        report = eda.class_balance_report(multi_record_df).set_index("level")
        assert report.loc["sample-level", "n_total"] > report.loc["patient-level", "n_total"]

    def test_age_bands_cover_cohort(self, synthetic_df):
        bands = eda.death_rate_by_age_band(synthetic_df)
        assert bands["n_samples"].sum() == len(synthetic_df)
        assert bands["death_rate"].between(0, 1).all()

    def test_ejection_fraction_strata_follow_esc_classes(self, synthetic_df):
        strata = eda.ejection_fraction_strata(synthetic_df)
        assert set(strata["ef_stratum"].astype(str)) <= {
            "<=40 (HFrEF)", "41-49 (HFmrEF)", ">=50 (HFpEF)"
        }
        assert strata["n_samples"].sum() == synthetic_df["ejection_fraction"].notna().sum()
        assert "risk_ratio_vs_reference" in strata.columns

    def test_ef_boundaries_are_exact(self):
        df = pd.DataFrame(
            {"ejection_fraction": [40, 41, 49, 50], config.TARGET_COLUMN: [1, 0, 0, 0]}
        )
        strata = eda.ejection_fraction_strata(df).set_index("ef_stratum")
        assert strata.loc["<=40 (HFrEF)", "n_samples"] == 1
        assert strata.loc["41-49 (HFmrEF)", "n_samples"] == 2
        assert strata.loc[">=50 (HFpEF)", "n_samples"] == 1

    def test_creatinine_and_sodium_strata(self, synthetic_df):
        creat = eda.serum_creatinine_strata(synthetic_df)
        sodium = eda.serum_sodium_strata(synthetic_df)
        assert creat["n_samples"].sum() == synthetic_df["serum_creatinine"].notna().sum()
        assert sodium["n_samples"].sum() == len(synthetic_df)

    def test_correlation_table_excludes_leakage_column(self, synthetic_df):
        corr = eda.correlation_with_target(synthetic_df)
        assert list(corr.columns) == ["feature", "corr_with_death"]
        assert set(corr["feature"]) == set(config.FEATURE_COLUMNS)

    def test_follow_up_leakage_ranks_time_first(self, synthetic_df):
        report = eda.follow_up_leakage_report(synthetic_df)
        top = report.iloc[0]
        assert top["feature"] == "time" and bool(top["is_leakage"])
        assert top["median_died"] < top["median_survived"]

    def test_demographic_bias_covers_all_binary_attributes(self, synthetic_df):
        report = eda.demographic_bias_report(synthetic_df)
        assert set(config.BINARY_FEATURES) == set(report["attribute"])
        assert report["share_of_cohort"].between(0, 1).all()

    def test_numeric_summary_splits_by_label(self, synthetic_df):
        summary = eda.numeric_summary(synthetic_df)
        assert set(summary[config.TARGET_COLUMN]) == {0, 1}
        assert set(summary["feature"]) == set(config.NUMERIC_FEATURES)


class TestEdaOrchestration:
    def test_run_full_eda_executes_end_to_end(self, synthetic_df, tmp_path):
        tables = eda.run_full_eda(synthetic_df, output_dir=tmp_path)
        expected = {
            "missing_values", "suspected_imputation", "class_balance", "numeric_summary",
            "age_bands", "ejection_fraction_strata", "serum_creatinine_strata",
            "serum_sodium_strata", "correlation", "follow_up_leakage", "demographic_bias",
        }
        assert expected == set(tables)
        for name, table in tables.items():
            assert isinstance(table, pd.DataFrame)
            assert not table.empty, f"Table '{name}' is empty"
            assert (tmp_path / f"eda_{name}.csv").exists()

    def test_eda_does_not_mutate_input(self, synthetic_df):
        before = synthetic_df.copy()
        eda.run_full_eda(synthetic_df)
        pd.testing.assert_frame_equal(synthetic_df, before)

    def test_overview_figure_is_saved(self, synthetic_df, tmp_path):
        path = tmp_path / "overview.png"
        eda.plot_eda_overview(synthetic_df, save_path=path)
        assert path.exists() and path.stat().st_size > 0
