"""Unit tests for the pooled UCI + Zigong missing-LVEF analysis (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src import harmonization as hz
from src import pooled_missing as pm
from src.data_loader import make_synthetic_dataset


@pytest.fixture(scope="module")
def pooled():
    return hz.harmonize_both(
        make_synthetic_dataset(n_rows=300, random_state=3),
        hz.make_synthetic_zigong(n_rows=600, death_rate=0.08, random_state=4),
        horizon_days=90,
    )


class TestScenarios:
    def test_complete_case_drops_rows_without_lvef(self, pooled):
        data, features = pm.scenario_data(pooled, "complete_case")
        assert data[pm.LVEF].notna().all()
        assert len(data) == pooled[pm.LVEF].notna().sum()
        assert pm.LVEF in features

    def test_drop_lvef_keeps_all_rows(self, pooled):
        data, features = pm.scenario_data(pooled, "drop_lvef")
        assert len(data) == len(pooled)
        assert pm.LVEF not in features

    def test_imputation_keeps_rows_and_feature(self, pooled):
        data, features = pm.scenario_data(pooled, "impute_mice")
        assert len(data) == len(pooled) and pm.LVEF in features

    def test_source_feature_is_optional(self, pooled):
        data, features = pm.scenario_data(pooled, "drop_lvef", source_feature=True)
        assert pm.SOURCE_FEATURE in features
        assert set(data[pm.SOURCE_FEATURE].unique()) <= {0.0, 1.0}
        assert pm.SOURCE_FEATURE not in pm.scenario_data(pooled, "drop_lvef")[1]

    def test_unknown_imputer_rejected(self):
        with pytest.raises(ValueError, match="Unknown imputer"):
            pm.make_imputer("mean")


class TestPreprocessor:
    @pytest.mark.parametrize("imputer", ["median", "mice", "knn"])
    def test_output_has_no_missing_values(self, pooled, imputer):
        features = list(hz.HARMONIZED_FEATURES)
        out = pm.make_pooled_preprocessor(features, imputer=imputer).fit_transform(pooled[features])
        assert not np.isnan(out).any()
        assert out.shape == (len(pooled), len(features))

    def test_indicator_adds_columns_for_missing_features(self, pooled):
        features = list(hz.HARMONIZED_FEATURES)
        out = pm.make_pooled_preprocessor(features, add_indicator=True).fit_transform(pooled[features])
        n_with_missing = int(pooled[features].isna().any().sum())
        assert out.shape[1] == len(features) + n_with_missing

    def test_log_transform_is_selected_by_name(self, pooled):
        """An all-missing column must not shift the log transform onto another column."""
        features = list(hz.HARMONIZED_FEATURES)
        X = pooled[features].copy()
        X[pm.LVEF] = np.nan
        pre = pm.make_pooled_preprocessor(features).fit(X)
        names = list(pre.named_steps["log"].get_feature_names_out())
        assert names[: len(pm.SKEWED)] == [c for c in features if c in pm.SKEWED]

    def test_fit_uses_train_rows_only(self, pooled):
        """Scaler/imputer statistics must not change when held-out rows are transformed."""
        features = list(hz.HARMONIZED_FEATURES)
        train, test = pooled.iloc[:500], pooled.iloc[500:]
        pre = pm.make_pooled_preprocessor(features, imputer="median").fit(train[features])
        mean_before = pre.named_steps["scale"].mean_.copy()
        stats_before = pre.named_steps["impute"].statistics_.copy()
        pre.transform(test[features])
        np.testing.assert_array_equal(mean_before, pre.named_steps["scale"].mean_)
        np.testing.assert_array_equal(stats_before, pre.named_steps["impute"].statistics_)


class TestEvaluation:
    def test_evaluate_scenario_row(self, pooled):
        row = pm.evaluate_scenario(pooled, scenario="impute_median", model_name="logistic_regression",
                                   n_repeats=2)
        assert row["n_rows"] == len(pooled)
        assert row["zigong_n"] + row["uci_n"] == len(pooled)
        assert row["common_n"] == row["uci_n"] + row["zigong_lvef_n"]
        for key in ("pooled_auc", "uci_auc", "zigong_auc"):
            assert 0 <= row[key] <= 1

    def test_complete_case_scores_only_lvef_rows(self, pooled):
        row = pm.evaluate_scenario(pooled, scenario="complete_case", model_name="logistic_regression",
                                   n_repeats=1)
        assert row["zigong_n"] == row["zigong_lvef_n"]

    def test_imputation_accuracy_median_has_no_spread(self, pooled):
        acc = pm.imputation_accuracy(pooled, imputers=("median", "mice"))
        median = acc.set_index("imputer").loc["median"]
        # One median per training fold: almost no spread compared with the true values.
        assert median["sd_imputed"] < 0.1 * median["sd_true"]
        assert (acc["n_masked"] == ((pooled["source"] == "zigong") & pooled[pm.LVEF].notna()).sum()).all()
        assert (acc["rmse"] >= acc["mae"]).all()

    def test_missingness_summary_is_aggregate(self, pooled):
        summary = pm.missingness_summary(pooled)
        assert summary["n"].sum() == (pooled["source"] == "zigong").sum()
        assert hz.ID_COLUMN not in summary.columns

    def test_script_runs_on_synthetic_data(self, tmp_path):
        import importlib.util

        path = config.PROJECT_ROOT / "scripts" / "run_pooled_missing.py"
        spec = importlib.util.spec_from_file_location("run_pooled_missing", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.main(["--synthetic", "--n-repeats", "1", "--models", "logistic_regression",
                            "--output", str(tmp_path)]) == 0
        for name in ("pooled_scenarios.csv", "lvef_imputation_accuracy.csv", "lvef_missingness.csv"):
            assert (tmp_path / name).exists()
        assert isinstance(pd.read_csv(tmp_path / "pooled_scenarios.csv"), pd.DataFrame)
