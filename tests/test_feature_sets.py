"""Unit tests for named feature sets and the paired feature-set comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, preprocessing, train
from src.data_loader import split_features_target
from src.models import get_model


class TestFeatureSetConfig:
    def test_all_is_the_default_feature_list(self):
        numeric, binary = config.get_feature_set("all")
        assert numeric + binary == config.FEATURE_COLUMNS

    def test_minimal_two_is_creatinine_and_ef(self):
        numeric, binary = config.get_feature_set("minimal_two")
        assert set(numeric) == {"ejection_fraction", "serum_creatinine"}
        assert binary == []

    def test_unknown_feature_set_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown feature set"):
            config.get_feature_set("everything")

    def test_returned_lists_are_copies(self):
        numeric, _ = config.get_feature_set("all")
        numeric.append("time")
        assert "time" not in config.get_feature_set("all")[0]

    @pytest.mark.parametrize("name", list(config.FEATURE_SETS))
    def test_no_feature_set_contains_leakage_or_id(self, name):
        numeric, binary = config.get_feature_set(name)
        forbidden = set(config.LEAKAGE_COLUMNS) | {config.ID_COLUMN, config.TARGET_COLUMN}
        assert not forbidden & set(numeric + binary)


class TestMinimalPipeline:
    def test_split_selects_two_columns(self, synthetic_df):
        X, y = split_features_target(synthetic_df, feature_set="minimal_two")
        assert list(X.columns) == ["ejection_fraction", "serum_creatinine"]
        assert len(y) == len(X)

    def test_log_applies_only_to_serum_creatinine(self, synthetic_df):
        X, y = split_features_target(synthetic_df, feature_set="minimal_two")
        X = X.fillna(X.median())
        pre = preprocessing.make_preprocessor(scale=False, feature_set="minimal_two").fit(X, y)
        frame = pd.DataFrame(pre.transform(X), columns=pre.get_feature_names_out())
        np.testing.assert_allclose(frame["serum_creatinine"], np.log1p(X["serum_creatinine"]))
        np.testing.assert_allclose(frame["ejection_fraction"], X["ejection_fraction"])

    def test_minimal_pipeline_fits_without_leakage(self, synthetic_df):
        split = preprocessing.patient_level_split(synthetic_df, verify=False)
        X_train, y_train = split_features_target(split.train, feature_set="minimal_two")
        X_test, _ = split_features_target(split.test, feature_set="minimal_two")
        pipeline = preprocessing.make_full_pipeline(
            get_model("logistic_regression"), feature_set="minimal_two"
        )
        pipeline.fit(X_train, y_train)
        assert list(pipeline.named_steps["preprocessor"].get_feature_names_out()) == [
            "ejection_fraction", "serum_creatinine"
        ]
        assert preprocessing.assert_no_fit_on_holdout(pipeline, X_test)

    def test_stacking_respects_feature_set(self, synthetic_df):
        X, y = split_features_target(synthetic_df, feature_set="minimal_two")
        stacking = train.build_pipeline("stacking", feature_set="minimal_two")
        stacking.fit(X, y)
        assert stacking.predict_proba(X).shape == (len(X), 2)


class TestPairedComparison:
    def test_identical_feature_sets_give_zero_difference(self, synthetic_df):
        """Same folds for both arms: comparing a set with itself must give exactly 0."""
        table, _ = train.compare_feature_sets(
            synthetic_df, ["logistic_regression"], feature_sets=("all", "all"), n_splits=3
        )
        assert set(table["metric"]) == set(train.COMPARISON_METRICS)
        np.testing.assert_allclose(table["mean_diff"], 0.0, atol=1e-12)

    def test_comparison_table_and_csv(self, synthetic_df, tmp_path):
        table, folds = train.compare_feature_sets(
            synthetic_df, ["logistic_regression"], n_splits=3, output_dir=tmp_path
        )
        assert (tmp_path / "minimal_vs_full_feature_comparison.csv").exists()
        assert {"mean_all", "mean_minimal_two", "mean_diff", "diff_ci95_low"} <= set(table.columns)
        assert (table["diff_ci95_low"] <= table["mean_diff"]).all()
        assert (table["mean_diff"] <= table["diff_ci95_high"]).all()
        assert len(folds[("logistic_regression", "all")]) == 3
        assert len(folds[("logistic_regression", "minimal_two")]) == 3

    def test_comparison_never_touches_the_test_split(self, synthetic_df, monkeypatch):
        """Only train+val patients may reach the cross-validation."""
        test_ids = set(preprocessing.patient_level_split(synthetic_df, verify=False).test[
            config.ID_COLUMN
        ])
        seen: list[set] = []
        original = train.evaluation.cross_validate_grouped

        def spy(pipeline, X, y, groups, **kwargs):
            seen.append(set(groups))
            return original(pipeline, X, y, groups, **kwargs)

        monkeypatch.setattr(train.evaluation, "cross_validate_grouped", spy)
        train.compare_feature_sets(synthetic_df, ["logistic_regression"], n_splits=3)
        assert seen and all(ids.isdisjoint(test_ids) for ids in seen)

    def test_cli_compare_mode(self, tmp_path):
        code = train.main(
            ["--synthetic", "--models", "logistic_regression", "--feature-set", "compare",
             "--output", str(tmp_path)]
        )
        assert code == 0
        assert (tmp_path / "minimal_vs_full_feature_comparison.csv").exists()
