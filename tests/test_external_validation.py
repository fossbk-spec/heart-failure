"""Unit tests for external validation (synthetic data only - Zigong is restricted)."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src import external_validation as ev
from src import harmonization as hz
from src.data_loader import make_synthetic_dataset


@pytest.fixture(scope="module")
def harmonized():
    uci = hz.harmonize_uci(make_synthetic_dataset(n_rows=400, random_state=5), horizon_days=90)
    zigong = hz.harmonize_zigong(
        hz.make_synthetic_zigong(n_rows=800, death_rate=0.08, random_state=6), horizon_days=90
    )
    return uci, zigong


class TestCalibrationMetrics:
    def test_perfectly_calibrated_predictions(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.02, 0.6, 20_000)
        y = rng.binomial(1, p)
        intercept, slope = ev.calibration_intercept_slope(y, p)
        assert intercept == pytest.approx(0.0, abs=0.05)
        assert slope == pytest.approx(1.0, abs=0.05)

    def test_overprediction_gives_negative_intercept(self):
        rng = np.random.default_rng(1)
        p_true = rng.uniform(0.01, 0.1, 20_000)
        y = rng.binomial(1, p_true)
        p_model = np.clip(p_true * 5, 0, 0.99)  # risk over-estimated 5x
        intercept, _ = ev.calibration_intercept_slope(y, p_model)
        assert intercept < -1.0

    def test_intercept_update_matches_observed_rate(self):
        rng = np.random.default_rng(2)
        y = rng.binomial(1, 0.03, 5_000)
        p = rng.uniform(0.2, 0.4, 5_000)
        recal = ev.recalibrate_intercept(y, p)
        assert recal.mean() == pytest.approx(y.mean(), abs=0.002)
        # A monotone shift on the logit scale never changes the ranking.
        assert np.array_equal(np.argsort(recal), np.argsort(p))

    def test_bootstrap_ci_brackets_point_estimate(self):
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(3)
        y = rng.binomial(1, 0.1, 1_000)
        p = np.clip(0.1 + 0.3 * y + rng.normal(0, 0.2, 1_000), 0, 1)
        low, high = ev.bootstrap_auc_ci(y, p, n_boot=200)
        assert low <= roc_auc_score(y, p) <= high


class TestFeatureSets:
    def test_split_columns(self):
        numeric, binary = ev.split_columns(ev.EXTERNAL_FEATURE_SETS["all9"])
        assert set(numeric) == set(hz.HARMONIZED_NUMERIC)
        assert set(binary) == set(hz.HARMONIZED_BINARY)

    def test_unknown_feature_is_rejected(self):
        with pytest.raises(ValueError, match="Not harmonized"):
            ev.split_columns(["smoking"])

    def test_no_ef_set_excludes_lvef(self):
        assert "ejection_fraction" not in ev.EXTERNAL_FEATURE_SETS["no_ef"]

    def test_lvef_models_only_apply_to_measured_lvef(self, harmonized):
        _, zigong = harmonized
        rows = ev.applicable_rows(zigong, ev.EXTERNAL_FEATURE_SETS["minimal_two"])
        assert rows.sum() == zigong["ejection_fraction"].notna().sum()
        assert ev.applicable_rows(zigong, ev.EXTERNAL_FEATURE_SETS["no_ef"]).all()


class TestValidation:
    @pytest.mark.parametrize("feature_set", list(ev.EXTERNAL_FEATURE_SETS))
    def test_validate_model_row(self, harmonized, feature_set):
        uci, zigong = harmonized
        result = ev.validate_model(
            uci, zigong, model_name="logistic_regression", feature_set=feature_set,
            horizon_days=90, n_boot=50,
        )
        row = result.row
        assert row["zigong_n"] == len(result.zigong_y)
        assert 0 <= row["external_auc"] <= 1
        assert row["auc_ci95_low"] <= row["external_auc"] <= row["auc_ci95_high"]
        assert row["brier_after_intercept_update"] <= row["brier"] + 1e-9
        assert 0 <= row["sensitivity_at_uci_threshold"] <= 1

    def test_zigong_rows_never_reach_fit(self, harmonized, monkeypatch):
        """Every fit (CV folds + final model) must only see UCI rows."""
        uci, zigong = harmonized
        seen = []
        original = ev.build_external_pipeline

        def spy_builder(model_name, features):
            pipeline = original(model_name, features)
            fit = type(pipeline).fit

            def spy_fit(self, X, y=None, **params):
                seen.append(X)
                return fit(self, X, y, **params)

            monkeypatch.setattr(type(pipeline), "fit", spy_fit)
            return pipeline

        monkeypatch.setattr(ev, "build_external_pipeline", spy_builder)
        ev.validate_model(
            uci, zigong, model_name="logistic_regression", feature_set="no_ef",
            horizon_days=90, n_boot=20,
        )
        assert len(seen) == config.N_SPLITS + 1
        uci_index = set(uci.index)
        for X in seen:
            assert set(X.index) <= uci_index
            assert len(X) <= len(uci)

    def test_full_grid_marks_primary_analysis(self):
        table = ev.run_external_validation(
            make_synthetic_dataset(n_rows=400, random_state=8),
            hz.make_synthetic_zigong(n_rows=600, death_rate=0.08, random_state=9),
            horizons=(90, 180), uci_like_options=(False, True),
            feature_sets=("no_ef",), model_names=("logistic_regression",), n_boot=20,
        )
        assert len(table) == 4
        primary = table[table["primary"]]
        assert len(primary) == 1
        assert primary.iloc[0]["horizon_days"] == 90 and not primary.iloc[0]["uci_like"]

    def test_script_runs_on_synthetic_data(self, tmp_path):
        import importlib.util

        path = config.PROJECT_ROOT / "scripts" / "run_external_validation.py"
        spec = importlib.util.spec_from_file_location("run_external_validation", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = ["--synthetic", "--n-boot", "10", "--models", "logistic_regression",
                "--output", str(tmp_path)]
        assert module.main(args) == 0
        assert (tmp_path / "external_validation.csv").exists()
