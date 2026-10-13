"""Unit tests for the model factory."""

from __future__ import annotations

import pytest

from src import models


class TestScalingRule:
    @pytest.mark.parametrize(
        "name", ["random_forest", "extra_trees", "gradient_boosting", "decision_tree", "xgboost"]
    )
    def test_tree_models_are_not_scaled(self, name):
        assert not models.needs_scaling(name)

    @pytest.mark.parametrize("name", ["logistic_regression", "svm", "knn", "naive_bayes"])
    def test_distance_and_linear_models_are_scaled(self, name):
        assert models.needs_scaling(name)

    def test_name_normalization(self):
        assert not models.needs_scaling("Random-Forest")


class TestFactory:
    def test_unknown_model_gives_valid_options(self):
        with pytest.raises(KeyError, match="Valid options"):
            models.get_model("magic_forest")


class TestNewModels:
    def test_new_models_are_listed(self):
        assert {"extra_trees", "mlp"} <= set(models.available_models())

    def test_extra_trees_honours_class_weight_and_overrides(self):
        model = models.get_model("extra_trees", class_weight="balanced", n_estimators=50)
        assert type(model).__name__ == "ExtraTreesClassifier"
        assert model.class_weight == "balanced"
        assert model.n_estimators == 50

    def test_mlp_has_no_early_stopping(self):
        model = models.get_model("mlp")
        assert type(model).__name__ == "MLPClassifier"
        assert model.early_stopping is False

    @pytest.mark.parametrize("name", ["extra_trees", "mlp"])
    def test_new_models_fit_inside_leakage_safe_pipeline(self, synthetic_df, name):
        from src.data_loader import split_features_target
        from src.preprocessing import (
            assert_no_fit_on_holdout,
            make_full_pipeline,
            patient_level_split,
        )

        split = patient_level_split(synthetic_df, verify=False)
        X_train, y_train = split_features_target(split.train)
        X_test, _ = split_features_target(split.test)
        pipeline = make_full_pipeline(
            models.get_model(name), scale=models.needs_scaling(name)
        )
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        assert proba.shape == (len(X_test),)
        assert ((proba >= 0) & (proba <= 1)).all()
        assert assert_no_fit_on_holdout(pipeline, X_test)

    def test_default_training_set_includes_extra_trees(self):
        from src.train import DEFAULT_MODELS

        assert "extra_trees" in DEFAULT_MODELS
        assert set(DEFAULT_MODELS) <= set(models.available_models())


class TestOverrides:
    @pytest.mark.parametrize(
        ("name", "param", "value"),
        [
            ("random_forest", "n_estimators", 25),
            ("logistic_regression", "max_iter", 50),
            ("gradient_boosting", "learning_rate", 0.2),
            ("mlp", "hidden_layer_sizes", (8, 4)),
        ],
    )
    def test_overrides_replace_defaults(self, name, param, value):
        """Overriding a default hyperparameter must not raise a duplicate-kwarg TypeError."""
        model = models.get_model(name, **{param: value})
        assert model.get_params()[param] == value

    def test_new_kwargs_are_passed_through(self):
        model = models.get_model("random_forest", max_features=0.5)
        assert model.max_features == 0.5
