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
