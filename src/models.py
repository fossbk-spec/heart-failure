"""Model factories: baseline, cost-sensitive, and advanced ensemble models.

Model-selection rationale for this problem:

- **Logistic Regression** (mandatory baseline): linear, coefficients can be
  read directly as odds ratios -> very easy to discuss clinically.
- **Cost-sensitive learning** (`class_weight="balanced"`): an imbalance
  countermeasure that does NOT synthesize fake data, so it carries no
  "SMOTE leakage" risk. This is an important control arm against the
  SMOTE-based branch.
- **Random Forest / Gradient Boosting / XGBoost**: capture nonlinear
  interactions (e.g. ejection fraction x serum creatinine) that a linear model misses.
- **Extra Trees**: the best model of Ishaq et al. (2021) on this dataset; more
  randomized splits than Random Forest, so lower variance on small samples.
- **MLP**: a small regularized neural network, kept as an exploratory arm —
  299 patients are far too few for deep models.
- **Stacking Ensemble**: base learners + a Logistic Regression meta-learner,
  with preprocessing + SMOTE nested inside each base learner so the
  stacking's internal CV never sees resampled data (a clean data-splitting
  protocol, unlike pipelines that oversample before CV).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import config

# --------------------------------------------------------------------------- #
# XGBoost is an OPTIONAL dependency (needs OpenMP/libomp on macOS).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - environment-dependent
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBOOST_AVAILABLE = False


def get_model(name: str, *, class_weight: str | None = None, **overrides: Any):
    """Return an estimator by name.

    Args:
        name: one of the keys in `available_models()`.
        class_weight: "balanced" to enable cost-sensitive learning (if the
            model supports it). Do not combine with SMOTE unless intentional.
        **overrides: override default hyperparameters.
    """
    name = name.lower().replace("-", "_")
    rs = config.RANDOM_STATE
    cw = {"class_weight": class_weight}
    # (estimator class, default hyperparameters); `overrides` win over defaults.
    specs: dict[str, tuple[type, dict[str, Any]]] = {
        "logistic_regression": (
            LogisticRegression,
            {"max_iter": 2000, "solver": "liblinear", "random_state": rs, **cw},
        ),
        "decision_tree": (
            DecisionTreeClassifier,
            {"max_depth": 6, "min_samples_leaf": 20, "random_state": rs, **cw},
        ),
        "random_forest": (
            RandomForestClassifier,
            {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 3, "n_jobs": -1,
             "random_state": rs, **cw},
        ),
        "extra_trees": (
            ExtraTreesClassifier,
            {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 3, "n_jobs": -1,
             "random_state": rs, **cw},
        ),
        # Small, regularized network without early stopping: with ~190 training
        # patients an internal early-stopping split would hold ~19 rows, part of
        # them SMOTE-synthesized. L-BFGS converges reliably on small datasets.
        # MLP has no `class_weight`; rely on SMOTE.
        "mlp": (
            MLPClassifier,
            {"hidden_layer_sizes": (16,), "alpha": 1e-2, "solver": "lbfgs", "max_iter": 2000,
             "random_state": rs},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier,
            {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "random_state": rs},
        ),
        "svm": (
            SVC,
            {"kernel": "rbf", "C": 1.0, "probability": True, "random_state": rs, **cw},
        ),
        "knn": (KNeighborsClassifier, {"n_neighbors": 15, "weights": "distance"}),
        "naive_bayes": (GaussianNB, {}),
    }
    builders: dict[str, Any] = {
        key: (lambda cls=cls, defaults=defaults: cls(**{**defaults, **overrides}))
        for key, (cls, defaults) in specs.items()
    }
    builders["xgboost"] = lambda: _build_xgboost(**overrides)
    builders["stacking"] = lambda: build_stacking_ensemble(class_weight=class_weight)
    if name not in builders:
        raise KeyError(f"Model '{name}' does not exist. Valid options: {sorted(builders)}")
    return builders[name]()


def _build_xgboost(**overrides: Any):
    """Instantiate XGBoost; raise a clear error if libomp is missing."""
    if not XGBOOST_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "Failed to load xgboost (missing OpenMP runtime). "
            "On macOS: `brew install libomp`. Or use 'gradient_boosting' instead."
        )
    params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "aucpr",
        "random_state": config.RANDOM_STATE,
        "n_jobs": -1,
    }
    params.update(overrides)
    return XGBClassifier(**params)


def build_stacking_ensemble(
    *,
    class_weight: str | None = None,
    base_wrapper: Callable[[Any, bool], Any] | None = None,
) -> StackingClassifier:
    """Stacking Ensemble: diverse base learners + Logistic Regression meta-learner.

    The internal `cv=5` in `StackingClassifier` ensures meta-features are
    generated from out-of-fold predictions => no leakage from base to meta.

    `base_wrapper(estimator, scale)` wraps each base learner (e.g. into a
    preprocessor + SMOTE pipeline) so that SMOTE runs only on the train part of
    each INTERNAL fold. Resampling before the internal CV would let synthetic
    samples and their source records fall into different folds, making the
    out-of-fold meta-features optimistic.
    """
    base_estimators = [
        ("rf", get_model("random_forest", class_weight=class_weight)),
        ("gb", get_model("gradient_boosting")),
        ("nb", get_model("naive_bayes")),
        ("dt", get_model("decision_tree", class_weight=class_weight)),
    ]
    if base_wrapper is not None:
        # Only distance/likelihood-based learners benefit from scaling.
        base_estimators = [
            (name, base_wrapper(est, name == "nb")) for name, est in base_estimators
        ]
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=2000, class_weight=class_weight),
        cv=5,
        stack_method="predict_proba",
        n_jobs=None,
    )


# Tree ensembles split on thresholds, so feature scaling changes nothing.
UNSCALED_MODELS = frozenset(
    {"random_forest", "extra_trees", "gradient_boosting", "decision_tree", "xgboost"}
)


def needs_scaling(name: str) -> bool:
    """Whether the model's preprocessing should standardize continuous features."""
    return name.lower().replace("-", "_") not in UNSCALED_MODELS


def available_models() -> list[str]:
    """List of models usable in the current environment."""
    names = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "svm",
        "mlp",
        "knn",
        "naive_bayes",
        "stacking",
    ]
    if XGBOOST_AVAILABLE:
        names.append("xgboost")
    return names
