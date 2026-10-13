"""External validation: develop on UCI, validate on PhysioNet Zigong (plan option A).

Protocol (docs/ACTION_PLAN_FOR_CLAUDE.md, Phase 4):

1. Harmonize both cohorts (`src.harmonization`) at a fixed death horizon.
2. Fit every model on the WHOLE harmonized UCI cohort only. Zigong rows never
   reach any `fit` (imputer, log1p, scaler and model are all learned on UCI).
3. No SMOTE: external validation judges predicted probabilities, and SMOTE
   shifts them upwards by construction.
4. Internal reference = grouped 5-fold CV on UCI (out-of-fold predictions);
   the decision threshold is the highest one keeping OOF Recall >= 0.80.
5. On Zigong report discrimination (ROC-AUC with stratified bootstrap CI,
   PR-AUC), calibration (calibration-in-the-large, slope, Brier) and the
   Brier score after an intercept-only update. The intercept update uses
   Zigong outcomes, so it shows what a local recalibration WOULD achieve;
   it is not an independent estimate.

Feature sets: models using ejection fraction can only be applied to Zigong
admissions with a measured LVEF (Cohort A, 32%); the `no_ef` model applies to
every admission (Cohort B). Missing LVEF is never imputed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src import config, evaluation, models, preprocessing
from src import harmonization as hz

EXTERNAL_FEATURE_SETS: dict[str, list[str]] = {
    "all9": hz.HARMONIZED_FEATURES,
    "no_ef": [c for c in hz.HARMONIZED_FEATURES if c != "ejection_fraction"],
    "minimal_two": ["ejection_fraction", "serum_creatinine"],
}
DEFAULT_EXTERNAL_MODELS = ["logistic_regression", "random_forest", "extra_trees"]


def split_columns(features: list[str]) -> tuple[list[str], list[str]]:
    """(numeric, binary) split of harmonized feature names, keeping the given order."""
    numeric = [c for c in features if c in hz.HARMONIZED_NUMERIC]
    binary = [c for c in features if c in hz.HARMONIZED_BINARY]
    unknown = set(features) - set(numeric) - set(binary)
    if unknown:
        raise ValueError(f"Not harmonized features: {sorted(unknown)}")
    return numeric, binary


def build_external_pipeline(model_name: str, features: list[str]):
    """Leakage-safe pipeline on harmonized columns (no SMOTE; see module docstring)."""
    return preprocessing.make_full_pipeline(
        models.get_model(model_name),
        use_smote=False,
        scale=models.needs_scaling(model_name),
        columns=split_columns(features),
    )


def applicable_rows(df: pd.DataFrame, features: list[str]) -> pd.Series:
    """Rows a model may be applied to: LVEF-based models need a measured LVEF."""
    if "ejection_fraction" in features:
        return df["ejection_fraction"].notna()
    return pd.Series(True, index=df.index)


# --------------------------------------------------------------------------- #
# Calibration & uncertainty
# --------------------------------------------------------------------------- #
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _fit_logistic(x: np.ndarray | None, offset: np.ndarray, y: np.ndarray, n_iter: int = 50):
    """Unpenalized logistic regression y ~ a (+ b * x) + offset, by Newton-Raphson."""
    design = np.ones((len(y), 1)) if x is None else np.column_stack([np.ones(len(y)), x])
    beta = np.zeros(design.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-(design @ beta + offset)))
        grad = design.T @ (y - p)
        hess = design.T @ (design * (p * (1 - p))[:, None])
        step = np.linalg.solve(hess + 1e-9 * np.eye(len(beta)), grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def calibration_intercept_slope(y_true, y_proba) -> tuple[float, float]:
    """Calibration-in-the-large (intercept with logit(p) as offset) and calibration slope.

    Perfect calibration gives intercept 0 and slope 1. Intercept < 0 means the
    model over-predicts risk; slope < 1 means predictions are too extreme.
    """
    y = np.asarray(y_true, dtype=float)
    lp = _logit(y_proba)
    intercept = float(_fit_logistic(None, lp, y)[0])
    slope = float(_fit_logistic(lp, np.zeros(len(y)), y)[1])
    return intercept, slope


def recalibrate_intercept(y_true, y_proba) -> np.ndarray:
    """Shift predictions on the logit scale so mean risk matches the observed rate."""
    intercept, _ = calibration_intercept_slope(y_true, y_proba)
    return 1 / (1 + np.exp(-(_logit(y_proba) + intercept)))


def bootstrap_auc_ci(
    y_true, y_proba, *, n_boot: int = 1000, alpha: float = 0.05,
    random_state: int = config.RANDOM_STATE,
) -> tuple[float, float]:
    """Percentile CI of ROC-AUC, resampling positives and negatives separately."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba, dtype=float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(random_state)
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aucs[i] = roc_auc_score(y[idx], p[idx])
    return float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2))


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
@dataclass
class ValidationResult:
    row: dict[str, float | str | int | bool]
    zigong_proba: np.ndarray
    zigong_y: np.ndarray


def internal_oof_proba(pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> np.ndarray:
    """Out-of-fold UCI predictions from grouped 5-fold CV."""
    from sklearn.base import clone

    oof = np.empty(len(y), dtype=float)
    for tr, va in preprocessing.cv_splitter().split(X, y, groups=groups):
        fold_model = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = evaluation.predict_proba_positive(fold_model, X.iloc[va])
    return oof


def validate_model(
    uci: pd.DataFrame,
    zigong: pd.DataFrame,
    *,
    model_name: str,
    feature_set: str,
    horizon_days: int,
    min_recall: float = 0.8,
    n_boot: int = 1000,
) -> ValidationResult:
    """Fit on harmonized UCI, evaluate on harmonized Zigong. Inputs are harmonized frames."""
    features = EXTERNAL_FEATURE_SETS[feature_set]
    label = hz.outcome_column(horizon_days)
    X_dev, y_dev = uci[features], uci[label].astype(int)

    pipeline = build_external_pipeline(model_name, features)
    oof = internal_oof_proba(pipeline, X_dev, y_dev, uci[hz.ID_COLUMN])
    threshold = evaluation.threshold_for_min_recall(y_dev, oof, min_recall=min_recall)
    pipeline.fit(X_dev, y_dev)

    ext = zigong[applicable_rows(zigong, features)]
    X_ext, y_ext = ext[features], ext[label].astype(int).to_numpy()
    preprocessing.assert_no_fit_on_holdout(pipeline, X_ext)
    proba = evaluation.predict_proba_positive(pipeline, X_ext)

    has_both = len(np.unique(y_ext)) == 2
    auc = roc_auc_score(y_ext, proba) if has_both else np.nan
    ci = bootstrap_auc_ci(y_ext, proba, n_boot=n_boot) if has_both else (np.nan, np.nan)
    citl, slope = calibration_intercept_slope(y_ext, proba) if has_both else (np.nan, np.nan)
    recal = recalibrate_intercept(y_ext, proba) if has_both else proba
    at_threshold = evaluation.compute_metrics(y_ext, proba, threshold=threshold)

    row = {
        "horizon_days": horizon_days,
        "feature_set": feature_set,
        "model": model_name,
        "uci_n": len(uci),
        "uci_deaths": int(y_dev.sum()),
        "internal_oof_auc": roc_auc_score(y_dev, oof),
        "zigong_n": len(ext),
        "zigong_deaths": int(y_ext.sum()),
        "zigong_rate": float(y_ext.mean()) if len(y_ext) else np.nan,
        "external_auc": auc,
        "auc_ci95_low": ci[0],
        "auc_ci95_high": ci[1],
        "external_pr_auc": average_precision_score(y_ext, proba) if has_both else np.nan,
        "mean_predicted_risk": float(proba.mean()) if len(proba) else np.nan,
        "calibration_in_the_large": citl,
        "calibration_slope": slope,
        "brier": brier_score_loss(y_ext, proba) if len(y_ext) else np.nan,
        "brier_after_intercept_update": brier_score_loss(y_ext, recal) if len(y_ext) else np.nan,
        "brier_null": float(y_ext.mean() * (1 - y_ext.mean())) if len(y_ext) else np.nan,
        "uci_threshold": threshold,
        "sensitivity_at_uci_threshold": at_threshold["recall"],
        "specificity_at_uci_threshold": at_threshold["specificity"],
    }
    return ValidationResult(row=row, zigong_proba=proba, zigong_y=y_ext)


def run_external_validation(
    uci_raw: pd.DataFrame,
    zigong_raw: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (90, 28, 180),
    uci_like_options: tuple[bool, ...] = (False, True),
    feature_sets: tuple[str, ...] = tuple(EXTERNAL_FEATURE_SETS),
    model_names: tuple[str, ...] = tuple(DEFAULT_EXTERNAL_MODELS),
    n_boot: int = 1000,
) -> pd.DataFrame:
    """Run the full grid; the first horizon / uci_like option is the primary analysis."""
    rows = []
    for horizon in horizons:
        uci = hz.harmonize_uci(uci_raw, horizon_days=horizon)
        for uci_like in uci_like_options:
            zigong = hz.harmonize_zigong(zigong_raw, horizon_days=horizon, uci_like=uci_like)
            for feature_set in feature_sets:
                for model_name in model_names:
                    result = validate_model(
                        uci, zigong, model_name=model_name, feature_set=feature_set,
                        horizon_days=horizon, n_boot=n_boot,
                    )
                    rows.append({"uci_like": uci_like, **result.row})
    table = pd.DataFrame(rows)
    table.insert(
        0, "primary",
        (table["horizon_days"] == horizons[0]) & (table["uci_like"] == uci_like_options[0]),
    )
    return table
