"""Pooled UCI + Zigong modelling: how to handle the missing LVEF.

LVEF (`ejection_fraction`) is the most informative predictor on UCI, but it is
missing in 68% of Zigong admissions (and never in UCI). Three strategies:

- `complete_case`  (ignore EXAMPLES): drop admissions without LVEF. Keeps the
  variable but only 256 UCI + 635 Zigong rows remain.
- `drop_lvef`      (ignore the FEATURE): keep all 2,264 rows, lose LVEF.
- `impute_*`       (imputation): keep all rows and LVEF, fill the gaps with a
  median, iterative (MICE-style) or KNN imputer, optionally with missing
  indicators. With 68% missing this mostly fabricates values, so we also
  measure how well each imputer recovers LVEF values that ARE observed.

Every scenario fits log1p, scaling and imputation inside the training folds
only. Evaluation is repeated stratified 5-fold CV on the pooled rows
(stratified by cohort x outcome), with metrics reported on the pooled data and
WITHIN each cohort: pooled ROC-AUC is inflated because the two cohorts differ
13-fold in mortality, so a model can score well just by telling them apart.

All outputs are aggregate metrics (Zigong is restricted-access).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import config, models
from src import external_validation as ev
from src import harmonization as hz

LVEF = "ejection_fraction"
SKEWED = [c for c in hz.HARMONIZED_NUMERIC if c in config.SKEWED_FEATURES]
SOURCE_FEATURE = "is_zigong"

# scenario -> (keep LVEF column, drop rows without LVEF, imputer, missing indicators)
SCENARIOS: dict[str, tuple[bool, bool, str, bool]] = {
    "complete_case": (True, True, "median", False),
    "drop_lvef": (False, False, "median", False),
    "impute_median": (True, False, "median", False),
    "impute_median_indicator": (True, False, "median", True),
    "impute_mice": (True, False, "mice", False),
    "impute_mice_indicator": (True, False, "mice", True),
    "impute_knn": (True, False, "knn", False),
}
DEFAULT_POOLED_MODELS = ["logistic_regression", "random_forest", "extra_trees"]


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _log1p(X):
    return np.log1p(np.clip(X, 0, None))


def make_imputer(kind: str, *, add_indicator: bool = False, random_state: int = config.RANDOM_STATE):
    """Imputer applied AFTER scaling (KNN distances need comparable scales)."""
    if kind == "median":
        return SimpleImputer(strategy="median", add_indicator=add_indicator)
    if kind == "mice":
        return IterativeImputer(max_iter=20, random_state=random_state,
                                add_indicator=add_indicator)
    if kind == "knn":
        return KNNImputer(n_neighbors=10, weights="distance", add_indicator=add_indicator)
    raise ValueError(f"Unknown imputer: {kind!r}")


def make_pooled_preprocessor(features: list[str], *, imputer: str = "median",
                             add_indicator: bool = False) -> Pipeline:
    """log1p (skewed labs, by NAME) -> StandardScaler -> imputer, all fitted on train only.

    Transforms are selected by column name before anything can drop a column,
    so an all-missing column cannot shift the log transform onto another one.
    Scaling ignores NaN in `fit`, so imputation sees scaled inputs.
    """
    skewed = [c for c in features if c in SKEWED]
    other = [c for c in features if c not in SKEWED]
    log_step = ColumnTransformer(
        [("log", FunctionTransformer(_log1p, feature_names_out="one-to-one"), skewed),
         ("rest", "passthrough", other)],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("log", log_step),
        ("scale", StandardScaler()),
        ("impute", make_imputer(imputer, add_indicator=add_indicator)),
    ])


def build_pooled_pipeline(model_name: str, features: list[str], scenario: str) -> Pipeline:
    _, _, imputer, indicator = SCENARIOS[scenario]
    return Pipeline([
        ("pre", make_pooled_preprocessor(features, imputer=imputer, add_indicator=indicator)),
        ("model", models.get_model(model_name)),
    ])


def scenario_data(pooled: pd.DataFrame, scenario: str, *, source_feature: bool = False):
    """Rows and feature list a scenario uses."""
    keep_lvef, drop_rows, _, _ = SCENARIOS[scenario]
    features = [c for c in hz.HARMONIZED_FEATURES if keep_lvef or c != LVEF]
    data = pooled[pooled[LVEF].notna()] if drop_rows else pooled
    data = data.copy()
    if source_feature:
        data[SOURCE_FEATURE] = (data[hz.SOURCE_COLUMN] == "zigong").astype(float)
        features = features + [SOURCE_FEATURE]
    return data, features


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _strata(df: pd.DataFrame, label: str) -> np.ndarray:
    return (df[hz.SOURCE_COLUMN].astype(str) + "_" + df[label].astype(int).astype(str)).to_numpy()


def oof_predictions(pipeline, X: pd.DataFrame, y: pd.Series, strata: np.ndarray, *,
                    n_splits: int = 5, seed: int = config.RANDOM_STATE) -> np.ndarray:
    """Out-of-fold probabilities from one stratified K-fold run (every fit on train folds only)."""
    oof = np.empty(len(y), dtype=float)
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, strata):
        fitted = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = fitted.predict_proba(X.iloc[va])[:, 1]
    return oof


def _subset_metrics(prefix: str, y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {f"{prefix}_n": len(y), f"{prefix}_deaths": int(np.sum(y)),
                f"{prefix}_auc": np.nan, f"{prefix}_pr_auc": np.nan}
    return {
        f"{prefix}_n": len(y),
        f"{prefix}_deaths": int(y.sum()),
        f"{prefix}_auc": roc_auc_score(y, p),
        f"{prefix}_pr_auc": average_precision_score(y, p),
    }


def evaluate_scenario(pooled: pd.DataFrame, *, scenario: str, model_name: str,
                      horizon_days: int = 90, n_repeats: int = 3, n_splits: int = 5,
                      source_feature: bool = False) -> dict[str, float | str | bool]:
    """Repeated CV of one scenario; metrics averaged over repeats."""
    label = hz.outcome_column(horizon_days)
    data, features = scenario_data(pooled, scenario, source_feature=source_feature)
    data = data.reset_index(drop=True)
    X, y = data[features], data[label].astype(int)
    strata = _strata(data, label)
    is_zg = (data[hz.SOURCE_COLUMN] == "zigong").to_numpy()
    has_lvef = data[LVEF].notna().to_numpy()
    pipeline = build_pooled_pipeline(model_name, features, scenario)

    rows = []
    for rep in range(n_repeats):
        p = oof_predictions(pipeline, X, y, strata, n_splits=n_splits,
                            seed=config.RANDOM_STATE + rep)
        yv = y.to_numpy()
        m: dict[str, float] = {
            "pooled_auc": roc_auc_score(yv, p),
            "pooled_pr_auc": average_precision_score(yv, p),
            "pooled_brier": brier_score_loss(yv, p),
        }
        m |= _subset_metrics("uci", yv[~is_zg], p[~is_zg])
        m |= _subset_metrics("zigong", yv[is_zg], p[is_zg])
        lv = is_zg & has_lvef
        m |= _subset_metrics("zigong_lvef", yv[lv], p[lv])
        common = ~is_zg | has_lvef  # UCI + Zigong with LVEF: rows every scenario can score
        m |= _subset_metrics("common", yv[common], p[common])
        zy, zp = yv[is_zg], p[is_zg]
        m["zigong_mean_pred"] = float(zp.mean())
        m["zigong_rate"] = float(zy.mean())
        m["zigong_citl"] = ev.calibration_intercept_slope(zy, zp)[0] if zy.sum() else np.nan
        rows.append(m)

    avg = pd.DataFrame(rows).mean().to_dict()
    sd = pd.DataFrame(rows)[["zigong_auc", "zigong_lvef_auc", "pooled_auc"]].std(ddof=1)
    return {
        "scenario": scenario,
        "model": model_name,
        "source_feature": source_feature,
        "n_rows": len(data),
        "n_features": len(features),
        **avg,
        "zigong_auc_sd_repeats": sd["zigong_auc"],
        "zigong_lvef_auc_sd_repeats": sd["zigong_lvef_auc"],
        "pooled_auc_sd_repeats": sd["pooled_auc"],
    }


def run_pooled_scenarios(pooled: pd.DataFrame, *, scenarios=tuple(SCENARIOS),
                         model_names=tuple(DEFAULT_POOLED_MODELS),
                         source_options=(False, True), n_repeats: int = 3,
                         horizon_days: int = 90) -> pd.DataFrame:
    rows = [
        evaluate_scenario(pooled, scenario=s, model_name=m, source_feature=src,
                          n_repeats=n_repeats, horizon_days=horizon_days)
        for src in source_options for s in scenarios for m in model_names
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# How well can LVEF be imputed?
# --------------------------------------------------------------------------- #
def imputation_accuracy(pooled: pd.DataFrame, *, imputers=("median", "mice", "knn"),
                        n_splits: int = 5, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Mask OBSERVED Zigong LVEF values in held-out folds and compare imputations to the truth.

    The imputer is fitted on the training folds (UCI + Zigong, LVEF as observed);
    held-out Zigong rows have their LVEF hidden before transform. This measures
    the best case: values missing in practice are probably missing NOT at random
    (no echocardiogram), so real errors can be larger.
    """
    features = list(hz.HARMONIZED_FEATURES)
    zg_obs = ((pooled[hz.SOURCE_COLUMN] == "zigong") & pooled[LVEF].notna()).to_numpy()
    idx_obs = np.flatnonzero(zg_obs)
    order = [c for c in features if c in SKEWED] + [c for c in features if c not in SKEWED]
    j = order.index(LVEF)  # the log step puts skewed columns first, then the rest
    rows = []
    for kind in imputers:
        errors, imputed_all, truth_all = [], [], []
        for _, va_obs in KFold(n_splits, shuffle=True, random_state=seed).split(idx_obs):
            held = idx_obs[va_obs]
            train = pooled.drop(index=pooled.index[held])
            pre = make_pooled_preprocessor(features, imputer=kind).fit(train[features])
            X_held = pooled.iloc[held][features].copy()
            truth = X_held[LVEF].to_numpy()
            X_held[LVEF] = np.nan
            scaled = pre.transform(X_held)
            scaler = pre.named_steps["scale"]
            imputed = scaled[:, j] * scaler.scale_[j] + scaler.mean_[j]
            errors.append(imputed - truth)
            imputed_all.append(imputed)
            truth_all.append(truth)
        err = np.concatenate(errors)
        imp, tru = np.concatenate(imputed_all), np.concatenate(truth_all)
        rows.append({
            "imputer": kind,
            "n_masked": len(err),
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "bias": float(err.mean()),
            "corr_with_truth": float(np.corrcoef(imp, tru)[0, 1]) if imp.std() > 0 else np.nan,
            "sd_imputed": float(imp.std()),
            "sd_true": float(tru.std()),
        })
    return pd.DataFrame(rows)


def missingness_summary(pooled: pd.DataFrame, horizon_days: int = 90) -> pd.DataFrame:
    """Aggregate: Zigong mortality and profile with vs without a measured LVEF."""
    label = hz.outcome_column(horizon_days)
    zg = pooled[pooled[hz.SOURCE_COLUMN] == "zigong"]
    out = []
    for measured, part in zg.groupby(zg[LVEF].notna()):
        out.append({
            "lvef_measured": bool(measured),
            "n": len(part),
            "deaths": int(part[label].sum()),
            "death_rate": float(part[label].mean()),
            "age_median": float(part["age"].median()),
            "creatinine_median": float(part["serum_creatinine"].median()),
            "sodium_median": float(part["serum_sodium"].median()),
        })
    return pd.DataFrame(out)
