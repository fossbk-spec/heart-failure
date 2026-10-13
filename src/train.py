"""End-to-end training script: load data -> split -> train -> evaluate.

Run:
    python -m src.train                          # run all default models
    python -m src.train --models logistic_regression random_forest
    python -m src.train --no-smote --class-weight balanced
    python -m src.train --synthetic               # run on synthetic data (CI/demo)
    python -m src.train --feature-set minimal_two # serum creatinine + EF only
    python -m src.train --feature-set compare     # paired CV: minimal_two vs all

Results are written to `reports/`:
    - model_comparison.csv : model comparison table on the TEST set
    - cv_<model>.csv       : per-fold cross-validation results
    - minimal_vs_full_feature_comparison.csv : paired feature-set comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import config, evaluation, models, preprocessing
from src.data_loader import (
    load_raw_data,
    make_synthetic_dataset,
    profile_dataset,
    split_features_target,
)

DEFAULT_MODELS = [
    "logistic_regression", "random_forest", "extra_trees", "gradient_boosting", "stacking",
]
COMPARISON_METRICS = ["recall", "specificity", "pr_auc", "roc_auc", "mcc", "brier", "f2"]


def build_pipeline(
    name: str,
    *,
    use_smote: bool = True,
    class_weight: str | None = None,
    feature_set: str = "all",
):
    """Leakage-safe pipeline (preprocessing + optional SMOTE + model) for a model name."""
    if name == "stacking":
        # Preprocessing + SMOTE live inside each base learner, so the
        # stacking's internal CV never sees resampled data.
        return models.build_stacking_ensemble(
            class_weight=class_weight,
            base_wrapper=lambda est, scale: preprocessing.make_full_pipeline(
                est, use_smote=use_smote, scale=scale, feature_set=feature_set
            ),
        )
    return preprocessing.make_full_pipeline(
        models.get_model(name, class_weight=class_weight),
        use_smote=use_smote,
        scale=models.needs_scaling(name),
        feature_set=feature_set,
    )


def run_experiment(
    df: pd.DataFrame,
    model_names: list[str],
    *,
    use_smote: bool = True,
    class_weight: str | None = None,
    beta: float = config.F_BETA,
    min_recall: float | None = None,
    run_cv: bool = True,
    feature_set: str = "all",
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Train + evaluate multiple models under the Zero Data Leakage protocol.

    Protocol:
        1. Patient-level train/val/test split (stratified by label, grouped by patient_id).
        2. Fit the pipeline (preprocessing + SMOTE + model) on train ONLY.
        3. Pick the decision threshold on VALIDATION.
        4. Report the final metrics ONCE on TEST.
    """
    split = preprocessing.patient_level_split(df)
    print("\n--- Data distribution after patient-level split ---")
    print(split.summary().to_string(index=False))

    X_train, y_train = split_features_target(split.train, feature_set=feature_set)
    X_val, y_val = split_features_target(split.val, feature_set=feature_set)
    X_test, y_test = split_features_target(split.test, feature_set=feature_set)

    results: list[evaluation.EvaluationResult] = []
    cv_tables: dict[str, pd.DataFrame] = {}

    for name in model_names:
        print(
            f"\n=== {name} (smote={use_smote}, class_weight={class_weight}, "
            f"features={feature_set}) ==="
        )
        pipeline = build_pipeline(
            name, use_smote=use_smote, class_weight=class_weight, feature_set=feature_set
        )

        pipeline.fit(X_train, y_train)

        # Verify: transforming the hold-out set does not alter the fitted state.
        fitted_pipelines = pipeline.estimators_ if name == "stacking" else [pipeline]
        for fitted in fitted_pipelines:
            preprocessing.assert_no_fit_on_holdout(fitted, X_test)

        # Pick the threshold on VALIDATION (never on TEST).
        val_proba = evaluation.predict_proba_positive(pipeline, X_val)
        if min_recall is not None:
            threshold = evaluation.threshold_for_min_recall(y_val, val_proba, min_recall=min_recall)
        else:
            threshold, val_fbeta = evaluation.find_best_threshold(y_val, val_proba, beta=beta)
            print(f"  Best threshold on validation: {threshold:.4f} (F{beta:g}={val_fbeta:.4f})")

        result = evaluation.evaluate_model(
            pipeline, X_test, y_test, model_name=name, threshold=threshold, beta=beta
        )
        print(result)
        results.append(result)

        if run_cv:
            train_val = pd.concat([split.train, split.val], ignore_index=True)
            X_tv, y_tv = split_features_target(train_val, feature_set=feature_set)
            cv_tables[name] = evaluation.cross_validate_grouped(
                pipeline,
                X_tv,
                y_tv,
                train_val[config.ID_COLUMN],
                beta=beta,
                min_recall=min_recall,
            )
            print("  CV (train+val):")
            print("   ", evaluation.summarize_cv(cv_tables[name]).to_dict())

    table = evaluation.comparison_table(results)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_dir / "model_comparison.csv", index=False)
        for name, cv in cv_tables.items():
            cv.to_csv(output_dir / f"cv_{name}.csv")
        (output_dir / "split_summary.json").write_text(
            json.dumps(split.summary().to_dict(orient="records"), indent=2), encoding="utf-8"
        )
        print(f"\nResults written to: {output_dir}")

    return table, cv_tables


def compare_feature_sets(
    df: pd.DataFrame,
    model_names: list[str],
    *,
    feature_sets: tuple[str, str] = ("all", "minimal_two"),
    use_smote: bool = True,
    class_weight: str | None = None,
    beta: float = config.F_BETA,
    min_recall: float | None = None,
    n_splits: int = config.N_SPLITS,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """Paired grouped-CV comparison of two feature sets (e.g. Chicco & Jurman 2020).

    Runs on train+val only (the test split stays untouched). Both feature
    sets use the SAME patient-grouped folds - the splitter is seeded and the
    rows are identical, only the columns differ - so per-fold differences
    (second - first) are paired. The 95% CI of the mean difference uses a t
    distribution with k-1 df; it is approximate because CV folds share
    training data and are not independent.

    Returns:
        (long-format summary table, {(model, feature_set): per-fold CV table})
    """
    split = preprocessing.patient_level_split(df, verify=False)
    train_val = pd.concat([split.train, split.val], ignore_index=True)
    groups = train_val[config.ID_COLUMN]
    base, other = feature_sets

    rows: list[dict[str, float | str]] = []
    folds: dict[tuple[str, str], pd.DataFrame] = {}
    for name in model_names:
        for feature_set in feature_sets:
            X, y = split_features_target(train_val, feature_set=feature_set)
            folds[(name, feature_set)] = evaluation.cross_validate_grouped(
                build_pipeline(
                    name, use_smote=use_smote, class_weight=class_weight, feature_set=feature_set
                ),
                X,
                y,
                groups,
                n_splits=n_splits,
                beta=beta,
                min_recall=min_recall,
            )
        for metric in COMPARISON_METRICS:
            a = folds[(name, base)][metric].to_numpy(dtype=float)
            b = folds[(name, other)][metric].to_numpy(dtype=float)
            diff = b - a
            half_width = (
                stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
                if len(diff) > 1
                else np.nan
            )
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    f"mean_{base}": a.mean(),
                    f"sd_{base}": a.std(ddof=1),
                    f"mean_{other}": b.mean(),
                    f"sd_{other}": b.std(ddof=1),
                    "mean_diff": diff.mean(),
                    "diff_ci95_low": diff.mean() - half_width,
                    "diff_ci95_high": diff.mean() + half_width,
                }
            )

    table = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_dir / "minimal_vs_full_feature_comparison.csv", index=False)
        print(f"\nResults written to: {output_dir}")
    return table, folds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train heart failure survival prediction models.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"List of models. Valid options: {models.available_models()}")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE (use cost-sensitive learning instead).")
    parser.add_argument("--class-weight", default=None, choices=[None, "balanced"],
                        help="Enable cost-sensitive learning.")
    parser.add_argument("--beta", type=float, default=config.F_BETA, help="Beta for the F-beta score.")
    parser.add_argument("--min-recall", type=float, default=None,
                        help="Set the threshold under a minimum Recall constraint (e.g. 0.8).")
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation.")
    parser.add_argument("--feature-set", default="all",
                        choices=[*config.FEATURE_SETS, "compare"],
                        help="Feature set, or 'compare' for a paired CV of minimal_two vs all.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (when the raw CSV isn't available).")
    parser.add_argument("--data", default=None, help="Optional custom CSV path.")
    parser.add_argument("--output", default=str(config.REPORTS_DIR), help="Output directory for results.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.synthetic:
        print("[WARNING] Using SYNTHETIC DATA - do not use for scientific reporting.")
        df = make_synthetic_dataset(n_rows=600)
    else:
        df = load_raw_data(args.data)

    print(f"\nDataset profile: {profile_dataset(df)}")

    if args.feature_set == "compare":
        table, _ = compare_feature_sets(
            df,
            args.models,
            use_smote=not args.no_smote,
            class_weight=args.class_weight,
            beta=args.beta,
            min_recall=args.min_recall,
            output_dir=Path(args.output),
        )
        print("\n========== PAIRED CV: minimal_two - all (95% CI of the difference) ==========")
        print(table.round(4).to_string(index=False))
        return 0

    table, _ = run_experiment(
        df,
        args.models,
        use_smote=not args.no_smote,
        class_weight=args.class_weight,
        beta=args.beta,
        min_recall=args.min_recall,
        run_cv=not args.no_cv,
        feature_set=args.feature_set,
        output_dir=Path(args.output),
    )

    print("\n================ MODEL COMPARISON ON TEST SET ================")
    cols = ["model", "threshold", "recall", "pr_auc", f"f{args.beta:g}", "mcc", "precision",
            f"f{args.beta:g}_flag_all", "roc_auc", "specificity", "accuracy", "null_accuracy",
            "n_false_negative"]
    print(table[[c for c in cols if c in table.columns]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
