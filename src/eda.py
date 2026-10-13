"""Reusable EDA helpers (requirement: must run end-to-end without errors).

Every function: (1) takes a DataFrame, (2) returns a DataFrame/Figure,
(3) never mutates the input data (no side effects), (4) works even without a
display backend (uses Agg for plotting).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config

TARGET = config.TARGET_COLUMN


# --------------------------------------------------------------------------- #
# 1. Descriptive tables
# --------------------------------------------------------------------------- #
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report missing values plus the most frequent value's share per column.

    The raw file has no NaN, but a single value repeated far more often than
    a continuous lab test allows (e.g. `platelets == 263358.03`, which is the
    column mean) suggests values that were imputed BEFORE publication. Such
    hidden imputation is reported via `top_value_share`.
    """
    rows = []
    for col in df.columns:
        series = df[col]
        n = len(df)
        n_nan = int(series.isna().sum())
        counts = series.value_counts(dropna=True)
        rows.append(
            {
                "feature": col,
                "dtype": str(series.dtype),
                "n_missing": n_nan,
                "pct_missing": n_nan / n if n else 0.0,
                "n_unique": int(series.nunique(dropna=True)),
                "top_value": counts.index[0] if len(counts) else np.nan,
                "top_value_share": counts.iloc[0] / n if len(counts) and n else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False).reset_index(drop=True)


def suspected_imputation_report(df: pd.DataFrame) -> pd.DataFrame:
    """Continuous columns whose single most frequent value looks like a fill-in.

    Flags a value as suspicious when it is non-integer-looking yet repeated
    (>= 5 records), or when it equals the column mean/median.
    """
    rows = []
    for col in config.NUMERIC_FEATURES:
        series = df[col].dropna()
        if series.empty:
            continue
        top_value, top_count = series.value_counts().index[0], int(series.value_counts().iloc[0])
        rows.append(
            {
                "feature": col,
                "top_value": top_value,
                "top_count": top_count,
                "equals_mean": bool(np.isclose(top_value, series.mean(), rtol=1e-4)),
                "equals_median": bool(np.isclose(top_value, series.median(), rtol=1e-6)),
                "non_integer_repeated": bool(top_value != round(top_value) and top_count >= 5),
            }
        )
    return pd.DataFrame(rows)


def class_balance_report(df: pd.DataFrame) -> pd.DataFrame:
    """Class imbalance table at BOTH levels: sample (record) and patient.

    Sample-level imbalance is the label skew across records; patient-level
    imbalance is the label skew across unique patients. This dataset has one
    record per patient so the two coincide — but the check would immediately
    flag a difference on real EHR data with repeated visits per patient.
    """
    n = len(df)
    pos = int((df[TARGET] == 1).sum())

    patient_level = df.groupby(config.ID_COLUMN)[TARGET].max()
    n_pat = len(patient_level)
    pos_pat = int((patient_level == 1).sum())

    return pd.DataFrame(
        [
            {
                "level": "sample-level",
                "n_total": n,
                "n_positive": pos,
                "n_negative": n - pos,
                "positive_rate": pos / n if n else 0.0,
                "imbalance_ratio": (n - pos) / pos if pos else np.inf,
            },
            {
                "level": "patient-level",
                "n_total": n_pat,
                "n_positive": pos_pat,
                "n_negative": n_pat - pos_pat,
                "positive_rate": pos_pat / n_pat if n_pat else 0.0,
                "imbalance_ratio": (n_pat - pos_pat) / pos_pat if pos_pat else np.inf,
            },
        ]
    )


def numeric_summary(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Descriptive statistics for continuous features, split by the `DEATH_EVENT` label."""
    features = features or config.NUMERIC_FEATURES
    frames = []
    for col in features:
        grouped = df.groupby(TARGET)[col]
        summary = grouped.agg(["count", "mean", "std", "median", "min", "max"]).round(3)
        summary.insert(0, "feature", col)
        frames.append(summary.reset_index())
    return pd.concat(frames, ignore_index=True)


def death_rate_by_category(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Death rate broken down by each category of a categorical/binary feature."""
    grouped = df.groupby(column, dropna=False)[TARGET]
    out = grouped.agg(n_samples="count", n_death="sum")
    out["death_rate"] = out["n_death"] / out["n_samples"]
    return out.sort_values("death_rate", ascending=False).reset_index()


def _death_rate_by_bins(
    df: pd.DataFrame, column: str, bins: list[float], labels: list[str], name: str
) -> pd.DataFrame:
    tmp = df[[column, TARGET]].copy()
    tmp[name] = pd.cut(tmp[column], bins=bins, labels=labels, right=False)
    grouped = tmp.groupby(name, observed=True)[TARGET]
    out = grouped.agg(n_samples="count", n_death="sum")
    out["death_rate"] = out["n_death"] / out["n_samples"]
    return out.reset_index()


def death_rate_by_age_band(
    df: pd.DataFrame, bins: tuple[float, ...] = (40, 50, 60, 70, 80, 120)
) -> pd.DataFrame:
    """Death rate by age band (the cohort is >= 40 years old by inclusion criteria)."""
    labels = [f"[{int(lo)}, {int(hi)})" for lo, hi in zip(bins[:-1], bins[1:], strict=True)]
    return _death_rate_by_bins(df, "age", list(bins), labels, "age_band")


def ejection_fraction_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Death rate by heart-failure phenotype from LVEF (ESC 2021 classification).

    <=40%: HFrEF (reduced) | 41-49%: HFmrEF (mildly reduced) | >=50%: HFpEF (preserved).
    EF values in this dataset are integers, so bins at 41/50 are exact.
    """
    out = _death_rate_by_bins(
        df,
        "ejection_fraction",
        [0, 41, 50, np.inf],
        ["<=40 (HFrEF)", "41-49 (HFmrEF)", ">=50 (HFpEF)"],
        "ef_stratum",
    )
    return _add_risk_ratio(out, reference="<=40 (HFrEF)", label_col="ef_stratum")


def serum_creatinine_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Death rate by serum creatinine level (mg/dL) - a renal-dysfunction proxy.

    <=1.2: within normal range | 1.2-1.5: mildly elevated | 1.5-2.0: elevated
    | >=2.0: markedly elevated (cardiorenal syndrome territory).
    """
    out = _death_rate_by_bins(
        df,
        "serum_creatinine",
        [0, 1.2001, 1.5, 2.0, np.inf],
        ["<=1.2 (normal)", "1.2-1.5 (mild)", "1.5-2.0 (elevated)", ">=2.0 (marked)"],
        "creatinine_stratum",
    )
    return _add_risk_ratio(out, reference="<=1.2 (normal)", label_col="creatinine_stratum")


def serum_sodium_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Death rate for hyponatremia (<135 mEq/L) vs. normal sodium."""
    out = _death_rate_by_bins(
        df,
        "serum_sodium",
        [0, 135, np.inf],
        ["<135 (hyponatremia)", ">=135 (normal)"],
        "sodium_stratum",
    )
    return _add_risk_ratio(out, reference=">=135 (normal)", label_col="sodium_stratum")


def _add_risk_ratio(out: pd.DataFrame, *, reference: str, label_col: str) -> pd.DataFrame:
    ref = out.loc[out[label_col].astype(str) == reference, "death_rate"]
    baseline = float(ref.iloc[0]) if len(ref) else np.nan
    out["risk_ratio_vs_reference"] = out["death_rate"] / baseline if baseline else np.nan
    return out


def correlation_with_target(df: pd.DataFrame) -> pd.DataFrame:
    """Point-biserial correlation between each baseline feature and `DEATH_EVENT`."""
    cols = config.FEATURE_COLUMNS
    corr = (
        df[cols + [TARGET]]
        .astype(float)
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=np.abs, ascending=False)
    )
    return corr.rename("corr_with_death").rename_axis("feature").reset_index()


def follow_up_leakage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Show why the follow-up `time` column is a LEAKAGE feature.

    Death ends follow-up, so deceased patients have systematically short
    `time`. Its univariate ROC-AUC (computed as 1 - AUC because the relation
    is inverse) is reported next to the best legitimate baseline feature, to
    make the inflation it would cause explicit.
    """
    from sklearn.metrics import roc_auc_score

    y = df[TARGET].astype(int)
    rows = []
    for col in config.LEAKAGE_COLUMNS + config.NUMERIC_FEATURES:
        x = df[col]
        mask = x.notna()
        if y[mask].nunique() < 2:
            continue
        auc = roc_auc_score(y[mask], x[mask])
        rows.append(
            {
                "feature": col,
                "is_leakage": col in config.LEAKAGE_COLUMNS,
                "median_survived": float(x[y == 0].median()),
                "median_died": float(x[y == 1].median()),
                "univariate_auc": max(auc, 1 - auc),
            }
        )
    return pd.DataFrame(rows).sort_values("univariate_auc", ascending=False).reset_index(drop=True)


def demographic_bias_report(df: pd.DataFrame) -> pd.DataFrame:
    """Check representation and outcome rate across sex / smoking / comorbidity axes.

    Purpose: detect under-represented groups — the model will perform worse
    on these groups once deployed in practice (e.g. women are ~35% of this
    cohort).
    """
    frames = []
    for col in config.BINARY_FEATURES:
        part = death_rate_by_category(df, col).rename(columns={col: "category"})
        part.insert(0, "attribute", col)
        part["share_of_cohort"] = part["n_samples"] / len(df)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def run_full_eda(df: pd.DataFrame, output_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Run the full EDA end-to-end, returning a dict of result tables.

    Args:
        output_dir: if given, save each table as a CSV for inclusion in the report.
    """
    tables: dict[str, pd.DataFrame] = {
        "missing_values": missing_value_report(df),
        "suspected_imputation": suspected_imputation_report(df),
        "class_balance": class_balance_report(df),
        "numeric_summary": numeric_summary(df),
        "age_bands": death_rate_by_age_band(df),
        "ejection_fraction_strata": ejection_fraction_strata(df),
        "serum_creatinine_strata": serum_creatinine_strata(df),
        "serum_sodium_strata": serum_sodium_strata(df),
        "correlation": correlation_with_target(df),
        "follow_up_leakage": follow_up_leakage_report(df),
        "demographic_bias": demographic_bias_report(df),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(out / f"eda_{name}.csv", index=False)
    return tables


# --------------------------------------------------------------------------- #
# 2. Plots
# --------------------------------------------------------------------------- #
def plot_eda_overview(df: pd.DataFrame, save_path: str | Path | None = None):  # pragma: no cover
    """Plot a 2x3 overview: class balance, EF, creatinine, age, sodium, follow-up leakage."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    survived, died = "#4C78A8", "#E45756"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    counts = df[TARGET].value_counts().sort_index()
    axes[0, 0].bar(["Survived (0)", "Died (1)"], counts.values, color=[survived, died])
    for i, v in enumerate(counts.values):
        axes[0, 0].text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom", fontsize=9)
    axes[0, 0].set_title("Class balance (DEATH_EVENT)")
    axes[0, 0].set_ylim(0, counts.max() * 1.25)

    def bar(ax, table, label_col, title, color):
        ax.bar(table[label_col].astype(str), table["death_rate"], color=color)
        ax.set_title(title)
        ax.set_ylabel("death rate")
        ax.tick_params(axis="x", rotation=25, labelsize=8)

    bar(axes[0, 1], ejection_fraction_strata(df), "ef_stratum", "Death rate by LVEF phenotype", died)
    bar(axes[0, 2], serum_creatinine_strata(df), "creatinine_stratum",
        "Death rate by serum creatinine (mg/dL)", "#F58518")
    bar(axes[1, 0], death_rate_by_age_band(df), "age_band", "Death rate by age band", died)
    bar(axes[1, 1], serum_sodium_strata(df), "sodium_stratum", "Death rate by serum sodium",
        "#72B7B2")

    for label, color in ((0, survived), (1, died)):
        subset = df.loc[df[TARGET] == label, "time"].dropna()
        axes[1, 2].hist(subset, bins=25, alpha=0.6, color=color, density=True,
                        label="died" if label else "survived")
    axes[1, 2].set_title("Follow-up `time` by outcome (LEAKAGE - excluded)")
    axes[1, 2].set_xlabel("days")
    axes[1, 2].legend(fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
