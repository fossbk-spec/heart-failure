# Heart Failure Survival Prediction using Machine Learning

Predicting death during follow-up of patients with heart failure from
baseline clinical data (Heart Failure Clinical Records, 299 patients,
Faisalabad 2015 — Ahmad et al. 2017; Chicco & Jurman 2020), under a strict
**Zero Data Leakage** discipline and prioritizing **Recall/PR-AUC** over
Accuracy for a clinical problem where missing a high-risk patient (FN) is far
more costly than a false alarm (FP). Death rate: 32.1% (96/299). Full
rationale: [`docs/DETAILED_EXAM_REPORT.md`](docs/DETAILED_EXAM_REPORT.md).

## Run on Google Colab / Kaggle Notebooks (offline — no Kaggle token needed)
The dataset is already bundled under `data/raw/`, so each notebook below
clones this repo and installs dependencies in its first cell — no manual
setup required.

| Notebook | Colab | Kaggle |
|---|---|---|
| 01 - Exploratory Data Analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fossbk-spec/heart-failure/blob/main/notebooks/01_exploratory_data_analysis.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/fossbk-spec/heart-failure/blob/main/notebooks/01_exploratory_data_analysis.ipynb) |
| 02 - Preprocessing & Imbalance | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fossbk-spec/heart-failure/blob/main/notebooks/02_preprocessing_and_imbalance.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/fossbk-spec/heart-failure/blob/main/notebooks/02_preprocessing_and_imbalance.ipynb) |
| 03 - Model Training & Evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fossbk-spec/heart-failure/blob/main/notebooks/03_model_training_and_evaluation.ipynb) | [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/fossbk-spec/heart-failure/blob/main/notebooks/03_model_training_and_evaluation.ipynb) |

## Quick start (local)
```bash
make setup           # create .venv + install requirements.txt
make verify          # check data/raw/ matches 299 records / 96 deaths / 0 missing / sha256
make test            # 138 unit tests (must pass, especially tests/test_data_leakage.py)
make eda             # export EDA tables + figures into reports/
make train           # train + compare 4 models, F2-tuned threshold -> reports/
make train-recall80  # same, threshold = highest one keeping validation Recall >= 0.80
make compare-features # paired CV: serum creatinine + EF only vs all 11 features
```
On Windows, replace `.venv/bin/python` with `.venv\Scripts\python` (or run
the commands in the Makefile by hand).

## Repository structure
```
heart-failure/
├── data/
│   ├── raw/                     # heart_failure_clinical_records_dataset.csv (bundled)
│   ├── processed/                # preprocessed data (generated, not committed)
│   └── SSoT_pdf_references/      # SSoT directory for paper PDFs (see its README)
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing_and_imbalance.ipynb
│   └── 03_model_training_and_evaluation.ipynb
├── src/
│   ├── config.py           # central config/schema constants (incl. leakage columns)
│   ├── data_loader.py      # load + patient_id + validate schema + synthetic data generator
│   ├── eda.py               # reusable EDA helpers (LVEF / creatinine / sodium strata, leakage report)
│   ├── preprocessing.py     # patient-level split, leakage-safe Pipeline
│   ├── models.py            # Logistic Regression, RF, Extra Trees, MLP ... Stacking
│   ├── evaluation.py        # Recall/PR-AUC/F2, threshold tuning, grouped CV
│   ├── explainability.py    # SHAP, permutation importance, odds ratios
│   └── train.py             # end-to-end training CLI
├── scripts/
│   ├── download_data.py     # fetch data from UCI (+ checksum verification)
│   └── run_eda.py
├── tests/                    # 138 unit tests, including the anti-leakage suite
├── docs/
│   ├── DETAILED_EXAM_REPORT.md   # detailed exam-solution report (Vietnamese)
│   └── literature/                # 7 paper summaries + dataset spec
└── .github/workflows/ci_cd_pipeline.yml
```

## Zero Data Leakage discipline (mandatory)
1. **Patient-level split**: the raw CSV has no identifier, so `data_loader`
   adds `patient_id` (1..299). Splits are stratified by `DEATH_EVENT` and
   grouped by `patient_id`; CV uses `StratifiedGroupKFold`.
2. **No post-outcome features**: `time` (follow-up days) is EXCLUDED from X.
   Death ends follow-up, so `time` encodes the label: adding it lifts grouped-CV
   ROC-AUC of a Random Forest from **0.79 → 0.92** without any real gain in
   prognostic skill (notebook 02). (In survival analysis — Cox models as in
   Ahmad et al. 2017 — `time` is the legitimate *outcome* time, never an input.)
3. **Fit-on-train-only**: imputer, `log1p`, `StandardScaler`, `SMOTE` are only
   `fit` on train; validation/test only call `.transform()`.
4. **SMOTE lives inside the Pipeline** (`imblearn.pipeline.Pipeline`), and
   inside each base learner of the stacking ensemble.
5. Every run can be asserted with `verify_zero_data_leakage()` and
   `assert_no_fit_on_holdout()` — see `tests/test_data_leakage.py`.

## Evaluation metrics
Priority order: **Recall (Sensitivity) > PR-AUC > F2-score > ROC-AUC > Specificity**.
Accuracy alone is never used: predicting "survives" for everyone already gives
67.9% accuracy with Recall = 0.

**Caveat specific to this dataset.** With 32% prevalence, the trivial rule
"flag every patient" already reaches F2 ≈ 0.70 (reported as `f2_flag_all`).
Maximizing F2 on the 48-patient validation set therefore drives the threshold
towards "flag almost everyone" (Recall 1.0, Specificity ≈ 0.1). The clinically
meaningful operating point is `--min-recall 0.8` (`make train-recall80`):
the highest threshold that still flags ≥ 80% of deaths on validation.

## Reference results — `make train-recall80`
Test set (60 patients, 19 deaths; threshold chosen on validation):

| Model | Recall | Specificity | PR-AUC | F2 | Precision | ROC-AUC |
|---|---|---|---|---|---|---|
| Stacking Ensemble | 0.74 | 0.66 | 0.63 | 0.67 | 0.50 | 0.76 |
| Gradient Boosting | 0.74 | 0.66 | 0.46 | 0.67 | 0.50 | 0.69 |
| Random Forest | 0.68 | 0.63 | 0.54 | 0.63 | 0.46 | 0.74 |
| Logistic Regression | 0.53 | 0.68 | 0.56 | 0.51 | 0.43 | 0.69 |

Grouped cross-validation on train+val (mean ± sample std across 5 folds):

| Model | Recall | Specificity | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.86 ± 0.07 | 0.49 ± 0.15 | 0.63 ± 0.03 | 0.77 ± 0.04 |
| Random Forest | 0.79 ± 0.15 | 0.68 ± 0.14 | 0.67 ± 0.04 | 0.78 ± 0.05 |
| Gradient Boosting | 0.83 ± 0.13 | 0.46 ± 0.14 | 0.63 ± 0.11 | 0.74 ± 0.06 |
| Stacking Ensemble | 0.78 ± 0.10 | 0.62 ± 0.14 | 0.63 ± 0.11 | 0.76 ± 0.08 |

The test set has only 19 deaths, so test differences of one or two patients
swing Recall by 0.05–0.10; CV is more reliable. With the default F2 policy
(`make train`), all models reach CV Recall 0.84–0.91 but Specificity only
0.27–0.45 — see `reports/train_log.txt`.

**Two-feature model (Chicco & Jurman 2020).** `make compare-features` runs a
paired grouped CV on train+val (same 5 folds for both arms, Recall ≥ 0.80
threshold policy). Serum creatinine + ejection fraction alone is not worse
than all 11 features; point estimates are slightly better, but most 95% CIs
of the difference include 0:

| Model | ROC-AUC (11 → 2 features) | PR-AUC (11 → 2) | Δ ROC-AUC [95% CI] |
|---|---|---|---|
| Logistic Regression | 0.77 → 0.77 | 0.63 → 0.65 | +0.01 [−0.04, 0.05] |
| Random Forest | 0.78 → 0.83 | 0.67 → 0.71 | +0.05 [−0.01, 0.11] |
| Extra Trees | 0.75 → 0.83 | 0.61 → 0.74 | +0.08 [−0.01, 0.16] |
| Gradient Boosting | 0.74 → 0.78 | 0.63 → 0.69 | +0.04 [0.02, 0.06] |

Extra Trees (the best model of Ishaq et al. 2021) performs on par with Random
Forest on all 11 features (CV ROC-AUC 0.75, PR-AUC 0.61); the MLP is weaker
(0.72 / 0.59).

**Explainability.** SHAP on Random Forest ranks `serum_creatinine` and
`ejection_fraction` first, matching Chicco & Jurman (2020). Logistic-regression
odds ratios (per 1 SD, after log1p for skewed labs): serum creatinine OR 3.7
(95% CI 2.1–6.6), ejection fraction OR 0.43 (0.28–0.67). CIs are indicative
only, since coefficients are learned on SMOTE-resampled data.

## License
Source code: MIT (see `LICENSE`). The dataset is distributed by the UCI
Machine Learning Repository under CC BY 4.0 (Chicco & Jurman 2020; original
data: Ahmad et al. 2017, PLoS ONE, CC BY). Paper PDFs belong to their
respective publishers (see `data/SSoT_pdf_references/README.md`).
