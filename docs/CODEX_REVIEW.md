# Code review by Codex (2026-10-14)

Reviewer: Codex CLI 0.162.0-alpha.2, read-only, on a clean checkout of commit `fb10cee` without restricted Zigong data. Codex could not run the tests.

The default UCI and external-validation paths largely follow the intended leakage controls: feature lists exclude `time`, preprocessing and SMOTE are fitted within training pipelines, and thresholds use validation or OOF predictions. I found **no Critical issue**, but several **Major issues** weaken the statistical conclusions or the broader “Zero Data Leakage” claim: stacking’s internal folds ignore patient groups, feature-comparison CIs treat dependent folds as independent, and calibration fitting can silently diverge. The harmonization arithmetic and fixed-horizon label rules look correct under the documented input assumptions. This review used only the supplied text; **I could not run tests**, verify reported results, or inspect restricted Zigong data. No files were modified.

## Major findings

### 1. Stacking’s internal CV ignores patient groups

**Location:** `src/models.py:165–168`; `src/train.py:49–57`

- **What is wrong:** `StackingClassifier(cv=5)` generates meta-training predictions using ordinary stratified folds, without patient groups. Grouped outer CV and patient-level holdouts do not prevent leakage inside this additional split.
- **Failure scenario:** With repeated records for one patient, one record can train a base learner while another supplies its supposedly OOF prediction to the meta-learner. The current one-row-per-patient UCI dataset avoids this, but the advertised repeated-record protection does not.
- **Suggested fix:** Pass explicitly constructed patient-grouped internal splits to stacking for each fit, or implement grouped OOF meta-feature generation. Until supported, reject repeated patient IDs for stacking.
- **Coverage gap:** `tests/test_data_leakage.py:230–244` checks SMOTE placement using single-record patients; it never checks stacking’s internal patient separation.

### 2. Feature-comparison CIs underestimate dependence; “not worse” is unsupported

**Location:** `src/train.py:204–220`; `README.md:118–129`; `docs/PROGRESS_REPORT.md:226`

- **What is wrong:** The CI uses `SD(differences)/sqrt(k)` as though the five paired fold differences were independent. Their training sets overlap substantially. The docstring acknowledges dependence, but the output and reports still present conventional 95% CIs and draw inferential conclusions.
- **Failure scenario:** A narrow interval such as Gradient Boosting’s reported `[0.02, 0.06]` can suggest convincing improvement without properly accounting for training-set dependence. Intervals containing zero also do **not** establish noninferiority.
- **Suggested fix:** Report these fold differences descriptively, or use a justified repeated-CV corrected comparison or patient-level resampling that repeats the fitting procedure. Replace “not worse” with “higher/similar point estimates”; a noninferiority claim requires a prespecified margin and appropriate analysis.
- **Coverage gap:** `tests/test_feature_sets.py:75–92` checks zero differences and interval ordering, not the validity of uncertainty estimates.

### 3. Calibration Newton iterations can silently diverge

**Location:** `src/external_validation.py:77–89`, `105–108`

- **What is wrong:** Undamped Newton steps have no line search, convergence status, finite-value checks, or separation handling. Adding `1e-9` to the Hessian does not ensure convergence. The final iterate is returned even after failure.
- **Failure scenario:** Suppose predictions are uniformly `0.99` but observed mortality is `0.02`. The first intercept step is approximately `−98`; the next Hessian is nearly zero and the next step can be enormous. Subsequent probabilities saturate or overflow, producing a meaningless CITL and intercept update. Perfect separation also has no finite unpenalized slope.
- **Suggested fix:** Solve the intercept-only score equation with a bracketed scalar root finder. Fit slope/intercept using a stable optimizer with convergence and separation checks, returning an explicit failure status where appropriate. Intercept recalibration should not unnecessarily fit the slope.
- **Coverage gap:** `tests/test_external_validation.py:24–47` exercises moderate probabilities only; add extreme overprediction, constant predictions, and separation cases.

## Minor findings

### 4. Empty external cohorts crash before the intended NaN reporting

**Location:** `src/external_validation.py:168–178`

- **What is wrong:** Prediction and the preprocessing audit run before checking whether any applicable external rows remain. Later empty-cohort guards therefore cannot protect this path.
- **Failure scenario:** Every Zigong LVEF is missing, or `uci_like` removes all admissions. LVEF-based validation passes zero rows to preprocessing and fails instead of reporting an unavailable analysis.
- **Suggested fix:** Handle an empty applicable cohort before transforming or predicting, returning counts of zero and unavailable metrics.
- **Coverage gap:** `tests/test_external_validation.py:72–90` covers partial missingness, not an entirely empty applicable cohort.

### 5. Entirely missing numeric columns can corrupt positional log transforms

**Location:** `src/preprocessing.py:216–232`

- **What is wrong:** Imputers can drop entirely missing columns, while `skewed_idx` remains based on the original column positions.
- **Failure scenario:** In the six-column numeric block, an entirely missing EF column is dropped. The original platelet and creatinine positions then refer to different columns, so the code can log-transform creatinine and sodium while leaving platelets untransformed. Other missing-column arrangements can cause an index error.
- **Suggested fix:** Preserve empty features explicitly and define their fill policy, or apply named feature transformations before any operation that can change column positions.
- **Coverage gap:** `tests/test_preprocessing.py:15–34` covers scattered missing values only.

### 6. External validation accepts stacking with a different, unsafe internal preprocessing arrangement

**Location:** `src/external_validation.py:52–59`; `src/models.py:111`, `138–168`; `scripts/run_external_validation.py:41–42`

- **What is wrong:** The external CLI accepts arbitrary model names. Selecting `stacking` uses `get_model("stacking")` without base wrappers, then places preprocessing outside stacking. Consequently, preprocessing learns from the full UCI fitting set before stacking creates its internal OOF meta-features.
- **Failure scenario:** `--models stacking` creates internal meta-training predictions whose preprocessing has seen the internal validation rows. The default LR/RF/Extra Trees analyses are unaffected.
- **Suggested fix:** Restrict external validation to supported models, or construct stacking with preprocessing inside every base learner and resolve its grouped internal CV.
- **Coverage gap:** External tests exercise Logistic Regression only.

### 7. The external fit spy does not establish cohort provenance

**Location:** `tests/test_external_validation.py:104–120`

- **What is wrong:** The test identifies UCI rows through integer DataFrame indices. Both harmonized cohorts reset their indices, so those indices overlap.
- **Failure scenario:** A regression fitting on `zigong.iloc[:len(uci)]` would satisfy both existing assertions: its indices are a subset of UCI indices and its length is no greater than UCI’s.
- **Suggested fix:** Give the test cohorts disjoint index namespaces or retain source-prefixed IDs in a test-visible index, then assert provenance for every fit. Also spy on threshold selection to establish that its labels and predictions come exclusively from UCI.
- **Qualification:** The supplied production code does fit predictive pipelines exclusively on UCI. This is a weakness in the claimed safeguard, not evidence of current cross-cohort fitting.

### 8. Nondefault beta breaks feature-comparison mode

**Location:** `src/train.py:38`, `201–203`, `239`

- **What is wrong:** `compare_feature_sets()` accepts `beta`, but its metric list always requires `"f2"`. `compute_metrics()` instead creates the key corresponding to the requested beta.
- **Failure scenario:** `--feature-set compare --beta 1` returns `"f1"` metrics and then raises `KeyError: 'f2'`.
- **Suggested fix:** Construct the comparison metric list using `f"f{beta:g}"`, or reject nondefault beta in this mode.
- **Coverage gap:** Comparison tests use the default beta exclusively.

### 9. Wald odds-ratio CIs omit uncertainty from the fitted intercept

**Location:** `src/explainability.py:139–147`; `README.md:178–180`

- **What is wrong:** The Fisher information matrix contains feature columns only, although the Logistic Regression model also estimates an intercept. Inverting this matrix treats that intercept as known. SMOTE and penalized estimation add further inferential limitations beyond this omission.
- **Failure scenario:** With binary predictors having nonzero means, ignoring intercept–coefficient covariance can understate coefficient uncertainty and make reported OR intervals appear more precise.
- **Suggested fix:** Include the intercept in the design and extract the coefficient covariance block. For inferential claims, use an appropriate unpenalized original-data model or resample the complete fitting procedure; strengthen the existing caveat.
- **Coverage gap:** `tests/test_explainability.py:38–43` only checks that intervals bracket their estimates.

### 10. Leakage audits check state changes, not previous fitting provenance

**Location:** `src/preprocessing.py:338–354`; `README.md:79–80`; `docs/PROGRESS_REPORT.md:79–82`

- **What is wrong:** `assert_no_fit_on_holdout()` detects changes during its own transform call. It cannot detect a preprocessor previously fitted on train plus holdout. `verify_zero_data_leakage()` checks ID overlap only.
- **Failure scenario:** Fit a pipeline on concatenated train and test, then call both audits using disjoint split frames. Both can pass.
- **Suggested fix:** Describe these checks narrowly and use fit-input spies or recorded fit provenance to substantiate train-only fitting. Avoid presenting them as comprehensive leakage certification.

## Verified as correct

- **Patient-level holdouts and ordinary nested CV:** IDs remain grouped in `src/preprocessing.py:87–95`; outer and threshold-tuning inner folds use grouped splitters in `src/evaluation.py:224–240`.
- **Feature exclusion:** Normal and harmonized feature lists exclude `time`, target, ID, and cohort source. `time` is used appropriately to construct the fixed-horizon outcome.
- **Preprocessing and SMOTE placement:** Default training fits pipelines on training data; SMOTE stays inside those pipelines and is skipped during prediction.
- **Threshold provenance:** Holdout experiments tune on validation; nested CV tunes on inner OOF predictions; external validation selects its threshold from UCI OOF predictions.
- **External default models:** Predictive preprocessing and model fitting use UCI only, without SMOTE. Zigong intercept updating intentionally uses external outcomes and is explicitly described as an apparent, nonindependent result.
- **Metrics:** MCC, F2, flag-all F2, specificity, Brier, and average precision are implemented correctly for valid, nonempty binary inputs. “PR-AUC” is explicitly identified as average precision.
- **Calibration definitions and bootstrap design:** CITL uses a fixed logit offset; slope is fitted with a free intercept. Bootstrap sampling preserves positive and negative counts and uses a seeded generator, assuming independent admissions.
- **Harmonization:** Creatinine division by `88.4`, UCI platelet division by `1000`, sex-specific strict Hb thresholds, and right-closed age mapping match the documented assumptions. The `uci_like` filter selects the intended age bands and NYHA III–IV, subject to the resolution of banded ages.
- **Censoring rules:** Death at or before the horizon is positive; death later or survival through the horizon is negative; alive censoring before the horizon remains unknown and is dropped.
- **Reproducibility and CI data:** Default random generators, model seeds, SMOTE, and grouped splitters are seeded. External CI commands explicitly use synthetic data.