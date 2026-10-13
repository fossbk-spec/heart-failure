# reports/

Results generated from source code (CSV/JSON/PNG/TXT). These are NOT committed —
rerun `make eda` / `make train` / `make train-recall80` to regenerate them.

- `figures/` : EDA overview, SHAP summary.
- `eda/`     : CSV tables from `src/eda.py` (class balance, LVEF / creatinine /
  sodium strata, follow-up leakage report, ...).
- `model_comparison.csv`, `cv_<model>.csv`, `split_summary.json`, `train_log.txt` :
  training results with the default F2 threshold policy.
- `recall80/` : the same outputs with `--min-recall 0.8`.
