# data/raw/

Contains `heart_failure_clinical_records_dataset.csv` (Heart Failure Clinical
Records, UCI Machine Learning Repository #519, CC BY 4.0; 299 x 13). This file
is BUNDLED in the repo so that notebooks run offline on Google Colab / Kaggle
Notebooks / local — no API token required.

SHA-256: `9c73cea7468ff5d517801ec050fe9993da5912fce4b56f296f8df3b38dd75912`

The file has no identifier column; `src/data_loader.load_raw_data()` adds
`patient_id` (1..299, file order) at load time. The CSV itself is never modified.

If the file is removed, re-download it with:
```
python scripts/download_data.py
```
