.PHONY: help setup data verify eda train train-recall80 compare-features download-zigong inspect-zigong external-validation train-fast test lint clean

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create a virtual environment and install dependencies
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

data:  ## Download the UCI dataset into data/raw/ (already bundled)
	.venv/bin/python scripts/download_data.py

verify:  ## Verify the raw data matches expectations (299 rows / 96 deaths / 0 missing / sha256)
	.venv/bin/python scripts/download_data.py --verify-only

eda:  ## Run EDA and export tables + figures into reports/
	.venv/bin/python scripts/run_eda.py

train:  ## Full training run (4 models + cross-validation, F2-tuned threshold)
	.venv/bin/python -m src.train

train-recall80:  ## Same, but threshold = highest one keeping validation Recall >= 0.80
	.venv/bin/python -m src.train --min-recall 0.8 --output reports/recall80

compare-features:  ## Paired CV: serum creatinine + EF only vs all features (Chicco & Jurman 2020)
	.venv/bin/python -m src.train --feature-set compare --min-recall 0.8 \
		--models logistic_regression random_forest extra_trees gradient_boosting

download-zigong:  ## Download restricted Zigong data with your PhysioNet account (PHYSIONET_USER=<name>)
	.venv/bin/python scripts/download_zigong.py --user $(PHYSIONET_USER)

inspect-zigong:  ## Aggregate profile of the restricted Zigong data in data/raw/zigong/
	.venv/bin/python scripts/inspect_zigong.py

external-validation:  ## Develop on UCI, validate on Zigong (90-day death + sensitivity analyses)
	.venv/bin/python scripts/run_external_validation.py

train-fast:  ## Fast training run, skipping cross-validation
	.venv/bin/python -m src.train --models logistic_regression random_forest --no-cv

test:  ## Run the full unit test suite
	.venv/bin/python -m pytest -v

lint:  ## Check source code quality
	.venv/bin/python -m ruff check src tests scripts

clean:  ## Remove temporary files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
