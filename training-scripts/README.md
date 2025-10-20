# Training CLI (CatBoost, LightGBM, XGBoost)

This directory contains the standalone training CLI used to train and evaluate gradient boosting models on binary classification datasets. It powers the example workflows in the repository and can optionally export dataset splits for use by the load-testing client.

- Script: training.py
- Algorithms: CatBoost, XGBoost, LightGBM
- Inputs: CSV or Parquet datasets
- Outputs: metrics (CSV), trained model artifacts, optional train/test splits and predicted probabilities

For an end-to-end overview of how training, serving, and testing connect, see the repository root README.

## Quick start

Option A — run all four example trainings via helper (recommended):

```bash
# from repository root
bash start.sh train
```

Option B — run a single training job manually (from repository root):

```bash
python training-scripts/training.py \
    --data credit_card_transactions.parquet \
    --target is_fraud \
    --test-size 0.2 \
    --drop-cols "Unnamed: 0" first last street city state zip lat long dob trans_num merch_zipcode merchant job \
    --save-splits
```

## Installation prerequisites

- Python 3.9+ recommended
- Install project requirements from the repository root:

```bash
pip install -r requirements.txt
```

Notes:
- Pickle loading and model training require the algorithm libraries (catboost, xgboost, lightgbm). These are included in the repository requirements.
- Reading Parquet files requires a Parquet engine for pandas (pyarrow recommended).

## What the script does

Given a dataset, the script:
- Loads the data (CSV or Parquet) and identifies the target column (provided via --target or inferred heuristically).
- Optionally drops specified columns and one-hot encodes categorical features (see --include-categorical and --max-categories).
- Splits into train/test sets with stratification.
- Trains CatBoost, XGBoost, and LightGBM with sensible defaults and class imbalance handling.
- Computes metrics (ROC AUC, PR AUC, Accuracy, Precision, Recall, F1, confusion matrix) and timing (train_time_s, infer_time_s).
- Appends results to experiment-results/metrics/all_metrics.csv and prints a clean table to the console.
- Saves trained model pickles to experiment-results/models/ using the naming convention <dataset>_<Algo>.pkl.
- Optionally saves train/test splits and a preprocessor object for downstream use (e.g., load testing).

## Inputs

- Dataset file in CSV or Parquet format.
- Target column name (optional). If not provided, the script attempts to infer a binary target from common names like [is_fraud, Class, target, label, fraud] or a binary 0/1 column.

Positive label mapping:
- Use --positive-label to indicate which raw value represents the positive class. Non-numeric labels will be mapped to 0/1 accordingly.

## Outputs

All outputs are written under experiment-results/ at the repository root.

- Metrics (always):
  - experiment-results/metrics/all_metrics.csv (appended per run)
- Models (always):
  - experiment-results/models/<dataset>_<Algo>.pkl
    - Example: credit_card_transactions_CatBoost.pkl
- Splits (when --save-splits is set):
  - experiment-results/splits/<dataset>/
    - X_train.parquet, X_test.parquet
    - y_train.parquet, y_test.parquet
    - preprocessor.pkl (scikit-learn ColumnTransformer)
    - split_info.json (summary with counts and ratios)
- Predicted probabilities (when --save-probs is set):
  - experiment-results/splits/<dataset>/predicted_probabilities.json

Dataset name derivation:
- The dataset name is inferred from the input filename stem (without extension) with some common suffixes removed (e.g., _data, _dataset, -data, -dataset, _processed, _cleaned). This name is used in the output filenames and folders.

## CLI arguments

- --data PATH (required): CSV or Parquet file path.
- --target NAME: Target column. If omitted, the script attempts to infer it.
- --positive-label VALUE: Value treated as the positive class (default 1). If the target is non-numeric, matching values are mapped to 1, others to 0.
- --drop-cols COL [COL ...]: Space-separated list of columns to drop.
- --id-col NAME: Optional ID column to drop.
- --test-size FLOAT: Test set fraction (default 0.2).
- --random-state INT: Random seed (default 42).
- --include-categorical: Enable one-hot encoding of categorical columns. If omitted, only numeric features are used.
- --max-categories INT: Maximum unique values to one-hot encode for a categorical column (default 50). Higher-cardinality categorical columns are dropped when encoding is enabled.
- --n-estimators INT: Number of trees for all models (default 300).
- --learning-rate FLOAT: Learning rate for all models (default 0.2).
- --max-depth INT: Max tree depth (model-specific defaults are used when not set).
- --n-jobs INT: Parallel threads (default: CPU count).
- --output-csv PATH: Optional path to write a copy of the metrics CSV. If a directory, a file named all_metrics.csv is created inside.
- --save-probs: Save predicted probabilities for the test set per model.
- --save-splits: Save train/test splits and the preprocessor for reuse.
- --verbose: Enable debug logging.

## Using trained models with the model server

The model server loads pickles from model-server/models/ using the filename convention <dataset>_<Algo>.pkl.

If you want to serve a newly trained model from experiment-results/models/, copy or symlink it into model-server/models/ and start the server with matching environment variables:

```bash
# from repository root
mkdir -p model-server/models
cp experiment-results/models/credit_card_transactions_CatBoost.pkl model-server/models/

# run the server
cd model-server
export DATASET_NAME=credit_card_transactions   # or diabetic, healthcare-dataset-stroke, UNSW_NB15_merged
export LOAD_MODEL=catboost                     # or lgbm, xgboost
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Check /health to confirm the server loaded the intended artifact and dataset.

## Using saved splits with the load tester

Locust (test-server) reads feature rows from:
- test-server/test_files/splits/<dataset>/X_test.parquet

If you used --save-splits, copy the generated Parquet to the expected location:

```bash
# from repository root
mkdir -p test-server/test_files/splits/credit_card_transactions
cp experiment-results/splits/credit_card_transactions/X_test.parquet \
   test-server/test_files/splits/credit_card_transactions/

# optional (not required by Locust):
cp experiment-results/splits/credit_card_transactions/y_test.parquet \
   test-server/test_files/splits/credit_card_transactions/
```

Then run the headless load test:

```bash
bash start.sh test http://localhost:8000 200 20 2m DEBUG
```

Tip: You can also set DATASET_NAME before running start.sh test to select a dataset for testing.

## Example commands (four datasets)

```bash
python training-scripts/training.py \
    --data diabetic_data.parquet \
    --target readmitted \
    --positive-label ">30" \
    --test-size 0.2 \
    --save-splits

python training-scripts/training.py \
    --data credit_card_transactions.parquet \
    --target is_fraud \
    --test-size 0.2 \
    --drop-cols "Unnamed: 0" first last street city state zip lat long dob trans_num merch_zipcode merchant job \
    --save-splits

python training-scripts/training.py \
    --data UNSW_NB15_merged.parquet \
    --target label \
    --test-size 0.2 \
    --save-splits

python training-scripts/training.py \
    --data healthcare-dataset-stroke-data.parquet \
    --target stroke \
    --test-size 0.2 \
    --save-splits
```

## Troubleshooting

- Target not found or inferred:
  - Provide --target explicitly. The script’s inference covers only common names and simple 0/1 heuristics.
- Positive class mapping seems off:
  - Verify --positive-label matches your dataset’s positive value (e.g., ">30" for diabetic_data.parquet).
- Parquet read errors:
  - Ensure pyarrow (recommended) or fastparquet is installed and importable by pandas.
- Library import errors for catboost/xgboost/lightgbm:
  - Reinstall from requirements: pip install -r requirements.txt
- Metric values are NaN:
  - Check that the predicted scores are valid and the labels are binary (0/1). Ensure there is at least one positive and one negative sample in the test split.

## Reproducibility

- Use --random-state to control the train/test split and model randomness.
- Class imbalance is handled via computed positive class weights; see the script (compute_pos_weight) for details.
