#!/usr/bin/env python3
"""
Train and evaluate CatBoost, XGBoost, and LightGBM models for fraud detection (or any binary classification).
Outputs a concise table of metrics to the console and optionally saves a CSV report.

Usage examples:
  python fraud_detection_catboost_xgboost_lightgbm.py \
      --data credit_card_transactions.csv \
      --target is_fraud \
      --drop-cols Unnamed: 0 first last street city state zip lat long dob trans_num merch_zipcode merchant job \
      --test-size 0.2 --random-state 42

Switch datasets by changing --data and --target (and optionally --drop-cols).
Only CatBoost, XGBoost, LightGBM algorithms are used.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Third-party model libraries
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pickle


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", required=False, default=None, help="Target column name (binary)")
    parser.add_argument(
        "--positive-label",
        required=False,
        default=1,
        help="Label to treat as positive class. If target is non-numeric, values equal to this will be mapped to 1, others to 0.",
    )
    parser.add_argument(
        "--drop-cols",
        nargs="*",
        default=None,
        help="Columns to drop before modeling. Space-separated names.",
    )
    parser.add_argument("--id-col", default=None, help="Optional ID column to drop")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size fraction (0-1)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--include-categorical",
        action="store_true",
        help="If provided, one-hot encode categorical columns (object/bool/category). Otherwise only numeric columns are used.",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=50,
        help="Maximum unique values to one-hot encode for a categorical column (columns with more unique values are dropped).",
    )
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of trees for all models")
    parser.add_argument("--learning-rate", type=float, default=0.2, help="Learning rate for all models")
    parser.add_argument("--max-depth", type=int, default=5, help="Max depth for trees (model defaults if not set)")
    parser.add_argument("--n-jobs", type=int, default=os.cpu_count() or 4, help="Parallel threads")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to write the metrics table CSV. If directory, a file will be created inside.",
    )
    parser.add_argument(
        "--save-probs",
        action="store_true",
        help="If set, saves per-model predicted probabilities for the test set into experiment-results directory.",
    )
    parser.add_argument(
        "--save-splits",
        action="store_true",
        help="If set, saves train/test splits as parquet files for later use.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["is_fraud", "Class", "target", "label", "fraud"]
    for c in candidates:
        if c in df.columns:
            return c
    # Heuristic: binary column with values {0,1}
    for col in df.columns:
        vals = set(df[col].dropna().unique().tolist()[:5])
        if vals.issubset({0, 1}) and df[col].nunique() <= 2:
            return col
    return None


def cast_binary_target(y: pd.Series, positive_label) -> pd.Series:
    # If numeric already 0/1, return
    if pd.api.types.is_numeric_dtype(y):
        # Map numeric positive label to 1 if needed
        if positive_label in (1, 1.0):
            # Ensure 0/1
            uniq = set(pd.Series(y).dropna().unique())
            if uniq.issubset({0, 1}):
                return y.astype(int)
        # Map equality to 1 else 0
        return (y == positive_label).astype(int)
    # Non-numeric: map positive_label to 1 else 0
    return (y.astype(str) == str(positive_label)).astype(int)


def get_dataset_name(data_path: str) -> str:
    """Extract dataset name from file path"""
    path = Path(data_path)
    # Remove extension and any versioning suffixes
    name = path.stem
    # Remove common suffixes
    for suffix in ['_data', '_dataset', '-data', '-dataset', '_processed', '_cleaned']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def create_output_dirs() -> Tuple[Path, Path, Path]:
    """Create metrics, models, and splits directories"""
    base_dir = Path("experiment-results")
    metrics_dir = base_dir / "metrics"
    models_dir = base_dir / "models"
    splits_dir = base_dir / "splits"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    return metrics_dir, models_dir, splits_dir


def train_test_preprocess(
        df: pd.DataFrame,
        target_col: str,
        include_categorical: bool,
        max_categories: int,
        drop_cols: Optional[List[str]],
        id_col: Optional[str],
        positive_label,
        test_size: float,
        random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, List[
    str], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Drop unwanted columns
    cols_to_drop = []
    if drop_cols:
        cols_to_drop.extend([c for c in drop_cols if c in df.columns])
    if id_col and id_col in df.columns:
        cols_to_drop.append(id_col)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logging.info("Dropped columns: %s", cols_to_drop)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data. Available: {list(df.columns)[:20]}...")

    y_raw = df[target_col]
    X_df = df.drop(columns=[target_col])
    y = cast_binary_target(y_raw, positive_label=positive_label)

    # Feature typing
    numeric_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
    cat_cols = [c for c in X_df.columns if c not in numeric_cols]

    if not include_categorical:
        cat_cols = []

    # Limit high-cardinality columns
    limited_cat_cols = []
    dropped_high_card = []
    for c in cat_cols:
        nunique = X_df[c].nunique(dropna=True)
        if nunique <= max_categories:
            limited_cat_cols.append(c)
        else:
            dropped_high_card.append(c)
    if dropped_high_card:
        logging.info("Dropped high-cardinality categorical columns (> %d uniques): %s", max_categories,
                     dropped_high_card)

    transformers = []
    if limited_cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                limited_cat_cols,
            )
        )
    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))

    preproc = ColumnTransformer(transformers=transformers, remainder="drop")

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit the preprocessor to compute transformed feature names (optional)
    preproc.fit(X_train_df)
    try:
        # scikit-learn >= 1.0
        feature_names = preproc.get_feature_names_out().tolist()
    except Exception:
        feature_names = []

    X_train = preproc.transform(X_train_df)
    X_test = preproc.transform(X_test_df)

    # Replace NaNs if any remaining
    X_train = np.nan_to_num(X_train, copy=False)
    X_test = np.nan_to_num(X_test, copy=False)

    return X_train, X_test, np.asarray(y_train), np.asarray(
        y_test), preproc, feature_names, X_train_df, X_test_df, y_train, y_test


def compute_pos_weight(y_train: np.ndarray) -> float:
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    if pos == 0:
        return 1.0
    return max(1.0, neg / pos)


def build_models(n_estimators: int, learning_rate: float, max_depth: Optional[int], n_jobs: int, pos_weight: float):
    # Shared tree depth default per model left to library defaults if None
    models = []

    cb_params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=n_estimators,
        learning_rate=learning_rate,
        depth=max_depth if max_depth is not None else 6,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        class_weights=[1.0, pos_weight],
    )
    models.append(("CatBoost", CatBoostClassifier(**cb_params)))

    xgb_params = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth if max_depth is not None else 6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=n_jobs,
        tree_method="hist",
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
    )
    models.append(("XGBoost", XGBClassifier(**xgb_params)))

    lgb_params = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=63 if max_depth is None else min(2 ** max_depth, 1024),
        max_depth=-1 if max_depth is None else max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        n_jobs=n_jobs,
        random_state=42,
        reg_lambda=1.0,
        scale_pos_weight=pos_weight,
        verbose=-1,
    )
    models.append(("LightGBM", LGBMClassifier(**lgb_params)))

    return models


def predict_proba_safe(model, X: np.ndarray) -> np.ndarray:
    # Try predict_proba, then decision_function, else predict
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if isinstance(proba, list):
            proba = np.asarray(proba)
        # Binary: take positive class column
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        if proba.ndim == 1:
            return proba
    if hasattr(model, "decision_function"):
        df = model.decision_function(X)
        if isinstance(df, list):
            df = np.asarray(df)
        # Scale to 0..1 with logistic for binary
        try:
            return 1.0 / (1.0 + np.exp(-df))
        except Exception:
            return df
    # Fallback to label predictions 0/1
    preds = model.predict(X)
    return preds.astype(float)


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test, dataset_name: str) -> dict:
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_proba = predict_proba_safe(model, X_test)
    infer_time = time.perf_counter() - t1

    # Choose threshold 0.5 for binary predictions
    y_pred = (y_proba >= 0.5).astype(int)

    # Metrics
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_test, y_proba)
    except Exception:
        pr_auc = float("nan")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel().tolist()

    return {
        "dataset": dataset_name,
        "model": name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "train_time_s": train_time,
        "infer_time_s": infer_time,
        "timestamp": datetime.now().isoformat(timespec='seconds'),
    }


def append_or_create_metrics_csv(metrics_dir: Path, new_metrics: pd.DataFrame) -> None:
    """Append new metrics to existing CSV or create new one"""
    metrics_file = metrics_dir / "all_metrics.csv"

    if metrics_file.exists():
        # Append to existing file
        existing_metrics = pd.read_csv(metrics_file)
        combined_metrics = pd.concat([existing_metrics, new_metrics], ignore_index=True)
        combined_metrics.to_csv(metrics_file, index=False)
        logging.info("Appended metrics to existing file: %s", metrics_file)
    else:
        # Create new file
        new_metrics.to_csv(metrics_file, index=False)
        logging.info("Created new metrics file: %s", metrics_file)


def save_train_test_splits(splits_dir: Path, dataset_name: str,
                           X_train_df: pd.DataFrame, X_test_df: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series,
                           preproc: ColumnTransformer) -> None:
    """Save train/test splits as parquet files for later use"""

    # Create dataset-specific subdirectory
    dataset_splits_dir = splits_dir / dataset_name
    dataset_splits_dir.mkdir(parents=True, exist_ok=True)

    # Save the feature splits
    X_train_df.to_parquet(dataset_splits_dir / "X_train.parquet", index=False)
    X_test_df.to_parquet(dataset_splits_dir / "X_test.parquet", index=False)

    # Save target variables
    y_train.to_frame().to_parquet(dataset_splits_dir / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(dataset_splits_dir / "y_test.parquet", index=False)

    # Save the preprocessor
    with open(dataset_splits_dir / "preprocessor.pkl", 'wb') as f:
        pickle.dump(preproc, f)

    # Save split info as JSON
    split_info = {
        "dataset_name": dataset_name,
        "train_samples": len(X_train_df),
        "test_samples": len(X_test_df),
        "features": X_train_df.shape[1],
        "positive_samples_train": int(y_train.sum()),
        "positive_samples_test": int(y_test.sum()),
        "positive_ratio_train": float(y_train.mean()),
        "positive_ratio_test": float(y_test.mean()),
        "test_size": len(X_test_df) / (len(X_train_df) + len(X_test_df)),
        "timestamp": datetime.now().isoformat(timespec='seconds')
    }

    with open(dataset_splits_dir / "split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)

    logging.info("Saved train/test splits to: %s", dataset_splits_dir)
    logging.info("Split info: %d train, %d test samples", len(X_train_df), len(X_test_df))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    data_path = Path(args.data)
    if not data_path.exists():
        logging.error("Data file not found: %s", data_path)
        return 2

    # Get dataset name and create output directories
    dataset_name = get_dataset_name(args.data)
    metrics_dir, models_dir, splits_dir = create_output_dirs()
    logging.info("Dataset name: %s", dataset_name)
    logging.info("Metrics directory: %s", metrics_dir)
    logging.info("Models directory: %s", models_dir)
    logging.info("Splits directory: %s", splits_dir)

    # Load dataset
    logging.info("Loading data: %s", data_path)
    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    logging.info("Loaded shape: %s columns: %d", df.shape, len(df.columns))

    target_col = args.target or infer_target_column(df)
    if not target_col:
        logging.error("Could not infer target column. Please specify --target.")
        return 2
    if args.target is None:
        logging.info("Inferred target column: %s", target_col)

    # Prepare data
    X_train, X_test, y_train, y_test, preproc, feat_names, X_train_df, X_test_df, y_train_series, y_test_series = train_test_preprocess(
        df=df,
        target_col=target_col,
        include_categorical=bool(args.include_categorical),
        max_categories=args.max_categories,
        drop_cols=args.drop_cols,
        id_col=args.id_col,
        positive_label=args.positive_label,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    logging.info(
        "Train/Test sizes: X_train=%s X_test=%s | Positives in train: %d (of %d)",
        X_train.shape,
        X_test.shape,
        int((y_train == 1).sum()),
        len(y_train),
    )

    # Save train/test splits if requested
    if args.save_splits:
        save_train_test_splits(splits_dir, dataset_name, X_train_df, X_test_df, y_train_series, y_test_series, preproc)

    pos_weight = compute_pos_weight(y_train)
    logging.info("Computed positive class weight (neg/pos): %.3f", pos_weight)

    models = build_models(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
        pos_weight=pos_weight,
    )

    results = []
    probs_to_save = {}

    for name, model in models:
        logging.info("Training model: %s", name)
        metrics = evaluate_model(name, model, X_train, y_train, X_test, y_test, dataset_name)
        results.append(metrics)

        # Save model with dataset name
        model_filename = models_dir / f"{dataset_name}_{name}.pkl"
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
        logging.info("Saved model to: %s", model_filename)

        if args.save_probs:
            y_proba = predict_proba_safe(model, X_test)
            probs_to_save[name] = y_proba.tolist()

    # Metrics table
    df_metrics = pd.DataFrame(results)
    order = [
        "dataset",
        "model",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "tn",
        "fn",
        "train_time_s",
        "infer_time_s",
        "timestamp",
    ]
    df_metrics = df_metrics[order]

    # Log as a clean table
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:0.4f}" if isinstance(v, (float, np.floating)) else str(v))
    logging.info("\n%s", df_metrics.to_string(index=False))

    # Save metrics to single consolidated CSV file
    append_or_create_metrics_csv(metrics_dir, df_metrics)

    # Also save to user-specified location if provided
    if args.output_csv:
        out_path = Path(args.output_csv)
        if out_path.is_dir() or str(out_path).endswith(os.sep):
            out_path = out_path / "all_metrics.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_metrics.to_csv(out_path, index=False)
        logging.info("Saved metrics CSV to additional location: %s", out_path)

    if args.save_probs:
        probs_filename = splits_dir / dataset_name / "predicted_probabilities.json"
        probs_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(probs_filename, "w", encoding="utf-8") as f:
            json.dump(probs_to_save, f)
        logging.info("Saved predicted probabilities to: %s", probs_filename)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())