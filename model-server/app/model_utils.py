import os
import pickle
from pathlib import Path
from typing import Any, Optional, Tuple


def resolve_model_path() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Determine which model file to load based on LOAD_MODEL and DATASET_NAME env vars.
    New convention: model files are placed under 'models' directory and named as
      <dataset_name>_<Algo>.pkl
    where Algo is one of: CatBoost, LightGBM, XGBoost.

    Returns: (selected_key, model_path, error)
      - selected_key: combined key, e.g., 'credit_card_transactions/catboost'
      - model_path: string path if resolved
      - error: error string if could not resolve
    """
    # repo root is the parent of the 'app' directory containing this file
    repo_root = Path(__file__).resolve().parent.parent
    models_dir = repo_root / "experiment-results" / "models"

    # Required: dataset name
    dataset = (os.getenv("DATASET_NAME") or os.getenv("DATASET") or "").strip()
    model_key = (os.getenv("LOAD_MODEL") or "").strip().lower()

    # Map model key to filename suffix used in artifacts
    algo_suffix = {
        "catboost": "CatBoost",
        "lgbm": "LightGBM",
        "xgboost": "XGBoost",
    }

    if not dataset:
        return None, None, "DATASET_NAME is not set. Set it to one of your available datasets (e.g., credit_card_transactions, diabetic, healthcare-dataset-stroke, UNSW_NB15_merged)."
    if not model_key:
        return None, None, "LOAD_MODEL is not set. Set it to one of: catboost, lgbm, xgboost."
    if model_key not in algo_suffix:
        return None, None, f"Unsupported LOAD_MODEL='{model_key}'. Use one of: catboost, lgbm, xgboost."

    candidate = models_dir / f"{dataset}_{algo_suffix[model_key]}.pkl"
    if not candidate.exists():
        return f"{dataset}/{model_key}", None, (
            f"Model file '{candidate}' not found. Ensure the file exists in model-server/models/ with naming '<dataset>_<Algo>.pkl'."
        )
    return f"{dataset}/{model_key}", str(candidate.resolve()), None


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
