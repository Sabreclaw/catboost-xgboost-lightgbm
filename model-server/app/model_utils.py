import os
import pickle
from pathlib import Path
from typing import Any, Optional, Tuple


def resolve_model_path() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Determine which model file to load based on LOAD_MODEL env var,
    assuming a fixed 'models' directory at the repository root.

    Returns: (selected_key, model_path, error)
      - selected_key: 'catboost' | 'lgbm' | 'xgboost' | None
      - model_path: string path if resolved
      - error: error string if could not resolve
    """
    # repo root is the parent of the 'app' directory containing this file
    repo_root = Path(__file__).resolve().parent.parent
    models_dir = repo_root / "models"

    key = (os.getenv("LOAD_MODEL") or "").strip().lower()
    mapping = {
        "catboost": "Catboost_model.pkl",
        "lgbm": "LGBM_model.pkl",
        "xgboost": "XGBoost_model.pkl",
    }
    if not key:
        return None, None, "LOAD_MODEL is not set. Set it to one of: catboost, lgbm, xgboost."
    if key not in mapping:
        return None, None, f"Unsupported LOAD_MODEL='{key}'. Use one of: catboost, lgbm, xgboost."

    candidate = models_dir / mapping[key]
    if not candidate.exists():
        return key, None, f"Model file '{candidate}' not found. Place the file in the repository 'models' folder."
    return key, str(candidate.resolve()), None


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
