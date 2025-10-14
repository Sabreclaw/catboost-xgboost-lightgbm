import os
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


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


def to_dataframe(payload: Any) -> pd.DataFrame:
    """
    Convert various JSON payload shapes into a pandas DataFrame.

    Supported shapes:
    - {"instances": [ {feature: value, ...}, ... ]}
    - {"inputs": [ ... ]}
    - {"columns": [...], "data": [[...], [...]]}
    - [ {feature: value, ...}, ... ]
    - {feature: value, ...}
    - [[...], [...]] with optional {"columns": [...]} wrapping
    """
    data = payload
    if isinstance(data, dict):
        # sklearn-like table: {"columns": [...], "data": [[...]]}
        if set(data.keys()) >= {"columns", "data"}:
            columns = data["columns"]
            rows = data["data"]
            return pd.DataFrame(rows, columns=columns)
        # common wrapper keys
        if "instances" in data:
            data = data["instances"]
        elif "inputs" in data:
            data = data["inputs"]
        else:
            # assume it's a single row dict
            return pd.DataFrame([data])

    # Now data is list-like (list of dicts or list of lists)
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return pd.DataFrame(data)
        # list of lists — build a DataFrame without column names
        return pd.DataFrame(data)

    # Empty or unrecognized — return empty DataFrame to let caller validate
    return pd.DataFrame()

# Configure logging level from env: LOG_LEVEL or DEBUG=1
_LOG_LEVEL = os.getenv("LOG_LEVEL")
if not _LOG_LEVEL:
    _LOG_LEVEL = "DEBUG" if str(os.getenv("DEBUG", "")).lower() in ("1", "true", "yes", "on") else "INFO"
logging.basicConfig(
    level=getattr(logging, str(_LOG_LEVEL).upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("model-server")

app = FastAPI(title="Model Inference Server", version="1.0.0")


@app.on_event("startup")
def startup_event() -> None:
    selected_key, model_path, err = resolve_model_path()
    app.state.model_key = selected_key
    app.state.model_path = model_path
    app.state.model_error = err
    app.state.model = None

    logger.info("startup: LOAD_MODEL=%s resolved to key=%s path=%s", os.getenv("LOAD_MODEL"), selected_key, model_path)
    if err:
        logger.error("startup: model resolution error: %s", err)

    if model_path and not err:
        try:
            model = load_pickle(model_path)
            app.state.model = model
            app.state.model_error = None
            logger.info("startup: model loaded successfully from %s", model_path)
        except Exception as e:
            app.state.model = None
            app.state.model_error = f"Failed to load model from '{model_path}': {e}"
            logger.exception("startup: failed to load model from %s", model_path)


@app.get("/health")
def health() -> JSONResponse:
    loaded = app.state.model is not None and app.state.model_error is None
    if not loaded:
        logger.warning("health: model not loaded. key=%s path=%s err=%s", app.state.model_key, app.state.model_path, app.state.model_error)
    return JSONResponse(
        {
            "status": "ok" if loaded else "error",
            "loaded": loaded,
            "model": app.state.model_key,
            "model_path": app.state.model_path,
            "error": app.state.model_error,
        }
    )


@app.post("/invocation")
async def invocation(request: Request) -> JSONResponse:
    # Check if model loaded
    if app.state.model is None or app.state.model_error is not None:
        logger.error("invocation: 503 model not ready. err=%s", app.state.model_error)
        raise HTTPException(status_code=503, detail=app.state.model_error or "Model not loaded")

    try:
        payload = await request.json()
        logger.debug(
            "invocation: received payload type=%s keys=%s query_method=%s",
            type(payload).__name__,
            list(payload.keys()) if isinstance(payload, dict) else None,
            request.query_params.get("method"),
        )
    except Exception:
        logger.exception("invocation: failed to parse JSON body")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Convert payload to DataFrame
    df = to_dataframe(payload)
    if df.empty and (not isinstance(payload, list) or len(payload) != 0):
        logger.warning("invocation: empty/unrecognized payload. type=%s body=%s", type(payload).__name__, payload)
        raise HTTPException(status_code=400, detail="Empty or unrecognized payload format")

    logger.debug("invocation: dataframe shape=%s columns=%s", tuple(df.shape), list(df.columns))

    model = app.state.model

    # Choose prediction method: default to predict; allow override via query param or body key
    method_name = None
    # Allow method override in body at top-level
    if isinstance(payload, dict) and "method" in payload:
        method_name = str(payload.get("method")).strip()
    # Fallback to query parameter
    if method_name is None:
        method_name = request.query_params.get("method") or "predict"
    method_name = method_name or "predict"
    logger.debug("invocation: using method=%s", method_name)

    # Resolve prediction method
    if not hasattr(model, method_name):
        # Fallbacks for common cases
        if method_name == "predict_proba" and hasattr(model, "predict"):
            method = getattr(model, "predict")
            logger.info("invocation: method predict_proba not found; falling back to predict")
        else:
            logger.error("invocation: unsupported method=%s on model=%s", method_name, type(model).__name__)
            raise HTTPException(status_code=400, detail=f"Model does not support method '{method_name}'")
    else:
        method = getattr(model, method_name)

    try:
        # Some models expect numpy arrays; most accept DataFrames
        X = df
        try:
            preds = method(X)
        except Exception:
            # Retry with numpy values if DataFrame fails
            logger.debug("invocation: retrying with numpy array due to DF call failure", exc_info=True)
            preds = method(X.values)

        # Convert to JSON-serializable
        if hasattr(preds, "tolist"):
            out = preds.tolist()
        elif isinstance(preds, (np.ndarray, pd.Series)):
            out = preds.tolist()
        elif isinstance(preds, (list, tuple)):
            out = list(preds)
        else:
            # scalar
            out = [preds]

        logger.debug("invocation: success. n_preds=%d", len(out))
        return JSONResponse({"predictions": out})
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("invocation: inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "Model Inference Server running.",
        "health": "/health",
        "invoke": "/invocation",
        "env": {
            "LOAD_MODEL": os.getenv("LOAD_MODEL"),
        },
        "models_dir": str(Path(__file__).resolve().parent.parent / "models"),
        "model_path": app.state.model_path,
    }
