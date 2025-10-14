import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .model_utils import resolve_model_path, load_pickle
from .data_utils import to_dataframe
from .energy import EnergyProfiler

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




# global profiler instance
energy_profiler = EnergyProfiler()


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


@app.post("/energy-measurement/start")
async def energy_start(request: Request) -> JSONResponse:
    # optional body: {"model": "name-to-use"}
    model_name = None
    try:
        if request.headers.get("content-length") and int(request.headers.get("content-length", 0)) > 0:
            body = await request.json()
            if isinstance(body, dict):
                model_name = body.get("model")
    except Exception:
        # ignore body parse errors; treat as no-override
        logger.debug("energy start: ignoring invalid JSON body", exc_info=True)
    try:
        default_name = app.state.model_key or os.getenv("LOAD_MODEL")
        effective_name = model_name or default_name
        info = energy_profiler.start(effective_name)
        return JSONResponse({"status": "started", **info})
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="energibridge binary not found on PATH")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/energy-measurement/stop")
async def energy_stop() -> JSONResponse:
    try:
        info = energy_profiler.stop()
        return JSONResponse({"status": "stopped", **info})
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))


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
