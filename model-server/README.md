# Model Inference Server (FastAPI)

A minimal FastAPI server that loads a pickled ML model at startup and serves predictions via HTTP.

Index
- [Prerequisites](#prerequisites)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Run the server](#run-the-server)
- [Health check](#health-check)
- [Invoke predictions](#invoke-predictions)
- [Choosing the prediction method](#choosing-the-prediction-method)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)
- [Debug logging](#debug-logging)

- Models are expected under the repository `models` folder.
- Select which dataset and algorithm to load via `DATASET_NAME` and `LOAD_MODEL` environment variables.
- Two endpoints are provided:
  - `GET /health` – server and model status
  - `POST /invocation` – run inference

Model filename convention (relative to `./models`):
- `<dataset_name>_<Algo>.pkl`
  - Algo ∈ {`CatBoost`, `LightGBM`, `XGBoost`}
  - Example: `credit_card_transactions_CatBoost.pkl`, `diabetic_LightGBM.pkl`

FastAPI interactive docs are available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Prerequisites
- Python 3.9 or newer (3.10+ recommended)
- pip

Note: The pickled models might require their respective libraries (catboost, xgboost, lightgbm) to be installed in the environment where you run the server. See “Install dependencies” below.

Large model artifacts may be stored using Git LFS in this repository. If models were pulled via LFS, ensure Git LFS is installed and run the following from the repository root before starting the server (see the root README for details):
```bash
git lfs install
git lfs pull
```

## Project structure
```
model-server/
├─ app/
│  └─ main.py          # FastAPI application
├─ models/
│  ├─ credit_card_transactions_CatBoost.pkl
│  ├─ credit_card_transactions_LightGBM.pkl
│  ├─ credit_card_transactions_XGBoost.pkl
│  ├─ diabetic_CatBoost.pkl
│  ├─ diabetic_LightGBM.pkl
│  ├─ diabetic_XGBoost.pkl
│  ├─ healthcare-dataset-stroke_CatBoost.pkl
│  ├─ healthcare-dataset-stroke_LightGBM.pkl
│  ├─ healthcare-dataset-stroke_XGBoost.pkl
│  ├─ UNSW_NB15_merged_CatBoost.pkl
│  ├─ UNSW_NB15_merged_LightGBM.pkl
│  └─ UNSW_NB15_merged_XGBoost.pkl
└─ requirements.txt
```

## Setup
### 1) Create and activate a virtual environment (from repository root)
macOS/Linux (bash/zsh):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies (from repository root)
```bash
pip install -r requirements.txt
```

If the pickled model requires additional libraries that are commented out in `requirements.txt`, install them as needed, for example:
```bash
# Install only the libraries required by the pickle
pip install catboost     # for CatBoost pickles
pip install xgboost      # for XGBoost pickles
pip install lightgbm     # for LightGBM pickles
```

## Run the server
Set which dataset and model to load using `DATASET_NAME` and `LOAD_MODEL`, then start uvicorn.

macOS/Linux (bash/zsh):
```bash
export DATASET_NAME=credit_card_transactions   # or diabetic, healthcare-dataset-stroke, UNSW_NB15_merged
export LOAD_MODEL=catboost                     # or lgbm, xgboost
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Windows (PowerShell):
```powershell
$env:DATASET_NAME = "credit_card_transactions"  # or diabetic, healthcare-dataset-stroke, UNSW_NB15_merged
$env:LOAD_MODEL    = "catboost"                  # or lgbm, xgboost
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

If everything is OK, the server will load `<dataset>_<Algo>.pkl` from `./models` at startup.

## Health check
```bash
curl -s http://localhost:8000/health | jq
```
Example response:
```json
{
  "status": "ok",
  "loaded": true,
  "model": "credit_card_transactions/catboost",
  "model_path": "/absolute/path/to/model-server/models/credit_card_transactions_CatBoost.pkl",
  "error": null
}
```
If the model is missing or cannot be loaded, `status` will be `error` and `error` will explain why.

## Invoke predictions
Endpoint: `POST /invocation`

The server accepts several JSON shapes and will internally convert them to a pandas DataFrame. Below are common examples. Replace feature names/values with those appropriate for your model.

### 1) Single JSON object (one row)
```bash
curl -X POST http://localhost:8000/invocation \
  -H 'Content-Type: application/json' \
  -d '{"feature_1": 1.2, "feature_2": 3, "feature_3": "A"}'
```
Response:
```json
{"predictions":[0.123]}
```

### 2) List of JSON objects (multiple rows)
```bash
curl -X POST http://localhost:8000/invocation \
  -H 'Content-Type: application/json' \
  -d '[{"feature_1": 1.2, "feature_2": 3}, {"feature_1": 0.7, "feature_2": 9}]'
```

### 3) Wrapped as `instances`
```bash
curl -X POST 'http://localhost:8000/invocation' \
  -H 'Content-Type: application/json' \
  -d '{"instances": [{"feature_1": 1.2, "feature_2": 3}, {"feature_1": 0.7, "feature_2": 9}]}'
```

### 4) Columns + data (sklearn-like)
```bash
curl -X POST 'http://localhost:8000/invocation' \
  -H 'Content-Type: application/json' \
  -d '{
        "columns": ["feature_1", "feature_2"],
        "data": [[1.2, 3], [0.7, 9]]
      }'
```

### 5) List of lists (unnamed columns)
```bash
curl -X POST 'http://localhost:8000/invocation' \
  -H 'Content-Type: application/json' \
  -d '[[1.2, 3], [0.7, 9]]'
```

## Choosing the prediction method
By default, the server calls `predict`. You can choose a different method (e.g., `predict_proba`) via query param or body field.

- Query parameter:
```bash
curl -X POST 'http://localhost:8000/invocation?method=predict_proba' \
  -H 'Content-Type: application/json' \
  -d '{"instances": [{"feature_1": 1.2, "feature_2": 3}]}'
```

- Body field:
```bash
curl -X POST 'http://localhost:8000/invocation' \
  -H 'Content-Type: application/json' \
  -d '{
        "method": "predict_proba",
        "instances": [{"feature_1": 1.2, "feature_2": 3}]
      }'
```

Note: If the model does not implement the requested method, the server returns HTTP 400 with a helpful message.

## Troubleshooting
- `GET /health` shows `error`:
  - Ensure `DATASET_NAME` is set (e.g., `credit_card_transactions`, `diabetic`, `healthcare-dataset-stroke`, `UNSW_NB15_merged`).
  - Ensure `LOAD_MODEL` is set to one of: `catboost`, `lgbm`, `xgboost`.
  - Ensure the matching pickle exists in `./models/` using the convention `<dataset>_<Algo>.pkl`, where Algo is one of `CatBoost`, `LightGBM`, `XGBoost`.
    - Examples: `credit_card_transactions_CatBoost.pkl`, `diabetic_LightGBM.pkl`, `UNSW_NB15_merged_XGBoost.pkl`.
  - If the pickle requires extra dependencies, install them (catboost/xgboost/lightgbm).

- `POST /invocation` returns `400 Invalid JSON body`:
  - Check your JSON is valid and you set `Content-Type: application/json`.

- `POST /invocation` returns `400 Empty or unrecognized payload format`:
  - Use one of the documented payload shapes above.

- `POST /invocation` returns `400 Model does not support method ...`:
  - Use a method implemented by your model (commonly `predict`, sometimes `predict_proba`).

- `POST /invocation` returns `500 Inference failed: ...`:
  - The model raised an exception while predicting. Check your feature names/order and types.

## Notes
- This server is a minimal reference and is not hardened for production. Consider adding input validation, authentication, request limits, logging/metrics, and proper error handling for production use.
- Interactive documentation is available at http://localhost:8000/docs when the server is running.

## Debug logging
Verbose server logs can be enabled without significant changes to the run command.

Options:
- Set environment variable LOG_LEVEL=DEBUG (takes precedence), or set DEBUG=1 for a quick toggle.
- Or pass --log-level debug to uvicorn.

Examples (bash/zsh):
```bash
export DATASET_NAME=credit_card_transactions
export LOAD_MODEL=catboost
export LOG_LEVEL=DEBUG           # or: export DEBUG=1
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level debug
```

What you will see:
- Startup: selected model key, resolved model path, success/failure loading.
- /health: warnings when model is not loaded and any stored error message.
- /invocation: payload summary, resolved prediction method, DataFrame shape/columns, and detailed errors with stack traces if inference fails.
