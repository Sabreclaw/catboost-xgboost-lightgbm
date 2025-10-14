# Benchmark Gradient Boosting Models

This repository contains:
- model-server/ – a FastAPI inference server that loads a selected model and serves predictions.
- test-server/ – a Locust load-testing setup that sends single-row POST /invocation requests to the model server (no /health checks from Locust).
- fraud-detection-with-catboost-xgboost-lightgbm.ipynb – a notebook exploring/modeling fraud detection.

For detailed setup and usage, see the component-specific READMEs:
- Model server: model-server/README.md
- Load testing: test-server/README.md

Project map (high level):
```
repo-root/
├─ model-server/           # FastAPI app and model loading
├─ test-server/            # Locust scenarios, config, helpers
├─ credit_card_transactions.csv
└─ fraud-detection-with-catboost-xgboost-lightgbm.ipynb
```

Flow overview (Mermaid):
```mermaid
flowchart LR
    subgraph Client/Tester
        A[User / Locust]
    end

    A -->|"POST /invocation (JSON row)"| S[FastAPI Model Server]
    S -->|"Load selected model at startup"| M[(Model: CatBoost / LGBM / XGBoost)]
    S -->|Predict / Predict_proba| P[Predictions JSON]

    %% Optional health endpoint exposed by server (not called by Locust)
    A -.->|GET /health| S
```

Notes:
- The test-server is configured to send only single-row JSON payloads to POST /invocation and does not perform periodic /health checks.
- You can configure test-server behavior via test-server/config.json or CLI flags (see test-server/README.md).
- The model server supports choosing a prediction method via the method query parameter (e.g., /invocation?method=predict_proba). See model-server/README.md for details.


## Asset setup (zip)
Some large files are provided as .zip archives. Use the helper script to decompress them into the expected locations.

Archives and target locations:
- models.zip → model-server/models/
- test_files.zip → test-server/test_files/
- credit_card_transactions.csv.zip → ./credit_card_transactions.csv

Run the setup script (requires the zip/unzip tool):
```bash
# from repository root
bash setup.sh
```

If unzip is not installed, install zip/unzip tools first:
- macOS (Homebrew):
  - brew install zip
- Ubuntu/Debian:
  - sudo apt-get update && sudo apt-get install -y unzip
- Fedora:
  - sudo dnf install -y unzip
- CentOS/RHEL (yum):
  - sudo yum install -y unzip

Manual alternative (without setup.sh):
- Extract models.zip and place the resulting models/ folder under model-server/ so that you have model-server/models/ with the model .pkl files inside.
- Extract test_files.zip and place the resulting test_files/ folder under test-server/ so that you have test-server/test_files/ with X_test.csv (and optionally y_test.csv).
- Extract credit_card_transactions.csv.zip at the repository root so that you have ./credit_card_transactions.csv.

After extraction, you can:
- Start the model server (see model-server/README.md) — ensure LOAD_MODEL is set (catboost/lgbm/xgboost) and corresponding .pkl exists in model-server/models/.
- Run load tests (see test-server/README.md) — ensure test-server/test_files/X_test.csv exists.



## Quick start script (start.sh)
A helper script is available at the repository root to streamline serving the model or running load tests. It uses interactive prompts so you can confirm/skip each step.

## Running the model-server with Experiment Runner

- Example config: experiment-runner/examples/model-server-run/RunnerConfig.py
- How to run (from repository root):
  - python experiment-runner/ experiment-runner/examples/model-server-run/RunnerConfig.py
- Prerequisites:
  - Extract models.zip (or run ./setup.sh) so model-server/models/ contains the .pkl files.
  - Install server deps: pip install -r model-server/requirements.txt (preferably in model-server/.venv via start.sh serve)
  - Install ER deps: pip install -r experiment-runner/requirements.txt
  - Also install requests if missing: pip install requests
- Notes:
  - The example will try combinations of model types (catboost/lgbm/xgboost) and log levels (info/debug), skipping any missing model file.
  - It launches uvicorn in the model-server directory so app.main and models path resolve, waits for /health to be loaded, optionally does a warmup POST /invocation, then shuts down.

Prerequisites:
- Python 3 installed (for serving and optional venv creation)
- For load testing: Locust (the script can install it via test-server/requirements.txt if missing)

Usage:
- Serve the FastAPI model server (creates model-server/.venv if you agree, installs dependencies if you agree, and starts uvicorn):
  - With prompts and defaults
    - bash start.sh serve
  - With some flags (still prompts; flags set initial defaults)
    - bash start.sh serve --host 127.0.0.1 --port 8000 --model lgbm
  - During the run you will be asked:
    - Create model-server/.venv? (skip to use system Python)
    - Install dependencies from model-server/requirements.txt?
    - Choose LOAD_MODEL (catboost/lgbm/xgboost)
    - Choose host/port
    - Enable debug logs? (sets LOG_LEVEL=DEBUG and passes --log-level debug to uvicorn)

- Run Locust load tests (headless):
  - bash start.sh test [HOST] [USERS] [SPAWN_RATE] [DURATION] [LOGLEVEL]
  - Example: bash start.sh test http://localhost:8000 200 20 2m DEBUG
  - If Locust is not installed, the script can install it via: pip install -r test-server/requirements.txt (prompted)
  - It will run test-server/run_locust_headless.sh with the chosen parameters.

Notes:
- The script defaults to using a virtual environment at model-server/.venv for both serving and (if available) testing. If that venv doesn’t exist or you skip creating it, your system Python/pip will be used.
- For model serving details and environment variables, see model-server/README.md.
- For load testing details and configuration, see test-server/README.md.
