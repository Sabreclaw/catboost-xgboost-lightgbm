# Load testing with Locust
This repository includes a simple Locust setup to stress test the FastAPI inference server.

## Index
- [Folder layout](#folder-layout)
- [Prerequisites](#prerequisites)
- [Prepare test data](#prepare-your-test-data)
- [Configuration (config.json)](#configuration-configjson)
- [Run Locust (headless)](#run-locust-headless)
- [Notes](#notes)
- [Example config.json](#example-configjson)

## Folder layout
- test-server/
  - locustfile.py – Locust scenarios that read test-server/test_files/X_test.csv, ignore the first column (id), and send requests to /invocation
  - test_files/ – place X_test.csv and y_test.csv here (files not provided)
  - run_locust_headless.sh – convenience script to run Locust in headless mode

## Prerequisites
- Server running (see earlier sections) on http://localhost:8000 or another host
- Dependencies installed (from repository root):
  - pip install -r requirements.txt

## Prepare your test data
- Create the folder test-server/test_files
- Put X_test.csv and y_test.csv inside it
- X_test.csv: the first column must be an id; all remaining columns are features to send in requests

## Configuration (config.json)
- Locust defaults can be configured in test-server/config.json
- Fields:
  - host: base URL of the model server (used if --host is not provided)
  - users: default total users (documentation only; use CLI flags to apply)
  - spawn_rate: default user spawn rate per second (documentation only)
  - duration: default test duration (documentation only)
  - stop_timeout: default graceful stop timeout in seconds (documentation only)
  - wait_time.min_seconds / max_seconds: controls per-user wait times between tasks
  - pred_method: optional prediction method to append as /invocation?method=...
  - test_files.x_test_path: optional override path to X_test.csv (relative to test-server/ or absolute)
- Precedence:
  - CLI flags/environment always override config.json.
  - PRED_METHOD env var overrides pred_method in config.

## Run Locust (headless)
Option A – use the helper script:
```bash
# from repository root
bash test-server/run_locust_headless.sh http://localhost:8000 200 20 2m DEBUG
```
- Arguments: HOST USERS SPAWN_RATE DURATION [LOGLEVEL]
  - HOST: default http://localhost:8000
  - USERS: total concurrent users (e.g., 200)
  - SPAWN_RATE: users spawned per second (e.g., 20)
  - DURATION: test time (e.g., 2m, 5m, 1h)
  - LOGLEVEL: Locust logging level (default INFO). Examples: INFO, DEBUG, WARNING
- Optional: export PRED_METHOD=predict_proba to have Locust call /invocation?method=predict_proba
- Alternatively, set the LOGLEVEL environment variable instead of passing it as the 5th argument.

Option B – raw Locust command:
```bash
locust -f test-server/locustfile.py \
  --headless \
  --host http://localhost:8000 \
  -u 200 \
  -r 20 \
  -t 2m \
  --stop-timeout 60
```

## Notes
- locustfile.py expects test-server/test_files/X_test.csv to exist; it ignores the first (id) column and uses the rest as features.
- The script sends only single-row requests to POST /invocation. No /health checks are performed.
- For a more aggressive test, increase USERS and SPAWN_RATE and reduce wait times if needed.

## Example config.json
```json
{
  "host": "http://localhost:8000",
  "users": 200,
  "spawn_rate": 20,
  "duration": "2m",
  "stop_timeout": 60,
  "wait_time": { "min_seconds": 0.01, "max_seconds": 0.1 },
  "pred_method": null,
  "test_files": { "x_test_path": "test-server/test_files/X_test.csv" }
}
```

