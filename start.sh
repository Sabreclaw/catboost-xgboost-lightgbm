#!/usr/bin/env bash
set -euo pipefail

# start.sh — helper to run the model server or load tests with interactive prompts
# Usage:
#   bash start.sh serve [--host 0.0.0.0] [--port 8000] [--model catboost]
#   bash start.sh test  [HOST] [USERS] [SPAWN_RATE] [DURATION] [LOGLEVEL]
#
# Notes:
# - In 'serve' mode, this script (optionally) creates model-server/.venv, installs
#   model dependencies, and starts the FastAPI server with optional debug logs.
# - In 'test' mode, it attempts to run Locust via test-server/run_locust_headless.sh
#   using an existing virtualenv if available, or the system Python. It will
#   offer to install test requirements if Locust is missing.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
MODEL_DIR="$REPO_ROOT/model-server"
VENV_DIR="$REPO_ROOT/.venv"
TEST_DIR="$REPO_ROOT/test-server"

usage() {
  cat <<EOF
Usage:
  $0 serve [--host HOST] [--port PORT] [--model MODEL_KEY]
  $0 test  [HOST] [USERS] [SPAWN_RATE] [DURATION] [LOGLEVEL] [MAX_REQUESTS]
  $0 train

Notes:
- MODEL_KEY can be one of: all, catboost, lgbm, xgboost (default: all)
- In 'all' mode the server loads all available models. Use dataset/model query params in /invocation to select per request.

Examples:
  # Serve with prompts (defaults host=0.0.0.0, port=8000, model=catboost)
  bash $0 serve

  # Serve with some flags
  bash $0 serve --host 127.0.0.1 --port 8000 --model lgbm

  # Run load test (headless)
  bash $0 test http://localhost:8000 200 20 2m DEBUG

  # Train all datasets with default options (saves splits)
  bash $0 train
EOF
}

confirm() {
  # confirm "Message" [default_y|default_n]
  local prompt="$1"
  local default="${2:-default_y}"
  local yn
  if [[ "$default" == "default_n" ]]; then
    read -r -p "$prompt [y/N]: " yn || true
    [[ -z "${yn:-}" ]] && yn="n"
  else
    read -r -p "$prompt [Y/n]: " yn || true
    [[ -z "${yn:-}" ]] && yn="y"
  fi
  case "${yn,,}" in
    y|yes) return 0 ;;
    *) return 1 ;;
  esac
}

ensure_python3() {
  if command -v python3 >/dev/null 2>&1; then
    return 0
  fi
  echo "ERROR: python3 not found in PATH." >&2
  echo "Please install Python 3 and retry." >&2
  exit 1
}

activate_venv() {
  # activates venv if it exists
  if [[ -d "$VENV_DIR" && -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
  fi
}

serve_mode() {
  local HOST="0.0.0.0"
  local PORT="8000"
  local MODEL_KEY="all"
  # Parse optional flags
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --host) HOST="$2"; shift 2 ;;
      --port) PORT="$2"; shift 2 ;;
      --model) MODEL_KEY="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
  done

  echo "== Serve mode =="
  ensure_python3

  # Step 1: Ensure virtual environment
  if [[ -d "$VENV_DIR" ]]; then
    echo "Virtual environment already exists at: $VENV_DIR"
  else
    if confirm "Create virtual environment at ./.venv?" default_y; then
      (cd "$REPO_ROOT" && python3 -m venv .venv)
      echo "Created venv at $VENV_DIR"
    else
      echo "Skipping venv creation (will use system Python)"
    fi
  fi

  # Step 2: Activate venv if present
  if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
    PYBIN="python"
    PIPBIN="pip"
  else
    PYBIN="python3"
    PIPBIN="pip3"
  fi

  # Step 3: Install root requirements (auto-skip if already satisfied in current Python env)
  REQ_FILE="$REPO_ROOT/requirements.txt"
  MISSING_PKGS="$("$PYBIN" - "$REQ_FILE" <<'PY'
import sys, re
try:
    from importlib import metadata
except ImportError:  # pragma: no cover
    import importlib_metadata as metadata  # type: ignore
path = sys.argv[1]
missing = []
with open(path, 'r') as f:
    for line in f:
        s = line.strip()
        if not s or s.startswith('#') or s.startswith('-r ') or s.startswith('--requirement'):
            continue
        m = re.match(r'^([A-Za-z0-9_.-]+)', s)
        if not m:
            continue
        name = m.group(1)
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
print(",".join(missing))
PY
)"
  if [[ -z "$MISSING_PKGS" ]]; then
    echo "All requirements already present in the current environment. Skipping dependency installation."
  else
    if [[ -d "$VENV_DIR" ]]; then
      echo "The following packages are missing in $VENV_DIR: $MISSING_PKGS"
    else
      echo "The following packages are missing: $MISSING_PKGS"
    fi
    if confirm "Install project requirements (pip install -r requirements.txt)?" default_y; then
      "$PIPBIN" install -r "$REQ_FILE"
    else
      echo "Skipping dependency installation"
    fi
  fi

  # Step 4: Choose model key first (default: all)
  read -r -p "Select LOAD_MODEL (all/catboost/lgbm/xgboost) [${MODEL_KEY}]: " MODEL_IN || true
  if [[ -n "${MODEL_IN:-}" ]]; then
    MODEL_KEY="$MODEL_IN"
  fi

  # Step 5: Choose dataset only if not 'all'
  local DATASET_NAME="${DATASET_NAME:-credit_card_transactions}"
  if [[ "${MODEL_KEY,,}" != "all" ]]; then
    read -r -p "Select DATASET_NAME (credit_card_transactions/diabetic/healthcare-dataset-stroke/UNSW_NB15_merged) [${DATASET_NAME}]: " DATASET_IN || true
    if [[ -n "${DATASET_IN:-}" ]]; then
      DATASET_NAME="$DATASET_IN"
    fi
  fi

  # Step 6: Choose host/port
  read -r -p "Server host [${HOST}]: " HOST_IN || true
  if [[ -n "${HOST_IN:-}" ]]; then
    HOST="$HOST_IN"
  fi
  read -r -p "Server port [${PORT}]: " PORT_IN || true
  if [[ -n "${PORT_IN:-}" ]]; then
    PORT="$PORT_IN"
  fi

  # Step 7: Debug logging?
  DEBUG_FLAGS=""
  if confirm "Enable DEBUG logging?" default_n; then
    export LOG_LEVEL=DEBUG
    DEBUG_FLAGS="--log-level debug"
  fi

  # Preflight: Verify model artifacts exist
  if [[ "${MODEL_KEY,,}" == "all" ]]; then
    local ANY_MODEL
    ANY_MODEL=$(ls "$REPO_ROOT/experiment-results/models"/*.pkl 2>/dev/null | head -n 1 || true)
    if [[ -z "$ANY_MODEL" ]]; then
      echo "ERROR: No model pickles found under $REPO_ROOT/experiment-results/models/. Run training first." >&2
      exit 1
    fi
    echo "Starting server in ALL-models mode. host=$HOST port=$PORT"
    export LOAD_MODEL="all"
    unset DATASET_NAME || true
  else
    local SUFFIX=""
    case "${MODEL_KEY,,}" in
      catboost) SUFFIX="CatBoost" ;;
      lgbm) SUFFIX="LightGBM" ;;
      xgboost) SUFFIX="XGBoost" ;;
      *) echo "ERROR: Unsupported model key: $MODEL_KEY (use all|catboost|lgbm|xgboost)" >&2; exit 1 ;;
    esac
    local MODEL_PATH="$REPO_ROOT/experiment-results/models/${DATASET_NAME}_${SUFFIX}.pkl"
    if [[ ! -f "$MODEL_PATH" ]]; then
      echo "ERROR: Model file not found: $MODEL_PATH" >&2
      echo "Ensure the model exists in $REPO_ROOT/experiment-results/models/ named '<dataset>_<Algo>.pkl'." >&2
      exit 1
    fi
    echo "Starting server with: DATASET_NAME=$DATASET_NAME LOAD_MODEL=$MODEL_KEY host=$HOST port=$PORT"
    export DATASET_NAME="$DATASET_NAME"
    export LOAD_MODEL="$MODEL_KEY"
  fi
  cd "$MODEL_DIR"
  exec uvicorn app.main:app --host "$HOST" --port "$PORT" $DEBUG_FLAGS
}

ensure_locust() {
  if command -v locust >/dev/null 2>&1; then
    return 0
  fi
  # Try venv locust if we activated earlier
  if command -v python >/dev/null 2>&1; then
    if python -c "import locust" >/dev/null 2>&1; then
      return 0
    fi
  fi
  echo "Locust is not available."
  if confirm "Install project requirements (pip install -r requirements.txt)?" default_y; then
    local PIPBIN="pip3"
    if command -v pip >/dev/null 2>&1; then PIPBIN="pip"; fi
    # If venv active, pip will install there
    $PIPBIN install -r "$REPO_ROOT/requirements.txt"
  else
    echo "Skipping installation; test mode may fail if Locust is missing."
  fi
}

test_mode() {
  echo "== Test mode (Locust) =="
  # If a venv exists for model-server, prefer to activate it
  if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
  fi

  ensure_locust

  local HOST="${1:-${HOST:-http://localhost:8000}}"
  local USERS="${2:-${USERS:-100}}"
  local SPAWN_RATE="${3:-${SPAWN_RATE:-100}}"
  local DURATION="${4:-${DURATION:-60s}}"
  local LOGLEVEL="${5:-${LOGLEVEL:-INFO}}"
  local MAX_REQUESTS="${6:-${MAX_REQUESTS:-15000}}"

  echo "Current settings:"
  echo "  HOST=$HOST"
  echo "  USERS=$USERS"
  echo "  SPAWN_RATE=$SPAWN_RATE"
  echo "  DURATION=$DURATION"
  echo "  LOGLEVEL=$LOGLEVEL"
  echo "  MAX_REQUESTS=$MAX_REQUESTS"

  if confirm "Edit settings before running?" default_n; then
    read -r -p "HOST [$HOST]: " IN || true; [[ -n "${IN:-}" ]] && HOST="$IN"
    read -r -p "USERS [$USERS]: " IN || true; [[ -n "${IN:-}" ]] && USERS="$IN"
    read -r -p "SPAWN_RATE [$SPAWN_RATE]: " IN || true; [[ -n "${IN:-}" ]] && SPAWN_RATE="$IN"
    read -r -p "DURATION [$DURATION]: " IN || true; [[ -n "${IN:-}" ]] && DURATION="$IN"
    read -r -p "LOGLEVEL [$LOGLEVEL]: " IN || true; [[ -n "${IN:-}" ]] && LOGLEVEL="$IN"
    read -r -p "MAX_REQUESTS [$MAX_REQUESTS]: " IN || true; [[ -n "${IN:-}" ]] && MAX_REQUESTS="$IN"
  fi

  if ! [[ -f "$TEST_DIR/run_locust_headless.sh" ]]; then
    echo "ERROR: $TEST_DIR/run_locust_headless.sh not found." >&2
    exit 1
  fi

  # Choose datasets for testing (default: all)
  local DATASET_NAME_TEST="all"
  read -r -p "Select DATASET_NAME for testing (all/credit_card_transactions/diabetic/healthcare-dataset-stroke/UNSW_NB15_merged) [${DATASET_NAME_TEST}]: " DATASET_TEST_IN || true
  if [[ -n "${DATASET_TEST_IN:-}" ]]; then
    DATASET_NAME_TEST="$DATASET_TEST_IN"
  fi

  # Choose models for testing (default: all)
  local MODEL_KEY_TEST="all"
  read -r -p "Select MODEL for testing (all/catboost/lgbm/xgboost) [${MODEL_KEY_TEST}]: " MODEL_TEST_IN || true
  if [[ -n "${MODEL_TEST_IN:-}" ]]; then
    MODEL_KEY_TEST="$MODEL_TEST_IN"
  fi

  # Ask how many times to run the experiment
  local RUNS="1"
  read -r -p "How many times do you want to run the experiment? [${RUNS}]: " RUNS_IN || true
  if [[ -n "${RUNS_IN:-}" ]]; then RUNS="$RUNS_IN"; fi
  if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [[ "$RUNS" -lt 1 ]]; then
    echo "Invalid number of runs '$RUNS'. Defaulting to 1."
    RUNS=1
  fi

  # Generate a batch GUID shared by all runs
  local BATCH_GUID
  if command -v uuidgen >/dev/null 2>&1; then
    BATCH_GUID="$(uuidgen)"
  elif command -v python3 >/dev/null 2>&1; then
    BATCH_GUID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
  else
    BATCH_GUID="batch-$(date +%s)-$RANDOM"
  fi
  echo "Batch GUID: $BATCH_GUID"

  if confirm "Run Locust now?" default_y; then
    # Build dataset list
    local DATASETS=()
    if [[ "${DATASET_NAME_TEST,,}" == "all" ]]; then
      DATASETS=("credit_card_transactions" "diabetic" "healthcare-dataset-stroke" "UNSW_NB15_merged")
    else
      DATASETS=("$DATASET_NAME_TEST")
    fi
    # Build model list
    local MODELS=()
    case "${MODEL_KEY_TEST,,}" in
      all) MODELS=("catboost" "lgbm" "xgboost") ;;
      catboost|lgbm|xgboost) MODELS=("${MODEL_KEY_TEST,,}") ;;
      *) echo "Unknown model selection: $MODEL_KEY_TEST" >&2; return 1 ;;
    esac

    local ds md i
    for ds in "${DATASETS[@]}"; do
      local X_PATH_CHECK="$TEST_DIR/../experiment-results/splits/${ds}/X_test.parquet"
      if [[ ! -f "$X_PATH_CHECK" ]]; then
        echo "WARNING: Skipping dataset '$ds' because test split not found at $X_PATH_CHECK" >&2
        continue
      fi
      for md in "${MODELS[@]}"; do
        for (( i=1; i<=RUNS; i++ )); do
          echo "\n===> Starting run $i of $RUNS (dataset=$ds, model=$md, host=$HOST, users=$USERS, spawn_rate=$SPAWN_RATE, duration=$DURATION, max_requests=$MAX_REQUESTS)"
          (cd "$TEST_DIR" && DATASET_NAME="$ds" LOAD_MODEL="$md" BATCH_GUID="$BATCH_GUID" bash ./run_locust_headless.sh "$HOST" "$USERS" "$SPAWN_RATE" "$DURATION" "$LOGLEVEL" "$MAX_REQUESTS")
          if [[ "$i" -lt "$RUNS" ]]; then
            echo "Run $i finished. Waiting 5 seconds before next run..."
            sleep 5
          fi
        done
      done
    done
  else
    echo "Skipped running Locust."
  fi
}

train_mode() {
  echo "== Train mode (all datasets) =="
  ensure_python3

  # Ensure virtual environment
  if [[ -d "$VENV_DIR" ]]; then
    echo "Virtual environment already exists at: $VENV_DIR"
  else
    if confirm "Create virtual environment at ./.venv?" default_y; then
      (cd "$REPO_ROOT" && python3 -m venv .venv)
      echo "Created venv at $VENV_DIR"
    else
      echo "Skipping venv creation (will use system Python)"
    fi
  fi

  # Activate venv if present
  if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
    PYBIN="python"
    PIPBIN="pip"
  else
    PYBIN="python3"
    PIPBIN="pip3"
  fi

  # Install requirements if needed
  REQ_FILE="$REPO_ROOT/requirements.txt"
  MISSING_PKGS="$($PYBIN - "$REQ_FILE" <<'PY'
import sys, re
try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata  # type: ignore
path = sys.argv[1]
missing = []
with open(path, 'r') as f:
    for line in f:
        s = line.strip()
        if not s or s.startswith('#') or s.startswith('-r ') or s.startswith('--requirement'):
            continue
        m = re.match(r'^([A-Za-z0-9_.-]+)', s)
        if not m:
            continue
        name = m.group(1)
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
print(",".join(missing))
PY
)"
  if [[ -n "$MISSING_PKGS" ]]; then
    echo "Installing missing packages: $MISSING_PKGS"
    $PIPBIN install -r "$REQ_FILE"
  fi

  TRAIN_SCRIPT="$REPO_ROOT/training-scripts/training.py"
  if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT" >&2
    exit 1
  fi

  echo "Running training for diabetic_data.parquet"
  $PYBIN "$TRAIN_SCRIPT" \
    --data "$REPO_ROOT/training-scripts/diabetic_data.parquet" \
    --target readmitted \
    --positive-label ">30" \
    --test-size 0.2 \
    --save-splits || { echo "Training failed for diabetic_data"; exit 1; }

  echo "Running training for credit_card_transactions.parquet"
  $PYBIN "$TRAIN_SCRIPT" \
    --data "$REPO_ROOT/training-scripts/credit_card_transactions.parquet" \
    --target is_fraud \
    --test-size 0.2 \
    --drop-cols "Unnamed: 0" first last street city state zip lat long dob trans_num merch_zipcode merchant job \
    --save-splits || { echo "Training failed for credit_card_transactions"; exit 1; }

  echo "Running training for UNSW_NB15_merged.parquet"
  $PYBIN "$TRAIN_SCRIPT" \
    --data "$REPO_ROOT/training-scripts/UNSW_NB15_merged.parquet" \
    --target label \
    --test-size 0.2 \
    --save-splits || { echo "Training failed for UNSW_NB15_merged"; exit 1; }

  echo "Running training for healthcare-dataset-stroke-data.parquet"
  $PYBIN "$TRAIN_SCRIPT" \
    --data "$REPO_ROOT/training-scripts/healthcare-dataset-stroke-data.parquet" \
    --target stroke \
    --test-size 0.2 \
    --save-splits || { echo "Training failed for healthcare-dataset-stroke-data"; exit 1; }

  echo "All trainings completed. Splits are saved under experiment-results/splits/<dataset>/"
}

main() {
  if [[ $# -lt 1 ]]; then
    usage; exit 1
  fi
  local mode="$1"; shift || true
  case "$mode" in
    serve) serve_mode "$@" ;;
    test)  test_mode  "$@" ;;
    train) train_mode "$@" ;;
    -h|--help) usage ;;
    *) echo "Unknown mode: $mode" >&2; usage; exit 1 ;;
  esac
}

main "$@"
