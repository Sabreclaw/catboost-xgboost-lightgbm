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
VENV_DIR="$MODEL_DIR/.venv"
TEST_DIR="$REPO_ROOT/test-server"

usage() {
  cat <<EOF
Usage:
  $0 serve [--host HOST] [--port PORT] [--model MODEL_KEY]
  $0 test  [HOST] [USERS] [SPAWN_RATE] [DURATION] [LOGLEVEL]

Examples:
  # Serve with prompts (defaults host=0.0.0.0, port=8000, model=catboost)
  bash $0 serve

  # Serve with some flags
  bash $0 serve --host 127.0.0.1 --port 8000 --model lgbm

  # Run load test (headless)
  bash $0 test http://localhost:8000 200 20 2m DEBUG
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
  local MODEL_KEY="catboost"
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
    if confirm "Create virtual environment at model-server/.venv?" default_y; then
      (cd "$MODEL_DIR" && python3 -m venv .venv)
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

  # Step 3: Install model-server requirements
  if confirm "Install model-server requirements (pip install -r model-server/requirements.txt)?" default_y; then
    "$PIPBIN" install -r "$MODEL_DIR/requirements.txt"
  else
    echo "Skipping dependency installation"
  fi

  # Step 4: Choose model key
  read -r -p "Select LOAD_MODEL (catboost/lgbm/xgboost) [${MODEL_KEY}]: " MODEL_IN || true
  if [[ -n "${MODEL_IN:-}" ]]; then
    MODEL_KEY="$MODEL_IN"
  fi

  # Step 5: Choose host/port
  read -r -p "Server host [${HOST}]: " HOST_IN || true
  if [[ -n "${HOST_IN:-}" ]]; then
    HOST="$HOST_IN"
  fi
  read -r -p "Server port [${PORT}]: " PORT_IN || true
  if [[ -n "${PORT_IN:-}" ]]; then
    PORT="$PORT_IN"
  fi

  # Step 6: Debug logging?
  DEBUG_FLAGS=""
  if confirm "Enable DEBUG logging?" default_n; then
    export LOG_LEVEL=DEBUG
    DEBUG_FLAGS="--log-level debug"
  fi

  echo "Starting server with: LOAD_MODEL=$MODEL_KEY host=$HOST port=$PORT"
  export LOAD_MODEL="$MODEL_KEY"
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
  if confirm "Install test-server requirements (pip install -r test-server/requirements.txt)?" default_y; then
    local PIPBIN="pip3"
    if command -v pip >/dev/null 2>&1; then PIPBIN="pip"; fi
    # If venv active, pip will install there
    $PIPBIN install -r "$TEST_DIR/requirements.txt"
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
  local SPAWN_RATE="${3:-${SPAWN_RATE:-10}}"
  local DURATION="${4:-${DURATION:-1m}}"
  local LOGLEVEL="${5:-${LOGLEVEL:-INFO}}"

  echo "Current settings:"
  echo "  HOST=$HOST"
  echo "  USERS=$USERS"
  echo "  SPAWN_RATE=$SPAWN_RATE"
  echo "  DURATION=$DURATION"
  echo "  LOGLEVEL=$LOGLEVEL"

  if confirm "Edit settings before running?" default_n; then
    read -r -p "HOST [$HOST]: " IN || true; [[ -n "${IN:-}" ]] && HOST="$IN"
    read -r -p "USERS [$USERS]: " IN || true; [[ -n "${IN:-}" ]] && USERS="$IN"
    read -r -p "SPAWN_RATE [$SPAWN_RATE]: " IN || true; [[ -n "${IN:-}" ]] && SPAWN_RATE="$IN"
    read -r -p "DURATION [$DURATION]: " IN || true; [[ -n "${IN:-}" ]] && DURATION="$IN"
    read -r -p "LOGLEVEL [$LOGLEVEL]: " IN || true; [[ -n "${IN:-}" ]] && LOGLEVEL="$IN"
  fi

  if ! [[ -f "$TEST_DIR/run_locust_headless.sh" ]]; then
    echo "ERROR: $TEST_DIR/run_locust_headless.sh not found." >&2
    exit 1
  fi

  if confirm "Run Locust now?" default_y; then
    (cd "$TEST_DIR" && bash ./run_locust_headless.sh "$HOST" "$USERS" "$SPAWN_RATE" "$DURATION" "$LOGLEVEL")
  else
    echo "Skipped running Locust."
  fi
}

main() {
  if [[ $# -lt 1 ]]; then
    usage; exit 1
  fi
  local mode="$1"; shift || true
  case "$mode" in
    serve) serve_mode "$@" ;;
    test)  test_mode  "$@" ;;
    -h|--help) usage ;;
    *) echo "Unknown mode: $mode" >&2; usage; exit 1 ;;
  esac
}

main "$@"
