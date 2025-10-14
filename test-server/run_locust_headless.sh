#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_locust_headless.sh [HOST] [USERS] [SPAWN_RATE] [DURATION] [LOGLEVEL]
#  HOST       - default http://localhost:8000
#  USERS      - total concurrent users (default 100)
#  SPAWN_RATE - users spawned per second (default 10)
#  DURATION   - total time to run, e.g., 1m, 5m, 1h (default 1m)
#  LOGLEVEL   - locust log level (default INFO). Examples: INFO, DEBUG, WARNING

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCUST_FILE="$SCRIPT_DIR/locustfile.py"

HOST="${1:-${HOST:-http://localhost:8000}}"
USERS="${2:-${USERS:-100}}"
SPAWN_RATE="${3:-${SPAWN_RATE:-10}}"
DURATION="${4:-${DURATION:-1m}}"
LOGLEVEL="${5:-${LOGLEVEL:-INFO}}"

if ! command -v locust >/dev/null 2>&1; then
  echo "ERROR: locust is not installed. Run: pip install -r ../requirements.txt" >&2
  exit 1
fi

locust \
  -f "$LOCUST_FILE" \
  --headless \
  --host "$HOST" \
  -u "$USERS" \
  -r "$SPAWN_RATE" \
  -t "$DURATION" \
  --stop-timeout 60 \
  --loglevel "$LOGLEVEL"
