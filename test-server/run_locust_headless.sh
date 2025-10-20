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
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_DIR="$REPO_ROOT/experiment-results"
mkdir -p "$EXP_DIR"

HOST="${1:-${HOST:-http://localhost:8000}}"
USERS="${2:-${USERS:-100}}"
SPAWN_RATE="${3:-${SPAWN_RATE:-10}}"
DURATION="${4:-${DURATION:-1m}}"
LOGLEVEL="${5:-${LOGLEVEL:-INFO}}"

if ! command -v locust >/dev/null 2>&1; then
  echo "ERROR: locust is not installed. Run: pip install -r ../requirements.txt" >&2
  exit 1
fi

# Preflight: Resolve Parquet split path (dataset required)
DATASET_NAME="${DATASET_NAME:-credit_card_transactions}"
X_PATH="$SCRIPT_DIR/../experiment-results/splits/${DATASET_NAME}/X_test.parquet"

if [[ ! -f "$X_PATH" ]]; then
  echo "ERROR: Test parquet not found: $X_PATH" >&2
  echo "- Aborting load test. Ensure train/test splits exist under test-server/test_files/splits/<dataset>/X_test.parquet." >&2
  exit 1
fi

# Prepare results prefix
TS="$(date +%Y%m%d-%H%M%S)"
PREFIX="$EXP_DIR/${TS}_${DATASET_NAME}_${LOAD_MODEL:-model}"

STOP_CALLED=false
cleanup() {
  if command -v curl >/dev/null 2>&1; then
    if [[ "$STOP_CALLED" != true ]]; then
      echo "Stopping energy profiling at $HOST/energy-measurement/stop (cleanup)"
      curl -sS -X POST "$HOST/energy-measurement/stop" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT INT TERM

# Start energy measurement (best effort)
if command -v curl >/dev/null 2>&1; then
  echo "Starting energy profiling at $HOST/energy-measurement/start"
  curl -sS -X POST "$HOST/energy-measurement/start" -H 'Content-Type: application/json' -d "{\"model\": \"${LOAD_MODEL:-}\"}" >/dev/null || true
else
  echo "WARN: curl not found; skipping energy profiler start" >&2
fi

# Run Locust and capture CSV stats into experiment-results
locust \
  -f "$LOCUST_FILE" \
  --headless \
  --host "$HOST" \
  -u "$USERS" \
  -r "$SPAWN_RATE" \
  -t "$DURATION" \
  --stop-timeout 60 \
  --csv "$PREFIX" \
  --csv-full-history \
  --loglevel "$LOGLEVEL"

# Stop energy profiling and capture JSON
STOP_JSON="${PREFIX}_energy_stop.json"
if command -v curl >/dev/null 2>&1; then
  echo "Stopping energy profiling at $HOST/energy-measurement/stop"
  if curl -sS -X POST "$HOST/energy-measurement/stop" -H 'Accept: application/json' >"$STOP_JSON"; then
    STOP_CALLED=true
  else
    echo "WARN: Failed to fetch energy stop JSON" >&2
  fi
fi

# Summarize and print a combined table and append to run_table.csv
python3 - <<'PY' "$PREFIX" "$DURATION" "$DATASET_NAME" "${LOAD_MODEL:-}" "$TS" "${BATCH_GUID:-}"
import csv, json, os, sys
from pathlib import Path
from datetime import datetime

prefix = sys.argv[1]
duration_conf = sys.argv[2]
dataset_name = sys.argv[3] if len(sys.argv) > 3 else os.getenv('DATASET_NAME') or ''
model_key = sys.argv[4] if len(sys.argv) > 4 else os.getenv('LOAD_MODEL') or ''
ts_str = sys.argv[5] if len(sys.argv) > 5 else datetime.now().strftime('%Y%m%d-%H%M%S')
batch_guid = sys.argv[6] if len(sys.argv) > 6 else os.getenv('BATCH_GUID') or ''

locust_csv = Path(prefix + "_stats.csv")
locust_csv_total = Path(prefix + "_stats_total.csv")
stop_json = Path(prefix + "_energy_stop.json")

# run_table.csv lives in the experiment-results directory
exp_dir = Path(prefix).parent
run_table = exp_dir / "run_table.csv"


def to_float(val):
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def read_stats_row():
    if not locust_csv.exists():
        return None, []
    with open(locust_csv, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    agg = None
    for r in reversed(rows):
        name = (r.get('Name') or r.get('name') or r.get('Label') or '').strip().lower()
        if name in ('aggregated', 'total', 'all'):
            agg = r
            break
    if agg is None:
        for r in reversed(rows):
            name = (r.get('Name') or r.get('name') or '').strip().lower()
            typ = (r.get('Type') or r.get('type') or '').strip().lower()
            if name in ('',) and (typ in ('aggregated','') or True):
                agg = r
                break
    return agg, rows


def read_stats_total():
    if not locust_csv_total.exists():
        return None
    with open(locust_csv_total, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            return rows[-1]
    return None


def extract_totals():
    agg, rows = read_stats_row()
    if agg is None:
        agg = read_stats_total()
    req_keys = ['Requests', '# requests', 'requests', 'Request Count', 'request_count']
    avg_keys = ['Average Response Time', 'Average response time', 'Avg Response Time', 'Avg response time', 'avg_response_time']
    total_requests = None
    avg_resp_ms = None
    if agg:
        for k in req_keys:
            if k in agg and agg[k] != '':
                v = to_float(agg[k])
                if v is not None:
                    total_requests = v
                    break
        for k in avg_keys:
            if k in agg and agg[k] != '':
                v = to_float(agg[k])
                if v is not None:
                    avg_resp_ms = v
                    break
    if (total_requests is None or avg_resp_ms is None) and rows:
        total = 0.0
        weighted_sum = 0.0
        for r in rows:
            name = (r.get('Name') or r.get('name') or '').strip()
            lname = name.lower()
            if lname in ('aggregated','total','all') or name == '':
                continue
            req = None
            for k in req_keys:
                if k in r and r[k] != '':
                    req = to_float(r[k])
                    if req is not None:
                        break
            avg = None
            for k in avg_keys:
                if k in r and r[k] != '':
                    avg = to_float(r[k])
                    if avg is not None:
                        break
            if req is not None:
                total += req
                if avg is not None:
                    weighted_sum += req * avg
        if total_requests is None and total > 0:
            total_requests = total
        if avg_resp_ms is None and total > 0:
            avg_resp_ms = weighted_sum / total
    return total_requests, avg_resp_ms


# Defaults
total_requests, avg_resp_ms = extract_totals()

# Parse energy stop JSON
metrics = {}
energy_j = duration_s = mean_cpu = mean_mem_gb = mean_power = None
if stop_json.exists():
    try:
        data = json.loads(stop_json.read_text())
        m = data.get('metrics') or {}
        metrics = m
        def pick(*keys):
            for k in keys:
                if k in data and data[k] is not None:
                    return data[k]
                if k in m and m[k] is not None:
                    return m[k]
            return None
        energy_j = to_float(pick('energy_joules'))
        duration_s = to_float(pick('duration_seconds'))
        mean_cpu = to_float(pick('mean_cpu_usage'))
        mean_mem_gb = to_float(pick('mean_memory_gb'))
        mean_power = to_float(pick('mean_power_watts'))
        if mean_power is None and energy_j is not None and duration_s and duration_s > 0:
            mean_power = energy_j / duration_s
    except Exception:
        pass

# Simple ASCII table
rows = [
    ("Total requests", f"{int(total_requests):,}" if total_requests is not None else "-"),
    ("Mean latency (ms)", f"{avg_resp_ms:.2f}" if avg_resp_ms is not None else "-"),
    ("Test duration (config)", duration_conf),
    ("Energy (J)", f"{energy_j:.3f}" if isinstance(energy_j, (int, float)) else "-"),
    ("Mean power (W)", f"{mean_power:.3f}" if isinstance(mean_power, (int, float)) else "-"),
    ("Mean CPU usage (%)", f"{mean_cpu:.3f}" if isinstance(mean_cpu, (int, float)) else "-"),
    ("Mean memory usage (GB)", f"{mean_mem_gb:.3f}" if isinstance(mean_mem_gb, (int, float)) else "-"),
]

width_key = max(len(k) for k, _ in rows) + 2
width_val = max(len(v) for _, v in rows) + 2
sep = "+" + ("-" * width_key) + "+" + ("-" * width_val) + "+"
print("\n" + sep)
print("|" + " Metric".ljust(width_key) + "|" + " Value".ljust(width_val) + "|")
print(sep)
for k, v in rows:
    print("|" + f" {k}".ljust(width_key) + "|" + f" {v}".ljust(width_val) + "|")
print(sep)
print(f"Saved Locust CSV: {locust_csv}")
if stop_json.exists():
    print(f"Saved Energy JSON: {stop_json}")

# Append a row to experiment-results/run_table.csv
headers = [
    'timestamp', 'batch_guid', 'database', 'model',
    'total_requests', 'mean_latency_ms', 'test_duration_config',
    'energy_j', 'mean_power_w', 'mean_cpu_percent', 'mean_memory_gb'
]
row = {
    'timestamp': ts_str,
    'batch_guid': batch_guid,
    'database': dataset_name,
    'model': model_key,
    'total_requests': int(total_requests) if isinstance(total_requests, (int, float)) and total_requests is not None else '',
    'mean_latency_ms': float(f"{avg_resp_ms:.6f}") if isinstance(avg_resp_ms, (int, float)) and avg_resp_ms is not None else '',
    'test_duration_config': duration_conf,
    'energy_j': float(f"{energy_j:.6f}") if isinstance(energy_j, (int, float)) and energy_j is not None else '',
    'mean_power_w': float(f"{mean_power:.6f}") if isinstance(mean_power, (int, float)) and mean_power is not None else '',
    'mean_cpu_percent': float(f"{mean_cpu:.6f}") if isinstance(mean_cpu, (int, float)) and mean_cpu is not None else '',
    'mean_memory_gb': float(f"{mean_mem_gb:.6f}") if isinstance(mean_mem_gb, (int, float)) and mean_mem_gb is not None else '',
}

run_table_exists = run_table.exists()
with open(run_table, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    if not run_table_exists:
        writer.writeheader()
    writer.writerow(row)
print(f"Appended run summary to: {run_table}")
PY
