import os
import random
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from locust import HttpUser, task, between, events


def load_rows_from_csv(csv_path: Path) -> List[Dict]:
    if not csv_path.exists():
        print(f"[locustfile] WARNING: Test file not found: {csv_path}")
        return []
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] <= 1:
            print("[locustfile] WARNING: X_test.csv must have at least 2 columns (first is id).")
            return []
        # Ignore the first column (id)
        feature_cols = df.columns[1:]
        rows = df[feature_cols].to_dict(orient="records")
        print(f"[locustfile] Loaded {len(rows)} rows and {len(feature_cols)} features from {csv_path}")
        return rows
    except Exception as e:
        print(f"[locustfile] ERROR: Failed to read {csv_path}: {e}")
        return []


# Resolve test files directory relative to this file
BASE_DIR = Path(__file__).resolve().parent

# Load optional config.json
import json
CONFIG_PATH = BASE_DIR / "config.json"
CONFIG = {}
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r") as f:
            CONFIG = json.load(f)
        print(f"[locustfile] Loaded config: {CONFIG_PATH}")
    except Exception as e:
        print(f"[locustfile] WARNING: Failed to read config.json: {e}")

# Test files
default_x_path = Path("test_files") / "X_test.csv"
config_x_path = (
    CONFIG.get("test_files", {}).get("x_test_path") if isinstance(CONFIG.get("test_files", {}), dict) else None
)
if config_x_path:
    p = Path(config_x_path)
    X_TEST_PATH = p if p.is_absolute() else (BASE_DIR / p)
else:
    X_TEST_PATH = BASE_DIR / default_x_path

# Preload rows once at import time
X_ROWS: List[Dict] = load_rows_from_csv(X_TEST_PATH)

# Prediction method: env var overrides config
_config_pred_method = CONFIG.get("pred_method") if isinstance(CONFIG, dict) else None
PRED_METHOD: Optional[str] = os.getenv("PRED_METHOD") or _config_pred_method  # e.g., "predict_proba"


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    if not X_ROWS:
        print("[locustfile] WARNING: No rows loaded from X_test.csv. Load test will run but no /invocation calls will be made.")


class InferenceUser(HttpUser):
    # Host can come from CLI (--host), env HOST, or config.json (lowest precedence here)
    host = os.getenv("HOST") or (CONFIG.get("host") if isinstance(CONFIG, dict) else None)
    # Wait time from config, default to previous aggressive values
    _wt = CONFIG.get("wait_time", {}) if isinstance(CONFIG, dict) else {}
    _wt_min = _wt.get("min_seconds", 0.01) if isinstance(_wt, dict) else 0.01
    _wt_max = _wt.get("max_seconds", 0.1) if isinstance(_wt, dict) else 0.1
    wait_time = between(_wt_min, _wt_max)

    @task(1)
    def invoke_single(self):
        if not X_ROWS:
            return
        row = random.choice(X_ROWS)
        # Send as a single JSON object (one row)
        path = "/invocation"
        if PRED_METHOD:
            path += f"?method={PRED_METHOD}"
        self.client.post(path, json=row, name="/invocation", timeout=30)
