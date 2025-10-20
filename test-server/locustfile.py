import os
import random
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from locust import HttpUser, task, between, events


def load_rows_from_parquet(pq_path: Path) -> List[Dict]:
    if not pq_path.exists():
        print(f"[locustfile] WARNING: Test parquet not found: {pq_path}")
        return []
    try:
        df = pd.read_parquet(pq_path)
        if df.shape[1] < 1:
            print("[locustfile] WARNING: X_test.parquet has no feature columns.")
            return []
        rows = df.to_dict(orient="records")
        print(f"[locustfile] Loaded {len(rows)} rows and {df.shape[1]} features from {pq_path}")
        return rows
    except Exception as e:
        print(f"[locustfile] ERROR: Failed to read {pq_path}: {e}")
        return []


# Resolve base dir
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

# Dataset selection: ENV wins over config; default to credit_card_transactions
DATASET_NAME = os.getenv("DATASET_NAME") or CONFIG.get("dataset_name") or "credit_card_transactions"
X_TEST_PATH = BASE_DIR / "test_files" / "splits" / DATASET_NAME / "X_test.parquet"

# Preload rows once at import time
X_ROWS: List[Dict] = load_rows_from_parquet(X_TEST_PATH)

# Prediction method: env var overrides config
_config_pred_method = CONFIG.get("pred_method") if isinstance(CONFIG, dict) else None
PRED_METHOD: Optional[str] = os.getenv("PRED_METHOD") or _config_pred_method  # e.g., "predict_proba"


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    if not X_ROWS:
        print(f"[locustfile] WARNING: No rows loaded from {X_TEST_PATH}. Load test will run but no /invocation calls will be made.")


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
