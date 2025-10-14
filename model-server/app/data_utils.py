from typing import Any
import pandas as pd


def to_dataframe(payload: Any) -> pd.DataFrame:
    """
    Convert various JSON payload shapes into a pandas DataFrame.

    Supported shapes:
    - {"instances": [ {feature: value, ...}, ... ]}
    - {"inputs": [ ... ]}
    - {"columns": [...], "data": [[...], [...]]}
    - [ {feature: value, ...}, ... ]
    - {feature: value, ...}
    - [[...], [...]] with optional {"columns": [...]} wrapping
    """
    data = payload
    if isinstance(data, dict):
        # sklearn-like table: {"columns": [...], "data": [[...]]}
        if set(data.keys()) >= {"columns", "data"}:
            columns = data["columns"]
            rows = data["data"]
            return pd.DataFrame(rows, columns=columns)
        # common wrapper keys
        if "instances" in data:
            data = data["instances"]
        elif "inputs" in data:
            data = data["inputs"]
        else:
            # assume it's a single row dict
            return pd.DataFrame([data])

    # Now data is list-like (list of dicts or list of lists)
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return pd.DataFrame(data)
        # list of lists — build a DataFrame without column names
        return pd.DataFrame(data)

    # Empty or unrecognized — return empty DataFrame to let caller validate
    return pd.DataFrame()
