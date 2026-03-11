import json
import sqlite3
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = PROJECT_ROOT / "monitoring" / "reference_stats.json"
PRIMARY_DB_PATH = PROJECT_ROOT / "artifacts" / "monitoring.db"
LEGACY_DB_PATH = PROJECT_ROOT / "monitoring" / "predictions.db"

DRIFT_THRESHOLD = 0.05


def resolve_db_path() -> Path:
    """Prefer legacy predictions.db if present, else use monitoring.db."""
    if LEGACY_DB_PATH.exists():
        return LEGACY_DB_PATH
    return PRIMARY_DB_PATH

def load_reference():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing reference stats at {REFERENCE_PATH}. "
            "Run: prime-run python monitoring/reference_data.py"
        )
    with open(REFERENCE_PATH, encoding="utf-8") as f:
        return json.load(f)

def load_live_data():
    db_path = resolve_db_path()
    conn = sqlite3.connect(db_path)

    df = pd.read_sql(
        """
        SELECT features
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT 500
        """,
        conn,
    )

    df["features"] = df["features"].apply(json.loads)
    features_df = pd.json_normalize(df["features"])

    return features_df

def detect_drift():
    """
    Determine drift with two-sample KS test (no Evidently):
    - Null hypothesis: reference and live feature values are from same distribution.
    - Drift rule: p_value < DRIFT_THRESHOLD.
    """
    reference = load_reference()
    live_df = load_live_data()

    drift_results = {}

    for col in live_df.columns:
        if col not in reference:
            continue

        ref_values = reference[col]["values"]
        live_values = live_df[col].dropna().values

        if len(live_values) < 30:
            continue

        stat, p_value = ks_2samp(ref_values, live_values)

        drift = p_value < DRIFT_THRESHOLD

        drift_results[col] = {
            "method": "ks_2samp",
            "threshold": DRIFT_THRESHOLD,
            "ks_stat": float(stat),
            "p_value": float(p_value),
            "sample_size": int(len(live_values)),
            "drift_detected": drift,
        }

    return drift_results

if __name__ == "__main__":
    results = detect_drift()

    for feature, result in results.items():
        if result["drift_detected"]:
            print(f"DRIFT detected in {feature}")
        else:   
            print(f"{feature}: OK")