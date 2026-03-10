import sqlite3
from pathlib import Path

import pandas as pd

MONITORING_DB = Path(__file__).resolve().parent.parent / "artifacts" / "monitoring.db"


def _load_predictions() -> pd.DataFrame:
    if not MONITORING_DB.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(str(MONITORING_DB))
    try:
        return pd.read_sql_query("SELECT * FROM predictions", conn)
    finally:
        conn.close()


def compute_latency_metrics() -> dict:
    df = _load_predictions()
    if df.empty or "latency_ms" not in df.columns:
        return {
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "mean_latency_ms": 0.0,
            "count": 0,
        }

    latency = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()
    if latency.empty:
        return {
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "mean_latency_ms": 0.0,
            "count": 0,
        }

    return {
        "p50_latency_ms": float(latency.quantile(0.50)),
        "p95_latency_ms": float(latency.quantile(0.95)),
        "p99_latency_ms": float(latency.quantile(0.99)),
        "mean_latency_ms": float(latency.mean()),
        "count": int(latency.shape[0]),
    }


def compute_error_rate() -> float:
    df = _load_predictions()
    if df.empty:
        return 0.0

    total = len(df)
    if "status" in df.columns:
        errors = int((df["status"].astype(str).str.lower() == "error").sum())
        return errors / total if total > 0 else 0.0

    # SQLite schema has no explicit `status`; treat invalid latency as an error proxy.
    if "latency_ms" in df.columns:
        latency = pd.to_numeric(df["latency_ms"], errors="coerce")
        errors = int(latency.isna().sum() + (latency < 0).sum())
        return errors / total if total > 0 else 0.0

    return 0.0

if __name__ == "__main__":
    latency_metrics = compute_latency_metrics()
    error_rate = compute_error_rate()

    print("Latency Metrics:", latency_metrics)
    print("Error Rate:", error_rate)
