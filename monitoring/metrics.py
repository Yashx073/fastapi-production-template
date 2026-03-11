"""Pure DataFrame-based metric helpers.

For database-backed metrics (no DataFrame in hand), use
monitoring.latency_metrics instead.
"""
import pandas as pd


def compute_latency_metrics(df: pd.DataFrame) -> tuple:
    """Return (p50_ms, p95_ms) latency from a predictions DataFrame."""
    p50 = float(df["latency_ms"].quantile(0.50))
    p95 = float(df["latency_ms"].quantile(0.95))
    return p50, p95


def compute_error_rate(df: pd.DataFrame) -> float:
    """Return the fraction of rows where status == 'error'."""
    errors = df["status"] == "error"
    return float(errors.mean())