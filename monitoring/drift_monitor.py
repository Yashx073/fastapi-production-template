from pathlib import Path
import json
import sqlite3
import argparse
import time

import pandas as pd
from prometheus_client import Gauge, start_http_server
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "credit_card_fraud_v1.parquet"
CURRENT_PATH = PROJECT_ROOT / "monitoring" / "recent_predictions.csv"
REPORT_PATH = PROJECT_ROOT / "monitoring" / "drift_reports" / "drift_report.html"
PRIMARY_DB_PATH = PROJECT_ROOT / "artifacts" / "monitoring.db"
LEGACY_DB_PATH = PROJECT_ROOT / "monitoring" / "predictions.db"

DRIFT_SCORE_GAUGE = Gauge("data_drift_score", "Overall data drift score")
DRIFT_DETECTED_GAUGE = Gauge("data_drift_detected", "Data drift detected flag (1=true, 0=false)")


def resolve_db_path() -> Path:
    if LEGACY_DB_PATH.exists():
        return LEGACY_DB_PATH
    return PRIMARY_DB_PATH


def build_current_csv_from_db(limit: int = 1000) -> bool:
    db_path = resolve_db_path()
    if not db_path.exists():
        return False

    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql(
            """
            SELECT features
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
    except Exception:
        conn.close()
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if df.empty:
        return False

    try:
        df["features"] = df["features"].apply(json.loads)
    except Exception:
        return False

    current_data = pd.json_normalize(df["features"])
    CURRENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    current_data.to_csv(CURRENT_PATH, index=False)
    return True


def validate_paths() -> None:
    missing_paths = []

    if not REFERENCE_PATH.exists():
        missing_paths.append(str(REFERENCE_PATH))

    if not CURRENT_PATH.exists():
        if not build_current_csv_from_db():
            missing_paths.append(str(CURRENT_PATH))

    if missing_paths:
        missing = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing required path(s):\n{missing}")


def compute_and_save_drift() -> tuple[float, bool]:
    validate_paths()

    reference_data = pd.read_parquet(REFERENCE_PATH)
    current_data = pd.read_csv(CURRENT_PATH)

    common_columns = [column for column in reference_data.columns if column in current_data.columns]
    if not common_columns:
        raise ValueError("No common columns found between reference and current datasets")

    reference_data = reference_data[common_columns]
    current_data = current_data[common_columns]

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(REPORT_PATH))

        result = report.as_dict()
        dataset_drift = result["metrics"][0]["result"].get("dataset_drift", False)
        drift_share = float(result["metrics"][0]["result"].get("drift_share", 0.0) or 0.0)
        if dataset_drift:
            print("⚠️ DATA DRIFT DETECTED")

        print(f"Drift report generated: {REPORT_PATH}")
        return drift_share, bool(dataset_drift)
    except Exception:
        pass

    # Fallback mode for environments where Evidently is incompatible (e.g., Python 3.14)
    numeric_columns = [
        column
        for column in common_columns
        if pd.api.types.is_numeric_dtype(reference_data[column]) and pd.api.types.is_numeric_dtype(current_data[column])
    ]

    if not numeric_columns:
        raise RuntimeError("No numeric common columns available for fallback drift detection")

    rows = []
    drift_detected = False
    for column in numeric_columns:
        ref = pd.to_numeric(reference_data[column], errors="coerce").dropna()
        cur = pd.to_numeric(current_data[column], errors="coerce").dropna()
        if len(ref) < 10 or len(cur) < 10:
            continue

        stat, p_value = ks_2samp(ref, cur)
        feature_drift = p_value < 0.05
        if feature_drift:
            drift_detected = True
        rows.append((column, float(stat), float(p_value), feature_drift))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html_rows = "\n".join(
        f"<tr><td>{name}</td><td>{ks:.4f}</td><td>{p:.6f}</td><td>{'YES' if drift else 'NO'}</td></tr>"
        for name, ks, p, drift in rows
    )
    html = f"""
    <html><body>
    <h2>Fallback Drift Report (KS test)</h2>
    <p>Generated because Evidently could not run in this Python environment.</p>
    <table border='1' cellpadding='6' cellspacing='0'>
      <tr><th>feature</th><th>ks_stat</th><th>p_value</th><th>drift_detected (p&lt;0.05)</th></tr>
      {html_rows}
    </table>
    </body></html>
    """
    REPORT_PATH.write_text(html, encoding="utf-8")
    drift_score = (sum(1 for _, _, _, detected in rows if detected) / len(rows)) if rows else 0.0
    if drift_detected:
        print("⚠️ DATA DRIFT DETECTED")
    print(f"Drift report generated: {REPORT_PATH}")
    return drift_score, drift_detected


def run_once() -> None:
    drift_score, drift_detected = compute_and_save_drift()
    DRIFT_SCORE_GAUGE.set(drift_score)
    DRIFT_DETECTED_GAUGE.set(1.0 if drift_detected else 0.0)


def monitor_loop(interval_seconds: int, metrics_port: int) -> None:
    start_http_server(metrics_port)
    while True:
        try:
            drift_score, drift_detected = compute_and_save_drift()
            DRIFT_SCORE_GAUGE.set(drift_score)
            DRIFT_DETECTED_GAUGE.set(1.0 if drift_detected else 0.0)
            if drift_score > 0.5:
                print("⚠️ DRIFT ALERT: data_drift_score > 0.5")
        except Exception as error:
            print(f"⚠️ Drift monitor iteration failed: {error}")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data drift monitoring")
    parser.add_argument("--loop", action="store_true", help="Run continuously and expose Prometheus metrics")
    parser.add_argument("--interval", type=int, default=60, help="Loop interval in seconds")
    parser.add_argument("--port", type=int, default=9000, help="Prometheus metrics port")
    args = parser.parse_args()

    if args.loop:
        monitor_loop(interval_seconds=args.interval, metrics_port=args.port)
    else:
        run_once()
