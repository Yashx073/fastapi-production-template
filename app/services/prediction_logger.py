from app.db.models import PredictionLog
from app.db.session import SessionLocal
from monitoring.prediction_store import get_store
import json
import csv
from datetime import datetime
from pathlib import Path
from threading import Thread


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RECENT_PREDICTIONS_CSV = PROJECT_ROOT / "monitoring" / "recent_predictions.csv"


def _to_native(obj):
    """Convert numpy/scalar types to native Python types for JSON/DB insertion."""
    try:
        return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    except Exception:
        return obj


def _async_log(model_version, features, prediction, probability, latency_ms, status="success"):
    """
    Background logging thread - never blocks inference.
    Logs to both SQLite (monitoring) and Postgres (audit trail).
    """
    # Log to SQLite monitoring store (primary)
    try:
        store = get_store()
        native_features = _to_native(features)
        native_prediction = int(bool(prediction)) if prediction is not None else None
        native_probability = float(probability) if probability is not None else None
        
        store.log_prediction(
            model_version=str(model_version),
            features=native_features,
            prediction=native_prediction,
            probability=native_probability,
            latency_ms=float(latency_ms),
            status=str(status),
        )
    except Exception as e:
        print(f"⚠️  SQLite logging failed (non-fatal): {e}")

    # Log to CSV for drift monitoring input dataset
    try:
        native_features = _to_native(features)
        row = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **(native_features if isinstance(native_features, dict) else {}),
            "prediction": int(bool(prediction)) if prediction is not None else None,
            "probability": float(probability) if probability is not None else None,
            "status": str(status),
        }

        RECENT_PREDICTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
        file_exists = RECENT_PREDICTIONS_CSV.exists()
        fieldnames = list(row.keys())

        with open(RECENT_PREDICTIONS_CSV, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"⚠️  CSV logging failed (non-fatal): {e}")
    
    # Log to Postgres audit table (secondary, if DB available)
    try:
        db = SessionLocal()
        native_features = _to_native(features)
        native_prediction = int(bool(prediction)) if prediction is not None else None
        native_probability = float(probability) if probability is not None else None
        native_latency = float(latency_ms) if latency_ms is not None else None

        record = PredictionLog(
            model_version=str(model_version),
            features=native_features,
            prediction=native_prediction,
            probability=native_probability,
            latency_ms=native_latency,
            status=str(status),
        )
        db.add(record)
        db.commit()
        db.close()
    except Exception as e:
        print(f"⚠️  Postgres logging failed (non-fatal): {e}")
        try:
            db.close()
        except Exception:
            pass


def log_prediction(model_version, features, prediction, probability, latency_ms, status="success"):
    """
    Non-blocking prediction logger.
    Spawns background thread to avoid blocking inference.
    
    ⚠️  Inference returns immediately; logging happens in background.
    """
    thread = Thread(
        target=_async_log,
        args=(model_version, features, prediction, probability, latency_ms, status),
        daemon=True,
    )
    thread.start()