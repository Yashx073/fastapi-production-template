from app.db.models import PredictionLog
from app.db.session import SessionLocal
from monitoring.prediction_store import get_store
import json
from threading import Thread


def _to_native(obj):
    """Convert numpy/scalar types to native Python types for JSON/DB insertion."""
    try:
        return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    except Exception:
        return obj


def _async_log(model_version, features, prediction, probability, latency_ms):
    """
    Background logging thread - never blocks inference.
    Logs to both SQLite (monitoring) and Postgres (audit trail).
    """
    # Log to SQLite monitoring store (primary)
    try:
        store = get_store()
        native_features = _to_native(features)
        native_prediction = int(bool(prediction))
        native_probability = float(probability) if probability is not None else None
        
        store.log_prediction(
            model_version=str(model_version),
            features=native_features,
            prediction=native_prediction,
            probability=native_probability,
            latency_ms=float(latency_ms),
        )
    except Exception as e:
        print(f"⚠️  SQLite logging failed (non-fatal): {e}")
    
    # Log to Postgres audit table (secondary, if DB available)
    try:
        db = SessionLocal()
        native_features = _to_native(features)
        native_prediction = int(bool(prediction))
        native_probability = float(probability) if probability is not None else None
        native_latency = float(latency_ms) if latency_ms is not None else None

        record = PredictionLog(
            model_version=str(model_version),
            features=native_features,
            prediction=native_prediction,
            probability=native_probability,
            latency_ms=native_latency,
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


def log_prediction(model_version, features, prediction, probability, latency_ms):
    """
    Non-blocking prediction logger.
    Spawns background thread to avoid blocking inference.
    
    ⚠️  Inference returns immediately; logging happens in background.
    """
    thread = Thread(
        target=_async_log,
        args=(model_version, features, prediction, probability, latency_ms),
        daemon=True,
    )
    thread.start()