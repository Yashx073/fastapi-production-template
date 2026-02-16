from app.db.models import PredictionLog
from app.db.session import SessionLocal
import json


def _to_native(obj):
    """Convert numpy/scalar types to native Python types for JSON/DB insertion."""
    try:
        return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    except Exception:
        return obj


def log_prediction(model_version, features, prediction, probability, latency):
    db = SessionLocal()
    try:
        native_features = _to_native(features)
        native_prediction = int(bool(prediction))
        native_probability = float(probability) if probability is not None else None
        native_latency = float(latency) if latency is not None else None

        record = PredictionLog(
            model_version=str(model_version),
            features=native_features,
            prediction=native_prediction,
            probability=native_probability,
            latency_ms=native_latency,
        )
        db.add(record)
        db.commit()
    finally:
        db.close()