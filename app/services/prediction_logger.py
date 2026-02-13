from app.db.models import PredictionLog
from app.db.session import SessionLocal

def log_prediction(model_version, features, prediction, probability, latency):
    db = SessionLocal()
    record = PredictionLog(
        model_version = model_version,
        features = features,
        prediction = prediction,
        probability = probability,
        latency_ms = latency
    )
    db.add(record)
    db.commit()
    db.close()