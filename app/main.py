from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
import pandas as pd
import mlflow
import mlflow.pyfunc
import time

from app.db.models import Base
from app.db.session import engine
from app.services.prediction_logger import log_prediction
from app.schemas import FraudRequest
from monitoring.prediction_store import get_store

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_URI = "models:/fraud-detection-model/Production"


model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    # Load model from Production stage - fail fast if not available
    print(f"Loading model from Production stage: {MODEL_URI}")
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print("✅ Model loaded successfully from Production stage")
    except Exception as e:
        print(f"❌ FATAL: Failed to load model from Production stage: {e}")
        print("   Ensure a model version is promoted to Production in MLflow Model Registry")
        raise RuntimeError(f"Production model not available: {e}")
    
    # Initialize database
    db_max_retries = 5
    db_retry_delay = 2
    for attempt in range(db_max_retries):
        try:
            Base.metadata.create_all(bind=engine)
            print("✓ Database tables created successfully")
            break
        except Exception as e:
            if attempt < db_max_retries - 1:
                print(f"⚠ Database connection attempt {attempt + 1}/{db_max_retries} failed: {e}")
                print(f"  Retrying in {db_retry_delay}s...")
                time.sleep(db_retry_delay)
            else:
                print(f"✗ Database connection failed after {db_max_retries} attempts: {e}")
                raise
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(transaction: FraudRequest, threshold: float = 0.8):
    import time
    start_time = time.time()
    
    if model is None:
        return {"error": "Model not loaded"}
    
    df = pd.DataFrame([transaction.dict()])
    prob = model.predict(df)[0][1]
    fraud = prob >= threshold
    latency = (time.time() - start_time) * 1000  # Convert to milliseconds

    # Extract model version from metadata
    try:
        model_version = model.metadata.get_model_info().version
    except Exception:
        model_version = "Production"

    log_prediction(
        model_version=str(model_version),
        features=df.to_dict(orient="records")[0],
        probability=float(prob),
        prediction=fraud,
        latency_ms=latency
    )
    
    return {
        "fraud_probability": float(prob),
        "is_fraud": bool(fraud),
        "threshold": threshold,
    }


# ============================================================================
# MONITORING ENDPOINTS (6.1 Foundation)
# ============================================================================

@app.get("/monitoring/latency")
def get_latency_metrics():
    """Return P50, P95 latency and other SLA metrics."""
    store = get_store()
    stats = store.get_latency_stats()
    return {
        "latency_metrics": stats,
        "unit": "milliseconds",
    }


@app.get("/monitoring/predictions")
def get_recent_predictions(limit: int = 50):
    """Return recent predictions for debugging and analysis."""
    store = get_store()
    predictions = store.get_recent_predictions(limit=limit)
    return {
        "count": len(predictions),
        "predictions": predictions,
    }


@app.get("/monitoring/distribution")
def get_prediction_distribution():
    """Return fraud vs non-fraud prediction distribution (last 1h)."""
    store = get_store()
    dist = store.get_prediction_distribution()
    return {
        "distribution": dist,
        "window": "1_hour",
    }
