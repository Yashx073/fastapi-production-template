from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd
import mlflow
import mlflow.sklearn
import time

from app.db.models import Base
from app.db.session import engine
from app.services.prediction_logger import log_prediction
from app.schemas import FraudRequest

mlflow.set_tracking_uri("http://mlflow:5000")
MODEL_URI = "models:/fraud-detection-model/8"


model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    # Try to load MLflow model; fall back to local storage if registry access fails
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting to load MLflow model from registry ({attempt + 1}/{max_retries})...")
            model = mlflow.sklearn.load_model(MODEL_URI)
            print("✓ MLflow model loaded from registry")
            break
        except Exception as e:
            print(f"⚠ Registry attempt {attempt + 1} failed: {str(e)[:100]}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print("✗ Registry loading failed, trying local artifact storage...")
                try:
                    # Fall back to loading from local artifacts
                    import joblib
                    from pathlib import Path
                    
                    # Find the latest model in mlruns
                    mlruns_models = Path("/mlruns/152946494954325001/models")
                    if mlruns_models.exists():
                        # Get the most recently modified model directory
                        model_dirs = sorted(mlruns_models.glob("m-*"), key=lambda p: p.stat().st_mtime, reverse=True)
                        if model_dirs:
                            latest_model_dir = model_dirs[0]
                            model_file = latest_model_dir / "artifacts" / "model.pkl"
                            if model_file.exists():
                                model = joblib.load(model_file)
                                print(f"✓ Model loaded from local storage: {model_file}")
                            else:
                                print(f"⚠ Model pickle file not found at {model_file}")
                                model = None
                        else:
                            print("⚠ No model directories found in mlruns")
                            model = None
                    else:
                        print(f"⚠ MLruns path does not exist: {mlruns_models}")
                        model = None
                except Exception as e2:
                    print(f"✗ Local loading also failed: {e2}")
                    model = None
    
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
    prob = model.predict_proba(df)[0][1]
    fraud = prob >= threshold
    latency = (time.time() - start_time) * 1000  # Convert to milliseconds

    log_prediction(
        model_version="8",
        features=df.to_dict(orient="records")[0],
        probability=float(prob),
        prediction=fraud,
        latency=latency
    )
    
    return {
        "fraud_probability": float(prob),
        "is_fraud": bool(fraud),
        "threshold": threshold,
    }
