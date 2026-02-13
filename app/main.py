from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn
from app.db.models import Base
from app.db.session import engine
from app.services.prediction_logger import log_prediction


app = FastAPI(title="Fraud Detection API")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

MODEL_NAME = "fraud-detection-model"

model = mlflow.sklearn.load_model(
    model_uri=f"models:/{MODEL_NAME}@production"
)

class Transaction(BaseModel):
    amount: float
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: int
    velocity_last_24h: int
    cardholder_age: int

@app.get("/")
def root():
    return {"status": "running"}

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind = engine)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(transaction: Transaction, threshold: float = 0.8):

    start_time = time.time()

    df = pd.DataFrame([{
        "amount": transaction.amount,
        "transaction_hour": transaction.transaction_hour,
        "merchant_category": transaction.merchant_category,
        "foreign_transaction": transaction.foreign_transaction,
        "location_mismatch": transaction.location_mismatch,
        "device_trust_score": transaction.device_trust_score,
        "velocity_last_24h": transaction.velocity_last_24h,
        "cardholder_age": transaction.cardholder_age,
    }])

    probability = float(model.predict_proba(df)[0][1])
    prediction = int(probability >= threshold)

    latency_ms = (time.time() - start_time) * 1000

    log_prediction(
        model_version = "production",
        features = df.to_dict(orient = "records")[0],
        prediction = prediction,
        probability = probability,
        latency = latency_ms
    )

    return {
        "fraud_probability": round(probability, 4),
        "is_fraud": prediction,
        "threshold_used": threshold
    }
