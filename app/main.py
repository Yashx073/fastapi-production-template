from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import mlflow
import mlflow.sklearn
import os

from app.db.models import Base
from app.db.session import engine
from app.services.prediction_logger import log_prediction

mlflow.set_tracking_uri("http://mlflow:5000")
MODEL_URI = "models:/fraud-detection-model/Production"


model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = mlflow.sklearn.load_model(MODEL_URI)
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(lifespan=lifespan)

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

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(transaction: Transaction, threshold: float = 0.8):
    df = pd.DataFrame([transaction.dict()])
    prob = model.predict_proba(df)[0][1]
    fraud = prob >= threshold

    log_prediction(
        features=df.to_dict(orient="records")[0],
        probability=float(prob),
        prediction=fraud
    )

    return {
        "fraud_probability": round(float(prob), 4),
        "fraud": fraud,
        "threshold": threshold
    }
