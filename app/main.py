from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()
 
MODEL_PATH = "ml/model.pkl"
model = joblib.load(MODEL_PATH)

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
    return {"status" : "running"}

@app.get("/health")
def health():
    return {"status" : "healthy"}

@app.post("/predict")
def predict(transaction: Transaction, threshold: float = 0.8):

    data = pd.DataFrame([{
        "amount": transaction.amount,
        "transaction_hour": transaction.transaction_hour,
        "merchant_category": transaction.merchant_category,
        "foreign_transaction": transaction.foreign_transaction,
        "location_mismatch": transaction.location_mismatch,
        "device_trust_score": transaction.device_trust_score,
        "velocity_last_24h": transaction.velocity_last_24h,
        "cardholder_age": transaction.cardholder_age,
    }], dtype = object)

    probability = model.predict_proba(data)[0][1]
    prediction = int(probability >= threshold)

    return {
        "fraud_probability" : round(float(probability), 4),
        "is_fraud" : prediction,
        "threshold_used" : threshold
    } 
    