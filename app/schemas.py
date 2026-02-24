from pydantic import BaseModel

class FraudRequest(BaseModel):
    amount: float
    transaction_hour: int
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: float
    velocity_last_24h: float
    cardholder_age: int
    merchant_category: str