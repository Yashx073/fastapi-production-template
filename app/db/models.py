from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone

Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key = True, index = True)
    timestamp = Column(DateTime(timezone=True), default = lambda: datetime.now(timezone.utc))
    model_version = Column(String)
    features = Column(JSON)
    prediction = Column(Integer)
    probability = Column(Float)
    latency_ms = Column(Float)
