"""SQLite-based prediction store for monitoring and drift detection."""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

MONITORING_DB = Path(__file__).resolve().parent.parent / "artifacts" / "monitoring.db"
MONITORING_DB.parent.mkdir(parents=True, exist_ok=True)

# SQL schema for predictions table
PREDICTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model_version TEXT NOT NULL,
    features TEXT NOT NULL,
    prediction INTEGER NOT NULL,
    probability REAL NOT NULL,
    latency_ms REAL NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_model_version ON predictions(model_version);
"""


class PredictionStore:
    """Thread-safe SQLite store for prediction logs."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or MONITORING_DB
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create schema if it doesn't exist."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            for statement in PREDICTIONS_SCHEMA.split(";"):
                if statement.strip():
                    conn.execute(statement)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️  Failed to initialize monitoring schema: {e}")
            raise
    
    def log_prediction(
        self,
        model_version: str,
        features: Dict[str, Any],
        prediction: int,
        probability: float,
        latency_ms: float,
        timestamp: Optional[str] = None,
    ) -> bool:
        """
        Log a single prediction (non-blocking best-effort).
        
        Args:
            model_version: Version/stage (e.g., "fraud-detection-model@Production")
            features: Input features dict
            prediction: Binary prediction (0 or 1)
            probability: Confidence score [0,1]
            latency_ms: Inference latency ms
            timestamp: ISO format timestamp (auto-generated if None)
        
        Returns:
            bool: True if logged successfully, False on error (non-fatal)
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat() + "Z"
        
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.execute(
                """
                INSERT INTO predictions 
                (timestamp, model_version, features, prediction, probability, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    model_version,
                    json.dumps(features, default=str),
                    int(prediction),
                    float(probability),
                    float(latency_ms),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"⚠️  Prediction logging failed (non-fatal): {e}")
            return False
    
    def get_recent_predictions(self, limit: int = 100) -> list:
        """Retrieve recent predictions for analysis."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            print(f"⚠️  Failed to fetch predictions: {e}")
            return []
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Compute P50, P95 latency from recent predictions."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as count,
                    AVG(latency_ms) as mean,
                    MIN(latency_ms) as min,
                    MAX(latency_ms) as max
                FROM predictions
                WHERE timestamp > datetime('now', '-1 hour')
                """
            )
            row = cursor.fetchone()
            conn.close()

            if row is None:
                return {
                    "count": 0,
                    "mean_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                }

            result = dict(row)
            
            return {
                "count": int(result.get("count") or 0),
                "mean_ms": float(result.get("mean") or 0.0),
                "min_ms": float(result.get("min") or 0.0),
                "max_ms": float(result.get("max") or 0.0),
            }
        except Exception as e:
            print(f"⚠️  Failed to compute latency stats: {e}")
            return {}
    
    def get_prediction_distribution(self) -> Dict[str, int]:
        """Get fraud=0 vs fraud=1 counts from recent predictions."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute(
                """
                SELECT 
                    prediction,
                    COUNT(*) as count
                FROM predictions
                WHERE timestamp > datetime('now', '-1 hour')
                GROUP BY prediction
                """
            )
            result = {str(row[0]): row[1] for row in cursor.fetchall()}
            conn.close()
            return result
        except Exception as e:
            print(f"⚠️  Failed to compute prediction distribution: {e}")
            return {}


# Global store instance
_store = None


def get_store() -> PredictionStore:
    """Get or create global prediction store."""
    global _store
    if _store is None:
        _store = PredictionStore()
    return _store
