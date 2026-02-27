# Monitoring Module (Step 6.1 - Foundation)

Production-grade prediction logging and latency tracking.

## Architecture

```
API Request
  ↓
[predict endpoint] → model.predict() → compute latency
  ↓
[non-blocking] → log_prediction() (background thread)
  ├→ SQLite monitoring.db (primary)
  └→ Postgres audit (secondary, optional)
  ↓
[immediate] → response (user sees result immediately)
```

## Key Design Points

✅ **Never blocks inference** - logging happens in background thread  
✅ **Fail-safe logging** - errors in logging don't crash API  
✅ **Dual-store** - SQLite for speed, Postgres for audit trail  
✅ **Production-ready schema** - timestamp, model_version, features, probability, latency

## Usage

### Log a Prediction (in API)

```python
from app.services.prediction_logger import log_prediction

log_prediction(
    model_version="fraud-detection-model@Production",
    features={"amount": 100.5, "velocity": 2},
    prediction=1,
    probability=0.92,
    latency_ms=18.4,
)
# Returns immediately; logging happens in background
```

### Query Monitoring Data (via API)

```bash
# Get latency metrics (P50, P95, etc)
curl http://localhost:8000/monitoring/latency

# Get recent predictions
curl http://localhost:8000/monitoring/predictions?limit=50

# Get fraud vs non-fraud distribution (last 1h)
curl http://localhost:8000/monitoring/distribution
```

## Schema

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,           -- ISO format, for time-based analysis
    model_version TEXT NOT NULL,       -- For rollback & audit
    features TEXT NOT NULL,            -- JSON string, for drift detection
    prediction INTEGER NOT NULL,       -- 0 or 1 (fraud)
    probability REAL NOT NULL,         -- Confidence [0,1], for confidence drift
    latency_ms REAL NOT NULL,          -- SLA monitoring
    created_at DATETIME                -- DB audit timestamp
);
```

## What's Working Now (6.1)

- ✅ Every inference logs: timestamp, model_version, features, prediction, probability, latency
- ✅ Logging is non-blocking (background thread)
- ✅ SQLite store at `artifacts/monitoring.db`
- ✅ Monitoring endpoints for latency & distribution

## What's Next (6.2+)

- 6.2: Latency percentiles (P50, P95, P99)
- 6.3: Data drift detection (KS test or Evidently)
- 6.4: Alerts on drift/latency/errors
- 6.5: Grafana dashboard

---

**Interview talking point**: "I implemented non-blocking prediction logging to ensure inference latency is unaffected by monitoring overhead. This is production-grade design."
