"""Thresholds and tunables for monitoring alerts (Step 6.4)."""

# Alert when p95 latency exceeds this value.
LATENCY_THRESHOLD_MS = 200

# Alert when the observed error rate exceeds this value.
ERROR_RATE_THRESHOLD = 0.05

# Alert when this fraction of tested features show drift.
DRIFTED_FEATURE_RATE_THRESHOLD = 0.20

# Minimum minutes before sending the same alert again.
ALERT_COOLDOWN_MINUTES = 15