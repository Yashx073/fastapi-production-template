"""Background alert worker for Step 6.4.

Checks latency, error rate, and drift periodically and sends alerts to Slack.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

try:
    from monitoring.alert_rules import (
        ALERT_COOLDOWN_MINUTES,
        DRIFTED_FEATURE_RATE_THRESHOLD,
        ERROR_RATE_THRESHOLD,
        LATENCY_THRESHOLD_MS,
    )
    from monitoring.drift_detector import detect_drift
    from monitoring.latency_metrics import compute_error_rate, compute_latency_metrics
except ModuleNotFoundError:
    # Support running as a script from inside `monitoring/`.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from monitoring.alert_rules import (
        ALERT_COOLDOWN_MINUTES,
        DRIFTED_FEATURE_RATE_THRESHOLD,
        ERROR_RATE_THRESHOLD,
        LATENCY_THRESHOLD_MS,
    )
    from monitoring.drift_detector import detect_drift
    from monitoring.latency_metrics import compute_error_rate, compute_latency_metrics

logger = logging.getLogger(__name__)

POLL_INTERVAL: int = int(os.getenv("ALERT_POLL_INTERVAL", "60"))
STATE_PATH = Path(__file__).resolve().parent / ".alert_state.json"


def send_alert(message: str) -> None:
    """Send *message* to Slack via the SLACK_WEBHOOK_URL env variable.

    Falls back to a console warning when the variable is not configured so
    that the worker is still functional during local development.
    """
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        logger.warning("SLACK_WEBHOOK_URL not set — printing alert instead: %s", message)
        print("[ALERT]", message)
        return

    try:
        resp = requests.post(webhook, json={"text": message}, timeout=10)
        resp.raise_for_status()
        logger.info("Slack alert sent: %s", message)
    except requests.RequestException as exc:
        logger.error("Failed to send Slack alert: %s", exc)


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}

    try:
        with STATE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        with STATE_PATH.open("w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception as exc:
        logger.error("Failed to persist alert state: %s", exc)


def _should_send(alert_key: str, now_ts: float, state: dict) -> bool:
    last_sent = float(state.get(alert_key, 0.0))
    cooldown_s = ALERT_COOLDOWN_MINUTES * 60
    return now_ts - last_sent >= cooldown_s


def _send_with_cooldown(alert_key: str, message: str, now_ts: float, state: dict) -> None:
    if not _should_send(alert_key, now_ts, state):
        logger.info("Skipping %s alert due to cooldown.", alert_key)
        return

    send_alert(message)
    state[alert_key] = now_ts


def check_alerts() -> None:
    """Evaluate all metric thresholds and fire alerts on any breaches."""
    now_ts = time.time()
    state = _load_state()

    metrics = compute_latency_metrics()
    error_rate = compute_error_rate()
    try:
        drift_results = detect_drift()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Drift check skipped due to error: %s", exc)
        drift_results = {}

    p95 = metrics.get("p95_latency_ms", 0.0)
    count = metrics.get("count", 0)

    if count == 0:
        logger.debug("No predictions logged yet — skipping alert check.")
        return

    if p95 > LATENCY_THRESHOLD_MS:
        _send_with_cooldown(
            "latency",
            f":warning: High p95 latency: {p95:.1f} ms "
            f"(threshold: {LATENCY_THRESHOLD_MS} ms, sample: {count} requests)",
            now_ts,
            state,
        )

    if error_rate > ERROR_RATE_THRESHOLD:
        _send_with_cooldown(
            "error_rate",
            f":red_circle: High error rate: {error_rate:.1%} "
            f"(threshold: {ERROR_RATE_THRESHOLD:.1%}, sample: {count} requests)",
            now_ts,
            state,
        )

    tested = len(drift_results)
    drifted = sum(1 for item in drift_results.values() if item.get("drift_detected"))
    drift_rate = (drifted / tested) if tested > 0 else 0.0

    if tested > 0 and drift_rate >= DRIFTED_FEATURE_RATE_THRESHOLD:
        _send_with_cooldown(
            "drift",
            (
                ":rotating_light: Data drift detected: "
                f"{drifted}/{tested} features drifted ({drift_rate:.1%}) "
                f"(threshold: {DRIFTED_FEATURE_RATE_THRESHOLD:.1%})"
            ),
            now_ts,
            state,
        )

    _save_state(state)


def run() -> None:
    """Run the alert worker loop indefinitely."""
    logger.info("Alert worker started (poll interval: %ss).", POLL_INTERVAL)
    while True:
        try:
            check_alerts()
        except Exception as exc:  # noqa: BLE001
            logger.error("Alert check failed: %s", exc)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()