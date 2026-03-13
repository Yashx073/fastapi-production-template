# Fraud Detection MLOps Platform

A production-grade ML system for real-time credit card fraud detection — covering model training, experiment tracking, containerised deployment to Kubernetes, autoscaling, drift detection, and a full Prometheus/Grafana observability stack.

---

## Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Getting Started — Docker Compose (Local)](#getting-started--docker-compose-local)
- [Model Training](#model-training)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Monitoring Stack](#monitoring-stack)
- [Drift Detection](#drift-detection)
- [API Reference](#api-reference)
- [Tests](#tests)
- [Make Targets](#make-targets)
- [Project Structure](#project-structure)

---

## Architecture

```
                        ┌─────────────────────────────────────────────────────┐
                        │                  Kubernetes (Minikube)               │
                        │                                                      │
  HTTP Request ──────►  │  fraud-api  (FastAPI · 2 replicas · HPA)            │
                        │       │                                              │
                        │       ├── postgres-service  (prediction logging)     │
                        │       ├── mlflow-service    (model registry)         │
                        │       └── /metrics          (Prometheus scrape)      │
                        │                                                      │
                        │  drift-monitor ──loop──► /metrics (drift gauges)    │
                        │                                                      │
                        │  Prometheus ──► Grafana        (dashboards)          │
                        │       └──────► Alertmanager    (DataDriftDetected)   │
                        └─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn |
| ML | scikit-learn, pandas, numpy |
| Experiment Tracking | MLflow |
| Orchestration | Prefect |
| Database | PostgreSQL, SQLAlchemy, psycopg2 |
| Containers | Docker, Docker Compose |
| Kubernetes | Minikube, kubectl, Helm, HPA |
| Monitoring | Prometheus, Grafana, Alertmanager |
| Drift Detection | Evidently, prometheus-client |
| Testing | pytest, pytest-env |

---

## Features

### API & Serving
- `POST /predict` — real-time fraud scoring from a registered MLflow model
- `GET /health` — liveness check with model + DB readiness flags
- `GET /metrics` — Prometheus text format (HTTP rate, latency histograms)
- `GET /monitoring/metrics` — JSON snapshot of p50/p95/p99 latency, error rate, drift score
- Graceful degraded-mode startup — API stays up even if DB or model is temporarily unreachable

### ML Pipeline
- Logistic Regression and Random Forest on 10 k credit card fraud records
- Config-driven splits and hyperparameters (`configs/model_config.yaml`)
- Feature preprocessing pipeline (`features/preprocess.py`)
- MLflow experiment tracking — metrics, params, plots, and model artifacts per run
- MLflow Model Registry with `Staging → Production` promotion gate
- Prefect flow for reproducible scheduled/manual re-training (`orchestration/train_flow.py`)
- Data schema validation — column order, ranges, null checks, binary flags (`validation/schema.py`)

### Infrastructure
- Multi-stage Docker image with health check
- `docker-compose.yml` — full local stack: API + PostgreSQL + MLflow + drift-monitor + alert-worker
- Kubernetes manifests for every component (`k8s/`)
- HPA (Horizontal Pod Autoscaler) — CPU-based, verified scale-up and scale-down under `ab` load

### Monitoring & Observability
- `kube-prometheus-stack` (Helm) — Prometheus, Grafana, Alertmanager in one release
- ServiceMonitor scrapes API `/metrics` and drift-monitor `/metrics` automatically
- Custom `PrometheusRule` — `DataDriftDetected` alert fires when `data_drift_score > 0.5` for 2 min
- Alert worker with Slack webhook support, per-alert cooldown, and p95 latency / error-rate thresholds

---

## Prerequisites

| Tool | Minimum version |
|---|---|
| Docker | 24+ |
| docker compose | v2 |
| Python | 3.11+ |
| kubectl | 1.28+ |
| Minikube | 1.32+ |
| Helm | 3.14+ |

---

## Getting Started — Docker Compose (Local)

### 1. Clone & configure

```bash
git clone <your-repo-url>
cd fastapi-production-template

# Optional: set Slack webhook for alerts
cp .env.example .env        # create if needed
echo "SLACK_WEBHOOK_URL=https://hooks.slack.com/..." >> .env
```

### 2. Start all services

```bash
docker compose up -d --build
```

Services started:
| Service | URL |
|---|---|
| API | http://localhost:8000 |
| MLflow UI | http://localhost:5000 |
| PostgreSQL | localhost:5432 |
| Drift Monitor metrics | http://localhost:9000/metrics |

### 3. Check everything is healthy

```bash
docker compose ps
curl http://localhost:8000/health
```

---

## Model Training

### Via Docker Compose (recommended)

```bash
docker compose exec api python ml/train_pipeline.py \
  --data data/raw/credit_card_fraud_10k.csv \
  --config configs/model_config.yaml
```

### Via Prefect (orchestrated)

```bash
python orchestration/train_flow.py \
  --data data/raw/credit_card_fraud_10k.csv \
  --config configs/model_config.yaml
```

### Promote model to Production

1. Open MLflow UI at http://localhost:5000
2. Navigate to **Models → fraud-detection-model**
3. Select the best run version and click **Register → Promote to Production**

The API loads the `Production` stage model on startup.

---

## Kubernetes Deployment

### 1. Start Minikube

```bash
minikube start
```

### 2. Build image inside Minikube's Docker daemon

```bash
eval $(minikube docker-env)
docker build -t fraud-api:2.0 .
```

### 3. Install monitoring stack via Helm

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install monitoring prometheus-community/kube-prometheus-stack \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```

### 4. Deploy all manifests

```bash
kubectl apply -f k8s/
```

This creates:

| Manifest | What it deploys |
|---|---|
| `api-deployment.yaml` | FastAPI app (2 replicas, CPU requests/limits) |
| `api-service.yaml` | NodePort service on port 30007 |
| `postgres-deployment.yaml` | PostgreSQL + ClusterIP service |
| `mlflow-deployment.yaml` | MLflow server + NodePort service |
| `hpa.yaml` | HPA — min 2, max 5 replicas at 50% CPU |
| `drift-monitor.yaml` | Drift detection loop deployment |
| `drift-service.yaml` | ClusterIP for drift-monitor on port 9000 |
| `servicemonitor.yaml` | Scrape API `/metrics` |
| `drift-servicemonitor.yaml` | Scrape drift-monitor `/metrics` |
| `drift-alert.yaml` | PrometheusRule — `DataDriftDetected` |

### 5. Verify pods are running

```bash
kubectl get pods
```

Expected output — all pods `Running`:
```
fraud-api-xxx            1/1   Running
postgres-xxx             1/1   Running
mlflow-xxx               1/1   Running
drift-monitor-xxx        1/1   Running
prometheus-xxx           2/2   Running
grafana-xxx              3/3   Running
alertmanager-xxx         2/2   Running
```

### 6. Access services via port-forward

```bash
# API
kubectl port-forward svc/fraud-api-service 8000:80

# Grafana (admin / prom-operator)
kubectl port-forward svc/monitoring-grafana 3000:80

# Prometheus
kubectl port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090

# MLflow
kubectl port-forward svc/mlflow-service 5000:5000

# Drift monitor metrics
kubectl port-forward svc/drift-monitor 9000:9000
```

---

## Monitoring Stack

### Prometheus — verify targets

Open http://localhost:9090 → **Status → Targets**

You should see:
- `serviceMonitor/default/fraud-api/0` — UP
- `serviceMonitor/default/drift-monitor/0` — UP

### Grafana — dashboards

Open http://localhost:3000 (admin / prom-operator)

Useful PromQL queries:
```promql
# Request rate
rate(http_requests_total{job="fraud-api"}[5m])

# p95 latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="fraud-api"}[5m]))

# Drift score
data_drift_score

# Drift detected flag
data_drift_detected
```

### Alert rules

```bash
# List all alert rules and their states
curl -s http://localhost:9090/api/v1/rules?type=alert | \
  python3 -c "
import json, sys
for g in json.load(sys.stdin)['data']['groups']:
    for r in g['rules']:
        print(r['name'], r['state'])
"
```

---

## Drift Detection

The drift monitor runs as a dedicated Kubernetes deployment and:

1. Loads reference data from `monitoring/reference_stats.json`
2. Computes drift against recent predictions every 60 seconds using Evidently
3. Exports two Prometheus gauges on port 9000:
   - `data_drift_score` — overall drift score (0–1)
   - `data_drift_detected` — binary flag (1 = drift detected)
4. Prints an alert line to stdout when score exceeds threshold

The `DataDriftDetected` PrometheusRule fires if `data_drift_score > 0.5` persists for 2 minutes and routes through Alertmanager.

### Run drift monitor locally

```bash
python monitoring/drift_monitor.py \
  --loop \
  --interval 60 \
  --port 9000
```

---

## API Reference

### `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "db_ready": true
}
```

### `POST /predict`

**Request body:**

```json
{
  "transaction_id": "txn_001",
  "amount": 250.0,
  "transaction_hour": 14,
  "merchant_category": "retail",
  "foreign_transaction": 0,
  "location_mismatch": 0,
  "device_trust_score": 85,
  "velocity_last_24h": 3,
  "cardholder_age": 34
}
```

**Response:**

```json
{
  "transaction_id": "txn_001",
  "fraud_probability": 0.07,
  "fraud_prediction": 0,
  "latency_ms": 12.4
}
```

### `GET /monitoring/metrics`

```json
{
  "p50_latency_ms": 11.2,
  "p95_latency_ms": 45.8,
  "p99_latency_ms": 102.1,
  "mean_latency_ms": 14.3,
  "error_rate": 0.0,
  "drift_score": 0.0,
  "drift_detected": false
}
```

---

## Tests

```bash
# Run all tests
make test

# With verbose output
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=app
```

The test suite uses `SKIP_STARTUP_TASKS=1` (set automatically via `pytest.ini`) to bypass MLflow and DB connections during testing.

---

## Make Targets

| Target | Description |
|---|---|
| `make run` | Start API with hot-reload (expects local DB/MLflow) |
| `make run-local` | Start DB + MLflow containers, then run API with auto-wired IPs |
| `make docker-build` | Build the Docker image |
| `make docker-run` | Run the Docker image on port 8000 |
| `make test` | Run pytest |

---

## Project Structure

```text
app/
  main.py             FastAPI app, lifespan, endpoints
  schemas.py          Pydantic request/response models
  db/
    models.py         SQLAlchemy ORM models
    session.py        DB engine + session factory
  services/
    prediction_logger.py  Async prediction logging (Postgres + CSV)

features/
  preprocess.py       Feature engineering pipeline

ml/
  train_pipeline.py   Training script (LR + RF, MLflow logging)

orchestration/
  train_flow.py       Prefect training flow with retry logic

validation/
  schema.py           Data schema validation (columns, ranges, nulls)

monitoring/
  drift_monitor.py    Evidently drift detection loop + Prometheus export
  alert_worker.py     Background alert worker (latency, error rate, drift)
  alert_rules.py      Alert thresholds config
  latency_metrics.py  p50/p95/p99 latency computation
  drift_detector.py   Drift detection helper
  prediction_store.py SQLite prediction store
  reference_data.py   Reference dataset loader
  reference_stats.json Baseline feature statistics

k8s/
  api-deployment.yaml
  api-service.yaml
  postgres-deployment.yaml
  mlflow-deployment.yaml
  mlflow-service.yaml
  hpa.yaml
  drift-monitor.yaml
  drift-service.yaml
  servicemonitor.yaml
  drift-servicemonitor.yaml
  drift-alert.yaml

configs/
  model_config.yaml   Hyperparameters, split config, experiment name

data/
  raw/                Source dataset (credit_card_fraud_10k.csv)
  metadata/           Dataset metadata JSON

tests/
  test_main.py        API endpoint unit tests
```
