# AI Engineering Fraud Detection

FastAPI service for fraud prediction with ML training and MLflow tracking.

## Features

- FastAPI REST API with /health and /predict
- ML training for Logistic Regression and Random Forest
- MLflow tracking for metrics and artifacts
- Optional Postgres logging for predictions
- Dockerized setup

## Services

- API: http://localhost:8000
- MLflow UI: http://localhost:5000
- Postgres: localhost:5432

## Quick Start

Create and start services:

```bash
docker-compose up -d --build
```

Train models and log to MLflow:

```bash
docker-compose exec -T api python ml/train.py
```

## Local Development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
make run
```

## Tests

```bash
make test
```

## Project Structure

```text
app/        FastAPI application
features/   Feature preprocessing
ml/         Training scripts
tests/      Tests
```
