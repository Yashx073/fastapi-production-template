run:
	uvicorn app.main:app --reload

run-local:
	docker compose up -d db mlflow
	DB_IP=$$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' fastapi-production-template-db-1) && \
	DATABASE_URL="postgresql://fraud:fraud@$$DB_IP:5432/frauddb" MLFLOW_TRACKING_URI="http://localhost:5000" prime-run uvicorn app.main:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t ai-engineering-app .

docker-run:
	docker run -p 8000:8000 ai-engineering-app

test:
	pytest
