run:
	uvicorn app.main:app --reload

docker-build:
	docker build -t ai-engineering-app .

docker-run:
	docker run -p 8000:8000 ai-engineering-app

test:
	pytest
