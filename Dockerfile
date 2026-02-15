FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies + curl for healthcheck
RUN apt-get update && \
    apt-get install -y gcc curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application and ML model
COPY app ./app
COPY ml ./ml
COPY features ./features

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
