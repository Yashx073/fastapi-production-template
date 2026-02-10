<<<<<<< HEAD
# FastAPI Production Template

A production-ready FastAPI backend template with:

- Docker containerization
- Health checks
- Automated testing (pytest)
- Clean project structure
- Makefile automation
- WSL2 + GPU-ready development setup

## 🚀 Run Locally

```bash
make run
=======
# FastAPI Production Template

A production-ready FastAPI backend template built for scalable API development and containerized deployment.

This repository demonstrates backend engineering fundamentals including clean architecture, automated testing, Docker containerization, and Linux-native development using WSL2.

---

## 🚀 Features

- FastAPI REST API
- Health check endpoint (`/health`)
- Automated testing with pytest
- Dockerized application
- Makefile-based workflow
- Clean project structure
- Production-ready foundation

---

## 📂 Project Structure

```text
fastapi-production-template/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── tests/
│   └── test_main.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── Makefile
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 🛠 Tech Stack

- Python 3.10+
- FastAPI
- Pytest
- Docker
- WSL2 (Linux-based development)

---

## ▶ Run Locally

Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start development server:

```bash
make run
```

Visit:

```
http://localhost:8000
```

---

## 🐳 Build Docker Image

```bash
make docker-build
```

---

## ▶ Run Docker Container

```bash
make docker-run
```

Visit:

```
http://localhost:8000
```

---

## 🧪 Run Tests

```bash
make test
```

---

## 🎯 Purpose

This template serves as a foundation for:

- ML model serving
- Microservices architecture
- Kubernetes deployment
- MLOps pipelines
- Production API systems

---

## 📌 Future Extensions

- CI/CD integration
- Kubernetes manifests
- Structured logging
- Environment-based configuration
- Cloud deployment

---

## 👤 Author

Yash Mohadikar  
GitHub: https://github.com/Yashx073
>>>>>>> c4c05dd (Fix README formatting and structure)
