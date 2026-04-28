# 🔍 Fake News Detection — End-to-End MLOps Pipeline

> **An end-to-end Machine Learning Operations (MLOps) pipeline for fake news detection**  
> Integrating MLflow, Docker, GitHub Actions, Prometheus, and Grafana for a production-ready NLP system.

[![CI/CD](https://github.com/tabidah-usmani/MLOPS-project/actions/workflows/mlops.yml/badge.svg)](https://github.com/tabidah-usmani/MLOPS-project/actions)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.11.1-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Track](https://img.shields.io/badge/Track-II%20Technical%20Research-green)

---



## 📌 Overview

This project demonstrates a **production-ready MLOps pipeline** for fake news detection.
The goal is not just to train a model — but to show how MLOps tools solve three
critical production problems:

| Problem | Solution | Tool Used |
|---|---|---|
| Poor reproducibility | Experiment tracking + model versioning | MLflow |
| Slow manual deployment | Containerization + CI/CD automation | Docker + GitHub Actions |
| Zero production visibility | Real-time metrics and dashboards | Prometheus + Grafana |

### ML Problem
Binary text classification — given a news article, predict whether it is **FAKE** or **REAL**.

### Model Performance
| Model | Accuracy | F1-Score | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 95.28% | 95.43% | 95.12% | 95.74% |
| **Random Forest** ✅ | **95.91%** | **96.05%** | **95.25%** | **96.87%** |

---

## 🔬 Research Question

> *"How does an automated MLOps pipeline improve reproducibility, deployment efficiency,
> and runtime performance of fake news detection systems compared to traditional manual workflows?"*

### Sub-Research Questions
- **RQ1:** How does MLflow experiment tracking improve model reproducibility?
- **RQ2:** What is the impact of Docker containerization on deployment consistency and latency?
- **RQ3:** How does CI/CD automation reduce deployment time vs manual processes?
- **RQ4:** Can Prometheus and Grafana effectively track runtime performance in production?

### Hypotheses — All Confirmed ✅
| Hypothesis | Claim | Result |
|---|---|---|
| H1 | Perfect reproducibility via MLflow | Variance = 0.000 across 3 runs ✅ |
| H2 | Docker reduces deployment time | 15 sec vs 15 min manual ✅ |
| H3 | CI/CD completes under 5 minutes | 78 seconds ✅ |
| H4 | Prometheus captures real-time metrics | 56 predictions monitored live ✅ |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    DATA LAYER                        │
│         WELFake Dataset (72,134 articles)            │
│         Text Preprocessing + TF-IDF (10,000 feat)   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              EXPERIMENT TRACKING                     │
│    MLflow — logs params, metrics, model versions     │
│    Model Registry — FakeNewsDetector v1, v2          │
│                   Port: 5001                         │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                MODEL SERVING                         │
│      Flask REST API — POST /predict                  │
│      Returns: label + confidence + latency           │
│                   Port: 5000                         │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              CONTAINERIZATION                        │
│         Docker + docker-compose                      │
│    4 containers on shared mlops-network              │
│    API | MLflow | Prometheus | Grafana               │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  CI/CD                               │
│           GitHub Actions — 3 Jobs                    │
│    Run Tests → Build Docker → Code Quality           │
│         Triggers on every push to main               │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               MONITORING                             │
│    Prometheus (Port 9090) — scrapes every 15s        │
│    Grafana (Port 3000) — live dashboards             │
│    Tracks: latency, predictions, request counts      │
└─────────────────────────────────────────────────────┘
```

### Docker Container Layout

| Container | Image | Port | Purpose |
|---|---|---|---|
| fakenews-api | fakenews-api:latest | 5000 | Flask REST API + model serving |
| fakenews-mlflow | python:3.11-slim | 5001 | MLflow tracking server |
| fakenews-prometheus | prom/prometheus:latest | 9090 | Metrics scraping |
| fakenews-grafana | grafana/grafana:latest | 3000 | Visualization dashboards |

---


## ✅ Prerequisites

Install these before starting:

| Tool | Download | Verify |
|---|---|---|
| Python 3.11+ | [python.org](https://python.org) | `python --version` |
| Docker Desktop | [docker.com](https://docker.com/products/docker-desktop) | `docker --version` |
| Git | [git-scm.com](https://git-scm.com) | `git --version` |

> ⚠️ Make sure **Docker Desktop is open and running** before any Docker commands.

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/tabidah-usmani/MLOPS-project.git
cd MLOPS-project

# 2. Setup environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 3. Add dataset (download WELFake_Dataset.csv from Kaggle → place in dataset/)

# 4. Train model
python src/preprocess.py
python src/train.py

# 5. Build and run everything
docker build -t fakenews-api -f DockerFile .
docker compose -f docker/docker-compose.yml --project-directory . up -d
```

Open in browser:

| Service | URL | Credentials |
|---|---|---|
| Flask API | http://localhost:5000 | — |
| MLflow UI | http://localhost:5001 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin123 |

---

## 📖 Step-by-Step Setup

### Step 1 — Clone and Install

```bash
git clone https://github.com/tabidah-usmani/MLOPS-project.git
cd MLOPS-project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Step 2 — Download the Dataset

1. Go to [kaggle.com/datasets/saurabhshahane/fake-news-classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
2. Download `WELFake_Dataset.csv`
3. Place it in the `dataset/` folder

### Step 3 — Preprocess and Train

```bash
python src/preprocess.py
python src/train.py
```

Expected output:
```
=== Logistic Regression ===
Accuracy: 0.9528 | F1: 0.9543 | Precision: 0.9512 | Recall: 0.9574

=== Random Forest ===
Accuracy: 0.9591 | F1: 0.9605 | Precision: 0.9525 | Recall: 0.9687
Model saved to models/model.pkl
```

### Step 4 — View MLflow UI (optional before Docker)

```bash
mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Open `http://127.0.0.1:5001`

### Step 5 — Build Docker Image

```bash
docker build -t fakenews-api -f DockerFile .
```

### Step 6 — Start All Services

```bash
docker compose -f docker/docker-compose.yml --project-directory . up -d
```

---

## 🐳 Running with Docker

### Two Terminal Method

```
Terminal 1 (keep running)                    Terminal 2 (commands)
─────────────────────────                    ──────────────────────
docker compose ... up                        pytest tests/ -v
                                             Invoke-WebRequest ...
```

### Background Mode (recommended)

```bash
# Start silently in background
docker compose -f docker/docker-compose.yml --project-directory . up -d

# Check status
docker ps

# View logs
docker compose -f docker/docker-compose.yml --project-directory . logs -f

# Stop everything
docker compose -f docker/docker-compose.yml --project-directory . down
```

---

## 🌐 API Usage

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/predict` | Predict FAKE or REAL |
| GET | `/metrics` | Prometheus metrics |

### Health Check

```powershell
Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing
```

Response:
```json
{"status": "healthy", "model": "models/model.pkl"}
```

### Make a Prediction

```powershell
Invoke-WebRequest -Uri "http://localhost:5000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text": "Your full news article text goes here"}' `
  -UseBasicParsing
```

Response:
```json
{
  "label": "FAKE",
  "confidence": 0.9821,
  "latency_ms": 35.51,
  "text_preview": "Your full news article text goes here"
}
```

> ⚠️ For best results send full article text (100+ words). The model was trained on full articles and performs best with rich text input.

---

## 📊 MLflow Experiment Tracking

### View UI

```bash
mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### What Gets Tracked Automatically

| Category | Values |
|---|---|
| Parameters | model_type, test_size, max_features, C, n_estimators |
| Metrics | accuracy, f1_score, precision, recall |
| Artifacts | trained model pipeline |
| Registry | FakeNewsDetector v1 (LR), v2 (RF) |

### Reproducibility — Zero Variance Across 3 Runs

| Run | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Run 1 | 0.953 | 0.954 | 0.951 | 0.957 |
| Run 2 | 0.953 | 0.954 | 0.951 | 0.957 |
| Run 3 | 0.953 | 0.954 | 0.951 | 0.957 |
| **Variance** | **0.000** | **0.000** | **0.000** | **0.000** |

---

## 📈 Monitoring with Prometheus and Grafana

### Prometheus — http://localhost:9090

Useful queries:

```promql
# Total predictions by label
fake_news_predictions_total

# Average latency
rate(fake_news_request_latency_seconds_sum[5m]) /
rate(fake_news_request_latency_seconds_count[5m])

# Request rate
rate(fake_news_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(fake_news_request_latency_seconds_bucket[5m]))
```

Check targets: `http://localhost:9090/targets` → `fake-news-api` should show **UP**

### Grafana — http://localhost:3000

1. Login: `admin` / `admin123`
2. `Connections` → `Data Sources` → `Add` → `Prometheus`
3. URL: `http://prometheus:9090` → `Save & Test`
4. Create dashboard panels using the queries above

### Custom Metrics

| Metric | Type | Description |
|---|---|---|
| `fake_news_predictions_total` | Counter | Predictions by label (FAKE/REAL) |
| `fake_news_request_latency_seconds` | Histogram | Per-request latency |
| `fake_news_requests_total` | Counter | Requests by HTTP status |

---

## 🔄 CI/CD Pipeline

### How It Works

```
git push origin main
        │
        ▼
GitHub Actions triggered automatically
        │
   ┌────┴──────────┐
   │               │
   ▼               ▼
Job 1           Job 3
Run Tests       Code Quality
(pytest)        (flake8)
   │
   ▼ (if pass)
Job 2
Build Docker Image
        │
        ▼
   Complete ✅
   ~78 seconds
```

### View Runs

```
https://github.com/tabidah-usmani/MLOPS-project/actions
```

### Trigger Pipeline

```bash
echo "" >> README.md
git add . && git commit -m "Trigger CI" && git push origin main
```

### Deployment Time Comparison

| Method | Time | Manual Steps |
|---|---|---|
| Manual first setup | ~900 sec | 12+ |
| Manual subsequent | ~300 sec | 8+ |
| Docker compose up | ~15 sec | 1 |
| **CI/CD Pipeline** | **78 sec** | **0** |

> **92% reduction** in deployment time

---

## 🧪 Running Tests

```bash
venv\Scripts\activate
pytest tests/ -v
```

Expected:
```
tests/test_model.py::test_clean_text_basic           PASSED
tests/test_model.py::test_clean_text_removes_urls    PASSED
tests/test_model.py::test_clean_text_handles_empty   PASSED
tests/test_model.py::test_clean_text_handles_none    PASSED
tests/test_model.py::test_clean_text_removes_numbers PASSED
5 passed in 1.23s
```

---

## 📊 Experimental Results Summary

### Model Performance

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 95.28% | 95.43% | 95.12% | 95.74% |
| **Random Forest** ✅ | **95.91%** | **96.05%** | **95.25%** | **96.87%** |

### API Latency (N=50 requests)

| Metric | Value |
|---|---|
| Average | 123.06 ms |
| Minimum | 75.20 ms |
| Cold start | 677.84 ms |
| Steady state | 85–100 ms |
| Monitored total | 56 predictions |

---

## 📦 Dataset

**WELFake Dataset** — 72,134 labeled news articles

| Split | Samples | % |
|---|---|---|
| Training | 57,616 | 80% |
| Test | 14,405 | 20% |
| Real (label=1) | 36,201 | 50.2% |
| Fake (label=0) | 35,820 | 49.8% |

**Download:** [Kaggle — WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

> ⚠️ Not included in repo (50MB). Download and place at `dataset/WELFake_Dataset.csv`

---


---

*Built with MLflow · Docker · GitHub Actions · Prometheus · Grafana*
