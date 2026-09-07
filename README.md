# 🔍 Fake News Detection System — MLOps Pipeline

> An end-to-end production-ready MLOps pipeline for fake news detection with
> automated retraining, drift monitoring, and full observability.

[![CI/CD](https://github.com/tabidah-usmani/MLOPS-project/actions/workflows/mlops.yml/badge.svg)](https://github.com/tabidah-usmani/MLOPS-project/actions)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.11.1-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Track](https://img.shields.io/badge/Track-II%20Technical%20Research-green)

---

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [API Documentation](#api-documentation)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Drift Detection and Auto-Retraining](#drift-detection-and-auto-retraining)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Experimental Results](#-experimental-results)
- [Dataset](#-dataset)
- [License](#-license)

---

## ✨ Features

| Feature | Description |
|---|---|
| **ML Pipeline** | TF-IDF + classical ML classifiers (Logistic Regression, Random Forest, LinearSVC) |
| **REST API** | Flask-based inference service with built-in drift detection |
| **Monitoring** | Prometheus metrics + Grafana dashboards + Alertmanager |
| **Auto-Retraining** | Drift-triggered retraining with quality gates (F1 ≥ 0.95) |
| **Experiment Tracking** | MLflow for model versioning and run tracking |
| **Containerized** | Docker Compose for easy local development |
| **CI/CD** | GitHub Actions for linting, testing, building, and deployment |

---

## ✅ Prerequisites

- Docker & Docker Compose (v2.0+)
- Python 3.11 (for local development)
- Git (for version control)
- 8GB+ RAM recommended

---

## ⚡ Quick Start

```bash
# 1. Clone repository
git clone https://github.com/tabidah-usmani/MLOPS-project.git
cd MLOPS-project

# 2. Train initial model
python src/preprocess.py
python src/train.py

# 3. Launch all services
docker compose -f docker/docker-compose.yml --project-directory . up --build

# 4. Test API
curl http://localhost:5000/health
```

---

## 📖 Detailed Setup

### Step 1 — Environment Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

### Step 2 — Dataset Preparation

Place dataset at `dataset/cleaned.csv` with this format:

```
clean_text,label
"preprocessed news article text",0
"another preprocessed article",1
```

> Label: `0` = FAKE, `1` = REAL

### Step 3 — Model Training

```bash
# Train both models
python src/train.py

# View MLflow experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open MLflow at `http://localhost:5001`

### Step 4 — Start All Services

```bash
# Foreground (see logs)
docker compose -f docker/docker-compose.yml --project-directory . up

# Background (silent)
docker compose -f docker/docker-compose.yml --project-directory . up -d

# View logs
docker compose -f docker/docker-compose.yml --project-directory . logs -f

# Stop everything
docker compose -f docker/docker-compose.yml --project-directory . down
```

---

## 🌐 API Documentation

**Base URL:** `http://localhost:5000`

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | News classification |
| `GET` | `/drift` | Drift detection status |
| `POST` | `/reset` | Reset drift window |
| `GET` | `/metrics` | Prometheus metrics |

### Prediction Example

**Request:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking: Scientists discover revolutionary cure."}'
```

**Response (success):**

```json
{
  "label": "REAL",
  "confidence": 0.9876,
  "latency_ms": 45.23,
  "word_count": 18,
  "text_preview": "Breaking: Scientists discover...",
  "drift_detected": false
}
```

**Response (text too short):**

```json
{
  "warning": "Text too short for reliable prediction",
  "word_count": 5,
  "minimum_recommended": 30,
  "tip": "Send at least 30 words for accurate results"
}
```

> ⚠️ Send at least **30 words** for accurate predictions. The model was trained on full news articles.

---

## 📈 Monitoring and Alerting

### Access Dashboards

| Service | URL | Credentials |
|---|---|---|
| Prometheus | http://localhost:9090 | No auth |
| Grafana | http://localhost:3000 | admin / admin123 |
| Alertmanager | http://localhost:9093 | No auth |
| MLflow | http://localhost:5001 | No auth |

### Key Prometheus Queries (PromQL)

```promql
# Request rate
rate(fake_news_requests_total[5m])

# Prediction latency (95th percentile)
histogram_quantile(0.95, rate(fake_news_request_latency_seconds_bucket[5m]))

# Average latency
rate(fake_news_request_latency_seconds_sum[5m]) /
rate(fake_news_request_latency_seconds_count[5m]) * 1000

# Drift status
fake_news_drift_detected

# Label distribution
fake_news_fake_ratio
fake_news_real_ratio

# Total predictions by label
fake_news_predictions_total

# API health
up{job="fake-news-api"}
```

### Grafana Dashboard Panels

| Panel | Query | Description |
|---|---|---|
| Total Predictions | `fake_news_predictions_total` | FAKE vs REAL count |
| Avg Latency | rate sum/count * 1000 | Response time in ms |
| P95 Latency | histogram_quantile 0.95 | Worst case latency |
| Request Rate | rate requests total | Requests per second |
| Drift Alert | `fake_news_drift_detected` | 1=drift, 0=normal |
| FAKE Ratio | `fake_news_fake_ratio` | Distribution trend |

---

## 🔄 Drift Detection and Auto-Retraining

### How Drift Detection Works

| Parameter | Value |
|---|---|
| Window Size | Last 50 predictions |
| Threshold | 70% majority for any label |
| Drift Trigger | FAKE ratio > 70% OR REAL ratio > 70% |
| Alert Delay | Fires after 2 minutes of persistent drift |

### Auto-Retraining Flow

```
User sends predictions → API
         │
         ▼
Drift detector checks ratio
         │
         ▼
Prometheus scrapes metrics (every 10s)
         │
         ▼
Alert fires when drift persists (2 min)
         │
         ▼
Alertmanager sends webhook to retrain-service
         │
         ▼
Retrain service triggers retraining
         │
         ▼
New model trained + validated (F1 ≥ 0.95)
         │
         ▼
Model saved → API restarted
```


### Prometheus Alert Rules

| Alert | Condition | Severity |
|---|---|---|
| `PredictionDriftDetected` | FAKE ratio drops below threshold | Warning — FIRING |
| `PersistentPredictionDrift` | Drift persists over 10 min window | Critical — PENDING |
| `RealNewsDropoff` | REAL prediction volume drops | Warning — INACTIVE |

---

## 🧪 Testing

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_model.py -v

# Run with coverage report
pytest --cov=. --cov-report=html
```

**Expected output:**

```
tests/test_model.py::test_clean_text_basic           PASSED
tests/test_model.py::test_clean_text_removes_urls    PASSED
tests/test_model.py::test_clean_text_handles_empty   PASSED
tests/test_model.py::test_clean_text_handles_none    PASSED
tests/test_model.py::test_clean_text_removes_numbers PASSED
5 passed in 1.23s
```

---


## 📁 Project Structure

```
MLOPS-project/
│
├── .github/
│   └── workflows/
│       ├── mlops.yml              # CI/CD pipeline (test, build, lint)
│       └── retrain.yml            # Auto-retraining workflow
│
├── api/
│   └── app.py                     # Flask API with drift detection
│
├── src/
│   ├── preprocess.py              # Text cleaning pipeline
│   └── train.py                   # Model training and selection
│
├── tests/
│   └── test_model.py              # Unit tests
│
├── models/
│   └── model.pkl                  # Trained pipeline
│
├── dataset/
│   └── cleaned.csv                # Preprocessed training data
│
├── monitoring/
│   ├── prometheus.yml             # Prometheus scrape configuration
│   ├── alerts.yml                 # Alerting rules (drift detection)
│   ├── alertmanager.yml           # Alertmanager routing config
│   └── grafana/
│       └── datasources.yml        # Grafana Prometheus datasource
│
├── docker/
│   └── docker-compose.yml         # Multi-container orchestration
│
├── DockerFile                     # API container build definition
├── Dockerfile.retrain             # Retrain service build definition
├── retrain_webhook.py             # Retraining trigger service
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 📊 Experimental Results

### Model Performance

| Model | Accuracy | F1-Score | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 95.28% | 95.43% | 95.12% | 95.74% |
| **Random Forest** ✅ | **95.91%** | **96.05%** | **95.25%** | **96.87%** |

### Reproducibility (3 MLflow-tracked runs)

| Run | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Run 1 | 0.953 | 0.954 | 0.951 | 0.957 |
| Run 2 | 0.953 | 0.954 | 0.951 | 0.957 |
| Run 3 | 0.953 | 0.954 | 0.951 | 0.957 |
| **Variance** | **0.000** | **0.000** | **0.000** | **0.000** |

### Deployment Efficiency

| Method | Time | Manual Steps |
|---|---|---|
| Manual first setup | ~900 sec | 12+ |
| Manual subsequent | ~300 sec | 8+ |
| Docker compose up | ~15 sec | 1 |
| **CI/CD Pipeline** | **78 sec** | **0** |

> 92% reduction in deployment time via CI/CD automation

### API Latency (N=50 requests)

| Metric | Value |
|---|---|
| Average | 31–53 ms |
| Minimum | 12 ms |
| P95 | 48–49 ms |
| Cold start | 677 ms (one-time) |

---

## 📦 Dataset

**WELFake Dataset** — 72,134 labeled news articles

**Download:** [Kaggle — WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

> ⚠️ Dataset not included in repo. Download `WELFake_Dataset.csv` and place at `dataset/WELFake_Dataset.csv`

---

## Credits

- **Tabidah Usmani**
- **Amna Javaid**
- **Sara Zahid**
