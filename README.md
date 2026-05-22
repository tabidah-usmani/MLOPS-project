# Fake News Detection (MLOps Project)

This repository implements an **end-to-end MLOps pipeline** for fake news detection:

- **Training**: TF-IDF + classical ML classifiers, logged to **MLflow**
- **Serving**: **Flask** REST API with drift detection + Prometheus metrics
- **Monitoring**: **Prometheus + Grafana + Alertmanager**
- **Retraining**: GitHub Actions workflow triggered by drift alerts (and nightly)

---

## Project Structure

- `src/`
  - `preprocess.py` — text cleaning + dataset preprocessing
  - `train.py` — model training, MLflow logging, model selection, saving to `models/model.pkl`
- `api/`
  - `app.py` — Flask API
    - `POST /predict` performs cleaning → inference → drift check
    - `/metrics` exposes Prometheus metrics
    - `/drift` and `/reset` support drift monitoring
- `models/`
  - `model.pkl` — the saved scikit-learn pipeline used by the API
- `dataset/`
  - `cleaned.csv` — expected training dataset with columns `clean_text` and `label`
- `monitoring/`
  - Prometheus and Alertmanager configuration
  - Grafana datasource configuration
- `.github/workflows/`
  - `mlops.yml` — lint + tests + Docker build
  - `retrain.yml` — drift-triggered retraining + quality gate + redeploy build

---

## Data Format

Training expects a CSV with:

- `clean_text`: preprocessed text
- `label`: integer class (`0` = FAKE, `1` = REAL)

`src/preprocess.py` can generate `dataset/cleaned.csv` from a WELFake-style dataset.

---

## Model Training

Entry point: `src/train.py`

High-level steps:

1. Load `dataset/cleaned.csv`
2. Split into train/test (stratified)
3. Build pipeline:
   - `TfidfVectorizer` with word n-grams
   - classifier chosen from:
     - Logistic Regression
     - Random Forest
     - LinearSVC wrapped with `CalibratedClassifierCV` (to support `predict_proba`)
4. Train and evaluate (Accuracy, F1, Precision, Recall)
5. Select the best model by **F1**
6. Save best pipeline to `models/model.pkl`
7. Log runs/metrics to **MLflow**

---

## Steps to Run the Project (Recommended)

### 1) Preprocess + Train

```bash
python src/preprocess.py
python src/train.py
```

This produces `dataset/cleaned.csv` and then `models/model.pkl`.

### 2) Start API + Monitoring stack (Docker Compose)

```bash
docker compose up --build
```

---

## Verify the API

- Health check:
  - `http://localhost:5000/health`

- Prediction:
  - `POST http://localhost:5000/predict`

- Prometheus metrics:
  - `http://localhost:5000/metrics`

Example request:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Breaking: Scientists discover revolutionary cure."}'
```

---

## Docker (Local) - Ports

- API: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Alertmanager: `http://localhost:9093`

---

## Running Tests

```bash
pytest -v
```

API tests are skipped if `models/model.pkl` is missing.

---

## CI/CD

### `mlops.yml`
- runs lint (`flake8`)
- runs tests (`pytest tests/test_model.py -v`)
- builds Docker image for API

### `retrain.yml`
- can run from a drift-triggered webhook (Grafana) and also nightly
- downloads dataset via Kaggle
- preprocesses data
- trains all candidate models and selects best by F1
- enforces a **quality gate** (F1 ≥ 0.95)
- if passed, builds/pushes a retrained Docker image

---

## Notes

- The API uses `nltk` stopwords; `DockerFile` downloads required NLTK resources at build time.
- Drift detection is based on predicted labels only (no ground-truth stream).

