import os
import time
import joblib
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import clean_text

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")
model = joblib.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

REQUEST_COUNT = Counter(
    'fake_news_requests_total',
    'Total prediction requests',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'fake_news_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint']
)
PREDICTION_COUNT = Counter(
    'fake_news_predictions_total',
    'Total predictions by label',
    ['label']
)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model": MODEL_PATH
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            REQUEST_COUNT.labels('POST', '/predict', '400').inc()
            return jsonify({"error": "Missing 'text' field in request body"}), 400

        raw_text = data['text']
        if not raw_text or not raw_text.strip():
            REQUEST_COUNT.labels('POST', '/predict', '400').inc()
            return jsonify({"error": "Text cannot be empty"}), 400

        cleaned = clean_text(raw_text)
        prediction = model.predict([cleaned])[0]
        probabilities = model.predict_proba([cleaned])[0]

        label = "REAL" if prediction == 1 else "FAKE"
        confidence = float(max(probabilities))

        PREDICTION_COUNT.labels(label).inc()
        REQUEST_COUNT.labels('POST', '/predict', '200').inc()

        latency = time.time() - start_time
        REQUEST_LATENCY.labels('/predict').observe(latency)

        return jsonify({
            "label": label,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency * 1000, 2),
            "text_preview": raw_text[:100] + "..." if len(raw_text) > 100 else raw_text
        }), 200

    except Exception as e:
        REQUEST_COUNT.labels('POST', '/predict', '500').inc()
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "Fake News Detector API",
        "version": "1.0",
        "endpoints": {
            "POST /predict": "Send news text, get FAKE or REAL",
            "GET /health":   "Check API health",
            "GET /metrics":  "Prometheus metrics"
        }
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)