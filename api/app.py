import os
import time
import joblib
from collections import deque
from flask import Flask, request, jsonify
from flask_cors import CORS
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Use relative import instead
from src.preprocess import clean_text

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")
model = joblib.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

# ─────────────────────────────────────────
# Drift Detection Configuration
# ─────────────────────────────────────────
DRIFT_THRESHOLD = 0.70  # alert if >70% predictions are same label
WINDOW_SIZE = 50  # sliding window of last N predictions
MIN_WORD_COUNT = 30  # minimum words for reliable prediction

prediction_window = deque(maxlen=WINDOW_SIZE)
drift_detected = False


def check_prediction_drift(label):
    """
    Sliding window drift detector.
    Appends the latest label, then checks if FAKE or REAL
    ratio exceeds DRIFT_THRESHOLD over the last WINDOW_SIZE predictions.
    Returns True if drift is detected, False otherwise.
    """
    global drift_detected
    prediction_window.append(label)

    if len(prediction_window) == WINDOW_SIZE:
        fake_ratio = prediction_window.count("FAKE") / WINDOW_SIZE
        real_ratio = prediction_window.count("REAL") / WINDOW_SIZE

        if fake_ratio > DRIFT_THRESHOLD or real_ratio > DRIFT_THRESHOLD:
            drift_detected = True
            print(
                f"[DRIFT ALERT] FAKE={fake_ratio:.2%} REAL={real_ratio:.2%} "
                f"— threshold={DRIFT_THRESHOLD:.0%}"
            )
        else:
            drift_detected = False

    return drift_detected


# ─────────────────────────────────────────
# Prometheus Metrics
# ─────────────────────────────────────────
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

# Drift metrics
DRIFT_GAUGE = Gauge(
    'fake_news_drift_detected',
    'Whether prediction drift has been detected (1=drift, 0=normal)'
)
FAKE_RATIO_GAUGE = Gauge(
    'fake_news_fake_ratio',
    'Ratio of FAKE predictions in last 50 requests (0.0 to 1.0)'
)
REAL_RATIO_GAUGE = Gauge(
    'fake_news_real_ratio',
    'Ratio of REAL predictions in last 50 requests (0.0 to 1.0)'
)

# Short text warning metric
SHORT_TEXT_COUNT = Counter(
    'fake_news_short_text_total',
    'Total requests rejected due to insufficient word count'
)


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model": MODEL_PATH,
        "version": "1.2"
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        data = request.get_json()

        # ── Validation: missing field ──────────────
        if not data or 'text' not in data:
            REQUEST_COUNT.labels('POST', '/predict', '400').inc()
            return jsonify({"error": "Missing 'text' field in request body"}), 400

        raw_text = data['text']

        # ── Validation: empty text ─────────────────
        if not raw_text or not raw_text.strip():
            REQUEST_COUNT.labels('POST', '/predict', '400').inc()
            return jsonify({"error": "Text cannot be empty"}), 400

        # ── Validation: short text warning ─────────
        word_count = len(raw_text.split())
        if word_count < MIN_WORD_COUNT:
            SHORT_TEXT_COUNT.inc()
            REQUEST_COUNT.labels('POST', '/predict', '200').inc()
            return jsonify({
                "warning": "Text too short for reliable prediction",
                "word_count": word_count,
                "minimum_recommended": MIN_WORD_COUNT,
                "tip": f"Send at least {MIN_WORD_COUNT} words for accurate results"
            }), 200

        # ── Predict ────────────────────────────────
        cleaned = clean_text(raw_text)
        prediction = model.predict([cleaned])[0]
        probabilities = model.predict_proba([cleaned])[0]

        label = "REAL" if prediction == 1 else "FAKE"
        confidence = float(max(probabilities))

        # ── Prometheus: prediction metrics ─────────
        PREDICTION_COUNT.labels(label).inc()
        REQUEST_COUNT.labels('POST', '/predict', '200').inc()

        latency = time.time() - start_time
        REQUEST_LATENCY.labels('/predict').observe(latency)

        # ── Drift Detection ────────────────────────
        is_drift = check_prediction_drift(label)

        DRIFT_GAUGE.set(1 if is_drift else 0)

        if len(prediction_window) > 0:
            fake_ratio = prediction_window.count("FAKE") / len(prediction_window)
            real_ratio = prediction_window.count("REAL") / len(prediction_window)
            FAKE_RATIO_GAUGE.set(round(fake_ratio, 4))
            REAL_RATIO_GAUGE.set(round(real_ratio, 4))

        return jsonify({
            "label": label,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency * 1000, 2),
            "word_count": word_count,
            "text_preview": raw_text[:100] + "..." if len(raw_text) > 100 else raw_text,
            "drift_detected": is_drift
        }), 200

    except Exception as e:
        REQUEST_COUNT.labels('POST', '/predict', '500').inc()
        return jsonify({"error": str(e)}), 500


@app.route('/drift', methods=['GET'])
def drift_status():
    """Returns current drift detection status and window statistics."""
    window_list = list(prediction_window)
    total = len(window_list)

    if total == 0:
        return jsonify({
            "drift_detected": False,
            "fake_ratio": 0.0,
            "real_ratio": 0.0,
            "fake_count": 0,
            "real_count": 0,
            "window_size": 0,
            "window_capacity": WINDOW_SIZE,
            "threshold": DRIFT_THRESHOLD,
            "status": "Insufficient data — need at least 1 prediction"
        }), 200

    fake_ratio = round(window_list.count("FAKE") / total, 4)
    real_ratio = round(window_list.count("REAL") / total, 4)

    return jsonify({
        "drift_detected": drift_detected,
        "fake_ratio": fake_ratio,
        "real_ratio": real_ratio,
        "fake_count": window_list.count("FAKE"),
        "real_count": window_list.count("REAL"),
        "window_size": total,
        "window_capacity": WINDOW_SIZE,
        "threshold": DRIFT_THRESHOLD,
        "status": (
            "DRIFT ALERT — prediction distribution has shifted"
            if drift_detected else
            "Normal — distribution within expected range"
        )
    }), 200


@app.route('/reset', methods=['POST'])
def reset_drift():
    """
    Resets the drift detection window and all drift gauges.
    Useful for demos and testing — clears sliding window to start fresh.
    """
    global drift_detected
    prediction_window.clear()
    drift_detected = False
    DRIFT_GAUGE.set(0)
    FAKE_RATIO_GAUGE.set(0)
    REAL_RATIO_GAUGE.set(0)
    return jsonify({
        "status": "Drift window reset successfully",
        "window_size": 0,
        "drift_detected": False
    }), 200


@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "Fake News Detector API",
        "version": "1.2",
        "endpoints": {
            "POST /predict": "Send news text, get FAKE or REAL (min 30 words recommended)",
            "GET  /health": "Check API health",
            "GET  /metrics": "Prometheus metrics",
            "GET  /drift": "Check prediction drift status",
            "POST /reset": "Reset drift detection window (for testing)"
        }
    }), 200


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
