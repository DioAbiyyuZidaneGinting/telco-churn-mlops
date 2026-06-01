"""
prometheus_exporter.py
=======================
Flask API + Prometheus metrics exporter untuk model Telco Churn.

Endpoint:
  POST /predict  — Menerima JSON, mengembalikan prediksi
  GET  /metrics  — Prometheus scrape endpoint
  GET  /health   — Health check

Metrics (10+):
  total_requests, success_requests, failed_requests,
  prediction_latency, cpu_usage, memory_usage, disk_usage,
  response_time, prediction_class_count, error_rate
"""

import os
import time
import json
import logging
import threading
import warnings
import psutil
import mlflow.sklearn
import pandas as pd
from flask import Flask, request, jsonify, Response
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, REGISTRY,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# ─── CONFIG ─────────────────────────────────────────────────────────────────
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MODEL_URI    = os.getenv("MODEL_URI", "models:/telco-churn-rf-tuned/latest")
HOST         = os.getenv("HOST", "0.0.0.0")
PORT         = int(os.getenv("PORT", 5001))

# ─── PROMETHEUS METRICS ─────────────────────────────────────────────────────

# 1. Total requests
TOTAL_REQUESTS = Counter(
    "total_requests_total",
    "Total number of prediction requests received",
)

# 2. Successful requests
SUCCESS_REQUESTS = Counter(
    "success_requests_total",
    "Number of successful prediction requests",
)

# 3. Failed requests
FAILED_REQUESTS = Counter(
    "failed_requests_total",
    "Number of failed prediction requests",
)

# 4. Prediction latency
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latency of model prediction in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# 5. CPU usage
CPU_USAGE = Gauge(
    "cpu_usage_percent",
    "Current CPU usage percentage",
)

# 6. Memory usage
MEMORY_USAGE = Gauge(
    "memory_usage_percent",
    "Current memory usage percentage",
)

# 7. Disk usage
DISK_USAGE = Gauge(
    "disk_usage_percent",
    "Current disk usage percentage",
)

# 8. Response time (end-to-end)
RESPONSE_TIME = Histogram(
    "response_time_seconds",
    "End-to-end HTTP response time in seconds",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)

# 9. Prediction class count (label)
PREDICTION_CLASS_COUNT = Counter(
    "prediction_class_count_total",
    "Total predictions per class label",
    ["label"],
)

# 10. Error rate (gauge, updated per request)
ERROR_RATE = Gauge(
    "error_rate_percent",
    "Rolling error rate as percentage of total requests",
)

# Bonus metrics
MODEL_LOAD_TIME = Gauge("model_load_time_seconds", "Time taken to load the model")
ACTIVE_REQUESTS = Gauge("active_requests", "Number of currently active requests")
INPUT_BATCH_SIZE = Histogram(
    "input_batch_size",
    "Number of rows per prediction request",
    buckets=[1, 5, 10, 50, 100, 500],
)


# ─── SYSTEM METRICS UPDATER ──────────────────────────────────────────────────

def update_system_metrics():
    """Background thread: update system metrics every 5 seconds."""
    while True:
        CPU_USAGE.set(psutil.cpu_percent(interval=1))
        MEMORY_USAGE.set(psutil.virtual_memory().percent)
        DISK_USAGE.set(psutil.disk_usage("/").percent)
        time.sleep(5)


# ─── MODEL LOADER ────────────────────────────────────────────────────────────

def load_model():
    log.info(f"Loading model: {MODEL_URI}")
    start = time.time()
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)
    elapsed = time.time() - start
    MODEL_LOAD_TIME.set(elapsed)
    log.info(f"  Model loaded in {elapsed:.2f}s")
    return model


# ─── ROUTES ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_URI}), 200


@app.route("/predict", methods=["POST"])
def predict():
    TOTAL_REQUESTS.inc()
    ACTIVE_REQUESTS.inc()
    start_total = time.time()

    try:
        body = request.get_json(force=True)
        if not body:
            raise ValueError("Request body kosong.")

        rows = body if isinstance(body, list) else [body]
        df   = pd.DataFrame(rows)
        INPUT_BATCH_SIZE.observe(len(df))

        # Predict + ukur latency
        with PREDICTION_LATENCY.time():
            predictions  = MODEL.predict(df)
            probabilities = MODEL.predict_proba(df)[:, 1]

        # Update class counter
        for pred in predictions:
            label = "Churn" if pred == 1 else "No_Churn"
            PREDICTION_CLASS_COUNT.labels(label=label).inc()

        SUCCESS_REQUESTS.inc()

        # Update error rate
        total   = TOTAL_REQUESTS._value.get()
        failed  = FAILED_REQUESTS._value.get()
        ERROR_RATE.set((failed / total * 100) if total > 0 else 0)

        elapsed = time.time() - start_total
        RESPONSE_TIME.observe(elapsed)

        return jsonify({
            "predictions" : predictions.tolist(),
            "probabilities": [round(float(p), 4) for p in probabilities],
            "labels"      : ["Churn" if p == 1 else "No Churn" for p in predictions],
            "latency_s"   : round(elapsed, 4),
        }), 200

    except Exception as exc:
        FAILED_REQUESTS.inc()
        total  = TOTAL_REQUESTS._value.get()
        failed = FAILED_REQUESTS._value.get()
        ERROR_RATE.set((failed / total * 100) if total > 0 else 0)
        log.error(f"Prediction error: {exc}")
        return jsonify({"error": str(exc)}), 500

    finally:
        ACTIVE_REQUESTS.dec()


@app.route("/metrics", methods=["GET"])
def metrics():
    data = generate_latest()
    return Response(data, mimetype=CONTENT_TYPE_LATEST)


# ─── ENTRYPOINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load model
    MODEL = load_model()

    # Start system metrics background thread
    t = threading.Thread(target=update_system_metrics, daemon=True)
    t.start()

    log.info(f"Starting server on {HOST}:{PORT}")
    log.info(f"  POST /predict  — Prediction endpoint")
    log.info(f"  GET  /metrics  — Prometheus scrape endpoint")
    log.info(f"  GET  /health   — Health check")
    app.run(host=HOST, port=PORT)
