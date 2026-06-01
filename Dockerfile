# ============================================================
# Dockerfile — Telco Churn MLflow Model Serving
# Python 3.12 kompatibel (menggantikan mlflow models build-docker)
# ============================================================

FROM python:3.12-slim

# ── System dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        mlflow==2.19.0 \
        scikit-learn==1.5.0 \
        pandas==2.2.2 \
        numpy==1.26.4 \
        matplotlib==3.9.0 \
        seaborn==0.13.2

# ── Working directory ────────────────────────────────────────
WORKDIR /app

# ── Environment variables (override at runtime) ──────────────
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_TRACKING_USERNAME=""
ENV MLFLOW_TRACKING_PASSWORD=""
ENV MODEL_URI="models:/telco-churn-rf-mlproject/latest"
ENV PORT=5001

# ── Expose serving port ──────────────────────────────────────
EXPOSE 5001

# ── Entrypoint: jalankan mlflow models serve ─────────────────
CMD mlflow models serve \
        --model-uri "${MODEL_URI}" \
        --host 0.0.0.0 \
        --port ${PORT} \
        --no-conda
