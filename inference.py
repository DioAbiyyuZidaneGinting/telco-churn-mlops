"""
inference.py
============
Load model dari MLflow dan jalankan prediksi.
Mendukung input dari CSV atau stdin JSON.
"""

import os
import sys
import json
import logging
import warnings
import pandas as pd
import mlflow.sklearn

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── CONFIG ─────────────────────────────────────────────────────────────────
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MODEL_URI    = os.getenv("MODEL_URI", "models:/telco-churn-rf-tuned/latest")


# ─── HELPERS ────────────────────────────────────────────────────────────────

def load_model(model_uri: str):
    log.info(f"Loading model from: {model_uri}")
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model(model_uri)
    log.info("  Model loaded successfully.")
    return model


def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    """Mengembalikan DataFrame dengan kolom prediction dan probability."""
    predictions  = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    result = df.copy()
    result["prediction"]  = predictions
    result["probability"] = probabilities.round(4)
    result["churn_label"] = result["prediction"].map({1: "Churn", 0: "No Churn"})
    return result


def predict_from_csv(model, csv_path: str) -> pd.DataFrame:
    log.info(f"Reading input CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    log.info(f"  Input shape: {df.shape}")
    result = predict(model, df)
    log.info(f"  Churn predicted: {result['prediction'].sum()} / {len(result)}")
    return result


def predict_from_json(model, json_input: str) -> dict:
    """Input: JSON string berisi list of dicts (satu row = satu record)."""
    data = json.loads(json_input)
    df = pd.DataFrame(data if isinstance(data, list) else [data])
    result = predict(model, df)
    return result[["prediction", "probability", "churn_label"]].to_dict(orient="records")


# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    model = load_model(MODEL_URI)

    if len(sys.argv) > 1:
        input_path = sys.argv[1]

        if input_path.endswith(".csv"):
            result = predict_from_csv(model, input_path)
            output_path = input_path.replace(".csv", "_predictions.csv")
            result.to_csv(output_path, index=False)
            log.info(f"Predictions saved to: {output_path}")
            print(result[["prediction", "probability", "churn_label"]].head(10).to_string())

        elif input_path.endswith(".json"):
            with open(input_path) as f:
                json_str = f.read()
            results = predict_from_json(model, json_str)
            print(json.dumps(results, indent=2))

        else:
            log.error("Unsupported input format. Use .csv or .json")
            sys.exit(1)

    else:
        # Demo: pakai data preprocessed
        log.info("No input provided. Running demo with telco_churn_preprocessed.csv...")
        result = predict_from_csv(model, "telco_churn_preprocessed.csv")
        print("\n=== Sample Predictions ===")
        print(result[["prediction", "probability", "churn_label"]].head(10).to_string())


if __name__ == "__main__":
    main()
