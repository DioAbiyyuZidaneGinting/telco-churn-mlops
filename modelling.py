"""
modelling.py
============
Random Forest + MLflow Local Tracking
Dataset : telco_churn_preprocessed.csv
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("modelling.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

DATA_PATH       = "telco_churn_preprocessed.csv"
EXPERIMENT_NAME = "telco-churn-classification"
RUN_NAME        = "random-forest-baseline"
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

RF_PARAMS = {
    "n_estimators"  : 100,
    "max_depth"     : 10,
    "min_samples_split": 5,
    "class_weight"  : "balanced",
    "random_state"  : RANDOM_STATE,
    "n_jobs"        : -1,
}


# ─── helpers ────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    log.info(f"Loading data: {path}")
    df = pd.read_csv(path)
    log.info(f"  Shape: {df.shape} | Target dist:\n{df['Churn'].value_counts().to_string()}")
    return df


def split(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log.info(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def train(X_train, y_train, params: dict) -> RandomForestClassifier:
    log.info(f"Training RandomForest | params: {params}")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    log.info("  Training complete.")
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall"   : recall_score(y_test, y_pred, zero_division=0),
        "f1_score" : f1_score(y_test, y_pred, zero_division=0),
        "roc_auc"  : roc_auc_score(y_test, y_prob),
    }
    log.info("=== Evaluation ===")
    for k, v in metrics.items():
        log.info(f"  {k}: {v:.4f}")
    log.info("\n" + classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    return metrics, y_pred, y_prob


def save_confusion_matrix(y_test, y_pred, path="confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path


def save_roc_curve(y_test, y_prob, path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend()
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path


def save_feature_importance(model, feature_names, path="feature_importance.png", top_n=20):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(top_n), imp[idx], color="steelblue")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
    ax.set_title(f"Top {top_n} Feature Importances"); ax.set_ylabel("Importance")
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path


# ─── main ───────────────────────────────────────────────────────────────────

def run_modelling():
    log.info("*" * 55)
    log.info("  MEMULAI MODELLING — modelling.py")
    log.info("*" * 55)

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, feat_names = split(df)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        log.info(f"MLflow Run ID: {run.info.run_id}")

        # Train
        model = train(X_train, y_train, RF_PARAMS)

        # Evaluate
        metrics, y_pred, y_prob = evaluate(model, X_test, y_test)

        # Log params & metrics
        mlflow.log_params(RF_PARAMS)
        mlflow.log_metrics(metrics)

        # Log artefak
        mlflow.log_artifact(save_confusion_matrix(y_test, y_pred))
        mlflow.log_artifact(save_roc_curve(y_test, y_prob))
        mlflow.log_artifact(save_feature_importance(model, feat_names))

        # Log dataset info
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size",  len(X_test))
        mlflow.log_param("n_features", len(feat_names))

        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model",
                                 registered_model_name="telco-churn-rf")
        log.info(f"Model logged to MLflow: {EXPERIMENT_NAME}/{RUN_NAME}")

    log.info("*" * 55)
    log.info("  MODELLING SELESAI")
    log.info("*" * 55)


if __name__ == "__main__":
    run_modelling()
