"""
MLProject/modelling.py
=======================
Versi MLProject dari modelling.py — menerima argumen CLI.
"""

import argparse
import logging
import warnings
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
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
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Telco Churn — RF MLflow Project")
    parser.add_argument("--data_path",    type=str,   default="telco_churn_preprocessed.csv")
    parser.add_argument("--n_estimators", type=int,   default=100)
    parser.add_argument("--max_depth",    type=int,   default=10)
    parser.add_argument("--test_size",    type=float, default=0.2)
    return parser.parse_args()


def save_cm(y_test, y_pred):
    path = "confusion_matrix.png"
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_title("Confusion Matrix"); ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path


def save_roc(y_test, y_prob):
    path = "roc_curve.png"
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC Curve"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(); plt.tight_layout(); plt.savefig(path); plt.close()
    return path


def save_feat_imp(model, feature_names, top_n=20):
    path = "feature_importance.png"
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(top_n), imp[idx], color="steelblue")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
    ax.set_title(f"Top {top_n} Feature Importances"); ax.set_ylabel("Importance")
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path


def main():
    args = parse_args()
    log.info(f"Args: {vars(args)}")

    df = pd.read_csv(args.data_path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    feat_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    params = {
        "n_estimators"  : args.n_estimators,
        "max_depth"     : args.max_depth,
        "class_weight"  : "balanced",
        "random_state"  : 42,
        "n_jobs"        : -1,
    }
    mlflow.log_params(params)
    mlflow.log_param("test_size", args.test_size)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall"   : recall_score(y_test, y_pred, zero_division=0),
        "f1_score" : f1_score(y_test, y_pred, zero_division=0),
        "roc_auc"  : roc_auc_score(y_test, y_prob),
    }
    mlflow.log_metrics(metrics)

    mlflow.log_artifact(save_cm(y_test, y_pred))
    mlflow.log_artifact(save_roc(y_test, y_prob))
    mlflow.log_artifact(save_feat_imp(model, feat_names))

    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "rf_model", signature=signature,
                             registered_model_name="telco-churn-rf-mlproject")

    log.info("=== Results ===")
    for k, v in metrics.items():
        log.info(f"  {k}: {v:.4f}")



if __name__ == "__main__":
    main()
