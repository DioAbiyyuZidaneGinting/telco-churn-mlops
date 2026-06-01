"""
modelling_tuning.py
===================
Random Forest + GridSearchCV + MLflow (DagsHub-compatible)

Setup DagsHub sebelum menjalankan:
1. Daftar di https://dagshub.com
2. Buat repo baru → Import from GitHub
3. Buka Settings → Integrations → MLflow
4. Salin Tracking URI, username, dan token
5. Set environment variables:
     export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
     export MLFLOW_TRACKING_USERNAME=<dagshub_username>
     export MLFLOW_TRACKING_PASSWORD=<dagshub_token>
6. Atau isi langsung pada bagian CONFIG di bawah (tidak disarankan untuk produksi)
"""

import os
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
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ─── LOGGING ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("modelling_tuning.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─── CONFIG ─────────────────────────────────────────────────────────────────
DATA_PATH = "telco_churn_preprocessed.csv"
EXPERIMENT_NAME = "telco-churn-tuning"
RUN_NAME = "rf-gridsearch"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# DagsHub — ambil dari env variable (isi via GitHub Secrets / .env)
DAGSHUB_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/<YOUR_USERNAME>/<YOUR_REPO>.mlflow",
)

PARAM_GRID = {
    "n_estimators"    : [50, 100, 200],
    "max_depth"       : [5, 10, None],
    "min_samples_split": [2, 5],
    "class_weight"    : ["balanced"],
}


# ─── HELPERS ────────────────────────────────────────────────────────────────

def load_data(path: str):
    log.info(f"Loading: {path}")
    df = pd.read_csv(path)
    log.info(f"  Shape: {df.shape}")
    return df


def split(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    log.info(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def run_gridsearch(X_train, y_train, param_grid: dict):
    log.info(f"Running GridSearchCV | param_grid: {param_grid}")
    base_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    gs = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    log.info(f"  Best params : {gs.best_params_}")
    log.info(f"  Best CV F1  : {gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_, gs.best_score_, gs.cv_results_


def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall"   : recall_score(y_test, y_pred, zero_division=0),
        "f1_score" : f1_score(y_test, y_pred, zero_division=0),
        "roc_auc"  : roc_auc_score(y_test, y_prob),
    }, y_pred, y_prob


def fig_confusion_matrix(y_test, y_pred) -> str:
    path = "cm_tuned.png"
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_title("Confusion Matrix — Tuned RF")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    return path


def fig_roc_curve(y_test, y_prob) -> str:
    path = "roc_tuned.png"
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, color="royalblue", label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve — Tuned RF"); ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    return path


def fig_feature_importance(model, feature_names, top_n=20) -> str:
    path = "feat_imp_tuned.png"
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(top_n), imp[idx], color="royalblue")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
    ax.set_title(f"Top {top_n} Feature Importances — Tuned RF")
    ax.set_ylabel("Importance")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    return path


def save_dataset_summary(df: pd.DataFrame) -> str:
    path = "dataset_summary.txt"
    with open(path, "w") as f:
        f.write(f"Shape           : {df.shape}\n")
        f.write(f"Target dist     :\n{df['Churn'].value_counts().to_string()}\n")
        f.write(f"Churn rate      : {df['Churn'].mean()*100:.2f}%\n\n")
        f.write(df.describe().to_string())
    return path


def save_classification_report(y_test, y_pred) -> str:
    path = "classification_report.txt"
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    with open(path, "w") as f:
        f.write(report)
    log.info(f"\n{report}")
    return path


# ─── MAIN ───────────────────────────────────────────────────────────────────

def run_tuning():
    log.info("*" * 55)
    log.info("  MEMULAI TUNING — modelling_tuning.py")
    log.info("*" * 55)

    # Setup MLflow → DagsHub
    mlflow.set_tracking_uri(DAGSHUB_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    log.info(f"MLflow Tracking URI : {DAGSHUB_TRACKING_URI}")

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, feat_names = split(df)

    best_model, best_params, best_cv_f1, cv_results = run_gridsearch(X_train, y_train, PARAM_GRID)
    metrics, y_pred, y_prob = compute_metrics(best_model, X_test, y_test)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        log.info(f"MLflow Run ID: {run.info.run_id}")

        # ── Log parameters ──────────────────────────────────────────────
        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring",  "f1")
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("n_features", len(feat_names))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples",  len(X_test))

        # ── Log metrics ──────────────────────────────────────────────────
        mlflow.log_metrics(metrics)
        mlflow.log_metric("best_cv_f1", best_cv_f1)

        # ── Log artefak visual ───────────────────────────────────────────
        mlflow.log_artifact(fig_confusion_matrix(y_test, y_pred))
        mlflow.log_artifact(fig_roc_curve(y_test, y_prob))
        mlflow.log_artifact(fig_feature_importance(best_model, feat_names))

        # ── Log teks artefak ─────────────────────────────────────────────
        mlflow.log_artifact(save_classification_report(y_test, y_pred))
        mlflow.log_artifact(save_dataset_summary(df))

        # ── Log model ────────────────────────────────────────────────────
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="rf_tuned_model",
            signature=signature,
            registered_model_name="telco-churn-rf-tuned",
        )

        log.info("=== Final Metrics ===")
        for k, v in metrics.items():
            log.info(f"  {k}: {v:.4f}")

    log.info("*" * 55)
    log.info("  TUNING SELESAI")
    log.info("*" * 55)


if __name__ == "__main__":
    run_tuning()
