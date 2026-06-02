"""
modelling.py
============
Random Forest + MLflow Local Tracking (autolog only)
Dataset : telco_churn_preprocessed.csv
"""

import argparse
import logging
import warnings
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train Random Forest for Telco Churn")
    parser.add_argument("--data_path",    type=str,   default=DATA_PATH,    help="Path to preprocessed CSV")
    parser.add_argument("--n_estimators", type=int,   default=100,          help="Number of trees")
    parser.add_argument("--max_depth",    type=int,   default=10,           help="Max tree depth")
    parser.add_argument("--test_size",    type=float, default=TEST_SIZE,    help="Test split ratio")
    return parser.parse_args()


def run_modelling(args):
    log.info("*" * 55)
    log.info("  MEMULAI MODELLING — modelling.py")
    log.info("*" * 55)
    log.info(f"  data_path    : {args.data_path}")
    log.info(f"  n_estimators : {args.n_estimators}")
    log.info(f"  max_depth    : {args.max_depth}")
    log.info(f"  test_size    : {args.test_size}")

    df = pd.read_csv(args.data_path)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    log.info(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    params = {
        "n_estimators"     : args.n_estimators,
        "max_depth"        : args.max_depth,
        "min_samples_split": 5,
        "class_weight"     : "balanced",
        "random_state"     : RANDOM_STATE,
        "n_jobs"           : -1,
    }

    # Enable autologging
    mlflow.sklearn.autolog()

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=RUN_NAME):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate on test set so metrics can be logged by autolog
        score = model.score(X_test, y_test)
        log.info(f"Model test score: {score:.4f}")

    log.info("*" * 55)
    log.info("  MODELLING SELESAI")
    log.info("*" * 55)


if __name__ == "__main__":
    run_modelling(parse_args())
