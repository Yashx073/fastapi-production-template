import os
import sys
import argparse
import tempfile
import hashlib
from pathlib import Path
import yaml
import random

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")

import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from validation.schema import validate_schema
from features.preprocess import build_preprocessor, load_data


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def set_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)


# --------------------------------------------------
# Training + MLflow logging
# --------------------------------------------------
def train_and_log(
    model_pipeline,
    model_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
    data_path: Path,
    data_sha256: str,
    config_path: Path,
    random_seed: int,
):
    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("framework", "sklearn")
        mlflow.set_tag("dataset", data_path.name)
        mlflow.set_tag("data_sha256", data_sha256)

        try:
            git_sha = os.popen("git rev-parse HEAD").read().strip()
            if git_sha:
                mlflow.set_tag("git_commit", git_sha)
        except Exception:
            pass

        mlflow.log_param("random_seed", random_seed)
        mlflow.log_artifact(str(config_path), artifact_path="config")

        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)
        y_prob = model_pipeline.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

        cm = confusion_matrix(y_test, y_pred)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("precision_class_1", report["1"]["precision"])
        mlflow.log_metric("recall_class_1", report["1"]["recall"])
        mlflow.log_metric("f1_class_1", report["1"]["f1-score"])
        mlflow.log_metric("false_negatives", cm[1, 0])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            cm_path = tmpdir / "confusion_matrix.png"
            plt.figure(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(str(cm_path), artifact_path="plots")

            pr_path = tmpdir / "precision_recall_curve.png"
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            plt.figure()
            plt.plot(recall, precision)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.savefig(pr_path)
            plt.close()
            mlflow.log_artifact(str(pr_path), artifact_path="plots")

        mlflow.sklearn.log_model(
            model_pipeline,
            artifact_path="model",
            registered_model_name="fraud-detection-model",
        )


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------
def main(args):
    config = load_config(Path(args.config))
    data_path = Path(args.data)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(config["experiment"]["name"])

    set_seeds(config["split"]["random_state"])

    df = pd.read_parquet(data_path)
    validate_schema(df, raise_on_error=True)

    data_sha256 = file_sha256(data_path)

    X = df.drop(columns=["transaction_id", "is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        stratify=y,
        random_state=config["split"]["random_state"],
    )

    with mlflow.start_run(run_name="training_pipeline"):
        mlflow.set_tag("pipeline", "fraud_detection_training")
        mlflow.set_tag("framework", "sklearn")
        mlflow.set_tag("problem_type", "binary_classification")
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("num_rows", len(df))
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("train_size", X_train.shape[0])
        mlflow.log_param("test_size", X_test.shape[0])

        lr_pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(X_train)),
                (
                    "model",
                    LogisticRegression(**config["models"]["logistic_regression"]),
                ),
            ]
        )

        train_and_log(
            lr_pipeline,
            "Logistic_Regression",
            X_train,
            X_test,
            y_train,
            y_test,
            data_path,
            data_sha256,
            Path(args.config),
            config["split"]["random_state"],
        )

        rf_pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(X_train)),
                (
                    "model",
                    RandomForestClassifier(**config["models"]["random_forest"]),
                ),
            ]
        )

        train_and_log(
            rf_pipeline,
            "Random_Forest",
            X_train,
            X_test,
            y_train,
            y_test,
            data_path,
            data_sha256,
            Path(args.config),
            config["split"]["random_state"],
        )


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic fraud model training")
    parser.add_argument("--data", required=True, help="Path to processed Parquet dataset")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    args = parser.parse_args()

    try:
        main(args)
        print("\n✅ Training pipeline completed successfully")
    except RuntimeError as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)