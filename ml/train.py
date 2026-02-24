import os
import sys
import tempfile
import hashlib
from pathlib import Path
import joblib

# Set environment variables BEFORE importing mlflow
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Configure MLflow to use remote server for tracking only (not registry)
mlflow.set_tracking_uri("http://mlflow:5000")

EXPERIMENT_NAME = "fraud_detection"
RANDOM_SEED = 42
mlflow.set_experiment(EXPERIMENT_NAME)

from features.preprocess import load_data, build_preprocessor

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "credit_card_fraud_v1.parquet"


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def train_and_log(model, model_name, X_train, X_test, y_train, y_test, data_sha256):
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("stage", "training")
        mlflow.set_tag("problem_type", "binary_classification")
        mlflow.set_tag("dataset", "credit_card_fraud_10k")
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("imbalance", "severe")

        mlflow.log_param("data_path", str(DATA_PATH))
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("random_seed", RANDOM_SEED)

        model_step = model.named_steps["model"]
        mlflow.log_params(model_step.get_params())

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

        cm = confusion_matrix(y_test, y_pred)
        false_negatives = cm[1, 0]

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("precision_class_1", report["1"]["precision"])
        mlflow.log_metric("recall_class_1", report["1"]["recall"])
        mlflow.log_metric("f1_class_1", report["1"]["f1-score"])
        mlflow.log_metric("false_negatives", false_negatives)

        with tempfile.TemporaryDirectory() as tmpdir:
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            plt.figure(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path, artifact_path="plots")

            pr_path = os.path.join(tmpdir, "precision_recall_curve.png")
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            plt.figure()
            plt.plot(recall, precision)
            plt.savefig(pr_path)
            plt.close()
            mlflow.log_artifact(pr_path, artifact_path="plots")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.pkl")
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path, artifact_path="model")
                print(f"✓ Model artifact logged for {model_name}")
        except Exception as e:
            print(f"⚠ Could not log model artifact: {e}")


def main():
    X, y = load_data(DATA_PATH)
    data_sha256 = _file_sha256(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    preprocessor = build_preprocessor(X_train)

    lr_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )),
    ])

    train_and_log(
        lr_model,
        "Logistic_Regression",
        X_train,
        X_test,
        y_train,
        y_test,
        data_sha256,
    )

    rf_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ])

    train_and_log(
        rf_model,
        "Random_Forest",
        X_train,
        X_test,
        y_train,
        y_test,
        data_sha256,
    )


if __name__ == "__main__":
    main()
