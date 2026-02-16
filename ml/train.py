import os
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

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

MLFLOW_ARTIFACT_ROOT = PROJECT_ROOT / "mlruns"
MLFLOW_TRACKING_URI = f"file://{MLFLOW_ARTIFACT_ROOT}"

EXPERIMENT_NAME = "fraud_detection"
REGISTERED_MODEL_NAME = "fraud-detection-model"

os.makedirs(MLFLOW_ARTIFACT_ROOT, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient()

from features.preprocess import load_data, build_preprocessor

DATA_PATH = PROJECT_ROOT / "ml" / "data" / "credit_card_fraud_10k.csv"


def train_and_log(model, model_name, X_train, X_test, y_train, y_test):

    with mlflow.start_run(run_name=model_name):

        mlflow.set_tag("stage", "training")
        mlflow.set_tag("problem_type", "binary_classification")
        mlflow.set_tag("dataset", "credit_card_fraud_10k")
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("imbalance", "severe")

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

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )



def main():

    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
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
    )

    rf_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    train_and_log(
        rf_model,
        "Random_Forest",
        X_train,
        X_test,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    main()
