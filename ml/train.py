from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, 
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    precision_score, 
    recall_score, 
    f1_score,
)
from sklearn.model_selection import train_test_split
from preprocess import load_data, build_preprocessor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow
import mlflow.sklearn

DATA_PATH = "ml/data/credit_card_fraud_10k.csv"

def train_and_log(model, model_name, X_train, X_test, y_train, y_test):

    with mlflow.start_run(run_name = model_name):

        mlflow.log_param("model_type", model_name)

        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] 

        roc_auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metric("roc-auc", roc_auc)
        mlflow.log_metric("Precision_class_1", report["1"]["precision"])
        mlflow.log_metric("Recall_class_1", report["1"]["recall"])
        mlflow.log_metric("F1_class_1", report["1"]["f1-score"])

        print("\nModel:", {model_name})
        print("ROC-AUC:", roc_auc)
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision Recall Curve")
        plt.savefig("precision_recall_curve.png")
        mlflow.log_artifact("precision_recall_curve.png")
        plt.close()

        result = mlflow.sklearn.log_model(
            sk_model = model,
            artifact_path = "model"
        )

        mlflow.register_model(
            model_uri = result.model_uri,
            name = "fraud-detection-model"
        )

        


def main():

    mlflow.set_experiment("fraud_detection")

    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    preprocessor = build_preprocessor(X_train)

    #Logistic Regression
    lr_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    train_and_log(lr_model, "Logistic Regression", X_train, X_test, y_train, y_test)

    # Random Forest
    rf_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            class_weight="balanced",
            random_state=42
        ))
    ])
    train_and_log(rf_model, "Random Forest", X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()


