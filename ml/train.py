import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from preprocess import load_data, build_preprocessor
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

DATA_PATH = "ml/data/credit_card_fraud_10k.csv"

def train_and_log(model, model_name, X_train, X_test, y_train, y_test):

    with mlflow.start_run(run_name = model_name):

        clf = Pipeline(steps=[
            ("Preprocessor", build_preprocessor(X_train)),
            ("model", model),
        ])

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_prob)

        print("\nModel:", {model_name})
        print("ROC-AUC:", roc_auc)
        print(classification_report(y_test, y_pred))

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("roc-auc", roc_auc)
        mlflow.log_param("Precision", precision)
        mlflow.log_param("Recall", recall)
        mlflow.log_param("F1", f1)

        mlflow.sklearn.log_model(
            clf,
            artifact_path = "model",
        )

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

        mlflow.register_model(
            model_uri = model_uri,
            name = "fraud-detection-model"
        )


def main():

    mlflow.set_experiment("fraud_detection")

    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    #Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    train_and_log(lr_model, "Logistic Regression", X_train, X_test, y_train, y_test)

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight="balanced",
        random_state=42
    )
    train_and_log(rf_model, "Random Forest", X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()


