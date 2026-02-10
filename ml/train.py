import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from preprocess import load_data, build_preprocessor
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

DATA_PATH = "data/credit_card_fraud_10k.csv"

def main():
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    preprocessor = build_preprocessor(X)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, "model.pkl")
    print("Model saved")

    import numpy as np

    threshold = 0.8
    y_pred_custom = (y_prob >= threshold).astype(int)

    print(f"\nCustom Threshold: {threshold}")   
    print(classification_report(y_test, y_pred_custom))

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall-Curve")
    plt.savefig("precision_recall_curve.png")


if __name__ == "__main__":
    main()


