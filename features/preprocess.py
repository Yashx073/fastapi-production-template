import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

path ="data/credit_card_fraud_10k.csv"

def load_data(path):
    df = pd.read_csv(path)

    df = df.drop("transaction_id", axis=1)

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    return X, y

def build_preprocessor(X):
    numerical_features = X.select_dtypes(include=["int64","float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor