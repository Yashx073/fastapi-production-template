import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_data(parquet_path: str | Path):
    parquet_path = Path(parquet_path)
    if parquet_path.suffix != ".parquet":
        raise ValueError("Training data must be a .parquet file")

    df = pd.read_parquet(parquet_path)

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
