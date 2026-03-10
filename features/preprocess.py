import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_data(data_path: str | Path):
    """Load data from CSV or Parquet file."""
    data_path = Path(data_path)
    
    if data_path.suffix.lower() == '.csv':
        df = pd.read_csv(data_path)
    elif data_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}. Use .csv or .parquet")

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
