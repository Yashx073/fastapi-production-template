import pandas as pd


EXPECTED_COLUMNS = [
    "transaction_id",
    "amount",
    "transaction_hour",
    "merchant_category",
    "foreign_transaction",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age",
    "is_fraud",
]


def validate_schema(df: pd.DataFrame) -> None:
    """Validate dataframe schema and constraints using pure pandas.
    
    Raises RuntimeError if validation fails.
    """
    failures = []
    
    # Column structure
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        failures.append(f"Missing columns: {missing}")
    
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra:
        failures.append(f"Extra columns: {extra}")
    
    # Uniqueness
    if "transaction_id" in df.columns and df["transaction_id"].duplicated().any():
        failures.append("transaction_id has duplicates")
    
    # Numeric ranges
    if "amount" in df.columns:
        invalid = (df["amount"] < 0) | (df["amount"] > 100000)
        if invalid.any():
            failures.append(f"amount out of range: {invalid.sum()} rows")
    
    if "transaction_hour" in df.columns:
        invalid = (df["transaction_hour"] < 0) | (df["transaction_hour"] > 23)
        if invalid.any():
            failures.append(f"transaction_hour out of range: {invalid.sum()} rows")
    
    if "device_trust_score" in df.columns:
        invalid = (df["device_trust_score"] < 0) | (df["device_trust_score"] > 100)
        if invalid.any():
            failures.append(f"device_trust_score out of range: {invalid.sum()} rows")
    
    if "velocity_last_24h" in df.columns:
        invalid = (df["velocity_last_24h"] < 0) | (df["velocity_last_24h"] > 500)
        if invalid.any():
            failures.append(f"velocity_last_24h out of range: {invalid.sum()} rows")
    
    if "cardholder_age" in df.columns:
        invalid = (df["cardholder_age"] < 18) | (df["cardholder_age"] > 100)
        if invalid.any():
            failures.append(f"cardholder_age out of range: {invalid.sum()} rows")
    
    # Binary flags
    for col in ["foreign_transaction", "location_mismatch", "is_fraud"]:
        if col in df.columns:
            invalid = ~df[col].isin([0, 1])
            if invalid.any():
                failures.append(f"{col} has non-binary values: {invalid.sum()} rows")
    
    # NULL checks
    for col in EXPECTED_COLUMNS:
        if col in df.columns and df[col].isna().any():
            failures.append(f"{col} has NULL values: {df[col].isna().sum()} rows")
    
    if failures:
        error_msg = "❌ Schema validation failed:\n  " + "\n  ".join(failures)
        raise RuntimeError(error_msg)
    
    print(f"✅ Schema validation passed ({len(df)} rows, {len(df.columns)} columns)")