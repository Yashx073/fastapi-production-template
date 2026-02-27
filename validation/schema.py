import pandas as pd


def validate_schema(df: pd.DataFrame) -> None:
    expected_columns = [
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

    # ---------- Column existence ----------
    if list(df.columns) != expected_columns:
        raise RuntimeError("❌ Schema validation failed: column order mismatch")

    # ---------- Uniqueness ----------
    if df["transaction_id"].duplicated().any():
        raise RuntimeError("❌ Schema validation failed: duplicate transaction_id")

    # ---------- Numeric ranges ----------
    range_checks = {
        "amount": (0.0, 100000.0),
        "transaction_hour": (0, 23),
        "device_trust_score": (0, 100),
        "velocity_last_24h": (0, 500),
        "cardholder_age": (18, 100),
    }
    for col, (min_val, max_val) in range_checks.items():
        if not df[col].between(min_val, max_val).all():
            raise RuntimeError(f"❌ Schema validation failed: {col} out of range")

    # ---------- Binary flags ----------
    for col in ["foreign_transaction", "location_mismatch", "is_fraud"]:
        if not df[col].isin([0, 1]).all():
            raise RuntimeError(f"❌ Schema validation failed: {col} not binary")

    # ---------- Null checks ----------
    if df[expected_columns].isnull().any().any():
        raise RuntimeError("❌ Schema validation failed: null values present")

    print("✅ Schema validation passed")