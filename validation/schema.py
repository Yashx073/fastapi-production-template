import pandas as pd
from great_expectations.dataset import PandasDataset


def validate_schema(df: pd.DataFrame) -> None:
    ge_df = PandasDataset(df)

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
    ge_df.expect_table_columns_to_match_ordered_list(expected_columns)

    # ---------- Uniqueness ----------
    ge_df.expect_column_values_to_be_unique("transaction_id")

    # ---------- Numeric ranges ----------
    ge_df.expect_column_values_to_be_between("amount", 0.0, 100000.0)
    ge_df.expect_column_values_to_be_between("transaction_hour", 0, 23)
    ge_df.expect_column_values_to_be_between("device_trust_score", 0, 100)
    ge_df.expect_column_values_to_be_between("velocity_last_24h", 0, 500)
    ge_df.expect_column_values_to_be_between("cardholder_age", 18, 100)

    # ---------- Binary flags ----------
    for col in ["foreign_transaction", "location_mismatch", "is_fraud"]:
        ge_df.expect_column_values_to_be_in_set(col, [0, 1])

    # ---------- Null checks ----------
    for col in expected_columns:
        ge_df.expect_column_values_to_not_be_null(col)

    # ---------- Final validation ----------
    result = ge_df.validate()

    if not result["success"]:
        raise RuntimeError("❌ Schema validation failed")

    print("✅ Schema validation passed")