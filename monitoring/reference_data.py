import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = PROJECT_ROOT / "monitoring" / "reference_stats.json"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "credit_card_fraud_v1.parquet"


def generate_reference_stats(dataset_path: Path) -> Path:
    df = pd.read_parquet(dataset_path)

    stats = {}
    for col in df.columns:
        if df[col].dtype != "object":
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            stats[col] = {
                "mean": float(values.mean()) if not values.empty else 0.0,
                "std": float(values.std()) if not values.empty else 0.0,
                "values": values.tolist(),
            }

    with open(REFERENCE_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f)

    print(f"Reference statistics saved to {REFERENCE_PATH}")
    return REFERENCE_PATH


if __name__ == "__main__":
    generate_reference_stats(DEFAULT_DATASET_PATH)