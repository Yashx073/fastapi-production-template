from pathlib import Path
import hashlib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "ml" / "data" / "credit_card_fraud_10k.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "credit_card_fraud_v1.parquet"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "credit_card_fraud_v1.json"


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df = df.drop_duplicates()
    df = df.dropna()

    df.to_parquet(PROCESSED_PATH, index=False)
    dataset_hash = _file_sha256(PROCESSED_PATH)

    METADATA_PATH.write_text(
        f"""{{
  \"dataset\": \"credit_card_fraud\",
  \"version\": \"v1\",
  \"rows\": {len(df)},
  \"columns\": {list(df.columns)},
  \"hash\": \"{dataset_hash}\"
}}"""
    )

    print("Dataset built successfully")
    print("Rows:", len(df))
    print("Hash:", dataset_hash)


if __name__ == "__main__":
    main()
