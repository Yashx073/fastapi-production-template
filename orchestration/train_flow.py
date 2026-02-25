from prefect import flow, task
from pathlib import Path
import argparse
import sys

from ml.train_pipeline import main as train_main

@task(
    name = "Run Training Pipeline",
    retries = 1,
    retry_delay_seconds = 30
)
def run_training(data_path: str, config_path: str):
    class Args:
        data = data_path
        config = config_path
    
    train_main(Args)

@flow(
    name = "fraud-detection-training-flow",
    log_prints= True,
)
def training_flow(data_path: str, config_path: str):
    run_training(data_path, config_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Perfect Training Flow")
    parser.add_argument("--data", required= True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    training_flow(
        data_path=args.data,
        config_path=args.config,
    )