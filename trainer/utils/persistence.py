# task.py
import argparse
import logging

from google.cloud import storage
from tensorflow import keras

import trainer.config as config

logger = logging.getLogger(__name__)

def save_model_and_reports(
    args: argparse.Namespace,
    model_name: str,
    history: keras.callbacks.History
) -> None:
    storage_client = storage.Client(project=args.project_id)
    bucket = storage_client.bucket(args.gcs_path)

    # Save model
    blob = bucket.blob(f"models/{model_name}")
    blob.upload_from_filename(f"{config.MODEL_PATH}/{model_name}")
    logger.info(f"Model saved to {args.gcs_path}/models/{model_name}")

    # Save training report
    if history:
        blob = bucket.blob(f"reports/{args.model_name}_{args.curr_date_str}.png")
        blob.upload_from_filename(f"{config.MODEL_PATH}/{args.model_name}_{args.curr_date_str}.png")
        logger.info(f"Training report saved to {args.gcs_path}/reports/{args.model_name}_{args.curr_date_str}.png")

    # Save threshold report
    blob = bucket.blob(f"reports/{args.model_name}_{args.curr_date_str}_error_distribution.png")
    blob.upload_from_filename(f"{config.MODEL_PATH}/{args.model_name}_{args.curr_date_str}_error_distribution.png")
    logger.info(f"Threshold report saved to {args.gcs_path}/reports/{args.model_name}_{args.curr_date_str}_error_distribution.png")

    storage_client.close()
