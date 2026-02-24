# task.py
import argparse
import logging
import json

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
    try:
        blob = bucket.blob(f"models/autoencoder/{model_name}")
        blob.upload_from_filename(f"{config.MODEL_PATH}/{model_name}")
        logger.info(f"Model saved to {args.gcs_path}/models/autoencoder/{model_name}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    # Save training report
    try:
        if history:
            blob = bucket.blob(f"reports/autoencoder/{args.curr_date_str}{args.postfix}/loss.png")
            blob.upload_from_filename(f"{config.MODEL_PATH}/loss.png")
            logger.info(f"Training report saved to {args.gcs_path}/reports/autoencoder/{args.curr_date_str}{args.postfix}/loss.png")
    except Exception as e:
        logger.error(f"Failed to save training report: {e}")

    # Save threshold report
    try:
        blob = bucket.blob(f"reports/autoencoder/{args.curr_date_str}{args.postfix}/error_distribution.png")
        blob.upload_from_filename(f"{config.MODEL_PATH}/error_distribution.png")
        logger.info(f"Threshold report saved to {args.gcs_path}/reports/autoencoder/{args.curr_date_str}{args.postfix}/error_distribution.png")
    except Exception as e:
        logger.error(f"Failed to save threshold report: {e}")

    # Save MMC encoding mapping
    try:
        blob = bucket.blob(f"reports/autoencoder/{args.curr_date_str}{args.postfix}/mmc_mapping.json")
        blob.upload_from_filename(f"{config.MODEL_PATH}/mmc_mapping.json")
        logger.info(f"MMC encoding mapping saved to {args.gcs_path}/reports/autoencoder/{args.curr_date_str}{args.postfix}/mmc_mapping.json")
    except Exception as e:
        logger.error(f"Failed to save MMC encoding mapping: {e}")

    storage_client.close()
