# task.py
import argparse
import logging
from datetime import datetime
import json

from google.cloud import storage
from google.cloud import bigquery
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
        # Save for historical reference
        blob = bucket.blob(f"models/autoencoder/{model_name}")
        blob.upload_from_filename(f"{config.MODEL_PATH}/{model_name}")

        # Save for latest reference (overwrites previous)
        blob = bucket.blob(f"models/autoencoder/{args.model_name}_00000000{args.postfix}.keras")
        blob.upload_from_filename(f"{config.MODEL_PATH}/{model_name}")

        logger.info(f"Model saved to {args.gcs_path}/models/autoencoder/{model_name}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    # Save training report
    try:
        if history:
            # Save for historical reference
            blob = bucket.blob(f"reports/autoencoder/{args.curr_date_str}{args.postfix}/loss.png")
            blob.upload_from_filename(f"{config.MODEL_PATH}/loss.png")

            # Save for latest reference (overwrites previous)
            blob = bucket.blob(f"reports/autoencoder/00000000{args.postfix}/loss.png")
            blob.upload_from_filename(f"{config.MODEL_PATH}/loss.png")

            logger.info(f"Training report saved to {args.gcs_path}/reports/autoencoder/{args.curr_date_str}{args.postfix}/loss.png")
            
    except Exception as e:
        logger.error(f"Failed to save training report: {e}")

    # Save threshold report
    try:
        # Save for historical reference
        blob = bucket.blob(f"reports/autoencoder/{args.curr_date_str}{args.postfix}/error_distribution.png")
        blob.upload_from_filename(f"{config.MODEL_PATH}/error_distribution.png")

        # Save for latest reference (overwrites previous)
        blob = bucket.blob(f"reports/autoencoder/00000000{args.postfix}/error_distribution.png")
        blob.upload_from_filename(f"{config.MODEL_PATH}/error_distribution.png")

        logger.info(f"Threshold report saved to {args.gcs_path}/reports/autoencoder/{args.curr_date_str}{args.postfix}/error_distribution.png")
    except Exception as e:
        logger.error(f"Failed to save threshold report: {e}")

    # Save MMC encoding mapping
    try:
        # Save for historical reference
        blob = bucket.blob(f"reports/autoencoder/{args.curr_date_str}{args.postfix}/mmc_mapping.json")
        blob.upload_from_filename(f"{config.MODEL_PATH}/mmc_mapping.json")

        # Save for latest reference (overwrites previous)
        blob = bucket.blob(f"reports/autoencoder/00000000{args.postfix}/mmc_mapping.json")
        blob.upload_from_filename(f"{config.MODEL_PATH}/mmc_mapping.json")

        logger.info(f"MMC encoding mapping saved to {args.gcs_path}/reports/autoencoder/{args.curr_date_str}{args.postfix}/mmc_mapping.json")
    except Exception as e:
        logger.error(f"Failed to save MMC encoding mapping: {e}")

    # Save hyperparameters
    try:
        # Save for historical reference
        blob = bucket.blob(f"reports/autoencoder/{args.curr_date_str}{args.postfix}/hyperparameters.json")
        blob.upload_from_filename(f"{config.MODEL_PATH}/hyperparameters.json")

        # Save for latest reference (overwrites previous)
        blob = bucket.blob(f"reports/autoencoder/00000000{args.postfix}/hyperparameters.json")
        blob.upload_from_filename(f"{config.MODEL_PATH}/hyperparameters.json")

        logger.info(f"Hyperparameters saved to {args.gcs_path}/reports/autoencoder/{args.curr_date_str}{args.postfix}/hyperparameters.json")

    except Exception as e:
        logger.error(f"Failed to save hyperparameters: {e}")
    

    # Specific for this model, save column info
    try:
        # Save for historical reference
        blob = bucket.blob(f"reports/autoencoder/{args.curr_date_str}{args.postfix}/columns_info.json")
        blob.upload_from_filename(f"{config.MODEL_PATH}/columns_info.json")

        # Save for latest reference (overwrites previous)
        blob = bucket.blob(f"reports/autoencoder/00000000{args.postfix}/columns_info.json")
        blob.upload_from_filename(f"{config.MODEL_PATH}/columns_info.json")

        logger.info(f"Columns info saved to {args.gcs_path}/reports/autoencoder/{args.curr_date_str}{args.postfix}/columns_info.json")

    except Exception as e:
        logger.error(f"Failed to save columns info: {e}")

    storage_client.close()


    # Save hyperparameters and columns info to big query
    try:
        bq_client = bigquery.Client(project=args.project_id)
        dataset_id, table_id = args.bq_report_path.split(".")[-2:]
        table_ref = bq_client.dataset(dataset_id).table(table_id)

        # Load hyperparameters and columns info from local files
        with open(f"{config.MODEL_PATH}/hyperparameters.json", "r") as f:
            hyperparams_data = json.load(f)
        with open(f"{config.MODEL_PATH}/columns_info.json", "r") as f:
            columns_info_data = json.load(f)

        # Combine into a single record
        record = {
            "training_date": datetime.strptime(args.curr_date_str, "%Y%m%d").date(),
            **hyperparams_data,
            **columns_info_data,
            "postfix": args.postfix
        }

        # Insert into BigQuery
        errors = bq_client.insert_rows_json(table_ref, [record])
        if errors:
            logger.error(f"Failed to insert record into BigQuery: {errors}")
        else:
            logger.info(f"Hyperparameters and columns info successfully inserted into BigQuery table {args.bq_dataset}.{args.bq_table}")

    except Exception as e:
        logger.error(f"Failed to save hyperparameters and columns info to BigQuery: {e}")
