import os
import uuid
from typing import List, Tuple, Generator
from datetime import timedelta
import logging
import gc

from google.cloud import bigquery, storage
import numpy as np
import pandas as pd
import tensorflow as tf

import trainer.config as config

logger = logging.getLogger(__name__)

# Util Functions
def fetch_data(
    query: str,
    temp_dataset: str,
    bucket_name: str,
    gcs_prefix: str,
    local_dir: str,
    project_id: str = None,
    location: str = None,
) -> List[str]:
    """
    Run filtered query -> save to temp table -> extract parquet to GCS ->
    download locally -> cleanup GCS and temp table.

    Fully scalable for very large datasets.

    Args:
        query: SQL query with filters
        temp_dataset: dataset to store temporary table
        bucket_name: GCS bucket
        gcs_prefix: prefix in GCS
        local_dir: local directory
        project_id: optional
        location: optional

    Returns:
        list of local parquet files
    """

    bq_client = bigquery.Client(project=project_id, location=location)
    storage_client = storage.Client(project=project_id)

    os.makedirs(local_dir, exist_ok=True)

    # create unique temp table
    temp_table_id = f"{project_id}.{temp_dataset}.temp_export_{uuid.uuid4().hex}"

    logger.info(f"Creating temp table: {temp_table_id}")

    # Run query -> temp table
    job_config = bigquery.QueryJobConfig(
        destination=temp_table_id,
        write_disposition="WRITE_TRUNCATE",
    )

    query_job = bq_client.query(query, job_config=job_config)
    query_job.result()

    logger.info("Query complete")

    # Extract temp table -> GCS parquet
    destination_uri = f"gs://{bucket_name}/{gcs_prefix}/part-*.parquet"

    logger.info(f"Extracting to {destination_uri}")

    extract_job = bq_client.extract_table(
        temp_table_id,
        destination_uri,
        job_config=bigquery.ExtractJobConfig(
            destination_format=bigquery.DestinationFormat.PARQUET,
            compression="SNAPPY",
        ),
    )

    extract_job.result()

    logger.info("Extract complete")

    # Download parquet files
    blobs = list(storage_client.list_blobs(bucket_name, prefix=gcs_prefix))

    local_files = []

    for blob in blobs:
        if blob.name.endswith(".parquet"):
            local_path = os.path.join(local_dir, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            local_files.append(local_path)
            logger.info(f"Downloaded {local_path}")

    # Delete GCS files
    logger.info("Deleting GCS temp files...")
    for blob in blobs:
        blob.delete()

    # Delete temp table
    logger.info("Deleting temp table...")
    bq_client.delete_table(temp_table_id, not_found_ok=True)

    logger.info("Cleanup complete")

    return local_files


def fetch_train_and_val(args):
        if args.validation_interval == 0:
            # No validation set, use all data for training
            end_date_train = args.end_train_date.strftime('%Y-%m-%d')
            start_date_train = (args.end_train_date - timedelta(days=args.start_train_interval - 1)).strftime('%Y-%m-%d')
            start_date_val = None
            end_date_val = None
        else:
            # Original logic for when validation_interval > 0
            end_date_train = (args.end_train_date - timedelta(days=args.validation_interval)).strftime('%Y-%m-%d')
            start_date_train = (args.end_train_date - timedelta(days=args.start_train_interval - 1)).strftime('%Y-%m-%d')
            start_date_val = (args.end_train_date - timedelta(days=args.validation_interval - 1)).strftime('%Y-%m-%d')
            end_date_val = args.end_train_date.strftime('%Y-%m-%d')
        
        query_train: str = f"SELECT * FROM `{args.bq_training_data_path}` WHERE (EXTRACT(DATE FROM {args.time_column}) BETWEEN '{start_date_train}' AND '{end_date_train}') AND "
        if start_date_val:
            query_val: str = f"SELECT * FROM `{args.bq_training_data_path}` WHERE (EXTRACT(DATE FROM {args.time_column}) BETWEEN '{start_date_val}' AND '{end_date_val}')"
        else:
            query_val = None

        temp_dataset: str = args.temp_dataset
        bucket_name: str = args.gcs_path
        local_dir_train: str = config.TRAIN_PATH
        local_dir_val:str = config.VAL_PATH
        project_id: str = "finnet-data-platform"
        location: str = "asia-southeast2"
    
        # Fetch training data
        fetch_data(
            query=query_train,
            temp_dataset=temp_dataset,
            bucket_name=bucket_name,
            gcs_prefix="temp",
            local_dir=local_dir_train,
            project_id=project_id,
            location=location,
        )
        # Fetch validation data
        if query_val:
            fetch_data(
                query=query_val,
                temp_dataset=temp_dataset,
                bucket_name=bucket_name,
                gcs_prefix="temp",
                local_dir=local_dir_val,
                project_id=project_id,
                location=location,
            )


def get_features(
    id_columns: List[str],
    drop_columns: List[str],
    impute_columns: List[str],
    log_scale_columns: List[str],
    mmc_encoding_columns: List[str],
    time_column: str
) -> List[str]:
    """
    Get the features of the training data as a list of strings of feature names
    """
    # Assert file exists
    filenames = os.listdir('data')
    assert filenames, "No data found."

    # Read data
    df = pd.read_parquet(f'data/{filenames[0]}')
    features = set(list(df.columns))

    # Dropped Columns
    features -= set(id_columns)
    features -= set(drop_columns)
    features -= set(mmc_encoding_columns)
    features -= {time_column}

    # New Columns
    features = list(features)
    for col in mmc_encoding_columns:
        features += [f'cat-{col}-mean', f'cat-{col}-median', f'cat-{col}-count']
    features += [time_column + '_sin', time_column + '_cos']

    # Free memory
    del df
    gc.collect()

    return features
    

def preprocess(
    df: pd.DataFrame,
    features: List[str],
    id_columns: List[str],
    drop_columns: List[str],
    impute_columns: List[str],
    log_scale_columns: List[str],
    mmc_encoding_columns: List[str],
    time_column: str
) -> pd.DataFrame:      
    """Preprocess the dataframe into all-numeric and normalized values"""
    # Drop ID columns
    # TODO: Save the ID mapping
    df.drop(id_columns, inplace=True, axis=1)

    # Drop drop-columns
    df.drop(drop_columns, inplace=True, axis=1)

    # Fill NA
    df.fillna(value=0.0, inplace=True)

    # Perform Mean - Median - Count encoding
    # TODO: Save the encoding mapping
    for col in mmc_encoding_columns:
        df[f'cat-{col}-mean'] = df[col].map(np.log10(df.groupby(col)['amount'].mean()+1).astype(np.float32).to_dict()).fillna(0.0)
        df[f'cat-{col}-median'] = df[col].map(np.log10(df.groupby(col)['amount'].quantile(0.5)+1).astype(np.float32).to_dict()).fillna(0.0)
        df[f'cat-{col}-count'] = df[col].map(np.log10(df.groupby(col)['amount'].count()+1).to_dict()).fillna(0.0)
    df.drop(mmc_encoding_columns, inplace=True, axis=1)

    # Perform Log Scaling
    df[log_scale_columns] = np.log10(df[log_scale_columns]+1)

    # Convert to time phasor
    df['time_second'] = df[time_column].dt.hour*3600 + df[time_column].dt.minute*60 + df[time_column].dt.second
    df[time_column + '_sin'] = np.sin(df['time_second'] * (np.pi/43200))
    df[time_column + '_cos'] = np.cos(df['time_second'] * (np.pi/43200))
    df.drop([time_column, 'time_second'], inplace=True, axis=1)

    # Create feature list
    if len(features) == 0:
        features = list(df.column)
    assert len(features) != 0, "Failed to build feature list"
    
    return df[features]


def get_train_generator_and_val_set(
    features: List[str],
    id_columns: List[str],
    drop_columns: List[str],
    impute_columns: List[str],
    log_scale_columns: List[str],
    mmc_encoding_columns: List[str],
    time_column: str,
) -> Tuple[Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None], pd.DataFrame]:
    """Return the generator function for tensorflow"""
    # Train Generator

    kwargs = {
        "features": features,
        "id_columns": id_columns,
        "drop_columns":drop_columns,
        "impute_columns": impute_columns,
        "log_scale_columns": log_scale_columns,
        "mmc_encoding_columns": mmc_encoding_columns,
        "time_column": time_column
    }

    def _generator():
        filenames = os.listdir(config.TRAIN_PATH)
    
        for filename in filenames:
            # Clean garbage
            gc.collect()

            # Preprocess data
            df = preprocess(
                df=pd.read_parquet(f'{config.TRAIN_PATH}/{filename}'),
                **kwargs
            )
            df = df.astype(np.float32)
            yield df, df  # Return twice for autoencoder architecture
            del df
            gc.collect()

    # Val Dataset
    val_df = preprocess(
        pd.read_parquet(f'{config.VAL_PATH}/'),
        **kwargs
    )
    return _generator, val_df


def create_tf_dataset(
        generator: Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None],
        val_df: pd.DataFrame,
        features: List[str],
        batch_size: int = 1024
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            # The 'None' here allows variable rows per file
            tf.TensorSpec(shape=(None, len(features)), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(features)), dtype=tf.float32)
        )
    )
    
    train_dataset = train_dataset.unbatch()
    train_dataset = train_dataset.shuffle(buffer_size=10000) 
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(val_df)
    
    return train_dataset, val_dataset
