import os
import uuid
from typing import Dict, List, Tuple, Generator, Union, Literal
from datetime import timedelta
import logging
import gc
import json
import re

from google.cloud import bigquery, storage
import numpy as np
import pandas as pd
import tensorflow as tf
from tdigest import TDigest  # For quantile estimation

import trainer.config as config

logger = logging.getLogger(__name__)

# Constants
NUMERICAL_PATTERN = r'-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'
PERIODIC_COLUMN_PATTERN = rf'^\s*({NUMERICAL_PATTERN}):({NUMERICAL_PATTERN})\s*$'

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
        
        # Compile filter
        group_filter = f"is_aggregator_merchant = {args.taken_group}" if args.taken_group >= 0 else "true"

        query_train: str = f"SELECT * FROM `{args.bq_training_data_path}` WHERE (EXTRACT(DATE FROM {args.time_column}) BETWEEN '{start_date_train}' AND '{end_date_train}') AND {group_filter}"
        if start_date_val:
            query_val: str = f"SELECT * FROM `{args.bq_training_data_path}` WHERE (EXTRACT(DATE FROM {args.time_column}) BETWEEN '{start_date_val}' AND '{end_date_val}') AND {group_filter}"
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
    stat_encoding_columns: List[str],
    periodic_columns: List[Tuple[str, Union[float, str, Literal["time", "year"]]]]
) -> List[str]:
    """
    Get the features of the training data as a list of strings of feature names
    """
    # Assert file exists
    filenames = os.listdir(config.TRAIN_PATH)
    assert filenames, "No data found."

    # Read data
    df = pd.read_parquet(os.path.join(config.TRAIN_PATH, filenames[0]))
    raw_features = list(df.columns)
    features = set(raw_features)

    # Dropped Columns
    features -= set(id_columns)
    features -= set(drop_columns)
    features -= set(stat_encoding_columns)
    features -= set(col for col, _ in periodic_columns)

    # New Columns
    features = list(features)
    for col in stat_encoding_columns:  # Stat Encoding
        features += [
            f'cat-{col}-{feat}'
            for feat in
            ['mean', 'count', 'min', 'max', *[f'q{i}' for i in range(10,91,10)]]
        ]
    for col, period in periodic_columns:  # Periodic columns
        infix = ""
        if period == "time":
            infix = "_time"
        elif period == "year":
            infix = "_year"
        features += [col + infix + "_sin", col + infix + "_cos"]

    # Free memory
    del df
    gc.collect()

    return features, raw_features
    

def preprocess(
    df: pd.DataFrame,
    features: List[str],
    id_columns: List[str],
    drop_columns: List[str],
    impute_columns: List[str],
    log_scale_columns: List[str],
    stat_encoding_columns: List[str],
    periodic_columns: List[Tuple[str, Union[float, str, Literal["time", "year"]]]],
    stat_mapping: Dict[str, Dict[str, Dict[str, float]]]
) -> pd.DataFrame:      
    """Preprocess the dataframe into all-numeric and normalized values"""
    # Drop ID columns
    df.drop(id_columns, inplace=True, axis=1)

    # Drop drop-columns
    df.drop(drop_columns, inplace=True, axis=1)

    # Fill NA
    df.fillna(value=0.0, inplace=True)


    # Perform stat encoding
    for col in stat_encoding_columns:
        for feat in ['mean', 'count', 'min', 'max', *[f'q{i}' for i in range(10,91,10)]]:
            df[f'cat-{col}-{feat}'] = df[col].map(stat_mapping[col][feat]).fillna(0.0)
    df.drop(stat_encoding_columns, inplace=True, axis=1)

    # Perform Log Scaling
    df[log_scale_columns] = np.log10(df[log_scale_columns]+1)

    # Convert periodic columns to phasor
    for col, period in periodic_columns:

        if isinstance(period, (float, int)):
            scaled = df[col] / period * 2*np.pi
            infix = ""
        elif isinstance(period, str) and period not in {"time", "year"} and re.search(PERIODIC_COLUMN_PATTERN, period):
            start, end = re.findall(PERIODIC_COLUMN_PATTERN, period)[0]
            start, end = float(start), float(end)
            scaled = (df[col] - start) / (end-start) * 2*np.pi
            infix = ""
        elif period == "time":
            scaled = df[col].dt.hour*3600 + df[col].dt.minute*60 + df[col].dt.second
            scaled /= 24 * 60 * 60  # Number of seconds in one day
            scaled *= 2*np.pi
            infix = "_time"
        elif period == "year":
            scaled = df[col].dt.dayofyear / 365.25 * 2*np.pi
            infix = "_year"
        else:
            raise RuntimeError(f"Periodic columns period info should be int, float, \"time\", \"year\", or {PERIODIC_COLUMN_PATTERN}. Found \"{period}\" instead.")
        
        df[col + infix + '_sin'] = np.sin(scaled)
        df[col + infix + '_cos'] = np.cos(scaled)

    df.drop(list(set(col for col, _ in periodic_columns)), inplace=True, axis=1)
    
    return df[features]


def create_stat_mapping(stat_encoding_columns: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Create the mapping for mean, median, and count encoding for each column in stat_encoding_columns"""
    stat_mapping = {}

    for col in stat_encoding_columns:
        mean_mapping = {}
        count_mapping = {}
        min_mapping = {}
        max_mapping = {}
        quantile_mapping = {}

        logger.info("Start computing statistics...")

        for filename in os.listdir(config.TRAIN_PATH):
            df = pd.read_parquet(
                f'{config.TRAIN_PATH}/{filename}',
                columns=[col, 'amount']
            )

            grouped = df.groupby(col)['amount']

            # sum and count (exact)
            amount_sum = grouped.sum()
            amount_count = grouped.count()
            amount_min = grouped.min()
            amount_max = grouped.max()

            # update sum, count, min, and max
            for key in amount_sum.index:
                mean_mapping[key] = mean_mapping.get(key, 0.0) + amount_sum[key]
                count_mapping[key] = count_mapping.get(key, 0) + amount_count[key]
                min_mapping[key] = amount_min[key]
                max_mapping[key] = amount_max[key]

            # update quantile sketch
            for key, values in grouped:
                if key not in quantile_mapping:
                    quantile_mapping[key] = TDigest()
                quantile_mapping[key].batch_update(values.values)

        # finalize statistics
        final_mean = {
            k: float(np.log10(mean_mapping[k] / count_mapping[k] + 1))
            for k in mean_mapping
        }

        final_count = {
            k: float(np.log10(count_mapping[k] + 1))
            for k in count_mapping
        }

        final_min = {
            k: float(np.log10(min_mapping[k] + 1))
            for k in min_mapping
        }

        final_max = {
            k: float(np.log10(max_mapping[k] + 1))
            for k in max_mapping
        }

        final_quantile = {}  # Final quantile is done differently as it may produce NaN
        for i in range(10,91,10):
            final_quantile[f'q{i}'] = {}
            for k in quantile_mapping:
                result = float(np.log10(quantile_mapping[k].percentile(i) + 1))
                if np.isnan(result):
                    result = min_mapping + (max_mapping[k] - min_mapping[k]) * i/100
                final_quantile[f'q{i}'][k] = np.log10(result+1)
        

        stat_mapping[col] = {
            'mean': final_mean,
            'count': final_count,
            'min': final_min,
            'max': final_max,
            **final_quantile
        }
    
    return stat_mapping


def get_train_generator_and_val_set(
    features: List[str],
    id_columns: List[str],
    drop_columns: List[str],
    impute_columns: List[str],
    log_scale_columns: List[str],
    stat_encoding_columns: List[str],
    periodic_columns: List[Tuple[str, Union[float, str, Literal["time", "year"]]]]

) -> Tuple[Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None], pd.DataFrame]:
    """Return the generator function for tensorflow"""
    # Train Generator

    kwargs = {
        "features": features,
        "id_columns": id_columns,
        "drop_columns": drop_columns,
        "impute_columns": impute_columns,
        "log_scale_columns": log_scale_columns,
        "stat_encoding_columns": stat_encoding_columns,
        "periodic_columns": periodic_columns
    }

    # Generate stat encoding mapping
    stat_mapping = create_stat_mapping(stat_encoding_columns)
    kwargs['stat_mapping'] = stat_mapping

    # Save the stat mapping to a local JSON file for later us in persistence.py
    stat_mapping_path = f"{config.MODEL_PATH}/stat_mapping.json"
    with open(stat_mapping_path, 'w') as f:
        json.dump(stat_mapping, f, indent=4)
             

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

    if len(val_df) > 0:
        val_dataset = tf.data.Dataset.from_tensor_slices(val_df).batch()
    else:
        val_dataset = None
    
    return train_dataset, val_dataset
