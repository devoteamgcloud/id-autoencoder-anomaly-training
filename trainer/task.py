# task.py
import argparse
from datetime import datetime
import logging
import os

from utils.data_loader import fetch_train_and_val, get_features, get_train_generator_and_val_set, create_tf_dataset
from utils.train import train_model, find_threshold
from utils.persistence import save_model_and_reports
from utils.report_generator import generate_training_report, generate_threshold_report
from args_validator import int_or_float, valid_bq_path, valid_datetime

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_training_pipeline(args: argparse.Namespace):
    """Orchestrator for training process"""

    # Step 1: Fetch data from BigQuery with date filters and save as local parquet files 
    logger.info("Starting data fetching process...")
    if args.get_new_data:
        fetch_train_and_val(args)
    else:
        data_exists = os.path.exists('train_data') and os.path.exists('val_data')
        if data_exists:
            logger.info("Using existing local parquet files for training and validation data.")
        else:
            logger.info("Local parquet files not found. Fetching data from BigQuery.")
            fetch_train_and_val(args)

    # Step 2: Generate generator and dataset object
    logger.info("Preparing data generators and TensorFlow datasets...")
    kwargs = {
        'id_columns': args.id_columns,
        'drop_columns': args.drop_columns,
        'impute_columns': args.impute_columns,
        'log_scale_columns': args.log_scale_columns,
        'mmc_encoding_columns': args.mmc_encoding_columns,
        'time_column': args.time_column
    }
    features = get_features(**kwargs)
    generator, val_df = get_train_generator_and_val_set(features, **kwargs)
    train, val = create_tf_dataset(generator, val_df, features, args.batch_size)

    # Step 3: Start training process, get threshold and reconstruction errors
    logger.info("Starting model training...")
    autoencoder, history, model_name = train_model(args, features, train, val)

    # Step 4: Generate training report
    logger.info("Generating training and threshold reports...")
    generate_training_report(args, history)
    threshold, all_errors_train, all_errors_val = find_threshold(autoencoder, train, val_df, history, args)
    generate_threshold_report(args, threshold, all_errors_train, all_errors_val)

    # Step 5: Save model and training report to GCS
    logger.info("Saving model and reports to GCS...")
    save_model_and_reports(args, model_name, history)


if __name__ == '__main__':

    # ########## Parse Arguments
    parser = argparse.ArgumentParser()

    # Project metadata
    parser.add_argument('--project-id', type=str, default=None, help='GCP Project ID')

    # Paths
    parser.add_argument('--gcs-path', type=str, required=True, help='Destination GCS path for model training results, temporary data dump, and evaluation')
    parser.add_argument('--bq-training-data-path', type=valid_bq_path, required=True, help='ML Training source table path in BigQuery')

    # Data date range
    parser.add_argument('--end-train-date', type=valid_datetime, default='', help='Ending date filter for the training data in the format of YYYY-MM-DD')
    parser.add_argument('--start-train-interval', type=int, default=90, help='Starting date filter for the training data in the form of how many days before the end date')
    parser.add_argument('--validation-interval', type=int, default=1, help='Number of days taken (from most recent data) for validation dataset')

    # Column Names
    parser.add_argument('--id-columns', nargs='+', required=True, help='List of identifier columns')
    parser.add_argument('--drop-columns', nargs='+', required=True, help='List of columns that want to be dropped')
    parser.add_argument('--impute-columns', nargs='+', required=True, help='List of columns that need to be imputed with 0')
    parser.add_argument('--log-scale-columns', nargs='+', required=True, help='List of columns that need log normalization')
    parser.add_argument('--mmc-encoding-columns', nargs='+', required=True, help='List of high cardinality categorical columns that need to be encoded to mean, median, and count')
    parser.add_argument('--time-column', type=str, required=True, help='Column that identifies transaction time')
    
    # Training Hyperparams
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the training process')
    parser.add_argument('--n-hidden', type=int, default=3, help='Number of hidden layers between (exclusive): (1) input layer and latent space layer, (2) latent space layer and output layer')
    parser.add_argument('--latent-dim', type=int_or_float, default=0.5, help='Dimension of latent space. If float between (0,1), it will set the dimension to d*input_dim. If int, it will set the dimension to the set value')
    parser.add_argument('--activation', type=str, default='relu', help='Type of activation function used in hidden layers')
    parser.add_argument('--quantile-threshold', type=float, default=0.95, help='Quantile threshold for determining anomaly based on reconstruction error distribution')

    # Training batch and epoch
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)

    # Model metadata
    parser.add_argument('--model-name', type=str, default='autoencoder', help='Name of the model to be saved')

    # Misc
    parser.add_argument('--get-new-data', type=bool, default=True, help='Whether to get new data from BigQuery or use existing local parquet files')
    
    args = parser.parse_args()

    # Derived values
    if args.bq_training_data_path:
        # Location for putting the temporary dataset
        args.temp_dataset = args.bq_training_data_path.split('.')[1]
        if not args.project_id:
            args.project_id = args.bq_training_data_path.split('.')[0]
    args.curr_date_str = datetime.now().strftime('%Y%m%d')
    

    # ########## Start process
    try:
        run_training_pipeline(args)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise e