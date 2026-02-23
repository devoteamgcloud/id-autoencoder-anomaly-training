import os
import glob
from google.cloud import aiplatform
from google.cloud import storage

# --- CONFIGURATION ---
PROJECT_ID = "finnet-data-platform"
REGION = "asia-southeast2"
BUCKET_URI = "gs://finnet-model-output-temp" 
APP_NAME = "autoencoder-anomaly-detection"

# GPU Image for Vertex AI
CONTAINER_URI = "asia-docker.pkg.dev/vertex-ai/training/tf-gpu.2-17.py310:latest"

# Task Arguments
kwargs = {
    "gcs-path": "finnet-model-output-temp",
    "bq-training-data-path": "finnet-data-platform.phase2_dev.ml_features",
    "id-columns": ["customer_id", "merchant_id", "cust_email", "cust_msisdn"],
    "drop-columns": ["sof_entropy", "payment_code_entropy", "payment_code", "trx_id", "is_aggregator_merchant"],
    "impute-columns": ["c_amount_std_30d", "c_amount_zscore_30d"],
    "log-scale-columns": [
        "amount", "m_amount_mean_30d", "m_amount_std_30d", "m_tx_count_24h",
        "c_amount_mean_30d", "c_amount_std_30d", "c_tx_count_24h",
        "m_tx_count_2h", "m_amount_sum_2h", "c_tx_count_2h", "c_amount_sum_2h",
        "m_tx_count_30m", "c_tx_count_30m",
        "amount_vs_m_mean", "amount_vs_c_mean", "cm_tx_count_7d"
    ],
    "mmc-encoding-columns": ["sof_id"],
    "time-column": "created_at",
    "epochs": 25,
    "quantile_threshold": 0.95,
    "n_hidden": 3,
    "latent_dim": 0.9,
}

def kwargs_to_list(kwargs):
    """Converts dict to flat list format: ['--arg', 'val', '--list', 'item1', 'item2']"""
    args_list = []
    for key, value in kwargs.items():
        arg_key = f"--{key}"
        args_list.append(arg_key)
        if isinstance(value, list):
            args_list.extend(map(str, value))
        else:
            args_list.append(str(value))
    return args_list

def build_and_upload_package():
    """Builds the source distribution and uploads it to GCS."""
    print("1. Building source distribution...")
    # Clean previous builds
    os.system("rm -rf dist/ *.egg-info")
    # Build package (requires setup.py)
    exit_code = os.system("python3 setup.py sdist --formats=gztar")
    if exit_code != 0:
        raise Exception("Build failed. Check setup.py errors.")

    # Find the built file
    dist_files = glob.glob("dist/*.tar.gz")
    if not dist_files:
        raise Exception("No .tar.gz found in dist/ folder.")
    local_path = dist_files[0]
    filename = os.path.basename(local_path)
    
    # Upload to GCS
    print(f"2. Uploading {filename} to {BUCKET_URI}/trainer_code/...")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket_name = BUCKET_URI.replace("gs://", "").split("/")[0]
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"trainer_code/{filename}")
    blob.upload_from_filename(local_path)
    
    gcs_uri = f"gs://{bucket_name}/trainer_code/{filename}"
    print(f"   Uploaded to: {gcs_uri}")
    return gcs_uri

def submit_package_job():
    # 1. Build and Upload Code First
    python_package_gcs_uri = build_and_upload_package()

    # 2. Initialize the AI Platform
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=BUCKET_URI
    )

    # 3. Define the Custom Python Package Job
    # NOW we have the required 'python_package_gcs_uri'
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=APP_NAME,
        python_package_gcs_uri=python_package_gcs_uri, # <--- FIXED HERE
        python_module_name="trainer.task", 
        container_uri=CONTAINER_URI,
        project=PROJECT_ID,
        staging_bucket=BUCKET_URI,
    )

    print(f"3. Submitting job to {REGION}...")

    # 4. Submit and Run the job
    job.run(
        machine_type="n1-standard-8", 
        accelerator_type="NVIDIA_TESLA_T4", 
        accelerator_count=1,                
        replica_count=1,
        args=kwargs_to_list(kwargs),
        sync=True 
    )

if __name__ == '__main__':
    submit_package_job()
