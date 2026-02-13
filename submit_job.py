# submit_job.py
from google.cloud import aiplatform

# --- CONFIGURATION ---
PROJECT_ID = "your-project-id"
REGION = "us-central1"
BUCKET_URI = "gs://your-bucket-name"  # Used for staging code and saving model artifacts
APP_NAME = "my-training-job"

# Select a pre-built container image based on your framework
# See full list: https://cloud.google.com/vertex-ai/docs/training/pre-built-containers
# Examples:
# Scikit-learn: us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest
# PyTorch:     us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest
# TensorFlow:  us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11:latest
CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest"

def submit_custom_job():
    # 1. Initialize the AI Platform
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=BUCKET_URI
    )

    # 2. Define the Custom Training Job
    # This configures how the code is packaged and what container runs it
    job = aiplatform.CustomTrainingJob(
        display_name=APP_NAME,
        script_path="task.py",        # Path to your local training script
        container_uri=CONTAINER_URI,  # The pre-built container
        requirements=["pandas", "numpy"], # Pip packages to install on the VM
    )

    print("Submitting job...")

    # 3. Submit and Run the job
    # This launches the VM, installs requirements, and runs task.py
    job.run(
        machine_type="n1-standard-4", # e.g., n1-standard-4, n1-standard-8
        # accelerator_type="NVIDIA_TESLA_T4", # Uncomment if using GPU
        # accelerator_count=1,                # Uncomment if using GPU
        replica_count=1,
        args=[
            "--epochs=10",
            "--learning-rate=0.05"
        ],
        sync=True # Set to False if you don't want to wait for the job to finish in the terminal
    )

if __name__ == '__main__':
    submit_custom_job()