# Anomaly Detection Using Autoencoder + Vertex AI

This project implements an anomaly detection system using autoencoders trained on Google Cloud Vertex AI. An autoencoder is a neural network that learns to compress data into a lower-dimensional representation and then reconstruct it back to the original form. Anomalies are detected by measuring reconstruction errors - data points that are poorly reconstructed are likely to be anomalous.

## Overview

The system trains an autoencoder on normal transaction data from BigQuery and uses reconstruction error thresholds to identify anomalous patterns. The training pipeline includes data preprocessing, feature engineering, model training, and evaluation, all orchestrated through Vertex AI.

## Quick Start

1. **Configure your training parameters** by filling out `config.json`:
   ```bash
   cp config.example.json config.json
   ```
   
   Edit `config.json` with your specific parameters:
   - GCP project and data paths
   - Column specifications for your dataset
   - Training hyperparameters
   - Data date ranges and filtering options

2. **Configure your Vertex AI job specifications** by filling out `jobspec.json`:
   ```bash
   cp jobspec.example.json jobspec.json
   ```
   
   Edit `jobspec.json` with your compute requirements:
   - Machine type and accelerators
   - Service account
   - Resource allocation

3. **Submit the training job** to Vertex AI:
   ```bash
   python3 submit_job.py
   ```

## Configuration Parameters

### Data Configuration
- `project-id`: GCP Project ID
- `gcs-path`: Destination GCS path for model artifacts and temporary data
- `bq-training-data-path`: BigQuery source table path
- `bq-report-path`: BigQuery target table path for report
- `end-train-date`: Training data end date (YYYY-MM-DD)
- `start-train-interval`: Days before end date to start training data (default: 90)
- `validation-interval`: Days for validation dataset (default: 1)

### Feature Engineering
- `id-columns`: List of identifier columns
- `drop-columns`: Columns to exclude from training
- `impute-columns`: Columns to impute with 0
- `log-scale-columns`: Columns requiring log normalization
- `mmc-encoding-columns`: High cardinality categorical columns for mean/median/count encoding
- `time-column`: Transaction timestamp column

### Model Hyperparameters
- `learning-rate`: Training learning rate (default: 0.001)
- `n-hidden`: Number of hidden layers (default: 3)
- `latent-dim`: Latent space dimension (float 0-1 for ratio, int for absolute size)
- `activation`: Hidden layer activation function (default: 'relu')
- `quantile-threshold`: Anomaly detection threshold percentile (default: 0.95)
- `epochs`: Training epochs (default: 100)
- `batch_size`: Training batch size (default: 1024)

### Output Configuration
- `model-name`: Saved model name (default: 'autoencoder')
- `postfix`: Additional identifier for model and reports
- `get-new-data`: Whether to fetch fresh data from BigQuery (default: true)

## Requirements

- Google Cloud SDK configured with appropriate permissions
- Access to BigQuery source data
- Vertex AI API enabled
- Required Python dependencies (see requirements.txt)