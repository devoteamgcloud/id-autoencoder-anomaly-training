# task.py
import logging
import argparse
from typing import Any, List
import os
import json

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

import trainer.config as config

logger = logging.getLogger(__name__)

matplotlib.use('Agg')

def generate_training_report(
    args: argparse.Namespace,
    history: Any
) -> None:
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot training & validation loss
        axes[0].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Plot training & validation R2 Score
        axes[1].plot(history.history['r2_score'], label='Training R2 Score')
        if 'val_r2_score' in history.history:
            axes[1].plot(history.history['val_r2_score'], label='Validation R2 Score')
        axes[1].set_title('Model R2 Score')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('R2 Score')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        if not os.path.exists(config.MODEL_PATH):
            os.makedirs(config.MODEL_PATH)
        plt.savefig(f"{config.MODEL_PATH}/loss.png")
        # plt.close(fig)


def generate_threshold_report(
    args: argparse.Namespace,
    threshold: float,
    all_errors_train: np.ndarray,
    all_errors_val: np.ndarray
) -> None:
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Data to be used for visualization
    train_size = min(len(all_errors_train), 1_500_000)
    val_size = min(len(all_errors_val), 1_500_000)
    
    train_data = np.random.choice(all_errors_train, size=train_size, replace=False) + 1e-9
    val_data = np.random.choice(all_errors_val, size=val_size, replace=False) + 1e-9
    
    if len(all_errors_train) >= 1_500_000:
        logger.info(f'Train data is too big ({len(all_errors_train):,} >= 1,500,000), using sample data')
    if len(all_errors_val) >= 1_500_000:
        logger.info(f'Validation data is too big ({len(all_errors_val):,} >= 1,500,000), using sample data')

    scales = [('Linear Scale', False), ('Log Scale', True)]
    
    for i, (title, use_log) in enumerate(scales):
        ax = axes[i]
        
        sns.histplot(train_data, bins=50, color='blue', label='Train', 
                     kde=False, log_scale=use_log, ax=ax)
        sns.histplot(val_data, bins=50, color='orange', label='Val', 
                     kde=False, log_scale=use_log, ax=ax)
        
        ax.axvline(threshold, color='red', linestyle='--', 
                   label=f'Threshold ({args.quantile_threshold})')
        
        ax.set_title(f'Error Distribution ({title})')
        ax.set_xlabel('Reconstruction Error')
        ax.set_ylabel('Count' if not use_log else 'Count (Log Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(f"{config.MODEL_PATH}/error_distribution.png")
    # plt.close(fig)


def generate_hyperparameters_report(
    args: argparse.Namespace,
    threshold: float,
    features: List[str]
) -> None:
    hyperparams = {
        'batch-size': args.batch_size,
        'epochs': args.epochs,
        'n-hidden': args.n_hidden,
        'latent-dim': args.latent_dim,
        'activation': args.activation,
        'learning-rate': args.learning_rate,
        'quantile': args.quantile_threshold,
        'threshold': threshold
    }

    with open(f"{config.MODEL_PATH}/hyperparameters.txt", "w") as f:
        f.write(json.dumps(hyperparams, indent=4))
    

    # Specific for this model, also safe the columns used for training & taken group info
    columns_info = {
        'features': features,
        'id-columns': args.id_columns,
        'drop-columns': args.drop_columns,
        'impute-columns': args.impute_columns,
        'log-scale-columns': args.log_scale_columns,
        'mmc-encoding-columns': args.mmc_encoding_columns,
        'time-column': args.time_column,
        'taken-group': args.taken_group
    }

    with open(f"{config.MODEL_PATH}/columns_info.json", "w") as f:
        f.write(json.dumps(columns_info, indent=4))
