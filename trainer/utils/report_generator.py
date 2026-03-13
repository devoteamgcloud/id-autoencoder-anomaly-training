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
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        # Plot 1: Training & Validation Loss
        axes[0].plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: Training & Validation Output-0 MAPE
        axes[1].plot(history.history['output-0_mape'], label='Training MAPE')
        if 'val_output-0_mape' in history.history:
            axes[1].plot(history.history['val_output-0_mape'], label='Validation MAPE')
        axes[1].set_title('Output-0 MAPE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAPE')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot 3: Average Training & Validation Accuracy for output-{i}
        # Find all accuracy metrics for output-{i} (assuming there might be multiple)
        train_acc_keys = [key for key in history.history.keys() if 'output-' in key and '_accuracy' in key and 'val_' not in key]
        val_acc_keys = [key for key in history.history.keys() if 'val_output-' in key and '_accuracy' in key]
        
        if train_acc_keys:
            # Calculate average accuracy across all output-{i} metrics
            train_acc_avg = []
            val_acc_avg = []
            
            epochs = len(history.history[train_acc_keys[0]])
            
            for epoch in range(epochs):
                # Average training accuracy for this epoch
                train_epoch_acc = [history.history[key][epoch] for key in train_acc_keys]
                train_acc_avg.append(sum(train_epoch_acc) / len(train_epoch_acc))
                
                # Average validation accuracy for this epoch (if available)
                if val_acc_keys:
                    val_epoch_acc = [history.history[key][epoch] for key in val_acc_keys]
                    val_acc_avg.append(sum(val_epoch_acc) / len(val_epoch_acc))
            
            axes[2].plot(train_acc_avg, label='Training Accuracy (avg)')
            if val_acc_avg:
                axes[2].plot(val_acc_avg, label='Validation Accuracy (avg)')
            
            axes[2].set_title('Average Output-{i} Accuracy')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Accuracy')
            axes[2].legend()
            axes[2].grid(True)
        
        plt.tight_layout()
        
        if not os.path.exists(config.MODEL_PATH):
            os.makedirs(config.MODEL_PATH)
        plt.savefig(f"{config.MODEL_PATH}/training_report.png", dpi=300, bbox_inches='tight')


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
    features: List[str],
    raw_features: List[str],
    feature_slices: List[slice]
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

    with open(f"{config.MODEL_PATH}/hyperparameters.json", "w") as f:
        f.write(json.dumps(hyperparams, indent=4))

    # Specific for this model, also safe the columns used for training & taken group info
    columns_info = {
        'features': features,
        'raw_features': raw_features,
        'feature_slices': [[s.start, s.stop, s.step] for s in feature_slices],
        'columns_dtypes': args.columns_dtypes,
        'id-columns': args.id_columns,
        'drop-columns': args.drop_columns,
        'impute-columns': args.impute_columns,
        'log-scale-columns': args.log_scale_columns,
        'stat-encoding-columns': args.stat_encoding_columns,
        'periodic-columns': args.periodic_columns,
        'ohe-columns': args.ohe_columns,
        'time-column': args.time_column,
        'taken-group': args.taken_group
    }

    with open(f"{config.MODEL_PATH}/columns_info.json", "w") as f:
        f.write(json.dumps(columns_info, indent=4))
