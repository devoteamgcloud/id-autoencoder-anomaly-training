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
        
        # Plot 2: Training & Validation MAPE
        # Key name differs: single-output model → 'mape'; multi-head → 'output-0_mape'
        mape_key     = 'mape'         if 'mape'         in history.history else 'output-0_mape'
        val_mape_key = 'val_mape'     if 'val_mape'     in history.history else 'val_output-0_mape'
        if mape_key in history.history:
            axes[1].plot(history.history[mape_key], label='Training MAPE')
            if val_mape_key in history.history:
                axes[1].plot(history.history[val_mape_key], label='Validation MAPE')
            axes[1].set_title('Output-0 MAPE')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAPE')
            axes[1].legend()
            axes[1].grid(True)

        # Plot 3: Accuracy (multi-head) or best-val-loss marker (single-head)
        train_acc_keys = [k for k in history.history if 'output-' in k and '_accuracy' in k and not k.startswith('val_')]
        val_acc_keys   = [k for k in history.history if k.startswith('val_output-') and '_accuracy' in k]

        if train_acc_keys:
            train_acc_avg, val_acc_avg = [], []
            n_epochs = len(history.history[train_acc_keys[0]])
            for ep in range(n_epochs):
                train_acc_avg.append(sum(history.history[k][ep] for k in train_acc_keys) / len(train_acc_keys))
                if val_acc_keys:
                    val_acc_avg.append(sum(history.history[k][ep] for k in val_acc_keys) / len(val_acc_keys))
            axes[2].plot(train_acc_avg, label='Training Accuracy (avg)')
            if val_acc_avg:
                axes[2].plot(val_acc_avg, label='Validation Accuracy (avg)')
            axes[2].set_title('Average Output-{i} Accuracy')
            axes[2].set_ylabel('Accuracy')
        else:
            # Single-head: show train vs val loss together with best-epoch marker
            axes[2].plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                axes[2].plot(history.history['val_loss'], label='Validation Loss')
                best_ep = int(np.argmin(history.history['val_loss']))
                best_val = history.history['val_loss'][best_ep]
                axes[2].scatter([best_ep], [best_val], color='red', zorder=5,
                                label=f'best val (ep {best_ep}, {best_val:.3f})')
            axes[2].set_title('Loss (best val epoch)')
            axes[2].set_ylabel('Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if not os.path.exists(config.MODEL_PATH):
            os.makedirs(config.MODEL_PATH)
        plt.savefig(f"{config.MODEL_PATH}/loss.png", dpi=300, bbox_inches='tight')


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
        'signed-log-columns': getattr(args, 'signed_log_columns', []),
        'stat-encoding-columns': args.stat_encoding_columns,
        'periodic-columns': args.periodic_columns,
        'ohe-columns': args.ohe_columns,
        'time-column': args.time_column,
        'taken-group': args.taken_group
    }

    with open(f"{config.MODEL_PATH}/columns_info.json", "w") as f:
        f.write(json.dumps(columns_info, indent=4))
