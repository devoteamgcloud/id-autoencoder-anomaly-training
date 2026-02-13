# task.py
import argparse
import os
import time
import uuid
import re
from typing import List
from datetime import datetime, timedelta
import logging
import argparse.Namespace

from google.cloud import bigquery, storage
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.backend import epsilon
import keras.callbacks.History
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def generate_training_report(
    args: argparse.Namespace,
    history: keras.callbacks.History
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
        plt.savefig(f"{args.model_name}_{args.curr_date_str}.png")


def generate_threshold_report(
    args: argparse.Namespace,
    threshold: float,
    all_errors_train: np.ndarray,
    all_errors_val: np.ndarray
) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Define the versions we want to plot
        scales = [('Linear Scale', False), ('Log Scale', True)]
        for i, (title, use_log) in enumerate(scales):
            ax = axes[i]
            sns.histplot(np.random.choice(all_errors_train, size=min(len(all_errors_train), len(all_errors_train)), replace=False), bins='auto', color='blue', label='Train', 
                        kde=True, log_scale=use_log, ax=ax)
            sns.histplot(all_errors_val, bins='auto', color='orange', label='Val', 
                        kde=True, log_scale=use_log, ax=ax)
            ax.axvline(threshold, color='red', linestyle='--', 
                    label=f'Threshold ({args.quantile_threshold})')
            ax.set_title(f'Error Distribution ({title})')
            ax.set_xlabel('Reconstruction Error')
            ax.set_ylabel('Count' if not use_log else 'Count (Log Scale)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{args.model_name}_{args.curr_date_str}_error_distribution.png")
