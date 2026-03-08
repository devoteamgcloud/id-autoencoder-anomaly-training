import argparse
from typing import List, Tuple, Dict, Callable
from datetime import datetime
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.backend import epsilon
import numpy as np
import pandas as pd

import trainer.config as config

logger = logging.getLogger(__name__)

import tensorflow as tf
from tqdm import tqdm
import numpy as np

class CleanProgress(tf.keras.callbacks.Callback):
    def __init__(self, num_heads, update_freq='epoch'):
        """
        Args:
            num_heads (int): Total number of output heads
            update_freq (str or int): 'epoch', 'batch', or int (every N batches)
        """
        self.num_heads = num_heads
        self.update_freq = update_freq
        self.pbar = None
        
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.steps = self.params['steps']
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        if self.update_freq != 'epoch':
            desc = f"Epoch {epoch+1}/{self.epochs}"
            self.pbar = tqdm(total=self.steps, desc=desc, leave=False)
            
    def on_batch_end(self, batch, logs=None):
        if self.update_freq == 'batch' or (isinstance(self.update_freq, int) and batch % self.update_freq == 0):
            if self.pbar:
                # Show key metrics in progress bar
                postfix = {
                    'loss': f"{logs.get('loss', 0):.4f}",
                    'mape': f"{logs.get('output-0_mape', 0):.3f}",
                }
                
                # Look for any classification metric
                if self.num_heads > 1:
                    # Try different possible metric names
                    for metric_suffix in ['accuracy', 'categorical_accuracy', 'sparse_categorical_accuracy']:
                        acc_key = f'output-1_{metric_suffix}'
                        if acc_key in logs and logs[acc_key] > 0:
                            postfix['acc'] = f"{logs.get(acc_key, 0):.3f}"
                            break
                    
                self.pbar.set_postfix(postfix)
                self.pbar.update(1)
    
    def on_epoch_end(self, epoch, logs=None):
        if self.pbar:
            self.pbar.close()
            
        print(f"\n{'='*90}")
        print(f"EPOCH {epoch+1:3d} SUMMARY")
        print(f"{'='*90}")
        
        # Overall metrics
        print(f"Total Loss: {logs.get('loss', 0):.6f} | Val Loss: {logs.get('val_loss', 0):.6f}")
        
        # Debug: Print all available metrics for first epoch
        if epoch == 0:
            print(f"\nDEBUG - Available metrics:")
            for key in sorted(logs.keys()):
                print(f"  {key}: {logs[key]:.6f}")
            print()
        
        # Detailed head-by-head breakdown
        print(f"\n{'Head':<6} {'Type':<12} {'Train Loss':<12} {'Val Loss':<12} {'Train Metric':<16} {'Val Metric':<16}")
        print(f"{'-'*90}")
        
        for i in range(self.num_heads):
            head_name = f"output-{i}"
            
            # Head type
            head_type = "Regression" if i == 0 else "Classification"
            
            # Loss values
            train_loss_key = f"{head_name}_loss"
            val_loss_key = f"val_{head_name}_loss"
            train_loss = logs.get(train_loss_key, 0)
            val_loss = logs.get(val_loss_key, 0)
            
            # Metric values
            if i == 0:  # Regression head
                train_metric_key = f"{head_name}_mape"
                val_metric_key = f"val_{head_name}_mape"
                train_metric = logs.get(train_metric_key, 0)
                val_metric = logs.get(val_metric_key, 0)
                train_metric_str = f"MAPE: {train_metric:.4f}"
                val_metric_str = f"MAPE: {val_metric:.4f}"
            else:  # Classification head
                # Try to find the correct metric name
                train_metric = 0
                val_metric = 0
                metric_found = False
                
                # Common metric names for classification
                for metric_suffix in ['accuracy', 'categorical_accuracy', 'sparse_categorical_accuracy']:
                    train_metric_key = f"{head_name}_{metric_suffix}"
                    val_metric_key = f"val_{head_name}_{metric_suffix}"
                    
                    if train_metric_key in logs:
                        train_metric = logs.get(train_metric_key, 0)
                        val_metric = logs.get(val_metric_key, 0)
                        metric_found = True
                        break
                
                if metric_found:
                    train_metric_str = f"Acc: {train_metric:.4f}"
                    val_metric_str = f"Acc: {val_metric:.4f}"
                else:
                    train_metric_str = "Acc: N/A"
                    val_metric_str = "Acc: N/A"
            
            # Format the row
            print(f"{i:<6} {head_type:<12} {train_loss:<12.6f} {val_loss:<12.6f} "
                  f"{train_metric_str:<16} {val_metric_str:<16}")
        
        print(f"{'='*90}\n")
        

def create_autoencoder(
    input_dim: int,
    latent_vec_dim: int,
    enc_hidden_layers_num: int,
    feature_slices: List[slice],
    dropout_rate: float = 0.1,
    activation: str = 'relu'
):
    """
    Create an autoencoder model with multiple encoding layers.
    
    Parameters:
    input_dim (int): Input layer dimension
    latent_vec_dim (int): Latent layer dimension
    enc_hidden_layers_num (int): number of layers between input layer and latent layer (exclusive)
    """
    # Input layer
    input_layer = keras.Input(shape=(input_dim,), name='input')

    # Compute dimensions of hidden layers
    hidden_dims = np.linspace(input_dim, latent_vec_dim, enc_hidden_layers_num+2)[1:-1]
    hidden_dims = np.ceil(hidden_dims).astype(int)
    
    # Encoder
    prev_layer = input_layer
    for i, n_dim in enumerate(hidden_dims):
        hidden = layers.Dense(n_dim, kernel_initializer='he_normal', name=f'encoder_{i+1}')(prev_layer)
        hidden = layers.BatchNormalization()(hidden) # Added for stability
        hidden = layers.Activation(activation)(hidden)
        hidden = layers.Dropout(dropout_rate)(hidden)
        prev_layer = hidden

    # Bottleneck layer
    bottleneck = layers.Dense(latent_vec_dim, kernel_initializer='he_normal', name='bottleneck')(prev_layer)
    bottleneck = layers.BatchNormalization()(bottleneck) 
    bottleneck = layers.Activation(activation)(bottleneck)
    
    # Decoder
    prev_layer = bottleneck
    for i, n_dim in enumerate(hidden_dims[::-1]):
        hidden = layers.Dense(n_dim, kernel_initializer='he_normal', name=f'decoder_{i+1}')(prev_layer)
        hidden = layers.BatchNormalization()(hidden)
        hidden = layers.Activation(activation)(hidden)
        hidden = layers.Dropout(dropout_rate)(hidden)
        prev_layer = hidden
    
    losses = {}
    weights = {}
    
    # Output Head: Regression
    ndim_reg = feature_slices[0].stop - feature_slices[0].start
    reg_layer = layers.Dense(ndim_reg, activation='softplus', name='output-0')(prev_layer)
    losses['output-0'] = 'mae'
    weights['output-0'] = 1.0 # Standardize weights first to debug

    # Output Heads: OHE / Binary
    output_layers = [reg_layer]
    for i, slice_ in enumerate(feature_slices[1:], start=1):
        ndim = slice_.stop - slice_.start
        act = 'sigmoid' if ndim == 1 else 'softmax'
        loss_type = 'binary_crossentropy' if ndim == 1 else 'categorical_crossentropy'
        
        out = layers.Dense(ndim, activation=act, name=f'output-{i}')(prev_layer)
        output_layers.append(out)
        
        losses[f'output-{i}'] = loss_type
        # If divergence continues, lower the weight of the categorical heads
        weights[f'output-{i}'] = 0.05
    
    autoencoder = keras.Model(inputs=input_layer, outputs=output_layers)
    return autoencoder, losses, weights


def train_model(
        args: argparse.Namespace, 
        features: List[str],
        train: tf.data.Dataset, 
        val: tf.data.Dataset,
        feature_slices: List[slice]
    ) -> Tuple[keras.Model, keras.callbacks.History, str]:
    """Train the autoencoder model."""

    INPUT_DIM = len(features)
    LATENT_VEC_DIM = args.latent_dim if isinstance(args.latent_dim, int) else int(args.latent_dim * INPUT_DIM)
    N_HIDDEN = args.n_hidden
    autoencoder, losses, weights = create_autoencoder(INPUT_DIM, LATENT_VEC_DIM, N_HIDDEN, feature_slices)

    def mape(y_true, y_pred):
        return tf.reduce_mean((tf.abs(y_true - y_pred) + 0.1) / (tf.abs(y_true) + 0.1))
    
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=losses,
        loss_weights=weights,
        # metrics=[r2_score],
        metrics = {'output-0': mape, **{f'output-{i}': 'accuracy' for i in range(1,len(feature_slices))}}
    )

    # -- Define callbacks
    model_name = f"{args.model_name}_{args.curr_date_str}{args.postfix}.keras"
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        f"{config.MODEL_PATH}/{model_name}",
        monitor='loss',
        save_best_only=True,
        verbose=1
    )
    csv_logger = tf.keras.callbacks.CSVLogger('training_log.csv', separator=',', append=False)
    # class CleanProgress(tf.keras.callbacks.Callback):
    #     def on_epoch_end(self, epoch, logs=None):
    #         print(f"Epoch {epoch+1}: Total Loss: {logs['loss']:.4f} - R2: {logs['output-0_r2_score']:.4f}")

    callbacks = [CleanProgress(len(feature_slices)), early_stopping, reduce_lr, model_checkpoint, csv_logger]

    logger.info("Starting training...")
    try:
        history = autoencoder.fit(
            train,
            validation_data=val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=0
        )
        logger.info("Training completed!")
    except Exception as e:
        logger.error(e)
        history = []
    
    return autoencoder, history, model_name


def find_threshold(
        args: argparse.Namespace,
        autoencoder: keras.Model,
        train: tf.data.Dataset,
        val_df: Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> Tuple[float, np.ndarray, np.ndarray]:

    # Function for reconcat separated dataframes
    def reconcat(result: List):
        np_list = []
        for tensor in result:
            np_list.append(tensor.numpy())
        concatenated = np.concatenate(np_list, axis=1)
        return concatenated

    # Compute train error
    all_errors_train = []
    for batch in train:
        # Unpack the batch, the data is in the form of (df, Dict[df])
        x_batch = batch[0]
        reconstruction = reconcat(autoencoder(x_batch, training=False))
        loss = np.sqrt(np.mean(np.square(x_batch - reconstruction), axis=1))
        all_errors_train.append(loss)
    all_errors_train = np.concatenate(all_errors_train, axis=0)
    # Compute val error
    try:
        val_reconstruction = reconcat(autoencoder(val_df[0], training=False))
        val_loss = np.sqrt(np.mean(np.square(val_reconstruction - val_df), axis=1))
        all_errors_val = val_loss
    except Exception as e:
        all_errors_val = np.array([])

    # Find threshold based on quantile of training error distribution
    threshold = np.quantile(all_errors_train, args.quantile_threshold)
    logger.info(f"Anomaly detection threshold set at: {threshold}")

    return threshold, all_errors_train, all_errors_val
import argparse
from typing import List, Tuple, Dict, Callable
from datetime import datetime
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.backend import epsilon
import numpy as np
import pandas as pd

import trainer.config as config

logger = logging.getLogger(__name__)

import tensorflow as tf
from tqdm import tqdm
import numpy as np

class CleanProgress(tf.keras.callbacks.Callback):
    def __init__(self, num_heads, update_freq='epoch'):
        """
        Args:
            num_heads (int): Total number of output heads
            update_freq (str or int): 'epoch', 'batch', or int (every N batches)
        """
        self.num_heads = num_heads
        self.update_freq = update_freq
        self.pbar = None
        
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.steps = self.params['steps']
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        if self.update_freq != 'epoch':
            desc = f"Epoch {epoch+1}/{self.epochs}"
            self.pbar = tqdm(total=self.steps, desc=desc, leave=False)
            
    def on_batch_end(self, batch, logs=None):
        if self.update_freq == 'batch' or (isinstance(self.update_freq, int) and batch % self.update_freq == 0):
            if self.pbar:
                # Show key metrics in progress bar
                postfix = {
                    'loss': f"{logs.get('loss', 0):.4f}",
                    'mape': f"{logs.get('output-0_mape', 0):.3f}",
                }
                
                # Look for any classification metric
                if self.num_heads > 1:
                    # Try different possible metric names
                    for metric_suffix in ['accuracy', 'categorical_accuracy', 'sparse_categorical_accuracy']:
                        acc_key = f'output-1_{metric_suffix}'
                        if acc_key in logs and logs[acc_key] > 0:
                            postfix['acc'] = f"{logs.get(acc_key, 0):.3f}"
                            break
                    
                self.pbar.set_postfix(postfix)
                self.pbar.update(1)
    
    def on_epoch_end(self, epoch, logs=None):
        if self.pbar:
            self.pbar.close()
            
        print(f"\n{'='*90}")
        print(f"EPOCH {epoch+1:3d} SUMMARY")
        print(f"{'='*90}")
        
        # Overall metrics
        print(f"Total Loss: {logs.get('loss', 0):.6f} | Val Loss: {logs.get('val_loss', 0):.6f}")
        
        # Debug: Print all available metrics for first epoch
        if epoch == 0:
            print(f"\nDEBUG - Available metrics:")
            for key in sorted(logs.keys()):
                print(f"  {key}: {logs[key]:.6f}")
            print()
        
        # Detailed head-by-head breakdown
        print(f"\n{'Head':<6} {'Type':<12} {'Train Loss':<12} {'Val Loss':<12} {'Train Metric':<16} {'Val Metric':<16}")
        print(f"{'-'*90}")
        
        for i in range(self.num_heads):
            head_name = f"output-{i}"
            
            # Head type
            head_type = "Regression" if i == 0 else "Classification"
            
            # Loss values
            train_loss_key = f"{head_name}_loss"
            val_loss_key = f"val_{head_name}_loss"
            train_loss = logs.get(train_loss_key, 0)
            val_loss = logs.get(val_loss_key, 0)
            
            # Metric values
            if i == 0:  # Regression head
                train_metric_key = f"{head_name}_mape"
                val_metric_key = f"val_{head_name}_mape"
                train_metric = logs.get(train_metric_key, 0)
                val_metric = logs.get(val_metric_key, 0)
                train_metric_str = f"MAPE: {train_metric:.4f}"
                val_metric_str = f"MAPE: {val_metric:.4f}"
            else:  # Classification head
                # Try to find the correct metric name
                train_metric = 0
                val_metric = 0
                metric_found = False
                
                # Common metric names for classification
                for metric_suffix in ['accuracy', 'categorical_accuracy', 'sparse_categorical_accuracy']:
                    train_metric_key = f"{head_name}_{metric_suffix}"
                    val_metric_key = f"val_{head_name}_{metric_suffix}"
                    
                    if train_metric_key in logs:
                        train_metric = logs.get(train_metric_key, 0)
                        val_metric = logs.get(val_metric_key, 0)
                        metric_found = True
                        break
                
                if metric_found:
                    train_metric_str = f"Acc: {train_metric:.4f}"
                    val_metric_str = f"Acc: {val_metric:.4f}"
                else:
                    train_metric_str = "Acc: N/A"
                    val_metric_str = "Acc: N/A"
            
            # Format the row
            print(f"{i:<6} {head_type:<12} {train_loss:<12.6f} {val_loss:<12.6f} "
                  f"{train_metric_str:<16} {val_metric_str:<16}")
        
        print(f"{'='*90}\n")
        

def create_autoencoder(
    input_dim: int,
    latent_vec_dim: int,
    enc_hidden_layers_num: int,
    feature_slices: List[slice],
    dropout_rate: float = 0.1,
    activation: str = 'relu'
):
    """
    Create an autoencoder model with multiple encoding layers.
    
    Parameters:
    input_dim (int): Input layer dimension
    latent_vec_dim (int): Latent layer dimension
    enc_hidden_layers_num (int): number of layers between input layer and latent layer (exclusive)
    """
    # Input layer
    input_layer = keras.Input(shape=(input_dim,), name='input')

    # Compute dimensions of hidden layers
    hidden_dims = np.linspace(input_dim, latent_vec_dim, enc_hidden_layers_num+2)[1:-1]
    hidden_dims = np.ceil(hidden_dims).astype(int)
    
    # Encoder
    prev_layer = input_layer
    for i, n_dim in enumerate(hidden_dims):
        hidden = layers.Dense(n_dim, kernel_initializer='he_normal', name=f'encoder_{i+1}')(prev_layer)
        hidden = layers.BatchNormalization()(hidden) # Added for stability
        hidden = layers.Activation(activation)(hidden)
        hidden = layers.Dropout(dropout_rate)(hidden)
        prev_layer = hidden

    # Bottleneck layer
    bottleneck = layers.Dense(latent_vec_dim, kernel_initializer='he_normal', name='bottleneck')(prev_layer)
    bottleneck = layers.BatchNormalization()(bottleneck) 
    bottleneck = layers.Activation(activation)(bottleneck)
    
    # Decoder
    prev_layer = bottleneck
    for i, n_dim in enumerate(hidden_dims[::-1]):
        hidden = layers.Dense(n_dim, kernel_initializer='he_normal', name=f'decoder_{i+1}')(prev_layer)
        hidden = layers.BatchNormalization()(hidden)
        hidden = layers.Activation(activation)(hidden)
        hidden = layers.Dropout(dropout_rate)(hidden)
        prev_layer = hidden
    
    losses = {}
    weights = {}
    
    # Output Head: Regression
    ndim_reg = feature_slices[0].stop - feature_slices[0].start
    reg_layer = layers.Dense(ndim_reg, activation='softplus', name='output-0')(prev_layer)
    losses['output-0'] = 'mae'
    weights['output-0'] = 1.0 # Standardize weights first to debug

    # Output Heads: OHE / Binary
    output_layers = [reg_layer]
    for i, slice_ in enumerate(feature_slices[1:], start=1):
        ndim = slice_.stop - slice_.start
        act = 'sigmoid' if ndim == 1 else 'softmax'
        loss_type = 'binary_crossentropy' if ndim == 1 else 'categorical_crossentropy'
        
        out = layers.Dense(ndim, activation=act, name=f'output-{i}')(prev_layer)
        output_layers.append(out)
        
        losses[f'output-{i}'] = loss_type
        # If divergence continues, lower the weight of the categorical heads
        weights[f'output-{i}'] = 0.05
    
    autoencoder = keras.Model(inputs=input_layer, outputs=output_layers)
    return autoencoder, losses, weights


def train_model(
        args: argparse.Namespace, 
        features: List[str],
        train: tf.data.Dataset, 
        val: tf.data.Dataset,
        feature_slices: List[slice]
    ) -> Tuple[keras.Model, keras.callbacks.History, str]:
    """Train the autoencoder model."""

    INPUT_DIM = len(features)
    LATENT_VEC_DIM = args.latent_dim if isinstance(args.latent_dim, int) else int(args.latent_dim * INPUT_DIM)
    N_HIDDEN = args.n_hidden
    autoencoder, losses, weights = create_autoencoder(INPUT_DIM, LATENT_VEC_DIM, N_HIDDEN, feature_slices)

    def mape(y_true, y_pred):
        return tf.reduce_mean((tf.abs(y_true - y_pred) + 0.1) / (tf.abs(y_true) + 0.1))
    
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=losses,
        loss_weights=weights,
        # metrics=[r2_score],
        metrics = {'output-0': mape, **{f'output-{i}': 'accuracy' for i in range(1,len(feature_slices))}}
    )

    # -- Define callbacks
    model_name = f"{args.model_name}_{args.curr_date_str}{args.postfix}.keras"
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        f"{config.MODEL_PATH}/{model_name}",
        monitor='loss',
        save_best_only=True,
        verbose=1
    )
    csv_logger = tf.keras.callbacks.CSVLogger('training_log.csv', separator=',', append=False)
    # class CleanProgress(tf.keras.callbacks.Callback):
    #     def on_epoch_end(self, epoch, logs=None):
    #         print(f"Epoch {epoch+1}: Total Loss: {logs['loss']:.4f} - R2: {logs['output-0_r2_score']:.4f}")

    callbacks = [CleanProgress(len(feature_slices)), early_stopping, reduce_lr, model_checkpoint, csv_logger]

    logger.info("Starting training...")
    try:
        history = autoencoder.fit(
            train,
            validation_data=val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=0
        )
        logger.info("Training completed!")
    except Exception as e:
        logger.error(e)
        history = []
    
    return autoencoder, history, model_name


def find_threshold(
        args: argparse.Namespace,
        autoencoder: keras.Model,
        train: tf.data.Dataset,
        val_df: Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> Tuple[float, np.ndarray, np.ndarray]:

    # Function for reconcat separated dataframes
    def reconcat(result: List):
        np_list = []
        for tensor in result:
            np_list.append(tensor.numpy())
        concatenated = np.concatenate(np_list, axis=1)
        return concatenated

    # Compute train error
    all_errors_train = []
    for batch in train:
        # Unpack the batch, the data is in the form of (df, Dict[df])
        x_batch = batch[0]
        reconstruction = reconcat(autoencoder(x_batch, training=False))
        loss = np.sqrt(np.mean(np.square(x_batch - reconstruction), axis=1))
        all_errors_train.append(loss)
    all_errors_train = np.concatenate(all_errors_train, axis=0)
    # Compute val error
    try:
        val_reconstruction = reconcat(autoencoder(val_df[0], training=False))
        val_loss = np.sqrt(np.mean(np.square(val_reconstruction - val_df), axis=1))
        all_errors_val = val_loss
    except Exception as e:
        all_errors_val = np.array([])

    # Find threshold based on quantile of training error distribution
    threshold = np.quantile(all_errors_train, args.quantile_threshold)
    logger.info(f"Anomaly detection threshold set at: {threshold}")

    return threshold, all_errors_train, all_errors_val
