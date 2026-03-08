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
    reg_layer = layers.Dense(ndim_reg, activation='linear', name='output-0')(prev_layer)
    losses['output-0'] = 'mse'
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
        weights[f'output-{i}'] = 0.5
    
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

    def r2_score(y_true, y_pred):
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        return 1 - ss_res / (ss_tot + epsilon())
    
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=losses,
        loss_weights=weights,
        metrics=[r2_score]
    )

    # -- Define callbacks
    model_name = f"{args.model_name}_{args.curr_date_str}{args.postfix}.keras"
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        f"{config.MODEL_PATH}/{model_name}",
        monitor='loss',
        save_best_only=True,
        verbose=1
    )
    class CleanProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch+1}: Total Loss: {logs['loss']:.4f} - R2: {logs['output-0_r2_score']:.4f}")

    callbacks = [CleanProgress(), early_stopping, reduce_lr, model_checkpoint]

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
        val_df: pd.DataFrame
    ) -> Tuple[float, np.ndarray, np.ndarray]:

    # Function for reconcat separated dataframes
    def reconcat(result: Dict[str, np.array]):
        ordered = []
        for i in range(len(result)):
            ordered.append(result[f'output-{i}'])
        concatenated = np.concatenate(ordered, axis=1)
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
        all_errors_val = np.sqrt(np.mean(np.square(autoencoder(val_df, training=False) - val_df), axis=1))
    except Exception as e:
        all_errors_val = np.array([])

    # Find threshold based on quantile of training error distribution
    threshold = np.quantile(all_errors_train, args.quantile_threshold)
    logger.info(f"Anomaly detection threshold set at: {threshold}")

    return threshold, all_errors_train, all_errors_val
