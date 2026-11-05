"""CNN confrontabili: input comune (frame temporali, coefficienti MFCC)."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_model(input_shape, n_classes, cfg, mean, variance, noise_std=.45):
    if cfg.family not in ('cnn1d', 'cnn2d'):
        raise ValueError('family deve essere cnn1d o cnn2d')
    if not cfg.filters or any(f <= 0 for f in cfg.filters) or cfg.kernel <= 0:
        raise ValueError('Filtri e kernel devono essere positivi')
    if not 0 <= cfg.dropout < 1 or cfg.head not in ('flatten','gap'):
        raise ValueError('Dropout/head non validi')
    inputs = layers.Input(shape=input_shape, name='mfcc_raw')
    # Nessuna normalizzazione aggiuntiva: le MFCC sono già normalizzate dal DSP EI.
    x = inputs
    if noise_std > 0:
        x = layers.GaussianNoise(noise_std, name='training_noise')(x)
    if cfg.family == 'cnn2d':
        x = layers.Reshape((*input_shape, 1))(x)
    for i, filters in enumerate(cfg.filters):
        conv = layers.Conv1D if cfg.family == 'cnn1d' else layers.Conv2D
        pool = layers.MaxPooling1D if cfg.family == 'cnn1d' else layers.MaxPooling2D
        kernel = cfg.kernel if cfg.family == 'cnn1d' else (cfg.kernel,cfg.kernel)
        x = conv(filters, kernel, padding='same', activation='relu',
                 name=f'conv_{i+1}')(x)
        x = pool(pool_size=2, strides=2, padding='same', name=f'pool_{i+1}')(x)
        x = layers.Dropout(cfg.dropout, name=f'dropout_{i+1}')(x)
    if cfg.head == 'flatten':
        x = layers.Flatten()(x)
    else:
        pool = layers.GlobalAveragePooling1D if cfg.family == 'cnn1d' else layers.GlobalAveragePooling2D
        x = pool()(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='y_pred')(x)
    return Model(inputs, outputs, name=cfg.family)
