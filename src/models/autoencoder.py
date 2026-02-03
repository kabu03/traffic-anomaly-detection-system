import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from .base import BaseDetector
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed  # type: ignore

class AutoencoderDetector(BaseDetector):
    """
    Anomaly detector supporting multiple autoencoder architectures.
    arch:
      - 'lstm-ae' : LSTM sequence autoencoder (current)
      - 'dense'   : simple per-window dense AE
    """
    def __init__(self, window_size=24, latent_dim=8, hidden_dim=32, n_layers=1,
                 batch_size=32, random_state=42, arch='lstm-ae'):
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.random_state = random_state
        self.arch = arch
        self.model_ = None

    def _create_sequences(self, X):
        """Helper function to convert a flat time series into sequences (windows)."""
        X_seq = []
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        for i in range(len(X) - self.window_size + 1):
            X_seq.append(X[i:(i + self.window_size)])
        return np.array(X_seq)

    def fit(self, X, y=None, normal_len=None):
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        X_seq = self._create_sequences(X)
        n_features = 1 if X.ndim == 1 else X.shape[1]

        # Optionally restrict training windows to normal prefix
        if normal_len is not None:
            max_start = max(0, normal_len - self.window_size + 1)
            X_train_seq = X_seq[:max_start] if max_start > 0 else X_seq
        else:
            X_train_seq = X_seq

        if self.arch == 'lstm-ae':
            input_layer = Input(shape=(self.window_size, n_features))
            x = input_layer
            # Encoder hidden(s)
            for _ in range(max(0, self.n_layers - 1)):
                x = LSTM(self.hidden_dim, return_sequences=True,
                         activation='tanh', recurrent_activation='sigmoid')(x)
            # Bottleneck
            x = LSTM(self.latent_dim, return_sequences=False,
                     activation='tanh', recurrent_activation='sigmoid')(x)
            # Decoder
            x = RepeatVector(self.window_size)(x)
            for _ in range(max(0, self.n_layers - 1)):
                x = LSTM(self.hidden_dim, return_sequences=True,
                         activation='tanh', recurrent_activation='sigmoid')(x)
            # Final decoder LSTM always hidden_dim for symmetry
            x = LSTM(self.hidden_dim, return_sequences=True,
                     activation='tanh', recurrent_activation='sigmoid')(x)
            output_layer = TimeDistributed(Dense(n_features, activation='linear'))(x)
            self.model_ = Model(input_layer, output_layer)
        else:
            # Placeholder dense variant (per window flatten + dense AE)
            input_layer = Input(shape=(self.window_size, n_features))
            flat = Flatten()(input_layer)
            # Simple bottleneck
            h = Dense(self.hidden_dim, activation='relu')(flat)
            z = Dense(self.latent_dim, activation='relu')(h)
            h2 = Dense(self.hidden_dim, activation='relu')(z)
            out = Dense(self.window_size * n_features, activation='linear')(h2)
            output_layer = Reshape((self.window_size, n_features))(out)
            self.model_ = Model(input_layer, output_layer)

        self.model_.compile(optimizer='adam', loss='mse')

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model_.fit(
            X_train_seq, X_train_seq,
            epochs=100,
            shuffle=False,
            batch_size=self.batch_size,
            validation_split=0.1 if len(X_train_seq) > 10 else 0.0,
            callbacks=[early_stopping],
            verbose=0
        )
        return self

    def decision_function(self, X):
        """
        Calculate the reconstruction error for each sequence as the anomaly score.
        """
        if self.model_ is None:
            raise RuntimeError("The model must be fitted before calling decision_function.")

        X_seq = self._create_sequences(X)
        X_reconstructed = self.model_.predict(X_seq, verbose=0)  # * suppress keras logging

        # Calculate MSE for each sequence (time × features)
        errors = np.mean(np.square(X_seq - X_reconstructed), axis=(1, 2))

        # Pad scores to match original input length. The score for a point corresponds
        # to the error of the window ending at that point.
        # Use NaN for the initial region (avoid biasing threshold); pipeline can drop/ignore NaNs
        pad = np.full(self.window_size - 1, np.nan)
        return np.concatenate([pad, errors])
