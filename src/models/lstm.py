import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential  # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from .base import BaseDetector

class LSTMDetector(BaseDetector):
    """
    Anomaly detector using an LSTM-based forecasting model.

    The model is trained to predict the next time step in a sequence. The anomaly
    score is the squared prediction error.

    Optional station embedding:
      - If station_idx (per-timestep integer indices) is provided to fit/decision_function,
        we embed with an Embedding layer (UNK=0), concatenate to numeric features at each
        timestep, and train/predict with the augmented feature tensor.
      - If station_idx is None, behavior matches the original numeric-only model.
    """
    def __init__(
        self,
        window_size=24,
        lstm_units=32,
        n_layers=1,
        batch_size=32,
        random_state=42,
        use_station_embedding=True,
        embedding_dim=8,
        station_vocab_size=None  # if None, inferred from data (max idx + 1)
    ):
        self.window_size = window_size
        self.lstm_units = lstm_units
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.random_state = random_state
        self.use_station_embedding = use_station_embedding
        self.embedding_dim = embedding_dim
        self.station_vocab_size = station_vocab_size

        self.model_ = None
        self._n_features_ = None  # numeric feature dimension at fit-time
        self._use_embed_path_ = False  # set true if we built the embedding path

    def _create_sequences(self, X, station_idx=None, incident_nums=None):
        """
        Convert flat time series into input/output sequences for forecasting.

        If incident_nums is provided (shape (N,) or (N,1)), windows are created
        only within contiguous regions sharing the same stream_id to prevent
        cross-stream leakage.

        Returns:
          X_seq: (N_windows, window_size, d)
          y_seq: (N_windows, d)
          S_seq: None or (N_windows, window_size) int32  (embedding indices)
        """
        X_seq, y_seq = [], []
        S_seq = [] if station_idx is not None else None

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        N = len(X)

        # Validate/normalize station_idx
        s = None
        if station_idx is not None:
            s = station_idx
            if s.ndim == 2 and s.shape[1] == 1:
                s = s.ravel()
            s = s.astype(np.int32)
            if len(s) != N:
                raise ValueError(f"station_idx length {len(s)} != X length {N}")

        # Validate/normalize incident_nums
        cids = None
        if incident_nums is not None:
            cids = incident_nums
            if getattr(cids, "ndim", 1) == 2 and cids.shape[1] == 1:
                cids = cids.ravel()
            if len(cids) != N:
                raise ValueError(f"incident_nums length {len(cids)} != X length {N}")

        # Helper: iterate contiguous segments by stream_id
        def iter_segments():
            if cids is None:
                yield 0, N
                return
            starts = np.flatnonzero(np.r_[True, cids[1:] != cids[:-1]])
            ends = np.r_[starts[1:], N]
            for st, en in zip(starts, ends):
                yield int(st), int(en)

        # Build windows per segment (no cross-stream)
        W = self.window_size
        for st, en in iter_segments():
            seg_len = en - st
            if seg_len <= W:
                continue
            for i in range(st, en - W):
                X_seq.append(X[i:(i + W)])
                y_seq.append(X[i + W])
                if S_seq is not None:
                    S_seq.append(s[i:(i + W)])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        if S_seq is not None:
            S_seq = np.array(S_seq, dtype=np.int32)
        return X_seq, y_seq, S_seq

    def _build_model(self, d_num_features, using_embedding):
        """
        Build either:
          - numeric-only Sequential model (backward compatible), or
          - two-input Functional model with Embedding + concat.
        """
        tf.random.set_seed(self.random_state)

        if not using_embedding:
            model = Sequential()
            model.add(Input(shape=(self.window_size, d_num_features)))
            for i in range(self.n_layers):
                return_sequences = (i < self.n_layers - 1)
                model.add(LSTM(self.lstm_units, activation='relu', return_sequences=return_sequences))
            model.add(Dense(d_num_features))
            model.compile(optimizer='adam', loss='mae')
            return model

        # Two-input model: numeric sequence + station index sequence
        x_in = Input(shape=(self.window_size, d_num_features), name="x_numeric")
        s_in = Input(shape=(self.window_size,), dtype='int32', name="s_index")

        if self.station_vocab_size is None:
            raise ValueError("station_vocab_size must be set when using station embedding.")

        emb = Embedding(
            input_dim=int(self.station_vocab_size),
            output_dim=int(self.embedding_dim),
            mask_zero=False,  # we do not use 0-padding masking; UNK=0 is a valid token
            name="station_embedding"
        )(s_in)  # -> (batch, window_size, embedding_dim)

        x_cat = Concatenate(axis=-1, name="concat_num_embed")([x_in, emb])

        x = x_cat
        for i in range(self.n_layers):
            return_sequences = (i < self.n_layers - 1)
            x = LSTM(self.lstm_units, activation='relu', return_sequences=return_sequences, name=f"lstm_{i+1}")(x)

        out = Dense(d_num_features, name="pred_next")(x)

        model = Model(inputs=[x_in, s_in], outputs=out, name="lstm_with_station_embedding")
        model.compile(optimizer='adam', loss='mae')
        return model

    def fit(self, X, y=None, station_idx=None, incident_nums=None):
        """
        Fit the forecasting model.
        - X: (N, d_num)
        - station_idx: optional (N,) or (N,1) int indices (UNK=0) for embedding per timestep.
        - incident_nums: optional (N,) int; if provided, training windows do not cross stream boundaries.
        """
        self._n_features_ = X.shape[1] if X.ndim == 2 else 1

        X_seq, y_seq, S_seq = self._create_sequences(X, station_idx=station_idx, incident_nums=incident_nums)
        using_embedding = self.use_station_embedding and (S_seq is not None)

        if using_embedding and self.station_vocab_size is None:
            max_idx = int(np.max(S_seq)) if S_seq.size > 0 else 0
            self.station_vocab_size = max_idx + 1

        self.model_ = self._build_model(self._n_features_, using_embedding)
        self._use_embed_path_ = using_embedding

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        if using_embedding:
            self.model_.fit(
                [X_seq, S_seq], y_seq,
                epochs=100,
                batch_size=self.batch_size,
                validation_split=0.1,
                callbacks=[early_stopping],
                shuffle=False,
                verbose=0
            )
        else:
            self.model_.fit(
                X_seq, y_seq,
                epochs=100,
                batch_size=self.batch_size,
                validation_split=0.1,
                callbacks=[early_stopping],
                shuffle=False,
                verbose=0
            )
        return self

    def decision_function(self, X, station_idx=None, incident_nums=None):
        """
        Compute anomaly scores (squared prediction error) per timestep.

        - If incident_nums is provided, windows and scores are computed per stream segment.
          The first window_size timesteps of each segment are zero-padded (no context).
        """
        if self.model_ is None:
            raise RuntimeError("The model must be fitted before calling decision_function.")

        # Build sequences (respects stream boundaries if incident_nums provided)
        X_seq, y_true, S_seq = self._create_sequences(X, station_idx=station_idx, incident_nums=incident_nums)

        # Predict
        if self._use_embed_path_:
            if S_seq is None:
                raise ValueError("This model expects station_idx for inference; got None.")
            y_pred = self.model_.predict([X_seq, S_seq], verbose=0)
        else:
            y_pred = self.model_.predict(X_seq, verbose=0)

        # Window-wise errors
        errors = np.square(y_true - y_pred)
        win_scores = np.sum(errors, axis=1)  # length = total number of windows

        # Scatter back to full-timestep scores with per-segment zero padding
        N = len(X) if X.ndim == 1 else X.shape[0]
        scores = np.zeros(N, dtype=float)

        if incident_nums is None:
            # Single segment: pad first window_size points
            W = self.window_size
            if N > W:
                scores[W:] = win_scores
            return scores

        # Multiple segments: pad each segment
        cids = incident_nums
        if getattr(cids, "ndim", 1) == 2 and cids.shape[1] == 1:
            cids = cids.ravel()
        starts = np.flatnonzero(np.r_[True, cids[1:] != cids[:-1]])
        ends = np.r_[starts[1:], N]

        W = self.window_size
        w_ptr = 0
        for st, en in zip(starts, ends):
            seg_len = en - st
            n_wins = max(seg_len - W, 0)
            if n_wins > 0:
                scores[st + W: en] = win_scores[w_ptr: w_ptr + n_wins]
                w_ptr += n_wins
            # else: entire segment stays zeros (too short to form a window)
        return scores