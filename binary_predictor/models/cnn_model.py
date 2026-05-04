"""
1D CNN model for candle direction prediction using TensorFlow/Keras.

Dual-input architecture:
  - Sequence input: last N candles as raw OHLCV (Conv1D pathway)
  - Tabular input: engineered features (Dense pathway)
  - Merged for final prediction
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CNN_SEQUENCE_LENGTH, CNN_EPOCHS, CNN_BATCH_SIZE, CNN_PATIENCE, CNN_LR, MODEL_DIR
from utils.logger import get_logger

logger = get_logger("binary_predictor")


def _build_cnn_model(seq_length: int, n_seq_features: int, n_tabular_features: int):
    """
    Build a dual-input 1D CNN model.

    Input 1: Sequence of OHLCV candles → Conv1D pathway
    Input 2: Tabular engineered features → Dense pathway
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model, Input
    except ImportError:
        logger.error("TensorFlow not installed. Install with: pip install tensorflow")
        raise

    # Sequence pathway
    seq_input = Input(shape=(seq_length, n_seq_features), name="seq_input")
    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(seq_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Tabular pathway
    tab_input = Input(shape=(n_tabular_features,), name="tab_input")
    y = layers.Dense(32, activation="relu")(tab_input)
    y = layers.Dropout(0.2)(y)

    # Merge
    merged = layers.Concatenate()([x, y])
    merged = layers.Dense(64, activation="relu")(merged)
    merged = layers.Dropout(0.3)(merged)
    output = layers.Dense(1, activation="sigmoid", name="output")(merged)

    model = Model(inputs=[seq_input, tab_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CNN_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def prepare_sequences(
    df: pd.DataFrame,
    seq_length: int = CNN_SEQUENCE_LENGTH,
    ohlcv_cols: list[str] = None,
    feature_cols: list[str] = None,
    target_col: str = "next_candle_direction",
) -> tuple:
    """
    Prepare sequential + tabular inputs for the CNN.

    Returns
    -------
    (X_seq, X_tab, y)
        X_seq: shape (n_samples, seq_length, n_ohlcv_cols)
        X_tab: shape (n_samples, n_feature_cols)
        y: shape (n_samples,)
    """
    if ohlcv_cols is None:
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
    if feature_cols is None:
        # Use session and pattern flags as tabular features
        feature_cols = [
            c for c in df.columns
            if c not in ohlcv_cols + ["time", target_col, "candle_direction"]
            and c in df.columns
        ]

    seq_data = df[ohlcv_cols].values
    tab_data = df[feature_cols].values
    target = df[target_col].values if target_col in df.columns else None

    X_seq_list = []
    X_tab_list = []
    y_list = []

    for i in range(seq_length, len(df)):
        X_seq_list.append(seq_data[i - seq_length : i])
        X_tab_list.append(tab_data[i])
        if target is not None:
            y_list.append(target[i])

    X_seq = np.array(X_seq_list, dtype=np.float32)
    X_tab = np.array(X_tab_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) if y_list else None

    return X_seq, X_tab, y, feature_cols


class CNNModel:
    """1D CNN dual-input model wrapper."""

    def __init__(self, seq_length: int = CNN_SEQUENCE_LENGTH):
        self.seq_length = seq_length
        self.model = None
        self.feature_cols: list[str] = []
        self.ohlcv_cols = ["open", "high", "low", "close", "volume"]
        self._is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "CNNModel":
        """
        Train the CNN model.
        
        X_train/X_val should be the FULL DataFrames (with OHLCV + features).
        """
        import tensorflow as tf

        # Prepare sequences from training data
        df_train = X_train.copy()
        df_train["next_candle_direction"] = y_train.values

        X_seq_tr, X_tab_tr, y_tr, self.feature_cols = prepare_sequences(
            df_train, self.seq_length, self.ohlcv_cols
        )

        if X_seq_tr is None or len(X_seq_tr) == 0:
            logger.warning("Not enough data to train CNN")
            return self

        # Build model
        self.model = _build_cnn_model(
            self.seq_length,
            n_seq_features=len(self.ohlcv_cols),
            n_tabular_features=len(self.feature_cols),
        )

        # Validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            df_val = X_val.copy()
            df_val["next_candle_direction"] = y_val.values
            X_seq_v, X_tab_v, y_v, _ = prepare_sequences(
                df_val, self.seq_length, self.ohlcv_cols, self.feature_cols
            )
            if len(X_seq_v) > 0:
                validation_data = ([X_seq_v, X_tab_v], y_v)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if validation_data else "loss",
                patience=CNN_PATIENCE,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, min_lr=1e-6
            ),
        ]

        # Normalize OHLCV sequences per-window
        X_seq_tr = self._normalize_sequences(X_seq_tr)
        if validation_data:
            X_seq_v = self._normalize_sequences(X_seq_v)
            validation_data = ([X_seq_v, X_tab_v], y_v)

        # Replace NaN/Inf
        X_seq_tr = np.nan_to_num(X_seq_tr, nan=0.0, posinf=1.0, neginf=-1.0)
        X_tab_tr = np.nan_to_num(X_tab_tr, nan=0.0, posinf=1.0, neginf=-1.0)

        self.model.fit(
            [X_seq_tr, X_tab_tr],
            y_tr,
            epochs=CNN_EPOCHS,
            batch_size=CNN_BATCH_SIZE,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0,
        )

        self._is_fitted = True
        logger.info(f"CNN trained: seq_len={self.seq_length}, samples={len(X_seq_tr):,}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for class 1 (UP).
        X should be the full DataFrame with OHLCV + features.
        """
        if not self._is_fitted or self.model is None:
            return np.full(len(X), 0.5)

        X_seq, X_tab, _, _ = prepare_sequences(
            X, self.seq_length, self.ohlcv_cols, self.feature_cols,
            target_col="_dummy_"
        )

        if len(X_seq) == 0:
            return np.full(len(X), 0.5)

        X_seq = self._normalize_sequences(X_seq)
        X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=1.0, neginf=-1.0)
        X_tab = np.nan_to_num(X_tab, nan=0.0, posinf=1.0, neginf=-1.0)

        preds = self.model.predict([X_seq, X_tab], verbose=0).flatten()

        # Pad to match original length (first seq_length rows can't be predicted)
        padded = np.full(len(X), 0.5)
        padded[self.seq_length:] = preds
        return padded

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def _normalize_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        """Normalize each sequence window independently."""
        X_norm = X_seq.copy()
        for i in range(len(X_norm)):
            window = X_norm[i]
            mean = window.mean(axis=0, keepdims=True)
            std = window.std(axis=0, keepdims=True) + 1e-9
            X_norm[i] = (window - mean) / std
        return X_norm

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model to disk."""
        path = path or MODEL_DIR / "cnn_model"
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save(str(path / "keras_model.keras"))

        meta = {
            "feature_cols": self.feature_cols,
            "ohlcv_cols": self.ohlcv_cols,
            "seq_length": self.seq_length,
            "is_fitted": self._is_fitted,
        }
        joblib.dump(meta, path / "cnn_meta.pkl")
        logger.info(f"CNN model saved → {path}")
        return path

    def load(self, path: Optional[Path] = None) -> "CNNModel":
        """Load model from disk."""
        path = path or MODEL_DIR / "cnn_model"
        path = Path(path)

        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(str(path / "keras_model.keras"))
        except Exception as e:
            logger.warning(f"Could not load CNN Keras model: {e}")
            self.model = None

        meta_path = path / "cnn_meta.pkl"
        if meta_path.exists():
            meta = joblib.load(meta_path)
            self.feature_cols = meta["feature_cols"]
            self.ohlcv_cols = meta["ohlcv_cols"]
            self.seq_length = meta["seq_length"]
            self._is_fitted = meta["is_fitted"]

        logger.info(f"CNN model loaded ← {path}")
        return self
