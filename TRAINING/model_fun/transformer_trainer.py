# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

import TRAINING.common.repro_bootstrap  # noqa: F401 - must be first

import numpy as np, logging, tensorflow as tf, sys
from typing import Any, Dict, List, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
from .base_trainer import BaseModelTrainer
from TRAINING.common.safety import configure_tf
logger = logging.getLogger(__name__)

# Add CONFIG directory to path for centralized config loading
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Try to import config loader
_USE_CENTRALIZED_CONFIG = False
try:
    from config_loader import load_model_config
    _USE_CENTRALIZED_CONFIG = True
except ImportError:
    logger.debug("config_loader not available; using hardcoded defaults")

class TransformerTrainer(BaseModelTrainer):
    def __init__(self, config: Dict[str, Any] = None):
        # Load centralized config if available and no config provided
        if config is None and _USE_CENTRALIZED_CONFIG:
            try:
                config = load_model_config("transformer")
                logger.info("✅ [Transformer] Loaded centralized config from CONFIG/model_config/transformer.yaml")
            except Exception as e:
                logger.warning(f"Failed to load centralized config: {e}. Using hardcoded defaults.")
                config = {}
        
        super().__init__(config or {})
        
        # DEPRECATED: Hardcoded defaults kept for backward compatibility
        # To change these, edit CONFIG/model_config/transformer.yaml
        self.config.setdefault("epochs", 50)
        self.config.setdefault("batch_size", 128)  # Reduced from 512 to prevent OOM
        self.config.setdefault("d_model", 128)
        self.config.setdefault("heads", 4)  # Reduced from 8 to prevent OOM with large sequences
        self.config.setdefault("ff_dim", 256)
        self.config.setdefault("dropout", 0.1)
        self.config.setdefault("learning_rate", 1e-3)
        self.config.setdefault("patience", 10)

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              X_va=None, y_va=None, feature_names: List[str] = None, **kwargs) -> Any:
        from common.threads import ensure_gpu_visible
        
        # Ensure GPU is visible (restore if hidden by prior CPU-only family)
        gpu_available = ensure_gpu_visible("Transformer")
        
        # 1) Preprocess data
        X_tr, y_tr = self.preprocess_data(X_tr, y_tr)
        
        # 2) Configure TensorFlow
        configure_tf(cpu_only=kwargs.get("cpu_only", False) or not gpu_available)
        
        # 3) Split only if no external validation provided
        if X_va is None or y_va is None:
            # Load test split params from config
            test_size, seed = self._get_test_split_params()
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tr, y_tr, test_size=test_size, random_state=seed
            )
        
        # 4) Reshape for Transformer
        # Check if input is already 3D (raw OHLCV sequence mode)
        if X_tr.ndim == 3:
            # Already 3D (N, seq_len, channels) - raw sequence mode
            self.feature_names = feature_names or [f"ch{i}" for i in range(X_tr.shape[2])]
            logger.info(f"[Transformer] Input already 3D: {X_tr.shape} (raw sequence mode)")
        else:
            # 2D (N, F) - traditional feature mode
            self.feature_names = feature_names or [f"f{i}" for i in range(X_tr.shape[1])]
            # Reshape to (N, F, 1) where features are treated as time steps
            X_tr = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
            X_va = X_va.reshape(X_va.shape[0], X_va.shape[1], 1)
        
        # 4.5) Adjust batch size dynamically based on sequence length to prevent OOM
        # NOTE: These thresholds are in BAR COUNT, not time. Memory usage scales with bar count:
        #   Attention matrix: O(batch_size * heads * seq_len^2)
        # At different intervals, the same bar count represents different time periods:
        #   - 1m: 200 bars = 200 min (3.3 hours)
        #   - 5m: 200 bars = 1000 min (16.7 hours)
        #   - 15m: 200 bars = 3000 min (50 hours)
        # This is intentional: Transformer memory scales quadratically with bar count.
        seq_len = X_tr.shape[1]
        base_batch_size = self.config["batch_size"]

        # Get threshold from config (value is in BARS)
        try:
            from CONFIG.config_loader import get_cfg
            max_seq_for_full_batch = self.config.get(
                "max_sequence_length_for_full_batch",
                int(get_cfg("models.transformer.max_seq_for_full_batch", default=200))
            )
            interval_minutes = get_cfg("pipeline.data.interval_minutes", default=5)
        except ImportError:
            max_seq_for_full_batch = self.config.get("max_sequence_length_for_full_batch", 200)
            interval_minutes = 5

        # For sequences > threshold bars, reduce batch size to prevent OOM
        if seq_len > max_seq_for_full_batch:
            # Log what this represents in time
            time_window_minutes = seq_len * interval_minutes
            threshold_time_minutes = max_seq_for_full_batch * interval_minutes
            logger.info(f"[Transformer] Sequence length {seq_len} bars ({time_window_minutes}m) exceeds "
                       f"threshold {max_seq_for_full_batch} bars ({threshold_time_minutes}m)")

            # Scale batch size inversely with sequence length
            adjusted_batch_size = max(32, int(base_batch_size * (max_seq_for_full_batch / seq_len)))
            if adjusted_batch_size < base_batch_size:
                logger.warning(f"⚠️ [Transformer] Reducing batch size from {base_batch_size} to {adjusted_batch_size} "
                             f"to prevent OOM (sequence length: {seq_len} bars)")
                self.config["batch_size"] = adjusted_batch_size
        
        # 5) Build model with correct input shape
        model = self._build_model(X_tr.shape[1], n_channels=X_tr.shape[2])
        
        # 6) Train with callbacks
        callbacks = [
            # Use callbacks from config if available
            *self.get_callbacks("Transformer")
        ]
        
        model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=callbacks,
            verbose=0
        )
        
        # 7) Store state and sanity check
        self.model = model
        self.is_trained = True
        self.post_fit_sanity(X_tr, "Transformer")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        from TRAINING.common.safety import guard_features
        if self._is_3d_input:
            # Raw sequence mode: skip 2D preprocessing, pass 3D directly
            Xp = guard_features(np.ascontiguousarray(X, dtype=np.float32))
        else:
            # Feature mode: existing 2D preprocessing path
            if X.ndim == 3:
                X = X.reshape(X.shape[0], X.shape[1])
            Xp, _ = self.preprocess_data(X, None)
            Xp = Xp.reshape(Xp.shape[0], Xp.shape[1], 1)
        preds = self.model.predict(Xp, verbose=0).ravel()
        return np.nan_to_num(preds, nan=0.0).astype(np.float32)

    def _build_model(self, input_dim: int, n_channels: int = 1) -> tf.keras.Model:
        """Build Transformer model with safe defaults"""
        import os
        # Set TF seed for determinism (TF_DETERMINISTIC_OPS=1 requires explicit seed)
        seed = int(os.environ.get("PYTHONHASHSEED", "42"))
        tf.random.set_seed(seed)

        inputs = tf.keras.Input(shape=(input_dim, n_channels), name="x")
        x = inputs
        
        # Project to d_model
        x = tf.keras.layers.Conv1D(self.config["d_model"], 1, activation="linear")(x)
        
        # Multi-head attention
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=self.config["heads"], 
            key_dim=self.config["d_model"]
        )(x, x)
        
        # Add & Norm
        x = tf.keras.layers.Add()([x, attn])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Feed forward
        ff = tf.keras.layers.Dense(self.config["ff_dim"], activation="relu")(x)
        ff = tf.keras.layers.Dropout(self.config["dropout"])(ff)
        ff = tf.keras.layers.Dense(self.config["d_model"])(ff)
        
        # Add & Norm
        x = tf.keras.layers.Add()([x, ff])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Global pooling and output
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.config["dropout"])(x)
        outputs = tf.keras.layers.Dense(1, name="y")(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Compile with gradient clipping (load from config if available)
        clipnorm = self._get_clipnorm()
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config["learning_rate"],
            clipnorm=clipnorm
        )
        
        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae"]
        )
        
        return model