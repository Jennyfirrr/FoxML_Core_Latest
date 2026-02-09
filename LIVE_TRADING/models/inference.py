"""
Inference Engine
================

Unified inference interface that routes to family-specific prediction methods.
Handles sequential models with buffer management.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from TRAINING.common.live.seq_ring_buffer import SeqBufferManager

from LIVE_TRADING.common.constants import (
    SEQUENTIAL_FAMILIES,
    TF_FAMILIES,
    TREE_FAMILIES,
)
from LIVE_TRADING.common.exceptions import InferenceError
from .loader import ModelLoader

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Unified inference engine for all model families.

    Routes predictions to appropriate family-specific methods.
    Manages sequential buffers for time-series models.

    Phase 17 (interval-agnostic pipeline): Validates that data interval matches
    model training interval to prevent garbage predictions from interval mismatch.
    """

    def __init__(
        self,
        loader: ModelLoader,
        device: str = "cpu",
        data_interval_minutes: Optional[float] = None,
        strict_interval_check: bool = True,
    ):
        """
        Initialize inference engine.

        Args:
            loader: ModelLoader instance
            device: Device for inference ("cpu" or "cuda")
            data_interval_minutes: Data bar interval in minutes (Phase 17).
                                   If provided, validates against model training interval.
            strict_interval_check: If True (default), raise error on interval mismatch.
                                  If False, log warning but continue.
        """
        self.loader = loader
        self.device = device
        self.data_interval_minutes = data_interval_minutes
        self.strict_interval_check = strict_interval_check

        # Loaded models cache
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        # Sequential buffer managers per target/family
        self._seq_buffers: Dict[str, SeqBufferManager] = {}

        # Track validated models to avoid repeated checks
        self._interval_validated: Dict[str, bool] = {}

        logger.info(f"InferenceEngine initialized on device: {device}")
        if data_interval_minutes is not None:
            logger.info(f"Phase 17: Data interval validation enabled ({data_interval_minutes}m)")

    def _validate_interval(
        self,
        cache_key: str,
        metadata: Dict[str, Any],
        target: str,
        family: str,
    ) -> bool:
        """
        Phase 17: Validate that data interval matches model training interval.

        Args:
            cache_key: Cache key for tracking validated models
            metadata: Model metadata dict
            target: Target name (for error messages)
            family: Model family (for error messages)

        Returns:
            True if validation passes or is skipped, False if strict check fails

        Raises:
            InferenceError: If strict_interval_check=True and intervals don't match
        """
        # Skip if no data interval provided or already validated
        if self.data_interval_minutes is None:
            return True
        if cache_key in self._interval_validated:
            return self._interval_validated[cache_key]

        training_interval = metadata.get("interval_minutes")

        if training_interval is None:
            # Model metadata doesn't have interval (older model)
            logger.warning(
                f"Phase 17: Model {target}:{family} has no interval_minutes in metadata. "
                f"Skipping interval validation (may produce incorrect predictions)"
            )
            self._interval_validated[cache_key] = True
            return True

        if float(training_interval) != float(self.data_interval_minutes):
            msg = (
                f"Phase 17: Interval mismatch for {target}:{family}. "
                f"Model trained at {training_interval}m, data is {self.data_interval_minutes}m. "
                f"Predictions may be incorrect."
            )
            if self.strict_interval_check:
                raise InferenceError(family, target, msg)
            else:
                logger.warning(msg)
                self._interval_validated[cache_key] = True
                return True

        logger.debug(f"Phase 17: Interval validated for {target}:{family} ({training_interval}m)")
        self._interval_validated[cache_key] = True
        return True

    def load_models_for_target(
        self,
        target: str,
        families: List[str] | None = None,
        view: str = "CROSS_SECTIONAL",
    ) -> None:
        """
        Pre-load models for a target.

        Args:
            target: Target name
            families: List of families to load (None = all available)
            view: View type
        """
        if families is None:
            families = self.loader.list_available_families(target, view)

        for family in families:
            try:
                model, metadata = self.loader.load_model(target, family, view)
                cache_key = f"{target}:{family}:{view}"

                # Phase 17: Validate interval before caching
                self._validate_interval(cache_key, metadata, target, family)

                self._models[cache_key] = model
                self._metadata[cache_key] = metadata

                # Initialize buffer for sequential models
                if family in SEQUENTIAL_FAMILIES:
                    self._init_sequential_buffer(target, family, metadata)

                logger.info(f"Loaded {family} model for {target}")

            except Exception as e:
                logger.warning(f"Failed to load {family} for {target}: {e}")

    def _init_sequential_buffer(
        self,
        target: str,
        family: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Initialize sequential buffer for a model."""
        seq_length = metadata.get("sequence_length", 20)
        input_mode = metadata.get("input_mode", "features")

        if input_mode == "raw_sequence":
            # Raw OHLCV: F = number of channels (typically 5)
            channels = metadata.get(
                "sequence_channels",
                ["open", "high", "low", "close", "volume"],
            )
            n_features = len(channels)
        else:
            # Feature-based: F = number of features
            feature_list = (
                metadata.get("feature_list")
                or metadata.get("features")
                or metadata.get("feature_names")
                or []
            )
            n_features = len(feature_list)

        if n_features == 0:
            logger.warning(f"No features/channels for {target}:{family}, skipping buffer init")
            return

        buffer_key = f"{target}:{family}"
        # Get TTL from sequential config (time-based, not hardcoded)
        from CONFIG.config_loader import get_cfg
        ttl_seconds = get_cfg("pipeline.training.sequential.live.ttl_seconds", default=300.0)
        self._seq_buffers[buffer_key] = SeqBufferManager(
            T=seq_length,
            F=n_features,
            ttl_seconds=ttl_seconds,
        )
        logger.debug(
            f"Initialized buffer for {buffer_key}: T={seq_length}, F={n_features}, "
            f"mode={input_mode}, ttl={ttl_seconds}s"
        )

    def predict(
        self,
        target: str,
        family: str,
        features: np.ndarray,
        symbol: str = "default",
        view: str = "CROSS_SECTIONAL",
    ) -> float:
        """
        Make prediction for a single sample.

        Args:
            target: Target name
            family: Model family
            features: Feature array (1D for cross-sectional, 2D for sequential)
            symbol: Symbol for sequential buffer tracking
            view: View type

        Returns:
            Prediction value

        Raises:
            InferenceError: If prediction fails
        """
        cache_key = f"{target}:{family}:{view}"

        if cache_key not in self._models:
            # Try to load on-demand
            try:
                model, metadata = self.loader.load_model(target, family, view)

                # Phase 17: Validate interval before caching
                self._validate_interval(cache_key, metadata, target, family)

                self._models[cache_key] = model
                self._metadata[cache_key] = metadata

                if family in SEQUENTIAL_FAMILIES:
                    self._init_sequential_buffer(target, family, metadata)
            except InferenceError:
                raise  # Re-raise interval validation errors
            except Exception as e:
                raise InferenceError(family, symbol, f"Failed to load model: {e}")

        model = self._models[cache_key]
        metadata = self._metadata[cache_key]

        input_mode = metadata.get("input_mode", "features")

        try:
            # Raw OHLCV sequence path
            if input_mode == "raw_sequence":
                if family in SEQUENTIAL_FAMILIES:
                    return self._predict_raw_sequential(
                        model, features, target, family, symbol, metadata
                    )
                else:
                    raise InferenceError(
                        family, symbol,
                        f"raw_sequence mode only supported for sequential families, got {family}",
                    )

            # Existing feature-based paths
            if family in TREE_FAMILIES:
                return self._predict_tree(model, features, family)
            elif family in SEQUENTIAL_FAMILIES:
                return self._predict_sequential(
                    model, features, target, family, symbol
                )
            elif family in TF_FAMILIES:
                return self._predict_keras(model, features)
            else:
                return self._predict_generic(model, features, family, symbol)

        except InferenceError:
            raise
        except Exception as e:
            raise InferenceError(family, symbol, str(e))

    def _predict_tree(self, model: Any, features: np.ndarray, family: str) -> float:
        """Predict with tree-based model (LightGBM, XGBoost, CatBoost)."""
        X = np.atleast_2d(features)

        # Check if it's a LightGBM Booster (native format)
        if hasattr(model, "predict") and hasattr(model, "num_trees"):
            # Native LightGBM Booster
            pred = model.predict(X)
        else:
            # sklearn-style interface
            pred = model.predict(X)

        return float(np.atleast_1d(pred)[0])

    def _predict_keras(self, model: Any, features: np.ndarray) -> float:
        """Predict with Keras model (non-sequential)."""
        X = np.atleast_2d(features).astype(np.float32)
        pred = model.predict(X, verbose=0)
        return float(pred.squeeze())

    def _predict_sequential(
        self,
        model: Any,
        features: np.ndarray,
        target: str,
        family: str,
        symbol: str,
    ) -> float:
        """Predict with sequential model using buffer."""
        buffer_key = f"{target}:{family}"
        buffer_manager = self._seq_buffers.get(buffer_key)

        if buffer_manager is None:
            raise InferenceError(family, symbol, "Buffer not initialized")

        # Push features to buffer
        features_1d = np.atleast_1d(features).astype(np.float32)
        buffer_manager.push_features(symbol, features_1d)

        # Check if buffer is ready
        if not buffer_manager.is_ready(symbol):
            # Return NaN while warming up
            return float("nan")

        # Get sequence and predict
        sequence = buffer_manager.get_sequence(symbol)
        if sequence is None:
            return float("nan")

        # Check if model is PyTorch or Keras
        if hasattr(model, "forward"):
            # PyTorch model
            import torch
            with torch.no_grad():
                seq_tensor = sequence.to(self.device)
                pred = model(seq_tensor)
                return float(pred.cpu().numpy().squeeze())
        else:
            # Keras model
            pred = model.predict(sequence.numpy(), verbose=0)
            return float(pred.squeeze())

    def _predict_raw_sequential(
        self,
        model: Any,
        ohlcv_row: np.ndarray,
        target: str,
        family: str,
        symbol: str,
        metadata: Dict[str, Any],
    ) -> float:
        """
        Predict with raw OHLCV sequential model.

        Pushes normalized OHLCV bar into ring buffer, predicts when buffer full.
        Same pattern as _predict_sequential() but with raw OHLCV data.

        Args:
            model: Loaded model
            ohlcv_row: Normalized OHLCV bar (5,) from predictor._prepare_raw_sequence()
            target: Target name
            family: Model family
            symbol: Symbol for buffer tracking
            metadata: Model metadata

        Returns:
            Prediction value (NaN during warmup)
        """
        buffer_key = f"{target}:{family}"
        buffer_manager = self._seq_buffers.get(buffer_key)

        if buffer_manager is None:
            raise InferenceError(family, symbol, "Buffer not initialized for raw sequence model")

        # Push normalized OHLCV bar to buffer
        ohlcv_1d = np.atleast_1d(ohlcv_row).astype(np.float32)
        buffer_manager.push_features(symbol, ohlcv_1d)

        # Check if buffer has enough bars
        if not buffer_manager.is_ready(symbol):
            return float("nan")  # Still warming up

        # Get full sequence and predict
        sequence = buffer_manager.get_sequence(symbol)
        if sequence is None:
            return float("nan")

        # Check if model is PyTorch or Keras
        if hasattr(model, "forward"):
            # PyTorch model
            import torch
            with torch.no_grad():
                seq_tensor = sequence.to(self.device)
                pred = model(seq_tensor)
                return float(pred.cpu().numpy().squeeze())
        else:
            # Keras model
            pred = model.predict(sequence.numpy(), verbose=0)
            return float(pred.squeeze())

    def _predict_generic(self, model: Any, features: np.ndarray, family: str, symbol: str) -> float:
        """Generic predict for unknown model types."""
        X = np.atleast_2d(features)
        if hasattr(model, "predict"):
            pred = model.predict(X)
            return float(np.atleast_1d(pred)[0])
        else:
            raise InferenceError(family, symbol, "Model has no predict method")

    def predict_all_families(
        self,
        target: str,
        features: np.ndarray,
        symbol: str = "default",
        view: str = "CROSS_SECTIONAL",
        families: List[str] | None = None,
    ) -> Dict[str, float]:
        """
        Make predictions from all loaded families for a target.

        Args:
            target: Target name
            features: Feature array
            symbol: Symbol for tracking
            view: View type
            families: Specific families (None = all loaded)

        Returns:
            Dict mapping family name to prediction
        """
        if families is None:
            families = self.loader.list_available_families(target, view)

        results = {}
        for family in families:
            try:
                pred = self.predict(target, family, features, symbol, view)
                if not np.isnan(pred):
                    results[family] = pred
            except InferenceError as e:
                logger.warning(f"Inference failed for {family}: {e}")

        return results

    def warmup_sequential(
        self,
        target: str,
        family: str,
        historical_features: np.ndarray,
        symbol: str = "default",
    ) -> int:
        """
        Warm up sequential buffer with historical data.

        Args:
            target: Target name
            family: Model family
            historical_features: Historical feature array (T x F)
            symbol: Symbol for buffer

        Returns:
            Number of samples pushed
        """
        buffer_key = f"{target}:{family}"
        buffer_manager = self._seq_buffers.get(buffer_key)

        if buffer_manager is None:
            raise InferenceError(family, symbol, "Buffer not initialized")

        count = 0
        for i in range(len(historical_features)):
            buffer_manager.push_features(symbol, historical_features[i])
            count += 1

        logger.info(f"Warmed up {buffer_key} with {count} samples for {symbol}")
        return count

    def reset_buffers(self, target: str | None = None) -> None:
        """Reset sequential buffers."""
        if target is None:
            for manager in self._seq_buffers.values():
                manager.reset_all()
            logger.debug("Reset all sequential buffers")
        else:
            for key, manager in self._seq_buffers.items():
                if key.startswith(f"{target}:"):
                    manager.reset_all()
            logger.debug(f"Reset sequential buffers for {target}")

    def get_buffer_status(
        self,
        target: str,
        family: str,
        symbol: str = "default",
    ) -> Dict[str, Any]:
        """
        Get buffer status for a specific target/family/symbol.

        Args:
            target: Target name
            family: Model family
            symbol: Symbol name

        Returns:
            Status dict with fill_count, capacity, is_ready, etc.
        """
        buffer_key = f"{target}:{family}"
        buffer_manager = self._seq_buffers.get(buffer_key)

        if buffer_manager is None:
            return {"initialized": False}

        buffer = buffer_manager.buffers.get(symbol)
        if buffer is None:
            return {"initialized": True, "symbol_exists": False}

        return buffer.get_status()

    def clear_cache(self) -> None:
        """Clear model and buffer caches."""
        self._models.clear()
        self._metadata.clear()
        for manager in self._seq_buffers.values():
            manager.reset_all()
        self._seq_buffers.clear()
        logger.debug("InferenceEngine cache cleared")


def predict(
    loader: ModelLoader,
    target: str,
    family: str,
    features: np.ndarray,
    view: str = "CROSS_SECTIONAL",
) -> float:
    """
    Convenience function for single prediction.

    Args:
        loader: ModelLoader instance
        target: Target name
        family: Model family
        features: Feature array
        view: View type

    Returns:
        Prediction value
    """
    engine = InferenceEngine(loader)
    return engine.predict(target, family, features, view=view)
