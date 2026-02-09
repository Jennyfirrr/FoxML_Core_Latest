"""
Multi-Horizon Predictor
=======================

Coordinates predictions across all horizons and model families.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from TRAINING.training_strategies.utils import _normalize_ohlcv_sequence

from LIVE_TRADING.common.constants import HORIZONS
from LIVE_TRADING.models.loader import ModelLoader
from LIVE_TRADING.models.inference import InferenceEngine
from LIVE_TRADING.models.feature_builder import FeatureBuilder
from .standardization import ZScoreStandardizer
from .confidence import ConfidenceScorer, ConfidenceComponents

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Single model prediction with metadata."""

    family: str
    horizon: str
    raw: float
    standardized: float
    confidence: ConfidenceComponents
    calibrated: float  # standardized × confidence

    @property
    def alpha(self) -> float:
        """Alias for calibrated prediction (used by barrier gate)."""
        return self.calibrated

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "family": self.family,
            "horizon": self.horizon,
            "raw": self.raw,
            "standardized": self.standardized,
            "confidence": self.confidence.to_dict(),
            "calibrated": self.calibrated,
        }


@dataclass
class HorizonPredictions:
    """Predictions for a single horizon."""

    horizon: str
    timestamp: datetime
    predictions: Dict[str, ModelPrediction] = field(default_factory=dict)

    @property
    def families(self) -> List[str]:
        """List of families with predictions."""
        return list(self.predictions.keys())

    @property
    def mean_calibrated(self) -> float:
        """Mean of calibrated predictions."""
        if not self.predictions:
            return 0.0
        return float(np.mean([p.calibrated for p in self.predictions.values()]))

    @property
    def mean_standardized(self) -> float:
        """Mean of standardized predictions."""
        if not self.predictions:
            return 0.0
        return float(np.mean([p.standardized for p in self.predictions.values()]))

    def get_calibrated_dict(self) -> Dict[str, float]:
        """Get dict of calibrated predictions (sorted by family)."""
        return {f: p.calibrated for f, p in sorted_items(self.predictions)}

    def get_standardized_dict(self) -> Dict[str, float]:
        """Get dict of standardized predictions (sorted by family)."""
        return {f: p.standardized for f, p in sorted_items(self.predictions)}

    def get_raw_dict(self) -> Dict[str, float]:
        """Get dict of raw predictions (sorted by family)."""
        return {f: p.raw for f, p in sorted_items(self.predictions)}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "horizon": self.horizon,
            "timestamp": self.timestamp.isoformat(),
            "predictions": {f: p.to_dict() for f, p in sorted_items(self.predictions)},
            "mean_calibrated": self.mean_calibrated,
            "mean_standardized": self.mean_standardized,
        }


@dataclass
class AllPredictions:
    """Predictions across all horizons."""

    symbol: str
    timestamp: datetime
    horizons: Dict[str, HorizonPredictions] = field(default_factory=dict)

    def get_horizon(self, horizon: str) -> Optional[HorizonPredictions]:
        """Get predictions for a specific horizon."""
        return self.horizons.get(horizon)

    @property
    def available_horizons(self) -> List[str]:
        """List of horizons with predictions."""
        return list(self.horizons.keys())

    def get_best_horizon(self) -> Optional[str]:
        """Get horizon with highest mean calibrated signal."""
        if not self.horizons:
            return None
        return max(
            self.horizons.keys(),
            key=lambda h: abs(self.horizons[h].mean_calibrated)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "horizons": {h: hp.to_dict() for h, hp in sorted_items(self.horizons)},
            "best_horizon": self.get_best_horizon(),
        }


class MultiHorizonPredictor:
    """
    Generates predictions across all horizons and model families.

    Pipeline:
    1. Build features from market data
    2. Run inference for each (horizon, family) pair
    3. Standardize predictions
    4. Calculate confidence
    5. Apply calibration
    """

    def __init__(
        self,
        run_root: str,
        horizons: List[str] | None = None,
        families: List[str] | None = None,
        device: str = "cpu",
    ):
        """
        Initialize multi-horizon predictor.

        Args:
            run_root: Path to TRAINING run artifacts
            horizons: Horizons to predict (default: all)
            families: Model families to use (default: all available)
            device: Device for inference
        """
        self.horizons = horizons or get_cfg(
            "live_trading.horizons", default=HORIZONS
        )
        self.families = families

        # Initialize components
        self.loader = ModelLoader(run_root)
        self.engine = InferenceEngine(self.loader, device=device)
        self.standardizer = ZScoreStandardizer()
        self.confidence_scorer = ConfidenceScorer()

        # Feature builders per target (cached)
        self._feature_builders: Dict[str, FeatureBuilder] = {}

        logger.info(f"MultiHorizonPredictor initialized: horizons={self.horizons}")

    def _get_feature_builder(
        self,
        target: str,
        family: str,
    ) -> FeatureBuilder:
        """Get or create feature builder for target/family."""
        key = f"{target}:{family}"
        if key not in self._feature_builders:
            feature_list = self.loader.get_feature_list(target, family)
            self._feature_builders[key] = FeatureBuilder(feature_list)
        return self._feature_builders[key]

    def predict_all_horizons(
        self,
        target: str,
        prices: pd.DataFrame,
        symbol: str,
        data_timestamp: datetime | None = None,
        adv: float = float("inf"),
        planned_dollars: float = 0.0,
        exclude_families: set | None = None,
    ) -> AllPredictions:
        """
        Generate predictions for all horizons.

        Args:
            target: Target name (e.g., "ret_5m")
            prices: OHLCV DataFrame
            symbol: Trading symbol
            data_timestamp: Data timestamp for freshness
            adv: Average daily volume
            planned_dollars: Planned trade size
            exclude_families: Families to skip (e.g., CS ranking families
                handled by CrossSectionalRankingPredictor)

        Returns:
            AllPredictions with all horizons
        """
        if data_timestamp is None:
            data_timestamp = datetime.now(timezone.utc)

        # Get available families for this target
        available_families = self.loader.list_available_families(target)
        families_to_use = self.families or available_families
        if exclude_families:
            families_to_use = [f for f in families_to_use if f not in exclude_families]

        all_preds = AllPredictions(
            symbol=symbol,
            timestamp=data_timestamp,
        )

        for horizon in self.horizons:
            horizon_preds = self._predict_horizon(
                target=target,
                horizon=horizon,
                prices=prices,
                symbol=symbol,
                families=families_to_use,
                data_timestamp=data_timestamp,
                adv=adv,
                planned_dollars=planned_dollars,
            )
            if horizon_preds.predictions:  # Only add if we got predictions
                all_preds.horizons[horizon] = horizon_preds

        return all_preds

    def predict_single_horizon(
        self,
        target: str,
        horizon: str,
        prices: pd.DataFrame,
        symbol: str,
        families: List[str] | None = None,
        data_timestamp: datetime | None = None,
        adv: float = float("inf"),
        planned_dollars: float = 0.0,
    ) -> HorizonPredictions:
        """
        Generate predictions for a single horizon.

        Args:
            target: Target name
            horizon: Horizon (e.g., "5m")
            prices: OHLCV DataFrame
            symbol: Trading symbol
            families: Model families to use
            data_timestamp: Data timestamp
            adv: Average daily volume
            planned_dollars: Planned trade size

        Returns:
            HorizonPredictions for the horizon
        """
        if data_timestamp is None:
            data_timestamp = datetime.now(timezone.utc)

        # Get available families
        available_families = self.loader.list_available_families(target)
        families_to_use = families or self.families or available_families

        return self._predict_horizon(
            target=target,
            horizon=horizon,
            prices=prices,
            symbol=symbol,
            families=families_to_use,
            data_timestamp=data_timestamp,
            adv=adv,
            planned_dollars=planned_dollars,
        )

    def _predict_horizon(
        self,
        target: str,
        horizon: str,
        prices: pd.DataFrame,
        symbol: str,
        families: List[str],
        data_timestamp: datetime,
        adv: float,
        planned_dollars: float,
    ) -> HorizonPredictions:
        """Generate predictions for a single horizon."""
        horizon_preds = HorizonPredictions(
            horizon=horizon,
            timestamp=data_timestamp,
        )

        for family in families:
            try:
                pred = self._predict_single(
                    target=target,
                    horizon=horizon,
                    family=family,
                    prices=prices,
                    symbol=symbol,
                    data_timestamp=data_timestamp,
                    adv=adv,
                    planned_dollars=planned_dollars,
                )
                if pred is not None:
                    horizon_preds.predictions[family] = pred

            except Exception as e:
                logger.warning(f"Prediction failed for {family}/{horizon}: {e}")

        return horizon_preds

    def _prepare_raw_sequence(
        self,
        prices: pd.DataFrame,
        target: str,
        family: str,
    ) -> Optional[np.ndarray]:
        """
        Prepare raw OHLCV sequence for inference.

        Extracts OHLCV columns from prices DataFrame, applies the same
        normalization used during training (SST), and returns the latest
        normalized bar to push into the ring buffer one bar at a time.

        CONTRACT: INTEGRATION_CONTRACTS.md v1.3
        - sequence_normalization must match training
        - sequence_channels must be ["open", "high", "low", "close", "volume"]

        Args:
            prices: OHLCV DataFrame
            target: Target name
            family: Model family

        Returns:
            Normalized OHLCV bar (5,) or None on error
        """
        seq_config = self.loader.get_sequence_config(target, family)
        if not seq_config:
            logger.error(f"No sequence config for {target}:{family}")
            return None

        seq_len = seq_config["sequence_length"]
        normalization = seq_config["sequence_normalization"]
        channels = seq_config["sequence_channels"]

        # Map channel names to DataFrame columns (case-insensitive)
        col_map = {}
        for ch in channels:
            matched = next(
                (c for c in prices.columns if c.lower() == ch.lower()), None
            )
            col_map[ch] = matched

        missing = [ch for ch in channels if col_map[ch] is None]
        if missing:
            logger.error(f"Missing OHLCV columns for {target}:{family}: {missing}")
            return None

        # Extract columns in channel order; +1 for normalization context
        ohlcv_cols = [col_map[ch] for ch in channels]
        ohlcv_df = prices[ohlcv_cols].tail(seq_len + 1)

        if len(ohlcv_df) < 2:
            logger.warning(
                f"Insufficient data for {target}:{family}: "
                f"need >= 2 bars, got {len(ohlcv_df)}"
            )
            return None

        ohlcv_array = ohlcv_df.values.astype(np.float32)

        # Normalize using the SST function from TRAINING
        normalized = _normalize_ohlcv_sequence(ohlcv_array, method=normalization)

        # Return the latest bar as a 1D array to push into the ring buffer.
        # The buffer accumulates bars over time; we push one per cycle.
        return normalized[-1]  # shape: (5,)

    def _predict_single(
        self,
        target: str,
        horizon: str,
        family: str,
        prices: pd.DataFrame,
        symbol: str,
        data_timestamp: datetime,
        adv: float,
        planned_dollars: float,
    ) -> Optional[ModelPrediction]:
        """Generate single model prediction."""

        # Check input mode
        input_mode = self.loader.get_input_mode(target, family)

        if input_mode == "raw_sequence":
            # Raw OHLCV path: prepare normalized bar instead of building features
            features = self._prepare_raw_sequence(prices, target, family)
        else:
            # Existing: build computed features
            builder = self._get_feature_builder(target, family)
            features = builder.build_features(prices, symbol)

        if features is None or np.any(np.isnan(features)):
            logger.warning(f"Invalid features for {family}/{symbol}")
            return None

        # Run inference
        raw_pred = self.engine.predict(target, family, features, symbol)

        if np.isnan(raw_pred):
            return None

        # Standardize
        std_pred = self.standardizer.standardize(raw_pred, family, horizon)

        # Calculate confidence
        confidence = self.confidence_scorer.calculate_confidence(
            model=family,
            horizon=horizon,
            data_timestamp=data_timestamp,
            adv=adv,
            planned_dollars=planned_dollars,
        )

        # Calibrated prediction
        calibrated = self.confidence_scorer.apply_confidence(
            std_pred, confidence.overall
        )

        return ModelPrediction(
            family=family,
            horizon=horizon,
            raw=raw_pred,
            standardized=std_pred,
            confidence=confidence,
            calibrated=calibrated,
        )

    def predict_single_target(
        self,
        target: str,
        prices: pd.DataFrame,
        symbol: str,
        data_timestamp: datetime | None = None,
    ) -> Optional[ModelPrediction]:
        """
        Get a single prediction for a target (first available horizon + family).

        Used by barrier gate for peak/valley predictions.

        Args:
            target: Target name (e.g., "will_peak_5m")
            prices: OHLCV DataFrame
            symbol: Trading symbol
            data_timestamp: Data timestamp for freshness

        Returns:
            ModelPrediction or None if no model available
        """
        if data_timestamp is None:
            data_timestamp = datetime.now(timezone.utc)

        # Use first configured horizon
        horizon = self.horizons[0] if self.horizons else "5m"

        available_families = self.loader.list_available_families(target)
        if not available_families:
            return None

        # Try first available family
        for family in available_families:
            try:
                pred = self._predict_single(
                    target=target,
                    horizon=horizon,
                    family=family,
                    prices=prices,
                    symbol=symbol,
                    data_timestamp=data_timestamp,
                    adv=float("inf"),
                    planned_dollars=0.0,
                )
                if pred is not None:
                    return pred
            except Exception as e:
                logger.debug(f"predict_single_target failed for {family}/{target}: {e}")
                continue

        return None

    def update_actuals(
        self,
        model: str,
        horizon: str,
        prediction: float,
        actual_return: float,
    ) -> None:
        """
        Update with actual returns for IC tracking.

        Args:
            model: Model family
            horizon: Horizon
            prediction: Previous prediction
            actual_return: Realized return
        """
        self.confidence_scorer.update_with_actual(
            model, horizon, prediction, actual_return
        )

    def reset(self) -> None:
        """Reset all stateful components."""
        self.standardizer.reset()
        self.confidence_scorer.reset()
        self.engine.reset_buffers()
        logger.debug("MultiHorizonPredictor reset")
