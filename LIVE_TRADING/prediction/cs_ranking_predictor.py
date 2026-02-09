"""
Cross-Sectional Ranking Predictor
==================================

Handles inference for models trained with cross-sectional ranking loss.

CS ranking models output relative scores (not absolute return forecasts).
Predictions only have meaning when compared across symbols at the same
timestamp. This module:

1. Collects raw predictions for all symbols in the universe
2. Ranks them cross-sectionally per (horizon, family)
3. Converts percentile ranks to z-score-like signals via probit transform
4. Produces AllPredictions compatible with the existing blending pipeline

CONTRACT: INTEGRATION_CONTRACTS.md v1.4
Consumer of: cross_sectional_ranking.* fields in model_meta.json
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import norm, rankdata

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.models.loader import ModelLoader
from LIVE_TRADING.models.inference import InferenceEngine
from .predictor import (
    AllPredictions,
    HorizonPredictions,
    ModelPrediction,
    MultiHorizonPredictor,
)
logger = logging.getLogger(__name__)

# Clip bounds for probit-transformed signal (matches ZScoreStandardizer)
_SIGNAL_CLIP_MIN = -3.0
_SIGNAL_CLIP_MAX = 3.0

# Percentile rank boundaries to avoid infinite probit values
_RANK_EPSILON = 0.001


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """
    Compute percentile ranks in (0, 1].

    Uses scipy.stats.rankdata with 'average' method for tied values.
    Divides by N to produce ranks in (0, 1].

    Args:
        values: 1D array of scores

    Returns:
        Percentile ranks array, same shape as values
    """
    n = len(values)
    if n == 0:
        return np.array([])
    ranks = rankdata(values, method="average")
    return ranks / n


def _rank_to_signal(percentile_rank: float) -> float:
    """
    Convert percentile rank [0, 1] to z-score-like signal via probit transform.

    Maps uniform distribution to standard normal:
    - 0.5 -> 0.0   (neutral)
    - 0.95 -> ~1.64  (strong long)
    - 0.05 -> ~-1.64 (strong short)

    Clipped to [-3, 3] to match ZScoreStandardizer output scale.

    Args:
        percentile_rank: Rank in [0, 1]

    Returns:
        Signal value in [-3, 3]
    """
    clipped = np.clip(percentile_rank, _RANK_EPSILON, 1.0 - _RANK_EPSILON)
    signal = norm.ppf(clipped)
    return float(np.clip(signal, _SIGNAL_CLIP_MIN, _SIGNAL_CLIP_MAX))


class CrossSectionalRankingPredictor:
    """
    Handles inference for models trained with cross-sectional ranking loss.

    Unlike pointwise models that produce absolute return forecasts,
    CS ranking models produce relative scores. This class:
    1. Collects raw predictions for all symbols at a timestamp
    2. Ranks them cross-sectionally per family
    3. Converts ranks to alpha signals compatible with blending pipeline

    CONTRACT: INTEGRATION_CONTRACTS.md v1.4
    Methods: predict(), _get_ranking_config()
    """

    def __init__(
        self,
        loader: ModelLoader,
        engine: InferenceEngine,
        predictor: MultiHorizonPredictor,
        min_universe_size: int | None = None,
    ):
        """
        Initialize CS ranking predictor.

        Args:
            loader: ModelLoader for metadata access
            engine: InferenceEngine for raw predictions
            predictor: MultiHorizonPredictor for feature building and raw scores
            min_universe_size: Minimum symbols required for ranking (default: config)
        """
        self.loader = loader
        self.engine = engine
        self.predictor = predictor
        self.min_universe_size = (
            min_universe_size if min_universe_size is not None
            else get_cfg("live_trading.cs_ranking.min_universe_size", default=5)
        )

        logger.info(
            f"CrossSectionalRankingPredictor initialized: "
            f"min_universe_size={self.min_universe_size}"
        )

    def _get_ranking_config(
        self,
        target: str,
        family: str,
    ) -> Dict[str, Any]:
        """
        Get cross-sectional ranking configuration from model metadata.

        CONTRACT: INTEGRATION_CONTRACTS.md v1.4
        Reads cross_sectional_ranking.* fields.

        Args:
            target: Target name
            family: Model family name

        Returns:
            CS ranking config dict, or {} if not a CS ranking model.
        """
        return self.loader.get_cs_ranking_config(target, family)

    def is_cs_ranking_model(self, target: str, family: str) -> bool:
        """Check if a model uses cross-sectional ranking."""
        return bool(self._get_ranking_config(target, family))

    def get_cs_families(self, target: str) -> List[str]:
        """Get families that are CS ranking models for a target."""
        families = self.loader.list_available_families(target)
        return [f for f in families if self.is_cs_ranking_model(target, f)]

    def predict(
        self,
        target: str,
        universe: Dict[str, Any],
        horizons: List[str],
        data_timestamp: datetime | None = None,
        adv_map: Dict[str, float] | None = None,
    ) -> Dict[str, AllPredictions]:
        """
        Generate cross-sectionally ranked predictions for all symbols.

        CONTRACT: INTEGRATION_CONTRACTS.md v1.4
        Uses ranking inference path for CS-trained models.

        Steps:
        1. Identify CS ranking families for this target
        2. Collect raw model scores for every symbol x family
        3. Rank symbols cross-sectionally per family
        4. Convert percentile ranks to z-score-like signals
        5. Build AllPredictions per symbol

        Args:
            target: Target name
            universe: Dict[symbol, pd.DataFrame] mapping symbols to price data
            horizons: List of horizon strings
            data_timestamp: Current timestamp
            adv_map: Optional Dict[symbol, float] for ADV values

        Returns:
            Dict[symbol, AllPredictions] with ranked predictions.
            Empty dict if no CS families or universe too small.
        """
        if data_timestamp is None:
            data_timestamp = datetime.now(timezone.utc)
        if adv_map is None:
            adv_map = {}

        cs_families = self.get_cs_families(target)
        if not cs_families:
            return {}

        # Step 1: Collect raw scores for each symbol x family
        # We use the same score for all horizons since CS ranking models
        # produce a single cross-sectional score (not horizon-specific)
        raw_scores = self._collect_raw_scores(
            target, universe, cs_families, data_timestamp
        )

        if not raw_scores:
            return {}

        # Step 2: Rank cross-sectionally per family
        ranked = self._rank_cross_sectionally(raw_scores, cs_families)

        # Step 3: Build AllPredictions per symbol
        return self._build_predictions(
            ranked, horizons, cs_families, data_timestamp, adv_map
        )

    def _collect_raw_scores(
        self,
        target: str,
        universe: Dict[str, Any],
        cs_families: List[str],
        data_timestamp: datetime,
    ) -> Dict[str, Dict[str, float]]:
        """
        Collect raw model predictions for all symbols.

        Returns:
            Dict[symbol, Dict[family, raw_score]]
        """
        raw_scores: Dict[str, Dict[str, float]] = {}

        for symbol, prices in sorted_items(universe):
            family_scores: Dict[str, float] = {}
            for family in cs_families:
                try:
                    raw = self._get_raw_score(
                        target, family, prices, symbol, data_timestamp
                    )
                    if raw is not None and not np.isnan(raw):
                        family_scores[family] = raw
                except Exception as e:
                    logger.warning(
                        f"CS raw score failed for {symbol}/{family}: {e}"
                    )

            if family_scores:
                raw_scores[symbol] = family_scores

        return raw_scores

    def _get_raw_score(
        self,
        target: str,
        family: str,
        prices: Any,
        symbol: str,
        data_timestamp: datetime,
    ) -> Optional[float]:
        """
        Get raw prediction from model (before standardization/ranking).

        Routes through the predictor's feature building and inference engine,
        but returns the raw score without z-score standardization.
        """
        input_mode = self.loader.get_input_mode(target, family)

        if input_mode == "raw_sequence":
            features = self.predictor._prepare_raw_sequence(prices, target, family)
        else:
            builder = self.predictor._get_feature_builder(target, family)
            features = builder.build_features(prices, symbol)

        if features is None or np.any(np.isnan(features)):
            return None

        return self.engine.predict(target, family, features, symbol)

    def _rank_cross_sectionally(
        self,
        raw_scores: Dict[str, Dict[str, float]],
        cs_families: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Rank symbols cross-sectionally per family.

        Returns percentile ranks in (0, 1]. Families with fewer than
        min_universe_size symbols are skipped.

        Args:
            raw_scores: Dict[symbol, Dict[family, raw_score]]
            cs_families: List of CS ranking families

        Returns:
            Dict[symbol, Dict[family, percentile_rank]]
        """
        ranked: Dict[str, Dict[str, float]] = {sym: {} for sym in raw_scores}

        for family in cs_families:
            # Collect valid scores for this family
            family_scores: Dict[str, float] = {}
            for sym, scores in sorted_items(raw_scores):
                score = scores.get(family)
                if score is not None:
                    family_scores[sym] = score

            if len(family_scores) < self.min_universe_size:
                logger.debug(
                    f"CS ranking: {family} has {len(family_scores)} symbols "
                    f"< min {self.min_universe_size}, skipping"
                )
                continue

            # Rank (deterministic: sorted symbol order)
            symbols = sorted(family_scores.keys())
            values = np.array([family_scores[s] for s in symbols])
            ranks = _percentile_rank(values)

            for sym, rank in zip(symbols, ranks):
                ranked[sym][family] = float(rank)

        return ranked

    def _build_predictions(
        self,
        ranked: Dict[str, Dict[str, float]],
        horizons: List[str],
        cs_families: List[str],
        data_timestamp: datetime,
        adv_map: Dict[str, float],
    ) -> Dict[str, AllPredictions]:
        """
        Build AllPredictions from ranked scores.

        Converts percentile ranks to z-score-like signals via probit
        transform, applies confidence scoring, and packages into
        standard AllPredictions/HorizonPredictions/ModelPrediction objects.

        Returns:
            Dict[symbol, AllPredictions]
        """
        results: Dict[str, AllPredictions] = {}

        for symbol, family_ranks in sorted_items(ranked):
            if not family_ranks:
                continue

            all_preds = AllPredictions(
                symbol=symbol,
                timestamp=data_timestamp,
            )

            for horizon in horizons:
                hp = HorizonPredictions(
                    horizon=horizon,
                    timestamp=data_timestamp,
                )

                for family, pct_rank in sorted_items(family_ranks):
                    signal = _rank_to_signal(pct_rank)

                    confidence = self.predictor.confidence_scorer.calculate_confidence(
                        model=family,
                        horizon=horizon,
                        data_timestamp=data_timestamp,
                        adv=adv_map.get(symbol, float("inf")),
                        planned_dollars=0.0,
                    )

                    calibrated = self.predictor.confidence_scorer.apply_confidence(
                        signal, confidence.overall
                    )

                    hp.predictions[family] = ModelPrediction(
                        family=family,
                        horizon=horizon,
                        raw=pct_rank,
                        standardized=signal,
                        confidence=confidence,
                        calibrated=calibrated,
                    )

                if hp.predictions:
                    all_preds.horizons[horizon] = hp

            if all_preds.horizons:
                results[symbol] = all_preds

        return results
