"""
Volatility Scaling
==================

Scales positions based on volatility estimates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from CONFIG.config_loader import get_cfg

from LIVE_TRADING.common.constants import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class VolatilityScaler:
    """
    Scales alpha to position weight using volatility.

    z = clip(α / σ, -z_max, z_max)
    weight = z × (max_weight / z_max)

    This converts the alpha signal (expected return) to a
    position weight by dividing by volatility, giving a
    Sharpe-ratio-like scaling.
    """

    def __init__(
        self,
        z_max: float | None = None,
        max_weight: float | None = None,
    ):
        """
        Initialize volatility scaler.

        Args:
            z_max: Maximum z-score (clips alpha/vol ratio)
            max_weight: Maximum position weight (e.g., 0.05 = 5%)
        """
        self.z_max = z_max if z_max is not None else get_cfg(
            "live_trading.sizing.z_max",
            default=DEFAULT_CONFIG["z_max"],
        )
        self.max_weight = max_weight if max_weight is not None else get_cfg(
            "live_trading.sizing.max_weight",
            default=DEFAULT_CONFIG["max_weight"],
        )
        # Guard against z_max=0 which would cause ZeroDivisionError in scale()
        if self.z_max <= 0:
            logger.warning(f"z_max={self.z_max} is invalid, using default 3.0")
            self.z_max = 3.0
        logger.info(f"VolatilityScaler: z_max={self.z_max}, max_weight={self.max_weight}")

    def scale(self, alpha: float, volatility: float) -> float:
        """
        Scale alpha to position weight.

        Args:
            alpha: Alpha signal (decimal return, e.g., 0.01 = 1%)
            volatility: Volatility estimate (annualized std, e.g., 0.20 = 20%)

        Returns:
            Target weight (-max_weight to +max_weight)
        """
        if volatility <= 0:
            return 0.0

        z = alpha / volatility
        z_clipped = np.clip(z, -self.z_max, self.z_max)
        weight = z_clipped * (self.max_weight / self.z_max)

        return float(weight)

    def scale_batch(
        self,
        alphas: Dict[str, float],
        volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Scale multiple symbols.

        Args:
            alphas: Dict mapping symbol to alpha
            volatilities: Dict mapping symbol to volatility

        Returns:
            Dict mapping symbol to target weight
        """
        return {
            symbol: self.scale(alpha, volatilities.get(symbol, 0.0))
            for symbol, alpha in alphas.items()
        }

    def get_z_score(self, alpha: float, volatility: float) -> float:
        """
        Get raw z-score before clipping.

        Args:
            alpha: Alpha signal
            volatility: Volatility estimate

        Returns:
            Raw z-score (alpha/volatility)
        """
        if volatility <= 0:
            return 0.0
        return alpha / volatility

    def inverse_scale(self, weight: float, volatility: float) -> float:
        """
        Convert weight back to alpha.

        Useful for understanding what alpha would be needed
        for a given position weight.

        Args:
            weight: Position weight
            volatility: Volatility estimate

        Returns:
            Alpha signal that would produce this weight
        """
        z = weight * (self.z_max / self.max_weight)
        return z * volatility

    def get_analysis(self, alpha: float, volatility: float) -> Dict[str, Any]:
        """
        Get detailed analysis of scaling.

        Args:
            alpha: Alpha signal
            volatility: Volatility

        Returns:
            Analysis dict
        """
        z = self.get_z_score(alpha, volatility)
        z_clipped = np.clip(z, -self.z_max, self.z_max)
        weight = self.scale(alpha, volatility)

        return {
            "alpha": alpha,
            "volatility": volatility,
            "z_score_raw": z,
            "z_score_clipped": float(z_clipped),
            "was_clipped": abs(z) > self.z_max,
            "weight": weight,
            "weight_pct": weight * 100,
            "z_max": self.z_max,
            "max_weight": self.max_weight,
        }
