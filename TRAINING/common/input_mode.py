# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Input Mode Handling for Training Pipeline

This module defines the InputMode enum and helper functions for determining
how data is fed to models:

- FEATURES: Traditional mode using computed technical indicators
- RAW_SEQUENCE: Raw OHLCV bars fed directly to sequence models

See INTEGRATION_CONTRACTS.md v1.3 for model_meta field definitions.
See .claude/plans/raw-ohlcv-sequence-mode.md for implementation details.

Usage:
    from TRAINING.common.input_mode import InputMode, get_input_mode, is_raw_sequence_mode

    mode = get_input_mode()  # Returns InputMode from config
    if is_raw_sequence_mode():
        # Skip feature selection, use raw OHLCV sequences
        ...
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional, Set

logger = logging.getLogger(__name__)


class InputMode(str, Enum):
    """
    Input mode for training pipeline.

    FEATURES: Traditional mode - compute technical indicators, select features
    RAW_SEQUENCE: Feed raw OHLCV bars directly to sequence models
    """

    FEATURES = "features"
    RAW_SEQUENCE = "raw_sequence"

    @classmethod
    def from_string(cls, value: str) -> "InputMode":
        """
        Convert string to InputMode enum.

        Args:
            value: String value ("features" or "raw_sequence")

        Returns:
            InputMode enum value

        Raises:
            ValueError: If value is not a valid input mode
        """
        try:
            return cls(value.lower().strip())
        except ValueError:
            valid = [m.value for m in cls]
            raise ValueError(
                f"Invalid input_mode: '{value}'. Must be one of: {valid}"
            )


# Model families that support raw sequence mode
# These families can accept (N, seq_len, 5) input directly
RAW_SEQUENCE_FAMILIES: Set[str] = {
    "LSTM",
    "Transformer",
    "CNN1D",
    "TabLSTM",
    "TabTransformer",
    "TabCNN",
}

# Normalized (lowercase) versions for lookup
_RAW_SEQUENCE_FAMILIES_NORMALIZED: Set[str] = {f.lower() for f in RAW_SEQUENCE_FAMILIES}


def get_input_mode(
    config_value: Optional[str] = None,
    experiment_config: Optional[dict] = None,
) -> InputMode:
    """
    Get input mode from config with proper precedence.

    Precedence (highest to lowest):
    1. Explicit config_value parameter
    2. Experiment config (experiment_config["pipeline"]["input_mode"])
    3. Pipeline config (pipeline.input_mode)
    4. Default: FEATURES

    Args:
        config_value: Explicit override value
        experiment_config: Experiment configuration dict

    Returns:
        InputMode enum value
    """
    # Priority 1: Explicit parameter
    if config_value is not None:
        return InputMode.from_string(config_value)

    # Priority 2: Experiment config
    if experiment_config is not None:
        if isinstance(experiment_config, dict):
            exp_mode = experiment_config.get("pipeline", {}).get("input_mode")
        else:
            exp_mode = getattr(experiment_config, "input_mode", None)
        if exp_mode is not None:
            return InputMode.from_string(exp_mode)

    # Priority 3: Pipeline config
    try:
        from CONFIG.config_loader import get_cfg

        cfg_mode = get_cfg("pipeline.input_mode", default="features")
        return InputMode.from_string(cfg_mode)
    except ImportError:
        pass

    # Priority 4: Default
    return InputMode.FEATURES


def is_raw_sequence_mode(
    config_value: Optional[str] = None,
    experiment_config: Optional[dict] = None,
) -> bool:
    """
    Check if we're in raw sequence mode.

    This is a convenience function for conditionals.

    Args:
        config_value: Explicit override value
        experiment_config: Experiment configuration dict

    Returns:
        True if input_mode is RAW_SEQUENCE
    """
    return get_input_mode(config_value, experiment_config) == InputMode.RAW_SEQUENCE


def is_family_raw_sequence_compatible(family: str) -> bool:
    """
    Check if a model family supports raw sequence mode.

    Args:
        family: Model family name (case-insensitive)

    Returns:
        True if family can accept raw OHLCV sequences
    """
    return family.lower() in _RAW_SEQUENCE_FAMILIES_NORMALIZED


def filter_families_for_input_mode(
    families: list,
    input_mode: InputMode,
) -> list:
    """
    Filter model families based on input mode.

    In FEATURES mode: All families are allowed
    In RAW_SEQUENCE mode: Only sequence-compatible families are allowed

    Args:
        families: List of model family names
        input_mode: Current input mode

    Returns:
        Filtered list of families
    """
    if input_mode == InputMode.FEATURES:
        return families

    # RAW_SEQUENCE mode: filter to compatible families only
    filtered = [f for f in families if is_family_raw_sequence_compatible(f)]

    if len(filtered) < len(families):
        removed = [f for f in families if not is_family_raw_sequence_compatible(f)]
        logger.info(
            f"RAW_SEQUENCE mode: Filtered out non-sequential families: {removed}"
        )

    if not filtered:
        logger.warning(
            f"RAW_SEQUENCE mode: No compatible families found! "
            f"Compatible families are: {RAW_SEQUENCE_FAMILIES}"
        )

    return filtered


def get_raw_sequence_config(
    experiment_config: Optional[dict] = None,
) -> dict:
    """
    Get raw sequence configuration from config.

    Returns:
        Dict with sequence config:
        {
            "length_minutes": int,
            "channels": List[str],
            "normalization": str,
            "gap_handling": str,
            "gap_tolerance": float,
            "auto_clamp": bool,
        }
    """
    try:
        from CONFIG.config_loader import get_cfg

        # Check experiment config first
        if experiment_config is not None:
            exp_seq = experiment_config.get("pipeline", {}).get("sequence", {})
            if exp_seq:
                return {
                    "length_minutes": exp_seq.get(
                        "length_minutes",
                        get_cfg("pipeline.sequence.default_length_minutes", default=320),
                    ),
                    "channels": exp_seq.get(
                        "channels",
                        get_cfg(
                            "pipeline.sequence.default_channels",
                            default=["open", "high", "low", "close", "volume"],
                        ),
                    ),
                    "normalization": exp_seq.get(
                        "normalization",
                        get_cfg("pipeline.sequence.normalization", default="returns"),
                    ),
                    "gap_handling": exp_seq.get(
                        "gap_handling",
                        get_cfg("pipeline.sequence.gap_handling", default="split"),
                    ),
                    "gap_tolerance": exp_seq.get(
                        "gap_tolerance",
                        get_cfg("pipeline.sequence.gap_tolerance", default=1.5),
                    ),
                    "auto_clamp": exp_seq.get(
                        "auto_clamp",
                        get_cfg("pipeline.sequence.auto_clamp", default=False),
                    ),
                }

        # Fall back to pipeline config
        return {
            "length_minutes": get_cfg("pipeline.sequence.default_length_minutes", default=320),
            "channels": get_cfg(
                "pipeline.sequence.default_channels",
                default=["open", "high", "low", "close", "volume"],
            ),
            "normalization": get_cfg("pipeline.sequence.normalization", default="returns"),
            "gap_handling": get_cfg("pipeline.sequence.gap_handling", default="split"),
            "gap_tolerance": get_cfg("pipeline.sequence.gap_tolerance", default=1.5),
            "auto_clamp": get_cfg("pipeline.sequence.auto_clamp", default=False),
        }

    except ImportError:
        # Config not available, use defaults
        return {
            "length_minutes": 320,
            "channels": ["open", "high", "low", "close", "volume"],
            "normalization": "returns",
            "gap_handling": "split",
            "gap_tolerance": 1.5,
            "auto_clamp": False,
        }
