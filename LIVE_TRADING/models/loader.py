"""
Model Loader
============

Loads trained models from TRAINING run artifacts.
Uses SST path helpers for artifact location.

Security:
- H2 FIX: Verifies model file checksums before pickle loading
- Checksums stored in model_meta.json during training
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import iterdir_sorted

from LIVE_TRADING.common.constants import (
    SEQUENTIAL_FAMILIES,
    TF_FAMILIES,
    TREE_FAMILIES,
)
from LIVE_TRADING.common.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


def _compute_file_checksum(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute checksum of a file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hex digest of file hash
    """
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_model_checksum(
    model_path: Path,
    expected_checksum: str | None,
    strict: bool = False,
) -> bool:
    """
    H2 FIX: Verify model file checksum before loading.

    Args:
        model_path: Path to model file
        expected_checksum: Expected hash (from metadata)
        strict: If True, raise on mismatch; if False, warn only

    Returns:
        True if checksum matches or not available

    Raises:
        ModelLoadError: If strict=True and checksum mismatch
    """
    if expected_checksum is None:
        logger.debug(f"No checksum available for {model_path.name}, skipping verification")
        return True

    actual_checksum = _compute_file_checksum(model_path)

    if actual_checksum != expected_checksum:
        msg = (
            f"Checksum mismatch for {model_path.name}: "
            f"expected {expected_checksum[:16]}..., got {actual_checksum[:16]}..."
        )
        if strict:
            raise ModelLoadError("N/A", "N/A", f"Security: {msg}")
        else:
            logger.warning(f"Security warning: {msg}")
            return False

    logger.debug(f"Checksum verified for {model_path.name}")
    return True


class ModelLoader:
    """
    Loads models from a TRAINING run's artifact directory.

    Uses SST-compliant paths to locate model files.
    """

    def __init__(
        self,
        run_root: Path | str,
        verify_checksums: bool | None = None,
        strict_checksums: bool = False,
    ):
        """
        Initialize model loader.

        Args:
            run_root: Path to run directory (e.g., RESULTS/runs/<run_id>/<timestamp>)
            verify_checksums: Whether to verify model checksums (default: from config)
            strict_checksums: If True, raise on checksum mismatch; if False, warn only
        """
        self.run_root = Path(run_root)

        if not self.run_root.exists():
            raise ModelLoadError("N/A", "N/A", f"Run root does not exist: {run_root}")

        # H2 FIX: Checksum verification settings
        self._verify_checksums = verify_checksums if verify_checksums is not None else get_cfg(
            "live_trading.models.verify_checksums", default=True
        )
        self._strict_checksums = strict_checksums

        # Cache loaded models
        self._model_cache: Dict[str, Tuple[Any, Dict[str, Any]]] = {}

        # Target index from manifest
        self._target_index = self._load_target_index()

        logger.info(f"ModelLoader initialized from {run_root} (verify_checksums={self._verify_checksums})")

    def _load_target_index(self) -> Dict[str, Any]:
        """Load target index from run manifest."""
        manifest_path = self.run_root / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
                return manifest.get("target_index", {})
        return {}

    def get_target_models_dir(
        self,
        target: str,
        view: str = "CROSS_SECTIONAL",
    ) -> Path:
        """
        Get the models directory for a target.

        Args:
            target: Target name
            view: View type (CROSS_SECTIONAL or SEQUENTIAL)

        Returns:
            Path to models directory
        """
        # SST path structure: targets/<target>/models/view=<view>/
        return self.run_root / "targets" / target / "models" / f"view={view}"

    def get_family_dir(
        self,
        target: str,
        family: str,
        view: str = "CROSS_SECTIONAL",
    ) -> Path:
        """
        Get the directory for a specific family's model.

        Args:
            target: Target name
            family: Model family name
            view: View type

        Returns:
            Path to family model directory
        """
        models_dir = self.get_target_models_dir(target, view)
        return models_dir / f"family={family}"

    def load_model(
        self,
        target: str,
        family: str,
        view: str = "CROSS_SECTIONAL",
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a model and its metadata.

        Args:
            target: Target name
            family: Model family name
            view: View type

        Returns:
            Tuple of (model, metadata_dict)

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        cache_key = f"{target}:{family}:{view}"

        if cache_key in self._model_cache:
            logger.debug(f"Using cached model: {cache_key}")
            return self._model_cache[cache_key]

        family_dir = self.get_family_dir(target, family, view)

        if not family_dir.exists():
            raise ModelLoadError(family, target, f"Family directory not found: {family_dir}")

        # Load metadata first
        meta_path = family_dir / "model_meta.json"
        if not meta_path.exists():
            raise ModelLoadError(family, target, f"Model metadata not found: {meta_path}")

        with open(meta_path) as f:
            metadata = json.load(f)

        # Load model based on family type
        model = self._load_model_by_family(family, family_dir, metadata, target)

        # Cache and return
        self._model_cache[cache_key] = (model, metadata)
        logger.info(f"Loaded model: {cache_key}")

        return model, metadata

    def _load_model_by_family(
        self,
        family: str,
        family_dir: Path,
        metadata: Dict[str, Any],
        target: str,
    ) -> Any:
        """Load model using family-specific loader."""

        if family in TREE_FAMILIES:
            return self._load_tree_model(family_dir, family, target, metadata)
        elif family in TF_FAMILIES:
            return self._load_keras_model(family_dir, metadata, family, target)
        else:
            # Fallback to pickle
            return self._load_pickle_model(family_dir, family, target, metadata)

    def _load_tree_model(
        self,
        family_dir: Path,
        family: str,
        target: str,
        metadata: Dict[str, Any],
    ) -> Any:
        """Load tree-based model (LightGBM, XGBoost, CatBoost)."""
        model_path = family_dir / "model.pkl"

        if not model_path.exists():
            # Try joblib format
            model_path = family_dir / "model.joblib"

        if not model_path.exists():
            # Try txt format (native LightGBM format)
            model_path = family_dir / "model.txt"
            if model_path.exists():
                return self._load_lightgbm_native(model_path, family, target)

        if not model_path.exists():
            raise ModelLoadError(family, target, f"No model file found in {family_dir}")

        # H2 FIX: Verify checksum before loading pickle
        if self._verify_checksums:
            expected_checksum = metadata.get("model_checksum")
            _verify_model_checksum(model_path, expected_checksum, self._strict_checksums)

        with open(model_path, "rb") as f:
            return pickle.load(f)

    def _load_lightgbm_native(self, model_path: Path, family: str, target: str) -> Any:
        """Load LightGBM model from native text format."""
        try:
            import lightgbm as lgb
            return lgb.Booster(model_file=str(model_path))
        except ImportError:
            raise ModelLoadError(family, target, "LightGBM not installed")
        except Exception as e:
            raise ModelLoadError(family, target, f"Failed to load LightGBM model: {e}")

    def _load_keras_model(
        self,
        family_dir: Path,
        metadata: Dict[str, Any],
        family: str,
        target: str,
    ) -> Any:
        """Load Keras/TensorFlow model."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ModelLoadError(family, target, "TensorFlow not installed")

        model_path = family_dir / "model.h5"

        if not model_path.exists():
            # Try SavedModel format
            model_path = family_dir / "model"

        if not model_path.exists():
            # Try keras format
            model_path = family_dir / "model.keras"

        if not model_path.exists():
            raise ModelLoadError(family, target, f"No model file found in {family_dir}")

        try:
            model = tf.keras.models.load_model(str(model_path), compile=False)
            return model
        except Exception as e:
            raise ModelLoadError(family, target, f"Failed to load Keras model: {e}")

    def _load_pickle_model(
        self,
        family_dir: Path,
        family: str,
        target: str,
        metadata: Dict[str, Any],
    ) -> Any:
        """Load generic pickle model."""
        model_path = family_dir / "model.pkl"

        if not model_path.exists():
            raise ModelLoadError(family, target, f"No model file found in {family_dir}")

        # H2 FIX: Verify checksum before loading pickle
        if self._verify_checksums:
            expected_checksum = metadata.get("model_checksum")
            _verify_model_checksum(model_path, expected_checksum, self._strict_checksums)

        with open(model_path, "rb") as f:
            return pickle.load(f)

    def get_input_mode(
        self,
        target: str,
        family: str,
        view: str = "CROSS_SECTIONAL",
    ) -> str:
        """
        Get the input mode for a model.

        CONTRACT: INTEGRATION_CONTRACTS.md v1.3
        - "features" (default): Traditional feature-based input
        - "raw_sequence": Raw OHLCV bar sequences

        Returns "features" for models without input_mode field (backward compat).

        Args:
            target: Target name
            family: Model family name
            view: View type

        Returns:
            Input mode string
        """
        _, metadata = self.load_model(target, family, view)
        return metadata.get("input_mode", "features")

    def get_sequence_config(
        self,
        target: str,
        family: str,
        view: str = "CROSS_SECTIONAL",
    ) -> Dict[str, Any]:
        """
        Get sequence configuration for raw_sequence models.

        CONTRACT: INTEGRATION_CONTRACTS.md v1.3
        - sequence_length: bars in sequence
        - sequence_channels: OHLCV channel names
        - sequence_normalization: normalization method

        Args:
            target: Target name
            family: Model family name
            view: View type

        Returns:
            Dict with sequence config, or empty dict for feature-based models.
        """
        _, metadata = self.load_model(target, family, view)
        if metadata.get("input_mode", "features") != "raw_sequence":
            return {}

        return {
            "sequence_length": metadata.get("sequence_length", 64),
            "sequence_channels": metadata.get(
                "sequence_channels",
                ["open", "high", "low", "close", "volume"],
            ),
            "sequence_normalization": metadata.get("sequence_normalization", "returns"),
        }

    def get_feature_list(
        self,
        target: str,
        family: str,
        view: str = "CROSS_SECTIONAL",
    ) -> List[str]:
        """
        Get the feature list for a model.

        CONTRACT: See INTEGRATION_CONTRACTS.md for schema
        - New models: feature_list field (sorted)
        - Legacy models: fallback to features or feature_names
        - raw_sequence models: empty list (no computed features)

        Args:
            target: Target name
            family: Model family name
            view: View type

        Returns:
            List of feature names in order
        """
        _, metadata = self.load_model(target, family, view)

        # Raw sequence models have empty feature_list by contract
        if metadata.get("input_mode", "features") == "raw_sequence":
            return []

        # CONTRACT: feature_list is canonical; fallback to legacy fields for old models
        feature_list = metadata.get("feature_list")
        if feature_list is not None:
            return list(feature_list)

        # Backward compatibility: check legacy field names
        features = metadata.get("features") or metadata.get("feature_names")
        if features is not None:
            logger.warning(
                f"Model {family}/{target} uses legacy 'features' field; "
                f"re-train for 'feature_list' compliance"
            )
            return list(features)

        logger.warning(f"No feature list found in metadata for {family}/{target}")
        return []

    def get_routing_decision(
        self,
        target: str,
        view: str = "CROSS_SECTIONAL",
    ) -> Dict[str, Any]:
        """
        Get the routing decision for a target.

        Args:
            target: Target name
            view: View type

        Returns:
            Routing decision dict with selected family, metrics, etc.
        """
        models_dir = self.get_target_models_dir(target, view)
        routing_path = models_dir / "routing_decision.json"

        if not routing_path.exists():
            return {}

        with open(routing_path) as f:
            return json.load(f)

    def list_available_targets(self) -> List[str]:
        """List all targets in the run."""
        targets_dir = self.run_root / "targets"
        if not targets_dir.exists():
            return []

        return sorted([
            d.name for d in iterdir_sorted(targets_dir)
            if d.is_dir()
        ])

    def list_available_families(
        self,
        target: str,
        view: str = "CROSS_SECTIONAL",
    ) -> List[str]:
        """List available model families for a target."""
        models_dir = self.get_target_models_dir(target, view)
        if not models_dir.exists():
            return []

        families = []
        for d in iterdir_sorted(models_dir):
            if d.is_dir() and d.name.startswith("family="):
                family_name = d.name.replace("family=", "")
                families.append(family_name)

        return families

    def get_model_metrics(
        self,
        target: str,
        family: str,
        view: str = "CROSS_SECTIONAL",
    ) -> Dict[str, float]:
        """Get model metrics (AUC, etc.) from metadata."""
        _, metadata = self.load_model(target, family, view)
        return metadata.get("metrics", {})

    def get_selected_family(
        self,
        target: str,
        view: str = "CROSS_SECTIONAL",
    ) -> Optional[str]:
        """
        Get the routing-selected family for a target.

        Args:
            target: Target name
            view: View type

        Returns:
            Selected family name or None
        """
        routing = self.get_routing_decision(target, view)
        return routing.get("selected_family")

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._model_cache.clear()
        logger.debug("Model cache cleared")


def load_model_from_run(
    run_root: Path | str,
    target: str,
    family: str,
    view: str = "CROSS_SECTIONAL",
) -> Tuple[Any, Dict[str, Any]]:
    """
    Convenience function to load a single model.

    Args:
        run_root: Path to run directory
        target: Target name
        family: Model family
        view: View type

    Returns:
        Tuple of (model, metadata)
    """
    loader = ModelLoader(run_root)
    return loader.load_model(target, family, view)
