# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Pipeline Stages Mixin for IntelligentTrainer.

Contains the main pipeline stage methods:
- _organize_by_cohort: Organize run directory by sample size
- rank_targets_auto: Automatic target ranking
- select_features_auto: Automatic feature selection
- _aggregate_feature_selection_summaries: Aggregate summaries

Extracted from intelligent_trainer.py for maintainability.

SST COMPLIANCE:
- Uses iterdir_sorted(), rglob_sorted() for deterministic iteration
- Uses sorted_items() for deterministic dict iteration
- Uses write_atomic_json() for atomic writes
- Uses View/Stage enums for consistent handling

DETERMINISM:
- All filesystem iterations use sorted variants
- All dict iterations use sorted_items()
"""

import hashlib
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

# SST: Import View/Stage enums for consistent handling
from TRAINING.orchestration.utils.scope_resolution import View, Stage

# DETERMINISM: Import deterministic iteration helpers
from TRAINING.common.utils.determinism_ordering import iterdir_sorted, rglob_sorted, sorted_items

# Import ranking/selection modules
from TRAINING.ranking import (
    rank_targets,
    discover_targets,
    load_target_configs,
    select_features_for_target,
    load_multi_model_config
)

# Import config helpers from sibling module
from TRAINING.orchestration.intelligent_trainer.config import (
    get_experiment_config_path as _get_experiment_config_path,
    load_experiment_config_safe as _load_experiment_config_safe,
)

# Import caching helpers from sibling module
from TRAINING.orchestration.intelligent_trainer.caching import (
    get_cache_key as _get_cache_key_impl,
    load_cached_rankings as _load_cached_rankings_impl,
    save_cached_rankings as _save_cached_rankings_impl,
    load_cached_features as _load_cached_features_impl,
    save_cached_features as _save_cached_features_impl,
)

# Import new config system (optional - for backward compatibility)
try:
    from CONFIG.config_builder import (
        build_feature_selection_config,
        build_target_ranking_config,
    )
    from CONFIG.config_schemas import ExperimentConfig, FeatureSelectionConfig, TargetRankingConfig
    _NEW_CONFIG_AVAILABLE = True
except ImportError:
    _NEW_CONFIG_AVAILABLE = False
    ExperimentConfig = None
    FeatureSelectionConfig = None
    TargetRankingConfig = None

# Import leakage sentinels
try:
    from TRAINING.common.leakage_sentinels import LeakageSentinel
    _SENTINELS_AVAILABLE = True
except ImportError:
    _SENTINELS_AVAILABLE = False
    LeakageSentinel = None

logger = logging.getLogger(__name__)


class PipelineStageMixin:
    """
    Mixin class providing pipeline stage methods for IntelligentTrainer.

    This mixin contains:
    - _organize_by_cohort: Organize run directory by sample size (n_effective)
    - rank_targets_auto: Automatic target ranking with caching
    - select_features_auto: Automatic feature selection per target
    - _aggregate_feature_selection_summaries: Aggregate summaries to globals/

    Methods in this mixin expect the following attributes on self:
    - output_dir: Path - Output directory for this run
    - _initial_output_dir: Path - Initial output directory (before cohort organization)
    - _run_name: str - Run name identifier
    - _n_effective: Optional[int] - Sample size (n_effective)
    - _bin_info: Optional[Dict] - Sample size bin information
    - data_dir: Path - Data directory
    - symbols: List[str] - List of symbols
    - experiment_config: Optional[ExperimentConfig] - Experiment configuration
    - cache_dir: Path - Cache directory
    - target_ranking_cache: Path - Target ranking cache file
    - feature_selection_cache: Path - Feature selection cache directory
    - sentinel: Optional[LeakageSentinel] - Leakage sentinel instance
    - fs_model_families: Optional[List[str]] - Feature selection model families

    Methods in this mixin call the following methods on self:
    - _get_sample_size_bin: Get sample size bin information
    - _get_cache_key: Generate cache key
    - _load_cached_rankings: Load cached rankings
    - _save_cached_rankings: Save rankings to cache
    - _load_cached_features: Load cached features
    - _save_cached_features: Save features to cache
    """

    # Type hints for expected attributes (set by the main class)
    output_dir: Path
    _initial_output_dir: Path
    _run_name: str
    _n_effective: Optional[int]
    _bin_info: Optional[Dict[str, Any]]
    data_dir: Path
    symbols: List[str]
    experiment_config: Any  # Optional[ExperimentConfig]
    cache_dir: Path
    target_ranking_cache: Path
    feature_selection_cache: Path
    sentinel: Any  # Optional[LeakageSentinel]
    fs_model_families: Optional[List[str]]

    def _organize_by_cohort(self):
        """
        Organize the run directory by sample size (n_effective) after first target is processed.
        Moves from RESULTS/_pending/{run_name}/ to RESULTS/{n_effective}/{run_name}/

        Example: RESULTS/25000/test_run_20251212_010000/

        Note: If n_effective was already determined in __init__, this is a no-op.
        """
        # If n_effective was already set and we're not in _pending/, we're already organized
        if self._n_effective is not None and "_pending" not in str(self.output_dir):
            return  # Already organized

        # If n_effective was set early but we're still in _pending/, move now
        if self._n_effective is not None and "_pending" in str(self.output_dir):
            repo_root = Path(__file__).parent.parent.parent.parent
            results_dir = repo_root / "RESULTS"
            bin_info = self._get_sample_size_bin(self._n_effective)
            bin_name = bin_info["bin_name"]
            if not hasattr(self, '_bin_info'):
                self._bin_info = bin_info
            new_output_dir = results_dir / bin_name / self._run_name

            if new_output_dir.exists():
                logger.warning(f"Sample size directory {new_output_dir} already exists, not moving")
                self.output_dir = new_output_dir
                self.cache_dir = self.output_dir / "cache"
                self.target_ranking_cache = self.cache_dir / "target_rankings.json"
                self.feature_selection_cache = self.cache_dir / "feature_selections"
                return

            import shutil
            bin_info = self._get_sample_size_bin(self._n_effective)
            bin_name = bin_info["bin_name"]
            if not hasattr(self, '_bin_info'):
                self._bin_info = bin_info
            new_output_dir.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Moving run from {self.output_dir} to {new_output_dir} (N={self._n_effective} determined early, bin={bin_name})")
            try:
                shutil.move(str(self.output_dir), str(new_output_dir))
                self.output_dir = new_output_dir
                self.cache_dir = self.output_dir / "cache"
                self.target_ranking_cache = self.cache_dir / "target_rankings.json"
                self.feature_selection_cache = self.cache_dir / "feature_selections"
                logger.info(f"Organized run by sample size bin (N={self._n_effective}, bin={bin_name}): {self.output_dir}")
                return
            except Exception as move_error:
                logger.error(f"Failed to move directory: {move_error}")
                # Stay in current location if move fails
                return

        try:
            # Try to find n_effective from target-first structure first, then legacy REPRODUCIBILITY
            # Target-first structure: targets/<target>/reproducibility/CROSS_SECTIONAL/cohort=<id>/metadata.json
            # Legacy structure: REPRODUCIBILITY/TARGET_RANKING/CROSS_SECTIONAL/<target>/cohort=<id>/metadata.json

            metadata_file = None

            # First, try target-first structure: search in targets/<target>/reproducibility/
            if (self._initial_output_dir / "targets").exists():
                # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                for target_dir in iterdir_sorted(self._initial_output_dir / "targets"):
                    if not target_dir.is_dir():
                        continue
                    repro_dir = target_dir / "reproducibility"
                    if repro_dir.exists():
                        # Check CROSS_SECTIONAL view first (most common)
                        cs_dir = repro_dir / View.CROSS_SECTIONAL.value
                        if cs_dir.exists():
                            # DETERMINISTIC: Use rglob_sorted to handle nested batch_/attempt_ structure deterministically
                            for cohort_dir in rglob_sorted(cs_dir, "cohort=*"):
                                if cohort_dir.is_dir() and cohort_dir.name.startswith("cohort="):
                                    candidate = cohort_dir / "metadata.json"
                                    if candidate.exists():
                                        metadata_file = candidate
                                        break
                        if metadata_file:
                            break

            # Fallback to legacy REPRODUCIBILITY structure
            if metadata_file is None:
                possible_repro_dirs = [
                    self.output_dir / "target_rankings" / "REPRODUCIBILITY",
                    self.output_dir / "REPRODUCIBILITY"
                ]

                target_ranking_dir = None
                for repro_dir in possible_repro_dirs:
                    candidate = repro_dir / "TARGET_RANKING"
                    if candidate.exists():
                        target_ranking_dir = candidate
                        break

                if target_ranking_dir is None:
                    logger.info(f"TARGET_RANKING directory not found at expected paths (checked: {[str(d / 'TARGET_RANKING') for d in possible_repro_dirs]})")
                    # Try recursive search as fallback (legacy structure only)
                    logger.info(f"Trying recursive search in {self._initial_output_dir}")
                    # DETERMINISM: Use rglob_sorted for deterministic iteration order
                    for candidate in rglob_sorted(self._initial_output_dir, "REPRODUCIBILITY/TARGET_RANKING/*/cohort=*/metadata.json"):
                        if candidate.exists():
                            try:
                                with open(candidate, 'r') as f:
                                    metadata = json.load(f)
                                n_effective = metadata.get('n_effective')
                                if n_effective is not None and n_effective > 0:
                                    self._n_effective = int(n_effective)
                                    logger.info(f"Found n_effective via recursive search: {self._n_effective} at {candidate.parent}")
                                    metadata_file = candidate
                                    break
                            except Exception as e:
                                logger.debug(f"Failed to read metadata from {candidate}: {e}")
                            continue
                    if self._n_effective is None:
                        logger.warning(f"Could not find n_effective via recursive search in {self._initial_output_dir}")
                        return
                else:
                    # Find first target's metadata.json to extract n_effective
                    # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                    for target_dir in iterdir_sorted(target_ranking_dir):
                        if not target_dir.is_dir():
                            continue

                        # Look for cohort= directories
                        # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                        for cohort_dir in iterdir_sorted(target_dir):
                            if cohort_dir.is_dir() and cohort_dir.name.startswith("cohort="):
                                metadata_file = cohort_dir / "metadata.json"
                                if metadata_file.exists():
                                    try:
                                        with open(metadata_file, 'r') as f:
                                            metadata = json.load(f)
                                        n_effective = metadata.get('n_effective')
                                        if n_effective is not None and n_effective > 0:
                                            self._n_effective = int(n_effective)
                                            logger.info(f"Found n_effective: {self._n_effective} from {metadata_file}")
                                            break
                                    except Exception as e:
                                        logger.debug(f"Failed to read metadata from {metadata_file}: {e}")
                                        continue
                        if self._n_effective is not None:
                            break

            # If we found n_effective, move the directory
            if self._n_effective is not None:
                # Move the entire run directory to sample-size-bin-organized location
                repo_root = Path(__file__).parent.parent.parent.parent
                results_dir = repo_root / "RESULTS"
                bin_info = self._get_sample_size_bin(self._n_effective)
                bin_name = bin_info["bin_name"]
                if not hasattr(self, '_bin_info'):
                    self._bin_info = bin_info
                new_output_dir = results_dir / bin_name / self._run_name

                if new_output_dir.exists():
                    logger.warning(f"Sample size directory {new_output_dir} already exists, not moving")
                    # Still update paths to point to existing directory
                    self.output_dir = new_output_dir
                    self.cache_dir = self.output_dir / "cache"
                    self.target_ranking_cache = self.cache_dir / "target_rankings.json"
                    self.feature_selection_cache = self.cache_dir / "feature_selections"
                    logger.info(f"Using existing sample size directory: {self.output_dir}")
                    return

                # Move the directory
                import shutil
                new_output_dir.parent.mkdir(parents=True, exist_ok=True)

                logger.info(f"Moving run from {self._initial_output_dir} to {new_output_dir}")
                try:
                    shutil.move(str(self._initial_output_dir), str(new_output_dir))
                    self.output_dir = new_output_dir

                    # Update cache_dir path and cache file paths
                    self.cache_dir = self.output_dir / "cache"
                    self.target_ranking_cache = self.cache_dir / "target_rankings.json"
                    self.feature_selection_cache = self.cache_dir / "feature_selections"

                    bin_info = self._get_sample_size_bin(self._n_effective)
                    bin_name = bin_info["bin_name"]
                    logger.info(f"Organized run by sample size bin (N={self._n_effective}, bin={bin_name}): {self.output_dir}")
                    return
                except Exception as move_error:
                    logger.error(f"Failed to move directory: {move_error}")
                    logger.debug(f"Move error traceback: {traceback.format_exc()}")
                    # Stay in _pending/ if move fails
                    return

            logger.debug(f"No metadata.json found to extract n_effective, waiting for first target")
        except Exception as e:
            logger.warning(f"Could not organize by sample size (will stay in _pending/): {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            # Stay in _pending/ if we can't determine cohort

        # If we still haven't organized, try one more time with more aggressive search
        if self._n_effective is None:
            try:
                # First try target-first structure: search in targets/<target>/reproducibility/
                if (self._initial_output_dir / "targets").exists():
                    # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                    for target_dir in iterdir_sorted(self._initial_output_dir / "targets"):
                        if not target_dir.is_dir():
                            continue
                        repro_dir = target_dir / "reproducibility"
                        if repro_dir.exists():
                            # Check CROSS_SECTIONAL view
                            cs_dir = repro_dir / View.CROSS_SECTIONAL.value
                            if cs_dir.exists():
                                # DETERMINISTIC: Use rglob_sorted to handle nested batch_/attempt_ structure deterministically
                                for cohort_dir in rglob_sorted(cs_dir, "cohort=*"):
                                    if cohort_dir.is_dir() and cohort_dir.name.startswith("cohort="):
                                        metadata_file = cohort_dir / "metadata.json"
                                        if metadata_file.exists():
                                            try:
                                                with open(metadata_file, 'r') as f:
                                                    metadata = json.load(f)
                                                n_effective = metadata.get('n_effective')
                                                if n_effective is not None and n_effective > 0:
                                                    self._n_effective = int(n_effective)
                                                    logger.info(f"Found n_effective from target-first structure: {self._n_effective} at {metadata_file.parent}")
                                                    break
                                            except Exception as e:
                                                logger.debug(f"Failed to read metadata from {metadata_file}: {e}")
                                    if self._n_effective is not None:
                                        break
                            if self._n_effective is not None:
                                break

                # Fallback: search legacy REPRODUCIBILITY structure
                if self._n_effective is None:
                    # DETERMINISM: Use rglob_sorted for deterministic iteration order
                    for metadata_file in rglob_sorted(self._initial_output_dir, "REPRODUCIBILITY/TARGET_RANKING/*/cohort=*/metadata.json"):
                        if metadata_file.exists():
                            try:
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                n_effective = metadata.get('n_effective')
                                if n_effective is not None and n_effective > 0:
                                    self._n_effective = int(n_effective)

                                    repo_root = Path(__file__).parent.parent.parent.parent
                                    results_dir = repo_root / "RESULTS"
                                    new_output_dir = results_dir / str(self._n_effective) / self._run_name

                                    if new_output_dir.exists():
                                        logger.warning(f"Sample size directory {new_output_dir} already exists, not moving")
                                        self.output_dir = new_output_dir
                                        self.cache_dir = self.output_dir / "cache"
                                        self.target_ranking_cache = self.cache_dir / "target_rankings.json"
                                        self.feature_selection_cache = self.cache_dir / "feature_selections"
                                        return

                                    import shutil
                                    new_output_dir.parent.mkdir(parents=True, exist_ok=True)
                                    bin_info = self._get_sample_size_bin(self._n_effective)
                                    bin_name = bin_info["bin_name"]
                                    if not hasattr(self, '_bin_info'):
                                        self._bin_info = bin_info
                                    logger.info(f"Moving run from {self._initial_output_dir} to {new_output_dir} (found via recursive search, N={self._n_effective}, bin={bin_name})")
                                    shutil.move(str(self._initial_output_dir), str(new_output_dir))
                                    self.output_dir = new_output_dir
                                    self.cache_dir = self.output_dir / "cache"
                                    self.target_ranking_cache = self.cache_dir / "target_rankings.json"
                                    self.feature_selection_cache = self.cache_dir / "feature_selections"
                                    logger.info(f"Organized run by sample size bin (N={self._n_effective}, bin={bin_name}): {self.output_dir}")
                                    return
                            except Exception as e:
                                logger.debug(f"Failed to read metadata from {metadata_file}: {e}")
                                continue
            except Exception as e2:
                logger.debug(f"Recursive search also failed: {e2}")

        # Create target-first structure
        from TRAINING.orchestration.utils.target_first_paths import initialize_run_structure
        initialize_run_structure(self.output_dir)

        # Create initial manifest with experiment config and run metadata
        from TRAINING.orchestration.utils.manifest import create_manifest
        try:
            experiment_config_dict = None
            if self.experiment_config:
                # Convert experiment config to dict for manifest
                try:
                    exp_name = self.experiment_config.name
                    exp_file = _get_experiment_config_path(exp_name)
                    if exp_file.exists():
                        experiment_config_dict = _load_experiment_config_safe(exp_name)
                except Exception as e:
                    logger.debug(f"Could not load experiment config for manifest: {e}")

            run_metadata_dict = {
                "data_dir": str(self.data_dir) if self.data_dir else None,
                "symbols": self.symbols if self.symbols else None,
                "output_dir": str(self.output_dir) if self.output_dir else None
            }

            # RI-001: Don't set run_id in initial manifest - will be updated after finalization
            # Let manifest handle derivation (defaults to None → unstable ID until run_identity is finalized)
            create_manifest(
                self.output_dir,
                run_id=None,  # Will be updated after feature selection finalizes run_identity
                experiment_config=experiment_config_dict,
                run_metadata=run_metadata_dict
            )
        except Exception as e:
            logger.warning(f"Failed to create initial manifest: {e}")

        # Keep legacy directories for backward compatibility during transition
        # NOTE: training_results/ directory removed - all models now go to targets/<target>/models/
        (self.output_dir / "leakage_diagnostics").mkdir(exist_ok=True)

        # Create new structure directories (keep for backward compatibility)
        (self.output_dir / "DECISION" / "TARGET_RANKING").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "DECISION" / "FEATURE_SELECTION").mkdir(parents=True, exist_ok=True)
        # Target-first structure only - no legacy REPRODUCIBILITY directory creation

        # Cache paths
        self.target_ranking_cache = self.cache_dir / "target_rankings.json"
        self.feature_selection_cache = self.cache_dir / "feature_selections"
        self.feature_selection_cache.mkdir(parents=True, exist_ok=True)

        # Initialize leakage sentinel if available
        if _SENTINELS_AVAILABLE:
            self.sentinel = LeakageSentinel()
        else:
            self.sentinel = None

    def rank_targets_auto(
        self,
        top_n: int = 5,
        model_families: Optional[List[str]] = None,
        multi_model_config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        use_cache: bool = True,
        max_targets_to_evaluate: Optional[int] = None,  # Limit number of targets to evaluate (for faster testing)
        targets_to_evaluate: Optional[List[str]] = None,  # NEW: Whitelist of specific targets to evaluate (works with auto_targets=true)
        target_ranking_config: Optional['TargetRankingConfig'] = None,  # New typed config (optional)
        min_cs: Optional[int] = None,  # Load from config if None
        max_cs_samples: Optional[int] = None,  # Load from config if None
        max_rows_per_symbol: Optional[int] = None,  # Load from config if None
        return_summary: bool = False  # NEW: If True, return (results, summary) tuple instead of just results
    ) -> Union[List[str], Tuple[List[str], Dict[str, Any]]]:
        """
        Automatically rank targets and return top N.

        Args:
            top_n: Number of top targets to return
            model_families: Optional list of model families to use [LEGACY]
            multi_model_config: Optional multi-model config [LEGACY]
            force_refresh: If True, ignore cache and re-rank
            use_cache: If True, use cached rankings if available
            max_targets_to_evaluate: Optional limit on number of targets to evaluate (for faster testing)
            targets_to_evaluate: Optional whitelist of specific targets to evaluate (works with auto_targets=true)
            target_ranking_config: Optional TargetRankingConfig object [NEW - preferred]
            min_cs: Minimum cross-sectional size per timestamp (loads from config if None)
            max_cs_samples: Maximum samples per timestamp for cross-sectional sampling (loads from config if None)
            max_rows_per_symbol: Maximum rows to load per symbol (loads from config if None)

        Returns:
            List of top N target names
        """
        logger.info(f"🎯 Ranking targets (top {top_n})...")
        if max_targets_to_evaluate is not None:
            logger.info(f"📊 max_targets_to_evaluate limit: {max_targets_to_evaluate}")

        # Generate cache key
        config_hash = hashlib.md5(
            json.dumps({
                'model_families': model_families or [],
                'symbols': sorted(self.symbols),
                'max_targets_to_evaluate': max_targets_to_evaluate,  # Include limit in cache key
                'targets_to_evaluate': sorted(targets_to_evaluate) if targets_to_evaluate else []  # NEW: Include whitelist in cache key
            }, sort_keys=True).encode()
        ).hexdigest()
        cache_key = self._get_cache_key(self.symbols, config_hash)

        # Check cache
        if not force_refresh and use_cache:
            cached = self._load_cached_rankings(cache_key, use_cache=True)
            if cached:
                # Respect max_targets_to_evaluate even when using cache
                if max_targets_to_evaluate is not None and max_targets_to_evaluate > 0:
                    if len(cached) > max_targets_to_evaluate:
                        logger.info(f"✅ Using cached rankings, truncating to {max_targets_to_evaluate} targets (cache had {len(cached)})")
                        cached = cached[:max_targets_to_evaluate]
                    else:
                        logger.info(f"✅ Using cached target rankings ({len(cached)} targets, limit={max_targets_to_evaluate})")
                else:
                    logger.info(f"✅ Using cached target rankings ({len(cached)} targets)")
                # Return top N from cache
                top_targets = [r['target'] for r in cached[:top_n]]
                return top_targets

        # Discover or load targets
        try:
            # Try to discover targets from data
            sample_symbol = self.symbols[0]
            targets_dict = discover_targets(sample_symbol, self.data_dir)
            logger.info(f"Discovered {len(targets_dict)} targets from data")
        except Exception as e:
            logger.warning(f"Target discovery failed: {e}, loading from config")
            # Fallback to config
            targets_config = load_target_configs()
            targets_dict = {
                name: config for name, config in sorted_items(targets_config)
                if config.get('enabled', False)
            }
            logger.info(f"Loaded {len(targets_dict)} enabled targets from config")

        if not targets_dict:
            logger.error("No targets found")
            return []

        # Filter out excluded target patterns (from experiment config)
        exclude_patterns = []
        if self.experiment_config:
            try:
                import yaml
                exp_name = self.experiment_config.name
                exp_file = _get_experiment_config_path(exp_name)
                if exp_file.exists():
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    intel_training = exp_yaml.get('intelligent_training', {})
                    if intel_training:
                        exclude_patterns = intel_training.get('exclude_target_patterns', [])
                        if exclude_patterns:
                            logger.info(f"📋 Loaded exclude_target_patterns from experiment config: {exclude_patterns}")
                else:
                    logger.debug(f"Experiment config file not found: {exp_file}")
            except Exception as e:
                logger.warning(f"Could not load exclude_target_patterns from experiment config: {e}")
        else:
            logger.debug("No experiment_config available, skipping exclude_target_patterns")

        if exclude_patterns:
            original_count = len(targets_dict)
            filtered_targets = {}
            for target, target_config in sorted_items(targets_dict):
                # Check if target matches any exclusion pattern
                excluded = False
                for pattern in exclude_patterns:
                    if pattern in target:
                        excluded = True
                        break
                if not excluded:
                    filtered_targets[target] = target_config
            targets_dict = filtered_targets
            excluded_count = original_count - len(targets_dict)
            if excluded_count > 0:
                logger.info(f"📋 Excluded {excluded_count} targets matching patterns: {exclude_patterns}")
                logger.info(f"📋 Remaining {len(targets_dict)} targets after exclusion")

        # NEW: Apply targets_to_evaluate whitelist if specified (works with auto_targets=true)
        if targets_to_evaluate:
            original_count = len(targets_dict)
            whitelisted_targets = {}
            targets_to_evaluate_set = set(targets_to_evaluate)
            for target, target_config in sorted_items(targets_dict):
                if target in targets_to_evaluate_set:
                    whitelisted_targets[target] = target_config
            targets_dict = whitelisted_targets
            filtered_count = original_count - len(targets_dict)
            if filtered_count > 0:
                logger.info(f"📋 Applied targets_to_evaluate whitelist: {len(targets_dict)} targets remain (filtered out {filtered_count})")
            if len(targets_dict) == 0:
                logger.warning(f"⚠️  targets_to_evaluate whitelist resulted in 0 targets. Check that whitelist targets exist in discovered targets.")

        # NEW: Build target ranking config from experiment config if available
        if target_ranking_config is None and self.experiment_config and _NEW_CONFIG_AVAILABLE:
            target_ranking_config = build_target_ranking_config(self.experiment_config)

        # Default model families if not provided
        if model_families is None:
            if target_ranking_config and _NEW_CONFIG_AVAILABLE:
                # Extract from typed config
                model_families = [
                    name for name, config in sorted_items(target_ranking_config.model_families)
                    if config.get('enabled', False)
                ]
            elif multi_model_config:
                model_families_dict = multi_model_config.get('model_families', {})
                model_families = [
                    name for name, config in sorted_items(model_families_dict)
                    if config and config.get('enabled', False)
                ]
            else:
                model_families = ['lightgbm', 'random_forest', 'neural_network']

        # Get explicit interval from experiment config if available
        explicit_interval = None
        experiment_config = None
        try:
            if hasattr(self, 'experiment_config') and self.experiment_config:
                experiment_config = self.experiment_config
                explicit_interval = getattr(self.experiment_config.data, 'bar_interval', None)
        except Exception as e:
            # Best-effort: experiment config access failed, continue without explicit interval
            logger.debug(f"Could not access experiment_config for explicit_interval: {e}")

        # SST: Record stage transition BEFORE any stage work
        try:
            from TRAINING.orchestration.utils.run_context import save_stage_transition
            save_stage_transition(self.output_dir, "TARGET_RANKING", reason="Starting target ranking phase")
        except Exception as e:
            logger.warning(f"Could not save TARGET_RANKING stage transition: {e}")

        # RI-002: Create partial RunIdentity for TARGET_RANKING and store on self
        # This will be finalized after feature selection completes (RI-003)
        try:
            from TRAINING.common.utils.fingerprinting import create_stage_identity
            self._partial_identity = create_stage_identity(
                stage=Stage.TARGET_RANKING,
                symbols=self.symbols,
                experiment_config=experiment_config,
                data_dir=self.data_dir,
            )
            logger.debug(f"Created partial identity, awaiting finalization: train_seed={self._partial_identity.train_seed}")
        except Exception as e:
            logger.warning(f"Failed to create TARGET_RANKING identity: {e}")
            self._partial_identity = None
        # Alias for backward compatibility with existing code that uses target_ranking_identity
        target_ranking_identity = self._partial_identity

        # Before rank_targets() call - build registry once
        from TRAINING.common.feature_registry import get_registry
        from TRAINING.orchestration.utils.target_first_paths import run_root

        # Get current_bar_minutes from explicit_interval or config
        # CRITICAL: Normalize to float|int|None before passing to registry (fail-closed for non-numeric)
        current_bar_minutes = None
        if explicit_interval:
            from TRAINING.ranking.utils.data_interval import normalize_interval
            current_bar_minutes = normalize_interval(explicit_interval)
        elif experiment_config:
            try:
                from CONFIG.config_loader import get_cfg
                raw_interval = get_cfg('data.bar_interval', default=None)
                # Normalize to numeric (fail-closed for non-numeric)
                if raw_interval is not None:
                    try:
                        current_bar_minutes = float(raw_interval)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid bar_interval config value: {raw_interval} (type: {type(raw_interval)}). Treating as None.")
                        current_bar_minutes = None
            except Exception as e:
                logger.debug(f"Config loading failed for bar_interval: {e}")

        # Build registry once (reuse same instance)
        # current_bar_minutes is now guaranteed to be float|int|None (not str/Decimal/etc)
        from TRAINING.common.determinism import is_strict_mode
        from TRAINING.common.exceptions import RegistryLoadError

        fail_closed_registry = is_strict_mode()

        try:
            registry = get_registry(current_bar_minutes=current_bar_minutes, strict=fail_closed_registry)
        except RegistryLoadError:
            # get_registry() itself failed - re-raise in strict mode, set None in best-effort
            if fail_closed_registry:
                raise
            logger.warning(
                "Registry load failed (best-effort mode). "
                "Continuing without registry. Coverage computation will be skipped."
            )
            registry = None
        except Exception as e:
            # Unexpected exception from get_registry() - wrap and re-raise in strict mode
            if fail_closed_registry:
                raise RegistryLoadError(
                    message=f"Unexpected error loading registry: {e}",
                    registry_path=None,
                    stage="TARGET_RANKING",
                    error_code="REGISTRY_LOAD_FAILED"
                ) from e
            logger.warning(
                f"Unexpected error loading registry (best-effort mode): {e}. "
                "Continuing without registry. Coverage computation will be skipped."
            )
            registry = None

        # Validate registry was loaded successfully (only if get_registry() succeeded)
        if registry is not None:
            if not hasattr(registry, 'config_path') or not registry.config_path:
                if fail_closed_registry:
                    raise RegistryLoadError(
                        message="Registry loaded but config_path is missing. Worker processes cannot reconstruct registry.",
                        registry_path=None,
                        stage="TARGET_RANKING",
                        error_code="REGISTRY_LOAD_FAILED"
                    )
                logger.warning(
                    "Registry loaded but config_path is missing. "
                    "Worker processes may fail to reconstruct registry. "
                    f"Registry type: {type(registry)}"
                )

        # SB-007: Capture registry hash for mutation detection between stages
        self._registry_hash_at_load = None
        if registry is not None and hasattr(registry, 'features'):
            try:
                self._registry_hash_at_load = hash(frozenset(registry.features.keys()))
                logger.debug(f"SB-007: Captured registry hash at load: {self._registry_hash_at_load}")
            except Exception as e:
                logger.debug(f"SB-007: Could not compute registry hash: {e}")

        # Collect coverage breakdowns
        coverage_breakdowns_by_target = {}

        # Check for flush_each_target config
        try:
            from CONFIG.config_loader import get_cfg
            flush_each_target = get_cfg('registry_autopatch.flush_each_target', default=False)
        except Exception as e:
            logger.debug(f"Config loading failed for flush_each_target: {e}")
            flush_each_target = False

        # Rank targets
        logger.info(f"Evaluating {len(targets_dict)} targets with {len(model_families)} model families...")
        rankings = rank_targets(
            targets=targets_dict,
            symbols=self.symbols,
            data_dir=self.data_dir,
            model_families=model_families,
            multi_model_config=multi_model_config,
            output_dir=self.output_dir,  # Pass base output_dir (not target_rankings subdir)
            top_n=None,  # Get all rankings for caching
            max_targets_to_evaluate=max_targets_to_evaluate,  # Limit evaluation if specified
            target_ranking_config=target_ranking_config,  # Pass typed config if available
            explicit_interval=explicit_interval,  # Pass explicit interval
            experiment_config=experiment_config,  # Pass experiment config
            min_cs=min_cs,  # Pass min_cs from config
            max_cs_samples=max_cs_samples,  # Pass max_cs_samples from config
            max_rows_per_symbol=max_rows_per_symbol,  # Pass max_rows_per_symbol from config
            run_identity=target_ranking_identity,  # SST: Pass identity for reproducibility tracking
            registry=registry,  # NEW: Pass same instance
            coverage_breakdowns_dict=coverage_breakdowns_by_target,  # NEW: Collect breakdowns
        )

        # After rankings complete, generate patches (final authoritative set)
        if coverage_breakdowns_by_target:
            try:
                from TRAINING.common.utils.registry_autopatch import aggregate_and_write_coverage_patches
                from TRAINING.orchestration.utils.manifest import derive_run_id_from_identity

                # Resolve run_root (SST helper)
                run_root_dir = run_root(self.output_dir)

                # Derive run_id from run_identity (SST - no ad-hoc string replacement)
                run_id = None
                if target_ranking_identity:
                    try:
                        run_id = derive_run_id_from_identity(
                            run_identity=target_ranking_identity
                        )
                    except Exception as e:
                        logger.debug(f"Failed to derive run_id from identity: {e}")
                        # Fallback to unstable run_id if identity derivation fails
                        from TRAINING.orchestration.utils.manifest import derive_unstable_run_id, generate_run_instance_id
                        run_id = derive_unstable_run_id(generate_run_instance_id())

                # Generate patches (final authoritative set)
                patch_files = aggregate_and_write_coverage_patches(
                    run_root=run_root_dir,
                    coverage_breakdowns_by_target=coverage_breakdowns_by_target,
                    registry=registry,  # Same instance used during evaluation
                    run_id=run_id
                )

                if patch_files:
                    logger.info(f"✅ Generated registry autopatch files: {list(patch_files.keys())}")
                    for key, path in sorted_items(patch_files):
                        if path:
                            logger.debug(f"  {key}: {path}")
                else:
                    logger.debug("No registry autopatch files generated (no suggestions or disabled)")
            except Exception as e:
                logger.warning(f"Failed to generate registry autopatch files: {e}")
                import traceback
                logger.debug(f"Registry autopatch traceback: {traceback.format_exc()}")

        # Build ranking summary if requested (for diagnostic purposes)
        ranking_summary = None
        if return_summary and rankings:
            valid_results = [r for r in rankings if getattr(r, 'valid_for_ranking', True)]
            invalid_results = [r for r in rankings if not getattr(r, 'valid_for_ranking', True)]

            # Group invalid reasons
            invalid_reasons_by_type = {}
            for r in invalid_results:
                reasons = getattr(r, 'invalid_reasons', [])
                for reason in reasons:
                    if reason not in invalid_reasons_by_type:
                        invalid_reasons_by_type[reason] = []
                    invalid_reasons_by_type[reason].append(r.target)

            ranking_summary = {
                'total_evaluated': len(rankings),
                'valid_for_ranking': len(valid_results),
                'invalid_for_ranking': len(invalid_results),
                'invalid_reasons_by_type': invalid_reasons_by_type,
                'top_valid_targets': [r.target for r in valid_results[:top_n]] if valid_results else []
            }

        # After target ranking completes, organize by sample size (n_effective)
        # This moves the entire directory (including all REPRODUCIBILITY data) to RESULTS/{n_effective}/{run_name}/
        if self._n_effective is None and rankings:
            logger.info("🔍 Attempting to organize run by sample size (n_effective)...")
            logger.info(f"   Current output_dir: {self.output_dir}")
            logger.info(f"   Initial output_dir: {self._initial_output_dir}")
            self._organize_by_cohort()
            if self._n_effective is not None:
                bin_info = self._get_sample_size_bin(self._n_effective)
                bin_name = bin_info["bin_name"]
                if not hasattr(self, '_bin_info'):
                    self._bin_info = bin_info
                logger.info(f"✅ Successfully organized run by sample size bin (N={self._n_effective}, bin={bin_name}): {self.output_dir}")
                logger.info(f"   Moved from: {self._initial_output_dir}")
                logger.info(f"   Moved to: {self.output_dir}")
            else:
                logger.warning("⚠️  Could not determine n_effective, run will stay in _pending/")
                logger.warning(f"   Run directory: {self._initial_output_dir}")
                # Try to help debug - check if REPRODUCIBILITY exists (check both new and old structures)
                repro_check_new = self._initial_output_dir / "REPRODUCIBILITY"
                repro_check_old = self._initial_output_dir / "target_rankings" / "REPRODUCIBILITY"
                if repro_check_new.exists():
                    logger.warning(f"   REPRODUCIBILITY found at: {repro_check_new}")
                elif repro_check_old.exists():
                    logger.warning(f"   REPRODUCIBILITY found at (old structure): {repro_check_old}")
                else:
                    logger.warning(f"   REPRODUCIBILITY not found at: {repro_check_new} or {repro_check_old}")

        # Generate metrics rollups after target ranking completes
        try:
            from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
            tracker = ReproducibilityTracker(output_dir=self.output_dir / "target_rankings")
            # Defensive: Ensure _run_name is a string before calling .replace()
            run_id = self._run_name.replace("_", "-") if self._run_name and isinstance(self._run_name, str) else None
            if not run_id:
                from datetime import datetime
                run_id = datetime.now().isoformat()
            tracker.generate_metrics_rollups(stage=Stage.TARGET_RANKING, run_id=run_id)
            logger.debug("✅ Generated metrics rollups for TARGET_RANKING")
        except Exception as e:
            logger.debug(f"Failed to generate metrics rollups: {e}")

        # Save to cache
        if use_cache:
            cache_data = [r.to_dict() for r in rankings]
            self._save_cached_rankings(cache_key, cache_data)

        # === DUAL RANKING: Filter by strict_viability_flag before promotion ===
        # Only promote targets where strict evaluation clears threshold (prevents false positives)
        viable_rankings = []
        for r in rankings:
            # Check strict_viability_flag if available
            if hasattr(r, 'strict_viability_flag') and r.strict_viability_flag is not None:
                if r.strict_viability_flag:
                    viable_rankings.append(r)
                else:
                    # Log warning for targets that failed strict evaluation
                    logger.warning(
                        f"⚠️  Target {r.target} failed strict viability check "
                        f"(screen_score={r.score_screen:.4f}, strict_score={r.score_strict:.4f if r.score_strict else 'N/A'})"
                    )
            else:
                # If strict evaluation not available, include target (backward compatibility)
                viable_rankings.append(r)

        # Log mismatch telemetry warnings
        for r in viable_rankings:
            if hasattr(r, 'mismatch_telemetry') and r.mismatch_telemetry:
                telemetry = r.mismatch_telemetry
                unknown_count = telemetry.get('unknown_feature_count', 0)
                rank_delta_val = r.rank_delta if hasattr(r, 'rank_delta') and r.rank_delta is not None else None

                if unknown_count > 0:
                    logger.warning(
                        f"⚠️  Target {r.target} has {unknown_count} unknown features in screen evaluation "
                        f"(registry_coverage={telemetry.get('registry_coverage_rate', 0.0):.2%})"
                    )
                if rank_delta_val is not None and rank_delta_val > 5:
                    logger.warning(
                        f"⚠️  Target {r.target} has large rank_delta={rank_delta_val} "
                        f"(screen ranks {rank_delta_val} positions higher than strict)"
                    )

        # Return top N from viable rankings
        top_targets = [r.target for r in viable_rankings[:top_n]]
        if len(viable_rankings) < len(rankings):
            logger.info(
                f"📊 Filtered {len(rankings)} rankings → {len(viable_rankings)} viable "
                f"(strict_viability_filter applied)"
            )
        logger.info(f"✅ Top {len(top_targets)} targets: {', '.join(top_targets)}")

        if return_summary:
            return top_targets, ranking_summary
        else:
            return top_targets

    def select_features_auto(
        self,
        target: str,
        top_m: int = 100,
        model_families_config: Optional[Dict[str, Any]] = None,
        multi_model_config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        use_cache: bool = True,
        feature_selection_config: Optional['FeatureSelectionConfig'] = None,  # New typed config (optional)
        view: Union[str, View] = View.CROSS_SECTIONAL,  # Must match target ranking view
        symbol: Optional[str] = None  # Required for SYMBOL_SPECIFIC view
    ) -> List[str]:
        """
        Automatically select top M features for a target.

        Args:
            target: Target column name
            top_m: Number of top features to return
            model_families_config: Optional model families config [LEGACY]
            multi_model_config: Optional multi-model config [LEGACY]
            force_refresh: If True, ignore cache and re-select
            use_cache: If True, use cached features if available
            feature_selection_config: Optional FeatureSelectionConfig object [NEW - preferred]

        Returns:
            List of top M feature names
        """
        logger.info(f"🔍 Selecting features for {target} (top {top_m})...")

        # Check cache
        if not force_refresh and use_cache:
            cached = self._load_cached_features(target)
            if cached:
                logger.info(f"✅ Using cached features for {target} ({len(cached)} features)")
                return cached[:top_m]

        # NEW: Build feature selection config from experiment config if available
        if feature_selection_config is None and self.experiment_config and _NEW_CONFIG_AVAILABLE:
            # Create a temporary experiment config with this target
            # Copy data config from original (includes bar_interval)
            from copy import deepcopy
            temp_data = deepcopy(self.experiment_config.data) if hasattr(self.experiment_config, 'data') and self.experiment_config.data else None

            # CRITICAL: If fs_model_families is set, use it for feature selection
            # This ensures feature_selection.model_families is used, not training.model_families
            feature_selection_overrides = {'top_n': top_m}
            if self.fs_model_families is not None:
                feature_selection_overrides['model_families'] = self.fs_model_families

            temp_exp = ExperimentConfig(
                name=self.experiment_config.name,
                data_dir=self.experiment_config.data_dir,
                symbols=self.experiment_config.symbols,
                target=target,
                data=temp_data,
                max_samples_per_symbol=self.experiment_config.max_samples_per_symbol,
                feature_selection_overrides=feature_selection_overrides
            )
            feature_selection_config = build_feature_selection_config(temp_exp)

        # LEGACY: Load config if not provided
        if multi_model_config is None and feature_selection_config is None:
            multi_model_config = load_multi_model_config()

        # Select features - write to target-first structure (targets/<target>/reproducibility/)
        # The feature selection code will handle the path construction based on view
        feature_output_dir = self.output_dir

        # Extract explicit_interval from experiment_config for feature selection
        explicit_interval = None
        if self.experiment_config is not None:
            # Try to get bar_interval from config
            if hasattr(self.experiment_config, 'data') and hasattr(self.experiment_config.data, 'bar_interval'):
                explicit_interval = self.experiment_config.data.bar_interval
            # Also check direct bar_interval property (convenience)
            elif hasattr(self.experiment_config, 'bar_interval'):
                explicit_interval = self.experiment_config.bar_interval
            # Legacy: check interval field
            elif hasattr(self.experiment_config, 'interval'):
                explicit_interval = self.experiment_config.interval

        # Filter symbols based on view
        symbols_to_use = self.symbols
        # Normalize view to enum for comparison
        view_enum = View.from_string(view) if isinstance(view, str) else view
        if view_enum == View.SYMBOL_SPECIFIC and symbol:
            symbols_to_use = [symbol]
        elif view == "LOSO" and symbol:
            # LOSO: train on all symbols except symbol
            symbols_to_use = [s for s in self.symbols if s != symbol]

        # Compute universe_sig for reproducibility tracking
        universe_sig = None
        try:
            from TRAINING.orchestration.utils.run_context import compute_universe_signature
            universe_sig = compute_universe_signature(symbols_to_use)
        except Exception as e:
            logger.debug(f"Failed to compute universe_sig: {e}")

        # Create partial RunIdentity using SST factory, then add target-specific fields
        partial_identity = None
        try:
            from TRAINING.common.utils.fingerprinting import (
                create_stage_identity,
                RunIdentity,
                compute_target_fingerprint,
                compute_routing_fingerprint,
            )
            from TRAINING.common.utils.config_hashing import sha256_full

            # Use SST factory for base identity (handles seed fallback chain)
            base_identity = create_stage_identity(
                stage=Stage.FEATURE_SELECTION,
                symbols=symbols_to_use,
                experiment_config=self.experiment_config,
                data_dir=self.data_dir,
            )

            # Target signature from target config
            target_signature = compute_target_fingerprint(target=target)
            # Convert 16-char to 64-char if needed
            if target_signature and len(target_signature) == 16:
                target_signature = sha256_full(target_signature)

            # Routing signature from view/symbol
            routing_signature, routing_payload = compute_routing_fingerprint(
                view=view,
                symbol=symbol,
            )

            # Resolve registry overlay signature (for reproducibility - must match what feature selection uses)
            registry_overlay_signature = None
            if feature_output_dir:
                try:
                    from TRAINING.orchestration.utils.target_first_paths import run_root
                    from TRAINING.ranking.utils.registry_overlay_resolver import resolve_registry_overlay_dir_for_feature_selection
                    from TRAINING.common.utils.fingerprinting import compute_registry_signature

                    # Convert explicit_interval to minutes if needed (for signature computation)
                    current_bar_minutes = None
                    if explicit_interval:
                        if isinstance(explicit_interval, str):
                            # Parse "5m" -> 5
                            if explicit_interval.endswith('m'):
                                try:
                                    current_bar_minutes = float(explicit_interval[:-1])
                                except ValueError:
                                    pass
                        elif isinstance(explicit_interval, (int, float)):
                            current_bar_minutes = float(explicit_interval)

                    run_output_root = run_root(feature_output_dir)
                    overlay_resolution = resolve_registry_overlay_dir_for_feature_selection(
                        run_output_root=run_output_root,
                        experiment_config=self.experiment_config,
                        target_column=target,
                        current_bar_minutes=current_bar_minutes
                    )
                    # Compute signature if overlay was found
                    if overlay_resolution.overlay_dir:
                        registry_overlay_signature = compute_registry_signature(
                            registry_overlay_dir=overlay_resolution.overlay_dir,
                            persistent_override_dir=None,  # Not used in feature selection
                            persistent_unblock_dir=None,  # Not used in feature selection
                            target_column=target,
                            current_bar_minutes=current_bar_minutes
                        )
                        if registry_overlay_signature:
                            logger.debug(f"Resolved registry overlay signature for {target}: {registry_overlay_signature[:16]}...")
                except Exception as e:
                    logger.debug(f"Could not resolve registry overlay signature for {target}: {e}")

            # FP-005: Extend base identity with target-specific fields
            # Use None not empty strings for missing signatures
            # (split and hparams computed in selector where folds/models are known)
            partial_identity = RunIdentity(
                dataset_signature=base_identity.dataset_signature,
                split_signature=None,  # Computed in selector after folds created
                target_signature=target_signature,  # FP-005: None not empty string
                feature_signature=None,  # Computed after feature selection
                hparams_signature=None,  # Computed per model family in selector
                routing_signature=routing_signature,  # FP-005: None not empty string
                routing_payload=routing_payload,
                registry_overlay_signature=registry_overlay_signature,  # Set from overlay resolution
                train_seed=base_identity.train_seed,
                is_final=False,
            )
            logger.debug(f"Created partial RunIdentity for {target} using SST factory")
        except Exception as e:
            logger.warning(f"Failed to create partial RunIdentity: {e}")
            partial_identity = None

        # OPTIMIZATION: Get candidate features via fast preflight (schema-only, ~1ms per file)
        # This enables column projection in feature selection, reducing memory by 5-10x
        candidate_features = None
        try:
            from TRAINING.ranking.utils.preflight_leakage import preflight_filter_features

            # Get interval_minutes for preflight
            preflight_interval_minutes = 5  # Default
            if explicit_interval:
                if isinstance(explicit_interval, str) and explicit_interval.endswith('m'):
                    try:
                        preflight_interval_minutes = int(explicit_interval[:-1])
                    except ValueError:
                        pass
                elif isinstance(explicit_interval, (int, float)):
                    preflight_interval_minutes = int(explicit_interval)

            # Run preflight filter (fast schema-only check)
            # DETERMINISM: Sort symbols before sampling to ensure consistent preflight results
            preflight_result = preflight_filter_features(
                data_dir=self.data_dir,
                symbols=sorted(symbols_to_use)[:20],  # Sample for schema (fast, sorted for determinism)
                targets=[target],
                interval_minutes=preflight_interval_minutes,
                for_ranking=True,  # Use permissive mode (feature selection filters further)
                verbose=False,
            )
            candidate_features = preflight_result.get(target, None)
            if candidate_features:
                logger.info(f"📋 Preflight: {len(candidate_features)} candidate features for column projection")
        except Exception as e:
            logger.debug(f"Preflight failed, will load all columns: {e}")
            candidate_features = None

        selected_features, _ = select_features_for_target(
            target_column=target,
            symbols=symbols_to_use,
            data_dir=self.data_dir,
            model_families_config=model_families_config,
            multi_model_config=multi_model_config,
            top_n=top_m,
            output_dir=feature_output_dir,
            feature_selection_config=feature_selection_config,  # Pass typed config if available
            explicit_interval=explicit_interval,  # Pass explicit interval to avoid auto-detection warnings
            experiment_config=self.experiment_config,  # Pass experiment config for data.bar_interval
            view=view,  # Pass view to ensure consistency
            symbol=symbol,  # Pass symbol for SYMBOL_SPECIFIC view
            universe_sig=universe_sig,  # Pass universe_sig for reproducibility tracking
            run_identity=partial_identity,  # Pass partial identity for reproducibility
            candidate_features=candidate_features,  # OPTIMIZATION: Pass preflight features for column projection
        )

        # Load confidence and apply routing
        try:
            from TRAINING.orchestration.target_routing import (
                load_target_confidence,
                classify_target_from_confidence,
                save_target_routing_metadata
            )

            # Get routing config from multi_model config
            routing_config = None
            if multi_model_config:
                routing_config = multi_model_config.get('confidence', {}).get('routing', {})
            elif feature_selection_config and hasattr(feature_selection_config, 'config'):
                # Try to extract from typed config
                routing_config = feature_selection_config.config.get('confidence', {}).get('routing', {})

            # Get target-specific directory for routing (functions will walk up to find base if needed)
            from TRAINING.orchestration.utils.target_first_paths import get_target_reproducibility_dir
            target_repro_dir = get_target_reproducibility_dir(self.output_dir, target)

            # Pass view to load_target_confidence so it checks the correct view-scoped location
            conf = load_target_confidence(target_repro_dir, target, view=view)
            if conf:
                routing = classify_target_from_confidence(conf, routing_config=routing_config)
                save_target_routing_metadata(self.output_dir, target, conf, routing, view=view)

                # Log routing decision
                logger.info(
                    f"🎯 Target {target}: confidence={conf['confidence']} "
                    f"(score_tier={conf.get('score_tier', 'LOW')}, "
                    f"reason={conf.get('low_confidence_reason', 'N/A')}) "
                    f"→ bucket={routing['bucket']}, "
                    f"allowed_in_production={routing['allowed_in_production']}"
                )
        except Exception as e:
            logger.debug(f"Failed to load/route confidence for {target}: {e}")

        # Save to cache
        if selected_features:
            self._save_cached_features(target, selected_features)

        logger.info(f"✅ Selected {len(selected_features)} features for {target}")

        return selected_features

    def _aggregate_feature_selection_summaries(self) -> None:
        """
        Aggregate feature selection summaries from all targets into globals/ directory.

        Collects per-target summaries from targets/*/reproducibility/ and creates
        aggregated summaries in globals/:
        - globals/feature_selection_summary.json (metadata only)
        - globals/selected_features_summary.json (actual feature lists per target per view)
        - globals/model_family_status_summary.json

        Feature lists are read from targets/{target}/reproducibility/selected_features.txt
        and organized by target and view (CROSS_SECTIONAL or SYMBOL_SPECIFIC).
        """
        from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
        from TRAINING.ranking.target_routing import load_routing_decisions

        globals_dir = get_globals_dir(self.output_dir)
        globals_dir.mkdir(parents=True, exist_ok=True)

        targets_dir = self.output_dir / "targets"
        if not targets_dir.exists():
            logger.debug("No targets directory found, skipping aggregation")
            return

        # Load routing decisions to determine view per target
        routing_decisions = {}
        try:
            routing_decisions = load_routing_decisions(output_dir=self.output_dir)
        except Exception as e:
            logger.debug(f"Could not load routing decisions for view determination: {e}")

        # Collect all per-target summaries
        all_summaries = {}
        all_family_statuses = {}
        feature_selections = {}  # New: actual feature lists per target per view

        # DETERMINISM: Use iterdir_sorted for deterministic iteration order
        for target_dir in iterdir_sorted(targets_dir):
            if not target_dir.is_dir():
                continue

            target = target_dir.name
            repro_dir = target_dir / "reproducibility"

            if not repro_dir.exists():
                continue

            # Determine view from routing decisions or check both views
            route_info = routing_decisions.get(target, {})
            route = route_info.get('route', View.CROSS_SECTIONAL.value)

            # Check both CROSS_SECTIONAL and SYMBOL_SPECIFIC views (files are now view-scoped)
            views_to_check = []
            if route in [View.CROSS_SECTIONAL.value, 'BOTH']:
                views_to_check.append(View.CROSS_SECTIONAL.value)
            if route in [View.SYMBOL_SPECIFIC.value, 'BOTH']:
                # For SYMBOL_SPECIFIC, check all symbol subdirectories
                sym_specific_dir = repro_dir / View.SYMBOL_SPECIFIC.value
                if sym_specific_dir.exists():
                    # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                    for sym_dir in iterdir_sorted(sym_specific_dir):
                        if sym_dir.is_dir() and sym_dir.name.startswith("symbol="):
                            views_to_check.append((View.SYMBOL_SPECIFIC.value, sym_dir.name.replace("symbol=", "")))

            # If no routing info, check both views
            if not views_to_check:
                views_to_check = [View.CROSS_SECTIONAL.value]
                sym_specific_dir = repro_dir / View.SYMBOL_SPECIFIC.value
                if sym_specific_dir.exists():
                    # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                    for sym_dir in iterdir_sorted(sym_specific_dir):
                        if sym_dir.is_dir() and sym_dir.name.startswith("symbol="):
                            views_to_check.append((View.SYMBOL_SPECIFIC.value, sym_dir.name.replace("symbol=", "")))

            # Load files from view-scoped locations
            for view_info in views_to_check:
                if isinstance(view_info, tuple):
                    view, symbol = view_info
                    view_dir = repro_dir / View.SYMBOL_SPECIFIC.value / f"symbol={symbol}"
                else:
                    view = view_info
                    symbol = None
                    view_dir = repro_dir / View.CROSS_SECTIONAL.value

                if not view_dir.exists():
                    continue

                # Load feature_selection_summary.json (view-scoped)
                summary_path = view_dir / "feature_selection_summary.json"
                if summary_path.exists() and target not in all_summaries:
                    try:
                        with open(summary_path) as f:
                            summary = json.load(f)
                            all_summaries[target] = summary
                    except Exception as e:
                        logger.debug(f"Failed to load summary for {target} ({view}): {e}")

                # Load model_family_status.json (view-scoped)
                status_path = view_dir / "model_family_status.json"
                if status_path.exists() and target not in all_family_statuses:
                    try:
                        with open(status_path) as f:
                            status_data = json.load(f)
                            all_family_statuses[target] = status_data
                    except Exception as e:
                        logger.debug(f"Failed to load family status for {target} ({view}): {e}")

                # Load actual feature list from selected_features.txt (view-scoped)
                selected_features_path = view_dir / "selected_features.txt"
                if selected_features_path.exists():
                    try:
                        with open(selected_features_path, 'r') as f:
                            features = [line.strip() for line in f if line.strip()]

                        if features:
                            # Create key: target:view[:symbol]
                            if symbol:
                                key = f"{target}:{view}:{symbol}"
                                view_display = f"{view}:{symbol}"
                            else:
                                key = f"{target}:{view}"
                                view_display = view

                            # SST: Construct relative path strings for metadata (consistent with target_routing.py pattern)
                            from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
                            target_clean = normalize_target_name(target)
                            stage_prefix = "stage=FEATURE_SELECTION"
                            base_path = f"targets/{target_clean}/reproducibility/{stage_prefix}/{view}"
                            symbol_suffix = f"symbol={symbol}/" if symbol else ""

                            feature_selections[key] = {
                                'target': target,
                                'view': view,
                                'symbol': symbol,
                                'n_features': len(features),
                                'features': features,
                                'selected_features_path': f"{base_path}/{symbol_suffix}selected_features.txt",
                                'feature_selection_summary_path': f"{base_path}/{symbol_suffix}feature_selection_summary.json"
                            }
                            logger.debug(f"Loaded {len(features)} features for {key}")
                    except Exception as e:
                        logger.debug(f"Failed to load selected features for {target} ({view_display}): {e}")

        # Write aggregated feature_selection_summary.json
        if all_summaries:
            aggregated_summary = {
                'total_targets': len(all_summaries),
                'targets': all_summaries,
                'summary': {
                    'total_features_selected': sum(
                        s.get('top_n', 0) if isinstance(s, dict) else 0
                        for s in all_summaries.values()
                    ),
                    'avg_features_per_target': sum(
                        s.get('top_n', 0) if isinstance(s, dict) else 0
                        for s in all_summaries.values()
                    ) / len(all_summaries) if all_summaries else 0
                }
            }

            from TRAINING.orchestration.utils.target_first_paths import run_root, globals_dir as globals_dir_func
            run_root_dir = run_root(self.output_dir)
            summaries_dir = globals_dir_func(run_root_dir, "summaries")
            summaries_dir.mkdir(parents=True, exist_ok=True)
            summary_path = summaries_dir / "feature_selection_summary.json"
            # SST: Use write_atomic_json for atomic write with canonical serialization
            from TRAINING.common.utils.file_utils import write_atomic_json
            write_atomic_json(summary_path, aggregated_summary)
            logger.info(f"✅ Aggregated feature selection summaries to {summary_path}")

        # Write aggregated model_family_status_summary.json
        if all_family_statuses:
            # Aggregate family statuses across all targets
            aggregated_family_status = {
                'total_targets': len(all_family_statuses),
                'targets': all_family_statuses,
                'summary': {}
            }

            # Aggregate summary statistics across all targets
            all_families = set()
            for status_data in all_family_statuses.values():
                if isinstance(status_data, dict) and 'summary' in status_data:
                    all_families.update(status_data['summary'].keys())

            for family in all_families:
                total_success = 0
                total_failed = 0
                total_fallback = 0
                for status_data in all_family_statuses.values():
                    if isinstance(status_data, dict) and 'summary' in status_data:
                        family_summary = status_data['summary'].get(family, {})
                        total_success += family_summary.get('success', 0)
                        total_failed += family_summary.get('failed', 0)
                        total_fallback += family_summary.get('no_signal_fallback', 0)

                aggregated_family_status['summary'][family] = {
                    'total_success': total_success,
                    'total_failed': total_failed,
                    'total_fallback': total_fallback
                }

            from TRAINING.orchestration.utils.target_first_paths import run_root, globals_dir as globals_dir_func
            run_root_dir = run_root(self.output_dir)
            summaries_dir = globals_dir_func(run_root_dir, "summaries")
            summaries_dir.mkdir(parents=True, exist_ok=True)
            status_path = summaries_dir / "model_family_status_summary.json"
            # SST: Use write_atomic_json for atomic write with canonical serialization
            from TRAINING.common.utils.file_utils import write_atomic_json
            write_atomic_json(status_path, aggregated_family_status)
            logger.info(f"✅ Aggregated model family status summaries to {status_path}")

        # NEW: Write selected_features_summary.json with actual feature lists
        if feature_selections:
            # Calculate summary statistics
            by_view = {}
            total_features = 0
            # DC-003: Use sorted_items for deterministic iteration
            for key, selection in sorted_items(feature_selections):
                view = selection['view']
                by_view[view] = by_view.get(view, 0) + 1
                total_features += selection['n_features']

            selected_features_summary = {
                'feature_selections': feature_selections,
                'summary': {
                    'total_targets': len(feature_selections),
                    'by_view': by_view,
                    'total_features': total_features
                }
            }

            summaries_dir = globals_dir_func(run_root_dir, "summaries")
            summaries_dir.mkdir(parents=True, exist_ok=True)
            summary_path = summaries_dir / "selected_features_summary.json"
            # SST: Use write_atomic_json for atomic write with canonical serialization
            from TRAINING.common.utils.file_utils import write_atomic_json
            write_atomic_json(summary_path, selected_features_summary)
            logger.info(f"✅ Aggregated selected features summary to {summary_path} ({len(feature_selections)} target/view combinations)")
