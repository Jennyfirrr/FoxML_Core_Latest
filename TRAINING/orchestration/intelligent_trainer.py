# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Intelligent Training Orchestrator

Integrates target ranking and feature selection into the training pipeline
while preserving all existing functionality and leakage-free behavior.

This is a new entry point that wraps train_with_strategies.py, adding:
- Automatic target ranking and selection
- Automatic feature selection per target
- Caching of ranking/selection results
- Backward compatibility with existing workflows
"""

# ============================================================================
# CRITICAL: Path setup MUST happen before any TRAINING imports
# ============================================================================
import sys
from pathlib import Path

# Add project root to path FIRST (before any TRAINING imports)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Add TRAINING to path
_TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

# ============================================================================
# CRITICAL: Import repro_bootstrap FIRST before ANY numeric libraries
# This sets thread env vars BEFORE numpy/torch/sklearn are imported.
# DO NOT move this import or add imports above it (except path setup)!
# ============================================================================
import TRAINING.common.repro_bootstrap  # noqa: F401 - side effects only

import os
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import hashlib
import time
import datetime
import numpy as np

# CRITICAL: Set global determinism BEFORE importing any ML libraries
# This ensures reproducible results across runs
from TRAINING.common.determinism import init_determinism_from_config

# Set determinism immediately (reads from config, respects REPRO_MODE env var)
init_determinism_from_config()

# Import input mode handling for raw OHLCV sequence mode
from TRAINING.common.input_mode import (
    InputMode,
    get_input_mode,
    is_raw_sequence_mode,
    filter_families_for_input_mode,
    get_raw_sequence_config,
)

# Import config loader for CS ranking mode check
from CONFIG.config_loader import get_cfg


def is_cs_ranking_enabled(experiment_config: Optional[Dict] = None) -> bool:
    """
    Check if cross-sectional ranking mode is enabled.

    Cross-sectional ranking mode changes training to optimize ranking quality
    within each timestamp rather than pointwise prediction accuracy.

    When enabled:
    - Target ranking stage is SKIPPED (ranking defines its own objective)
    - Feature selection is SKIPPED (uses raw OHLCV sequences)
    - Model training uses ranking loss (pairwise/listwise)
    - Metrics are ranking-aligned (Spearman IC, top-bottom spread)

    Args:
        experiment_config: Optional experiment config dict (for overrides)

    Returns:
        True if CS ranking mode is enabled
    """
    # Check experiment config first (highest priority)
    if experiment_config:
        cs_cfg = experiment_config.get("pipeline", {}).get("cross_sectional_ranking", {})
        if cs_cfg.get("enabled"):
            return True

    # Fall back to pipeline config
    return bool(get_cfg("pipeline.cross_sectional_ranking.enabled", default=False))


def get_cs_ranking_config(experiment_config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get cross-sectional ranking configuration.

    Args:
        experiment_config: Optional experiment config dict (for overrides)

    Returns:
        CS ranking config dict with target, loss, batching, and metrics settings
    """
    # Start with pipeline config defaults
    config = {
        "target": get_cfg("pipeline.cross_sectional_ranking.target", default={}),
        "loss": get_cfg("pipeline.cross_sectional_ranking.loss", default={}),
        "batching": get_cfg("pipeline.cross_sectional_ranking.batching", default={}),
        "metrics": get_cfg("pipeline.cross_sectional_ranking.metrics", default={}),
    }

    # Override with experiment config if available
    if experiment_config:
        exp_cs_cfg = experiment_config.get("pipeline", {}).get("cross_sectional_ranking", {})
        for key in ["target", "loss", "batching", "metrics"]:
            if key in exp_cs_cfg:
                config[key].update(exp_cs_cfg[key])

    return config

# Import training event emitter for dashboard progress monitoring
from TRAINING.orchestration.utils.training_events import (
    init_training_events,
    close_training_events,
    emit_progress,
    emit_stage_change,
    emit_target_start,
    emit_target_complete,
    emit_run_complete,
    emit_error,
)

# Import ranking/selection modules
from TRAINING.ranking import (
    rank_targets,
    discover_targets,
    load_target_configs,
    select_features_for_target,
    load_multi_model_config
)

# Import new config system (optional - for backward compatibility)
try:
    from CONFIG.config_builder import (
        build_feature_selection_config,
        build_target_ranking_config,
        build_training_config
    )
    from CONFIG.config_schemas import ExperimentConfig, FeatureSelectionConfig, TargetRankingConfig, TrainingConfig
    _NEW_CONFIG_AVAILABLE = True
except ImportError:
    _NEW_CONFIG_AVAILABLE = False
    # Logger not yet initialized, will be set up below
    pass

# Import config loader for centralized path resolution
try:
    from CONFIG.config_loader import (
        get_experiment_config_path,
        load_experiment_config,
        load_training_config,
        CONFIG_DIR
    )
    _CONFIG_LOADER_AVAILABLE = True
except ImportError:
    _CONFIG_LOADER_AVAILABLE = False
    # Logger not yet initialized, will be set up below
    pass

# Import existing training pipeline functions
# We call functions directly, not the main() entry point
# NOTE: load_mtf_data removed - Stage 3 now always uses lazy loading with column projection
from TRAINING.train_with_strategies import (
    train_models_for_interval_comprehensive,
    ALL_FAMILIES
)

# Import UnifiedDataLoader for lazy loading (Phase 4 memory optimization)
from TRAINING.data.loading.unified_loader import UnifiedDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Helper functions to get/load experiment config (with fallback)
# NOTE: Implementation moved to intelligent_trainer/config.py
# These are thin wrappers for backward compatibility
def _get_experiment_config_path(exp_name: str) -> Path:
    """Get experiment config path. Implementation in intelligent_trainer/config.py."""
    return _get_experiment_config_path_submodule(exp_name)

def _load_experiment_config_safe(exp_name: str) -> Dict[str, Any]:
    """Load experiment config safely. Implementation in intelligent_trainer/config.py."""
    return _load_experiment_config_safe_submodule(exp_name)

# Import leakage sentinels
try:
    from TRAINING.common.leakage_sentinels import LeakageSentinel, SentinelResult
    _SENTINELS_AVAILABLE = True
except ImportError:
    _SENTINELS_AVAILABLE = False
    logger.debug("Leakage sentinels not available")

# Import pandas for data handling (used throughout the module)
import pandas as pd

# Import from modular components
from TRAINING.orchestration.intelligent_trainer.utils import json_default as _json_default
from TRAINING.orchestration.intelligent_trainer.cli import create_argument_parser
from TRAINING.orchestration.intelligent_trainer.config import (
    get_experiment_config_path as _get_experiment_config_path_submodule,
    load_experiment_config_safe as _load_experiment_config_safe_submodule,
)
from TRAINING.orchestration.intelligent_trainer.caching import (
    get_cache_key as _get_cache_key_impl,
    load_cached_rankings as _load_cached_rankings_impl,
    save_cached_rankings as _save_cached_rankings_impl,
    get_feature_cache_path as _get_feature_cache_path_impl,
    load_cached_features as _load_cached_features_impl,
    save_cached_features as _save_cached_features_impl,
)

# SST: Import View enum for consistent view handling
from TRAINING.orchestration.utils.scope_resolution import View, Stage
# DETERMINISM: Import deterministic iteration helpers
from TRAINING.common.utils.determinism_ordering import iterdir_sorted, rglob_sorted, sorted_items

# Import pipeline stage mixin (extracted for maintainability)
from TRAINING.orchestration.intelligent_trainer.pipeline_stages import PipelineStageMixin


class IntelligentTrainer(PipelineStageMixin):
    """
    Intelligent training orchestrator with integrated ranking and selection.
    """
    
    def __init__(
        self,
        data_dir: Path,
        symbols: List[str],
        output_dir: Path,
        cache_dir: Optional[Path] = None,
        add_timestamp: bool = True,
        experiment_config: Optional['ExperimentConfig'] = None,  # New typed config (optional)
        max_rows_per_symbol: Optional[int] = None,  # For output directory binning
        max_cs_samples: Optional[int] = None,  # For output directory binning
        fs_model_families: Optional[List[str]] = None,  # Feature selection model families (separate from training)
        user_specified_output_dir: bool = False  # If True, respect user's --output-dir path
    ):
        """
        Initialize the intelligent trainer.
        
        Args:
            data_dir: Directory containing symbol data
            symbols: List of symbols to train on
            output_dir: Output directory for training results
            cache_dir: Optional cache directory for ranking/selection results
            add_timestamp: If True, append timestamp to output_dir to make runs distinguishable
            experiment_config: Optional ExperimentConfig object [NEW - preferred]
        """
        from datetime import datetime
        
        # NEW: Use experiment config if provided
        if experiment_config is not None and _NEW_CONFIG_AVAILABLE:
            self.data_dir = experiment_config.data_dir
            self.experiment_config = experiment_config
            
            # Auto-discover symbols if empty/missing (CRITICAL: do BEFORE fingerprint computation)
            if not experiment_config.symbols:
                from TRAINING.orchestration.utils.symbol_discovery import (
                    discover_symbols_from_data_dir,
                    select_symbol_batch
                )
                
                # Get interval from config
                interval = experiment_config.data.bar_interval if experiment_config.data else None
                
                # Discover symbols from data directory
                discovered_symbols, search_paths = discover_symbols_from_data_dir(
                    self.data_dir, interval
                )
                
                if not discovered_symbols:
                    raise ValueError(
                        f"symbols is empty, so auto-discovery is enabled; "
                        f"no symbols found under {self.data_dir} "
                        f"(searched: {search_paths})"
                    )
                
                # Apply batch selection if configured
                symbol_batch_size = experiment_config.data.symbol_batch_size if experiment_config.data else None
                random_seed = experiment_config.data.seed if experiment_config.data else 42
                
                resolved_symbols = select_symbol_batch(
                    discovered_symbols, symbol_batch_size, random_seed
                )
                
                # SST: Write resolved symbols back to canonical location
                experiment_config.symbols = resolved_symbols
                
                logger.info(
                    f"Auto-discovered {len(discovered_symbols)} symbols, "
                    f"selected {len(resolved_symbols)} (seed={random_seed})"
                )
            
            # Always read from canonical location (SST)
            self.symbols = experiment_config.symbols
        else:
            self.data_dir = Path(data_dir)
            self.symbols = symbols
            self.experiment_config = None
        
        # Store config limits for output directory binning
        self._max_rows_per_symbol = max_rows_per_symbol
        self._max_cs_samples = max_cs_samples

        # Run identity tracking (Phase 1 - Run Identity Lifecycle)
        # _partial_identity: Created early, holds identity before feature signature is known
        # run_identity: Finalized identity after feature selection (is_final=True)
        self._partial_identity: Optional[Any] = None
        self.run_identity: Optional[Any] = None

        # Generate run instance ID for directory name (includes timestamp + UUID for uniqueness)
        output_dir = Path(output_dir)
        if add_timestamp:
            # Use new generate_run_instance_id() for directory name (includes UUID suffix)
            from TRAINING.orchestration.utils.manifest import generate_run_instance_id
            output_dir_name = generate_run_instance_id()
        else:
            # Use provided name as-is (for backward compatibility)
            output_dir_name = output_dir.name

        # FIX: Respect user-specified --output-dir paths
        # If user explicitly specified --output-dir, use it as the base (optionally with timestamp)
        # Only use RESULTS/runs/ comparison group structure when no explicit path given
        if user_specified_output_dir:
            # User specified explicit path: use it as-is or with timestamp appended
            if add_timestamp:
                # Nest timestamped run inside user's specified directory
                # e.g., --output-dir my_run/ -> my_run/20260120_123456_abc123/
                self.output_dir = output_dir / output_dir_name
                logger.info(f"📁 Output directory: {self.output_dir} (user-specified with timestamp)")
            else:
                # Use exact path user specified
                self.output_dir = output_dir
                logger.info(f"📁 Output directory: {self.output_dir} (user-specified)")
            self._n_effective = self._estimate_n_effective_early()  # Still compute for logging
        else:
            # Default behavior: organize runs under RESULTS/runs/ with comparison group structure
            # Structure: RESULTS/runs/{comparison_group_dir}/{date_time_run_name}/
            # Comparison group dir is computed from configs at startup (data, symbols, n_effective, etc.)
            repo_root = Path(__file__).parent.parent.parent  # Go up from TRAINING/orchestration/ to repo root
            results_dir = repo_root / "RESULTS"
            runs_dir = results_dir / "runs"  # Organize all runs under runs/ subdirectory

            # Try to estimate n_effective early (before first target is processed)
            self._n_effective = self._estimate_n_effective_early()

            # Compute comparison group directory from configs available at startup
            # This organizes runs by metadata from the start, not moved later
            comparison_group_dir = self._compute_comparison_group_dir_at_startup()

            if comparison_group_dir:
                # Create directory structure: RESULTS/runs/{comparison_group_dir}/{date_time_run_name}/
                self.output_dir = runs_dir / comparison_group_dir / output_dir_name
                logger.info(f"📁 Output directory: {self.output_dir} (organized by comparison group: {comparison_group_dir})")
            elif self._n_effective is not None:
                # Fallback: Use sample size bin if comparison group can't be computed
                bin_info = self._get_sample_size_bin(self._n_effective)
                bin_name = bin_info["bin_name"]
                self.output_dir = runs_dir / bin_name / output_dir_name
                logger.info(f"📁 Output directory: {self.output_dir} (organized by sample size bin: {bin_name}, N={self._n_effective})")
                self._bin_info = bin_info
            else:
                # Final fallback: start in _pending/ - will be organized after first target
                self.output_dir = runs_dir / "_pending" / output_dir_name
                logger.info(f"📁 Output directory: {self.output_dir} (will be organized after first target)")
        
        self._run_name = output_dir_name  # Store for reference

        # Create directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store initial output dir for cohort organization (before any moves)
        self._initial_output_dir = self.output_dir
        
        # Enable persistent run logging (captures all output to {output_dir}/logs/run.log)
        try:
            from TRAINING.orchestration.utils.logging_setup import enable_run_logging
            self._run_log_path = enable_run_logging(
                output_dir=self.output_dir,
                log_filename="run.log",
                also_capture_stdout=True
            )
        except Exception as e:
            logger.warning(f"Could not enable run logging: {e}")
            self._run_log_path = None
        
        self.cache_dir = Path(cache_dir) if cache_dir else self.output_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache file paths
        self.target_ranking_cache = self.cache_dir / "target_rankings.json"
        self.feature_selection_cache = self.cache_dir / "feature_selections"
        self.feature_selection_cache.mkdir(parents=True, exist_ok=True)
        
        # Store feature selection model families (separate from training families)
        self.fs_model_families = fs_model_families
    
    def _estimate_n_effective_early(self) -> Optional[int]:
        """
        Try to estimate n_effective early from config limits, existing metadata, or data files.
        
        Priority:
        1. Use configured max_rows_per_symbol * num_symbols (if config limits are set)
        2. Check existing metadata from previous runs
        3. Estimate from data files
        
        Returns:
            Estimated n_effective or None if cannot be determined
        """
        logger.info("🔍 Attempting early n_effective estimation...")
        
        # Method 0: Use configured limits if available (PREFERRED - reflects actual data limits)
        if self._max_rows_per_symbol is not None and self._max_rows_per_symbol > 0:
            # Calculate expected n_effective based on config: max_rows_per_symbol * num_symbols
            # This reflects the actual data that will be loaded, not the full dataset size
            expected_n = self._max_rows_per_symbol * len(self.symbols)
            logger.info(f"🔍 Using configured max_rows_per_symbol={self._max_rows_per_symbol} × {len(self.symbols)} symbols = {expected_n} for output directory binning")
            return expected_n
        
        # Also check experiment config if available
        if self.experiment_config:
            try:
                exp_name = self.experiment_config.name
                if _CONFIG_LOADER_AVAILABLE:
                    exp_file = get_experiment_config_path(exp_name)
                    if exp_file.exists():
                        exp_yaml = load_experiment_config(exp_name)
                        exp_data = exp_yaml.get('data', {})
                        config_max_rows = exp_data.get('max_rows_per_symbol') or exp_data.get('max_samples_per_symbol')
                        if config_max_rows is not None and config_max_rows > 0:
                            expected_n = config_max_rows * len(self.symbols)
                            logger.info(f"🔍 Using experiment config max_rows_per_symbol={config_max_rows} × {len(self.symbols)} symbols = {expected_n} for output directory binning")
                            return expected_n
                else:
                    # Fallback for when config loader is not available
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    exp_data = exp_yaml.get('data', {})
                    config_max_rows = exp_data.get('max_rows_per_symbol') or exp_data.get('max_samples_per_symbol')
                    if config_max_rows is not None and config_max_rows > 0:
                        expected_n = config_max_rows * len(self.symbols)
                        logger.info(f"🔍 Using experiment config max_rows_per_symbol={config_max_rows} × {len(self.symbols)} symbols = {expected_n} for output directory binning")
                        return expected_n
            except Exception as e:
                logger.debug(f"Could not read max_rows_per_symbol from experiment config: {e}")
        
        # Method 1: Check if there's existing metadata from a previous run with same symbols/data
        # (This handles the case where you're re-running with same data)
        try:
            repo_root = Path(__file__).parent.parent.parent
            results_dir = repo_root / "RESULTS"
            
            # Look for existing runs with same symbols (quick check)
            if results_dir.exists():
                logger.debug(f"Checking existing runs in {results_dir} for matching symbols: {self.symbols}")
                # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                for n_dir in iterdir_sorted(results_dir):
                    if n_dir.is_dir() and n_dir.name.isdigit():
                        # Check if there's a recent run with similar structure
                        # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                        for run_dir in iterdir_sorted(n_dir):
                            if run_dir.is_dir():
                                # Check metadata.json in REPRODUCIBILITY
                                # DETERMINISM: Use rglob_sorted for deterministic iteration order
                                for metadata_file in rglob_sorted(run_dir, "REPRODUCIBILITY/TARGET_RANKING/*/cohort=*/metadata.json"):
                                    try:
                                        import json
                                        with open(metadata_file, 'r') as f:
                                            metadata = json.load(f)
                                        # Check if symbols match (rough check)
                                        existing_symbols = metadata.get('symbols', [])
                                        if existing_symbols and set(existing_symbols) == set(self.symbols):
                                            n_effective = metadata.get('n_effective')
                                            if n_effective and n_effective > 0:
                                                logger.info(f"🔍 Found matching n_effective={n_effective} from previous run with same symbols")
                                                return int(n_effective)
                                    except Exception as e:
                                        logger.debug(f"Failed to read {metadata_file}: {e}")
                                        continue
        except Exception as e:
            logger.debug(f"Could not check existing metadata for n_effective: {e}")
        
        # Method 2: Quick sample from data files to estimate sample size
        try:
            import pandas as pd
            total_rows = 0
            
            # Sample first few symbols to estimate
            sample_symbols = self.symbols[:3] if len(self.symbols) > 3 else self.symbols
            logger.info(f"🔍 Sampling {len(sample_symbols)} symbols from {self.data_dir} to estimate n_effective")
            
            for symbol in sample_symbols:
                # Try multiple possible paths
                possible_paths = [
                    self.data_dir / f"symbol={symbol}" / f"{symbol}.parquet",
                    self.data_dir / symbol / f"{symbol}.parquet",
                    self.data_dir / f"{symbol}.parquet"
                ]
                
                data_path = None
                for path in possible_paths:
                    if path.exists():
                        data_path = path
                        break
                
                if data_path is None:
                    logger.warning(f"⚠️  Data file not found for {symbol} (tried: {[str(p) for p in possible_paths]})")
                    continue
                
                logger.info(f"  ✓ Found data file for {symbol}: {data_path}")
                
                try:
                    # Use parquet metadata if available (faster - no data load)
                    try:
                        import pyarrow.parquet as pq
                        parquet_file = pq.ParquetFile(data_path)
                        symbol_rows = parquet_file.metadata.num_rows
                        total_rows += symbol_rows
                        logger.debug(f"  {symbol}: {symbol_rows} rows (from parquet metadata)")
                    except ImportError:
                        # pyarrow not available, try pandas
                        logger.debug(f"  pyarrow not available, using pandas for {symbol}")
                        symbol_rows = len(pd.read_parquet(data_path))
                        total_rows += symbol_rows
                        logger.debug(f"  {symbol}: {symbol_rows} rows (from pandas)")
                    except Exception as e:
                        logger.debug(f"  Could not read parquet metadata for {symbol}: {e}, trying pandas")
                        # Fallback: actually count rows (slower but works)
                        symbol_rows = len(pd.read_parquet(data_path))
                        total_rows += symbol_rows
                        logger.debug(f"  {symbol}: {symbol_rows} rows (from pandas fallback)")
                except Exception as e:
                    logger.debug(f"Could not read {data_path} for sample size estimation: {e}")
                    continue
            
            # Extrapolate to all symbols
            if total_rows > 0 and len(sample_symbols) > 0:
                avg_per_symbol = total_rows / len(sample_symbols)
                estimated_total = int(avg_per_symbol * len(self.symbols))
                logger.info(f"🔍 Estimated n_effective={estimated_total} from data file sampling ({len(sample_symbols)} symbols sampled, {total_rows} total rows)")
                return estimated_total
            else:
                logger.debug(f"Could not estimate n_effective: total_rows={total_rows}, sample_symbols={len(sample_symbols)}")
        except Exception as e:
            logger.warning(f"Could not estimate n_effective from data files: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        logger.info("⚠️  Could not determine n_effective early, will use _pending/ and organize after first target")
        return None
    
    def _compute_comparison_group_dir_at_startup(self) -> Optional[str]:
        """
        Compute comparison group directory name from configs available at startup.
        
        Uses metadata available at config load time:
        - data_dir, symbols, min_cs, max_cs_samples → dataset_signature
        - n_effective (estimated) → n_effective
        - view/routing (if known) → routing_signature
        
        Note: task_signature, model_family, feature_signature come later and are not
        included in the directory name (they're in the comparison group key for diff telemetry).
        
        Returns:
            Comparison group directory name (e.g., "data-012e801c_route-fcabc6e9_n-988")
            or None if cannot be computed
        """
        import hashlib
        
        try:
            from TRAINING.orchestration.utils.diff_telemetry import ComparisonGroup
            
            # Compute dataset signature from available configs
            dataset_parts = []
            
            # Experiment config path (if available)
            if self.experiment_config:
                if _CONFIG_LOADER_AVAILABLE:
                    exp_file = get_experiment_config_path(self.experiment_config.name)
                else:
                    exp_file = _get_experiment_config_path(self.experiment_config.name)
                if exp_file.exists():
                    dataset_parts.append(f"exp:{self.experiment_config.name}")
            
            # Data directory path (normalized)
            if self.data_dir:
                data_dir_str = str(self.data_dir.resolve())
                dataset_parts.append(f"data_dir={data_dir_str}")
            
            # Symbols (sorted for consistency)
            if self.symbols:
                symbols_str = "|".join(sorted(self.symbols))
                dataset_parts.append(f"symbols={symbols_str}")
            
            # Config limits (if available)
            if self._max_cs_samples is not None:
                dataset_parts.append(f"max_cs_samples={self._max_cs_samples}")
            
            # Also check experiment config for min_cs, max_cs_samples
            min_cs = None
            max_cs_samples = self._max_cs_samples
            if self.experiment_config:
                try:
                    exp_name = self.experiment_config.name
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    if exp_yaml:
                        exp_data = exp_yaml.get('data', {})
                        min_cs = exp_data.get('min_cs')
                        if max_cs_samples is None:
                            max_cs_samples = exp_data.get('max_cs_samples') or exp_data.get('max_rows_per_symbol')
                        if min_cs is not None:
                            dataset_parts.append(f"min_cs={min_cs}")
                except Exception as e:
                    # Best-effort: dataset metadata extraction failed, continue without it
                    logger.debug(f"Could not extract dataset metadata from experiment config: {e}")
            
            # Compute dataset signature hash
            dataset_signature = None
            if dataset_parts:
                dataset_str = "|".join(sorted(dataset_parts))
                dataset_signature = hashlib.sha256(dataset_str.encode()).hexdigest()[:16]
            
            # Routing signature (default to CROSS_SECTIONAL if not known)
            # This can be refined later when view is determined
            routing_signature = None
            routing_str = "view=CROSS_SECTIONAL"  # Default
            routing_signature = hashlib.sha256(routing_str.encode()).hexdigest()[:16]
            
            # Build partial comparison group (what we know at startup)
            comparison_group = ComparisonGroup(
                dataset_signature=dataset_signature,
                routing_signature=routing_signature,
                n_effective=self._n_effective
            )
            
            # Generate directory name
            dir_name = comparison_group.to_dir_name()
            return dir_name if dir_name != "default" else None
            
        except Exception as e:
            logger.debug(f"Could not compute comparison group directory at startup: {e}")
            return None
    
    def _get_sample_size_bin(self, n_effective: int) -> Dict[str, Any]:
        """
        Bin n_effective into readable ranges for grouping similar sample sizes.
        
        **Boundary Rules (CRITICAL - DO NOT CHANGE WITHOUT VERSIONING):**
        - Boundaries are EXCLUSIVE upper bounds: `bin_min <= n_effective < bin_max`
        - Example: `sample_25k-50k` means `25000 <= n_effective < 50000`
        - This ensures unambiguous binning (50,000 always goes to `sample_50k-100k`, never `sample_25k-50k`)
        
        **Binning Scheme Version:** `sample_bin_v1`
        - If you change thresholds, increment version and update this docstring
        - Old runs retain their original bin metadata for backward compatibility
        
        **Bins (v1):**
        - sample_0-5k: 0 <= N < 5,000
        - sample_5k-10k: 5,000 <= N < 10,000
        - sample_10k-25k: 10,000 <= N < 25,000
        - sample_25k-50k: 25,000 <= N < 50,000
        - sample_50k-100k: 50,000 <= N < 100,000
        - sample_100k-250k: 100,000 <= N < 250,000
        - sample_250k-500k: 250,000 <= N < 500,000
        - sample_500k-1M: 500,000 <= N < 1,000,000
        - sample_1M+: N >= 1,000,000
        
        This groups runs with similar cross-sectional sample sizes together for easy comparison.
        **Note:** Bin is for directory organization only. Trend series keys use stable identity (cohort_id, stage, target)
        and do NOT include bin_name to prevent fragmentation when binning scheme changes.
        
        Args:
            n_effective: Effective sample size
            
        Returns:
            Dict with keys: bin_name, bin_min, bin_max, binning_scheme_version
        """
        BINNING_SCHEME_VERSION = "sample_bin_v1"
        
        # Define bins with EXCLUSIVE upper bounds (bin_min <= N < bin_max)
        bins = [
            (0, 5000, "sample_0-5k"),
            (5000, 10000, "sample_5k-10k"),
            (10000, 25000, "sample_10k-25k"),
            (25000, 50000, "sample_25k-50k"),
            (50000, 100000, "sample_50k-100k"),
            (100000, 250000, "sample_100k-250k"),
            (250000, 500000, "sample_250k-500k"),
            (500000, 1000000, "sample_500k-1M"),
            (1000000, float('inf'), "sample_1M+")
        ]
        
        for bin_min, bin_max, bin_name in bins:
            if bin_min <= n_effective < bin_max:
                return {
                    "bin_name": bin_name,
                    "bin_min": bin_min,
                    "bin_max": bin_max if bin_max != float('inf') else None,
                    "binning_scheme_version": BINNING_SCHEME_VERSION
                }
        
        # Fallback (should never reach here)
        return {
            "bin_name": "sample_unknown",
            "bin_min": None,
            "bin_max": None,
            "binning_scheme_version": BINNING_SCHEME_VERSION
        }

    # =========================================================================
    # Caching wrapper methods (called by PipelineStageMixin)
    # These provide the instance context (cache paths) to standalone functions
    # =========================================================================

    def _get_cache_key(self, symbols: List[str], config_hash: str) -> str:
        """Generate cache key from symbols and config hash."""
        return _get_cache_key_impl(symbols, config_hash)

    def _load_cached_rankings(self, cache_key: str, use_cache: bool = True) -> Optional[List[Dict[str, Any]]]:
        """Load cached target rankings from instance cache path."""
        return _load_cached_rankings_impl(self.target_ranking_cache, cache_key, use_cache)

    def _save_cached_rankings(self, cache_key: str, cache_data: List[Dict[str, Any]]) -> None:
        """Save target rankings to instance cache path."""
        _save_cached_rankings_impl(self.target_ranking_cache, cache_key, cache_data)

    def _load_cached_features(self, target: str) -> Optional[List[str]]:
        """Load cached feature selection for a target from instance cache path."""
        return _load_cached_features_impl(self.feature_selection_cache, target)

    def _save_cached_features(self, target: str, features: List[str]) -> None:
        """Save feature selection results to instance cache path."""
        _save_cached_features_impl(self.feature_selection_cache, target, features)

    # =========================================================================
    # Run Identity Lifecycle Methods (Phase 1 - Architecture Remediation)
    # =========================================================================

    def _compute_feature_signature_from_target_features(
        self,
        target_features: Dict[str, Any]
    ) -> str:
        """
        Compute a combined feature signature from all target features.

        RI-003: This provides the feature_signature needed to finalize RunIdentity.

        Args:
            target_features: Dict mapping target -> features (list or dict for symbol-specific)

        Returns:
            64-char SHA256 hash of combined feature set
        """
        import hashlib

        # Collect all unique features across all targets
        all_features = set()
        for target in sorted(target_features.keys()):
            features = target_features[target]
            if isinstance(features, list):
                all_features.update(features)
            elif isinstance(features, dict):
                # Symbol-specific or BOTH view
                if 'cross_sectional' in features:
                    all_features.update(features.get('cross_sectional', []))
                if 'symbol_specific' in features:
                    for sym_features in features.get('symbol_specific', {}).values():
                        all_features.update(sym_features)
                else:
                    # Plain dict mapping symbol -> features
                    for sym_features in features.values():
                        if isinstance(sym_features, list):
                            all_features.update(sym_features)

        # Sort and join for deterministic hashing
        feature_list_str = "|".join(sorted(all_features))
        return hashlib.sha256(feature_list_str.encode()).hexdigest()

    def _finalize_run_identity(self, target_features: Dict[str, Any]) -> None:
        """
        Finalize the partial run identity after feature selection completes.

        RI-003: This is the critical step that was missing - calling finalize()
        on the partial identity created at TARGET_RANKING stage.

        Args:
            target_features: Dict mapping target -> selected features
        """
        if self._partial_identity is None:
            logger.warning("No partial identity to finalize (was TARGET_RANKING identity creation successful?)")
            return

        if target_features is None or len(target_features) == 0:
            logger.warning("Cannot finalize identity with empty target_features")
            return

        try:
            # Compute feature signature from all selected features
            feature_signature = self._compute_feature_signature_from_target_features(target_features)

            # Finalize the identity (creates new object with is_final=True)
            self.run_identity = self._partial_identity.finalize(feature_signature)
            logger.info(f"✅ Finalized RunIdentity: strict_key={self.run_identity.strict_key[:16]}...")

            # Update manifest with the finalized identity
            try:
                from TRAINING.orchestration.utils.manifest import create_manifest
                create_manifest(
                    self.output_dir,
                    run_id=None,  # Let manifest derive from run_identity
                    run_identity=self.run_identity,
                    run_instance_id=self.output_dir.name
                )
                logger.debug("Updated manifest with finalized RunIdentity")
            except Exception as e:
                logger.warning(f"Failed to update manifest with finalized identity: {e}")

        except ValueError as e:
            # finalize() raises ValueError if required signatures are missing
            logger.warning(f"Cannot finalize RunIdentity: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error finalizing RunIdentity: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")

    def _get_stable_run_id(self) -> str:
        """
        Get stable run_id from finalized identity.

        RI-004: Single canonical function for run_id derivation with strict mode enforcement.

        Returns:
            Stable run_id derived from finalized RunIdentity

        Raises:
            RuntimeError: In strict mode if run_identity is not finalized
        """
        if self.run_identity is not None:
            from TRAINING.orchestration.utils.manifest import derive_run_id_from_identity
            return derive_run_id_from_identity(run_identity=self.run_identity)

        from TRAINING.common.determinism import is_strict_mode
        if is_strict_mode():
            raise RuntimeError(
                "run_identity not finalized - cannot derive stable run_id in strict mode. "
                "Call _finalize_run_identity() after feature selection completes."
            )

        # Best-effort fallback (non-strict mode)
        logger.warning("Using unstable run_id from output_dir.name (run_identity not finalized)")
        return self.output_dir.name

    def _run_leakage_diagnostics(
        self,
        training_results: Dict[str, Any],
        targets: List[str],
        mtf_data: Any,
        train_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run leakage diagnostics using leakage sentinels.

        This method validates that models don't exhibit leakage patterns
        by testing with known leaky and clean targets.

        Args:
            training_results: Results from training pipeline
            targets: List of target names that were trained
            mtf_data: Multi-timeframe data loader
            train_kwargs: Training kwargs used

        Returns:
            Dictionary with leakage diagnostic results
        """
        if not _SENTINELS_AVAILABLE:
            logger.warning("Leakage sentinels not available - skipping diagnostics")
            return {"skipped": True, "reason": "sentinels_unavailable"}

        try:
            logger.info("🔍 Running leakage diagnostics with sentinels...")

            # Import the sentinel runner
            from TRAINING.ranking.predictability.leakage_sentinels import (
                run_leakage_sentinel_validation
            )

            # Run validation
            diagnostics = run_leakage_sentinel_validation(
                training_results=training_results,
                targets=targets,
                mtf_data=mtf_data,
                output_dir=self.output_dir,
            )

            # Log summary
            if diagnostics.get("passed", False):
                logger.info("✅ Leakage diagnostics passed")
            else:
                logger.warning(f"⚠️ Leakage diagnostics flagged issues: {diagnostics.get('issues', [])}")

            return diagnostics

        except ImportError as e:
            logger.warning(f"Leakage sentinels import failed: {e}")
            return {"skipped": True, "reason": f"import_error: {e}"}
        except Exception as e:
            logger.error(f"Leakage diagnostics failed: {e}")
            return {"skipped": True, "reason": f"error: {e}"}

    def train_with_intelligence(
        self,
        auto_targets: bool = False,
        top_n_targets: int = 5,
        auto_features: bool = False,
        top_m_features: int = 100,
        decision_apply_mode: bool = False,  # NEW: Enable apply mode for decisions
        decision_dry_run: bool = False,  # NEW: Dry-run mode (show patch without applying)
        decision_min_level: int = 2,  # NEW: Minimum decision level to apply
        targets: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        families: Optional[List[str]] = None,
        strategy: str = 'single_task',
        use_cache: bool = True,
        run_leakage_diagnostics: bool = False,
        max_targets_to_evaluate: Optional[int] = None,  # Limit number of targets to evaluate (for faster testing)
        targets_to_evaluate: Optional[List[str]] = None,  # NEW: Whitelist of specific targets to evaluate (works with auto_targets=true)
        **train_kwargs
    ) -> Dict[str, Any]:
        """
        Train models with intelligent target/feature selection.
        
        Args:
            auto_targets: If True, automatically rank and select targets
            top_n_targets: Number of top targets to select (if auto_targets=True)
            auto_features: If True, automatically select features per target
            top_m_features: Number of top features per target (if auto_features=True)
            targets: Manual target list (overrides auto_targets if provided)
            max_targets_to_evaluate: Optional limit on number of targets to evaluate (for faster testing)
            features: Manual feature list (overrides auto_features if provided)
            families: Model families to train
            strategy: Training strategy ('single_task', 'multi_task', 'cascade')
            run_leakage_diagnostics: If True, run leakage sentinel tests after training
            **train_kwargs: Additional arguments passed to train_with_strategies
        
        Returns:
            Training results dictionary
        """
        logger.info("🚀 Starting intelligent training pipeline")

        # Initialize training event emitter for dashboard monitoring
        run_id = getattr(self, 'run_identity', None)
        run_id_str = run_id.run_id if run_id and hasattr(run_id, 'run_id') else str(self.output_dir.name)
        init_training_events(run_id_str)
        emit_progress("initializing", 0, message="Starting intelligent training pipeline")
        _run_start_time = time.time()

        # Cache SST YAML once for entire run (single load, reuse everywhere in this method)
        # This prevents inconsistency from multiple loads and ensures we read the same config
        self._exp_yaml_sst = {}
        if self.experiment_config:
            try:
                exp_name = self.experiment_config.name
                # Log the identifier being used (catch path vs name bugs)
                try:
                    exp_path = _get_experiment_config_path(exp_name)
                    logger.debug(f"Loading SST YAML: name={exp_name}, resolved_path={exp_path}")
                except Exception:
                    logger.debug(f"Loading SST YAML: name={exp_name}")
                
                self._exp_yaml_sst = _load_experiment_config_safe(exp_name) or {}
                if not isinstance(self._exp_yaml_sst, dict):
                    logger.warning(f"SST YAML loaded but not a dict: {type(self._exp_yaml_sst)}")
                    self._exp_yaml_sst = {}
                else:
                    logger.debug(f"Cached SST YAML: keys={list(self._exp_yaml_sst.keys())}")
                    # Log critical sections for debugging
                    if 'training' in self._exp_yaml_sst:
                        logger.debug(f"  training.model_families={self._exp_yaml_sst['training'].get('model_families')}")
                    if 'feature_selection' in self._exp_yaml_sst:
                        logger.debug(f"  feature_selection.model_families={self._exp_yaml_sst['feature_selection'].get('model_families')}")
            except Exception as e:
                logger.warning(f"Failed to load SST YAML for {self.experiment_config.name}: {e}")
                self._exp_yaml_sst = {}
        
        # Pre-run decision hook: Load latest decision and optionally apply to config
        resolved_config_patch = {}
        decision_artifact_dir = None
        try:
            from TRAINING.decisioning.decision_engine import DecisionEngine
            from TRAINING.orchestration.utils.cohort_metadata_extractor import extract_cohort_metadata
            
            # Try to extract cohort metadata early (for decision loading)
            # This is approximate - actual cohort_id will be computed later
            try:
                cohort_metadata = extract_cohort_metadata(
                    symbols=self.symbols,
                    min_cs=train_kwargs.get('min_cs', 10),
                    max_cs_samples=train_kwargs.get('max_cs_samples')
                )
                cohort_id = None
                segment_id = None
                if cohort_metadata:
                    # Compute approximate cohort_id (will be refined later)
                    # Use view from metadata if available, default to CROSS_SECTIONAL
                    from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
                    temp_tracker = ReproducibilityTracker(output_dir=self.output_dir)
                    # Normalize view from metadata (may be string from JSON)
                    approx_view_str = cohort_metadata.get('view', View.CROSS_SECTIONAL.value)
                    approx_view = View.from_string(approx_view_str) if isinstance(approx_view_str, str) else approx_view_str
                    cohort_id = temp_tracker._compute_cohort_id(cohort_metadata, view=approx_view)
                    
                    # Try to get segment_id from index (target-first structure first)
                    from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
                    globals_dir = get_globals_dir(self.output_dir)
                    index_file = globals_dir / "index.parquet"
                    if not index_file.exists():
                        # Fallback to legacy REPRODUCIBILITY structure
                        repro_dir = self.output_dir / "REPRODUCIBILITY"
                        index_file = repro_dir / "index.parquet"
                    if index_file.exists():
                        try:
                            import pandas as pd
                            df = pd.read_parquet(index_file)
                            cohort_mask = df['cohort_id'] == cohort_id
                            if cohort_mask.any() and 'segment_id' in df.columns:
                                # Get latest segment_id for this cohort
                                segment_id = int(df[cohort_mask]['segment_id'].iloc[-1])
                        except Exception as e:
                            # Diagnostic: segment_id extraction failed, continue without it
                            logger.debug(f"Could not extract segment_id from index: {e}")
                
                if cohort_id and (decision_apply_mode or decision_dry_run):
                    # Read index from target-first structure first
                    from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
                    globals_dir = get_globals_dir(self.output_dir)
                    index_file = globals_dir / "index.parquet"
                    if not index_file.exists():
                        # Fallback to legacy REPRODUCIBILITY structure
                        repro_dir = self.output_dir / "REPRODUCIBILITY"
                        index_file = repro_dir / "index.parquet"
                    if index_file.exists():
                        engine = DecisionEngine(index_file, apply_mode=decision_apply_mode)
                        latest_decision = engine.load_latest(cohort_id, base_dir=self.output_dir.parent)
                        
                        if latest_decision:
                            logger.info(f"📊 Decision selection: cohort_id={cohort_id}, segment_id={segment_id}, "
                                      f"decision_level={latest_decision.decision_level}, "
                                      f"actions={latest_decision.decision_action_mask}, "
                                      f"reasons={latest_decision.decision_reason_codes}")
                            
                            if latest_decision.decision_level >= decision_min_level:
                                # Create artifact directory for receipts (one location, one format)
                                from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
                                globals_dir = get_globals_dir(self.output_dir)
                                decision_artifact_dir = globals_dir / "decision_patches"
                                decision_artifact_dir.mkdir(parents=True, exist_ok=True)
                                
                                # Always save receipts (dry-run or apply) - NON-NEGOTIABLE
                                patched_config, patch, warnings = engine.apply_patch(train_kwargs, latest_decision)
                                
                                # Receipt 1: Decision used (always)
                                decision_used_file = decision_artifact_dir / "decision_used.json"
                                # SST: Use write_atomic_json for atomic write with canonical serialization
                                from TRAINING.common.utils.file_utils import write_atomic_json
                                write_atomic_json(decision_used_file, latest_decision.to_dict())
                                
                                # Receipt 2: Resolved config baseline (always)
                                try:
                                    # DETERMINISM: Use canonical_yaml() for deterministic YAML output
                                    from TRAINING.common.utils.determinism_serialization import write_canonical_yaml
                                    resolved_config_file = decision_artifact_dir / "resolved_config.yaml"
                                    write_canonical_yaml(resolved_config_file, train_kwargs.copy())
                                except ImportError:
                                    resolved_config_file = decision_artifact_dir / "resolved_config.json"
                                    # SST: Use write_atomic_json for atomic write with canonical serialization
                                    from TRAINING.common.utils.file_utils import write_atomic_json
                                    write_atomic_json(resolved_config_file, train_kwargs.copy())
                                
                                # Receipt 3: Applied patch (or "none" if no patch)
                                patch_file = decision_artifact_dir / "applied_patch.json"
                                
                                if decision_dry_run:
                                    # Dry-run: show patch without applying
                                    # SST: Use write_atomic_json for atomic write with canonical serialization
                                    from TRAINING.common.utils.file_utils import write_atomic_json
                                    write_atomic_json(patch_file, {
                                        "mode": "dry_run",
                                        "decision_run_id": latest_decision.run_id,
                                        "cohort_id": cohort_id,
                                        "segment_id": segment_id,
                                        "patch": patch if patch else "none",
                                        "warnings": warnings,
                                        "keys_changed": list(patch.keys()) if patch else [],
                                    })
                                    
                                    logger.info("="*80)
                                    logger.info("🔍 DRY RUN: Decision Application Preview")
                                    logger.info("="*80)
                                    logger.info(f"Decision selected: {latest_decision.run_id}")
                                    logger.info(f"  Cohort: {cohort_id}, Segment: {segment_id}")
                                    logger.info(f"  Level: {latest_decision.decision_level}, Actions: {latest_decision.decision_action_mask}")
                                    logger.info(f"  Reasons: {latest_decision.decision_reason_codes}")
                                    if patch:
                                        logger.info(f"Patch that WOULD be applied:")
                                        for key, value in patch.items():
                                            old_val = train_kwargs.get(key.split('.')[0]) if '.' in key else train_kwargs.get(key)
                                            logger.info(f"  {key}: {old_val} → {value}")
                                        logger.info(f"Keys that would change: {list(patch.keys())}")
                                    else:
                                        logger.info("No patch (no actions or all actions skipped)")
                                    if warnings:
                                        for w in warnings:
                                            logger.warning(f"  ⚠️  {w}")
                                    logger.info(f"📄 Receipts saved to: {decision_artifact_dir.relative_to(self.output_dir)}/")
                                    logger.info("  - decision_used.json")
                                    logger.info("  - resolved_config.yaml")
                                    logger.info(f"  - applied_patch.json (mode: dry_run)")
                                    logger.info("="*80)
                                elif decision_apply_mode:
                                    # Apply decision patch to config
                                    resolved_config_patch = patch
                                    train_kwargs.update(patched_config)
                                    
                                    # Save patched config (receipt - overwrites resolved_config.yaml)
                                    try:
                                        # DETERMINISM: Use canonical_yaml() for deterministic YAML output
                                        from TRAINING.common.utils.determinism_serialization import write_canonical_yaml
                                        patched_config_file = decision_artifact_dir / "resolved_config.yaml"
                                        write_canonical_yaml(patched_config_file, patched_config)
                                    except ImportError:
                                        patched_config_file = decision_artifact_dir / "resolved_config.json"
                                        # SST: Use write_atomic_json for atomic write with canonical serialization
                                        from TRAINING.common.utils.file_utils import write_atomic_json
                                        write_atomic_json(patched_config_file, patched_config)
                                    
                                    # Update config_hash to reflect patch
                                    import hashlib
                                    patch_hash = hashlib.sha256(json.dumps(patch, sort_keys=True).encode()).hexdigest()[:8]
                                    train_kwargs['decision_patch_hash'] = patch_hash
                                    
                                    logger.info("="*80)
                                    logger.info("🔧 APPLY MODE: Decision Patch Applied")
                                    logger.info("="*80)
                                    logger.info(f"Decision: {latest_decision.run_id} (cohort={cohort_id}, segment={segment_id})")
                                    if patch:
                                        logger.info(f"Patch applied:")
                                        for key, value in patch.items():
                                            logger.info(f"  {key}: → {value}")
                                        logger.info(f"Keys changed: {list(patch.keys())}")
                                    else:
                                        logger.info("No patch (patch='none')")
                                    if warnings:
                                        for w in warnings:
                                            logger.warning(f"  ⚠️  {w}")
                                    logger.info(f"🔑 Config hash updated with patch_hash: {patch_hash}")
                                    logger.info(f"📄 Receipts saved to: {decision_artifact_dir.relative_to(self.output_dir)}/")
                                    logger.info("  - decision_used.json")
                                    logger.info("  - resolved_config.yaml (patched)")
                                    logger.info(f"  - applied_patch.json (mode: apply)")
                                    logger.info("="*80)
                            else:
                                logger.info(f"⏭️  Decision level {latest_decision.decision_level} < {decision_min_level}, skipping application")
                        else:
                            logger.debug(f"No decision found for cohort_id={cohort_id}")
            except Exception as e:
                logger.debug(f"Pre-run decision loading failed (non-critical): {e}")
                import traceback
                logger.debug(traceback.format_exc())
        except ImportError:
            logger.debug("Decision engine not available, skipping pre-run hook")
        
        # Create resolved config for run reproduction (after all configs are loaded)
        try:
            from TRAINING.orchestration.utils.manifest import create_resolved_config
            
            # Load multi_model_config if not already loaded
            multi_model_config_for_resolved = None
            if 'multi_model_config' in locals():
                multi_model_config_for_resolved = multi_model_config
            else:
                try:
                    multi_model_config_for_resolved = load_multi_model_config()
                except Exception as e:
                    logger.debug(f"Could not load multi_model_config for resolved config: {e}")
            
            # Get base seed from config
            base_seed = None
            try:
                from CONFIG.config_loader import get_cfg
                base_seed = int(get_cfg("pipeline.determinism.base_seed", default=42))
            except Exception as e:
                # Best-effort: config access failed, use default seed
                logger.debug(f"Could not get base_seed from config: {e}, using default 42")
                base_seed = 42
            
            # Determine task type (default to regression, will be refined per-target)
            task_type = "regression"  # Default, will be determined per-target
            
            # Get model families from config
            model_families_for_resolved = None
            if multi_model_config_for_resolved and 'model_families' in multi_model_config_for_resolved:
                model_families_dict = multi_model_config_for_resolved.get('model_families', {})
                if isinstance(model_families_dict, dict):
                    model_families_for_resolved = sorted([
                        name for name, config in model_families_dict.items()
                        if isinstance(config, dict) and config.get('enabled', False)
                    ])
            
            # Convert experiment_config to dict if it's an object
            experiment_config_dict = None
            if self.experiment_config:
                if hasattr(self.experiment_config, '__dict__'):
                    data_dir = None
                    if hasattr(self.experiment_config, 'data_dir'):
                        data_dir_val = getattr(self.experiment_config, 'data_dir')
                        data_dir = str(data_dir_val) if data_dir_val else None
                    
                    interval = None
                    if hasattr(self.experiment_config, 'data') and hasattr(self.experiment_config.data, 'bar_interval'):
                        interval = self.experiment_config.data.bar_interval
                    elif hasattr(self.experiment_config, 'bar_interval'):
                        interval = getattr(self.experiment_config, 'bar_interval')
                    elif hasattr(self.experiment_config, 'interval'):
                        interval = getattr(self.experiment_config, 'interval')
                    
                    experiment_config_dict = {
                        "name": getattr(self.experiment_config, 'name', None),
                        "data_dir": data_dir,
                        "symbols": getattr(self.experiment_config, 'symbols', None),
                        "interval": interval,
                    }
                elif isinstance(self.experiment_config, dict):
                    experiment_config_dict = self.experiment_config
                elif self._exp_yaml_sst:
                    # Use cached YAML if available
                    experiment_config_dict = self._exp_yaml_sst
            
            # Create resolved config
            # Note: RunIdentity may not be finalized yet (only finalized after feature selection)
            # If identity not available, create_resolved_config() will use unstable run_id
            resolved_config = create_resolved_config(
                output_dir=self.output_dir,
                run_id=None,  # Will be generated from identity or output_dir fallback
                run_identity=None,  # Not available yet (only created per-target after feature selection)
                experiment_config=experiment_config_dict,
                multi_model_config=multi_model_config_for_resolved,
                model_families=model_families_for_resolved,
                task_type=task_type,
                base_seed=base_seed,
                overrides=resolved_config_patch if resolved_config_patch else None
            )
            
            # CRITICAL: Validate that config.resolved.json was created successfully
            resolved_config_path = self.output_dir / "globals" / "config.resolved.json"
            if not resolved_config_path.exists():
                logger.error(
                    f"❌ CRITICAL: config.resolved.json was not created at {resolved_config_path}. "
                    f"Run will continue but snapshots may lack config fingerprints, breaking determinism."
                )
            elif resolved_config and 'config_fingerprint' in resolved_config:
                logger.info(f"✅ Resolved config created: {resolved_config_path}")
                logger.debug(f"   Config fingerprint: {resolved_config['config_fingerprint'][:16]}...")
                if 'deterministic_config_fingerprint' in resolved_config:
                    logger.debug(f"   Deterministic fingerprint: {resolved_config['deterministic_config_fingerprint'][:16]}...")
            else:
                logger.warning(
                    f"⚠️ Resolved config created but missing fingerprints. "
                    f"This may break determinism tracking."
                )
            
            # Save user config if available
            try:
                from TRAINING.orchestration.utils.manifest import save_user_config
                experiment_config_path = None
                if self.experiment_config and hasattr(self.experiment_config, 'name'):
                    try:
                        from TRAINING.orchestration.intelligent_trainer import _get_experiment_config_path
                        experiment_config_path = _get_experiment_config_path(self.experiment_config.name)
                    except Exception as e:
                        # Best-effort: config path resolution failed, continue without it
                        logger.debug(f"Could not resolve experiment_config_path: {e}")
                save_user_config(
                    output_dir=self.output_dir,
                    experiment_config_path=experiment_config_path,
                    experiment_config_dict=experiment_config_dict
                )
            except Exception as e:
                logger.debug(f"Could not save user config: {e}")
            
            # Save overrides if any were applied
            if resolved_config_patch:
                try:
                    from TRAINING.orchestration.utils.manifest import save_overrides_config
                    save_overrides_config(
                        output_dir=self.output_dir,
                        overrides=resolved_config_patch
                    )
                except Exception as e:
                    logger.debug(f"Could not save overrides config: {e}")
            
            # Save all config files to globals/configs/ for run recreation
            try:
                from TRAINING.orchestration.utils.manifest import save_all_configs
                experiment_config_name = None
                if self.experiment_config and hasattr(self.experiment_config, 'name'):
                    experiment_config_name = self.experiment_config.name
                save_all_configs(
                    output_dir=self.output_dir,
                    experiment_config_name=experiment_config_name
                )
            except Exception as e:
                logger.debug(f"Could not save all configs: {e}")
            
            # Update manifest with config_fingerprint
            if resolved_config and 'config_fingerprint' in resolved_config:
                try:
                    from TRAINING.orchestration.utils.manifest import create_manifest
                    create_manifest(
                        output_dir=self.output_dir,
                        config_digest=resolved_config.get('deterministic_config_fingerprint') or resolved_config.get('config_fingerprint'),
                        experiment_config=experiment_config_dict
                    )
                    # Validate manifest was created
                    manifest_path = self.output_dir / "manifest.json"
                    if not manifest_path.exists():
                        logger.warning(f"⚠️ Manifest creation reported success but manifest.json not found at {manifest_path}")
                    else:
                        logger.debug(f"✅ Manifest created: {manifest_path}")
                except Exception as e:
                    logger.warning(f"Could not update manifest with config fingerprint: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
        except Exception as e:
            logger.error(
                f"❌ CRITICAL: Failed to create resolved config: {e}. "
                f"Run will continue but snapshots may lack config fingerprints, breaking determinism and reproducibility."
            )
            import traceback
            logger.debug(traceback.format_exc())
        
        # Extract data limits from train_kwargs (passed from main, potentially patched by decisions)
        min_cs = train_kwargs.get('min_cs', 10)
        max_cs_samples = train_kwargs.get('max_cs_samples')
        max_rows_per_symbol = train_kwargs.get('max_rows_per_symbol')
        
        # Step 1: Target selection
        # ================================================================
        # Skip target ranking when:
        # 1. RAW_SEQUENCE mode: Target ranking uses feature importance which doesn't apply
        # 2. CS_RANKING mode: Ranking defines its own objective, no need for target ranking
        # ================================================================
        input_mode = get_input_mode(experiment_config=self._exp_yaml_sst)
        cs_ranking_enabled = is_cs_ranking_enabled(self._exp_yaml_sst)

        # Check if we should skip target ranking
        skip_target_ranking = (
            (input_mode == InputMode.RAW_SEQUENCE and auto_targets and targets is None)
            or (cs_ranking_enabled and auto_targets and targets is None)
        )

        if skip_target_ranking:
            skip_reason = "Cross-Sectional Ranking Mode" if cs_ranking_enabled else "Raw Sequence Mode"
            logger.warning("="*80)
            logger.warning(f"STEP 1: SKIPPED - {skip_reason} (target ranking disabled)")
            logger.warning("="*80)
            if cs_ranking_enabled:
                logger.warning(
                    "Cross-sectional ranking mode is enabled. "
                    "Target ranking uses feature importance which doesn't apply to ranking objectives. "
                    "Please set 'intelligent_training.manual_targets' in your experiment config "
                    "or pass --targets on the command line."
                )
            else:
                logger.warning(
                    "RAW_SEQUENCE mode requires manual target specification. "
                    "Target ranking uses feature importance which doesn't apply to raw OHLCV data. "
                    "Please set 'intelligent_training.manual_targets' in your experiment config "
                    "or pass --targets on the command line."
                )
            # Try to discover targets from a sample symbol
            if self.symbols:
                sample_symbol = sorted(self.symbols)[0]
                try:
                    targets_dict = discover_targets(sample_symbol, self.data_dir)
                    all_targets = sorted(targets_dict.keys())
                    if all_targets:
                        # Use top_n or all discovered targets
                        n_targets = top_n_targets if top_n_targets else len(all_targets)
                        targets = all_targets[:n_targets]
                        logger.info(
                            f"RAW_SEQUENCE mode: Discovered {len(all_targets)} targets from {sample_symbol}, "
                            f"using first {len(targets)}: {targets}"
                        )
                except Exception as e:
                    logger.warning(f"Could not discover targets: {e}")

            if not targets:
                raise ValueError(
                    "RAW_SEQUENCE mode: No targets specified and none discovered. "
                    "Set 'intelligent_training.manual_targets' in experiment config."
                )
            emit_stage_change("initializing", "training")  # Skip ranking and feature_selection

        elif auto_targets and targets is None:
            logger.info("="*80)
            logger.info("STEP 1: Automatic Target Ranking")
            logger.info("="*80)
            emit_stage_change("initializing", "ranking")
            emit_progress("ranking", 0, message="Starting target ranking")
            if max_targets_to_evaluate is not None:
                logger.info(f"🔢 max_targets_to_evaluate={max_targets_to_evaluate} (type: {type(max_targets_to_evaluate).__name__})")
            # Request summary for diagnostic purposes
            ranking_result = self.rank_targets_auto(
                top_n=top_n_targets,
                use_cache=use_cache,
                max_targets_to_evaluate=max_targets_to_evaluate,
                targets_to_evaluate=targets_to_evaluate,  # NEW: Pass whitelist
                min_cs=min_cs,
                max_cs_samples=max_cs_samples,
                max_rows_per_symbol=max_rows_per_symbol,
                return_summary=True  # Request summary for diagnostics
            )
            
            # Handle return_summary=True case
            if isinstance(ranking_result, tuple):
                targets, ranking_summary = ranking_result
            else:
                targets = ranking_result
                ranking_summary = None

            # Emit ranking progress
            emit_progress(
                "ranking", 100,
                targets_total=len(targets) if targets else 0,
                message=f"Target ranking complete: {len(targets) if targets else 0} targets selected"
            )

            if not targets:
                # Check dev_mode before raising error
                dev_mode = False
                try:
                    from CONFIG.dev_mode import get_dev_mode
                    dev_mode = get_dev_mode()
                except Exception as e:
                    # Best-effort: dev_mode check failed, continue with False
                    logger.debug(f"Could not check dev_mode: {e}, defaulting to False")
                    dev_mode = False
                
                if dev_mode:
                    # In dev_mode: log diagnostic summary and allow fallback
                    logger.warning("⚠️  [DEV_MODE] No targets selected after ranking. Allowing fallback to all discovered targets.")
                    if ranking_summary:
                        logger.warning(f"   Ranking summary: {ranking_summary['total_evaluated']} evaluated, "
                                     f"{ranking_summary['valid_for_ranking']} valid, "
                                     f"{ranking_summary['invalid_for_ranking']} invalid")
                        if ranking_summary['invalid_reasons_by_type']:
                            logger.warning("   Invalid reasons breakdown:")
                            # FIX ISSUE-006: Sort for determinism
                            for reason, target_list in sorted(ranking_summary['invalid_reasons_by_type'].items()):
                                logger.warning(f"     {reason}: {len(target_list)} targets ({', '.join(target_list[:5])}{'...' if len(target_list) > 5 else ''})")
                    
                    # Fallback: use all discovered targets
                    # FIX ISSUE-017: Add bounds check for empty symbols list
                    if not self.symbols:
                        error_msg = "Critical: No symbols available for target discovery fallback."
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    sample_symbol = self.symbols[0]
                    targets_dict = discover_targets(sample_symbol, self.data_dir)
                    # FIX ISSUE-003: Sort for determinism
                    targets = sorted(targets_dict.keys())[:top_n_targets]  # Limit to top_n, sorted for determinism
                    logger.warning(f"   [DEV_MODE] Using {len(targets)} discovered targets as fallback")
                else:
                    # In prod: raise error with diagnostic summary if available
                    error_msg = (
                        "No targets selected after ranking. This usually means:\n"
                        "  1. All targets have insufficient features (short-horizon targets need features with allowed_horizons)\n"
                        "  2. All targets were degenerate (single class, zero variance, extreme imbalance)\n"
                        "  3. All targets failed evaluation\n\n"
                    )
                    
                    if ranking_summary:
                        error_msg += f"Diagnostic summary:\n"
                        error_msg += f"  - Total evaluated: {ranking_summary['total_evaluated']}\n"
                        error_msg += f"  - Valid for ranking: {ranking_summary['valid_for_ranking']}\n"
                        error_msg += f"  - Invalid for ranking: {ranking_summary['invalid_for_ranking']}\n"
                        if ranking_summary['invalid_reasons_by_type']:
                            error_msg += f"  - Invalid reasons:\n"
                            # FIX ISSUE-006: Sort for determinism
                            for reason, target_list in sorted(ranking_summary['invalid_reasons_by_type'].items()):
                                error_msg += f"    * {reason}: {len(target_list)} targets\n"
                    
                    error_msg += (
                        "\nConsider:\n"
                        "  - Using targets with longer horizons (more features available)\n"
                        "  - Adding more features to CONFIG/feature_registry.yaml with shorter allowed_horizons\n"
                        "  - Checking CONFIG/excluded_features.yaml (may be too restrictive)\n"
                        "  - Using --no-auto-targets and providing manual --targets list"
                    )
                    
                    raise ValueError(error_msg)
        elif targets is None:
            # Fallback: discover all targets
            logger.info("Discovering all targets from data...")
            # FIX ISSUE-017: Add bounds check for empty symbols list
            if not self.symbols:
                error_msg = "Critical: No symbols available for target discovery."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            sample_symbol = self.symbols[0]
            targets_dict = discover_targets(sample_symbol, self.data_dir)
            # FIX ISSUE-003: Sort for determinism
            targets = sorted(targets_dict.keys())
            logger.info(f"Using all {len(targets)} discovered targets")
        
        logger.info(f"📋 Selected {len(targets)} targets: {', '.join(targets[:5])}{'...' if len(targets) > 5 else ''}")
        
        # Step 1.5: Apply training plan filter BEFORE feature selection (if available)
        # This avoids wasting time selecting features for targets that will be filtered out
        filtered_targets = targets
        filtered_symbols_by_target = {t: self.symbols for t in targets}
        training_plan = None
        training_plan_dir = None
        
        # Check if training plan exists (from previous run or generated earlier)
        # Check globals/ first (new structure), then METRICS/ as fallback (legacy)
        from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
        potential_plan_dir_globals = get_globals_dir(self.output_dir) / "training_plan"
        potential_plan_dir_legacy = self.output_dir / "METRICS" / "training_plan"
        if potential_plan_dir_globals.exists():
            training_plan_dir = potential_plan_dir_globals
        elif potential_plan_dir_legacy.exists():
            training_plan_dir = potential_plan_dir_legacy
        if training_plan_dir:
            try:
                from TRAINING.orchestration.training_plan_consumer import (
                    apply_training_plan_filter,
                    load_training_plan
                )
                training_plan = load_training_plan(training_plan_dir)
                if training_plan:
                    filtered_targets, filtered_symbols_by_target = apply_training_plan_filter(
                        targets=targets,
                        symbols=self.symbols,
                        training_plan_dir=training_plan_dir,
                        use_cs_plan=True,
                        use_symbol_plan=True
                    )
                    if len(filtered_targets) < len(targets):
                        logger.info(f"📋 Training plan filter applied BEFORE feature selection: {len(targets)} → {len(filtered_targets)} targets")
            except Exception as e:
                logger.debug(f"Could not apply training plan filter before feature selection: {e}, will filter after")
                filtered_targets = targets
        
        # Guard: Skip feature selection if no targets remain after filtering
        if not filtered_targets:
            logger.warning(
                "⚠️  All targets were filtered out before feature selection. "
                "Skipping feature selection phase."
            )
            if training_plan:
                logger.warning(
                    "   This may indicate an issue with the training plan or routing decisions. "
                    f"Training plan filtered {len(targets)} targets → 0 targets."
                )
            target_features = {}  # Empty dict - no features selected
        else:
            # Step 2: Feature selection (per target if auto_features)
            # CRITICAL: Use same view as target ranking for consistency
            # Only select features for filtered_targets to avoid waste
            # 
            # FEATURE STORAGE DOCUMENTATION:
            # Features are stored in multiple locations:
            # 1. Memory: target_features dict (built here, lines 1738-1867)
            #    - Structure: {target: [features]} for CROSS_SECTIONAL
            #    - Structure: {target: {symbol: [features]}} for SYMBOL_SPECIFIC
            #    - Structure: {target: {'cross_sectional': [...], 'symbol_specific': {...}}} for BOTH
            # 2. Disk: targets/{target}/reproducibility/selected_features.txt
            #    - Saved by TRAINING/ranking/feature_selection_reporting.py (line 338-342)
            #    - One feature per line, plain text format
            # 3. Routing decisions: targets/{target}/decision/routing_decision.json
            #    - Contains 'selected_features_path' field (line 275 in target_routing.py)
            # 4. Global summary: globals/selected_features_summary.json (created by _aggregate_feature_selection_summaries)
            #    - Aggregated view of all features per target per view for auditing
            # 
            # PASSING TO PHASE 3:
            # target_features dict is passed to train_models_for_interval_comprehensive via
            # target_features parameter (line 2173). Training function uses this to filter
            # features before training models.
            target_features = {}

            # ================================================================
            # RAW SEQUENCE MODE: Skip feature selection entirely
            # In raw_sequence mode, we feed raw OHLCV bars directly to models
            # without computing technical indicators or selecting features.
            # ================================================================
            # Skip feature selection when:
            # 1. RAW_SEQUENCE mode: Uses raw OHLCV sequences as input
            # 2. CS_RANKING mode: Uses raw OHLCV sequences with ranking loss
            # ================================================================
            input_mode = get_input_mode(experiment_config=self._exp_yaml_sst)
            cs_ranking_enabled = is_cs_ranking_enabled(self._exp_yaml_sst)

            # Check if we should skip feature selection
            skip_feature_selection = (
                input_mode == InputMode.RAW_SEQUENCE or cs_ranking_enabled
            )

            if skip_feature_selection:
                skip_reason = "Cross-Sectional Ranking Mode" if cs_ranking_enabled else "Raw Sequence Mode"
                logger.info("="*80)
                logger.info(f"STEP 2: SKIPPED - {skip_reason} (no feature selection)")
                logger.info("="*80)

                if cs_ranking_enabled:
                    logger.info(
                        "Cross-sectional ranking mode is enabled: skipping feature selection. "
                        "Models will receive raw OHLCV sequences with ranking-aligned targets."
                    )
                    # Get CS ranking config for logging
                    cs_config = get_cs_ranking_config(self._exp_yaml_sst)
                    logger.info(
                        f"CS Ranking config: target_type={cs_config['target'].get('type', 'cs_percentile')}, "
                        f"loss_type={cs_config['loss'].get('type', 'pairwise_logistic')}, "
                        f"batch_size={cs_config['batching'].get('timestamps_per_batch', 32)}"
                    )
                else:
                    logger.info(
                        f"Input mode is RAW_SEQUENCE: skipping feature selection. "
                        f"Models will receive raw OHLCV sequences directly."
                    )
                    # Get sequence config for logging
                    seq_config = get_raw_sequence_config(self._exp_yaml_sst)
                    logger.info(
                        f"Sequence config: length={seq_config['length_minutes']}min, "
                        f"channels={seq_config['channels']}, normalization={seq_config['normalization']}"
                    )

                # Mark all targets as having "raw" features (placeholder)
                # The actual sequence building happens in the training stage
                for target in filtered_targets:
                    if cs_ranking_enabled:
                        target_features[target] = ["__CS_RANKING_RAW_OHLCV__"]
                    else:
                        target_features[target] = ["__RAW_OHLCV_SEQUENCE__"]

                # Skip to next stage (no feature selection needed)
                emit_stage_change("ranking", "training")  # Skip feature_selection stage

            elif auto_features and features is None:
                logger.info("="*80)
                logger.info("STEP 2: Automatic Feature Selection")
                logger.info("="*80)
                emit_stage_change("ranking", "feature_selection")
                emit_progress(
                    "feature_selection", 0,
                    targets_total=len(filtered_targets) if filtered_targets else 0,
                    message="Starting feature selection"
                )

                # SST: Record stage transition BEFORE any stage work
                try:
                    from TRAINING.orchestration.utils.run_context import save_stage_transition
                    save_stage_transition(self.output_dir, "FEATURE_SELECTION", reason="Starting feature selection phase")
                except Exception as e:
                    logger.warning(f"Could not save FEATURE_SELECTION stage transition: {e}")
                
                # Load routing decisions to determine view per target
                routing_decisions = {}
                try:
                    from TRAINING.ranking.target_routing import load_routing_decisions
                    # load_routing_decisions now automatically checks new and legacy locations
                    # Note: filtered_targets may not be available yet at this stage, so skip fingerprint validation
                    routing_decisions = load_routing_decisions(
                        output_dir=self.output_dir,
                        expected_targets=None,  # Not available at feature selection stage
                        validate_fingerprint=False  # Skip validation at this stage
                    )
                    if routing_decisions:
                        # CRITICAL FIX: Log routing decision count and validate consistency
                        n_decisions = len(routing_decisions)
                        logger.info(f"Loaded routing decisions for {n_decisions} targets")
                        # Log summary of routes for debugging
                        route_counts = {}
                        # DC-006: Use sorted_items for deterministic iteration
                        for target, decision in sorted_items(routing_decisions):
                            route = decision.get('route', 'UNKNOWN')
                            route_counts[route] = route_counts.get(route, 0) + 1
                        # DC-006: Sort route_counts for deterministic logging
                        logger.debug(f"Routing decision summary: {dict(sorted(route_counts.items()))}")

                        # SB-006: Validate routing decisions BEFORE feature selection in strict mode
                        if filtered_targets:
                            from TRAINING.common.determinism import is_strict_mode
                            routing_targets = set(routing_decisions.keys())
                            filtered_targets_set = set(filtered_targets)
                            missing = filtered_targets_set - routing_targets
                            if missing:
                                if is_strict_mode():
                                    from TRAINING.common.exceptions import ConfigError
                                    raise ConfigError(
                                        f"SB-006: Missing routing decisions for {len(missing)} targets in strict mode: "
                                        f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}. "
                                        f"Run routing stage first or disable strict mode."
                                    )
                                logger.warning(
                                    f"⚠️ Missing routing decisions for {len(missing)} targets: "
                                    f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}, defaulting to CROSS_SECTIONAL"
                                )
                except Exception as e:
                    # SB-006: In strict mode, routing failures should raise
                    from TRAINING.common.determinism import is_strict_mode
                    if is_strict_mode():
                        from TRAINING.common.exceptions import ConfigError
                        raise ConfigError(f"SB-006: Could not load routing decisions in strict mode: {e}") from e
                    logger.debug(f"Could not load routing decisions: {e}, using CROSS_SECTIONAL for all targets")

                # Only select features for filtered targets (avoids waste)
                n_targets = len(filtered_targets)
                for target_idx, target in enumerate(filtered_targets):
                    # Emit progress for dashboard
                    progress_pct = (target_idx / n_targets) * 100 if n_targets > 0 else 0
                    emit_progress(
                        "feature_selection", progress_pct,
                        current_target=target,
                        targets_complete=target_idx,
                        targets_total=n_targets,
                        message=f"Selecting features for {target}"
                    )

                    # Determine view from routing decision (defaults to CROSS_SECTIONAL if not found)
                    route_info = routing_decisions.get(target, {})
                    route = route_info.get('route', View.CROSS_SECTIONAL.value)

                    # CRITICAL FIX: Check if cross-sectional is explicitly DISABLED in routing plan
                    cs_info = route_info.get('cross_sectional', {})
                    cs_route_status = cs_info.get('route', 'ENABLED') if isinstance(cs_info, dict) else 'ENABLED'
                    
                    # Handle different route types
                    if route == View.CROSS_SECTIONAL.value:
                        # CRITICAL FIX: Respect routing plan - skip CS feature selection if DISABLED
                        if cs_route_status == 'DISABLED':
                            logger.warning(
                                f"Skipping cross-sectional feature selection for {target}: "
                                f"CS route is DISABLED in routing plan (reason: {cs_info.get('reason', 'unknown')})"
                            )
                            # Don't select features for this target
                            continue
                        # Cross-sectional feature selection only
                        target_features[target] = self.select_features_auto(
                            target=target,
                            top_m=top_m_features,
                            use_cache=use_cache,
                            view=View.CROSS_SECTIONAL,
                            symbol=None
                        )
                    elif route == View.SYMBOL_SPECIFIC.value:
                        # Symbol-specific feature selection only
                        # CRITICAL FIX: winner_symbols should come from routing plan, but validate it's not empty
                        winner_symbols = route_info.get('winner_symbols', [])
                        
                        # If winner_symbols is empty or None, check if we should use all symbols
                        # This can happen if routing plan didn't populate winner_symbols correctly
                        if not winner_symbols or len(winner_symbols) == 0:
                            logger.warning(
                                f"⚠️ SYMBOL_SPECIFIC route for {target} has no winner_symbols in routing plan. "
                                f"Falling back to all symbols: {self.symbols}"
                            )
                            winner_symbols = self.symbols
                        else:
                            # Validate winner_symbols are actually in our symbol list
                            valid_symbols = [s for s in winner_symbols if s in self.symbols]
                            if len(valid_symbols) < len(winner_symbols):
                                invalid = set(winner_symbols) - set(self.symbols)
                                logger.warning(
                                    f"⚠️ SYMBOL_SPECIFIC route for {target} has invalid symbols in winner_symbols: {invalid}. "
                                    f"Using only valid symbols: {valid_symbols}"
                                )
                            winner_symbols = valid_symbols if valid_symbols else self.symbols
                        
                        if not winner_symbols:
                            logger.error(f"❌ No valid symbols for SYMBOL_SPECIFIC route for {target}, skipping")
                            continue

                        # DC-005: Sort winner_symbols for deterministic iteration order
                        winner_symbols = sorted(winner_symbols)
                        logger.info(f"📊 SYMBOL_SPECIFIC route for {target}: training {len(winner_symbols)} symbols: {winner_symbols}")
                        target_features[target] = {}
                        for symbol in winner_symbols:
                            target_features[target][symbol] = self.select_features_auto(
                                target=target,
                                top_m=top_m_features,
                                use_cache=use_cache,
                                view=View.SYMBOL_SPECIFIC,
                                symbol=symbol
                            )
                    elif route == 'BOTH':
                        # Both cross-sectional and symbol-specific
                        # Store in a structured format: {'cross_sectional': [...], 'symbol_specific': {symbol: [...]}}
                        cs_features = self.select_features_auto(
                            target=target,
                            top_m=top_m_features,
                            use_cache=use_cache,
                            view=View.CROSS_SECTIONAL,
                            symbol=None
                        )
                        winner_symbols = route_info.get('winner_symbols', self.symbols)
                        if not winner_symbols:
                            winner_symbols = self.symbols
                        # DC-005: Sort winner_symbols for deterministic iteration order
                        winner_symbols = sorted(winner_symbols)
                        symbol_features = {}
                        for symbol in winner_symbols:
                            symbol_features[symbol] = self.select_features_auto(
                                target=target,
                                top_m=top_m_features,
                                use_cache=use_cache,
                                view=View.SYMBOL_SPECIFIC,
                                symbol=symbol
                            )
                        target_features[target] = {
                            'cross_sectional': cs_features,
                            'symbol_specific': symbol_features,
                            'route': 'BOTH'
                        }
                    elif route == 'BLOCKED':
                        logger.warning(f"Skipping feature selection for {target} (BLOCKED: {route_info.get('reason', 'suspicious score')})")
                        # Don't select features for blocked targets - skip this target
                        target_features[target] = []  # Empty list to avoid KeyError downstream
            elif features:
                # Use same features for all filtered targets
                for target in filtered_targets:
                    target_features[target] = features
        
        # Aggregate feature selection summaries after all targets are processed
        if auto_features and target_features:
            try:
                self._aggregate_feature_selection_summaries()
            except Exception as e:
                logger.warning(f"Failed to aggregate feature selection summaries: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")

        # RI-003: Finalize run identity after feature selection completes
        # This is the critical step that was missing - now we have all feature signatures
        if auto_features and target_features:
            self._finalize_run_identity(target_features)

        # Generate metrics rollups after feature selection completes (all targets processed)
        if auto_features and target_features:
            try:
                from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
                tracker = ReproducibilityTracker(output_dir=self.output_dir)
                # RI-004/RI-005: Use _get_stable_run_id() for consistent run_id derivation
                try:
                    run_id = self._get_stable_run_id()
                except RuntimeError:
                    # Strict mode but identity not finalized - fallback to _run_name
                    run_id = self._run_name.replace("_", "-") if self._run_name and isinstance(self._run_name, str) else None
                if not run_id:
                    # Final fallback: Use self.run_identity (RI-005: was using 'trainer' instead of 'self')
                    run_identity = self.run_identity or self._partial_identity
                    try:
                        from TRAINING.orchestration.utils.manifest import derive_run_id_from_identity
                        run_id = derive_run_id_from_identity(
                            run_identity=run_identity
                        )
                    except Exception as e:
                        # CRITICAL: run_id affects artifacts - use centralized error handling policy
                        # Fallback to unstable run_id if identity derivation fails
                        from TRAINING.orchestration.utils.manifest import derive_unstable_run_id, generate_run_instance_id
                        run_id = derive_unstable_run_id(generate_run_instance_id())
                        from TRAINING.common.exceptions import handle_error_with_policy
                        # In deterministic mode, this will fail closed (raise)
                        # In best-effort mode, will log warning and use fallback
                        from datetime import datetime
                        fallback_run_id = datetime.now().isoformat()
                        run_id = handle_error_with_policy(
                            error=e,
                            stage="FEATURE_SELECTION",
                            error_type="run_id_derivation",
                            affects_artifact=True,
                            affects_manifest=True,
                            fallback_value=fallback_run_id,
                            logger_instance=logger
                        )
                tracker.generate_metrics_rollups(stage=Stage.FEATURE_SELECTION, run_id=run_id)
                logger.debug("✅ Generated metrics rollups for FEATURE_SELECTION")
            except Exception as e:
                logger.debug(f"Failed to generate metrics rollups for FEATURE_SELECTION: {e}")
        
        # Step 2.5: Generate training routing plan (if feature selection completed)
        # Note: training_plan_dir may already be set from Step 1.5
        if target_features and not training_plan_dir:
            try:
                from TRAINING.orchestration.routing_integration import generate_routing_plan_after_feature_selection
                
                # Extract training.model_families from cached SST YAML
                # Using self._exp_yaml_sst (cached at start of train_with_intelligence)
                exp_training = self._exp_yaml_sst.get("training", {}) if isinstance(self._exp_yaml_sst, dict) else {}
                cfg_families = exp_training.get("model_families")
                
                # SST precedence: config always wins when present
                if cfg_families:
                    logger.info(f"📋 Using training.model_families from config (SST): {cfg_families}")
                    train_families = cfg_families
                else:
                    logger.info(f"📋 training.model_families not set in config; using provided families: {families}")
                    train_families = families
                
                # Assert training families match config (if config provided)
                # Uses same YAML already loaded above to ensure consistency
                if exp_training and 'model_families' in exp_training:
                    expected = set(exp_training['model_families'])
                    actual = set(train_families) if train_families else set()
                    if expected != actual:
                        raise RuntimeError(
                            f"Training families mismatch: expected {sorted(expected)} from config, "
                            f"got {sorted(actual)}. This indicates a bug in family resolution."
                        )
                
                # Log families source for debugging
                logger.info(f"📋 Training plan generation: families parameter={train_families}")
                logger.info(f"📋 Training plan generation: families type={type(train_families)}, length={len(train_families) if isinstance(train_families, list) else 'N/A'}")
                if isinstance(train_families, list):
                    logger.info(f"📋 Training plan generation: families contents={train_families}")
                
                routing_plan = generate_routing_plan_after_feature_selection(
                    output_dir=self.output_dir,
                    targets=list(target_features.keys()),
                    symbols=self.symbols,
                    generate_training_plan=True,  # Generate training plan
                    model_families=train_families  # Use training families (SST)
                )
                logger.info(f"📋 Passed model_families={train_families} to routing_integration for training plan generation")
                if routing_plan:
                    logger.info("✅ Training routing plan generated - see globals/routing_plan/ for details (legacy: METRICS/routing_plan/)")
                    # Set training plan directory for filtering
                    from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
                    training_plan_dir = get_globals_dir(self.output_dir) / "training_plan"
                    
                    # FIX: Verify plans were actually saved
                    routing_plan_path = get_globals_dir(self.output_dir) / "routing_plan" / "routing_plan.json"
                    training_plan_path = training_plan_dir / "master_training_plan.json"
                    if not routing_plan_path.exists():
                        logger.warning(f"⚠️ Routing plan was generated but file not found at {routing_plan_path}")
                    if not training_plan_path.exists() and not (training_plan_dir / "training_plan.json").exists():
                        logger.warning(f"⚠️ Training plan was generated but file not found at {training_plan_path}")
                    
                    # FIX: Update manifest with plan hashes after plans are created
                    try:
                        from TRAINING.orchestration.utils.manifest import update_manifest_with_plan_hashes
                        update_manifest_with_plan_hashes(self.output_dir)
                    except Exception as e:
                        # CRITICAL: manifest update affects artifacts - use centralized error handling policy
                        from TRAINING.common.exceptions import handle_error_with_policy
                        handle_error_with_policy(
                            error=e,
                            stage="TRAINING",
                            error_type="manifest_update",
                            affects_artifact=True,
                            affects_manifest=True,
                            logger_instance=logger
                        )
                else:
                    logger.warning("⚠️ Routing plan generation returned None - plans may not have been created")
            except Exception as e:
                # FIX: Log at WARNING level so failures are visible (not hidden at debug level)
                logger.warning(f"⚠️ Failed to generate routing plan: {e}")
                logger.debug(f"Routing plan generation traceback:", exc_info=True)
        
        # Write auto-enable audit payload (after feature selection, early in run)
        # CRITICAL FIX #1: Use module-level tracking (works with per-target instances)
        try:
            from TRAINING.common.feature_registry import get_registry
            from TRAINING.orchestration.utils.target_first_paths import run_root, get_globals_dir
            from TRAINING.common.utils.file_utils import safe_json_dump
            
            # CRITICAL FIX #1: Get any registry instance (tracking is module-level, shared across all instances)
            # All registry instances share the same _auto_enabled_features dict (references _AUTO_ENABLED_FEATURES_GLOBAL)
            # This ensures tracking works even when get_registry() creates new instances per-target
            registry = get_registry()
            if registry and hasattr(registry, 'get_auto_enable_audit'):
                audit = registry.get_auto_enable_audit()
                
                if audit['n_features_enabled'] > 0:
                    # Log summary
                    sample_features = [f['feature_name'] for f in audit['enabled_features'][:5]]
                    sample_str = ', '.join(sample_features)
                    if len(audit['enabled_features']) > 5:
                        sample_str += f", ... ({len(audit['enabled_features']) - 5} more)"
                    
                    logger.info(
                        f"Auto-enabled {audit['n_features_enabled']} features with empty allowed_horizons via family defaults "
                        f"(flag enabled). Sample: [{sample_str}] "
                        f"(see globals/registry_effective_auto_enabled_features.json for full list)"
                    )
                    
                    # Threshold warning
                    if audit['threshold_warning_triggered']:
                        logger.warning(
                            f"Registry contains unusually many disabled-but-healable features: "
                            f"{audit['n_features_enabled']} auto-enabled ({audit['enabled_percent']:.1f}% of registry). "
                            f"Consider cleaning registry by setting allowed_horizons: null for these features."
                        )
                    
                    # Write audit payload
                    run_root_dir = run_root(self.output_dir)
                    globals_dir = get_globals_dir(run_root_dir)
                    globals_dir.mkdir(parents=True, exist_ok=True)
                    audit_path = globals_dir / "registry_effective_auto_enabled_features.json"
                    
                    # SST: Use write_atomic_json for atomic write with canonical serialization
                    from TRAINING.common.utils.file_utils import write_atomic_json
                    write_atomic_json(audit_path, audit)
                    
                    logger.debug(f"✅ Auto-enable audit payload written to: {audit_path}")
        except Exception as e:
            # CRITICAL FIX #5: Log at WARNING level so failures are visible (not hidden at debug)
            logger.warning(f"Could not write auto-enable audit payload: {e}")
            logger.debug(f"Auto-enable audit payload write traceback:", exc_info=True)
        
        # Step 3: Training
        logger.info("="*80)
        logger.info("STEP 3: Model Training")
        logger.info("="*80)
        
        # Apply training plan filter if available (may have been applied earlier)
        # If not already filtered, try to filter now
        if training_plan_dir is None:
            # Try to find training plan directory - check globals/ first, then METRICS/ as fallback
            from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
            potential_plan_dir_globals = get_globals_dir(self.output_dir) / "training_plan"
            potential_plan_dir_legacy = self.output_dir / "METRICS" / "training_plan"
            if potential_plan_dir_globals.exists():
                training_plan_dir = potential_plan_dir_globals
            elif potential_plan_dir_legacy.exists():
                training_plan_dir = potential_plan_dir_legacy
        
        if training_plan_dir:
            try:
                # Validate training_plan_dir
                training_plan_dir = Path(training_plan_dir)
                if not training_plan_dir.exists():
                    logger.debug(f"Training plan directory does not exist: {training_plan_dir}")
                elif training_plan_dir.exists():
                    from TRAINING.orchestration.training_plan_consumer import (
                        apply_training_plan_filter,
                        load_training_plan,
                        get_model_families_for_job
                    )
                    
                    # Load training plan with error handling
                    try:
                        training_plan = load_training_plan(training_plan_dir)
                    except Exception as e:
                        logger.warning(f"Failed to load training plan: {e}, proceeding without filtering")
                        training_plan = None
                    
                    if training_plan:
                        # Fail-fast check: abort if training plan has 0 jobs unless dev_mode
                        # CRITICAL: This must not be caught by broad exception handlers
                        jobs = training_plan.get("jobs", [])
                        total_jobs = training_plan.get("metadata", {}).get("total_jobs", len(jobs))
                        if total_jobs == 0:
                            # FIX ISSUE-004: Use centralized dev_mode helper instead of direct get_cfg
                            dev_mode = False
                            try:
                                from CONFIG.dev_mode import get_dev_mode
                                dev_mode = get_dev_mode()
                            except Exception:
                                pass
                            
                            # Load view for diagnostics
                            run_view = "UNKNOWN"
                            try:
                                from TRAINING.orchestration.utils.run_context import load_run_context
                                context = load_run_context(self.output_dir)
                                if context:
                                    run_view = context.get("view", "UNKNOWN")
                            except Exception as e:
                                # Diagnostic: run_context load failed, continue without view
                                logger.debug(f"Could not load run_context for view: {e}")

                            if not dev_mode:
                                raise ValueError(
                                    f"FATAL: Training plan has 0 jobs. Routing diagnostics: "
                                    f"view={run_view}, targets_checked={len(targets)}, "
                                    f"symbols={len(self.symbols) if self.symbols else 'N/A'}. "
                                    f"Check globals/routing/routing_candidates.json for details."
                                )
                            else:
                                logger.warning(
                                    f"⚠️ Training plan has 0 jobs (dev_mode=true). "
                                    f"view={run_view}, targets={len(targets)}. "
                                    f"Using fallback families from metadata/SST."
                                )
                        
                        try:
                            filtered_targets, filtered_symbols_by_target = apply_training_plan_filter(
                                targets=targets,
                                symbols=self.symbols,
                                training_plan_dir=training_plan_dir,
                                use_cs_plan=True,
                                use_symbol_plan=True
                            )
                            
                            # Validate filtered results
                            if not isinstance(filtered_targets, list):
                                logger.warning(f"apply_training_plan_filter returned invalid filtered_targets: {type(filtered_targets)}, using original")
                                filtered_targets = targets
                            
                            if not isinstance(filtered_symbols_by_target, dict):
                                logger.warning(f"apply_training_plan_filter returned invalid filtered_symbols_by_target: {type(filtered_symbols_by_target)}, using default")
                                filtered_symbols_by_target = {t: self.symbols for t in filtered_targets}
                            
                            if len(filtered_targets) < len(targets):
                                logger.info(f"📋 Training plan filter applied: {len(targets)} → {len(filtered_targets)} targets")
                            
                            # Log symbol filtering per target - safely
                            for target in filtered_targets[:10]:  # Limit logging to first 10
                                try:
                                    filtered_symbols = filtered_symbols_by_target.get(target, self.symbols)
                                    if isinstance(filtered_symbols, list) and len(filtered_symbols) < len(self.symbols):
                                        logger.info(f"📋 Filtered symbols for {target}: {len(self.symbols)} → {len(filtered_symbols)} symbols")
                                except Exception as e:
                                    logger.debug(f"Error logging symbol filter for {target}: {e}")
                            
                            # Update target_features to only include filtered targets - safely
                            # Also exclude BLOCKED targets (they have empty feature lists)
                            if isinstance(target_features, dict):
                                try:
                                    filtered_target_features = {}
                                    # DC-004: Use sorted_items for deterministic iteration
                                    for t, f in sorted_items(target_features):
                                        if t in filtered_targets:
                                            # Skip BLOCKED targets (empty list) and targets with no features
                                            if isinstance(f, list) and len(f) == 0:
                                                logger.debug(f"Skipping {t} from target_features (BLOCKED or no features)")
                                                continue
                                            filtered_target_features[t] = f
                                    target_features = filtered_target_features
                                except Exception as e:
                                    logger.warning(f"Failed to filter target_features: {e}, keeping original")
                        except ValueError:
                            # Re-raise ValueError (fail-fast for 0 jobs)
                            raise
                        except Exception as e:
                            logger.warning(f"Failed to apply training plan filter: {e}, using all targets")
                            filtered_targets = targets
                            filtered_symbols_by_target = {t: self.symbols for t in targets}
            except ValueError:
                # Re-raise ValueError (fail-fast for 0 jobs, critical errors)
                raise
            except Exception as e:
                logger.warning(f"Failed to apply training plan filter: {e}", exc_info=True)
        
        # =====================================================================
        # Data Loading: Eager (all upfront) or Lazy (per-target)
        # =====================================================================
        # Check if lazy loading is enabled
        # CRITICAL: Must check experiment config first (has precedence over base config)
        lazy_loading_config = {}
        lazy_loading_enabled = False

        # First try experiment config (has precedence)
        if self.experiment_config:
            exp_name = self.experiment_config.name if hasattr(self.experiment_config, 'name') else str(self.experiment_config)
            try:
                exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                if exp_file.exists():
                    import yaml
                    with open(exp_file, 'r') as f:
                        exp_yaml = yaml.safe_load(f) or {}
                    lazy_cfg = exp_yaml.get('intelligent_training', {}).get('lazy_loading', {})
                    if lazy_cfg:
                        lazy_loading_config = lazy_cfg
                        lazy_loading_enabled = lazy_cfg.get('enabled', False)
                        logger.info(f"🔍 Lazy loading config loaded from experiment: {exp_name}")
            except Exception as e:
                logger.debug(f"Could not load lazy loading config from experiment: {e}")

        # Fallback to base config if not set by experiment
        if not lazy_loading_enabled:
            lazy_loading_config = get_cfg("intelligent_training.lazy_loading", default={})
            lazy_loading_enabled = lazy_loading_config.get('enabled', False)

        mtf_data = None
        data_loader = None

        # OPTIMIZATION: Always use lazy loading with column projection for Stage 3
        # This ensures we only load the features selected in Stage 2, not all 531 columns
        # The deprecated load_mtf_data() eager path has been removed

        # Get interval from data path or config
        interval_minutes = get_cfg("pipeline.data.interval_minutes", default=5)
        interval_str = f"{interval_minutes}m"

        data_loader = UnifiedDataLoader(
            data_dir=self.data_dir,
            interval=interval_str,
        )

        # Verify symbols exist via schema read (fast, ~1ms/symbol)
        schemas = data_loader.read_schema(self.symbols)
        if not schemas:
            raise ValueError(f"Failed to read schema for any symbols: {self.symbols}")

        # Always use lazy loading for memory efficiency
        # Even if lazy_loading.enabled=false in config, we use lazy loading for Stage 3
        # because we have the selected features and can do column projection
        if lazy_loading_enabled:
            logger.info(f"🔋 LAZY LOADING ENABLED: Data will be loaded per-target with column projection")
            logger.info(f"   Config: verify_memory_release={lazy_loading_config.get('verify_memory_release', False)}, "
                       f"log_memory_usage={lazy_loading_config.get('log_memory_usage', True)}")
        else:
            # Force lazy loading even if not explicitly enabled
            # This is safe because we have target_features for column projection
            lazy_loading_enabled = True
            lazy_loading_config = lazy_loading_config or {}
            lazy_loading_config['enabled'] = True
            logger.info(f"🔋 LAZY LOADING FORCED: Data will be loaded per-target with column projection")
            logger.info(f"   (Stage 3 always uses lazy loading for memory efficiency)")

        logger.info(f"✅ Schema validated for {len(schemas)}/{len(self.symbols)} symbols (lazy loading ready)")
        
        # Prepare training parameters
        interval = 'cross_sectional'  # Use cross-sectional training
        
        # =====================================================================
        # Phase 3: Extract model families with correct precedence
        # Precedence: job families → plan metadata → SST training → param → fail
        # =====================================================================
        from TRAINING.common.utils.sst_contract import filter_trainers, FEATURE_SELECTORS
        
        # Log parameter for debugging (note: this may be overridden by SST config below)
        if families:
            logger.debug(f"📋 Training phase: families parameter provided={families} (may be overridden by SST config)")
        
        # Get SST training.model_families from cached YAML
        sst_training_families = None
        if self._exp_yaml_sst:
            exp_training = self._exp_yaml_sst.get("training", {})
            sst_training_families = exp_training.get("model_families")
            if sst_training_families:
                sst_training_families = filter_trainers(sst_training_families)
                logger.info(f"📋 SST training.model_families={sst_training_families}")
        
        # Normalize parameter families if provided
        param_families = filter_trainers(families) if families else None
        
        # Build per-target families map via consumer (handles job → metadata fallback)
        target_families_map = {}
        if training_plan and filtered_targets:
            for target in filtered_targets:
                if not isinstance(target, str) or not target:
                    continue
                
                try:
                    # Consumer handles: job families → plan metadata fallback
                    plan_families = get_model_families_for_job(
                        training_plan,
                        target=target,
                        symbol=None,
                        training_type="cross_sectional"
                    )
                    
                    if plan_families and isinstance(plan_families, list):
                        target_families_map[target] = plan_families
                        logger.debug(f"📋 Target {target}: families from plan = {plan_families}")
                except Exception as e:
                    logger.warning(f"Failed to get model families for {target}: {e}")
                    continue
        
        # Resolve families with correct precedence
        # NOTE: Using union across targets. If per-target routing is supported,
        # pass target_families_map to executor instead of global families_list.
        # Union is the safe compromise when global list is required.
        
        if target_families_map:
            # Use UNION of all target families (not intersection - don't drop families)
            all_families = set()
            for target_fams in target_families_map.values():
                all_families.update(target_fams)
            families_list = sorted(all_families)
            logger.info(f"📋 Using model families from training plan (union of {len(target_families_map)} targets): {families_list}")
        elif sst_training_families:
            # Precedence 3: SST training.model_families
            families_list = sst_training_families
            logger.info(f"📋 No plan families, using SST training.model_families: {families_list}")
        elif param_families:
            # Precedence 4: Parameter families (fallback)
            families_list = param_families
            logger.info(f"📋 No plan/SST families, using parameter families: {families_list}")
        else:
            # Precedence 5: Fail (configuration error - no silent ALL_FAMILIES fallback)
            raise ValueError(
                "No model families available: training plan has no jobs/metadata, "
                "no SST training.model_families, and no families parameter provided. "
                "This is a configuration error. Check your experiment config."
            )
        
        # Validate not empty
        if not families_list:
            raise ValueError("families_list is empty after resolution. Configuration error.")
        
        # Log final families list for debugging
        logger.info(f"📋 Final families_list for training: {families_list} (count: {len(families_list)})")
        
        # Invariant: families_list must not contain feature selectors (defensive check)
        selector_violations = set(families_list) & FEATURE_SELECTORS
        if selector_violations:
            raise RuntimeError(
                f"🚨 INVARIANT VIOLATION: families_list contains feature selectors: {selector_violations}. "
                f"These should have been filtered. Full list: {families_list}"
            )
        
        # Invariant: families_list must only contain config-enabled trainers (if SST training families provided)
        # NOTE: Validate against sst_training_families (training.model_families), NOT the families parameter
        # (which comes from feature_selection.model_families - a different config section)
        if sst_training_families is not None:
            from TRAINING.common.utils.sst_contract import normalize_family
            
            # Normalize both sides using single-source-of-truth function (includes alias resolution)
            sst_set = {normalize_family(f) for f in sst_training_families}
            plan_set = {normalize_family(f) for f in families_list}
            
            extra_families = plan_set - sst_set
            if extra_families:
                raise RuntimeError(
                    f"🚨 INVARIANT VIOLATION: families not in training.model_families: {extra_families}. "
                    f"SST training families (normalized): {sorted(sst_set)}, "
                    f"Plan families (normalized): {sorted(plan_set)}"
                )
        
        # Validate families_list is not empty
        if not families_list:
            logger.warning("⚠️ No model families available after filtering! Using default families.")
            fallback_families = ALL_FAMILIES if not families else families
            families_list = [f for f in fallback_families if f not in FEATURE_SELECTORS]

        # ================================================================
        # RAW SEQUENCE MODE: Filter to sequence-compatible families only
        # In raw_sequence mode, only LSTM, Transformer, CNN1D, etc. can be used
        # ================================================================
        input_mode = get_input_mode(experiment_config=self._exp_yaml_sst)
        if input_mode == InputMode.RAW_SEQUENCE:
            original_count = len(families_list)
            families_list = filter_families_for_input_mode(families_list, input_mode)
            if len(families_list) < original_count:
                logger.info(
                    f"RAW_SEQUENCE mode: Filtered families from {original_count} to {len(families_list)}: {families_list}"
                )
            if not families_list:
                logger.error(
                    f"❌ RAW_SEQUENCE mode: No compatible model families! "
                    f"Sequence-compatible families are: LSTM, Transformer, CNN1D, TabLSTM, TabTransformer, TabCNN"
                )
                return {
                    "status": "failed",
                    "reason": "No sequence-compatible model families for raw_sequence mode"
                }

        # Validate filtered_targets is not empty
        if not filtered_targets:
            logger.warning("⚠️ All targets were filtered out by training plan! Training will be skipped.")
            logger.warning("   This may indicate an issue with the training plan or routing decisions.")
        
        # Use run root directly (no training_results/ subdirectory)
        output_dir_str = str(self.output_dir)
        
        # Get training parameters from kwargs or config (min_cs and max_cs_samples already extracted above)
        max_rows_train = train_kwargs.get('max_rows_train')
        
        # Early return if no targets to train
        if not filtered_targets:
            logger.error("❌ No targets to train after filtering. Exiting.")
            return {
                "status": "skipped",
                "reason": "All targets filtered out by training plan",
                "targets_requested": len(targets),
                "targets_filtered": 0
            }
        
        logger.info(f"Training {len(filtered_targets)} targets with strategy '{strategy}'")
        logger.info(f"Model families: {len(families_list)} families")
        if target_features:
            logger.info(f"Using selected features per target (top {top_m_features} per target)")
            # Log feature counts per target (handle different structures)
            # DC-004: Use sorted for deterministic logging order
            for target, feat_data in sorted(target_features.items())[:3]:
                if isinstance(feat_data, list):
                    logger.info(f"  {target}: {len(feat_data)} features (CROSS_SECTIONAL)")
                elif isinstance(feat_data, dict):
                    if 'cross_sectional' in feat_data and 'symbol_specific' in feat_data:
                        # BOTH route
                        cs_count = len(feat_data['cross_sectional']) if isinstance(feat_data['cross_sectional'], list) else 0
                        sym_count = len(feat_data['symbol_specific']) if isinstance(feat_data['symbol_specific'], dict) else 0
                        logger.info(f"  {target}: {cs_count} CS features + {sym_count} symbol-specific sets (BOTH)")
                    else:
                        # SYMBOL_SPECIFIC route
                        sym_count = len(feat_data) if isinstance(feat_data, dict) else 0
                        total_feat_count = sum(len(v) if isinstance(v, list) else 0 for v in feat_data.values()) if isinstance(feat_data, dict) else 0
                        logger.info(f"  {target}: {total_feat_count} features across {sym_count} symbols (SYMBOL_SPECIFIC)")
                else:
                    logger.info(f"  {target}: {type(feat_data).__name__} structure")
            if len(target_features) > 3:
                logger.info(f"  ... and {len(target_features) - 3} more targets")
        
        # Pass selected features and routing decisions to training pipeline
        # If target_features is empty, training will auto-discover features
        features_to_use = target_features if target_features else None
        
        # Pass routing decisions to training so it knows which view to use
        routing_decisions_for_training = {}
        try:
            from TRAINING.ranking.target_routing import load_routing_decisions
            # load_routing_decisions now automatically checks new and legacy locations
            # Pass filtered_targets for fingerprint validation
            routing_decisions_for_training = load_routing_decisions(
                output_dir=self.output_dir,
                expected_targets=filtered_targets if 'filtered_targets' in locals() and filtered_targets else None,
                validate_fingerprint=True
            )
            if routing_decisions_for_training:
                # CRITICAL FIX: Log routing decision count and validate consistency
                n_decisions = len(routing_decisions_for_training)
                logger.info(f"Loaded routing decisions for training: {n_decisions} targets")
                # Validate that routing decisions match filtered targets
                if 'filtered_targets' in locals() and filtered_targets:
                    routing_targets = set(routing_decisions_for_training.keys())
                    filtered_targets_set = set(filtered_targets)
                    
                    if routing_targets != filtered_targets_set:
                        missing = filtered_targets_set - routing_targets
                        extra = routing_targets - filtered_targets_set
                        if missing:
                            logger.warning(
                                f"⚠️ Routing decisions missing for {len(missing)} targets: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
                            )
                        if extra:
                            logger.warning(
                                f"⚠️ Routing decisions contain {len(extra)} unexpected targets: {sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}. "
                                f"These may be from a previous run."
                            )
                    else:
                        logger.debug(f"✅ Routing decisions match filtered targets: {n_decisions} targets")
        except Exception as e:
            logger.debug(f"Could not load routing decisions for training: {e}")

        # SB-007: Validate registry was not mutated between stages
        if hasattr(self, '_registry_hash_at_load') and self._registry_hash_at_load is not None:
            try:
                from TRAINING.common.feature_registry import get_registry
                current_registry = get_registry()
                if current_registry is not None and hasattr(current_registry, 'features'):
                    current_hash = hash(frozenset(current_registry.features.keys()))
                    if current_hash != self._registry_hash_at_load:
                        from TRAINING.common.determinism import is_strict_mode
                        from TRAINING.common.exceptions import RegistryLoadError
                        msg = (
                            f"SB-007: Registry was mutated between stages! "
                            f"Original hash: {self._registry_hash_at_load}, current hash: {current_hash}. "
                            f"This can cause non-deterministic behavior."
                        )
                        if is_strict_mode():
                            raise RegistryLoadError(
                                message=msg,
                                stage="TRAINING",
                                error_code="REGISTRY_MUTATION_DETECTED"
                            )
                        logger.warning(msg)
                    else:
                        logger.debug(f"SB-007: Registry hash unchanged: {current_hash}")
            except ImportError:
                pass  # Registry not available, skip check

        # Call the training function
        logger.info("Starting model training...")
        emit_stage_change("feature_selection", "training")
        emit_progress(
            "training", 0,
            targets_total=len(filtered_targets) if filtered_targets else 0,
            message="Starting model training"
        )

        # SST: Record stage transition BEFORE any stage work
        try:
            from TRAINING.orchestration.utils.run_context import save_stage_transition
            save_stage_transition(self.output_dir, "TRAINING", reason="Starting model training phase")
        except Exception as e:
            logger.warning(f"Could not save TRAINING stage transition: {e}")
        
        # SST: Create partial identity for TRAINING stage (mirrors FEATURE_SELECTION pattern)
        training_identity = None
        try:
            from TRAINING.common.utils.fingerprinting import create_stage_identity
            training_identity = create_stage_identity(
                stage=Stage.TRAINING,
                symbols=self.symbols,  # Full universe
                experiment_config=self.experiment_config,
                data_dir=self.data_dir,
            )
            logger.debug(f"Created TRAINING identity with train_seed={training_identity.train_seed}")
        except Exception as e:
            logger.debug(f"Failed to create TRAINING identity: {e}")
        
        # Prepare CS ranking config if enabled
        cs_ranking_config_for_training = None
        if is_cs_ranking_enabled(self._exp_yaml_sst):
            cs_ranking_config_for_training = get_cs_ranking_config(self._exp_yaml_sst)
            logger.info(
                f"🎯 Cross-sectional ranking mode enabled: "
                f"loss={cs_ranking_config_for_training['loss'].get('type', 'pairwise_logistic')}, "
                f"target={cs_ranking_config_for_training['target'].get('type', 'cs_percentile')}"
            )

        training_results = train_models_for_interval_comprehensive(
            interval=interval,
            targets=filtered_targets,  # Use filtered targets
            mtf_data=mtf_data,  # None if lazy loading enabled
            families=families_list,
            strategy=strategy,
            output_dir=output_dir_str,
            min_cs=min_cs,
            max_cs_samples=max_cs_samples,
            max_rows_train=max_rows_train,
            target_features=features_to_use,
            target_families=target_families_map if target_families_map else None,  # Per-target families from plan
            routing_decisions=routing_decisions_for_training,  # Pass routing decisions
            run_identity=training_identity,  # SST: Pass identity for reproducibility tracking
            experiment_config=self._exp_yaml_sst,  # SST: Raw YAML dict (not ExperimentConfig dataclass) so downstream get_input_mode/get_raw_sequence_config can detect pipeline.input_mode
            # Lazy loading parameters (Phase 4 memory optimization)
            data_loader=data_loader,  # UnifiedDataLoader instance (or None if eager)
            symbols=self.symbols if lazy_loading_enabled else None,  # Required for lazy loading
            lazy_loading_config=lazy_loading_config if lazy_loading_enabled else None,
            # Cross-sectional ranking parameters (Phase 5)
            cs_ranking_config=cs_ranking_config_for_training,  # None if CS ranking disabled
        )
        
        logger.info("="*80)
        
        # Count trained models
        total_models = sum(
            len(target_results) 
            for target_results in training_results.get('models', {}).values()
        )
        
        # CRITICAL: Fail loudly if 0 models were trained
        # training_results is the dict returned from train_models_for_interval_comprehensive
        # It should have 'models', 'failed_targets', 'failed_reasons' keys
        failed_targets = training_results.get('failed_targets', [])
        failed_reasons = training_results.get('failed_reasons', {})
        
        # If not found at top level, check if it's nested in a 'results' key
        if not failed_targets and 'results' in training_results:
            failed_targets = training_results['results'].get('failed_targets', [])
            failed_reasons = training_results['results'].get('failed_reasons', {})
        
        if total_models == 0:
            logger.error("="*80)
            logger.error("❌ TRAINING RUN FAILED: 0 models trained across %d targets", len(targets))
            logger.error("="*80)
            logger.error("Failed targets: %d / %d", len(failed_targets), len(targets))
            if failed_targets:
                logger.error("Failed target list: %s", failed_targets[:10])
                # Log most common failure reason
                if failed_reasons:
                    reason_counts = {}
                    for reason in failed_reasons.values():
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    most_common = max(reason_counts.items(), key=lambda x: x[1])
                    logger.error("Most common failure reason: %s (%d targets)", most_common[0], most_common[1])
            logger.error("="*80)
            logger.error("This indicates a critical data preparation issue.")
            logger.error("Check logs above for 'all-NaN feature columns' or 'No valid data after cleaning' messages.")
            logger.error("="*80)
            status = 'failed_no_models'
        else:
            logger.info("✅ Training completed successfully")
            logger.info("="*80)
            status = 'completed'
        
        logger.info(f"Trained {total_models} models across {len(targets)} targets")
        if failed_targets:
            logger.warning(f"⚠️ {len(failed_targets)} targets failed data preparation and were skipped")
        
        # Final status summary
        if status == 'failed_no_models':
            logger.error("="*80)
            logger.error("❌ TRAINING PIPELINE FAILED - NO MODELS TRAINED")
            logger.error("="*80)
            logger.error("Action required: Check diagnostic logs above to identify why all features became NaN")
            logger.error("="*80)
        
        # Create run-level confidence summary
        try:
            from TRAINING.orchestration.target_routing import collect_run_level_confidence_summary
            # NOTE: load_multi_model_config already imported at module level (line 63)

            # Get routing config
            multi_model_config = load_multi_model_config()
            routing_config = None
            if multi_model_config:
                routing_config = multi_model_config.get('confidence', {}).get('routing', {})
            
            # Look for feature selection results in target-first structure (targets/<target>/reproducibility/)
            # MetricsAggregator will check target-first structure first, then fall back to legacy
            feature_selections_dir = self.output_dir
            if feature_selections_dir.exists():
                all_confidence = collect_run_level_confidence_summary(
                    feature_selections_dir=feature_selections_dir,
                    output_dir=self.output_dir,
                    routing_config=routing_config
                )
                
                if all_confidence:
                    # Log summary stats
                    high_conf = sum(1 for c in all_confidence if c.get('confidence') == 'HIGH')
                    medium_conf = sum(1 for c in all_confidence if c.get('confidence') == 'MEDIUM')
                    low_conf = sum(1 for c in all_confidence if c.get('confidence') == 'LOW')
                    logger.info(f"📊 Confidence summary: {high_conf} HIGH, {medium_conf} MEDIUM, {low_conf} LOW")
        except Exception as e:
            logger.debug(f"Failed to create run-level confidence summary: {e}")
        
        # Run leakage diagnostics if enabled
        sentinel_results = {}
        if run_leakage_diagnostics:
            logger.info("="*80)
            logger.info("STEP 4: Leakage Diagnostics (Sentinels)")
            logger.info("="*80)
            try:
                sentinel_results = self._run_leakage_diagnostics(
                    training_results, targets, mtf_data, train_kwargs
                )
            except Exception as e:
                logger.warning(f"Leakage diagnostics failed: {e}")
                sentinel_results = {'error': str(e)}
        
        # Generate trend summary (if reproducibility tracking is available)
        trend_summary = None
        try:
            from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
            # Path is already imported at top of file
            # Check for target-first structure (targets/ and globals/) first
            has_target_first = (self.output_dir / "targets").exists() or (self.output_dir / "globals").exists()
            has_legacy = (self.output_dir / "REPRODUCIBILITY").exists()
            
            if not has_target_first:
                # Try alternative location (backward compatibility for old structure)
                # Only check parent if output_dir is a module subdirectory
                if self.output_dir.name in ["target_rankings", "feature_selections", "training_results"]:
                    has_legacy = (self.output_dir.parent / "REPRODUCIBILITY").exists()
            
            if has_target_first or has_legacy:
                # Create tracker to access trend summary method
                tracker = ReproducibilityTracker(output_dir=self.output_dir)
                trend_summary = tracker.generate_trend_summary(view="STRICT", min_runs_for_trend=2)
                
                if trend_summary.get("status") == "ok":
                    logger.info("="*80)
                    logger.info("TREND ANALYSIS SUMMARY")
                    logger.info("="*80)
                    logger.info(f"Series analyzed: {trend_summary.get('n_series', 0)}")
                    logger.info(f"Trends computed: {trend_summary.get('n_trends', 0)}")
                    
                    if trend_summary.get("declining_trends"):
                        logger.warning(f"⚠️  {len(trend_summary['declining_trends'])} declining trends detected")
                        for decl in trend_summary["declining_trends"][:5]:
                            logger.warning(f"  - {decl['metric']}: slope={decl['slope']:.6f}/day ({decl['series'][:50]}...)")
                    
                    if trend_summary.get("alerts"):
                        logger.info(f"ℹ️  {len(trend_summary['alerts'])} trend alerts")
                        for alert in trend_summary["alerts"][:3]:  # Show first 3
                            severity_icon = "⚠️" if alert.get('severity') == 'warning' else "ℹ️"
                            logger.info(f"  {severity_icon} {alert.get('message', '')[:100]}")
                    
                    logger.info("="*80)
        except Exception as e:
            logger.debug(f"Could not generate trend summary: {e}")
            # Don't fail if trend analysis fails
        
        # Create per-target metadata files and update final manifest
        try:
            from TRAINING.orchestration.utils.manifest import create_target_metadata, update_manifest
            for target in targets:
                try:
                    create_target_metadata(self.output_dir, target)
                except Exception as e:
                    logger.debug(f"Failed to create metadata for target {target}: {e}")
            
            # Update manifest with final targets list
            try:
                # Use cached SST YAML for manifest (consistency)
                experiment_config_dict = self._exp_yaml_sst if hasattr(self, '_exp_yaml_sst') and self._exp_yaml_sst else None
                
                # Update manifest with final information
                from TRAINING.orchestration.utils.manifest import create_manifest
                # RI-005: Get run_identity if available, fallback to partial identity
                run_identity = self.run_identity or self._partial_identity
                create_manifest(
                    self.output_dir,
                    run_id=None,  # Will be derived from run_identity if available
                    targets=targets,
                    experiment_config=experiment_config_dict,
                    run_metadata={
                        "data_dir": str(self.data_dir) if self.data_dir else None,
                        "symbols": self.symbols if self.symbols else None,
                        "n_effective": getattr(self, '_n_effective', None)
                    },
                    run_identity=run_identity,
                    run_instance_id=self.output_dir.name  # Directory name is the run instance ID
                )
            except Exception as e:
                logger.debug(f"Failed to update final manifest: {e}")
        except Exception as e:
            logger.debug(f"Failed to create target metadata files: {e}")
        
        # PERFORMANCE AUDIT: Save audit report at end of run
        try:
            from TRAINING.common.utils.performance_audit import get_auditor
            auditor = get_auditor()
            if auditor.enabled and auditor.calls:
                audit_report_path = self.output_dir / "globals" / "performance_audit_report.json"
                auditor.save_report(audit_report_path)
                
                # Also generate summary log
                summary = auditor.report_summary()
                multipliers = auditor.report_multipliers(min_calls=2)
                nested_loops = auditor.report_nested_loops()
                
                logger.info("="*80)
                logger.info("📊 PERFORMANCE AUDIT SUMMARY")
                logger.info("="*80)
                logger.info(f"Total function calls tracked: {summary.get('total_calls', 0)}")
                logger.info(f"Unique functions: {summary.get('unique_functions', 0)}")
                
                if multipliers:
                    logger.info(f"\n⚠️  MULTIPLIERS FOUND: {len(multipliers)} functions called multiple times with same input")
                    for func_name, findings in multipliers.items():
                        for finding in findings:
                            logger.info(f"  - {func_name}: {finding['call_count']}× calls, {finding['total_duration']:.2f}s total "
                                      f"(wasted: {finding['wasted_duration']:.2f}s, stage: {finding['stage']})")
                
                if nested_loops:
                    logger.info(f"\n⚠️  NESTED LOOP PATTERNS: {len(nested_loops)} potential nested loop issues")
                    for finding in nested_loops[:5]:  # Show top 5
                        logger.info(f"  - {finding['func_name']}: {finding['consecutive_calls']} consecutive calls "
                                  f"in {finding['time_span']:.2f}s (stage: {finding['stage']})")
                
                logger.info(f"\n💾 Full audit report saved to: {audit_report_path}")
                logger.info("="*80)
        except Exception as e:
            logger.debug(f"Failed to save performance audit report: {e}")

        # Emit run completion event for dashboard
        run_duration = time.time() - _run_start_time
        successful_targets = len(targets) - len(failed_targets) if targets else 0
        emit_run_complete(
            status="success" if status == "completed" else "failed",
            total_targets=len(targets) if targets else 0,
            successful_targets=successful_targets,
            duration_seconds=run_duration,
        )
        emit_progress(
            "completed", 100,
            targets_complete=successful_targets,
            targets_total=len(targets) if targets else 0,
            message=f"Training complete: {total_models} models trained"
        )
        close_training_events()

        return {
            'targets': targets,
            'target_features': target_features,
            'strategy': strategy,
            'training_results': training_results,
            'total_models': total_models,
            'sentinel_results': sentinel_results,
            'status': status,  # Use status from above (either 'completed' or 'failed_no_models')
            'failed_targets': failed_targets,
            'failed_reasons': failed_reasons,
            'trend_summary': trend_summary
        }


def main():
    """Main entry point for intelligent training orchestrator."""
    # NOTE: ArgumentParser setup moved to intelligent_trainer/cli.py
    parser = create_argument_parser()
    args = parser.parse_args()

    # FIX: Track if user explicitly specified --output-dir before any fallback logic
    # This is checked BEFORE config/default fallbacks are applied
    user_specified_output_dir = args.output_dir is not None

    # NEW: Load experiment config if provided (PREFERRED)
    experiment_config = None
    if args.experiment_config and _NEW_CONFIG_AVAILABLE:
        try:
            from CONFIG.config_builder import load_experiment_config
            experiment_config = load_experiment_config(args.experiment_config)
            logger.info(f"✅ Loaded experiment config: {experiment_config.name}")
            # Use experiment config values if CLI args not provided
            if not args.data_dir:
                args.data_dir = experiment_config.data_dir
            if not args.symbols:
                args.symbols = experiment_config.symbols
        except Exception as e:
            logger.error(f"Failed to load experiment config '{args.experiment_config}': {e}")
            raise
    
    # Load intelligent training config (NEW - allows simple command-line usage)
    # Use config loader if available, otherwise fallback to direct path
    intel_config_data = {}
    if _CONFIG_LOADER_AVAILABLE:
        try:
            intel_config_data = load_training_config("intelligent_training_config")
            logger.info("✅ Loaded intelligent training config using config loader")
        except Exception as e:
            logger.warning(f"Could not load intelligent training config via loader: {e}")
    else:
        # Fallback: Use canonical path
        intel_config_file = Path("CONFIG/pipeline/training/intelligent.yaml")
        if intel_config_file.exists():
            try:
                import yaml
                with open(intel_config_file, 'r') as f:
                    intel_config_data = yaml.safe_load(f) or {}
                logger.info(f"✅ Loaded intelligent training config from {intel_config_file}")
            except Exception as e:
                logger.warning(f"Could not load intelligent training config: {e}")
    
    # Get config file path for logging (needed for trace output)
    if _CONFIG_LOADER_AVAILABLE:
        try:
            from CONFIG.config_loader import get_config_path
            intel_config_file = get_config_path("intelligent_training_config")
        except Exception as e:
            # Convenience path: fallback to default config path if get_config_path fails
            logger.debug(f"Could not get config path via get_config_path: {e}, using default")
            intel_config_file = Path("CONFIG/pipeline/training/intelligent.yaml")
    else:
        intel_config_file = Path("CONFIG/pipeline/training/intelligent.yaml")
    
    # Apply config values if CLI args not provided
    if not args.data_dir and intel_config_data.get('data', {}).get('data_dir'):
        args.data_dir = Path(intel_config_data['data']['data_dir'])
    if not args.symbols and intel_config_data.get('data', {}).get('symbols'):
        args.symbols = intel_config_data['data']['symbols']
    if not args.output_dir and intel_config_data.get('output', {}).get('output_dir'):
        args.output_dir = Path(intel_config_data['output']['output_dir'])
    if not args.cache_dir and intel_config_data.get('output', {}).get('cache_dir'):
        args.cache_dir = Path(intel_config_data['output']['cache_dir']) if intel_config_data['output']['cache_dir'] else None
    
    # Apply quick/full presets
    if args.quick:
        logger.info("🚀 Quick test mode enabled")
        intel_config_data.setdefault('targets', {})['max_targets_to_evaluate'] = 3
        intel_config_data.setdefault('targets', {})['top_n_targets'] = 3
        intel_config_data.setdefault('features', {})['top_m_features'] = 50
    elif args.full:
        logger.info("🏭 Full production mode enabled")
        # Use all config defaults
    
    # Validate required args (either from CLI, config, or experiment config)
    if not args.data_dir:
        parser.error("--data-dir is required (or set in CONFIG/pipeline/training/intelligent.yaml)")
    if not args.symbols:
        parser.error("--symbols is required (or set in CONFIG/pipeline/training/intelligent.yaml)")
    if not args.output_dir:
        args.output_dir = Path('intelligent_output')
    
    # Load intelligent training settings (prioritize new config file, fallback to old system)
    try:
        from CONFIG.config_loader import get_cfg
        _CONFIG_AVAILABLE = True
    except ImportError:
        _CONFIG_AVAILABLE = False
        logger.warning("Config loader not available, using hardcoded defaults")
    
    # Use new intelligent_training_config.yaml if available, otherwise fallback to pipeline_config.yaml
    if intel_config_data:
        # Use new config file
        targets_cfg = intel_config_data.get('targets', {})
        features_cfg = intel_config_data.get('features', {})
        data_cfg = intel_config_data.get('data', {})
        advanced_cfg = intel_config_data.get('advanced', {})
        cache_cfg = intel_config_data.get('cache', {})
        
        # NEW: Merge experiment config data section into data_cfg (experiment config takes priority)
        if experiment_config:
            # Merge experiment config data into data_cfg (experiment config overrides intelligent_training_config)
            if hasattr(experiment_config, 'data') and experiment_config.data:
                # Merge experiment config data values
                exp_data_dict = {
                    'data_dir': str(experiment_config.data_dir) if experiment_config.data_dir else None,
                    'symbols': experiment_config.symbols if experiment_config.symbols else None,
                    'interval': experiment_config.data.bar_interval if experiment_config.data.bar_interval else None,
                    'max_rows_per_symbol': experiment_config.max_samples_per_symbol if hasattr(experiment_config, 'max_samples_per_symbol') else None,
                    'max_samples_per_symbol': experiment_config.max_samples_per_symbol if hasattr(experiment_config, 'max_samples_per_symbol') else None,
                }
                # Update data_cfg with experiment config values (only non-None values)
                for key, value in exp_data_dict.items():
                    if value is not None:
                        data_cfg[key] = value
                logger.debug(f"📋 Merged experiment config data into data_cfg: {exp_data_dict}")
        
            # Also read data section directly from YAML file to get min_cs, max_cs_samples, max_rows_train
            # (these aren't in ExperimentConfig object, so read from YAML)
            try:
                import yaml
                exp_name = experiment_config.name
                exp_file = _get_experiment_config_path(exp_name)
                if exp_file.exists():
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    exp_data_section = exp_yaml.get('data', {})
                    # Merge ALL data section keys from YAML (experiment config takes priority)
                    # This includes: min_cs, max_cs_samples, max_rows_train, max_samples_per_symbol, max_rows_per_symbol
                    for key in ['min_cs', 'max_cs_samples', 'max_rows_train', 'max_samples_per_symbol', 'max_rows_per_symbol']:
                        if key in exp_data_section:
                            data_cfg[key] = exp_data_section[key]
                            logger.debug(f"📋 Loaded {key}={exp_data_section[key]} from experiment config YAML")
            except Exception as e:
                logger.debug(f"Could not load data section from experiment config YAML: {e}")
        
        auto_targets = targets_cfg.get('auto_targets', True)
        top_n_targets = targets_cfg.get('top_n_targets', 10)
        max_targets_to_evaluate = targets_cfg.get('max_targets_to_evaluate', None)
        manual_targets = targets_cfg.get('manual_targets', [])
        targets_to_evaluate = targets_cfg.get('targets_to_evaluate', [])  # NEW: Whitelist support
        
        # Track config source for debug logging
        config_sources = {
            'max_targets_to_evaluate': 'base_config',
            'top_n_targets': 'base_config',
            'targets_to_evaluate': 'base_config'
        }
        
        # NEW: Extract manual_targets, max_targets_to_evaluate, etc. from experiment config if available (overrides config file)
        if experiment_config:
            try:
                import yaml
                # Path is already imported at top of file
                exp_name = experiment_config.name
                exp_file = _get_experiment_config_path(exp_name)
                if exp_file.exists():
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    intel_training = exp_yaml.get('intelligent_training', {})
                    if intel_training:
                        exp_manual_targets = intel_training.get('manual_targets', [])
                        if exp_manual_targets:
                            manual_targets = exp_manual_targets
                            logger.info(f"📋 Using manual targets from experiment config: {manual_targets}")
                        exp_auto_targets = intel_training.get('auto_targets', True)
                        if not exp_auto_targets:
                            auto_targets = False
                            logger.info(f"📋 Disabled auto_targets from experiment config (using manual targets)")
                        # Extract max_targets_to_evaluate from experiment config (overrides base config)
                        exp_max_targets = intel_training.get('max_targets_to_evaluate')
                        logger.debug(f"🔍 DEBUG: Found max_targets_to_evaluate in experiment config: {exp_max_targets} (type: {type(exp_max_targets).__name__})")
                        if exp_max_targets is not None:
                            # Ensure it's an integer (YAML might load as int or string)
                            try:
                                max_targets_to_evaluate = int(exp_max_targets)
                                config_sources['max_targets_to_evaluate'] = 'experiment_config'
                                logger.info(f"📋 Using max_targets_to_evaluate={max_targets_to_evaluate} from experiment config")
                            except (ValueError, TypeError) as e:
                                logger.warning(f"⚠️  Invalid max_targets_to_evaluate value '{exp_max_targets}' in experiment config, ignoring: {e}")
                                # Keep existing value (from base config or default)
                        else:
                            logger.debug(f"🔍 DEBUG: max_targets_to_evaluate not found in experiment config intelligent_training section (current value: {max_targets_to_evaluate})")
                        # Extract top_n_targets from experiment config (overrides base config)
                        exp_top_n = intel_training.get('top_n_targets')
                        if exp_top_n is not None:
                            top_n_targets = exp_top_n
                            config_sources['top_n_targets'] = 'experiment_config'
                            logger.info(f"📋 Using top_n_targets={top_n_targets} from experiment config")
                        # Extract targets_to_evaluate whitelist from experiment config (NEW)
                        exp_targets_whitelist = intel_training.get('targets_to_evaluate', [])
                        if exp_targets_whitelist:
                            targets_to_evaluate = exp_targets_whitelist if isinstance(exp_targets_whitelist, list) else [exp_targets_whitelist]
                            config_sources['targets_to_evaluate'] = 'experiment_config'
                            logger.info(f"📋 Using targets_to_evaluate whitelist from experiment config: {targets_to_evaluate}")
            except Exception as e:
                logger.debug(f"Could not load intelligent_training from experiment config: {e}")
            
            # Fallback: Use targets.primary ONLY if auto_targets is False and no manual_targets specified
            # If auto_targets is True, we should auto-discover targets, not use primary as fallback
            if not manual_targets and not auto_targets and hasattr(experiment_config, 'target') and experiment_config.target:
                manual_targets = [experiment_config.target]
                logger.info(f"📋 Using primary target from experiment config (auto_targets=false): {manual_targets}")
        
        auto_features = features_cfg.get('auto_features', True)
        top_m_features = features_cfg.get('top_m_features', 100)
        manual_features = features_cfg.get('manual_features', [])
        
        # Model families from config (can be overridden by CLI or experiment config)
        # Default to None (not empty list) so we can distinguish "not set" from "empty list"
        config_families = intel_config_data.get('model_families', None)
        
        strategy = intel_config_data.get('strategy', 'single_task')
        min_cs = data_cfg.get('min_cs', 10)
        # Support both max_rows_per_symbol and max_samples_per_symbol (backward compatibility)
        max_rows_per_symbol = data_cfg.get('max_rows_per_symbol') or data_cfg.get('max_samples_per_symbol', None)
        max_rows_train = data_cfg.get('max_rows_train', None)
        max_cs_samples = data_cfg.get('max_cs_samples', 1000)
        run_leakage_diagnostics = advanced_cfg.get('run_leakage_diagnostics', False)
        
        # Decision application mode
        decisions_cfg = intel_config_data.get('decisions', {})
        decision_apply_mode_config = decisions_cfg.get('apply_mode', 'off')
        decision_min_level = decisions_cfg.get('min_level_to_apply', 2)
        
        use_cache = cache_cfg.get('use_cache', True)
        force_refresh_config = cache_cfg.get('force_refresh', False)
        
        # Check for test mode override (for E2E testing)
        # NOTE: Experiment config takes priority over test config
        use_test_config = args.output_dir and 'test' in str(args.output_dir).lower()
        if use_test_config and intel_config_data.get('test'):
            test_cfg = intel_config_data['test']
            logger.info("📋 Using test configuration (detected 'test' in output-dir)")
            # Only apply test config if experiment config didn't override these values
            # Check if experiment config already set these (experiment config takes priority)
            exp_has_max_targets = config_sources.get('max_targets_to_evaluate') == 'experiment_config'
            exp_has_top_n = config_sources.get('top_n_targets') == 'experiment_config'
            exp_has_top_m = False  # Check separately for top_m_features
            if experiment_config:
                try:
                    exp_name = experiment_config.name
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    if exp_yaml:
                        intel_training = exp_yaml.get('intelligent_training', {})
                        if intel_training:
                            # Double-check by looking at the YAML directly
                            if 'max_targets_to_evaluate' in intel_training:
                                exp_has_max_targets = True
                            if 'top_n_targets' in intel_training:
                                exp_has_top_n = True
                            if 'top_m_features' in intel_training:
                                exp_has_top_m = True
                except Exception as e:
                    # Best-effort: experiment config access failed, continue without flags
                    logger.debug(f"Could not check experiment config for intel_training flags: {e}")
            
            # Only override if experiment config didn't set these values
            if 'max_targets_to_evaluate' in test_cfg and not exp_has_max_targets:
                max_targets_to_evaluate = test_cfg.get('max_targets_to_evaluate')
                config_sources['max_targets_to_evaluate'] = 'test_config'
                logger.info(f"📋 Using max_targets_to_evaluate={max_targets_to_evaluate} from test config (experiment config did not override)")
            elif exp_has_max_targets:
                logger.debug(f"📋 Skipping test config max_targets_to_evaluate (experiment config value={max_targets_to_evaluate} takes priority)")
            if 'top_n_targets' in test_cfg and not exp_has_top_n:
                top_n_targets = test_cfg.get('top_n_targets')
                config_sources['top_n_targets'] = 'test_config'
                logger.info(f"📋 Using top_n_targets={top_n_targets} from test config (experiment config did not override)")
            elif exp_has_top_n:
                logger.debug(f"📋 Skipping test config top_n_targets (experiment config value={top_n_targets} takes priority)")
            if 'top_m_features' in test_cfg and not exp_has_top_m:
                top_m_features = test_cfg.get('top_m_features')
                logger.info(f"📋 Using top_m_features={top_m_features} from test config")
        
        # Debug logging: Show final config values and their sources
        logger.debug(f"🔍 Config precedence summary:")
        logger.debug(f"   max_targets_to_evaluate={max_targets_to_evaluate} (source: {config_sources.get('max_targets_to_evaluate', 'unknown')})")
        logger.debug(f"   top_n_targets={top_n_targets} (source: {config_sources.get('top_n_targets', 'unknown')})")
        if targets_to_evaluate:
            logger.debug(f"   targets_to_evaluate={targets_to_evaluate} (source: {config_sources.get('targets_to_evaluate', 'unknown')})")
    elif _CONFIG_AVAILABLE:
        # Fallback to old config system
        use_test_config = args.output_dir and 'test' in str(args.output_dir).lower()

        if use_test_config:
            test_cfg = get_cfg("test.intelligent_training", default={}, config_name="pipeline_config")
            if test_cfg:
                logger.info("📋 Using test configuration (detected 'test' in output-dir)")
                intel_cfg = test_cfg
            else:
                intel_cfg = get_cfg("intelligent_training", default={}, config_name="pipeline_config")
        else:
            intel_cfg = get_cfg("intelligent_training", default={}, config_name="pipeline_config")

        auto_targets = intel_cfg.get('auto_targets', True)
        top_n_targets = intel_cfg.get('top_n_targets', 5)
        max_targets_to_evaluate = intel_cfg.get('max_targets_to_evaluate', None)
        targets_to_evaluate = intel_cfg.get('targets_to_evaluate', [])  # NEW: Whitelist support
        auto_features = intel_cfg.get('auto_features', True)
        top_m_features = intel_cfg.get('top_m_features', 100)
        strategy = intel_cfg.get('strategy', 'single_task')
        min_cs = intel_cfg.get('min_cs', 10)
        
        # Experiment config overrides test config (experiment config takes priority)
        if experiment_config:
            try:
                import yaml
                exp_name = experiment_config.name
                exp_file = _get_experiment_config_path(exp_name)
                if exp_file.exists():
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    intel_training = exp_yaml.get('intelligent_training', {})
                    if intel_training:
                        # Override with experiment config values (experiment config takes priority)
                        if 'max_targets_to_evaluate' in intel_training:
                            max_targets_to_evaluate = intel_training.get('max_targets_to_evaluate')
                            logger.info(f"📋 Using max_targets_to_evaluate={max_targets_to_evaluate} from experiment config (overrides test config)")
                        if 'top_n_targets' in intel_training:
                            top_n_targets = intel_training.get('top_n_targets')
                            logger.info(f"📋 Using top_n_targets={top_n_targets} from experiment config (overrides test config)")
                        if 'top_m_features' in intel_training:
                            top_m_features = intel_training.get('top_m_features')
                            logger.info(f"📋 Using top_m_features={top_m_features} from experiment config (overrides test config)")
                        # Extract targets_to_evaluate whitelist from experiment config (NEW)
                        exp_targets_whitelist = intel_training.get('targets_to_evaluate', [])
                        if exp_targets_whitelist:
                            targets_to_evaluate = exp_targets_whitelist if isinstance(exp_targets_whitelist, list) else [exp_targets_whitelist]
                            logger.info(f"📋 Using targets_to_evaluate whitelist from experiment config: {targets_to_evaluate}")
            except Exception as e:
                logger.debug(f"Could not load intelligent_training from experiment config: {e}")
        
        # Priority: experiment_config object > intel_cfg dict (backward compatibility)
        # Support both max_rows_per_symbol and max_samples_per_symbol
        if experiment_config and hasattr(experiment_config, 'max_samples_per_symbol'):
            max_rows_per_symbol = experiment_config.max_samples_per_symbol
        else:
            max_rows_per_symbol = intel_cfg.get('max_rows_per_symbol') or intel_cfg.get('max_samples_per_symbol', None)
        
        max_rows_train = intel_cfg.get('max_rows_train', None)
        max_cs_samples = intel_cfg.get('max_cs_samples', None)
        run_leakage_diagnostics = intel_cfg.get('run_leakage_diagnostics', False)
        
        # Decision application mode (legacy config path)
        decisions_cfg = intel_cfg.get('decisions', {})
        decision_apply_mode_config = decisions_cfg.get('apply_mode', 'off')
        decision_min_level = decisions_cfg.get('min_level_to_apply', 2)
        
        use_cache = True
        force_refresh_config = False
        manual_targets = []
        manual_features = []
        config_families = None  # None = not set, [] = explicitly empty, [list] = explicitly set
        
        # NEW: Extract manual_targets from experiment config if available
        # Load the raw YAML to access intelligent_training section
        if experiment_config:
            try:
                import yaml
                # Path is already imported at top of file
                # Find the experiment config file
                exp_name = experiment_config.name
                exp_file = _get_experiment_config_path(exp_name)
                if exp_file.exists():
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    # Extract intelligent_training section
                    intel_training = exp_yaml.get('intelligent_training', {})
                    if intel_training:
                        exp_manual_targets = intel_training.get('manual_targets', [])
                        if exp_manual_targets:
                            manual_targets = exp_manual_targets
                            logger.info(f"📋 Using manual targets from experiment config: {manual_targets}")
                        # Also check auto_targets setting
                        exp_auto_targets = intel_training.get('auto_targets', True)
                        if not exp_auto_targets:
                            auto_targets = False
                            logger.info(f"📋 Disabled auto_targets from experiment config (using manual targets)")
                        # Extract max_targets_to_evaluate from experiment config (overrides base config)
                        exp_max_targets = intel_training.get('max_targets_to_evaluate')
                        if exp_max_targets is not None:
                            # Ensure it's an integer (YAML might load as int or string)
                            try:
                                max_targets_to_evaluate = int(exp_max_targets)
                                logger.info(f"📋 Using max_targets_to_evaluate={max_targets_to_evaluate} from experiment config")
                            except (ValueError, TypeError) as e:
                                logger.warning(f"⚠️  Invalid max_targets_to_evaluate value '{exp_max_targets}' in experiment config, ignoring: {e}")
                                # Keep existing value (from base config or default)
                        # Extract top_n_targets from experiment config (overrides base config)
                        exp_top_n = intel_training.get('top_n_targets')
                        if exp_top_n is not None:
                            top_n_targets = exp_top_n
                            logger.info(f"📋 Using top_n_targets={top_n_targets} from experiment config")
                    
                    # CRITICAL: Split model families config - training vs feature selection
                    # This ensures training.model_families is used for training, not feature selection
                    # Extract training.model_families from experiment config (NEW - SST)
                    exp_training = exp_yaml.get('training', {})
                    training_families = None
                    if exp_training:
                        exp_model_families = exp_training.get('model_families', None)
                        if exp_model_families is not None:
                            # Explicitly set in config - use it (even if empty list)
                            if isinstance(exp_model_families, list):
                                training_families = exp_model_families
                                logger.info(f"📋 Using training.model_families from experiment config (SST): {training_families}")
                            else:
                                logger.warning(f"⚠️ training.model_families in experiment config is not a list: {type(exp_model_families)}, ignoring")
                    
                    # Extract feature_selection.model_families from experiment config (NEW - SST)
                    exp_feature_selection = exp_yaml.get('feature_selection', {})
                    fs_families = None
                    if exp_feature_selection:
                        exp_fs_families = exp_feature_selection.get('model_families', None)
                        if exp_fs_families is not None:
                            if isinstance(exp_fs_families, list):
                                fs_families = exp_fs_families
                                logger.info(f"📋 Using feature_selection.model_families from experiment config (SST): {fs_families}")
                            else:
                                logger.warning(f"⚠️ feature_selection.model_families in experiment config is not a list: {type(exp_fs_families)}, ignoring")
                    
                    # Fallback: Use intelligent_training.model_families if neither training nor feature_selection specified
                    if training_families is None and fs_families is None:
                        intel_training = exp_yaml.get('intelligent_training', {})
                        if intel_training:
                            fallback_families = intel_training.get('model_families', None)
                            if fallback_families is not None and isinstance(fallback_families, list):
                                training_families = fallback_families
                                fs_families = fallback_families
                                logger.info(f"📋 Using intelligent_training.model_families as fallback for both training and feature selection: {fallback_families}")
                    
                    # Set config_families to training_families (for backward compatibility)
                    # Feature selection will use fs_families separately
                    # CRITICAL: Only set config_families if training_families was actually found
                    # If experiment config exists but training.model_families is not set,
                    # keep config_families from intelligent_training_config (set at line 3798)
                    # as fallback, but train_with_intelligence will use SST training.model_families
                    # from self._exp_yaml_sst instead
                    if training_families is not None:
                        config_families = training_families
                    # else: config_families remains from intelligent_training_config (line 3798)
                else:
                    # No experiment config - initialize to None
                    training_families = None
                    fs_families = None
            except Exception as e:
                logger.debug(f"Could not load intelligent_training from experiment config: {e}")
            
            # Fallback: Use targets.primary ONLY if auto_targets is False and no manual_targets specified
            # If auto_targets is True, we should auto-discover targets, not use primary as fallback
            if not manual_targets and not auto_targets and hasattr(experiment_config, 'target') and experiment_config.target:
                manual_targets = [experiment_config.target]
                logger.info(f"📋 Using primary target from experiment config (auto_targets=false): {manual_targets}")
        
        if max_cs_samples is None:
            max_cs_samples = get_cfg("pipeline.data_limits.max_cs_samples", default=None, config_name="pipeline_config")
    else:
        # Fallback defaults (must match CONFIG/pipeline/training/intelligent.yaml)
        auto_targets = True
        top_n_targets = 10  # Matches intelligent.yaml → targets.top_n_targets
        max_targets_to_evaluate = None
        targets_to_evaluate = []  # NEW: Initialize whitelist
        auto_features = True
        top_m_features = 100  # Matches intelligent.yaml → features.top_m_features
        strategy = 'single_task'  # Matches intelligent.yaml → strategy
        min_cs = 10  # Matches intelligent.yaml → data.min_cs
        max_rows_per_symbol = None
        max_rows_train = None
        max_cs_samples = None
        run_leakage_diagnostics = False
        
        # Decision application mode (fallback defaults)
        decision_apply_mode_config = 'off'
        decision_min_level = 2
        
        use_cache = True
        force_refresh_config = False
        manual_targets = []
        manual_features = []
        config_families = None  # None = not set, [] = explicitly empty, [list] = explicitly set
    
    # ============================================================================
    # CONFIG TRACE: Comprehensive logging of config loading and precedence
    # ============================================================================
    logger.info("=" * 80)
    logger.info("📋 CONFIG TRACE: Configuration Loading and Precedence")
    logger.info("=" * 80)
    
    # Track loaded files
    loaded_files = []
    if intel_config_file.exists():
        loaded_files.append(("intelligent_training_config.yaml", str(intel_config_file.resolve())))
    if experiment_config:
        exp_file = _get_experiment_config_path(experiment_config.name)
        if exp_file.exists():
            loaded_files.append((f"experiment: {experiment_config.name}.yaml", str(exp_file.resolve())))
    
    logger.info(f"📁 Loaded config files (in order):")
    for i, (name, path) in enumerate(loaded_files, 1):
        logger.info(f"   {i}. {name}")
        logger.info(f"      → {path}")
    
    # Track config value sources (before CLI overrides)
    config_trace = {}
    
    def trace_value(key: str, value: Any, source: str, section: str = ""):
        """Track where a config value came from"""
        full_key = f"{section}.{key}" if section else key
        if full_key not in config_trace:
            config_trace[full_key] = []
        config_trace[full_key].append({
            'value': value,
            'source': source
        })
    
    # Trace key config values from intelligent_training_config.yaml
    if intel_config_data:
        data_cfg = intel_config_data.get('data', {})
        targets_cfg = intel_config_data.get('targets', {})
        features_cfg = intel_config_data.get('features', {})
        
        trace_value("min_cs", min_cs, 
                    f"intelligent_training_config.yaml → data.min_cs = {data_cfg.get('min_cs', 'default=10')}",
                    "data")
        trace_value("max_cs_samples", max_cs_samples,
                    f"intelligent_training_config.yaml → data.max_cs_samples = {data_cfg.get('max_cs_samples', 'default=1000')}",
                    "data")
        # Check if experiment config overrode this value
        exp_max_rows = None
        if experiment_config:
            try:
                import yaml
                exp_name = experiment_config.name
                exp_file = _get_experiment_config_path(exp_name)
                if exp_file.exists():
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    exp_data = exp_yaml.get('data', {})
                    exp_max_rows = exp_data.get('max_rows_per_symbol') or exp_data.get('max_samples_per_symbol')
            except Exception as e:
                # Best-effort: experiment config load failed, continue without max_rows
                logger.debug(f"Could not load experiment config for max_rows: {e}")
        
        if exp_max_rows is not None:
            trace_value("max_rows_per_symbol", max_rows_per_symbol,
                        f"intelligent_training_config.yaml → data.max_rows_per_symbol = {data_cfg.get('max_rows_per_symbol') or data_cfg.get('max_samples_per_symbol', 'default=None')}",
                        "data")
            trace_value("max_rows_per_symbol", exp_max_rows,
                        f"experiment: {experiment_config.name}.yaml → data.max_rows_per_symbol = {exp_max_rows} (OVERRIDE)",
                        "data")
        else:
            trace_value("max_rows_per_symbol", max_rows_per_symbol,
                        f"intelligent_training_config.yaml → data.max_rows_per_symbol = {data_cfg.get('max_rows_per_symbol') or data_cfg.get('max_samples_per_symbol', 'default=None')}",
                        "data")
        trace_value("max_rows_train", max_rows_train,
                    f"intelligent_training_config.yaml → data.max_rows_train = {data_cfg.get('max_rows_train', 'default=None')}",
                    "data")
        trace_value("auto_targets", auto_targets,
                    f"intelligent_training_config.yaml → targets.auto_targets = {targets_cfg.get('auto_targets', 'default=True')}",
                    "targets")
        trace_value("top_n_targets", top_n_targets,
                    f"intelligent_training_config.yaml → targets.top_n_targets = {targets_cfg.get('top_n_targets', 'default=10')}",
                    "targets")
        trace_value("max_targets_to_evaluate", max_targets_to_evaluate,
                    f"intelligent_training_config.yaml → targets.max_targets_to_evaluate = {targets_cfg.get('max_targets_to_evaluate', 'default=None')}",
                    "targets")
        trace_value("auto_features", auto_features,
                    f"intelligent_training_config.yaml → features.auto_features = {features_cfg.get('auto_features', 'default=True')}",
                    "features")
        trace_value("top_m_features", top_m_features,
                    f"intelligent_training_config.yaml → features.top_m_features = {features_cfg.get('top_m_features', 'default=100')}",
                    "features")
    
    # Check for experiment config overrides
    if experiment_config:
        exp_file = _get_experiment_config_path(experiment_config.name)
        if exp_file.exists():
            try:
                exp_yaml = _load_experiment_config_safe(experiment_config.name)
                exp_data = exp_yaml.get('data', {})
                # Trace all data section keys that exist in experiment config
                for key in ['min_cs', 'max_cs_samples', 'max_rows_train', 'max_samples_per_symbol', 'max_rows_per_symbol']:
                    if key in exp_data:
                        trace_value(key, exp_data[key],
                                    f"experiment: {experiment_config.name}.yaml → data.{key} = {exp_data[key]} (OVERRIDE)",
                                    "data")
                # Trace intelligent_training section overrides
                exp_intel = exp_yaml.get('intelligent_training', {})
                if exp_intel:
                    for key in ['max_targets_to_evaluate', 'top_n_targets', 'auto_targets']:
                        if key in exp_intel:
                            trace_value(key, exp_intel[key],
                                        f"experiment: {experiment_config.name}.yaml → intelligent_training.{key} = {exp_intel[key]} (OVERRIDE)",
                                        "targets" if key in ['max_targets_to_evaluate', 'top_n_targets', 'auto_targets'] else "")
                
                # Trace training.model_families from experiment config
                exp_training = exp_yaml.get('training', {})
                if exp_training and 'model_families' in exp_training:
                    trace_value("model_families", exp_training['model_families'],
                                f"experiment: {experiment_config.name}.yaml → training.model_families = {exp_training['model_families']} (OVERRIDE)",
                                "training")
            except Exception as e:
                logger.debug(f"Could not trace experiment config: {e}")
    
    # CLI overrides (for testing/debugging only - warn user)
    if args.override_max_samples:
        logger.warning("⚠️  Using CLI override for max_samples (testing only - not SST compliant)")
        max_rows_per_symbol = args.override_max_samples
        trace_value("max_rows_per_symbol", max_rows_per_symbol, 
                    f"CLI --override-max-samples = {args.override_max_samples} (OVERRIDE)",
                    "data")
    if args.override_max_rows:
        logger.warning("⚠️  Using CLI override for max_rows (testing only - not SST compliant)")
        max_rows_per_symbol = args.override_max_rows
        trace_value("max_rows_per_symbol", max_rows_per_symbol,
                    f"CLI --override-max-rows = {args.override_max_rows} (OVERRIDE)",
                    "data")
    if args.targets:
        trace_value("manual_targets", args.targets, "CLI --targets (OVERRIDE)", "targets")
    if args.features:
        trace_value("manual_features", args.features, "CLI --features (OVERRIDE)", "features")
    if args.families:
        trace_value("model_families", args.families, "CLI --families (OVERRIDE)", "training")
    
    # Log final resolved values with source chain
    logger.info("")
    logger.info("🔍 Key Config Values (with source chain):")
    key_configs = [
        ("data.min_cs", min_cs),
        ("data.max_cs_samples", max_cs_samples),
        ("data.max_rows_per_symbol", max_rows_per_symbol),
        ("data.max_rows_train", max_rows_train),
        ("targets.auto_targets", auto_targets),
        ("targets.top_n_targets", top_n_targets),
        ("targets.max_targets_to_evaluate", max_targets_to_evaluate),
        ("features.auto_features", auto_features),
        ("features.top_m_features", top_m_features),
    ]
    
    for key, final_value in key_configs:
        if key in config_trace:
            sources = config_trace[key]
            logger.info(f"   {key}: {final_value}")
            for i, source_info in enumerate(sources, 1):
                arrow = "→" if i < len(sources) else "✓"
                logger.info(f"      {i}. {arrow} {source_info['source']}")
        else:
            logger.info(f"   {key}: {final_value} (no trace - using default or hardcoded)")
    
    # Check for conflicts (same key from multiple sources with different values)
    logger.info("")
    logger.info("⚠️  Conflict Detection:")
    conflicts = []
    for key, sources in config_trace.items():
        if len(sources) > 1:
            values = [s['value'] for s in sources]
            # Check if values are actually different (handle None, int/str conversions)
            unique_values = set(str(v) if v is not None else 'None' for v in values)
            if len(unique_values) > 1:
                conflicts.append((key, sources))
    
    if conflicts:
        logger.warning(f"   Found {len(conflicts)} potential conflicts:")
        for key, sources in conflicts:
            logger.warning(f"      {key}:")
            for source_info in sources:
                logger.warning(f"         - {source_info['source']}")
    else:
        logger.info("   ✅ No conflicts detected (all sources agree or override cleanly)")
    
    # ============================================================================
    # Runtime Settings section
    # ============================================================================
    logger.info("")
    logger.info("⚙️  Runtime Settings:")
    logger.info("")
    
    # Determinism settings
    logger.info("   🔒 Determinism:")
    try:
        from TRAINING.common.determinism import BASE_SEED
        base_seed = BASE_SEED if BASE_SEED is not None else 42
        logger.info(f"      Base seed: {base_seed}")
    except Exception:
        logger.info("      Base seed: 42 (default)")
    
    try:
        from CONFIG.config_loader import get_cfg
        threads = os.environ.get('OMP_NUM_THREADS', None)
        det_cfg = get_cfg("pipeline.determinism", default={}, config_name="pipeline_config")
        if threads:
            logger.info(f"      Threads: {threads} (from OMP_NUM_THREADS env var)")
        else:
            threads = det_cfg.get('threads', 'auto')
            logger.info(f"      Threads: {threads}")
        
        deterministic_mode = det_cfg.get('deterministic_algorithms', False)
        logger.info(f"      Deterministic algorithms: {deterministic_mode}")
    except Exception:
        threads = os.environ.get('OMP_NUM_THREADS', 'auto')
        logger.info(f"      Threads: {threads} (from env)")
        logger.info("      Deterministic algorithms: False (default)")
    
    # Parallelism settings
    logger.info("")
    logger.info("   🔀 Parallelism:")
    try:
        from CONFIG.config_loader import get_cfg
        
        # Multi-target parallelism
        multi_target_cfg = get_cfg("multi_target", default={}, config_name="target_configs")
        parallel_targets = multi_target_cfg.get('parallel_targets', False)
        logger.info(f"      parallel_targets: {parallel_targets}")
        
        # Multi-model feature selection parallelism
        multi_model_cfg = get_cfg("multi_model_feature_selection", default={}, config_name="multi_model_feature_selection")
        parallel_symbols = multi_model_cfg.get('parallel_symbols', False)
        logger.info(f"      parallel_symbols: {parallel_symbols}")
        
        # Threading config
        threading_cfg = get_cfg("threading.parallel", default={}, config_name="threading_config")
        parallel_enabled = threading_cfg.get('enabled', True)
        max_workers_process = threading_cfg.get('max_workers_process', None)
        max_workers_thread = threading_cfg.get('max_workers_thread', None)
        logger.info(f"      threading.parallel.enabled: {parallel_enabled}")
        if max_workers_process is not None:
            logger.info(f"      max_workers_process: {max_workers_process}")
        if max_workers_thread is not None:
            logger.info(f"      max_workers_thread: {max_workers_thread}")
    except Exception as e:
        logger.info(f"      ⚠️  Could not load parallelism config: {e}")
    
    # Routing settings
    logger.info("")
    logger.info("   🎯 Routing:")
    try:
        from CONFIG.config_loader import get_cfg
        
        routing_cfg = get_cfg("target_ranking.routing", default={}, config_name="target_ranking_config")
        dev_mode = routing_cfg.get('dev_mode', False)
        # Use centralized helper for consistency
        try:
            from CONFIG.dev_mode import get_dev_mode
            dev_mode = get_dev_mode()
        except Exception as e:
            # Best-effort: dev_mode check failed, fallback to config value
            logger.debug(f"Could not check dev_mode: {e}, using config value")
        dev_mode_indicator = " [DEV_MODE]" if dev_mode else ""
        logger.info(f"      dev_mode: {dev_mode}{dev_mode_indicator}")
        
        if dev_mode:
            # Show active thresholds (relaxed for dev mode)
            T_cs_base = float(routing_cfg.get('skill01_threshold', routing_cfg.get('auc_threshold', 0.65)))
            T_sym_base = float(routing_cfg.get('symbol_skill01_threshold', routing_cfg.get('symbol_auc_threshold', 0.60)))
            T_frac_base = float(routing_cfg.get('frac_symbols_good_threshold', 0.5))
            T_suspicious_cs_base = float(routing_cfg.get('suspicious_skill01', routing_cfg.get('suspicious_auc', 0.90)))
            T_suspicious_sym_base = float(routing_cfg.get('suspicious_symbol_skill01', routing_cfg.get('suspicious_symbol_auc', 0.95)))
            
            # Apply dev mode adjustments (same logic as in target_routing.py)
            T_cs = max(0.40, T_cs_base - 0.25)
            T_sym = max(0.35, T_sym_base - 0.25)
            T_frac = max(0.2, T_frac_base - 0.3)
            T_suspicious_cs = min(0.98, T_suspicious_cs_base + 0.05)
            T_suspicious_sym = min(0.98, T_suspicious_sym_base + 0.03)
            
            logger.info("      Active thresholds (dev_mode relaxed):")
            logger.info(f"         T_cs (skill01_threshold): {T_cs:.2f} (base: {T_cs_base:.2f})")
            logger.info(f"         T_sym (symbol_skill01_threshold): {T_sym:.2f} (base: {T_sym_base:.2f})")
            logger.info(f"         T_frac (frac_symbols_good_threshold): {T_frac:.2f} (base: {T_frac_base:.2f})")
            logger.info(f"         T_suspicious_cs: {T_suspicious_cs:.2f} (base: {T_suspicious_cs_base:.2f})")
            logger.info(f"         T_suspicious_sym: {T_suspicious_sym:.2f} (base: {T_suspicious_sym_base:.2f})")
        else:
            # Show base thresholds
            T_cs = float(routing_cfg.get('skill01_threshold', routing_cfg.get('auc_threshold', 0.65)))
            T_sym = float(routing_cfg.get('symbol_skill01_threshold', routing_cfg.get('symbol_auc_threshold', 0.60)))
            T_frac = float(routing_cfg.get('frac_symbols_good_threshold', 0.5))
            T_suspicious_cs = float(routing_cfg.get('suspicious_skill01', routing_cfg.get('suspicious_auc', 0.90)))
            T_suspicious_sym = float(routing_cfg.get('suspicious_symbol_skill01', routing_cfg.get('suspicious_symbol_auc', 0.95)))
            
            logger.info("      Active thresholds (production):")
            logger.info(f"         T_cs (skill01_threshold): {T_cs:.2f}")
            logger.info(f"         T_sym (symbol_skill01_threshold): {T_sym:.2f}")
            logger.info(f"         T_frac (frac_symbols_good_threshold): {T_frac:.2f}")
            logger.info(f"         T_suspicious_cs: {T_suspicious_cs:.2f}")
            logger.info(f"         T_suspicious_sym: {T_suspicious_sym:.2f}")
    except Exception as e:
        logger.info(f"      ⚠️  Could not load routing config: {e}")
    
    # Log working directory and config paths
    logger.info("")
    logger.info("📂 Environment:")
    logger.info(f"   Working directory: {os.getcwd()}")
    logger.info(f"   Project root: {_PROJECT_ROOT}")
    if _CONFIG_LOADER_AVAILABLE:
        logger.info(f"   Config directory: {CONFIG_DIR.resolve()}")
    else:
        logger.info(f"   Config directory: {Path('CONFIG').resolve()}")
    
    logger.info("=" * 80)
    logger.info("")
    
    # ============================================================================
    # End of config trace
    # ============================================================================
    
    # Manual overrides (targets/features/families) - CLI > config (SST) > defaults
    # Priority: CLI args > experiment config (SST) > intelligent_training_config > hardcoded defaults
    targets = args.targets if args.targets else (manual_targets if manual_targets else None)
    features = args.features if args.features else (manual_features if manual_features else None)
    
    # Families: CLI overrides config, but config is SST if CLI not provided
    # CRITICAL: CLI --families ONLY affects training families, NOT feature selection
    # FS families are intentionally different (include selectors like mutual_information)
    if args.families:
        families = args.families
        logger.info(f"📋 Using model families from CLI (overrides config): {families}")
        # CLI ONLY overrides training families - FS families come from config
        # fs_families is already set from config above (around line 3203)
        if 'fs_families' not in locals() or fs_families is None:
            # Fallback: if fs_families wasn't set from config, keep it separate
            # Use default FS families that include selectors
            fs_families = ['lightgbm', 'xgboost', 'random_forest', 'mutual_information', 'univariate_selection']
            logger.info(f"📋 CLI override: using default feature selection families: {fs_families}")
    elif config_families is not None:
        # Config explicitly set (even if empty list) - use it as SST
        families = config_families if config_families else []
        if families:
            logger.info(f"📋 Using training model families from config (SST): {families}")
        else:
            logger.warning(f"⚠️ Config specifies empty model_families list - no families will be trained")
        # For feature selection, use fs_families if it was set, otherwise use training_families
        if 'fs_families' not in locals() or fs_families is None:
            fs_families = families
    else:
        # No config specified - use defaults (backward compatibility)
        families = ['lightgbm', 'xgboost', 'random_forest']
        logger.warning(f"⚠️ No model families in config, using defaults: {families}")
        fs_families = families
    
    # Log both resolved sets once per run
    # IMPORTANT: These are SEPARATE and should NEVER be mixed:
    # - families: Training families (trainers like lightgbm, xgboost, ensemble)
    # - fs_families: Feature selection families (may include selectors like mutual_information)
    logger.info(f"📋 Resolved model families - Training: {families}, Feature Selection: {fs_families}")
    
    # Create orchestrator
    # Pass config limits for output directory binning (use configured values, not full dataset size)
    trainer = IntelligentTrainer(
        data_dir=args.data_dir,
        symbols=args.symbols,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        experiment_config=experiment_config,  # Pass experiment config if loaded
        max_rows_per_symbol=max_rows_per_symbol,  # For output directory binning
        max_cs_samples=max_cs_samples,  # For output directory binning
        fs_model_families=fs_families,  # Pass feature selection families separately
        user_specified_output_dir=user_specified_output_dir  # FIX: Respect user's --output-dir
    )
    
    # Load configs (legacy support)
    target_ranking_config = None
    if args.target_ranking_config:
        target_ranking_config = load_target_configs(args.target_ranking_config)
    
    multi_model_config = None
    if args.multi_model_config:
        multi_model_config = load_multi_model_config(args.multi_model_config)
    elif not experiment_config:
        # Only load default if no experiment config
        multi_model_config = load_multi_model_config()
    
    # Determine cache usage (CLI overrides config)
    if args.no_cache:
        use_cache = False
    elif args.force_refresh:
        use_cache = True  # Still use cache, but force refresh
    else:
        use_cache = use_cache  # From config
    
    # Force refresh from config or CLI
    force_refresh = args.force_refresh or force_refresh_config
    
    # Decision application mode (CLI overrides config)
    decision_mode = args.apply_decisions if hasattr(args, 'apply_decisions') and args.apply_decisions else decision_apply_mode_config
    decision_apply_mode = (decision_mode == 'apply')
    decision_dry_run = (decision_mode == 'dry_run')
    
    # Run training with config-driven settings
    try:
        results = trainer.train_with_intelligence(
            auto_targets=auto_targets,
            top_n_targets=top_n_targets,
            max_targets_to_evaluate=max_targets_to_evaluate,
            targets_to_evaluate=targets_to_evaluate,  # NEW: Pass whitelist
            auto_features=auto_features,
            top_m_features=top_m_features,
            targets=targets,  # Manual override if provided
            features=features,  # Manual override if provided
            families=families,  # Manual override if provided
            strategy=strategy,
            force_refresh=force_refresh,
            use_cache=use_cache,
            run_leakage_diagnostics=run_leakage_diagnostics,
            min_cs=min_cs,
            max_rows_per_symbol=max_rows_per_symbol,
            max_rows_train=max_rows_train,
            max_cs_samples=max_cs_samples,
            decision_apply_mode=decision_apply_mode,
            decision_dry_run=decision_dry_run,
            decision_min_level=decision_min_level if 'decision_min_level' in locals() else 2
        )
        
        logger.info("="*80)
        logger.info("✅ Intelligent training pipeline completed")
        logger.info("="*80)
        logger.info(f"Targets: {len(results['targets'])}")
        logger.info(f"Strategy: {results['strategy']}")
        logger.info(f"Status: {results['status']}")
        
        # Compute and save run hash with change detection
        try:
            from TRAINING.orchestration.utils.diff_telemetry import save_run_hash, DiffTelemetry
            from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
            
            # FIX: Find previous run ID by searching parent/sibling directories for previous runs
            # Don't look in current run's run_hash.json (it doesn't exist yet on first run)
            prev_run_id = None
            try:
                # Path is already imported at module level, no need to re-import
                import json
                
                # Find RESULTS directory by walking up from output_dir
                results_dir = trainer.output_dir
                for _ in range(10):
                    if results_dir.name == "RESULTS":
                        break
                    if not results_dir.parent.exists():
                        break
                    results_dir = results_dir.parent
                
                # Search for previous runs in RESULTS/runs/cg-*/ or RESULTS/sample_*/
                if results_dir.name == "RESULTS":
                    runs_dir = results_dir / "runs"
                    if runs_dir.exists():
                        # New structure: RESULTS/runs/cg-*/
                        search_base = runs_dir
                    else:
                        # Old structure: RESULTS/sample_*/
                        search_base = results_dir
                    
                    # Find all intelligent_output_* directories (deterministic sorting by run_id timestamp)
                    prev_runs = []
                    # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                    for cg_dir in iterdir_sorted(search_base):
                        if not cg_dir.is_dir():
                            continue
                        # DETERMINISM: Use iterdir_sorted for deterministic iteration order
                        for run_dir in iterdir_sorted(cg_dir):
                            if not run_dir.is_dir():
                                continue
                            # Use parser to handle old underscore, old dash, and new suffix forms
                            from TRAINING.orchestration.utils.manifest import parse_run_instance_dirname
                            parsed = parse_run_instance_dirname(run_dir.name)
                            if parsed is None:
                                continue  # Skip invalid formats
                            
                            prev_hash_file = run_dir / "globals" / "run_hash.json"
                            if prev_hash_file.exists() and prev_hash_file != (get_globals_dir(trainer.output_dir) / "run_hash.json"):
                                try:
                                    with open(prev_hash_file, 'r') as f:
                                        prev_data = json.load(f)
                                        run_id = prev_data.get('run_id', '')
                                        # DETERMINISTIC: Parse timestamp from parsed directory name for sorting
                                        # Use parsed date_str and time_str for consistent sorting
                                        ts_int = -1
                                        if parsed.date_str and parsed.time_str:
                                            # Format: YYYYMMDD_HHMMSS
                                            try:
                                                ts_int = int(parsed.date_str + parsed.time_str)
                                            except ValueError:
                                                pass
                                        
                                        # Fallback: try to parse from run_id if directory parsing failed
                                        if ts_int == -1:
                                            import re
                                            match = re.match(r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', run_id)
                                            if match:
                                                # Convert to integer: YYYYMMDDHHMMSS
                                                ts_int = int(f"{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}{match.group(5)}{match.group(6)}")
                                            else:
                                                # Try ISO format: YYYY-MM-DDTHH:MM:SS
                                                match = re.match(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})', run_id)
                                                if match:
                                                    ts_int = int(f"{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}{match.group(5)}{match.group(6)}")
                                        
                                        prev_runs.append((ts_int, run_id, prev_data.get('run_hash')))
                                except Exception as e:
                                    # Diagnostic: run_id parsing failed, skip this run
                                    logger.debug(f"Could not parse run_id timestamp: {e}, skipping")
                                    continue
                    
                    # DETERMINISTIC: Sort by parsed timestamp (not mtime)
                    if prev_runs:
                        prev_runs.sort(reverse=True)  # Most recent first (highest timestamp)
                        prev_run_id = prev_runs[0][1]
                        logger.debug(f"Found previous run ID: {prev_run_id} from {len(prev_runs)} previous runs")
            except Exception as e:
                logger.debug(f"Failed to find previous run ID: {e}")
            
            # Create DiffTelemetry instance for change detection
            diff_telemetry = DiffTelemetry(output_dir=trainer.output_dir)
            
            # Compute and save run hash
            # Read run_id from manifest.json (authoritative source - SST pattern)
            run_id = None
            run_id_source = None  # Track which source provided run_id (for debugging)
            
            # Guard: Normalize output_dir from str/PathLike → Path before building manifest_path
            raw_output_dir = getattr(trainer, 'output_dir', None)
            try:
                output_dir = Path(raw_output_dir) if raw_output_dir else None
            except (TypeError, ValueError):
                # Invalid path type - skip manifest read
                output_dir = None
            
            if output_dir:
                from TRAINING.orchestration.utils.manifest import read_run_id_from_manifest
                manifest_path = output_dir / "manifest.json"
                run_id = read_run_id_from_manifest(manifest_path)
                if run_id:
                    run_id_source = "manifest"
                    logger.debug(f"Using run_id from manifest.json: {run_id}")
            
            # Fallback 1: Use output_dir.name (most likely to match reality if manifest missing)
            if not run_id and output_dir:
                dir_name = output_dir.name
                if dir_name and isinstance(dir_name, str) and dir_name.strip():
                    run_id = dir_name.strip()
                    run_id_source = "output_dir.name"
                    logger.debug(f"Using run_id from output_dir.name: {run_id}")
            
            # Fallback 2: Use _run_name (as-is, no format mutations)
            if not run_id:
                if hasattr(trainer, '_run_name') and trainer._run_name and isinstance(trainer._run_name, str):
                    run_id = trainer._run_name.strip()  # Use as-is, don't convert underscores to dashes
                    run_id_source = "_run_name"
                    logger.debug(f"Using run_id from _run_name: {run_id}")
            
            # Fallback 3: Derive from RunIdentity if available (deterministic, matches what snapshots use)
            # RI-005: Fixed attribute names - use run_identity (finalized) or _partial_identity (partial)
            if not run_id:
                run_identity = getattr(trainer, 'run_identity', None) or getattr(trainer, '_partial_identity', None)
                if run_identity:
                    try:
                        from TRAINING.orchestration.utils.manifest import derive_run_id_from_identity
                        run_id = derive_run_id_from_identity(run_identity=run_identity)
                        run_id_source = "RunIdentity"
                        logger.debug(f"Derived run_id from identity: {run_id}")
                    except ValueError:
                        # Identity not finalized - can't derive, continue to next fallback
                        logger.debug("RunIdentity not finalized, cannot derive run_id")
                        pass
            
            # Final fallback: Disable filtering or raise (strict mode policy)
            if not run_id:
                from TRAINING.common.determinism import is_strict_mode
                if is_strict_mode():
                    # Strict mode: raise error (fail closed)
                    raise RuntimeError(
                        f"Cannot determine run_id from manifest, directory name, _run_name, or RunIdentity. "
                        f"output_dir={output_dir}, _run_name={getattr(trainer, '_run_name', None)}. "
                        f"In strict mode, run_id is required for deterministic run hash computation."
                    )
                else:
                    # Best-effort mode: disable filtering + loud warning
                    logger.warning(
                        f"⚠️ Cannot determine run_id from manifest, directory name, _run_name, or RunIdentity. "
                        f"Computing run hash without run_id filter. This may aggregate snapshots from multiple runs. "
                        f"output_dir={output_dir}, _run_name={getattr(trainer, '_run_name', None)}, "
                        f"attempted_sources={['manifest', 'output_dir.name', '_run_name', 'RunIdentity']}"
                    )
                    run_id = None  # Explicitly None - compute_full_run_hash will use all snapshots
                    run_id_source = "none"
            
            # Log which source won (for debugging run mismatches) - explicit at end, not in else block
            if run_id:
                logger.debug(f"run_id lookup succeeded: source={run_id_source}, run_id={run_id}")
            
            saved_path = save_run_hash(
                output_dir=trainer.output_dir,
                run_id=run_id,
                prev_run_id=prev_run_id,
                diff_telemetry=diff_telemetry
            )
            
            if saved_path:
                logger.info(f"✅ Run hash saved to: {saved_path}")
                # Log change summary if available
                try:
                    import json
                    with open(saved_path, 'r') as f:
                        run_hash_data = json.load(f)
                        if run_hash_data.get('changes'):
                            changes = run_hash_data['changes']
                            logger.info(f"   Changes: {changes.get('severity_summary', 'none')} severity, "
                                      f"{len(changes.get('changed_snapshots', []))} snapshots changed")
                        if run_hash_data.get('run_hash'):
                            logger.info(f"   Run hash: {run_hash_data['run_hash']}")
                except Exception as e:
                    # Diagnostic: run_hash logging failed, continue without it
                    logger.debug(f"Could not log run_hash data: {e}")
                
                # Update manifest with run_hash (reuse existing function)
                try:
                    from TRAINING.orchestration.utils.manifest import update_manifest_with_run_hash
                    update_manifest_with_run_hash(trainer.output_dir)
                except Exception as e:
                    logger.debug(f"Failed to update manifest with run_hash: {e}")
            else:
                logger.warning(
                    "⚠️ Run hash computation returned None. This may indicate no snapshots were found. "
                    "Run will continue but reproducibility tracking may be incomplete."
                )
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to compute run hash: {e}. "
                f"Run will continue but reproducibility tracking may be incomplete."
            )
            import traceback
            logger.debug(f"Run hash computation traceback: {traceback.format_exc()}")
        
        # Write registry autopatch suggestions if enabled
        try:
            from TRAINING.common.utils.registry_autopatch import get_autopatch
            autopatch = get_autopatch()
            
            if autopatch.enabled and autopatch.write:
                patch_file = autopatch.write_patch_file(trainer.output_dir)
                if patch_file:
                    logger.info(f"✅ Registry autopatch suggestions written to: {patch_file}")
                    
                    # Record in manifest if available
                    try:
                        from TRAINING.orchestration.utils.manifest import update_manifest
                        import json
                        manifest_path = trainer.output_dir / "globals" / "manifest.json"
                        if manifest_path.exists():
                            with open(manifest_path, 'r') as f:
                                manifest = json.load(f)
                            
                            manifest['registry_autopatch'] = {
                                'suggestions_written': True,
                                'suggestion_file': str(patch_file.relative_to(trainer.output_dir)),
                                'n_features': len(autopatch._suggestions) if hasattr(autopatch, '_suggestions') else 0
                            }
                            
                            # SST: Use write_atomic_json for atomic write with canonical serialization
                            from TRAINING.common.utils.file_utils import write_atomic_json
                            write_atomic_json(manifest_path, manifest)
                    except Exception as e:
                        logger.debug(f"Failed to record autopatch in manifest: {e}")
        except Exception as e:
            logger.debug(f"Registry autopatch not available or failed: {e}")
        
        return 0
    
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

