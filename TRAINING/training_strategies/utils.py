# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

# ---- PATH BOOTSTRAP: ensure project root on sys.path in parent AND children ----
import os, sys
from pathlib import Path

# CRITICAL: Set LD_LIBRARY_PATH for conda CUDA libraries BEFORE any imports
# This must happen before TensorFlow tries to load CUDA libraries
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    conda_lib = os.path.join(conda_prefix, "lib")
    conda_targets_lib = os.path.join(conda_prefix, "targets", "x86_64-linux", "lib")
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = []
    if conda_lib not in current_ld_path:
        new_paths.append(conda_lib)
    if conda_targets_lib not in current_ld_path:
        new_paths.append(conda_targets_lib)
    if new_paths:
        updated_ld_path = ":".join(new_paths + [current_ld_path] if current_ld_path else new_paths)
        os.environ["LD_LIBRARY_PATH"] = updated_ld_path

# Show TensorFlow warnings so user knows if GPU isn't working
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Removed - show warnings
# os.environ.setdefault("TF_LOGGING_VERBOSITY", "ERROR")  # Removed - show warnings

# project root: TRAINING/training_strategies/utils.py -> parents[2] = repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make sure Python can import `common`, `model_fun`, etc.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Propagate to spawned processes (spawned interpreter reads PYTHONPATH at startup)
os.environ.setdefault("PYTHONPATH", str(_PROJECT_ROOT))

# Now we can import path setup utilities
from TRAINING.common.utils.path_setup import setup_all_paths
_PROJECT_ROOT, _TRAINING_ROOT, _CONFIG_DIR = setup_all_paths(_PROJECT_ROOT)

# Import config loader (CONFIG is already in sys.path from setup_all_paths)
try:
    from config_loader import get_pipeline_config, get_family_timeout, get_cfg, get_system_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    import logging
    # Only log at debug level to avoid misleading warnings
    logging.getLogger(__name__).debug("Config loader not available; using hardcoded defaults")

from TRAINING.common.safety import set_global_numeric_guards
set_global_numeric_guards()

# ---- JOBLIB/LOKY CLEANUP: prevent resource tracker warnings ----
from TRAINING.common.utils.process_cleanup import setup_loky_cleanup_from_config
setup_loky_cleanup_from_config()

"""
Enhanced Training Script with Multiple Strategies - Full Original Functionality

Replicates ALL functionality from train_mtf_cross_sectional_gpu.py but with:
- Modular architecture
- 3 training strategies (single-task, multi-task, cascade)
- All 20 model families from original script
- GPU acceleration
- Memory management
- Batch processing
- Cross-sectional training
- Target discovery
- Data validation
"""

# ANTI-DEADLOCK: Process-level safety (before importing TF/XGB/sklearn)
import time as _t
# Make thread pools predictable (also avoids weird deadlocks)


# Import the isolation runner (moved to TRAINING/common/isolation_runner.py)
# Paths are already set up above

from TRAINING.common.isolation_runner import child_isolated
from TRAINING.common.threads import temp_environ, child_env_for_family, plan_for_family, thread_guard, set_estimator_threads
from TRAINING.common.tf_runtime import ensure_tf_initialized
from TRAINING.common.tf_setup import tf_thread_setup

# Family classifications - import from centralized constants
from TRAINING.common.family_constants import TF_FAMS, TORCH_FAMS, CPU_FAMS, TORCH_SEQ_FAMILIES


"""Utility functions for training strategies."""

def setup_logging(log_level: str = "INFO", logfile: str = "training_strategies.log"):
    import queue
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper()))
    # remove any existing handlers
    for h in list(root.handlers): root.removeHandler(h)
    q = queue.SimpleQueue()  # lock-free
    qh = logging.handlers.QueueHandler(q)
    root.addHandler(qh)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(logfile);      fh.setFormatter(fmt)
    listener = logging.handlers.QueueListener(q, sh, fh)
    listener.daemon = True
    listener.start()
    return listener

# --- THREAD/OMP GUARD: put this at the VERY top, once ---
import os, time as _t
def _now(): return _t.perf_counter()

def safe_duration(t0):
    # EH-010: Add debug logging for duration computation errors
    try:
        return f"{_t.perf_counter()-t0:.2f}s"
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"EH-010: Failed to compute duration: {e}")
        return "n/a"

# global knobs filled by main()
# Load from config if available, otherwise use defaults
if _CONFIG_AVAILABLE:
    pipeline_cfg = get_pipeline_config()
    threading_cfg = get_cfg("threading.defaults.default_threads", config_name="threading_config")
    if threading_cfg is None:
        THREADS = max(1, (os.cpu_count() or 2) - 1)
    else:
        THREADS = threading_cfg if isinstance(threading_cfg, int) else max(1, (os.cpu_count() or 2) - 1)
    MKL_THREADS_DEFAULT = int(get_cfg("threading.defaults.mkl_threads", default=1, config_name="threading_config"))
else:
    THREADS = max(1, (os.cpu_count() or 2) - 1)
    MKL_THREADS_DEFAULT = 1
CPU_ONLY = False

def _pkg_ver(name):
    """Dynamic package version detection"""
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "missing"

def _env_guard(omp_threads: int, mkl_threads: int = 1):
    os.environ["OMP_NUM_THREADS"]   = str(omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(mkl_threads)
    os.environ["MKL_NUM_THREADS"]      = str(mkl_threads)
    os.environ["NUMEXPR_NUM_THREADS"]  = "1"
    os.environ["KMP_AFFINITY"]         = "disabled"
    os.environ["KMP_BLOCKTIME"]        = "0"
    os.environ["MKL_THREADING_LAYER"]  = "GNU"
    os.environ["JOBLIB_START_METHOD"]  = "spawn"

_env_guard(max(1, (os.cpu_count() or 2) - 1), mkl_threads=1)

# multiprocessing start method BEFORE heavy imports

# TF env before any TF import
# Load from config if available
if _CONFIG_AVAILABLE:
    python_hash_seed = get_cfg("pipeline.determinism.python_hash_seed", default="42")
    tf_deterministic = get_cfg("pipeline.determinism.tf_deterministic_ops", default="1")
    os.environ.setdefault("PYTHONHASHSEED", python_hash_seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", tf_deterministic)
else:
    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# TF_CPP_MIN_LOG_LEVEL already set at top of file

# CRITICAL: Import determinism FIRST before any ML libraries
from TRAINING.common.determinism import init_determinism_from_config, stable_seed_from, seed_for, get_deterministic_params, log_determinism_info

# Set global determinism immediately (reads from config, respects REPRO_MODE env var)
BASE_SEED = init_determinism_from_config()

import argparse
import logging
# Removed duplicate import: numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Type alias for consistent 8-tuple returns
Eight = Tuple[np.ndarray, np.ndarray, List[str], List[str], np.ndarray, List[str], Optional[np.ndarray], Any]
import sys
# Removed duplicate import: os
import warnings
import joblib
from datetime import datetime
# Removed unused import: glob
import time

# Polars optimization (same as original script)
# Use global THREADS variable (already loaded from config if available)
DEFAULT_THREADS = str(THREADS)
USE_POLARS = os.getenv("USE_POLARS", "1") == "1"
if USE_POLARS:
    try:
        os.environ.setdefault("POLARS_MAX_THREADS", DEFAULT_THREADS)
        import polars as pl
        pl.enable_string_cache()
        print("🚀 Polars optimization enabled")
    except ImportError:
        USE_POLARS = False
        print("Polars not available, falling back to pandas")

# Set up environment variables for determinism (same as original)
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# NOTE: Do NOT set np.random.seed() or random.seed() at module level.
# The determinism framework (repro_bootstrap) handles seeding.
# Module-level seeds would overwrite the framework's controlled seeding on every import.
import random

# Suppress warnings
warnings.filterwarnings("ignore", message=r"Protobuf gencode version .* is exactly one major version older")

# Import modular training system
# Note: Strategy classes (SingleTaskStrategy, MultiTaskStrategy, CascadeStrategy) are not currently defined
# They may have been removed during refactoring or are defined elsewhere
# Removed unused imports: ModelFactory, DataPreprocessor, TargetResolver, ValidationUtils

# Import target router for enhanced target support
from TRAINING.orchestration.routing.target_router import route_target, spec_from_target, TaskSpec

# Import existing utilities
try:
    from target_resolver import safe_target_extraction, pick_single_target_column
    from memory_manager import aggressive_cleanup, monitor_memory
except ImportError:
    # Fallback if scripts module not available
    def safe_target_extraction(df, target):
        return df[target], target
    
    def pick_single_target_column(df, target):
        return target
    
    def aggressive_cleanup():
        import gc
        gc.collect()
    
    def monitor_memory():
        return {}

logger = logging.getLogger(__name__)

# All 20 model families from original script
ALL_FAMILIES = [
    'LightGBM', 'XGBoost', 'MLP', 'CNN1D', 'LSTM', 'Transformer',
    'TabCNN', 'TabLSTM', 'TabTransformer', 'RewardBased',
    'QuantileLightGBM', 'NGBoost', 'GMMRegime', 'ChangePoint',
    'FTRLProximal', 'VAE', 'GAN', 'Ensemble', 'MetaLearning', 'MultiTask'
]

# Model type classification for efficient training
CROSS_SECTIONAL_MODELS = [
    'LightGBM', 'XGBoost', 'MLP', 'Ensemble', 'RewardBased',
    'QuantileLightGBM', 'NGBoost', 'GMMRegime', 'ChangePoint',
    'FTRLProximal', 'VAE', 'GAN', 'MetaLearning', 'MultiTask'
]

SEQUENTIAL_MODELS = [
    'CNN1D', 'LSTM', 'Transformer', 'TabCNN', 'TabLSTM', 'TabTransformer'
]

# PyTorch sequential families (for better performance)
TORCH_SEQ_FAMILIES = {"CNN1D", "LSTM", "Transformer", "TabCNN", "TabLSTM", "TabTransformer"}

def normalize_family_name(family: str) -> str:
    """
    Canonicalize model family name to snake_case lowercase for registry lookups.
    
    DEPRECATED: Use TRAINING.utils.sst_contract.normalize_family() instead.
    This function is kept for backward compatibility but delegates to SST contract.
    
    Args:
        family: Family name (can be any case/variant)
    
    Returns:
        Normalized family name in snake_case lowercase
    """
    try:
        # Use SST contract for consistency
        from TRAINING.common.utils.sst_contract import normalize_family
        return normalize_family(family)
    except ImportError:
        # Fallback to original logic if SST contract not available
        if not family or not isinstance(family, str):
            return str(family).lower() if family else ""
        
        import re
        
        # Normalize input: strip, replace hyphens/spaces with underscores
        family_clean = family.strip().replace("-", "_").replace(" ", "_")
        
        # If already snake_case (has underscores), just lowercase
        if "_" in family_clean:
            return family_clean.lower().replace("__", "_")
        
        # Convert TitleCase/CamelCase to snake_case
        parts = re.split(r'(?=[A-Z])', family_clean)
        parts = [p for p in parts if p]  # Remove empty strings
        
        if len(parts) == 1:
            return parts[0].lower()
        
        result = "_".join(p.lower() for p in parts)
        return result.replace("__", "_")


# Family capabilities map (from original script)
# All keys must be canonical snake_case (enforced by _assert_canonical_keys)
# 
# supported_tasks: Optional list of task types this family supports.
#   - If missing/None: family supports ALL task types (default-allow)
#   - Values: "regression", "binary", "multiclass", "ranking"
#   - Families are skipped if task_type not in supported_tasks
FAMILY_CAPS = {
    # Tree-based: support all tasks natively (no supported_tasks = all allowed)
    "lightgbm": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "xgboost": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "catboost": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "random_forest": {"nan_ok": True, "needs_tf": False, "experimental": False},
    
    # Neural nets: support all tasks (with proper objective selection)
    "mlp": {"nan_ok": False, "needs_tf": True, "experimental": False, "preprocess_in_family": True},
    "cnn1d": {"nan_ok": False, "needs_tf": False, "backend": "torch", "experimental": False, "preprocess_in_family": True},
    "lstm": {"nan_ok": False, "needs_tf": False, "backend": "torch", "experimental": False, "preprocess_in_family": True},
    "transformer": {"nan_ok": False, "needs_tf": False, "backend": "torch", "experimental": False, "preprocess_in_family": True},
    "tabcnn": {"nan_ok": False, "needs_tf": False, "backend": "torch", "experimental": False, "preprocess_in_family": True},
    "tablstm": {"nan_ok": False, "needs_tf": False, "backend": "torch", "experimental": False, "preprocess_in_family": True},
    "tabtransformer": {"nan_ok": False, "needs_tf": False, "backend": "torch", "experimental": False, "preprocess_in_family": True},
    "neural_network": {"nan_ok": False, "needs_tf": True, "experimental": False, "preprocess_in_family": True},
    
    # Linear models: regression only (no classification API)
    "lasso": {"nan_ok": False, "needs_tf": False, "experimental": False,
              "supported_tasks": ["regression"]},
    "ridge": {"nan_ok": False, "needs_tf": False, "experimental": False,
              "supported_tasks": ["regression"]},
    "elastic_net": {"nan_ok": False, "needs_tf": False, "experimental": False,
                    "supported_tasks": ["regression"]},
    
    # Logistic regression: classification only
    "logistic_regression": {"nan_ok": False, "needs_tf": False, "experimental": False,
                            "supported_tasks": ["binary", "multiclass"]},
    
    # NGBoost: regression + binary (multiclass not well-supported)
    "ngboost": {"nan_ok": False, "needs_tf": False, "experimental": True,
                "supported_tasks": ["regression", "binary"]},
    
    # Quantile models: regression only (quantile prediction)
    "quantile_lightgbm": {"nan_ok": True, "needs_tf": False, "experimental": False,
                         "supported_tasks": ["regression"]},
    
    # Specialized models
    "reward_based": {"nan_ok": False, "needs_tf": False, "experimental": False},
    "gmm_regime": {"nan_ok": False, "needs_tf": False, "experimental": True, "feature_emitter": False},
    "change_point": {"nan_ok": False, "needs_tf": False, "experimental": True, "feature_emitter": False},
    "ftrl_proximal": {"nan_ok": False, "needs_tf": False, "experimental": False,
                      "supported_tasks": ["binary"]},  # Online logistic regression
    "vae": {"nan_ok": False, "needs_tf": True, "experimental": True},
    "gan": {"nan_ok": False, "needs_tf": True, "experimental": True},
    "ensemble": {"nan_ok": False, "needs_tf": False, "experimental": False},
    "meta_learning": {"nan_ok": False, "needs_tf": True, "experimental": True},
    "multi_task": {"nan_ok": False, "needs_tf": True, "experimental": True},
    
    # Feature selection methods (not trainers, but need compatibility for importance extraction)
    "mutual_information": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "univariate_selection": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "rfe": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "boruta": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "stability_selection": {"nan_ok": True, "needs_tf": False, "experimental": False},
}


def is_family_compatible(family: str, task_type) -> tuple:
    """
    Check if family supports the given task type.
    
    This is the SINGLE SOURCE OF TRUTH for task-type filtering.
    Used by all 3 stages: TARGET_RANKING, FEATURE_SELECTION, TRAINING.
    
    Args:
        family: Model family name (will be normalized)
        task_type: TaskType enum, string, or None
    
    Returns:
        Tuple of (is_compatible: bool, skip_reason: Optional[str])
        - (True, None) if compatible or no restriction
        - (False, "unsupported_task:binary") if not compatible
    """
    if task_type is None:
        return True, None  # No task type specified = allow all
    
    normalized = normalize_family_name(family)
    caps = FAMILY_CAPS.get(normalized, {})
    supported = caps.get("supported_tasks")
    
    if supported is None:
        return True, None  # No restriction = all tasks allowed
    
    # Normalize task_type to string key
    if hasattr(task_type, 'name'):
        # TaskType enum: REGRESSION, BINARY_CLASSIFICATION, etc.
        task_str = task_type.name.lower()
    else:
        task_str = str(task_type).lower()
    
    # Map enum names to config keys
    # BINARY_CLASSIFICATION -> binary, MULTICLASS_CLASSIFICATION -> multiclass
    task_key = task_str.replace("_classification", "")
    
    if task_key in supported:
        return True, None
    else:
        return False, f"unsupported_task:{task_key}"


# ===========================================================================
# Sequence Building Utilities
# ===========================================================================


def _create_rolling_windows(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Create rolling windows using stride tricks (zero-copy view).

    Args:
        arr: (N, C) array where N is samples and C is channels
        window_size: Window length (must be <= N)

    Returns:
        (N - window_size + 1, window_size, C) array

    Note:
        Uses np.lib.stride_tricks.sliding_window_view for efficiency.
        For 2D arrays, we transpose to get (N', window, C) from (N', C, window).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    if arr.ndim == 1:
        # 1D array: return (N - window_size + 1, window_size)
        return sliding_window_view(arr, window_shape=window_size)
    else:
        # 2D array: sliding_window_view returns (N - window + 1, C, window)
        # We need (N - window + 1, window, C) for sequence models
        windows = sliding_window_view(arr, window_shape=window_size, axis=0)
        # Transpose last two dimensions: (N', C, window) -> (N', window, C)
        return np.transpose(windows, (0, 2, 1))


def _normalize_ohlcv_sequence(
    seq: np.ndarray,
    method: str = "returns"
) -> np.ndarray:
    """
    Normalize a single OHLCV sequence.

    This is the SINGLE SOURCE OF TRUTH for OHLCV normalization.
    Used by both TRAINING (sequence building) and LIVE_TRADING (inference).

    Args:
        seq: (seq_len, 5) array of [O, H, L, C, V]
        method: Normalization method:
            - "returns": (price / first_close) - 1.0 for prices,
                        (volume / first_volume) - 1.0 for volume
            - "log_returns": log(price / first_close) for prices,
                            log(volume / first_volume) for volume
            - "minmax": Per-channel [0, 1] scaling within the sequence
            - "none": No normalization (raw prices/volumes)

    Returns:
        Normalized (seq_len, 5) array

    Notes:
        - Price columns (O, H, L, C) are normalized relative to first bar's close
        - Volume is normalized separately (different magnitude from prices)
        - Epsilon (1e-8) prevents division by zero

    Contract (INTEGRATION_CONTRACTS.md v1.3):
        LIVE_TRADING must use the SAME normalization method stored in
        model_meta.sequence_normalization to ensure inference consistency.
    """
    eps = 1e-8

    # "none" normalization works with any number of channels
    if method == "none":
        return seq.copy()

    # All other normalization methods require exactly 5 channels (OHLCV)
    if seq.shape[1] != 5:
        raise ValueError(
            f"Expected 5 channels (OHLCV) for normalization='{method}', got {seq.shape[1]}. "
            f"Use normalization='none' for custom channel configurations."
        )

    if method == "returns":
        # Price channels: relative to first close
        first_close = seq[0, 3] + eps  # close is index 3
        prices = seq[:, :4]  # O, H, L, C
        prices_norm = (prices / first_close) - 1.0

        # Volume: relative to first volume
        first_vol = seq[0, 4] + eps
        vol_norm = (seq[:, 4] / first_vol) - 1.0

        return np.column_stack([prices_norm, vol_norm])

    elif method == "log_returns":
        first_close = seq[0, 3] + eps
        prices = seq[:, :4]
        prices_norm = np.log((prices / first_close) + eps)

        first_vol = seq[0, 4] + eps
        vol_norm = np.log((seq[:, 4] / first_vol) + eps)

        return np.column_stack([prices_norm, vol_norm])

    elif method == "minmax":
        # Per-channel min-max scaling to [0, 1]
        mins = seq.min(axis=0, keepdims=True)
        maxs = seq.max(axis=0, keepdims=True)
        return (seq - mins) / (maxs - mins + eps)

    else:
        raise ValueError(
            f"Unknown normalization method: {method}. "
            f"Supported: 'returns', 'log_returns', 'minmax', 'none'"
        )


def _detect_timestamp_gaps(
    timestamps: np.ndarray,
    expected_interval_seconds: int,
    tolerance: float = 1.5
) -> np.ndarray:
    """
    Detect gaps in timestamp series.

    Args:
        timestamps: Array of timestamps (datetime64 or compatible)
        expected_interval_seconds: Expected interval between bars in seconds
        tolerance: Gap detection tolerance multiplier (default 1.5 = 50% tolerance)

    Returns:
        Boolean array of length N-1, True where gap detected between timestamps[i] and timestamps[i+1]
    """
    if len(timestamps) < 2:
        return np.array([], dtype=bool)

    # Convert to seconds for diff calculation
    ts_seconds = timestamps.astype('datetime64[s]').astype(np.int64)
    diffs = np.diff(ts_seconds)

    return diffs > (expected_interval_seconds * tolerance)


def _split_on_gaps(
    data: np.ndarray,
    gap_mask: np.ndarray
) -> list:
    """
    Split array into contiguous segments based on gap mask.

    Args:
        data: (N, ...) array to split
        gap_mask: Boolean array of length N-1 indicating gaps

    Returns:
        List of arrays, each a contiguous segment
    """
    if len(gap_mask) == 0 or not gap_mask.any():
        return [data]

    split_indices = np.where(gap_mask)[0] + 1
    return np.split(data, split_indices)


def build_sequences_from_ohlcv(
    df,
    seq_len: int = None,
    seq_len_minutes: int = None,
    interval_minutes: int = None,
    channels: list = None,
    normalization: str = "returns",
    symbol: str = None,
    handle_gaps: bool = True,
    gap_tolerance: float = 1.5,
    auto_clamp: bool = False,
) -> tuple:
    """
    Build sequences from raw OHLCV data for sequence model training.

    This function creates rolling windows of raw price bars for training
    sequence models (Transformer, LSTM, CNN1D) without computed features.

    Args:
        df: DataFrame with columns [ts, open, high, low, close, volume, ...]
            MUST be sorted by timestamp (validated).
        seq_len: Sequence length in bars (mutually exclusive with seq_len_minutes)
        seq_len_minutes: Sequence length in minutes (preferred, interval-agnostic)
        interval_minutes: Data interval for conversion (required if seq_len_minutes)
        channels: OHLCV columns to use. Default: ["open", "high", "low", "close", "volume"]
        normalization: Normalization method ("returns", "log_returns", "minmax", "none")
        symbol: Optional symbol for logging
        handle_gaps: If True, split sequences at time gaps (don't span gaps)
        gap_tolerance: Gap detection tolerance multiplier (default 1.5)
        auto_clamp: If True and gap_handling="split", clamp seq_len to fit within
            a single US trading session (390 minutes). Uses 90% of max bars to
            allow rolling windows. Prevents 0-sequence output when seq_len
            exceeds session length.

    Returns:
        Tuple of:
            X: (N, seq_len, num_channels) sequences
            timestamps: (N,) timestamp of last bar in each sequence
            indices: (N,) original DataFrame indices for alignment with targets

    Raises:
        ValueError: If df is not sorted by timestamp
        ValueError: If required columns missing
        ValueError: If both seq_len and seq_len_minutes provided

    DRY/SST Compliance:
        - Uses minutes_to_bars() for interval conversion
        - Uses get_cfg() for config defaults
        - Sorted iteration for determinism
        - No random sampling

    Contract (INTEGRATION_CONTRACTS.md v1.3):
        When input_mode="raw_sequence":
        - model_meta.sequence_length = seq_len
        - model_meta.sequence_channels = channels
        - model_meta.sequence_normalization = normalization
    """
    import logging
    local_logger = logging.getLogger(__name__)

    # Default channels
    if channels is None:
        if _CONFIG_AVAILABLE:
            channels = get_cfg(
                "pipeline.sequence.default_channels",
                default=["open", "high", "low", "close", "volume"]
            )
        else:
            channels = ["open", "high", "low", "close", "volume"]

    # Validate mutual exclusivity
    if seq_len is not None and seq_len_minutes is not None:
        raise ValueError(
            "Cannot specify both seq_len and seq_len_minutes. "
            "Use seq_len_minutes for interval-agnostic configuration."
        )

    # Resolve sequence length
    if seq_len is None and seq_len_minutes is None:
        if _CONFIG_AVAILABLE:
            seq_len_minutes = get_cfg("pipeline.sequence.default_length_minutes", default=320)
        else:
            seq_len_minutes = 320  # ~64 bars @ 5m

    if seq_len_minutes is not None:
        if interval_minutes is None:
            if _CONFIG_AVAILABLE:
                interval_minutes = get_cfg("pipeline.data.interval_minutes", default=5)
            else:
                interval_minutes = 5
                local_logger.warning(
                    f"No interval_minutes provided, using default={interval_minutes}m. "
                    f"Pass interval_minutes for explicit control."
                )
        from TRAINING.common.interval import minutes_to_bars
        seq_len = minutes_to_bars(seq_len_minutes, interval_minutes)
        local_logger.debug(f"Derived seq_len={seq_len} bars from {seq_len_minutes}m @ {interval_minutes}m")

    # Auto-clamp seq_len to fit within a single US trading session
    if auto_clamp and handle_gaps and interval_minutes is not None:
        session_minutes = 390  # US market: 9:30-16:00
        max_bars = session_minutes // interval_minutes
        clamped = int(max_bars * 0.9)
        if seq_len > clamped:
            symbol_str = f" [{symbol}]" if symbol else ""
            local_logger.warning(
                f"Auto-clamp{symbol_str}: seq_len={seq_len} exceeds session capacity "
                f"(max_bars={max_bars} @ {interval_minutes}m, 90%={clamped}). "
                f"Clamping to {clamped} bars."
            )
            seq_len = clamped

    # Validate required columns
    required_cols = ["ts"] + channels
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. DataFrame has: {list(df.columns)}")

    # Validate sorted by timestamp (determinism requirement)
    ts_values = df["ts"].values
    if len(ts_values) > 1:
        diffs = np.diff(ts_values.astype('datetime64[ns]').astype(np.int64))
        if (diffs < 0).any():
            raise ValueError(
                "DataFrame must be sorted by timestamp for deterministic sequence building. "
                "Use df.sort_values('ts') before calling this function."
            )

    symbol_str = f" [{symbol}]" if symbol else ""
    local_logger.info(
        f"Building OHLCV sequences{symbol_str}: seq_len={seq_len}, "
        f"channels={channels}, normalization={normalization}"
    )

    # Extract OHLCV data
    ohlcv_data = df[channels].values.astype(np.float32)
    timestamps = df["ts"].values
    original_indices = df.index.values

    N = len(ohlcv_data)
    if N < seq_len:
        local_logger.warning(
            f"Insufficient data{symbol_str}: {N} rows < seq_len={seq_len}. "
            f"Returning empty sequences."
        )
        return (
            np.zeros((0, seq_len, len(channels)), dtype=np.float32),
            np.array([], dtype=timestamps.dtype),
            np.array([], dtype=original_indices.dtype),
        )

    # Handle gaps: split data into contiguous segments
    if handle_gaps and interval_minutes is not None:
        gap_mask = _detect_timestamp_gaps(
            timestamps,
            expected_interval_seconds=interval_minutes * 60,
            tolerance=gap_tolerance,
        )

        if gap_mask.any():
            n_gaps = gap_mask.sum()
            local_logger.info(f"Detected {n_gaps} time gaps{symbol_str}; splitting sequences")

            # Split all arrays at gaps
            ohlcv_segments = _split_on_gaps(ohlcv_data, gap_mask)
            ts_segments = _split_on_gaps(timestamps, gap_mask)
            idx_segments = _split_on_gaps(original_indices, gap_mask)

            # Build sequences from each segment, then concatenate
            all_X = []
            all_ts = []
            all_idx = []

            for seg_ohlcv, seg_ts, seg_idx in zip(ohlcv_segments, ts_segments, idx_segments):
                if len(seg_ohlcv) >= seq_len:
                    windows = _create_rolling_windows(seg_ohlcv, seq_len)
                    # Normalize each window
                    normalized = np.array([
                        _normalize_ohlcv_sequence(w, normalization)
                        for w in windows
                    ], dtype=np.float32)

                    # Last timestamp and index for each window
                    win_ts = seg_ts[seq_len - 1:]
                    win_idx = seg_idx[seq_len - 1:]

                    all_X.append(normalized)
                    all_ts.append(win_ts)
                    all_idx.append(win_idx)

            if all_X:
                X = np.concatenate(all_X, axis=0)
                out_ts = np.concatenate(all_ts, axis=0)
                out_idx = np.concatenate(all_idx, axis=0)
            else:
                X = np.zeros((0, seq_len, len(channels)), dtype=np.float32)
                out_ts = np.array([], dtype=timestamps.dtype)
                out_idx = np.array([], dtype=original_indices.dtype)

            local_logger.info(f"Built {len(X)} sequences from {len(ohlcv_segments)} segments{symbol_str}")
            return X, out_ts, out_idx

    # No gaps or gap handling disabled: build sequences from full array
    windows = _create_rolling_windows(ohlcv_data, seq_len)

    # Normalize each window
    X = np.array([
        _normalize_ohlcv_sequence(w, normalization)
        for w in windows
    ], dtype=np.float32)

    # Timestamps and indices correspond to last bar in each window
    out_ts = timestamps[seq_len - 1:]
    out_idx = original_indices[seq_len - 1:]

    local_logger.info(f"Built {len(X)} sequences{symbol_str}")
    return X, out_ts, out_idx


def build_sequences_from_features(X, lookback=None, lookback_minutes=None, interval_minutes=None):
    """
    Convert 2D features (N, F) to 3D sequences (N', T, F) using rolling windows.

    Args:
        X: (N, F) feature matrix
        lookback: sequence length T in bars (DEPRECATED, use lookback_minutes)
        lookback_minutes: sequence length in minutes (preferred, interval-agnostic)
        interval_minutes: data interval in minutes (required if lookback_minutes set)

    Returns:
        X_seq: (N', T, F) where N' = N - lookback + 1
    """
    import logging
    logger = logging.getLogger(__name__)

    # Resolve lookback from config if not provided
    if lookback is None and lookback_minutes is None:
        if _CONFIG_AVAILABLE:
            # Prefer time-based lookback_minutes
            lookback_minutes = get_cfg("pipeline.sequential.lookback_minutes", default=None)
            if lookback_minutes is None:
                # Fallback to bar-based
                lookback = int(get_cfg("pipeline.sequential.default_lookback", default=64))
        else:
            lookback = 64

    # Convert lookback_minutes to bars if provided
    if lookback_minutes is not None:
        if interval_minutes is None:
            if _CONFIG_AVAILABLE:
                interval_minutes = get_cfg("pipeline.data.interval_minutes", default=5)
            else:
                interval_minutes = 5
                logger.warning(
                    f"No interval_minutes provided, using default={interval_minutes}m. "
                    f"Pass interval_minutes for explicit control."
                )
        from TRAINING.common.interval import minutes_to_bars
        lookback = minutes_to_bars(lookback_minutes, interval_minutes)
        logger.debug(f"Derived lookback={lookback} bars from {lookback_minutes}m @ {interval_minutes}m")
    N, F = X.shape
    if N <= lookback:
        # If not enough data, pad with zeros
        X_seq = np.zeros((1, lookback, F), dtype=np.float32)
        X_seq[0, :N, :] = X
        return X_seq
    
    # Create rolling windows
    X_seq = np.zeros((N - lookback + 1, lookback, F), dtype=np.float32)
    for i in range(N - lookback + 1):
        X_seq[i] = X[i:i + lookback]
    
    return X_seq

def tf_available():
    """Check if TensorFlow is available.
    
    Note: This is a lenient check - we only verify the module can be imported.
    Full initialization happens in child processes, so we don't need to verify
    GPU availability or full library loading here.
    """
    try:
        import tensorflow as tf  # noqa: F401
        return True
    except (ImportError, Exception):
        # Catch all exceptions - TensorFlow might fail to import for various reasons
        # (library loading, CUDA issues, etc.) but child processes will handle it
        return False

def ngboost_available():
    """Check if NGBoost is available."""
    try:
        import ngboost  # noqa
        return True
    except Exception:
        return False

def pick_tf_device():
    """Dynamically detect best TensorFlow device (GPU if available, CPU otherwise)."""
    if os.getenv("CPU_ONLY", "0") == "1":
        return "/CPU:0"
    try:
        import tensorflow as tf
        gpus = tf.config.list_logical_devices("GPU")
        return "/GPU:0" if gpus else "/CPU:0"
    except Exception:
        return "/CPU:0"

TF_DEVICE = pick_tf_device()
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"TF device: {TF_DEVICE} | GPUs: {len(gpus)}")
except Exception:
    logger.info(f"TF device: {TF_DEVICE}")


