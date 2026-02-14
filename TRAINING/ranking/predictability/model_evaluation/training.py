# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Model Training and Evaluation Logic

This module contains the train_and_evaluate_models function and related helpers
for training multiple model families and extracting feature importances.

Extracted from model_evaluation.py as part of Phase 1 modular decomposition.
"""

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Target Predictability Ranking

Uses multiple model families to evaluate which of your 63 targets are most predictable.
This helps prioritize compute: train models on high-predictability targets first.

Methodology:
1. For each target, train multiple model families on sample data
2. Calculate predictability scores:
   - Model R² scores (cross-validated)
   - Feature importance magnitude (mean absolute SHAP/importance)
   - Consistency across models (low std = high confidence)
3. Rank targets by composite predictability score
4. Output ranked list with recommendations

Usage:
  # Rank all enabled targets
  python SCRIPTS/rank_target_predictability.py
  
  # Test on specific symbols first
  python SCRIPTS/rank_target_predictability.py --symbols AAPL,MSFT,GOOGL
  
  # Rank specific targets
  python SCRIPTS/rank_target_predictability.py --targets peak_60m,valley_60m,swing_high_15m
"""


import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np

# SST: Import View and Stage enums for consistent view/stage handling
from TRAINING.orchestration.utils.scope_resolution import View, Stage
# DETERMINISM: Import deterministic iteration helpers
from TRAINING.common.utils.determinism_ordering import sorted_unique
from dataclasses import dataclass
import yaml
import json
from collections import defaultdict
import warnings

# Add project root FIRST (before any scripts.* imports)
# TRAINING/ranking/rank_target_predictability.py -> parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Add CONFIG directory to path for centralized config loading
_CONFIG_DIR = _REPO_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_cfg, get_safety_config, get_experiment_config_path, load_experiment_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass  # Logger not yet initialized, will be set up below

# Import from modular components
from TRAINING.ranking.predictability.model_evaluation.config_helpers import get_importance_top_fraction as _get_importance_top_fraction

# Import logging config utilities
try:
    from CONFIG.logging_config_utils import get_module_logging_config, get_backend_logging_config
    _LOGGING_CONFIG_AVAILABLE = True
except ImportError:
    _LOGGING_CONFIG_AVAILABLE = False
    # Fallback: create a simple config-like object
    class _DummyLoggingConfig:
        def __init__(self):
            self.gpu_detail = False
            self.cv_detail = False
            self.edu_hints = False
            self.detail = False

# Import checkpoint utility (after path is set)
from TRAINING.orchestration.utils.checkpoint import CheckpointManager

# Import unified task type system
from TRAINING.common.utils.task_types import (
    TaskType, TargetConfig, ModelConfig, 
    is_compatible, create_model_configs_from_yaml
)
from TRAINING.common.utils.task_metrics import evaluate_by_task, compute_composite_score
from TRAINING.ranking.utils.target_validation import validate_target, check_cv_compatibility

# Suppress expected warnings (harmless)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

# Setup logging with journald support
from TRAINING.orchestration.utils.logging_setup import setup_logging
logger = setup_logging(
    script_name="rank_target_predictability",
    level=logging.INFO,
    use_journald=True
)



# Import dependencies
from TRAINING.ranking.predictability.scoring import TargetPredictabilityScore
from TRAINING.ranking.predictability.composite_score import calculate_composite_score
from TRAINING.ranking.predictability.data_loading import load_sample_data, prepare_features_and_target, get_model_config
from TRAINING.ranking.predictability.leakage_detection import detect_leakage, _save_feature_importances, _log_suspicious_features, find_near_copy_features, _detect_leaking_features
from TRAINING.ranking.predictability.model_evaluation.leakage_helpers import detect_and_fix_leakage, LeakageArtifacts


# Import from modular components
from TRAINING.ranking.predictability.model_evaluation.leakage_helpers import compute_suspicion_score as _compute_suspicion_score


# Import from modular components
from TRAINING.ranking.predictability.model_evaluation.reporting import log_canonical_summary as _log_canonical_summary

# Import safety gate from modular components
from TRAINING.ranking.predictability.model_evaluation.safety import enforce_final_safety_gate as _enforce_final_safety_gate

# Import threading utilities for smart thread management
try:
    from TRAINING.common.threads import plan_for_family, thread_guard, default_threads
    _THREADING_UTILITIES_AVAILABLE = True
except ImportError:
    _THREADING_UTILITIES_AVAILABLE = False

# Initialize determinism system and get BASE_SEED
try:
    from TRAINING.common.determinism import init_determinism_from_config
    BASE_SEED = init_determinism_from_config()
except ImportError:
    # Fallback: load from config directly if determinism module not available
    try:
        from CONFIG.config_loader import get_cfg
        BASE_SEED = int(get_cfg("pipeline.determinism.base_seed", default=42, config_name="pipeline_config"))
    except Exception:
        BASE_SEED = 42  # Final fallback matches pipeline.yaml default

# NOTE: _enforce_final_safety_gate is now imported from the safety submodule (see import above)
# The function was extracted as part of Phase 1 modular decomposition.



def train_and_evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: TaskType,
    model_families: List[str] = None,
    multi_model_config: Dict[str, Any] = None,
    target_column: str = None,  # For leak reporting and horizon extraction
    data_interval_minutes: int = 5,  # Data bar interval (default: 5-minute bars)
    time_vals: Optional[np.ndarray] = None,  # Timestamps for each sample (for fold timestamp tracking)
    explicit_interval: Optional[Union[int, str]] = None,  # Explicit interval from config (for consistency)
    experiment_config: Optional[Any] = None,  # Optional ExperimentConfig (for data.bar_interval)
    output_dir: Optional[Path] = None,  # Optional output directory for stability snapshots
    resolved_config: Optional[Any] = None,  # NEW: ResolvedConfig with correct purge/embargo (post-pruning)
    dropped_tracker: Optional[Any] = None,  # NEW: Optional DroppedFeaturesTracker for telemetry
    view: Union[str, View] = View.CROSS_SECTIONAL,  # View enum or "CROSS_SECTIONAL" - View type for REPRODUCIBILITY structure
    symbol: Optional[str] = None,  # Symbol name for SYMBOL_SPECIFIC view
    run_identity: Optional[Any] = None,  # RunIdentity for snapshot storage
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], float, Dict[str, List[Tuple[str, float]]], Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """
    Train multiple models and return task-aware metrics + importance magnitude
    
    Args:
        X: Feature matrix
        y: Target array
        feature_names: List of feature names
        task_type: TaskType enum (REGRESSION, BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION)
        model_families: List of model family names to use
        multi_model_config: Multi-model config dict
    
    Returns:
        model_metrics: Dict of {model_name: {metric_name: value}} per model (full metrics)
        model_scores: Dict of {model_name: primary_score} per model (backward compat)
        mean_importance: Mean absolute feature importance
        all_suspicious_features: Dict of {model_name: [(feature, importance), ...]}
        all_feature_importances: Dict of {model_name: {feature: importance}}
        fold_timestamps: List of {fold_idx, train_start, train_end, test_start, test_end} per fold
        perfect_correlation_models: Set of model names that triggered perfect correlation warnings
    
    Always returns 7 values, even on error (returns empty dicts, 0.0, empty list, and empty set)
    """
    # Get logging config for this module (at function start)
    if _LOGGING_CONFIG_AVAILABLE:
        log_cfg = get_module_logging_config('rank_target_predictability')
        lgbm_backend_cfg = get_backend_logging_config('lightgbm')
        catboost_backend_cfg = get_backend_logging_config('catboost')
    else:
        log_cfg = _DummyLoggingConfig()
        lgbm_backend_cfg = type('obj', (object,), {'native_verbosity': -1, 'show_sparse_warnings': True})()
        catboost_backend_cfg = type('obj', (object,), {'native_verbosity': 1, 'show_sparse_warnings': True})()
    
    # Initialize return values (ensures we always return 6 values)
    model_metrics = {}
    model_scores = {}
    importance_magnitudes = []
    all_suspicious_features = {}  # {model_name: [(feature, importance), ...]}
    all_feature_importances = {}  # {model_name: {feature: importance}} for detailed export
    fold_timestamps = []  # List of fold timestamp info
    
    # SST: Extract universe_sig from run_identity for consistent artifact scoping
    # NOTE: This will be updated to canonical universe_sig_for_writes after resolved_data_config is available
    train_universe_sig = getattr(run_identity, 'dataset_signature', None) if run_identity else None
    
    # Timing instrumentation for all importance producers (SST: thresholds from config)
    import time
    timing_data = {}  # {model_family: elapsed_seconds}
    overall_start_time = time.time()
    
    # Load timing config (SST)
    timing_log_enabled = True
    timing_log_threshold_seconds = 1.0  # Only log if > 1 second
    try:
        from CONFIG.config_loader import get_cfg
        timing_log_enabled = get_cfg('preprocessing.multi_model_feature_selection.timing.enabled', default=True, config_name='preprocessing_config')
        timing_log_threshold_seconds = get_cfg('preprocessing.multi_model_feature_selection.timing.log_threshold_seconds', default=1.0, config_name='preprocessing_config')
    except Exception:
        pass  # Use defaults if config not available
    
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        import lightgbm as lgb
        from TRAINING.ranking.utils.purged_time_series_split import PurgedTimeSeriesSplit
        from TRAINING.ranking.utils.leakage_filtering import _extract_horizon, _load_leakage_config
        from TRAINING.ranking.utils.feature_pruning import quick_importance_prune
    except Exception as e:
        logger.warning(f"Failed to import required libraries: {e}")
        return {}, {}, 0.0, {}, {}, []
    
    # Helper function for CV with early stopping (for gradient boosting models)
    def cross_val_score_with_early_stopping(model, X, y, cv, scoring, early_stopping_rounds=None, n_jobs=1):
        # Load default early stopping rounds from config
        if early_stopping_rounds is None:
            if _CONFIG_AVAILABLE:
                try:
                    early_stopping_rounds = int(get_cfg("preprocessing.validation.early_stopping_rounds", default=50, config_name="preprocessing_config"))
                except Exception:
                    early_stopping_rounds = 50
            else:
                early_stopping_rounds = 50
        """
        Cross-validation with early stopping support for gradient boosting models.
        
        cross_val_score doesn't support early stopping callbacks, so we need a manual loop.
        This prevents overfitting by stopping when validation performance plateaus.
        """
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            try:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Clone model for this fold
                from sklearn.base import clone
                fold_model = clone(model)
                
                # Train with early stopping
                # Check if model supports early stopping (LightGBM/XGBoost)
                supports_eval_set = hasattr(fold_model, 'fit') and 'eval_set' in fold_model.fit.__code__.co_varnames
                supports_early_stopping = hasattr(fold_model, 'fit') and 'early_stopping_rounds' in fold_model.fit.__code__.co_varnames
                
                if supports_eval_set:
                    # LightGBM style: uses callbacks
                    # Check by module name for reliability (str(type()) can be fragile)
                    model_module = type(fold_model).__module__
                    if 'lightgbm' in model_module.lower():
                        fold_model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                        )
                    # XGBoost style: early_stopping_rounds is set in constructor (XGBoost 2.0+)
                    # Don't pass it to fit() - it's already in the model
                    elif 'xgboost' in model_module.lower():
                        import xgboost as xgb
                        # XGBoost 2.0+ has early_stopping_rounds in constructor, not fit()
                        # Check if model already has it set, otherwise use eval_set only
                        fold_model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )
                    else:
                        # Fallback: try eval_set without callbacks
                        fold_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                else:
                    # Standard fit for models without early stopping
                    fold_model.fit(X_train, y_train)
                
                # Evaluate on validation set
                if scoring == 'r2':
                    from sklearn.metrics import r2_score
                    y_pred = fold_model.predict(X_val)
                    score = r2_score(y_val, y_pred)
                elif scoring == 'roc_auc':
                    from sklearn.metrics import roc_auc_score
                    y_proba = fold_model.predict_proba(X_val)[:, 1] if hasattr(fold_model, 'predict_proba') else fold_model.predict(X_val)
                    if len(np.unique(y_val)) == 2:
                        score = roc_auc_score(y_val, y_proba)
                    else:
                        score = np.nan
                elif scoring == 'accuracy':
                    from sklearn.metrics import accuracy_score
                    y_pred = fold_model.predict(X_val)
                    score = accuracy_score(y_val, y_pred)
                else:
                    # Fallback to default scorer
                    from sklearn.metrics import get_scorer
                    scorer = get_scorer(scoring)
                    score = scorer(fold_model, X_val, y_val)
                
                scores.append(score)
            except Exception as e:
                logger.debug(f"  Fold {fold_idx + 1} failed: {e}")
                scores.append(np.nan)
        
        return np.array(scores)
    
    # NEW: Initialize dropped features tracker for telemetry
    if dropped_tracker is None:
        from TRAINING.ranking.utils.dropped_features_tracker import DroppedFeaturesTracker
        dropped_tracker = DroppedFeaturesTracker()
    
    # ARCHITECTURAL IMPROVEMENT: Pre-prune low-importance features before expensive training
    # This reduces noise and prevents "Curse of Dimensionality" issues
    # Drop features with < 0.01% cumulative importance using a fast LightGBM model
    original_feature_count = len(feature_names)
    # Load feature count threshold from config
    try:
        from CONFIG.config_loader import get_cfg
        feature_count_threshold = int(get_cfg("safety.leakage_detection.model_evaluation.feature_count_pruning_threshold", default=100, config_name="safety_config"))
    except Exception:
        feature_count_threshold = 100
    if original_feature_count > feature_count_threshold:  # Only prune if we have many features
        logger.info(f"  Pre-pruning features: {original_feature_count} features")
        
        # Determine task type string for pruning
        if task_type == TaskType.REGRESSION:
            task_str = 'regression'
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            task_str = 'classification'
        else:
            task_str = 'classification'
        
        try:
            # Generate deterministic seed for feature pruning based on target
            from TRAINING.common.determinism import stable_seed_from
            # Use target_column if available, otherwise use default
            target_for_seed = target_column if target_column else 'pruning'
            prune_seed = stable_seed_from([target_for_seed, 'feature_pruning'])
            
            # Load feature pruning config
            if _CONFIG_AVAILABLE:
                try:
                    cumulative_threshold = get_cfg("preprocessing.feature_pruning.cumulative_threshold", default=0.0001, config_name="preprocessing_config")
                    min_features = get_cfg("preprocessing.feature_pruning.min_features", default=50, config_name="preprocessing_config")
                    n_estimators = get_cfg("preprocessing.feature_pruning.n_estimators", default=50, config_name="preprocessing_config")
                except Exception:
                    cumulative_threshold = 0.0001
                    min_features = 50
                    n_estimators = 50
            else:
                cumulative_threshold = 0.0001
                min_features = 50
                n_estimators = 50
            
            X_pruned, feature_names_pruned, pruning_stats = quick_importance_prune(
                X, y, feature_names,
                cumulative_threshold=cumulative_threshold,
                min_features=min_features,
                task_type=task_str,
                n_estimators=n_estimators,
                seed=prune_seed
            )
            
            # NEW: Track pruning drops for telemetry with stage record
            if dropped_tracker is not None and 'dropped_features' in pruning_stats:
                # Get config provenance
                config_provenance_dict = {
                    "cumulative_threshold": cumulative_threshold,
                    "min_features": min_features,
                    "n_estimators": n_estimators,
                    "task_type": task_str
                }
                
                dropped_tracker.add_pruning_drops(
                    pruning_stats['dropped_features'],
                    pruning_stats,
                    input_features=feature_names,
                    output_features=feature_names_pruned,
                    config_provenance=config_provenance_dict
                )
            
            if pruning_stats.get('dropped_count', 0) > 0:
                logger.info(f"  ✅ Pruned: {original_feature_count} → {len(feature_names_pruned)} features "
                          f"(dropped {pruning_stats['dropped_count']} low-importance features)")
                
                # Check for duplicates before assignment
                if len(feature_names_pruned) != len(set(feature_names_pruned)):
                    # DETERMINISM: Use sorted_unique for deterministic iteration order
                    duplicates = [name for name in sorted_unique(feature_names_pruned) if feature_names_pruned.count(name) > 1]
                    logger.error(f"  🚨 DUPLICATE COLUMN NAMES in pruned features: {duplicates}")
                    raise ValueError(f"Duplicate feature names after pruning: {duplicates}")
                
                feature_names_before_prune = feature_names.copy()
                X = X_pruned
                feature_names = feature_names_pruned
                
                # Log feature set transition
                from TRAINING.ranking.utils.cross_sectional_data import _log_feature_set
                _log_feature_set("PRUNER_SELECTED", feature_names, previous_names=feature_names_before_prune, logger_instance=logger)
                
                # CRITICAL: Re-run gatekeeper after pruning (pruning can surface long-lookback features)
                # Pruning drops low-importance features, which might have been masking long-lookback features
                # We must re-enforce the lookback cap on the pruned set
                if resolved_config is not None:
                    logger.info(f"  🔄 Re-running gatekeeper on pruned feature set (pruning may have surfaced long-lookback features)")
                    
                    # Extract horizon from target_column or use resolved_config
                    target_horizon_minutes = resolved_config.horizon_minutes if resolved_config else None
                    if target_horizon_minutes is None and target_column:
                        try:
                            from TRAINING.ranking.utils.leakage_filtering import _extract_horizon, _load_leakage_config
                            leakage_config = _load_leakage_config()
                            target_horizon_minutes = _extract_horizon(target_column, leakage_config)
                        except Exception:
                            pass
                    
                    # Load config and compute policy cap
                    from TRAINING.ranking.utils.leakage_budget import load_lookback_budget_spec, compute_policy_cap_minutes
                    spec, spec_warnings = load_lookback_budget_spec("safety_config")
                    for spec_warning in spec_warnings:
                        logger.warning(f"Config validation: {spec_warning}")
                    policy_cap_result = compute_policy_cap_minutes(spec, target_horizon_minutes, data_interval_minutes)
                    
                    X, feature_names, gate_report = _enforce_final_safety_gate(
                        X, feature_names,
                        policy_cap_minutes=policy_cap_result.cap_minutes,
                        interval_minutes=data_interval_minutes,
                        feature_time_meta_map=resolved_config.feature_time_meta_map if hasattr(resolved_config, 'feature_time_meta_map') else None,
                        base_interval_minutes=resolved_config.base_interval_minutes if hasattr(resolved_config, 'base_interval_minutes') else None,
                        logger=logger,
                        dropped_tracker=dropped_tracker
                    )
                    
                    # Update resolved_config with gate_report
                    if gate_report.get("enforced_feature_set"):
                        resolved_config._gatekeeper_enforced = gate_report["enforced_feature_set"]
                    
                    logger.info(f"  ✅ After post-prune gatekeeper: {len(feature_names)} features remaining")
                    
                    # CRITICAL: Get EnforcedFeatureSet from gatekeeper (if available)
                    # This is the authoritative feature set after post-prune gatekeeper
                    post_prune_gatekeeper_enforced = None
                    if hasattr(resolved_config, '_gatekeeper_enforced'):
                        post_prune_gatekeeper_enforced = resolved_config._gatekeeper_enforced
                        # Validate that feature_names matches enforced.features
                        if feature_names != post_prune_gatekeeper_enforced.features:
                            logger.warning(
                                f"  ⚠️ Post-prune gatekeeper: feature_names != enforced.features. "
                                f"This indicates a bug - X was sliced but feature_names wasn't updated correctly."
                            )
                            # Fix it: use enforced.features (the truth)
                            feature_names = post_prune_gatekeeper_enforced.features.copy()
            else:
                logger.info(f"  No features pruned (all above threshold)")
                from TRAINING.ranking.utils.cross_sectional_data import _log_feature_set
                _log_feature_set("PRUNER_SELECTED", feature_names, previous_names=None, logger_instance=logger)
            
            # CRITICAL: Recompute resolved_config with feature_lookback_max from PRUNED features
            # This prevents paying 1440m purge for features we don't even use
            from TRAINING.ranking.utils.leakage_budget import compute_budget
            from TRAINING.ranking.utils.resolved_config import compute_feature_lookback_max, create_resolved_config
            
            # Get registry for lookback calculation
            registry = None
            try:
                from TRAINING.common.feature_registry import get_registry
                registry = get_registry()
            except Exception:
                pass
            
            # Compute budget from PRUNED feature set
            if resolved_config and resolved_config.horizon_minutes:
                budget, _, _ = compute_budget(
                    feature_names,
                    data_interval_minutes,
                    resolved_config.horizon_minutes,
                    registry=registry,
                    stage="pre_gatekeeper_prune_check"
                )
                resolved_config.feature_lookback_max_minutes = budget.max_feature_lookback_minutes
                
                # Enforce leakage policy after pruning (final feature set)
                # Design: purge covers feature lookback, embargo covers target horizon
                if resolved_config.purge_minutes is not None:
                    purge_minutes = resolved_config.purge_minutes
                    embargo_minutes = resolved_config.embargo_minutes if resolved_config.embargo_minutes is not None else purge_minutes
                    
                    # Load policy and buffer from config
                    policy = "strict"
                    buffer_minutes = 5.0  # Default
                    try:
                        from CONFIG.config_loader import get_cfg
                        policy = get_cfg("safety.leakage_detection.policy", default="drop", config_name="safety_config")
                        buffer_minutes = float(get_cfg("safety.leakage_detection.lookback_buffer_minutes", default=5.0, config_name="safety_config"))
                    except Exception:
                        pass
                    
                    # Constraint 1: purge must cover feature lookback
                    purge_required = budget.max_feature_lookback_minutes + buffer_minutes
                    purge_violation = purge_minutes < purge_required
                    
                    # Constraint 2: embargo must cover target horizon
                    # Guard: horizon_minutes may be None (e.g., for some target types)
                    if budget.horizon_minutes is not None:
                        embargo_required = budget.horizon_minutes + buffer_minutes
                        embargo_violation = embargo_minutes < embargo_required
                    else:
                        # If horizon is None, skip embargo validation (not applicable)
                        embargo_violation = False
                        embargo_required = None
                    
                    if purge_violation or embargo_violation:
                        violations = []
                        if purge_violation:
                            violations.append(
                                f"purge ({purge_minutes:.1f}m) < lookback_requirement ({purge_required:.1f}m) "
                                f"[max_lookback={budget.max_feature_lookback_minutes:.1f}m + buffer={buffer_minutes:.1f}m]"
                            )
                        if embargo_violation:
                            violations.append(
                                f"embargo ({embargo_minutes:.1f}m) < horizon_requirement ({embargo_required:.1f}m) "
                                f"[horizon={budget.horizon_minutes:.1f}m + buffer={buffer_minutes:.1f}m]"
                            )
                        
                        msg = f"🚨 LEAKAGE VIOLATION (post-pruning): {'; '.join(violations)}"
                        
                        if policy == "strict":
                            raise RuntimeError(msg + " (policy: strict - training blocked)")
                        elif policy == "warn":
                            logger.error(msg + " (policy: warn - continuing with violation - NOT RECOMMENDED)")
                        # Note: drop_features policy already handled in gatekeeper, so we just warn here
                    elif embargo_required is None:
                        # Log that embargo validation was skipped due to missing horizon
                        logger.debug(f"   ℹ️  Embargo validation skipped: horizon_minutes is None (not applicable for this target type)")
            
            # Get n_symbols_available from mtf_data
            n_symbols_available = len(mtf_data) if 'mtf_data' in locals() else 1
            
            # Load lookback cap from config
            # Priority: 1) lookback_budget_minutes (new explicit cap), 2) ranking_mode_max_lookback_minutes (legacy)
            max_lookback_cap = None
            try:
                from CONFIG.config_loader import get_cfg
                # Try new explicit cap first
                budget_cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
                if budget_cap_raw != "auto" and isinstance(budget_cap_raw, (int, float)):
                    max_lookback_cap = float(budget_cap_raw)
                    logger.debug(f"Using lookback_budget_minutes cap: {max_lookback_cap:.1f}m")
                else:
                    # Fallback to legacy ranking_mode_max_lookback_minutes
                    max_lookback_cap = get_cfg("safety.leakage_detection.ranking_mode_max_lookback_minutes", default=None, config_name="safety_config")
                    if max_lookback_cap is not None:
                        max_lookback_cap = float(max_lookback_cap)
                        logger.debug(f"Using ranking_mode_max_lookback_minutes cap: {max_lookback_cap:.1f}m (legacy)")
            except Exception:
                pass
            
            # Compute feature lookback from PRUNED features
            # Get fingerprint for validation
            from TRAINING.ranking.utils.cross_sectional_data import _log_feature_set, _compute_feature_fingerprint
            _log_feature_set("POST_PRUNE", feature_names, previous_names=None, logger_instance=logger)
            post_prune_fp, post_prune_order_fp = _compute_feature_fingerprint(feature_names, set_invariant=True)
            # CRITICAL: Store feature_names for invariant check later
            post_prune_feature_names = feature_names.copy()  # Store for later comparison
            
            # CRITICAL: Use lookback_budget_minutes cap (if set) for POST_PRUNE recompute
            # This ensures consistency with gatekeeper threshold
            lookback_budget_cap_for_recompute = None
            try:
                from CONFIG.config_loader import get_cfg
                budget_cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
                if budget_cap_raw != "auto" and isinstance(budget_cap_raw, (int, float)):
                    lookback_budget_cap_for_recompute = float(budget_cap_raw)
            except Exception:
                pass
            
            # Use lookback_budget_minutes cap if set, else use ranking_mode_max_lookback_minutes
            effective_cap = lookback_budget_cap_for_recompute if lookback_budget_cap_for_recompute is not None else max_lookback_cap
            
            # CRITICAL: Use apply_lookback_cap() to enforce (quarantine unknowns), not just compute
            # This ensures unknowns are dropped at POST_PRUNE, not just logged
            from TRAINING.ranking.utils.lookback_cap_enforcement import apply_lookback_cap
            from CONFIG.config_loader import get_cfg
            
            # Load policy
            policy = "drop"  # Default: drop (over_budget_action)
            try:
                policy = get_cfg("safety.leakage_detection.policy", default="drop", config_name="safety_config")
                over_budget_action = get_cfg("safety.leakage_detection.over_budget_action", default="drop", config_name="safety_config")
                # Use over_budget_action for behavior, policy for logging
                if over_budget_action == "hard_stop":
                    policy = "strict"
                elif over_budget_action == "drop":
                    policy = "drop"
            except Exception:
                pass
            
            # Enforce cap (this will quarantine unknowns in strict mode, drop them in drop mode)
            cap_result = apply_lookback_cap(
                features=feature_names,
                interval_minutes=data_interval_minutes,
                cap_minutes=effective_cap,
                policy=policy,
                stage="POST_PRUNE",
                registry=registry,
                log_mode="summary"
            )
            
            # CRITICAL: Convert to EnforcedFeatureSet (SST contract)
            enforced_post_prune = cap_result.to_enforced_set(stage="POST_PRUNE", cap_minutes=effective_cap)
            
            # PHASE 2: Create and store FeatureSet artifact for reuse (eliminates recomputation)
            post_prune_artifact = None
            if output_dir is not None:
                try:
                    from TRAINING.ranking.utils.feature_set_artifact import create_artifact_from_enforced
                    post_prune_artifact = create_artifact_from_enforced(
                        enforced_post_prune,
                        stage="POST_PRUNE",
                        # DETERMINISM: Use sorted() for deterministic iteration order in dict comprehension
                        removal_reasons={f: "pruned" for f in sorted(set(feature_names) - set(enforced_post_prune.features))}
                    )
                    # Save to target-first structure (targets/<target>/reproducibility/featureset_artifacts/)
                    if target_column:
                        # Find base run directory
                        base_output_dir = output_dir
                        from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
                        base_output_dir = get_run_root(base_output_dir)
                        
                        if base_output_dir.exists():
                            try:
                                from TRAINING.orchestration.utils.target_first_paths import (
                                    ensure_scoped_artifact_dir, ensure_target_structure
                                )
                                from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
                                target_clean = normalize_target_name(target_column)
                                ensure_target_structure(base_output_dir, target_clean)
                                
                                # CRITICAL FIX: Validate view matches symbol count before path construction
                                symbol_for_artifact = symbol if ('symbol' in locals() and symbol) else None
                                view_for_artifact = view
                                view_enum_for_artifact = View.from_string(view) if isinstance(view, str) else view
                                
                                # Determine number of symbols from available data
                                n_symbols_for_validation = 1  # Default assumption
                                if 'symbols_array' in locals() and symbols_array is not None and len(symbols_array) > 0:
                                    unique_symbols = set(symbols_array)
                                    n_symbols_for_validation = len(unique_symbols)
                                    # If single symbol and SYMBOL_SPECIFIC, derive symbol
                                    if view_enum_for_artifact == View.SYMBOL_SPECIFIC and symbol_for_artifact is None and n_symbols_for_validation == 1:
                                        symbol_for_artifact = list(unique_symbols)[0]
                                        logger.debug(f"Derived symbol={symbol_for_artifact} from symbols_array for POST_PRUNE artifact path")
                                
                                # CRITICAL: If view is SYMBOL_SPECIFIC but we have multiple symbols, force CROSS_SECTIONAL
                                if view_enum_for_artifact == View.SYMBOL_SPECIFIC and n_symbols_for_validation > 1:
                                    logger.warning(
                                        f"⚠️  Invalid view=SYMBOL_SPECIFIC for multi-symbol run (n_symbols={n_symbols_for_validation}) "
                                        f"when saving POST_PRUNE artifact. Forcing view=CROSS_SECTIONAL."
                                    )
                                    view_for_artifact = View.CROSS_SECTIONAL
                                    symbol_for_artifact = None  # Clear symbol for CROSS_SECTIONAL
                                
                                # CRITICAL: Use canonical universe_sig_for_writes if available, fallback to train_universe_sig
                                # This ensures featureset_artifacts uses the same universe_sig as cohort metadata
                                universe_sig_for_artifact = universe_sig_for_writes if 'universe_sig_for_writes' in locals() and universe_sig_for_writes else train_universe_sig
                                
                                target_artifact_dir = ensure_scoped_artifact_dir(
                                    base_output_dir, target_clean, "featureset_artifacts",
                                    view=view_for_artifact, symbol=symbol_for_artifact, universe_sig=universe_sig_for_artifact,
                                    stage=Stage.TARGET_RANKING  # Explicit stage for proper scoping
                                )
                                post_prune_artifact.save(target_artifact_dir)
                                logger.debug(f"Saved POST_PRUNE artifact to view-scoped location: {target_artifact_dir}")
                            except Exception as e2:
                                logger.debug(f"Failed to save POST_PRUNE artifact to target-first location: {e2}")
                    
                    # Target-first structure only - no legacy writes
                except Exception as e:
                    logger.debug(f"  ⚠️  Failed to persist POST_PRUNE artifact: {e}")
            
            # Extract results (for backward compatibility)
            safe_features_post_prune = enforced_post_prune.features  # Use enforced.features (the truth)
            quarantined_post_prune = list(enforced_post_prune.quarantined.keys()) + enforced_post_prune.unknown
            canonical_map_from_post_prune = enforced_post_prune.canonical_map
            
            # CRITICAL: Slice X immediately using enforced.features (no rediscovery)
            # The enforced.features list IS the authoritative order - X columns must match it
            if len(safe_features_post_prune) < len(feature_names):
                logger.info(
                    f"  🔄 POST_PRUNE enforcement: {len(feature_names)} → {len(safe_features_post_prune)} "
                    f"(quarantined={len(quarantined_post_prune)})"
                )
                # Slice X to match enforced.features
                if X is not None and len(X.shape) == 2:
                    # Build indices for safe features (enforced.features)
                    feature_indices = [i for i, f in enumerate(feature_names) if f in enforced_post_prune.features]
                    if feature_indices and len(feature_indices) == len(enforced_post_prune.features):
                        X = X[:, feature_indices]
                    else:
                        logger.warning(
                            f"  ⚠️ POST_PRUNE: Could not slice X (indices mismatch). "
                            f"Expected {len(enforced_post_prune.features)} features, got {len(feature_indices)} indices."
                        )
                feature_names = enforced_post_prune.features.copy()  # Use enforced.features (the truth)
                # Update fingerprint after enforcement
                post_prune_fp, post_prune_order_fp = _compute_feature_fingerprint(feature_names, set_invariant=True)
                post_prune_feature_names = feature_names.copy()  # Update stored list
                # Store EnforcedFeatureSet for downstream use
                post_prune_enforced = enforced_post_prune
                
                # CRITICAL: Hard-fail check: POST_PRUNE must have ZERO unknowns in strict mode
                # This is the contract: post-enforcement stages should never see unknowns
                if len(enforced_post_prune.unknown) > 0:
                    policy = "strict"
                    try:
                        from CONFIG.config_loader import get_cfg
                        policy = get_cfg("safety.leakage_detection.policy", default="drop", config_name="safety_config")
                    except Exception:
                        pass
                    
                    if policy == "strict":
                        error_msg = (
                            f"🚨 POST_PRUNE CONTRACT VIOLATION: {len(enforced_post_prune.unknown)} features have unknown lookback (inf). "
                            f"In strict mode, post-enforcement stages must have ZERO unknowns. "
                            f"These should have been quarantined at gatekeeper. "
                            f"Sample: {enforced_post_prune.unknown[:10]}"
                        )
                        logger.error(error_msg)
                        raise RuntimeError(f"{error_msg} (policy: strict - training blocked)")
                    else:
                        logger.warning(
                            f"⚠️ POST_PRUNE: {len(enforced_post_prune.unknown)} features have unknown lookback (inf). "
                            f"Policy={policy} allows this, but this is unexpected after enforcement."
                        )
                
                # CRITICAL: Boundary assertion - validate feature_names matches POST_PRUNE EnforcedFeatureSet
                from TRAINING.ranking.utils.lookback_policy import assert_featureset_hash
                try:
                    assert_featureset_hash(
                        label="POST_PRUNE",
                        expected=post_prune_enforced,
                        actual_features=feature_names,
                        logger_instance=logger,
                        allow_reorder=False  # Strict order check
                    )
                except RuntimeError as e:
                    # This should never happen if we used enforced.features.copy() above
                    logger.error(f"POST_PRUNE assertion failed (unexpected): {e}")
                    # Fix it: use enforced.features (the truth)
                    feature_names = post_prune_enforced.features.copy()
                    logger.info(f"Fixed: Updated feature_names to match post_prune_enforced.features")
            
            # Now compute lookback from SAFE features only (no unknowns)
            # Use the canonical map from enforcement (already computed)
            lookback_result = compute_feature_lookback_max(
                safe_features_post_prune, data_interval_minutes, max_lookback_cap_minutes=effective_cap,
                expected_fingerprint=post_prune_fp,
                stage="POST_PRUNE",
                canonical_lookback_map=canonical_map_from_post_prune  # Use same map from enforcement
            )
            # Handle dataclass return
            if hasattr(lookback_result, 'max_minutes'):
                computed_lookback = lookback_result.max_minutes
                top_offenders = lookback_result.top_offenders
                lookback_fingerprint = lookback_result.fingerprint
                canonical_map_from_post_prune = lookback_result.canonical_lookback_map if hasattr(lookback_result, 'canonical_lookback_map') else None
            else:
                # Tuple return (backward compatibility)
                computed_lookback, top_offenders = lookback_result
                lookback_fingerprint = None
                canonical_map_from_post_prune = None
            
            # Validate fingerprint
            if lookback_fingerprint and lookback_fingerprint != post_prune_fp:
                logger.error(
                    f"🚨 FINGERPRINT MISMATCH (POST_PRUNE): computed={lookback_fingerprint} != expected={post_prune_fp}"
                )
            
            if computed_lookback is not None:
                feature_lookback_max_minutes = computed_lookback
                
                # CRITICAL INVARIANT CHECK: max(lookback_map[features]) == actual_max_from_features
                # This prevents regression and ensures canonical map consistency
                if canonical_map_from_post_prune is not None:
                    from TRAINING.ranking.utils.leakage_budget import _feat_key
                    
                    # Extract lookbacks for current features from canonical map
                    feature_lookbacks_from_map = []
                    for feat_name in feature_names:
                        feat_key = _feat_key(feat_name)
                        lookback = canonical_map_from_post_prune.get(feat_key)
                        if lookback is not None and lookback != float("inf"):
                            feature_lookbacks_from_map.append(lookback)
                    
                    if feature_lookbacks_from_map:
                        max_from_map = max(feature_lookbacks_from_map)
                        # Allow small floating-point differences (1.0 minute tolerance)
                        if abs(max_from_map - computed_lookback) > 1.0:
                            error_msg = (
                                f"🚨 INVARIANT VIOLATION (POST_PRUNE): "
                                f"max(canonical_map[features])={max_from_map:.1f}m != "
                                f"computed_lookback={computed_lookback:.1f}m. "
                                f"This indicates canonical map inconsistency. "
                                f"Feature set: {len(feature_names)} features, "
                                f"canonical map entries: {len([k for k in canonical_map_from_post_prune.keys() if k in [_feat_key(f) for f in feature_names]])}"
                            )
                            logger.error(error_msg)
                            
                            # Hard-fail in strict mode
                            policy = "strict"
                            try:
                                from CONFIG.config_loader import get_cfg
                                policy = get_cfg("safety.leakage_detection.policy", default="drop", config_name="safety_config")
                            except Exception:
                                pass
                            
                            if policy == "strict":
                                raise RuntimeError(error_msg)
                        else:
                            logger.debug(
                                f"✅ INVARIANT CHECK (POST_PRUNE): "
                                f"max(canonical_map[features])={max_from_map:.1f}m == "
                                f"computed_lookback={computed_lookback:.1f}m ✓"
                            )
                
                # SANITY CHECK: Verify top_offenders matches reported max and is from current feature set
                if top_offenders:
                    actual_max_in_list = top_offenders[0][1]
                    current_feature_set = set(feature_names)
                    
                    # Verify all top features are in current feature set (should always be true now)
                    top_feature_names = {f for f, _ in top_offenders[:5]}
                    missing = top_feature_names - current_feature_set
                    if missing:
                        logger.error(
                            f"🚨 CRITICAL: Top lookback features not in current feature set: {missing}. "
                            f"This indicates top_offenders was built from wrong feature set."
                        )
                    
                    # Only warn about max mismatch if fingerprint validation passed (invariant-checked stage)
                    # For POST_PRUNE stage, mismatch is a real error if fingerprint matches
                    if lookback_fingerprint and lookback_fingerprint == post_prune_fp:
                        # This is an invariant-checked stage, so mismatch is a real error
                        if abs(actual_max_in_list - computed_lookback) > 1.0:
                            logger.error(
                                f"🚨 Lookback max mismatch (POST_PRUNE): reported={computed_lookback:.1f}m "
                                f"but top feature={actual_max_in_list:.1f}m. "
                                f"This indicates lookback computation bug."
                            )
                    
                    # CRITICAL: Update resolved_config with the recomputed lookback (from pruned features)
                    # This ensures the budget object reflects the actual feature set
                    if resolved_config is not None:
                        resolved_config.feature_lookback_max_minutes = computed_lookback
                    
                    # Log top features (only if > 4 hours for debugging)
                    if computed_lookback > 240:
                        fingerprint_str = lookback_fingerprint if lookback_fingerprint else (lookback_result.fingerprint if hasattr(lookback_result, 'fingerprint') else 'N/A')
                        logger.info(f"  📊 Feature lookback (POST_PRUNE): max={computed_lookback:.1f}m, fingerprint={fingerprint_str}, n_features={len(feature_names)}")
                        logger.info(f"    Top lookback features: {', '.join([f'{f}({m:.0f}m)' for f, m in top_offenders[:5]])}")
                        
                        # Check if lookback_budget_minutes cap is set
                        try:
                            from CONFIG.config_loader import get_cfg
                            budget_cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
                            if budget_cap_raw != "auto" and isinstance(budget_cap_raw, (int, float)):
                                budget_cap = float(budget_cap_raw)
                                if computed_lookback > budget_cap:
                                    exceeding_features = [(f, m) for f, m in top_offenders if m > budget_cap + 1.0]
                                    exceeding_count = len(exceeding_features)
                                    
                                    # CRITICAL: In strict mode, this is a hard-stop
                                    policy = "strict"
                                    try:
                                        from CONFIG.config_loader import get_cfg
                                        policy = get_cfg("safety.leakage_detection.policy", default="drop", config_name="safety_config")
                                    except Exception:
                                        pass
                                    
                                    error_msg = (
                                        f"🚨 POST_PRUNE CAP VIOLATION: actual_max={computed_lookback:.1f}m > cap={budget_cap:.1f}m. "
                                        f"Feature set contains {exceeding_count} features exceeding cap. "
                                        f"Gatekeeper should have dropped these features. "
                                        f"Top offenders: {', '.join([f'{f}({m:.0f}m)' for f, m in exceeding_features[:10]])}"
                                    )
                                    
                                    if policy == "strict":
                                        raise RuntimeError(error_msg + " (policy: strict - training blocked)")
                                    else:
                                        logger.error(error_msg + " (policy: warn - continuing with violation - NOT RECOMMENDED)")
                        except Exception:
                            pass
            else:
                feature_lookback_max_minutes = None
            
            # Recompute resolved_config with actual pruned feature lookback
            # This overrides the baseline config created earlier
            # CRITICAL: Use computed_lookback (from POST_PRUNE recompute) not feature_lookback_max_minutes variable
            # The variable might be stale if computed_lookback was None
            if resolved_config is not None:
                # Use the value we just computed and stored in resolved_config (line 867)
                # OR use feature_lookback_max_minutes if computed_lookback was None
                final_lookback = resolved_config.feature_lookback_max_minutes if resolved_config.feature_lookback_max_minutes is not None else feature_lookback_max_minutes
                
                # Override with post-prune config
                resolved_config = create_resolved_config(
                    requested_min_cs=resolved_config.requested_min_cs,
                    n_symbols_available=n_symbols_available,
                    max_cs_samples=resolved_config.max_cs_samples,
                    interval_minutes=resolved_config.interval_minutes,
                    horizon_minutes=resolved_config.horizon_minutes,
                    feature_lookback_max_minutes=final_lookback,  # Use final computed value
                    purge_buffer_bars=resolved_config.purge_buffer_bars,
                    default_purge_minutes=None,  # Loads from safety_config.yaml (SST)
                    features_safe=resolved_config.features_safe,
                    features_dropped_nan=resolved_config.features_dropped_nan,
                    features_final=len(feature_names),  # Updated count
                    view=resolved_config.view,
                    symbol=resolved_config.symbol,
                    feature_names=feature_names,  # Pruned features
                    recompute_lookback=False,  # Already computed above
                    experiment_config=experiment_config  # NEW: Pass experiment_config for base_interval_minutes
                )
                if log_cfg.cv_detail:
                    logger.info(f"  ✅ Resolved config (post-prune): purge={resolved_config.purge_minutes:.1f}m, embargo={resolved_config.embargo_minutes:.1f}m")
                
                # CRITICAL: Enforce leakage policy after pruning (final feature set)
                if resolved_config.purge_minutes is not None and resolved_config.feature_lookback_max_minutes is not None:
                    from TRAINING.ranking.utils.leakage_budget import compute_budget
                    
                    # Get registry
                    registry = None
                    try:
                        from TRAINING.common.feature_registry import get_registry
                        registry = get_registry()
                    except Exception:
                        pass
                    
                    # Compute budget from final pruned features
                    # CRITICAL: This is the ACTUAL budget for the final feature set
                    # Use lookback_budget_minutes cap (if set) for consistency
                    lookback_budget_cap_for_budget = None
                    budget_cap_provenance_budget = None
                    try:
                        from CONFIG.config_loader import get_cfg, get_config_path
                        budget_cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
                        config_path = get_config_path("safety_config")
                        budget_cap_provenance_budget = f"safety_config.yaml:{config_path} → safety.leakage_detection.lookback_budget_minutes = {budget_cap_raw} (default='auto')"
                        if budget_cap_raw != "auto" and isinstance(budget_cap_raw, (int, float)):
                            lookback_budget_cap_for_budget = float(budget_cap_raw)
                    except Exception as e:
                        budget_cap_provenance_budget = f"config lookup failed: {e}"
                    
                    # PHASE 2: Reuse POST_PRUNE artifact to eliminate recomputation
                    canonical_map_from_post_prune = None
                    if 'post_prune_artifact' in locals() and post_prune_artifact is not None:
                        # Use canonical map from artifact (single source of truth)
                        canonical_map_from_post_prune = post_prune_artifact.canonical_lookback_map
                        logger.debug(f"  ✅ POST_PRUNE_policy_check: Reusing canonical map from POST_PRUNE artifact (n_features={len(post_prune_artifact.features)})")
                    elif 'lookback_result' in locals() and hasattr(lookback_result, 'canonical_lookback_map'):
                        # Fallback: use from lookback_result (backward compatibility)
                        canonical_map_from_post_prune = lookback_result.canonical_lookback_map
                    elif 'lookback_result' in locals() and hasattr(lookback_result, 'lookback_map'):
                        # Backward compatibility
                        canonical_map_from_post_prune = lookback_result.lookback_map
                    
                    # If we don't have the canonical map, we MUST recompute using compute_feature_lookback_max
                    # to ensure we get the same result as POST_PRUNE
                    if canonical_map_from_post_prune is None:
                        logger.warning(
                            f"⚠️ POST_PRUNE_policy_check: No canonical map available from POST_PRUNE artifact. "
                            f"Recomputing using compute_feature_lookback_max to ensure consistency."
                        )
                        # Recompute using the same function as POST_PRUNE
                        lookback_result_for_policy = compute_feature_lookback_max(
                            feature_names,
                            data_interval_minutes,
                            max_lookback_cap_minutes=lookback_budget_cap_for_budget,
                            expected_fingerprint=post_prune_fp if 'post_prune_fp' in locals() else None,
                            stage="POST_PRUNE_policy_check",
                            registry=registry
                        )
                        if hasattr(lookback_result_for_policy, 'canonical_lookback_map'):
                            canonical_map_from_post_prune = lookback_result_for_policy.canonical_lookback_map
                        elif hasattr(lookback_result_for_policy, 'lookback_map'):
                            canonical_map_from_post_prune = lookback_result_for_policy.lookback_map
                    
                    # Log config trace for budget compute (only if not using artifact)
                    if 'post_prune_artifact' not in locals() or post_prune_artifact is None:
                        logger.info(f"📋 CONFIG TRACE (POST_PRUNE_policy_check budget): {budget_cap_provenance_budget}")
                        logger.info(f"   → max_lookback_cap_minutes passed to compute_budget: {lookback_budget_cap_for_budget}")
                        logger.info(f"   → Computing budget from {len(feature_names)} features, expected fingerprint: {post_prune_fp if 'post_prune_fp' in locals() else 'None'}")
                        logger.info(f"   → Using canonical map from POST_PRUNE: {'YES' if canonical_map_from_post_prune is not None else 'NO (will recompute)'}")
                    
                    budget, budget_fp, budget_order_fp = compute_budget(
                        feature_names,
                        data_interval_minutes,
                        resolved_config.horizon_minutes,
                        registry=registry,
                        max_lookback_cap_minutes=lookback_budget_cap_for_budget,  # Pass cap to compute_budget
                        expected_fingerprint=post_prune_fp if 'post_prune_fp' in locals() else None,
                        stage="POST_PRUNE_policy_check",
                        canonical_lookback_map=canonical_map_from_post_prune,  # CRITICAL: Use same map as POST_PRUNE (from artifact if available)
                        feature_time_meta_map=resolved_config.feature_time_meta_map if resolved_config and hasattr(resolved_config, 'feature_time_meta_map') else None,
                        base_interval_minutes=resolved_config.base_interval_minutes if resolved_config else None
                    )
                    
                    # Log the computed budget for debugging
                    logger.info(f"   → Budget computed: actual_max={budget.max_feature_lookback_minutes:.1f}m, cap={budget.cap_max_lookback_minutes}, fingerprint={budget_fp}")
                    
                    # CRITICAL: Update resolved_config with the NEW budget (from pruned features)
                    # This ensures budget.actual_max reflects the actual feature set
                    resolved_config.feature_lookback_max_minutes = budget.max_feature_lookback_minutes
                    
                    # Validate fingerprint
                    if 'post_prune_fp' in locals() and budget_fp != post_prune_fp:
                        logger.error(
                            f"🚨 FINGERPRINT MISMATCH (POST_PRUNE_policy_check): budget={budget_fp} != expected={post_prune_fp}"
                        )
                    purge_minutes = resolved_config.purge_minutes
                    embargo_minutes = resolved_config.embargo_minutes if resolved_config.embargo_minutes is not None else purge_minutes
                    
                    # Load policy and buffer from config
                    policy = "strict"
                    buffer_minutes = 5.0  # Default
                    try:
                        from CONFIG.config_loader import get_cfg
                        policy = get_cfg("safety.leakage_detection.policy", default="drop", config_name="safety_config")
                        buffer_minutes = float(get_cfg("safety.leakage_detection.lookback_buffer_minutes", default=5.0, config_name="safety_config"))
                    except Exception:
                        pass
                    
                    # Constraint 1: purge must cover feature lookback
                    purge_required = budget.max_feature_lookback_minutes + buffer_minutes
                    purge_violation = purge_minutes < purge_required
                    
                    # Constraint 2: embargo must cover target horizon
                    # Guard: horizon_minutes may be None (e.g., for some target types)
                    if budget.horizon_minutes is not None:
                        embargo_required = budget.horizon_minutes + buffer_minutes
                        embargo_violation = embargo_minutes < embargo_required
                    else:
                        # If horizon is None, skip embargo validation (not applicable)
                        embargo_violation = False
                        embargo_required = None
                    
                    if purge_violation or embargo_violation:
                        violations = []
                        if purge_violation:
                            violations.append(
                                f"purge ({purge_minutes:.1f}m) < lookback_requirement ({purge_required:.1f}m) "
                                f"[max_lookback={budget.max_feature_lookback_minutes:.1f}m + buffer={buffer_minutes:.1f}m]"
                            )
                        if embargo_violation:
                            violations.append(
                                f"embargo ({embargo_minutes:.1f}m) < horizon_requirement ({embargo_required:.1f}m) "
                                f"[horizon={budget.horizon_minutes:.1f}m + buffer={buffer_minutes:.1f}m]"
                            )
                        
                        msg = f"🚨 LEAKAGE VIOLATION (post-pruning): {'; '.join(violations)}"
                        
                        if policy == "strict":
                            raise RuntimeError(msg + " (policy: strict - training blocked)")
                        elif policy == "warn":
                            logger.error(msg + " (policy: warn - continuing with violation - NOT RECOMMENDED)")
                        # Note: drop_features policy already handled in gatekeeper, so we just warn here
                    elif embargo_required is None:
                        # Log that embargo validation was skipped due to missing horizon
                        logger.debug(f"   ℹ️  Embargo validation skipped: horizon_minutes is None (not applicable for this target type)")
            
            # Save stability snapshot for quick pruning (non-invasive hook)
            # Only save if output_dir is available (optional feature)
            if 'full_importance_dict' in pruning_stats and output_dir is not None:
                try:
                    from TRAINING.stability.feature_importance import save_snapshot_hook
                    # Use target-first structure for snapshots
                    from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
                    target_clean = normalize_target_name(target_column if target_column else 'unknown')
                    
                    # Find base run directory
                    base_output_dir = output_dir
                    if base_output_dir:
                        from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
                        base_output_dir = get_run_root(base_output_dir)
                        
                        if base_output_dir.exists():
                            from TRAINING.orchestration.utils.target_first_paths import (
                                get_target_reproducibility_dir, ensure_target_structure
                            )
                            ensure_target_structure(base_output_dir, target_clean)
                            target_repro_dir = get_target_reproducibility_dir(base_output_dir, target_clean)
                            
                            # NOTE: Quick_pruner snapshots are DISABLED for production use.
                            # Reason: Quick_pruner runs BEFORE CV splits are created, so it cannot
                            # have a complete RunIdentity (missing split_signature). Saving it as a
                            # "stability snapshot" would be semantically incorrect - it's a pre-CV
                            # preprocessing step, not a model evaluation result.
                            # The pruning stats are still computed and used internally; we just
                            # don't persist them to the identity-tracked snapshot system.
                            logger.debug(f"Quick_pruner stats computed (not saved to stability snapshots - pre-CV data)")
                except Exception as e:
                    logger.debug(f"Stability snapshot save failed for quick_pruner (non-critical): {e}")
        except RuntimeError as e:
            # CRITICAL: Re-raise RuntimeError (strict mode violations, etc.)
            # These are safety-critical and should not be swallowed
            if "policy: strict" in str(e) or "training blocked" in str(e):
                logger.error(f"  🚨 Feature pruning failed with strict policy violation: {e}")
                raise  # Re-raise - strict mode violations must abort
            else:
                # Use centralized error handling policy
                from TRAINING.common.exceptions import handle_error_with_policy
                handle_error_with_policy(
                    error=e,
                    stage="FEATURE_SELECTION",
                    error_type="feature_pruning",
                    affects_artifact=True,
                    affects_selection=True,
                    fallback_value=None  # Will use original features below
                )
        except Exception as e:
            # Use centralized error handling policy
            from TRAINING.common.exceptions import handle_error_with_policy
            handle_error_with_policy(
                error=e,
                stage="FEATURE_SELECTION",
                error_type="feature_pruning",
                affects_artifact=True,
                affects_selection=True,
                fallback_value=None  # Will use original features below
            )
            # Continue with original features (baseline resolved_config already assigned)
    
    # CRITICAL: Create resolved_config AFTER pruning (or if pruning skipped)
    # This ensures feature_lookback_max is computed from actual features used in training
    if resolved_config is None:
        from TRAINING.ranking.utils.resolved_config import compute_feature_lookback_max, create_resolved_config
        
        # Get n_symbols_available from cohort_context
        n_symbols_available = len(mtf_data) if 'mtf_data' in locals() else 1
        
        # Load ranking mode cap from config
        max_lookback_cap = None
        try:
            from CONFIG.config_loader import get_cfg
            max_lookback_cap = get_cfg("safety.leakage_detection.ranking_mode_max_lookback_minutes", default=None, config_name="safety_config")
            if max_lookback_cap is not None:
                max_lookback_cap = float(max_lookback_cap)
        except Exception:
            pass
        
        # Compute feature lookback from actual features (pruned or unpruned)
        from TRAINING.ranking.utils.cross_sectional_data import _compute_feature_fingerprint
        current_fp, current_order_fp = _compute_feature_fingerprint(feature_names, set_invariant=True)
        
        # CRITICAL INVARIANT CHECK: Verify featureset matches POST_PRUNE (if it exists)
        # This detects featureset mis-wire: if feature_names changed between POST_PRUNE and strict check
        if 'post_prune_fp' in locals() and post_prune_fp is not None:
            # Use reusable invariant check helper (if EnforcedFeatureSet available)
            if 'post_prune_enforced' in locals():
                from TRAINING.ranking.utils.lookback_policy import assert_featureset_hash
                assert_featureset_hash(
                    label="MODEL_TRAIN_INPUT",
                    expected=post_prune_enforced,
                    actual_features=feature_names,
                    logger_instance=logger,
                    allow_reorder=False  # Strict order check (default)
                )
            else:
                # Fallback: manual check (for backward compatibility)
                # Check exact list equality first (not just hash)
                if 'post_prune_feature_names' in locals() and feature_names == post_prune_feature_names:
                    logger.debug(
                        f"✅ INVARIANT CHECK PASSED: exact list match, n_features={len(feature_names)}"
                    )
                elif current_fp != post_prune_fp:
                    logger.error(
                        f"🚨 FEATURESET MIS-WIRE DETECTED: current fingerprint={current_fp[:16]} != POST_PRUNE={post_prune_fp[:16]}. "
                        f"Feature list passed to strict check differs from POST_PRUNE. "
                        f"Current n_features={len(feature_names)}, POST_PRUNE fingerprint={post_prune_fp[:16]}. "
                        f"This indicates feature_names was modified or wrong variable passed."
                    )
                    # Log sample differences for debugging
                    if 'post_prune_feature_names' in locals():
                        current_set = set(feature_names)
                        post_prune_set = set(post_prune_feature_names)
                        added = current_set - post_prune_set
                        removed = post_prune_set - current_set
                        if added:
                            logger.error(f"   Added features: {list(added)[:10]}")
                        if removed:
                            logger.error(f"   Removed features: {list(removed)[:10]}")
                        # Check order divergence
                        if not added and not removed and len(feature_names) == len(post_prune_feature_names):
                            for i, (exp, act) in enumerate(zip(post_prune_feature_names, feature_names)):
                                if exp != act:
                                    logger.error(
                                        f"   Order divergence at index {i}: expected={exp}, actual={act}"
                                    )
                                    break
                    raise RuntimeError(
                        f"FEATURESET MIS-WIRE: feature_names passed to strict check (fingerprint={current_fp[:16]}) "
                        f"does not match POST_PRUNE (fingerprint={post_prune_fp[:16]}). "
                        f"This indicates a bug: feature list was modified or wrong variable passed."
                    )
                else:
                    logger.debug(
                        f"✅ INVARIANT CHECK PASSED: current fingerprint={current_fp[:16]} == POST_PRUNE={post_prune_fp[:16]}, "
                        f"n_features={len(feature_names)}"
                    )
        
        lookback_result = compute_feature_lookback_max(
            feature_names, data_interval_minutes, max_lookback_cap_minutes=max_lookback_cap,
            expected_fingerprint=current_fp,
            stage="fallback_lookback_compute"
        )
        # Handle dataclass return
        if hasattr(lookback_result, 'max_minutes'):
            computed_lookback = lookback_result.max_minutes
            top_offenders = lookback_result.top_offenders
            lookback_fingerprint = lookback_result.fingerprint
        else:
            # Tuple return (backward compatibility)
            computed_lookback, top_offenders = lookback_result
            lookback_fingerprint = None
        
        # Validate fingerprint (only if we have it)
        if lookback_fingerprint and lookback_fingerprint != current_fp:
            logger.error(
                f"🚨 FINGERPRINT MISMATCH (fallback): computed={lookback_fingerprint} != expected={current_fp}"
            )
        
        if computed_lookback is not None:
            feature_lookback_max_minutes = computed_lookback
            # SANITY CHECK: Verify top_offenders matches reported max and is from current feature set
            if top_offenders:
                actual_max_in_list = top_offenders[0][1]
                current_feature_set = set(feature_names)
                
                # Verify all top features are in current feature set (should always be true now)
                top_feature_names = {f for f, _ in top_offenders[:5]}
                missing = top_feature_names - current_feature_set
                if missing:
                    logger.error(
                        f"🚨 CRITICAL: Top lookback features not in current feature set: {missing}. "
                        f"This indicates top_offenders was built from wrong feature set."
                    )
                
                # Only warn about max mismatch if fingerprint validation passed (invariant-checked stage)
                # For fallback stage, mismatch might be expected
                if lookback_fingerprint and lookback_fingerprint == current_fp:
                    # This is an invariant-checked stage, so mismatch is a real error
                    if abs(actual_max_in_list - computed_lookback) > 1.0:
                        logger.error(
                            f"🚨 Lookback max mismatch (fallback): reported={computed_lookback:.1f}m "
                            f"but top feature={actual_max_in_list:.1f}m. "
                            f"This indicates lookback computation bug."
                        )
                
                # Log top features (only if > 4 hours for debugging)
                if computed_lookback > 240:
                    fingerprint_str = lookback_fingerprint if lookback_fingerprint else (lookback_result.fingerprint if hasattr(lookback_result, 'fingerprint') else 'N/A')
                    logger.info(f"  📊 Feature lookback analysis: max={computed_lookback:.1f}m, fingerprint={fingerprint_str}")
                    logger.info(f"    Top lookback features (from {len(feature_names)} features): {', '.join([f'{f}({m:.0f}m)' for f, m in top_offenders[:5]])}")
        else:
            # Fallback: use conservative estimate if cannot compute
            # Use time-based value (1 day = 1440 minutes) - interval-agnostic
            if data_interval_minutes is not None and data_interval_minutes > 0:
                feature_lookback_max_minutes = 1440.0  # 1 day conservative default
            else:
                feature_lookback_max_minutes = None
        
        # Extract horizon from target_column if available
        target_horizon_minutes = None
        if target_column:
            try:
                from TRAINING.ranking.utils.leakage_filtering import _extract_horizon, _load_leakage_config
                leakage_config = _load_leakage_config()
                target_horizon_minutes = _extract_horizon(target_column, leakage_config)
            except Exception:
                pass
        
        # Create resolved config with actual feature lookback
        resolved_config = create_resolved_config(
            requested_min_cs=1,  # Not used in train_and_evaluate_models context
            n_symbols_available=n_symbols_available,
            max_cs_samples=None,
            interval_minutes=data_interval_minutes,
            horizon_minutes=target_horizon_minutes,
            feature_lookback_max_minutes=feature_lookback_max_minutes,
            purge_buffer_bars=5,
            default_purge_minutes=None,  # Loads from safety_config.yaml (SST)
            features_safe=original_feature_count,
            features_dropped_nan=0,
            features_final=len(feature_names),
            view=View.CROSS_SECTIONAL,  # Default for train_and_evaluate_models
            symbol=None,
            feature_names=feature_names,
            recompute_lookback=False,  # Already computed above
            experiment_config=experiment_config  # NEW: Pass experiment_config for base_interval_minutes
        )
        
        if log_cfg.cv_detail:
            logger.info(f"  ✅ Resolved config created: purge={resolved_config.purge_minutes:.1f}m, embargo={resolved_config.embargo_minutes:.1f}m")
    
    # Get CV config (with fallback if multi_model_config is None or cross_validation is None)
    if multi_model_config is None:
        cv_config = {}
        # Try to load from config if multi_model_config not provided
        try:
            from CONFIG.config_loader import get_cfg
            folds = int(get_cfg("training.folds", default=3, config_name="intelligent_training_config"))
            cv_n_jobs = int(get_cfg("training.cv_n_jobs", default=1, config_name="intelligent_training_config"))
        except Exception:
            folds = 3
            cv_n_jobs = 1
    else:
        cv_config = multi_model_config.get('cross_validation', {})
        # Ensure cv_config is never None (handle case where key exists but value is None)
        if cv_config is None:
            cv_config = {}
        # SST: Try to get from config first, then fallback to cv_config or defaults
        try:
            from CONFIG.config_loader import get_cfg
            folds = int(get_cfg("training.folds", default=cv_config.get('folds', 3), config_name="intelligent_training_config"))
            cv_n_jobs = int(get_cfg("training.cv_n_jobs", default=cv_config.get('n_jobs', 1), config_name="intelligent_training_config"))
        except Exception:
            # Fallback to cv_config or defaults if config loader fails
            folds = cv_config.get('folds', 3)
            cv_n_jobs = cv_config.get('n_jobs', 1)
    
    # CRITICAL: Use PurgedTimeSeriesSplit to prevent temporal leakage
    # Standard K-Fold shuffles data randomly, which destroys time patterns
    # TimeSeriesSplit respects time order but doesn't prevent overlap leakage
    # PurgedTimeSeriesSplit enforces a gap between train/test = target horizon
    
    # Calculate purge_overlap based on target horizon
    # Extract target horizon (in minutes) from target column name
    leakage_config = _load_leakage_config()
    target_horizon_minutes = _extract_horizon(target_column, leakage_config) if target_column else None
    
    # Auto-detect data interval from timestamps if available, otherwise use parameter
    # CRITICAL: Using wrong interval causes data leakage (e.g., 1m data with 5m assumption leaks 4 minutes)
    if time_vals is not None and len(time_vals) > 1:
        try:
            # Convert to pandas Timestamp if needed
            # Handle both numeric (nanoseconds) and datetime timestamps
            if isinstance(time_vals[0], (int, float, np.integer, np.floating)):
                # Handle numeric timestamps (nanoseconds or Unix timestamp)
                time_series = pd.to_datetime(time_vals, unit='ns')
            elif isinstance(time_vals, np.ndarray) and time_vals.dtype.kind == 'M':
                # Already datetime64 array
                time_series = pd.Series(time_vals)
            else:
                time_series = pd.Series(time_vals)
            
            # Ensure time_series is datetime type for proper diff calculation
            if not pd.api.types.is_datetime64_any_dtype(time_series):
                time_series = pd.to_datetime(time_series)
            
            # CRITICAL: For panel data, multiple rows share the same timestamp
            # Calculate diff on UNIQUE timestamps, not all rows (otherwise median will be 0)
            unique_times = time_series.unique()
            unique_times_sorted = pd.Series(unique_times).sort_values()
            
            # Calculate median time difference between unique timestamps
            time_diffs = unique_times_sorted.diff().dropna()
            # time_diffs should be TimedeltaIndex when time_series is datetime
            if isinstance(time_diffs, pd.TimedeltaIndex) and len(time_diffs) > 0:
                median_diff_minutes = abs(time_diffs.median().total_seconds()) / 60.0
            elif len(time_diffs) > 0:
                # Fallback: if diff didn't produce Timedeltas, calculate manually
                median_diff = time_diffs.median()
                if isinstance(median_diff, pd.Timedelta):
                    median_diff_minutes = abs(median_diff.total_seconds()) / 60.0
                elif isinstance(median_diff, (int, float, np.integer, np.floating)):
                    # Assume nanoseconds if numeric (use abs to handle unsorted timestamps)
                    median_diff_minutes = abs(float(median_diff)) / 1e9 / 60.0
                else:
                    raise ValueError(f"Unexpected median_diff type: {type(median_diff)}")
            else:
                # No differences (all timestamps identical) - use default
                median_diff_minutes = data_interval_minutes
                logger.warning(f"  All timestamps identical, cannot detect interval, using parameter: {data_interval_minutes}m")
            
            # Round to common intervals (1m, 5m, 15m, 30m, 60m)
            common_intervals = [1, 5, 15, 30, 60]
            detected_interval = min(common_intervals, key=lambda x: abs(x - median_diff_minutes))
            
            # Only use auto-detection if it's close to a common interval (load tolerance from config)
            try:
                from CONFIG.config_loader import get_cfg
                tolerance = float(get_cfg("safety.leakage_detection.model_evaluation.interval_detection_tolerance", default=0.2, config_name="safety_config"))
            except Exception:
                tolerance = 0.2
            if abs(median_diff_minutes - detected_interval) / detected_interval < tolerance:
                data_interval_minutes = detected_interval
                logger.info(f"  Auto-detected data interval: {median_diff_minutes:.1f}m → {data_interval_minutes}m (from timestamps)")
            else:
                # Fall back to parameter if detection is unclear
                logger.warning(f"  Auto-detection unclear ({median_diff_minutes:.1f}m), using parameter: {data_interval_minutes}m")
        except Exception as e:
            logger.warning(f"  Failed to auto-detect interval from timestamps: {e}, using parameter: {data_interval_minutes}m")
    else:
        # Use parameter value (default: 5)
        logger.info(f"  Using data interval from parameter: {data_interval_minutes}m")
    
    # CRITICAL FIX: Recompute purge_minutes from FINAL featureset (post-gatekeeper + post-prune)
    # The resolved_config.purge_minutes may have been computed from pre-prune featureset
    # We need to ensure purge is computed from the ACTUAL features used in training
    from TRAINING.ranking.utils.leakage_budget import compute_budget
    from TRAINING.ranking.utils.resolved_config import derive_purge_embargo
    from TRAINING.ranking.utils.cross_sectional_data import _compute_feature_fingerprint
    
    # Compute fingerprint of final featureset for validation
    final_featureset_fp, _ = _compute_feature_fingerprint(feature_names, set_invariant=True)
    
    # Get registry and feature_time_meta_map for budget computation
    registry = None
    feature_time_meta_map = None
    try:
        from TRAINING.common.feature_registry import get_registry
        registry = get_registry()
    except Exception:
        pass
    
    # Get feature_time_meta_map from resolved_config if available
    if resolved_config is not None and hasattr(resolved_config, 'feature_time_meta_map'):
        feature_time_meta_map = resolved_config.feature_time_meta_map
    
    # Compute budget from FINAL featureset (the one actually used in training)
    # This ensures purge is computed from the correct featureset
    budget_final, budget_fp, _ = compute_budget(
        feature_names,
        data_interval_minutes,
        target_horizon_minutes if target_horizon_minutes is not None else 60.0,
        registry=registry,
        max_lookback_cap_minutes=None,  # Don't cap - we want actual max for purge computation
        stage="CV_SPLITTER_CREATION",
        feature_time_meta_map=feature_time_meta_map,
        base_interval_minutes=resolved_config.base_interval_minutes if resolved_config is not None else None
    )
    
    # Validate fingerprint matches
    if budget_fp != final_featureset_fp:
        logger.error(
            f"🚨 FINGERPRINT MISMATCH (CV_SPLITTER): budget={budget_fp} != final_featureset={final_featureset_fp}. "
            f"This indicates a bug in feature set tracking."
        )
    else:
        logger.debug(f"✅ CV_SPLITTER: purge computed from fingerprint={budget_fp[:8]} (matches MODEL_TRAIN_INPUT)")
    
    # Load purge settings from config
    if _CONFIG_AVAILABLE:
        try:
            purge_buffer_bars = int(get_cfg("pipeline.leakage.purge_buffer_bars", default=5, config_name="pipeline_config"))
            purge_include_feature_lookback = get_cfg("safety.leakage_detection.purge_include_feature_lookback", default=True, config_name="safety_config")
        except Exception:
            purge_buffer_bars = 5
            purge_include_feature_lookback = True
    else:
        purge_buffer_bars = 5
        purge_include_feature_lookback = True
    
    # Compute purge from FINAL featureset lookback
    # If purge_include_feature_lookback is True, purge must cover feature lookback
    feature_lookback_max_minutes = budget_final.max_feature_lookback_minutes
    
    # Use centralized derivation function (base purge from horizon)
    purge_minutes_val, embargo_minutes_val = derive_purge_embargo(
        horizon_minutes=target_horizon_minutes,
        interval_minutes=data_interval_minutes,
        feature_lookback_max_minutes=None,  # derive_purge_embargo doesn't use this - we apply it separately below
        purge_buffer_bars=purge_buffer_bars,
        default_purge_minutes=85.0
    )
    
    # CRITICAL: Apply purge_include_feature_lookback policy (same logic as create_resolved_config)
    # If purge_include_feature_lookback=True, purge must be >= feature_lookback_max + interval
    if purge_include_feature_lookback and feature_lookback_max_minutes is not None:
        from TRAINING.common.utils.duration_parser import enforce_purge_audit_rule, format_duration
        
        purge_in = purge_minutes_val
        lookback_in = feature_lookback_max_minutes
        interval_for_rule = data_interval_minutes
        
        # Enforce audit rule: purge >= lookback_max (with interval-aware rounding)
        purge_out, min_purge, changed = enforce_purge_audit_rule(
            purge_in * 60.0,  # Convert minutes to seconds
            lookback_in * 60.0,  # Convert minutes to seconds
            interval=interval_for_rule * 60.0 if interval_for_rule is not None else None,
            buffer_frac=0.01,  # 1% safety buffer
            strict_greater=True
        )
        
        if changed:
            purge_minutes_val = purge_out.to_minutes()
            logger.info(
                f"⚠️  CV_SPLITTER: Increased purge from {purge_in:.1f}m to {purge_minutes_val:.1f}m "
                f"(min required: {format_duration(min_purge)}) to satisfy purge_include_feature_lookback=True. "
                f"Feature lookback: {lookback_in:.1f}m"
            )
    
    # CRITICAL ASSERT: Verify purge_include_feature_lookback policy is correctly applied
    if purge_include_feature_lookback and feature_lookback_max_minutes is not None:
        min_required_purge = feature_lookback_max_minutes + (data_interval_minutes if data_interval_minutes is not None else 5.0)
        assert purge_minutes_val >= min_required_purge, (
            f"🚨 BUG: purge_include_feature_lookback=True but purge ({purge_minutes_val:.1f}m) < "
            f"required ({min_required_purge:.1f}m = lookback {feature_lookback_max_minutes:.1f}m + interval {data_interval_minutes:.1f}m). "
            f"This indicates the purge_include_feature_lookback logic is not being applied correctly."
        )
    
    # Log purge computation with fingerprint for validation
    logger.info(
        f"📊 CV_SPLITTER: purge_minutes={purge_minutes_val:.1f}m computed from final_featureset "
        f"(fingerprint={final_featureset_fp[:8]}, actual_max_lookback={feature_lookback_max_minutes:.1f}m, "
        f"purge_include_feature_lookback={purge_include_feature_lookback}, "
        f"min_required={feature_lookback_max_minutes + (data_interval_minutes if data_interval_minutes is not None else 5.0):.1f}m if include_lookback=True)"
    )
    
    # CRITICAL: Validate purge doesn't exceed data span (hard-stop if invalid)
    # Check for explicit override config
    allow_invalid_cv = False
    try:
        from CONFIG.config_loader import get_cfg
        allow_invalid_cv = get_cfg("safety.leakage_detection.cv.allow_invalid_cv", default=False, config_name="safety_config")
    except Exception:
        pass
    
    if time_vals is not None and len(time_vals) > 0:
        time_series = pd.Series(time_vals) if not isinstance(time_vals, pd.Series) else time_vals
        if hasattr(time_series, 'min') and hasattr(time_series, 'max'):
            try:
                time_min = time_series.min()
                time_max = time_series.max()
                # Handle both datetime and numeric (nanoseconds) timestamps
                if isinstance(time_min, (pd.Timestamp, pd.DatetimeTZDtype)):
                    # Already datetime - use total_seconds()
                    data_span_minutes = (time_max - time_min).total_seconds() / 60.0
                elif isinstance(time_min, (int, float, np.integer, np.floating)):
                    # Numeric (likely nanoseconds) - convert to timedelta
                    time_min_dt = pd.to_datetime(time_min, unit='ns')
                    time_max_dt = pd.to_datetime(time_max, unit='ns')
                    data_span_minutes = (time_max_dt - time_min_dt).total_seconds() / 60.0
                else:
                    # Try to convert to datetime
                    time_min_dt = pd.to_datetime(time_min)
                    time_max_dt = pd.to_datetime(time_max)
                    data_span_minutes = (time_max_dt - time_min_dt).total_seconds() / 60.0
                
                if purge_minutes_val >= data_span_minutes:
                    error_msg = (
                        f"🚨 INVALID CV CONFIGURATION: purge_minutes ({purge_minutes_val:.1f}m) >= data_span ({data_span_minutes:.1f}m). "
                        f"This will produce empty/invalid CV folds. "
                        f"Either: 1) Set lookback_budget_minutes cap to drop long-lookback features, "
                        f"2) Load more data (≥ {purge_minutes_val/1440:.1f} trading days), or "
                        f"3) Disable purge_include_feature_lookback in config."
                    )
                    if allow_invalid_cv:
                        logger.error(f"{error_msg} (override: allow_invalid_cv=true - proceeding anyway)")
                    else:
                        raise RuntimeError(error_msg)
            except RuntimeError:
                raise  # Re-raise RuntimeError (our hard-stop)
            except Exception as e:
                # Other exceptions (type conversion, etc.) - log but don't hard-stop
                logger.warning(f"  Failed to validate purge vs data span: {e}, skipping validation")
    
    purge_minutes = purge_minutes_val  # Keep numeric value for simpler API
    
    # Check for duplicate column names before training
    if len(feature_names) != len(set(feature_names)):
        # DETERMINISM: Use sorted_unique for deterministic iteration order
        duplicates = [name for name in sorted_unique(feature_names) if feature_names.count(name) > 1]
        logger.error(f"  🚨 DUPLICATE COLUMN NAMES before training: {duplicates}")
        raise ValueError(f"Duplicate feature names before training: {duplicates}")
    
    # Log feature set before training and compute fingerprint
    # CRITICAL: This fingerprint represents the ACTUAL features used in training (POST_PRUNE, not just post-gatekeeper)
    # Pruning happens earlier in this function (line ~635), so feature_names here is the final pruned set
    # All subsequent lookback computations must use this same fingerprint for validation
    from TRAINING.ranking.utils.cross_sectional_data import _log_feature_set, _compute_feature_fingerprint
    _log_feature_set("MODEL_TRAIN_INPUT", feature_names, previous_names=None, logger_instance=logger)
    model_train_input_fingerprint, model_train_input_order_fp = _compute_feature_fingerprint(feature_names, set_invariant=True)
    logger.info(f"📊 MODEL_TRAIN_INPUT fingerprint={model_train_input_fingerprint} (n_features={len(feature_names)}, POST_PRUNE)")
    
    # CRITICAL: Validate that purge was computed from the same featureset
    if 'final_featureset_fp' in locals() and final_featureset_fp != model_train_input_fingerprint:
        logger.error(
            f"🚨 FINGERPRINT MISMATCH: purge computed from {final_featureset_fp[:8]} but MODEL_TRAIN_INPUT={model_train_input_fingerprint[:8]}. "
            f"This indicates purge was computed from wrong featureset!"
        )
    elif 'final_featureset_fp' in locals():
        logger.debug(f"✅ Purge fingerprint validation: purge={final_featureset_fp[:8]} == MODEL_TRAIN_INPUT={model_train_input_fingerprint[:8]}")
    
    # Create purged time series split with time-based purging
    # CRITICAL: Validate time_vals alignment and sorting before using time-based purging
    if time_vals is not None and len(time_vals) == len(X):
        # Ensure time_vals is sorted (required for binary search in PurgedTimeSeriesSplit)
        time_series = pd.Series(time_vals) if not isinstance(time_vals, pd.Series) else time_vals
        if not time_series.is_monotonic_increasing:
            logger.warning("⚠️  time_vals is not sorted! Sorting X, y, and time_vals together")
            sort_idx = np.argsort(time_vals)
            X = X[sort_idx]
            y = y[sort_idx]
            time_vals = time_series.iloc[sort_idx].values if isinstance(time_series, pd.Series) else time_series[sort_idx]
            logger.info(f"  Sorted data by timestamp (preserving alignment)")
        
        # PHASE 1: Pre-CV compatibility check for degenerate folds (first-class handling)
        # Check if target is compatible with CV before creating splitter
        from TRAINING.ranking.utils.target_validation import check_cv_compatibility
        is_cv_compatible, cv_compatibility_reason = check_cv_compatibility(y, task_type, folds)
        
        # Get degenerate fold fallback policy from config
        cv_degenerate_fallback = "reduce_folds"  # Default
        cv_mifolds = 2  # Default minimum folds
        try:
            from CONFIG.config_loader import get_cfg
            cv_degenerate_fallback = get_cfg("training.cv_degenerate_fallback", default="reduce_folds", config_name="intelligent_training_config")
            cv_mifolds = int(get_cfg("training.cv_mifolds", default=2, config_name="intelligent_training_config"))
        except Exception:
            pass
        
        # Apply fallback policy if target is not CV-compatible
        original_folds = folds
        if not is_cv_compatible:
            logger.info(
                f"  ℹ️  CV compatibility check: {cv_compatibility_reason}. "
                f"Using fallback policy: {cv_degenerate_fallback}"
            )
            
            if cv_degenerate_fallback == "reduce_folds":
                # Reduce folds until compatible or reach minimum
                while folds > cv_mifolds:
                    folds -= 1
                    is_compatible, reason = check_cv_compatibility(y, task_type, folds)
                    if is_compatible:
                        logger.info(
                            f"  ℹ️  Reduced CV folds from {original_folds} to {folds} to handle degenerate target. "
                            f"Reason: {cv_compatibility_reason}"
                        )
                        break
                    cv_compatibility_reason = reason
                
                # If still not compatible at minimum folds, skip CV
                if folds == cv_mifolds and not check_cv_compatibility(y, task_type, folds)[0]:
                    logger.info(
                        f"  ℹ️  Target still degenerate at minimum folds ({cv_mifolds}). "
                        f"Will skip CV and train on full dataset for importance only."
                    )
                    folds = 0  # Signal to skip CV
            elif cv_degenerate_fallback == "skip_cv":
                logger.info(
                    f"  ℹ️  Skipping CV due to degenerate target. "
                    f"Will train on full dataset for importance only. Reason: {cv_compatibility_reason}"
                )
                folds = 0  # Signal to skip CV
            elif cv_degenerate_fallback == "different_splitter":
                # For classification, use StratifiedKFold if available
                logger.info(
                    f"  ℹ️  Using alternative splitter for degenerate target. "
                    f"Reason: {cv_compatibility_reason}"
                )
                # Note: This would require implementing alternative splitter logic
                # For now, fall back to reduce_folds
                folds = max(cv_mifolds, folds - 1)
                logger.info(f"  ℹ️  Falling back to reduce_folds: {folds} folds")
        
        # Create splitter only if we have valid folds
        skip_cv = False
        if folds > 0:
            # Uses purge_overlap_minutes (simpler API) instead of purge_overlap_time
            tscv = PurgedTimeSeriesSplit(
                n_splits=folds,
                purge_overlap_minutes=purge_minutes,
                time_column_values=time_vals
            )
            if log_cfg.cv_detail:
                logger.info(f"  Using PurgedTimeSeriesSplit (TIME-BASED): {folds} folds, purge_minutes={purge_minutes:.1f}")
        else:
            # Skip CV - will train on full dataset
            tscv = None
            skip_cv = True
            logger.info(f"  ℹ️  Skipping CV (degenerate target). Will train on full dataset for importance only.")
        
        # CRITICAL: Validate CV folds before training to prevent IndexError
        # Convert splitter generator to list to inspect all folds
        if skip_cv:
            # Skip fold validation if CV is skipped
            all_folds = []
            folds_generated = 0
            valid_folds = []
            n_valid_folds = 0
        else:
            all_folds = list(tscv.split(X, y))
            folds_generated = len(all_folds)
        
        if folds_generated == 0 and not skip_cv:
            raise RuntimeError(
                f"🚨 No CV folds generated. This usually means purge/embargo ({purge_time:.1f}m) is too large "
                f"relative to data span. Either: 1) Reduce lookback_budget_minutes cap to drop long-lookback features, "
                f"2) Load more data (≥ {purge_time/1440:.1f} trading days), or "
                f"3) Disable purge_include_feature_lookback in config."
            )
        
        # Determine if this is a classification task
        is_binary = task_type == TaskType.BINARY_CLASSIFICATION
        is_multiclass = task_type == TaskType.MULTICLASS_CLASSIFICATION
        is_classification = is_binary or is_multiclass
        
        # Validate each fold (skip if CV is skipped)
        if not skip_cv:
            valid_folds = []
            for fold_idx, (train_idx, test_idx) in enumerate(all_folds):
                # Check that indices are non-empty
                if len(train_idx) == 0:
                    logger.warning(f"  ⚠️  Fold {fold_idx + 1}: Empty training set (skipping)")
                    continue
                if len(test_idx) == 0:
                    logger.warning(f"  ⚠️  Fold {fold_idx + 1}: Empty test set (skipping)")
                    continue
                
                # For classification, check that both classes are present in training set
                if is_classification:
                    train_y = y[train_idx]
                    unique_classes = np.unique(train_y[~np.isnan(train_y)])
                    if len(unique_classes) < 2:
                        logger.warning(
                            f"  ⚠️  Fold {fold_idx + 1}: Training set has only {len(unique_classes)} class(es) "
                            f"(classes: {unique_classes.tolist()}), skipping"
                        )
                        continue
                
                valid_folds.append((train_idx, test_idx))
            
            n_valid_folds = len(valid_folds)
        else:
            n_valid_folds = 0
        
        if n_valid_folds == 0 and not skip_cv:
            raise RuntimeError(
                f"🚨 No valid CV folds after validation. Generated {folds_generated} folds, but all were invalid. "
                f"This usually means: 1) purge/embargo ({purge_time:.1f}m) is too large relative to data span, "
                f"2) Target is degenerate (single class or extreme imbalance), or "
                f"3) Data span is insufficient. "
                f"Either: 1) Reduce lookback_budget_minutes cap, 2) Load more data, or "
                f"3) Check target distribution."
            )
        
        if not skip_cv:
            if n_valid_folds < folds_generated:
                logger.warning(
                    f"  ⚠️  Only {n_valid_folds}/{folds_generated} folds are valid. "
                    f"Proceeding with {n_valid_folds} folds."
                )
            
            # Create a wrapper splitter that only yields valid folds
            class ValidatedSplitter:
                def __init__(self, valid_folds):
                    self.valid_folds = valid_folds
                    self.n_splits = len(valid_folds)
                
                def split(self, X, y=None, groups=None):
                    for train_idx, test_idx in self.valid_folds:
                        yield train_idx, test_idx
                
                def get_n_splits(self, X=None, y=None, groups=None):
                    return self.n_splits
            
            tscv = ValidatedSplitter(valid_folds)
            if log_cfg.cv_detail:
                logger.info(f"  ✅ CV fold validation: {n_valid_folds} valid folds (from {folds_generated} generated)")
    else:
        # CRITICAL: Row-count based purging is INVALID for panel data (multiple symbols per timestamp)
        # With 50 symbols, 1 bar = 50 rows. Using row counts causes catastrophic leakage.
        # We MUST fail loudly rather than silently producing invalid results.
        raise ValueError(
            f"CRITICAL: time_vals is required for panel data. "
            f"Row-count based purging is INVALID when multiple symbols share the same timestamp. "
            f"With {len(X)} samples, row-count purging would cause 100% data leakage. "
            f"Please ensure cross_sectional_data.py returns time_vals."
        )
    
    # Capture fold timestamps if time_vals is provided
    if time_vals is not None and len(time_vals) == len(X):
        try:
            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X, y)):
                train_times = time_vals[train_idx]
                test_times = time_vals[test_idx]
                fold_timestamps.append({
                    'fold_idx': fold_idx + 1,
                    'train_start': pd.Timestamp(train_times.min()) if len(train_times) > 0 else None,
                    'train_end': pd.Timestamp(train_times.max()) if len(train_times) > 0 else None,
                    'test_start': pd.Timestamp(test_times.min()) if len(test_times) > 0 else None,
                    'test_end': pd.Timestamp(test_times.max()) if len(test_times) > 0 else None,
                    'train_samples': len(train_idx),
                    'test_samples': len(test_idx)
                })
            if log_cfg.cv_detail:
                logger.info(f"  Captured timestamps for {len(fold_timestamps)} folds")
        except Exception as e:
            logger.warning(f"  Failed to capture fold timestamps: {e}")
            fold_timestamps = []
    
    if model_families is None:
        # Load from multi-model config if available
        if multi_model_config:
            model_families_dict = multi_model_config.get('model_families', {})
            if model_families_dict is None or not isinstance(model_families_dict, dict):
                logger.warning("model_families in config is None or not a dict. Using defaults.")
                model_families = ['lightgbm', 'random_forest', 'neural_network']
            else:
                model_families = [
                    name for name, config in model_families_dict.items()
                    if config is not None and isinstance(config, dict) and config.get('enabled', False)
                ]
                # Sort for deterministic order (ensures reproducible aggregations)
                model_families = sorted(model_families)
            logger.debug(f"Using {len(model_families)} models from config: {', '.join(model_families)}")
        else:
            model_families = ['lightgbm', 'random_forest', 'neural_network']
    
    # Filter families by task type compatibility (prevents garbage scores in aggregations)
    from TRAINING.training_strategies.utils import is_family_compatible
    compatible_families = []
    skipped_families = []
    for family in model_families:
        compatible, skip_reason = is_family_compatible(family, task_type)
        if compatible:
            compatible_families.append(family)
        else:
            skipped_families.append((family, skip_reason))
            logger.info(f"⏭️ Skipping {family} for target ranking: {skip_reason}")
    if skipped_families:
        logger.info(f"📋 Filtered {len(skipped_families)} incompatible families for task={task_type}")
    model_families = compatible_families
    
    # Create ModelConfig objects for this task type
    model_configs = create_model_configs_from_yaml(multi_model_config, task_type) if multi_model_config else []
    # Filter to only enabled model families
    model_configs = [mc for mc in model_configs if mc.name in model_families]
    
    # Note: model_metrics, model_scores, importance_magnitudes already initialized at function start
    
    # Determine task characteristics
    unique_vals = np.unique(y[~np.isnan(y)])
    is_binary = task_type == TaskType.BINARY_CLASSIFICATION
    is_multiclass = task_type == TaskType.MULTICLASS_CLASSIFICATION
    is_classification = is_binary or is_multiclass
    
    # Select scoring metric based on task type
    if task_type == TaskType.REGRESSION:
        scoring = 'r2'
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        scoring = 'roc_auc'
    else:  # MULTICLASS_CLASSIFICATION
        scoring = 'accuracy'
    
    # Helper function to detect perfect correlation (data leakage)
    # Track which models had perfect correlation warnings (for auto-fixer)
    _perfect_correlation_models = set()
    
    # Load thresholds from config
    if _CONFIG_AVAILABLE:
        try:
            safety_cfg = get_safety_config()
            # safety_config.yaml has a top-level 'safety' key
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            _correlation_threshold = float(leakage_cfg.get('auto_fix_thresholds', {}).get('perfect_correlation', 0.999))
            _suspicious_score_threshold = float(leakage_cfg.get('model_alerts', {}).get('suspicious_score', 0.99))
        except Exception:
            _correlation_threshold = 0.999  # FALLBACK_DEFAULT_OK
            _suspicious_score_threshold = 0.99  # FALLBACK_DEFAULT_OK
    else:
        # Load from safety config
        if _CONFIG_AVAILABLE:
            try:
                safety_cfg = get_safety_config()
                # safety_config.yaml has a top-level 'safety' key
                safety_section = safety_cfg.get('safety', {})
                leakage_cfg = safety_section.get('leakage_detection', {})
                _correlation_threshold = float(leakage_cfg.get('auto_fix_thresholds', {}).get('perfect_correlation', 0.999))
                _suspicious_score_threshold = float(leakage_cfg.get('model_alerts', {}).get('suspicious_score', 0.99))
            except Exception:
                _correlation_threshold = 0.999  # FALLBACK_DEFAULT_OK
                _suspicious_score_threshold = 0.99  # FALLBACK_DEFAULT_OK
        else:
            _correlation_threshold = 0.999  # FALLBACK_DEFAULT_OK
            _suspicious_score_threshold = 0.99  # FALLBACK_DEFAULT_OK
    
    # NOTE: Removed _critical_leakage_detected flag - training accuracy alone is not
    # a reliable leakage signal for tree-based models. Real defense: schema filters + pre-scan.
    
    def _check_for_perfect_correlation(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> bool:
        """
        Check if predictions are perfectly correlated with targets.
        
        NOTE: High training accuracy alone is NOT a reliable signal for leakage, especially
        for tree-based models (Random Forest, LightGBM) which can overfit to 100% training
        accuracy through memorization even without leakage.
        
        This function now only logs a warning for debugging purposes. Real leakage defense
        comes from:
        - Explicit feature filters (schema, pattern-based exclusions)
        - Pre-training near-copy scan
        - Time-purged cross-validation
        
        Returns True if perfect correlation detected (for tracking), but does NOT trigger
        early exit or mark target as LEAKAGE_DETECTED.
        """
        try:
            # Tree-based models can easily overfit to 100% training accuracy
            tree_models = {'random_forest', 'lightgbm', 'xgboost', 'catboost'}
            is_tree_model = model_name.lower() in tree_models
            
            # For classification, check if predictions match exactly
            if task_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
                if len(y_true) == len(y_pred):
                    accuracy = np.mean(y_true == y_pred)
                    # Use > with epsilon to prevent false triggers from rounding
                    # Load epsilon from config (default: 1e-6)
                    try:
                        from CONFIG.config_loader import get_cfg
                        epsilon = float(get_cfg("safety.leakage_detection.model_evaluation.comparison_epsilon", default=1e-6, config_name="safety_config"))
                    except Exception:
                        epsilon = 1e-6  # FALLBACK_DEFAULT_OK
                    if accuracy > (_correlation_threshold + epsilon):  # Configurable threshold (default: 99.9%)
                        metric_name = "training accuracy"
                        
                        if is_tree_model:
                            # Tree models: This is likely just overfitting, not leakage
                            logger.warning(
                                f"  ⚠️  {model_name} reached {accuracy:.1%} {metric_name} "
                                f"(threshold: {_correlation_threshold:.1%}). "
                                f"This may just be overfitting - tree ensembles can memorize training data. "
                                f"Check CV metrics instead. Real leakage defense: schema filters + pre-scan."
                            )
                        else:
                            # Non-tree models: Still suspicious but less likely to be false positive
                            logger.warning(
                                f"  ⚠️  {model_name} reached {accuracy:.1%} {metric_name} "
                                f"(threshold: {_correlation_threshold:.1%}). "
                                f"High training accuracy detected - investigate if CV metrics are also suspiciously high."
                            )
                        
                        _perfect_correlation_models.add(model_name)  # Track for debugging/auto-fixer
                        return True  # Return True for tracking, but don't trigger early exit
            
            # For regression, check correlation
            elif task_type == TaskType.REGRESSION:
                if len(y_true) == len(y_pred):
                    corr = np.corrcoef(y_true, y_pred)[0, 1]
                    # Use > with epsilon to prevent false triggers from rounding
                    # Load epsilon from config (default: 1e-6)
                    try:
                        from CONFIG.config_loader import get_cfg
                        epsilon = float(get_cfg("safety.leakage_detection.model_evaluation.comparison_epsilon", default=1e-6, config_name="safety_config"))
                    except Exception:
                        epsilon = 1e-6  # FALLBACK_DEFAULT_OK
                    if not np.isnan(corr) and abs(corr) > (_correlation_threshold + epsilon):
                        if is_tree_model:
                            logger.warning(
                                f"  ⚠️  {model_name} has correlation {corr:.4f} "
                                f"(threshold: {_correlation_threshold:.4f}). "
                                f"This may just be overfitting - check CV metrics instead."
                            )
                        else:
                            logger.warning(
                                f"  ⚠️  {model_name} has correlation {corr:.4f} "
                                f"(threshold: {_correlation_threshold:.4f}). "
                                f"High correlation detected - investigate if CV metrics are also suspiciously high."
                            )
                        
                        _perfect_correlation_models.add(model_name)  # Track for debugging
                        return True  # Return True for tracking, but don't trigger early exit
        except Exception:
            pass
        return False
    
    # Helper function to compute and store full task-aware metrics
    def _compute_and_store_metrics(model_name: str, model, X: np.ndarray, y: np.ndarray,
                                   primary_score: float, task_type: TaskType):
        """
        Compute full task-aware metrics and store in both model_metrics and model_scores.
        
        Args:
            model_name: Name of the model
            model: Fitted model
            X: Feature matrix (for predictions)
            y: True target values
            primary_score: Primary score from CV (R², AUC, or accuracy)
            task_type: TaskType enum
        """
        # Defensive check: ensure model_scores and model_metrics are dicts
        nonlocal model_scores, model_metrics
        if model_scores is None or not isinstance(model_scores, dict):
            logger.warning(f"model_scores is None or not a dict in _compute_and_store_metrics, reinitializing")
            model_scores = {}
        if model_metrics is None or not isinstance(model_metrics, dict):
            logger.warning(f"model_metrics is None or not a dict in _compute_and_store_metrics, reinitializing")
            model_metrics = {}
        
        # Store primary score for backward compatibility
        model_scores[model_name] = primary_score
        
            # Compute full task-aware metrics
        try:
            # Calculate training accuracy/correlation BEFORE checking for perfect correlation
            # This is needed for auto-fixer to detect high training scores
            training_accuracy = None
            if task_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
                if hasattr(model, 'predict_proba'):
                    if task_type == TaskType.BINARY_CLASSIFICATION:
                        y_proba = model.predict_proba(X)[:, 1]
                        try:
                            from CONFIG.config_loader import get_cfg
                            binary_threshold = float(get_cfg("safety.leakage_detection.model_evaluation.binary_classification_threshold", default=0.5, config_name="safety_config"))
                        except Exception:
                            binary_threshold = 0.5
                        y_pred_train = (y_proba >= binary_threshold).astype(int)
                    else:
                        y_proba = model.predict_proba(X)
                        y_pred_train = y_proba.argmax(axis=1)
                else:
                    y_pred_train = model.predict(X)
                if len(y) == len(y_pred_train):
                    training_accuracy = np.mean(y == y_pred_train)
            elif task_type == TaskType.REGRESSION:
                y_pred_train = model.predict(X)
                if len(y) == len(y_pred_train):
                    corr = np.corrcoef(y, y_pred_train)[0, 1]
                    if not np.isnan(corr):
                        training_accuracy = abs(corr)  # Store absolute correlation for regression
            
            if task_type == TaskType.REGRESSION:
                y_pred = model.predict(X)
                # Check for perfect correlation (leakage) - this sets _critical_leakage_detected flag
                if _check_for_perfect_correlation(y, y_pred, model_name):
                    logger.error(f"  CRITICAL: {model_name} shows signs of data leakage! Check feature filtering.")
                    # Early exit: don't compute more metrics, return immediately
                    return
                full_metrics = evaluate_by_task(task_type, y, y_pred, return_ic=True)
            elif task_type == TaskType.BINARY_CLASSIFICATION:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)[:, 1]  # Probability of class 1
                    # Load binary classification threshold from config
                    try:
                        from CONFIG.config_loader import get_cfg
                        binary_threshold = float(get_cfg("safety.leakage_detection.model_evaluation.binary_classification_threshold", default=0.5, config_name="safety_config"))
                    except Exception:
                        binary_threshold = 0.5
                    y_pred = (y_proba >= binary_threshold).astype(int)
                else:
                    # Fallback for models without predict_proba
                    y_pred = model.predict(X)
                    y_proba = np.clip(y_pred, 0, 1)  # Assume predictions are probabilities
                # Check for perfect correlation (for debugging/tracking only - not a leakage signal)
                _check_for_perfect_correlation(y, y_pred, model_name)
                full_metrics = evaluate_by_task(task_type, y, y_proba)
            else:  # MULTICLASS_CLASSIFICATION
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)
                    y_pred = y_proba.argmax(axis=1)
                else:
                    # Fallback: one-hot encode predictions
                    y_pred = model.predict(X)
                    n_classes = len(np.unique(y[~np.isnan(y)]))
                    y_proba = np.eye(n_classes)[y_pred.astype(int)]
                # Check for perfect correlation (for debugging/tracking only - not a leakage signal)
                _check_for_perfect_correlation(y, y_pred, model_name)
                full_metrics = evaluate_by_task(task_type, y, y_proba)
            
            # Store full metrics (training metrics from evaluate_by_task)
            model_metrics[model_name] = full_metrics
            
            # CRITICAL: Overwrite training metrics with CV scores (primary_score is from CV)
            # This ensures model_metrics contains CV scores, not training scores
            if task_type == TaskType.REGRESSION:
                model_metrics[model_name]['r2'] = primary_score  # CV R²
            elif task_type == TaskType.BINARY_CLASSIFICATION:
                model_metrics[model_name]['roc_auc'] = primary_score  # CV AUC
            else:  # MULTICLASS_CLASSIFICATION
                model_metrics[model_name]['accuracy'] = primary_score  # CV accuracy
            
            # Also store training accuracy/correlation for auto-fixer detection
            # This is the in-sample training score (not CV), which is what triggers leakage warnings
            if training_accuracy is not None:
                if task_type == TaskType.REGRESSION:
                    model_metrics[model_name]['training_r2'] = training_accuracy
                else:
                    model_metrics[model_name]['training_accuracy'] = training_accuracy
            
            # Compute prediction fingerprint for determinism tracking
            try:
                from TRAINING.common.utils.prediction_hashing import compute_prediction_fingerprint_for_model
                # Get strict mode from identity config
                strict_mode = False
                try:
                    from TRAINING.common.utils.fingerprinting import get_identity_mode
                    strict_mode = get_identity_mode() == "strict"
                except Exception:
                    pass
                
                # y_proba may not be defined for regression - use locals()
                proba_for_hash = locals().get('y_proba', None)
                
                fp_dict = compute_prediction_fingerprint_for_model(
                    preds=y_pred,
                    proba=proba_for_hash,
                    model=model,
                    task_type=str(task_type),
                    X=X,
                    strict_mode=strict_mode,
                )
                if fp_dict:
                    model_metrics[model_name]['prediction_fingerprint'] = fp_dict
                    logger.info(f"✅ Computed prediction_fingerprint for {model_name}: hash={fp_dict.get('prediction_hash', '')[:12]}...")
                else:
                    logger.warning(f"⚠️ compute_prediction_fingerprint_for_model returned None for {model_name}")
            except Exception as fp_e:
                logger.warning(f"Prediction fingerprint failed for {model_name}: {fp_e}")
        except Exception as e:
            logger.warning(f"Failed to compute full metrics for {model_name}: {e}")
            # Fallback to primary score only
            if task_type == TaskType.REGRESSION:
                model_metrics[model_name] = {'r2': primary_score}
            elif task_type == TaskType.BINARY_CLASSIFICATION:
                model_metrics[model_name] = {'roc_auc': primary_score}
            else:
                model_metrics[model_name] = {'accuracy': primary_score}
    
    # Helper function to update both model_scores and model_metrics
    # NOTE: This is now mainly for backward compat - full metrics computed after training
    def _update_model_score(model_name: str, score: float):
        """Update model_scores (backward compat) - full metrics computed separately"""
        model_scores[model_name] = score
    
    # Check for degenerate target BEFORE training models
    # A target is degenerate if it has < 2 unique values or one class has < 2 samples
    unique_vals = np.unique(y[~np.isnan(y)])
    if len(unique_vals) < 2:
        logger.debug(f"    Skipping: Target has only {len(unique_vals)} unique value(s)")
        return {}, {}, 0.0, {}, {}, [], set()  # model_metrics, model_scores, mean_importance, suspicious_features, feature_importances, fold_timestamps, perfect_correlation_models
    
    # For classification, check class balance
    if is_binary or is_multiclass:
        class_counts = np.bincount(y[~np.isnan(y)].astype(int))
        min_class_count = class_counts[class_counts > 0].min()
        if min_class_count < 2:
            logger.debug(f"    Skipping: Smallest class has only {min_class_count} sample(s)")
            return {}, {}, 0.0, {}, {}, [], set()  # model_metrics, model_scores, mean_importance, suspicious_features, feature_importances, fold_timestamps, perfect_correlation_models
    
    # LightGBM
    if 'lightgbm' in model_families:
        lightgbm_start_time = time.time()
        logger.info(f"  🚀 Starting LightGBM training...")
        try:
            # GPU settings (will fallback to CPU if GPU not available)
            gpu_params = {}
            
            # STRICT MODE: Force CPU for determinism
            from TRAINING.common.determinism import is_strict_mode
            if is_strict_mode():
                logger.info("  ℹ️  Strict mode: forcing CPU for LightGBM (GPU disabled for determinism)")
                # Skip GPU detection entirely - gpu_params stays empty (CPU)
            else:
                try:
                    from CONFIG.config_loader import get_cfg
                    # SST: All values from config, no hardcoded defaults
                    test_enabled = get_cfg('gpu.lightgbm.test_enabled', default=True, config_name='gpu_config')
                    test_n_estimators = get_cfg('gpu.lightgbm.test_n_estimators', default=1, config_name='gpu_config')
                    test_samples = get_cfg('gpu.lightgbm.test_samples', default=10, config_name='gpu_config')
                    test_features = get_cfg('gpu.lightgbm.test_features', default=5, config_name='gpu_config')
                    gpu_device_id = get_cfg('gpu.lightgbm.gpu_device_id', default=0, config_name='gpu_config')
                    gpu_platform_id = get_cfg('gpu.lightgbm.gpu_platform_id', default=0, config_name='gpu_config')
                    try_cuda_first = get_cfg('gpu.lightgbm.try_cuda_first', default=True, config_name='gpu_config')
                    preferred_device = get_cfg('gpu.lightgbm.device', default='cuda', config_name='gpu_config')
                    
                    if test_enabled and try_cuda_first:
                        # Try CUDA first (fastest)
                        try:
                            test_model = lgb.LGBMRegressor(device='cuda', n_estimators=test_n_estimators, gpu_device_id=gpu_device_id, verbose=lgbm_backend_cfg.native_verbosity)
                            test_model.fit(np.random.rand(test_samples, test_features), np.random.rand(test_samples))
                            gpu_params = {'device': 'cuda', 'gpu_device_id': gpu_device_id}
                            logger.info(f"  ✅ Using GPU (CUDA) for LightGBM (device_id={gpu_device_id})")
                        except Exception as cuda_error:
                            # Try OpenCL
                            try:
                                test_model = lgb.LGBMRegressor(device='gpu', n_estimators=test_n_estimators, gpu_platform_id=gpu_platform_id, gpu_device_id=gpu_device_id, verbose=lgbm_backend_cfg.native_verbosity)
                                test_model.fit(np.random.rand(test_samples, test_features), np.random.rand(test_samples))
                                gpu_params = {'device': 'gpu', 'gpu_platform_id': gpu_platform_id, 'gpu_device_id': gpu_device_id}
                                logger.info(f"  ✅ Using GPU (OpenCL) for LightGBM (platform_id={gpu_platform_id}, device_id={gpu_device_id})")
                            except Exception as opencl_error:
                                logger.warning(f"  ⚠️  LightGBM GPU not available (CUDA: {cuda_error}, OpenCL: {opencl_error}), using CPU")
                    elif test_enabled and preferred_device in ['cuda', 'gpu']:
                        # Use preferred device directly
                        try:
                            if preferred_device == 'cuda':
                                test_model = lgb.LGBMRegressor(device='cuda', n_estimators=test_n_estimators, gpu_device_id=gpu_device_id, verbose=lgbm_backend_cfg.native_verbosity)
                                gpu_params = {'device': 'cuda', 'gpu_device_id': gpu_device_id}
                            else:
                                test_model = lgb.LGBMRegressor(device='gpu', n_estimators=test_n_estimators, gpu_platform_id=gpu_platform_id, gpu_device_id=gpu_device_id, verbose=lgbm_backend_cfg.native_verbosity)
                                gpu_params = {'device': 'gpu', 'gpu_platform_id': gpu_platform_id, 'gpu_device_id': gpu_device_id}
                            test_model.fit(np.random.rand(test_samples, test_features), np.random.rand(test_samples))
                            logger.info(f"  ✅ Using GPU ({preferred_device.upper()}) for LightGBM")
                        except Exception as gpu_error:
                            logger.warning(f"  ⚠️  LightGBM GPU ({preferred_device}) not available: {gpu_error}, using CPU")
                    else:
                        # Skip test, use preferred device from config
                        if preferred_device in ['cuda', 'gpu']:
                            if preferred_device == 'cuda':
                                gpu_params = {'device': 'cuda', 'gpu_device_id': gpu_device_id}
                            else:
                                gpu_params = {'device': 'gpu', 'gpu_platform_id': gpu_platform_id, 'gpu_device_id': gpu_device_id}
                            logger.info(f"  Using GPU ({preferred_device.upper()}) for LightGBM (test disabled)")
                except Exception as e:
                    logger.warning(f"  ⚠️  LightGBM GPU config error: {e}, using CPU")
            
            # Get config values
            lgb_config = get_model_config('lightgbm', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(lgb_config, dict):
                lgb_config = {}
            # Remove objective, device, and verbose from config (we set these explicitly)
            # CRITICAL: Remove verbose to prevent double argument error
            lgb_config_clean = {k: v for k, v in lgb_config.items() if k not in ['device', 'objective', 'metric', 'verbose', 'deterministic', 'force_row_wise']}
            
            # CRITICAL: Force deterministic mode for reproducibility
            # This is essential for consistent results across runs, especially with GPU
            lgb_config_clean['deterministic'] = True
            lgb_config_clean['force_row_wise'] = True  # Required for deterministic=True

            # Set verbose level from backend config
            # Note: verbose is a model constructor parameter, not fit() parameter
            verbose_level = lgbm_backend_cfg.native_verbosity
            
            if is_binary:
                model = lgb.LGBMClassifier(
                    objective='binary',
                    verbose=verbose_level,  # Enable verbose for GPU verification
                    **lgb_config_clean,
                    **gpu_params
                )
            elif is_multiclass:
                n_classes = len(unique_vals)
                model = lgb.LGBMClassifier(
                    objective='multiclass',
                    num_class=n_classes,
                    verbose=verbose_level,  # Enable verbose for GPU verification
                    **lgb_config_clean,
                    **gpu_params
                )
            else:
                model = lgb.LGBMRegressor(
                    objective='regression',
                    verbose=verbose_level,  # Enable verbose for GPU verification
                    **lgb_config_clean,
                    **gpu_params
                )
            
            # CRITICAL FIX: Use manual CV loop with early stopping for gradient boosting
            # Get early stopping rounds from config (default: 50)
            early_stopping_rounds = lgb_config.get('early_stopping_rounds', 50) if isinstance(lgb_config, dict) else 50
            
            if log_cfg.cv_detail:
                logger.info(f"  Using CV with early stopping (rounds={early_stopping_rounds}) for LightGBM")
            scores = cross_val_score_with_early_stopping(
                model, X, y, cv=tscv, scoring=scoring, 
                early_stopping_rounds=early_stopping_rounds, n_jobs=1  # n_jobs=1 for early stopping compatibility
            )
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # Train once on full data (with early stopping on a validation split) to get importance
            # CRITICAL: Use time-aware split (load ratio from config) - don't shuffle time series data
            # Guard against empty arrays
            try:
                from CONFIG.config_loader import get_cfg
                time_split_ratio = float(get_cfg("preprocessing.validation.time_aware_split_ratio", default=0.8, config_name="preprocessing_config"))
                min_samples_for_split = int(get_cfg("preprocessing.validation.min_samples_for_split", default=10, config_name="preprocessing_config"))
            except Exception:
                time_split_ratio = 0.8
                min_samples_for_split = 10
            
            if len(X) < min_samples_for_split:
                logger.warning(f"  ⚠️  Too few samples ({len(X)}) for train/val split, fitting on all data")
                split_idx = len(X)
            else:
                split_idx = int(len(X) * time_split_ratio)
                split_idx = max(1, split_idx)  # Ensure at least 1 sample in validation
            
            if split_idx < len(X):
                X_train_final, X_val_final = X[:split_idx], X[split_idx:]
                y_train_final, y_val_final = y[:split_idx], y[split_idx:]
            else:
                # Fallback: use all data if too small
                X_train_final, X_val_final = X, X
                y_train_final, y_val_final = y, y
            # Log GPU usage if available (controlled by config)
            if 'device' in gpu_params and log_cfg.gpu_detail:
                logger.info(f"  🚀 Training LightGBM on {gpu_params['device'].upper()} (device_id={gpu_params.get('gpu_device_id', 0)})")
                logger.info(f"  📊 Dataset size: {len(X_train_final)} samples, {X_train_final.shape[1]} features")
                if log_cfg.edu_hints:
                    logger.info(f"  💡 Note: GPU is most efficient for large datasets (>100k samples)")
            
            # Use threading utilities for smart thread management
            if _THREADING_UTILITIES_AVAILABLE:
                # Get thread plan based on family and GPU usage
                plan = plan_for_family('LightGBM', total_threads=default_threads())
                # Set num_threads from plan (OMP threads for LightGBM)
                model.set_params(num_threads=plan['OMP'])
                # Use thread_guard context manager for safe thread control
                with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                    model.fit(
                        X_train_final, y_train_final,
                        eval_set=[(X_val_final, y_val_final)],
                        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                    )
            else:
                # Fallback: manual thread management
                model.fit(
                    X_train_final, y_train_final,
                    eval_set=[(X_val_final, y_val_final)],
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                )
            
            # Verify GPU was actually used (only if gpu_detail enabled)
            if 'device' in gpu_params and log_cfg.gpu_detail:
                # Check model parameters to see what device was actually used
                try:
                    model_params = model.get_params()
                    actual_device = model_params.get('device', 'unknown')
                    if actual_device != 'cpu':
                        logger.info(f"  ✅ LightGBM confirmed using {actual_device.upper()}")
                    else:
                        logger.warning(f"  ⚠️  LightGBM fell back to CPU despite GPU params")
                        logger.warning(f"     This can happen if dataset is too small or GPU not properly configured")
                except Exception:
                    logger.debug("  Could not verify device from model params")
            
            # CRITICAL: Check for suspiciously high scores (likely leakage)
            has_leak = False
            if not np.isnan(primary_score) and primary_score >= _suspicious_score_threshold:
                # Use task-appropriate metric name
                if task_type == TaskType.REGRESSION:
                    metric_name = "R²"
                elif task_type == TaskType.BINARY_CLASSIFICATION:
                    metric_name = "ROC-AUC"
                else:
                    metric_name = "Accuracy"
                logger.error(f"  🚨 LEAKAGE ALERT: lightgbm {metric_name}={primary_score:.4f} >= 0.99 - likely data leakage!")
                logger.error(f"    Features: {len(feature_names)} features")
                logger.error(f"    Analyzing feature importances to identify leaks...")
                has_leak = True
            
            # LEAK DETECTION: Analyze feature importance for suspicious patterns
            importances = model.feature_importances_
            # Load importance threshold from config
            if _CONFIG_AVAILABLE:
                try:
                    safety_cfg = get_safety_config()
                    # safety_config.yaml has a top-level 'safety' key
                    safety_section = safety_cfg.get('safety', {})
                    leakage_cfg = safety_section.get('leakage_detection', {})
                    importance_threshold = float(leakage_cfg.get('importance', {}).get('single_feature_threshold', 0.50))
                except Exception:
                    importance_threshold = 0.50
            else:
                importance_threshold = 0.50
            
            suspicious_features = _detect_leaking_features(
                feature_names, importances, model_name='lightgbm',
                threshold=importance_threshold,
                force_report=has_leak  # Always report top features if score indicates leak
            )
            if suspicious_features:
                all_suspicious_features['lightgbm'] = suspicious_features
            
            # Store all feature importances for detailed export
            # CRITICAL: Align importance to feature_names order to ensure fingerprint match
            importance_series = pd.Series(importances, index=feature_names[:len(importances)] if len(importances) <= len(feature_names) else feature_names)
            # Reindex to match exact feature_names order (fills missing with 0.0)
            importance_series = importance_series.reindex(feature_names, fill_value=0.0)
            importance_dict = importance_series.to_dict()
            all_feature_importances['lightgbm'] = importance_dict
            
            # Log importance keys vs train input (now guaranteed to match order)
            importance_keys = list(importance_dict.keys())  # Use list to preserve order
            train_input_keys = feature_names  # Already a list
            if len(importance_keys) != len(train_input_keys):
                missing = set(train_input_keys) - set(importance_keys)
                logger.warning(f"  ⚠️  IMPORTANCE_KEYS mismatch: {len(importance_keys)} keys vs {len(train_input_keys)} train features")
                logger.warning(f"    Missing from importance: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
            elif importance_keys == train_input_keys:
                # Keys match AND order matches - safe to log fingerprint
                from TRAINING.ranking.utils.cross_sectional_data import _log_feature_set
                _log_feature_set("IMPORTANCE_KEYS", importance_keys, previous_names=feature_names, logger_instance=logger)
            
            # Compute and store full task-aware metrics
            _compute_and_store_metrics('lightgbm', model, X, y, primary_score, task_type)
            
            # Use percentage of total importance in top fraction features (0-1 scale, interpretable)
            total_importance = np.sum(importances)
            if total_importance > 0:
                top_fraction = _get_importance_top_fraction()
                top_k = max(1, int(len(importances) * top_fraction))
                top_importance_sum = np.sum(np.sort(importances)[-top_k:])
                # Normalize to 0-1: what % of total importance is in top 10%?
                importance_ratio = top_importance_sum / total_importance
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
            
            # Log timing
            lightgbm_elapsed = time.time() - lightgbm_start_time
            timing_data['lightgbm'] = lightgbm_elapsed
            if timing_log_enabled and lightgbm_elapsed >= timing_log_threshold_seconds:
                logger.info(f"  ⏱️  LightGBM timing: {lightgbm_elapsed:.2f} seconds")
            
        except Exception as e:
            lightgbm_elapsed = time.time() - lightgbm_start_time
            timing_data['lightgbm'] = lightgbm_elapsed
            if timing_log_enabled:
                logger.warning(f"LightGBM failed after {lightgbm_elapsed:.2f} seconds: {e}")
    
    # Random Forest
    if 'random_forest' in model_families:
        random_forest_start_time = time.time()
        logger.info(f"  🚀 Starting Random Forest training...")
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            
            # Get config values
            rf_config = get_model_config('random_forest', multi_model_config)
            
            # FIX: Convert seed → random_state (sklearn uses random_state, config uses seed)
            if 'seed' in rf_config:
                rf_config = rf_config.copy()
                rf_config['random_state'] = rf_config.pop('seed')
            elif 'random_state' not in rf_config:
                rf_config = rf_config.copy()
                rf_config['random_state'] = BASE_SEED
            
            if is_binary or is_multiclass:
                model = RandomForestClassifier(**rf_config)
            else:
                model = RandomForestRegressor(**rf_config)
            
            # CRITICAL: Use same CV splitter as all other models (tscv) to ensure OOF evaluation
            # cross_val_score trains on train folds and evaluates on test folds - this is truly OOF
            scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # Log CV fold details for debugging
            if len(valid_scores) > 0:
                logger.debug(f"Random Forest CV scores: {valid_scores.tolist()} (mean={primary_score:.4f}, std={valid_scores.std():.4f})")
                if len(valid_scores) < len(scores):
                    logger.warning(f"Random Forest: {len(scores) - len(valid_scores)} folds returned NaN scores")
            
            # ⚠️ IMPORTANCE BIAS WARNING: This fits on the full dataset (in-sample)
            # Deep trees/GBMs can memorize noise, making feature importance biased.
            # TODO: Future enhancement - use permutation importance calculated on CV test folds
            # For now, this is acceptable but be aware that importance may be inflated
            # CRITICAL: The evaluation score (primary_score) is OOF, but importance is in-sample
            # Use threading utilities for smart thread management
            if _THREADING_UTILITIES_AVAILABLE:
                # Get thread plan based on family
                plan = plan_for_family('RandomForest', total_threads=default_threads())
                # Set n_jobs from plan (OMP threads for RandomForest)
                model.set_params(n_jobs=plan['OMP'])
                # Use thread_guard context manager for safe thread control
                with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                    model.fit(X, y)
            else:
                # Fallback: manual thread management
                model.fit(X, y)
            
            # Check for suspicious scores (OOF score - this is the truth)
            has_leak = not np.isnan(primary_score) and primary_score >= _suspicious_score_threshold
            if has_leak:
                logger.warning(
                    f"⚠️ Random Forest OOF score {primary_score:.4f} >= {_suspicious_score_threshold:.4f} "
                    f"(suspiciously high). This may indicate: "
                    f"(A) Target definition is trivial (e.g., fwd_ret_oc_same_day sampled at wrong time), "
                    f"(B) Feature leakage (check top correlated features), or "
                    f"(C) CV splitter misconfiguration. "
                    f"CV folds: {len(valid_scores)}/{len(scores)} valid"
                )
            
            # LEAK DETECTION: Analyze feature importance
            importances = model.feature_importances_
            suspicious_features = _detect_leaking_features(
                feature_names, importances, model_name='random_forest', 
                threshold=0.50, force_report=has_leak
            )
            if suspicious_features:
                all_suspicious_features['random_forest'] = suspicious_features
            
            # Store all feature importances for detailed export
            # CRITICAL: Align importance to feature_names order to ensure fingerprint match
            importance_series = pd.Series(importances, index=feature_names[:len(importances)] if len(importances) <= len(feature_names) else feature_names)
            # Reindex to match exact feature_names order (fills missing with 0.0)
            importance_series = importance_series.reindex(feature_names, fill_value=0.0)
            importance_dict = importance_series.to_dict()
            all_feature_importances['random_forest'] = importance_dict
            
            # Log importance keys vs train input (only once per model, use random_forest as representative)
            # Now guaranteed to match order
            if 'random_forest' not in all_feature_importances or len(all_feature_importances) == 1:
                importance_keys = list(importance_dict.keys())  # Use list to preserve order
                train_input_keys = feature_names  # Already a list
                if len(importance_keys) != len(train_input_keys):
                    missing = set(train_input_keys) - set(importance_keys)
                    logger.warning(f"  ⚠️  IMPORTANCE_KEYS mismatch (random_forest): {len(importance_keys)} keys vs {len(train_input_keys)} train features")
                    logger.warning(f"    Missing from importance: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
                elif importance_keys == train_input_keys:
                    # Keys match AND order matches - safe to log fingerprint
                    from TRAINING.ranking.utils.cross_sectional_data import _log_feature_set
                    _log_feature_set("IMPORTANCE_KEYS", importance_keys, previous_names=feature_names, logger_instance=logger)
            
            # Compute and store full task-aware metrics
            _compute_and_store_metrics('random_forest', model, X, y, primary_score, task_type)
            
            # Use percentage of total importance in top fraction features (0-1 scale, interpretable)
            total_importance = np.sum(importances)
            if total_importance > 0:
                top_fraction = _get_importance_top_fraction()
                top_k = max(1, int(len(importances) * top_fraction))
                top_importance_sum = np.sum(np.sort(importances)[-top_k:])
                # Normalize to 0-1: what % of total importance is in top 10%?
                importance_ratio = top_importance_sum / total_importance
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
            
            # Log timing
            random_forest_elapsed = time.time() - random_forest_start_time
            timing_data['random_forest'] = random_forest_elapsed
            if timing_log_enabled and random_forest_elapsed >= timing_log_threshold_seconds:
                logger.info(f"  ⏱️  Random Forest timing: {random_forest_elapsed:.2f} seconds")
            
        except Exception as e:
            random_forest_elapsed = time.time() - random_forest_start_time
            timing_data['random_forest'] = random_forest_elapsed
            if timing_log_enabled:
                logger.warning(f"Random Forest failed after {random_forest_elapsed:.2f} seconds: {e}")
    
    # Neural Network
    if 'neural_network' in model_families:
        neural_network_start_time = time.time()
        logger.info(f"  🚀 Starting Neural Network training...")
        try:
            from sklearn.neural_network import MLPRegressor, MLPClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.compose import TransformedTargetRegressor
            from sklearn.pipeline import Pipeline
            
            # Get config values
            nn_config = get_model_config('neural_network', multi_model_config)
            
            # FIX: Convert seed → random_state (sklearn uses random_state, config uses seed)
            if 'seed' in nn_config:
                nn_config = nn_config.copy()
                nn_config['random_state'] = nn_config.pop('seed')
            elif 'random_state' not in nn_config:
                nn_config = nn_config.copy()
                nn_config['random_state'] = BASE_SEED
            
            if is_binary or is_multiclass:
                # For classification: Pipeline handles imputation and scaling within CV folds
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('model', MLPClassifier(**nn_config))
                ]
                pipeline = Pipeline(steps)
                model = pipeline
                y_for_training = y
            else:
                # For regression: Pipeline for features + TransformedTargetRegressor for target
                # This ensures no data leakage - all scaling/imputation happens within CV folds
                feature_steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('model', MLPRegressor(**nn_config))
                ]
                feature_pipeline = Pipeline(feature_steps)
                model = TransformedTargetRegressor(
                    regressor=feature_pipeline,
                    transformer=StandardScaler()
                )
                y_for_training = y
            
            # Neural networks need special handling for degenerate targets
            # Suppress convergence warnings (they're noisy and we handle failures gracefully)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                try:
                    # Pipeline ensures imputation/scaling happens within each CV fold (no leakage)
                    scores = cross_val_score(model, X, y_for_training, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                    valid_scores = scores[~np.isnan(scores)]
                    primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
                except ValueError as e:
                    if "least populated class" in str(e) or "too few" in str(e):
                        logger.debug(f"    Neural Network: Target too imbalanced for CV")
                        primary_score = np.nan
                        model_metrics['neural_network'] = {'roc_auc': np.nan} if task_type == TaskType.BINARY_CLASSIFICATION else {'r2': np.nan} if task_type == TaskType.REGRESSION else {'accuracy': np.nan}
                        model_scores['neural_network'] = np.nan
                    else:
                        raise
            
            # Fit on raw data (Pipeline handles preprocessing internally)
            # ⚠️ IMPORTANCE BIAS WARNING: This fits on the full dataset (in-sample)
            # See comment above for details
            if not np.isnan(primary_score):
                # Use threading utilities for smart thread management
                if _THREADING_UTILITIES_AVAILABLE:
                    # Get thread plan based on family (neural networks are GPU families, so OMP=1, MKL=1)
                    plan = plan_for_family('MLP', total_threads=default_threads())
                    # Use thread_guard context manager for safe thread control
                    with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                        model.fit(X, y_for_training)
                else:
                    # Fallback: manual thread management
                    model.fit(X, y_for_training)
                
                # Compute and store full task-aware metrics (Pipeline handles preprocessing)
                _compute_and_store_metrics('neural_network', model, X, y_for_training, primary_score, task_type)
            
            baseline_score = model.score(X, y_for_training)
            
            # PERFORMANCE AUDIT: Track permutation importance computation
            import time
            perm_start_time = time.time()
            try:
                from TRAINING.common.utils.performance_audit import get_auditor
                auditor = get_auditor()
                if auditor.enabled:
                    # Include target/symbol/view so different evaluation contexts
                    # get distinct fingerprints (avoids false-positive "redundancy" alerts)
                    view_str = view.value if hasattr(view, 'value') else str(view)
                    fingerprint_kwargs = {
                        'data_shape': X.shape,
                        'n_features_sampled': min(10, X.shape[1]),
                        'stage': 'target_ranking',
                        'target': target_column,
                        'symbol': symbol,
                        'view': view_str,
                    }
                    fingerprint = auditor._compute_fingerprint('neural_network.permutation_importance', **fingerprint_kwargs)
            except Exception:
                auditor = None
                fingerprint = None
            
            perm_scores = []
            importance = np.zeros(X.shape[1])  # Initialize importance array
            for i in range(min(10, X.shape[1])):  # Sample 10 features
                X_perm = X.copy()
                # Use deterministic seed for permutation
                from TRAINING.common.determinism import stable_seed_from
                perm_seed = stable_seed_from(['permutation', target_column if 'target_column' in locals() else 'default', f'feature_{i}'])
                np.random.seed(perm_seed)
                np.random.shuffle(X_perm[:, i])
                perm_score = model.score(X_perm, y_for_training)
                perm_importance = abs(baseline_score - perm_score)
                perm_scores.append(perm_importance)
                importance[i] = perm_importance
            
            # Track call
            if auditor and auditor.enabled:
                perm_elapsed = time.time() - perm_start_time
                view_str = view.value if hasattr(view, 'value') else str(view)
                auditor.track_call(
                    func_name='neural_network.permutation_importance',
                    duration=perm_elapsed,
                    rows=X.shape[0],
                    cols=X.shape[1],
                    stage='target_ranking',
                    cache_hit=False,
                    input_fingerprint=fingerprint,
                    target=target_column,
                    symbol=symbol,
                    view=view_str,
                )
            
            # Normalize importance to match sampled features
            if len(perm_scores) > 0 and np.max(perm_scores) > 0:
                # Scale sampled features to full feature set
                importance = importance / np.max(perm_scores) if np.max(perm_scores) > 0 else importance
            
            # Store all feature importances for detailed export (same pattern as other models)
            # CRITICAL: Align importance to feature_names order to ensure fingerprint match
            importance_series = pd.Series(importance, index=feature_names[:len(importance)] if len(importance) <= len(feature_names) else feature_names)
            importance_series = importance_series.reindex(feature_names, fill_value=0.0)
            importance_dict = importance_series.to_dict()
            all_feature_importances['neural_network'] = importance_dict
            
            importance_magnitudes.append(np.mean(perm_scores))
            
            # Log timing
            neural_network_elapsed = time.time() - neural_network_start_time
            timing_data['neural_network'] = neural_network_elapsed
            if timing_log_enabled and neural_network_elapsed >= timing_log_threshold_seconds:
                logger.info(f"  ⏱️  Neural Network timing: {neural_network_elapsed:.2f} seconds")
            
        except ImportError as e:
            neural_network_elapsed = time.time() - neural_network_start_time
            timing_data['neural_network'] = neural_network_elapsed
            logger.error(f"❌ Neural Network not available: {e} - timing: {neural_network_elapsed:.2f} seconds")
            all_feature_importances['neural_network'] = {}  # Record failure
        except Exception as e:
            neural_network_elapsed = time.time() - neural_network_start_time
            timing_data['neural_network'] = neural_network_elapsed
            if timing_log_enabled:
                logger.error(f"❌ Neural Network failed after {neural_network_elapsed:.2f} seconds: {e}", exc_info=True)
            all_feature_importances['neural_network'] = {}  # Record failure
    
    # XGBoost
    if 'xgboost' in model_families:
        xgboost_start_time = time.time()
        logger.info(f"  🚀 Starting XGBoost training...")
        try:
            import xgboost as xgb
            
            # GPU settings (will fallback to CPU if GPU not available)
            gpu_params = {}
            
            # STRICT MODE: Force CPU for determinism
            if is_strict_mode():
                logger.info("  ℹ️  Strict mode: forcing CPU for XGBoost (GPU disabled for determinism)")
                # Skip GPU detection entirely - gpu_params stays empty (CPU)
            else:
                try:
                    from CONFIG.config_loader import get_cfg
                    # SST: All values from config, no hardcoded defaults
                    xgb_device = get_cfg('gpu.xgboost.device', default='cpu', config_name='gpu_config')
                    xgb_tree_method = get_cfg('gpu.xgboost.tree_method', default='hist', config_name='gpu_config')
                    # Note: gpu_id removed in XGBoost 3.1+, use device='cuda:0' format if needed
                    # For now, just use 'cuda' for default GPU
                    test_enabled = get_cfg('gpu.xgboost.test_enabled', default=True, config_name='gpu_config')
                    test_n_estimators = get_cfg('gpu.xgboost.test_n_estimators', default=1, config_name='gpu_config')
                    test_samples = get_cfg('gpu.xgboost.test_samples', default=10, config_name='gpu_config')
                    test_features = get_cfg('gpu.xgboost.test_features', default=5, config_name='gpu_config')
                    
                    if xgb_device == 'cuda':
                        if test_enabled:
                            # XGBoost 3.1+ uses device='cuda' with tree_method='hist' (gpu_id removed)
                            try:
                                test_model = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=test_n_estimators, verbosity=0)
                                test_model.fit(np.random.rand(test_samples, test_features), np.random.rand(test_samples))
                                gpu_params = {'tree_method': xgb_tree_method, 'device': 'cuda'}
                                logger.info("  ✅ Using GPU (CUDA) for XGBoost")
                            except Exception as gpu_test_error:
                                # Try legacy API: tree_method='gpu_hist' (for XGBoost < 2.0)
                                try:
                                    test_model = xgb.XGBRegressor(tree_method='gpu_hist', n_estimators=test_n_estimators, verbosity=0)
                                    test_model.fit(np.random.rand(test_samples, test_features), np.random.rand(test_samples))
                                    gpu_params = {'tree_method': 'gpu_hist'}  # Legacy API doesn't use device parameter
                                    logger.info("  ✅ Using GPU (CUDA) for XGBoost (legacy API: gpu_hist)")
                                except Exception as legacy_error:
                                    logger.warning(f"  ⚠️  XGBoost GPU test failed (new API: {gpu_test_error}, legacy API: {legacy_error}), falling back to CPU")
                        else:
                            # Skip test, use config values directly
                            gpu_params = {'tree_method': xgb_tree_method, 'device': 'cuda'}
                            logger.info("  Using GPU (CUDA) for XGBoost (test disabled)")
                    else:
                        logger.info("  Using CPU for XGBoost (device='cpu' in config)")
                except Exception as e:
                    logger.warning(f"  ⚠️  XGBoost GPU config error, using CPU: {e}")
            
            # Get config values
            xgb_config = get_model_config('xgboost', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(xgb_config, dict):
                xgb_config = {}
            # Remove task-specific parameters (we set these explicitly based on task type)
            # CRITICAL: Extract early_stopping_rounds from config - it goes in constructor for XGBoost 2.0+
            # Also remove tree_method and device if present (we set these from GPU config)
            early_stopping_rounds = xgb_config.get('early_stopping_rounds', None)
            xgb_config_clean = {k: v for k, v in xgb_config.items() 
                              if k not in ['objective', 'eval_metric', 'early_stopping_rounds', 'tree_method', 'device', 'gpu_id']}
            
            # XGBoost 2.0+ requires early_stopping_rounds in constructor, not fit()
            if early_stopping_rounds is not None:
                xgb_config_clean['early_stopping_rounds'] = early_stopping_rounds
            
            # Add GPU params if available (will override any tree_method/device in config)
            xgb_config_clean.update(gpu_params)
            
            if is_binary:
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    **xgb_config_clean
                )
            elif is_multiclass:
                n_classes = len(unique_vals)
                model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=n_classes,
                    **xgb_config_clean
                )
            else:
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    **xgb_config_clean
                )
            
            # Log GPU usage if available (controlled by config)
            if 'device' in gpu_params and gpu_params.get('device') == 'cuda' and log_cfg.gpu_detail:
                logger.info("  🚀 Training XGBoost on CUDA")
                logger.info(f"  📊 Dataset size: {len(X)} samples, {X.shape[1]} features")
                if log_cfg.edu_hints:
                    logger.info(f"  💡 Note: GPU is most efficient for large datasets (>100k samples)")
            
            # CRITICAL FIX: Use manual CV loop with early stopping for gradient boosting
            # Get early stopping rounds from config (default: 50)
            # NOTE: For XGBoost 2.0+, early_stopping_rounds is set in constructor above, not passed to fit()
            early_stopping_rounds = xgb_config.get('early_stopping_rounds', 50) if isinstance(xgb_config, dict) else 50
            
            logger.info(f"  Using CV with early stopping (rounds={early_stopping_rounds}) for XGBoost")
            try:
                # XGBoost uses same early stopping interface as LightGBM
                scores = cross_val_score_with_early_stopping(
                    model, X, y, cv=tscv, scoring=scoring,
                    early_stopping_rounds=early_stopping_rounds, n_jobs=1
                )
                valid_scores = scores[~np.isnan(scores)]
                primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            except ValueError as e:
                if "Invalid classes" in str(e) or "Expected" in str(e):
                    logger.debug(f"    XGBoost: Target degenerate in some CV folds")
                    primary_score = np.nan
                    model_metrics['xgboost'] = {'roc_auc': np.nan} if task_type == TaskType.BINARY_CLASSIFICATION else {'r2': np.nan} if task_type == TaskType.REGRESSION else {'accuracy': np.nan}
                    model_scores['xgboost'] = np.nan
                else:
                    raise
            
            # Train once on full data (with early stopping) to get importance and full metrics
            # CRITICAL: Use time-aware split (last 20% as validation) - don't shuffle time series data
            if not np.isnan(primary_score):
                # Guard against empty arrays
                if len(X) < 10:
                    logger.warning(f"  ⚠️  Too few samples ({len(X)}) for train/val split, fitting on all data")
                    split_idx = len(X)
                else:
                    # Load time-aware split ratio from config
                    try:
                        from CONFIG.config_loader import get_cfg
                        time_split_ratio = float(get_cfg("preprocessing.validation.time_aware_split_ratio", default=0.8, config_name="preprocessing_config"))
                    except Exception:
                        time_split_ratio = 0.8
                    split_idx = int(len(X) * time_split_ratio)
                    split_idx = max(1, split_idx)  # Ensure at least 1 sample in validation
                
                if split_idx < len(X):
                    X_train_final, X_val_final = X[:split_idx], X[split_idx:]
                    y_train_final, y_val_final = y[:split_idx], y[split_idx:]
                else:
                    # Fallback: use all data if too small
                    X_train_final, X_val_final = X, X
                    y_train_final, y_val_final = y, y
                # XGBoost 2.0+: early_stopping_rounds is set in constructor, not passed to fit()
                # The model already has it configured from the constructor above
                # Use threading utilities for smart thread management
                if _THREADING_UTILITIES_AVAILABLE:
                    # Get thread plan based on family and GPU usage
                    plan = plan_for_family('XGBoost', total_threads=default_threads())
                    # Set n_jobs from plan (OMP threads for XGBoost)
                    model.set_params(n_jobs=plan['OMP'])
                    # Use thread_guard context manager for safe thread control
                    with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                        model.fit(
                            X_train_final, y_train_final,
                            eval_set=[(X_val_final, y_val_final)],
                            verbose=False
                        )
                else:
                    # Fallback: manual thread management
                    model.fit(
                        X_train_final, y_train_final,
                        eval_set=[(X_val_final, y_val_final)],
                        verbose=False
                    )
                
                # Check for suspicious scores
                has_leak = primary_score >= _suspicious_score_threshold
                
                # Compute and store full task-aware metrics
                _compute_and_store_metrics('xgboost', model, X, y, primary_score, task_type)
                
                # LEAK DETECTION: Analyze feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    suspicious_features = _detect_leaking_features(
                        feature_names, importances, model_name='xgboost', 
                        threshold=0.50, force_report=has_leak
                    )
                    if suspicious_features:
                        all_suspicious_features['xgboost'] = suspicious_features
                    
                    # Store all feature importances for detailed export
                    # CRITICAL: Align importance to feature_names order to ensure fingerprint match
                    importance_series = pd.Series(importances, index=feature_names[:len(importances)] if len(importances) <= len(feature_names) else feature_names)
                    # Reindex to match exact feature_names order (fills missing with 0.0)
                    importance_series = importance_series.reindex(feature_names, fill_value=0.0)
                    importance_dict = importance_series.to_dict()
                    all_feature_importances['xgboost'] = importance_dict
            if hasattr(model, 'feature_importances_'):
                # Use percentage of total importance in top 10% features (0-1 scale, interpretable)
                importances = model.feature_importances_
                total_importance = np.sum(importances)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importances) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importances)[-top_k:])
                    # Normalize to 0-1: what % of total importance is in top 10%?
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
                importance_magnitudes.append(importance_ratio)
            
            # Log timing
            xgboost_elapsed = time.time() - xgboost_start_time
            timing_data['xgboost'] = xgboost_elapsed
            if timing_log_enabled and xgboost_elapsed >= timing_log_threshold_seconds:
                logger.info(f"  ⏱️  XGBoost timing: {xgboost_elapsed:.2f} seconds")
        except Exception as e:
            xgboost_elapsed = time.time() - xgboost_start_time
            timing_data['xgboost'] = xgboost_elapsed
            if timing_log_enabled:
                logger.warning(f"XGBoost failed after {xgboost_elapsed:.2f} seconds: {e}")
    
    # CatBoost
    if 'catboost' in model_families:
        catboost_start_time = time.time()
        logger.info(f"  🚀 Starting CatBoost training...")
        try:
            import catboost as cb
            from catboost import Pool
            from TRAINING.ranking.utils.target_utils import is_classification_target, is_binary_classification_target
            
            # Determine task characteristics (use task_type, not y inspection for consistency)
            is_binary = task_type == TaskType.BINARY_CLASSIFICATION
            is_classification = task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]
            
            # GPU settings (will fallback to CPU if GPU not available)
            gpu_params = {}
            
            # STRICT MODE: Force CPU for determinism
            if is_strict_mode():
                logger.info("  ℹ️  Strict mode: forcing CPU for CatBoost (GPU disabled for determinism)")
                # Skip GPU detection entirely - gpu_params stays empty (CPU)
            else:
                try:
                    from CONFIG.config_loader import get_cfg
                    # SST: All values from config, no hardcoded defaults
                    # FIX: Rename to catboost_task_type to avoid overwriting task_type (TaskType enum)
                    catboost_task_type = get_cfg('gpu.catboost.task_type', default='CPU', config_name='gpu_config')
                    devices = get_cfg('gpu.catboost.devices', default='0', config_name='gpu_config')
                    thread_count = get_cfg('gpu.catboost.thread_count', default=8, config_name='gpu_config')
                    test_enabled = get_cfg('gpu.catboost.test_enabled', default=True, config_name='gpu_config')
                    test_iterations = get_cfg('gpu.catboost.test_iterations', default=1, config_name='gpu_config')
                    test_samples = get_cfg('gpu.catboost.test_samples', default=10, config_name='gpu_config')
                    test_features = get_cfg('gpu.catboost.test_features', default=5, config_name='gpu_config')

                    if catboost_task_type == 'GPU':
                        if test_enabled:
                            # Try GPU (CatBoost uses task_type='GPU' or devices parameter)
                            # Test if GPU is available
                            try:
                                test_model = cb.CatBoostRegressor(task_type='GPU', devices=devices, iterations=test_iterations, verbose=False)
                                # FIX: GPU mode requires Pool objects, not numpy arrays
                                test_X = np.random.rand(test_samples, test_features).astype('float32')
                                test_y = np.random.rand(test_samples).astype('float32')
                                test_pool = Pool(data=test_X, label=test_y)
                                test_model.fit(test_pool)
                                gpu_params = {'task_type': 'GPU', 'devices': devices}
                                logger.info(f"  ✅ Using GPU (CUDA) for CatBoost (devices={devices})")
                            except Exception as gpu_test_error:
                                logger.warning(f"  ⚠️  CatBoost GPU test failed, falling back to CPU: {gpu_test_error}")
                                gpu_params = {}  # Fallback to CPU
                        else:
                            # Skip test, use config values directly
                            gpu_params = {'task_type': 'GPU', 'devices': devices}
                            logger.info(f"  Using GPU (CUDA) for CatBoost (devices={devices}, test disabled)")
                    else:
                        gpu_params = {}  # Use CPU (no GPU params)
                        logger.info("  Using CPU for CatBoost (task_type='CPU' in config)")
                except Exception as e:
                    logger.warning(f"  ⚠️  CatBoost GPU config error, using CPU: {e}")
            
            # Get config values
            cb_config = get_model_config('catboost', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(cb_config, dict):
                cb_config = {}
            
            # Build params dict (copy to avoid mutating original)
            params = dict(cb_config)
            
            # Remove task_type, devices, and thread_count if present (we set these from GPU config)
            params.pop('task_type', None)
            params.pop('devices', None)
            params.pop('thread_count', None)  # Remove if present, we'll set from GPU config when using GPU
            
            # Add GPU params if available (will override any task_type/devices in config)
            params.update(gpu_params)
            
            # Use threading utilities for smart thread management
            if _THREADING_UTILITIES_AVAILABLE:
                # Get thread plan based on family and GPU usage
                plan = plan_for_family('CatBoost', total_threads=default_threads())
                # Set thread_count from plan (OMP threads for CatBoost)
                if 'thread_count' not in params:
                    params['thread_count'] = plan['OMP']
            else:
                # Fallback: use thread_count from config variable
                if gpu_params and gpu_params.get('task_type') == 'GPU' and 'thread_count' not in params:
                    params['thread_count'] = thread_count
            
            # CatBoost Performance Diagnostics and Optimizations
            # Check for common issues that cause slow training (>20min for 50k samples)
            warnings_issued = []
            
            # 1. Check for excessive depth (exponential complexity: 2^d)
            depth = params.get('depth', 6)  # Default is 6
            if depth > 8:
                warnings_issued.append(f"⚠️  CatBoost depth={depth} is high (exponential complexity 2^{depth}). Consider depth ≤ 8 for faster training.")
            
            # 2. Check for text-like features (object/string dtype columns)
            # Convert X to DataFrame temporarily to check dtypes if feature_names available
            text_features_detected = []
            high_cardinality_features = []
            if feature_names and len(feature_names) == X.shape[1]:
                try:
                    # Create temporary DataFrame to check dtypes (pandas already imported at top)
                    X_df = pd.DataFrame(X, columns=feature_names)
                    
                    # Check for object/string dtype columns (potential text features)
                    object_cols = X_df.select_dtypes(include=['object', 'string']).columns.tolist()
                    if object_cols:
                        text_features_detected = object_cols
                        if 'text_features' not in params or not params.get('text_features'):
                            warnings_issued.append(
                                f"⚠️  CatBoost: Detected {len(object_cols)} text/object columns: {object_cols[:5]}{'...' if len(object_cols) > 5 else ''}. "
                                f"Add text_features=['col_name'] to params to avoid treating them as high-cardinality categoricals."
                            )
                    
                    # Check for high cardinality categoricals (potential ID columns)
                    # Only flag for DROP when multiple ID signals agree (categorical + high unique ratio + ID-like name)
                    # Numeric columns with high cardinality are normal (continuous features) - just warn, don't suggest dropping
                    cat_features_list = params.get('cat_features', [])
                    if isinstance(cat_features_list, (list, tuple)):
                        cat_features_set = set(cat_features_list)
                    else:
                        cat_features_set = set()
                    
                    for col in feature_names:
                        if col in X_df.columns:
                            try:
                                unique_count = X_df[col].nunique()
                                unique_ratio = unique_count / len(X_df) if len(X_df) > 0 else 0
                                
                                # Check if column is treated as categorical
                                is_categorical = (
                                    col in cat_features_set or
                                    X_df[col].dtype.name in ['object', 'category', 'string'] or
                                    str(X_df[col].dtype).startswith('category')
                                )
                                
                                # Check if it's numeric (float/int) - high cardinality is normal for continuous features
                                is_numeric = pd.api.types.is_numeric_dtype(X_df[col])
                                
                                # ID-like name patterns
                                id_patterns = ['_id', '_ID', 'id_', 'ID_', 'user_', 'User_', 'ip_', 'IP_', 'row_', 'Row_', 
                                              'uuid', 'UUID', 'tx_', 'order_', 'session_', 'hash_', '_key', '_Key']
                                has_id_name = any(pattern in col for pattern in id_patterns)
                                
                                # Check if values mostly occur once (median count per value <= 2)
                                value_counts = X_df[col].value_counts()
                                median_count = value_counts.median() if len(value_counts) > 0 else float('inf')
                                
                                # Only suggest DROP when multiple ID signals agree:
                                # 1. Treated as categorical (not numeric)
                                # 2. High unique ratio (>0.2 or >0.5 for strict)
                                # 3. Values mostly unique (median count <= 2) OR unique_ratio > 0.8
                                # 4. ID-like name OR near-perfect uniqueness
                                should_drop = (
                                    is_categorical and  # Must be categorical (not numeric)
                                    unique_ratio > 0.2 and  # High unique ratio
                                    (median_count <= 2 or unique_ratio > 0.8) and  # Mostly unique values
                                    (has_id_name or unique_ratio > 0.95)  # ID-like name OR near-perfect uniqueness
                                )
                                
                                if should_drop:
                                    high_cardinality_features.append((col, unique_count, unique_ratio, is_categorical))
                                elif is_numeric and unique_ratio > 0.8 and unique_count > 1000:
                                    # Numeric column with high cardinality - this is normal for continuous features
                                    # Just log a debug message, don't warn (this is expected behavior)
                                    logger.debug(f"  CatBoost: Column '{col}' is numeric with high cardinality ({unique_count} unique, {unique_ratio:.1%} unique ratio) - this is normal for continuous features")
                            except Exception:
                                pass  # Skip if can't compute unique count
                    
                    if high_cardinality_features:
                        id_cols = [col for col, _, _, _ in high_cardinality_features[:5]]
                        warnings_issued.append(
                            f"⚠️  CatBoost: Detected {len(high_cardinality_features)} high-cardinality ID-like CATEGORICAL columns: {id_cols}{'...' if len(high_cardinality_features) > 5 else ''}. "
                            f"These are treated as categorical with high unique ratios and ID-like names. Consider dropping or encoding differently (they don't generalize and slow training)."
                        )
                except Exception as e:
                    # If DataFrame conversion fails, skip diagnostics (non-critical)
                    logger.debug(f"  CatBoost diagnostics skipped (non-critical): {e}")
            
            # 3. Automatic metric_period injection for eval_set (reduces evaluation overhead)
            # Note: We use cross_val_score which doesn't use eval_set directly, but if params has eval_set,
            # we should add metric_period to reduce overhead
            # SST: Check config first, then use default
            if 'metric_period' not in params:
                # SST: Try to get from intelligent_training_config first, then model config, then default
                try:
                    from CONFIG.config_loader import get_cfg
                    # Check intelligent_training_config first (SST)
                    metric_period_from_config = get_cfg('training.catboost.metric_period', default=None, config_name='intelligent_training_config')
                    if metric_period_from_config is not None:
                        params['metric_period'] = metric_period_from_config
                    else:
                        # Fallback to model config (from multi_model.yaml) or default
                        params['metric_period'] = cb_config.get('metric_period', 50)  # Default: 50 if not in any config
                except Exception:
                    # If config loader fails, use model config or default
                    params['metric_period'] = cb_config.get('metric_period', 50)
                if log_cfg.edu_hints:
                    logger.debug(f"  CatBoost: Added metric_period={params['metric_period']} to reduce evaluation overhead (SST: from config or default)")
            
            # Log warnings if any
            if warnings_issued:
                logger.warning(f"  CatBoost Performance Warnings:")
                for warning in warnings_issued:
                    logger.warning(f"    {warning}")
                if log_cfg.edu_hints:
                    logger.info(f"  💡 See KNOWN_ISSUES.md for CatBoost slow training troubleshooting")
            
            # CRITICAL: Enforce consistent loss/metric pairs for CatBoost
            # Binary classification: Logloss + AUC (not RMSE + roc_auc)
            # Regression: RMSE + RMSE
            # This prevents NaN from loss/metric mismatch
            
            # Auto-detect target type and set loss_function if not specified
            if "loss_function" not in params:
                if is_classification_target(y):
                    if is_binary_classification_target(y):
                        params["loss_function"] = "Logloss"  # Binary classification loss
                    else:
                        params["loss_function"] = "MultiClass"
                else:
                    params["loss_function"] = "RMSE"  # Regression loss
            
            # CRITICAL: Override config if it has inconsistent loss/metric pair
            # If task is binary classification but loss is RMSE, fix it
            if is_binary and params.get("loss_function") == "RMSE":
                logger.warning(
                    f"  ⚠️  CatBoost: Config has RMSE loss for binary classification. "
                    f"Overriding to Logloss to prevent NaN."
                )
                params["loss_function"] = "Logloss"
            
            # Set eval_metric to match loss_function (CatBoost uses 'AUC' not 'roc_auc')
            if "eval_metric" not in params:
                if is_binary:
                    params["eval_metric"] = "AUC"  # CatBoost's internal metric name
                elif is_classification_target(y):
                    params["eval_metric"] = "Accuracy"
                else:
                    params["eval_metric"] = "RMSE"
            
            # Ensure eval_metric is consistent with loss_function
            if is_binary and params.get("eval_metric") not in ["AUC", "Logloss"]:
                logger.warning(
                    f"  ⚠️  CatBoost: Config has eval_metric={params.get('eval_metric')} for binary classification. "
                    f"Overriding to AUC to prevent NaN."
                )
                params["eval_metric"] = "AUC"
            
            # If loss_function is specified in config, respect it (YAML in charge)
            # But we've already validated consistency above
            
            # CRITICAL: Verify GPU params are in params dict before instantiation
            # CatBoost REQUIRES task_type='GPU' to actually use GPU (devices alone is ignored)
            if gpu_params and 'task_type' in gpu_params:
                # Ensure GPU params are definitely in params (defensive check)
                params.update(gpu_params)
                # Explicit verification that task_type is set
                if params.get('task_type') != 'GPU':
                    logger.warning(f"  ⚠️  CatBoost GPU params updated but task_type is '{params.get('task_type')}', expected 'GPU'")
                else:
                    logger.debug(f"  ✅ CatBoost GPU verified: task_type={params.get('task_type')}, devices={params.get('devices')}")
            elif gpu_params:
                # GPU was requested but task_type missing - this is a bug
                logger.error(f"  ❌ CatBoost GPU requested but task_type missing from gpu_params: {gpu_params}")
            
            # Log final params for debugging (only if GPU was requested)
            if gpu_params and gpu_params.get('task_type') == 'GPU' and log_cfg.gpu_detail:
                logger.debug(f"  CatBoost final params (sample): task_type={params.get('task_type')}, devices={params.get('devices')}, iterations={params.get('iterations', 'default')}")
            
            # Set verbose level from backend config (similar to LightGBM)
            # CRITICAL: Use logging_level='Silent' instead of verbose=0 to avoid
            # "Verbose period should be nonnegative" errors in newer CatBoost versions.
            # CatBoost internally maps verbose to verbose_period, and some versions
            # reject verbose=0 as an invalid period.
            verbose_val = params.pop('verbose', None)
            if verbose_val is None:
                verbose_val = catboost_backend_cfg.native_verbosity
            # Determine if we should be silent
            if isinstance(verbose_val, bool):
                is_silent = not verbose_val
            elif isinstance(verbose_val, (int, float)):
                is_silent = verbose_val <= 0
            else:
                is_silent = True
            if is_silent:
                params['logging_level'] = 'Silent'
                params.pop('verbose_period', None)
            else:
                params['verbose'] = max(1, int(verbose_val))

            # CatBoost: verbose_period must be >= 0 (if present)
            # Remove if negative to prevent "Verbose period should be nonnegative" error
            # This matches the logic in config_cleaner.py but applies it directly in the model evaluation path
            if 'verbose_period' in params:
                verbose_period_val = params.get('verbose_period')
                if isinstance(verbose_period_val, (int, float)) and verbose_period_val < 0:
                    params.pop('verbose_period', None)
                    logger.debug(f"  CatBoost: Removed invalid verbose_period={verbose_period_val} (must be >= 0)")
            
            # CRITICAL: Choose model class based on task_type (not y inspection)
            # This ensures consistency: BINARY_CLASSIFICATION → CatBoostClassifier, REGRESSION → CatBoostRegressor
            # Using is_classification_target(y) can be inconsistent with task_type
            if is_binary:
                # Binary classification: must use CatBoostClassifier
                if params.get("loss_function") == "RMSE":
                    # This should have been fixed above, but double-check
                    logger.error(
                        f"  ❌ CatBoost: Binary classification but loss_function=RMSE. "
                        f"This should have been overridden. Fixing now."
                    )
                    params["loss_function"] = "Logloss"
                base_model = cb.CatBoostClassifier(**params)
                # Hard-stop: verify we got the right class
                if not isinstance(base_model, cb.CatBoostClassifier):
                    raise ValueError(
                        f"BINARY_CLASSIFICATION requires CatBoostClassifier, but got {type(base_model)}. "
                        f"This is a programming error."
                    )
            elif is_classification:
                # Multiclass classification: must use CatBoostClassifier
                base_model = cb.CatBoostClassifier(**params)
            else:
                # Regression: must use CatBoostRegressor
                base_model = cb.CatBoostRegressor(**params)
                # Hard-stop: verify we got the right class
                if not isinstance(base_model, cb.CatBoostRegressor):
                    raise ValueError(
                        f"REGRESSION requires CatBoostRegressor, but got {type(base_model)}. "
                        f"This is a programming error."
                    )
            
            # FIX: When GPU mode is enabled, CatBoost requires Pool objects instead of numpy arrays
            # Create a wrapper class that converts numpy arrays to Pool objects in fit() method
            use_gpu = 'task_type' in params and params.get('task_type') == 'GPU'
            
            if use_gpu:
                # Create a wrapper class that handles Pool conversion for GPU mode
                # FIX: Make sklearn-compatible by implementing get_params/set_params
                class CatBoostGPUWrapper:
                    """Wrapper for CatBoost models that converts numpy arrays to Pool objects when GPU is enabled."""
                    def __init__(self, base_model=None, cat_features=None, use_gpu=True, _model_class=None, **kwargs):
                        # If base_model is provided, use it; otherwise create from kwargs (for sklearn cloning)
                        if base_model is not None:
                            self.base_model = base_model
                            # Store the model class for sklearn cloning
                            self._model_class = type(base_model)
                        else:
                            # Recreate base model from kwargs (for sklearn clone)
                            # Determine model class from loss_function or use stored class
                            if _model_class is not None:
                                model_class = _model_class
                            else:
                                # Infer from loss_function in kwargs
                                loss_fn = kwargs.get('loss_function', 'RMSE')
                                if loss_fn in ['Logloss', 'MultiClass']:
                                    model_class = cb.CatBoostClassifier
                                else:
                                    model_class = cb.CatBoostRegressor
                            
                            # CRITICAL: Verify model class matches loss_function
                            # If loss_function is Logloss but model_class is Regressor, fix it
                            loss_fn = kwargs.get('loss_function', 'RMSE')
                            if loss_fn in ['Logloss', 'MultiClass'] and model_class == cb.CatBoostRegressor:
                                logger.warning(
                                    f"  ⚠️  CatBoost GPU wrapper: loss_function={loss_fn} but model_class=Regressor. "
                                    f"Fixing to Classifier."
                                )
                                model_class = cb.CatBoostClassifier
                            elif loss_fn == 'RMSE' and model_class == cb.CatBoostClassifier:
                                logger.warning(
                                    f"  ⚠️  CatBoost GPU wrapper: loss_function={loss_fn} but model_class=Classifier. "
                                    f"Fixing to Regressor."
                                )
                                model_class = cb.CatBoostRegressor
                            
                            self.base_model = model_class(**kwargs)
                            self._model_class = model_class
                        # FIX: For sklearn clone validation, ensure cat_features is set exactly as passed
                        # If None, use empty list; if already a list, use it directly; otherwise convert
                        if cat_features is None:
                            self.cat_features = []
                        elif isinstance(cat_features, list):
                            # Already a list - use it directly (sklearn expects this for clone validation)
                            self.cat_features = cat_features
                        else:
                            # Convert to list if it's not already
                            self.cat_features = list(cat_features)
                        self.use_gpu = use_gpu
                    
                    def get_params(self, deep=True):
                        """Get parameters for sklearn compatibility."""
                        # Get base model params and add wrapper-specific params
                        params = self.base_model.get_params(deep=deep)
                        # FIX: Return cat_features as-is (it's already a list from __init__)
                        # Sklearn's clone validation requires exact round-trip: get_params() -> __init__(**params) -> get_params()
                        params['cat_features'] = self.cat_features
                        params['use_gpu'] = self.use_gpu
                        params['_model_class'] = self._model_class
                        # Remove base_model from params (it's not a constructor arg)
                        params.pop('base_model', None)
                        return params
                    
                    def set_params(self, **params):
                        """Set parameters for sklearn compatibility."""
                        # Extract wrapper-specific params
                        cat_features = params.pop('cat_features', None)
                        use_gpu = params.pop('use_gpu', None)
                        model_class = params.pop('_model_class', None)
                        if cat_features is not None:
                            # FIX: Set exactly as passed (sklearn clone validation requires this)
                            if isinstance(cat_features, list):
                                self.cat_features = cat_features
                            else:
                                self.cat_features = list(cat_features) if cat_features else []
                        if use_gpu is not None:
                            self.use_gpu = use_gpu
                        if model_class is not None:
                            self._model_class = model_class
                        # Update base model params
                        self.base_model.set_params(**params)
                        return self
                    
                    def fit(self, X, y=None, **kwargs):
                        """Convert numpy arrays to Pool objects when GPU is enabled."""
                        # Convert X and y to Pool objects for GPU mode
                        if isinstance(X, np.ndarray):
                            train_pool = Pool(data=X, label=y, cat_features=self.cat_features)
                            return self.base_model.fit(train_pool, **kwargs)
                        elif isinstance(X, Pool):
                            # Already a Pool object
                            return self.base_model.fit(X, y, **kwargs)
                        else:
                            # Fallback: try direct fit (for other data types)
                            return self.base_model.fit(X, y, **kwargs)
                    
                    def predict(self, X, **kwargs):
                        """Delegate predict to base model."""
                        if isinstance(X, np.ndarray) and self.use_gpu:
                            # Convert to Pool for consistency, though predict may work with arrays
                            test_pool = Pool(data=X, cat_features=self.cat_features)
                            return self.base_model.predict(test_pool, **kwargs)
                        return self.base_model.predict(X, **kwargs)
                    
                    def score(self, X, y, **kwargs):
                        """Delegate score to base model."""
                        if isinstance(X, np.ndarray) and self.use_gpu:
                            test_pool = Pool(data=X, label=y, cat_features=self.cat_features)
                            return self.base_model.score(test_pool, **kwargs)
                        return self.base_model.score(X, y, **kwargs)
                    
                    def __getattr__(self, name):
                        """Delegate all other attributes to base model."""
                        return getattr(self.base_model, name)
                
                # Get categorical features from params if specified
                cat_features = params.get('cat_features', [])
                if isinstance(cat_features, list) and len(cat_features) > 0:
                    # If cat_features are column names, convert to indices
                    if feature_names and isinstance(cat_features[0], str):
                        cat_feature_indices = [feature_names.index(f) for f in cat_features if f in feature_names]
                    else:
                        cat_feature_indices = cat_features
                else:
                    cat_feature_indices = []
                
                model = CatBoostGPUWrapper(base_model=base_model, cat_features=cat_feature_indices, use_gpu=use_gpu)
            else:
                # CPU mode: use model directly (no Pool conversion needed)
                model = base_model

            # Log GPU usage if available (always log, not just when gpu_detail enabled)
            if 'task_type' in params and params.get('task_type') == 'GPU':
                logger.info(f"  🚀 Training CatBoost on GPU (devices={params.get('devices', '0')})")
                logger.info(f"  📊 Dataset size: {len(X)} samples, {X.shape[1]} features")
                if log_cfg.edu_hints:
                    logger.info(f"  💡 Note: CatBoost does quantization on CPU first, then trains on GPU")
                    logger.info(f"  💡 Watch GPU memory allocation (not just utilization %) to verify GPU usage")
            elif gpu_params and gpu_params.get('task_type') == 'GPU':
                # Fallback: log if GPU was requested but not in final params
                logger.warning(f"  ⚠️  CatBoost GPU requested but task_type not in final params (check config cleaning)")
                logger.warning(f"  ⚠️  Final params task_type: {params.get('task_type', 'MISSING')}")
                if log_cfg.edu_hints:
                    logger.info(f"  💡 Note: GPU is most efficient for large datasets (>100k samples)")
            
            # Check for outer parallelism that might cause CPU bottleneck
            # If CV is parallelized (cv_n_jobs > 1), this can cause CPU to peg even with thread_count limited
            if cv_n_jobs and cv_n_jobs > 1 and gpu_params and gpu_params.get('task_type') == 'GPU':
                logger.warning(
                    f"  ⚠️  CatBoost GPU training with parallel CV (n_jobs={cv_n_jobs}). "
                    f"Outer parallelism can cause CPU bottleneck even with thread_count limited. "
                    f"Consider setting cv_n_jobs=1 for GPU training."
                )
            
            # PHASE 1: Handle skipped CV (degenerate folds policy)
            if tscv is None:
                # CV was skipped due to degenerate target - skip CV and fit on full dataset
                primary_score = np.nan
                logger.info(f"  ℹ️  CatBoost: Skipping CV (degenerate target detected pre-CV). Fitting on full dataset for importance only.")
            else:
                # CRITICAL: Fold health check before CV to diagnose NaN issues
                # Log fold health for each fold and hard-fail on invalid folds
                logger.info(f"  🔍 CatBoost CV fold health check:")
                logger.info(f"     Objective: {params.get('loss_function', 'auto')}, Metric: {scoring}, Task: {task_type.name}")
                
                # Extract purge/embargo from resolved_config if available
                purge_minutes_val = None
                embargo_minutes_val = None
                if resolved_config:
                    purge_minutes_val = getattr(resolved_config, 'purge_minutes', None)
                    embargo_minutes_val = getattr(resolved_config, 'embargo_minutes', None)
                
                # Also try to get from purge_time if available (for logging)
                # purge_time is defined earlier in the function as pd.Timedelta
                try:
                    if purge_time is not None:
                        # purge_time is a Timedelta, convert to minutes
                        if hasattr(purge_time, 'total_seconds'):
                            purge_minutes_val = purge_time.total_seconds() / 60.0
                        elif isinstance(purge_time, (int, float)):
                            purge_minutes_val = purge_time
                except NameError:
                    # purge_time not in scope, use resolved_config values only
                    pass
                
                if purge_minutes_val and embargo_minutes_val:
                    logger.info(f"     Purge: {purge_minutes_val:.1f}m, Embargo: {embargo_minutes_val:.1f}m")
                elif purge_minutes_val:
                    logger.info(f"     Purge: {purge_minutes_val:.1f}m, Embargo: unknown")
                else:
                    logger.info(f"     Purge/Embargo: from resolved_config or defaults")
                
                # Check each fold before CV
                fold_violations = []
                all_folds_list = list(tscv.split(X, y))
                
                for fold_idx, (train_idx, val_idx) in enumerate(all_folds_list):
                    train_n = len(train_idx)
                    val_n = len(val_idx)
                    
                    # Basic checks
                    if val_n == 0:
                        fold_violations.append(f"Fold {fold_idx + 1}: val_n=0 (empty validation set)")
                        continue
                    
                    # Binary classification: check class balance in BOTH train and validation sets
                    if is_binary:
                        # Check validation set
                        val_y = y[val_idx]
                        val_y_clean = val_y[~np.isnan(val_y)]
                        val_unique = np.unique(val_y_clean) if len(val_y_clean) > 0 else np.array([])
                        val_pos_count = np.sum(val_y_clean == 1) if len(val_y_clean) > 0 else 0
                        val_neg_count = np.sum(val_y_clean == 0) if len(val_y_clean) > 0 else 0
                        
                        # Check training set (CRITICAL: single-class training causes NaN)
                        train_y = y[train_idx]
                        train_y_clean = train_y[~np.isnan(train_y)]
                        train_unique = np.unique(train_y_clean) if len(train_y_clean) > 0 else np.array([])
                        train_pos_count = np.sum(train_y_clean == 1) if len(train_y_clean) > 0 else 0
                        train_neg_count = np.sum(train_y_clean == 0) if len(train_y_clean) > 0 else 0
                        
                        # Check for violations
                        val_degenerate = val_pos_count == 0 or val_neg_count == 0
                        train_degenerate = train_pos_count == 0 or train_neg_count == 0
                        
                        if train_degenerate:
                            fold_violations.append(
                                f"Fold {fold_idx + 1}: Binary classification with degenerate TRAINING set "
                                f"(train_pos={train_pos_count}, train_neg={train_neg_count}, train_unique={train_unique.tolist()})"
                            )
                            logger.warning(
                                f"     ⚠️  Fold {fold_idx + 1}: train_n={train_n}, train_pos={train_pos_count}, train_neg={train_neg_count}, "
                                f"train_unique={train_unique.tolist()}, val_n={val_n}, val_pos={val_pos_count}, val_neg={val_neg_count}"
                            )
                        elif val_degenerate:
                            fold_violations.append(
                                f"Fold {fold_idx + 1}: Binary classification with degenerate validation set "
                                f"(val_pos={val_pos_count}, val_neg={val_neg_count}, val_unique={val_unique.tolist()})"
                            )
                            logger.warning(
                                f"     ⚠️  Fold {fold_idx + 1}: train_n={train_n}, train_pos={train_pos_count}, train_neg={train_neg_count}, "
                                f"val_n={val_n}, val_pos={val_pos_count}, val_neg={val_neg_count}, val_unique={val_unique.tolist()}"
                            )
                        else:
                            logger.info(
                                f"     ✅ Fold {fold_idx + 1}: train_n={train_n}, train_pos={train_pos_count}, train_neg={train_neg_count}, "
                                f"val_n={val_n}, val_pos={val_pos_count}, val_neg={val_neg_count}"
                            )
                    else:
                        # Regression or multiclass: just log sizes
                        logger.info(f"     ✅ Fold {fold_idx + 1}: train_n={train_n}, val_n={val_n}")
                    
                    # Ranking/group structure check (if groups are used)
                    # Note: For panel data ranking, groups are typically timestamps
                    # If CatBoost is using ranking objective, we'd need group IDs here
                    # For now, just log if we detect ranking mode
                    if 'objective' in params and 'ranking' in str(params.get('objective', '')).lower():
                        # Ranking mode: would need group IDs to check group sizes
                        logger.warning(f"     ⚠️  Fold {fold_idx + 1}: Ranking mode detected but group structure not validated")
                
                # Log violations but don't hard-fail (let CV proceed and return NaN if needed)
                # This allows us to diagnose the issue while maintaining current behavior
                if fold_violations:
                    error_msg = (
                        f"🚨 CatBoost CV fold health check FAILED. Invalid folds detected:\n"
                        f"   " + "\n   ".join(fold_violations) + "\n"
                        f"   This will likely cause NaN scores. Fix by:\n"
                    )
                    if purge_minutes_val and embargo_minutes_val:
                        error_msg += f"   1) Reducing purge/embargo ({purge_minutes_val:.1f}m/{embargo_minutes_val:.1f}m) if too large\n"
                    else:
                        error_msg += f"   1) Reducing purge/embargo if too large\n"
                    error_msg += (
                        f"   2) Loading more data to ensure sufficient validation set size\n"
                        f"   3) For binary classification: ensure validation sets have both classes"
                    )
                    logger.error(error_msg)
                    logger.warning(f"     ⚠️  Proceeding with CV anyway (will likely return NaN)")
                else:
                    logger.info(f"     ✅ All {len(all_folds_list)} folds passed health check")
                
                try:
                    scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                    valid_scores = scores[~np.isnan(scores)]
                    primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
                except (ValueError, TypeError) as e:
                    error_str = str(e)
                    if "Invalid classes" in error_str or "Expected" in error_str:
                        logger.debug(f"    CatBoost: Target degenerate in some CV folds")
                        primary_score = np.nan
                        model_metrics['catboost'] = {'roc_auc': np.nan} if task_type == TaskType.BINARY_CLASSIFICATION else {'r2': np.nan} if task_type == TaskType.REGRESSION else {'accuracy': np.nan}
                        model_scores['catboost'] = np.nan
                    elif "Invalid data type" in error_str and "catboost.Pool" in error_str:
                        # FIX: If Pool conversion failed, log and re-raise with context
                        logger.error(f"  ❌ CatBoost GPU Pool conversion error: {e}")
                        logger.error(f"  💡 This may indicate a CatBoost version compatibility issue with GPU mode")
                        raise
                    else:
                        raise
            
            # Fit model and compute importance even if CV failed (NaN score)
            # Classification targets often fail CV due to degenerate folds, but we can still compute importance
            # from a model fit on the full dataset
            model_fitted = False
            if not np.isnan(primary_score):
                # Use threading utilities for smart thread management
                if _THREADING_UTILITIES_AVAILABLE:
                    # Get thread plan based on family and GPU usage
                    plan = plan_for_family('CatBoost', total_threads=default_threads())
                    # Use thread_guard context manager for safe thread control
                    with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                        model.fit(X, y)
                else:
                    # Fallback: manual thread management
                    model.fit(X, y)
                model_fitted = True
                
                # Verify GPU is actually being used (post-fit check)
                if gpu_params and gpu_params.get('task_type') == 'GPU':
                    try:
                        actual_params = model.get_all_params()
                        actual_task_type = actual_params.get('task_type', 'UNKNOWN')
                        if actual_task_type != 'GPU':
                            logger.warning(
                                f"  ⚠️  CatBoost GPU requested but model reports task_type='{actual_task_type}'. "
                                f"GPU may not be active. Check model.get_all_params() for actual configuration."
                            )
                        elif log_cfg.gpu_detail:
                            logger.debug(f"  ✅ CatBoost GPU verified post-fit: task_type={actual_task_type}, devices={actual_params.get('devices', 'UNKNOWN')}")
                    except Exception as e:
                        logger.debug(f"  CatBoost post-fit GPU verification skipped (non-critical): {e}")

                # Compute and store full task-aware metrics
                _compute_and_store_metrics('catboost', model, X, y, primary_score, task_type)
            else:
                # CV failed (NaN score) - still try to fit and compute importance
                # This is especially important for classification targets that may fail CV due to degenerate folds
                # but can still provide useful feature importance from full-dataset fit
                logger.info(f"  ℹ️  CatBoost CV returned NaN (likely degenerate folds), but fitting on full dataset to compute importance")
                try:
                    # Use threading utilities for smart thread management
                    if _THREADING_UTILITIES_AVAILABLE:
                        # Get thread plan based on family and GPU usage
                        plan = plan_for_family('CatBoost', total_threads=default_threads())
                        # Use thread_guard context manager for safe thread control
                        with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                            model.fit(X, y)
                    else:
                        # Fallback: manual thread management
                        model.fit(X, y)
                    model_fitted = True
                except Exception as e:
                    logger.warning(f"  ⚠️  CatBoost failed to fit on full dataset: {e}")
                    model_fitted = False
            
            # CatBoost requires training dataset to compute feature importance
            # FIX: For GPU wrapper, need to access base_model and handle Pool conversion
            # CRITICAL: Always compute and store importance if model trained successfully (even if CV failed)
            importance = None
            if model_fitted:
                # PERFORMANCE AUDIT: Track CatBoost importance computation
                import time
                importance_start_time = time.time()
                try:
                    from TRAINING.common.utils.performance_audit import get_auditor
                    auditor = get_auditor()
                    if auditor.enabled:
                        # Include target/symbol/view so different evaluation contexts
                        # get distinct fingerprints (avoids false-positive "redundancy" alerts)
                        view_str = view.value if hasattr(view, 'value') else str(view)
                        fingerprint_kwargs = {
                            'data_shape': X.shape,
                            'n_features': len(feature_names),
                            'importance_type': 'PredictionValuesChange',
                            'stage': 'target_ranking',
                            'target': target_column,
                            'symbol': symbol,
                            'view': view_str,
                        }
                        fingerprint = auditor._compute_fingerprint('catboost.get_feature_importance', **fingerprint_kwargs)
                except Exception:
                    auditor = None
                    fingerprint = None
                
                try:
                    # CRITICAL: CatBoost get_feature_importance() requires Pool objects in many cases
                    # Even in CPU mode, some CatBoost versions require Pool for feature importance
                    # Always convert to Pool to ensure compatibility
                    if isinstance(X, np.ndarray):
                        # Get categorical features if available
                        cat_features = []
                        if hasattr(model, 'cat_features'):
                            cat_features = model.cat_features
                        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'get_cat_feature_indices'):
                            try:
                                cat_features = model.base_model.get_cat_feature_indices()
                            except Exception:
                                cat_features = []
                        
                        importance_data = Pool(data=X, cat_features=cat_features if cat_features else None)
                    else:
                        importance_data = X
                    
                    if hasattr(model, 'base_model'):
                        # Wrapper model - use base model
                        importance = model.base_model.get_feature_importance(data=importance_data, type='PredictionValuesChange')
                    else:
                        # Direct model (CPU mode)
                        importance = model.get_feature_importance(data=importance_data, type='PredictionValuesChange')
                    
                    # Track call
                    if auditor and auditor.enabled:
                        importance_elapsed = time.time() - importance_start_time
                        view_str = view.value if hasattr(view, 'value') else str(view)
                        auditor.track_call(
                            func_name='catboost.get_feature_importance',
                            duration=importance_elapsed,
                            rows=X.shape[0],
                            cols=len(feature_names),
                            stage='target_ranking',
                            cache_hit=False,
                            input_fingerprint=fingerprint,
                            target=target_column,
                            symbol=symbol,
                            view=view_str,
                        )
                except Exception as e:
                    logger.warning(f"  ⚠️  CatBoost feature importance computation failed: {e}")
                    logger.debug(f"  CatBoost importance error details:", exc_info=True)
                    # Try fallback: ensure Pool object is used (CatBoost requires Pool for get_feature_importance)
                    try:
                        # Always use Pool for fallback too
                        if isinstance(X, np.ndarray):
                            cat_features = []
                            if hasattr(model, 'cat_features'):
                                cat_features = model.cat_features
                            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'get_cat_feature_indices'):
                                try:
                                    cat_features = model.base_model.get_cat_feature_indices()
                                except Exception:
                                    cat_features = []
                            importance_data = Pool(data=X, cat_features=cat_features if cat_features else None)
                        else:
                            importance_data = X
                        
                        if hasattr(model, 'base_model'):
                            importance = model.base_model.get_feature_importance(data=importance_data, type='PredictionValuesChange')
                        else:
                            importance = model.get_feature_importance(data=importance_data, type='PredictionValuesChange')
                        logger.info(f"  ✅ CatBoost importance computed using fallback method (Pool conversion)")
                    except Exception as e2:
                        logger.warning(f"  ⚠️  CatBoost importance fallback also failed: {e2}")
                        importance = None
                
                # Store all feature importances for detailed export (same pattern as other models)
                # CRITICAL: Align importance to feature_names order to ensure fingerprint match
                if importance is not None and len(importance) > 0:
                    importance_series = pd.Series(importance, index=feature_names[:len(importance)] if len(importance) <= len(feature_names) else feature_names)
                    # Reindex to match exact feature_names order (fills missing with 0.0)
                    importance_series = importance_series.reindex(feature_names, fill_value=0.0)
                    importance_dict = importance_series.to_dict()
                    all_feature_importances['catboost'] = importance_dict
                    logger.debug(f"  ✅ CatBoost feature importance stored: {len(importance_dict)} features")
                else:
                    if importance is None:
                        logger.warning(f"  ⚠️  CatBoost feature importance is None (computation failed)")
                    else:
                        logger.warning(f"  ⚠️  CatBoost feature importance is empty (len={len(importance)})")
                    # Store empty dict to ensure CatBoost appears in output (even if empty)
                    # This ensures consistency - all models that train should have entries
                    all_feature_importances['catboost'] = {}
            else:
                # Model didn't fit - can't compute importance
                importance = np.array([])
                all_feature_importances['catboost'] = {}
            if importance is not None and len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
            
            # Log timing
            catboost_elapsed = time.time() - catboost_start_time
            timing_data['catboost'] = catboost_elapsed
            if timing_log_enabled and catboost_elapsed >= timing_log_threshold_seconds:
                logger.info(f"  ⏱️  CatBoost timing: {catboost_elapsed:.2f} seconds")
        except ImportError:
            catboost_elapsed = time.time() - catboost_start_time
            timing_data['catboost'] = catboost_elapsed
            logger.warning(f"CatBoost not available (pip install catboost) - timing: {catboost_elapsed:.2f} seconds")
        except Exception as e:
            catboost_elapsed = time.time() - catboost_start_time
            timing_data['catboost'] = catboost_elapsed
            error_str = str(e)
            # Log detailed error information for debugging
            logger.error(f"❌ CatBoost failed after {catboost_elapsed:.2f} seconds: {error_str}")
            # Log full exception traceback for verbose_period and other parameter errors
            if "verbose_period" in error_str.lower() or "verbose" in error_str.lower():
                logger.error(f"   CatBoost verbose/verbose_period error details:", exc_info=True)
            else:
                logger.debug(f"   CatBoost error details:", exc_info=True)
            # Ensure CatBoost appears in results with empty dict (so it's tracked as failed)
            all_feature_importances['catboost'] = {}
            model_metrics['catboost'] = {'roc_auc': np.nan} if task_type == TaskType.BINARY_CLASSIFICATION else {'r2': np.nan} if task_type == TaskType.REGRESSION else {'accuracy': np.nan}
            model_scores['catboost'] = np.nan
    
    # Lasso
    if 'lasso' in model_families:
        lasso_start_time = time.time()
        logger.info(f"  🚀 Starting Lasso training...")
        try:
            from sklearn.linear_model import Lasso
            from sklearn.pipeline import Pipeline
            from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
            
            # Get config values
            lasso_config = get_model_config('lasso', multi_model_config)
            
            # FIX: Convert seed → random_state (sklearn uses random_state, config uses seed)
            if 'seed' in lasso_config:
                lasso_config = lasso_config.copy()
                lasso_config['random_state'] = lasso_config.pop('seed')
            elif 'random_state' not in lasso_config:
                lasso_config = lasso_config.copy()
                lasso_config['random_state'] = BASE_SEED
            
            # Use sklearn-safe conversion (handles NaNs, dtypes, infs)
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # CRITICAL FIX: Pipeline ensures scaling happens within each CV fold (no leakage)
            # Lasso requires scaling for proper convergence (features must be on similar scales)
            # Note: X_dense is already imputed by make_sklearn_dense_X, so we only need scaler
            steps = [
                ('scaler', StandardScaler()),  # Required for Lasso convergence
                ('model', Lasso(**lasso_config))
            ]
            pipeline = Pipeline(steps)
            
            scores = cross_val_score(pipeline, X_dense, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # ⚠️ IMPORTANCE BIAS WARNING: This fits on the full dataset (in-sample)
            # See comment above for details
            # Use threading utilities for smart thread management
            if _THREADING_UTILITIES_AVAILABLE:
                # Get thread plan based on family (linear models use MKL threads)
                plan = plan_for_family('Lasso', total_threads=default_threads())
                # Use thread_guard context manager for safe thread control
                with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                    pipeline.fit(X_dense, y)
            else:
                # Fallback: manual thread management
                pipeline.fit(X_dense, y)
            
            # Compute and store full task-aware metrics (Lasso is regression-only)
            if not np.isnan(primary_score) and task_type == TaskType.REGRESSION:
                _compute_and_store_metrics('lasso', pipeline, X_dense, y, primary_score, task_type)
            
            # Extract coefficients from the fitted model
            model = pipeline.named_steps['model']
            importance = np.abs(model.coef_)
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # Store all feature importances for detailed export (same pattern as other models)
            # CRITICAL: Align importance to feature_names order to ensure fingerprint match
            importance_series = pd.Series(importance, index=feature_names)
            importance_series = importance_series.reindex(feature_names, fill_value=0.0)
            importance_dict = importance_series.to_dict()
            all_feature_importances['lasso'] = importance_dict
            
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
            
            # Log timing
            lasso_elapsed = time.time() - lasso_start_time
            timing_data['lasso'] = lasso_elapsed
            if timing_log_enabled and lasso_elapsed >= timing_log_threshold_seconds:
                logger.info(f"  ⏱️  Lasso timing: {lasso_elapsed:.2f} seconds")
        except ImportError as e:
            lasso_elapsed = time.time() - lasso_start_time
            timing_data['lasso'] = lasso_elapsed
            logger.error(f"❌ Lasso not available: {e} - timing: {lasso_elapsed:.2f} seconds")
            all_feature_importances['lasso'] = {}  # Record failure
        except Exception as e:
            lasso_elapsed = time.time() - lasso_start_time
            timing_data['lasso'] = lasso_elapsed
            error_str = str(e)
            # Log detailed error information for debugging
            logger.error(f"❌ Lasso failed after {lasso_elapsed:.2f} seconds: {error_str}")
            logger.debug(f"   Lasso error details:", exc_info=True)
            # Ensure Lasso appears in results with empty dict (so it's tracked as failed)
            all_feature_importances['lasso'] = {}
            model_metrics['lasso'] = {'r2': np.nan} if task_type == TaskType.REGRESSION else {'roc_auc': np.nan} if task_type == TaskType.BINARY_CLASSIFICATION else {'accuracy': np.nan}
            model_scores['lasso'] = np.nan
    
    # Ridge
    if 'ridge' in model_families:
        ridge_start_time = time.time()
        logger.info(f"  🚀 Starting Ridge training...")
        try:
            from sklearn.linear_model import Ridge, RidgeClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
            
            # Ridge doesn't handle NaNs - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Get config values
            ridge_config = get_model_config('ridge', multi_model_config)
            
            # FIX: Convert seed → random_state (sklearn uses random_state, config uses seed)
            if 'seed' in ridge_config:
                ridge_config = ridge_config.copy()
                ridge_config['random_state'] = ridge_config.pop('seed')
            elif 'random_state' not in ridge_config:
                ridge_config = ridge_config.copy()
                ridge_config['random_state'] = BASE_SEED
            
            # CRITICAL: Use correct estimator based on task type
            # For classification: RidgeClassifier (not Ridge regression)
            # For regression: Ridge
            if is_binary or is_multiclass:
                est_cls = RidgeClassifier
            else:
                est_cls = Ridge
            
            # CRITICAL: Ridge requires scaling for proper convergence
            # Pipeline ensures scaling happens within each CV fold (no leakage)
            steps = [
                ('scaler', StandardScaler()),  # Required for Ridge convergence
                ('model', est_cls(**ridge_config))
            ]
            pipeline = Pipeline(steps)
            
            scores = cross_val_score(pipeline, X_dense, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # Fit on full data for importance extraction (CV is done elsewhere)
            # Use threading utilities for smart thread management
            if _THREADING_UTILITIES_AVAILABLE:
                # Get thread plan based on family (linear models use MKL threads)
                plan = plan_for_family('Ridge', total_threads=default_threads())
                # Use thread_guard context manager for safe thread control
                with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                    pipeline.fit(X_dense, y)
            else:
                # Fallback: manual thread management
                pipeline.fit(X_dense, y)
            
            # Compute and store full task-aware metrics
            if not np.isnan(primary_score):
                _compute_and_store_metrics('ridge', pipeline, X_dense, y, primary_score, task_type)
            
            # Extract coefficients from the fitted model
            model = pipeline.named_steps['model']
            # FIX: Handle both 1D (binary) and 2D (multiclass) coef_ shapes
            coef = model.coef_
            if len(coef.shape) > 1:
                # Multiclass: use max absolute coefficient across classes
                importance = np.abs(coef).max(axis=0)
            else:
                # Binary or regression: use absolute coefficients
                importance = np.abs(coef)
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # Store all feature importances for detailed export (same pattern as other models)
            # CRITICAL: Align importance to feature_names order to ensure fingerprint match
            importance_series = pd.Series(importance, index=feature_names)
            importance_series = importance_series.reindex(feature_names, fill_value=0.0)
            importance_dict = importance_series.to_dict()
            all_feature_importances['ridge'] = importance_dict
            
            # Validate importance is not all zeros
            if np.all(importance == 0) or np.sum(importance) == 0:
                logger.warning(f"Ridge: All coefficients are zero (over-regularized or no signal)")
                importance_ratio = 0.0
            else:
                if len(importance) > 0:
                    total_importance = np.sum(importance)
                    if total_importance > 0:
                        top_fraction = _get_importance_top_fraction()
                        top_k = max(1, int(len(importance) * top_fraction))
                        top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                        importance_ratio = top_importance_sum / total_importance
                    else:
                        importance_ratio = 0.0
                else:
                    importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
            
            # Log timing
            ridge_elapsed = time.time() - ridge_start_time
            timing_data['ridge'] = ridge_elapsed
            if timing_log_enabled and ridge_elapsed >= timing_log_threshold_seconds:
                logger.info(f"  ⏱️  Ridge timing: {ridge_elapsed:.2f} seconds")
        except ImportError as e:
            ridge_elapsed = time.time() - ridge_start_time
            timing_data['ridge'] = ridge_elapsed
            logger.error(f"❌ Ridge not available: {e} - timing: {ridge_elapsed:.2f} seconds")
            all_feature_importances['ridge'] = {}  # Record failure
        except Exception as e:
            ridge_elapsed = time.time() - ridge_start_time
            timing_data['ridge'] = ridge_elapsed
            if timing_log_enabled:
                logger.error(f"❌ Ridge failed after {ridge_elapsed:.2f} seconds: {e}", exc_info=True)
            all_feature_importances['ridge'] = {}  # Record failure
    
    # Elastic Net
    if 'elastic_net' in model_families:
        elastic_net_start_time = time.time()
        logger.info(f"  🚀 Starting Elastic Net training...")
        try:
            from sklearn.linear_model import ElasticNet, LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
            
            # Elastic Net doesn't handle NaNs - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Get config values
            elastic_net_config = get_model_config('elastic_net', multi_model_config)
            
            # FIX: Convert seed → random_state (sklearn uses random_state, config uses seed)
            if 'seed' in elastic_net_config:
                elastic_net_config = elastic_net_config.copy()
                elastic_net_config['random_state'] = elastic_net_config.pop('seed')
            elif 'random_state' not in elastic_net_config:
                elastic_net_config = elastic_net_config.copy()
                elastic_net_config['random_state'] = BASE_SEED
            
            # CRITICAL: Use correct estimator based on task type
            # For classification: LogisticRegression with penalty='elasticnet' and solver='saga'
            # For regression: ElasticNet
            if is_binary or is_multiclass:
                # LogisticRegression with elasticnet penalty
                est_cls = LogisticRegression
                # ElasticNet requires solver='saga' for penalty='elasticnet'
                elastic_net_config = elastic_net_config.copy()
                elastic_net_config['penalty'] = 'elasticnet'
                elastic_net_config['solver'] = 'saga'  # Required for elasticnet penalty
                # l1_ratio maps to ElasticNet's l1_ratio (0 = pure L2, 1 = pure L1)
                if 'l1_ratio' not in elastic_net_config:
                    elastic_net_config['l1_ratio'] = elastic_net_config.get('l1_ratio', 0.5)
                # alpha maps to C (inverse regularization strength)
                if 'alpha' in elastic_net_config:
                    # Convert alpha to C (C = 1/alpha for consistency with sklearn)
                    alpha = elastic_net_config.pop('alpha')
                    elastic_net_config['C'] = 1.0 / alpha if alpha > 0 else 1.0
                elif 'C' not in elastic_net_config:
                    elastic_net_config['C'] = 1.0  # Default C=1.0
            else:
                # ElasticNet regression
                est_cls = ElasticNet
            
            # FAIL-FAST: Set reasonable max_iter limit to avoid long-running fits
            # Default max_iter is 1000, but saga solver can be very slow
            # Use a lower limit (500) to fail faster if it's not converging or going to zero
            original_max_iter = elastic_net_config.get('max_iter', 1000)
            if 'max_iter' not in elastic_net_config:
                elastic_net_config['max_iter'] = 500  # Reduced from default 1000 for fail-fast
            elif elastic_net_config.get('max_iter', 1000) > 500:
                # Cap at 500 for fail-fast behavior
                logger.debug(f"Elastic Net: Capping max_iter at 500 for fail-fast (was {elastic_net_config['max_iter']})")
                elastic_net_config['max_iter'] = 500
            
            # CRITICAL: Elastic Net requires scaling for proper convergence
            # Pipeline ensures scaling happens within each CV fold (no leakage)
            steps = [
                ('scaler', StandardScaler()),  # Required for ElasticNet convergence
                ('model', est_cls(**elastic_net_config))
            ]
            pipeline = Pipeline(steps)
            
            # FAIL-FAST: Quick pre-check with very small max_iter to detect obvious failures early
            # This catches over-regularization cases that would zero out quickly
            elastic_net_failed = False  # Flag to skip expensive operations if quick check fails
            if original_max_iter > 50:
                try:
                    quick_config = elastic_net_config.copy()
                    quick_config['max_iter'] = 50  # Very quick check
                    quick_steps = [
                        ('scaler', StandardScaler()),
                        ('model', est_cls(**quick_config))
                    ]
                    quick_pipeline = Pipeline(quick_steps)
                    quick_pipeline.fit(X_dense, y)
                    quick_model = quick_pipeline.named_steps['model']
                    quick_coef = quick_model.coef_
                    if len(quick_coef.shape) > 1:
                        quick_importance = np.abs(quick_coef).max(axis=0)
                    else:
                        quick_importance = np.abs(quick_coef)
                    
                    # If quick check shows all zeros, fail immediately without full fit
                    if np.all(quick_importance == 0) or np.sum(quick_importance) == 0:
                        raise ValueError("Elastic Net: All coefficients are zero (over-regularized or no signal). Model invalid.")
                except ValueError as e:
                    # Handle fail-fast signal gracefully - skip expensive operations
                    error_msg = str(e)
                    if "All coefficients are zero" in error_msg or "over-regularized" in error_msg or "no signal" in error_msg:
                        logger.debug(f"    Elastic Net: {error_msg} (skipping full fit)")
                        primary_score = np.nan
                        model_metrics['elastic_net'] = {'roc_auc': np.nan} if task_type == TaskType.BINARY_CLASSIFICATION else {'r2': np.nan} if task_type == TaskType.REGRESSION else {'accuracy': np.nan}
                        model_scores['elastic_net'] = np.nan
                        all_feature_importances['elastic_net'] = {}  # Record failure
                        importance_magnitudes.append(0.0)
                        elastic_net_failed = True  # Set flag to skip expensive operations
                    else:
                        # Other ValueErrors - re-raise
                        raise
                except Exception:
                    # Other exceptions from quick check - continue with full fit
                    pass
            
            # Skip expensive operations if quick check detected failure
            if not elastic_net_failed:
                scores = cross_val_score(pipeline, X_dense, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                valid_scores = scores[~np.isnan(scores)]
                primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
                
                # Fit on full data for importance extraction (CV is done elsewhere)
                # Use threading utilities for smart thread management
                if _THREADING_UTILITIES_AVAILABLE:
                    # Get thread plan based on family (linear models use MKL threads)
                    plan = plan_for_family('ElasticNet', total_threads=default_threads())
                    # Use thread_guard context manager for safe thread control
                    with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                        pipeline.fit(X_dense, y)
                else:
                    # Fallback: manual thread management
                    pipeline.fit(X_dense, y)
                
                # Compute and store full task-aware metrics
                if not np.isnan(primary_score):
                    _compute_and_store_metrics('elastic_net', pipeline, X_dense, y, primary_score, task_type)
                
                # Extract coefficients from the fitted model
                model = pipeline.named_steps['model']
                # FIX: Handle both 1D (binary) and 2D (multiclass) coef_ shapes
                coef = model.coef_
                if len(coef.shape) > 1:
                    # Multiclass: use max absolute coefficient across classes
                    importance = np.abs(coef).max(axis=0)
                else:
                    # Binary or regression: use absolute coefficients
                    importance = np.abs(coef)
                
                # Update feature_names to match dense array
                feature_names = feature_names_dense
                
                # FAIL-FAST: Validate importance immediately and handle gracefully if all zeros
                if np.all(importance == 0) or np.sum(importance) == 0:
                    # Handle gracefully instead of raising
                    logger.debug(f"    Elastic Net: All coefficients are zero after full fit (over-regularized or no signal)")
                    primary_score = np.nan
                    model_metrics['elastic_net'] = {'roc_auc': np.nan} if task_type == TaskType.BINARY_CLASSIFICATION else {'r2': np.nan} if task_type == TaskType.REGRESSION else {'accuracy': np.nan}
                    model_scores['elastic_net'] = np.nan
                    all_feature_importances['elastic_net'] = {}
                    importance_magnitudes.append(0.0)
                    # Don't raise - just mark as failed and continue
                else:
                    # Store all feature importances for detailed export (same pattern as other models)
                    # CRITICAL: Align importance to feature_names order to ensure fingerprint match
                    importance_series = pd.Series(importance, index=feature_names)
                    importance_series = importance_series.reindex(feature_names, fill_value=0.0)
                    importance_dict = importance_series.to_dict()
                    all_feature_importances['elastic_net'] = importance_dict
                    
                    # Calculate importance ratio (for metrics)
                    if len(importance) > 0:
                        total_importance = np.sum(importance)
                        if total_importance > 0:
                            top_fraction = _get_importance_top_fraction()
                            top_k = max(1, int(len(importance) * top_fraction))
                            top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                            importance_ratio = top_importance_sum / total_importance
                        else:
                            importance_ratio = 0.0
                    else:
                        importance_ratio = 0.0
                    importance_magnitudes.append(importance_ratio)
            
            # Log timing
            elastic_net_elapsed = time.time() - elastic_net_start_time
            timing_data['elastic_net'] = elastic_net_elapsed
            if timing_log_enabled and elastic_net_elapsed >= timing_log_threshold_seconds:
                logger.info(f"  ⏱️  Elastic Net timing: {elastic_net_elapsed:.2f} seconds")
        except ImportError as e:
            elastic_net_elapsed = time.time() - elastic_net_start_time
            timing_data['elastic_net'] = elastic_net_elapsed
            logger.error(f"❌ Elastic Net not available: {e} - timing: {elastic_net_elapsed:.2f} seconds")
            all_feature_importances['elastic_net'] = {}  # Record failure
        except Exception as e:
            elastic_net_elapsed = time.time() - elastic_net_start_time
            timing_data['elastic_net'] = elastic_net_elapsed
            if timing_log_enabled:
                logger.error(f"❌ Elastic Net failed after {elastic_net_elapsed:.2f} seconds: {e}", exc_info=True)
            all_feature_importances['elastic_net'] = {}  # Record failure
    
    # Mutual Information
    if 'mutual_information' in model_families:
        mutual_information_start_time = time.time()
        logger.info(f"  🚀 Starting Mutual Information training...")
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
            
            # Mutual information doesn't handle NaN - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Get config values
            mi_config = get_model_config('mutual_information', multi_model_config)
            
            # Get seed from SST (determinism system) - no hardcoded defaults
            mi_seed = mi_config.get('seed')
            if mi_seed is None:
                from TRAINING.common.determinism import stable_seed_from
                mi_seed = stable_seed_from(['mutual_information', target_column if target_column else 'default'])
            
            # Suppress warnings for zero-variance features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if is_binary or is_multiclass:
                    importance = mutual_info_classif(X_dense, y, 
                                                    random_state=mi_seed,
                                                    discrete_features=mi_config.get('discrete_features', 'auto'))
                else:
                    importance = mutual_info_regression(X_dense, y, 
                                                       random_state=mi_seed,
                                                       discrete_features=mi_config.get('discrete_features', 'auto'))
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # Handle NaN/inf
            importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Store all feature importances for detailed export (same pattern as other models)
            # CRITICAL: Align importance to feature_names order to ensure fingerprint match
            importance_series = pd.Series(importance, index=feature_names)
            importance_series = importance_series.reindex(feature_names, fill_value=0.0)
            importance_dict = importance_series.to_dict()
            all_feature_importances['mutual_information'] = importance_dict
            
            # Mutual information doesn't have R², so we use a proxy based on max MI
            # Normalize to 0-1 scale for importance
            if len(importance) > 0 and np.max(importance) > 0:
                importance_normalized = importance / np.max(importance)
                total_importance = np.sum(importance_normalized)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance_normalized) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importance_normalized)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            
            # For mutual information, we can't compute R² directly
            # Use a proxy: higher MI concentration = better predictability
            # Scale to approximate R² range (0-0.3 for good targets)
            model_scores['mutual_information'] = min(0.3, importance_ratio * 0.3)
            importance_magnitudes.append(importance_ratio)
            
            # Log timing
            mutual_information_elapsed = time.time() - mutual_information_start_time
            timing_data['mutual_information'] = mutual_information_elapsed
            if timing_log_enabled and mutual_information_elapsed >= timing_log_threshold_seconds:
                logger.info(f"  ⏱️  Mutual Information timing: {mutual_information_elapsed:.2f} seconds")
        except ImportError as e:
            mutual_information_elapsed = time.time() - mutual_information_start_time
            timing_data['mutual_information'] = mutual_information_elapsed
            logger.error(f"❌ Mutual Information not available: {e} - timing: {mutual_information_elapsed:.2f} seconds")
            all_feature_importances['mutual_information'] = {}  # Record failure
        except Exception as e:
            mutual_information_elapsed = time.time() - mutual_information_start_time
            timing_data['mutual_information'] = mutual_information_elapsed
            if timing_log_enabled:
                logger.error(f"❌ Mutual Information failed after {mutual_information_elapsed:.2f} seconds: {e}", exc_info=True)
            all_feature_importances['mutual_information'] = {}  # Record failure
    
    # Univariate Selection
    if 'univariate_selection' in model_families:
        try:
            from sklearn.feature_selection import f_regression, f_classif
            from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
            
            # F-tests don't handle NaN - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Suppress division by zero warnings (expected for zero-variance features)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if is_binary or is_multiclass:
                    scores, pvalues = f_classif(X_dense, y)
                else:
                    scores, pvalues = f_regression(X_dense, y)
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # Handle NaN/inf in scores (from zero-variance features)
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize F-statistics
            if len(scores) > 0 and np.max(scores) > 0:
                importance = scores / np.max(scores)
            else:
                importance = np.zeros(len(feature_names))
            
            # Store all feature importances for detailed export (same pattern as other models)
            # CRITICAL: Align importance to feature_names order to ensure fingerprint match
            importance_series = pd.Series(importance, index=feature_names)
            importance_series = importance_series.reindex(feature_names, fill_value=0.0)
            importance_dict = importance_series.to_dict()
            all_feature_importances['univariate_selection'] = importance_dict
            
            if len(importance) > 0 and np.sum(importance) > 0:
                total_importance = np.sum(importance)
                top_fraction = _get_importance_top_fraction()
                top_k = max(1, int(len(importance) * top_fraction))
                top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                importance_ratio = top_importance_sum / total_importance
            else:
                importance_ratio = 0.0
            
            # F-statistics don't have R², use proxy
            model_scores['univariate_selection'] = min(0.3, importance_ratio * 0.3)
            importance_magnitudes.append(importance_ratio)
        except ImportError as e:
            logger.error(f"❌ Univariate Selection not available: {e}")
            all_feature_importances['univariate_selection'] = {}  # Record failure
        except Exception as e:
            logger.error(f"❌ Univariate Selection failed: {e}", exc_info=True)
            all_feature_importances['univariate_selection'] = {}  # Record failure
    
    # RFE
    if 'rfe' in model_families:
        try:
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.impute import SimpleImputer
            
            # RFE uses RandomForest which handles NaN, but let's impute for consistency
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Get config values with defaults
            rfe_config = get_model_config('rfe', multi_model_config)
            # FIX: Use .get() with default to prevent KeyError
            # Default to 20% of features or top_k if available, but at least 1
            default_n_features = max(1, int(0.2 * X_imputed.shape[1]))
            n_features_to_select = min(rfe_config.get('n_features_to_select', default_n_features), X_imputed.shape[1])
            step = rfe_config.get('step', 5)
            
            # Use random_forest config for RFE estimator
            rf_config = get_model_config('random_forest', multi_model_config)
            
            if is_binary or is_multiclass:
                estimator = RandomForestClassifier(**rf_config)
            else:
                estimator = RandomForestRegressor(**rf_config)
            
            selector = RFE(estimator, n_features_to_select=n_features_to_select, step=step)
            # Use threading utilities for smart thread management
            if _THREADING_UTILITIES_AVAILABLE:
                # Get thread plan based on family (RFE uses RandomForest estimator)
                plan = plan_for_family('RandomForest', total_threads=default_threads())
                # Set n_jobs on estimator from plan (OMP threads for RandomForest)
                if hasattr(estimator, 'set_params') and 'n_jobs' in estimator.get_params():
                    estimator.set_params(n_jobs=plan['OMP'])
                # Use thread_guard context manager for safe thread control
                with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                    selector.fit(X_imputed, y)
            else:
                # Fallback: manual thread management
                selector.fit(X_imputed, y)
            
            # Get R² using cross-validation on selected features (proper validation)
            selected_features = selector.support_
            if np.any(selected_features):
                X_selected = X_imputed[:, selected_features]
                # Quick RF for scoring (use smaller config)
                quick_rf_config = get_model_config('random_forest', multi_model_config).copy()
                # Use smaller model for quick scoring
                quick_rf_config['n_estimators'] = 50
                quick_rf_config['max_depth'] = 8
                
                if is_binary or is_multiclass:
                    quick_rf = RandomForestClassifier(**quick_rf_config)
                else:
                    quick_rf = RandomForestRegressor(**quick_rf_config)
                
                # Use cross-validation for proper validation (not training score)
                scores = cross_val_score(quick_rf, X_selected, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                valid_scores = scores[~np.isnan(scores)]
                model_scores['rfe'] = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            else:
                model_scores['rfe'] = np.nan
            
            # Convert ranking to importance
            ranking = selector.ranking_
            importance = 1.0 / (ranking + 1e-6)
            
            # Store all feature importances for detailed export (same pattern as other models)
            # CRITICAL: Align importance to feature_names order to ensure fingerprint match
            importance_series = pd.Series(importance, index=feature_names)
            importance_series = importance_series.reindex(feature_names, fill_value=0.0)
            importance_dict = importance_series.to_dict()
            all_feature_importances['rfe'] = importance_dict
            
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except ImportError as e:
            logger.error(f"❌ RFE not available: {e}")
            all_feature_importances['rfe'] = {}  # Record failure
        except Exception as e:
            logger.error(f"❌ RFE failed: {e}", exc_info=True)
            all_feature_importances['rfe'] = {}  # Record failure
    
    # Boruta
    if 'boruta' in model_families:
        boruta_start_time = time.time()
        logger.info(f"  🚀 Starting Boruta training...")
        boruta_iteration_times = []  # Track time per iteration
        
        # Conditional execution gate (SST: all thresholds from config)
        boruta_should_run = True
        skip_reason = None
        
        try:
            boruta_config_check = get_model_config('boruta', multi_model_config)
            boruta_enabled = boruta_config_check.get('enabled', True)
            if not boruta_enabled:
                boruta_should_run = False
                skip_reason = "disabled in config"
            
            # Check dataset size thresholds (SST: from config)
            if boruta_should_run:
                # Load thresholds from config (SST)
                # TARGET_RANKING should use multi_model_config (ranking configs) first, then fallback to preprocessing_config
                max_features_threshold = 200  # Default
                max_samples_threshold = 20000  # Default
                try:
                    # SST: Use multi_model_config first (TARGET_RANKING should use ranking configs, not preprocessing)
                    # multi_model_config comes from CONFIG/ranking/targets/multi_model.yaml or CONFIG/ranking/features/multi_model.yaml
                    if multi_model_config:
                        boruta_config_check = get_model_config('boruta', multi_model_config)
                        if boruta_config_check and 'config' in boruta_config_check:
                            boruta_cfg_local = boruta_config_check.get('config', {})
                            max_features_threshold = boruta_cfg_local.get('max_features_threshold', 200)
                            max_samples_threshold = boruta_cfg_local.get('max_samples_threshold', 20000)
                    
                    # Only fallback to preprocessing_config if multi_model_config not available or missing values
                    if max_features_threshold == 200 and max_samples_threshold == 20000:
                        from CONFIG.config_loader import get_cfg
                        boruta_cfg = get_cfg("preprocessing.multi_model_feature_selection.boruta", default={}, config_name="preprocessing_config")
                        max_features_threshold = boruta_cfg.get('max_features_threshold', 200)
                        max_samples_threshold = boruta_cfg.get('max_samples_threshold', 20000)
                except Exception:
                    # Fallback if config not available
                    max_features_threshold = 200
                    max_samples_threshold = 20000
                
                n_features = len(feature_names) if feature_names else X.shape[1] if X is not None else 0
                n_samples = len(y) if y is not None else X.shape[0] if X is not None else 0
                
                if n_features > max_features_threshold:
                    boruta_should_run = False
                    skip_reason = f"too many features ({n_features} > {max_features_threshold})"
                elif n_samples > max_samples_threshold:
                    # Check if subsampling is enabled (will be checked later)
                    subsample_cfg = boruta_cfg.get('subsample_large_datasets', {}) if 'boruta_cfg' in locals() else {}
                    subsample_enabled = subsample_cfg.get('enabled', True) if subsample_cfg else True
                    if not subsample_enabled:
                        boruta_should_run = False
                        skip_reason = f"too many samples ({n_samples} > {max_samples_threshold}) and subsampling disabled"
        except Exception as e:
            logger.debug(f"Failed to load Boruta conditional execution config: {e}, proceeding with Boruta")
            # If config load fails, proceed with Boruta (graceful degradation)
        
        if not boruta_should_run:
            boruta_elapsed = time.time() - boruta_start_time
            timing_data['boruta'] = boruta_elapsed
            logger.info(f"  ⏱️  Boruta SKIPPED - {skip_reason} (timing: {boruta_elapsed:.2f}s)")
            all_feature_importances['boruta'] = {}  # Record skip
        else:
            try:
                from boruta import BorutaPy
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
                
                # Boruta doesn't support NaN - use sklearn-safe conversion
                X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
                
                # Get config values
                boruta_config = get_model_config('boruta', multi_model_config)
                
                # Use random_forest config for Boruta estimator
                rf_config = get_model_config('random_forest', multi_model_config)
                
                # Get seed from SST (determinism system) - no hardcoded defaults
                boruta_seed = boruta_config.get('seed')
                if boruta_seed is None:
                    from TRAINING.common.determinism import stable_seed_from
                    boruta_seed = stable_seed_from(['boruta', target_column if target_column else 'default'])
                
                # Remove all seed keys from rf_config to prevent double argument error
                rf_config_clean = rf_config.copy()
                for seed_key in ['seed', 'random_state', 'random_seed']:
                    rf_config_clean.pop(seed_key, None)
                
                if is_binary or is_multiclass:
                    rf = RandomForestClassifier(**rf_config_clean, random_state=boruta_seed)
                else:
                    rf = RandomForestRegressor(**rf_config_clean, random_state=boruta_seed)
                
                boruta = BorutaPy(rf, n_estimators='auto', verbose=0, 
                                random_state=boruta_seed,
                                max_iter=boruta_config.get('max_iter', 100))
                
                # Track Boruta fit time and apply time-budget (SST: budget from config)
                import signal
                boruta_fit_start = time.time()
                boruta_max_time_seconds = None
                max_time_minutes = 10  # Default
                try:
                    # SST: Use multi_model_config first (TARGET_RANKING should use ranking configs, not preprocessing)
                    # multi_model_config comes from CONFIG/ranking/targets/multi_model.yaml or CONFIG/ranking/features/multi_model.yaml
                    if multi_model_config and boruta_config:
                        max_time_minutes = boruta_config.get('max_time_minutes', 10)
                    elif multi_model_config:
                        # Try to get boruta config from multi_model_config directly
                        boruta_config_check = get_model_config('boruta', multi_model_config)
                        if boruta_config_check and 'config' in boruta_config_check:
                            boruta_cfg_local = boruta_config_check.get('config', {})
                            max_time_minutes = boruta_cfg_local.get('max_time_minutes', 10)
                    # Only fallback to preprocessing_config if multi_model_config not available
                    if max_time_minutes == 10:  # Still using default
                        from CONFIG.config_loader import get_cfg
                        boruta_cfg_time = get_cfg('preprocessing.multi_model_feature_selection.boruta', default={}, config_name='preprocessing_config')
                        max_time_minutes = boruta_cfg_time.get('max_time_minutes', 10)
                    boruta_max_time_seconds = max_time_minutes * 60
                except Exception:
                    pass  # Use defaults if config not available
                
                # Time-budget wrapper for Boruta fit (SST: budget from config)
                boruta_timed_out = False
                boruta_budget_hit = False
                
                def boruta_timeout_handler(signum, frame):
                    nonlocal boruta_timed_out, boruta_budget_hit
                    boruta_timed_out = True
                    boruta_budget_hit = True
                    raise TimeoutError(f"Boruta training exceeded {boruta_max_time_seconds/60:.0f} minute time budget")
                
                # Set up timeout if budget is configured (Unix only - Windows will use soft check)
                timeout_set = False
                if boruta_max_time_seconds is not None:
                    try:
                        signal.signal(signal.SIGALRM, boruta_timeout_handler)
                        signal.alarm(int(boruta_max_time_seconds))
                        timeout_set = True
                    except (AttributeError, OSError):
                        # Windows doesn't support SIGALRM - will use soft timeout check
                        logger.debug(f"  Boruta: Timeout signal not available on this platform, using soft timeout check")
                
                try:
                    boruta.fit(X_dense, y)
                    
                    # Cancel timeout if it was set
                    if timeout_set:
                        try:
                            signal.alarm(0)
                        except (AttributeError, OSError):
                            pass
                    
                    # Soft timeout check (for Windows or if signal didn't fire)
                    boruta_fit_elapsed = time.time() - boruta_fit_start
                    if boruta_max_time_seconds is not None and boruta_fit_elapsed > boruta_max_time_seconds:
                        boruta_budget_hit = True
                        logger.warning(f"  ⚠️  Boruta BORUTA_BUDGET_HIT - Training took {boruta_fit_elapsed/60:.1f} minutes (exceeded {boruta_max_time_seconds/60:.0f} min budget)")
                        # Continue with current results (quality-safe: use what we have)
                        
                except (TimeoutError, ValueError) as e:
                    # TimeoutError can be raised directly, or Boruta may catch it and re-raise as ValueError
                    # Check if this is actually a timeout by examining the error message
                    error_msg = str(e)
                    is_timeout = (
                        isinstance(e, TimeoutError) or
                        "timeout" in error_msg.lower() or
                        "time budget" in error_msg.lower() or
                        "exceeded" in error_msg.lower()
                    )
                    
                    boruta_fit_elapsed = time.time() - boruta_fit_start
                    
                    if is_timeout:
                        boruta_budget_hit = True
                        logger.warning(
                            f"  ⚠️  Boruta BORUTA_BUDGET_HIT - Training exceeded {boruta_max_time_seconds/60:.0f} minute budget "
                            f"after {boruta_fit_elapsed/60:.1f} minutes. Error: {error_msg}"
                        )
                        # Cancel timeout
                        if timeout_set:
                            try:
                                signal.alarm(0)
                            except (AttributeError, OSError):
                                pass
                        # Check if Boruta has partial results we can use
                        if hasattr(boruta, 'ranking_') and boruta.ranking_ is not None:
                            logger.info(f"  Boruta: Using partial results from interrupted fit")
                            # Continue with partial results (quality-safe: better than nothing)
                        else:
                            # No partial results - log and skip Boruta gracefully
                            logger.warning(f"  Boruta: No partial results available, skipping Boruta feature selection")
                            boruta_timed_out = True
                            boruta_budget_hit = True
                            # Skip the rest of Boruta processing
                            raise  # Re-raise to be caught by outer handler which will skip Boruta
                    else:
                        # Not a timeout - re-raise the original error
                        raise
                
                # Only continue if Boruta completed successfully (not timed out)
                if boruta_timed_out or boruta_budget_hit:
                    raise ValueError(f"Boruta timed out or exceeded budget - skipping")
                
                boruta_fit_elapsed = time.time() - boruta_fit_start
                boruta_iteration_times.append(boruta_fit_elapsed)  # Store overall fit time
                
                # Get R² using cross-validation on selected features (proper validation)
                selected_features = boruta.support_
                if np.any(selected_features):
                    X_selected = X_dense[:, selected_features]
                    # Quick RF for scoring (use smaller config)
                    quick_rf_config = get_model_config('random_forest', multi_model_config).copy()
                    # Use smaller model for quick scoring
                    quick_rf_config['n_estimators'] = 50
                    quick_rf_config['max_depth'] = 8
                    
                    if is_binary or is_multiclass:
                        quick_rf = RandomForestClassifier(**quick_rf_config)
                    else:
                        quick_rf = RandomForestRegressor(**quick_rf_config)
                    
                    # Use cross-validation for proper validation (not training score)
                    scores = cross_val_score(quick_rf, X_selected, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                    valid_scores = scores[~np.isnan(scores)]
                    model_scores['boruta'] = valid_scores.mean() if len(valid_scores) > 0 else np.nan
                else:
                    model_scores['boruta'] = np.nan
                
                # Update feature_names to match dense array
                feature_names = feature_names_dense
                
                # Convert to importance
                ranking = boruta.ranking_
                selected = boruta.support_
                importance = np.where(selected, 1.0, np.where(ranking == 2, 0.5, 0.1))
                if len(importance) > 0:
                    total_importance = np.sum(importance)
                    if total_importance > 0:
                        top_fraction = _get_importance_top_fraction()
                        top_k = max(1, int(len(importance) * top_fraction))
                        top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                        importance_ratio = top_importance_sum / total_importance
                    else:
                        importance_ratio = 0.0
                else:
                    importance_ratio = 0.0
                importance_magnitudes.append(importance_ratio)
                
                # Store all feature importances for detailed export (same pattern as other models)
                # CRITICAL: Align importance to feature_names order to ensure fingerprint match
                importance_series = pd.Series(importance, index=feature_names)
                importance_series = importance_series.reindex(feature_names, fill_value=0.0)
                importance_dict = importance_series.to_dict()
                all_feature_importances['boruta'] = importance_dict
                
                # Log detailed Boruta timing
                boruta_elapsed = time.time() - boruta_start_time
                timing_data['boruta'] = boruta_elapsed
                if timing_log_enabled and boruta_elapsed >= timing_log_threshold_seconds:
                    fit_time_str = f"{boruta_fit_elapsed:.2f}" if boruta_fit_elapsed else "N/A"
                    logger.info(f"  ⏱️  Boruta timing: {boruta_elapsed:.2f} seconds (fit: {fit_time_str}s)")
                    if hasattr(boruta, 'n_iter_'):
                        logger.info(f"     Boruta iterations: {boruta.n_iter_}/{boruta_config.get('max_iter', 100)}")
            except ImportError as e:
                boruta_elapsed = time.time() - boruta_start_time
                timing_data['boruta'] = boruta_elapsed
                logger.error(f"❌ Boruta not available (pip install Boruta): {e} - timing: {boruta_elapsed:.2f} seconds")
                all_feature_importances['boruta'] = {}  # Record failure
            except Exception as e:
                boruta_elapsed = time.time() - boruta_start_time
                timing_data['boruta'] = boruta_elapsed
                error_msg = str(e)
                # Check if this is a timeout/budget error (already handled gracefully)
                # Boruta may catch TimeoutError and re-raise as ValueError with "Please check your X and y variable"
                # We detect this by checking if the error occurred after a timeout period
                is_timeout_error = (
                    "timeout" in error_msg.lower() or
                    "time budget" in error_msg.lower() or
                    "exceeded" in error_msg.lower() or
                    "Boruta timed out" in error_msg or
                    "BORUTA_BUDGET_HIT" in error_msg or
                    ("Please check your X and y variable" in error_msg and boruta_elapsed >= (boruta_max_time_seconds or 600))
                )
                if is_timeout_error:
                    logger.warning(f"⚠️  Boruta skipped due to timeout/budget after {boruta_elapsed:.2f} seconds: {error_msg}")
                else:
                    if timing_log_enabled:
                        logger.error(f"❌ Boruta failed after {boruta_elapsed:.2f} seconds: {e}", exc_info=True)
                all_feature_importances['boruta'] = {}  # Record failure
    
    # Stability Selection
    if 'stability_selection' in model_families:
        stability_selection_start_time = time.time()
        logger.info(f"  🚀 Starting Stability Selection training...")
        try:
            from sklearn.linear_model import LassoCV, LogisticRegressionCV
            from TRAINING.common.utils.sklearn_safe import make_sklearn_dense_X
            
            # Stability selection uses Lasso/LogisticRegression which don't handle NaN
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Get config values
            stability_config = get_model_config('stability_selection', multi_model_config)
            n_bootstrap = stability_config.get('n_bootstrap', 50)
            # Get seed from SST (determinism system) - no hardcoded defaults
            seed = stability_config.get('seed')
            if seed is None:
                from TRAINING.common.determinism import stable_seed_from
                seed = stable_seed_from(['stability_selection', target_column if target_column else 'default'])
            stability_cv = stability_config.get('cv', 3)
            stability_n_jobs = stability_config.get('n_jobs', 1)
            stability_cs = stability_config.get('Cs', 10)
            stability_scores = np.zeros(X_dense.shape[1])
            bootstrap_r2_scores = []
            
            # Use lasso config for stability selection models
            lasso_config = get_model_config('lasso', multi_model_config)
            
            for i in range(n_bootstrap):
                # Use deterministic seed for bootstrap sampling
                from TRAINING.common.determinism import stable_seed_from
                bootstrap_seed = stable_seed_from(['bootstrap', target_column if 'target_column' in locals() else 'default', f'iter_{i}'])
                np.random.seed(bootstrap_seed)
                indices = np.random.choice(len(X_dense), size=len(X_dense), replace=True)
                X_boot, y_boot = X_dense[indices], y[indices]
                
                try:
                    # Use TimeSeriesSplit for internal CV (even though bootstrap breaks temporal order,
                    # this maintains consistency with the rest of the codebase)
                    # Clean config to prevent double seed argument
                    from TRAINING.common.utils.config_cleaner import clean_config_for_estimator
                    if is_binary or is_multiclass:
                        lr_config = {'Cs': stability_cs, 'cv': tscv, 'max_iter': lasso_config.get('max_iter', 1000), 'n_jobs': stability_n_jobs}
                        lr_config_clean = clean_config_for_estimator(LogisticRegressionCV, lr_config, extra_kwargs={'random_state': seed}, family_name='stability_selection')
                        model = LogisticRegressionCV(**lr_config_clean, random_state=seed)
                    else:
                        lasso_config_clean_dict = {'cv': tscv, 'max_iter': lasso_config.get('max_iter', 1000), 'n_jobs': stability_n_jobs}
                        lasso_config_clean = clean_config_for_estimator(LassoCV, lasso_config_clean_dict, extra_kwargs={'random_state': seed}, family_name='stability_selection')
                        model = LassoCV(**lasso_config_clean, random_state=seed)
                    
                    # Use threading utilities for smart thread management
                    if _THREADING_UTILITIES_AVAILABLE:
                        # Get thread plan based on family (linear models use MKL threads)
                        plan = plan_for_family('Lasso', total_threads=default_threads())
                        # Set n_jobs from plan if model supports it
                        if hasattr(model, 'set_params') and 'n_jobs' in model.get_params():
                            model.set_params(n_jobs=plan['OMP'])
                        # Use thread_guard context manager for safe thread control
                        with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                            model.fit(X_boot, y_boot)
                    else:
                        # Fallback: manual thread management
                        model.fit(X_boot, y_boot)
                    coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                    stability_scores += (np.abs(coef) > 1e-6).astype(int)
                    
                    # Get R² using cross-validation (proper validation, not training score)
                    # Note: Bootstrap samples break temporal order, but we still use TimeSeriesSplit
                    # for consistency (it won't help here, but maintains the pattern)
                    # Use a quick model for CV scoring
                    if is_binary or is_multiclass:
                        lr_cv_config = {'Cs': [1.0], 'cv': tscv, 'max_iter': lasso_config.get('max_iter', 1000), 'n_jobs': 1}
                        lr_cv_config_clean = clean_config_for_estimator(LogisticRegressionCV, lr_cv_config, extra_kwargs={'random_state': seed}, family_name='stability_selection')
                        cv_model = LogisticRegressionCV(**lr_cv_config_clean, random_state=seed)
                    else:
                        lasso_cv_config = {'cv': tscv, 'max_iter': lasso_config.get('max_iter', 1000), 'n_jobs': 1}
                        lasso_cv_config_clean = clean_config_for_estimator(LassoCV, lasso_cv_config, extra_kwargs={'random_state': seed}, family_name='stability_selection')
                        cv_model = LassoCV(**lasso_cv_config_clean, random_state=seed)
                    cv_scores = cross_val_score(cv_model, X_boot, y_boot, cv=tscv, scoring=scoring, n_jobs=1, error_score=np.nan)
                    valid_cv_scores = cv_scores[~np.isnan(cv_scores)]
                    if len(valid_cv_scores) > 0:
                        bootstrap_r2_scores.append(valid_cv_scores.mean())
                except Exception:
                    continue
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # Average R² across bootstraps
            if bootstrap_r2_scores:
                model_scores['stability_selection'] = np.mean(bootstrap_r2_scores)
            else:
                model_scores['stability_selection'] = np.nan
            
            # Normalize stability scores to importance
            importance = stability_scores / n_bootstrap
            
            # Store all feature importances for detailed export (same pattern as other models)
            # CRITICAL: Align importance to feature_names order to ensure fingerprint match
            importance_series = pd.Series(importance, index=feature_names)
            importance_series = importance_series.reindex(feature_names, fill_value=0.0)
            importance_dict = importance_series.to_dict()
            all_feature_importances['stability_selection'] = importance_dict
            
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
            
            # Log timing
            stability_selection_elapsed = time.time() - stability_selection_start_time
            timing_data['stability_selection'] = stability_selection_elapsed
            if timing_log_enabled and stability_selection_elapsed >= timing_log_threshold_seconds:
                logger.info(f"  ⏱️  Stability Selection timing: {stability_selection_elapsed:.2f} seconds")
        except ImportError as e:
            stability_selection_elapsed = time.time() - stability_selection_start_time
            timing_data['stability_selection'] = stability_selection_elapsed
            logger.error(f"❌ Stability Selection not available: {e} - timing: {stability_selection_elapsed:.2f} seconds")
            all_feature_importances['stability_selection'] = {}  # Record failure
        except Exception as e:
            stability_selection_elapsed = time.time() - stability_selection_start_time
            timing_data['stability_selection'] = stability_selection_elapsed
            if timing_log_enabled:
                logger.error(f"❌ Stability Selection failed after {stability_selection_elapsed:.2f} seconds: {e}", exc_info=True)
            all_feature_importances['stability_selection'] = {}  # Record failure
    
    # Histogram Gradient Boosting
    if 'histogram_gradient_boosting' in model_families:
        histogram_gb_start_time = time.time()
        logger.info(f"  🚀 Starting Histogram Gradient Boosting training...")
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
            
            # Get config values
            hgb_config = get_model_config('histogram_gradient_boosting', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(hgb_config, dict):
                hgb_config = {}
            # Remove task-specific parameters (loss is set automatically by classifier/regressor)
            hgb_config_clean = {k: v for k, v in hgb_config.items() if k != 'loss'}
            
            if is_binary or is_multiclass:
                model = HistGradientBoostingClassifier(**hgb_config_clean)
            else:
                model = HistGradientBoostingRegressor(**hgb_config_clean)
            
            scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # Train once to get importance and full metrics
            # Use threading utilities for smart thread management
            if _THREADING_UTILITIES_AVAILABLE:
                # Get thread plan based on family (HistGradientBoosting is OMP-heavy)
                plan = plan_for_family('HistGradientBoosting', total_threads=default_threads())
                # Set n_jobs from plan (OMP threads for HistGradientBoosting)
                model.set_params(n_jobs=plan['OMP'])
                # Use thread_guard context manager for safe thread control
                with thread_guard(omp=plan['OMP'], mkl=plan['MKL']):
                    model.fit(X, y)
            else:
                # Fallback: manual thread management
                model.fit(X, y)
            
            # Compute and store full task-aware metrics
            if not np.isnan(primary_score):
                _compute_and_store_metrics('histogram_gradient_boosting', model, X, y, primary_score, task_type)
            if hasattr(model, 'feature_importances_'):
                # Use percentage of total importance in top 10% features (0-1 scale, interpretable)
                importances = model.feature_importances_
                total_importance = np.sum(importances)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importances) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importances)[-top_k:])
                    # Normalize to 0-1: what % of total importance is in top 10%?
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
                importance_magnitudes.append(importance_ratio)
            
            # Log timing
            histogram_gb_elapsed = time.time() - histogram_gb_start_time
            timing_data['histogram_gradient_boosting'] = histogram_gb_elapsed
            if timing_log_enabled and histogram_gb_elapsed >= timing_log_threshold_seconds:
                logger.info(f"  ⏱️  Histogram Gradient Boosting timing: {histogram_gb_elapsed:.2f} seconds")
        except Exception as e:
            histogram_gb_elapsed = time.time() - histogram_gb_start_time
            timing_data['histogram_gradient_boosting'] = histogram_gb_elapsed
            if timing_log_enabled:
                logger.warning(f"Histogram Gradient Boosting failed after {histogram_gb_elapsed:.2f} seconds: {e}")
    
    mean_importance = np.mean(importance_magnitudes) if importance_magnitudes else 0.0
    
    # model_scores already contains primary scores (backward compatible)
    # model_metrics contains full metrics dict
    # all_suspicious_features contains leak detection results (aggregated across all models)
    # all_feature_importances contains detailed per-feature importances for export
    # Log summary timing for all importance producers
    overall_elapsed = time.time() - overall_start_time
    if timing_log_enabled and overall_elapsed >= timing_log_threshold_seconds:
        logger.info(f"⏱️  Total importance producer timing: {overall_elapsed:.2f} seconds")
        if timing_data:
            sorted_timing = sorted(timing_data.items(), key=lambda x: x[1], reverse=True)
            for model_family, elapsed in sorted_timing:
                if elapsed >= timing_log_threshold_seconds:
                    percentage = (elapsed / overall_elapsed) * 100 if overall_elapsed > 0 else 0
                    logger.info(f"   {model_family}: {elapsed:.2f}s ({percentage:.1f}%)")
    
    return model_metrics, model_scores, mean_importance, all_suspicious_features, all_feature_importances, fold_timestamps, _perfect_correlation_models


# Import from modular components
from TRAINING.ranking.predictability.model_evaluation.reporting import (
    save_feature_importances as _save_feature_importances,
    log_suspicious_features as _log_suspicious_features
)
# REMOVED: Conflicting import - detect_leakage is already imported from leakage_detection.py at line 127
# from TRAINING.ranking.predictability.model_evaluation.leakage_helpers import detect_leakage


# calculate_composite_score is imported from TRAINING.ranking.predictability.composite_score
# (removed duplicate definition to use the imported version with definition/version support)


