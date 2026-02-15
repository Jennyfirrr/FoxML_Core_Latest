# Changelog

All notable changes to FoxML Core will be documented in this file.

## 2026-02-14

### Documentation

#### Added
- **HTML documentation book** — `bin/build-book.sh` compiles all ~800 markdown files into a browsable HTML book with MkDocs Material theme, full-text search, dark/light mode, and hierarchical navigation across 11 parts (executive overview through appendices)
- **PDF documentation pipeline** — `bin/build-book-pdf.sh` produces a single PDF with table of contents via Pandoc + XeLaTeX, covering key documents in reading order
- **mkdocs.yml configuration** — Complete navigation tree covering DOCS/, CONFIG/, TRAINING/, LIVE_TRADING/, DASHBOARD/, INTERNAL/, skills, and implementation plans

### RAW_SEQUENCE Mode

#### Fixed
- **3D raw OHLCV preprocessing corruption in TF sequence trainers** — `BaseModelTrainer.preprocess_data()` assumed 2D `(N, F)` input, silently destroying 3D `(N, T, 5)` raw OHLCV sequences: `colmask` became a 2D matrix, boolean indexing flattened the array to `(N, T*5)`, and the imputer fit on corrupted data. Added a 3D early-return branch that skips column masking and imputation, preserving sequence structure for CNN1D, LSTM, and Transformer trainers
- **`predict()` crash with raw OHLCV models** — CNN1D, LSTM, and Transformer `predict()` methods unconditionally flattened 3D input to 2D for preprocessing, then reshaped to `(N, T*5, 1)` instead of `(N, T, 5)`, causing `IndexError` during `post_fit_sanity`. Now uses `_is_3d_input` flag to bypass 2D preprocessing in raw sequence mode
- **Hardcoded `Input(shape=(dim, 1))` in sequence model architectures** — CNN1D, LSTM, and Transformer `_build_model()` always used 1 channel, ignoring multi-channel raw OHLCV input. Added `n_channels` parameter so models build `Input(shape=(seq_len, 5))` for raw OHLCV
- **`input_mode` missing from `model_meta.json`** — Training artifacts lacked `input_mode`, `sequence_length`, `sequence_channels`, and `sequence_normalization` fields required by LIVE_TRADING inference. Now populated from `routing_meta` when `input_mode=raw_sequence`

## 2026-02-13

### RAW_SEQUENCE Mode

#### Added
- **Auto-clamp sequence length to trading session** — New `auto_clamp` config knob under `pipeline.sequence`. When enabled with `gap_handling: "split"`, automatically reduces `seq_len` to fit within a single US trading session (390 min × 90%), preventing 0-sequence output when configured length exceeds session capacity. At 5m interval: clamps to 70 bars.

### Configuration

#### Changed
- **Default GPU config set to CPU-only** — `gpu.cuda_visible_devices` changed from `"0"` to `"-1"` in `CONFIG/pipeline/gpu.yaml` for environments without CUDA-enabled GPUs

### Experiment Configs

#### Added
- **Raw OHLCV 4GB experiment config** (`raw_ohlcv_4gb.yaml`) — Trains LSTM/Transformer/CNN1D on all 10 symbols with raw OHLCV bars, sized for a ~4GB RAM footprint (96-bar sequences, medium model capacity, lazy loading)
- **Raw OHLCV experiment template** (`_template_raw_ohlcv.yaml`) — Self-contained template for creating raw OHLCV sequence experiments with memory sizing guide, parameter documentation, and tuning knobs

### Performance Audit Improvements

#### Fixed
- **False-positive redundancy alerts in performance audit** - Fingerprints for `catboost.get_feature_importance`, `neural_network.permutation_importance`, and `RankingHarness.build_panel` now include target/symbol/view context, so structurally necessary calls across different targets and symbols are no longer flagged as wasted compute
- **Audit report now shows context** - Multiplier findings include `targets`, `symbols`, and `views` arrays; `all_calls` entries include `target`, `symbol`, and `view` fields for easier diagnosis

### RAW_SEQUENCE Mode

#### Fixed
- **`discover_targets` UnboundLocalError in RAW_SEQUENCE mode** — A redundant local import of `discover_targets` inside an `elif` branch caused Python to shadow the module-level import throughout the entire method, making the auto-discovery fallback crash with "cannot access local variable" when target ranking is skipped
- **RAW_SEQUENCE mode silently falling back to FEATURES mode** — Training function received `ExperimentConfig` dataclass (no `pipeline` field) instead of raw YAML dict, so `get_input_mode()` couldn't detect `pipeline.input_mode: raw_sequence` and defaulted to FEATURES; all targets then failed with 0 features from registry
- **`write_atomic_json` UnboundLocalError in training summary** — A local import from non-existent `TRAINING.common.utils.file_io` shadowed the working module-level import from `file_utils`, causing the training results summary write to crash
- **Lazy loading column projection missing OHLCV columns** — In RAW_SEQUENCE mode with lazy loading, the `__RAW_OHLCV_SEQUENCE__` sentinel placeholder was passed to `load_for_target` as a feature name; the loader silently dropped the non-existent column, never loading actual OHLCV data. Now detects RAW_SEQUENCE mode early and substitutes real channel names (open, high, low, close, volume)

### Pipeline Fixes

#### Fixed
- **`ret_1` lookback inference hard-fail** - 1-bar return feature was rejected by leakage budget because the `bars >= 2` guard excluded single-bar indicators; changed to `bars >= 1` for known indicator families (ret, sma, ema, vol, rsi, etc.)
- **Demo experiment config data path** - Updated `demo.yaml` to point at `data_labeled_v2` instead of non-existent `data_labeled_v3`
- **`get_input_mode` crash with ExperimentConfig object** - Function assumed dict input but received a dataclass; now handles both dict and ExperimentConfig types
- **Missing xgboost crashes all model training** - `model_fun/__init__.py` imported all CPU trainers unconditionally, so a missing `xgboost` package prevented even `lightgbm` from loading; now uses conditional imports per trainer
- **`release_data` crash on None DataFrame** - `unified_loader.py` called `len(df)` without guarding against None values in the data dict
- **CatBoost "Verbose period should be nonnegative"** - Newer CatBoost versions reject `verbose=0`; replaced with `logging_level='Silent'` for silent mode
- **NoneType `.get()` errors in reproducibility tracking** - Fixed chained `.get('key', {}).get(...)` patterns that fail when nested values (date_range, cs_config, prediction_fingerprint, best_metrics) are explicitly None instead of missing
- **PosixPath.startswith crash in run hash computation** - `iterdir_sorted()` returns Path objects but `.startswith()` was called directly instead of `.name.startswith()`
- **`horizon_minutes is None` for `fwd_ret_N` targets** - Added fallback pattern matching bare integers in target names (e.g., `fwd_ret_15` → 15 minutes)
- **Manifest path mismatch warning** - Validation check looked at `globals/manifest.json` but manifest is saved at `manifest.json`
- **Infinite loop freeze in directory walk** - `while results_dir.parent.exists()` loops in diff_telemetry and target_routing never terminate at filesystem root; replaced with bounded walks

## 2026-02-09

### Dashboard Hardening & Bug Fixes (Rounds 1-4)

Four rounds of systematic audits and fixes across the Rust TUI and Python IPC bridge, addressing 47 issues total.

#### Added
- **Polygon.io data download script** - `bin/download_polygon_data.py` downloads historical 5-minute OHLCV bars into FoxML-compatible parquet format with pagination, RTH filtering, rate limiting, and custom symbol lists

#### Fixed
- **Fake command palette commands** - 5 commands (`trading.pause`, `trading.resume`, `training.stop`, `config.edit`, `nav.models`) now call real logic instead of showing fake success notifications
- **Position table highlight when scrolled** - Selection tracking now uses scroll offset so the correct row is highlighted
- **WebSocket connection hang** - Added 5-second timeout so the connect button recovers if bridge is unreachable
- **View render errors silenced** - All `let _ =` on render/update calls replaced with `tracing::warn!` logging
- **Bridge timezone inconsistency** - 3 control endpoints (`pause`, `resume`, `kill_switch`) now use `datetime.now(timezone.utc)` matching all other endpoints
- **`get_metrics()` error schema incomplete** - Error fallback now returns all expected fields plus `"error"` field
- **Overview health indicator overflow** - Early return when `x >= area.right()` prevents wasted rendering cycles
- **Training event queue log level** - Changed from `debug` to `warning` to match trading/alpaca queue levels
- **Decision statistics mismatch** - `trade_count`/`hold_count`/`blocked_count` now computed before pagination limit, matching `total_decisions`
- **Type coercion crash in decision explanation** - Numeric format strings wrapped with `float()` coercion and try/except for corrupted trace data
- **`get_recent_events(count=0)` returns all events** - Added guard: `if count <= 0: return []`
- **Timestamp sorting with empty string default** - Changed default from `""` to `"0"` so missing timestamps sort last
- **Silent date parse error** - Invalid `since` parameter now logs a warning instead of bare `pass`
- **Timezone-naive datetimes in Alpaca stream** - 6 occurrences of `datetime.now()` changed to `datetime.now(timezone.utc)` in alpaca_stream.py
- **UTF-8 string slicing panic** - Event log and risk gauge truncation now uses `.chars().take(n)` instead of byte-index slicing that panics on multi-byte characters

#### Added
- **Bearer token auth on control endpoints** - Bridge generates random token on startup, writes to `/tmp/foxml_bridge_token`; Rust client reads and sends `Authorization: Bearer <token>` on POST requests
- **Sharpe ratio calculation** - Rolling P&L history with annualized Sharpe (`mean/std * sqrt(252)`) replaces the `None` stub
- **`src/config.rs` module** - Environment variable configuration for bridge URL, tmp paths, and project root:
  - `FOXML_BRIDGE_URL` (default: `127.0.0.1:8765`)
  - `FOXML_TMP_DIR` (default: `/tmp`)
  - `FOXML_ROOT` (default: `.`)
- **Desktop notification failure logging** - Failed `notify-send` calls now logged at `debug` level

#### Changed
- 15+ Rust files updated to use `crate::config::*` helpers instead of hardcoded strings
- Model selector paths now use `config::project_root()` instead of relative `PathBuf::from()`
- `DASHBOARD/dashboard/src/api/client.rs` - Added `auth_token` field and `authenticated_post()` helper

#### Files Created
- `DASHBOARD/dashboard/src/config.rs` - Centralized environment variable configuration

#### Files Modified
- `DASHBOARD/dashboard/src/app.rs` - Command palette wiring, render error logging
- `DASHBOARD/dashboard/src/views/trading.rs` - WS timeout, config bridge URL
- `DASHBOARD/dashboard/src/views/overview.rs` - Overflow guard, config paths
- `DASHBOARD/dashboard/src/views/model_selector.rs` - Config project root paths
- `DASHBOARD/dashboard/src/views/training.rs` - Config paths
- `DASHBOARD/dashboard/src/views/log_viewer.rs` - Config paths
- `DASHBOARD/dashboard/src/views/config_editor.rs` - Config paths
- `DASHBOARD/dashboard/src/views/file_browser.rs` - Config paths
- `DASHBOARD/dashboard/src/views/training_launcher.rs` - Config paths
- `DASHBOARD/dashboard/src/views/service_manager.rs` - Config paths
- `DASHBOARD/dashboard/src/views/settings.rs` - Config paths
- `DASHBOARD/dashboard/src/widgets/position_table.rs` - Scroll offset fix
- `DASHBOARD/dashboard/src/api/client.rs` - Auth token, authenticated POST
- `DASHBOARD/dashboard/src/ui/notification.rs` - Failure logging
- `DASHBOARD/dashboard/src/launcher/*.rs` - Config paths (6 files)
- `DASHBOARD/bridge/server.py` - Timezone, Sharpe, auth, error schema, decision stats, type safety
- `DASHBOARD/bridge/alpaca_stream.py` - Event count guard

---

## 2026-02-08

### Codebase Bug Fix Campaign

Systematic code review and fix of 36 bugs across all components, with 17 false positives identified and documented.

#### Fixed
- **Dashboard (16 bugs)** - Python bridge async I/O, CORS headers, systemd detection, timestamp handling, and Rust TUI rendering issues
- **CRITICAL (7 bugs)** - Cross-component issues including data corruption, race conditions, and import errors
- **LIVE_TRADING (7 bugs)** - Inference pipeline crashes, position corruption, short selling logic errors
- **TRAINING (6 bugs)** - Determinism violations, import errors, and code quality issues

#### Files Modified
- See commit messages `df3a1d7` through `a190726` for full file lists

---

### LIVE_TRADING Inference Pipeline

Wired raw OHLCV and cross-sectional ranking model support into the live inference pipeline.

#### Added
- **Input mode awareness** - `loader.py`, `inference.py`, `predictor.py` now check `input_mode` field from model metadata
- **Raw OHLCV inference path** - `_prepare_raw_sequence()` with case-insensitive OHLCV column matching and SST normalization
- **Cross-sectional ranking inference** - `predict_cross_section()` method for batch ranking across symbols
- **Sequential buffer manager** - `SeqBufferManager(T=seq_len, F=n_features)` for streaming sequence construction
- **Barrier gate fix** - Added `predict_single_target()` to `MultiHorizonPredictor` fixing broken barrier gate

#### Changed
- `LIVE_TRADING/models/loader.py` - Returns `input_mode` in metadata tuple
- `LIVE_TRADING/models/inference.py` - Routes to raw OHLCV or feature-based prediction based on input mode
- `LIVE_TRADING/prediction/predictor.py` - Cross-sectional ranking support

---

### Determinism & Thread Safety Fixes

#### Fixed
- Thread safety violations in shared state
- Non-deterministic dict iteration in artifact code paths
- Data corruption bugs in concurrent access patterns

#### Files Modified
- See commit `9e06343` for full details

---

## 2026-01-21

### Rust TUI Dashboard

Added a comprehensive terminal-based dashboard for monitoring training and trading operations. The dashboard is built with Rust and ratatui for high performance and low resource usage.

#### Added
- `DASHBOARD/dashboard/` - Rust TUI application with ratatui framework
  - **Launcher View** - Main dashboard with system status, live training/trading metrics
  - **Training Monitor** - Live progress tracking, run discovery, model performance grid
  - **Config Editor** - YAML editing with syntax highlighting and validation
  - **Log Viewer** - Real-time log tailing with filtering
  - **Service Manager** - systemd user service control (start/stop/restart)
  - **Settings View** - Theme selection and preferences

- `DASHBOARD/bridge/` - Python IPC bridge (FastAPI)
  - REST API for trading engine communication
  - WebSocket streaming for real-time events
  - Endpoints: `/api/state`, `/api/metrics`, `/api/positions`, `/api/control`

- `TRAINING/orchestration/utils/training_events.py` - Training event emitter
  - JSONL event file for dashboard polling (`/tmp/foxml_training_events.jsonl`)
  - PID file for process detection (`/tmp/foxml_training.pid`)
  - Event types: `progress`, `stage_change`, `target_start`, `target_complete`, `run_complete`, `error`

- `bin/foxml` - Dashboard launcher script

#### Features
- **Theme System** - Auto-detects colors from terminal/WM (kitty, tmux, hyprland, waybar)
- **File-based Events** - No WebSocket needed for training monitoring
- **UTC to Local Time** - Timestamps converted for display
- **PID Detection** - Dashboard detects running training processes

#### Documentation
- `.claude/skills/dashboard-overview.md` - Architecture and features
- `.claude/skills/dashboard-development.md` - Development guide
- `.claude/skills/dashboard-ipc-bridge.md` - IPC bridge API reference
- `.claude/skills/dashboard-event-integration.md` - Event integration patterns
- `DASHBOARD/README.md`, `DASHBOARD/FEATURES.md` - User documentation

---

### Training Event Integration

Improved training progress tracking for the dashboard with intermediate progress events and proper stage handling.

#### Fixed
- **Stage change handling** - `stage_change` events now use `new_stage` field correctly via `effective_stage()` method
- **UTC timestamps** - Converted to local time for display (was showing UTC causing 6-hour offset)
- **Progress during ranking** - Added intermediate `emit_progress()` calls in `target_ranker.py` for each target evaluated

#### Changed
- `TRAINING/ranking/target_ranker.py` - Now emits progress events during target evaluation loop
- `DASHBOARD/dashboard/src/api/events.rs` - Added `new_stage`, `previous_stage` fields and `effective_stage()` method
- `DASHBOARD/dashboard/src/launcher/live_dashboard.rs` - UTC to local time conversion using chrono

---

### Large-Scale Dataset Optimizations (70M+ rows)

Optimized the pipeline for very large datasets through column projection and streaming concatenation.

#### Added
- **Column Projection** - Load only required columns at all pipeline stages
  - `TRAINING/data_processing/data_loader.py` - `load_all_data()` accepts `columns` parameter
  - 60-75% memory reduction for large datasets
  - Prevents OOM on datasets with 200+ columns

- **Streaming Concat** - Incremental DataFrame construction
  - Replaced `pd.concat(dfs)` with streaming append pattern
  - Constant memory usage regardless of number of files
  - GC between file loads

#### Changed
- `TRAINING/training_strategies/execution/data_preparation.py` - Column projection throughout
- `TRAINING/ranking/target_ranker.py` - Only loads features + target columns
- `TRAINING/ranking/feature_selector.py` - Column filtering at load time

---

### Cross-Sectional Ranking Objective (Planning Complete)

Designed comprehensive plan for implementing true cross-sectional ranking objectives (ListMLE, ListNet, LambdaRank) to replace pointwise binary classification in target ranking.

#### Added
- `.claude/plans/cross-sectional-ranking-objective.md` - Master plan
- `.claude/plans/cs-ranking-phase1-targets.md` - Target/label generation
- `.claude/plans/cs-ranking-phase2-batching.md` - Cross-sectional batch construction
- `.claude/plans/cs-ranking-phase3-losses.md` - Ranking loss implementations
- `.claude/plans/cs-ranking-phase4-metrics.md` - NDCG/MRR metrics
- `.claude/plans/cs-ranking-phase5-integration.md` - Pipeline integration

---

### Raw OHLCV Sequence Mode (Planning Complete)

Designed input mode for feeding raw OHLCV sequences directly to neural models (LSTM, Transformer, CNN1D) without feature engineering.

#### Added
- `.claude/plans/raw-ohlcv-sequence-mode.md` - Master plan
- `.claude/plans/raw-ohlcv-sequence-data.md` - Data preparation details
- `TRAINING/common/input_mode.py` - Input mode enum (FEATURES, RAW_SEQUENCE, HYBRID)

---

## 2026-01-19

### Modular Decomposition Bug Fixes

Fixed 4 runtime errors discovered during decomposition verification. Root cause: imports/exports added without runtime verification.

#### Fixed
- `model_evaluation.py` - Removed import of non-existent `load_evaluation_config` function
- `model_evaluation.py` - Removed import of non-existent `build_predictability_report`, added correct exports (`save_feature_importances`, `log_suspicious_features`)
- `target_ranker.py` - Fixed `UnboundLocalError` for `valid_results` by initializing before conditional block
- `leakage_detection.py` - `_save_feature_importances()` already had correct parameters (`run_identity`, `model_metrics`, `attempt_id`)

#### Added
- `.claude/skills/decomposition-verification.md` - Comprehensive checklist for future decomposition work
- `repro_tracker_modules/` - Extracted types from `reproducibility_tracker.py` (DriftCategory, ComparisonStatus, DriftMetrics, CohortMetadata, ComparisonResult)
- `feature_selector_modules/` - Delegating wrappers for feature selection (config.py, core.py)

#### Lesson Learned
Static analysis is NOT sufficient for verifying decomposition. Always run actual Python imports to catch circular imports, missing dependencies, and parameter mismatches.

---

### Code Review and Additional Bug Fixes

Comprehensive code review of all refactored modules identified and fixed several additional issues.

#### Fixed
- `diff_telemetry.py` - **snapshot.json path mismatch**: `save_snapshot()` now returns the resolved cohort_dir, and `finalize_run()` uses this for `save_diff()` to ensure both files are written to the same directory
- `leakage_detection.py` - **Leak detection report path**: `_log_suspicious_features()` now accepts `output_dir` parameter to write reports to per-run directory instead of global `TRAINING/results/`
- `model_evaluation/reporting.py` - Same fix for `log_suspicious_features()`
- `leakage_detection/reporting.py` - Same fix for `log_suspicious_features()`
- `test_comparison_group.py` - Updated 3 tests to include `feature_signature` (now required for TARGET_RANKING)

#### Root Cause Analysis
- **snapshot.json missing warning**: Path repair logic in `save_diff()` was executed AFTER `save_snapshot()`, causing files to be written to different directories when path repair was needed. Fix: `save_snapshot()` now returns the resolved path for use by `save_diff()`.
- **Contract test failures**: `feature_signature` was added as a required field for TARGET_RANKING but tests weren't updated.

#### Verification
- All 22 decomposed modules import correctly
- All 16 contract tests pass (2 skipped - require special env vars)
- No circular imports detected
- All function signatures match call sites

---

### Model Evaluation Modular Decomposition (Phase 1 - Partial)

Began modular decomposition of `model_evaluation.py` (9,656 lines) into focused submodules. This is part of the broader modular decomposition plan to improve maintainability.

#### Added
- `TRAINING/ranking/predictability/model_evaluation/safety.py` - Safety gate validation (~300 lines)
  - `enforce_final_safety_gate()` extracted from main file
  - Final gatekeeper that validates features before model training
  - Policy cap enforcement and dropped feature tracking

- `TRAINING/ranking/predictability/model_evaluation/autofix.py` - Automatic leakage fixing (~310 lines)
  - `evaluate_target_with_autofix()` with auto-rerun logic
  - Iterative leakage detection and patch application

- `TRAINING/ranking/predictability/model_evaluation/training.py` - Stub module
  - Re-exports `train_and_evaluate_models` from parent (full extraction deferred)

- `TRAINING/ranking/predictability/model_evaluation/ranking.py` - Stub module
  - Re-exports `evaluate_target_predictability` from parent (full extraction deferred)

#### Changed
- `model_evaluation/__init__.py` - Updated with proper exports from all submodules
- `model_evaluation.py` - Now imports `_enforce_final_safety_gate` from safety submodule

#### Line Count Reduction
- Before: 9,656 lines
- After: 9,359 lines (-297 lines extracted to safety.py)

#### Notes
- Full extraction of `train_and_evaluate_models` (~4,800 lines) and `evaluate_target_predictability` (~4,100 lines) deferred to future sessions due to size
- Stub modules establish the modular structure while maintaining backward compatibility
- See `.claude/plans/mod-phase1-model-evaluation.md` for detailed status

---

### Integration Contracts & Cross-Module Communication

Established formal integration contracts between TRAINING and LIVE_TRADING modules, fixing 4 critical issues.

#### Fixed
- `feature_list` vs `features` field name mismatch - TRAINING now writes `feature_list` (sorted) in all metadata paths
- `interval_minutes` not written in symbol-specific path - added to all 4 metadata blocks in training.py
- Features not guaranteed sorted - added `_get_sorted_feature_list()` helper for determinism
- `model_checksum` not always written - added SHA256 computation after model save

#### Added
- `INTEGRATION_CONTRACTS.md` - Cross-module artifact schemas (model_meta.json, manifest.json, routing_decision.json)
- `.claude/skills/integration-contracts.md` - Integration guidelines for Claude Code
- `.claude/skills/changelog-maintenance.md` - Guidelines for maintaining changelog after tasks
- `.claude/prompts/integration-audit.md` - Fresh context audit prompt

#### Files Modified
- `TRAINING/training_strategies/execution/training.py` - Added `_get_sorted_feature_list()`, `_compute_model_checksum()`, feature_list and interval_minutes to all paths
- `TRAINING/models/specialized/core.py` - Added sorted `feature_list` field
- `LIVE_TRADING/models/loader.py` - Added backward-compatible fallback for feature_list
- `LIVE_TRADING/models/inference.py` - Added fallback for feature_list
- `LIVE_TRADING/README.md` - Added cross-module integration section linking to INTEGRATION_CONTRACTS.md

---

### Interval-Agnostic Pipeline - Phases 8-10

Implementation of multi-horizon training, cross-horizon ensemble, and multi-interval experiments as part of the interval-agnostic pipeline plan.

#### Phase 8: Multi-Horizon Training
- **HorizonBundle Dataclass**: Created `TRAINING/training_strategies/types/horizon_types.py`
  - `HorizonBundle` dataclass for grouping related horizons with validation
  - `HorizonSpec` for individual horizon metadata (minutes, bars, type)
  - Factory functions: `make_horizon_spec()`, `make_horizon_bundle()`
  - Validation: mutual exclusivity, interval alignment, parent-child relationships

- **MultiHorizonTrainer**: Created `TRAINING/training_strategies/execution/multi_horizon_trainer.py`
  - Train multiple horizons in single pass with shared feature computation
  - Automatic feature alignment across horizons
  - Per-horizon metrics and fingerprinting
  - SST-compliant artifact organization

- **Tests**: 49 tests in `tests/test_horizon_bundle.py` and `tests/test_multi_horizon_trainer.py`

#### Phase 9: Cross-Horizon Ensemble
- **CrossHorizonEnsemble**: Created `TRAINING/model_fun/cross_horizon_ensemble.py`
  - Ridge risk-parity weighting: `w ∝ (Σ + λI)^{-1} μ`
  - Optional horizon decay (exponential, configurable half-life)
  - IC-based weighting with lookback window
  - Save/load for production deployment
  - Convenience functions: `calculate_horizon_weights()`, `blend_horizon_predictions()`

- **Config**: Added `cross_horizon` section to `CONFIG/models/ensemble.yaml`
  ```yaml
  cross_horizon:
    enabled: false
    ridge_lambda: 0.15
    decay_function: exponential
    decay_half_life_minutes: 30
    use_ic_weighting: true
  ```

- **Tests**: 18 tests in `tests/test_cross_horizon_ensemble.py`

#### Phase 10: Multi-Interval Experiments
- **MultiIntervalExperiment**: Created `TRAINING/orchestration/multi_interval_experiment.py`
  - Train models at multiple data intervals (1m, 5m, 15m, 60m, etc.)
  - Cross-interval validation (train at 5m, validate at 1m/15m/60m)
  - Degradation tracking with configurable thresholds
  - Comparison reports across intervals

- **IntervalDataLoader**: Discovers and loads interval-partitioned data
  - Supports `data_root/interval={N}m/` directory structure
  - Symbol filtering and sample limits

- **CrossIntervalValidator**: Validates model generalization
  - Computes degradation metrics vs baseline
  - Generates warnings for severe degradation (>50%)

- **FeatureTransfer**: Warm-start support between intervals
  - Compatible interval detection (divisibility check)
  - Feature name mapping between intervals
  - Scale factor computation for warm-start

- **IntervalComparator**: Compares results across intervals
  - Per-metric rankings
  - Best interval detection
  - Human-readable comparison reports

- **Example Config**: Created `CONFIG/experiments/multi_interval_example.yaml`
  ```yaml
  multi_interval:
    intervals: [5, 15, 60]
    primary_interval: 5
    cross_validation:
      enabled: true
      train_intervals: [5]
      validate_intervals: [1, 5, 15, 60]
  ```

- **Tests**: 29 tests in `tests/test_multi_interval_experiment.py`

#### Files Created
- `TRAINING/training_strategies/types/horizon_types.py`
- `TRAINING/training_strategies/execution/multi_horizon_trainer.py`
- `TRAINING/model_fun/cross_horizon_ensemble.py`
- `TRAINING/orchestration/multi_interval_experiment.py`
- `CONFIG/experiments/multi_interval_example.yaml`
- `tests/test_horizon_bundle.py`
- `tests/test_multi_horizon_trainer.py`
- `tests/test_cross_horizon_ensemble.py`
- `tests/test_multi_interval_experiment.py`

#### Files Modified
- `CONFIG/models/ensemble.yaml` - Added cross_horizon section
- `.claude/plans/interval_agnostic_pipeline.md` - Updated phase status

#### Test Summary
- Phase 8: 49 tests (HorizonBundle + MultiHorizonTrainer)
- Phase 9: 18 tests (CrossHorizonEnsemble)
- Phase 10: 29 tests (MultiIntervalExperiment)
- **Total**: 96 new tests for Phases 8-10

## 2026-01-18

### Training Pipeline Code Review - Phase 2

#### Unit Tests Added
- **Target Routing Tests**: Created comprehensive tests for `_compute_target_routing_decisions()` and `_compute_single_target_routing_decision()` in `tests/test_target_routing.py`
  - Tests for CS route with good coverage
  - Tests for SYMBOL_SPECIFIC route with weak CS
  - Tests for BOTH route with concentrated performance
  - Tests for BLOCKED route with suspicious scores
  - Tests for dev mode threshold relaxation
  - Tests for fingerprint validation
- **Cross-Sectional Feature Ranker Tests**: Created tests for public functions in `tests/test_cross_sectional_feature_ranker.py`
  - Tests for `normalize_cross_sectional_per_date()` with zscore/rank methods
  - Tests for `train_panel_model()` for LightGBM/XGBoost
  - Tests for `compute_cross_sectional_importance()`
  - Tests for `tag_features_by_importance()`
  - Tests for `compute_cross_sectional_stability()`
- **Model Factory Tests**: Created tests for model factory in `tests/test_model_factory.py`
  - Singleton pattern tests
  - `create_model()` for regression/classification
  - `create_models_for_targets()` batch creation
  - `_auto_select_model_type()` logic
  - `validate_model_config()`

#### Performance Fixes
- **Fixed O(n²) Correlation Computation**: Optimized `select_features_by_correlation()` in `TRAINING/ranking/utils/feature_selection.py`
  - Now uses vectorized correlation matrix computation (`np.corrcoef(X.T)`)
  - O(F²) time with BLAS acceleration vs previous O(F² × N) per-pair computation
  - Vectorized target correlation using dot product
  - For 5000 features: ~25M operations → single matrix operation
- **Replaced .iterrows() with .itertuples()**: Fixed performance anti-pattern in `TRAINING/ranking/feature_selection_reporting.py`
  - ~100x faster iteration for DataFrame processing
  - Affected lines 191-210 and 245-260

#### Code Quality Improvements
- **Centralized Config Helpers**: Created `TRAINING/common/utils/config_helpers.py` with DRY helpers
  - `load_threshold()` - Generic threshold loading with fallback
  - `load_routing_thresholds()` - Routing thresholds with dev mode support
  - `apply_dev_mode_relaxation()` - Dev mode threshold relaxation
  - `load_feature_selection_thresholds()` - Feature selection thresholds
  - `load_data_limits()` - Data limit thresholds
- **Centralized Routing Config**: Created `CONFIG/routing/thresholds.yaml`
  - CS/symbol skill01 thresholds
  - Suspicious score thresholds for leakage detection
  - Dev mode relaxation settings
  - Documented routing decision rules

#### API Design Fixes
- **Parameter Validation in Feature Selector**: Added validation in `TRAINING/ranking/feature_selector.py`
  - Warns when multiple config sources provided (feature_selection_config, model_families_config, multi_model_config)
  - Raises error when SYMBOL_SPECIFIC view is missing required symbol parameter
- **Debug Logging in Parallel Exec**: Added debug logging for config loading failures in `TRAINING/common/parallel_exec.py`
  - Previously silently swallowed exceptions

#### Logging Improvements
- **Removed Emojis from Logs**: Replaced emojis with structured prefixes in `TRAINING/ranking/target_routing.py`
  - `🔧` → `[DEV]`
  - `⚠️` → `[WARN]`
  - `✅` → `[OK]`
  - `🚨` → `[ERROR]`

#### Files Modified
- `TRAINING/ranking/utils/feature_selection.py` - O(n²) correlation fix
- `TRAINING/ranking/feature_selection_reporting.py` - .iterrows() replacement
- `TRAINING/ranking/feature_selector.py` - Parameter validation
- `TRAINING/ranking/target_routing.py` - Logging fixes
- `TRAINING/common/parallel_exec.py` - Debug logging

#### Files Created
- `tests/test_target_routing.py` - Unit tests for routing
- `tests/test_cross_sectional_feature_ranker.py` - Unit tests for CS ranker
- `tests/test_model_factory.py` - Unit tests for model factory
- `TRAINING/common/utils/config_helpers.py` - Centralized config helpers
- `CONFIG/routing/thresholds.yaml` - Centralized routing thresholds

## 2026-01-17

### Run ID Lookup from Manifest Fix
- **Fixed Run ID Lookup**: Fixed `run_id` lookup in `intelligent_trainer.py` to read from `manifest.json` (authoritative SST source) instead of deriving from directory name with format mutations
  - Added `read_run_id_from_manifest()` helper in `manifest.py` for centralized manifest reading
  - Normalized `output_dir` from str/PathLike → Path with try/except guard (prevents early throw)
  - Correct fallback order: manifest → output_dir.name → _run_name → RunIdentity → None/raise
  - Removed underscore-to-dash conversion (no format mutations)
  - Removed `derive_unstable_run_id()` fallback (no ID fabrication)
  - Added strict mode policy: raise error in strict mode, disable filtering + warning in best-effort
  - Added source tracking: logs which source provided `run_id` for debugging
- **Impact**:
  - Fixes snapshot filtering failures (snapshots now found when manifest contains correct `run_id`)
  - Eliminates format corruption (no more underscore→dash mutations)
  - Prevents ID fabrication (never generates new `run_id` when sources unavailable)
  - SST compliance (manifest is authoritative source)
  - Strict mode safety (fail-closed behavior prevents silent aggregation)
- **Files**: `TRAINING/orchestration/utils/manifest.py`, `TRAINING/orchestration/intelligent_trainer.py`
- **Detailed Changelog**: [2026-01-17-run-id-lookup-from-manifest-fix.md](DOCS/02_reference/changelog/2026-01-17-run-id-lookup-from-manifest-fix.md)

### Run ID Normalization and Run Organization Improvements
- **Normalized Run ID Generation**: Implemented deterministic, hash-based `run_id` generation across all pipeline stages
  - New format: `ridv1_{sha256(strict_key + ":" + replicate_key)[:20]}` for stable runs
  - Fallback format: `rid_unstable_{run_instance_id}` when identity unavailable
  - All stages (TARGET_RANKING, FEATURE_SELECTION, TRAINING) now use consistent run_id derivation
  - Eliminates format mismatches that caused run matching failures
- **Improved Run Organization**: Changed directory naming from `cg-{hash}_n-{sample_size}_fam-{model_family}` to `cg-{cg_hash}_u-{universe_sig[:8]}_c-{config_sig[:8]}`
  - Groups runs by universe and config signatures instead of just sample count
  - Allows runs with different sample sizes but same config to be grouped together
  - `n_effective` moved to run leaf metadata (not in directory name)
- **Comparability Flags**: Added `is_comparable` and `run_id_kind` flags to `manifest.json`
  - Authoritative comparability checking via manifest flags
  - Falls back to legacy normalization for old runs
  - Explicitly refuses unstable-vs-stable comparisons
- **Directory Parser**: Added `parse_run_instance_dirname()` to handle multiple directory name formats
  - Handles: old underscore, old dash, new suffix forms
  - Replaces hardcoded `startswith()` checks
- **Config Signature**: Added canonical `compute_config_signature()` helper
  - Centralized definition of config signature
  - Includes all behavior-changing knobs except universe membership
  - **Fixed**: Added `feature_signature` to config signature (CRITICAL: different features = different outcomes)
  - Bumped `sig_version` from 1 to 2 when `feature_signature` was added
- **Impact**:
  - Deterministic run IDs (same inputs → same ID)
  - Better run organization (comparable runs grouped together)
  - Improved run matching in diff telemetry
  - Backward compatible (old runs still accessible)
- **Files**: `TRAINING/orchestration/utils/manifest.py`, `TRAINING/orchestration/utils/diff_telemetry/types.py`, `TRAINING/orchestration/utils/diff_telemetry.py`, `TRAINING/orchestration/intelligent_trainer.py`, `TRAINING/stability/feature_importance/schema.py`, `TRAINING/training_strategies/reproducibility/schema.py`, `TRAINING/ranking/target_ranker.py`, `TRAINING/orchestration/utils/reproducibility_tracker.py`, `TRAINING/orchestration/training_plan_generator.py`
- **Detailed Changelog**: [2026-01-17-run-id-normalization-and-organization.md](DOCS/02_reference/changelog/2026-01-17-run-id-normalization-and-organization.md)

### Feature Selection Fixes
- **Fixed ExperimentConfig AttributeError**: Fixed `AttributeError: 'ExperimentConfig' object has no attribute 'get'` in `load_lookback_budget_spec()` function
  - Added type check to handle both dict and `ExperimentConfig` objects before calling `.get()` method
  - `ExperimentConfig` objects (dataclasses) don't have `.get()` method, so experiment config override is skipped for non-dict objects
  - Follows same pattern used elsewhere in codebase (e.g., `target_ranker.py`, `model_evaluation.py`)
  - Training pipeline now works correctly when `ExperimentConfig` objects are passed to feature selection functions
- **Impact**:
  - Feature selection no longer crashes with AttributeError when `ExperimentConfig` objects are used
  - Function gracefully handles both dict and `ExperimentConfig` object inputs
- **Files**: `TRAINING/ranking/utils/leakage_budget.py`

## 2026-01-15

### Critical Training Pipeline Fixes
- **Fixed Path UnboundLocalError**: Removed redundant local `Path` import that shadowed module-level import, causing `UnboundLocalError: local variable 'Path' referenced before assignment` at line 1386 in training.py
  - Training pipeline now starts successfully without crashes
  - All `Path()` usages correctly reference module-level import
- **Fixed Routing Decisions Fingerprint Mismatch**: Added `symbols` and `symbol_count` to expected fingerprint computation to match stored fingerprint structure
  - Fingerprint validation now works correctly when targets match
  - Eliminates false "stale routing decisions" warnings
- **Fixed CatBoost Verbose Period Error**: Converted boolean `verbose=False` from config to integer `verbose=0` for CatBoost compatibility
  - CatBoost now trains successfully without `verbose_period should be nonnegative` error
  - Added validation to remove negative `verbose_period` values
  - Applied fix in config cleaner, feature selection, and model evaluation paths
- **Improved Missing Families Error Logging**: Enhanced error logging for CatBoost and Lasso failures with full traceback
  - Failed families are now properly tracked in results (not silently dropped)
  - Error messages clearly indicate why families failed
  - Debug logs point to specific error types for easier debugging
- **Improved Training Families Logging**: Changed misleading INFO log to DEBUG with clarification that parameter may be overridden by SST config
  - Log messages now clearly distinguish parameter vs final resolved families
- **Impact**:
  - Training pipeline no longer crashes with Path errors
  - Routing decisions fingerprint validation works correctly
  - CatBoost trains successfully with proper parameter handling
  - Better visibility into feature selection failures
  - Clearer logging for training families resolution
- **Files**: `TRAINING/training_strategies/execution/training.py`, `TRAINING/ranking/target_routing.py`, `TRAINING/common/utils/config_cleaner.py`, `TRAINING/ranking/multi_model_feature_selection.py`, `TRAINING/ranking/predictability/model_evaluation.py`, `TRAINING/ranking/feature_selector.py`, `TRAINING/orchestration/intelligent_trainer.py`
- **Detailed Changelog**: [2026-01-15-path-fingerprint-catboost-fixes.md](DOCS/02_reference/changelog/2026-01-15-path-fingerprint-catboost-fixes.md)

## 2026-01-14

### Feature Selection Pipeline Fixes
- **Fixed route_enum NameError**: Removed undefined `route_enum` variable references in feature selection logic
  - Changed from `route_enum == View.CROSS_SECTIONAL or route == View.CROSS_SECTIONAL.value` to `route == View.CROSS_SECTIONAL.value`
  - Changed from `route_enum == View.SYMBOL_SPECIFIC or route == View.SYMBOL_SPECIFIC.value` to `route == View.SYMBOL_SPECIFIC.value`
  - Route values from routing decisions are always strings, not View enums
  - Consistent with existing codebase patterns (lines 1778, 2668, 2695)
  - Fixes `NameError: name 'route_enum' is not defined` when processing routing decisions
- **Added Guard for Empty Target Lists**: Skip feature selection entirely when all targets are filtered out
  - Prevents wasteful setup work (loading routing decisions, logging) when no targets remain
  - Logs clear warning message explaining why feature selection was skipped
  - Consistent with training phase guards (lines 3143, 3154)
  - No breaking changes: downstream code already handles empty `target_features` correctly
- **Impact**:
  - Feature selection no longer crashes with NameError when routing decisions are missing
  - Pipeline correctly handles cases where target ranking or training plan filtering removes all targets
  - Improved efficiency by skipping unnecessary work when no targets remain
- **Files**: `TRAINING/orchestration/intelligent_trainer.py`

## 2026-01-13

### Leakage Auto-Fixer Fixes and Serialization Compliance
- **Fixed Missing output_dir Attribute**: Fixed `AttributeError: 'LeakageAutoFixer' object has no attribute 'output_dir'` in `_write_per_target_patch()`
  - Added `self.output_dir` initialization in `LeakageAutoFixer.__init__()`
  - Prevents runtime failure when writing per-target registry patches
- **Atomic Write for excluded_features.yaml**: Replaced direct `yaml.dump()` with `write_atomic_yaml()` in `_apply_excluded_features_updates()`
  - Ensures crash safety and power-loss resilience for config file updates
  - Uses canonical YAML serialization for determinism compliance
- **Canonical YAML Serialization for Patch Files**: Updated `_write_per_target_patch()` to use `canonical_yaml()` helper
  - Replaced direct `yaml.dump()` with canonical serialization
  - Maintains atomic write pattern (temp file + fsync + rename) while ensuring deterministic output
  - All registry patch files now use canonical serialization for consistency
- **Impact**:
  - Leakage auto-fixer no longer crashes when writing patches
  - All config and patch file writes are now crash-safe and deterministic
  - Consistent serialization format across all leakage auto-fixer outputs
- **Files**: `TRAINING/common/leakage_auto_fixer.py`

### Documentation Updates
- **README Maintenance Status**: Removed "unmaintained" warnings, added note about grey area maintenance status
- **Data Interval Configuration**: Added notes about manual config editing requirements and partial automatic interval handling
- **Custom Data Sets**: Added notes about manual leaky feature detection editing (including regex patterns) for custom naming schemes
- **Aspirational Features**: Added note about eventual goal of automatic feature/target discovery from data path
- **Privacy & Security**: Added section clarifying no networking/phone-home functions and security considerations
- **Internal Planning Documents**: Added section explaining inclusion of planning docs and `.cursor/plans/` files for transparency
- **AI Coding Agents**: Added note about performance degradation with too many rule constraints
- **Files**: `README.md`

## 2026-01-11

### Comprehensive Determinism, SST, and Error Handling Cleanup
- **Atomic Writes for All Artifacts**: Fixed all JSON/YAML artifact writes to use atomic write helpers with crash durability
  - All snapshot artifacts (manifests, routing plans, training plans, stats, audit reports) now use `write_atomic_json()`
  - Ensures power-loss safety: temp file → fsync → atomic replace → directory fsync
  - Fixed 25+ non-atomic writes across orchestration code
- **Centralized Error Handling**: Implemented fail-closed policy for artifact-shaping code
  - Created `EXCEPTIONS_MATRIX.md` categorizing all exception handling patterns
  - Fixed 6 critical artifact-shaping exceptions to use `handle_error_with_policy()`
  - Fixed 3 bare except clauses with proper logging
  - All artifact-shaping errors now fail closed in deterministic mode
- **Deterministic Dict Iteration**: Fixed dict iteration in artifact-shaping code paths
  - Fixed 13+ dict iterations in `diff_telemetry.py` and `reproducibility_tracker.py`
  - All artifact-shaping dict iterations now use `sorted_items()` helper
  - Ensures consistent serialization order for hashing and comparison
- **Filesystem Enumeration Verification**: Verified all filesystem operations use sorted helpers
  - All `.glob()`, `.rglob()`, `.iterdir()` replaced with `glob_sorted()`, `rglob_sorted()`, `iterdir_sorted()`
  - Ensures deterministic file discovery order across runs
- **Circular Import Prevention**: Verified error handling helpers don't create circular imports
  - `handle_error_with_policy()` lives in Level 1 (Core Utils) only
  - Import smoke test passes for all critical modules
- **Impact**:
  - All artifact writes are now crash-safe and power-loss resilient
  - Artifact-shaping errors properly fail closed in deterministic mode
  - Dict iteration order is deterministic in all artifact-shaping paths
  - Filesystem enumeration is deterministic across machines
  - Codebase ready for E2E testing with full determinism guarantees
- **Files**: `intelligent_trainer.py`, `reproducibility_tracker.py`, `diff_telemetry.py`, `training_plan_generator.py`, `training_router.py`, `target_routing.py`, `routing_candidates.py`, `run_context.py`, `checkpoint.py`, `file_utils.py`, `exceptions.py`
- **Documentation**: `INTERNAL_DOCS/references/EXCEPTIONS_MATRIX.md`, `INTERNAL_DOCS/references/WRITE_LOCATIONS_MATRIX.md`

## 2026-01-XX

### Determinism Hardening: Phase 1 (Financial Outputs)
- **Created Determinism Helper Modules**: Added canonical helpers for deterministic ordering and serialization
  - `TRAINING/common/utils/determinism_ordering.py` - Filesystem and container iteration helpers
  - `TRAINING/common/utils/determinism_serialization.py` - Canonical JSON serialization
  - `TRAINING/common/determinism_policy.py` - Tier A file list SST and waiver validation
- **Fixed Non-Deterministic Filesystem Operations**: 
  - `target_routing.py`: Fixed 2 unsorted filesystem operations (iterdir, rglob)
  - `decision_engine.py`: Fixed 1 unsorted glob operation
- **Fixed Non-Deterministic Dictionary Iterations**:
  - `target_ranker.py`: Fixed 8 dict iterations + parallel results sorting
  - `training_plan_generator.py`: Fixed 4 dict iterations
  - `feature_selector.py`: Fixed 2 dict iterations + 2 next(iter()) calls + 3 list comprehensions
- **Fixed Score-Based Sorting**: Added tie-breakers to prevent non-determinism on equal scores
  - `target_ranker.py`: Added target name tie-breakers to screen/strict score sorting
- **Fixed JSON Serialization**: Default `sort_keys=True` in `safe_json_dump()` for deterministic JSON output
- **Created Enforcement Tools**:
  - `bin/check_determinism_patterns.sh` - Pattern scanner for regression detection
  - `bin/verify_determinism_init.py` - Entry point verification script
- **Impact**:
  - Same inputs now produce identical outputs for target rankings, feature selection, routing decisions, and training plans
  - Eliminates "why did run A evaluate targets in different order than run B?" diff noise
  - Ensures deterministic target evaluation, feature selection, and training plan generation
  - All Tier A files (financial outputs) now use canonical helpers for deterministic ordering
- **Files**: `target_routing.py`, `decision_engine.py`, `target_ranker.py`, `training_plan_generator.py`, `feature_selector.py`, `file_utils.py`
- **Documentation**: `INTERNAL_DOCS/fixes-determinism-hardening-phase1-2026-01.md`

### Determinism Hardening: Phase 1 (Financial Outputs)
- **Created Determinism Helper Modules**: Added canonical helpers for deterministic ordering and serialization
  - `TRAINING/common/utils/determinism_ordering.py` - 8 helpers for filesystem and container iteration
  - `TRAINING/common/utils/determinism_serialization.py` - Canonical JSON serialization helpers
  - `TRAINING/common/determinism_policy.py` - Tier A file list SST and waiver validation
- **Fixed Non-Deterministic Filesystem Operations**: 
  - `target_routing.py`: Fixed 2 unsorted filesystem operations (iterdir, rglob) affecting routing decisions
  - `decision_engine.py`: Fixed 1 unsorted glob operation affecting decision loading
- **Fixed Non-Deterministic Dictionary Iterations**:
  - `target_ranker.py`: Fixed 8 dict iterations + parallel results sorting + score-based sorting with tie-breakers
  - `training_plan_generator.py`: Fixed 4 dict iterations affecting training plan generation
  - `feature_selector.py`: Fixed 2 dict iterations + 2 next(iter()) calls + 3 list comprehensions affecting feature selection
- **Fixed Score-Based Sorting**: Added tie-breakers to prevent non-determinism on equal scores
  - `target_ranker.py`: Added target name tie-breakers to screen/strict score sorting
- **Fixed JSON Serialization**: Default `sort_keys=True` in `safe_json_dump()` for deterministic JSON output
- **Created Enforcement Tools**:
  - `bin/check_determinism_patterns.sh` - Pattern scanner for regression detection
  - `bin/verify_determinism_init.py` - Entry point verification script
- **Impact**:
  - Same inputs now produce identical outputs for target rankings, feature selection, routing decisions, and training plans
  - Eliminates "why did run A evaluate targets in different order than run B?" diff noise
  - Ensures deterministic target evaluation, feature selection, and training plan generation
  - All Tier A files (financial outputs) now use canonical helpers for deterministic ordering
  - Complete determinism guarantees for critical iteration points
- **Files**: `target_routing.py`, `decision_engine.py`, `target_ranker.py`, `training_plan_generator.py`, `feature_selector.py`, `file_utils.py`
- **Documentation**: `INTERNAL_DOCS/fixes-determinism-hardening-phase1-2026-01.md`

### Dev Mode and Coverage Review Fixes (21 Issues Resolved)
- **Critical Fixes (3)**: Fixed uninitialized variable crash, SST violation from mismatch telemetry bypass, and missing `get_dev_mode_source()` function
- **High Priority Fixes (5)**: Fixed IndexError on empty symbols, division by zero in telemetry, interval zero guard, non-deterministic fallback selection, and defensive formatting
- **Medium Priority Fixes (10)**: Fixed telemetry variable naming, duplicate coverage computation, type normalization, error handling, direct get_cfg calls, non-deterministic logging, gate bypass visibility
- **Low Priority Fixes (3)**: Documented quality score fallback, fixed dev mode delta logic, added explicit None mode handling
- **Impact**:
  - All eligibility gates now use canonical `CoverageBreakdown` when `safe_columns` exists
  - No runtime crashes for empty lists/sets, zero intervals, or None formatting
  - Deterministic target selection and stable log ordering
  - Complete SST compliance for coverage computation and dev_mode resolution
- **Files**: `composite_score.py`, `model_evaluation.py`, `intelligent_trainer.py`, `registry_coverage.py`, `dev_mode.py`, `training_plan_generator.py`, `leakage_filtering.py`
- **Documentation**: See `INTERNAL_DOCS/fixes-dev-mode-coverage-review-2026-01.md` for complete details

## 2026-01-10

### Complete Metric Delta Auditability and Diff File Creation
- **Include All Metric Deltas**: Changed `_compute_metric_deltas()` to include ALL deltas (not just significant ones), providing complete auditability even when all metrics are identical
- **Always Create Diff Files When Comparable**: Updated `save_diff()` to create `metric_deltas.json` whenever `diff.comparable == true`, even if all metrics are identical or no metrics were compared
- **Complete Visibility**: Users can now see "we compared X metrics, all identical" instead of "no file created"
- **Impact**:
  - Complete auditability: all metrics that were compared are visible, not just significant ones
  - Better debugging: can distinguish between "no previous snapshot" vs "previous snapshot exists but no metrics to compare"
  - Consistent behavior: files are created whenever there's a comparison, regardless of results
- **Files**: `diff_telemetry.py`

### Dev Mode for Target Ranking Routing
- **Relaxed Thresholds for Testing**: Added `dev_mode` flag to target ranking routing that relaxes thresholds for testing with small datasets (10 symbols, 40k CS samples)
- **Threshold Adjustments**: When `dev_mode: true`:
  - T_cs: 0.65 → 0.40 (allows more cross-sectional routes)
  - T_sym: 0.60 → 0.35 (allows more symbol-specific routes)
  - T_frac: 0.5 → 0.2 (only requires 2/10 symbols instead of 5/10)
  - T_suspicious_cs: 0.90 → 0.95 (less aggressive blocking)
  - T_suspicious_sym: 0.95 → 0.98 (less aggressive blocking)
- **Determinism Test Config**: Updated `determinism_test.yaml` to use 4k per symbol and 40k CS samples for E2E testing
- **Impact**:
  - Targets can now pass through routing to feature selection for E2E testing with small datasets
  - Production behavior unchanged (dev_mode defaults to false)
  - Complete visibility into active thresholds in runtime settings
- **Files**: `target_routing.py`, `configs.yaml`, `determinism_test.yaml`

### Runtime Settings in Config Trace
- **Enhanced Config Trace**: Added "Runtime Settings" section to config trace showing:
  - Determinism: base seed, threads, deterministic algorithms flag
  - Parallelism: parallel_targets, parallel_symbols, max_workers, threading.parallel.enabled
  - Routing: dev_mode status, active thresholds (with base values when dev_mode is on)
- **Config Verification**: Fixed feature selection config name to read from correct file (`multi_model.yaml` instead of `config.yaml`)
- **Impact**:
  - Better verification of test environment settings
  - Clear visibility into which thresholds are active
  - Helps ensure determinism and parallelism settings match expectations
- **Files**: `intelligent_trainer.py`

### Enhanced Auto-Fix Skip Reason Logging with T-stat and Metrics
- **Complete Gate Coverage**: Added shared gate evaluator `_auto_fix_decision()` that computes both `should_fix` and `failed_gates` from the same logic, ensuring `should_auto_fix()` and `auto_fix_reason()` never drift
- **Correct Metric Naming**: Extract CV metric name from scoring source (same logic as `train_and_evaluate_models()`), not from task type inference
- **T-stat and CV Metrics in Skip Reasons**: Auto-fix skip reasons now include:
  - CV metric value (e.g., `cv_r2=0.0112` for regression, `cv_roc_auc=0.6701` for classification)
  - Universal T-stat (e.g., `tstat=0.563`) for cross-task comparability
- **Finite Value Filtering**: Use `np.isfinite()` not just `np.isnan()` to filter out `inf/-inf` values
- **Improved Label Semantics**: Changed `leak_scan_passed` → `no_leak_signals` (clearer blocker semantics)
- **Routing Traceability**: Added `auto_fix_reason` and `tstat_cs` fields to routing decision dicts for better traceability
- **Schema Version Bump**: Bumped `metrics_schema_version` from `1.1` to `1.2` (new `auto_fix_reason` field in `TargetPredictabilityScore`)
- **Impact**:
  - Skip reasons now show actual metric values, making regression vs classification comparable
  - Complete gate coverage ensures all skip reasons are logged correctly
  - T-stat provides universal metric for cross-task comparison
  - All changes maintain determinism (no sorting operations, deterministic max(), fixed ordering)
- **Files**: `leakage_assessment.py`, `model_evaluation.py`, `scoring.py`, `target_routing.py`

## 2026-01-10

### Added Cache Invalidation, Config Hashing, and Decision Tracking for Auto-Fixer Reruns
- **Cache Invalidation**: Added explicit cache invalidation after auto-fixer writes to `excluded_features.yaml` via `reload_feature_configs()` call, ensuring reruns immediately pick up new exclusions without waiting for mtime checks
- **Config Hashing**: Added SHA256 hash computation before/after writing to `excluded_features.yaml` using existing SST function `compute_config_hash_from_file()`, enabling hash verification between attempts
- **Deterministic YAML Output**: Changed `sort_keys=False` to `sort_keys=True` in YAML dump for deterministic file output (required for consistent hashing)
- **Enhanced Cache Reload Logging**: Added config hash logging when cache is invalidated (mtime change) and when config is loaded/reloaded, with mtime values for debugging
- **Comprehensive Decision Tracking**: Added attempt history tracking in `evaluate_target_with_autofix()` with:
  1. **Per-attempt state tracking**: Config hash (before/after), metrics (AUC, composite, status), detected leaks, exclusions added
  2. **Comparison logging**: Detailed comparison between attempt N and attempt N+1 (status, metrics, config hash verification)
  3. **Decision point logging**: Clear logging of why rerun was triggered, why stopped, or why resolved
  4. **Detailed detection logging**: Breakdown of detected leaks (features, confidence scores, reasons, sources) - top 10 sorted by confidence
  5. **Detailed exclusion logging**: Exact counts and samples of exclusions added (exact patterns, prefix patterns, registry rejections)
- **View-Agnostic Implementation**: All changes work for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views (view parameters correctly passed through, decision tracking is view-agnostic)
- **Autofixer Integration Fixes**: Fixed dead code accessing non-existent `TargetContext` attributes, added validation for required parameters before auto-fixer calls, improved TaskType enum handling with robust fallbacks, enhanced error handling with detailed logging
- **Impact**:
  - Reruns immediately see updated exclusions (no cache staleness)
  - Hash verification ensures config changes are correctly applied between attempts
  - Complete audit trail of all decisions made during rerun loops
  - Detailed visibility into what features were excluded and why
  - Works correctly for both CS and SS views
  - All changes maintain determinism (use existing SST functions, deterministic sorting)
- **Files**: `leakage_auto_fixer.py`, `leakage_filtering.py`, `model_evaluation.py`, `model_evaluation/leakage_helpers.py`

### Added Comprehensive Root Cause Debugging
- **Debugging Enhancements**: Added extensive logging at key points to diagnose why previous fixes didn't work:
  1. **universe_sig_for_writes logging**: Log value after `resolve_write_scope()` to verify it's set correctly
  2. **feature_importances condition logging**: Log whether `feature_importances` and `output_dir` conditions are met
  3. **universe_sig_for_save logging**: Log final value after all fallbacks to identify if it's still None
  4. **SCOPE BUG enhanced logging**: Log at ERROR level with full context when `universe_sig` is None
  5. **Exception logging**: Log full traceback when feature importances save fails
  6. **Artifacts manifest logging**: Log when `feature_importances_dir` doesn't exist (directory never created)
  7. **Skip reasons logging**: Log why feature importances are skipped (invalid_for_ranking, no importances, or universe_sig None)
- **Improved universe_sig fallback**: Added additional fallback to extract `universe_sig` from `cohort_context` and `output_dir` path using `parse_reproducibility_path()` if `universe_sig_for_writes` and `resolved_data_config.get('universe_sig')` are both None
- **Impact**: Next run will show exactly where the flow breaks and why feature importances aren't being saved, enabling targeted fixes
- **Files**: `model_evaluation.py`, `model_evaluation/reporting.py`, `diff_telemetry.py`

### Fixed Root Cause: Feature Importances Not Being Saved, Artifacts Manifest SHA256, Boruta Config, and Replicate Folders
- **Critical Root Cause Fixes**: Fixed four issues preventing feature importances from being saved, causing null artifacts_manifest_sha256, Boruta not using ranking configs, and replicate folders missing
  1. **Feature importances not being saved (ROOT CAUSE)**: Fixed unreliable `'universe_sig_for_writes' in locals()` check at line 6939. Removed `locals()` check and use variable directly with try/except, with explicit fallback to `resolved_data_config.get('universe_sig')`. Also improved fallback at line 7564-7566 to check `resolved_data_config` if `universe_sig_for_writes` is None
  2. **Artifacts manifest SHA256 null**: Replaced manual path traversal in `_compute_artifacts_manifest_digest()` with existing SST function `get_scoped_artifact_dir()`. Uses `parse_reproducibility_path()`, `parse_attempt_id_from_cohort_dir()`, `run_root()`, and `normalize_target_name()` for consistent path resolution
  3. **Boruta not using ranking configs**: Updated Boruta time limit and threshold loading to prioritize `multi_model_config` (from `CONFIG/ranking/targets/multi_model.yaml` or `CONFIG/ranking/features/multi_model.yaml`) first, then fallback to `preprocessing_config` only if ranking configs unavailable. Uses existing `get_model_config()` SST function
  4. **Replicate folders missing/wrong location**: Added `attempt_id` parameter to `get_snapshot_base_dir()` and `save_snapshot_hook()`, passing it through the entire chain from `save_feature_importances()` → `save_snapshot_hook()` → `get_snapshot_base_dir()` to ensure replicate folders are created in correct `attempt_{id}/` directories
- **Impact**: 
  - Feature importances CSV files are now correctly saved (universe_sig properly resolved)
  - `artifacts_manifest_sha256` is correctly computed using SST path resolution
  - Boruta uses ranking configs (20 minutes for TARGET_RANKING, 45 minutes for FEATURE_SELECTION) instead of preprocessing config defaults
  - Replicate folders are created in correct `attempt_{id}/` directories instead of always `attempt_0/`
  - All fixes maintain determinism (use existing SST functions, no new non-deterministic sources)
- **Files**: `model_evaluation.py`, `diff_telemetry.py`, `model_evaluation/reporting.py`, `stability/feature_importance/hooks.py`, `stability/feature_importance/io.py`, `CONFIG/ranking/targets/multi_model.yaml`

### Fixed Feature Importances Not Being Saved and Artifacts Manifest Issues
- **Critical Fixes**: Fixed three issues preventing feature importances from being saved and causing null artifacts_manifest_sha256
  1. **Feature importances not being saved**: Fixed `universe_sig_for_importances` being set to `None` before `universe_sig_for_writes` was available. Updated to use `universe_sig_for_writes` directly when building `_feature_importances_to_save`, and added check before save call to update `universe_sig` if it's None but `universe_sig_for_writes` is available
  2. **Artifacts manifest SHA256 path resolution**: Added debug logging when `feature_importances_dir` is not found and when manifest is empty, improving diagnostics for path resolution issues in `_compute_artifacts_manifest_digest()`
  3. **Model scores not saved in snapshots**: Extracted model scores (AUC, R², IC, etc.) from `model_metrics` and passed them to `save_snapshot_hook()`, storing them in snapshot `outputs` field
- **Impact**: 
  - Feature importances CSV files are now correctly saved to `batch_{sig}/attempt_{id}/feature_importances/` (CROSS_SECTIONAL) and `symbol={sym}/attempt_{id}/feature_importances/` (SYMBOL_SPECIFIC)
  - `artifacts_manifest_sha256` will be correctly computed with better diagnostics if issues occur
  - Model scores are now stored in stability snapshots for better tracking of model performance alongside feature importance stability
  - All fixes maintain determinism (use existing SST functions, no new non-deterministic sources)
- **Files**: `model_evaluation.py`, `diff_telemetry.py`, `model_evaluation/reporting.py`, `stability/feature_importance/hooks.py`

### Fixed Reproducibility Tracking Issues
- **Critical Fixes**: Fixed six issues affecting reproducibility tracking and leakage auto-fixer
  1. **Missing horizon_minutes**: Added fallback to extract `horizon_minutes` from target column name using SST function `resolve_target_horizon_minutes()` when `target_horizon_minutes` is not available, preventing COHORT_AWARE mode downgrade to NON_COHORT
  2. **Array truth value errors (two locations)**: Fixed "ambiguous truth value of an array" errors in two locations:
     - Line 4355: Fallback detection by safely checking for None/empty arrays
     - Line 4417: `ctx.symbols` boolean check by using `isinstance()` and `len()` instead of direct boolean evaluation
  3. **ExperimentConfig type error**: Fixed `TypeError: argument of type 'ExperimentConfig' is not iterable` in `resolve_target_horizon_minutes()` by adding `isinstance(config, dict)` check before using `in` operator (ExperimentConfig objects don't have `horizon_extraction` anyway)
  4. **LOW_N_CS check using wrong metric**: Fixed `n_cs_valid` calculation to use number of unique timestamps (from `time_vals`) instead of number of models, and made LOW_N_CS check view-aware (only applies to CROSS_SECTIONAL, not SYMBOL_SPECIFIC)
  5. **Feature importances saved for invalid targets**: Moved feature importances save to after `valid_for_ranking` check, ensuring they're only saved for valid targets
  6. **Diff telemetry not finding comparable runs**: Investigation added - may be related to missing `horizon_minutes` causing `task_signature` mismatch (horizon is part of task_signature computation)
- **Impact**: 
  - COHORT_AWARE mode now works correctly for all targets (horizon extracted from target name)
  - Leakage auto-fixer now works correctly (no more ExperimentConfig type errors)
  - LOW_N_CS check uses correct metric (n_timestamps) and only applies to CROSS_SECTIONAL view
  - Feature importances only saved for valid targets, reducing clutter
  - All fixes maintain determinism (no new non-deterministic sources)
- **Files**: `model_evaluation.py`, `reproducibility_tracker.py`, `sst_contract.py`, `composite_score.py`

### Fixed Scoring Schema Version Inconsistency
- **Bug Fix**: Fixed inconsistency where `snapshot.json` had `scoring_schema_version: "1.1"` but `metrics.json` had `schema.scoring: "1.2"`
  - Updated `normalize_snapshot()` in `diff_telemetry.py` to correctly extract scoring version from nested `outputs['metrics']['schema']['scoring']` path
  - Updated default `scoring_schema_version` from `"1.1"` to `"1.2"` to match `get_scoring_schema_version()` (current version)
  - Added fallback logic to check nested `schema.scoring` path first, then top-level `scoring_schema_version` for backward compatibility
  - Applied same fix to `run_data` and `additional_data` fallback paths
- **Impact**: `snapshot.json` top-level `scoring_schema_version` now correctly matches `outputs.metrics.schema.scoring` and `metrics.json` `schema.scoring`. All three will consistently show `"1.2"` (or whatever version is in config).
- **Files**: `diff_telemetry.py`

### Fixed Duplicate Cohort Folders and Missing artifacts_manifest_sha256
- **Critical Fix**: Fixed duplicate cohort folder creation and null `artifacts_manifest_sha256` in snapshots
  - Fixed `log_run()` in `reproducibility_tracker.py` to use canonical `build_target_cohort_dir()` instead of manual path construction
  - Fixed `save_snapshot()` in `diff_telemetry.py` to use canonical path builder for target-first structure
  - Fixed path resolution in `_compute_artifacts_manifest_digest()` to correctly identify `view_dir` (3 levels up) and `batch_or_symbol_dir` (2 levels up)
  - Fixed feature_importances directory lookup to use `batch_or_symbol_dir` instead of incorrect `view_dir`
  - Added `attempt_id` extraction to metrics data for better traceability
- **Impact**: Eliminates duplicate cohort folders in both CROSS_SECTIONAL and SYMBOL_SPECIFIC views. All cohorts now correctly stored in `batch_*/attempt_*/cohort=...` or `symbol=*/attempt_*/cohort=...` structure. `artifacts_manifest_sha256` now correctly computed and no longer null in snapshots.
- **Files**: `reproducibility_tracker.py`, `diff_telemetry.py`, `metrics.py`

### Fixed Universe Signature Consistency for Batch Folders
- **Critical Fix**: Ensured all artifacts use canonical `universe_sig_for_writes` from `resolved_data_config`, preventing multiple `batch_` folders for the same universe
  - Updated `early_universe_sig` and `train_universe_sig` to match canonical `universe_sig_for_writes` after data preparation
  - Fixed `feature_exclusions` to move to canonical `batch_` directory if initially saved with wrong `universe_sig`
  - Fixed `featureset_artifacts` to use `universe_sig_for_writes` when available
  - Fixed `feature_importances` to use `universe_sig_for_importances` (derived from `universe_sig_for_writes`)
  - Ensured CROSS_SECTIONAL always requires `batch_` level (extracts from `cohort_id` if `universe_sig` missing)
- **Impact**: Eliminates duplicate `batch_` folders for the same universe. All artifacts for the same loaded symbols are now consistently stored in the same `batch_{universe_sig[:12]}` folder
- **Files**: `model_evaluation.py`, `target_first_paths.py`

### Fixed OutputLayout attempt_id Attribute Error
- **Bug Fix**: Fixed `'OutputLayout' object has no attribute 'attempt_id'` error when saving feature importances
  - Added `self.attempt_id` storage in `OutputLayout.__init__()` (defaults to 0 for backward compatibility)
  - Fixed missing `attempt_id` parameter in `multi_model_feature_selection.py` OutputLayout instantiations
  - Ensures consistent per-attempt artifact paths for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views
- **Impact**: Feature importances now save correctly for all stages (TARGET_RANKING, FEATURE_SELECTION) and views
- **Files**: `output_layout.py`, `multi_model_feature_selection.py`

### Output Structure Cleanup and Backward Compatibility
- **Infrastructure**: Comprehensive cleanup of output directory structure with human-readable batch IDs and per-attempt artifact storage
  - Replaced `universe={long_hash}` with `batch_{short_hash[:12]}` for CROSS_SECTIONAL directories (cleaner, more readable)
  - Moved `feature_importances`, `featureset_artifacts`, and `feature_exclusions` into `attempt_{id}/` subdirectories (preserves history across auto-fix reruns)
  - Fixed `find_cohort_dir_by_id()` to search in `batch_*/attempt_*/cohort={id}/` structure for CROSS_SECTIONAL
  - Fixed all `iterdir()` usages to use `rglob("cohort=*")` for backward compatibility (12 fixes across 5 files)
  - Fixed `drift.json` path construction to use `build_target_cohort_dir()` for canonical paths
  - Fixed `manifest.py` cohort scanning to use `rglob("cohort=*")` for nested structures
  - Fixed `artifacts_manifest_sha256` lookup to find artifacts in `attempt_*/feature_importances/` structure
  - Fixed `early_universe_sig` fallback to compute from `symbols` if `run_identity.dataset_signature` is missing
  - Fixed `_sanitize_for_json` import error by adding to `diff_telemetry/__init__.py` exports
- **Impact**: Cleaner output structure, preserved artifact history, comprehensive backward compatibility. All readers now correctly handle nested `batch_*/attempt_*/cohort=*` structures.
- **Files**: `target_first_paths.py`, `output_layout.py`, `reproducibility_tracker.py`, `diff_telemetry.py`, `manifest.py`, `trend_analyzer.py`, `metrics.py`, `metrics_aggregator.py`, `training_strategies/reproducibility/io.py`, `intelligent_trainer.py`, `model_evaluation.py`, `diff_telemetry/__init__.py`
- **Details**: See `DOCS/02_reference/changelog/2026-01-10-output-structure-cleanup-and-backward-compatibility.md`

### Complete Non-Determinism Elimination
- **Critical Fix**: Eliminated all remaining non-determinism in feature ordering, DataFrame column ordering, and permutation importance
  - Fixed non-deterministic `mtf_data` dictionary iteration affecting `pd.concat()` column order
  - Fixed non-deterministic sample DataFrame selection for column discovery
  - Fixed non-deterministic feature auto-discovery (now always sorted)
  - Fixed non-deterministic symbol iteration in feature alignment
  - Added deterministic seeds for permutation importance shuffle operations
- **Impact**: Ensures "same inputs → same features → same data → same models → same outputs" end-to-end. All model inputs are now fully deterministic.
- **Files**: `cross_sectional_data.py`, `strategy_functions.py`, `shared_ranking_harness.py`, `leakage_detection.py`
- **Details**: See `DOCS/02_reference/changelog/2026-01-10-complete-non-determinism-elimination.md`

## 2026-01-09

### Fix save_rankings() Path Resolution Consistency
- **Fixed Missing run_root() Call in save_rankings()**: Added `run_root()` call to `save_rankings()` function for consistency with `_save_dual_view_rankings()`. Ensures both functions use the same path resolution logic to find the run root directory before writing to `globals/` directory
- **Impact**: Rankings files (`target_predictability_rankings.csv`, `target_prioritization.yaml`) now correctly resolve run root directory, ensuring they're written to the correct `globals/` location even if `output_dir` parameter isn't exactly the run root
- **Files**: `reporting.py`

### Globals Outputs and Manifest View/Stage Fixes
- **Fixed routing_decisions.json Fingerprint View Determination**: Replaced legacy `context.get("view")` with symbol-count-based view determination using established SST pattern (`len(symbols) > 1` → CROSS_SECTIONAL, `len(symbols) == 1` → SYMBOL_SPECIFIC). Ensures fingerprint correctly reflects actual view for multi-symbol runs
- **Fixed routing_decision.json Paths**: Added `stage=FEATURE_SELECTION/` prefix to paths in per-target routing decision files. Paths now use stage-scoped structure: `targets/{target}/reproducibility/stage=FEATURE_SELECTION/{view}/...`. Uses View enum for normalization (SST pattern)
- **Fixed metrics_aggregator.py View Determination**: Replaced `context.get("view")` with symbol-count-based determination in two locations:
  - Line 77: Uses `symbols` parameter directly from `aggregate_routing_candidates()` method
  - Line 934: Extracts unique symbols from `candidates_df` DataFrame for view determination
- **Fixed training_router.py View Determination**: Replaced `context.get("view")` with symbol-count-based determination by extracting unique symbols from `routing_candidates` DataFrame
- **Fixed Fingerprint Validation**: Updated `load_routing_decisions()` to use symbols from stored `fingerprint_data` in routing_decisions.json file for consistent view determination during validation
- **Impact**: All globals outputs (`routing_decisions.json`, `routing_decision.json`, routing plan metadata) now use correct view determination based on actual symbol count, not potentially stale legacy global view. Manifest `_find_files()` already recursively handles stage-scoped structure correctly
- **Maintains SST Principles**: All fixes use established SST patterns (symbol-count-based view determination, View enum normalization, stage-scoped paths, fallback to validated cache entries)
- **Files**: `target_routing.py` (ranking and orchestration), `metrics_aggregator.py`, `training_router.py`

## 2026-01-09

### Scoring Calibration Fixes: Eligibility Gates and Quality Formula
- **Fixed Double-Counting of Standard Error**: Removed stability from quality calculation since SE is already in t-stat. Quality now = coverage × registry_coverage × sample_size (multiplicative)
- **Added Eligibility Gates**: Targets with registry_coverage < 0.95 or n_slices_valid < 20 are marked `valid_for_ranking = False` with explicit reasons. Prevents low-quality targets from ranking high
- **Sample Size Penalty**: Targets with n_slices_valid between 20-30 get quality penalty (0.7-1.0), but still valid for ranking
- **Scoring Version Bumped to 1.2**: Updated scoring signature to include eligibility params for determinism
- **Files**: `composite_score.py`, `scoring.py`, `model_evaluation.py`, `target_ranker.py`, `metrics_schema.yaml`

### View Cache Conflict Fix: SS→CS Promotion Root Cause
- **Fixed View Cache Overriding Explicitly Requested Views**: Cache now only reused when requested_view matches cached view or is None (auto mode). On conflict, resolves fresh and logs warning
- **Root Cause**: Cached CROSS_SECTIONAL view was unconditionally reused even when SYMBOL_SPECIFIC was explicitly requested, causing SS→CS promotion warnings
- **Impact**: Explicit SYMBOL_SPECIFIC requests no longer overridden by cached CROSS_SECTIONAL. Cache still reused when appropriate
- **Files**: `cross_sectional_data.py`

### Multi-Symbol SYMBOL_SPECIFIC View Validation Fix
- **Fixed Multi-Symbol Runs Routing to Wrong Directories**: Added validation to prevent SYMBOL_SPECIFIC view when `n_symbols > 1`. Multi-symbol runs (e.g., universe `f517a23ce02cdcad4887b95107f165cc69f15796ccfd07c3b8e1466fbd2102f5`) now correctly route to `CROSS_SECTIONAL/universe={universe_sig}/` instead of `SYMBOL_SPECIFIC/symbol=.../universe={universe_sig}/`
- **Root Cause**: Auto-resolution logic didn't validate that `requested_view=SYMBOL_SPECIFIC` requires `n_symbols=1`. Multi-symbol runs incorrectly resolved to SYMBOL_SPECIFIC view
- **Fix**: Added validation before auto-resolution to clear invalid SYMBOL_SPECIFIC requests for multi-symbol runs. Added cache compatibility check. Added single-symbol validation in FEATURE_SELECTION stage
- **Impact**: Multi-symbol runs correctly route to CROSS_SECTIONAL directories. Files no longer incorrectly written to SYMBOL_SPECIFIC folders. Clear warnings when invalid requests are overridden
- **Files**: `cross_sectional_data.py`, `feature_selector.py`, `multi_model_feature_selection.py`

### Production Hardening: Prev Run Selection and Stage-Scoping Warnings
- **Improved Prev Run Selection**: Changed from `date` (timestamp) to `run_started_at` (monotonic) for more reliable ordering. Handles clock skew and resumed runs better. Falls back to `date` for backward compatibility with old index files
- **Added Stage-Scoping Warnings**: Early detection of cross-stage contamination by validating `full_metadata['stage']` matches current stage before calling `finalize_run()`. Provides early warning (diff_telemetry also validates, but early is better)
- **Files**: `reproducibility_tracker.py`

### Fix CROSS_SECTIONAL View Detection for Feature Importances
- **Fixed Incorrect SYMBOL_SPECIFIC Detection for Multi-Symbol CROSS_SECTIONAL Runs**: Resolved issue where CROSS_SECTIONAL runs (10 symbols) were incorrectly detected as SYMBOL_SPECIFIC when saving feature importances, causing "SYMBOL_SPECIFIC view requires symbol" errors
  - **Root Cause**: Auto-detection logic at function entry (line 5309-5313) converted CROSS_SECTIONAL to SYMBOL_SPECIFIC if `symbol` parameter was provided, without checking if it was actually a single-symbol run. For multi-symbol CROSS_SECTIONAL runs, `symbol` could be set from a previous target's state, causing incorrect auto-detection
  - **Fix #1 - Function Entry Auto-Detection** (`model_evaluation.py` lines 5309-5318): Added validation to check `symbols` parameter length before auto-detecting SYMBOL_SPECIFIC. Only triggers auto-detection if `len(symbols) == 1`. For multi-symbol runs, clears the `symbol` parameter to prevent incorrect detection
  - **Fix #2 - Feature Importances Auto-Detection** (`model_evaluation.py` lines 6575-6587): Added defensive validation in feature importances saving logic to check `resolved_data_config.symbols` or `symbols_array` length before auto-detecting SYMBOL_SPECIFIC. Clears `symbol_for_importances` for multi-symbol CROSS_SECTIONAL runs
  - **Impact**: CROSS_SECTIONAL runs with multiple symbols now correctly maintain CROSS_SECTIONAL view and `symbol=None`. Feature importances save to correct directories. No "SS→CS promotion detected" warnings for legitimate CROSS_SECTIONAL runs. SYMBOL_SPECIFIC runs remain unaffected (explicit SYMBOL_SPECIFIC bypasses auto-detection, single-symbol runs still auto-detect correctly)
  - **Maintains SST Principles**: Uses same validation pattern, preserves backward compatibility for legitimate SYMBOL_SPECIFIC runs

### Symbol Parameter Propagation Fix - All Stages (FEATURE_SELECTION and TRAINING)
- **Fixed Symbol Parameter Propagation Across All Stages**: Applied comprehensive fixes to FEATURE_SELECTION and TRAINING stages to match TARGET_RANKING fixes
  - **FEATURE_SELECTION - View Auto-Detection** (`feature_selector.py` line 264): Added auto-detection to convert `CROSS_SECTIONAL` to `SYMBOL_SPECIFIC` when symbol is provided, matching TARGET_RANKING behavior
  - **FEATURE_SELECTION - OutputLayout Validation** (`feature_selection_reporting.py` lines 437-443): Added auto-detection and validation in `save_feature_importances_for_reproducibility()` to ensure symbol is provided before creating OutputLayout, preventing "SYMBOL_SPECIFIC view requires symbol" errors
  - **FEATURE_SELECTION - Multi-Model Results Auto-Detection** (`multi_model_feature_selection.py` lines 5116-5120): Added view auto-detection to `save_multi_model_results()` using same pattern as other FEATURE_SELECTION fixes, ensuring consistent behavior across all entry points
  - **TRAINING - Path Validation** (`training_strategies/reproducibility/io.py` lines 81-87): Added validation warning when `SYMBOL_SPECIFIC` view is used without symbol parameter in `get_training_snapshot_dir()`
  - **TRAINING - Verified Symbol Parameter**: Confirmed `ArtifactPaths.model_dir()` correctly receives symbol parameter for SYMBOL_SPECIFIC routes (line 763) and None for CROSS_SECTIONAL routes (line 1832)
  - **Impact**: All three stages now have consistent symbol parameter propagation, view auto-detection, and validation. FEATURE_SELECTION and TRAINING stages will correctly route symbol-specific data to `SYMBOL_SPECIFIC/symbol=.../` directories
  - **Maintains SST Principles**: Uses same auto-detection and validation patterns as TARGET_RANKING, ensuring consistency across all stages

### Symbol Parameter Propagation Fix - Complete Path Construction
- **Fixed Symbol Parameter Propagation for All Path Construction**: Resolved multiple issues where symbol parameter was not properly propagated, causing: 1) "SYMBOL_SPECIFIC view requires symbol" error in `save_feature_importances()`, 2) Universe directories created under `SYMBOL_SPECIFIC/` instead of `SYMBOL_SPECIFIC/symbol=.../`, 3) Missing feature importance snapshots per symbol
  - **Fix #1 - resolve_write_scope()** (`model_evaluation.py` line 5671): Pass `symbol` parameter to `resolve_write_scope()` instead of `None`, enabling proper symbol derivation from SST config
  - **Fix #2 - symbol_for_importances fallback** (`model_evaluation.py` lines 6548-6558): Added fallback logic to ensure `symbol_for_importances` is set when view is `SYMBOL_SPECIFIC`, extracting from function parameter or `resolved_data_config.symbols` if available
  - **Fix #3 - ensure_scoped_artifact_dir() calls** (`model_evaluation.py` lines 939-943, 5395-5399, 6026-6030): Added symbol derivation logic for all artifact directory calls, extracting symbol from `symbols_array` or `symbols_to_load` when view is `SYMBOL_SPECIFIC` but symbol parameter is None
  - **Fix #4 - get_scoped_artifact_dir() validation** (`target_first_paths.py` lines 254-260): Added warning when `SYMBOL_SPECIFIC` view is used without symbol parameter, helping identify routing issues
  - **Impact**: All artifact directories (featureset_artifacts, feature_exclusions) now correctly route to `SYMBOL_SPECIFIC/symbol=.../universe=.../` instead of `SYMBOL_SPECIFIC/universe=.../`. Feature importances save successfully with correct symbol parameter. All path construction now properly includes symbol component for SYMBOL_SPECIFIC view
  - **Maintains SST Principles**: Uses existing symbol derivation patterns from `resolve_write_scope()`, preserves backward compatibility

### Symbol-Specific Routing Fix - View Propagation and Auto-Detection
- **Fixed Single-Symbol Runs Routing to CROSS_SECTIONAL**: Resolved critical bug where single-symbol runs were incorrectly routed to `CROSS_SECTIONAL` directories instead of `SYMBOL_SPECIFIC/symbol=.../` directories
  - **Root Cause**: Auto-detection at line 5300 correctly set `view = View.SYMBOL_SPECIFIC`, but `requested_view_from_context` was loaded from run context (which could be `CROSS_SECTIONAL`), overriding the auto-detected view before it was passed to data preparation function
  - **Fix #1 - View Propagation** (`model_evaluation.py` lines 5313-5318): After auto-detection sets `view = View.SYMBOL_SPECIFIC`, ensure `requested_view_from_context` uses the auto-detected view instead of loading from run context. This ensures `prepare_cross_sectional_data_for_ranking()` receives `SYMBOL_SPECIFIC` as `requested_view`, which then propagates to `resolved_data_config` and `view_for_writes` via `resolve_write_scope()`
  - **Fix #2 - Feature Importances Auto-Detection** (`model_evaluation.py` lines 6542-6546): Added auto-detection for `view_for_importances` to check if `symbol_for_importances` is set and force `SYMBOL_SPECIFIC` view, even if `view_for_writes` has wrong value. This ensures feature importances are saved to correct directory as a safety net
  - **Downstream Impact**: Fixes propagate through entire pipeline:
    - `requested_view_from_context` → `prepare_cross_sectional_data_for_ranking()` → `resolved_data_config.view` → `resolve_write_scope()` → `view_for_writes` → all path construction
    - Feature importances, metrics, snapshots, and all artifacts now route to correct `SYMBOL_SPECIFIC/symbol=.../` directories
  - **Impact**: Single-symbol runs now correctly route to `SYMBOL_SPECIFIC/symbol=.../universe=.../` directories. Log messages now show `SYMBOL_SPECIFIC` instead of `CROSS_SECTIONAL (symbol=AMZN)`. All downstream path construction uses correct view
  - **Maintains SST Principles**: Uses existing auto-detection pattern, preserves backward compatibility, and ensures consistent view handling throughout pipeline

### Comprehensive JSON/Parquet Serialization Fixes - SST Solution
- **Fixed JSON Serialization with Enum Objects**: Resolved critical issue where Stage/View enum objects were written directly to JSON, causing serialization failures and missing output files
  - **New SST Helpers in `file_utils.py`**: Created centralized `sanitize_for_serialization()`, `safe_json_dump()`, and `safe_dataframe_from_dict()` helpers that recursively convert enum objects to strings and handle pandas Timestamps
  - **Comprehensive Migration**: Replaced all direct `json.dump()` calls with `safe_json_dump()` across 11 files (136 total instances):
    - `intelligent_trainer.py`: Feature selection summary, model family status, selected features summary, target ranking cache, decision files (decision_used.json, resolved_config.json, applied_patch.json)
    - `target_routing.py`: Target confidence summary, routing path, feature routing file
    - `training_plan_generator.py`: Master training plan, JSON views (by_target, by_symbol, by_type, by_route)
    - `training_router.py`: Routing plan JSON
    - `routing_candidates.py`: Routing candidates JSON
    - `manifest.py`: Manifest updates, target metadata, resolved config, overrides config (4 locations)
    - `run_context.py`: Run context JSON saves (2 locations)
    - `reproducibility_tracker.py`: Audit report JSON
    - `checkpoint.py`: Checkpoint JSON writes
  - **Parquet Serialization**: Replaced `pd.DataFrame([data])` with `safe_dataframe_from_dict(data)` in `metrics.py` for drift_results, rollup_data, and metrics.parquet writes
  - **Impact**: All JSON and Parquet files now write successfully, fixing broken outputs in globals/ directory and all stage outputs
  - **Maintains SST Principles**: Centralized helpers ensure consistent enum handling across entire codebase

### Metrics Duplication Fix
- **Fixed Duplicate Metrics in metrics.json**: Resolved issue where metrics were written twice (nested under `'metrics'` key and at root level)
  - `reproducibility_tracker.py` (lines 2339-2349): Modified `write_cohort_metrics()` call to extract nested `'metrics'` dict from `run_data` or filter out non-metric keys before passing to metrics writer
  - `metrics.py` (lines 336-353): Enhanced `_write_metrics()` to detect and extract nested `'metrics'` dict as defensive safety net
  - **Impact**: `metrics.json` files now contain clean, non-duplicated metric data
  - **Backward Compatible**: Handles both nested and flat metric structures

### Symbol-Specific Routing Auto-Detection Fixes
- **Fixed Symbol-Specific View Auto-Detection**: Fixed bug where symbol-specific runs were being labeled as CROSS_SECTIONAL instead of SYMBOL_SPECIFIC
  - `model_evaluation.py` (lines 5297-5300): Auto-detects SYMBOL_SPECIFIC view when symbol is provided instead of nullifying symbol
  - `model_evaluation/reporting.py` (lines 180-184): Auto-detects SYMBOL_SPECIFIC view in `save_feature_importances()` when symbol is provided
  - `reproducibility_tracker.py` (line 2016): Fixed logic bug where FEATURE_SELECTION view determination happened after default CROSS_SECTIONAL assignment
  - Single-symbol runs now automatically route to SYMBOL_SPECIFIC directories instead of CROSS_SECTIONAL
  - Feature importances and other artifacts now correctly written to `SYMBOL_SPECIFIC/symbol=.../` paths
  - All fixes maintain SST principles and preserve backward compatibility

### Root Cause Fixes - NoneType Errors and Path Construction
- **Fixed NoneType.replace() Error - Root Cause**: Resolved persistent `'NoneType' object has no attribute 'replace'` error by passing `additional_data` parameter to `extract_run_id()` call at line 1196, enabling multi-source extraction from `run_data`, `additional_data`, and `metrics` dictionaries
- **Fixed Symbol-Specific Data Path Construction**: Fixed critical bug where symbol-specific data was being written to CROSS_SECTIONAL directories instead of SYMBOL_SPECIFIC/symbol=<symbol>/ directories
  - `reproducibility_tracker.py` (lines 1984-2029): Symbol check now happens FIRST before any view determination - if `symbol` is set, forces `SYMBOL_SPECIFIC` view immediately
  - `reproducibility_tracker.py` (lines 2048-2060): Path construction ensures symbol is included in path for SYMBOL_SPECIFIC view
  - `reproducibility_tracker.py` (lines 4078-4082, 4409-4419, 4562-4572): Fixed drift.json, metadata lookup, and metrics rollup path construction
  - `diff_telemetry.py` (lines 2904-2908): Fixed snapshot path construction to prioritize symbol presence
- **Additional NoneType Safety**: Added defensive check in `cohort_id.py` (line 107) to ensure `leak_ver` is never None before calling `.replace()` method
- All fixes maintain SST principles (enum usage, centralized helpers) and preserve hash verification data
- Verified all files compile successfully and symbol-specific data now goes to correct directories

### NoneType Replace Error Fixes - All Stages
- **Fixed Persistent NoneType Replace Error**: Resolved `'NoneType' object has no attribute 'replace'` error across all three stages (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
  - `reproducibility_tracker.py` (line 4002-4026): Multi-source `run_id` extraction from `run_data`, `additional_data`, and `metrics` with `NameError` handling
  - `cross_sectional_feature_ranker.py` (line 613-622): Defensive check for `audit_result.get('run_id')` before `.replace()` in TARGET_RANKING stage
  - `diff_telemetry.py` (line 4362-4370): Defensive check for `timestamp` before `.replace('Z', '+00:00')` across all stages
  - `intelligent_trainer.py` (lines 1239, 2411, 4030): Defensive checks for `_run_name` before `.replace()` in all three stages
  - All fixes include fallback to `datetime.now().isoformat()` if extraction fails
  - Preserves all hash verification data and determinism tracking (no impact on reproducibility)
  - Verified all files compile successfully and all stages are protected

### Comprehensive File Write Fixes - Enum Normalization and NoneType Error Resolution
- **Fixed Snapshot Creation Enum Normalization**: All snapshot creation functions now normalize Stage and View enums to strings before storing in dataclass fields
  - `TrainingSnapshot.from_training_result()`: Normalizes Stage.TRAINING and View enum inputs to strings
  - `FeatureSelectionSnapshot.from_importance_snapshot()`: Normalizes Stage and View enum inputs to strings
  - `NormalizedSnapshot` creation in `diff_telemetry.py`: Normalizes Stage enum to string
  - `create_aggregated_training_snapshot()`: Normalizes View enum to string
  - Prevents enum objects from being stored in snapshot dataclasses, ensuring JSON serialization works correctly

- **Fixed JSON Write Sanitization**: Added enum sanitization to all JSON write operations across the codebase
  - `write_atomic_json()` in `file_utils.py`: Now sanitizes data to normalize enums before JSON serialization
  - `manifest.py`: All manifest.json writes now sanitize enum values
  - `target_routing.py`: Confidence summary JSON writes sanitize enums
  - `ranking/target_routing.py`: Routing decisions JSON writes sanitize enums
  - `stability/feature_importance/io.py`: Snapshot index JSON writes sanitize enums
  - `training_strategies/reproducibility/io.py`: Training snapshot index and summary JSON writes sanitize enums
  - `routing_candidates.py`: Routing candidates JSON writes sanitize enums
  - `training_plan_generator.py`: Training plan JSON writes sanitize enums
  - `training_router.py`: Routing plan JSON writes sanitize enums
  - `run_context.py`: Run context JSON writes sanitize enums
  - All JSON writes now use local `_sanitize_for_json()` helper that recursively converts Enum objects to their `.value` property
  - Fixes missing JSON files in globals directory and other output locations

- **Fixed NoneType Replace Error**: Resolved persistent `'NoneType' object has no attribute 'replace'` error in reproducibility tracking
  - `reproducibility_tracker.py` (line 4004-4016): Added comprehensive defensive checks for run_id/timestamp extraction
  - Handles cases where `run_data` is not a dict, `run_id`/`timestamp` are None/empty, or values are not strings
  - Always falls back to `datetime.now().isoformat()` if extraction fails
  - Prevents crashes when run data is malformed or missing required fields

- **Overwrite Protection Verified**: Confirmed all fixes maintain existing overwrite protection mechanisms
  - Idempotency checks in `_update_index()` still deduplicate by (run_id, phase)
  - File locking in `_write_atomic_json_with_lock()` still prevents concurrent writes
  - Atomic writes ensure crash consistency
  - No overwriting issues reintroduced

### Comprehensive SST Path Construction Fixes - All Stages
- **Fixed Path Construction with Enum Values**: Resolved enum-to-string conversion issues in path construction across all three stages
  - `target_first_paths.py`: Fixed `get_target_reproducibility_dir()`, `find_cohort_dir_by_id()`, and `target_repro_dir()` to normalize Stage/View enums to strings before path construction
  - `output_layout.py`: Fixed `__init__()` and `repro_dir()` to normalize View and Stage enums to strings
  - `training_strategies/reproducibility/io.py`: Fixed `get_training_snapshot_dir()` to normalize View and Stage enums to strings
  - All path construction functions now explicitly convert enum values to strings using `.value` property before building paths
  - Fixes missing metric artifacts and JSON files across TARGET_RANKING, FEATURE_SELECTION, and TRAINING stages
  - Maintains backward compatibility with string inputs
  - Verified all three stages work correctly with enum inputs

### Metric Output JSON Serialization Fixes
- **Fixed Broken Metric Outputs**: Resolved JSON serialization issues where Stage/View enum objects were written directly to JSON
  - `metrics.py`: Updated `write_cohort_metrics()`, `_write_metrics()`, `_write_drift()`, `generate_view_rollup()`, and `generate_stage_rollup()` to normalize enum inputs to strings before JSON serialization
  - `reproducibility_tracker.py`: Updated `generate_metrics_rollups()` and fixed `ctx.stage`/`ctx.view` normalization before passing to `write_cohort_metrics()`
  - All metric output functions now accept `Union[str, Stage]` and `Union[str, View]` for SST compatibility
  - All enum values are explicitly converted to strings using `.value` property before being added to JSON dictionaries
  - Fixes broken metric outputs in CROSS_SECTIONAL view for target ranking and all other stages
  - Maintains backward compatibility with string inputs
  - Verified all JSON outputs contain string values (not enum objects)

### SST Implementation Progress Summary - Complete
- **Phase 1 Complete**: View and Stage enum migrations (29 files total)
- **Phase 2 Complete**: WriteScope function migration (4 functions updated)
- **Phase 3 Complete**: Helper function audits and standardization
  - Phase 3.1: Scope resolution migration - Complete
  - Phase 3.2: RunIdentity factory audit - Complete
  - Phase 3.3: Cohort ID unification - Complete
  - Phase 3.4: Config hash audit - Complete
  - Phase 3.5: Universe signature audit - Complete
- **All SST candidates implemented**: Complete migration to SST-centric architecture with consistent enums, WriteScope objects, and unified helpers

## 2026-01-09

### Syntax and Import Fixes
- **Fixed Syntax Errors**: Resolved all syntax and indentation errors in TRAINING pipeline
  - `model_evaluation.py`: Fixed indentation error in try block (line 6011-6026)
  - `cross_sectional_feature_ranker.py`: Fixed indentation error after try statement (line 703)
  - `multi_model_feature_selection.py`: Fixed orphaned else block and try block indentation (lines 4325-4327, 5453)
  - `intelligent_trainer.py`: Fixed `UnboundLocalError` for `Path` - removed redundant local import that shadowed global import (line 3970)
  - All Python files now compile without syntax errors
  - All critical modules import successfully

### SST Import Shadowing Fixes
- **Fixed UnboundLocalError Issues**: Resolved all import shadowing issues from SST refactoring
  - `model_evaluation.py:8129`: Removed `Stage` from local import (already imported globally at line 41) - fixes `UnboundLocalError: local variable 'Stage' referenced before assignment` at line 5390
  - `shared_ranking_harness.py:286`: Added global `Stage` import and removed redundant local import - prevents `UnboundLocalError` in `create_run_context` method
  - Removed redundant local `Path` imports in `diff_telemetry.py` (3 instances), `target_routing.py`, and `training.py` (5 instances) where `Path` is already imported globally
  - Verified all path construction functions correctly convert enum values to strings using `str(enum)` which returns `.value`
  - Verified all JSON serialization correctly handles enum values (enums inherit from `str` so serialize correctly, and `_sanitize_for_json` explicitly converts enums to `.value` for safety)
  - All critical modules now import without `UnboundLocalError` issues
  - All path construction and file writes verified to work correctly with enum values

### Additional SST Improvements
- **String Literal to Enum Migration**: Replaced hardcoded string comparisons with enum comparisons
  - `metrics.py`: 7+ instances - replaced `view == "SYMBOL_SPECIFIC"` with `view_enum == View.SYMBOL_SPECIFIC`
  - `trend_analyzer.py`: 4+ instances - replaced `stage == "TARGET_RANKING"` with `stage_enum == Stage.TARGET_RANKING`
  - `target_routing.py`: 3+ instances - replaced string lists and comparisons with View enum values
  - `cache_manager.py`, `artifact_mirror.py`, `leakage_detection/reporting.py`, `model_evaluation/reporting.py`, `reproducibility/io.py`, `manifest.py`, `hooks.py`, `metrics_schema.py`: All string comparisons migrated to enum comparisons
  - All enum comparisons use `View.from_string()` or `Stage.from_string()` for normalization, ensuring backward compatibility
  - All path construction uses `view_enum.value` to ensure string output for filesystem paths
  - JSON output format unchanged - enum values serialize as strings via `.value` property

- **Config Hashing Standardization**: Replaced manual hashlib calls with canonical_json/sha256 helpers
  - `fingerprinting.py`: Universe signature computation now uses `canonical_json()` + `sha256_short()`
  - `cohort_metadata_extractor.py`: Data fingerprinting now uses `sha256_short()` helper
  - `reproducibility/utils.py`: Comparison key hashing now uses `sha256_short()` helper
  - `diff_telemetry/types.py`: Hash computation now uses `canonical_json()` + `sha256_short()` helpers
  - All changes maintain same hash output format (backward compatible)
  - Binary file hashing (lock files, binary data) kept as-is (appropriate use of hashlib)

- **Verification**: All changes verified to maintain JSON output format and metric tracking
  - Enum comparisons work correctly with both string and enum inputs
  - Enum values serialize as strings in JSON (via `.value` property)
  - Path construction produces identical paths (enum `.value` matches original strings)
  - All test suites pass: imports, enum access, path construction, JSON serialization

### SST Implementation Complete - All Phases Verified
- **Comprehensive Verification**: All SST migration phases completed and verified
  - **Enum Migration**: 29 files migrated to use View and Stage enums (only 2 stage strings remain in comments, 23 view strings in appropriate contexts)
  - **WriteScope Migration**: 4 functions now accept WriteScope objects with backward compatibility
  - **Helper Unification**: All scope resolution, cohort ID, config hashing, and universe signature computations use unified SST helpers
  - **Backward Compatibility**: All changes maintain full backward compatibility with existing JSON files, snapshots, and metrics
  - **No Breaking Changes**: All file paths, JSON serialization, and existing data formats remain unchanged

### SST Config Hash Audit (Phase 3.4) - Complete
- **Config Hash Standardization**: Updated manual config hash computations to use shared helpers from `config_hashing.py`
  - `reproducibility_tracker.py`: Replaced manual `json.dumps()` + `hashlib.sha256()` with `canonical_json()` + `sha256_short()`
  - `diff_telemetry.py`: Replaced manual `hashlib.sha256()` calls with `sha256_short()` helper
  - All config hashing now uses consistent logic: `canonical_json()` for normalization, `sha256_full()` or `sha256_short()` for hashing
  - Hash lengths standardized: 8 chars for short hashes, 16 chars for medium, 64 chars for full identity keys

### SST Cohort ID Unification (Phase 3.3) - Complete
- **Cohort ID Generation Unification**: Created unified `compute_cohort_id()` helper in `cohort_id.py`
  - Extracted duplicate logic from `ReproducibilityTracker._compute_cohort_id()` and `compute_cohort_id_from_metadata()`
  - Both implementations now delegate to unified helper (SST-compliant)
  - Uses View enum for consistent view handling
  - Uses `extract_universe_sig()` helper for SST-compliant universe signature access
  - All cohort ID generation now uses single source of truth

### SST RunIdentity Factory Audit (Phase 3.2) - Complete
- **RunIdentity Construction Audit**: Verified all RunIdentity constructions follow SST patterns
  - Factory `create_stage_identity()` is used for creating new identities from scratch (9 instances)
  - Manual constructions are for legitimate use cases: updating existing identities, finalizing partial identities, or copying with modifications
  - All new identity creation uses factory pattern (SST-compliant)
  - Remaining manual constructions are for identity updates/copies (correct pattern)

### SST Universe Signature Audit (Phase 3.5) - Complete
- **Universe Signature Consistency**: Verified all universe signature computations use `compute_universe_signature()` helper
  - All 22 instances across 10 files verified to use `compute_universe_signature()` from `run_context.py`
  - Fallback manual computation in `fingerprinting.py` is defensive and acceptable (only used if helper unavailable)
  - One instance in `model_evaluation.py` is for `symbols_digest` (metadata), not universe signature - correctly different
  - All universe signature computations now consistent and SST-compliant

### SST Scope Resolution Migration (Phase 3.1) - Complete
- **Manual Scope Resolution Replacement**: Replaced manual `resolved_data_config.get('view')` and `resolved_data_config.get('universe_sig')` patterns with `resolve_write_scope()` helper
  - `feature_selector.py`: Consolidated manual universe_sig and view extraction into single `resolve_write_scope()` call
  - `model_evaluation.py`: Replaced manual view/universe_sig extraction with `resolve_write_scope()` for canonical scope resolution
  - All scope resolution now uses SST helper, ensuring consistent scope handling across codebase
  - Remaining `.get()` calls are for telemetry/metadata purposes (not scope resolution)

### SST WriteScope Migration (Phase 2.2) - Complete
- **WriteScope Function Migration**: Migrating functions to accept WriteScope objects for SST consistency
  - `get_scoped_artifact_dir()` and `ensure_scoped_artifact_dir()` in `target_first_paths.py` now accept `scope: WriteScope` parameter
  - `model_output_dir()` in `target_first_paths.py` now accepts `scope: WriteScope` parameter
  - `build_cohort_metadata()` in `cohort_metadata.py` now accepts `scope: WriteScope` parameter
  - All functions maintain backward compatibility with loose (view, symbol, universe_sig, stage) parameters
  - When `scope` is provided, extracts view, symbol, universe_sig, and stage from WriteScope
  - All call sites remain compatible (using deprecated parameters for now)

### SST WriteScope Migration (Phase 2.1) - Identification
- **WriteScope Adoption Planning**: Identified functions accepting loose (view, symbol, universe_sig) tuples for WriteScope migration
  - Functions identified for migration:
    - `get_scoped_artifact_dir()` and `ensure_scoped_artifact_dir()` in `target_first_paths.py` - accept view, symbol, universe_sig, stage
    - `model_output_dir()` in `target_first_paths.py` - accepts view, symbol, universe_sig
    - `build_cohort_metadata()` in `cohort_metadata.py` - accepts view, universe_sig, symbol
  - Migration strategy: Start with internal functions (lowest call sites), work outward to public APIs
  - Maintain backward compatibility with wrapper functions where needed

### SST Stage Enum Migration (Phase 1.2) - Complete
- **Stage Enum Adoption**: Migrated 17 files to use `Stage` enum from `scope_resolution.py` instead of hardcoded string literals
  - Function signatures updated to accept `Union[str, Stage]` for backward compatibility
  - All comparisons use enum instances: `stage_enum == Stage.TARGET_RANKING` instead of `stage == "TARGET_RANKING"`
  - All JSON serialization uses `str(stage_enum)` which returns `.value` via `__str__`
  - All path construction uses `str(stage_enum)` for consistent string conversion
- **Backward Compatibility Guaranteed**:
  - `Stage.from_string()` normalizes strings from JSON/metadata to enum instances (handles "MODEL_TRAINING" → "TRAINING" alias)
  - Existing JSON files (metadata.json, snapshot.json, metrics.json) continue to work unchanged
  - File paths remain identical (enum `__str__` returns `.value`)
- **Files Updated**:
  - `reproducibility_tracker.py` - Complete Stage enum migration with proper normalization and comparisons
  - `target_first_paths.py` - Already accepts `Union[str, Stage]` for stage parameters
  - `predictability/main.py`, `model_evaluation/reporting.py`, `leakage_detection.py` - TARGET_RANKING stage enum
  - `multi_model_feature_selection.py`, `feature_selector.py` - FEATURE_SELECTION stage enum
  - `target_ranker.py`, `dominance_quarantine.py` - TARGET_RANKING stage enum
  - `training.py`, `reproducibility/io.py`, `reproducibility/schema.py` - TRAINING stage enum
  - `intelligent_trainer.py` - TARGET_RANKING and FEATURE_SELECTION stage enum
- **No Breaking Changes**: All existing snapshots, metrics, and JSON files remain fully compatible

### SST View Enum Migration (Phase 1.1)
- **View Enum Adoption**: Migrated 17 files to use `View` enum from `scope_resolution.py` instead of hardcoded string literals
  - All function signatures now accept `Union[str, View]` for backward compatibility
  - All comparisons use enum instances: `view_enum == View.CROSS_SECTIONAL` instead of `view == "CROSS_SECTIONAL"`
  - All JSON serialization uses `view_enum.value` to ensure string output
  - All path construction uses `str(view_enum)` which returns `.value` via `__str__`
- **Backward Compatibility Guaranteed**:
  - `_sanitize_for_json()` explicitly converts Enum types to `.value` for JSON serialization
  - `View.from_string()` normalizes strings from JSON/metadata to enum instances
  - Existing JSON files (metadata.json, snapshot.json, metrics.json) continue to work unchanged
  - File paths remain identical (enum `__str__` returns `.value`)
- **Files Updated**:
  - `feature_selection_reporting.py`, `multi_model_feature_selection.py`, `metrics_aggregator.py`
  - `training.py`, `artifact_paths.py`, `cohort_metadata.py`, `cross_sectional_data.py`
  - `shared_ranking_harness.py`, `output_layout.py`, `diff_telemetry.py`, `target_first_paths.py`
  - `intelligent_trainer.py`, `target_ranker.py`, `feature_selector.py`, `cross_sectional_feature_ranker.py`
  - `model_evaluation.py`, `reproducibility_tracker.py`
- **No Breaking Changes**: All existing snapshots, metrics, and JSON files remain fully compatible

### SST Compliance Fixes
- **Target Name Normalization**: Added `normalize_target_name()` helper function and replaced **ALL** remaining instances (39+ total) of manual target normalization across **ALL** files - ensures consistent filesystem-safe target names across all path construction
- **Path Resolution Consistency**: Replaced **ALL** remaining custom path resolution loops (30+ total) with `run_root()` helper across **ALL** files for consistent run root directory resolution
- **Cross-Sectional Stability SST Parameters**: Added `view` and `symbol` parameters to `compute_cross_sectional_stability()` to use SST-resolved values instead of hardcoded defaults - ensures consistency with main feature selection stage
- **Universe Signature Fix**: Fixed hardcoded `universe_sig="ALL"` in `metrics_aggregator.py` - now extracts from cohort metadata with proper fallback chain
- **Internal Document Cleanup**: Removed all references to internal documentation from public-facing changelogs
- **SST Audit**: Verified RunIdentity construction patterns and TRAINING stage SST usage - all verified as correct
- **Determinism Verification**: All helper replacements verified to produce identical output as manual code, ensuring no non-determinism introduced
- **Complete Migration**: **ALL** remaining SST helper opportunities have been migrated - the codebase now uses SST helpers consistently throughout
- See [detailed changelog](DOCS/02_reference/changelog/2026-01-09-sst-consistency-fixes.md) for full details

## 2026-01-08

### FEATURE_SELECTION Cohort Consolidation
- **Consolidated Duplicate Cohort Directories**: Fixed duplicate cohort directories in FEATURE_SELECTION stage - cross-sectional panel now writes `metrics_cs_panel.json` to the same cohort directory as main feature selection (`metrics.json`), eliminating duplicate directories and making output structure cleaner
- **Cohort ID Passing**: Main feature selection now passes `cohort_id` to cross-sectional panel computation to ensure both use the same cohort directory
- **Bug Fixes**: Fixed null pointer errors when `cohort_dir` or `audit_result` are `None` - added proper null checks before accessing attributes
- **Backward Compatibility**: If `cohort_id` is not provided, CS panel falls back to creating its own cohort (legacy behavior)
- See [detailed changelog](DOCS/02_reference/changelog/2026-01-08-feature-selection-reproducibility-fixes.md) for full details

### FEATURE_SELECTION Reproducibility Fixes
- **CatBoost Missing from Results:** Fixed CatBoost disappearing from results - removed `importance.sum() > 0` filter that excluded failed models, handle empty dicts by creating zero importance Series, ensure failed models appear in aggregation with zero consensus score
- **Training Snapshot Validation:** Added validation that snapshot files actually exist after creation, improved error logging with full traceback at warning level
- **Duplicate Cohort Directories:** Fixed inconsistent `cs_config` structure causing different `config_hash` values - normalize to always include all keys (even if None) for consistent hashing
- **Missing universe_sig:** Fixed duplicate assignment overwriting `universe_sig` in metadata
- **Missing snapshot/diff files:** Added validation after `finalize_run()` to verify required files are created
- **Duplicate universe scopes:** Removed hardcoded `universe_sig="ALL"` default, use SST universe signature consistently, added fallback to extract from `run_identity.dataset_signature`
- **Missing per-model snapshots:** Improved error logging from debug to warning level for per-model snapshot failures
- **Missing deterministic_config_fingerprint:** Fixed path resolution to walk up directory tree to find run root
- **Documentation:** Created comprehensive guide explaining which snapshots exist (`multi_model_aggregated` = source of truth, `cross_sectional_panel` = optional stability analysis) and which one to use
- See [detailed changelog](DOCS/02_reference/changelog/2026-01-08-feature-selection-reproducibility-fixes.md) for full details

### File Overwrite and Plan Creation Fixes
- **run_context.json:** Fixed stage history loss - now preserves `current_stage` and `stage_history` when `save_run_context()` is called after `save_stage_transition()`
- **run_hash.json:** Fixed creation issues - improved error logging, fixed previous run lookup to search parent directories, added validation for missing snapshot indices
- **Routing/Training Plans:** Fixed plan creation - improved error logging (visible warnings instead of debug), added plan save verification, fixed manifest update to occur after plans are created
- See [detailed changelog](DOCS/02_reference/changelog/2026-01-08-file-overwrite-and-plan-creation-fixes.md) for full details

### Commercial License Clarity and Support Documentation
- **README:** Clarified commercial license requirements - now explicitly states "required for proprietary/closed deployments or to avoid AGPL obligations (especially SaaS/network use)"
- **SUPPORT.md:** Added root-level support documentation for easier discovery
- See [detailed changelog](DOCS/02_reference/changelog/2026-01-08-commercial-license-clarity-and-support.md) for full details

> **Note**: This project is under active development. See [NOTICE.md](NOTICE.md) for more information.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**For detailed technical changes:**
- [Changelog Index](DOCS/02_reference/changelog/README.md) – Per-day detailed changelogs with file paths, config keys, and implementation notes.

---

## [Unreleased]

### Recent Highlights (Last 7 Days)

#### 2026-01-09
**Registry Patch System - Automatic Per-Target Feature Exclusion** - Implemented comprehensive registry patch system for automatic, per-target, per-horizon feature exclusion to prevent data leakage. Replaces previous over-aggressive global rejection with granular exclusions. Features include: automatic patch writing during target evaluation, automatic patch loading across all stages (ranking, feature selection, training), auto-fix rerun wrapper with config-driven behavior, patch promotion to persistent storage, unblocking system, and query/explanation system. All patches are policy-only (deterministic), use two-phase eligibility checks (base eligibility → overlays), and maintain SST compliance. Auto-rerun is off by default in experiment config (must explicitly enable). See detailed changelog for complete implementation details.

#### 2026-01-08
**Comprehensive Config Dump** - Added automatic copying of all CONFIG files to `globals/configs/` when runs are created, preserving directory structure. Enables easy run recreation without needing original CONFIG folder access.

**Documentation Enhancements** - Updated README and tutorials to reflect 3-stage pipeline capabilities, dual-view support, and hardware requirements. Added detailed CPU/GPU recommendations for optimal performance and stability.

**File Locking for JSON Writes** - Added file locking around all JSON writes to prevent race conditions. Applied across all pipeline stages.

**Metrics SHA256 Structure Fix** - Fixed metrics digest computation by ensuring metrics are nested under `'metrics'` key in `run_data`.

**Task-Aware Routing Fix** - Fixed routing to use unified `skill01` score normalizing both regression IC and classification AUC-excess to [0,1] range.

**Dual Ranking and Filtering Mismatch Fix** - Fixed filtering mismatch between TARGET_RANKING and FEATURE_SELECTION, added dual ranking with mismatch telemetry.

**Cross-Stage Consistency Fixes** - Fixed Path imports, type casting, and config loading across all stages.

**Manifest and Determinism Fixes** - Fixed manifest schema consistency and deterministic fingerprint computation.

**Config Cleanup** - Removed all symlinks from CONFIG directory, code now uses canonical paths directly.

**Metrics Restructuring** - Grouped metrics structure for cleaner, non-redundant output with task-gating.

**Full Run Hash** - Deterministic run identifier with change detection and aggregation.

**Phase 3.1 Composite Score Fixes** - SE-based stability and skill-gating for family-correct, comparable scoring.

**Snapshot Contract Unification** - P0/P1 correctness fixes for TARGET_RANKING and FEATURE_SELECTION.

> **For detailed technical changes:** See [Changelog Index](DOCS/02_reference/changelog/README.md) for per-day detailed changelogs with file paths, config keys, and implementation notes.

#### 2026-01-07
**Expanded Model Families for TARGET_RANKING/FEATURE_SELECTION** - Full task-type coverage.
- **NEW**: `logistic_regression` family - Standalone classification baseline (binary/multiclass)
- **NEW**: `ftrl_proximal` family - Online learning approximation for binary classification
- **NEW**: `ngboost` family - Probabilistic gradient boosting with uncertainty estimation
- **ENABLED**: `ridge` and `elastic_net` families (were disabled, now enabled for regression)
- All new families:
  - Use `stable_seed_from()` for deterministic seeds (SST pattern)
  - Integrate with existing fingerprinting (prediction, feature, hparams)
  - Automatically populate `fs_snapshot_index.json` via wrapper code
  - Support task-type filtering via `FAMILY_CAPS.supported_tasks`
- **NEW**: Feature selection families added to `FAMILY_CAPS`: `rfe`, `boruta`, `stability_selection`

**Seed Parameter Normalization** - Fixes `unexpected keyword argument 'seed'` errors.
- **FIX**: Convert `seed` → `random_state` for sklearn models in TARGET_RANKING (`model_evaluation.py`)
  - Lasso, Ridge, Elastic Net, Random Forest, Neural Network
- **FIX**: Convert `seed` → `random_state` for sklearn models in leakage detection (`leakage_detection.py`)
  - Lasso, Random Forest, Neural Network
- **FIX**: Add `BASE_SEED` initialization to both files via `init_determinism_from_config()`
- FEATURE_SELECTION already correct (uses `_clean_config_for_estimator` which strips `seed`)

**Task-Type Model Filtering** - Prevents incompatible families from polluting aggregations.
- **NEW**: `supported_tasks` field in `FAMILY_CAPS` for constrained families
  - `elastic_net`, `ridge`, `lasso`: regression only
  - `logistic_regression`: binary, multiclass only
  - `ngboost`: regression, binary only
  - `quantile_lightgbm`: regression only
- **NEW**: `is_family_compatible()` helper in `utils.py` (SST single source of truth)
- **FIX**: Filter applied in all 3 stages before training:
  - Stage 1 (TARGET_RANKING): `model_evaluation.py`
  - Stage 2 (FEATURE_SELECTION): `multi_model_feature_selection.py`
  - Stage 3 (TRAINING): `training.py`
- Tree families (lightgbm, xgboost, catboost) have no restriction - all tasks allowed

**Task-Aware Metrics Schema** - No more `pos_rate: 0.0` on regression targets.
- **NEW**: `CONFIG/ranking/metrics_schema.yaml` with task-specific metric definitions
- **NEW**: `compute_target_stats()` in `metrics_schema.py` (cached schema loader)
- **FIX**: Regression targets emit `y_mean`, `y_std`, `y_min`, `y_max`, `y_finite_pct`
- **FIX**: Binary classification emits `pos_rate` (with configurable `pos_label`)
- **FIX**: Multiclass emits `class_balance` dict, `n_classes` (no `pos_rate`)
- Replaced 2 unconditional `pos_rate` writes in `model_evaluation.py`

**Canonical Metric Naming** - Unambiguous metric names across all stages.
- **NEW**: Naming scheme `<metric_base>__<view>__<aggregation>` (e.g., `spearman_ic__cs__mean`)
- **NEW**: `canonical_names` section in `metrics_schema.yaml` with task+view mappings
- **NEW**: `get_canonical_metric_name(task_type, view)` helper in `metrics_schema.py`
- **NEW**: `get_canonical_metric_names_for_output()` for snapshot metrics population
- **FIX**: `TargetPredictabilityScore` now includes `view` field and `primary_metric_name` property
- **FIX**: All stages emit canonical names alongside deprecated `auc` field for backward compat
  - Regression: `spearman_ic__cs__mean`, `r2__sym__mean`
  - Binary: `roc_auc__cs__mean`, `roc_auc__sym__mean`
  - Multiclass: `accuracy__cs__mean`, `accuracy__sym__mean`
- **DEPRECATED**: `auc` field preserved for backward compatibility (will be removed in v2.0)

**Classification Target Metrics Serialization Fix** - Fixes empty `outputs` for classification targets.
- **FIX**: `class_balance` dict keys now use strings instead of integers
  - PyArrow/Parquet doesn't support integer dict keys, causing silent serialization failures
  - Affected: `compute_target_stats()` in `metrics_schema.py` for binary/multiclass classification
- **FIX**: `_write_metrics()` now writes JSON first, then Parquet
  - JSON is more resilient; ensures metrics.json exists even if Parquet fails
- **NEW**: `_prepare_for_parquet()` helper recursively stringifies nested dict keys
- **FIX**: Shadowed `view` variable bug in `reproducibility_tracker.py` `_save_to_cohort()`
  - Was setting `view = None` then checking `if view:` (always False)
  - Now uses `metrics_view` to avoid shadowing the function parameter

**FEATURE_SELECTION Stability Analysis and Diff Telemetry Fixes**
- **FIX**: `io.py` now skips `manifest.json` when loading snapshots from `replicate/` directories
  - `manifest.json` has different schema (no top-level `run_id`), causing KeyError during stability analysis
  - Added to skip list alongside `fs_snapshot.json` in both `load_snapshots` functions
- **FIX**: `feature_selector.py` now populates `library_versions` in `additional_data`
  - Required for diff telemetry `ComparisonGroup` validation (FEATURE_SELECTION stage)
  - Collects Python version, lightgbm, sklearn, numpy, pandas versions
  - Fixes `ComparisonGroup missing required fields: ['hyperparameters_signature', 'library_versions_signature']` warning
- **FIX**: `get_snapshot_base_dir()` now accepts `ensure_exists` parameter (default True)
  - When False, returns path without creating directories (for read operations)
  - Prevents empty `reproducibility/CROSS_SECTIONAL/feature_importance_snapshots/` directories
  - `metrics_aggregator.py` now passes `ensure_exists=False` when searching for snapshots

**Sample Limit Consistency Across Stages** - Consistent data for TR/FS/TRAINING.
- **FIX**: `cross_sectional_feature_ranker.py` now respects `max_rows_per_symbol`
  - Was loading ALL data (188k samples) instead of config limit (2k per symbol)
- **FIX**: `compute_cross_sectional_importance()` accepts `max_rows_per_symbol` parameter
- **FIX**: `feature_selector.py` passes `max_samples_per_symbol` to CS ranker
- All stages now use consistent `.tail(N)` sampling for reproducibility

**TRAINING Stage Full Parity Tracking** - Complete audit trail for Stage 3.
- **NEW**: `TrainingSnapshot` schema in `TRAINING/training_strategies/reproducibility/schema.py`
  - Model artifact hash (`model_artifact_sha256`) for tamper detection
  - Prediction fingerprint (`predictions_sha256`) for determinism verification
  - Full comparison_group parity with TR/FS stages
- **NEW**: `training_snapshot_index.json` global index for all training runs
- **NEW**: `create_and_save_training_snapshot()` SST-compliant entry point
- **FIX**: Training snapshots created for both CROSS_SECTIONAL and SYMBOL_SPECIFIC models
- End-to-end chain: TR snapshot → FS snapshot → Training snapshot

**FS Snapshot Full Parity with TARGET_RANKING** - Complete audit trail for FEATURE_SELECTION stage.
- **FIX**: Seed derivation now uses `base_seed` (42) directly instead of deriving from `universe_sig`
  - Ensures TR/FS/TRAINING stages have consistent seeds for determinism verification
- **NEW**: `FeatureSelectionSnapshot` now includes full parity fields:
  - `snapshot_seq`: Sequence number for this run
  - `metrics_sha256`: Hash of outputs.metrics for drift detection
  - `artifacts_manifest_sha256`: Hash of output artifacts for tampering detection
  - `fingerprint_sources`: Documentation of what each fingerprint means
  - Full `comparison_group` with `n_effective`, `hyperparameters_signature`, `feature_registry_hash`, `comparable_key`
- **NEW**: Hooks (`save_snapshot_hook`, `save_snapshot_from_series_hook`) accept full parity fields
- **NEW**: `create_fs_snapshot_from_importance` accepts and passes through all parity fields

**OutputLayout & Path Functions Stage Support** - Complete stage-scoped path coverage.
- **NEW**: `OutputLayout` now accepts `stage` parameter and includes `stage=` in `repro_dir()` paths
- **NEW**: `target_repro_dir()` and `target_repro_file_path()` accept `stage` parameter
- **FIX**: All 12 `OutputLayout` callers now pass explicit stage (TARGET_RANKING/FEATURE_SELECTION)
- **FIX**: Dominance quarantine paths use stage-aware paths
- **FIX**: `artifacts_manifest_sha256` now computes correctly (artifacts in expected stage-scoped paths)
- **FIX**: `analyze_all_stability_hook` now uses `iter_stage_dirs()` for proper stage-aware scanning
- **FIX**: Stability metrics now keyed by stage (`TARGET_RANKING/target/method` vs `FEATURE_SELECTION/target/method`)
- **FIX**: `save_snapshot_hook` now passes `stage` to `get_snapshot_base_dir()` (was ignored)
- **FIX**: `feature_selector.py` callers now pass `stage="FEATURE_SELECTION"` explicitly

#### 2026-01-06 (Updated)
**SST Stage Factory & Identity Passthrough** - Stage-aware reproducibility tracking.
- **NEW**: SST stage factory in `run_context.py`: `save_stage_transition()`, `get_current_stage()`, `resolve_stage()`
- **NEW**: Stage-aware reproducibility paths: `stage=TARGET_RANKING/`, `stage=FEATURE_SELECTION/`
- **NEW**: Path scanning helpers for dual-structure support: `iter_stage_dirs()`, `find_cohort_dirs()`, `parse_reproducibility_path()`
- **FIX**: Identity passthrough to `log_run()` in `reproducibility_tracker.py`
- **FIX**: FEATURE_SELECTION identity finalization now logs at WARNING level (was silent DEBUG)
- **FIX**: Partial identity signatures used as fallback when finalization fails
- **FIX**: `fs_snapshot_index.json` fingerprints now populated from FEATURE_SELECTION stage data
- **FIX**: `cross_sectional_panel` snapshots now use partial fallback (was silently failing)
- **FIX**: `multi_model_feature_selection.py` per-family snapshots now use partial fallback
- [Full details →](DOCS/02_reference/changelog/2026-01-06-sst-stage-factory-identity-passthrough.md)

**Comprehensive Determinism Tracking** - Complete end-to-end tracking chain.
- All 8 model families get snapshots (was only XGBoost)
- Training stage now tracks prediction fingerprints
- Feature selection tracks input vs output signatures (`feature_signature_input` / `feature_signature_output`)
- Stage dependencies explicit in snapshots (`selected_targets`, `selected_features`)
- Seeds derived from identity for true determinism
- **FIX**: `allow_legacy=True` now respected for partial RunIdentity (was being ignored)
- **FIX**: Defensive model_metrics handling to ensure fingerprints reach aggregation
- **FIX**: Per-model RunIdentity in TARGET_RANKING prevents replicate folder overwrites
- **FIX**: `predictions_sha256` now populated via `log_run` API path (was only in fallback path)
- [Full details →](DOCS/02_reference/changelog/2026-01-06-determinism-tracking-comprehensive.md)

**View-Scoped Artifact Paths** - Proper separation by view/symbol.
- Artifacts scoped: `targets/<target>/reproducibility/<VIEW>/[symbol=<symbol>/]<artifact_type>/`
- CROSS_SECTIONAL vs SYMBOL_SPECIFIC no longer collide
- Backwards compatible with unscoped paths

**Snapshot Output Fixes** - Critical stage case mismatch resolved.
- Fixed FEATURE_SELECTION snapshots not being written (case mismatch)
- Human-readable manifests for hash-based directories
- Per-model prediction hashes in TARGET_RANKING

#### 2026-01-05
**Determinism and Seed Fixes** - Feature ordering and seed injection.
- Fixed non-deterministic feature ordering (`list(set(...))` → `sorted(set(...))`)
- Automatic seed injection to all model configs
- `feature_signature` added to TARGET_RANKING required fields
- [Full details →](DOCS/02_reference/changelog/2026-01-05-determinism-and-seed-fixes.md)

#### 2026-01-04
**Reproducibility File Output Fixes** - All files now written correctly.
- Fixed `snapshot.json`, `baseline.json`, diff files not being written
- Path reconstruction for target-first structure
- [Full details →](DOCS/02_reference/changelog/2026-01-04-reproducibility-file-output-fixes.md)

**GPU/CPU Determinism Config Fix** - Config settings now respected.
- Replaced hardcoded `set_global_determinism()` with config-aware `init_determinism_from_config()`
- GPU detection respects strict mode
- [Full details →](DOCS/02_reference/changelog/2026-01-04-gpu-cpu-determinism-config-fix.md)

#### 2026-01-03
**Determinism SST** - Production-grade reproducibility.
- `RunIdentity` SST with two-phase construction
- Strict/replicate key separation
- `bin/run_deterministic.sh` launcher
- [Full details →](DOCS/02_reference/changelog/2026-01-03-deterministic-run-identity.md)

---

### Older Updates

See the [Changelog Index](DOCS/02_reference/changelog/README.md) for detailed changelogs organized by date:

- **2026-01-02**: Horizon-aware routing, telemetry comparison fixes
- **2025-12-30**: Prediction hashing for determinism verification
- **2025-12-23**: Dominance quarantine, leakage safety, model timing
- **2025-12-22**: CatBoost/Boruta optimizations, performance audit
- **2025-12-21**: CatBoost fixes, feature selection routing
- **2025-12-20**: Threading utilities, target-first structure
- **2025-12-19**: Target-first migration, config fixes
- **2025-12-18**: TRAINING folder reorganization
- **2025-12-17**: Training pipeline audit, licensing
- **2025-12-16**: Diff telemetry integration
- **2025-12-15**: CatBoost GPU fixes, metrics rename
- **2025-12-14**: Drift tracking, lookahead bias fixes
- **2025-12-13**: SST enforcement, fingerprint tracking
- **2025-12-10–12**: Initial infrastructure setup

---

## Version History

### v0.1.0 (In Development)
- Initial release of FoxML Core
- Multi-model feature selection pipeline
- Target ranking with predictability scoring
- Comprehensive reproducibility tracking system
- Deterministic training with strict mode support
