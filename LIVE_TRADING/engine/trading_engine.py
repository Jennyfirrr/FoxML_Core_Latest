"""
Trading Engine
==============

Main orchestrator for the live trading pipeline.
Coordinates prediction, blending, arbitration, gating, sizing, and risk.

Pipeline Flow:
    Market Data
        ↓
    1. PREDICTION: Multi-horizon predictions
        ↓
    2. BLENDING: Ridge risk-parity per horizon
        ↓
    3. ARBITRATION: Select best horizon
        ↓
    4. GATING: Barrier + spread gates
        ↓
    5. SIZING: Volatility-scaled weights
        ↓
    6. RISK: Final validation
        ↓
    Orders → Broker

SST Compliance:
- Uses get_cfg() for configuration
- Uses sorted_items() for deterministic dict iteration
- Atomic state persistence
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.common.clock import Clock, get_clock
from LIVE_TRADING.common.constants import (
    DECISION_TRADE,
    DECISION_HOLD,
    DECISION_BLOCKED,
    HORIZONS,
    DEFAULT_CONFIG,
)
from LIVE_TRADING.common.types import (
    TradeDecision,
    PipelineTrace,
    MarketSnapshot,
    PositionState,
)
from LIVE_TRADING.brokers.interface import Broker
from LIVE_TRADING.prediction.predictor import MultiHorizonPredictor
from LIVE_TRADING.prediction.cs_ranking_predictor import CrossSectionalRankingPredictor
from LIVE_TRADING.blending.horizon_blender import HorizonBlender
from LIVE_TRADING.arbitration.horizon_arbiter import HorizonArbiter
from LIVE_TRADING.gating.barrier_gate import BarrierGate
from LIVE_TRADING.gating.spread_gate import SpreadGate
from LIVE_TRADING.sizing.position_sizer import PositionSizer
from LIVE_TRADING.risk.guardrails import RiskGuardrails

from .state import EngineState, CycleResult
from .data_provider import DataProvider

# Import event emitters for observability
from LIVE_TRADING.observability.events import (
    emit_trade,
    emit_decision,
    emit_error,
    emit_cycle_start,
    emit_cycle_end,
    emit_stage_change,
)

# CILS: Online learning components
from LIVE_TRADING.learning import (
    Exp3IXBandit,
    RewardTracker,
    EnsembleWeightOptimizer,
    BanditPersistence,
)

logger = logging.getLogger(__name__)

# Trade cooldown to prevent duplicate orders (C4 fix)
DEFAULT_TRADE_COOLDOWN_SECONDS = 5.0


@dataclass
class EngineConfig:
    """Configuration for the trading engine."""

    # State persistence
    state_path: Path = field(default_factory=lambda: Path("state/engine_state.json"))
    history_path: Path = field(default_factory=lambda: Path("state/history.json"))
    save_state: bool = True
    save_history: bool = True

    # Pipeline configuration
    horizons: List[str] = field(default_factory=lambda: HORIZONS.copy())
    families: Optional[List[str]] = None  # None = use all available

    # Trading configuration
    enable_barrier_gate: bool = True
    enable_spread_gate: bool = True
    include_trace: bool = True  # Include full pipeline trace in decisions

    # Target configuration
    default_target: str = "ret_5m"

    # Trade safety (C4: duplicate prevention)
    trade_cooldown_seconds: float = DEFAULT_TRADE_COOLDOWN_SECONDS

    # Position reconciliation (C2: broker sync)
    reconciliation_interval_cycles: int = 100  # Reconcile every N cycles
    reconciliation_mode: str = "warn"  # "strict", "warn", "auto_sync"

    # CILS: Online learning configuration
    enable_online_learning: bool = True
    cils_state_dir: str = "state/cils"
    cils_save_interval: int = 100  # Save CILS state every N updates
    cils_blend_ratio: float = 0.3  # 30% bandit weights, 70% static weights

    # Dynamic horizon discovery
    enable_horizon_discovery: bool = True  # Discover new horizons at runtime
    discovery_interval_cycles: int = 1000  # Check for new horizons every N cycles

    @classmethod
    def from_config(cls) -> "EngineConfig":
        """Load configuration from config files."""
        return cls(
            state_path=Path(get_cfg("live_trading.engine.state_path", default="state/engine_state.json")),
            history_path=Path(get_cfg("live_trading.engine.history_path", default="state/history.json")),
            save_state=get_cfg("live_trading.engine.save_state", default=True),
            save_history=get_cfg("live_trading.engine.save_history", default=True),
            horizons=get_cfg("live_trading.horizons", default=HORIZONS),
            enable_barrier_gate=get_cfg("live_trading.engine.enable_barrier_gate", default=True),
            enable_spread_gate=get_cfg("live_trading.engine.enable_spread_gate", default=True),
            include_trace=get_cfg("live_trading.engine.include_trace", default=True),
            default_target=get_cfg("live_trading.engine.default_target", default="ret_5m"),
            trade_cooldown_seconds=get_cfg("live_trading.risk.trade_cooldown_seconds", default=DEFAULT_TRADE_COOLDOWN_SECONDS),
            reconciliation_interval_cycles=get_cfg("live_trading.reconciliation.periodic_interval", default=100),
            reconciliation_mode=get_cfg("live_trading.reconciliation.mode", default="warn"),
            enable_online_learning=get_cfg("live_trading.online_learning.enabled", default=True),
            cils_state_dir=get_cfg("live_trading.bandit.state_dir", default="state/cils"),
            cils_save_interval=get_cfg("live_trading.bandit.save_interval", default=100),
            cils_blend_ratio=get_cfg("live_trading.online_learning.blend_ratio", default=0.3),
            enable_horizon_discovery=get_cfg("live_trading.online_learning.enable_horizon_discovery", default=True),
            discovery_interval_cycles=get_cfg("live_trading.online_learning.discovery_interval_cycles", default=1000),
        )


class TradingEngine:
    """
    Main trading engine orchestrator.

    Coordinates all pipeline components:
    - Prediction: Multi-horizon model predictions
    - Blending: Ridge risk-parity ensemble per horizon
    - Arbitration: Cost-aware horizon selection
    - Gating: Barrier and spread gates
    - Sizing: Volatility-scaled position sizing
    - Risk: Kill switches and exposure limits

    Example:
        >>> broker = get_broker("paper", initial_cash=100_000)
        >>> data_provider = SimulatedDataProvider()
        >>> engine = TradingEngine(
        ...     broker=broker,
        ...     data_provider=data_provider,
        ...     run_root="/path/to/training/run",
        ... )
        >>> result = engine.run_cycle(["AAPL", "MSFT"])
        >>> print(f"Executed {result.num_trades} trades")
    """

    def __init__(
        self,
        broker: Broker,
        data_provider: DataProvider,
        run_root: str | None = None,
        targets: List[str] | None = None,
        config: EngineConfig | None = None,
        predictor: MultiHorizonPredictor | None = None,
        clock: Clock | None = None,
    ):
        """
        Initialize trading engine.

        Args:
            broker: Broker instance for order execution
            data_provider: Data provider for market data
            run_root: Path to TRAINING run artifacts (required if predictor not provided)
            targets: Target names to trade (default: discover from run)
            config: Engine configuration
            predictor: Optional pre-configured predictor (for testing)
            clock: Clock instance for time (default: system clock)
        """
        self.broker = broker
        self.data_provider = data_provider
        self.config = config or EngineConfig.from_config()
        self._clock = clock or get_clock()

        # Initialize pipeline components
        if predictor is not None:
            self.predictor = predictor
        elif run_root is not None:
            self.predictor = MultiHorizonPredictor(
                run_root=run_root,
                horizons=self.config.horizons,
                families=self.config.families,
            )
        else:
            # No predictor - engine will only work for testing without predictions
            self.predictor = None
            logger.warning("No predictor or run_root provided - engine in limited mode")

        # CS ranking predictor (wraps the regular predictor)
        if self.predictor is not None:
            self.cs_predictor = CrossSectionalRankingPredictor(
                loader=self.predictor.loader,
                engine=self.predictor.engine,
                predictor=self.predictor,
            )
        else:
            self.cs_predictor = None

        self.blender = HorizonBlender()
        self.arbiter = HorizonArbiter()
        self.barrier_gate = BarrierGate() if self.config.enable_barrier_gate else None
        self.spread_gate = SpreadGate() if self.config.enable_spread_gate else None
        self.sizer = PositionSizer()
        self.risk_guardrails = RiskGuardrails(initial_capital=broker.get_cash())

        # Targets to trade
        self.targets = targets or [self.config.default_target]

        # State management
        self._state: Optional[EngineState] = None

        # C4: Track last trade time per symbol to prevent duplicates
        self._last_trade_time: Dict[str, datetime] = {}

        # C2: Track reconciliation
        self._cycles_since_reconciliation: int = 0

        # CILS: Online learning initialization
        self._cils_optimizer: Optional[EnsembleWeightOptimizer] = None
        self._cils_reward_tracker: Optional[RewardTracker] = None
        self._cils_persistence: Optional[BanditPersistence] = None
        self._cils_updates_since_save: int = 0
        self._cycles_since_discovery: int = 0  # Track cycles for horizon discovery

        if self.config.enable_online_learning:
            self._init_cils()

        # Stage tracking for dashboard observability
        self._current_stage: str = "idle"
        self._stage_lock = threading.Lock()

        logger.info(
            f"TradingEngine initialized: targets={self.targets}, "
            f"horizons={self.config.horizons}, "
            f"online_learning={self.config.enable_online_learning}"
        )

    @property
    def state(self) -> EngineState:
        """Get or initialize state."""
        if self._state is None:
            if self.config.state_path.exists():
                try:
                    self._state = EngineState.load(self.config.state_path)
                except Exception as e:
                    logger.warning(f"Failed to load state: {e}, creating new state")
                    self._state = self._create_initial_state()
            else:
                self._state = self._create_initial_state()
        return self._state

    def _create_initial_state(self) -> EngineState:
        """Create initial engine state."""
        cash = self.broker.get_cash()
        return EngineState(
            portfolio_value=cash,
            cash=cash,
        )

    def _set_stage(self, stage: str, symbol: str | None = None) -> None:
        """
        Update current pipeline stage.

        Args:
            stage: Stage name (idle, prediction, blending, arbitration, gating, sizing, risk, execution)
            symbol: Optional symbol being processed
        """
        with self._stage_lock:
            self._current_stage = stage
        # Emit event for dashboard
        emit_stage_change(
            stage=stage,
            symbol=symbol,
            cycle_id=self.state.cycle_count if self._state else None,
        )

    def get_current_stage(self) -> str:
        """
        Get current pipeline stage.

        Returns:
            Current stage name
        """
        with self._stage_lock:
            return self._current_stage

    def _init_cils(self) -> None:
        """
        Initialize CILS (Continuous Integrated Learning System).

        Creates or restores:
        - Exp3-IX bandit for online weight adaptation
        - Reward tracker for P&L feedback
        - Persistence manager for state saving
        """
        # Create persistence manager
        self._cils_persistence = BanditPersistence(
            state_dir=self.config.cils_state_dir,
            clock=self._clock,
        )

        # Try to restore existing state
        restored_optimizer, restored_tracker = self._cils_persistence.load(
            clock=self._clock
        )

        if restored_optimizer is not None:
            self._cils_optimizer = restored_optimizer
            self._cils_reward_tracker = restored_tracker
            logger.info(
                f"CILS restored: {self._cils_optimizer.bandit.total_steps} steps, "
                f"effective_blend={self._cils_optimizer.effective_blend_ratio:.3f}"
            )
        else:
            # Create new CILS components
            self._cils_optimizer = EnsembleWeightOptimizer(
                arm_names=self.config.horizons,
                blend_ratio=self.config.cils_blend_ratio,
            )
            self._cils_reward_tracker = RewardTracker(clock=self._clock)
            logger.info(f"CILS initialized fresh with {len(self.config.horizons)} horizon arms")

    def _cils_record_trade_entry(
        self,
        decision: TradeDecision,
        fill_price: float,
        current_time: datetime,
    ) -> Optional[str]:
        """
        Record trade entry for CILS reward tracking.

        Args:
            decision: Trade decision with horizon info
            fill_price: Execution price
            current_time: Trade timestamp

        Returns:
            Trade ID for exit tracking, or None if CILS disabled
        """
        if not self.config.enable_online_learning or self._cils_reward_tracker is None:
            return None

        if decision.horizon is None:
            return None

        try:
            arm_idx = self._cils_optimizer.get_arm_index(decision.horizon)
        except ValueError:
            logger.warning(f"Unknown horizon {decision.horizon} for CILS")
            return None

        trade_id = self._cils_reward_tracker.record_entry(
            arm=arm_idx,
            arm_name=decision.horizon,
            symbol=decision.symbol,
            side="BUY" if decision.shares > 0 else "SELL",
            entry_price=fill_price,
            qty=abs(decision.shares),
            entry_time=current_time,
        )

        return trade_id

    def _cils_record_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        fees: float = 0.0,
    ) -> None:
        """
        Record trade exit and update CILS bandit weights.

        Args:
            trade_id: Trade ID from entry
            exit_price: Exit price
            fees: Actual fees paid
        """
        if not self.config.enable_online_learning or self._cils_reward_tracker is None:
            return

        try:
            # Get the pending trade to find the arm
            pending = self._cils_reward_tracker.get_pending_by_id(trade_id)
            if pending is None:
                logger.warning(f"Trade {trade_id} not found in CILS tracker")
                return

            arm_name = pending.arm_name

            # Record exit and get net P&L
            net_pnl_bps = self._cils_reward_tracker.record_exit(
                trade_id=trade_id,
                exit_price=exit_price,
                fees=fees,
            )

            # Update bandit weights
            self._cils_optimizer.update_from_pnl(arm_name, net_pnl_bps)

            # Save state periodically
            self._cils_updates_since_save += 1
            if self._cils_updates_since_save >= self.config.cils_save_interval:
                self._save_cils_state()
                self._cils_updates_since_save = 0

            logger.debug(
                f"CILS update: arm={arm_name}, pnl={net_pnl_bps:.2f}bps, "
                f"effective_blend={self._cils_optimizer.effective_blend_ratio:.3f}"
            )

        except Exception as e:
            logger.error(f"CILS exit recording failed: {e}")

    def _save_cils_state(self) -> None:
        """Save CILS state to disk."""
        if self._cils_persistence is None or self._cils_optimizer is None:
            return

        try:
            self._cils_persistence.save(
                self._cils_optimizer,
                self._cils_reward_tracker,
            )
            logger.debug("CILS state saved")
        except Exception as e:
            logger.error(f"Failed to save CILS state: {e}")

    def _cils_get_ensemble_weights(
        self,
        static_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Get blended weights from CILS optimizer.

        Args:
            static_weights: Static weights from ridge regression

        Returns:
            Blended weights (static + bandit)
        """
        if not self.config.enable_online_learning or self._cils_optimizer is None:
            return static_weights

        return self._cils_optimizer.get_ensemble_weights(static_weights)

    def _extract_horizon_from_target(self, target: str) -> Optional[str]:
        """
        Extract horizon suffix from target name.

        Examples:
            "fwd_ret_5m" -> "5m"
            "ret_30m" -> "30m"
            "spread_1h" -> "1h"

        Args:
            target: Target name

        Returns:
            Horizon string or None if not found
        """
        # Match patterns like 5m, 30m, 1h, 2h, 1d at the end of target name
        match = re.search(r'(\d+[mhd])$', target)
        if match:
            return match.group(1)
        return None

    def _discover_new_horizons(self) -> int:
        """
        Discover new horizons from available model targets.

        Scans the model loader for targets, extracts horizons,
        and adds any new ones to the CILS optimizer.

        Returns:
            Number of new horizons discovered
        """
        if not self.config.enable_horizon_discovery:
            return 0

        if not self.config.enable_online_learning or self._cils_optimizer is None:
            return 0

        if self.predictor is None:
            return 0

        # Get available targets from the model loader
        try:
            loader = self.predictor.loader
            available_targets = loader.list_available_targets()
        except Exception as e:
            logger.debug(f"Could not list available targets: {e}")
            return 0

        # Extract horizons and find new ones
        discovered_count = 0
        for target in available_targets:
            horizon = self._extract_horizon_from_target(target)
            if horizon is None:
                continue

            # Check if horizon already exists in CILS
            if not self._cils_optimizer.has_arm(horizon):
                try:
                    new_idx = self._cils_optimizer.add_arm(horizon)
                    discovered_count += 1
                    logger.info(
                        f"CILS discovered new horizon '{horizon}' from target '{target}' "
                        f"(arm index {new_idx})"
                    )

                    # Also add to config horizons list for blender/arbiter
                    if horizon not in self.config.horizons:
                        self.config.horizons.append(horizon)

                except ValueError as e:
                    logger.debug(f"Could not add horizon {horizon}: {e}")

        if discovered_count > 0:
            # Save CILS state after discovery
            self._save_cils_state()
            logger.info(
                f"CILS horizon discovery complete: {discovered_count} new horizons, "
                f"total {self._cils_optimizer.n_arms} arms"
            )

        return discovered_count

    def _check_control_state(self) -> tuple[bool, Optional[str]]:
        """
        Check control state file for pause/kill switch.
        
        Returns:
            (is_paused, kill_switch_reason)
        """
        control_file = Path(os.getenv("FOXML_STATE_DIR", "state")) / "control_state.json"
        if not control_file.exists():
            return (False, None)
        
        try:
            with open(control_file, "r") as f:
                state = json.load(f)
                paused = state.get("paused", False)
                kill_switch = state.get("kill_switch_active", False)
                reason = state.get("kill_switch_reason") if kill_switch else None
                return (paused, reason)
        except Exception as e:
            logger.warning(f"Error reading control state: {e}")
            return (False, None)

    def run_cycle(
        self,
        symbols: List[str],
        current_time: datetime | None = None,
    ) -> CycleResult:
        """
        Run one trading cycle.

        Processes all symbols through the pipeline:
        1. Check control state (pause/kill switch)
        2. Check risk limits
        3. For each symbol: predict → blend → arbitrate → gate → size → risk → execute

        Args:
            symbols: Symbols to process
            current_time: Cycle timestamp (default: now)

        Returns:
            CycleResult with all decisions
        """
        current_time = current_time or self._clock.now()
        cycle_start_time = current_time

        # Reset stage at cycle start
        self._set_stage("idle")

        # Check control state (pause/kill switch)
        is_paused, kill_switch_reason = self._check_control_state()
        if is_paused:
            logger.info("Trading engine paused - skipping cycle")
            # Return empty cycle result
            return CycleResult(
                cycle_number=self.state.cycle_count,
                timestamp=current_time,
                decisions=[],
                portfolio_value=self.state.portfolio_value,
                cash=self.state.cash,
                is_trading_allowed=False,
                kill_switch_reason="paused",
            )

        # Emit cycle start event
        emit_cycle_start(self.state.cycle_count + 1, symbols)

        # Update state
        self.state.increment_cycle(current_time)
        self.state.update_daily_tracking(current_time)

        # C2: Periodic position reconciliation
        self._cycles_since_reconciliation += 1
        if self._cycles_since_reconciliation >= self.config.reconciliation_interval_cycles:
            self._reconcile_positions()
            self._cycles_since_reconciliation = 0

        # CILS: Periodic horizon discovery
        self._cycles_since_discovery += 1
        if self._cycles_since_discovery >= self.config.discovery_interval_cycles:
            self._discover_new_horizons()
            self._cycles_since_discovery = 0

        decisions = []
        trades_count = 0

        # Calculate current portfolio value
        position_value = self._get_position_value()
        portfolio_value = self.state.cash + position_value
        self.state.portfolio_value = portfolio_value

        # Check manual kill switch from control state
        _, manual_kill_switch_reason = self._check_control_state()
        if manual_kill_switch_reason:
            logger.warning(f"Manual kill switch active: {manual_kill_switch_reason}")
            for symbol in symbols:
                decisions.append(self._create_blocked_decision(
                    symbol,
                    manual_kill_switch_reason,
                    current_time,
                ))

            return CycleResult(
                cycle_number=self.state.cycle_count,
                timestamp=current_time,
                decisions=decisions,
                portfolio_value=portfolio_value,
                cash=self.state.cash,
                is_trading_allowed=False,
                kill_switch_reason=manual_kill_switch_reason,
            )

        # Check if trading is allowed (risk guardrails)
        risk_status = self.risk_guardrails.check_trading_allowed(
            portfolio_value,
            self.state.get_current_weights(),
            current_time,
        )

        if not risk_status.is_trading_allowed:
            logger.warning(f"Trading blocked: {risk_status.kill_switch_reason}")
            for symbol in symbols:
                decisions.append(self._create_blocked_decision(
                    symbol,
                    risk_status.kill_switch_reason or "kill_switch",
                    current_time,
                ))

            return CycleResult(
                cycle_number=self.state.cycle_count,
                timestamp=current_time,
                decisions=decisions,
                portfolio_value=portfolio_value,
                cash=self.state.cash,
                is_trading_allowed=False,
                kill_switch_reason=risk_status.kill_switch_reason,
            )

        # CS ranking pre-prediction: collect and rank all symbols cross-sectionally
        cs_ranked_preds: Dict[str, Any] = {}
        cs_exclude: set | None = None  # Families to exclude from pointwise prediction
        if self.cs_predictor and self.predictor:
            target = self.targets[0] if self.targets else self.config.default_target
            cs_families = self.cs_predictor.get_cs_families(target)
            if cs_families:
                cs_exclude = set(cs_families)
                self._set_stage("prediction")
                universe: Dict[str, Any] = {}
                adv_map: Dict[str, float] = {}
                for sym in symbols:
                    try:
                        universe[sym] = self.data_provider.get_historical(sym, period="1mo")
                        adv_map[sym] = self.data_provider.get_adv(sym)
                    except Exception as e:
                        logger.debug(f"CS ranking: no data for {sym}: {e}")
                try:
                    cs_ranked_preds = self.cs_predictor.predict(
                        target=target,
                        universe=universe,
                        horizons=self.config.horizons,
                        data_timestamp=current_time,
                        adv_map=adv_map,
                    )
                    if cs_ranked_preds:
                        logger.info(
                            f"CS ranking: ranked {len(cs_ranked_preds)} symbols "
                            f"across {len(cs_families)} families"
                        )
                except Exception as e:
                    logger.warning(f"CS ranking pre-prediction failed: {e}")

        # Process each symbol
        for symbol in symbols:
            try:
                decision = self._process_symbol(
                    symbol, current_time,
                    cs_ranked_preds.get(symbol), cs_exclude,
                )
                decisions.append(decision)

                # Execute if TRADE
                if decision.decision == DECISION_TRADE and decision.shares != 0:
                    if self._execute_trade(decision, current_time):
                        trades_count += 1

                # Record decision
                self.state.record_decision(decision)

                # Emit decision event for observability
                emit_decision(
                    symbol=decision.symbol,
                    decision=decision.decision,
                    alpha=decision.alpha,
                    horizon=decision.horizon,
                    shares=decision.shares,
                    target_weight=decision.target_weight,
                    reason=decision.reason,
                )

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                emit_error(f"Error processing {symbol}", exception=e)
                decisions.append(self._create_hold_decision(
                    symbol,
                    f"error: {e}",
                    current_time,
                ))

        # Save state
        if self.config.save_state:
            self.state.save(self.config.state_path)

        if self.config.save_history:
            self.state.save_history(self.config.history_path)

        # Reset stage at cycle end
        self._set_stage("idle")

        # Emit cycle end event
        cycle_duration = (self._clock.now() - cycle_start_time).total_seconds()
        emit_cycle_end(
            cycle_id=self.state.cycle_count,
            duration_seconds=cycle_duration,
            decisions_count=len(decisions),
            trades_count=trades_count,
        )

        return CycleResult(
            cycle_number=self.state.cycle_count,
            timestamp=current_time,
            decisions=decisions,
            portfolio_value=portfolio_value,
            cash=self.state.cash,
            is_trading_allowed=True,
        )

    def _process_symbol(
        self,
        symbol: str,
        current_time: datetime,
        cs_predictions: Optional[Any] = None,
        cs_exclude: set | None = None,
    ) -> TradeDecision:
        """
        Process a single symbol through the pipeline.

        Args:
            symbol: Trading symbol
            current_time: Current timestamp
            cs_predictions: Pre-computed CS ranking AllPredictions (optional)
            cs_exclude: CS family names to exclude from pointwise prediction
                (these are handled by CrossSectionalRankingPredictor instead)

        Returns:
            TradeDecision
        """
        current_weight = self.state.get_current_weights().get(symbol, 0.0)

        # Initialize trace if enabled
        trace = None
        if self.config.include_trace:
            trace = PipelineTrace(
                market_snapshot=MarketSnapshot(
                    symbol=symbol,
                    timestamp=current_time,
                    open=0.0, high=0.0, low=0.0, close=0.0,
                    volume=0.0, bid=0.0, ask=0.0,
                    spread_bps=0.0, volatility=0.0,
                )
            )

        # 1. Get market data
        try:
            quote = self.data_provider.get_quote(symbol)
            prices = self.data_provider.get_historical(symbol, period="1mo")
            adv = self.data_provider.get_adv(symbol)
        except Exception as e:
            logger.warning(f"Failed to get market data for {symbol}: {e}")
            return self._create_hold_decision(symbol, f"no_data: {e}", current_time)

        spread_bps = quote.spread_bps
        mid_price = quote.mid
        volatility = self._calculate_volatility(prices)

        # Update trace with market data
        if trace:
            trace.market_snapshot = MarketSnapshot(
                symbol=symbol,
                timestamp=current_time,
                open=float(prices["Open"].iloc[-1]) if "Open" in prices else mid_price,
                high=float(prices["High"].iloc[-1]) if "High" in prices else mid_price,
                low=float(prices["Low"].iloc[-1]) if "Low" in prices else mid_price,
                close=float(prices["Close"].iloc[-1]) if "Close" in prices else mid_price,
                volume=float(prices["Volume"].iloc[-1]) if "Volume" in prices else 0.0,
                bid=quote.bid,
                ask=quote.ask,
                spread_bps=spread_bps,
                volatility=volatility,
            )

        # 2. Spread gate
        if self.spread_gate:
            spread_result = self.spread_gate.evaluate(
                spread_bps=spread_bps,
                quote_timestamp=quote.timestamp,
            )
            if trace:
                trace.spread_gate_result = spread_result.to_dict()

            if not spread_result.allowed:
                return TradeDecision(
                    symbol=symbol,
                    decision=DECISION_HOLD,
                    horizon=None,
                    target_weight=current_weight,
                    current_weight=current_weight,
                    alpha=0.0,
                    shares=0,
                    reason=spread_result.reason,
                    timestamp=current_time,
                    trace=trace,
                )

        # 3. Multi-horizon prediction
        self._set_stage("prediction", symbol)
        if self.predictor is None:
            return self._create_hold_decision(symbol, "no_predictor", current_time, trace)

        target = self.targets[0] if self.targets else self.config.default_target

        try:
            all_preds = self.predictor.predict_all_horizons(
                target=target,
                prices=prices,
                symbol=symbol,
                data_timestamp=current_time,
                adv=adv,
                exclude_families=cs_exclude,
            )
        except Exception as e:
            logger.warning(f"Prediction failed for {symbol}: {e}")
            return self._create_hold_decision(symbol, f"prediction_error: {e}", current_time, trace)

        # Merge CS ranking predictions into pointwise predictions
        if cs_predictions is not None:
            for horizon, cs_hp in sorted_items(cs_predictions.horizons):
                if horizon in all_preds.horizons:
                    # Add CS families alongside pointwise families
                    all_preds.horizons[horizon].predictions.update(cs_hp.predictions)
                else:
                    all_preds.horizons[horizon] = cs_hp

        # Update trace with predictions
        if trace:
            for horizon, hp in sorted_items(all_preds.horizons):
                trace.raw_predictions[horizon] = hp.get_raw_dict()
                trace.standardized_predictions[horizon] = hp.get_standardized_dict()
                trace.confidences[horizon] = {
                    f: p.confidence.overall for f, p in sorted_items(hp.predictions)
                }

        # 4. Blend per horizon
        self._set_stage("blending", symbol)
        blended = self.blender.blend_all_horizons(
            {h: all_preds.horizons[h] for h in all_preds.available_horizons}
        )

        if trace:
            for horizon, ba in sorted_items(blended):
                trace.blended_alphas[horizon] = ba.alpha
                trace.blend_weights[horizon] = ba.weights

        # 5. Arbitrate
        self._set_stage("arbitration", symbol)
        arb_result = self.arbiter.arbitrate(
            blended_alphas=blended,
            spread_bps=spread_bps,
            volatility=volatility,
            adv=adv,
        )

        if trace:
            trace.horizon_scores = arb_result.horizon_scores
            trace.selected_horizon = arb_result.selected_horizon
            if arb_result.costs:
                trace.cost_breakdown[arb_result.selected_horizon or "none"] = {
                    "impact": arb_result.costs.impact_cost,
                    "spread": arb_result.costs.spread_cost,
                    "timing": arb_result.costs.timing_cost,
                    "total": arb_result.costs.total_cost,
                }

        if arb_result.decision != DECISION_TRADE:
            return TradeDecision(
                symbol=symbol,
                decision=DECISION_HOLD,
                horizon=arb_result.selected_horizon,
                target_weight=current_weight,
                current_weight=current_weight,
                alpha=arb_result.alpha,
                shares=0,
                reason=arb_result.reason,
                timestamp=current_time,
                trace=trace,
            )

        # 6. Barrier gate (if enabled)
        self._set_stage("gating", symbol)
        gate_result = None
        if self.barrier_gate:
            # C1 FIX: Get actual barrier model predictions
            p_peak, p_valley = self._get_barrier_predictions(symbol, prices, current_time)

            if arb_result.alpha > 0:
                gate_result = self.barrier_gate.evaluate_long_entry(p_peak=p_peak, p_valley=p_valley)
            else:
                gate_result = self.barrier_gate.evaluate_short_entry(p_peak=p_valley, p_valley=p_peak)

            if trace:
                trace.barrier_gate_result = gate_result.to_dict()

            if not gate_result.allowed:
                return TradeDecision(
                    symbol=symbol,
                    decision=DECISION_BLOCKED,
                    horizon=arb_result.selected_horizon,
                    target_weight=current_weight,
                    current_weight=current_weight,
                    alpha=arb_result.alpha,
                    shares=0,
                    reason=gate_result.reason,
                    timestamp=current_time,
                    trace=trace,
                )

        # 7. Position sizing
        self._set_stage("sizing", symbol)
        sizing_result = self.sizer.size_single(
            symbol=symbol,
            alpha=arb_result.alpha,
            volatility=volatility,
            current_weight=current_weight,
            price=mid_price,
            portfolio_value=self.state.portfolio_value,
            gate_result=gate_result,
        )

        if trace:
            trace.raw_weight = self.sizer.calculate_target_weight(
                arb_result.alpha, volatility, None
            )
            trace.gate_adjusted_weight = sizing_result.target_weight
            trace.final_weight = sizing_result.target_weight

        # Check if trade is within no-trade band
        if sizing_result.reason == "no_trade_band":
            return TradeDecision(
                symbol=symbol,
                decision=DECISION_HOLD,
                horizon=arb_result.selected_horizon,
                target_weight=sizing_result.target_weight,
                current_weight=current_weight,
                alpha=arb_result.alpha,
                shares=0,
                reason="no_trade_band",
                timestamp=current_time,
                trace=trace,
            )

        # 8. Risk validation
        self._set_stage("risk", symbol)
        risk_check = self.risk_guardrails.validate_trade(
            symbol,
            sizing_result.target_weight,
            self.state.get_current_weights(),
        )

        if trace:
            trace.risk_checks = {"passed": risk_check.passed, "reason": risk_check.message}

        if not risk_check.passed:
            return TradeDecision(
                symbol=symbol,
                decision=DECISION_BLOCKED,
                horizon=arb_result.selected_horizon,
                target_weight=sizing_result.target_weight,
                current_weight=current_weight,
                alpha=arb_result.alpha,
                shares=0,
                reason=risk_check.message,
                timestamp=current_time,
                trace=trace,
            )

        return TradeDecision(
            symbol=symbol,
            decision=DECISION_TRADE,
            horizon=arb_result.selected_horizon,
            target_weight=sizing_result.target_weight,
            current_weight=current_weight,
            alpha=arb_result.alpha,
            shares=sizing_result.shares,
            reason=arb_result.reason,
            timestamp=current_time,
            trace=trace,
        )

    def _can_trade_symbol(self, symbol: str, current_time: datetime) -> bool:
        """
        C4 FIX: Check if symbol is eligible for trading (not in cooldown).

        Args:
            symbol: Trading symbol
            current_time: Current timestamp

        Returns:
            True if symbol can be traded
        """
        last_trade = self._last_trade_time.get(symbol)
        if last_trade is None:
            return True

        elapsed = (current_time - last_trade).total_seconds()
        return elapsed >= self.config.trade_cooldown_seconds

    def _execute_trade(
        self,
        decision: TradeDecision,
        current_time: datetime,
    ) -> bool:
        """
        Execute a trade decision.

        Returns:
            True if trade was executed successfully, False otherwise
        """
        # Set stage for dashboard
        self._set_stage("execution", decision.symbol)

        # C4 FIX: Check cooldown to prevent duplicate orders
        if not self._can_trade_symbol(decision.symbol, current_time):
            logger.warning(
                f"Trade blocked for {decision.symbol}: in cooldown period "
                f"({self.config.trade_cooldown_seconds}s)"
            )
            emit_error(
                "trade_cooldown",
                f"Symbol {decision.symbol} in cooldown, trade skipped"
            )
            return False

        side = "BUY" if decision.shares > 0 else "SELL"
        qty = abs(decision.shares)

        try:
            result = self.broker.submit_order(
                symbol=decision.symbol,
                side=side,
                qty=qty,
            )

            # C3 FIX: Verify fill status before updating state
            fill_status = result.get("status", "unknown")
            if fill_status not in ("filled", "partial_fill"):
                logger.error(
                    f"Order not filled: {decision.symbol} status={fill_status}, "
                    f"order_id={result.get('order_id')}"
                )
                emit_error(
                    "order_not_filled",
                    f"Order {result.get('order_id')} for {decision.symbol} "
                    f"status: {fill_status}"
                )
                return False

            logger.info(f"Trade executed: {side} {qty} {decision.symbol}")

            # Update last trade time (C4)
            self._last_trade_time[decision.symbol] = current_time

            # Update state with verified fill
            fill_price = result.get("fill_price", 0.0)
            filled_qty = result.get("filled_qty", qty)  # Use actual filled quantity

            # Get current position shares
            current_pos = self.state.get_position(decision.symbol)
            current_shares = current_pos.shares if current_pos else 0.0

            # Use signed filled quantity for position update
            signed_filled = filled_qty if side == "BUY" else -filled_qty
            new_shares = current_shares + signed_filled

            if abs(new_shares) < 1e-8:
                # Position closed
                self.state.remove_position(decision.symbol)
            else:
                self.state.update_position(
                    symbol=decision.symbol,
                    weight=decision.target_weight,
                    shares=new_shares,
                    price=fill_price,
                    timestamp=current_time,
                )

            # Update cash using actual filled quantity
            trade_value = filled_qty * fill_price
            if side == "BUY":
                self.state.cash -= trade_value
            else:
                self.state.cash += trade_value

            # Record trade with actual filled quantity
            self.state.record_trade(
                symbol=decision.symbol,
                side=side,
                shares=filled_qty,
                price=fill_price,
                horizon=decision.horizon,
                order_id=result.get("order_id"),
            )

            # CILS: Record trade entry for P&L tracking
            cils_trade_id = self._cils_record_trade_entry(
                decision=decision,
                fill_price=fill_price,
                current_time=current_time,
            )
            if cils_trade_id:
                # Store trade_id in decision metadata for exit tracking
                if not hasattr(decision, 'metadata') or decision.metadata is None:
                    decision.metadata = {}
                decision.metadata["cils_trade_id"] = cils_trade_id

            # Emit trade event for observability
            emit_trade(
                symbol=decision.symbol,
                side=side,
                qty=filled_qty,
                price=fill_price,
                order_id=result.get("order_id"),
                horizon=decision.horizon,
                alpha=decision.alpha,
            )

            return True

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            emit_error(f"Trade execution failed for {decision.symbol}", exception=e)
            return False

    def _get_position_value(self) -> float:
        """Get total position value."""
        value = 0.0
        for symbol, pos in sorted_items(self.state.positions):
            try:
                quote = self.data_provider.get_quote(symbol)
                value += pos.shares * quote.mid
            except Exception:
                # Use last known price
                value += pos.shares * pos.current_price
        return value

    def _calculate_volatility(self, prices: pd.DataFrame, window: int = 20) -> float:
        """Calculate annualized volatility."""
        if len(prices) < 2:
            return 0.02  # Default 2% daily vol

        if "Close" not in prices.columns:
            return 0.02

        returns = prices["Close"].pct_change().dropna()
        if len(returns) < window:
            window = len(returns)

        daily_vol = returns.tail(window).std()
        return float(daily_vol * np.sqrt(252))  # Annualize

    def _reconcile_positions(self) -> None:
        """
        C2 FIX: Reconcile internal positions with broker reality.

        Detects drift between internal state and actual broker positions.
        Behavior depends on reconciliation_mode:
        - "strict": Raise error on mismatch
        - "warn": Log warning but continue
        - "auto_sync": Automatically sync to broker state
        """
        try:
            broker_positions = self.broker.get_positions()
        except Exception as e:
            logger.error(f"Failed to get broker positions for reconciliation: {e}")
            return

        qty_tolerance = get_cfg("live_trading.reconciliation.qty_tolerance", default=0.01)

        # Check each broker position
        for symbol, broker_qty in sorted_items(broker_positions):
            internal_pos = self.state.positions.get(symbol)
            internal_qty = internal_pos.shares if internal_pos else 0.0

            if abs(broker_qty - internal_qty) > qty_tolerance:
                msg = (
                    f"Position drift detected: {symbol} "
                    f"internal={internal_qty:.4f} broker={broker_qty:.4f}"
                )

                if self.config.reconciliation_mode == "strict":
                    logger.error(msg)
                    emit_error("position_drift", msg)
                    raise RuntimeError(msg)
                elif self.config.reconciliation_mode == "auto_sync":
                    logger.warning(f"{msg} - auto-syncing to broker")
                    if abs(broker_qty) < 1e-8:
                        self.state.remove_position(symbol)
                    else:
                        # Get current price for update
                        try:
                            quote = self.data_provider.get_quote(symbol)
                            price = quote.mid
                        except Exception:
                            price = internal_pos.current_price if internal_pos else 0.0

                        self.state.update_position(
                            symbol=symbol,
                            weight=0.0,  # Will be recalculated
                            shares=broker_qty,
                            price=price,
                            timestamp=self._clock.now(),
                        )
                else:  # "warn"
                    logger.warning(msg)

        # Check for internal positions not in broker
        for symbol, internal_pos in sorted_items(self.state.positions):
            if symbol not in broker_positions:
                msg = f"Orphaned internal position: {symbol} qty={internal_pos.shares:.4f}"

                if self.config.reconciliation_mode == "strict":
                    logger.error(msg)
                    emit_error("orphaned_position", msg)
                    raise RuntimeError(msg)
                elif self.config.reconciliation_mode == "auto_sync":
                    logger.warning(f"{msg} - removing from internal state")
                    self.state.remove_position(symbol)
                else:  # "warn"
                    logger.warning(msg)

        logger.debug("Position reconciliation completed")

    def _get_barrier_predictions(
        self,
        symbol: str,
        prices: pd.DataFrame,
        current_time: datetime,
    ) -> tuple[float, float]:
        """
        C1 FIX: Get barrier model predictions (p_peak, p_valley).

        Attempts to get predictions from barrier models. Falls back to
        neutral values if models not available.

        Args:
            symbol: Trading symbol
            prices: Price DataFrame
            current_time: Current timestamp

        Returns:
            Tuple of (p_peak, p_valley) probabilities
        """
        # Default neutral values
        default_p_peak = 0.3
        default_p_valley = 0.3

        if self.predictor is None:
            return default_p_peak, default_p_valley

        try:
            # Try to get barrier model predictions
            # Barrier targets are typically "will_peak_5m" and "will_valley_5m"
            barrier_targets = get_cfg(
                "live_trading.barrier_gate.targets",
                default={"peak": "will_peak_5m", "valley": "will_valley_5m"}
            )

            p_peak = default_p_peak
            p_valley = default_p_valley

            # Try peak prediction
            try:
                peak_preds = self.predictor.predict_single_target(
                    target=barrier_targets.get("peak", "will_peak_5m"),
                    prices=prices,
                    symbol=symbol,
                    data_timestamp=current_time,
                )
                if peak_preds and hasattr(peak_preds, 'alpha'):
                    # Convert alpha to probability (sigmoid-like)
                    p_peak = 1.0 / (1.0 + np.exp(-peak_preds.alpha * 5))
            except Exception as e:
                logger.debug(f"No peak barrier prediction for {symbol}: {e}")

            # Try valley prediction
            try:
                valley_preds = self.predictor.predict_single_target(
                    target=barrier_targets.get("valley", "will_valley_5m"),
                    prices=prices,
                    symbol=symbol,
                    data_timestamp=current_time,
                )
                if valley_preds and hasattr(valley_preds, 'alpha'):
                    p_valley = 1.0 / (1.0 + np.exp(-valley_preds.alpha * 5))
            except Exception as e:
                logger.debug(f"No valley barrier prediction for {symbol}: {e}")

            return p_peak, p_valley

        except Exception as e:
            logger.debug(f"Barrier prediction failed for {symbol}: {e}")
            return default_p_peak, default_p_valley

    def _create_blocked_decision(
        self,
        symbol: str,
        reason: str,
        timestamp: datetime,
        trace: Optional[PipelineTrace] = None,
    ) -> TradeDecision:
        """Create a BLOCKED decision."""
        return TradeDecision(
            symbol=symbol,
            decision=DECISION_BLOCKED,
            horizon=None,
            target_weight=0.0,
            current_weight=self.state.get_current_weights().get(symbol, 0.0),
            alpha=0.0,
            shares=0,
            reason=reason,
            timestamp=timestamp,
            trace=trace,
        )

    def _create_hold_decision(
        self,
        symbol: str,
        reason: str,
        timestamp: datetime,
        trace: Optional[PipelineTrace] = None,
    ) -> TradeDecision:
        """Create a HOLD decision."""
        current_weight = self.state.get_current_weights().get(symbol, 0.0)
        return TradeDecision(
            symbol=symbol,
            decision=DECISION_HOLD,
            horizon=None,
            target_weight=current_weight,
            current_weight=current_weight,
            alpha=0.0,
            shares=0,
            reason=reason,
            timestamp=timestamp,
            trace=trace,
        )

    def reset(self, initial_value: float | None = None) -> None:
        """
        Reset engine state.

        Args:
            initial_value: New initial capital (default: broker cash)
        """
        value = initial_value or self.broker.get_cash()
        self._state = EngineState(
            portfolio_value=value,
            cash=value,
        )
        self.risk_guardrails.reset(value)
        self.blender.reset()
        if self.predictor:
            self.predictor.reset()

        # C4: Clear trade cooldowns
        self._last_trade_time.clear()
        # C2: Reset reconciliation counter
        self._cycles_since_reconciliation = 0

        # CILS: Reset (optionally re-initialize)
        if self.config.enable_online_learning and self._cils_optimizer is not None:
            self._cils_optimizer.reset()
            if self._cils_reward_tracker is not None:
                self._cils_reward_tracker.reset()
            self._cils_updates_since_save = 0
            logger.info("CILS reset")

        logger.info(f"TradingEngine reset: initial_value=${value:,.2f}")

    def get_state_summary(self) -> Dict[str, Any]:
        """Get state summary for monitoring."""
        summary = {
            "cash": self.state.cash,
            "cycle_count": self.state.cycle_count,
            "daily_pnl": self.state.daily_pnl,
            "kill_switch_active": self.risk_guardrails.is_kill_switch_active,
            "last_update": self.state.last_update.isoformat() if self.state.last_update else None,
            "num_positions": len(self.state.positions),
            "portfolio_value": self.state.portfolio_value,
            "position_value": self._get_position_value(),
        }

        # Add CILS stats if enabled
        if self.config.enable_online_learning and self._cils_optimizer is not None:
            summary["cils"] = {
                "enabled": True,
                "total_steps": self._cils_optimizer.bandit.total_steps,
                "effective_blend_ratio": self._cils_optimizer.effective_blend_ratio,
                "bandit_weights": self._cils_optimizer.get_bandit_weights(),
                "pending_trades": (
                    self._cils_reward_tracker.pending_count
                    if self._cils_reward_tracker else 0
                ),
            }
        else:
            summary["cils"] = {"enabled": False}

        return summary

    def get_cils_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get detailed CILS statistics.

        Returns:
            Dict with CILS stats or None if disabled
        """
        if not self.config.enable_online_learning or self._cils_optimizer is None:
            return None

        stats = self._cils_optimizer.get_stats()

        if self._cils_reward_tracker is not None:
            stats["reward_tracker"] = {
                "pending_count": self._cils_reward_tracker.pending_count,
                "completed_count": self._cils_reward_tracker.completed_count,
            }
            # Add per-arm stats
            for i, arm_name in enumerate(self._cils_optimizer.arm_names):
                arm_stats = self._cils_reward_tracker.get_arm_stats(i)
                stats["bandit_stats"]["arm_stats"][i]["reward_stats"] = arm_stats

        return stats

    def record_position_exit(
        self,
        symbol: str,
        exit_price: float,
        fees: float = 0.0,
    ) -> None:
        """
        Record a position exit for CILS reward calculation.

        Call this when a position is closed (either by hitting horizon
        or manual exit) to provide P&L feedback to the bandit.

        Args:
            symbol: Trading symbol
            exit_price: Exit price
            fees: Actual fees paid
        """
        if not self.config.enable_online_learning or self._cils_reward_tracker is None:
            return

        # Find pending trades for this symbol
        pending_trades = self._cils_reward_tracker.get_pending_for_symbol(symbol)

        for trade in pending_trades:
            self._cils_record_trade_exit(
                trade_id=trade.trade_id,
                exit_price=exit_price,
                fees=fees,
            )
            logger.info(
                f"CILS recorded exit for {symbol}: trade_id={trade.trade_id}"
            )
