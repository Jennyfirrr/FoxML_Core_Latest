#!/usr/bin/env python3
"""
IPC Bridge Server
=================

FastAPI server exposing EventBus and Metrics via WebSocket/HTTP for Rust TUI dashboard.

This is a lightweight bridge - no changes to trading engine code needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging early to suppress uvicorn access logs
# These logs interfere with the TUI when printed to stdout/stderr
logging.basicConfig(
    level=logging.WARNING,  # Default to WARNING to suppress INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress uvicorn access logs (they interfere with TUI)
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.setLevel(logging.WARNING)

uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Try to import observability (may not be available if trading engine not running)
try:
    from LIVE_TRADING.observability import events, metrics
    from LIVE_TRADING.observability.events import Event, EventType
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

# Import Alpaca stream client
try:
    from alpaca_stream import (
        AlpacaEvent,
        AlpacaEventType,
        AlpacaStreamClient,
        get_alpaca_client,
        start_alpaca_stream,
        stop_alpaca_stream,
    )
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca stream module not available")

# App will be created with lifespan context manager

# WebSocket connections
active_connections: List[WebSocket] = []

# Event queue for WebSocket streaming (internal events)
event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)

# Alpaca event queue (separate for Alpaca-specific events)
alpaca_event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)

# Control state file path
CONTROL_STATE_FILE = Path(os.getenv("FOXML_STATE_DIR", "state")) / "control_state.json"


def on_event(event: Event) -> None:
    """Callback for EventBus - forwards events to WebSocket queue."""
    if OBSERVABILITY_AVAILABLE:
        try:
            event_dict = event.to_dict()
            # Put event in queue (non-blocking)
            try:
                event_queue.put_nowait(event_dict)
            except asyncio.QueueFull:
                logger.warning("Event queue full, dropping event")
        except Exception as e:
            logger.error(f"Error processing event: {e}")


def on_alpaca_event(event: "AlpacaEvent") -> None:
    """Callback for Alpaca WebSocket events."""
    try:
        event_dict = event.to_dict()
        # Put event in both queues (for WebSocket and polling)
        try:
            alpaca_event_queue.put_nowait(event_dict)
            # Also put in main event queue for unified streaming
            event_queue.put_nowait(event_dict)
        except asyncio.QueueFull:
            logger.warning("Alpaca event queue full, dropping event")
    except Exception as e:
        logger.error(f"Error processing Alpaca event: {e}")


def _read_control_state() -> Dict[str, Any]:
    """Read control state from file."""
    if CONTROL_STATE_FILE.exists():
        try:
            with open(CONTROL_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading control state: {e}")
    
    # Default state
    return {
        "paused": False,
        "kill_switch_active": False,
        "kill_switch_reason": None,
        "last_updated": None,
    }


def _write_control_state(state: Dict[str, Any]) -> None:
    """Write control state to file (atomic write with temp+rename)."""
    try:
        CONTROL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        tmp = CONTROL_STATE_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        tmp.rename(CONTROL_STATE_FILE)
    except Exception as e:
        logger.error(f"Error writing control state: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("IPC Bridge starting up...")

    if OBSERVABILITY_AVAILABLE:
        # Subscribe to all events
        events.subscribe(None, on_event)
        logger.info("Subscribed to EventBus")
    else:
        logger.warning("Running in mock mode - EventBus not available")

    # Start Alpaca WebSocket stream
    if ALPACA_AVAILABLE:
        try:
            await start_alpaca_stream(on_event=on_alpaca_event)
            logger.info("Alpaca WebSocket stream started")
        except Exception as e:
            logger.error(f"Failed to start Alpaca stream: {e}")
    else:
        logger.warning("Alpaca stream not available")

    yield

    # Shutdown
    logger.info("IPC Bridge shutting down...")

    # Stop Alpaca stream
    if ALPACA_AVAILABLE:
        try:
            await stop_alpaca_stream()
            logger.info("Alpaca WebSocket stream stopped")
        except Exception as e:
            logger.error(f"Error stopping Alpaca stream: {e}")

    if OBSERVABILITY_AVAILABLE:
        # Unsubscribe (if needed)
        pass

app = FastAPI(title="FoxML Dashboard IPC Bridge", lifespan=lifespan)


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """
    WebSocket endpoint for streaming events.
    
    Streams all events from EventBus to connected clients.
    """
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket client connected. Total connections: {len(active_connections)}")
    
    try:
        # Send recent events first
        if OBSERVABILITY_AVAILABLE:
            recent = events.get_recent(count=50)
            for event in recent:
                await websocket.send_json(event.to_dict())
        
        # Stream new events
        while True:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping", "timestamp": datetime.now(timezone.utc).isoformat()})
                
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(active_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.get("/api/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get current metrics.
    
    Returns:
        Dictionary of current metric values
    """
    if not OBSERVABILITY_AVAILABLE:
        return {
            "portfolio_value": 0.0,
            "daily_pnl": 0.0,
            "cash_balance": 0.0,
            "positions_count": 0,
            "sharpe_ratio": None,
            "trades_total": 0,
            "cycles_total": 0,
            "errors_total": 0,
        }
    
    # Read from MetricsRegistry
    try:
        return {
            "portfolio_value": metrics.portfolio_value.get(),
            "daily_pnl": metrics.daily_pnl.get(),
            "cash_balance": metrics.cash_balance.get(),
            "positions_count": metrics.positions_count.get(),
            "sharpe_ratio": None,  # TODO: Calculate from metrics
            "trades_total": metrics.trades_total.get(),
            "cycles_total": metrics.cycles_total.get(),
            "errors_total": metrics.errors_total.get(),
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {
            "error": str(e),
            "portfolio_value": 0.0,
            "daily_pnl": 0.0,
            "cash_balance": 0.0,
            "positions_count": 0,
        }


@app.get("/api/state")
async def get_state() -> Dict[str, Any]:
    """
    Get engine state including live pipeline stage.

    Returns:
        Dictionary of current engine state with live stage tracking
    """
    # Try to get live state from engine first
    try:
        from LIVE_TRADING.engine import get_engine
        engine = get_engine()

        if engine is not None:
            state_summary = engine.get_state_summary()
            return {
                "status": "running",
                "current_stage": engine.get_current_stage(),
                "last_cycle": state_summary.get("last_update"),
                "uptime_seconds": state_summary.get("uptime_seconds", 0),
                "cycle_count": state_summary.get("cycle_count", 0),
                "symbols_active": state_summary.get("num_positions", 0),
                "portfolio_value": state_summary.get("portfolio_value", 0),
                "cash": state_summary.get("cash", 0),
                "daily_pnl": state_summary.get("daily_pnl", 0),
                "kill_switch_active": state_summary.get("kill_switch_active", False),
            }
    except ImportError:
        pass  # Engine module not available
    except Exception as e:
        logger.error(f"Error getting live engine state: {e}")

    # Fallback: Try to read from engine state file
    state_file = Path(os.getenv("FOXML_STATE_DIR", ".")) / "engine_state.json"
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                file_state = json.load(f)
                file_state["status"] = "file_fallback"
                file_state.setdefault("current_stage", "idle")
                return file_state
        except Exception as e:
            logger.error(f"Error reading state file: {e}")

    # Default state
    return {
        "status": "stopped",
        "current_stage": "idle",
        "last_cycle": None,
        "uptime_seconds": 0,
        "cycle_count": 0,
        "symbols_active": 0,
    }


@app.get("/api/events/recent")
async def get_recent_events(count: int = 100) -> List[Dict[str, Any]]:
    """
    Get recent events.

    Args:
        count: Number of events to return

    Returns:
        List of recent events
    """
    if not OBSERVABILITY_AVAILABLE:
        return []

    recent = events.get_recent(count=count)
    return [event.to_dict() for event in recent]


# =============================================================================
# Dashboard Trading Endpoints
# =============================================================================


@app.get("/api/positions")
async def get_positions() -> Dict[str, Any]:
    """
    Get detailed position information.

    Returns list of positions with entry price, current price, P&L, weight.
    """
    try:
        from LIVE_TRADING.engine import get_engine_state
        state = get_engine_state()

        if state is None:
            return {"positions": [], "total_positions": 0}

        positions = []
        for symbol in sorted(state.positions.keys()):
            pos = state.positions[symbol]
            entry_price = getattr(pos, 'entry_price', pos.current_price)
            current_price = pos.current_price
            shares = pos.shares
            market_value = shares * current_price

            # Calculate unrealized P&L
            if entry_price > 0:
                unrealized_pnl = (current_price - entry_price) * shares
                unrealized_pnl_pct = ((current_price / entry_price) - 1) * 100
            else:
                unrealized_pnl = 0.0
                unrealized_pnl_pct = 0.0

            positions.append({
                "symbol": symbol,
                "shares": shares,
                "entry_price": entry_price,
                "current_price": current_price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "weight": getattr(pos, 'weight', 0.0),
                "entry_time": pos.timestamp.isoformat() if hasattr(pos, 'timestamp') and pos.timestamp else None,
                "side": "long" if shares > 0 else "short",
            })

        total_value = sum(p["market_value"] for p in positions)
        return {
            "positions": positions,
            "total_positions": len(positions),
            "long_count": sum(1 for p in positions if p["shares"] > 0),
            "short_count": sum(1 for p in positions if p["shares"] < 0),
            "total_market_value": total_value,
        }
    except ImportError:
        return {"positions": [], "total_positions": 0, "error": "Engine not available"}
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return {"positions": [], "total_positions": 0, "error": str(e)}


@app.get("/api/trades/history")
async def get_trade_history(
    limit: int = 100,
    offset: int = 0,
    symbol: Optional[str] = None,
    since: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get trade history with optional filtering.

    Args:
        limit: Max trades to return (default 100)
        offset: Pagination offset
        symbol: Filter by symbol
        since: Filter by timestamp (ISO format)
    """
    try:
        from LIVE_TRADING.engine import get_engine_state
        state = get_engine_state()

        if state is None:
            return {"trades": [], "total_trades": 0}

        # Get trade history from state
        trades = list(getattr(state, 'trade_history', []))

        # Apply filters
        if symbol:
            trades = [t for t in trades if t.get("symbol") == symbol]
        if since:
            try:
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
                trades = [t for t in trades if datetime.fromisoformat(t.get("timestamp", "1970-01-01")) >= since_dt]
            except ValueError:
                pass  # Invalid date format, skip filter

        # Sort by timestamp descending
        trades.sort(key=lambda t: t.get("timestamp", ""), reverse=True)

        # Paginate
        total = len(trades)
        trades = trades[offset:offset + limit]

        return {
            "trades": trades,
            "total_trades": total,
            "total_value": sum(t.get("value", 0) for t in trades),
        }
    except ImportError:
        return {"trades": [], "total_trades": 0, "error": "Engine not available"}
    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        return {"trades": [], "total_trades": 0, "error": str(e)}


@app.get("/api/cils/stats")
async def get_cils_stats() -> Dict[str, Any]:
    """
    Get CILS (online learning) optimizer statistics.

    Returns bandit arm weights, pull counts, and rewards.
    """
    try:
        from LIVE_TRADING.engine import get_engine
        engine = get_engine()

        if engine is None:
            return {"enabled": False}

        stats = engine.get_cils_stats()
        if stats is None:
            return {"enabled": False}

        # Format arms for dashboard display
        arms = []
        bandit_stats = stats.get("bandit_stats", {})
        arm_stats_list = bandit_stats.get("arm_stats", [])

        for arm in arm_stats_list:
            arms.append({
                "horizon": arm.get("name", f"arm_{arm.get('index', 0)}"),
                "weight": arm.get("weight", 0),
                "pull_count": arm.get("pull_count", 0),
                "cumulative_reward": arm.get("cumulative_reward", 0),
            })

        # Sort by horizon name
        arms.sort(key=lambda a: a.get("horizon", ""))

        return {
            "enabled": True,
            "algorithm": stats.get("algorithm", "exp3ix"),
            "arms": arms,
            "total_pulls": bandit_stats.get("total_steps", 0),
            "exploration_rate": bandit_stats.get("exploration_rate", 0),
            "effective_blend_ratio": stats.get("effective_blend_ratio", 0),
            "last_update": datetime.now(timezone.utc).isoformat(),
        }
    except ImportError:
        return {"enabled": False, "error": "Engine not available"}
    except Exception as e:
        logger.error(f"Error getting CILS stats: {e}")
        return {"enabled": False, "error": str(e)}


@app.get("/api/decisions/recent")
async def get_recent_decisions(
    limit: int = 50,
    symbol: Optional[str] = None,
    decision_type: Optional[str] = None,
    include_trace: bool = True,
) -> Dict[str, Any]:
    """
    Get recent trading decisions with full pipeline traces.

    This endpoint exposes the complete decision-making trail:
    - Why a trade was triggered or blocked
    - All model predictions and their weights
    - Blending calculations
    - Arbitration scores
    - Gate evaluations (spread, barrier)
    - Sizing calculations
    - Risk check results

    Args:
        limit: Max decisions to return (default 50)
        symbol: Filter by symbol (optional)
        decision_type: Filter by decision (TRADE, HOLD, BLOCKED)
        include_trace: Include full pipeline trace (default True)
    """
    try:
        from LIVE_TRADING.engine import get_engine_state
        state = get_engine_state()

        if state is None:
            return {"decisions": [], "total_decisions": 0}

        # Get decision history
        decisions = list(getattr(state, 'decision_history', []))

        # Apply filters
        if symbol:
            decisions = [d for d in decisions if d.get("symbol") == symbol]
        if decision_type:
            decisions = [d for d in decisions if d.get("decision") == decision_type.upper()]

        # Sort by timestamp descending (most recent first)
        decisions.sort(key=lambda d: d.get("timestamp", ""), reverse=True)

        # Limit
        total = len(decisions)
        decisions = decisions[:limit]

        # Optionally strip traces to reduce payload size
        if not include_trace:
            decisions = [
                {k: v for k, v in d.items() if k != "trace"}
                for d in decisions
            ]

        # Add summary statistics
        trade_count = sum(1 for d in decisions if d.get("decision") == "TRADE")
        hold_count = sum(1 for d in decisions if d.get("decision") == "HOLD")
        blocked_count = sum(1 for d in decisions if d.get("decision") == "BLOCKED")

        return {
            "decisions": decisions,
            "total_decisions": total,
            "trade_count": trade_count,
            "hold_count": hold_count,
            "blocked_count": blocked_count,
        }
    except ImportError:
        return {"decisions": [], "total_decisions": 0, "error": "Engine not available"}
    except Exception as e:
        logger.error(f"Error getting decisions: {e}")
        return {"decisions": [], "total_decisions": 0, "error": str(e)}


@app.get("/api/decisions/{symbol}/latest")
async def get_latest_decision(symbol: str) -> Dict[str, Any]:
    """
    Get the most recent decision for a specific symbol.

    Returns the full decision with pipeline trace showing exactly
    why the last trading decision was made.

    Args:
        symbol: Trading symbol (e.g., "AAPL")
    """
    try:
        from LIVE_TRADING.engine import get_engine_state
        state = get_engine_state()

        if state is None:
            return {"decision": None, "error": "Engine not running"}

        # Find most recent decision for this symbol
        decisions = [
            d for d in getattr(state, 'decision_history', [])
            if d.get("symbol") == symbol
        ]

        if not decisions:
            return {"decision": None, "message": f"No decisions found for {symbol}"}

        # Sort by timestamp and get most recent
        decisions.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
        latest = decisions[0]

        # Generate human-readable explanation if trace exists
        explanation = None
        if latest.get("trace"):
            trace = latest["trace"]
            explanation = _generate_decision_explanation(latest, trace)

        return {
            "decision": latest,
            "explanation": explanation,
        }
    except ImportError:
        return {"decision": None, "error": "Engine not available"}
    except Exception as e:
        logger.error(f"Error getting latest decision for {symbol}: {e}")
        return {"decision": None, "error": str(e)}


def _generate_decision_explanation(decision: Dict[str, Any], trace: Dict[str, Any]) -> str:
    """Generate a human-readable explanation of a trading decision."""
    lines = [
        f"Decision: {decision.get('decision')} for {decision.get('symbol')}",
        f"Reason: {decision.get('reason')}",
        "",
    ]

    # Market context
    snapshot = trace.get("market_snapshot", {})
    if snapshot:
        lines.extend([
            "Market Context:",
            f"  Price: {snapshot.get('close', 0):.2f}",
            f"  Spread: {snapshot.get('spread_bps', 0):.1f} bps",
            f"  Volatility: {snapshot.get('volatility', 0):.2%}",
            "",
        ])

    # Horizon analysis
    horizon_scores = trace.get("horizon_scores", {})
    blended_alphas = trace.get("blended_alphas", {})
    selected = trace.get("selected_horizon")
    if horizon_scores:
        lines.append("Horizon Analysis:")
        for h in sorted(horizon_scores.keys()):
            alpha_bps = blended_alphas.get(h, 0) * 10000
            score = horizon_scores.get(h, 0)
            marker = " <-- SELECTED" if h == selected else ""
            lines.append(f"  {h}: alpha={alpha_bps:.1f}bps, score={score:.2f}{marker}")
        lines.append("")

    # Gate results
    spread_gate = trace.get("spread_gate_result", {})
    barrier_gate = trace.get("barrier_gate_result", {})
    if spread_gate or barrier_gate:
        lines.append("Gate Evaluations:")
        if spread_gate:
            lines.append(f"  Spread Gate: allowed={spread_gate.get('allowed', True)}, "
                        f"spread={spread_gate.get('spread_bps', 0):.1f}bps")
        if barrier_gate:
            lines.append(f"  Barrier Gate: allowed={barrier_gate.get('allowed', True)}, "
                        f"p_peak={barrier_gate.get('p_peak', 0):.2f}, "
                        f"p_valley={barrier_gate.get('p_valley', 0):.2f}")
        lines.append("")

    # Sizing
    raw_weight = trace.get("raw_weight", 0)
    final_weight = trace.get("final_weight", 0)
    if raw_weight or final_weight:
        lines.extend([
            "Sizing:",
            f"  Raw Weight: {raw_weight:.4f}",
            f"  Final Weight: {final_weight:.4f}",
            f"  Shares: {decision.get('shares', 0)}",
            "",
        ])

    # Risk checks
    risk_checks = trace.get("risk_checks", {})
    if risk_checks:
        lines.append("Risk Checks:")
        for check, result in sorted(risk_checks.items()):
            status = "✓" if result else "✗"
            lines.append(f"  {status} {check}")

    return "\n".join(lines)


@app.get("/api/risk/status")
async def get_risk_status_endpoint() -> Dict[str, Any]:
    """
    Get current risk status including drawdown, exposure, and warnings.
    """
    try:
        from LIVE_TRADING.risk import get_risk_status
        status = get_risk_status()

        if status is None:
            return {
                "trading_allowed": False,
                "kill_switch_active": False,
                "warnings": [],
                "error": "Risk guardrails not available",
            }

        return {
            "trading_allowed": status.trading_allowed,
            "kill_switch_active": status.kill_switch_active,
            "kill_switch_reason": status.kill_switch_reason,

            "daily_pnl_pct": status.daily_pnl_pct,
            "daily_loss_limit_pct": status.daily_loss_limit_pct,
            "daily_loss_remaining_pct": status.daily_loss_remaining_pct,

            "drawdown_pct": status.drawdown_pct,
            "max_drawdown_limit_pct": status.max_drawdown_limit_pct,
            "drawdown_remaining_pct": status.drawdown_remaining_pct,

            "gross_exposure": status.gross_exposure,
            "net_exposure": status.net_exposure,
            "max_gross_exposure": status.max_gross_exposure,

            "warnings": [w.to_dict() for w in status.warnings],

            "last_check": datetime.now(timezone.utc).isoformat(),
        }
    except ImportError:
        return {
            "trading_allowed": False,
            "kill_switch_active": False,
            "warnings": [],
            "error": "Risk module not available",
        }
    except Exception as e:
        logger.error(f"Error getting risk status: {e}")
        return {
            "trading_allowed": False,
            "error": str(e),
            "warnings": [],
        }


# Alpaca endpoints

@app.get("/api/alpaca/status")
async def get_alpaca_status() -> Dict[str, Any]:
    """
    Get Alpaca WebSocket connection status.

    Returns:
        Connection status and configuration info
    """
    if not ALPACA_AVAILABLE:
        return {
            "available": False,
            "connected": False,
            "error": "Alpaca module not available",
        }

    client = get_alpaca_client()
    if not client:
        return {
            "available": True,
            "connected": False,
            "error": "Client not initialized",
        }

    return {
        "available": True,
        "connected": client.is_connected,
        "paper": client.config.paper,
        "has_credentials": bool(client.config.api_key and client.config.api_secret),
    }


@app.get("/api/alpaca/events/recent")
async def get_recent_alpaca_events(count: int = 50) -> List[Dict[str, Any]]:
    """
    Get recent Alpaca events (trade updates, account updates).

    Args:
        count: Number of events to return

    Returns:
        List of recent Alpaca events
    """
    if not ALPACA_AVAILABLE:
        return []

    client = get_alpaca_client()
    if not client:
        return []

    recent = client.get_recent_events(count=count)
    return [event.to_dict() for event in recent]


@app.websocket("/ws/alpaca")
async def websocket_alpaca(websocket: WebSocket):
    """
    WebSocket endpoint for streaming Alpaca events.

    Streams trade updates and account updates from Alpaca.
    """
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"Alpaca WebSocket client connected. Total connections: {len(active_connections)}")

    try:
        # Send recent events first
        if ALPACA_AVAILABLE:
            client = get_alpaca_client()
            if client:
                recent = client.get_recent_events(count=20)
                for event in recent:
                    await websocket.send_json(event.to_dict())

        # Stream new events from Alpaca queue
        while True:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(alpaca_event_queue.get(), timeout=5.0)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping", "source": "alpaca"})

    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"Alpaca WebSocket client disconnected. Total connections: {len(active_connections)}")
    except Exception as e:
        logger.error(f"Alpaca WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


# =============================================================================
# Training Progress WebSocket
# =============================================================================

# Training event queue for WebSocket streaming
training_event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)


def on_training_event(event: Dict[str, Any]) -> None:
    """Callback for training progress events."""
    try:
        training_event_queue.put_nowait(event)
    except asyncio.QueueFull:
        logger.debug("Training event queue full, dropping event")


@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    """
    WebSocket endpoint for training progress streaming.

    Streams training progress events including:
    - target_start: When training starts for a target
    - target_complete: When training completes for a target
    - model_complete: When a model family completes
    - stage_change: Pipeline stage changes
    - progress: Percentage progress updates
    """
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"Training WebSocket client connected. Total connections: {len(active_connections)}")

    try:
        while True:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(
                    training_event_queue.get(),
                    timeout=5.0
                )
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({
                    "type": "ping",
                    "source": "training",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"Training WebSocket client disconnected. Total connections: {len(active_connections)}")
    except Exception as e:
        logger.error(f"Training WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.post("/api/training/event")
async def post_training_event(event: Dict[str, Any]) -> Dict[str, str]:
    """
    Post a training event to be broadcast via WebSocket.

    This allows the training pipeline to push events to the dashboard.

    Args:
        event: Event data to broadcast

    Returns:
        Status response
    """
    try:
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = datetime.now(timezone.utc).isoformat()

        on_training_event(event)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error posting training event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint."""
    alpaca_connected = False
    if ALPACA_AVAILABLE:
        client = get_alpaca_client()
        if client:
            alpaca_connected = client.is_connected

    return {
        "status": "ok",
        "observability_available": OBSERVABILITY_AVAILABLE,
        "alpaca_available": ALPACA_AVAILABLE,
        "alpaca_connected": alpaca_connected,
        "active_connections": len(active_connections),
    }


# Control endpoints

class KillSwitchRequest(BaseModel):
    """Request body for kill switch toggle."""
    action: str  # "enable" or "disable"
    reason: Optional[str] = None


@app.post("/api/control/pause")
async def pause_engine() -> Dict[str, Any]:
    """
    Pause trading engine (stop processing cycles).
    
    Returns:
        Status response
    """
    state = _read_control_state()
    state["paused"] = True
    _write_control_state(state)
    
    logger.info("Trading engine paused via API")
    
    return {
        "status": "paused",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/control/resume")
async def resume_engine() -> Dict[str, Any]:
    """
    Resume trading engine.
    
    Returns:
        Status response
    """
    state = _read_control_state()
    state["paused"] = False
    _write_control_state(state)
    
    logger.info("Trading engine resumed via API")
    
    return {
        "status": "resumed",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/control/kill_switch")
async def toggle_kill_switch(request: KillSwitchRequest) -> Dict[str, Any]:
    """
    Toggle manual kill switch.
    
    Args:
        request: Kill switch action (enable/disable) and optional reason
        
    Returns:
        Current kill switch state
    """
    if request.action not in ("enable", "disable"):
        raise HTTPException(status_code=400, detail="action must be 'enable' or 'disable'")
    
    state = _read_control_state()
    
    if request.action == "enable":
        state["kill_switch_active"] = True
        state["kill_switch_reason"] = request.reason or "Manual kill switch enabled via API"
        logger.warning(f"Manual kill switch enabled: {state['kill_switch_reason']}")
    else:
        state["kill_switch_active"] = False
        state["kill_switch_reason"] = None
        logger.info("Manual kill switch disabled via API")
    
    _write_control_state(state)
    
    return {
        "kill_switch_active": state["kill_switch_active"],
        "kill_switch_reason": state["kill_switch_reason"],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/control/status")
async def get_control_status() -> Dict[str, Any]:
    """
    Get current control state.
    
    Returns:
        Dictionary with paused, kill_switch_active, and kill_switch_reason
    """
    state = _read_control_state()
    return {
        "paused": state.get("paused", False),
        "kill_switch_active": state.get("kill_switch_active", False),
        "kill_switch_reason": state.get("kill_switch_reason"),
        "last_updated": state.get("last_updated"),
    }


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging to file, not stdout
    log_dir = Path(os.getenv("FOXML_STATE_DIR", ".")) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "bridge.log"
    
    # Set up file logging for our application logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Add file handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
    
    # Run uvicorn with access logs disabled (they go to stdout and interfere with TUI)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8765,
        access_log=False,  # Disable access logs to prevent TUI interference
        log_config=None,  # Use our own logging config (already set up above)
    )
