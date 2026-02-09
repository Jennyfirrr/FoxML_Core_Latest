"""
Paper Trading Broker
====================

Simulated broker for paper trading with configurable slippage and fees.
Implements the Broker protocol for testing and simulation.

SST Compliance:
- Uses get_cfg() for configuration with fallback defaults
- Uses sorted iteration for deterministic behavior
- Logs trades to JSONL files for audit
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from CONFIG.config_loader import get_cfg
from TRAINING.common.utils.determinism_ordering import sorted_items

from LIVE_TRADING.common.clock import Clock, get_clock
from LIVE_TRADING.common.time_utils import parse_iso
from LIVE_TRADING.common.constants import (
    DEFAULT_CONFIG,
    SIDE_BUY,
    SIDE_SELL,
)
from LIVE_TRADING.common.exceptions import (
    BrokerError,
    InsufficientFundsError,
    OrderRejectedError,
)

logger = logging.getLogger(__name__)


class PaperBroker:
    """
    Paper trading broker with slippage simulation.

    Implements the Broker protocol for testing and simulation.
    Orders fill immediately at the current quote price plus slippage.

    Features:
    - Configurable slippage and fees
    - Position tracking
    - Fill history
    - Trade logging to JSONL files

    Example:
        >>> broker = PaperBroker(initial_cash=100_000)
        >>> broker.set_quote("AAPL", bid=150.0, ask=150.10)
        >>> result = broker.submit_order("AAPL", "BUY", 100)
        >>> broker.get_positions()
        {"AAPL": 100.0}
    """

    def __init__(
        self,
        *,
        slippage_bps: float | None = None,
        fee_bps: float | None = None,
        initial_cash: float | None = None,
        log_dir: str | Path | None = None,
        clock: Clock | None = None,
    ) -> None:
        """
        Initialize paper broker.

        Args:
            slippage_bps: Slippage in basis points (default from config)
            fee_bps: Fee in basis points (default from config)
            initial_cash: Initial cash balance (default from config)
            log_dir: Directory for trade logs (default: logs/paper_trades)
            clock: Clock instance for time (default: system clock)
        """
        self._clock = clock or get_clock()
        # Load from config with defaults
        self._slippage_bps = slippage_bps if slippage_bps is not None else get_cfg(
            "live_trading.paper.slippage_bps",
            default=DEFAULT_CONFIG["slippage_bps"],
        )
        self._fee_bps = fee_bps if fee_bps is not None else get_cfg(
            "live_trading.paper.fee_bps",
            default=DEFAULT_CONFIG["fee_bps"],
        )
        initial = initial_cash if initial_cash is not None else get_cfg(
            "live_trading.paper.initial_cash",
            default=DEFAULT_CONFIG["initial_cash"],
        )

        # Convert bps to decimal multiplier
        self._slip = float(self._slippage_bps) * 1e-4
        self._fee = float(self._fee_bps) * 1e-4

        # State
        self._positions: dict[str, float] = {}
        self._cash: float = float(initial)
        self._initial_cash: float = float(initial)
        self._fills: list[dict[str, Any]] = []
        self._orders: dict[str, dict[str, Any]] = {}
        self._quotes: dict[str, dict[str, Any]] = {}

        # Logging
        log_path = log_dir or get_cfg(
            "live_trading.paper.log_dir",
            default="logs/paper_trades",
        )
        self._log_dir = Path(log_path)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = self._clock.now().strftime("%Y-%m-%dT%H%M%SZ")

        logger.info(
            f"PaperBroker initialized: cash=${initial:,.2f}, "
            f"slippage={self._slippage_bps}bps, fee={self._fee_bps}bps"
        )

    # =========================================================================
    # Broker Protocol Implementation
    # =========================================================================

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Submit order with slippage simulation.

        Orders fill immediately at the current quote price plus slippage.
        For buys, we pay ask + slippage. For sells, we receive bid - slippage.

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            qty: Quantity to trade
            order_type: Order type (only "market" supported currently)
            limit_price: Limit price (not used for market orders)

        Returns:
            Fill result dict

        Raises:
            OrderRejectedError: If order is invalid
            InsufficientFundsError: If not enough cash
            BrokerError: If no quote available
        """
        symbol = symbol.upper()
        side = side.upper()
        qty = float(qty)

        # Validate side
        if side not in (SIDE_BUY, SIDE_SELL):
            raise OrderRejectedError(symbol, f"Invalid side: {side}")

        # Validate quantity
        if qty <= 0:
            raise OrderRejectedError(symbol, f"Invalid quantity: {qty}")

        # Get quote for fill price
        quote = self._quotes.get(symbol)
        if quote is None:
            raise BrokerError(f"No quote available for {symbol}")

        # Calculate fill price with slippage
        if side == SIDE_BUY:
            base_price = quote["ask"]
            fill_price = base_price * (1 + self._slip)
        else:
            base_price = quote["bid"]
            fill_price = base_price * (1 - self._slip)

        # Calculate trade value and fee
        notional = qty * fill_price
        fee = notional * self._fee

        # Check cash for buy orders
        if side == SIDE_BUY:
            required_cash = notional + fee
            if required_cash > self._cash:
                raise InsufficientFundsError(
                    f"Insufficient cash: need ${required_cash:,.2f}, "
                    f"have ${self._cash:,.2f}"
                )

        # Check cash for sell short orders (margin requirement)
        if side == SIDE_SELL:
            current_pos = self._positions.get(symbol, 0.0)
            if qty > current_pos:
                # Short selling: need cash to cover the margin
                short_qty = qty - current_pos
                margin_required = short_qty * fill_price + fee
                if margin_required > self._cash:
                    raise InsufficientFundsError(
                        f"Insufficient margin for short: need ${margin_required:,.2f}, "
                        f"have ${self._cash:,.2f}"
                    )

        # Execute the order
        order_id = str(uuid.uuid4())[:8]
        timestamp = self._clock.now()

        # Update positions
        if side == SIDE_BUY:
            self._positions[symbol] = self._positions.get(symbol, 0.0) + qty
            self._cash -= notional + fee
        else:
            self._positions[symbol] = self._positions.get(symbol, 0.0) - qty
            self._cash += notional - fee
            # Clean up zero positions
            if abs(self._positions[symbol]) < 1e-9:
                del self._positions[symbol]

        # Create fill record
        fill = {
            "order_id": order_id,
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "fill_price": fill_price,
            "notional": notional,
            "fee": fee,
            "slippage_bps": self._slippage_bps,
            "fee_bps": self._fee_bps,
        }

        self._fills.append(fill)
        self._orders[order_id] = fill
        self._append_log(fill)

        logger.info(
            f"Order filled: {side} {qty} {symbol} @ ${fill_price:.2f} "
            f"(fee: ${fee:.2f})"
        )

        return {
            "order_id": order_id,
            "status": "filled",
            "fill_price": fill_price,
            "filled_qty": qty,
            "qty": qty,
            "timestamp": timestamp,
        }

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """
        Cancel order (no-op for paper broker since orders fill instantly).

        Args:
            order_id: Order ID to cancel

        Returns:
            Status dict
        """
        return {
            "order_id": order_id,
            "status": "cancelled" if order_id not in self._orders else "filled",
            "timestamp": self._clock.now(),
        }

    def get_positions(self) -> dict[str, float]:
        """
        Get current positions.

        Returns:
            Dict mapping symbol to share count
        """
        # Return a copy with sorted keys for determinism
        return dict(sorted_items(self._positions))

    def get_cash(self) -> float:
        """
        Get available cash.

        Returns:
            Cash balance
        """
        return float(self._cash)

    def get_fills(self, since: datetime | None = None) -> list[dict[str, Any]]:
        """
        Get fills, optionally filtered by time.

        Args:
            since: Only return fills after this time

        Returns:
            List of fill records
        """
        if since is None:
            return list(self._fills)

        return [
            f
            for f in self._fills
            if parse_iso(f["timestamp"]) >= since
        ]

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """
        Get current quote for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Quote dict

        Raises:
            BrokerError: If no quote available
        """
        symbol = symbol.upper()
        if symbol not in self._quotes:
            raise BrokerError(f"No quote for {symbol}")
        return dict(self._quotes[symbol])

    def now(self) -> datetime:
        """
        Get current time.

        Returns:
            Current timezone.utc datetime
        """
        return self._clock.now()

    # =========================================================================
    # Paper Broker Specific Methods
    # =========================================================================

    def set_quote(
        self,
        symbol: str,
        bid: float,
        ask: float,
        bid_size: float = 100,
        ask_size: float = 100,
    ) -> None:
        """
        Set quote for a symbol (for testing/simulation).

        Args:
            symbol: Trading symbol
            bid: Bid price
            ask: Ask price
            bid_size: Bid size
            ask_size: Ask size
        """
        symbol = symbol.upper()
        spread_bps = 0.0
        if (ask + bid) > 0:
            spread_bps = (ask - bid) / ((ask + bid) / 2) * 10000

        self._quotes[symbol] = {
            "symbol": symbol,
            "bid": float(bid),
            "ask": float(ask),
            "bid_size": float(bid_size),
            "ask_size": float(ask_size),
            "timestamp": self._clock.now(),
            "spread_bps": spread_bps,
        }

    def set_quotes(self, quotes: dict[str, tuple[float, float]]) -> None:
        """
        Set multiple quotes at once.

        Args:
            quotes: Dict mapping symbol to (bid, ask) tuple
        """
        for symbol, (bid, ask) in sorted_items(quotes):
            self.set_quote(symbol, bid, ask)

    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value (cash + positions).

        Returns:
            Total portfolio value
        """
        value = self._cash
        for symbol, qty in sorted_items(self._positions):
            if symbol in self._quotes:
                mid = (
                    self._quotes[symbol]["bid"] + self._quotes[symbol]["ask"]
                ) / 2
                value += qty * mid
        return value

    def get_position_weights(self) -> dict[str, float]:
        """
        Get current position weights.

        Returns:
            Dict mapping symbol to weight (as fraction of portfolio)
        """
        portfolio_value = self.get_portfolio_value()
        if portfolio_value <= 0:
            return {}

        weights = {}
        for symbol, qty in sorted_items(self._positions):
            if symbol in self._quotes:
                mid = (
                    self._quotes[symbol]["bid"] + self._quotes[symbol]["ask"]
                ) / 2
                weights[symbol] = (qty * mid) / portfolio_value

        return weights

    def get_daily_pnl(self) -> float:
        """
        Get P&L since start of session.

        Returns:
            P&L in dollars
        """
        return self.get_portfolio_value() - self._initial_cash

    def get_daily_pnl_pct(self) -> float:
        """
        Get P&L percentage since start of session.

        Returns:
            P&L as percentage
        """
        if self._initial_cash <= 0:
            return 0.0
        return (self.get_portfolio_value() / self._initial_cash - 1) * 100

    def reset(self, initial_cash: float | None = None) -> None:
        """
        Reset broker state.

        Args:
            initial_cash: New starting cash (or use original)
        """
        self._positions.clear()
        self._fills.clear()
        self._orders.clear()
        cash = initial_cash if initial_cash is not None else get_cfg(
            "live_trading.paper.initial_cash",
            default=DEFAULT_CONFIG["initial_cash"],
        )
        self._cash = float(cash)
        self._initial_cash = float(cash)
        logger.info(f"PaperBroker reset: cash=${cash:,.2f}")

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary of current state.

        Returns:
            Summary dict
        """
        return {
            "cash": self._cash,
            "portfolio_value": self.get_portfolio_value(),
            "daily_pnl": self.get_daily_pnl(),
            "daily_pnl_pct": self.get_daily_pnl_pct(),
            "positions": self.get_positions(),
            "position_count": len(self._positions),
            "fill_count": len(self._fills),
        }

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _append_log(self, record: dict[str, Any]) -> None:
        """
        Append record to daily log file.

        Args:
            record: Record to log
        """
        date_str = self._clock.now().strftime("%Y-%m-%d")
        log_file = self._log_dir / f"{date_str}.jsonl"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
