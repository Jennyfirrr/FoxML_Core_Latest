#!/usr/bin/env python3
"""
Alpaca WebSocket Stream Client
==============================

Connects to Alpaca's WebSocket streams for real-time:
- Trade updates (fills, cancels, rejects)
- Account updates (balance changes)

This module handles authentication, reconnection, and event forwarding.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)


class AlpacaStreamType(str, Enum):
    """Alpaca WebSocket stream types."""
    TRADING = "trading"      # Trade updates, account updates
    # Future expansion:
    # SIP = "sip"            # Real-time market data (paid)
    # IEX = "iex"            # Free market data


class AlpacaEventType(str, Enum):
    """Event types from Alpaca streams."""
    # Trading stream events
    TRADE_UPDATE = "trade_update"
    ACCOUNT_UPDATE = "account_update"

    # Connection events (internal)
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    AUTHENTICATED = "authenticated"


@dataclass
class AlpacaEvent:
    """Normalized event from Alpaca WebSocket."""
    event_type: AlpacaEventType
    timestamp: datetime
    data: Dict[str, Any]
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": "alpaca",
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


@dataclass
class TradeUpdate:
    """Parsed trade update from Alpaca."""
    event: str           # new, fill, partial_fill, canceled, expired, etc.
    order_id: str
    symbol: str
    side: str            # buy, sell
    qty: float
    filled_qty: float
    filled_avg_price: Optional[float]
    order_type: str      # market, limit, stop, stop_limit
    status: str          # new, partially_filled, filled, canceled, etc.
    timestamp: datetime

    @classmethod
    def from_alpaca(cls, data: Dict[str, Any]) -> "TradeUpdate":
        """Parse from Alpaca trade_updates message."""
        order = data.get("order", {})
        return cls(
            event=data.get("event", "unknown"),
            order_id=order.get("id", ""),
            symbol=order.get("symbol", ""),
            side=order.get("side", ""),
            qty=float(order.get("qty", 0)),
            filled_qty=float(order.get("filled_qty", 0)),
            filled_avg_price=float(order["filled_avg_price"]) if order.get("filled_avg_price") else None,
            order_type=order.get("type", ""),
            status=order.get("status", ""),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now(timezone.utc).isoformat()).replace("Z", "+00:00")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "filled_qty": self.filled_qty,
            "filled_avg_price": self.filled_avg_price,
            "order_type": self.order_type,
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AlpacaStreamConfig:
    """Configuration for Alpaca WebSocket connection."""
    api_key: str
    api_secret: str
    paper: bool = True

    # Stream URLs
    PAPER_TRADING_URL = "wss://paper-api.alpaca.markets/stream"
    LIVE_TRADING_URL = "wss://api.alpaca.markets/stream"

    @property
    def trading_url(self) -> str:
        return self.PAPER_TRADING_URL if self.paper else self.LIVE_TRADING_URL

    @classmethod
    def from_env(cls) -> "AlpacaStreamConfig":
        """Load config from environment variables."""
        api_key = os.getenv("APCA_API_KEY_ID", "")
        api_secret = os.getenv("APCA_API_SECRET_KEY", "")
        paper = os.getenv("APCA_PAPER", "true").lower() == "true"

        if not api_key or not api_secret:
            logger.warning("Alpaca API credentials not found in environment")

        return cls(api_key=api_key, api_secret=api_secret, paper=paper)


class AlpacaStreamClient:
    """
    WebSocket client for Alpaca trading stream.

    Handles:
    - Authentication
    - Automatic reconnection
    - Event parsing and forwarding
    """

    def __init__(
        self,
        config: Optional[AlpacaStreamConfig] = None,
        on_event: Optional[Callable[[AlpacaEvent], None]] = None,
    ):
        self.config = config or AlpacaStreamConfig.from_env()
        self.on_event = on_event

        self._ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._authenticated = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0

        # Event history (for dashboard polling fallback)
        self._recent_events: List[AlpacaEvent] = []
        self._max_events = 100

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._authenticated

    def get_recent_events(self, count: int = 50) -> List[AlpacaEvent]:
        """Get recent events (for polling fallback)."""
        if count <= 0:
            return []
        return self._recent_events[-count:]

    async def connect(self) -> None:
        """Connect to Alpaca WebSocket and start listening."""
        if not self.config.api_key or not self.config.api_secret:
            logger.error("Cannot connect: Alpaca API credentials not configured")
            self._emit_event(AlpacaEvent(
                event_type=AlpacaEventType.ERROR,
                timestamp=datetime.now(timezone.utc),
                data={"error": "API credentials not configured"},
            ))
            return

        self._running = True

        while self._running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self._emit_event(AlpacaEvent(
                    event_type=AlpacaEventType.DISCONNECTED,
                    timestamp=datetime.now(timezone.utc),
                    data={"error": str(e)},
                ))

                if self._running:
                    logger.info(f"Reconnecting in {self._reconnect_delay}s...")
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2,
                        self._max_reconnect_delay
                    )

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._authenticated = False

    async def _connect_and_listen(self) -> None:
        """Internal: Connect, authenticate, and listen for messages."""
        url = self.config.trading_url
        logger.info(f"Connecting to Alpaca WebSocket: {url}")

        async with websockets.connect(url) as ws:
            self._ws = ws
            self._reconnect_delay = 1.0  # Reset on successful connect

            self._emit_event(AlpacaEvent(
                event_type=AlpacaEventType.CONNECTED,
                timestamp=datetime.now(timezone.utc),
                data={"url": url, "paper": self.config.paper},
            ))

            # Authenticate
            await self._authenticate()

            # Listen for messages
            async for message in ws:
                await self._handle_message(message)

    async def _authenticate(self) -> None:
        """Send authentication message."""
        if not self._ws:
            return

        auth_msg = {
            "action": "auth",
            "key": self.config.api_key,
            "secret": self.config.api_secret,
        }
        await self._ws.send(json.dumps(auth_msg))
        logger.debug("Authentication message sent")

    async def _subscribe(self) -> None:
        """Subscribe to trade updates after authentication."""
        if not self._ws:
            return

        subscribe_msg = {
            "action": "listen",
            "data": {
                "streams": ["trade_updates"]
            }
        }
        await self._ws.send(json.dumps(subscribe_msg))
        logger.info("Subscribed to trade_updates stream")

    async def _handle_message(self, raw_message: str) -> None:
        """Parse and handle incoming WebSocket message."""
        try:
            msg = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON received: {raw_message[:100]}")
            return

        stream = msg.get("stream")
        data = msg.get("data", {})

        # Handle authentication response
        if stream == "authorization":
            if data.get("status") == "authorized":
                self._authenticated = True
                logger.info("Alpaca WebSocket authenticated")
                self._emit_event(AlpacaEvent(
                    event_type=AlpacaEventType.AUTHENTICATED,
                    timestamp=datetime.now(timezone.utc),
                    data={"status": "authorized"},
                ))
                # Subscribe to streams after auth
                await self._subscribe()
            else:
                logger.error(f"Authentication failed: {data}")
                self._emit_event(AlpacaEvent(
                    event_type=AlpacaEventType.ERROR,
                    timestamp=datetime.now(timezone.utc),
                    data={"error": "Authentication failed", "details": data},
                ))

        # Handle listening confirmation
        elif stream == "listening":
            streams = data.get("streams", [])
            logger.info(f"Now listening to streams: {streams}")

        # Handle trade updates
        elif stream == "trade_updates":
            trade_update = TradeUpdate.from_alpaca(data)
            event = AlpacaEvent(
                event_type=AlpacaEventType.TRADE_UPDATE,
                timestamp=trade_update.timestamp,
                data=trade_update.to_dict(),
                raw=msg,
            )
            self._emit_event(event)
            logger.info(f"Trade update: {trade_update.event} {trade_update.symbol} {trade_update.side}")

        else:
            logger.debug(f"Unhandled stream: {stream}")

    def _emit_event(self, event: AlpacaEvent) -> None:
        """Emit event to callback and store in history."""
        # Store in history
        self._recent_events.append(event)
        if len(self._recent_events) > self._max_events:
            self._recent_events = self._recent_events[-self._max_events:]

        # Call callback if set
        if self.on_event:
            try:
                self.on_event(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")


# Singleton instance for use by bridge
_alpaca_client: Optional[AlpacaStreamClient] = None


def get_alpaca_client() -> Optional[AlpacaStreamClient]:
    """Get the global Alpaca client instance."""
    return _alpaca_client


async def start_alpaca_stream(
    on_event: Optional[Callable[[AlpacaEvent], None]] = None,
) -> AlpacaStreamClient:
    """
    Start the Alpaca WebSocket stream.

    Args:
        on_event: Callback for received events

    Returns:
        The AlpacaStreamClient instance
    """
    global _alpaca_client

    config = AlpacaStreamConfig.from_env()
    _alpaca_client = AlpacaStreamClient(config=config, on_event=on_event)

    # Start connection in background task
    asyncio.create_task(_alpaca_client.connect())

    return _alpaca_client


async def stop_alpaca_stream() -> None:
    """Stop the Alpaca WebSocket stream."""
    global _alpaca_client

    if _alpaca_client:
        await _alpaca_client.disconnect()
        _alpaca_client = None
