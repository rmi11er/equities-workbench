"""Kalshi exchange connector for REST and WebSocket communication."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from contextlib import asynccontextmanager

import aiohttp
import websockets
from websockets.client import WebSocketClientProtocol

from .auth import KalshiAuth
from .config import Config
from .constants import (
    Environment,
    WS_URLS,
    REST_URLS,
    WS_PATH,
    MAX_RECONNECT_ATTEMPTS,
    INITIAL_BACKOFF_SEC,
    MAX_BACKOFF_SEC,
    WS_HEARTBEAT_INTERVAL,
)
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


@dataclass
class WSMessage:
    """Parsed WebSocket message."""
    type: str
    sid: Optional[int] = None
    seq: Optional[int] = None
    msg: dict = field(default_factory=dict)


class KalshiConnector:
    """
    Manages all I/O with the Kalshi exchange.

    Handles:
    - RSA authentication
    - REST API calls with rate limiting
    - WebSocket connection with auto-reconnect
    - Heartbeat monitoring
    - Sequence gap detection
    """

    def __init__(self, config: Config):
        self.config = config
        self.env = config.environment

        self._auth = KalshiAuth(
            api_key_id=config.credentials.api_key_id,
            private_key_path=config.credentials.private_key_path,
        )
        self._rate_limiter = RateLimiter(
            read_rate=config.rate_limit.read_rate,
            write_rate=config.rate_limit.write_rate,
        )

        self._ws: Optional[WebSocketClientProtocol] = None
        self._ws_lock = asyncio.Lock()
        self._session: Optional[aiohttp.ClientSession] = None

        # Subscription state
        self._subscriptions: dict[int, dict] = {}  # sid -> subscription params
        self._next_cmd_id = 1
        self._sequence_numbers: dict[int, int] = {}  # sid -> last seq

        # Callbacks
        self._message_handlers: list[Callable[[WSMessage], None]] = []
        self._reconnect_handlers: list[Callable[[], None]] = []

        # State
        self._running = False
        self._last_message_time = 0.0
        self._reconnect_count = 0

    @property
    def ws_url(self) -> str:
        return WS_URLS[self.env]

    @property
    def rest_url(self) -> str:
        return REST_URLS[self.env]

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the connector (create HTTP session)."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        self._running = True
        logger.info(f"Connector started for {self.env.name}")

    async def stop(self) -> None:
        """Stop the connector and clean up resources."""
        self._running = False

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("Connector stopped")

    @asynccontextmanager
    async def session(self):
        """Context manager for connector lifecycle."""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

    # -------------------------------------------------------------------------
    # REST API
    # -------------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        data: Optional[dict] = None,
        is_write: bool = False,
    ) -> dict:
        """Make an authenticated REST API request."""
        if self._session is None:
            raise RuntimeError("Connector not started")

        # Rate limiting
        if is_write:
            await self._rate_limiter.acquire_write()
        else:
            await self._rate_limiter.acquire_read()

        url = f"{self.rest_url}{path}"
        headers = self._auth.get_auth_headers(method, path)
        headers["Content-Type"] = "application/json"

        kwargs: dict[str, Any] = {"headers": headers}
        if data:
            kwargs["json"] = data

        async with self._session.request(method, url, **kwargs) as resp:
            if resp.status == 429:
                logger.warning("Rate limit hit (429)")
                raise RateLimitError("Rate limit exceeded")

            resp_data = await resp.json()

            if resp.status >= 400:
                logger.error(f"API error {resp.status}: {resp_data}")
                raise APIError(resp.status, resp_data)

            return resp_data

    async def get_balance(self) -> dict:
        """Get account balance."""
        return await self._request("GET", "/portfolio/balance")

    async def get_positions(self) -> dict:
        """Get all positions."""
        return await self._request("GET", "/portfolio/positions")

    async def get_orders(self, ticker: Optional[str] = None) -> dict:
        """Get open orders."""
        path = "/portfolio/orders"
        if ticker:
            path += f"?ticker={ticker}"
        return await self._request("GET", path)

    async def create_order(
        self,
        ticker: str,
        side: str,
        price: int,
        count: int,
        client_order_id: Optional[str] = None,
    ) -> dict:
        """
        Create a new order.

        Args:
            ticker: Market ticker
            side: "buy" or "sell"
            price: Price in cents (1-99)
            count: Number of contracts
            client_order_id: Optional client-side order ID

        Returns:
            Order response with order_id
        """
        data = {
            "ticker": ticker,
            "side": side,
            "type": "limit",
            "action": "buy",  # Always buy yes/no contracts
            "count": count,
            "yes_price": price if side == "yes" else None,
            "no_price": price if side == "no" else None,
        }

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        if client_order_id:
            data["client_order_id"] = client_order_id

        return await self._request("POST", "/portfolio/orders", data, is_write=True)

    async def amend_order(
        self,
        order_id: str,
        price: Optional[int] = None,
        count: Optional[int] = None,
    ) -> dict:
        """
        Amend an existing order.

        Args:
            order_id: The order to amend
            price: New price (optional)
            count: New max fillable count (optional)

        Returns:
            Amended order response
        """
        data = {}
        if price is not None:
            data["price"] = price
        if count is not None:
            data["count"] = count

        return await self._request(
            "POST",
            f"/portfolio/orders/{order_id}/amend",
            data,
            is_write=True,
        )

    async def cancel_order(self, order_id: str) -> dict:
        """Cancel an order."""
        await self._rate_limiter.acquire_cancel()
        return await self._request(
            "DELETE",
            f"/portfolio/orders/{order_id}",
            is_write=False,  # Already acquired cancel token
        )

    async def cancel_all_orders(self, ticker: Optional[str] = None) -> dict:
        """Cancel all orders, optionally filtered by ticker."""
        data = {}
        if ticker:
            data["ticker"] = ticker
        return await self._request("DELETE", "/portfolio/orders", data, is_write=True)

    # -------------------------------------------------------------------------
    # WebSocket
    # -------------------------------------------------------------------------

    def on_message(self, handler: Callable[[WSMessage], None]) -> None:
        """Register a message handler."""
        self._message_handlers.append(handler)

    def on_reconnect(self, handler: Callable[[], None]) -> None:
        """Register a reconnect handler (called after successful reconnect)."""
        self._reconnect_handlers.append(handler)

    async def connect_ws(self) -> None:
        """Establish WebSocket connection."""
        headers = self._auth.get_auth_headers("GET", WS_PATH)

        logger.info(f"Connecting to WebSocket: {self.ws_url}")

        self._ws = await websockets.connect(
            self.ws_url,
            additional_headers=headers,
            ping_interval=WS_HEARTBEAT_INTERVAL,
            ping_timeout=WS_HEARTBEAT_INTERVAL * 2,
        )

        self._last_message_time = time.monotonic()
        self._reconnect_count = 0
        logger.info("WebSocket connected")

    async def disconnect_ws(self) -> None:
        """Close WebSocket connection."""
        if self._ws:
            await self._ws.close()
            self._ws = None
            logger.info("WebSocket disconnected")

    async def subscribe_orderbook(self, tickers: list[str]) -> int:
        """
        Subscribe to orderbook updates for given tickers.

        Returns:
            Subscription ID (sid)
        """
        cmd_id = self._next_cmd_id
        self._next_cmd_id += 1

        msg = {
            "id": cmd_id,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
            },
        }

        if len(tickers) == 1:
            msg["params"]["market_ticker"] = tickers[0]
        else:
            msg["params"]["market_tickers"] = tickers

        await self._send_ws(msg)

        # Store subscription for reconnect
        self._subscriptions[cmd_id] = msg["params"]

        return cmd_id

    async def subscribe_fills(self) -> int:
        """Subscribe to fill notifications."""
        cmd_id = self._next_cmd_id
        self._next_cmd_id += 1

        msg = {
            "id": cmd_id,
            "cmd": "subscribe",
            "params": {
                "channels": ["fill"],
            },
        }

        await self._send_ws(msg)
        self._subscriptions[cmd_id] = msg["params"]

        return cmd_id

    async def _send_ws(self, msg: dict) -> None:
        """Send a message over WebSocket."""
        if not self._ws:
            raise RuntimeError("WebSocket not connected")

        await self._ws.send(json.dumps(msg))
        logger.debug(f"WS send: {msg}")

    async def _receive_ws(self) -> Optional[WSMessage]:
        """Receive and parse a WebSocket message."""
        if not self._ws:
            return None

        try:
            raw = await self._ws.recv()
            self._last_message_time = time.monotonic()

            data = json.loads(raw)
            logger.debug(f"WS recv: {data}")

            return WSMessage(
                type=data.get("type", "unknown"),
                sid=data.get("sid"),
                seq=data.get("seq"),
                msg=data.get("msg", {}),
            )

        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            return None

    def _check_sequence(self, msg: WSMessage) -> bool:
        """
        Check for sequence gaps.

        Returns:
            True if sequence is OK, False if gap detected
        """
        if msg.sid is None or msg.seq is None:
            return True

        last_seq = self._sequence_numbers.get(msg.sid, 0)

        if msg.seq <= last_seq:
            # Duplicate or old message, ignore
            return True

        if last_seq > 0 and msg.seq != last_seq + 1:
            logger.warning(f"Sequence gap: expected {last_seq + 1}, got {msg.seq}")
            return False

        self._sequence_numbers[msg.sid] = msg.seq
        return True

    async def run_ws_loop(self) -> None:
        """
        Main WebSocket event loop with auto-reconnect.

        Handles:
        - Message receiving and dispatch
        - Sequence checking
        - Automatic reconnection on failure
        """
        backoff = INITIAL_BACKOFF_SEC

        while self._running:
            try:
                if not self._ws:
                    await self.connect_ws()
                    await self._resubscribe()

                    # Notify reconnect handlers
                    for handler in self._reconnect_handlers:
                        handler()

                msg = await self._receive_ws()

                if msg is None:
                    # Connection lost
                    self._ws = None
                    continue

                # Check sequence
                if not self._check_sequence(msg):
                    # Gap detected - need to reset book
                    msg = WSMessage(type="RESET_BOOK", msg={"reason": "sequence_gap"})

                # Dispatch to handlers
                for handler in self._message_handlers:
                    try:
                        handler(msg)
                    except Exception as e:
                        logger.exception(f"Handler error: {e}")

                # Reset backoff on successful message
                backoff = INITIAL_BACKOFF_SEC

            except websockets.ConnectionClosed as e:
                logger.warning(f"WebSocket closed: {e}")
                self._ws = None

            except Exception as e:
                logger.exception(f"WebSocket error: {e}")
                self._ws = None

                # Exponential backoff
                self._reconnect_count += 1
                if self._reconnect_count > MAX_RECONNECT_ATTEMPTS:
                    logger.error("Max reconnect attempts exceeded")
                    raise

                logger.info(f"Reconnecting in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_SEC)

    async def _resubscribe(self) -> None:
        """Resubscribe to all channels after reconnect."""
        for cmd_id, params in self._subscriptions.items():
            msg = {
                "id": cmd_id,
                "cmd": "subscribe",
                "params": params,
            }
            await self._send_ws(msg)
            logger.info(f"Resubscribed: {params}")


class APIError(Exception):
    """API error with status code and response."""

    def __init__(self, status: int, response: dict):
        self.status = status
        self.response = response
        super().__init__(f"API error {status}: {response}")


class RateLimitError(APIError):
    """Rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(429, {"error": message})
