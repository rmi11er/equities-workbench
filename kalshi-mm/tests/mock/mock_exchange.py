"""Mock Kalshi exchange for testing."""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime


@dataclass
class MockOrder:
    """Mock order representation."""
    order_id: str
    ticker: str
    side: str  # "yes" or "no"
    price: int
    count: int
    remaining: int
    filled: int = 0
    status: str = "resting"
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MockPosition:
    """Mock position."""
    ticker: str
    yes_contracts: int = 0
    no_contracts: int = 0


class MockKalshiExchange:
    """
    Mock Kalshi exchange for testing.

    Simulates:
    - Order placement, amendment, cancellation
    - Order matching
    - WebSocket message generation
    - Order book state
    """

    def __init__(self):
        self._orders: dict[str, MockOrder] = {}
        self._positions: dict[str, MockPosition] = {}
        self._orderbooks: dict[str, dict] = {}
        self._balance: float = 10000.0  # $100 in cents

        # WebSocket simulation
        self._ws_handlers: list[Callable] = []
        self._sequence: int = 0

        # Initialize a test order book
        self.set_orderbook("TEST-TICKER", {
            "yes": [[30, 100], [35, 200], [40, 150]],
            "no": [[55, 100], [60, 200], [65, 150]],
        })

    # -------------------------------------------------------------------------
    # Order Book
    # -------------------------------------------------------------------------

    def set_orderbook(self, ticker: str, book: dict) -> None:
        """Set order book state."""
        self._orderbooks[ticker] = book
        self._send_ws_message({
            "type": "orderbook_snapshot",
            "seq": self._next_seq(),
            "msg": {
                "market_ticker": ticker,
                **book,
            },
        })

    def update_orderbook(self, ticker: str, side: str, price: int, delta: int) -> None:
        """Send an order book delta."""
        self._send_ws_message({
            "type": "orderbook_delta",
            "seq": self._next_seq(),
            "msg": {
                "market_ticker": ticker,
                "side": side,
                "price": price,
                "delta": delta,
            },
        })

    def _next_seq(self) -> int:
        self._sequence += 1
        return self._sequence

    # -------------------------------------------------------------------------
    # REST API Simulation
    # -------------------------------------------------------------------------

    async def get_balance(self) -> dict:
        """Get mock balance."""
        return {"balance": self._balance}

    async def get_positions(self) -> dict:
        """Get mock positions."""
        return {
            "positions": [
                {
                    "ticker": p.ticker,
                    "yes_contracts": p.yes_contracts,
                    "no_contracts": p.no_contracts,
                }
                for p in self._positions.values()
            ]
        }

    async def get_orders(self, ticker: Optional[str] = None) -> dict:
        """Get mock orders."""
        orders = list(self._orders.values())
        if ticker:
            orders = [o for o in orders if o.ticker == ticker]

        return {
            "orders": [
                {
                    "order_id": o.order_id,
                    "ticker": o.ticker,
                    "side": o.side,
                    "price": o.price,
                    "count": o.count,
                    "remaining_count": o.remaining,
                    "fill_count": o.filled,
                    "status": o.status,
                }
                for o in orders
            ]
        }

    async def create_order(
        self,
        ticker: str,
        side: str,
        price: int,
        count: int,
        client_order_id: Optional[str] = None,
    ) -> dict:
        """Create a mock order."""
        order_id = str(uuid.uuid4())[:8]

        order = MockOrder(
            order_id=order_id,
            ticker=ticker,
            side=side,
            price=price,
            count=count,
            remaining=count,
        )

        self._orders[order_id] = order

        # Simulate matching (simplified)
        await self._try_match(order)

        return {
            "order": {
                "order_id": order_id,
                "ticker": ticker,
                "side": side,
                "price": price,
                "count": count,
                "remaining_count": order.remaining,
                "fill_count": order.filled,
                "status": order.status,
            }
        }

    async def amend_order(
        self,
        order_id: str,
        price: Optional[int] = None,
        count: Optional[int] = None,
    ) -> dict:
        """Amend a mock order."""
        if order_id not in self._orders:
            raise ValueError(f"Order not found: {order_id}")

        order = self._orders[order_id]

        if price is not None:
            order.price = price
        if count is not None:
            order.remaining = count - order.filled

        return {
            "order": {
                "order_id": order_id,
                "price": order.price,
                "remaining_count": order.remaining,
            }
        }

    async def cancel_order(self, order_id: str) -> dict:
        """Cancel a mock order."""
        if order_id not in self._orders:
            raise ValueError(f"Order not found: {order_id}")

        order = self._orders[order_id]
        order.status = "cancelled"
        order.remaining = 0

        return {"order_id": order_id, "status": "cancelled"}

    async def cancel_all_orders(self, ticker: Optional[str] = None) -> dict:
        """Cancel all mock orders."""
        cancelled = []
        for order in self._orders.values():
            if ticker is None or order.ticker == ticker:
                if order.status == "resting":
                    order.status = "cancelled"
                    order.remaining = 0
                    cancelled.append(order.order_id)

        return {"cancelled": cancelled}

    # -------------------------------------------------------------------------
    # Matching Engine (Simplified)
    # -------------------------------------------------------------------------

    async def _try_match(self, order: MockOrder) -> None:
        """Try to match an order against the book."""
        # Simplified: no actual matching, just leave as resting
        # In a more complete mock, we'd check against the book
        pass

    def fill_order(self, order_id: str, size: int, price: int) -> None:
        """Simulate a fill (for testing)."""
        if order_id not in self._orders:
            return

        order = self._orders[order_id]
        fill_size = min(size, order.remaining)

        order.filled += fill_size
        order.remaining -= fill_size

        if order.remaining == 0:
            order.status = "executed"

        # Update position
        if order.ticker not in self._positions:
            self._positions[order.ticker] = MockPosition(ticker=order.ticker)

        pos = self._positions[order.ticker]
        if order.side == "yes":
            pos.yes_contracts += fill_size
        else:
            pos.no_contracts += fill_size

        # Send fill message
        self._send_ws_message({
            "type": "fill",
            "msg": {
                "order_id": order_id,
                "ticker": order.ticker,
                "side": order.side,
                "price": price,
                "count": fill_size,
            },
        })

    # -------------------------------------------------------------------------
    # WebSocket Simulation
    # -------------------------------------------------------------------------

    def on_ws_message(self, handler: Callable) -> None:
        """Register a WebSocket message handler."""
        self._ws_handlers.append(handler)

    def _send_ws_message(self, msg: dict) -> None:
        """Send a message to all handlers."""
        for handler in self._ws_handlers:
            handler(msg)

    async def simulate_price_move(self, ticker: str, new_mid: float) -> None:
        """Simulate a price move (update order book)."""
        # Clear and set new book around new mid
        spread = 5
        yes_levels = [
            [int(new_mid + i), 100] for i in range(1, 4)
        ]
        no_levels = [
            [100 - int(new_mid) + i, 100] for i in range(1, 4)
        ]

        self.set_orderbook(ticker, {
            "yes": yes_levels,
            "no": no_levels,
        })

    async def simulate_volatility_spike(self, ticker: str) -> None:
        """Simulate rapid price moves to spike volatility."""
        for _ in range(10):
            import random
            delta = random.choice([-3, -2, -1, 1, 2, 3])
            self.update_orderbook(ticker, "yes", 50 + delta, 100)
            await asyncio.sleep(0.01)


class MockConnector:
    """
    Mock connector that wraps MockKalshiExchange.

    Drop-in replacement for KalshiConnector in tests.
    """

    def __init__(self, exchange: MockKalshiExchange):
        self._exchange = exchange
        self._message_handlers: list[Callable] = []

        # Wire up exchange messages
        exchange.on_ws_message(self._on_exchange_message)

    def _on_exchange_message(self, msg: dict) -> None:
        """Handle exchange message and convert to WSMessage."""
        from src.connector import WSMessage

        ws_msg = WSMessage(
            type=msg.get("type", "unknown"),
            seq=msg.get("seq"),
            sid=msg.get("sid"),
            msg=msg.get("msg", {}),
        )

        for handler in self._message_handlers:
            handler(ws_msg)

    def on_message(self, handler: Callable) -> None:
        self._message_handlers.append(handler)

    def on_reconnect(self, handler: Callable) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def connect_ws(self) -> None:
        pass

    async def subscribe_orderbook(self, tickers: list[str]) -> int:
        return 1

    async def subscribe_fills(self) -> int:
        return 2

    async def get_balance(self) -> dict:
        return await self._exchange.get_balance()

    async def get_positions(self) -> dict:
        return await self._exchange.get_positions()

    async def get_orders(self, ticker: Optional[str] = None) -> dict:
        return await self._exchange.get_orders(ticker)

    async def create_order(self, ticker: str, side: str, price: int, count: int, **kwargs) -> dict:
        return await self._exchange.create_order(ticker, side, price, count)

    async def amend_order(self, order_id: str, price: Optional[int] = None, count: Optional[int] = None) -> dict:
        return await self._exchange.amend_order(order_id, price, count)

    async def cancel_order(self, order_id: str) -> dict:
        return await self._exchange.cancel_order(order_id)

    async def cancel_all_orders(self, ticker: Optional[str] = None) -> dict:
        return await self._exchange.cancel_all_orders(ticker)
