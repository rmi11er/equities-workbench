"""Core type definitions for the market maker."""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from .constants import Side, OrderSide, OrderStatus


@dataclass
class PriceLevel:
    """A single price level in the order book."""
    price: int  # cents (1-99)
    size: int   # contracts


@dataclass
class OrderBookState:
    """Snapshot of order book state for one side."""
    levels: dict[int, int] = field(default_factory=dict)  # price -> size

    def best_price(self, is_bid: bool) -> Optional[int]:
        """Get best bid (max) or ask (min) price."""
        if not self.levels:
            return None
        return max(self.levels.keys()) if is_bid else min(self.levels.keys())

    def size_at(self, price: int) -> int:
        """Get size at a price level, 0 if not present."""
        return self.levels.get(price, 0)


@dataclass
class Quote:
    """A quote to place in the market."""
    side: OrderSide
    price: int  # cents
    size: int   # contracts

    def __post_init__(self):
        if not 1 <= self.price <= 99:
            raise ValueError(f"Price must be 1-99, got {self.price}")
        if self.size < 0:
            raise ValueError(f"Size must be non-negative, got {self.size}")


@dataclass
class Order:
    """Represents an order in the market."""
    order_id: str
    ticker: str
    side: OrderSide
    price: int
    size: int
    remaining: int
    filled: int
    status: OrderStatus
    client_order_id: Optional[str] = None  # For Kalshi amend API
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.RESTING, OrderStatus.PENDING)


@dataclass
class Fill:
    """Represents an order fill."""
    order_id: str
    ticker: str
    side: OrderSide
    price: int
    size: int
    timestamp: datetime


@dataclass
class Position:
    """Current position in a market."""
    ticker: str
    yes_contracts: int = 0
    no_contracts: int = 0

    @property
    def net_position(self) -> int:
        """Net position: positive = long yes, negative = long no."""
        return self.yes_contracts - self.no_contracts

    @property
    def total_exposure(self) -> int:
        """Total contracts held (absolute)."""
        return self.yes_contracts + self.no_contracts


@dataclass
class StrategyOutput:
    """Output from the strategy engine."""
    bid_price: int
    ask_price: int
    bid_size: int
    ask_size: int
    reservation_price: float
    spread: float
    inventory_skew: float


@dataclass
class TapeEntry:
    """Single entry for the data tape (CSV logging)."""
    ts: datetime
    ticker: str
    mid: float
    my_bid: Optional[int]
    my_ask: Optional[int]
    inventory: int
    unrealized_pnl: float
    realized_pnl: float
    latency_ms: float
    volatility: float
