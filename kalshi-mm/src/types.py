"""Core type definitions for the market maker."""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from enum import IntEnum

from .constants import Side, OrderSide, OrderStatus


class LayerIndex(IntEnum):
    """Layer indices for quote ladder."""
    CANARY = 0  # Small size, tight price - detects sweeps
    BODY = 1    # Standard size - main inventory
    WALL = 2    # Large size, deep price - capital preservation


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
class QuoteLevel:
    """A single level in the quote ladder."""
    price: int           # cents (1-99)
    size: int            # contracts
    layer_index: int     # 0=Canary, 1=Body, 2=Wall
    is_active: bool      # Whether this layer should be placed
    side: OrderSide      # BUY or SELL

    def __post_init__(self):
        if self.is_active and not 1 <= self.price <= 99:
            raise ValueError(f"Active quote price must be 1-99, got {self.price}")
        if self.size < 0:
            raise ValueError(f"Size must be non-negative, got {self.size}")


@dataclass
class QuoteLadder:
    """Multi-layer quote structure for a single market."""
    bids: list[QuoteLevel] = field(default_factory=list)  # Sorted best to worst (descending price)
    asks: list[QuoteLevel] = field(default_factory=list)  # Sorted best to worst (ascending price)
    reservation_price: float = 50.0

    @property
    def active_bids(self) -> list[QuoteLevel]:
        """Get only active bid levels."""
        return [q for q in self.bids if q.is_active]

    @property
    def active_asks(self) -> list[QuoteLevel]:
        """Get only active ask levels."""
        return [q for q in self.asks if q.is_active]

    @property
    def best_bid(self) -> Optional[QuoteLevel]:
        """Get best (highest) active bid."""
        active = self.active_bids
        return max(active, key=lambda q: q.price) if active else None

    @property
    def best_ask(self) -> Optional[QuoteLevel]:
        """Get best (lowest) active ask."""
        active = self.active_asks
        return min(active, key=lambda q: q.price) if active else None


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
    layer_index: Optional[int] = None  # 0=Canary, 1=Body, 2=Wall (v2.2)

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
