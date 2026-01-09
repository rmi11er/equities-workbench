"""Order book manager with volatility estimation."""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

from .connector import WSMessage
from .config import VolatilityConfig

logger = logging.getLogger(__name__)


@dataclass
class OrderBook:
    """
    Maintains authoritative market state for a single ticker.

    Kalshi binary markets have YES and NO sides. We track both.
    Prices are in cents (1-99).

    The YES bid/ask spread and NO bid/ask spread are related:
    - YES price + NO price = 100 (approximately, minus spread)
    """
    ticker: str

    # YES side: bids want to buy YES, asks want to sell YES
    yes_bids: dict[int, int] = field(default_factory=dict)  # price -> size
    yes_asks: dict[int, int] = field(default_factory=dict)  # price -> size

    # NO side: bids want to buy NO, asks want to sell NO
    no_bids: dict[int, int] = field(default_factory=dict)
    no_asks: dict[int, int] = field(default_factory=dict)

    # Sequence tracking
    version: int = 0

    def apply_snapshot(self, msg: dict) -> None:
        """
        Apply a full orderbook snapshot.

        Expected format:
        {
            "market_ticker": "...",
            "yes": [[price, size], ...],  # asks (offers to sell YES)
            "no": [[price, size], ...]    # asks (offers to sell NO)
        }

        Note: Kalshi's snapshot shows the offer side.
        The bid side is derived: if someone offers YES at 60,
        that's equivalent to bidding NO at 40.
        """
        self.yes_bids.clear()
        self.yes_asks.clear()
        self.no_bids.clear()
        self.no_asks.clear()

        # YES offers (these are asks on the YES side)
        for level in msg.get("yes", []):
            price, size = level[0], level[1]
            self.yes_asks[price] = size
            # Equivalent to NO bid at (100 - price)
            self.no_bids[100 - price] = size

        # NO offers (these are asks on the NO side)
        for level in msg.get("no", []):
            price, size = level[0], level[1]
            self.no_asks[price] = size
            # Equivalent to YES bid at (100 - price)
            self.yes_bids[100 - price] = size

        self.version += 1
        logger.debug(f"Snapshot applied for {self.ticker}, version={self.version}")

    def apply_delta(self, msg: dict) -> None:
        """
        Apply an incremental orderbook update.

        Expected format:
        {
            "market_ticker": "...",
            "price": 96,
            "delta": -54,  # negative = size decrease, positive = increase
            "side": "yes"  # or "no"
        }
        """
        price = msg.get("price")
        delta = msg.get("delta", 0)
        side = msg.get("side")

        if price is None or side is None:
            logger.warning(f"Invalid delta message: {msg}")
            return

        # Determine which book to update
        if side == "yes":
            book = self.yes_asks
            mirror_book = self.no_bids
            mirror_price = 100 - price
        else:
            book = self.no_asks
            mirror_book = self.yes_bids
            mirror_price = 100 - price

        # Update the book
        current = book.get(price, 0)
        new_size = current + delta

        if new_size <= 0:
            book.pop(price, None)
            mirror_book.pop(mirror_price, None)
        else:
            book[price] = new_size
            mirror_book[mirror_price] = new_size

        self.version += 1

    def best_yes_bid(self) -> Optional[int]:
        """Best price someone will pay for YES contracts."""
        return max(self.yes_bids.keys()) if self.yes_bids else None

    def best_yes_ask(self) -> Optional[int]:
        """Best price someone will sell YES contracts."""
        return min(self.yes_asks.keys()) if self.yes_asks else None

    def best_no_bid(self) -> Optional[int]:
        """Best price someone will pay for NO contracts."""
        return max(self.no_bids.keys()) if self.no_bids else None

    def best_no_ask(self) -> Optional[int]:
        """Best price someone will sell NO contracts."""
        return min(self.no_asks.keys()) if self.no_asks else None

    def mid_price(self) -> Optional[float]:
        """
        Calculate mid price for YES contracts.

        Uses the YES bid/ask if available, otherwise derives from NO.
        Returns None if no quotes available.
        """
        bid = self.best_yes_bid()
        ask = self.best_yes_ask()

        if bid is not None and ask is not None:
            return (bid + ask) / 2.0

        # Try to derive from NO side
        no_bid = self.best_no_bid()
        no_ask = self.best_no_ask()

        if no_bid is not None and no_ask is not None:
            # YES mid = 100 - NO mid
            no_mid = (no_bid + no_ask) / 2.0
            return 100.0 - no_mid

        # Partial information
        if bid is not None:
            return float(bid)
        if ask is not None:
            return float(ask)

        return None

    def spread(self) -> Optional[float]:
        """Calculate YES bid-ask spread in cents."""
        bid = self.best_yes_bid()
        ask = self.best_yes_ask()

        if bid is not None and ask is not None:
            return float(ask - bid)
        return None

    def is_valid(self) -> bool:
        """
        Check if the orderbook is in a valid, tradeable state.

        Returns False if:
        - Book is empty (no bids or asks)
        - Book is crossed (bid >= ask)
        - Prices are out of bounds

        This MUST be checked before quoting to prevent trading on garbage data.
        """
        bid = self.best_yes_bid()
        ask = self.best_yes_ask()

        # Empty book
        if bid is None or ask is None:
            return False

        # Crossed book (bid >= ask means impossible market)
        if bid >= ask:
            logger.warning(f"CROSSED BOOK DETECTED: bid={bid} >= ask={ask}")
            return False

        # Sanity check: prices should be 1-99
        if bid < 1 or bid > 99 or ask < 1 or ask > 99:
            logger.warning(f"INVALID PRICES: bid={bid}, ask={ask}")
            return False

        return True

    def get_ofi(self, levels: int = 3) -> float:
        """
        Calculate Order Flow Imbalance.

        OFI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        Returns value in [-1, 1], positive = more bid pressure.
        """
        # Get top N levels of bids
        bid_prices = sorted(self.yes_bids.keys(), reverse=True)[:levels]
        bid_volume = sum(self.yes_bids[p] for p in bid_prices)

        # Get top N levels of asks
        ask_prices = sorted(self.yes_asks.keys())[:levels]
        ask_volume = sum(self.yes_asks[p] for p in ask_prices)

        total = bid_volume + ask_volume
        if total == 0:
            return 0.0

        return (bid_volume - ask_volume) / total

    def get_effective_quote(self, min_depth: int) -> tuple[int, int]:
        """
        Get effective bid/ask prices that represent "real" liquidity.

        Prevents the bot from reacting to "dust" (1-2 contracts) in thin markets.
        Walks through the order book until cumulative volume >= min_depth.

        Args:
            min_depth: Minimum contracts required to define "real" price

        Returns:
            (effective_bid, effective_ask) tuple. If entire book < min_depth,
            returns worst available prices or (1, 99) as fallback.
        """
        # Walk bids (descending price) until we hit min_depth
        effective_bid = 1  # Fallback
        cumulative_bid = 0
        for price in sorted(self.yes_bids.keys(), reverse=True):
            cumulative_bid += self.yes_bids[price]
            effective_bid = price
            if cumulative_bid >= min_depth:
                break

        # If we didn't find enough depth, use worst bid or fallback
        if cumulative_bid < min_depth:
            if self.yes_bids:
                effective_bid = min(self.yes_bids.keys())
            else:
                effective_bid = 1

        # Walk asks (ascending price) until we hit min_depth
        effective_ask = 99  # Fallback
        cumulative_ask = 0
        for price in sorted(self.yes_asks.keys()):
            cumulative_ask += self.yes_asks[price]
            effective_ask = price
            if cumulative_ask >= min_depth:
                break

        # If we didn't find enough depth, use worst ask or fallback
        if cumulative_ask < min_depth:
            if self.yes_asks:
                effective_ask = max(self.yes_asks.keys())
            else:
                effective_ask = 99

        return effective_bid, effective_ask

    def get_effective_spread(self, min_depth: int) -> float:
        """
        Calculate effective spread based on depth-weighted prices.

        Args:
            min_depth: Minimum contracts to consider for effective price

        Returns:
            Effective spread in cents
        """
        eff_bid, eff_ask = self.get_effective_quote(min_depth)
        return float(eff_ask - eff_bid)

    def get_effective_mid(self, min_depth: int) -> float:
        """
        Calculate effective mid price based on depth-weighted prices.

        Args:
            min_depth: Minimum contracts to consider for effective price

        Returns:
            Effective mid price
        """
        eff_bid, eff_ask = self.get_effective_quote(min_depth)
        return (eff_bid + eff_ask) / 2.0


class VolatilityEstimator:
    """
    Estimates realized tick volatility using EMA.

    Uses price changes between ticks to estimate volatility,
    with exponential smoothing based on time elapsed.
    """

    def __init__(self, config: VolatilityConfig):
        self.config = config
        self._variance_ema: float = config.initial_volatility ** 2
        self._last_price: Optional[float] = None
        self._last_time: float = 0.0

    @property
    def volatility(self) -> float:
        """Current volatility estimate (standard deviation in cents)."""
        vol = math.sqrt(self._variance_ema)
        return max(vol, self.config.min_volatility)

    def update(self, price: float) -> float:
        """
        Update volatility estimate with new price tick.

        Args:
            price: New mid price in cents

        Returns:
            Updated volatility estimate
        """
        now = time.monotonic()

        if self._last_price is not None:
            # Calculate return (price change)
            price_change = price - self._last_price

            # Time-weighted alpha
            # Alpha decays with half-life, but we need to adjust for tick timing
            dt = now - self._last_time if self._last_time > 0 else 1.0

            # Compute alpha based on time elapsed
            # alpha = 1 - exp(-ln(2) * dt / halflife)
            alpha = 1 - math.exp(-math.log(2) * dt / self.config.ema_halflife_sec)

            # EMA update: variance = alpha * (return^2) + (1-alpha) * variance
            self._variance_ema = (
                alpha * (price_change ** 2) +
                (1 - alpha) * self._variance_ema
            )

        self._last_price = price
        self._last_time = now

        return self.volatility

    def reset(self, initial_price: Optional[float] = None) -> None:
        """Reset the estimator."""
        self._variance_ema = self.config.initial_volatility ** 2
        self._last_price = initial_price
        self._last_time = time.monotonic()


class OrderBookManager:
    """
    Manages order books for multiple tickers with volatility estimation.
    """

    def __init__(self, volatility_config: VolatilityConfig):
        self._books: dict[str, OrderBook] = {}
        self._volatility: dict[str, VolatilityEstimator] = {}
        self._volatility_config = volatility_config

    def get_or_create(self, ticker: str) -> OrderBook:
        """Get or create an order book for a ticker."""
        if ticker not in self._books:
            self._books[ticker] = OrderBook(ticker=ticker)
            self._volatility[ticker] = VolatilityEstimator(self._volatility_config)
        return self._books[ticker]

    def get(self, ticker: str) -> Optional[OrderBook]:
        """Get order book for a ticker, or None if not exists."""
        return self._books.get(ticker)

    def get_volatility(self, ticker: str) -> float:
        """Get current volatility estimate for a ticker."""
        if ticker in self._volatility:
            return self._volatility[ticker].volatility
        return self._volatility_config.initial_volatility

    def handle_message(self, msg: WSMessage) -> None:
        """
        Process a WebSocket message and update order books.

        Handles:
        - orderbook_snapshot
        - orderbook_delta
        - RESET_BOOK (internal signal for sequence gaps)
        """
        if msg.type == "orderbook_snapshot":
            ticker = msg.msg.get("market_ticker")
            if ticker:
                book = self.get_or_create(ticker)
                book.apply_snapshot(msg.msg)

                # Update volatility with new mid
                mid = book.mid_price()
                if mid is not None:
                    self._volatility[ticker].reset(mid)

                logger.info(f"OrderBook snapshot: {ticker}, mid={mid}")

        elif msg.type == "orderbook_delta":
            ticker = msg.msg.get("market_ticker")
            if ticker:
                book = self.get_or_create(ticker)
                book.apply_delta(msg.msg)

                # Update volatility
                mid = book.mid_price()
                if mid is not None:
                    vol = self._volatility[ticker].update(mid)
                    logger.debug(f"OrderBook delta: {ticker}, mid={mid:.2f}, vol={vol:.3f}")

        elif msg.type == "RESET_BOOK":
            # Clear all books - will be repopulated on next snapshot
            logger.warning(f"Resetting order books: {msg.msg.get('reason')}")
            for book in self._books.values():
                book.yes_bids.clear()
                book.yes_asks.clear()
                book.no_bids.clear()
                book.no_asks.clear()
                book.version = 0

    def reset(self, ticker: Optional[str] = None) -> None:
        """Reset order book(s) and volatility estimator(s)."""
        if ticker:
            if ticker in self._books:
                del self._books[ticker]
            if ticker in self._volatility:
                del self._volatility[ticker]
        else:
            self._books.clear()
            self._volatility.clear()
