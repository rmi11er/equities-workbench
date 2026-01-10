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
            "yes": [[price, size], ...],  # bids (resting buy orders for YES)
            "no": [[price, size], ...]    # bids (resting buy orders for NO)
        }

        Note: Kalshi's snapshot shows resting bid orders.
        The ask side is derived from the opposite side's bids:
        - If someone bids YES at 38, equivalent to offering NO at 62
        - If someone bids NO at 61, equivalent to offering YES at 39
        """
        self.yes_bids.clear()
        self.yes_asks.clear()
        self.no_bids.clear()
        self.no_asks.clear()

        # YES bids (resting buy orders for YES contracts)
        for level in msg.get("yes", []):
            price, size = level[0], level[1]
            self.yes_bids[price] = size
            # Implied NO ask at (100 - price)
            # If someone bids YES at 38, they'd sell NO at 62
            self.no_asks[100 - price] = size

        # NO bids (resting buy orders for NO contracts)
        for level in msg.get("no", []):
            price, size = level[0], level[1]
            self.no_bids[price] = size
            # Implied YES ask at (100 - price)
            # If someone bids NO at 61, they'd sell YES at 39
            self.yes_asks[100 - price] = size

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

        Deltas update the bid book for the specified side,
        and the derived ask book on the opposite side.
        """
        price = msg.get("price")
        delta = msg.get("delta", 0)
        side = msg.get("side")

        if price is None or side is None:
            logger.warning(f"Invalid delta message: {msg}")
            return

        # Determine which book to update
        # Deltas are for bid orders on each side
        if side == "yes":
            book = self.yes_bids
            mirror_book = self.no_asks  # YES bid implies NO ask
            mirror_price = 100 - price
        else:
            book = self.no_bids
            mirror_book = self.yes_asks  # NO bid implies YES ask
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
        Get effective bid/ask prices that represent "real" liquidity (BBO-clamped).

        Used for MID PRICE calculation. Returns depth-weighted prices but clamped
        to BBO so the mid price never falls outside the actual market spread.

        Args:
            min_depth: Minimum contracts required to define "real" price

        Returns:
            (effective_bid, effective_ask) tuple, clamped to BBO.
        """
        best_bid = self.best_yes_bid()
        best_ask = self.best_yes_ask()

        # Get unclamped depth prices
        depth_bid, depth_ask = self.get_depth_levels(min_depth)

        # Clamp to BBO for mid price calculation
        effective_bid = max(depth_bid, best_bid) if best_bid else depth_bid
        effective_ask = min(depth_ask, best_ask) if best_ask else depth_ask

        return effective_bid, effective_ask

    def get_depth_levels(self, min_depth: int) -> tuple[int, int]:
        """
        Get the price levels where min_depth contracts exist (UNCLAMPED).

        Used for SPREAD BOUNDS. Walks the book to find where real liquidity
        exists, without clamping to BBO. This tells us the bounds within which
        we should quote - we shouldn't bid above where real bid depth exists,
        and shouldn't ask below where real ask depth exists.

        Args:
            min_depth: Minimum contracts required to define "real" price

        Returns:
            (depth_bid, depth_ask) tuple - the prices where min_depth is found.
            In thin books, these will be worse than BBO.
        """
        best_bid = self.best_yes_bid()
        best_ask = self.best_yes_ask()

        # Default to BBO (or fallback if no book)
        depth_bid = best_bid if best_bid is not None else 1
        depth_ask = best_ask if best_ask is not None else 99

        # Walk bids (descending price) to find where min_depth exists
        cumulative_bid = 0
        for price in sorted(self.yes_bids.keys(), reverse=True):
            cumulative_bid += self.yes_bids[price]
            if cumulative_bid >= min_depth:
                depth_bid = price
                break

        # Walk asks (ascending price) to find where min_depth exists
        cumulative_ask = 0
        for price in sorted(self.yes_asks.keys()):
            cumulative_ask += self.yes_asks[price]
            if cumulative_ask >= min_depth:
                depth_ask = price
                break

        return depth_bid, depth_ask

    def clamp_quotes_to_depth(
        self,
        strategy_bid: int,
        strategy_ask: int,
        min_depth: int,
    ) -> tuple[int, int]:
        """
        Clamp strategy quotes based on market liquidity depth.

        This is the core quoting logic:
        1. LIQUID BBO (depth >= min_depth at BBO): Tighten to BBO
           - In liquid markets, we want to quote AT the BBO
           - bid = max(strategy_bid, best_bid) - push UP to BBO
           - ask = min(strategy_ask, best_ask) - push DOWN to BBO

        2. THIN BBO (depth < min_depth at BBO): Widen to depth levels
           - In thin markets, we quote at where real liquidity exists
           - bid = min(strategy_bid, depth_bid) - push DOWN to depth
           - ask = max(strategy_ask, depth_ask) - push UP to depth

        Args:
            strategy_bid: The bid price from strategy (e.g., Stoikov)
            strategy_ask: The ask price from strategy
            min_depth: Minimum contracts to consider "liquid"

        Returns:
            (clamped_bid, clamped_ask) - the final quote prices
        """
        best_bid = self.best_yes_bid()
        best_ask = self.best_yes_ask()

        # Handle empty/one-sided books
        if best_bid is None and best_ask is None:
            return 1, 99

        # Get depth levels (where min_depth contracts exist)
        depth_bid, depth_ask = self.get_depth_levels(min_depth)

        # Determine liquidity state for each side and clamp accordingly
        clamped_bid = self._clamp_bid(strategy_bid, best_bid, depth_bid)
        clamped_ask = self._clamp_ask(strategy_ask, best_ask, depth_ask)

        # Ensure quotes don't cross
        if clamped_bid >= clamped_ask:
            # Try to use depth levels
            if depth_bid < depth_ask:
                clamped_bid = depth_bid
                clamped_ask = depth_ask
            # Fall back to BBO if available
            elif best_bid is not None and best_ask is not None and best_bid < best_ask:
                clamped_bid = best_bid
                clamped_ask = best_ask
            # Last resort
            else:
                clamped_bid = 1
                clamped_ask = 99

        return clamped_bid, clamped_ask

    def _clamp_bid(
        self,
        strategy_bid: int,
        best_bid: Optional[int],
        depth_bid: int,
    ) -> int:
        """
        Clamp bid based on liquidity.

        - If BBO is liquid (depth_bid == best_bid): push UP to BBO
        - If BBO is thin (depth_bid < best_bid): push DOWN to depth
        """
        if best_bid is None:
            return 1  # No bids, use fallback

        if depth_bid == best_bid:
            # Liquid at BBO - tighten to BBO (push bid UP)
            return max(strategy_bid, best_bid)
        else:
            # Thin at BBO - widen to depth (push bid DOWN)
            return min(strategy_bid, depth_bid)

    def _clamp_ask(
        self,
        strategy_ask: int,
        best_ask: Optional[int],
        depth_ask: int,
    ) -> int:
        """
        Clamp ask based on liquidity.

        - If BBO is liquid (depth_ask == best_ask): push DOWN to BBO
        - If BBO is thin (depth_ask > best_ask): push UP to depth
        """
        if best_ask is None:
            return 99  # No asks, use fallback

        if depth_ask == best_ask:
            # Liquid at BBO - tighten to BBO (push ask DOWN)
            return min(strategy_ask, best_ask)
        else:
            # Thin at BBO - widen to depth (push ask UP)
            return max(strategy_ask, depth_ask)

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
