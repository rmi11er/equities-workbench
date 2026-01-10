"""Fill snapshot logger for post-trade analysis."""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Correlated Market Detection
# -----------------------------------------------------------------------------

def parse_correlated_tickers(ticker: str) -> tuple[list[str], Optional[str]]:
    """
    Parse a ticker to find correlated market tickers.

    Handles patterns like:
    - Game markets: KXNFLGAME-26JAN10LACAR-CAR → KXNFLGAME-26JAN10LACAR-LA
    - Championship/MVP: KXNFLMVP-26-MSTA → root "KXNFLMVP-26-" (need API lookup)

    Returns:
        (correlated_tickers, market_root)
        - correlated_tickers: list of known correlated tickers (for 2-team games)
        - market_root: prefix for multi-outcome markets (MVP, championship)
                       Use this to find other outcomes via API or subscribed markets
    """
    correlated = []
    market_root = None

    # Pattern 1: Two-team game markets
    # Format: {PREFIX}-{DATE}{TEAMS}-{OUTCOME}
    # Example: KXNFLGAME-26JAN10LACAR-CAR (LA vs CAR, outcome=CAR)
    # The OUTCOME at the end tells us who this ticker represents
    game_match = re.match(
        r'^(KX[A-Z]+GAME-\d{2}[A-Z]{3}\d{2})([A-Z]+)-([A-Z]{2,4})$',
        ticker
    )
    if game_match:
        prefix = game_match.group(1)  # KXNFLGAME-26JAN10
        teams = game_match.group(2)    # LACAR
        outcome = game_match.group(3)  # CAR

        # The outcome is one team, find the other team in the combined string
        # LACAR with outcome CAR -> other team is LA
        # LACAR with outcome LA -> other team is CAR (but wait, LA isn't at start/end cleanly)
        # Try both positions
        if teams.endswith(outcome):
            other_team = teams[:-len(outcome)]
        elif teams.startswith(outcome):
            other_team = teams[len(outcome):]
        else:
            # Can't parse, skip
            return correlated, None

        if other_team:
            correlated.append(f"{prefix}{teams}-{other_team}")
        return correlated, None

    # Pattern 2: Multi-outcome markets (MVP, Championship, Winner)
    # Format: {PREFIX}-{YEAR}-{OUTCOME}
    # Example: KXNFLMVP-26-MSTA, KXNFLAFCCHAMP-25-BUF
    # For these, we can't enumerate outcomes from ticker alone - return root for lookup
    multi_match = re.match(
        r'^(KX[A-Z]+(?:CHAMP|WINNER|MVP|CONF)-\d{2})-([A-Z0-9]+)$',
        ticker
    )
    if multi_match:
        market_root = multi_match.group(1) + "-"
        # Can't determine correlated tickers without API/subscription lookup
        # Caller should use market_root to find others
        return correlated, market_root

    # Pattern 3: Total points markets (over/under)
    # Format: {PREFIX}-{DETAILS}-{NUMBER}
    # Example: KXNCAAFTOTAL-26JAN09OREIND-46
    # These don't have simple correlations - skip for now

    return correlated, market_root


@dataclass
class CorrelatedMarketSnapshot:
    """Snapshot of a correlated market at fill time."""
    ticker: str
    best_bid: Optional[int] = None
    best_ask: Optional[int] = None
    mid: Optional[float] = None
    spread: Optional[int] = None

    # Implied fair value comparison
    # For two-team games: our_mid + correlated_mid should ≈ 100
    implied_sum: Optional[float] = None  # our_mid + this_mid
    arb_signal: Optional[float] = None   # deviation from 100


@dataclass
class OrderBookSnapshot:
    """Snapshot of orderbook state at fill time."""
    best_bid: Optional[int] = None
    best_ask: Optional[int] = None
    spread: Optional[int] = None
    mid: Optional[float] = None

    # Depth at best levels
    bid_size_at_best: int = 0
    ask_size_at_best: int = 0

    # Cumulative depth (top 5 levels)
    bid_depth_5: int = 0  # Total contracts within 5 cents of best bid
    ask_depth_5: int = 0  # Total contracts within 5 cents of best ask

    # Full top-of-book (list of [price, size] pairs)
    bid_levels: list = field(default_factory=list)  # [(price, size), ...]
    ask_levels: list = field(default_factory=list)  # [(price, size), ...]


@dataclass
class QuoteSnapshot:
    """Snapshot of our quotes at fill time."""
    bid_price: Optional[int] = None
    bid_size: Optional[int] = None
    ask_price: Optional[int] = None
    ask_size: Optional[int] = None

    # Distance from BBO
    bid_distance_from_best: Optional[int] = None  # negative = behind best, 0 = at best
    ask_distance_from_best: Optional[int] = None


@dataclass
class FillSnapshot:
    """Complete snapshot of market state at fill time."""
    # Metadata
    timestamp: str
    ticker: str

    # Fill details
    order_id: str
    trade_id: str
    side: str  # "yes" or "no"
    action: str  # "buy" or "sell"
    fill_price: int
    fill_size: int
    is_taker: bool

    # Position
    inventory_before: int
    inventory_after: int

    # Market state
    orderbook: OrderBookSnapshot = field(default_factory=OrderBookSnapshot)
    our_quotes: QuoteSnapshot = field(default_factory=QuoteSnapshot)

    # Context
    seconds_since_last_fill: Optional[float] = None
    fills_in_last_minute: int = 0

    # Correlated markets (added post-fill for analysis)
    correlated_markets: list = field(default_factory=list)  # list[CorrelatedMarketSnapshot]
    market_root: Optional[str] = None  # For multi-outcome markets (MVP, etc.) - use for API lookup


class FillLogger:
    """
    Logs detailed fill snapshots for post-trade analysis.

    Writes JSON-lines format for easy parsing with pandas:
        df = pd.read_json('fills.jsonl', lines=True)
    """

    def __init__(self, log_path: Path, orderbook_manager=None):
        """
        Initialize the fill logger.

        Args:
            log_path: Path to the fills.jsonl file
            orderbook_manager: Optional OrderBookManager to look up correlated markets
                               we're already subscribed to
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.log_path, 'a')
        self._last_fill_time: dict[str, float] = {}  # ticker -> timestamp
        self._recent_fills: list[float] = []  # timestamps of recent fills
        self._orderbook_manager = orderbook_manager

        logger.info(f"Fill logger initialized: {self.log_path}")

    def log_fill(
        self,
        ticker: str,
        fill_data: dict,
        orderbook,  # OrderBook instance
        quote_state,  # QuoteState from execution engine
        inventory_before: int,
        inventory_after: int,
    ) -> None:
        """
        Log a fill with full market context.

        Args:
            ticker: Market ticker
            fill_data: Raw fill data from WebSocket
            orderbook: OrderBook instance for this ticker
            quote_state: QuoteState from execution engine
            inventory_before: Position before fill
            inventory_after: Position after fill
        """
        now = datetime.now()
        now_ts = now.timestamp()

        # Build orderbook snapshot
        book_snapshot = self._snapshot_orderbook(orderbook)

        # Build quote snapshot
        quote_snapshot = self._snapshot_quotes(quote_state, book_snapshot)

        # Calculate timing context
        last_fill = self._last_fill_time.get(ticker)
        seconds_since_last = (now_ts - last_fill) if last_fill else None

        # Count recent fills (last 60 seconds)
        cutoff = now_ts - 60
        self._recent_fills = [t for t in self._recent_fills if t > cutoff]
        self._recent_fills.append(now_ts)

        # Look up correlated markets
        correlated_tickers, market_root = parse_correlated_tickers(ticker)
        correlated_snapshots = self._snapshot_correlated_markets(
            correlated_tickers, market_root, book_snapshot.mid
        )

        # Build snapshot
        snapshot = FillSnapshot(
            timestamp=now.isoformat(),
            ticker=ticker,
            order_id=fill_data.get("order_id", ""),
            trade_id=fill_data.get("trade_id", ""),
            side=fill_data.get("purchased_side", fill_data.get("side", "")),
            action=fill_data.get("action", ""),
            fill_price=fill_data.get("yes_price", 0),
            fill_size=fill_data.get("count", 0),
            is_taker=fill_data.get("is_taker", False),
            inventory_before=inventory_before,
            inventory_after=inventory_after,
            orderbook=book_snapshot,
            our_quotes=quote_snapshot,
            seconds_since_last_fill=seconds_since_last,
            fills_in_last_minute=len(self._recent_fills),
            correlated_markets=correlated_snapshots,
            market_root=market_root,
        )

        # Update last fill time
        self._last_fill_time[ticker] = now_ts

        # Write to file
        self._write(snapshot)

        logger.debug(f"Logged fill snapshot: {ticker} {snapshot.side} {snapshot.fill_size}@{snapshot.fill_price}")

    def _snapshot_orderbook(self, orderbook) -> OrderBookSnapshot:
        """Extract orderbook state."""
        if orderbook is None:
            return OrderBookSnapshot()

        # Use correct method names for OrderBook class
        best_bid = orderbook.best_yes_bid()
        best_ask = orderbook.best_yes_ask()

        # Get bid levels (sorted high to low)
        # OrderBook uses yes_bids dict directly, not bids.levels
        bid_levels = []
        bid_depth_5 = 0
        bid_size_at_best = 0
        if orderbook.yes_bids:
            sorted_bids = sorted(orderbook.yes_bids.items(), reverse=True)
            for price, size in sorted_bids[:10]:  # Top 10 levels
                bid_levels.append([price, size])
                if best_bid and price >= best_bid - 5:
                    bid_depth_5 += size
                if price == best_bid:
                    bid_size_at_best = size

        # Get ask levels (sorted low to high)
        # OrderBook uses yes_asks dict directly
        ask_levels = []
        ask_depth_5 = 0
        ask_size_at_best = 0
        if orderbook.yes_asks:
            sorted_asks = sorted(orderbook.yes_asks.items())
            for price, size in sorted_asks[:10]:  # Top 10 levels
                ask_levels.append([price, size])
                if best_ask and price <= best_ask + 5:
                    ask_depth_5 += size
                if price == best_ask:
                    ask_size_at_best = size

        spread = (best_ask - best_bid) if (best_bid and best_ask) else None
        mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else None

        return OrderBookSnapshot(
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            mid=mid,
            bid_size_at_best=bid_size_at_best,
            ask_size_at_best=ask_size_at_best,
            bid_depth_5=bid_depth_5,
            ask_depth_5=ask_depth_5,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
        )

    def _snapshot_quotes(self, quote_state, book: OrderBookSnapshot) -> QuoteSnapshot:
        """Extract our quote state."""
        if quote_state is None:
            return QuoteSnapshot()

        bid_price = None
        bid_size = None
        ask_price = None
        ask_size = None

        if quote_state.bid_order and quote_state.bid_order.is_active:
            bid_price = quote_state.bid_order.price
            bid_size = quote_state.bid_order.remaining

        if quote_state.ask_order and quote_state.ask_order.is_active:
            ask_price = quote_state.ask_order.price
            ask_size = quote_state.ask_order.remaining

        # Calculate distance from BBO
        bid_distance = None
        ask_distance = None
        if bid_price and book.best_bid:
            bid_distance = bid_price - book.best_bid  # negative = behind
        if ask_price and book.best_ask:
            ask_distance = book.best_ask - ask_price  # negative = behind

        return QuoteSnapshot(
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
            bid_distance_from_best=bid_distance,
            ask_distance_from_best=ask_distance,
        )

    def _snapshot_correlated_markets(
        self,
        correlated_tickers: list[str],
        market_root: Optional[str],
        our_mid: Optional[float],
    ) -> list[dict]:
        """
        Snapshot correlated markets for arb analysis.

        Args:
            correlated_tickers: Known correlated tickers (from 2-team games)
            market_root: Prefix for multi-outcome markets (to find via subscriptions)
            our_mid: Our market's mid price (for arb signal calculation)

        Returns:
            List of correlated market snapshots as dicts
        """
        snapshots = []

        if not self._orderbook_manager:
            return snapshots

        # Look up explicitly correlated tickers (e.g., other team in game)
        for corr_ticker in correlated_tickers:
            snapshot = self._get_correlated_snapshot(corr_ticker, our_mid)
            if snapshot:
                snapshots.append(asdict(snapshot))

        # For multi-outcome markets, find other subscribed markets with same root
        if market_root:
            # Get all books from the manager and find those matching the root
            try:
                # Access the internal _books dict if available
                all_books = getattr(self._orderbook_manager, '_books', {})
                for ticker in all_books.keys():
                    if ticker.startswith(market_root) and ticker not in correlated_tickers:
                        snapshot = self._get_correlated_snapshot(ticker, our_mid)
                        if snapshot:
                            snapshots.append(asdict(snapshot))
            except Exception as e:
                logger.debug(f"Could not enumerate correlated markets: {e}")

        return snapshots

    def _get_correlated_snapshot(
        self,
        ticker: str,
        our_mid: Optional[float],
    ) -> Optional[CorrelatedMarketSnapshot]:
        """Get snapshot of a single correlated market."""
        if not self._orderbook_manager:
            return None

        book = self._orderbook_manager.get(ticker)
        if book is None:
            return None

        best_bid = book.best_yes_bid()
        best_ask = book.best_yes_ask()

        if best_bid is None or best_ask is None:
            return None

        corr_mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        # Calculate arb signal for binary outcomes
        # For two-team games: our_mid + correlated_mid should ≈ 100
        implied_sum = None
        arb_signal = None
        if our_mid is not None:
            implied_sum = our_mid + corr_mid
            arb_signal = implied_sum - 100.0  # Positive = overpriced, negative = underpriced

        return CorrelatedMarketSnapshot(
            ticker=ticker,
            best_bid=best_bid,
            best_ask=best_ask,
            mid=corr_mid,
            spread=spread,
            implied_sum=implied_sum,
            arb_signal=arb_signal,
        )

    def _write(self, snapshot: FillSnapshot) -> None:
        """Write snapshot to file."""
        data = asdict(snapshot)
        self._file.write(json.dumps(data) + '\n')
        self._file.flush()

    def close(self) -> None:
        """Close the log file."""
        if self._file:
            self._file.close()
            self._file = None
