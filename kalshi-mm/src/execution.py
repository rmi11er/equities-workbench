"""Execution engine with order diffing and debouncing."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

from .config import StrategyConfig
from .connector import KalshiConnector, APIError, RateLimitError
from .constants import OrderSide, OrderStatus
from .types import Order, StrategyOutput
from .volatility_regime import VolatilityRegime

if TYPE_CHECKING:
    from .decision_log import DecisionLogger, FillEvent

logger = logging.getLogger(__name__)


class OrderAction(Enum):
    """Action to take on an order."""
    NONE = auto()       # No change needed
    CREATE = auto()     # Create new order
    AMEND = auto()      # Amend existing order
    CANCEL = auto()     # Cancel order


@dataclass
class OrderDiff:
    """Computed difference between target and actual state."""
    action: OrderAction
    order_id: Optional[str] = None  # For AMEND/CANCEL
    side: Optional[OrderSide] = None
    price: Optional[int] = None
    size: Optional[int] = None


@dataclass
class QuoteState:
    """Current state of our quotes in the market."""
    bid_order: Optional[Order] = None
    ask_order: Optional[Order] = None
    last_update: float = 0.0
    last_bid_price: Optional[int] = None
    last_ask_price: Optional[int] = None


class ExecutionEngine:
    """
    Manages order execution with diff-based updates and debouncing.

    Responsibilities:
    - Track current quote state
    - Compute minimal actions to reach target state
    - Debounce rapid quote updates
    - Execute orders via connector
    - VALIDATE orders before submission (guardrails)
    """

    def __init__(
        self,
        config: StrategyConfig,
        connector: KalshiConnector,
        decision_logger: Optional["DecisionLogger"] = None,
        max_consecutive_errors: int = 5,
        volatility_regime: Optional[VolatilityRegime] = None,
    ):
        self.config = config
        self.connector = connector
        self.decision_logger = decision_logger
        self.max_consecutive_errors = max_consecutive_errors
        self.volatility_regime = volatility_regime

        # State per ticker
        self._quotes: dict[str, QuoteState] = {}

        # P&L tracking
        self._realized_pnl: float = 0.0
        self._avg_entry_price: Optional[float] = None
        self._last_mid: dict[str, float] = {}  # Track mid for fill context

        # API error tracking for circuit breaker
        self._consecutive_api_errors: int = 0
        self._total_api_errors: int = 0

        # Market state for validation (set by market_maker before updates)
        self._best_bid: dict[str, int] = {}
        self._best_ask: dict[str, int] = {}
        self._book_depth: dict[str, dict[int, int]] = {}  # ticker -> {price: size}
        self._mid_price: dict[str, float] = {}

    # -------------------------------------------------------------------------
    # Guardrails / Validation
    # -------------------------------------------------------------------------

    def set_market_state(
        self,
        ticker: str,
        best_bid: int,
        best_ask: int,
        book_depth: Optional[dict[int, int]] = None,
    ) -> None:
        """
        Update market state for order validation.
        Must be called before update_quotes.

        Args:
            ticker: Market ticker
            best_bid: Best bid price
            best_ask: Best ask price
            book_depth: Optional dict of {price: size} for depth checking
        """
        self._best_bid[ticker] = best_bid
        self._best_ask[ticker] = best_ask
        if best_bid and best_ask:
            self._mid_price[ticker] = (best_bid + best_ask) / 2
        if book_depth is not None:
            self._book_depth[ticker] = book_depth

    def get_depth_at_price(self, ticker: str, price: int) -> int:
        """Get the book depth at a specific price level."""
        depth = self._book_depth.get(ticker, {})
        return depth.get(price, 0)

    def get_min_depth_contracts(self, ticker: str, price: int) -> int:
        """
        Convert dollar-based depth threshold to contract count.

        At 1c: $200 / $0.01 = 20,000 contracts needed
        At 50c: $200 / $0.50 = 400 contracts needed

        In high-vol regime, a decaying multiplier is applied.

        Args:
            ticker: Market ticker (for regime lookup)
            price: Price in cents (1-99)

        Returns:
            Minimum contracts required at this price level
        """
        if self.config.min_join_depth_dollars <= 0:
            return 0
        if price <= 0:
            return 0

        # Get regime multiplier (1.0 in normal, higher in high-vol)
        multiplier = 1.0
        if self.volatility_regime:
            multiplier = self.volatility_regime.get_depth_multiplier(ticker)

        # Apply multiplier to dollar threshold
        effective_dollars = self.config.min_join_depth_dollars * multiplier

        # Convert to contracts
        price_dollars = price / 100.0
        return int(effective_dollars / price_dollars)

    def should_skip_for_low_depth(
        self,
        ticker: str,
        price: int,
        is_bid: bool,
    ) -> tuple[bool, str]:
        """
        Check if we should skip placing an order due to low cumulative depth.

        We avoid being alone on a price level unless:
        1. There's sufficient cumulative depth from best to our level (dollar-based)
        2. We have good edge to mid (allow_solo_if_edge)

        Returns:
            (should_skip, reason)
        """
        min_depth = self.get_min_depth_contracts(ticker, price)
        if min_depth <= 0:
            return False, ""  # Feature disabled

        # Check cumulative depth from best to target price
        cumulative_depth = self.get_cumulative_depth(ticker, price, is_bid)
        if cumulative_depth >= min_depth:
            return False, ""  # Enough cumulative liquidity to join

        # Check if we have enough edge to go solo
        mid = self._mid_price.get(ticker)
        if mid is not None:
            if is_bid:
                edge = mid - price  # How far below mid we're bidding
            else:
                edge = price - mid  # How far above mid we're asking

            if edge >= self.config.allow_solo_if_edge:
                return False, ""  # Good edge, OK to be alone

        return True, f"Low cumulative depth ({cumulative_depth}) to {price} (need {min_depth}), would be exposed"

    def get_cumulative_depth(
        self,
        ticker: str,
        price: int,
        is_bid: bool,
    ) -> int:
        """
        Get cumulative depth from best price to the given price level.

        For bids: sum depth from best_bid down to price (inclusive)
        For asks: sum depth from best_ask up to price (inclusive)

        Args:
            ticker: Market ticker
            price: Target price level
            is_bid: True for bid side, False for ask side

        Returns:
            Cumulative contracts from best to this level
        """
        book_depth = self._book_depth.get(ticker, {})
        if not book_depth:
            return 0

        cumulative = 0
        if is_bid:
            best = self._best_bid.get(ticker)
            if best is None:
                return 0
            # Sum from best_bid down to price
            for p in range(best, price - 1, -1):
                cumulative += book_depth.get(p, 0)
        else:
            best = self._best_ask.get(ticker)
            if best is None:
                return 0
            # Sum from best_ask up to price
            for p in range(best, price + 1):
                cumulative += book_depth.get(p, 0)

        return cumulative

    def find_joinable_price(
        self,
        ticker: str,
        target_price: int,
        is_bid: bool,
    ) -> Optional[int]:
        """
        Find the nearest price level with sufficient cumulative depth to join.

        Instead of skipping a quote entirely when depth is low, we "retreat"
        to a worse price where cumulative depth from best to that level meets
        our dollar-based threshold (with volatility regime multiplier applied).

        Args:
            ticker: Market ticker
            target_price: The ideal price from strategy
            is_bid: True for bids (retreat = lower prices), False for asks (retreat = higher)

        Returns:
            Adjusted price with sufficient cumulative depth, or None if no joinable level found
        """
        min_depth = self.get_min_depth_contracts(ticker, target_price)
        if min_depth <= 0:
            return target_price  # Feature disabled, use original price

        max_retreat = self.config.max_retreat
        mid = self._mid_price.get(ticker)

        # Check cumulative depth at original price first
        cumulative = self.get_cumulative_depth(ticker, target_price, is_bid)
        if cumulative >= min_depth:
            return target_price

        # Check if we have enough edge to go solo at target price
        if mid is not None:
            if is_bid:
                edge = mid - target_price
            else:
                edge = target_price - mid
            if edge >= self.config.allow_solo_if_edge:
                return target_price  # Good edge, OK to be alone

        # Walk backwards to find a level with sufficient cumulative depth
        # For bids: look at lower prices (worse for us as buyer)
        # For asks: look at higher prices (worse for us as seller)
        for offset in range(1, max_retreat + 1):
            if is_bid:
                check_price = target_price - offset
                if check_price < 1:
                    break
            else:
                check_price = target_price + offset
                if check_price > 99:
                    break

            # Recalculate min_depth for retreated price (it changes with price and regime)
            min_depth_at_price = self.get_min_depth_contracts(ticker, check_price)
            cumulative = self.get_cumulative_depth(ticker, check_price, is_bid)
            if cumulative >= min_depth_at_price:
                # Don't log here - caller will log if they actually use the retreated price
                return check_price

        # No joinable level found within max_retreat
        return None

    def validate_order(
        self,
        ticker: str,
        side: str,  # "yes" or "no"
        price: int,
        action: str = "create",
    ) -> tuple[bool, str]:
        """
        Validate an order BEFORE submission.

        Returns:
            (is_valid, error_message)
        """
        # Basic price bounds
        if price < 1 or price > 99:
            return False, f"Price {price} out of bounds [1-99]"

        # Get market state
        market_bid = self._best_bid.get(ticker)
        market_ask = self._best_ask.get(ticker)

        if market_bid is None or market_ask is None:
            return False, "No market state available for validation"

        # Check for spread crossing
        # YES buy order at price >= market_ask would take liquidity
        # NO buy order at price >= (100 - market_bid) would take liquidity
        if side == "yes":
            if price >= market_ask:
                return False, f"BID {price} would cross spread (ask={market_ask})"
        else:  # side == "no"
            no_crossing_threshold = 100 - market_bid
            if price >= no_crossing_threshold:
                # NO buy at this price = YES sell at (100-price) which is <= market_bid
                yes_equiv = 100 - price
                return False, f"ASK {yes_equiv} would cross spread (bid={market_bid})"

        return True, ""

    def validate_quote_pair(
        self,
        bid_price: Optional[int],
        ask_price: Optional[int],
    ) -> tuple[bool, str]:
        """
        Validate that bid/ask don't cross each other.
        """
        if bid_price is not None and ask_price is not None:
            if bid_price >= ask_price:
                return False, f"Quotes crossed: bid={bid_price} >= ask={ask_price}"
        return True, ""

    def record_api_success(self) -> None:
        """Record successful API call - resets consecutive error counter."""
        self._consecutive_api_errors = 0

    def record_api_error(self) -> None:
        """Record API error - increments counters."""
        self._consecutive_api_errors += 1
        self._total_api_errors += 1
        logger.warning(
            f"API error #{self._consecutive_api_errors} "
            f"(total: {self._total_api_errors}, max: {self.max_consecutive_errors})"
        )

    def should_halt_on_errors(self) -> bool:
        """Check if we've hit too many consecutive API errors."""
        return self._consecutive_api_errors >= self.max_consecutive_errors

    def get_quote_state(self, ticker: str) -> QuoteState:
        """Get or create quote state for a ticker."""
        if ticker not in self._quotes:
            self._quotes[ticker] = QuoteState()
        return self._quotes[ticker]

    # -------------------------------------------------------------------------
    # Diffing Logic
    # -------------------------------------------------------------------------

    def compute_diff(
        self,
        ticker: str,
        target: StrategyOutput,
        should_bid: bool,
        should_ask: bool,
    ) -> tuple[OrderDiff, OrderDiff]:
        """
        Compute the diff between target quotes and current state.

        Returns:
            (bid_diff, ask_diff)
        """
        state = self.get_quote_state(ticker)

        bid_diff = self._compute_side_diff(
            current=state.bid_order,
            target_price=target.bid_price if should_bid else None,
            target_size=target.bid_size if should_bid else None,
            side=OrderSide.BUY,
        )

        ask_diff = self._compute_side_diff(
            current=state.ask_order,
            target_price=target.ask_price if should_ask else None,
            target_size=target.ask_size if should_ask else None,
            side=OrderSide.SELL,
        )

        return bid_diff, ask_diff

    def _compute_side_diff(
        self,
        current: Optional[Order],
        target_price: Optional[int],
        target_size: Optional[int],
        side: OrderSide,
    ) -> OrderDiff:
        """Compute diff for one side (bid or ask)."""

        # No target quote desired
        if target_price is None or target_size is None or target_size == 0:
            if current and current.is_active:
                return OrderDiff(
                    action=OrderAction.CANCEL,
                    order_id=current.order_id,
                )
            return OrderDiff(action=OrderAction.NONE)

        # No current order, need to create
        if current is None or not current.is_active:
            return OrderDiff(
                action=OrderAction.CREATE,
                side=side,
                price=target_price,
                size=target_size,
            )

        # Current order exists, check if amendment needed
        price_changed = current.price != target_price
        size_changed = current.remaining != target_size

        if price_changed or size_changed:
            return OrderDiff(
                action=OrderAction.AMEND,
                order_id=current.order_id,
                price=target_price if price_changed else None,
                size=target_size if size_changed else None,
            )

        return OrderDiff(action=OrderAction.NONE)

    # -------------------------------------------------------------------------
    # Debouncing
    # -------------------------------------------------------------------------

    def should_update(
        self,
        ticker: str,
        target: StrategyOutput,
        should_bid: bool = True,
        should_ask: bool = True,
    ) -> bool:
        """
        Check if we should update quotes (debouncing logic).

        Updates only when:
        - Missing an order that we should have (immediate re-quote after fill)
        - Price change exceeds threshold, OR
        - Time since last update exceeds threshold
        """
        state = self.get_quote_state(ticker)
        now = time.monotonic()

        # CRITICAL: If we should have an order but don't, update immediately
        # This ensures we re-quote after fills instead of waiting for debounce
        if should_bid and state.bid_order is None:
            return True
        if should_ask and state.ask_order is None:
            return True

        # Time-based: always update after debounce_seconds
        time_elapsed = now - state.last_update
        if time_elapsed >= self.config.debounce_seconds:
            return True

        # Price-based: update if price changed significantly
        bid_change = abs(target.bid_price - (state.last_bid_price or target.bid_price))
        ask_change = abs(target.ask_price - (state.last_ask_price or target.ask_price))

        if bid_change >= self.config.debounce_cents:
            return True
        if ask_change >= self.config.debounce_cents:
            return True

        return False

    def mark_updated(self, ticker: str, target: StrategyOutput) -> None:
        """Mark that quotes were updated."""
        state = self.get_quote_state(ticker)
        state.last_update = time.monotonic()
        state.last_bid_price = target.bid_price
        state.last_ask_price = target.ask_price

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    async def execute_diff(
        self,
        ticker: str,
        bid_diff: OrderDiff,
        ask_diff: OrderDiff,
    ) -> None:
        """
        Execute the computed diffs.

        Prioritizes cancels over creates to free up margin.
        """
        state = self.get_quote_state(ticker)

        # Execute cancels first
        for diff, attr in [(bid_diff, "bid_order"), (ask_diff, "ask_order")]:
            if diff.action == OrderAction.CANCEL and diff.order_id:
                try:
                    await self.connector.cancel_order(diff.order_id)
                    setattr(state, attr, None)
                    logger.info(f"Cancelled order {diff.order_id}")
                    self.record_api_success()
                except APIError as e:
                    logger.error(f"Cancel failed: {e}")
                    self.record_api_error()

        # Execute amends (with queue-position-aware logic)
        for diff, attr, side_str in [
            (bid_diff, "bid_order", "yes"),   # Bids are YES orders
            (ask_diff, "ask_order", "no"),    # Asks are NO orders
        ]:
            if diff.action == OrderAction.AMEND and diff.order_id:
                try:
                    # Get current order for fallback values and client_order_id
                    order = getattr(state, attr)

                    # CRITICAL: Need client_order_id for Kalshi amend API
                    if not order or not order.client_order_id:
                        logger.warning(
                            f"Cannot amend order {diff.order_id}: no client_order_id. "
                            f"Cancelling and recreating instead."
                        )
                        # Cancel and let next tick recreate with proper tracking
                        try:
                            await self.connector.cancel_order(diff.order_id)
                        except APIError:
                            pass
                        # Always clear state so we create fresh on next tick
                        setattr(state, attr, None)
                        continue

                    # Calculate price delta to determine if this is a significant change
                    price_delta = abs(diff.price - order.price) if diff.price is not None else 0
                    size_delta = abs(diff.size - order.remaining) if diff.size is not None else 0

                    # Queue-aware amendment logic:
                    # - Significant price change (>= debounce_cents): amend immediately
                    # - Small change: only amend if we're at the back of the queue
                    queue_threshold = self.config.queue_position_threshold
                    should_amend = False
                    queue_position = None

                    if price_delta >= self.config.debounce_cents:
                        # Significant price move - amend regardless of queue position
                        should_amend = True
                        logger.debug(
                            f"Amend triggered by price delta: {price_delta}c >= {self.config.debounce_cents}c"
                        )
                    elif queue_threshold > 0:
                        # Small change - check queue position first
                        try:
                            queue_position = await self.connector.get_queue_position(diff.order_id)
                            if queue_position > queue_threshold:
                                # Bad queue position, OK to amend
                                should_amend = True
                                logger.debug(
                                    f"Amend allowed: queue_pos={queue_position} > threshold={queue_threshold}"
                                )
                            else:
                                # Good queue position - preserve it
                                logger.info(
                                    f"Skipping amend to preserve queue priority: "
                                    f"queue_pos={queue_position} <= threshold={queue_threshold}, "
                                    f"price_delta={price_delta}c, size_delta={size_delta}"
                                )
                                continue
                        except APIError as e:
                            # If we can't get queue position, default to amending
                            logger.warning(f"Failed to get queue position: {e}, proceeding with amend")
                            should_amend = True
                    else:
                        # queue_threshold=0 means always amend (disabled)
                        should_amend = True

                    if not should_amend:
                        continue

                    # Apply retreat logic to find joinable price (same as CREATE)
                    is_bid = (side_str == "yes")
                    target_price = diff.price if diff.price is not None else order.price
                    adjusted_price = self.find_joinable_price(
                        ticker=ticker,
                        target_price=target_price,
                        is_bid=is_bid,
                    )

                    if adjusted_price is None:
                        # No joinable level found - keep current order rather than amend to exposed price
                        logger.info(
                            f"Skipping amend: no joinable level found for "
                            f"{'bid' if is_bid else 'ask'} at {target_price}, keeping order at {order.price}"
                        )
                        continue

                    # If adjusted price equals current order price, no need to amend
                    if adjusted_price == order.price and (diff.size is None or diff.size == order.remaining):
                        logger.debug(
                            f"Skipping amend: adjusted price {adjusted_price} equals current order price"
                        )
                        continue

                    # Log the amend decision
                    logger.info(
                        f"Amend: {ticker} {side_str}, "
                        f"price {order.price}->{adjusted_price} (target={target_price}, delta={price_delta}), "
                        f"size {order.remaining}->{diff.size or order.remaining}, "
                        f"queue_pos={queue_position}"
                    )

                    # CRITICAL: Kalshi API requires BOTH price AND count on amends
                    # Use adjusted price (with retreat logic applied)
                    yes_price = adjusted_price
                    amend_count = diff.size if diff.size is not None else order.remaining

                    # Safeguard: skip amend if we can't determine price or count
                    if yes_price is None or amend_count is None:
                        logger.error(
                            f"Cannot amend order {diff.order_id}: "
                            f"price={yes_price}, count={amend_count}, order={order}"
                        )
                        continue

                    # Convert YES price to NO price for ask side
                    api_price = yes_price
                    if side_str == "no":
                        api_price = 100 - yes_price

                    resp = await self.connector.amend_order(
                        order_id=diff.order_id,
                        ticker=ticker,
                        side=side_str,
                        client_order_id=order.client_order_id,
                        price=api_price,
                        count=amend_count,
                    )

                    # Update our state including new client_order_id
                    # Use adjusted_price (not diff.price) since we may have retreated
                    order.price = adjusted_price
                    if diff.size is not None:
                        order.remaining = diff.size
                    # Update client_order_id for future amends
                    if "updated_client_order_id" in resp:
                        order.client_order_id = resp["updated_client_order_id"]

                    logger.info(f"Amended order {diff.order_id}: price={adjusted_price}, size={diff.size}")
                    self.record_api_success()
                except APIError as e:
                    logger.error(f"Amend failed: {e}")
                    self.record_api_error()
                    # On failure, cancel and recreate
                    # Always clear state - if amend failed, order may be gone (filled/cancelled)
                    try:
                        await self.connector.cancel_order(diff.order_id)
                    except APIError:
                        # Cancel failed too (order already gone) - that's fine
                        pass
                    # Clear state regardless so we don't keep trying to amend a dead order
                    setattr(state, attr, None)

        # Execute creates
        for diff, attr, side_str in [
            (bid_diff, "bid_order", "yes"),  # Buying YES = bid
            (ask_diff, "ask_order", "no"),   # Selling YES = bid NO (equivalent)
        ]:
            if diff.action == OrderAction.CREATE:
                try:
                    # CRITICAL: Convert YES ask price to NO buy price
                    # Selling YES at X = Buying NO at (100-X)
                    api_price = diff.price
                    if side_str == "no":
                        api_price = 100 - diff.price
                        logger.debug(f"Converting ask: YES@{diff.price} -> NO@{api_price}")

                    # GUARDRAIL: Validate order before submission
                    is_valid, error_msg = self.validate_order(
                        ticker=ticker,
                        side=side_str,
                        price=api_price,
                    )
                    if not is_valid:
                        logger.warning(f"Quote skipped (would cross): {error_msg}")
                        continue  # Skip this order

                    # GUARDRAIL: Find joinable price level (avoid being alone)
                    # Use YES price for depth check (diff.price is always in YES terms)
                    is_bid = (side_str == "yes")
                    adjusted_price = self.find_joinable_price(
                        ticker=ticker,
                        target_price=diff.price,
                        is_bid=is_bid,
                    )
                    if adjusted_price is None:
                        logger.info(
                            f"Quote skipped: no joinable level found within retreat range "
                            f"for {'bid' if is_bid else 'ask'} at {diff.price}"
                        )
                        continue  # Skip only if no joinable level exists

                    # Use adjusted price for the order
                    final_yes_price = adjusted_price
                    final_api_price = adjusted_price
                    if side_str == "no":
                        final_api_price = 100 - adjusted_price

                    # Re-validate with adjusted price (in case retreat crossed spread)
                    is_valid, error_msg = self.validate_order(
                        ticker=ticker,
                        side=side_str,
                        price=final_api_price,
                    )
                    if not is_valid:
                        logger.warning(f"Quote skipped (adjusted price would cross): {error_msg}")
                        continue

                    resp = await self.connector.create_order(
                        ticker=ticker,
                        side=side_str,
                        price=final_api_price,
                        count=diff.size,
                    )

                    # Create Order object from response
                    order_data = resp.get("order", resp)
                    order = Order(
                        order_id=order_data["order_id"],
                        ticker=ticker,
                        side=diff.side,
                        price=final_yes_price,  # Store the actual price we used
                        size=diff.size,
                        remaining=diff.size,
                        filled=0,
                        status=OrderStatus.RESTING,
                        client_order_id=order_data.get("client_order_id"),
                    )
                    setattr(state, attr, order)
                    if final_yes_price != diff.price:
                        logger.info(
                            f"Created order {order.order_id}: {diff.side.value} {diff.size}@{final_yes_price} "
                            f"(retreated from {diff.price})"
                        )
                    else:
                        logger.info(f"Created order {order.order_id}: {diff.side.value} {diff.size}@{final_yes_price}")
                    self.record_api_success()

                except APIError as e:
                    logger.error(f"Create order failed: {e}")
                    self.record_api_error()

    async def update_quotes(
        self,
        ticker: str,
        target: StrategyOutput,
        should_bid: bool,
        should_ask: bool,
        force: bool = False,
    ) -> None:
        """
        Main entry point: update quotes to match target.

        Args:
            ticker: Market ticker
            target: Strategy output with target quotes
            should_bid: Whether to maintain a bid
            should_ask: Whether to maintain an ask
            force: Skip debouncing check
        """
        # Check debouncing (pass should_bid/ask to detect missing orders after fills)
        if not force and not self.should_update(ticker, target, should_bid, should_ask):
            return

        # Compute diff
        bid_diff, ask_diff = self.compute_diff(ticker, target, should_bid, should_ask)

        # Skip if no changes
        if bid_diff.action == OrderAction.NONE and ask_diff.action == OrderAction.NONE:
            return

        # Execute
        await self.execute_diff(ticker, bid_diff, ask_diff)

        # Mark updated
        self.mark_updated(ticker, target)

    # -------------------------------------------------------------------------
    # Emergency Actions
    # -------------------------------------------------------------------------

    async def cancel_all(self, ticker: Optional[str] = None) -> None:
        """
        Cancel all orders (panic mode).

        Uses individual cancels with staggered timing to avoid rate limits.
        Includes retry logic for transient failures.

        Args:
            ticker: If provided, only cancel for this ticker
        """
        logger.warning(f"CANCEL ALL triggered for ticker={ticker}")

        cancelled_count = 0
        failed_count = 0
        max_retries = 3
        base_delay = 0.15  # 150ms between cancels to stay under rate limit

        try:
            orders = await self.connector.get_orders(ticker)
            resting_orders = [o for o in orders.get("orders", []) if o.get("status") == "resting"]

            if not resting_orders:
                logger.info("No resting orders to cancel")
            else:
                logger.info(f"Found {len(resting_orders)} resting orders to cancel")

                for i, order in enumerate(resting_orders):
                    order_id = order.get("order_id")

                    # Stagger requests to avoid rate limit
                    if i > 0:
                        await asyncio.sleep(base_delay)

                    # Retry logic for transient failures
                    for attempt in range(max_retries):
                        try:
                            await self.connector.cancel_order(order_id)
                            cancelled_count += 1
                            logger.info(f"Cancelled order {order_id}")
                            break
                        except RateLimitError:
                            # Rate limited - wait longer and retry
                            wait_time = 0.5 * (attempt + 1)
                            logger.warning(f"Rate limited cancelling {order_id}, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                        except APIError as cancel_err:
                            if attempt < max_retries - 1:
                                logger.warning(f"Retry {attempt + 1}/{max_retries} for {order_id}: {cancel_err}")
                                await asyncio.sleep(0.2)
                            else:
                                failed_count += 1
                                logger.error(f"Failed to cancel order {order_id} after {max_retries} attempts: {cancel_err}")
                        except Exception as e:
                            if attempt < max_retries - 1:
                                logger.warning(f"Retry {attempt + 1}/{max_retries} for {order_id}: {e}")
                                await asyncio.sleep(0.2)
                            else:
                                failed_count += 1
                                logger.error(f"Network error cancelling order {order_id}: {e}")

                logger.info(f"Cancel complete: {cancelled_count} cancelled, {failed_count} failed")

        except APIError as e:
            logger.error(f"Failed to fetch orders for cancel: {e}")
        except Exception as e:
            # Network errors (DNS timeout, connection refused, etc.)
            logger.error(f"Network error during cancel_all: {type(e).__name__}: {e}")

        # Clear our state regardless of success (assume orders may be gone)
        if ticker:
            self._quotes.pop(ticker, None)
        else:
            self._quotes.clear()

    async def pull_quotes(self, ticker: str) -> None:
        """Pull quotes for a specific ticker (data stale, etc.)."""
        logger.warning(f"Pulling quotes for {ticker}")
        await self.cancel_all(ticker)

    async def check_and_pull_exposed_quotes(
        self,
        ticker: str,
        best_bid: Optional[int],
        best_ask: Optional[int],
        book_depth: Optional[dict[int, int]] = None,
    ) -> bool:
        """
        Check if our quotes are exposed and cancel them immediately.

        A quote is "exposed" when:
        - Our bid price > market best_bid (we're improving the market, first to get hit)
        - Our ask price < market best_ask (we're improving the market, first to get hit)
        - Depth at our price level dropped below threshold (we lost our cover)

        This is called reactively from the orderbook handler, not the tick loop,
        for faster reaction to adverse market moves.

        Args:
            ticker: Market ticker
            best_bid: Current market best bid
            best_ask: Current market best ask
            book_depth: Optional dict of {price: size} for depth checking

        Returns:
            True if any quotes were pulled
        """
        state = self._quotes.get(ticker)
        if not state:
            return False

        pulled = False

        # Check if bid is exposed (our bid > market best bid)
        if (
            state.bid_order
            and state.bid_order.is_active
            and best_bid is not None
            and state.bid_order.price > best_bid
        ):
            exposure = state.bid_order.price - best_bid
            logger.warning(
                f"[{ticker}] BID EXPOSED by {exposure}c: our_bid={state.bid_order.price} > best_bid={best_bid} - CANCELLING"
            )
            try:
                await self.connector.cancel_order(state.bid_order.order_id)
                state.bid_order = None
                pulled = True
            except Exception as e:
                logger.error(f"Failed to cancel exposed bid: {e}")

        # Check if ask is exposed (our ask < market best ask)
        if (
            state.ask_order
            and state.ask_order.is_active
            and best_ask is not None
            and state.ask_order.price < best_ask
        ):
            exposure = best_ask - state.ask_order.price
            logger.warning(
                f"[{ticker}] ASK EXPOSED by {exposure}c: our_ask={state.ask_order.price} < best_ask={best_ask} - CANCELLING"
            )
            try:
                await self.connector.cancel_order(state.ask_order.order_id)
                state.ask_order = None
                pulled = True
            except Exception as e:
                logger.error(f"Failed to cancel exposed ask: {e}")

        # Check if depth at our price levels has thinned dangerously
        # Only check if we have a depth threshold configured and book_depth provided
        if book_depth and self.config.min_join_depth_dollars > 0:
            # Check bid depth
            if state.bid_order and state.bid_order.is_active:
                our_bid = state.bid_order.price
                min_depth = self.get_min_depth_contracts(ticker, our_bid)
                # Get cumulative depth from best_bid to our level
                cumulative = 0
                if best_bid:
                    for p in range(best_bid, our_bid - 1, -1):
                        cumulative += book_depth.get(p, 0)

                # If depth dropped below half our threshold, pull the quote
                if cumulative < min_depth // 2:
                    logger.warning(
                        f"[{ticker}] BID DEPTH THINNED: cumulative={cumulative} < {min_depth // 2} "
                        f"at price={our_bid} - CANCELLING"
                    )
                    try:
                        await self.connector.cancel_order(state.bid_order.order_id)
                        state.bid_order = None
                        pulled = True
                    except Exception as e:
                        logger.error(f"Failed to cancel thin-depth bid: {e}")

            # Check ask depth
            if state.ask_order and state.ask_order.is_active:
                our_ask = state.ask_order.price
                min_depth = self.get_min_depth_contracts(ticker, our_ask)
                # Get cumulative depth from best_ask to our level
                cumulative = 0
                if best_ask:
                    for p in range(best_ask, our_ask + 1):
                        cumulative += book_depth.get(p, 0)

                # If depth dropped below half our threshold, pull the quote
                if cumulative < min_depth // 2:
                    logger.warning(
                        f"[{ticker}] ASK DEPTH THINNED: cumulative={cumulative} < {min_depth // 2} "
                        f"at price={our_ask} - CANCELLING"
                    )
                    try:
                        await self.connector.cancel_order(state.ask_order.order_id)
                        state.ask_order = None
                        pulled = True
                    except Exception as e:
                        logger.error(f"Failed to cancel thin-depth ask: {e}")

        return pulled

    async def send_ioc_simulation(
        self,
        ticker: str,
        side: str,
        price: int,
        count: int,
    ) -> dict:
        """
        Simulate an IOC (Immediate-or-Cancel) order.

        Kalshi REST API doesn't support true IOC. We simulate by:
        1. Send aggressive limit order (crossing the spread)
        2. Immediately send cancel request

        The matching engine executes what it can before the cancel arrives.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            price: Aggressive limit price (should cross spread)
            count: Number of contracts to try to fill

        Returns:
            Dict with order_id and status
        """
        order_id = None
        try:
            # Step 1: Create aggressive limit order
            resp = await self.connector.create_order(
                ticker=ticker,
                side=side,
                price=price,
                count=count,
            )
            order_id = resp.get("order", resp).get("order_id")

            if not order_id:
                logger.error("IOC simulation: No order_id in response")
                return {"success": False, "error": "No order_id"}

            logger.info(f"IOC simulation: Created order {order_id} {side} {count}@{price}")

            # Step 2: Immediately cancel to prevent resting
            # Small sleep to let matching engine process (optional, can remove)
            # await asyncio.sleep(0.01)  # 10ms

            try:
                await self.connector.cancel_order(order_id)
                logger.info(f"IOC simulation: Cancelled remainder of {order_id}")
            except APIError as cancel_err:
                # Cancel might fail if order was fully filled - that's OK
                if cancel_err.status == 404:
                    logger.info(f"IOC simulation: Order {order_id} already gone (likely filled)")
                else:
                    logger.warning(f"IOC simulation: Cancel failed: {cancel_err}")

            return {
                "success": True,
                "order_id": order_id,
                "side": side,
                "price": price,
                "count": count,
            }

        except APIError as e:
            logger.error(f"IOC simulation failed: {e}")
            # Try to cancel if we got an order_id
            if order_id:
                try:
                    await self.connector.cancel_order(order_id)
                except APIError:
                    pass
            return {"success": False, "error": str(e)}

    # -------------------------------------------------------------------------
    # Fill Handling & P&L
    # -------------------------------------------------------------------------

    def handle_fill(
        self,
        ticker: str,
        order_id: str,
        side: OrderSide,
        price: int,
        size: int,
        inventory_before: int = 0,
    ) -> None:
        """
        Handle a fill notification.

        Updates order state and P&L tracking.
        """
        state = self.get_quote_state(ticker)

        # Update order state and clear fully-filled orders
        for attr in ["bid_order", "ask_order"]:
            order = getattr(state, attr)
            if order and order.order_id == order_id:
                order.filled += size
                order.remaining -= size
                if order.remaining <= 0:
                    order.status = OrderStatus.EXECUTED
                    # Clear from state so we don't try to amend a dead order
                    setattr(state, attr, None)
                    logger.info(f"Order {order_id} fully filled, cleared from state")
                break

        # Calculate P&L impact
        # Buy YES = inventory increases, pay price
        # Sell YES (buy NO) = inventory decreases, receive 100-price
        pnl_delta = 0.0
        if side == OrderSide.BUY:
            # Bought YES at price, position gets longer
            pnl_delta = -price * size  # Spent money
            inventory_after = inventory_before + size
        else:
            # Sold YES at price, position gets shorter
            pnl_delta = price * size  # Received money
            inventory_after = inventory_before - size

        self._realized_pnl += pnl_delta

        # Log the fill with full context
        if self.decision_logger:
            from .decision_log import FillEvent
            fill_event = FillEvent(
                timestamp=datetime.now().isoformat(),
                order_id=order_id,
                side=side.value,
                price=price,
                size=size,
                inventory_before=inventory_before,
                inventory_after=inventory_after,
                realized_pnl_delta=pnl_delta,
                total_realized_pnl=self._realized_pnl,
                mid_at_fill=self._last_mid.get(ticker, 50.0),
                our_bid_at_fill=state.last_bid_price,
                our_ask_at_fill=state.last_ask_price,
            )
            self.decision_logger.log_fill(fill_event)

        logger.info(
            f"Fill: {side.value} {size}@{price} (order {order_id}) "
            f"inv: {inventory_before}->{inventory_after}, pnl_delta={pnl_delta:.0f}c"
        )

    def update_market_context(self, ticker: str, mid: float) -> None:
        """Update market context for fill logging."""
        self._last_mid[ticker] = mid

    @property
    def realized_pnl(self) -> float:
        """Get realized P&L."""
        return self._realized_pnl

    def get_unrealized_pnl(self, ticker: str, mid_price: float, inventory: int) -> float:
        """
        Calculate unrealized P&L.

        Simplified: assumes mid price is fair value.
        """
        if self._avg_entry_price is None or inventory == 0:
            return 0.0

        if inventory > 0:
            # Long YES: profit if mid > entry
            return (mid_price - self._avg_entry_price) * inventory
        else:
            # Short YES (long NO): profit if mid < entry
            return (self._avg_entry_price - mid_price) * abs(inventory)
