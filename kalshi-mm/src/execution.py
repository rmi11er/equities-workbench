"""Execution engine with order diffing and debouncing."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING

from .config import StrategyConfig
from .connector import KalshiConnector, APIError
from .constants import OrderSide, OrderStatus
from .types import Order, StrategyOutput

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
    ):
        self.config = config
        self.connector = connector
        self.decision_logger = decision_logger
        self.max_consecutive_errors = max_consecutive_errors

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

    # -------------------------------------------------------------------------
    # Guardrails / Validation
    # -------------------------------------------------------------------------

    def set_market_state(self, ticker: str, best_bid: int, best_ask: int) -> None:
        """
        Update market state for order validation.
        Must be called before update_quotes.
        """
        self._best_bid[ticker] = best_bid
        self._best_ask[ticker] = best_ask

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

    def should_update(self, ticker: str, target: StrategyOutput) -> bool:
        """
        Check if we should update quotes (debouncing logic).

        Updates only when:
        - Price change exceeds threshold, OR
        - Time since last update exceeds threshold
        """
        state = self.get_quote_state(ticker)
        now = time.monotonic()

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

        # Execute amends
        for diff, attr, side_str in [
            (bid_diff, "bid_order", "yes"),   # Bids are YES orders
            (ask_diff, "ask_order", "no"),    # Asks are NO orders
        ]:
            if diff.action == OrderAction.AMEND and diff.order_id:
                try:
                    # CRITICAL: Convert YES ask price to NO buy price
                    api_price = diff.price
                    if side_str == "no" and diff.price is not None:
                        api_price = 100 - diff.price
                        logger.debug(f"Converting amend ask: YES@{diff.price} -> NO@{api_price}")

                    resp = await self.connector.amend_order(
                        order_id=diff.order_id,
                        side=side_str,
                        price=api_price,
                        count=diff.size,
                    )
                    # Update our state
                    order = getattr(state, attr)
                    if order:
                        if diff.price is not None:
                            order.price = diff.price
                        if diff.size is not None:
                            order.remaining = diff.size
                    logger.info(f"Amended order {diff.order_id}: price={diff.price}, size={diff.size}")
                    self.record_api_success()
                except APIError as e:
                    logger.error(f"Amend failed: {e}")
                    self.record_api_error()
                    # On failure, cancel and recreate
                    try:
                        await self.connector.cancel_order(diff.order_id)
                        setattr(state, attr, None)
                    except APIError:
                        pass

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

                    resp = await self.connector.create_order(
                        ticker=ticker,
                        side=side_str,
                        price=api_price,
                        count=diff.size,
                    )

                    # Create Order object from response
                    order_data = resp.get("order", resp)
                    order = Order(
                        order_id=order_data["order_id"],
                        ticker=ticker,
                        side=diff.side,
                        price=diff.price,
                        size=diff.size,
                        remaining=diff.size,
                        filled=0,
                        status=OrderStatus.RESTING,
                    )
                    setattr(state, attr, order)
                    logger.info(f"Created order {order.order_id}: {diff.side.value} {diff.size}@{diff.price}")
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
        # Check debouncing
        if not force and not self.should_update(ticker, target):
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

        Uses individual cancels as primary method since batch cancel
        endpoint may not exist on all Kalshi API versions.

        Args:
            ticker: If provided, only cancel for this ticker
        """
        logger.warning(f"CANCEL ALL triggered for ticker={ticker}")

        # Always use individual cancels - more reliable
        cancelled_count = 0
        failed_count = 0

        try:
            orders = await self.connector.get_orders(ticker)
            resting_orders = [o for o in orders.get("orders", []) if o.get("status") == "resting"]

            if not resting_orders:
                logger.info("No resting orders to cancel")
            else:
                logger.info(f"Found {len(resting_orders)} resting orders to cancel")

                for order in resting_orders:
                    order_id = order.get("order_id")
                    try:
                        await self.connector.cancel_order(order_id)
                        cancelled_count += 1
                        logger.info(f"Cancelled order {order_id}")
                    except APIError as cancel_err:
                        failed_count += 1
                        logger.error(f"Failed to cancel order {order_id}: {cancel_err}")
                    except Exception as e:
                        # Network errors (DNS, connection, etc.)
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

        # Update order state
        for order in [state.bid_order, state.ask_order]:
            if order and order.order_id == order_id:
                order.filled += size
                order.remaining -= size
                if order.remaining <= 0:
                    order.status = OrderStatus.EXECUTED
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
