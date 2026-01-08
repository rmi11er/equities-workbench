"""Execution engine with order diffing and debouncing."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .config import StrategyConfig
from .connector import KalshiConnector, APIError
from .constants import OrderSide, OrderStatus
from .types import Order, StrategyOutput

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
    """

    def __init__(self, config: StrategyConfig, connector: KalshiConnector):
        self.config = config
        self.connector = connector

        # State per ticker
        self._quotes: dict[str, QuoteState] = {}

        # P&L tracking
        self._realized_pnl: float = 0.0
        self._avg_entry_price: Optional[float] = None

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
                except APIError as e:
                    logger.error(f"Cancel failed: {e}")

        # Execute amends
        for diff, attr in [(bid_diff, "bid_order"), (ask_diff, "ask_order")]:
            if diff.action == OrderAction.AMEND and diff.order_id:
                try:
                    resp = await self.connector.amend_order(
                        order_id=diff.order_id,
                        price=diff.price,
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
                except APIError as e:
                    logger.error(f"Amend failed: {e}")
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
                    resp = await self.connector.create_order(
                        ticker=ticker,
                        side=side_str,
                        price=diff.price,
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

                except APIError as e:
                    logger.error(f"Create order failed: {e}")

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

        Args:
            ticker: If provided, only cancel for this ticker
        """
        logger.warning(f"CANCEL ALL triggered for ticker={ticker}")

        try:
            await self.connector.cancel_all_orders(ticker)
        except APIError as e:
            logger.error(f"Cancel all failed: {e}")

        # Clear our state
        if ticker:
            self._quotes.pop(ticker, None)
        else:
            self._quotes.clear()

    async def pull_quotes(self, ticker: str) -> None:
        """Pull quotes for a specific ticker (data stale, etc.)."""
        logger.warning(f"Pulling quotes for {ticker}")
        await self.cancel_all(ticker)

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

        # P&L tracking (simplified)
        # TODO: More sophisticated P&L calculation with position tracking
        logger.info(f"Fill: {side.value} {size}@{price} (order {order_id})")

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
