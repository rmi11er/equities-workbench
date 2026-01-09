"""Unit tests for the execution engine."""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock

from src.execution import ExecutionEngine, OrderAction, OrderDiff, QuoteState
from src.config import StrategyConfig
from src.types import Order, StrategyOutput
from src.constants import OrderSide, OrderStatus


class TestOrderDiff:
    """Test order diff computation."""

    @pytest.fixture
    def config(self):
        return StrategyConfig(
            risk_aversion=0.05,
            max_inventory=500,
            max_order_size=100,
            base_spread=2.0,
            min_absolute_spread=2.0,
            quote_size=10,
            time_normalization_sec=86400.0,
            debounce_cents=2,
            debounce_seconds=5.0,
        )

    @pytest.fixture
    def mock_connector(self):
        connector = MagicMock()
        connector.create_order = AsyncMock(return_value={"order": {"order_id": "test-123"}})
        connector.amend_order = AsyncMock(return_value={})
        connector.cancel_order = AsyncMock(return_value={})
        connector.cancel_all_orders = AsyncMock(return_value={})
        return connector

    @pytest.fixture
    def engine(self, config, mock_connector):
        return ExecutionEngine(config, mock_connector)

    def test_diff_no_current_creates(self, engine):
        """Test diff creates new order when none exists."""
        target = StrategyOutput(
            bid_price=48,
            ask_price=52,
            bid_size=10,
            ask_size=10,
            reservation_price=50.0,
            spread=4.0,
            inventory_skew=0.0,
        )

        bid_diff, ask_diff = engine.compute_diff(
            "TEST",
            target,
            should_bid=True,
            should_ask=True,
        )

        assert bid_diff.action == OrderAction.CREATE
        assert bid_diff.price == 48
        assert bid_diff.size == 10

        assert ask_diff.action == OrderAction.CREATE
        assert ask_diff.price == 52
        assert ask_diff.size == 10

    def test_diff_cancel_when_not_wanted(self, engine):
        """Test diff cancels when quote not wanted."""
        # Setup existing order
        state = engine.get_quote_state("TEST")
        state.bid_order = Order(
            order_id="bid-123",
            ticker="TEST",
            side=OrderSide.BUY,
            price=48,
            size=10,
            remaining=10,
            filled=0,
            status=OrderStatus.RESTING,
        )

        target = StrategyOutput(
            bid_price=48,
            ask_price=52,
            bid_size=10,
            ask_size=10,
            reservation_price=50.0,
            spread=4.0,
            inventory_skew=0.0,
        )

        bid_diff, _ = engine.compute_diff(
            "TEST",
            target,
            should_bid=False,  # Don't want to bid
            should_ask=True,
        )

        assert bid_diff.action == OrderAction.CANCEL
        assert bid_diff.order_id == "bid-123"

    def test_diff_amend_on_price_change(self, engine):
        """Test diff amends when price changes."""
        state = engine.get_quote_state("TEST")
        state.bid_order = Order(
            order_id="bid-123",
            ticker="TEST",
            side=OrderSide.BUY,
            price=48,
            size=10,
            remaining=10,
            filled=0,
            status=OrderStatus.RESTING,
        )

        target = StrategyOutput(
            bid_price=47,  # Changed
            ask_price=52,
            bid_size=10,
            ask_size=10,
            reservation_price=50.0,
            spread=4.0,
            inventory_skew=0.0,
        )

        bid_diff, _ = engine.compute_diff(
            "TEST",
            target,
            should_bid=True,
            should_ask=True,
        )

        assert bid_diff.action == OrderAction.AMEND
        assert bid_diff.order_id == "bid-123"
        assert bid_diff.price == 47

    def test_diff_no_action_when_same(self, engine):
        """Test diff returns NONE when order matches target."""
        state = engine.get_quote_state("TEST")
        state.bid_order = Order(
            order_id="bid-123",
            ticker="TEST",
            side=OrderSide.BUY,
            price=48,
            size=10,
            remaining=10,
            filled=0,
            status=OrderStatus.RESTING,
        )

        target = StrategyOutput(
            bid_price=48,
            ask_price=52,
            bid_size=10,
            ask_size=10,
            reservation_price=50.0,
            spread=4.0,
            inventory_skew=0.0,
        )

        bid_diff, _ = engine.compute_diff(
            "TEST",
            target,
            should_bid=True,
            should_ask=True,
        )

        assert bid_diff.action == OrderAction.NONE


class TestDebouncing:
    """Test debouncing logic."""

    @pytest.fixture
    def config(self):
        return StrategyConfig(
            debounce_cents=2,
            debounce_seconds=5.0,
        )

    @pytest.fixture
    def mock_connector(self):
        return MagicMock()

    @pytest.fixture
    def engine(self, config, mock_connector):
        return ExecutionEngine(config, mock_connector)

    def test_should_update_on_price_change(self, engine):
        """Test update triggers on significant price change."""
        state = engine.get_quote_state("TEST")
        state.last_bid_price = 48
        state.last_ask_price = 52
        state.last_update = time.monotonic()

        target = StrategyOutput(
            bid_price=45,  # 3 cent change > 2 cent threshold
            ask_price=52,
            bid_size=10,
            ask_size=10,
            reservation_price=50.0,
            spread=4.0,
            inventory_skew=0.0,
        )

        assert engine.should_update("TEST", target)

    def test_should_not_update_small_change(self, engine):
        """Test update doesn't trigger on small price change."""
        state = engine.get_quote_state("TEST")
        state.last_bid_price = 48
        state.last_ask_price = 52
        state.last_update = time.monotonic()

        target = StrategyOutput(
            bid_price=47,  # 1 cent change < 2 cent threshold
            ask_price=52,
            bid_size=10,
            ask_size=10,
            reservation_price=50.0,
            spread=4.0,
            inventory_skew=0.0,
        )

        assert not engine.should_update("TEST", target)

    def test_should_update_after_time(self, engine):
        """Test update triggers after time threshold."""
        state = engine.get_quote_state("TEST")
        state.last_bid_price = 48
        state.last_ask_price = 52
        state.last_update = time.monotonic() - 10.0  # 10 seconds ago

        target = StrategyOutput(
            bid_price=48,  # No price change
            ask_price=52,
            bid_size=10,
            ask_size=10,
            reservation_price=50.0,
            spread=4.0,
            inventory_skew=0.0,
        )

        assert engine.should_update("TEST", target)


class TestExecution:
    """Test order execution."""

    @pytest.fixture
    def config(self):
        return StrategyConfig()

    @pytest.fixture
    def mock_connector(self):
        connector = MagicMock()
        connector.create_order = AsyncMock(return_value={"order": {"order_id": "new-123"}})
        connector.amend_order = AsyncMock(return_value={})
        connector.cancel_order = AsyncMock(return_value={})
        connector.cancel_all_orders = AsyncMock(return_value={})
        connector.get_orders = AsyncMock(return_value={"orders": []})
        return connector

    @pytest.fixture
    def engine(self, config, mock_connector):
        return ExecutionEngine(config, mock_connector)

    @pytest.mark.asyncio
    async def test_execute_create(self, engine, mock_connector):
        """Test executing a create diff."""
        # Set market state for validation (bid=45, ask=52)
        engine.set_market_state("TEST", best_bid=45, best_ask=52)

        bid_diff = OrderDiff(
            action=OrderAction.CREATE,
            side=OrderSide.BUY,
            price=48,  # Below ask, won't cross
            size=10,
        )
        ask_diff = OrderDiff(action=OrderAction.NONE)

        await engine.execute_diff("TEST", bid_diff, ask_diff)

        mock_connector.create_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_cancel(self, engine, mock_connector):
        """Test executing a cancel diff."""
        # Setup existing order
        state = engine.get_quote_state("TEST")
        state.bid_order = Order(
            order_id="bid-123",
            ticker="TEST",
            side=OrderSide.BUY,
            price=48,
            size=10,
            remaining=10,
            filled=0,
            status=OrderStatus.RESTING,
        )

        bid_diff = OrderDiff(
            action=OrderAction.CANCEL,
            order_id="bid-123",
        )
        ask_diff = OrderDiff(action=OrderAction.NONE)

        await engine.execute_diff("TEST", bid_diff, ask_diff)

        mock_connector.cancel_order.assert_called_once_with("bid-123")

    @pytest.mark.asyncio
    async def test_cancel_all(self, engine, mock_connector):
        """Test cancel all orders uses individual cancels."""
        # Setup mock to return resting orders
        mock_connector.get_orders = AsyncMock(return_value={
            "orders": [
                {"order_id": "order-1", "status": "resting"},
                {"order_id": "order-2", "status": "resting"},
                {"order_id": "order-3", "status": "executed"},  # Should be skipped
            ]
        })

        await engine.cancel_all("TEST")

        # Should fetch orders and cancel only resting ones
        mock_connector.get_orders.assert_called_once_with("TEST")
        assert mock_connector.cancel_order.call_count == 2
        mock_connector.cancel_order.assert_any_call("order-1")
        mock_connector.cancel_order.assert_any_call("order-2")
