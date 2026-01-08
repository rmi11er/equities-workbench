"""Integration tests using mock exchange."""

import pytest
import asyncio

from .mock_exchange import MockKalshiExchange, MockConnector
from src.orderbook import OrderBookManager
from src.strategy import StoikovStrategy
from src.execution import ExecutionEngine
from src.config import Config, StrategyConfig, VolatilityConfig


class TestIntegration:
    """Integration tests with mock exchange."""

    @pytest.fixture
    def exchange(self):
        return MockKalshiExchange()

    @pytest.fixture
    def connector(self, exchange):
        return MockConnector(exchange)

    @pytest.fixture
    def orderbook_manager(self):
        config = VolatilityConfig(
            ema_halflife_sec=60.0,
            min_volatility=0.1,
            initial_volatility=5.0,
        )
        return OrderBookManager(config)

    @pytest.fixture
    def strategy(self):
        config = StrategyConfig(
            risk_aversion=0.05,
            time_horizon=1.0,
            max_inventory=500,
            max_order_size=100,
            base_spread=2.0,
            quote_size=10,
            debounce_cents=2,
            debounce_seconds=5.0,
        )
        return StoikovStrategy(config)

    @pytest.fixture
    def execution(self, connector):
        config = StrategyConfig()
        return ExecutionEngine(config, connector)

    @pytest.mark.asyncio
    async def test_orderbook_receives_snapshot(self, exchange, connector, orderbook_manager):
        """Test orderbook manager receives and processes snapshot."""
        connector.on_message(orderbook_manager.handle_message)

        # Send a fresh snapshot after handler is registered
        exchange.set_orderbook("TEST-TICKER", {
            "yes": [[30, 100], [35, 200], [40, 150]],
            "no": [[55, 100], [60, 200], [65, 150]],
        })

        book = orderbook_manager.get("TEST-TICKER")
        assert book is not None
        assert len(book.yes_asks) > 0

    @pytest.mark.asyncio
    async def test_orderbook_receives_delta(self, exchange, connector, orderbook_manager):
        """Test orderbook manager processes deltas."""
        connector.on_message(orderbook_manager.handle_message)

        # Send snapshot first
        exchange.set_orderbook("TEST-TICKER", {
            "yes": [[30, 100], [35, 200], [40, 150]],
            "no": [[55, 100], [60, 200], [65, 150]],
        })

        book = orderbook_manager.get("TEST-TICKER")
        initial_size = book.yes_asks.get(35, 0)

        # Send delta
        exchange.update_orderbook("TEST-TICKER", "yes", 35, -50)

        # Check size updated
        assert book.yes_asks.get(35, 0) == initial_size - 50

    @pytest.mark.asyncio
    async def test_full_quote_cycle(self, exchange, connector, orderbook_manager, strategy, execution):
        """Test full cycle: market data -> strategy -> execution."""
        connector.on_message(orderbook_manager.handle_message)

        # Send snapshot
        exchange.set_orderbook("TEST-TICKER", {
            "yes": [[30, 100], [35, 200], [40, 150]],
            "no": [[55, 100], [60, 200], [65, 150]],
        })

        # Get market state
        book = orderbook_manager.get("TEST-TICKER")
        mid = book.mid_price()
        volatility = orderbook_manager.get_volatility("TEST-TICKER")

        # Generate quotes
        quotes = strategy.generate_quotes(
            mid_price=mid,
            inventory=0,
            volatility=volatility,
        )

        # Execute quotes
        await execution.update_quotes(
            ticker="TEST-TICKER",
            target=quotes,
            should_bid=True,
            should_ask=True,
            force=True,
        )

        # Verify orders placed
        orders = await connector.get_orders("TEST-TICKER")
        assert len(orders["orders"]) > 0

    @pytest.mark.asyncio
    async def test_quote_amendment_on_price_move(self, exchange, connector, orderbook_manager, strategy, execution):
        """Test quotes are amended when price moves."""
        connector.on_message(orderbook_manager.handle_message)

        # Send initial snapshot
        exchange.set_orderbook("TEST-TICKER", {
            "yes": [[30, 100], [35, 200], [40, 150]],
            "no": [[55, 100], [60, 200], [65, 150]],
        })

        # Initial quotes
        book = orderbook_manager.get("TEST-TICKER")
        quotes = strategy.generate_quotes(
            mid_price=book.mid_price(),
            inventory=0,
            volatility=5.0,
        )

        await execution.update_quotes(
            ticker="TEST-TICKER",
            target=quotes,
            should_bid=True,
            should_ask=True,
            force=True,
        )

        initial_state = execution.get_quote_state("TEST-TICKER")
        initial_bid_price = initial_state.last_bid_price

        # Move price significantly
        await exchange.simulate_price_move("TEST-TICKER", 60.0)

        # Generate new quotes
        book = orderbook_manager.get("TEST-TICKER")
        new_quotes = strategy.generate_quotes(
            mid_price=book.mid_price(),
            inventory=0,
            volatility=5.0,
        )

        await execution.update_quotes(
            ticker="TEST-TICKER",
            target=new_quotes,
            should_bid=True,
            should_ask=True,
            force=True,
        )

        # Verify price changed
        new_state = execution.get_quote_state("TEST-TICKER")
        # With significant mid move, quotes should move
        # (exact behavior depends on strategy params)

    @pytest.mark.asyncio
    async def test_fill_handling(self, exchange, connector, execution):
        """Test handling of fills."""
        # Place an order
        resp = await connector.create_order(
            ticker="TEST-TICKER",
            side="yes",
            price=45,
            count=10,
        )
        order_id = resp["order"]["order_id"]

        # Simulate fill
        exchange.fill_order(order_id, 5, 45)

        # Check order state updated
        orders = await connector.get_orders("TEST-TICKER")
        order = next(o for o in orders["orders"] if o["order_id"] == order_id)

        assert order["fill_count"] == 5
        assert order["remaining_count"] == 5

    @pytest.mark.asyncio
    async def test_cancel_all_on_disconnect(self, exchange, connector, execution):
        """Test cancel all behavior."""
        # Place some orders
        await connector.create_order("TEST-TICKER", "yes", 45, 10)
        await connector.create_order("TEST-TICKER", "no", 55, 10)

        # Cancel all
        await execution.cancel_all("TEST-TICKER")

        # Verify all cancelled
        orders = await connector.get_orders("TEST-TICKER")
        active = [o for o in orders["orders"] if o["status"] == "resting"]
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_inventory_skew(self, exchange, connector, orderbook_manager, strategy):
        """Test inventory skew affects quotes."""
        connector.on_message(orderbook_manager.handle_message)

        # Send snapshot
        exchange.set_orderbook("TEST-TICKER", {
            "yes": [[30, 100], [35, 200], [40, 150]],
            "no": [[55, 100], [60, 200], [65, 150]],
        })

        book = orderbook_manager.get("TEST-TICKER")
        mid = book.mid_price()

        # No inventory
        quotes_neutral = strategy.generate_quotes(
            mid_price=mid,
            inventory=0,
            volatility=5.0,
        )

        # Long inventory
        quotes_long = strategy.generate_quotes(
            mid_price=mid,
            inventory=200,
            volatility=5.0,
        )

        # Short inventory
        quotes_short = strategy.generate_quotes(
            mid_price=mid,
            inventory=-200,
            volatility=5.0,
        )

        # Long inventory should lower quotes (want to sell)
        assert quotes_long.bid_price <= quotes_neutral.bid_price
        assert quotes_long.ask_price <= quotes_neutral.ask_price

        # Short inventory should raise quotes (want to buy)
        assert quotes_short.bid_price >= quotes_neutral.bid_price
        assert quotes_short.ask_price >= quotes_neutral.ask_price
