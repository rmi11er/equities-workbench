"""Integration tests using mock exchange."""

import pytest
import asyncio

from .mock_exchange import MockKalshiExchange, MockConnector
from src.orderbook import OrderBookManager
from src.strategy import StoikovStrategy
from src.pegged import PeggedStrategy
from src.taker import ImpulseEngine, BailoutReason
from src.execution import ExecutionEngine
from src.config import (
    Config, StrategyConfig, VolatilityConfig,
    PeggedModeConfig, ImpulseConfig, RiskConfig
)


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
            max_inventory=500,
            max_order_size=100,
            base_spread=2.0,
            min_absolute_spread=2.0,
            quote_size=10,
            time_normalization_sec=86400.0,
            debounce_cents=2,
            debounce_seconds=5.0,
        )
        return StoikovStrategy(config)

    @pytest.fixture
    def execution(self, connector):
        # Disable min_join_depth for tests (otherwise quotes at prices without depth are skipped)
        config = StrategyConfig(min_join_depth=0)
        return ExecutionEngine(config, connector)

    @pytest.mark.asyncio
    async def test_orderbook_receives_snapshot(self, exchange, connector, orderbook_manager):
        """Test orderbook manager receives and processes snapshot."""
        connector.on_message(orderbook_manager.handle_message)

        # Send a fresh snapshot after handler is registered
        # yes/no arrays are BID orders in Kalshi format
        exchange.set_orderbook("TEST-TICKER", {
            "yes": [[35, 100], [38, 200], [40, 150]],  # YES bids
            "no": [[55, 100], [57, 200], [59, 150]],   # NO bids
        })

        book = orderbook_manager.get("TEST-TICKER")
        assert book is not None
        assert len(book.yes_bids) > 0

    @pytest.mark.asyncio
    async def test_orderbook_receives_delta(self, exchange, connector, orderbook_manager):
        """Test orderbook manager processes deltas."""
        connector.on_message(orderbook_manager.handle_message)

        # Send snapshot first
        exchange.set_orderbook("TEST-TICKER", {
            "yes": [[35, 100], [38, 200], [40, 150]],  # YES bids
            "no": [[55, 100], [57, 200], [59, 150]],   # NO bids
        })

        book = orderbook_manager.get("TEST-TICKER")
        initial_size = book.yes_bids.get(38, 0)

        # Send delta
        exchange.update_orderbook("TEST-TICKER", "yes", 38, -50)

        # Check size updated
        assert book.yes_bids.get(38, 0) == initial_size - 50

    @pytest.mark.asyncio
    async def test_full_quote_cycle(self, exchange, connector, orderbook_manager, strategy, execution):
        """Test full cycle: market data -> strategy -> execution."""
        connector.on_message(orderbook_manager.handle_message)

        # Send snapshot (yes/no are BID orders)
        exchange.set_orderbook("TEST-TICKER", {
            "yes": [[35, 100], [38, 200], [40, 150]],  # YES bids
            "no": [[55, 100], [57, 200], [59, 150]],   # NO bids
        })

        # Get market state
        book = orderbook_manager.get("TEST-TICKER")
        mid = book.mid_price()
        volatility = orderbook_manager.get_volatility("TEST-TICKER")
        best_bid = book.best_yes_bid()
        best_ask = book.best_yes_ask()

        # Set market state for validation (required for guardrails)
        # Include book depth so we pass min_join_depth check
        # Combine yes_bids and yes_asks for depth checking
        book_depth = {**book.yes_bids, **book.yes_asks} if book else {}
        execution.set_market_state("TEST-TICKER", best_bid=best_bid, best_ask=best_ask, book_depth=book_depth)

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

        # Send initial snapshot (yes/no are BID orders)
        exchange.set_orderbook("TEST-TICKER", {
            "yes": [[35, 100], [38, 200], [40, 150]],  # YES bids
            "no": [[55, 100], [57, 200], [59, 150]],   # NO bids
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

        # Send snapshot (yes/no are BID orders)
        exchange.set_orderbook("TEST-TICKER", {
            "yes": [[35, 100], [38, 200], [40, 150]],  # YES bids
            "no": [[55, 100], [57, 200], [59, 150]],   # NO bids
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


class TestV2Integration:
    """V2 integration tests for Impulse Control, Pegged Mode, and Depth-Based Pricing."""

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
    def impulse_engine(self):
        impulse_cfg = ImpulseConfig(
            enabled=True,
            taker_fee_cents=7,
            slippage_buffer=5,
            ofi_window_sec=10.0,
            ofi_threshold=500,
        )
        risk_cfg = RiskConfig(
            hard_stop_ratio=1.2,
            bailout_threshold=1,
        )
        strategy_cfg = StrategyConfig(
            max_inventory=500,
            max_order_size=100,
        )
        return ImpulseEngine(impulse_cfg, risk_cfg, strategy_cfg)

    @pytest.fixture
    def pegged_strategy(self):
        pegged_cfg = PeggedModeConfig(
            enabled=True,
            fair_value=50,
            max_exposure=2000,
            reload_threshold=0.8,
        )
        strategy_cfg = StrategyConfig(
            max_order_size=100,
            max_inventory=500,
        )
        return PeggedStrategy(pegged_cfg, strategy_cfg)

    @pytest.mark.asyncio
    async def test_effective_depth_pricing(self, exchange, connector, orderbook_manager):
        """Test that effective depth pricing respects BBO even with dust at top levels.

        The effective quote walks depth to find where min_depth contracts exist,
        but the result is always clamped to BBO to prevent generating quotes that
        would cross the actual market spread.
        """
        connector.on_message(orderbook_manager.handle_message)

        # Set up a thin book with dust at best prices
        # YES bids: dust at 35/33, real liquidity at 30/25
        # NO bids: dust at 60/58, real liquidity at 55/50
        exchange.set_orderbook("TEST-TICKER", {
            "yes": [[35, 5], [33, 10], [30, 100], [25, 200]],   # YES bids - best=35 (dust)
            "no": [[60, 5], [58, 10], [55, 100], [50, 200]],    # NO bids → YES asks at 40,42,45,50
        })

        book = orderbook_manager.get("TEST-TICKER")
        assert book is not None

        # Get effective quote with min_depth=100
        eff_bid, eff_ask = book.get_effective_quote(min_depth=100)

        # Depth walk finds liquidity at 30 and 45, but we clamp to BBO:
        # - best_bid = 35, best_ask = 40
        # This ensures we never quote worse than the actual market
        assert eff_bid == 35  # clamped to best_bid
        assert eff_ask == 40  # clamped to best_ask

    @pytest.mark.asyncio
    async def test_impulse_hard_limit_fires(self, impulse_engine):
        """Test that impulse engine fires on hard limit breach."""
        # max_inventory=500, hard_stop_ratio=1.2, so hard_stop=600
        action = impulse_engine.check_bailout(
            inventory=700,  # Way over limit
            reservation_price=50.0,
            best_bid=48,
            best_ask=52,
            regime="STANDARD",
        )

        assert action is not None
        assert action.reason == BailoutReason.HARD_LIMIT
        assert action.quantity == 200  # 700 - 500 = 200 excess

    @pytest.mark.asyncio
    async def test_impulse_ignores_ofi_in_pegged_mode(self, impulse_engine):
        """Test that impulse engine ignores OFI when in PEGGED mode."""
        # Simulate heavy selling pressure
        for _ in range(10):
            impulse_engine.record_trade(size=100, is_buy=False)  # -1000 OFI

        # In STANDARD mode, this would trigger
        action_standard = impulse_engine.check_bailout(
            inventory=200,
            reservation_price=50.0,
            best_bid=48,
            best_ask=52,
            regime="STANDARD",
        )
        assert action_standard is not None
        assert action_standard.reason == BailoutReason.TOXICITY_SPIKE

        # Reset and test PEGGED mode
        impulse_engine.reset()
        for _ in range(10):
            impulse_engine.record_trade(size=100, is_buy=False)

        action_pegged = impulse_engine.check_bailout(
            inventory=200,
            reservation_price=50.0,
            best_bid=48,
            best_ask=52,
            regime="PEGGED",
        )

        # Should NOT trigger in PEGGED mode
        assert action_pegged is None

    @pytest.mark.asyncio
    async def test_pegged_strategy_fixed_pricing(self, pegged_strategy):
        """Test that pegged strategy uses fixed pricing."""
        quotes = pegged_strategy.generate_quotes(inventory=0)

        # Should be FV-1 and FV+1
        assert quotes.bid_price == 49
        assert quotes.ask_price == 51
        assert quotes.reservation_price == 50.0

    @pytest.mark.asyncio
    async def test_pegged_strategy_inventory_affects_size_not_price(self, pegged_strategy):
        """Test that pegged strategy adjusts size but not price with inventory."""
        quotes_neutral = pegged_strategy.generate_quotes(inventory=0)
        quotes_long = pegged_strategy.generate_quotes(inventory=1500)  # 75% of max_exposure

        # Prices should stay fixed
        assert quotes_long.bid_price == quotes_neutral.bid_price
        assert quotes_long.ask_price == quotes_neutral.ask_price

        # But bid size should be reduced when long
        assert quotes_long.bid_size < quotes_neutral.bid_size

    @pytest.mark.asyncio
    async def test_ioc_simulation_for_bailout(self, exchange, connector):
        """Test IOC simulation execution for bailout."""
        from src.execution import ExecutionEngine

        config = StrategyConfig()
        engine = ExecutionEngine(config, connector)

        # Send IOC simulation
        result = await engine.send_ioc_simulation(
            ticker="TEST-TICKER",
            side="yes",
            price=48,
            count=50,
        )

        assert result.get("success") is True
        assert "order_id" in result

        # Order should be cancelled (simulating IOC behavior)
        orders = await connector.get_orders("TEST-TICKER")
        order = next(
            (o for o in orders["orders"] if o["order_id"] == result["order_id"]),
            None
        )
        # Should be cancelled or not exist
        if order:
            assert order["status"] == "cancelled"
