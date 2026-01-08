"""Unit tests for the Stoikov strategy engine."""

import pytest
from src.strategy import StoikovStrategy, StoikovParams
from src.pegged import PeggedStrategy
from src.config import StrategyConfig, PeggedModeConfig


class TestStoikovStrategy:
    """Test Stoikov strategy calculations."""

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
        )
        return StoikovStrategy(config)

    def test_reservation_price_no_inventory(self, strategy):
        """Reservation price equals mid when inventory is zero."""
        params = StoikovParams(
            mid_price=50.0,
            inventory=0,
            volatility=5.0,
            gamma=0.05,
            time_horizon=1.0,
            min_spread=2.0,
        )

        r = strategy.compute_reservation_price(params)
        assert r == 50.0

    def test_reservation_price_long_inventory(self, strategy):
        """Reservation price below mid when long (want to sell)."""
        params = StoikovParams(
            mid_price=50.0,
            inventory=100,  # Long 100 contracts
            volatility=5.0,
            gamma=0.05,
            time_horizon=1.0,
            min_spread=2.0,
        )

        r = strategy.compute_reservation_price(params)

        # r = S - q * gamma * sigma^2 * (T-t)
        # r = 50 - 100 * 0.05 * 25 * 1 = 50 - 125 = -75
        # But this is extreme, showing inventory skew working
        assert r < 50.0  # Should be below mid
        expected = 50.0 - (100 * 0.05 * 25 * 1.0)
        assert r == expected

    def test_reservation_price_short_inventory(self, strategy):
        """Reservation price above mid when short (want to buy)."""
        params = StoikovParams(
            mid_price=50.0,
            inventory=-100,  # Short 100 contracts
            volatility=5.0,
            gamma=0.05,
            time_horizon=1.0,
            min_spread=2.0,
        )

        r = strategy.compute_reservation_price(params)
        assert r > 50.0  # Should be above mid

    def test_generate_quotes_no_inventory(self, strategy):
        """Test quote generation with no inventory."""
        output = strategy.generate_quotes(
            mid_price=50.0,
            inventory=0,
            volatility=5.0,
        )

        # With no inventory, quotes should be symmetric around mid
        assert output.bid_price < 50
        assert output.ask_price > 50
        assert output.bid_price < output.ask_price

    def test_generate_quotes_clamped_to_bounds(self, strategy):
        """Test that quotes are clamped to valid range."""
        # Very high mid price
        output = strategy.generate_quotes(
            mid_price=98.0,
            inventory=0,
            volatility=5.0,
        )

        assert output.ask_price <= 99
        assert output.bid_price >= 1
        assert output.bid_price < output.ask_price

    def test_generate_quotes_inventory_reduces_size(self, strategy):
        """Test that inventory reduces quote size on the risky side."""
        # Long inventory should reduce bid size
        output_long = strategy.generate_quotes(
            mid_price=50.0,
            inventory=400,  # 80% of max
            volatility=5.0,
        )

        output_neutral = strategy.generate_quotes(
            mid_price=50.0,
            inventory=0,
            volatility=5.0,
        )

        assert output_long.bid_size < output_neutral.bid_size

    def test_should_quote_at_max_inventory(self, strategy):
        """Test quoting disabled at max inventory."""
        # At max long
        should_bid, should_ask = strategy.should_quote(500)
        assert not should_bid  # Don't add to long
        assert should_ask      # Can still sell

        # At max short
        should_bid, should_ask = strategy.should_quote(-500)
        assert should_bid      # Can still buy
        assert not should_ask  # Don't add to short

    def test_quotes_never_cross(self, strategy):
        """Test that bid never equals or exceeds ask."""
        test_cases = [
            (50.0, 0, 5.0),
            (50.0, 100, 5.0),
            (50.0, -100, 5.0),
            (10.0, 0, 1.0),
            (90.0, 0, 1.0),
            (50.0, 0, 20.0),  # High volatility
        ]

        for mid, inv, vol in test_cases:
            output = strategy.generate_quotes(mid, inv, vol)
            assert output.bid_price < output.ask_price, \
                f"Crossed quotes at mid={mid}, inv={inv}, vol={vol}"


class TestPeggedStrategy:
    """Test Pegged strategy for solved markets."""

    @pytest.fixture
    def pegged_config(self):
        return PeggedModeConfig(
            enabled=True,
            fair_value=50,
            max_exposure=2000,
            reload_threshold=0.8,
        )

    @pytest.fixture
    def strategy_config(self):
        return StrategyConfig(
            max_order_size=100,
            max_inventory=500,
        )

    @pytest.fixture
    def pegged_strategy(self, pegged_config, strategy_config):
        return PeggedStrategy(pegged_config, strategy_config)

    def test_fixed_pricing_ignores_volatility(self, pegged_strategy, pegged_config):
        """Test that pegged strategy uses fixed pricing regardless of market state."""
        # Generate quotes with different inventory levels
        quotes1 = pegged_strategy.generate_quotes(inventory=0)
        quotes2 = pegged_strategy.generate_quotes(inventory=100)
        quotes3 = pegged_strategy.generate_quotes(inventory=-100)

        # Prices should always be FV-1 and FV+1
        fv = pegged_config.fair_value
        assert quotes1.bid_price == fv - 1
        assert quotes1.ask_price == fv + 1
        assert quotes2.bid_price == fv - 1
        assert quotes2.ask_price == fv + 1
        assert quotes3.bid_price == fv - 1
        assert quotes3.ask_price == fv + 1

    def test_reservation_price_is_fair_value(self, pegged_strategy, pegged_config):
        """Test that reservation price equals fair value."""
        quotes = pegged_strategy.generate_quotes(inventory=0)
        assert quotes.reservation_price == float(pegged_config.fair_value)

    def test_should_quote_at_max_exposure(self, pegged_strategy, pegged_config):
        """Test quoting disabled at max exposure."""
        max_exp = pegged_config.max_exposure

        # At max long
        should_bid, should_ask = pegged_strategy.should_quote(max_exp)
        assert not should_bid  # Don't add to long
        assert should_ask      # Can still sell

        # At max short
        should_bid, should_ask = pegged_strategy.should_quote(-max_exp)
        assert should_bid      # Can still buy
        assert not should_ask  # Don't add to short

    def test_size_reduces_with_high_inventory(self, pegged_strategy):
        """Test that sizes reduce when inventory is very high."""
        quotes_neutral = pegged_strategy.generate_quotes(inventory=0)
        quotes_long = pegged_strategy.generate_quotes(inventory=1500)  # 75% of max_exposure

        # When very long, bid size should be reduced
        assert quotes_long.bid_size < quotes_neutral.bid_size

    def test_quotes_never_cross(self, pegged_strategy):
        """Test that bid never equals or exceeds ask."""
        test_cases = [0, 500, -500, 1000, -1000]

        for inv in test_cases:
            quotes = pegged_strategy.generate_quotes(inventory=inv)
            assert quotes.bid_price < quotes.ask_price, \
                f"Crossed quotes at inv={inv}"
