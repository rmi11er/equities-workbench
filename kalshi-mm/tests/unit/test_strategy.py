"""Unit tests for the Stoikov strategy engine."""

import pytest
from src.strategy import StoikovStrategy, StoikovParams
from src.config import StrategyConfig


class TestStoikovStrategy:
    """Test Stoikov strategy calculations."""

    @pytest.fixture
    def strategy(self):
        config = StrategyConfig(
            risk_aversion=0.05,
            time_horizon=1.0,
            max_inventory=500,
            max_order_size=100,
            base_spread=2.0,
            quote_size=10,
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
            base_spread=2.0,
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
            base_spread=2.0,
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
            base_spread=2.0,
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
