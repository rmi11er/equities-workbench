"""Unit tests for the Stoikov strategy engine."""

import pytest
from src.strategy import StoikovStrategy, StoikovParams
from src.pegged import PeggedStrategy
from src.config import StrategyConfig, PeggedModeConfig, LayeringConfig
from src.types import LayerIndex


class TestStoikovStrategy:
    """Test Stoikov strategy calculations."""

    @pytest.fixture
    def strategy(self):
        config = StrategyConfig(
            risk_aversion=0.05,
            max_inventory_dollars=250.0,  # 500 contracts at 50c
            max_order_size_dollars=50.0,  # 100 contracts at 50c
            base_spread=2.0,
            min_absolute_spread=2.0,
            quote_size_dollars=5.0,  # $5 = 10 contracts at 50c
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
            max_spread=0,
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
            max_spread=0,
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
            max_spread=0,
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
        """Test that inventory scaling affects quote size on the risky side."""
        # Test the _compute_sizes method directly to verify inventory scaling
        # With dollar-based sizing, prices affect contract counts, but the
        # inventory SCALE factor should still be applied

        # At mid=50, bid ~48c, ask ~52c
        # $5 at 50c = 10 contracts base

        # Test: inventory scaling reduces bid when long
        # max_inventory_dollars=250 at 50c = 500 contracts
        bid_size, ask_size = strategy._compute_sizes(
            inventory=250,  # 50% of max_inventory (500 contracts at 50c)
            bid_price=50,
            ask_price=50,
        )
        # With 50% inventory: bid_scale = max(0.1, 1 - 0.5) = 0.5
        # Base at 50c: $5 / 0.50 = 10 contracts
        # Scaled bid: 10 * 0.5 = 5
        assert bid_size == 5
        assert ask_size == 10  # Ask not scaled when long

        # Test: inventory scaling reduces ask when short
        bid_size, ask_size = strategy._compute_sizes(
            inventory=-250,  # 50% short
            bid_price=50,
            ask_price=50,
        )
        assert bid_size == 10  # Bid not scaled when short
        assert ask_size == 5   # Ask scaled down

    def test_should_quote_at_max_inventory(self, strategy):
        """Test quoting disabled at max inventory."""
        # max_inventory_dollars=250
        # At mid=50c: long YES uses 50c, short YES (long NO) uses 50c too
        # At max long (500 * 0.50 = $250 = max)
        should_bid, should_ask = strategy.should_quote(500, mid_price=50.0)
        assert not should_bid  # Don't add to long (at limit)
        assert should_ask      # Can still sell

        # At max short (500 * 0.50 = $250 = max, using NO price = 100-50 = 50c)
        should_bid, should_ask = strategy.should_quote(-500, mid_price=50.0)
        assert should_bid      # Can still buy
        assert not should_ask  # Don't add to short (at limit)

        # Below max - should allow quoting both sides
        should_bid, should_ask = strategy.should_quote(400, mid_price=50.0)  # $200 < $250
        assert should_bid      # Can still bid
        assert should_ask      # Can still sell

        # Test with asymmetric pricing (YES at 30c, NO at 70c)
        # Short 358 contracts * 70c = $250.60 (over limit of $250)
        should_bid, should_ask = strategy.should_quote(-358, mid_price=30.0)
        assert should_bid      # Can still buy
        assert not should_ask  # Don't add to short (over limit)

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
            max_order_size_dollars=50.0,  # 100 contracts at 50c
            max_inventory_dollars=250.0,  # 500 contracts at 50c
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


class TestLayeringStrategy:
    """Test v2.2 layering system for quote ladders."""

    @pytest.fixture
    def strategy(self):
        config = StrategyConfig(
            risk_aversion=0.05,
            max_inventory_dollars=250.0,
            max_order_size_dollars=50.0,
            base_spread=2.0,
            min_absolute_spread=2.0,
            quote_size_dollars=5.0,
            time_normalization_sec=86400.0,
        )
        return StoikovStrategy(config)

    @pytest.fixture
    def layering_config(self):
        return LayeringConfig(
            enabled=True,
            layer_count=3,
            base_tick_step=1,
            canary_size_ratio=0.5,
            body_size_ratio=1.0,
            wall_size_ratio=2.0,
            canary_timeout_ms=50,
            high_vol_spread_multiplier=2.0,
            wall_only_in_high_vol=True,
            high_vol_timeout_sec=30.0,
        )

    def test_generate_ladder_normal_mode(self, strategy, layering_config):
        """Test ladder generation in normal (non-high-vol) mode."""
        ladder = strategy.generate_ladder(
            mid_price=50.0,
            inventory=0,
            volatility=5.0,
            layering_config=layering_config,
            is_high_vol=False,
        )

        # Should have 3 bid levels and 3 ask levels
        assert len(ladder.bids) == 3
        assert len(ladder.asks) == 3

        # All layers should be active in normal mode
        for bid in ladder.bids:
            assert bid.is_active is True
        for ask in ladder.asks:
            assert ask.is_active is True

        # Check layer indices
        assert ladder.bids[0].layer_index == 0  # Canary
        assert ladder.bids[1].layer_index == 1  # Body
        assert ladder.bids[2].layer_index == 2  # Wall

        # Canary should have smallest size, Wall should have largest
        assert ladder.bids[0].size < ladder.bids[1].size  # Canary < Body
        assert ladder.bids[2].size > ladder.bids[1].size  # Wall > Body

    def test_generate_ladder_high_vol_wall_only(self, strategy, layering_config):
        """Test that only Wall layer is active in high-vol mode."""
        ladder = strategy.generate_ladder(
            mid_price=50.0,
            inventory=0,
            volatility=5.0,
            layering_config=layering_config,
            is_high_vol=True,
        )

        # Should still have 3 levels, but only Wall active
        assert len(ladder.bids) == 3
        assert len(ladder.asks) == 3

        # Only Wall (layer 2) should be active
        assert ladder.bids[0].is_active is False  # Canary disabled
        assert ladder.bids[1].is_active is False  # Body disabled
        assert ladder.bids[2].is_active is True   # Wall active

        assert ladder.asks[0].is_active is False
        assert ladder.asks[1].is_active is False
        assert ladder.asks[2].is_active is True

    def test_ladder_prices_spread_apart(self, strategy, layering_config):
        """Test that layer prices are spread apart by tick_step."""
        ladder = strategy.generate_ladder(
            mid_price=50.0,
            inventory=0,
            volatility=5.0,
            layering_config=layering_config,
            is_high_vol=False,
        )

        # Bids should decrease in price (layer 0 is best, layer 2 is worst)
        assert ladder.bids[0].price >= ladder.bids[1].price
        assert ladder.bids[1].price >= ladder.bids[2].price

        # Asks should increase in price (layer 0 is best, layer 2 is worst)
        assert ladder.asks[0].price <= ladder.asks[1].price
        assert ladder.asks[1].price <= ladder.asks[2].price

    def test_ladder_never_crosses(self, strategy, layering_config):
        """Test that ladder bids never exceed asks."""
        test_cases = [
            (50.0, 0, 5.0),
            (50.0, 100, 5.0),
            (50.0, -100, 5.0),
            (10.0, 0, 1.0),
            (90.0, 0, 1.0),
        ]

        for mid, inv, vol in test_cases:
            ladder = strategy.generate_ladder(
                mid_price=mid,
                inventory=inv,
                volatility=vol,
                layering_config=layering_config,
                is_high_vol=False,
            )

            # Best bid should be less than best ask
            best_bid = ladder.best_bid
            best_ask = ladder.best_ask

            if best_bid and best_ask:
                assert best_bid.price < best_ask.price, \
                    f"Ladder crossed at mid={mid}, inv={inv}, vol={vol}"

    def test_layering_disabled_fallback(self, strategy):
        """Test that disabled layering falls back to single quote."""
        disabled_config = LayeringConfig(enabled=False)

        ladder = strategy.generate_ladder(
            mid_price=50.0,
            inventory=0,
            volatility=5.0,
            layering_config=disabled_config,
            is_high_vol=False,
        )

        # Should have single bid and ask
        assert len(ladder.bids) == 1
        assert len(ladder.asks) == 1
        assert ladder.bids[0].layer_index == LayerIndex.BODY
        assert ladder.asks[0].layer_index == LayerIndex.BODY

    def test_high_vol_wider_spread(self, strategy, layering_config):
        """Test that high-vol mode uses wider tick spacing."""
        normal_ladder = strategy.generate_ladder(
            mid_price=50.0,
            inventory=0,
            volatility=5.0,
            layering_config=layering_config,
            is_high_vol=False,
        )

        high_vol_ladder = strategy.generate_ladder(
            mid_price=50.0,
            inventory=0,
            volatility=5.0,
            layering_config=layering_config,
            is_high_vol=True,
        )

        # Wall layer in high-vol should be at wider price than normal
        normal_wall_bid = normal_ladder.bids[2].price
        high_vol_wall_bid = high_vol_ladder.bids[2].price

        # High-vol wall should be at a worse (lower) bid price
        # because tick_step is multiplied by spread_multiplier
        assert high_vol_wall_bid <= normal_wall_bid

    def test_layer_index_enum(self):
        """Test LayerIndex enum values."""
        assert LayerIndex.CANARY == 0
        assert LayerIndex.BODY == 1
        assert LayerIndex.WALL == 2
