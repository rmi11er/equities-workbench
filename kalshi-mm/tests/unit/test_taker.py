"""Unit tests for the Impulse Engine (Taker)."""

import pytest
import time
from src.taker import ImpulseEngine, OFITracker, BailoutReason
from src.config import ImpulseConfig, RiskConfig, StrategyConfig


class TestOFITracker:
    """Test Order Flow Imbalance tracking."""

    @pytest.fixture
    def tracker(self):
        return OFITracker(window_sec=10.0)

    def test_initial_ofi_is_zero(self, tracker):
        """Test OFI starts at zero."""
        assert tracker.rolling_ofi == 0

    def test_buy_trade_increases_ofi(self, tracker):
        """Test that buy trades increase OFI."""
        tracker.record_trade(size=100, is_buy=True)
        assert tracker.rolling_ofi == 100

    def test_sell_trade_decreases_ofi(self, tracker):
        """Test that sell trades decrease OFI."""
        tracker.record_trade(size=100, is_buy=False)
        assert tracker.rolling_ofi == -100

    def test_balanced_trades_net_zero(self, tracker):
        """Test that balanced buy/sell nets to zero."""
        tracker.record_trade(size=100, is_buy=True)
        tracker.record_trade(size=100, is_buy=False)
        assert tracker.rolling_ofi == 0

    def test_reset_clears_ofi(self, tracker):
        """Test that reset clears all trades."""
        tracker.record_trade(size=100, is_buy=True)
        tracker.record_trade(size=50, is_buy=True)
        tracker.reset()
        assert tracker.rolling_ofi == 0


class TestImpulseEngine:
    """Test Impulse Engine bailout logic."""

    @pytest.fixture
    def impulse_config(self):
        return ImpulseConfig(
            enabled=True,
            taker_fee_cents=7,
            slippage_buffer=5,
            ofi_window_sec=10.0,
            ofi_threshold=500,
        )

    @pytest.fixture
    def risk_config(self):
        return RiskConfig(
            hard_stop_ratio=1.2,
            bailout_threshold=1,
        )

    @pytest.fixture
    def strategy_config(self):
        return StrategyConfig(
            max_inventory=500,
            max_order_size=100,
        )

    @pytest.fixture
    def engine(self, impulse_config, risk_config, strategy_config):
        return ImpulseEngine(impulse_config, risk_config, strategy_config)

    def test_no_bailout_normal_conditions(self, engine):
        """Test no bailout under normal conditions."""
        action = engine.check_bailout(
            inventory=100,
            reservation_price=50.0,
            best_bid=48,
            best_ask=52,
            regime="STANDARD",
        )
        assert action is None

    def test_hard_limit_bailout_long(self, engine):
        """Test hard limit bailout when very long."""
        # max_inventory=500, hard_stop_ratio=1.2, so hard_stop=600
        action = engine.check_bailout(
            inventory=650,  # > 600
            reservation_price=50.0,
            best_bid=48,
            best_ask=52,
            regime="STANDARD",
        )

        assert action is not None
        assert action.reason == BailoutReason.HARD_LIMIT
        assert action.side == "yes"  # Selling YES to reduce long
        assert action.quantity == 150  # 650 - 500 = 150 excess

    def test_hard_limit_bailout_short(self, engine):
        """Test hard limit bailout when very short."""
        action = engine.check_bailout(
            inventory=-650,  # > 600 in abs
            reservation_price=50.0,
            best_bid=48,
            best_ask=52,
            regime="STANDARD",
        )

        assert action is not None
        assert action.reason == BailoutReason.HARD_LIMIT
        assert action.side == "no"  # Buying YES by selling NO
        assert action.quantity == 150  # 650 - 500 = 150 excess

    def test_reservation_crossing_bailout_long(self, engine):
        """Test reservation crossing bailout when long."""
        # Long position, reservation below best bid
        action = engine.check_bailout(
            inventory=200,
            reservation_price=40.0,  # Way below best bid of 48
            best_bid=48,
            best_ask=52,
            regime="STANDARD",
        )

        assert action is not None
        assert action.reason == BailoutReason.RESERVATION_CROSSING
        assert action.side == "yes"
        assert action.quantity == 200  # Flatten completely

    def test_reservation_crossing_bailout_short(self, engine):
        """Test reservation crossing bailout when short."""
        # Short position, reservation above best ask
        action = engine.check_bailout(
            inventory=-200,
            reservation_price=60.0,  # Way above best ask of 52
            best_bid=48,
            best_ask=52,
            regime="STANDARD",
        )

        assert action is not None
        assert action.reason == BailoutReason.RESERVATION_CROSSING
        assert action.side == "no"
        assert action.quantity == 200  # Flatten completely

    def test_toxicity_spike_bailout_long(self, engine):
        """Test toxicity spike bailout when long with selling pressure."""
        # Simulate heavy selling pressure
        for _ in range(6):  # 6 * 100 = 600 > threshold of 500
            engine.record_trade(size=100, is_buy=False)

        action = engine.check_bailout(
            inventory=200,
            reservation_price=50.0,
            best_bid=48,
            best_ask=52,
            regime="STANDARD",
        )

        assert action is not None
        assert action.reason == BailoutReason.TOXICITY_SPIKE
        assert action.side == "yes"

    def test_toxicity_spike_bailout_short(self, engine):
        """Test toxicity spike bailout when short with buying pressure."""
        # Simulate heavy buying pressure
        for _ in range(6):  # 6 * 100 = 600 > threshold of 500
            engine.record_trade(size=100, is_buy=True)

        action = engine.check_bailout(
            inventory=-200,
            reservation_price=50.0,
            best_bid=48,
            best_ask=52,
            regime="STANDARD",
        )

        assert action is not None
        assert action.reason == BailoutReason.TOXICITY_SPIKE
        assert action.side == "no"

    def test_pegged_mode_ignores_reservation_crossing(self, engine):
        """Test that pegged mode ignores reservation crossing."""
        action = engine.check_bailout(
            inventory=200,
            reservation_price=40.0,  # Would trigger in STANDARD mode
            best_bid=48,
            best_ask=52,
            regime="PEGGED",  # But we're in PEGGED mode
        )

        # Should not trigger (only hard limit works in PEGGED mode)
        assert action is None

    def test_pegged_mode_ignores_toxicity(self, engine):
        """Test that pegged mode ignores toxicity spikes."""
        # Simulate heavy selling pressure
        for _ in range(6):
            engine.record_trade(size=100, is_buy=False)

        action = engine.check_bailout(
            inventory=200,
            reservation_price=50.0,
            best_bid=48,
            best_ask=52,
            regime="PEGGED",  # Pegged mode
        )

        # Should not trigger in PEGGED mode
        assert action is None

    def test_pegged_mode_still_triggers_hard_limit(self, engine):
        """Test that pegged mode still triggers on hard limits."""
        action = engine.check_bailout(
            inventory=650,  # > hard stop
            reservation_price=50.0,
            best_bid=48,
            best_ask=52,
            regime="PEGGED",
        )

        # Hard limit should still trigger
        assert action is not None
        assert action.reason == BailoutReason.HARD_LIMIT

    def test_disabled_engine_never_triggers(self, risk_config, strategy_config):
        """Test that disabled engine never triggers bailouts."""
        disabled_config = ImpulseConfig(enabled=False)
        engine = ImpulseEngine(disabled_config, risk_config, strategy_config)

        action = engine.check_bailout(
            inventory=1000,  # Way over limit
            reservation_price=10.0,  # Way off
            best_bid=48,
            best_ask=52,
            regime="STANDARD",
        )

        assert action is None
