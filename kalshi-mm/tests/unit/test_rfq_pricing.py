"""Unit tests for RFQ pricing engine."""

import pytest
from typing import Optional

from src.rfq.types import RFQ, RFQLeg
from src.rfq.config import PricingConfig
from src.rfq.pricing import PricingEngine, PriceSource


class MockPriceSource:
    """Mock price source for testing."""

    def __init__(self, prices: dict[str, float]):
        """
        Args:
            prices: Dict mapping market_ticker -> YES probability (0-1)
        """
        self._prices = prices

    def get_yes_price(self, market_ticker: str) -> Optional[float]:
        return self._prices.get(market_ticker)

    def get_no_price(self, market_ticker: str) -> Optional[float]:
        yes_price = self.get_yes_price(market_ticker)
        if yes_price is None:
            return None
        return 1.0 - yes_price


@pytest.fixture
def pricing_config():
    """Default pricing config for tests."""
    return PricingConfig(
        default_spread_pct=0.05,
        min_spread_cents=2,
        max_spread_cents=15,
        use_bbo_mid=True,
    )


@pytest.fixture
def mock_prices():
    """Mock prices for common test markets."""
    return {
        "MARKET-A": 0.60,  # 60% YES probability
        "MARKET-B": 0.40,  # 40% YES probability
        "MARKET-C": 0.80,  # 80% YES probability
        "MARKET-D": 0.25,  # 25% YES probability
    }


class TestPricingEngine:
    """Tests for PricingEngine."""

    def test_single_market_pricing(self, pricing_config, mock_prices):
        """Test pricing a single market RFQ."""
        source = MockPriceSource(mock_prices)
        engine = PricingEngine(pricing_config, source)

        from datetime import datetime
        rfq = RFQ(
            id="test-rfq-1",
            market_ticker="MARKET-A",
            contracts=100,
            created_at=datetime.now(),
        )

        theo = engine.compute_fair_value(rfq)
        assert theo == 0.60

    def test_two_leg_parlay_pricing(self, pricing_config, mock_prices):
        """Test pricing a 2-leg parlay (independent events)."""
        source = MockPriceSource(mock_prices)
        engine = PricingEngine(pricing_config, source)

        from datetime import datetime
        rfq = RFQ(
            id="test-rfq-2",
            contracts=100,
            created_at=datetime.now(),
            mve_collection_ticker="TEST-COLLECTION",
            mve_selected_legs=[
                RFQLeg(
                    event_ticker="EVENT-A",
                    market_ticker="MARKET-A",
                    side="yes",
                ),
                RFQLeg(
                    event_ticker="EVENT-B",
                    market_ticker="MARKET-B",
                    side="yes",
                ),
            ],
        )

        # Expected: 0.60 * 0.40 = 0.24
        theo = engine.compute_fair_value(rfq)
        assert theo == pytest.approx(0.24)

    def test_three_leg_parlay_pricing(self, pricing_config, mock_prices):
        """Test pricing a 3-leg parlay."""
        source = MockPriceSource(mock_prices)
        engine = PricingEngine(pricing_config, source)

        from datetime import datetime
        rfq = RFQ(
            id="test-rfq-3",
            contracts=100,
            created_at=datetime.now(),
            mve_collection_ticker="TEST-COLLECTION",
            mve_selected_legs=[
                RFQLeg(
                    event_ticker="EVENT-A",
                    market_ticker="MARKET-A",
                    side="yes",
                ),
                RFQLeg(
                    event_ticker="EVENT-B",
                    market_ticker="MARKET-B",
                    side="yes",
                ),
                RFQLeg(
                    event_ticker="EVENT-C",
                    market_ticker="MARKET-C",
                    side="yes",
                ),
            ],
        )

        # Expected: 0.60 * 0.40 * 0.80 = 0.192
        theo = engine.compute_fair_value(rfq)
        assert theo == pytest.approx(0.192)

    def test_parlay_with_no_sides(self, pricing_config, mock_prices):
        """Test parlay where some legs bet on NO."""
        source = MockPriceSource(mock_prices)
        engine = PricingEngine(pricing_config, source)

        from datetime import datetime
        rfq = RFQ(
            id="test-rfq-4",
            contracts=100,
            created_at=datetime.now(),
            mve_collection_ticker="TEST-COLLECTION",
            mve_selected_legs=[
                RFQLeg(
                    event_ticker="EVENT-A",
                    market_ticker="MARKET-A",
                    side="yes",  # P(YES) = 0.60
                ),
                RFQLeg(
                    event_ticker="EVENT-B",
                    market_ticker="MARKET-B",
                    side="no",   # P(NO) = 1 - 0.40 = 0.60
                ),
            ],
        )

        # Expected: 0.60 * 0.60 = 0.36
        theo = engine.compute_fair_value(rfq)
        assert theo == pytest.approx(0.36)

    def test_missing_leg_price_returns_none(self, pricing_config, mock_prices):
        """Test that missing leg prices cause pricing to fail."""
        source = MockPriceSource(mock_prices)
        engine = PricingEngine(pricing_config, source)

        from datetime import datetime
        rfq = RFQ(
            id="test-rfq-5",
            contracts=100,
            created_at=datetime.now(),
            mve_collection_ticker="TEST-COLLECTION",
            mve_selected_legs=[
                RFQLeg(
                    event_ticker="EVENT-A",
                    market_ticker="MARKET-A",
                    side="yes",
                ),
                RFQLeg(
                    event_ticker="EVENT-UNKNOWN",
                    market_ticker="UNKNOWN-MARKET",  # Not in mock_prices
                    side="yes",
                ),
            ],
        )

        theo = engine.compute_fair_value(rfq)
        assert theo is None

    def test_fallback_source(self, pricing_config):
        """Test fallback to secondary price source."""
        primary = MockPriceSource({"MARKET-A": 0.60})  # Only has A
        fallback = MockPriceSource({"MARKET-B": 0.40})  # Only has B
        engine = PricingEngine(pricing_config, primary, fallback)

        from datetime import datetime
        rfq = RFQ(
            id="test-rfq-6",
            contracts=100,
            created_at=datetime.now(),
            mve_collection_ticker="TEST-COLLECTION",
            mve_selected_legs=[
                RFQLeg(
                    event_ticker="EVENT-A",
                    market_ticker="MARKET-A",
                    side="yes",
                ),
                RFQLeg(
                    event_ticker="EVENT-B",
                    market_ticker="MARKET-B",
                    side="yes",
                ),
            ],
        )

        # A from primary (0.60), B from fallback (0.40)
        # Expected: 0.60 * 0.40 = 0.24
        theo = engine.compute_fair_value(rfq)
        assert theo == pytest.approx(0.24)


class TestQuotePricing:
    """Tests for quote price calculation."""

    def test_compute_quote_prices_basic(self, pricing_config):
        """Test basic quote price calculation."""
        source = MockPriceSource({})
        engine = PricingEngine(pricing_config, source)

        # Fair value 0.50 with 5% spread
        # yes_bid = 0.50 - 0.025 = 0.475
        # no_bid = 0.50 - 0.025 = 0.475
        yes_bid, no_bid = engine.compute_quote_prices(0.50, 100)

        assert float(yes_bid) == pytest.approx(0.475, abs=0.001)
        assert float(no_bid) == pytest.approx(0.475, abs=0.001)

    def test_compute_quote_prices_asymmetric(self, pricing_config):
        """Test quote prices with asymmetric fair value."""
        source = MockPriceSource({})
        engine = PricingEngine(pricing_config, source)

        # Fair value 0.70 with 5% spread
        # yes_bid = 0.70 - 0.025 = 0.675
        # no_bid = (1 - 0.70) - 0.025 = 0.275
        yes_bid, no_bid = engine.compute_quote_prices(0.70, 100)

        assert float(yes_bid) == pytest.approx(0.675, abs=0.001)
        assert float(no_bid) == pytest.approx(0.275, abs=0.001)

    def test_quote_prices_bounded(self):
        """Test that quote prices stay in valid range."""
        config = PricingConfig(default_spread_pct=0.50)  # Very wide spread
        source = MockPriceSource({})
        engine = PricingEngine(config, source)

        # Extreme case: fair value near edge
        yes_bid, no_bid = engine.compute_quote_prices(0.05, 100)

        # Should not go below 0.01
        assert float(yes_bid) >= 0.01
        assert float(no_bid) >= 0.01


class TestGetLegPrices:
    """Tests for leg price extraction."""

    def test_get_leg_prices(self, pricing_config, mock_prices):
        """Test extracting individual leg prices."""
        source = MockPriceSource(mock_prices)
        engine = PricingEngine(pricing_config, source)

        from datetime import datetime
        rfq = RFQ(
            id="test-rfq",
            contracts=100,
            created_at=datetime.now(),
            mve_selected_legs=[
                RFQLeg(
                    event_ticker="EVENT-A",
                    market_ticker="MARKET-A",
                    side="yes",
                ),
                RFQLeg(
                    event_ticker="EVENT-B",
                    market_ticker="MARKET-B",
                    side="no",  # NO side
                ),
                RFQLeg(
                    event_ticker="EVENT-UNKNOWN",
                    market_ticker="UNKNOWN",
                    side="yes",
                ),
            ],
        )

        leg_prices = engine.get_leg_prices(rfq)

        assert leg_prices["MARKET-A"] == pytest.approx(0.60)
        assert leg_prices["MARKET-B"] == pytest.approx(0.60)  # NO = 1 - 0.40
        assert leg_prices["UNKNOWN"] is None
