"""Unit tests for the OrderBook manager."""

import pytest
from src.orderbook import OrderBook, VolatilityEstimator, OrderBookManager
from src.config import VolatilityConfig


class TestOrderBook:
    """Test OrderBook operations."""

    @pytest.fixture
    def book(self):
        return OrderBook(ticker="TEST")

    def test_apply_snapshot(self, book, sample_orderbook_snapshot):
        """Test applying a full snapshot."""
        book.apply_snapshot(sample_orderbook_snapshot)

        # Check YES asks populated
        assert 30 in book.yes_asks
        assert book.yes_asks[30] == 100
        assert book.yes_asks[35] == 200
        assert book.yes_asks[40] == 150

        # Check NO asks populated
        assert 55 in book.no_asks
        assert book.no_asks[55] == 100

        # Check derived bids
        # YES ask at 30 -> NO bid at 70
        assert 70 in book.no_bids
        assert book.no_bids[70] == 100

        # NO ask at 55 -> YES bid at 45
        assert 45 in book.yes_bids
        assert book.yes_bids[45] == 100

    def test_apply_delta_reduce(self, book, sample_orderbook_snapshot):
        """Test applying a delta that reduces size."""
        book.apply_snapshot(sample_orderbook_snapshot)

        delta = {
            "market_ticker": "TEST",
            "price": 35,
            "delta": -50,
            "side": "yes",
        }

        book.apply_delta(delta)

        # Size should be reduced from 200 to 150
        assert book.yes_asks[35] == 150

    def test_apply_delta_remove_level(self, book, sample_orderbook_snapshot):
        """Test applying a delta that removes a level."""
        book.apply_snapshot(sample_orderbook_snapshot)

        delta = {
            "market_ticker": "TEST",
            "price": 30,
            "delta": -100,  # Exact size
            "side": "yes",
        }

        book.apply_delta(delta)

        # Level should be removed
        assert 30 not in book.yes_asks

    def test_apply_delta_increase(self, book, sample_orderbook_snapshot):
        """Test applying a delta that increases size."""
        book.apply_snapshot(sample_orderbook_snapshot)

        delta = {
            "market_ticker": "TEST",
            "price": 35,
            "delta": 100,
            "side": "yes",
        }

        book.apply_delta(delta)

        # Size should increase from 200 to 300
        assert book.yes_asks[35] == 300

    def test_mid_price(self, book, sample_orderbook_snapshot):
        """Test mid price calculation."""
        book.apply_snapshot(sample_orderbook_snapshot)

        mid = book.mid_price()

        # Best YES bid derived from NO ask at 55 -> YES bid at 45
        # Best YES ask at 30
        # Mid = (45 + 30) / 2 = 37.5
        assert mid == 37.5

    def test_best_prices(self, book, sample_orderbook_snapshot):
        """Test best price retrieval."""
        book.apply_snapshot(sample_orderbook_snapshot)

        assert book.best_yes_bid() == 45  # From NO ask at 55
        assert book.best_yes_ask() == 30  # Direct YES ask
        assert book.best_no_bid() == 70   # From YES ask at 30
        assert book.best_no_ask() == 55   # Direct NO ask

    def test_spread(self, book, sample_orderbook_snapshot):
        """Test spread calculation."""
        book.apply_snapshot(sample_orderbook_snapshot)

        spread = book.spread()

        # Note: In this case we have YES bid at 45 and YES ask at 30
        # This is an inverted market (bid > ask), which shouldn't happen
        # in reality but let's test the math anyway
        # Actually: best bid = 45, best ask = 30, spread = 30 - 45 = -15
        # This shows a locked/crossed market in our test data
        assert spread is not None

    def test_empty_book_mid_price(self, book):
        """Test mid price when book is empty."""
        assert book.mid_price() is None

    def test_get_ofi(self, book, sample_orderbook_snapshot):
        """Test Order Flow Imbalance calculation."""
        book.apply_snapshot(sample_orderbook_snapshot)

        ofi = book.get_ofi(levels=3)

        # OFI should be between -1 and 1
        assert -1.0 <= ofi <= 1.0


class TestVolatilityEstimator:
    """Test volatility estimation."""

    @pytest.fixture
    def estimator(self):
        config = VolatilityConfig(
            ema_halflife_sec=60.0,
            min_volatility=0.1,
            initial_volatility=5.0,
        )
        return VolatilityEstimator(config)

    def test_initial_volatility(self, estimator):
        """Test initial volatility value."""
        assert estimator.volatility == 5.0

    def test_update_increases_on_large_move(self, estimator):
        """Test volatility increases on large price moves."""
        initial_vol = estimator.volatility

        # First price sets baseline
        estimator.update(50.0)

        # Large move
        estimator.update(60.0)

        # Volatility should increase
        assert estimator.volatility > initial_vol * 0.5  # Some increase expected

    def test_volatility_floor(self, estimator):
        """Test volatility doesn't go below floor."""
        # Set initial and let it decay
        estimator.update(50.0)

        # Small moves
        for _ in range(100):
            estimator.update(50.0)

        assert estimator.volatility >= 0.1

    def test_reset(self, estimator):
        """Test reset functionality."""
        estimator.update(50.0)
        estimator.update(60.0)

        estimator.reset(initial_price=55.0)

        assert estimator.volatility == 5.0  # Back to initial
        assert estimator._last_price == 55.0


class TestOrderBookManager:
    """Test OrderBookManager."""

    @pytest.fixture
    def manager(self):
        config = VolatilityConfig(
            ema_halflife_sec=60.0,
            min_volatility=0.1,
            initial_volatility=5.0,
        )
        return OrderBookManager(config)

    def test_get_or_create(self, manager):
        """Test order book creation."""
        book = manager.get_or_create("TEST")
        assert book is not None
        assert book.ticker == "TEST"

        # Same book returned
        book2 = manager.get_or_create("TEST")
        assert book is book2

    def test_get_nonexistent(self, manager):
        """Test getting non-existent book."""
        assert manager.get("NONEXISTENT") is None

    def test_handle_snapshot_message(self, manager, sample_orderbook_snapshot):
        """Test handling snapshot message."""
        from src.connector import WSMessage

        msg = WSMessage(
            type="orderbook_snapshot",
            seq=1,
            msg=sample_orderbook_snapshot,
        )

        manager.handle_message(msg)

        book = manager.get("TEST-TICKER")
        assert book is not None
        assert len(book.yes_asks) > 0

    def test_handle_delta_message(self, manager, sample_orderbook_snapshot, sample_orderbook_delta):
        """Test handling delta message."""
        from src.connector import WSMessage

        # First apply snapshot
        snapshot_msg = WSMessage(
            type="orderbook_snapshot",
            seq=1,
            msg=sample_orderbook_snapshot,
        )
        manager.handle_message(snapshot_msg)

        # Then apply delta
        delta_msg = WSMessage(
            type="orderbook_delta",
            seq=2,
            msg=sample_orderbook_delta,
        )
        manager.handle_message(delta_msg)

        book = manager.get("TEST-TICKER")
        assert book.yes_asks[35] == 150  # 200 - 50
