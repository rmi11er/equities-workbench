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

        # Check YES bids populated (from "yes" array)
        assert 35 in book.yes_bids
        assert book.yes_bids[35] == 100
        assert book.yes_bids[38] == 200
        assert book.yes_bids[40] == 150

        # Check NO bids populated (from "no" array)
        assert 55 in book.no_bids
        assert book.no_bids[55] == 100

        # Check derived asks
        # YES bid at 40 -> NO ask at 60
        assert 60 in book.no_asks
        assert book.no_asks[60] == 150

        # NO bid at 59 -> YES ask at 41
        assert 41 in book.yes_asks
        assert book.yes_asks[41] == 150

    def test_apply_delta_reduce(self, book, sample_orderbook_snapshot):
        """Test applying a delta that reduces size."""
        book.apply_snapshot(sample_orderbook_snapshot)

        delta = {
            "market_ticker": "TEST",
            "price": 38,
            "delta": -50,
            "side": "yes",
        }

        book.apply_delta(delta)

        # Size should be reduced from 200 to 150
        assert book.yes_bids[38] == 150

    def test_apply_delta_remove_level(self, book, sample_orderbook_snapshot):
        """Test applying a delta that removes a level."""
        book.apply_snapshot(sample_orderbook_snapshot)

        delta = {
            "market_ticker": "TEST",
            "price": 35,
            "delta": -100,  # Exact size
            "side": "yes",
        }

        book.apply_delta(delta)

        # Level should be removed
        assert 35 not in book.yes_bids

    def test_apply_delta_increase(self, book, sample_orderbook_snapshot):
        """Test applying a delta that increases size."""
        book.apply_snapshot(sample_orderbook_snapshot)

        delta = {
            "market_ticker": "TEST",
            "price": 38,
            "delta": 100,
            "side": "yes",
        }

        book.apply_delta(delta)

        # Size should increase from 200 to 300
        assert book.yes_bids[38] == 300

    def test_mid_price(self, book, sample_orderbook_snapshot):
        """Test mid price calculation."""
        book.apply_snapshot(sample_orderbook_snapshot)

        mid = book.mid_price()

        # Best YES bid = 40 (max of yes array)
        # Best YES ask = 41 (100 - max of no array = 100 - 59)
        # Mid = (40 + 41) / 2 = 40.5
        assert mid == 40.5

    def test_best_prices(self, book, sample_orderbook_snapshot):
        """Test best price retrieval."""
        book.apply_snapshot(sample_orderbook_snapshot)

        assert book.best_yes_bid() == 40  # Max of yes array
        assert book.best_yes_ask() == 41  # 100 - max(no) = 100 - 59
        assert book.best_no_bid() == 59   # Max of no array
        assert book.best_no_ask() == 60   # 100 - max(yes) = 100 - 40

    def test_spread(self, book, sample_orderbook_snapshot):
        """Test spread calculation."""
        book.apply_snapshot(sample_orderbook_snapshot)

        spread = book.spread()

        # Best YES bid = 40, Best YES ask = 41
        # Spread = 41 - 40 = 1
        assert spread == 1.0

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
        assert len(book.yes_bids) > 0

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
        assert book.yes_bids[38] == 150  # 200 - 50


class TestEffectiveQuote:
    """Test effective quote calculation (V2 depth-based pricing)."""

    @pytest.fixture
    def empty_book(self):
        return OrderBook(ticker="TEST")

    @pytest.fixture
    def thin_book(self):
        """Book with dust orders at top levels."""
        book = OrderBook(ticker="TEST")
        # YES asks (sell YES): small dust at top, real liquidity deeper
        book.yes_asks = {
            40: 5,    # 5 contracts at 40 (dust)
            42: 10,   # 10 contracts at 42 (dust)
            45: 100,  # 100 contracts at 45 (real liquidity)
            50: 200,  # 200 contracts at 50
        }
        # YES bids (buy YES): small dust at top, real liquidity deeper
        book.yes_bids = {
            35: 5,    # 5 contracts at 35 (dust)
            33: 10,   # 10 contracts at 33 (dust)
            30: 100,  # 100 contracts at 30 (real liquidity)
            25: 200,  # 200 contracts at 25
        }
        return book

    @pytest.fixture
    def deep_book(self):
        """Book with plenty of liquidity at top levels."""
        book = OrderBook(ticker="TEST")
        book.yes_asks = {
            40: 150,  # 150 contracts at 40
            42: 200,  # 200 contracts at 42
            45: 300,  # 300 contracts at 45
        }
        book.yes_bids = {
            38: 150,  # 150 contracts at 38
            35: 200,  # 200 contracts at 35
            30: 300,  # 300 contracts at 30
        }
        return book

    def test_empty_book_returns_fallback(self, empty_book):
        """Test that empty book returns (1, 99) fallback."""
        eff_bid, eff_ask = empty_book.get_effective_quote(min_depth=100)
        assert eff_bid == 1
        assert eff_ask == 99

    def test_thin_book_ignores_dust(self, thin_book):
        """Test that thin book uses BBO - we never quote worse than best prices."""
        # With min_depth=100, even if depth is found at worse prices,
        # we clamp to best_bid/best_ask to avoid crossing the spread
        eff_bid, eff_ask = thin_book.get_effective_quote(min_depth=100)

        # Effective bid should be at best_bid (35), never worse
        assert eff_bid == 35

        # Effective ask should be at best_ask (40), never worse
        assert eff_ask == 40

    def test_deep_book_uses_best_price(self, deep_book):
        """Test that deep book uses best prices when liquidity is sufficient."""
        eff_bid, eff_ask = deep_book.get_effective_quote(min_depth=100)

        # Should use best prices since they have enough liquidity
        assert eff_bid == 38
        assert eff_ask == 40

    def test_effective_mid_calculation(self, thin_book):
        """Test effective mid price calculation."""
        eff_mid = thin_book.get_effective_mid(min_depth=100)

        # With BBO clamping: effective_bid=35, effective_ask=40
        # (35 + 40) / 2 = 37.5
        assert eff_mid == 37.5

    def test_effective_spread_calculation(self, thin_book):
        """Test effective spread calculation."""
        eff_spread = thin_book.get_effective_spread(min_depth=100)

        # With BBO clamping: effective_bid=35, effective_ask=40
        # Spread = 40 - 35 = 5
        assert eff_spread == 5.0

    def test_cumulative_depth_across_levels(self):
        """Test that depth is accumulated across multiple levels but clamped to BBO."""
        book = OrderBook(ticker="TEST")
        book.yes_asks = {
            40: 30,   # 30 contracts at 40
            42: 30,   # 30 contracts at 42
            45: 30,   # 30 contracts at 45
            50: 30,   # 30 contracts at 50
        }
        book.yes_bids = {
            35: 30,
            33: 30,
            30: 30,
            25: 30,
        }

        # With min_depth=100, depth is found at worse prices but clamped to BBO
        eff_bid, eff_ask = book.get_effective_quote(min_depth=100)

        # Ask: Depth found at 50 but clamped to best_ask=40
        assert eff_ask == 40

        # Bid: Depth found at 25 but clamped to best_bid=35
        assert eff_bid == 35

    def test_book_with_insufficient_total_depth(self):
        """Test book with less total depth than min_depth."""
        book = OrderBook(ticker="TEST")
        book.yes_asks = {
            40: 20,
            45: 30,
        }
        book.yes_bids = {
            35: 20,
            30: 30,
        }

        # Total ask depth = 50, total bid depth = 50
        # min_depth = 100 exceeds total depth
        eff_bid, eff_ask = book.get_effective_quote(min_depth=100)

        # Should return BEST available prices (actual BBO) for thin books
        # This ensures we never generate quotes that cross the actual market
        assert eff_bid == 35  # Best (highest) bid
        assert eff_ask == 40  # Best (lowest) ask


class TestDepthLevels:
    """Test get_depth_levels() - unclamped depth price discovery."""

    def test_liquid_bbo_returns_bbo(self):
        """When BBO has sufficient depth, depth_bid/ask equals BBO."""
        book = OrderBook(ticker="TEST")
        book.yes_bids = {36: 40000}  # 40k contracts at 36
        book.yes_asks = {37: 40000}  # 40k contracts at 37

        depth_bid, depth_ask = book.get_depth_levels(min_depth=300)

        assert depth_bid == 36  # Depth found at BBO
        assert depth_ask == 37  # Depth found at BBO

    def test_thin_bbo_returns_deeper_levels(self):
        """When BBO has dust, return where real depth exists."""
        book = OrderBook(ticker="TEST")
        book.yes_bids = {
            36: 50,    # Dust at BBO
            35: 50,    # More dust
            34: 500,   # Real depth here
        }
        book.yes_asks = {
            37: 50,    # Dust at BBO
            38: 50,    # More dust
            39: 500,   # Real depth here
        }

        depth_bid, depth_ask = book.get_depth_levels(min_depth=300)

        assert depth_bid == 34  # Had to go deeper to find 300
        assert depth_ask == 39  # Had to go deeper to find 300

    def test_cumulative_depth_accumulates(self):
        """Depth accumulates across levels until threshold met."""
        book = OrderBook(ticker="TEST")
        book.yes_bids = {
            36: 100,   # 100 at 36
            35: 100,   # +100 = 200 at 35
            34: 100,   # +100 = 300 at 34 (threshold met)
            33: 500,
        }
        book.yes_asks = {
            37: 150,   # 150 at 37
            38: 150,   # +150 = 300 at 38 (threshold met)
            39: 500,
        }

        depth_bid, depth_ask = book.get_depth_levels(min_depth=300)

        assert depth_bid == 34  # Cumulative 300 reached at 34
        assert depth_ask == 38  # Cumulative 300 reached at 38


class TestQuoteClamping:
    """
    Test the quote clamping logic for depth-based spread adjustment.

    The rules are:
    1. LIQUID BBO (depth at BBO >= min_depth): Quote AT the BBO
       - bid = max(strategy_bid, best_bid) - push UP to BBO
       - ask = min(strategy_ask, best_ask) - push DOWN to BBO

    2. THIN BBO (depth at BBO < min_depth): Quote at depth levels
       - bid = min(strategy_bid, depth_bid) - push DOWN to depth
       - ask = max(strategy_ask, depth_ask) - push UP to depth
    """

    @pytest.fixture
    def liquid_book(self):
        """Book with 40k contracts at BBO - very liquid."""
        book = OrderBook(ticker="TEST")
        book.yes_bids = {36: 40000}
        book.yes_asks = {37: 40000}
        return book

    @pytest.fixture
    def thin_book_with_depth(self):
        """Book with dust at BBO, real liquidity deeper."""
        book = OrderBook(ticker="TEST")
        book.yes_bids = {
            36: 50,    # Dust at BBO
            34: 500,   # Real depth
        }
        book.yes_asks = {
            37: 50,    # Dust at BBO
            39: 500,   # Real depth
        }
        return book

    @pytest.fixture
    def penny_wide_liquid(self):
        """Penny-wide market with good liquidity."""
        book = OrderBook(ticker="TEST")
        book.yes_bids = {45: 1000}
        book.yes_asks = {46: 1000}
        return book

    @pytest.fixture
    def wide_liquid(self):
        """Wide market (3 cents) with good liquidity."""
        book = OrderBook(ticker="TEST")
        book.yes_bids = {45: 1000}
        book.yes_asks = {48: 1000}
        return book

    # =========================================================================
    # LIQUID MARKET TESTS - Should quote AT BBO
    # =========================================================================

    def test_liquid_market_strategy_too_wide_tightens_to_bbo(self, liquid_book):
        """
        When market is liquid and strategy wants to quote wider than BBO,
        we should tighten to BBO.

        Market: 36 bid (40k) / 37 ask (40k)
        Strategy wants: 35 bid / 38 ask (too wide)
        Result: 36 bid / 37 ask (tightened to BBO)
        """
        clamped_bid, clamped_ask = liquid_book.clamp_quotes_to_depth(
            strategy_bid=35,
            strategy_ask=38,
            min_depth=300,
        )

        assert clamped_bid == 36, "Should tighten bid UP to BBO in liquid market"
        assert clamped_ask == 37, "Should tighten ask DOWN to BBO in liquid market"

    def test_liquid_market_strategy_at_bbo_stays_at_bbo(self, liquid_book):
        """
        When market is liquid and strategy already at BBO, stay there.

        Market: 36 bid (40k) / 37 ask (40k)
        Strategy wants: 36 bid / 37 ask
        Result: 36 bid / 37 ask (unchanged)
        """
        clamped_bid, clamped_ask = liquid_book.clamp_quotes_to_depth(
            strategy_bid=36,
            strategy_ask=37,
            min_depth=300,
        )

        assert clamped_bid == 36
        assert clamped_ask == 37

    def test_liquid_penny_wide_quotes_penny_wide(self, penny_wide_liquid):
        """
        Penny-wide liquid market should result in penny-wide quotes.

        Market: 45 bid (1k) / 46 ask (1k)
        Strategy wants: 44 bid / 47 ask (too wide)
        Result: 45 bid / 46 ask (penny-wide at BBO)
        """
        clamped_bid, clamped_ask = penny_wide_liquid.clamp_quotes_to_depth(
            strategy_bid=44,
            strategy_ask=47,
            min_depth=300,
        )

        assert clamped_bid == 45, "Should be at BBO bid"
        assert clamped_ask == 46, "Should be at BBO ask"
        assert clamped_ask - clamped_bid == 1, "Should be penny-wide"

    def test_liquid_wide_market_matches_market_spread(self, wide_liquid):
        """
        Wide but liquid market - match the market spread.

        Market: 45 bid (1k) / 48 ask (1k) - 3 cent spread
        Strategy wants: 44 bid / 49 ask (too wide)
        Result: 45 bid / 48 ask (match market)
        """
        clamped_bid, clamped_ask = wide_liquid.clamp_quotes_to_depth(
            strategy_bid=44,
            strategy_ask=49,
            min_depth=300,
        )

        assert clamped_bid == 45
        assert clamped_ask == 48
        assert clamped_ask - clamped_bid == 3, "Should match market spread"

    # =========================================================================
    # THIN MARKET TESTS - Should quote at depth levels (wider)
    # =========================================================================

    def test_thin_market_widens_to_depth_levels(self, thin_book_with_depth):
        """
        When market has dust at BBO, widen to where real depth exists.

        Market: 36 bid (50) / 37 ask (50) - dust
                34 bid (500) / 39 ask (500) - real depth
        Strategy wants: 35 bid / 38 ask
        Result: 34 bid / 39 ask (widened to depth)
        """
        clamped_bid, clamped_ask = thin_book_with_depth.clamp_quotes_to_depth(
            strategy_bid=35,
            strategy_ask=38,
            min_depth=300,
        )

        assert clamped_bid == 34, "Should widen bid DOWN to depth level"
        assert clamped_ask == 39, "Should widen ask UP to depth level"

    def test_thin_market_strategy_already_at_depth_stays(self, thin_book_with_depth):
        """
        When strategy already at depth levels, stay there.

        Market: dust at 36/37, depth at 34/39
        Strategy wants: 34 bid / 39 ask (already at depth)
        Result: 34 bid / 39 ask (unchanged)
        """
        clamped_bid, clamped_ask = thin_book_with_depth.clamp_quotes_to_depth(
            strategy_bid=34,
            strategy_ask=39,
            min_depth=300,
        )

        assert clamped_bid == 34
        assert clamped_ask == 39

    def test_thin_market_strategy_wider_than_depth_uses_strategy(self, thin_book_with_depth):
        """
        When strategy wants to quote wider than depth, allow it.

        Market: dust at 36/37, depth at 34/39
        Strategy wants: 33 bid / 40 ask (wider than depth)
        Result: 33 bid / 40 ask (strategy wins, it's more conservative)
        """
        clamped_bid, clamped_ask = thin_book_with_depth.clamp_quotes_to_depth(
            strategy_bid=33,
            strategy_ask=40,
            min_depth=300,
        )

        assert clamped_bid == 33, "Strategy can quote wider than depth"
        assert clamped_ask == 40, "Strategy can quote wider than depth"

    # =========================================================================
    # EDGE CASES
    # =========================================================================

    def test_empty_book_returns_fallback(self):
        """Empty book should return wide fallback quotes."""
        book = OrderBook(ticker="TEST")

        clamped_bid, clamped_ask = book.clamp_quotes_to_depth(
            strategy_bid=50,
            strategy_ask=50,
            min_depth=300,
        )

        # Should return safe fallback
        assert clamped_bid == 1
        assert clamped_ask == 99

    def test_one_sided_book_bid_only(self):
        """Book with only bids should handle gracefully."""
        book = OrderBook(ticker="TEST")
        book.yes_bids = {36: 1000}

        clamped_bid, clamped_ask = book.clamp_quotes_to_depth(
            strategy_bid=35,
            strategy_ask=40,
            min_depth=300,
        )

        assert clamped_bid == 36  # Liquid bid side, tighten to BBO
        assert clamped_ask == 99  # No asks, fallback

    def test_one_sided_book_ask_only(self):
        """Book with only asks should handle gracefully."""
        book = OrderBook(ticker="TEST")
        book.yes_asks = {37: 1000}

        clamped_bid, clamped_ask = book.clamp_quotes_to_depth(
            strategy_bid=35,
            strategy_ask=40,
            min_depth=300,
        )

        assert clamped_bid == 1   # No bids, fallback
        assert clamped_ask == 37  # Liquid ask side, tighten to BBO

    def test_prevents_crossed_quotes(self):
        """Should never return crossed quotes."""
        book = OrderBook(ticker="TEST")
        book.yes_bids = {50: 1000}
        book.yes_asks = {45: 1000}  # Crossed book (unusual)

        clamped_bid, clamped_ask = book.clamp_quotes_to_depth(
            strategy_bid=48,
            strategy_ask=47,
            min_depth=300,
        )

        # Should ensure bid < ask
        assert clamped_bid < clamped_ask, "Quotes must not cross"

    def test_seattle_nfc_scenario(self):
        """
        Real scenario: Seattle NFC Championship.
        BBO: 36 bid with 40k contracts / 37 ask with 40k contracts
        We should quote 36/37, NOT 35/38.
        """
        book = OrderBook(ticker="KXNFLNFCCHAMP-25-SEA")
        book.yes_bids = {36: 40000}
        book.yes_asks = {37: 40000}

        # Strategy (Stoikov) wants to quote wider due to volatility
        clamped_bid, clamped_ask = book.clamp_quotes_to_depth(
            strategy_bid=35,
            strategy_ask=38,
            min_depth=300,
        )

        assert clamped_bid == 36, "Must bid at BBO when liquid"
        assert clamped_ask == 37, "Must ask at BBO when liquid"

    def test_mvp_scenario_with_dust(self):
        """
        Real scenario: MVP market with dust at top.
        BBO: 61 bid (50 contracts) / 62 ask (50 contracts) - dust
        Real depth: 59 bid (500) / 64 ask (500)
        We should quote 59/64 (wider), NOT 61/62.
        """
        book = OrderBook(ticker="KXNFLMVP-26-MSTA")
        book.yes_bids = {
            61: 50,   # Dust
            60: 50,   # Dust
            59: 500,  # Real depth
        }
        book.yes_asks = {
            62: 50,   # Dust
            63: 50,   # Dust
            64: 500,  # Real depth
        }

        clamped_bid, clamped_ask = book.clamp_quotes_to_depth(
            strategy_bid=60,
            strategy_ask=63,
            min_depth=300,
        )

        assert clamped_bid == 59, "Should widen to depth level"
        assert clamped_ask == 64, "Should widen to depth level"
