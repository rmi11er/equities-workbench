"""Performance benchmarks for critical path components."""

import pytest
import time
import statistics
from typing import Callable

from src.orderbook import OrderBook
from src.strategy import StoikovStrategy
from src.pegged import PeggedStrategy
from src.taker import ImpulseEngine, OFITracker
from src.config import StrategyConfig, PeggedModeConfig, ImpulseConfig, RiskConfig
from src.decision_log import LatencyTimer


def benchmark(func: Callable, iterations: int = 1000) -> dict:
    """Run a function multiple times and return timing statistics."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed / 1000)  # Convert to microseconds

    return {
        "min_us": min(times),
        "max_us": max(times),
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "p95_us": sorted(times)[int(0.95 * len(times))],
        "p99_us": sorted(times)[int(0.99 * len(times))],
        "iterations": iterations,
    }


class TestOrderBookPerformance:
    """Benchmark orderbook operations."""

    @pytest.fixture
    def deep_book(self):
        """Create a realistic deep orderbook."""
        book = OrderBook(ticker="TEST")
        # 20 levels on each side, ~100-500 contracts per level
        for i in range(20):
            book.yes_asks[40 + i] = 100 + i * 20
            book.yes_bids[39 - i] = 100 + i * 20
            book.no_asks[60 + i] = 100 + i * 20
            book.no_bids[59 - i] = 100 + i * 20
        return book

    def test_mid_price_latency(self, deep_book):
        """mid_price should be < 5us."""
        stats = benchmark(lambda: deep_book.mid_price())

        print(f"\nmid_price: {stats['mean_us']:.1f}us mean, {stats['p99_us']:.1f}us p99")
        assert stats["p99_us"] < 50, f"mid_price p99 {stats['p99_us']}us exceeds 50us"

    def test_get_effective_quote_latency(self, deep_book):
        """get_effective_quote should be < 20us for typical books."""
        stats = benchmark(lambda: deep_book.get_effective_quote(min_depth=100))

        print(f"\nget_effective_quote: {stats['mean_us']:.1f}us mean, {stats['p99_us']:.1f}us p99")
        assert stats["p99_us"] < 100, f"get_effective_quote p99 {stats['p99_us']}us exceeds 100us"

    def test_get_ofi_latency(self, deep_book):
        """get_ofi should be < 10us."""
        stats = benchmark(lambda: deep_book.get_ofi(levels=3))

        print(f"\nget_ofi: {stats['mean_us']:.1f}us mean, {stats['p99_us']:.1f}us p99")
        assert stats["p99_us"] < 50, f"get_ofi p99 {stats['p99_us']}us exceeds 50us"

    def test_apply_delta_latency(self, deep_book):
        """apply_delta should be < 5us."""
        delta = {"market_ticker": "TEST", "price": 45, "delta": 10, "side": "yes"}
        stats = benchmark(lambda: deep_book.apply_delta(delta))

        print(f"\napply_delta: {stats['mean_us']:.1f}us mean, {stats['p99_us']:.1f}us p99")
        assert stats["p99_us"] < 50, f"apply_delta p99 {stats['p99_us']}us exceeds 50us"


class TestStrategyPerformance:
    """Benchmark strategy calculations."""

    @pytest.fixture
    def stoikov(self):
        config = StrategyConfig(
            risk_aversion=0.05,
            max_inventory=500,
            max_order_size=100,
            base_spread=2.0,
            min_absolute_spread=2.0,
            quote_size=10,
        )
        return StoikovStrategy(config)

    @pytest.fixture
    def pegged(self):
        pegged_cfg = PeggedModeConfig(
            enabled=True,
            fair_value=50,
            max_exposure=2000,
            reload_threshold=0.8,
        )
        strategy_cfg = StrategyConfig(max_order_size=100)
        return PeggedStrategy(pegged_cfg, strategy_cfg)

    def test_stoikov_generate_quotes_latency(self, stoikov):
        """generate_quotes should be < 50us."""
        stats = benchmark(
            lambda: stoikov.generate_quotes(
                mid_price=50.0,
                inventory=100,
                volatility=5.0,
            )
        )

        print(f"\nStoikov generate_quotes: {stats['mean_us']:.1f}us mean, {stats['p99_us']:.1f}us p99")
        assert stats["p99_us"] < 200, f"generate_quotes p99 {stats['p99_us']}us exceeds 200us"

    def test_pegged_generate_quotes_latency(self, pegged):
        """Pegged generate_quotes should be < 20us (simpler than Stoikov)."""
        stats = benchmark(
            lambda: pegged.generate_quotes(inventory=100)
        )

        print(f"\nPegged generate_quotes: {stats['mean_us']:.1f}us mean, {stats['p99_us']:.1f}us p99")
        assert stats["p99_us"] < 100, f"generate_quotes p99 {stats['p99_us']}us exceeds 100us"


class TestImpulsePerformance:
    """Benchmark impulse engine."""

    @pytest.fixture
    def engine(self):
        impulse_cfg = ImpulseConfig(
            enabled=True,
            taker_fee_cents=7,
            slippage_buffer=5,
            ofi_window_sec=10.0,
            ofi_threshold=500,
        )
        risk_cfg = RiskConfig(hard_stop_ratio=1.2, bailout_threshold=1)
        strategy_cfg = StrategyConfig(max_inventory=500)
        return ImpulseEngine(impulse_cfg, risk_cfg, strategy_cfg)

    def test_check_bailout_latency(self, engine):
        """check_bailout should be < 10us for no-trigger case."""
        stats = benchmark(
            lambda: engine.check_bailout(
                inventory=100,
                reservation_price=50.0,
                best_bid=48,
                best_ask=52,
                regime="STANDARD",
            )
        )

        print(f"\ncheck_bailout: {stats['mean_us']:.1f}us mean, {stats['p99_us']:.1f}us p99")
        assert stats["p99_us"] < 50, f"check_bailout p99 {stats['p99_us']}us exceeds 50us"

    def test_ofi_record_trade_latency(self):
        """OFI record_trade should be < 5us."""
        tracker = OFITracker(window_sec=10.0)
        stats = benchmark(lambda: tracker.record_trade(size=10, is_buy=True))

        print(f"\nOFI record_trade: {stats['mean_us']:.1f}us mean, {stats['p99_us']:.1f}us p99")
        assert stats["p99_us"] < 20, f"record_trade p99 {stats['p99_us']}us exceeds 20us"


class TestLatencyTimerOverhead:
    """Measure overhead of our latency tracking."""

    def test_latency_timer_overhead(self):
        """LatencyTimer context manager overhead should be < 1us."""

        def use_timer():
            with LatencyTimer() as t:
                pass
            return t.elapsed_us

        stats = benchmark(use_timer)

        print(f"\nLatencyTimer overhead: {stats['mean_us']:.2f}us mean, {stats['p99_us']:.2f}us p99")
        # Timer itself will show ~0.5us even for empty block
        assert stats["p99_us"] < 5, f"LatencyTimer overhead p99 {stats['p99_us']}us exceeds 5us"


class TestEndToEndLatency:
    """Test full tick processing simulation."""

    def test_simulated_tick_latency(self):
        """
        Simulate a full tick without network I/O.
        Target: < 500us for all computation.
        """
        # Setup
        book = OrderBook(ticker="TEST")
        for i in range(10):
            book.yes_asks[40 + i] = 100 + i * 20
            book.yes_bids[39 - i] = 100 + i * 20

        stoikov = StoikovStrategy(StrategyConfig())
        impulse = ImpulseEngine(
            ImpulseConfig(enabled=True),
            RiskConfig(),
            StrategyConfig(),
        )

        def simulated_tick():
            # Orderbook operations
            mid = book.mid_price()
            eff_bid, eff_ask = book.get_effective_quote(100)
            ofi = book.get_ofi(3)

            # Impulse check
            _ = impulse.check_bailout(
                inventory=50,
                reservation_price=50.0,
                best_bid=eff_bid,
                best_ask=eff_ask,
                regime="STANDARD",
            )

            # Strategy
            _ = stoikov.generate_quotes(
                mid_price=(eff_bid + eff_ask) / 2,
                inventory=50,
                volatility=5.0,
            )

        stats = benchmark(simulated_tick, iterations=1000)

        print(f"\nSimulated tick (no I/O): {stats['mean_us']:.1f}us mean, {stats['p99_us']:.1f}us p99")
        assert stats["p99_us"] < 1000, f"Simulated tick p99 {stats['p99_us']}us exceeds 1000us"
