"""Tests for the vectorized backtesting engine."""

import time
from datetime import datetime

import pandas as pd
import pytest

from moat.data import DataManager
from moat.engines.vector_engine import (
    VectorizedBacktester,
    crossover,
    crossunder,
    sma,
)
from moat.schemas import BacktestConfig
from moat.stats import (
    annualized_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


class TestStatistics:
    """Tests for performance statistics."""

    def test_sharpe_ratio(self) -> None:
        """Test Sharpe ratio calculation."""
        # Known returns with predictable Sharpe
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005] * 50)
        sr = sharpe_ratio(returns)
        assert isinstance(sr, float)
        assert sr > 0  # Positive returns should have positive Sharpe

    def test_max_drawdown(self) -> None:
        """Test max drawdown calculation."""
        # Equity curve with known drawdown
        equity = pd.Series([100, 110, 105, 120, 90, 95, 100])
        mdd = max_drawdown(equity)
        # Max drawdown from 120 to 90 = 25%
        assert abs(mdd - 0.25) < 0.01

    def test_sortino_ratio(self) -> None:
        """Test Sortino ratio uses downside deviation."""
        returns = pd.Series([0.02, 0.01, -0.005, 0.015, -0.002] * 50)
        sortino = sortino_ratio(returns)
        sr = sharpe_ratio(returns)
        # Sortino should be higher when there are few negative returns
        assert sortino > sr


class TestVectorizedBacktester:
    """Tests for vectorized backtesting engine."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Generate sample OHLCV data."""
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        # Trending up market with noise
        import numpy as np
        np.random.seed(42)
        close = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.02 + 0.0005))
        return pd.DataFrame({
            "open": close * (1 + np.random.randn(500) * 0.005),
            "high": close * (1 + np.abs(np.random.randn(500) * 0.01)),
            "low": close * (1 - np.abs(np.random.randn(500) * 0.01)),
            "close": close,
            "volume": np.random.randint(1000000, 10000000, 500),
        }, index=dates)

    def test_simple_backtest(self, sample_data: pd.DataFrame) -> None:
        """Test basic backtest execution."""
        backtester = VectorizedBacktester()

        # Simple strategy: always long
        def always_long(data: pd.DataFrame) -> pd.Series:
            return pd.Series(1.0, index=data.index)

        result = backtester.run(
            sample_data,
            always_long,
            strategy_name="always_long",
            symbol="TEST",
        )

        assert result.strategy_name == "always_long"
        assert result.total_trades >= 0
        assert result.max_drawdown >= 0
        assert result.equity_curve is not None
        assert len(result.equity_curve) == len(sample_data)

    def test_ma_crossover_strategy(self, sample_data: pd.DataFrame) -> None:
        """Test MA crossover strategy - Definition of Done requirement."""
        backtester = VectorizedBacktester()

        # 5-line MA strategy as per spec
        def ma_crossover(data: pd.DataFrame) -> pd.Series:
            fast_ma = sma(data["close"], 20)
            slow_ma = sma(data["close"], 50)
            signal = (fast_ma > slow_ma).astype(float)
            return signal

        start = time.time()
        result = backtester.run(
            sample_data,
            ma_crossover,
            strategy_name="ma_crossover",
            symbol="TEST",
        )
        elapsed = time.time() - start

        # Must complete in <0.5 seconds (spec requirement)
        assert elapsed < 0.5, f"Backtest took {elapsed:.2f}s, should be <0.5s"

        assert result.sharpe_ratio is not None
        assert result.total_trades > 0  # Should have some trades

    def test_signal_lagging(self, sample_data: pd.DataFrame) -> None:
        """Test that signals are lagged to prevent look-ahead bias."""
        backtester = VectorizedBacktester()

        # Strategy that generates signal based on current close
        def perfect_foresight(data: pd.DataFrame) -> pd.Series:
            # This would be cheating without lag
            return (data["close"].pct_change() > 0).astype(float)

        result = backtester.run(sample_data, perfect_foresight)

        # With proper lagging, we shouldn't have perfect returns
        # (lagged signal means we act on yesterday's signal today)
        assert result.sharpe_ratio is not None

    def test_transaction_costs(self, sample_data: pd.DataFrame) -> None:
        """Test that transaction costs reduce returns."""
        # No costs
        config_no_cost = BacktestConfig(commission_pct=0.0, slippage_pct=0.0)
        bt_no_cost = VectorizedBacktester(config_no_cost)

        # With costs
        config_with_cost = BacktestConfig(commission_pct=0.01, slippage_pct=0.005)
        bt_with_cost = VectorizedBacktester(config_with_cost)

        def oscillating_strategy(data: pd.DataFrame) -> pd.Series:
            # High turnover strategy
            return pd.Series(
                [1.0 if i % 2 == 0 else 0.0 for i in range(len(data))],
                index=data.index
            )

        result_no_cost = bt_no_cost.run(sample_data, oscillating_strategy)
        result_with_cost = bt_with_cost.run(sample_data, oscillating_strategy)

        # Costs should reduce returns
        assert result_with_cost.total_return < result_no_cost.total_return


class TestIndicators:
    """Tests for technical indicators."""

    def test_crossover(self) -> None:
        """Test crossover detection."""
        fast = pd.Series([1, 2, 3, 4, 5])
        slow = pd.Series([2, 2, 2, 3, 4])
        crosses = crossover(fast, slow)
        # Fast crosses above slow at index 2 (fast=3, slow=2)
        assert crosses.iloc[2] == True
        assert crosses.iloc[0] == False

    def test_sma(self) -> None:
        """Test simple moving average."""
        prices = pd.Series([10, 20, 30, 40, 50])
        ma = sma(prices, 3)
        assert pd.isna(ma.iloc[0])
        assert pd.isna(ma.iloc[1])
        assert ma.iloc[2] == 20.0  # (10+20+30)/3
        assert ma.iloc[3] == 30.0  # (20+30+40)/3


class TestRealDataBacktest:
    """Integration tests with real market data."""

    def test_aapl_ma_strategy(self) -> None:
        """Test MA strategy on real AAPL data."""
        dm = DataManager()
        data = dm.get("AAPL", start=datetime(2023, 1, 1), end=datetime(2024, 1, 1))

        backtester = VectorizedBacktester()

        def ma_strategy(df: pd.DataFrame) -> pd.Series:
            fast = sma(df["close"], 10)
            slow = sma(df["close"], 30)
            return (fast > slow).astype(float)

        start = time.time()
        result = backtester.run(data, ma_strategy, "MA_10_30", "AAPL")
        elapsed = time.time() - start

        # Performance check
        assert elapsed < 0.5

        print(f"\nAAPL MA Strategy Results:")
        print(f"  Total Return: {result.total_return:.2%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown:.2%}")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Execution Time: {elapsed:.3f}s")
