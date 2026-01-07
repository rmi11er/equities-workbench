"""
Vectorized backtesting engine for rapid strategy validation.

Uses NumPy/Pandas vectorized operations for maximum speed.
No loops - pure matrix operations.
"""

import logging
from datetime import datetime
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from moat.schemas import BacktestConfig, BacktestResult
from moat.stats import (
    calculate_all_metrics,
    calculate_returns,
    max_drawdown,
)

logger = logging.getLogger(__name__)

# Type alias for strategy functions
StrategyFn = Callable[[pd.DataFrame], pd.Series]


class VectorizedBacktester:
    """High-speed vectorized backtesting engine.

    Executes strategies using matrix operations for rapid hypothesis
    rejection. Handles signal lagging (prevent look-ahead bias),
    transaction costs, and basic position sizing.
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        """Initialize backtester.

        Args:
            config: Backtest configuration. Uses defaults if None.
        """
        self.config = config or BacktestConfig()

    def run(
        self,
        data: pd.DataFrame,
        strategy: Union[StrategyFn, pd.Series],
        strategy_name: str = "unnamed",
        symbol: str = "unknown",
    ) -> BacktestResult:
        """Run backtest on OHLCV data.

        Args:
            data: DataFrame with OHLCV columns, indexed by date.
            strategy: Either a function that takes data and returns
                      position signals, or a pre-computed signal Series.
            strategy_name: Name for reporting.
            symbol: Symbol being traded.

        Returns:
            BacktestResult with performance metrics.
        """
        data = data.copy()

        # Generate signals
        if callable(strategy):
            signals = strategy(data)
        else:
            signals = strategy

        # Validate signals
        signals = self._validate_signals(signals, data.index)

        # Lag signals by 1 period to prevent look-ahead bias
        # (trade on next bar's open after signal generated)
        signals = signals.shift(1).fillna(0)

        # Clip to leverage limits
        if self.config.max_leverage > 0:
            signals = signals.clip(-self.config.max_leverage, self.config.max_leverage)

        if not self.config.allow_shorting:
            signals = signals.clip(lower=0)

        # Calculate returns
        returns = self._calculate_strategy_returns(data, signals)

        # Apply transaction costs
        returns = self._apply_costs(returns, signals)

        # Build equity curve
        equity_curve = self.config.initial_capital * (1 + returns).cumprod()

        # Calculate metrics
        metrics = calculate_all_metrics(returns)

        # Count trades (position changes)
        position_changes = signals.diff().abs()
        total_trades = int((position_changes > 0.01).sum())

        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            config=self.config,
            start_date=data.index[0].to_pydatetime()
            if hasattr(data.index[0], "to_pydatetime")
            else datetime.fromisoformat(str(data.index[0])),
            end_date=data.index[-1].to_pydatetime()
            if hasattr(data.index[-1], "to_pydatetime")
            else datetime.fromisoformat(str(data.index[-1])),
            total_return=metrics["total_return"],
            annualized_return=metrics["annualized_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"],
            max_drawdown=metrics["max_drawdown"],
            calmar_ratio=metrics["calmar_ratio"],
            total_trades=total_trades,
            win_rate=metrics["win_rate"],
            equity_curve=equity_curve.tolist(),
        )

    def _validate_signals(
        self,
        signals: pd.Series,
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        """Validate and align signals with data index."""
        if not isinstance(signals, pd.Series):
            signals = pd.Series(signals, index=index)

        # Align to data index
        signals = signals.reindex(index)

        # Fill NaN with 0 (no position)
        signals = signals.fillna(0)

        # Ensure numeric
        signals = pd.to_numeric(signals, errors="coerce").fillna(0)

        return signals

    def _calculate_strategy_returns(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
    ) -> pd.Series:
        """Calculate strategy returns from signals.

        Assumes execution at close price. For more realistic
        simulation, use the event-driven engine.
        """
        # Use close-to-close returns
        price_returns = calculate_returns(data["close"])

        # Strategy return = position * market return
        strategy_returns = signals * price_returns

        return strategy_returns.fillna(0)

    def _apply_costs(
        self,
        returns: pd.Series,
        signals: pd.Series,
    ) -> pd.Series:
        """Apply transaction costs to returns.

        Costs are applied when position changes.
        """
        # Calculate turnover (absolute change in position)
        turnover = signals.diff().abs().fillna(0)

        # Total cost per bar = commission + slippage
        cost_per_trade = self.config.commission_pct + self.config.slippage_pct
        costs = turnover * cost_per_trade

        return returns - costs


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def std(series: pd.Series, period: int) -> pd.Series:
    """Rolling standard deviation."""
    return series.rolling(window=period).std()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Returns:
        Tuple of (middle, upper, lower) bands.
    """
    middle = sma(series, period)
    rolling_std = std(series, period)
    upper = middle + (rolling_std * num_std)
    lower = middle - (rolling_std * num_std)
    return middle, upper, lower


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD indicator.

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Detect when series1 crosses above series2.

    Returns:
        Boolean series, True on crossover bars.
    """
    above = series1 > series2
    prev_above = above.shift(1).fillna(False).astype(bool)
    return above & ~prev_above


def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Detect when series1 crosses below series2.

    Returns:
        Boolean series, True on crossunder bars.
    """
    below = series1 < series2
    prev_below = below.shift(1).fillna(False).astype(bool)
    return below & ~prev_below


def rank(series: pd.Series) -> pd.Series:
    """Percentile rank of values (0-1)."""
    return series.rank(pct=True)


def zscore(series: pd.Series, period: int) -> pd.Series:
    """Rolling z-score."""
    mean = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    return (series - mean) / rolling_std


# Export indicators for easy access
INDICATORS = {
    "sma": sma,
    "ema": ema,
    "std": std,
    "rsi": rsi,
    "bollinger_bands": bollinger_bands,
    "macd": macd,
    "crossover": crossover,
    "crossunder": crossunder,
    "rank": rank,
    "zscore": zscore,
}
