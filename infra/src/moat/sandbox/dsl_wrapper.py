"""
Domain-Specific Language (DSL) wrapper for strategy development.

Exposes only allowed financial primitives to the sandbox environment.
Agents can use these functions to build strategies safely.
"""

from typing import Any

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average.

    Args:
        series: Price or indicator series.
        period: Lookback period.

    Returns:
        Moving average series.
    """
    return series.rolling(window=int(period)).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average.

    Args:
        series: Price or indicator series.
        period: Lookback period (span).

    Returns:
        EMA series.
    """
    return series.ewm(span=int(period), adjust=False).mean()


def std(series: pd.Series, period: int) -> pd.Series:
    """Rolling standard deviation.

    Args:
        series: Price or indicator series.
        period: Lookback period.

    Returns:
        Rolling std series.
    """
    return series.rolling(window=int(period)).std()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index.

    Args:
        series: Price series (typically close).
        period: RSI period (default 14).

    Returns:
        RSI series (0-100).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=int(period)).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=int(period)).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD indicator.

    Args:
        series: Price series.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line period.

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    fast_ema = ema(series, int(fast))
    slow_ema = ema(series, int(slow))
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, int(signal))
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Args:
        series: Price series.
        period: Moving average period.
        num_std: Number of standard deviations for bands.

    Returns:
        Tuple of (middle, upper, lower) bands.
    """
    middle = sma(series, int(period))
    rolling_std = std(series, int(period))
    upper = middle + (rolling_std * num_std)
    lower = middle - (rolling_std * num_std)
    return middle, upper, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: ATR period.

    Returns:
        ATR series.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=int(period)).mean()


def rank(series: pd.Series) -> pd.Series:
    """Percentile rank of values (0-1).

    Useful for cross-sectional comparisons.

    Args:
        series: Any numeric series.

    Returns:
        Percentile rank series.
    """
    return series.rank(pct=True)


def zscore(series: pd.Series, period: int) -> pd.Series:
    """Rolling z-score.

    Args:
        series: Price or indicator series.
        period: Lookback period.

    Returns:
        Z-score series.
    """
    mean = series.rolling(window=int(period)).mean()
    rolling_std = series.rolling(window=int(period)).std()
    return (series - mean) / rolling_std.replace(0, np.nan)


def pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Percentage change.

    Args:
        series: Price series.
        periods: Number of periods for change calculation.

    Returns:
        Percentage change series.
    """
    return series.pct_change(periods=int(periods))


def lag(series: pd.Series, periods: int = 1) -> pd.Series:
    """Lag (shift) a series.

    Args:
        series: Any series.
        periods: Number of periods to shift (positive = backward).

    Returns:
        Lagged series.
    """
    return series.shift(int(periods))


def diff(series: pd.Series, periods: int = 1) -> pd.Series:
    """Difference of series.

    Args:
        series: Any series.
        periods: Number of periods for difference.

    Returns:
        Differenced series.
    """
    return series.diff(int(periods))


def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Detect when series1 crosses above series2.

    Args:
        series1: First series.
        series2: Second series.

    Returns:
        Boolean series (True on crossover).
    """
    above = series1 > series2
    shifted = above.shift(1)
    prev_above = shifted.fillna(False).infer_objects(copy=False).astype(bool)
    return above & ~prev_above


def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Detect when series1 crosses below series2.

    Args:
        series1: First series.
        series2: Second series.

    Returns:
        Boolean series (True on crossunder).
    """
    below = series1 < series2
    shifted = below.shift(1)
    prev_below = shifted.fillna(False).infer_objects(copy=False).astype(bool)
    return below & ~prev_below


def where(condition: pd.Series, if_true: Any, if_false: Any) -> pd.Series:
    """Conditional selection (like np.where).

    Args:
        condition: Boolean series.
        if_true: Value when True.
        if_false: Value when False.

    Returns:
        Series with selected values.
    """
    return pd.Series(np.where(condition, if_true, if_false), index=condition.index)


def clip(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """Clip values to range.

    Args:
        series: Any series.
        lower: Minimum value.
        upper: Maximum value.

    Returns:
        Clipped series.
    """
    return series.clip(lower=lower, upper=upper)


def rolling_max(series: pd.Series, period: int) -> pd.Series:
    """Rolling maximum.

    Args:
        series: Any series.
        period: Lookback period.

    Returns:
        Rolling max series.
    """
    return series.rolling(window=int(period)).max()


def rolling_min(series: pd.Series, period: int) -> pd.Series:
    """Rolling minimum.

    Args:
        series: Any series.
        period: Lookback period.

    Returns:
        Rolling min series.
    """
    return series.rolling(window=int(period)).min()


def sign(series: pd.Series) -> pd.Series:
    """Sign of values (-1, 0, or 1).

    Args:
        series: Any series.

    Returns:
        Sign series.
    """
    return np.sign(series)


def abs_val(series: pd.Series) -> pd.Series:
    """Absolute value.

    Args:
        series: Any series.

    Returns:
        Absolute value series.
    """
    return series.abs()


# Export all DSL functions
DSL_FUNCTIONS = {
    # Moving averages
    "sma": sma,
    "ema": ema,
    # Volatility
    "std": std,
    "atr": atr,
    "bollinger_bands": bollinger_bands,
    # Momentum
    "rsi": rsi,
    "macd": macd,
    # Transformations
    "rank": rank,
    "zscore": zscore,
    "pct_change": pct_change,
    "lag": lag,
    "diff": diff,
    # Signals
    "crossover": crossover,
    "crossunder": crossunder,
    # Utilities
    "where": where,
    "clip": clip,
    "rolling_max": rolling_max,
    "rolling_min": rolling_min,
    "sign": sign,
    "abs": abs_val,  # Override built-in abs for series
}


def get_dsl_globals() -> dict[str, Any]:
    """Get all DSL functions for sandbox execution.

    Returns:
        Dict of function name to function object.
    """
    return DSL_FUNCTIONS.copy()


def list_available_functions() -> list[str]:
    """List all available DSL functions.

    Returns:
        Sorted list of function names.
    """
    return sorted(DSL_FUNCTIONS.keys())
