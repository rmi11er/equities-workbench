"""
Performance statistics and metrics for backtesting.

Provides vectorized implementations of common financial metrics.
"""

from typing import Optional

import numpy as np
import pandas as pd


def calculate_returns(
    prices: pd.Series,
    method: str = "simple",
) -> pd.Series:
    """Calculate returns from a price series.

    Args:
        prices: Series of prices indexed by date.
        method: 'simple' for arithmetic returns, 'log' for log returns.

    Returns:
        Series of returns.
    """
    if method == "log":
        return np.log(prices / prices.shift(1))
    return prices.pct_change()


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe ratio.

    Args:
        returns: Series of periodic returns.
        risk_free_rate: Annual risk-free rate (default 0).
        periods_per_year: Trading periods per year (252 for daily).

    Returns:
        Annualized Sharpe ratio.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_return = excess_returns.mean()
    std_return = excess_returns.std(ddof=1)

    if std_return == 0:
        return 0.0

    return float(mean_return / std_return * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sortino ratio.

    Uses downside deviation (volatility of negative returns only).

    Args:
        returns: Series of periodic returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Trading periods per year.

    Returns:
        Annualized Sortino ratio.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_return = excess_returns.mean()

    # Downside deviation: std of returns below target (0)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return float("inf") if mean_return > 0 else 0.0

    downside_std = downside_returns.std(ddof=1)
    if downside_std == 0:
        return float("inf") if mean_return > 0 else 0.0

    return float(mean_return / downside_std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Series of portfolio values or cumulative returns.

    Returns:
        Maximum drawdown as a positive decimal (e.g., 0.20 for 20%).
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return float(-drawdown.min())


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Calculate drawdown at each point in the equity curve.

    Args:
        equity_curve: Series of portfolio values.

    Returns:
        Series of drawdowns (negative values).
    """
    peak = equity_curve.expanding().max()
    return (equity_curve - peak) / peak


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> Optional[float]:
    """Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Series of periodic returns.
        periods_per_year: Trading periods per year.

    Returns:
        Calmar ratio, or None if drawdown is 0.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return None

    # Build equity curve
    equity = (1 + returns).cumprod()

    mdd = max_drawdown(equity)
    if mdd == 0:
        return None

    annual_return = (equity.iloc[-1] ** (periods_per_year / len(returns))) - 1
    return float(annual_return / mdd)


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized return from periodic returns.

    Args:
        returns: Series of periodic returns.
        periods_per_year: Trading periods per year.

    Returns:
        Annualized return as decimal.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0

    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    return float((1 + total_return) ** (periods_per_year / n_periods) - 1)


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized volatility.

    Args:
        returns: Series of periodic returns.
        periods_per_year: Trading periods per year.

    Returns:
        Annualized volatility as decimal.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def win_rate(returns: pd.Series) -> float:
    """Calculate percentage of positive returns.

    Args:
        returns: Series of trade or period returns.

    Returns:
        Win rate as decimal (e.g., 0.55 for 55%).
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0

    return float((returns > 0).sum() / len(returns))


def profit_factor(returns: pd.Series) -> Optional[float]:
    """Calculate profit factor (gross profits / gross losses).

    Args:
        returns: Series of returns.

    Returns:
        Profit factor, or None if no losses.
    """
    returns = returns.dropna()
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()

    if losses == 0:
        return None if gains == 0 else float("inf")

    return float(gains / losses)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate Information Ratio vs benchmark.

    Args:
        returns: Strategy returns.
        benchmark_returns: Benchmark returns (same index).
        periods_per_year: Trading periods per year.

    Returns:
        Annualized Information Ratio.
    """
    # Align series
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0

    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = excess.std(ddof=1)

    if tracking_error == 0:
        return 0.0

    return float(excess.mean() / tracking_error * np.sqrt(periods_per_year))


def calculate_all_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    benchmark_returns: Optional[pd.Series] = None,
    periods_per_year: int = 252,
) -> dict:
    """Calculate comprehensive performance metrics.

    Args:
        returns: Series of periodic returns.
        risk_free_rate: Annual risk-free rate.
        benchmark_returns: Optional benchmark for comparison.
        periods_per_year: Trading periods per year.

    Returns:
        Dict of metric names to values.
    """
    returns = returns.dropna()
    equity = (1 + returns).cumprod()

    metrics = {
        "total_return": float((1 + returns).prod() - 1),
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "total_periods": len(returns),
    }

    if benchmark_returns is not None:
        metrics["information_ratio"] = information_ratio(
            returns, benchmark_returns, periods_per_year
        )

    return metrics
