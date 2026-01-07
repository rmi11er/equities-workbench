"""
Utility functions for common research workflows.

Consolidates repeated patterns for strategy testing and analysis.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from moat.data import DataManager
from moat.engines import VectorizedBacktester
from moat.schemas import BacktestConfig, BacktestResult
from moat.stress_test import RobustnessTestSuite, StressTestConfig, RobustnessReport


StrategyFn = Callable[[pd.DataFrame], pd.Series]


@dataclass
class StrategyReport:
    """Complete report for a strategy including backtest and robustness."""

    name: str
    symbol: str
    backtest: BacktestResult
    robustness: Optional[RobustnessReport] = None
    benchmark_return: Optional[float] = None
    exposure: Optional[float] = None


def fetch_universe(
    symbols: list[str],
    align: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """Fetch data for multiple symbols.

    Args:
        symbols: List of ticker symbols.
        align: If True, also return aligned close prices.
        verbose: Print progress.

    Returns:
        Tuple of (data dict, aligned closes DataFrame or None).
    """
    dm = DataManager()
    data = {}

    for sym in symbols:
        try:
            df = dm.get(sym)
            data[sym] = df
            if verbose:
                print(f"  {sym}: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
        except Exception as e:
            if verbose:
                print(f"  {sym}: FAILED - {e}")

    aligned = None
    if align and len(data) > 1:
        aligned = pd.DataFrame({sym: data[sym]["close"] for sym in data.keys()})
        aligned = aligned.dropna()
        if verbose:
            print(f"\nAligned: {len(aligned)} common trading days")

    return data, aligned


def backtest_strategy(
    data: pd.DataFrame,
    strategy: StrategyFn | pd.Series,
    name: str = "strategy",
    symbol: str = "unknown",
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """Run backtest on a strategy.

    Args:
        data: OHLCV DataFrame.
        strategy: Strategy function or pre-computed signals.
        name: Strategy name.
        symbol: Symbol being traded.
        config: Backtest configuration.

    Returns:
        BacktestResult.
    """
    bt = VectorizedBacktester(config or BacktestConfig())
    return bt.run(data, strategy, name, symbol)


def test_robustness(
    data: pd.DataFrame,
    strategy: StrategyFn,
    name: str = "strategy",
    monte_carlo_runs: int = 100,
    noise_levels: Optional[list[float]] = None,
) -> RobustnessReport:
    """Run robustness tests on a strategy.

    Args:
        data: OHLCV DataFrame.
        strategy: Strategy function.
        name: Strategy name.
        monte_carlo_runs: Number of MC simulations.
        noise_levels: Noise levels to test.

    Returns:
        RobustnessReport.
    """
    config = StressTestConfig(
        monte_carlo_runs=monte_carlo_runs,
        noise_levels=noise_levels or [0.001, 0.005, 0.01, 0.02],
    )
    suite = RobustnessTestSuite(stress_config=config)
    return suite.run_full_suite(data, strategy, name)


def full_strategy_test(
    data: pd.DataFrame,
    strategy: StrategyFn | pd.Series,
    name: str = "strategy",
    symbol: str = "unknown",
    run_robustness: bool = True,
    monte_carlo_runs: int = 100,
    config: Optional[BacktestConfig] = None,
    verbose: bool = True,
) -> StrategyReport:
    """Run complete strategy test with backtest and robustness.

    Args:
        data: OHLCV DataFrame.
        strategy: Strategy function or signals.
        name: Strategy name.
        symbol: Symbol being traded.
        run_robustness: Whether to run robustness tests.
        monte_carlo_runs: MC runs for robustness.
        config: Backtest configuration.
        verbose: Print results.

    Returns:
        StrategyReport with all results.
    """
    # Backtest
    bt_result = backtest_strategy(data, strategy, name, symbol, config)

    # Calculate exposure
    if callable(strategy):
        signals = strategy(data)
    else:
        signals = strategy
    exposure = signals.abs().mean()

    # Benchmark
    benchmark_return = (data["close"].iloc[-1] / data["close"].iloc[0]) - 1

    # Robustness (only if strategy is callable)
    rob_result = None
    if run_robustness and callable(strategy):
        if verbose:
            print("Running robustness tests...")
        rob_result = test_robustness(data, strategy, name, monte_carlo_runs)

    report = StrategyReport(
        name=name,
        symbol=symbol,
        backtest=bt_result,
        robustness=rob_result,
        benchmark_return=benchmark_return,
        exposure=exposure,
    )

    if verbose:
        print_strategy_report(report)

    return report


def print_strategy_report(report: StrategyReport) -> None:
    """Print formatted strategy report."""
    bt = report.backtest
    rob = report.robustness

    print()
    print("=" * 55)
    print(f"STRATEGY REPORT: {report.name} ({report.symbol})")
    print("=" * 55)

    print("\n--- Backtest ---")
    print(f"Total Return:     {bt.total_return:>+10.2%}")
    print(f"Annual Return:    {bt.annualized_return:>+10.2%}")
    print(f"Sharpe Ratio:     {bt.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:    {bt.sortino_ratio:>10.2f}")
    print(f"Max Drawdown:     {bt.max_drawdown:>10.2%}")
    print(f"Calmar Ratio:     {bt.calmar_ratio:>10.2f}" if bt.calmar_ratio else "Calmar Ratio:     N/A")
    print(f"Win Rate:         {bt.win_rate:>10.1%}")
    print(f"Total Trades:     {bt.total_trades:>10}")

    if report.exposure:
        print(f"Avg Exposure:     {report.exposure:>10.1%}")

    if report.benchmark_return is not None:
        print(f"\n--- vs Benchmark ---")
        print(f"Buy & Hold:       {report.benchmark_return:>+10.2%}")
        print(f"Excess Return:    {bt.total_return - report.benchmark_return:>+10.2%}")
        if report.exposure and report.exposure > 0:
            leverage_adj = bt.total_return / report.exposure
            print(f"Leverage-Adj:     {leverage_adj:>+10.2%}")

    if rob:
        print(f"\n--- Robustness ---")
        print(f"Score:            {rob.robustness_score:>10.1f} / 100")
        print(f"MC Sharpe:        {rob.monte_carlo_mean_sharpe:>7.2f} ± {rob.monte_carlo_std_sharpe:.2f}")
        print(f"Noise Sens:       {rob.noise_sensitivity:>10.2f}")
        print(f"Max Drawdown:     {bt.max_drawdown:>10.2%}")
        print(f"\nAssessment: {rob.notes}")


def compare_strategies(
    reports: list[StrategyReport],
) -> pd.DataFrame:
    """Create comparison table of multiple strategies.

    Args:
        reports: List of strategy reports.

    Returns:
        DataFrame with comparison metrics.
    """
    rows = []
    for r in reports:
        row = {
            "Strategy": r.name,
            "Symbol": r.symbol,
            "Return": r.backtest.total_return,
            "Sharpe": r.backtest.sharpe_ratio,
            "Sortino": r.backtest.sortino_ratio,
            "MaxDD": r.backtest.max_drawdown,
            "Trades": r.backtest.total_trades,
            "Exposure": r.exposure,
        }
        if r.robustness:
            row["Robustness"] = r.robustness.robustness_score
            row["MC_Sharpe"] = r.robustness.monte_carlo_mean_sharpe
        rows.append(row)

    return pd.DataFrame(rows)


def print_comparison_table(reports: list[StrategyReport]) -> None:
    """Print formatted comparison of strategies."""
    print(f"\n{'Strategy':<20} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Robust':>8}")
    print("-" * 58)
    for r in reports:
        rob_score = f"{r.robustness.robustness_score:.0f}" if r.robustness else "N/A"
        print(f"{r.name:<20} {r.backtest.total_return:>+10.2%} {r.backtest.sharpe_ratio:>8.2f} "
              f"{r.backtest.max_drawdown:>8.2%} {rob_score:>8}")
