"""
Robustness testing suite - "The Torture Chamber" for strategies.

Provides Monte Carlo simulation, noise injection, and parameter
sensitivity analysis to detect overfitting and fragile strategies.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from moat.engines.vector_engine import VectorizedBacktester
from moat.schemas import BacktestConfig, BacktestResult, RobustnessReport
from moat.stats import sharpe_ratio

logger = logging.getLogger(__name__)

StrategyFn = Callable[[pd.DataFrame], pd.Series]
ParameterizedStrategyFn = Callable[[pd.DataFrame, dict], pd.Series]


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""

    # Monte Carlo settings
    monte_carlo_runs: int = 100
    shuffle_returns: bool = True  # Shuffle daily returns
    shuffle_blocks: bool = False  # Shuffle blocks of returns (preserves autocorrelation)
    block_size: int = 20

    # Noise injection settings
    noise_levels: list[float] = field(default_factory=lambda: [0.001, 0.005, 0.01, 0.02])

    # Parameter scan settings
    param_variations: int = 5  # Number of variations per parameter


class RobustnessTestSuite:
    """Suite for testing strategy robustness.

    Tests strategies against:
    1. Monte Carlo shuffling - Does performance persist with randomized trade order?
    2. Noise injection - How sensitive is the strategy to price noise?
    3. Parameter sensitivity - Do small param changes break the strategy?
    """

    def __init__(
        self,
        backtest_config: Optional[BacktestConfig] = None,
        stress_config: Optional[StressTestConfig] = None,
    ) -> None:
        """Initialize robustness test suite.

        Args:
            backtest_config: Config for underlying backtester.
            stress_config: Config for stress tests.
        """
        self.backtest_config = backtest_config or BacktestConfig()
        self.stress_config = stress_config or StressTestConfig()
        self.backtester = VectorizedBacktester(self.backtest_config)

    def run_full_suite(
        self,
        data: pd.DataFrame,
        strategy: StrategyFn,
        strategy_name: str = "unnamed",
        param_ranges: Optional[dict[str, tuple[float, float]]] = None,
        parameterized_strategy: Optional[ParameterizedStrategyFn] = None,
    ) -> RobustnessReport:
        """Run complete robustness test suite.

        Args:
            data: OHLCV data.
            strategy: Strategy function for basic tests.
            strategy_name: Name for reporting.
            param_ranges: Dict of param_name -> (min, max) for param scan.
            parameterized_strategy: Strategy function that accepts params dict.

        Returns:
            RobustnessReport with composite score.
        """
        # Run baseline backtest
        baseline = self.backtester.run(data, strategy, strategy_name)
        baseline_sharpe = baseline.sharpe_ratio or 0.0

        # Monte Carlo analysis
        mc_result = self.monte_carlo_test(data, strategy)

        # Noise sensitivity
        noise_result = self.noise_injection_test(data, strategy)

        # Parameter sensitivity (if parameterized strategy provided)
        param_stability = None
        if parameterized_strategy and param_ranges:
            param_result = self.parameter_scan(
                data, parameterized_strategy, param_ranges
            )
            param_stability = param_result["stability_score"]

        # Calculate composite robustness score (0-100)
        score = self._calculate_robustness_score(
            baseline_sharpe=baseline_sharpe,
            mc_mean_sharpe=mc_result["mean_sharpe"],
            mc_std_sharpe=mc_result["std_sharpe"],
            noise_sensitivity=noise_result["sensitivity"],
            param_stability=param_stability,
        )

        return RobustnessReport(
            strategy_name=strategy_name,
            robustness_score=score,
            monte_carlo_mean_sharpe=mc_result["mean_sharpe"],
            monte_carlo_std_sharpe=mc_result["std_sharpe"],
            noise_sensitivity=noise_result["sensitivity"],
            parameter_stability=param_stability,
            notes=self._generate_notes(score, mc_result, noise_result, param_stability),
        )

    def monte_carlo_test(
        self,
        data: pd.DataFrame,
        strategy: StrategyFn,
    ) -> dict:
        """Run Monte Carlo simulation by shuffling returns.

        Tests whether strategy performance is robust to different
        orderings of returns (detects sequence-dependent strategies).

        Returns:
            Dict with mean_sharpe, std_sharpe, percentiles.
        """
        config = self.stress_config
        sharpe_ratios = []

        # Get original signals (computed once)
        original_signals = strategy(data)

        for i in range(config.monte_carlo_runs):
            # Shuffle returns while keeping signals aligned
            shuffled_data = self._shuffle_data(data, config)

            # Re-run strategy on shuffled data
            shuffled_signals = strategy(shuffled_data)

            # Calculate returns
            returns = shuffled_signals.shift(1) * shuffled_data["close"].pct_change()
            returns = returns.dropna()

            if len(returns) > 0:
                sr = sharpe_ratio(returns)
                sharpe_ratios.append(sr)

        sharpe_array = np.array(sharpe_ratios)

        return {
            "mean_sharpe": float(np.mean(sharpe_array)) if len(sharpe_array) > 0 else 0.0,
            "std_sharpe": float(np.std(sharpe_array)) if len(sharpe_array) > 0 else 0.0,
            "percentile_5": float(np.percentile(sharpe_array, 5)) if len(sharpe_array) > 0 else 0.0,
            "percentile_95": float(np.percentile(sharpe_array, 95)) if len(sharpe_array) > 0 else 0.0,
            "n_runs": len(sharpe_ratios),
        }

    def _shuffle_data(self, data: pd.DataFrame, config: StressTestConfig) -> pd.DataFrame:
        """Shuffle OHLCV data returns."""
        df = data.copy()

        # Calculate returns
        returns = df["close"].pct_change()

        if config.shuffle_blocks:
            # Block shuffle (preserves some autocorrelation)
            returns_arr = returns.dropna().values
            n_blocks = len(returns_arr) // config.block_size
            if n_blocks > 1:
                blocks = np.array_split(returns_arr[: n_blocks * config.block_size], n_blocks)
                np.random.shuffle(blocks)
                shuffled_returns = np.concatenate(blocks)
                # Pad back to original length
                shuffled_returns = np.concatenate(
                    [shuffled_returns, returns_arr[n_blocks * config.block_size :]]
                )
            else:
                shuffled_returns = returns_arr
        else:
            # Simple shuffle
            shuffled_returns = returns.dropna().values.copy()
            np.random.shuffle(shuffled_returns)

        # Reconstruct prices from shuffled returns
        new_close = df["close"].iloc[0] * (1 + pd.Series(shuffled_returns)).cumprod()
        new_close = pd.concat([pd.Series([df["close"].iloc[0]]), new_close]).reset_index(drop=True)

        # Scale other OHLC columns proportionally
        scale = new_close.values / df["close"].values
        df["close"] = new_close.values
        df["open"] = df["open"] * scale
        df["high"] = df["high"] * scale
        df["low"] = df["low"] * scale

        return df

    def noise_injection_test(
        self,
        data: pd.DataFrame,
        strategy: StrategyFn,
    ) -> dict:
        """Test strategy sensitivity to price noise.

        Adds varying levels of random noise to prices and measures
        degradation in strategy performance.

        Returns:
            Dict with sensitivity score and per-level results.
        """
        # Baseline performance
        baseline_result = self.backtester.run(data, strategy)
        baseline_sharpe = baseline_result.sharpe_ratio or 0.0

        results = []
        for noise_level in self.stress_config.noise_levels:
            noisy_data = self._add_noise(data, noise_level)
            noisy_result = self.backtester.run(noisy_data, strategy)
            noisy_sharpe = noisy_result.sharpe_ratio or 0.0

            # Calculate degradation
            if baseline_sharpe != 0:
                degradation = (baseline_sharpe - noisy_sharpe) / abs(baseline_sharpe)
            else:
                degradation = 0.0 if noisy_sharpe == 0 else -1.0

            results.append({
                "noise_level": noise_level,
                "sharpe": noisy_sharpe,
                "degradation": degradation,
            })

        # Overall sensitivity: average degradation per unit noise
        if len(results) > 0 and self.stress_config.noise_levels[-1] > 0:
            avg_degradation = np.mean([r["degradation"] for r in results])
            sensitivity = avg_degradation / self.stress_config.noise_levels[-1]
        else:
            sensitivity = 0.0

        return {
            "sensitivity": float(sensitivity),
            "baseline_sharpe": baseline_sharpe,
            "per_level": results,
        }

    def _add_noise(self, data: pd.DataFrame, noise_level: float) -> pd.DataFrame:
        """Add Gaussian noise to price data."""
        df = data.copy()
        n = len(df)

        for col in ["open", "high", "low", "close"]:
            noise = np.random.normal(0, noise_level, n)
            df[col] = df[col] * (1 + noise)

        # Ensure OHLC validity
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

        return df

    def parameter_scan(
        self,
        data: pd.DataFrame,
        strategy: ParameterizedStrategyFn,
        param_ranges: dict[str, tuple[float, float]],
    ) -> dict:
        """Scan parameter space for strategy stability.

        Tests strategy with varied parameters to detect overfitting
        to specific parameter values.

        Args:
            data: OHLCV data.
            strategy: Function(data, params) -> signals.
            param_ranges: Dict of param_name -> (min, max).

        Returns:
            Dict with stability score and parameter heatmap.
        """
        n_variations = self.stress_config.param_variations
        results = []

        # Generate parameter grid
        param_grid = {}
        for param_name, (p_min, p_max) in param_ranges.items():
            param_grid[param_name] = np.linspace(p_min, p_max, n_variations)

        # Run backtest for each parameter combination
        # For simplicity, vary one parameter at a time (not full grid)
        for param_name, values in param_grid.items():
            for value in values:
                params = {
                    k: (v[len(v) // 2] if k != param_name else value)
                    for k, v in param_grid.items()
                }

                signals = strategy(data, params)
                result = self.backtester.run(data, signals)
                results.append({
                    "params": params.copy(),
                    "sharpe": result.sharpe_ratio or 0.0,
                    "return": result.total_return,
                })

        # Calculate stability score
        sharpe_values = [r["sharpe"] for r in results]
        if len(sharpe_values) > 1 and np.mean(sharpe_values) != 0:
            # Stability = 1 - (coefficient of variation)
            cv = np.std(sharpe_values) / abs(np.mean(sharpe_values))
            stability = max(0, 1 - cv)
        else:
            stability = 0.0

        return {
            "stability_score": float(stability),
            "mean_sharpe": float(np.mean(sharpe_values)),
            "std_sharpe": float(np.std(sharpe_values)),
            "results": results,
        }

    def _calculate_robustness_score(
        self,
        baseline_sharpe: float,
        mc_mean_sharpe: float,
        mc_std_sharpe: float,
        noise_sensitivity: float,
        param_stability: Optional[float],
    ) -> float:
        """Calculate composite robustness score (0-100).

        Components:
        - Baseline Sharpe contribution (0-30 points)
        - Monte Carlo consistency (0-30 points)
        - Noise resilience (0-20 points)
        - Parameter stability (0-20 points)
        """
        score = 0.0

        # 1. Baseline Sharpe (0-30 points)
        # Sharpe of 2+ gets full points
        sharpe_score = min(30, max(0, baseline_sharpe * 15))
        score += sharpe_score

        # 2. Monte Carlo consistency (0-30 points)
        # High mean and low std is good
        if mc_std_sharpe > 0:
            mc_consistency = mc_mean_sharpe / mc_std_sharpe  # Information ratio-like
        else:
            mc_consistency = mc_mean_sharpe
        mc_score = min(30, max(0, mc_consistency * 10))
        score += mc_score

        # 3. Noise resilience (0-20 points)
        # Lower sensitivity is better
        noise_score = max(0, 20 * (1 - abs(noise_sensitivity)))
        score += noise_score

        # 4. Parameter stability (0-20 points)
        if param_stability is not None:
            param_score = param_stability * 20
        else:
            # If no param test, distribute points to other categories
            param_score = 10  # Neutral
        score += param_score

        return float(min(100, max(0, score)))

    def _generate_notes(
        self,
        score: float,
        mc_result: dict,
        noise_result: dict,
        param_stability: Optional[float],
    ) -> str:
        """Generate human-readable notes about robustness."""
        notes = []

        if score >= 80:
            notes.append("Strategy shows strong robustness characteristics.")
        elif score >= 60:
            notes.append("Strategy is moderately robust with some concerns.")
        elif score >= 40:
            notes.append("Strategy shows signs of fragility.")
        else:
            notes.append("Strategy may be overfit or unreliable.")

        # Monte Carlo insights
        if mc_result["std_sharpe"] > abs(mc_result["mean_sharpe"]):
            notes.append(
                "High variance in Monte Carlo suggests sequence-dependent performance."
            )

        # Noise insights
        if abs(noise_result["sensitivity"]) > 10:
            notes.append(
                "Strategy is highly sensitive to price noise - may fail with real data."
            )

        # Parameter insights
        if param_stability is not None and param_stability < 0.5:
            notes.append(
                "Low parameter stability suggests overfitting to specific values."
            )

        return " ".join(notes)


def quick_robustness_check(
    data: pd.DataFrame,
    strategy: StrategyFn,
    n_monte_carlo: int = 50,
) -> float:
    """Quick robustness score without full suite.

    Useful for rapid filtering of strategy candidates.

    Returns:
        Robustness score (0-100).
    """
    suite = RobustnessTestSuite(
        stress_config=StressTestConfig(
            monte_carlo_runs=n_monte_carlo,
            noise_levels=[0.005, 0.01],
        )
    )

    report = suite.run_full_suite(data, strategy, "quick_check")
    return report.robustness_score
