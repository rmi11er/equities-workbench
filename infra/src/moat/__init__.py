"""
The Moat - Quantitative Research & Execution Infrastructure.

A high-performance, modular, vendor-agnostic infrastructure for
quantitative finance strategy validation.
"""

__version__ = "0.1.0"

from moat.utils import (
    fetch_universe,
    backtest_strategy,
    test_robustness,
    full_strategy_test,
    print_strategy_report,
    compare_strategies,
    print_comparison_table,
)

__all__ = [
    "fetch_universe",
    "backtest_strategy",
    "test_robustness",
    "full_strategy_test",
    "print_strategy_report",
    "compare_strategies",
    "print_comparison_table",
]


def main() -> None:
    """CLI entry point."""
    print(f"The Moat v{__version__}")
    print("Quantitative Research Infrastructure")
