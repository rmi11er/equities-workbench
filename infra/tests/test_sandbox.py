"""Tests for the agent sandbox and DSL wrapper."""

import numpy as np
import pandas as pd
import pytest

from moat.sandbox import (
    DSL_FUNCTIONS,
    Sandbox,
    SandboxError,
    StrategyRunner,
    list_available_functions,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Generate sample OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    close = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
    return pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(100) * 0.005),
            "high": close * (1 + np.abs(np.random.randn(100) * 0.01)),
            "low": close * (1 - np.abs(np.random.randn(100) * 0.01)),
            "close": close,
            "volume": np.random.randint(1000000, 10000000, 100),
        },
        index=dates,
    )


class TestSandboxSecurity:
    """Tests for sandbox security features."""

    def test_blocks_imports(self) -> None:
        """Test that imports are blocked."""
        sandbox = Sandbox()

        # import statement
        is_valid, errors = sandbox.validate("import os")
        assert not is_valid
        assert any("import" in e.lower() for e in errors)

        # from...import
        is_valid, errors = sandbox.validate("from sys import exit")
        assert not is_valid

    def test_blocks_exec_eval(self) -> None:
        """Test that exec/eval are blocked."""
        sandbox = Sandbox()

        is_valid, errors = sandbox.validate("exec('print(1)')")
        assert not is_valid

        is_valid, errors = sandbox.validate("eval('1+1')")
        assert not is_valid

    def test_blocks_open(self) -> None:
        """Test that file operations are blocked."""
        sandbox = Sandbox()

        is_valid, errors = sandbox.validate("open('/etc/passwd')")
        assert not is_valid

    def test_blocks_dunder_attributes(self) -> None:
        """Test that dangerous dunder attributes are blocked."""
        sandbox = Sandbox()

        codes = [
            "x.__class__",
            "x.__globals__",
            "x.__code__",
            "x.__builtins__",
        ]

        for code in codes:
            is_valid, errors = sandbox.validate(code)
            assert not is_valid, f"Should block: {code}"

    def test_allows_safe_code(self) -> None:
        """Test that safe code passes validation."""
        sandbox = Sandbox()

        safe_codes = [
            "x = 1 + 1",
            "result = [i * 2 for i in range(10)]",
            "result = sum([1, 2, 3])",
            "result = max(1, 2, 3)",
        ]

        for code in safe_codes:
            is_valid, errors = sandbox.validate(code)
            assert is_valid, f"Should allow: {code}, errors: {errors}"


class TestSandboxExecution:
    """Tests for sandbox code execution."""

    def test_basic_execution(self) -> None:
        """Test basic code execution in sandbox."""
        sandbox = Sandbox()

        code = "result = 1 + 2"
        result = sandbox.execute(code)
        assert result == 3

    def test_uses_local_vars(self) -> None:
        """Test that local variables are accessible."""
        sandbox = Sandbox()

        code = "result = x * 2"
        result = sandbox.execute(code, {"x": 5})
        assert result == 10

    def test_blocks_unsafe_execution(self) -> None:
        """Test that unsafe code raises error."""
        sandbox = Sandbox()

        with pytest.raises(SandboxError):
            sandbox.execute("import os")

    def test_allowed_globals(self) -> None:
        """Test that allowed globals are accessible."""
        def custom_func(x):
            return x * 10

        sandbox = Sandbox(allowed_globals={"my_func": custom_func})

        code = "result = my_func(5)"
        result = sandbox.execute(code)
        assert result == 50


class TestStrategyRunner:
    """Tests for strategy runner."""

    def test_simple_strategy(self, sample_data: pd.DataFrame) -> None:
        """Test running a simple strategy."""
        runner = StrategyRunner()

        code = """
result = (sma(close, 20) > sma(close, 50)).astype(float)
"""
        signals = runner.run(code, sample_data)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        assert signals.max() <= 1
        assert signals.min() >= -1

    def test_ma_crossover_strategy(self, sample_data: pd.DataFrame) -> None:
        """Test MA crossover strategy - spec example."""
        runner = StrategyRunner()

        # Spec requirement: "return sma(close, 50) > close" executes
        code = """
result = (sma(close, 50) > close).astype(float)
"""
        signals = runner.run(code, sample_data)

        assert isinstance(signals, pd.Series)
        # Should have both 0 and 1 values
        assert set(signals.dropna().unique()).issubset({0.0, 1.0})

    def test_blocks_import_sys(self, sample_data: pd.DataFrame) -> None:
        """Test spec requirement: 'import sys' fails."""
        runner = StrategyRunner()

        with pytest.raises(SandboxError):
            runner.run("import sys", sample_data)

    def test_complex_strategy(self, sample_data: pd.DataFrame) -> None:
        """Test more complex strategy using multiple DSL functions."""
        runner = StrategyRunner()

        code = """
# Multi-factor strategy
ma_signal = sma(close, 20) > sma(close, 50)
rsi_signal = rsi(close, 14) < 30

# Combine signals
result = where(ma_signal & rsi_signal, 1.0, 0.0)
"""
        signals = runner.run(code, sample_data)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)

    def test_validation_without_execution(self, sample_data: pd.DataFrame) -> None:
        """Test validating code without running it."""
        runner = StrategyRunner()

        is_valid, errors = runner.validate("result = sma(close, 20)")
        assert is_valid

        is_valid, errors = runner.validate("import os")
        assert not is_valid


class TestDSLFunctions:
    """Tests for DSL function library."""

    def test_all_functions_available(self) -> None:
        """Test that all expected functions are in DSL."""
        expected = [
            "sma",
            "ema",
            "std",
            "rsi",
            "macd",
            "bollinger_bands",
            "rank",
            "zscore",
            "crossover",
            "crossunder",
        ]

        available = list_available_functions()
        for func in expected:
            assert func in available, f"Missing function: {func}"

    def test_sma_calculation(self) -> None:
        """Test SMA produces correct values."""
        prices = pd.Series([10, 20, 30, 40, 50])
        ma = DSL_FUNCTIONS["sma"](prices, 3)

        assert pd.isna(ma.iloc[0])
        assert pd.isna(ma.iloc[1])
        assert ma.iloc[2] == 20.0  # (10+20+30)/3
        assert ma.iloc[3] == 30.0  # (20+30+40)/3

    def test_rsi_bounds(self, sample_data: pd.DataFrame) -> None:
        """Test RSI is bounded 0-100."""
        rsi_values = DSL_FUNCTIONS["rsi"](sample_data["close"], 14)

        valid_rsi = rsi_values.dropna()
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100

    def test_crossover_detection(self) -> None:
        """Test crossover detection."""
        fast = pd.Series([1, 2, 3, 4, 5])
        slow = pd.Series([2, 2, 2, 3, 4])

        crosses = DSL_FUNCTIONS["crossover"](fast, slow)

        # Fast crosses above slow at index 2
        assert crosses.iloc[2] == True
        assert crosses.iloc[0] == False
        assert crosses.iloc[4] == False  # Already above, not a new cross


class TestIntegration:
    """Integration tests combining sandbox with backtest engine."""

    def test_sandbox_strategy_backtest(self, sample_data: pd.DataFrame) -> None:
        """Test running a sandboxed strategy through the backtester."""
        from moat.engines.vector_engine import VectorizedBacktester

        runner = StrategyRunner()
        backtester = VectorizedBacktester()

        # Define strategy in sandbox
        code = """
fast = sma(close, 10)
slow = sma(close, 30)
result = where(fast > slow, 1.0, 0.0)
"""
        signals = runner.run(code, sample_data)

        # Run backtest with sandbox signals
        result = backtester.run(
            sample_data,
            signals,
            strategy_name="sandboxed_ma",
            symbol="TEST",
        )

        assert result.strategy_name == "sandboxed_ma"
        assert result.total_trades >= 0
        assert result.sharpe_ratio is not None
