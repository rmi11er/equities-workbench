"""Tests for the data abstraction layer."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from moat.data import DataManager, YFinanceAdapter
from moat.schemas import OHLCVSeries


class TestYFinanceAdapter:
    """Tests for YFinance data fetching."""

    def test_fetch_aapl(self) -> None:
        """Test fetching AAPL data from YFinance."""
        adapter = YFinanceAdapter()
        df = adapter.get("AAPL", start=datetime(2024, 1, 1), end=datetime(2024, 12, 31))

        # Check we got data
        assert not df.empty
        assert len(df) > 200  # ~252 trading days

        # Check columns
        assert set(df.columns) >= {"open", "high", "low", "close", "volume"}

        # Check data quality
        assert df["close"].notna().all()
        assert (df["high"] >= df["low"]).all()
        assert (df["volume"] >= 0).all()

    def test_source_name(self) -> None:
        """Test source identifier."""
        adapter = YFinanceAdapter()
        assert adapter.source_name == "yfinance"


class TestDataManager:
    """Tests for the unified DataManager."""

    def test_get_from_yfinance(self, tmp_path: Path) -> None:
        """Test fetching data through DataManager."""
        # Create minimal config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text("""
paths:
  data_incoming: data/incoming
  data_processed: data/processed
data:
  default_source: yfinance
  cache_days: 1
""")
        (config_dir / "schema_map.yaml").write_text("""
generic:
  date: date
  open: open
  high: high
  low: low
  close: close
  volume: volume
""")

        # Create data dirs
        (tmp_path / "data" / "incoming").mkdir(parents=True)
        (tmp_path / "data" / "processed").mkdir(parents=True)

        dm = DataManager(config_path=config_dir / "config.yaml", base_dir=tmp_path)
        df = dm.get("AAPL", start=datetime(2024, 6, 1), end=datetime(2024, 6, 30))

        assert not df.empty
        assert "close" in df.columns


class TestOHLCVSeries:
    """Tests for OHLCV schema conversion."""

    def test_dataframe_roundtrip(self) -> None:
        """Test converting DataFrame to OHLCVSeries and back."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5),
            "open": [100.0, 101.0, 102.0, 101.5, 103.0],
            "high": [101.0, 102.0, 103.0, 102.5, 104.0],
            "low": [99.0, 100.0, 101.0, 100.5, 102.0],
            "close": [100.5, 101.5, 102.5, 102.0, 103.5],
            "volume": [1000, 1100, 1200, 1150, 1300],
        })

        series = OHLCVSeries.from_dataframe(df, symbol="TEST", source="test")

        assert series.symbol == "TEST"
        assert len(series.bars) == 5
        assert series.bars[0].close == 100.5

        # Convert back
        df_out = series.to_dataframe()
        assert len(df_out) == 5
        assert df_out["close"].iloc[0] == 100.5
