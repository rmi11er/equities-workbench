"""
Pydantic models for data validation and contracts.

This module defines the canonical data structures used throughout The Moat.
All data flowing through the system must conform to these schemas.
"""

from datetime import datetime
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class OHLCVBar(BaseModel):
    """Single OHLCV bar representing one time period."""

    model_config = ConfigDict(frozen=True)

    date: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    adjusted_close: Optional[float] = Field(default=None, gt=0)

    @field_validator("high")
    @classmethod
    def high_gte_low(cls, v: float, info) -> float:
        """Ensure high >= low."""
        if "low" in info.data and v < info.data["low"]:
            raise ValueError("high must be >= low")
        return v

    @field_validator("high")
    @classmethod
    def high_gte_open_close(cls, v: float, info) -> float:
        """Ensure high >= open and high >= close."""
        if "open" in info.data and v < info.data["open"]:
            raise ValueError("high must be >= open")
        if "close" in info.data and v < info.data["close"]:
            raise ValueError("high must be >= close")
        return v

    @field_validator("low")
    @classmethod
    def low_lte_open_close(cls, v: float, info) -> float:
        """Ensure low <= open and low <= close."""
        if "open" in info.data and v > info.data["open"]:
            raise ValueError("low must be <= open")
        if "close" in info.data and v > info.data["close"]:
            raise ValueError("low must be <= close")
        return v


class OHLCVSeries(BaseModel):
    """Time series of OHLCV data with metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbol: str = Field(min_length=1)
    source: str = Field(min_length=1)  # e.g., "yfinance", "local:futures_data"
    bars: List[OHLCVBar] = Field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with standard column names."""
        if not self.bars:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume", "adjusted_close"]
            )

        data = [bar.model_dump() for bar in self.bars]
        df = pd.DataFrame(data)
        df = df.set_index("date").sort_index()
        return df

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        symbol: str,
        source: str,
    ) -> "OHLCVSeries":
        """Create OHLCVSeries from a DataFrame."""
        df = df.copy()

        # Handle index
        if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if df.columns[0] != "date":
                df = df.rename(columns={df.columns[0]: "date"})

        # Ensure datetime
        df["date"] = pd.to_datetime(df["date"])

        # Handle adjusted_close
        if "adjusted_close" not in df.columns:
            df["adjusted_close"] = None

        bars = []
        for _, row in df.iterrows():
            bar = OHLCVBar(
                date=row["date"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                adjusted_close=float(row["adjusted_close"])
                if pd.notna(row["adjusted_close"])
                else None,
            )
            bars.append(bar)

        return cls(
            symbol=symbol,
            source=source,
            bars=bars,
            start_date=bars[0].date if bars else None,
            end_date=bars[-1].date if bars else None,
        )


class StrategySignal(BaseModel):
    """Output signal from a strategy for a single bar."""

    model_config = ConfigDict(frozen=True)

    date: datetime
    position: float = Field(ge=-1.0, le=1.0)  # -1 = full short, 0 = flat, 1 = full long
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Optional[dict] = None


class BacktestConfig(BaseModel):
    """Configuration for a backtest run."""

    initial_capital: float = Field(default=100000.0, gt=0)
    commission_pct: float = Field(default=0.001, ge=0)  # 0.1%
    slippage_pct: float = Field(default=0.0005, ge=0)  # 0.05%
    max_leverage: float = Field(default=1.0, ge=0)
    allow_shorting: bool = True


class BacktestResult(BaseModel):
    """Results from a backtest run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy_name: str
    symbol: str
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: float
    calmar_ratio: Optional[float] = None
    total_trades: int
    win_rate: Optional[float] = None
    equity_curve: Optional[List[float]] = None


class RobustnessReport(BaseModel):
    """Output from the robustness testing suite."""

    strategy_name: str
    robustness_score: float = Field(ge=0, le=100)
    monte_carlo_mean_sharpe: Optional[float] = None
    monte_carlo_std_sharpe: Optional[float] = None
    noise_sensitivity: Optional[float] = None  # % degradation with noise
    parameter_stability: Optional[float] = None  # Consistency across param changes
    notes: Optional[str] = None
