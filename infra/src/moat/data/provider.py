"""
Abstract base class for data providers.

All data sources must implement this interface to ensure
consistent behavior across the system.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd


class DataProvider(ABC):
    """Abstract base class for all data providers.

    Implementers must return standardized pd.DataFrame with columns:
    - date (index): datetime
    - open: float
    - high: float
    - low: float
    - close: float
    - volume: float
    - adjusted_close: float (optional, may be NaN)
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return identifier for this data source."""
        ...

    @abstractmethod
    def get(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol.

        Args:
            symbol: Ticker symbol or dataset name.
            start: Start date (inclusive). None means earliest available.
            end: End date (inclusive). None means latest available.

        Returns:
            DataFrame with standardized OHLCV columns, indexed by date.

        Raises:
            DataNotFoundError: If symbol/dataset not found.
            DataFetchError: If data retrieval fails.
        """
        ...

    @abstractmethod
    def list_available(self) -> list[str]:
        """List available symbols/datasets from this provider.

        Returns:
            List of available symbol names.
        """
        ...

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize a DataFrame to standard format.

        Args:
            df: Raw DataFrame to validate.

        Returns:
            Validated DataFrame with standard column names.

        Raises:
            ValueError: If required columns missing or data invalid.
        """
        required_cols = {"open", "high", "low", "close", "volume"}

        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()

        # Check for required columns
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Add adjusted_close if missing
        if "adjusted_close" not in df.columns:
            if "adj close" in df.columns:
                df["adjusted_close"] = df["adj close"]
                df = df.drop(columns=["adj close"])
            else:
                df["adjusted_close"] = df["close"]

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            else:
                raise ValueError("No date column or DatetimeIndex found")

        # Sort by date
        df = df.sort_index()

        # Select only standard columns
        standard_cols = ["open", "high", "low", "close", "volume", "adjusted_close"]
        df = df[standard_cols]

        return df


class DataNotFoundError(Exception):
    """Raised when requested data is not available."""

    pass


class DataFetchError(Exception):
    """Raised when data fetching fails."""

    pass
