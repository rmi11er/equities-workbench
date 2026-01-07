"""
YFinance data adapter for fetching market data from Yahoo Finance.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from moat.data.provider import DataFetchError, DataNotFoundError, DataProvider

logger = logging.getLogger(__name__)


class YFinanceAdapter(DataProvider):
    """Data provider using Yahoo Finance API.

    Fetches OHLCV data for any ticker supported by Yahoo Finance.
    Includes optional local caching to reduce API calls.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_days: int = 1,
    ) -> None:
        """Initialize YFinance adapter.

        Args:
            cache_dir: Directory for caching data. None disables caching.
            cache_days: Number of days to keep cached data fresh.
        """
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._cache_days = cache_days

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def source_name(self) -> str:
        """Return identifier for this data source."""
        return "yfinance"

    def get(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a ticker symbol.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'SPY').
            start: Start date. Defaults to 5 years ago.
            end: End date. Defaults to today.

        Returns:
            DataFrame with standardized OHLCV columns.

        Raises:
            DataNotFoundError: If ticker not found.
            DataFetchError: If API call fails.
        """
        symbol = symbol.upper()

        # Check cache first
        cached = self._load_from_cache(symbol, start, end)
        if cached is not None:
            logger.info(f"Loaded {symbol} from cache")
            return cached

        # Fetch from API
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start,
                end=end,
                period="5y" if start is None else None,
            )
        except Exception as e:
            raise DataFetchError(f"Failed to fetch {symbol}: {e}") from e

        if df.empty:
            raise DataNotFoundError(f"No data found for symbol: {symbol}")

        # Normalize column names
        df.columns = df.columns.str.lower()

        # Rename 'adj close' if present
        if "adj close" in df.columns:
            df = df.rename(columns={"adj close": "adjusted_close"})

        # Keep only OHLCV columns
        cols_to_keep = ["open", "high", "low", "close", "volume"]
        if "adjusted_close" in df.columns:
            cols_to_keep.append("adjusted_close")

        df = df[[c for c in cols_to_keep if c in df.columns]]

        # Add adjusted_close if not present
        if "adjusted_close" not in df.columns:
            df["adjusted_close"] = df["close"]

        # Ensure proper types
        for col in ["open", "high", "low", "close", "volume", "adjusted_close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Save to cache
        self._save_to_cache(symbol, df)

        logger.info(f"Fetched {len(df)} bars for {symbol} from YFinance")
        return df

    def list_available(self) -> list[str]:
        """List available symbols.

        For YFinance, returns cached symbols since there's no
        practical way to list all available tickers.
        """
        if not self._cache_dir:
            return []

        return [
            f.stem.upper()
            for f in self._cache_dir.glob("*.parquet")
        ]

    def _cache_path(self, symbol: str) -> Optional[Path]:
        """Get cache file path for a symbol."""
        if not self._cache_dir:
            return None
        return self._cache_dir / f"{symbol.lower()}.parquet"

    def _load_from_cache(
        self,
        symbol: str,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if fresh enough."""
        cache_path = self._cache_path(symbol)
        if not cache_path or not cache_path.exists():
            return None

        # Check freshness
        import time
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days > self._cache_days:
            return None

        try:
            df = pd.read_parquet(cache_path)

            # Filter by date range
            if start:
                df = df[df.index >= pd.Timestamp(start)]
            if end:
                df = df[df.index <= pd.Timestamp(end)]

            return df
        except Exception as e:
            logger.warning(f"Failed to load cache for {symbol}: {e}")
            return None

    def _save_to_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """Save data to cache."""
        cache_path = self._cache_path(symbol)
        if not cache_path:
            return

        try:
            df.to_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Failed to cache {symbol}: {e}")
