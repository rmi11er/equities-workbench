"""Yahoo Finance data source using yfinance library."""

import asyncio
from datetime import date, datetime, timedelta
from typing import Any

import yfinance as yf

from src.data.sources.base import DataSource


class YFinanceSource(DataSource):
    """Data source using Yahoo Finance via yfinance library.

    Note: yfinance is not async-native, so we run it in a thread pool.
    """

    async def fetch_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeframe: str = "daily",
    ) -> list[dict[str, Any]]:
        """Fetch price data from Yahoo Finance.

        Args:
            symbol: Stock symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            timeframe: 'daily' or 'hourly'

        Returns:
            List of price records
        """

        def _fetch():
            ticker = yf.Ticker(symbol)

            # yfinance uses different intervals
            interval = "1d" if timeframe == "daily" else "1h"

            # Add a day to end_date to make it inclusive
            end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())
            start_dt = datetime.combine(start_date, datetime.min.time())

            # For hourly data, yfinance has limitations on how far back we can go
            if timeframe == "hourly":
                # Max 730 days for hourly data
                max_start = datetime.now() - timedelta(days=729)
                if start_dt < max_start:
                    start_dt = max_start

            df = ticker.history(start=start_dt, end=end_dt, interval=interval)

            if df.empty:
                return []

            records = []
            for idx, row in df.iterrows():
                # idx is a DatetimeIndex
                timestamp = idx.to_pydatetime()
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)

                records.append(
                    {
                        "timestamp": timestamp,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"]),
                    }
                )

            return records

        self.logger.debug(f"Fetching {timeframe} prices for {symbol} from {start_date} to {end_date}")
        return await asyncio.to_thread(_fetch)

    async def fetch_company_info(self, symbol: str) -> dict[str, Any] | None:
        """Fetch company information from Yahoo Finance."""

        def _fetch():
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or "symbol" not in info:
                return None

            return {
                "name": info.get("shortName") or info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
            }

        self.logger.debug(f"Fetching company info for {symbol}")
        return await asyncio.to_thread(_fetch)

    async def fetch_multiple_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        timeframe: str = "daily",
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch prices for multiple symbols concurrently.

        Args:
            symbols: List of stock symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            timeframe: 'daily' or 'hourly'

        Returns:
            Dict mapping symbol to list of price records
        """

        async def fetch_one(symbol: str) -> tuple[str, list[dict[str, Any]]]:
            try:
                prices = await self.fetch_prices(symbol, start_date, end_date, timeframe)
                return (symbol, prices)
            except Exception as e:
                self.logger.warning(f"Failed to fetch prices for {symbol}: {e}")
                return (symbol, [])

        # Fetch all symbols concurrently
        results = await asyncio.gather(*[fetch_one(s) for s in symbols])
        return dict(results)

    async def fetch_multiple_company_info(
        self, symbols: list[str]
    ) -> dict[str, dict[str, Any] | None]:
        """Fetch company info for multiple symbols concurrently."""

        async def fetch_one(symbol: str) -> tuple[str, dict[str, Any] | None]:
            try:
                info = await self.fetch_company_info(symbol)
                return (symbol, info)
            except Exception as e:
                self.logger.warning(f"Failed to fetch company info for {symbol}: {e}")
                return (symbol, None)

        results = await asyncio.gather(*[fetch_one(s) for s in symbols])
        return dict(results)

    async def fetch_earnings_dates(self, symbol: str) -> list[dict[str, Any]]:
        """Fetch upcoming earnings dates from Yahoo Finance.

        Returns:
            List of earnings records with date info
        """

        def _fetch():
            ticker = yf.Ticker(symbol)
            records = []

            # Get earnings dates from calendar
            try:
                calendar = ticker.calendar
                if calendar is not None:
                    # calendar can be a dict or DataFrame
                    if isinstance(calendar, dict):
                        earnings_dates = calendar.get('Earnings Date', [])
                        if earnings_dates:
                            # Can be a single date or list of dates
                            if not isinstance(earnings_dates, list):
                                earnings_dates = [earnings_dates]
                            for ed in earnings_dates:
                                if ed is not None:
                                    if isinstance(ed, date):
                                        records.append({
                                            "symbol": symbol,
                                            "event_date": ed,
                                            "event_time": None,
                                        })
                                    elif hasattr(ed, 'date'):
                                        records.append({
                                            "symbol": symbol,
                                            "event_date": ed.date(),
                                            "event_time": None,
                                        })
            except Exception as e:
                pass

            return records

        self.logger.debug(f"Fetching earnings dates for {symbol}")
        return await asyncio.to_thread(_fetch)

    async def fetch_multiple_earnings_dates(
        self, symbols: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch earnings dates for multiple symbols."""

        async def fetch_one(symbol: str) -> tuple[str, list[dict[str, Any]]]:
            try:
                dates = await self.fetch_earnings_dates(symbol)
                return (symbol, dates)
            except Exception as e:
                self.logger.warning(f"Failed to fetch earnings dates for {symbol}: {e}")
                return (symbol, [])

        results = await asyncio.gather(*[fetch_one(s) for s in symbols])
        return dict(results)
