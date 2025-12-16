"""Financial Modeling Prep (FMP) data source."""

import asyncio
from datetime import date, datetime
from typing import Any
from uuid import uuid4

from src.data.sources.base import DataSource
from src.utils.config import get_settings


class FMPSource(DataSource):
    """Data source using Financial Modeling Prep API.

    Free tier includes:
    - Earnings calendar
    - Company profiles
    - Historical prices (limited)
    """

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self._api_key = api_key or get_settings().fmp_api_key

    @property
    def api_key(self) -> str:
        if not self._api_key:
            raise ValueError("FMP API key not configured. Set FMP_API_KEY environment variable.")
        return self._api_key

    async def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """Make authenticated GET request to FMP API."""
        url = f"{self.BASE_URL}/{endpoint}"
        request_params = {"apikey": self.api_key}
        if params:
            request_params.update(params)

        response = await self.client.get(url, params=request_params)
        response.raise_for_status()
        return response.json()

    async def fetch_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeframe: str = "daily",
    ) -> list[dict[str, Any]]:
        """Fetch historical price data from FMP.

        Note: Free tier has limited historical data.
        """
        endpoint = f"historical-price-full/{symbol}"
        params = {
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
        }

        self.logger.debug(f"Fetching {timeframe} prices for {symbol} from FMP")

        try:
            data = await self._get(endpoint, params)
        except Exception as e:
            self.logger.error(f"Failed to fetch prices from FMP for {symbol}: {e}")
            return []

        if not data or "historical" not in data:
            return []

        records = []
        for item in data["historical"]:
            try:
                timestamp = datetime.strptime(item["date"], "%Y-%m-%d")
                records.append(
                    {
                        "timestamp": timestamp,
                        "open": float(item["open"]),
                        "high": float(item["high"]),
                        "low": float(item["low"]),
                        "close": float(item["close"]),
                        "volume": int(item["volume"]),
                    }
                )
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Skipping malformed price record: {e}")
                continue

        # FMP returns newest first, reverse to chronological order
        records.reverse()
        return records

    async def fetch_company_info(self, symbol: str) -> dict[str, Any] | None:
        """Fetch company profile from FMP."""
        endpoint = f"profile/{symbol}"

        self.logger.debug(f"Fetching company info for {symbol} from FMP")

        try:
            data = await self._get(endpoint)
        except Exception as e:
            self.logger.error(f"Failed to fetch company info from FMP for {symbol}: {e}")
            return None

        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        profile = data[0]
        return {
            "name": profile.get("companyName"),
            "sector": profile.get("sector"),
            "industry": profile.get("industry"),
            "market_cap": profile.get("mktCap"),
        }

    async def fetch_earnings_calendar(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch earnings calendar from FMP.

        Returns list of dicts with keys:
        - symbol: str
        - event_date: date
        - event_time: str (BMO/AMC)
        - eps_estimate: float
        - revenue_estimate: int
        - fiscal_quarter: str
        - fiscal_year: int
        """
        endpoint = "earning_calendar"
        params = {}

        if start_date:
            params["from"] = start_date.isoformat()
        if end_date:
            params["to"] = end_date.isoformat()

        self.logger.debug(f"Fetching earnings calendar from FMP: {start_date} to {end_date}")

        try:
            data = await self._get(endpoint, params if params else None)
        except Exception as e:
            self.logger.error(f"Failed to fetch earnings calendar from FMP: {e}")
            return []

        if not data or not isinstance(data, list):
            return []

        records = []
        for item in data:
            try:
                event_date = datetime.strptime(item["date"], "%Y-%m-%d").date()

                # Determine event time (BMO/AMC)
                time_str = item.get("time", "")
                if time_str.lower() in ("bmo", "before market open"):
                    event_time = "BMO"
                elif time_str.lower() in ("amc", "after market close"):
                    event_time = "AMC"
                else:
                    event_time = time_str or None

                # Parse fiscal period
                fiscal_end = item.get("fiscalDateEnding")
                fiscal_quarter = None
                fiscal_year = None

                if fiscal_end:
                    try:
                        fiscal_dt = datetime.strptime(fiscal_end, "%Y-%m-%d")
                        fiscal_year = fiscal_dt.year
                        # Determine quarter from month
                        month = fiscal_dt.month
                        if month <= 3:
                            fiscal_quarter = "Q1"
                        elif month <= 6:
                            fiscal_quarter = "Q2"
                        elif month <= 9:
                            fiscal_quarter = "Q3"
                        else:
                            fiscal_quarter = "Q4"
                        fiscal_quarter = f"{fiscal_quarter} {fiscal_year}"
                    except ValueError:
                        pass

                records.append(
                    {
                        "event_id": str(uuid4()),
                        "symbol": item["symbol"],
                        "event_date": event_date,
                        "event_time": event_time,
                        "eps_estimate": item.get("epsEstimated"),
                        "eps_actual": item.get("eps"),
                        "revenue_estimate": item.get("revenueEstimated"),
                        "revenue_actual": item.get("revenue"),
                        "fiscal_quarter": fiscal_quarter,
                        "fiscal_year": fiscal_year,
                    }
                )
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Skipping malformed earnings record: {e}")
                continue

        return records

    async def fetch_earnings_for_symbol(
        self,
        symbol: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Fetch historical earnings for a specific symbol."""
        endpoint = f"historical/earning_calendar/{symbol}"
        params = {"limit": limit}

        self.logger.debug(f"Fetching earnings history for {symbol} from FMP")

        try:
            data = await self._get(endpoint, params)
        except Exception as e:
            self.logger.error(f"Failed to fetch earnings for {symbol} from FMP: {e}")
            return []

        if not data or not isinstance(data, list):
            return []

        records = []
        for item in data:
            try:
                event_date = datetime.strptime(item["date"], "%Y-%m-%d").date()

                # Parse fiscal period
                fiscal_end = item.get("fiscalDateEnding")
                fiscal_quarter = None
                fiscal_year = None

                if fiscal_end:
                    try:
                        fiscal_dt = datetime.strptime(fiscal_end, "%Y-%m-%d")
                        fiscal_year = fiscal_dt.year
                        month = fiscal_dt.month
                        if month <= 3:
                            fiscal_quarter = "Q1"
                        elif month <= 6:
                            fiscal_quarter = "Q2"
                        elif month <= 9:
                            fiscal_quarter = "Q3"
                        else:
                            fiscal_quarter = "Q4"
                        fiscal_quarter = f"{fiscal_quarter} {fiscal_year}"
                    except ValueError:
                        pass

                # Calculate surprise percentage if we have both estimate and actual
                eps_surprise_pct = None
                eps_estimate = item.get("epsEstimated")
                eps_actual = item.get("eps")
                if eps_estimate and eps_actual and eps_estimate != 0:
                    eps_surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100

                revenue_surprise_pct = None
                rev_estimate = item.get("revenueEstimated")
                rev_actual = item.get("revenue")
                if rev_estimate and rev_actual and rev_estimate != 0:
                    revenue_surprise_pct = ((rev_actual - rev_estimate) / abs(rev_estimate)) * 100

                records.append(
                    {
                        "event_id": str(uuid4()),
                        "symbol": symbol,
                        "event_date": event_date,
                        "event_time": item.get("time"),
                        "eps_estimate": eps_estimate,
                        "eps_actual": eps_actual,
                        "eps_surprise_pct": eps_surprise_pct,
                        "revenue_estimate": rev_estimate,
                        "revenue_actual": rev_actual,
                        "revenue_surprise_pct": revenue_surprise_pct,
                        "fiscal_quarter": fiscal_quarter,
                        "fiscal_year": fiscal_year,
                    }
                )
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Skipping malformed earnings record for {symbol}: {e}")
                continue

        return records

    async def fetch_earnings_transcript(
        self,
        symbol: str,
        year: int,
        quarter: int,
    ) -> dict[str, Any] | None:
        """Fetch earnings call transcript from FMP.

        Note: Transcripts may require paid tier.
        """
        endpoint = f"earning_call_transcript/{symbol}"
        params = {"year": year, "quarter": quarter}

        self.logger.debug(f"Fetching transcript for {symbol} Q{quarter} {year} from FMP")

        try:
            data = await self._get(endpoint, params)
        except Exception as e:
            self.logger.error(f"Failed to fetch transcript from FMP: {e}")
            return None

        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        transcript = data[0]
        return {
            "symbol": symbol,
            "quarter": quarter,
            "year": year,
            "date": transcript.get("date"),
            "content": transcript.get("content"),
        }
