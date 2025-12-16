"""Base class for data sources."""

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import httpx

from src.utils.logging import get_logger


class DataSource(ABC):
    """Abstract base class for data sources."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @abstractmethod
    async def fetch_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeframe: str = "daily",
    ) -> list[dict[str, Any]]:
        """Fetch price data for a symbol.

        Returns list of dicts with keys:
        - timestamp: datetime
        - open, high, low, close: float
        - volume: int
        """
        pass

    @abstractmethod
    async def fetch_company_info(self, symbol: str) -> dict[str, Any] | None:
        """Fetch company information.

        Returns dict with keys:
        - name: str
        - sector: str
        - industry: str
        - market_cap: int
        """
        pass
