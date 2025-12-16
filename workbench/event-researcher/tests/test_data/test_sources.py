"""Tests for data sources.

Note: These tests make actual API calls. Some may be skipped in CI
or if API keys are not configured.
"""

from datetime import date, timedelta

import pytest

from src.data.sources.yfinance_source import YFinanceSource


class TestYFinanceSource:
    """Tests for Yahoo Finance data source."""

    @pytest.fixture
    def source(self):
        """Create YFinance source."""
        return YFinanceSource()

    async def test_fetch_prices_daily(self, source: YFinanceSource):
        """Test fetching daily price data."""
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=30)

        prices = await source.fetch_prices("AAPL", start_date, end_date, "daily")

        assert len(prices) > 0
        assert all(
            key in prices[0] for key in ["timestamp", "open", "high", "low", "close", "volume"]
        )
        assert prices[0]["open"] > 0
        assert prices[0]["volume"] > 0

    async def test_fetch_prices_invalid_symbol(self, source: YFinanceSource):
        """Test fetching prices for invalid symbol."""
        end_date = date.today()
        start_date = end_date - timedelta(days=7)

        prices = await source.fetch_prices("INVALIDSYMBOL123", start_date, end_date)

        assert prices == []

    async def test_fetch_company_info(self, source: YFinanceSource):
        """Test fetching company information."""
        info = await source.fetch_company_info("AAPL")

        assert info is not None
        assert info["name"] is not None
        assert "Apple" in info["name"]
        assert info["sector"] == "Technology"
        assert info["market_cap"] is not None
        assert info["market_cap"] > 0

    async def test_fetch_company_info_invalid_symbol(self, source: YFinanceSource):
        """Test fetching info for invalid symbol."""
        info = await source.fetch_company_info("INVALIDSYMBOL123")

        assert info is None

    async def test_fetch_multiple_prices(self, source: YFinanceSource):
        """Test fetching prices for multiple symbols."""
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=7)

        results = await source.fetch_multiple_prices(
            ["AAPL", "MSFT", "GOOGL"], start_date, end_date
        )

        assert len(results) == 3
        assert "AAPL" in results
        assert "MSFT" in results
        assert "GOOGL" in results
        assert all(len(prices) > 0 for prices in results.values())

    async def test_fetch_multiple_company_info(self, source: YFinanceSource):
        """Test fetching company info for multiple symbols."""
        results = await source.fetch_multiple_company_info(["AAPL", "MSFT"])

        assert len(results) == 2
        assert results["AAPL"] is not None
        assert results["MSFT"] is not None
        assert "Apple" in results["AAPL"]["name"]
        assert "Microsoft" in results["MSFT"]["name"]


class TestFMPSource:
    """Tests for FMP data source.

    These tests require FMP_API_KEY to be configured.
    """

    @pytest.fixture
    def source(self):
        """Create FMP source."""
        from src.data.sources.fmp import FMPSource
        from src.utils.config import get_settings

        settings = get_settings()
        if not settings.fmp_api_key:
            pytest.skip("FMP API key not configured")
        return FMPSource()

    async def test_fetch_earnings_calendar(self, source):
        """Test fetching earnings calendar."""
        start_date = date.today()
        end_date = start_date + timedelta(days=14)

        earnings = await source.fetch_earnings_calendar(start_date, end_date)

        # Should have some earnings in the next 2 weeks
        assert isinstance(earnings, list)
        # Can't guarantee there will be earnings, but structure should be correct
        if earnings:
            assert "symbol" in earnings[0]
            assert "event_date" in earnings[0]

    async def test_fetch_earnings_for_symbol(self, source):
        """Test fetching historical earnings for a symbol."""
        try:
            earnings = await source.fetch_earnings_for_symbol("AAPL", limit=4)
        except Exception as e:
            if "403" in str(e) or "Forbidden" in str(e):
                pytest.skip("FMP API returned 403 - API key may be invalid or rate limited")
            raise

        if len(earnings) == 0:
            pytest.skip("FMP API returned no data - API key may be invalid or rate limited")

        assert earnings[0]["symbol"] == "AAPL"
        assert "eps_estimate" in earnings[0]
        assert "eps_actual" in earnings[0]

    async def test_fetch_company_info(self, source):
        """Test fetching company profile."""
        try:
            info = await source.fetch_company_info("AAPL")
        except Exception as e:
            if "403" in str(e) or "Forbidden" in str(e):
                pytest.skip("FMP API returned 403 - API key may be invalid or rate limited")
            raise

        if info is None:
            pytest.skip("FMP API returned no data - API key may be invalid")

        assert info["name"] is not None
        assert info["sector"] is not None
