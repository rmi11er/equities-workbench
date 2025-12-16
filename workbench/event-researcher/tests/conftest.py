"""Pytest configuration and fixtures."""

import asyncio
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from src.data.database import Database
from src.data.models import Earnings, Event, Price, WatchlistItem


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db():
    """Create an in-memory database for testing."""
    database = Database(":memory:")
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def sample_prices() -> list[Price]:
    """Sample price data for testing."""
    return [
        Price(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 2, 0, 0),
            timeframe="daily",
            open=Decimal("185.00"),
            high=Decimal("186.50"),
            low=Decimal("184.00"),
            close=Decimal("185.50"),
            volume=50000000,
            source="test",
        ),
        Price(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 3, 0, 0),
            timeframe="daily",
            open=Decimal("185.50"),
            high=Decimal("187.00"),
            low=Decimal("185.00"),
            close=Decimal("186.75"),
            volume=48000000,
            source="test",
        ),
        Price(
            symbol="NVDA",
            timestamp=datetime(2024, 1, 2, 0, 0),
            timeframe="daily",
            open=Decimal("480.00"),
            high=Decimal("485.00"),
            low=Decimal("478.00"),
            close=Decimal("484.00"),
            volume=40000000,
            source="test",
        ),
    ]


@pytest.fixture
def sample_event() -> Event:
    """Sample event for testing."""
    return Event(
        event_id="test-event-001",
        symbol="AAPL",
        event_type="earnings",
        event_date=date(2024, 1, 25),
        event_time="AMC",
        title="AAPL Q1 2024 Earnings",
        description="Q1 FY24 earnings release",
        metadata={"fiscal_quarter": "Q1", "fiscal_year": 2024},
    )


@pytest.fixture
def sample_earnings() -> Earnings:
    """Sample earnings data for testing."""
    return Earnings(
        event_id="test-event-001",
        symbol="AAPL",
        fiscal_quarter="Q1 2024",
        fiscal_year=2024,
        eps_estimate=Decimal("2.10"),
        eps_actual=Decimal("2.18"),
        eps_surprise_pct=Decimal("3.81"),
        revenue_estimate=118000000000,
        revenue_actual=119575000000,
        revenue_surprise_pct=Decimal("1.34"),
        guidance_direction="maintained",
    )


@pytest.fixture
def sample_watchlist_item() -> WatchlistItem:
    """Sample watchlist item for testing."""
    return WatchlistItem(
        symbol="AAPL",
        name="Apple Inc.",
        sector="Technology",
        industry="Consumer Electronics",
        market_cap=3000000000000,
        notes="Core holding",
    )
