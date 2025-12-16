"""Tests for database operations."""

from datetime import date, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from src.data.database import Database
from src.data.models import (
    ConversationMessage,
    Earnings,
    Event,
    Price,
    ResearchSession,
    Transcript,
    WatchlistItem,
)


class TestDatabaseInitialization:
    """Tests for database initialization."""

    async def test_initialize_creates_tables(self, db: Database):
        """Test that initialization creates all required tables."""
        result = await db._fetchall(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        )
        tables = {r[0] for r in result}

        expected_tables = {
            "prices",
            "events",
            "earnings",
            "watchlist",
            "research_sessions",
            "conversation_history",
            "transcripts",
            "cache",
            "schema_version",
        }
        assert expected_tables.issubset(tables)

    async def test_schema_version_recorded(self, db: Database):
        """Test that schema version is recorded."""
        result = await db._fetchone("SELECT version FROM schema_version")
        assert result is not None
        assert result[0] >= 1


class TestPriceOperations:
    """Tests for price-related database operations."""

    async def test_insert_prices(self, db: Database, sample_prices: list[Price]):
        """Test inserting price records."""
        count = await db.insert_prices(sample_prices)
        assert count == 3

    async def test_insert_prices_empty_list(self, db: Database):
        """Test inserting empty price list."""
        count = await db.insert_prices([])
        assert count == 0

    async def test_get_prices(self, db: Database, sample_prices: list[Price]):
        """Test retrieving price records."""
        await db.insert_prices(sample_prices)

        df = await db.get_prices(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert len(df) == 2
        assert df["symbol"].to_list() == ["AAPL", "AAPL"]

    async def test_get_prices_multiple_symbols(self, db: Database, sample_prices: list[Price]):
        """Test retrieving prices for multiple symbols."""
        await db.insert_prices(sample_prices)

        df = await db.get_prices(
            symbols=["AAPL", "NVDA"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert len(df) == 3
        symbols = set(df["symbol"].to_list())
        assert symbols == {"AAPL", "NVDA"}

    async def test_get_latest_price_date(self, db: Database, sample_prices: list[Price]):
        """Test getting latest price date for a symbol."""
        await db.insert_prices(sample_prices)

        latest = await db.get_latest_price_date("AAPL", "daily")
        assert latest == date(2024, 1, 3)

    async def test_get_latest_price_date_no_data(self, db: Database):
        """Test getting latest price date when no data exists."""
        latest = await db.get_latest_price_date("UNKNOWN", "daily")
        assert latest is None

    async def test_insert_prices_upsert(self, db: Database, sample_prices: list[Price]):
        """Test that inserting duplicate prices updates them."""
        await db.insert_prices(sample_prices)

        # Update one price
        updated_price = Price(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 2, 0, 0),
            timeframe="daily",
            open=Decimal("186.00"),  # Changed
            high=Decimal("186.50"),
            low=Decimal("184.00"),
            close=Decimal("185.50"),
            volume=50000000,
            source="test",
        )
        await db.insert_prices([updated_price])

        df = await db.get_prices(
            symbols=["AAPL"],
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
        )

        assert len(df) == 1
        assert float(df["open"][0]) == 186.00


class TestEventOperations:
    """Tests for event-related database operations."""

    async def test_insert_event(self, db: Database, sample_event: Event):
        """Test inserting an event."""
        await db.insert_event(sample_event)

        df = await db.get_events(symbols=["AAPL"])
        assert len(df) == 1
        assert df["event_id"][0] == "test-event-001"

    async def test_get_events_by_type(self, db: Database, sample_event: Event):
        """Test filtering events by type."""
        await db.insert_event(sample_event)

        # Add another event type
        conference = Event(
            event_id="test-event-002",
            symbol="AAPL",
            event_type="conference",
            event_date=date(2024, 2, 1),
            title="AAPL Investor Day",
        )
        await db.insert_event(conference)

        df = await db.get_events(event_types=["earnings"])
        assert len(df) == 1
        assert df["event_type"][0] == "earnings"

    async def test_get_events_by_date_range(self, db: Database, sample_event: Event):
        """Test filtering events by date range."""
        await db.insert_event(sample_event)

        # Event outside range
        df = await db.get_events(
            start_date=date(2024, 2, 1),
            end_date=date(2024, 2, 28),
        )
        assert len(df) == 0

        # Event inside range
        df = await db.get_events(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )
        assert len(df) == 1


class TestEarningsOperations:
    """Tests for earnings-related database operations."""

    async def test_insert_earnings(
        self, db: Database, sample_event: Event, sample_earnings: Earnings
    ):
        """Test inserting earnings data."""
        await db.insert_event(sample_event)
        await db.insert_earnings(sample_earnings)

        df = await db.get_earnings(symbols=["AAPL"])
        assert len(df) == 1
        assert df["symbol"][0] == "AAPL"
        assert float(df["eps_actual"][0]) == 2.18

    async def test_get_earnings_with_actuals_only(
        self, db: Database, sample_event: Event, sample_earnings: Earnings
    ):
        """Test filtering earnings to only those with actuals."""
        await db.insert_event(sample_event)
        await db.insert_earnings(sample_earnings)

        # Add upcoming earnings without actuals
        future_event = Event(
            event_id="test-event-003",
            symbol="AAPL",
            event_type="earnings",
            event_date=date(2024, 4, 25),
        )
        await db.insert_event(future_event)
        future_earnings = Earnings(
            event_id="test-event-003",
            symbol="AAPL",
            eps_estimate=Decimal("2.20"),
        )
        await db.insert_earnings(future_earnings)

        df = await db.get_earnings(symbols=["AAPL"], with_actuals_only=True)
        assert len(df) == 1
        assert df["event_id"][0] == "test-event-001"


class TestWatchlistOperations:
    """Tests for watchlist operations."""

    async def test_add_to_watchlist(self, db: Database, sample_watchlist_item: WatchlistItem):
        """Test adding item to watchlist."""
        await db.add_to_watchlist(sample_watchlist_item)

        df = await db.get_watchlist()
        assert len(df) == 1
        assert df["symbol"][0] == "AAPL"
        assert df["name"][0] == "Apple Inc."

    async def test_get_watchlist_by_sector(self, db: Database, sample_watchlist_item: WatchlistItem):
        """Test filtering watchlist by sector."""
        await db.add_to_watchlist(sample_watchlist_item)

        # Add item in different sector
        financials = WatchlistItem(
            symbol="JPM",
            name="JPMorgan Chase",
            sector="Financials",
        )
        await db.add_to_watchlist(financials)

        df = await db.get_watchlist(sectors=["Technology"])
        assert len(df) == 1
        assert df["symbol"][0] == "AAPL"

    async def test_remove_from_watchlist(self, db: Database, sample_watchlist_item: WatchlistItem):
        """Test removing item from watchlist."""
        await db.add_to_watchlist(sample_watchlist_item)
        await db.remove_from_watchlist("AAPL")

        df = await db.get_watchlist()
        assert len(df) == 0


class TestSessionOperations:
    """Tests for research session operations."""

    async def test_create_session(self, db: Database):
        """Test creating a research session."""
        session = ResearchSession(
            session_id="session-001",
            target_symbol="NVDA",
            title="NVDA Earnings Research",
            status="active",
            scenarios={"bull": 0.4, "base": 0.4, "bear": 0.2},
        )
        await db.create_session(session)

        result = await db.get_session("session-001")
        assert result is not None
        assert result.target_symbol == "NVDA"
        assert result.scenarios["bull"] == 0.4

    async def test_update_session(self, db: Database):
        """Test updating a research session."""
        session = ResearchSession(
            session_id="session-001",
            target_symbol="NVDA",
            title="NVDA Earnings Research",
        )
        await db.create_session(session)

        session.status = "archived"
        session.context_summary = "Analyzed Q3 earnings, decided to pass"
        await db.update_session(session)

        result = await db.get_session("session-001")
        assert result.status == "archived"
        assert "pass" in result.context_summary

    async def test_get_nonexistent_session(self, db: Database):
        """Test getting a session that doesn't exist."""
        result = await db.get_session("nonexistent")
        assert result is None


class TestConversationOperations:
    """Tests for conversation history operations."""

    async def test_add_and_get_messages(self, db: Database):
        """Test adding and retrieving conversation messages."""
        session = ResearchSession(
            session_id="session-001",
            target_symbol="NVDA",
        )
        await db.create_session(session)

        msg1 = ConversationMessage(
            message_id="msg-001",
            session_id="session-001",
            role="user",
            content="How did NVDA move after last earnings?",
        )
        msg2 = ConversationMessage(
            message_id="msg-002",
            session_id="session-001",
            role="assistant",
            content="NVDA moved +5.2% after Q2 earnings...",
            tool_calls={"query_prices": {"symbol": "NVDA"}},
        )

        await db.add_message(msg1)
        await db.add_message(msg2)

        messages = await db.get_conversation("session-001")
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[1].tool_calls is not None


class TestCacheOperations:
    """Tests for cache operations."""

    async def test_set_and_get_cache(self, db: Database):
        """Test setting and getting cache entries."""
        await db.set_cache("test-key", {"data": "value"})

        result = await db.get_cache("test-key")
        assert result == {"data": "value"}

    async def test_cache_expiration(self, db: Database):
        """Test that expired cache entries are not returned."""
        import asyncio

        # Set cache with 1 second TTL
        await db.set_cache("expired-key", {"data": "old"}, ttl_seconds=1)

        # Should exist immediately
        result = await db.get_cache("expired-key")
        assert result == {"data": "old"}

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired now
        result = await db.get_cache("expired-key")
        assert result is None

    async def test_get_nonexistent_cache(self, db: Database):
        """Test getting cache entry that doesn't exist."""
        result = await db.get_cache("nonexistent")
        assert result is None

    async def test_clear_expired_cache(self, db: Database):
        """Test clearing expired cache entries."""
        import asyncio

        await db.set_cache("expired", {"data": "old"}, ttl_seconds=1)
        await db.set_cache("valid", {"data": "new"}, ttl_seconds=3600)

        # Wait for first entry to expire
        await asyncio.sleep(1.1)

        count = await db.clear_expired_cache()
        assert count >= 1

        # Valid entry should still exist
        result = await db.get_cache("valid")
        assert result == {"data": "new"}
