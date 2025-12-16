"""Tests for event surfacing logic."""

from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from src.data.database import Database
from src.data.models import Earnings, Event, Price, WatchlistItem
from src.monitor.flags import (
    InterestFlag,
    calculate_momentum_flag,
    calculate_streak_flag,
)
from src.monitor.surfacer import EventSurfacer, InterestTier, SurfacedEvent


@pytest.fixture
async def db_with_test_data():
    """Create database with test data for surfacing tests."""
    db = Database(":memory:")
    await db.initialize()

    # Add watchlist entries
    await db.add_to_watchlist(
        WatchlistItem(
            symbol="AAPL",
            name="Apple Inc.",
            sector="Technology",
            market_cap=3000000000000,
        )
    )
    await db.add_to_watchlist(
        WatchlistItem(
            symbol="NVDA",
            name="NVIDIA Corp",
            sector="Technology",
            market_cap=1500000000000,
        )
    )
    await db.add_to_watchlist(
        WatchlistItem(
            symbol="JPM",
            name="JPMorgan Chase",
            sector="Financial Services",
            market_cap=500000000000,
        )
    )

    # Add price data for momentum calculation
    today = date.today()
    prices = []
    base_prices = {"AAPL": 150.0, "NVDA": 500.0, "JPM": 180.0}

    for i in range(100):
        d = today - timedelta(days=100 - i)
        if d.weekday() >= 5:
            continue

        for symbol, base in base_prices.items():
            # AAPL has upward momentum
            if symbol == "AAPL":
                price = base * (1 + i * 0.005)
            # NVDA flat
            elif symbol == "NVDA":
                price = base * (1 + (i % 5 - 2) * 0.002)
            # JPM downward
            else:
                price = base * (1 - i * 0.003)

            prices.append(
                Price(
                    symbol=symbol,
                    timestamp=datetime(d.year, d.month, d.day),
                    timeframe="daily",
                    open=Decimal(str(price)),
                    high=Decimal(str(price * 1.01)),
                    low=Decimal(str(price * 0.99)),
                    close=Decimal(str(price)),
                    volume=10000000,
                    source="test",
                )
            )

    await db.insert_prices(prices)

    # Add upcoming events
    events = [
        Event(
            event_id="aapl-earnings-q1",
            symbol="AAPL",
            event_type="earnings",
            event_date=today + timedelta(days=3),
            event_time="AMC",
            title="Apple Q1 Earnings",
        ),
        Event(
            event_id="nvda-earnings-q4",
            symbol="NVDA",
            event_type="earnings",
            event_date=today + timedelta(days=5),
            event_time="BMO",
            title="NVIDIA Q4 Earnings",
        ),
        Event(
            event_id="jpm-earnings-q1",
            symbol="JPM",
            event_type="earnings",
            event_date=today + timedelta(days=7),
            event_time="BMO",
            title="JPMorgan Q1 Earnings",
        ),
    ]

    for event in events:
        await db.insert_event(event)

    # Add earnings history for streak calculation
    for symbol in ["AAPL", "NVDA", "JPM"]:
        for q in range(4):
            # AAPL has beats, JPM has misses
            if symbol == "AAPL":
                surprise = 5.0 + q
            elif symbol == "JPM":
                surprise = -3.0 - q
            else:
                surprise = 1.0 if q % 2 == 0 else -1.0

            event_id = f"{symbol}-hist-earnings-{q}"
            event_date = today - timedelta(days=90 * (q + 1))

            # Create corresponding event
            await db.insert_event(
                Event(
                    event_id=event_id,
                    symbol=symbol,
                    event_type="earnings",
                    event_date=event_date,
                    event_time="AMC",
                    title=f"{symbol} Q{4 - q} 2023 Earnings",
                )
            )

            # Add earnings data
            await db.insert_earnings(
                Earnings(
                    event_id=event_id,
                    symbol=symbol,
                    fiscal_quarter=f"Q{4 - q} 2023",
                    eps_estimate=Decimal("2.00"),
                    eps_actual=Decimal(str(2.00 * (1 + surprise / 100))),
                    eps_surprise_pct=Decimal(str(surprise)),
                )
            )

    yield db
    await db.close()


class TestSurfacedEvent:
    """Tests for SurfacedEvent dataclass."""

    def test_to_dict(self):
        """Test converting surfaced event to dictionary."""
        event = SurfacedEvent(
            event_id="test-1",
            symbol="AAPL",
            event_type="earnings",
            event_date=date(2024, 3, 15),
            event_time="AMC",
            title="Apple Earnings",
            description=None,
            company_name="Apple Inc.",
            sector="Technology",
            interest_score=5,
            tier=InterestTier.HIGH,
            flags=[InterestFlag("momentum", "up 30%", 30.0)],
            days_until=3,
        )

        d = event.to_dict()

        assert d["event_id"] == "test-1"
        assert d["symbol"] == "AAPL"
        assert d["tier"] == "high"
        assert d["interest_score"] == 5
        assert len(d["flags"]) == 1
        assert d["flags"][0]["name"] == "momentum"


class TestEventSurfacer:
    """Tests for EventSurfacer class."""

    async def test_get_surfaced_events(self, db_with_test_data: Database):
        """Test basic event surfacing."""
        surfacer = EventSurfacer(db_with_test_data)
        events = await surfacer.get_surfaced_events(lookahead_days=14)

        assert len(events) >= 3
        # Events should be sorted by date then score
        dates = [e.event_date for e in events]
        assert dates == sorted(dates)

    async def test_surfaced_events_have_enrichment(self, db_with_test_data: Database):
        """Test that surfaced events have company info enrichment."""
        surfacer = EventSurfacer(db_with_test_data)
        events = await surfacer.get_surfaced_events(lookahead_days=14)

        aapl_event = next((e for e in events if e.symbol == "AAPL"), None)
        assert aapl_event is not None
        assert aapl_event.company_name == "Apple Inc."
        assert aapl_event.sector == "Technology"

    async def test_min_tier_filtering(self, db_with_test_data: Database):
        """Test filtering by minimum tier."""
        surfacer = EventSurfacer(db_with_test_data)

        # Get all events
        all_events = await surfacer.get_surfaced_events(
            lookahead_days=14, min_tier=InterestTier.LOW
        )

        # Get only standard and above
        standard_events = await surfacer.get_surfaced_events(
            lookahead_days=14, min_tier=InterestTier.STANDARD
        )

        # Standard should be subset of all
        assert len(standard_events) <= len(all_events)

    async def test_get_events_by_tier(self, db_with_test_data: Database):
        """Test grouping events by tier."""
        surfacer = EventSurfacer(db_with_test_data)
        grouped = await surfacer.get_events_by_tier(lookahead_days=14)

        assert InterestTier.HIGH in grouped
        assert InterestTier.STANDARD in grouped
        assert InterestTier.LOW in grouped

        # All events should be accounted for
        total = sum(len(events) for events in grouped.values())
        all_events = await surfacer.get_surfaced_events(lookahead_days=14)
        assert total == len(all_events)

    async def test_get_event_summary(self, db_with_test_data: Database):
        """Test event summary generation."""
        surfacer = EventSurfacer(db_with_test_data)
        summary = await surfacer.get_event_summary(lookahead_days=14)

        assert "total" in summary
        assert "high_interest" in summary
        assert "standard" in summary
        assert "low" in summary
        assert "events_by_tier" in summary

        # Total should match sum of tiers
        assert summary["total"] == (
            summary["high_interest"] + summary["standard"] + summary["low"]
        )


class TestInterestFlags:
    """Tests for individual interest flag calculations."""

    async def test_momentum_flag_upward(self, db_with_test_data: Database):
        """Test momentum flag for upward movement."""
        flag = await calculate_momentum_flag(
            db_with_test_data,
            "AAPL",
            threshold_pct=20,
            window_days=90,
        )

        # AAPL should have upward momentum based on test data
        assert flag is not None
        assert flag.name == "momentum"
        assert "up" in flag.description.lower()

    async def test_momentum_flag_downward(self, db_with_test_data: Database):
        """Test momentum flag for downward movement."""
        flag = await calculate_momentum_flag(
            db_with_test_data,
            "JPM",
            threshold_pct=10,
            window_days=90,
        )

        # JPM should have downward momentum
        assert flag is not None
        assert flag.name == "momentum"
        assert "down" in flag.description.lower()

    async def test_streak_flag_beats(self, db_with_test_data: Database):
        """Test streak flag for consecutive beats."""
        flag = await calculate_streak_flag(
            db_with_test_data,
            "AAPL",
            threshold=3,
        )

        # AAPL should have consecutive beats
        assert flag is not None
        assert flag.name == "streak"
        assert "beats" in flag.description.lower()

    async def test_streak_flag_misses(self, db_with_test_data: Database):
        """Test streak flag for consecutive misses."""
        flag = await calculate_streak_flag(
            db_with_test_data,
            "JPM",
            threshold=3,
        )

        # JPM should have consecutive misses
        assert flag is not None
        assert flag.name == "streak"
        assert "misses" in flag.description.lower()

    async def test_no_streak_flag_mixed(self, db_with_test_data: Database):
        """Test no streak flag for mixed results."""
        flag = await calculate_streak_flag(
            db_with_test_data,
            "NVDA",
            threshold=3,
        )

        # NVDA has mixed results - should not have streak
        assert flag is None
