"""Tests for event filtering utilities."""

from datetime import date

import pytest

from src.monitor.filters import (
    filter_by_date_range,
    filter_by_event_type,
    filter_by_flag,
    filter_by_sector,
    filter_by_symbol,
    filter_by_tier,
    group_by_date,
    group_by_symbol,
    sort_by_date,
    sort_by_interest,
)
from src.monitor.flags import InterestFlag
from src.monitor.surfacer import InterestTier, SurfacedEvent


@pytest.fixture
def sample_events() -> list[SurfacedEvent]:
    """Create sample events for testing."""
    return [
        SurfacedEvent(
            event_id="1",
            symbol="AAPL",
            event_type="earnings",
            event_date=date(2024, 3, 15),
            event_time="AMC",
            title="Apple Q1 Earnings",
            description=None,
            sector="Technology",
            interest_score=5,
            tier=InterestTier.HIGH,
            flags=[InterestFlag("momentum", "up 30%")],
        ),
        SurfacedEvent(
            event_id="2",
            symbol="MSFT",
            event_type="earnings",
            event_date=date(2024, 3, 16),
            event_time="BMO",
            title="Microsoft Q1 Earnings",
            description=None,
            sector="Technology",
            interest_score=3,
            tier=InterestTier.STANDARD,
            flags=[],
        ),
        SurfacedEvent(
            event_id="3",
            symbol="JPM",
            event_type="earnings",
            event_date=date(2024, 3, 14),
            event_time="BMO",
            title="JPMorgan Q1 Earnings",
            description=None,
            sector="Financial Services",
            interest_score=2,
            tier=InterestTier.STANDARD,
            flags=[InterestFlag("streak", "3 beats")],
        ),
        SurfacedEvent(
            event_id="4",
            symbol="NVDA",
            event_type="conference",
            event_date=date(2024, 3, 18),
            event_time=None,
            title="GTC Conference",
            description=None,
            sector="Technology",
            interest_score=1,
            tier=InterestTier.LOW,
            flags=[],
        ),
    ]


class TestFilterBySymbol:
    """Tests for filter_by_symbol."""

    def test_single_symbol(self, sample_events: list[SurfacedEvent]):
        """Test filtering to a single symbol."""
        result = filter_by_symbol(sample_events, ["AAPL"])
        assert len(result) == 1
        assert result[0].symbol == "AAPL"

    def test_multiple_symbols(self, sample_events: list[SurfacedEvent]):
        """Test filtering to multiple symbols."""
        result = filter_by_symbol(sample_events, ["AAPL", "MSFT"])
        assert len(result) == 2
        assert {e.symbol for e in result} == {"AAPL", "MSFT"}

    def test_case_insensitive(self, sample_events: list[SurfacedEvent]):
        """Test that symbol filtering is case-insensitive."""
        result = filter_by_symbol(sample_events, ["aapl", "msft"])
        assert len(result) == 2

    def test_no_matches(self, sample_events: list[SurfacedEvent]):
        """Test filtering with no matching symbols."""
        result = filter_by_symbol(sample_events, ["XYZ"])
        assert len(result) == 0


class TestFilterBySector:
    """Tests for filter_by_sector."""

    def test_single_sector(self, sample_events: list[SurfacedEvent]):
        """Test filtering to a single sector."""
        result = filter_by_sector(sample_events, ["Technology"])
        assert len(result) == 3

    def test_case_insensitive(self, sample_events: list[SurfacedEvent]):
        """Test that sector filtering is case-insensitive."""
        result = filter_by_sector(sample_events, ["technology"])
        assert len(result) == 3

    def test_financial_sector(self, sample_events: list[SurfacedEvent]):
        """Test filtering to financial sector."""
        result = filter_by_sector(sample_events, ["Financial Services"])
        assert len(result) == 1
        assert result[0].symbol == "JPM"


class TestFilterByEventType:
    """Tests for filter_by_event_type."""

    def test_earnings_only(self, sample_events: list[SurfacedEvent]):
        """Test filtering to earnings events."""
        result = filter_by_event_type(sample_events, ["earnings"])
        assert len(result) == 3

    def test_conference_only(self, sample_events: list[SurfacedEvent]):
        """Test filtering to conference events."""
        result = filter_by_event_type(sample_events, ["conference"])
        assert len(result) == 1
        assert result[0].symbol == "NVDA"


class TestFilterByDateRange:
    """Tests for filter_by_date_range."""

    def test_start_date_only(self, sample_events: list[SurfacedEvent]):
        """Test filtering with start date only."""
        result = filter_by_date_range(sample_events, start_date=date(2024, 3, 16))
        assert len(result) == 2
        assert {e.symbol for e in result} == {"MSFT", "NVDA"}

    def test_end_date_only(self, sample_events: list[SurfacedEvent]):
        """Test filtering with end date only."""
        result = filter_by_date_range(sample_events, end_date=date(2024, 3, 15))
        assert len(result) == 2
        assert {e.symbol for e in result} == {"AAPL", "JPM"}

    def test_date_range(self, sample_events: list[SurfacedEvent]):
        """Test filtering with both start and end dates."""
        result = filter_by_date_range(
            sample_events,
            start_date=date(2024, 3, 15),
            end_date=date(2024, 3, 16),
        )
        assert len(result) == 2
        assert {e.symbol for e in result} == {"AAPL", "MSFT"}


class TestFilterByTier:
    """Tests for filter_by_tier."""

    def test_high_tier_only(self, sample_events: list[SurfacedEvent]):
        """Test filtering to high interest tier."""
        result = filter_by_tier(sample_events, InterestTier.HIGH)
        assert len(result) == 1
        assert result[0].symbol == "AAPL"

    def test_standard_tier_includes_high(self, sample_events: list[SurfacedEvent]):
        """Test that standard tier includes high tier events."""
        result = filter_by_tier(sample_events, InterestTier.STANDARD)
        assert len(result) == 3
        tiers = {e.tier for e in result}
        assert InterestTier.HIGH in tiers
        assert InterestTier.STANDARD in tiers

    def test_low_tier_includes_all(self, sample_events: list[SurfacedEvent]):
        """Test that low tier includes all events."""
        result = filter_by_tier(sample_events, InterestTier.LOW)
        assert len(result) == 4


class TestFilterByFlag:
    """Tests for filter_by_flag."""

    def test_momentum_flag(self, sample_events: list[SurfacedEvent]):
        """Test filtering by momentum flag."""
        result = filter_by_flag(sample_events, "momentum")
        assert len(result) == 1
        assert result[0].symbol == "AAPL"

    def test_streak_flag(self, sample_events: list[SurfacedEvent]):
        """Test filtering by streak flag."""
        result = filter_by_flag(sample_events, "streak")
        assert len(result) == 1
        assert result[0].symbol == "JPM"

    def test_no_matching_flag(self, sample_events: list[SurfacedEvent]):
        """Test filtering with no matching flag."""
        result = filter_by_flag(sample_events, "vix_regime")
        assert len(result) == 0


class TestSortByInterest:
    """Tests for sort_by_interest."""

    def test_descending(self, sample_events: list[SurfacedEvent]):
        """Test sorting by interest score descending."""
        result = sort_by_interest(sample_events, descending=True)
        scores = [e.interest_score for e in result]
        assert scores == [5, 3, 2, 1]

    def test_ascending(self, sample_events: list[SurfacedEvent]):
        """Test sorting by interest score ascending."""
        result = sort_by_interest(sample_events, descending=False)
        scores = [e.interest_score for e in result]
        assert scores == [1, 2, 3, 5]


class TestSortByDate:
    """Tests for sort_by_date."""

    def test_ascending(self, sample_events: list[SurfacedEvent]):
        """Test sorting by date ascending."""
        result = sort_by_date(sample_events, descending=False)
        dates = [e.event_date for e in result]
        assert dates == sorted(dates)

    def test_descending(self, sample_events: list[SurfacedEvent]):
        """Test sorting by date descending."""
        result = sort_by_date(sample_events, descending=True)
        dates = [e.event_date for e in result]
        assert dates == sorted(dates, reverse=True)


class TestGroupByDate:
    """Tests for group_by_date."""

    def test_grouping(self, sample_events: list[SurfacedEvent]):
        """Test grouping events by date."""
        result = group_by_date(sample_events)
        assert len(result) == 4  # 4 different dates
        assert date(2024, 3, 15) in result
        assert len(result[date(2024, 3, 15)]) == 1


class TestGroupBySymbol:
    """Tests for group_by_symbol."""

    def test_grouping(self, sample_events: list[SurfacedEvent]):
        """Test grouping events by symbol."""
        result = group_by_symbol(sample_events)
        assert len(result) == 4  # 4 different symbols
        assert "AAPL" in result
        assert len(result["AAPL"]) == 1
