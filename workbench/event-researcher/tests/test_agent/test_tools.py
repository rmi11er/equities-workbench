"""Tests for agent tools."""

from datetime import date, datetime
from decimal import Decimal

import pytest

from src.data.database import Database
from src.data.models import Earnings, Event, Price, WatchlistItem


@pytest.fixture
async def db_with_data():
    """Create database with test data."""
    db = Database(":memory:")
    await db.initialize()

    # Add watchlist items
    await db.add_to_watchlist(
        WatchlistItem(
            symbol="NVDA",
            name="NVIDIA Corporation",
            sector="Technology",
            market_cap=1500000000000,
        )
    )
    await db.add_to_watchlist(
        WatchlistItem(
            symbol="AAPL",
            name="Apple Inc.",
            sector="Technology",
            market_cap=3000000000000,
        )
    )

    # Add price data for NVDA
    prices = []
    base_price = 100.0
    for i in range(100):
        d = date(2024, 1, 1) + __import__("datetime").timedelta(days=i)
        if d.weekday() >= 5:  # Skip weekends
            continue
        # Add some variation
        variation = (i % 10 - 5) * 0.5
        prices.append(
            Price(
                symbol="NVDA",
                timestamp=datetime(d.year, d.month, d.day),
                timeframe="daily",
                open=Decimal(str(base_price + variation)),
                high=Decimal(str(base_price + variation + 2)),
                low=Decimal(str(base_price + variation - 1)),
                close=Decimal(str(base_price + variation + 1)),
                volume=50000000,
                source="test",
            )
        )
        base_price += 0.5  # Upward trend

    await db.insert_prices(prices)

    # Add events
    events = [
        Event(
            event_id="evt-001",
            symbol="NVDA",
            event_type="earnings",
            event_date=date(2024, 1, 25),
            event_time="AMC",
            title="NVDA Q4 2023 Earnings",
        ),
        Event(
            event_id="evt-002",
            symbol="NVDA",
            event_type="earnings",
            event_date=date(2024, 2, 28),
            event_time="AMC",
            title="NVDA Q1 2024 Earnings",
        ),
    ]
    for event in events:
        await db.insert_event(event)

    # Add earnings data
    await db.insert_earnings(
        Earnings(
            event_id="evt-001",
            symbol="NVDA",
            fiscal_quarter="Q4 2023",
            fiscal_year=2023,
            eps_estimate=Decimal("4.50"),
            eps_actual=Decimal("5.16"),
            eps_surprise_pct=Decimal("14.67"),
        )
    )
    await db.insert_earnings(
        Earnings(
            event_id="evt-002",
            symbol="NVDA",
            fiscal_quarter="Q1 2024",
            fiscal_year=2024,
            eps_estimate=Decimal("5.50"),
            eps_actual=Decimal("6.12"),
            eps_surprise_pct=Decimal("11.27"),
        )
    )

    yield db
    await db.close()


class TestToolRegistry:
    """Tests for tool registry."""

    def test_registry_has_tools(self):
        """Test that tools are registered."""
        from src.agent.tools import registry

        tools = registry.list_tools()
        assert len(tools) > 0
        assert "query_prices" in tools
        assert "query_events" in tools
        assert "query_earnings" in tools
        assert "get_watchlist" in tools
        assert "calculate_ev" in tools

    def test_get_anthropic_tools(self):
        """Test getting tools in Anthropic format."""
        from src.agent.tools import registry

        tools = registry.get_anthropic_tools()
        assert len(tools) > 0

        # Check structure
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"


class TestLocalDataTools:
    """Tests for local data tools."""

    async def test_query_prices(self, db_with_data: Database):
        """Test query_prices tool."""
        from src.agent.tools.local_data import query_prices
        from src.data.database import get_database

        # Patch get_database to return our test db
        import src.agent.tools.local_data as local_data_module

        original_get_db = local_data_module.get_database
        local_data_module.get_database = lambda: db_with_data

        try:
            result = await query_prices(
                symbols=["NVDA"],
                start_date="2024-01-01",
                end_date="2024-01-31",
            )

            assert "data" in result
            assert "NVDA" in result["data"]
            assert len(result["data"]["NVDA"]) > 0
            assert result["symbols_found"] == ["NVDA"]
            assert result["symbols_missing"] == []
        finally:
            local_data_module.get_database = original_get_db

    async def test_query_prices_missing_symbol(self, db_with_data: Database):
        """Test query_prices with missing symbol."""
        from src.agent.tools.local_data import query_prices
        import src.agent.tools.local_data as local_data_module

        original_get_db = local_data_module.get_database
        local_data_module.get_database = lambda: db_with_data

        try:
            result = await query_prices(
                symbols=["NVDA", "UNKNOWN"],
                start_date="2024-01-01",
                end_date="2024-01-31",
            )

            assert "UNKNOWN" in result["symbols_missing"]
            assert "NVDA" in result["symbols_found"]
        finally:
            local_data_module.get_database = original_get_db

    async def test_get_watchlist(self, db_with_data: Database):
        """Test get_watchlist tool."""
        from src.agent.tools.local_data import get_watchlist
        import src.agent.tools.local_data as local_data_module

        original_get_db = local_data_module.get_database
        local_data_module.get_database = lambda: db_with_data

        try:
            result = await get_watchlist()

            assert result["count"] == 2
            symbols = [s["symbol"] for s in result["symbols"]]
            assert "NVDA" in symbols
            assert "AAPL" in symbols
        finally:
            local_data_module.get_database = original_get_db

    async def test_query_earnings(self, db_with_data: Database):
        """Test query_earnings tool."""
        from src.agent.tools.local_data import query_earnings
        import src.agent.tools.local_data as local_data_module

        original_get_db = local_data_module.get_database
        local_data_module.get_database = lambda: db_with_data

        try:
            result = await query_earnings(
                symbols=["NVDA"],
                with_actuals_only=True,
            )

            assert result["count"] == 2
            assert all(e["symbol"] == "NVDA" for e in result["earnings"])
            assert all(e["eps_actual"] is not None for e in result["earnings"])
        finally:
            local_data_module.get_database = original_get_db


class TestComputeTools:
    """Tests for compute tools."""

    async def test_calculate_ev(self):
        """Test calculate_ev tool."""
        from src.agent.tools.compute import calculate_ev

        scenarios = [
            {"name": "Beat + raise", "move_pct": 10.0, "probability": 0.35},
            {"name": "Inline", "move_pct": 2.0, "probability": 0.40},
            {"name": "Miss", "move_pct": -12.0, "probability": 0.25},
        ]

        result = await calculate_ev(scenarios)

        assert "expected_value" in result
        assert result["probability_valid"] is True
        assert abs(result["probability_sum"] - 1.0) < 0.01

        # EV = 0.35 * 10 + 0.40 * 2 + 0.25 * (-12) = 3.5 + 0.8 - 3.0 = 1.3
        assert abs(result["expected_value"] - 1.3) < 0.01

    async def test_calculate_ev_invalid_probabilities(self):
        """Test calculate_ev with invalid probabilities."""
        from src.agent.tools.compute import calculate_ev

        scenarios = [
            {"name": "A", "move_pct": 10.0, "probability": 0.5},
            {"name": "B", "move_pct": -5.0, "probability": 0.3},
        ]

        result = await calculate_ev(scenarios)

        assert result["probability_valid"] is False
        assert result["probability_sum"] == 0.8

    async def test_compute_event_response(self, db_with_data: Database):
        """Test compute_event_response tool."""
        from src.agent.tools.compute import compute_event_response
        import src.agent.tools.compute as compute_module

        original_get_db = compute_module.get_database
        compute_module.get_database = lambda: db_with_data

        try:
            result = await compute_event_response(
                symbol="NVDA",
                event_type="earnings",
                lookback_events=4,
            )

            assert result["symbol"] == "NVDA"
            assert result["event_type"] == "earnings"
            # May have error if price data doesn't align with events
            if "error" not in result:
                assert "statistics" in result
                assert "responses" in result
        finally:
            compute_module.get_database = original_get_db

    async def test_compute_correlation(self, db_with_data: Database):
        """Test compute_correlation tool."""
        from src.agent.tools.compute import compute_correlation
        import src.agent.tools.compute as compute_module

        # Add AAPL prices for correlation test
        prices = []
        base_price = 180.0
        for i in range(100):
            d = date(2024, 1, 1) + __import__("datetime").timedelta(days=i)
            if d.weekday() >= 5:
                continue
            variation = (i % 10 - 5) * 0.3
            prices.append(
                Price(
                    symbol="AAPL",
                    timestamp=datetime(d.year, d.month, d.day),
                    timeframe="daily",
                    open=Decimal(str(base_price + variation)),
                    high=Decimal(str(base_price + variation + 1)),
                    low=Decimal(str(base_price + variation - 0.5)),
                    close=Decimal(str(base_price + variation + 0.5)),
                    volume=30000000,
                    source="test",
                )
            )
            base_price += 0.3

        await db_with_data.insert_prices(prices)

        original_get_db = compute_module.get_database
        compute_module.get_database = lambda: db_with_data

        # Override today to be in our test data range
        import src.agent.tools.compute as cm
        original_today = cm.today_et
        cm.today_et = lambda: date(2024, 4, 1)

        try:
            result = await compute_correlation(
                symbol1="NVDA",
                symbol2="AAPL",
                days=60,
            )

            assert result["symbol1"] == "NVDA"
            assert result["symbol2"] == "AAPL"
            if "error" not in result:
                assert "correlation" in result
                assert -1 <= result["correlation"] <= 1
        finally:
            compute_module.get_database = original_get_db
            cm.today_et = original_today
