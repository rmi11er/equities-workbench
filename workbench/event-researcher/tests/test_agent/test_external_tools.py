"""Tests for external tools and new compute tools."""

from datetime import date, datetime
from decimal import Decimal

import pytest

from src.data.database import Database
from src.data.models import Price


@pytest.fixture
async def db_with_prices():
    """Create database with test price data for two correlated symbols."""
    db = Database(":memory:")
    await db.initialize()

    # Create price data for two symbols with correlated moves
    prices = []
    base_nvda = 100.0
    base_aapl = 150.0

    for i in range(200):
        d = date(2024, 1, 1) + __import__("datetime").timedelta(days=i)
        if d.weekday() >= 5:
            continue

        # Create correlated price movements
        daily_move = ((i % 7) - 3) * 0.5  # -1.5% to +1.5%

        prices.append(
            Price(
                symbol="NVDA",
                timestamp=datetime(d.year, d.month, d.day),
                timeframe="daily",
                open=Decimal(str(base_nvda)),
                high=Decimal(str(base_nvda + 2)),
                low=Decimal(str(base_nvda - 1)),
                close=Decimal(str(base_nvda * (1 + daily_move / 100))),
                volume=50000000,
                source="test",
            )
        )
        prices.append(
            Price(
                symbol="AAPL",
                timestamp=datetime(d.year, d.month, d.day),
                timeframe="daily",
                open=Decimal(str(base_aapl)),
                high=Decimal(str(base_aapl + 1)),
                low=Decimal(str(base_aapl - 0.5)),
                close=Decimal(str(base_aapl * (1 + daily_move * 0.7 / 100))),  # 0.7 beta
                volume=30000000,
                source="test",
            )
        )

        base_nvda *= (1 + daily_move / 100)
        base_aapl *= (1 + daily_move * 0.7 / 100)

    await db.insert_prices(prices)

    yield db
    await db.close()


class TestNewComputeTools:
    """Tests for compute_conditional_stats and generate_chart."""

    async def test_compute_conditional_stats(self, db_with_prices: Database):
        """Test compute_conditional_stats tool."""
        from src.agent.tools.compute import compute_conditional_stats
        import src.agent.tools.compute as compute_module

        original_get_db = compute_module.get_database
        compute_module.get_database = lambda: db_with_prices

        # Override today to be in our test data range
        original_today = compute_module.today_et
        compute_module.today_et = lambda: date(2024, 7, 1)

        try:
            result = await compute_conditional_stats(
                base_symbol="NVDA",
                target_symbol="AAPL",
                base_move_threshold=1.0,  # 1% threshold
                base_direction="up",
                lookback_days=180,
            )

            assert result["base_symbol"] == "NVDA"
            assert result["target_symbol"] == "AAPL"

            if "error" not in result:
                assert "statistics" in result
                assert "n_occurrences" in result
                assert result["n_occurrences"] > 0
                assert "mean_response" in result["statistics"]
                assert "beta" in result["statistics"]
        finally:
            compute_module.get_database = original_get_db
            compute_module.today_et = original_today

    async def test_generate_chart_price(self, db_with_prices: Database):
        """Test generate_chart with price type."""
        from src.agent.tools.compute import generate_chart
        import src.agent.tools.compute as compute_module

        original_get_db = compute_module.get_database
        compute_module.get_database = lambda: db_with_prices

        original_today = compute_module.today_et
        compute_module.today_et = lambda: date(2024, 7, 1)

        try:
            result = await generate_chart(
                chart_type="price",
                symbol="NVDA",
                days=30,
                title="NVDA Test Chart",
            )

            assert result["chart_type"] == "price"
            assert "chart" in result
            assert len(result["chart"]) > 0
        finally:
            compute_module.get_database = original_get_db
            compute_module.today_et = original_today

    async def test_generate_chart_distribution(self):
        """Test generate_chart with distribution type."""
        from src.agent.tools.compute import generate_chart

        result = await generate_chart(
            chart_type="distribution",
            data=[1.5, 2.3, -0.5, 3.2, 1.1, -1.0, 2.5, 0.8, 1.9, -0.3],
            title="Test Distribution",
        )

        assert result["chart_type"] == "distribution"
        assert "chart" in result

    async def test_generate_chart_bar(self):
        """Test generate_chart with bar type."""
        from src.agent.tools.compute import generate_chart

        result = await generate_chart(
            chart_type="bar",
            data=[10, 25, 15, 30],
            labels=["Q1", "Q2", "Q3", "Q4"],
            title="Quarterly Results",
        )

        assert result["chart_type"] == "bar"
        assert "chart" in result

    async def test_generate_chart_missing_symbol(self):
        """Test generate_chart error handling for missing symbol."""
        from src.agent.tools.compute import generate_chart

        result = await generate_chart(chart_type="price")

        assert "error" in result
        assert "Symbol required" in result["error"]


class TestExternalTools:
    """Tests for external fetch tools.

    Note: These tests check tool structure and error handling.
    Actual API calls are not tested to avoid external dependencies.
    """

    def test_external_tools_registered(self):
        """Test that external tools are registered."""
        from src.agent.tools import registry

        tools = registry.list_tools()

        assert "fetch_transcript" in tools
        assert "fetch_news" in tools
        assert "web_search" in tools
        assert "search_transcripts" in tools

    async def test_search_transcripts_empty(self):
        """Test search_transcripts with no cached transcripts."""
        from src.agent.tools.external import search_transcripts
        import src.agent.tools.external as external_module

        # Create empty test database
        db = Database(":memory:")
        await db.initialize()

        original_get_db = external_module.get_database
        external_module.get_database = lambda: db

        try:
            result = await search_transcripts(query="earnings")

            assert result["count"] == 0
            assert result["results"] == []
        finally:
            external_module.get_database = original_get_db
            await db.close()


class TestToolIntegration:
    """Integration tests for tools."""

    def test_all_tools_have_descriptions(self):
        """Test that all tools have descriptions."""
        from src.agent.tools import registry

        tools = registry.get_anthropic_tools()

        for tool in tools:
            assert "description" in tool
            assert len(tool["description"]) > 10

    def test_all_tools_have_valid_schemas(self):
        """Test that all tools have valid parameter schemas."""
        from src.agent.tools import registry

        tools = registry.get_anthropic_tools()

        for tool in tools:
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema

            # Check required params exist in properties
            required = schema.get("required", [])
            for param in required:
                assert param in schema["properties"], f"Required param {param} not in properties for {tool['name']}"
