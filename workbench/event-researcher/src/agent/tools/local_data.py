"""Local data tools for querying the database."""

from datetime import date, datetime, timedelta
from typing import Any

from src.agent.tools.registry import registry
from src.data.database import get_database
from src.utils.datetime_utils import today_et


def _date_from_str(date_str: str) -> date:
    """Parse date string to date object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _format_price_data(df) -> list[dict[str, Any]]:
    """Format price DataFrame to list of dicts."""
    if len(df) == 0:
        return []

    records = []
    for row in df.iter_rows(named=True):
        timestamp = row["timestamp"]
        date_str = (
            timestamp.strftime("%Y-%m-%d")
            if hasattr(timestamp, "strftime")
            else str(timestamp)[:10]
        )
        records.append(
            {
                "date": date_str,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }
        )
    return records


@registry.register(
    name="query_prices",
    description="Retrieve historical price data for one or more symbols over a date range. Returns OHLCV data.",
    parameters={
        "symbols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of stock symbols (e.g., ['NVDA', 'AAPL'])",
        },
        "start_date": {
            "type": "string",
            "description": "Start date in YYYY-MM-DD format",
        },
        "end_date": {
            "type": "string",
            "description": "End date in YYYY-MM-DD format",
        },
        "timeframe": {
            "type": "string",
            "enum": ["daily", "hourly"],
            "description": "Price timeframe (default: daily)",
        },
    },
    required=["symbols", "start_date", "end_date"],
)
async def query_prices(
    symbols: list[str],
    start_date: str,
    end_date: str,
    timeframe: str = "daily",
) -> dict[str, Any]:
    """Query price data from the database."""
    db = get_database()

    start = _date_from_str(start_date)
    end = _date_from_str(end_date)

    df = await db.get_prices(symbols, start, end, timeframe)

    # Group by symbol
    result_by_symbol = {}
    symbols_found = set()

    for row in df.iter_rows(named=True):
        symbol = row["symbol"]
        symbols_found.add(symbol)

        if symbol not in result_by_symbol:
            result_by_symbol[symbol] = []

        timestamp = row["timestamp"]
        date_str = (
            timestamp.strftime("%Y-%m-%d")
            if hasattr(timestamp, "strftime")
            else str(timestamp)[:10]
        )

        result_by_symbol[symbol].append(
            {
                "date": date_str,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }
        )

    symbols_missing = [s for s in symbols if s not in symbols_found]

    return {
        "data": result_by_symbol,
        "symbols_found": list(symbols_found),
        "symbols_missing": symbols_missing,
        "count": len(df),
    }


@registry.register(
    name="query_events",
    description="Retrieve upcoming or historical events (earnings, conferences, macro) matching filters.",
    parameters={
        "symbols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Filter by symbols (optional, None = all)",
        },
        "event_types": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Filter by event type: 'earnings', 'conference', 'analyst_day', 'macro'",
        },
        "start_date": {
            "type": "string",
            "description": "Start date in YYYY-MM-DD format (optional)",
        },
        "end_date": {
            "type": "string",
            "description": "End date in YYYY-MM-DD format (optional)",
        },
        "include_past": {
            "type": "boolean",
            "description": "Include events before today (default: false)",
        },
    },
    required=[],
)
async def query_events(
    symbols: list[str] | None = None,
    event_types: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    include_past: bool = False,
) -> dict[str, Any]:
    """Query events from the database."""
    db = get_database()

    # Default to today if not including past
    if start_date:
        start = _date_from_str(start_date)
    elif not include_past:
        start = today_et()
    else:
        start = None

    end = _date_from_str(end_date) if end_date else None

    df = await db.get_events(
        symbols=symbols,
        event_types=event_types,
        start_date=start,
        end_date=end,
    )

    events = []
    for row in df.iter_rows(named=True):
        events.append(
            {
                "event_id": row["event_id"],
                "symbol": row["symbol"],
                "event_type": row["event_type"],
                "event_date": str(row["event_date"]),
                "event_time": row["event_time"],
                "title": row["title"],
                "description": row["description"],
            }
        )

    return {"events": events, "count": len(events)}


@registry.register(
    name="query_earnings",
    description="Retrieve earnings data with estimates, actuals, and surprise percentages.",
    parameters={
        "symbols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Filter by symbols (optional)",
        },
        "start_date": {
            "type": "string",
            "description": "Start date in YYYY-MM-DD format (optional)",
        },
        "end_date": {
            "type": "string",
            "description": "End date in YYYY-MM-DD format (optional)",
        },
        "with_actuals_only": {
            "type": "boolean",
            "description": "Only return earnings with reported actuals (default: false)",
        },
    },
    required=[],
)
async def query_earnings(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    with_actuals_only: bool = False,
) -> dict[str, Any]:
    """Query earnings data from the database."""
    db = get_database()

    start = _date_from_str(start_date) if start_date else None
    end = _date_from_str(end_date) if end_date else None

    df = await db.get_earnings(
        symbols=symbols,
        start_date=start,
        end_date=end,
        with_actuals_only=with_actuals_only,
    )

    earnings = []
    for row in df.iter_rows(named=True):
        earnings.append(
            {
                "symbol": row["symbol"],
                "event_date": str(row["event_date"]) if row.get("event_date") else None,
                "fiscal_quarter": row["fiscal_quarter"],
                "fiscal_year": row["fiscal_year"],
                "eps_estimate": float(row["eps_estimate"]) if row["eps_estimate"] else None,
                "eps_actual": float(row["eps_actual"]) if row["eps_actual"] else None,
                "eps_surprise_pct": (
                    float(row["eps_surprise_pct"]) if row["eps_surprise_pct"] else None
                ),
                "revenue_estimate": row["revenue_estimate"],
                "revenue_actual": row["revenue_actual"],
                "revenue_surprise_pct": (
                    float(row["revenue_surprise_pct"])
                    if row["revenue_surprise_pct"]
                    else None
                ),
                "guidance_direction": row["guidance_direction"],
            }
        )

    return {"earnings": earnings, "count": len(earnings)}


@registry.register(
    name="get_watchlist",
    description="Retrieve the current watchlist of tracked symbols with company info.",
    parameters={
        "sectors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Filter by sectors (optional)",
        },
    },
    required=[],
)
async def get_watchlist(sectors: list[str] | None = None) -> dict[str, Any]:
    """Get the watchlist from the database."""
    db = get_database()

    df = await db.get_watchlist(sectors=sectors)

    symbols = []
    for row in df.iter_rows(named=True):
        symbols.append(
            {
                "symbol": row["symbol"],
                "name": row["name"],
                "sector": row["sector"],
                "industry": row["industry"],
                "market_cap": row["market_cap"],
            }
        )

    return {"symbols": symbols, "count": len(symbols)}


@registry.register(
    name="get_price_change",
    description="Calculate price change for a symbol over a period. Useful for checking momentum.",
    parameters={
        "symbol": {
            "type": "string",
            "description": "Stock symbol",
        },
        "days": {
            "type": "integer",
            "description": "Number of trading days to look back",
        },
    },
    required=["symbol", "days"],
)
async def get_price_change(symbol: str, days: int) -> dict[str, Any]:
    """Calculate price change over a period."""
    db = get_database()

    end_date = today_et()
    start_date = end_date - timedelta(days=int(days * 1.5))  # Buffer for weekends

    df = await db.get_prices([symbol], start_date, end_date, "daily")

    if len(df) < 2:
        return {
            "symbol": symbol,
            "error": "Insufficient price data",
            "days_requested": days,
        }

    # Get first and last prices
    closes = df["close"].to_list()
    first_close = float(closes[0])
    last_close = float(closes[-1])
    change_pct = ((last_close - first_close) / first_close) * 100

    # Get actual number of trading days
    actual_days = len(df)

    return {
        "symbol": symbol,
        "start_price": first_close,
        "end_price": last_close,
        "change_pct": round(change_pct, 2),
        "trading_days": actual_days,
        "period_requested": days,
    }
