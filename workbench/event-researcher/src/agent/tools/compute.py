"""Compute tools for analysis and calculations."""

import statistics
from datetime import date, datetime, timedelta
from typing import Any

from src.agent.tools.registry import registry
from src.data.database import get_database
from src.utils.datetime_utils import today_et


def _date_from_str(date_str: str) -> date:
    """Parse date string to date object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _calculate_return(prices: list[dict], start_idx: int, end_idx: int) -> float | None:
    """Calculate return between two price points."""
    if start_idx < 0 or end_idx >= len(prices) or start_idx >= end_idx:
        return None

    start_price = prices[start_idx]["close"]
    end_price = prices[end_idx]["close"]

    if start_price == 0:
        return None

    return ((end_price - start_price) / start_price) * 100


def _find_price_index_for_date(prices: list[dict], target_date: date) -> int | None:
    """Find the index of the price closest to the target date."""
    for i, p in enumerate(prices):
        p_date = p["date"]
        if isinstance(p_date, str):
            p_date = _date_from_str(p_date)
        if p_date >= target_date:
            return i
    return None


@registry.register(
    name="compute_event_response",
    description="""Calculate price response statistics around past events for a symbol.
Returns statistics like mean move, median, std dev, and individual event responses.
Useful for understanding how a stock typically moves around earnings or other events.""",
    parameters={
        "symbol": {
            "type": "string",
            "description": "Stock symbol",
        },
        "event_type": {
            "type": "string",
            "enum": ["earnings", "conference", "analyst_day"],
            "description": "Type of event to analyze",
        },
        "lookback_events": {
            "type": "integer",
            "description": "Number of past events to analyze (default: 8)",
        },
        "response_days": {
            "type": "integer",
            "description": "Number of days after event to measure response (default: 1)",
        },
    },
    required=["symbol", "event_type"],
)
async def compute_event_response(
    symbol: str,
    event_type: str,
    lookback_events: int = 8,
    response_days: int = 1,
) -> dict[str, Any]:
    """Compute price response statistics around events."""
    db = get_database()

    # Get historical events for the symbol
    events_df = await db.get_events(
        symbols=[symbol],
        event_types=[event_type],
        start_date=None,  # All history
        end_date=today_et() - timedelta(days=1),  # Exclude future/today
    )

    if len(events_df) == 0:
        return {
            "symbol": symbol,
            "event_type": event_type,
            "error": f"No historical {event_type} events found for {symbol}",
        }

    # Convert to list and sort by date descending (most recent first)
    events = []
    for row in events_df.iter_rows(named=True):
        events.append(
            {
                "event_id": row["event_id"],
                "event_date": row["event_date"],
                "title": row["title"],
            }
        )

    events.sort(key=lambda x: x["event_date"], reverse=True)
    events = events[:lookback_events]

    if not events:
        return {
            "symbol": symbol,
            "event_type": event_type,
            "error": "No events found in lookback period",
        }

    # Get price data covering all events
    oldest_event = min(e["event_date"] for e in events)
    newest_event = max(e["event_date"] for e in events)

    price_start = oldest_event - timedelta(days=10)
    price_end = newest_event + timedelta(days=response_days + 5)

    prices_df = await db.get_prices([symbol], price_start, price_end, "daily")

    if len(prices_df) == 0:
        return {
            "symbol": symbol,
            "event_type": event_type,
            "error": "No price data available for analysis",
        }

    # Convert prices to list
    prices = []
    for row in prices_df.iter_rows(named=True):
        timestamp = row["timestamp"]
        if hasattr(timestamp, "date"):
            p_date = timestamp.date()
        elif hasattr(timestamp, "strftime"):
            p_date = _date_from_str(timestamp.strftime("%Y-%m-%d"))
        else:
            p_date = _date_from_str(str(timestamp)[:10])

        prices.append(
            {
                "date": p_date,
                "close": float(row["close"]),
            }
        )

    # Calculate response for each event
    responses = []
    moves = []

    for event in events:
        event_date = event["event_date"]

        # Find price index at/after event date
        event_idx = _find_price_index_for_date(prices, event_date)
        if event_idx is None:
            continue

        # Find close before event (day before)
        pre_idx = event_idx - 1 if event_idx > 0 else None

        # Find close after response_days
        post_idx = event_idx + response_days if event_idx + response_days < len(prices) else None

        if pre_idx is None or post_idx is None:
            continue

        # Calculate 1-day move (close to close)
        move = _calculate_return(prices, pre_idx, post_idx)

        if move is not None:
            moves.append(move)
            responses.append(
                {
                    "event_date": str(event_date),
                    "title": event["title"],
                    "move_pct": round(move, 2),
                    "pre_close": prices[pre_idx]["close"],
                    "post_close": prices[post_idx]["close"],
                }
            )

    if not moves:
        return {
            "symbol": symbol,
            "event_type": event_type,
            "error": "Could not calculate responses - insufficient price data",
        }

    # Calculate statistics
    stats = {
        "mean": round(statistics.mean(moves), 2),
        "median": round(statistics.median(moves), 2),
        "std_dev": round(statistics.stdev(moves), 2) if len(moves) > 1 else 0,
        "min": round(min(moves), 2),
        "max": round(max(moves), 2),
        "positive_rate": round(sum(1 for m in moves if m > 0) / len(moves) * 100, 1),
        "avg_absolute_move": round(statistics.mean(abs(m) for m in moves), 2),
    }

    return {
        "symbol": symbol,
        "event_type": event_type,
        "n_events": len(responses),
        "response_days": response_days,
        "responses": responses,
        "statistics": stats,
    }


@registry.register(
    name="find_analogs",
    description="""Find historical events similar to a target scenario based on criteria.
Can match on same symbol history, momentum, sector performance, etc.
Returns similar past events with their outcomes.""",
    parameters={
        "symbol": {
            "type": "string",
            "description": "Target symbol",
        },
        "event_type": {
            "type": "string",
            "enum": ["earnings", "conference", "analyst_day"],
            "description": "Type of event",
        },
        "momentum_min": {
            "type": "number",
            "description": "Minimum stock momentum (% change over 90 days) to match",
        },
        "momentum_max": {
            "type": "number",
            "description": "Maximum stock momentum (% change over 90 days) to match",
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of analogs to return (default: 10)",
        },
    },
    required=["symbol", "event_type"],
)
async def find_analogs(
    symbol: str,
    event_type: str,
    momentum_min: float | None = None,
    momentum_max: float | None = None,
    max_results: int = 10,
) -> dict[str, Any]:
    """Find historical events similar to target criteria."""
    db = get_database()

    # Get all historical events for the symbol
    events_df = await db.get_events(
        symbols=[symbol],
        event_types=[event_type],
        start_date=None,
        end_date=today_et() - timedelta(days=5),
    )

    if len(events_df) == 0:
        return {
            "symbol": symbol,
            "error": f"No historical {event_type} events found",
        }

    # Get price data for calculating momentum
    oldest_possible = today_et() - timedelta(days=365 * 5)
    prices_df = await db.get_prices([symbol], oldest_possible, today_et(), "daily")

    if len(prices_df) == 0:
        return {"symbol": symbol, "error": "No price data for analysis"}

    # Build price lookup by date
    price_by_date = {}
    for row in prices_df.iter_rows(named=True):
        timestamp = row["timestamp"]
        if hasattr(timestamp, "date"):
            p_date = timestamp.date()
        else:
            p_date = _date_from_str(str(timestamp)[:10])
        price_by_date[p_date] = float(row["close"])

    # Find analogs matching criteria
    analogs = []

    for row in events_df.iter_rows(named=True):
        event_date = row["event_date"]

        # Calculate 90-day momentum before event
        momentum_start = event_date - timedelta(days=90)
        momentum_end = event_date - timedelta(days=1)

        start_price = None
        end_price = None

        # Find closest prices
        for d in range(5):
            check_date = momentum_start + timedelta(days=d)
            if check_date in price_by_date:
                start_price = price_by_date[check_date]
                break

        for d in range(5):
            check_date = momentum_end - timedelta(days=d)
            if check_date in price_by_date:
                end_price = price_by_date[check_date]
                break

        if start_price is None or end_price is None or start_price == 0:
            continue

        momentum = ((end_price - start_price) / start_price) * 100

        # Check momentum filters
        if momentum_min is not None and momentum < momentum_min:
            continue
        if momentum_max is not None and momentum > momentum_max:
            continue

        # Calculate post-event move
        post_date = event_date + timedelta(days=1)
        pre_price = None
        post_price = None

        for d in range(3):
            check_date = event_date - timedelta(days=d)
            if check_date in price_by_date:
                pre_price = price_by_date[check_date]
                break

        for d in range(5):
            check_date = post_date + timedelta(days=d)
            if check_date in price_by_date:
                post_price = price_by_date[check_date]
                break

        if pre_price is None or post_price is None or pre_price == 0:
            continue

        move_1d = ((post_price - pre_price) / pre_price) * 100

        analogs.append(
            {
                "event_date": str(event_date),
                "title": row["title"],
                "momentum_90d": round(momentum, 1),
                "move_1d": round(move_1d, 2),
                "matching_criteria": ["same_symbol"],
            }
        )

    # Sort by date descending
    analogs.sort(key=lambda x: x["event_date"], reverse=True)
    analogs = analogs[:max_results]

    # Calculate aggregate stats
    if analogs:
        moves = [a["move_1d"] for a in analogs]
        stats = {
            "mean_move": round(statistics.mean(moves), 2),
            "median_move": round(statistics.median(moves), 2),
            "positive_rate": round(sum(1 for m in moves if m > 0) / len(moves) * 100, 1),
        }
    else:
        stats = {}

    return {
        "symbol": symbol,
        "event_type": event_type,
        "criteria": {
            "momentum_min": momentum_min,
            "momentum_max": momentum_max,
        },
        "analogs": analogs,
        "count": len(analogs),
        "aggregate_stats": stats,
    }


@registry.register(
    name="calculate_ev",
    description="""Calculate expected value from a scenario distribution.
Each scenario has a move percentage and probability. Returns EV and variance.""",
    parameters={
        "scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "move_pct": {"type": "number"},
                    "probability": {"type": "number"},
                },
            },
            "description": "List of scenarios with name, move_pct, and probability (0-1)",
        },
    },
    required=["scenarios"],
)
async def calculate_ev(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate expected value from scenario distribution."""
    if not scenarios:
        return {"error": "No scenarios provided"}

    # Validate and calculate
    total_prob = sum(s.get("probability", 0) for s in scenarios)

    results = []
    ev = 0
    variance = 0

    for s in scenarios:
        name = s.get("name", "Unnamed")
        move = s.get("move_pct", 0)
        prob = s.get("probability", 0)

        contribution = move * prob
        ev += contribution

        results.append(
            {
                "name": name,
                "move_pct": move,
                "probability": prob,
                "contribution": round(contribution, 3),
            }
        )

    # Calculate variance
    for s in scenarios:
        move = s.get("move_pct", 0)
        prob = s.get("probability", 0)
        variance += prob * ((move - ev) ** 2)

    std_dev = variance ** 0.5

    return {
        "expected_value": round(ev, 3),
        "scenarios": results,
        "probability_sum": round(total_prob, 3),
        "probability_valid": abs(total_prob - 1.0) < 0.01,
        "variance": round(variance, 3),
        "std_dev": round(std_dev, 3),
    }


@registry.register(
    name="compute_correlation",
    description="""Compute correlation between two symbols over a time period.
Useful for understanding how correlated two assets are.""",
    parameters={
        "symbol1": {
            "type": "string",
            "description": "First symbol",
        },
        "symbol2": {
            "type": "string",
            "description": "Second symbol",
        },
        "days": {
            "type": "integer",
            "description": "Number of days to analyze (default: 90)",
        },
    },
    required=["symbol1", "symbol2"],
)
async def compute_correlation(
    symbol1: str,
    symbol2: str,
    days: int = 90,
) -> dict[str, Any]:
    """Compute correlation between two symbols."""
    db = get_database()

    end_date = today_et()
    start_date = end_date - timedelta(days=int(days * 1.5))

    # Get prices for both symbols
    df1 = await db.get_prices([symbol1], start_date, end_date, "daily")
    df2 = await db.get_prices([symbol2], start_date, end_date, "daily")

    if len(df1) < 10 or len(df2) < 10:
        return {
            "symbol1": symbol1,
            "symbol2": symbol2,
            "error": "Insufficient price data for correlation calculation",
        }

    # Build daily returns by date
    def get_returns(df):
        prices = []
        for row in df.iter_rows(named=True):
            timestamp = row["timestamp"]
            if hasattr(timestamp, "date"):
                p_date = timestamp.date()
            else:
                p_date = _date_from_str(str(timestamp)[:10])
            prices.append({"date": p_date, "close": float(row["close"])})

        returns = {}
        for i in range(1, len(prices)):
            if prices[i - 1]["close"] != 0:
                ret = (prices[i]["close"] - prices[i - 1]["close"]) / prices[i - 1]["close"]
                returns[prices[i]["date"]] = ret
        return returns

    returns1 = get_returns(df1)
    returns2 = get_returns(df2)

    # Find common dates
    common_dates = set(returns1.keys()) & set(returns2.keys())

    if len(common_dates) < 10:
        return {
            "symbol1": symbol1,
            "symbol2": symbol2,
            "error": "Insufficient overlapping data",
        }

    # Calculate correlation
    r1 = [returns1[d] for d in sorted(common_dates)]
    r2 = [returns2[d] for d in sorted(common_dates)]

    mean1 = statistics.mean(r1)
    mean2 = statistics.mean(r2)

    numerator = sum((r1[i] - mean1) * (r2[i] - mean2) for i in range(len(r1)))
    std1 = (sum((x - mean1) ** 2 for x in r1)) ** 0.5
    std2 = (sum((x - mean2) ** 2 for x in r2)) ** 0.5

    if std1 == 0 or std2 == 0:
        correlation = 0
    else:
        correlation = numerator / (std1 * std2)

    return {
        "symbol1": symbol1,
        "symbol2": symbol2,
        "correlation": round(correlation, 3),
        "observations": len(common_dates),
        "period_days": days,
    }


@registry.register(
    name="compute_conditional_stats",
    description="""Compute statistics for target symbol conditioned on base asset behavior.
Example: 'How does MSTR move when BTC is up >5%?'
Returns response statistics for the target when base meets the condition.""",
    parameters={
        "base_symbol": {
            "type": "string",
            "description": "Symbol to condition on (e.g., BTC, SPY)",
        },
        "target_symbol": {
            "type": "string",
            "description": "Symbol to measure response (e.g., MSTR, QQQ)",
        },
        "base_move_threshold": {
            "type": "number",
            "description": "Minimum absolute move % for base to trigger (e.g., 5 for 5%)",
        },
        "base_direction": {
            "type": "string",
            "enum": ["up", "down", "any"],
            "description": "Direction of base move to consider (default: any)",
        },
        "lookback_days": {
            "type": "integer",
            "description": "Days of history to analyze (default: 365)",
        },
    },
    required=["base_symbol", "target_symbol", "base_move_threshold"],
)
async def compute_conditional_stats(
    base_symbol: str,
    target_symbol: str,
    base_move_threshold: float,
    base_direction: str = "any",
    lookback_days: int = 365,
) -> dict[str, Any]:
    """Compute conditional response statistics."""
    db = get_database()

    end_date = today_et()
    start_date = end_date - timedelta(days=int(lookback_days * 1.5))

    # Get prices for both symbols
    base_df = await db.get_prices([base_symbol], start_date, end_date, "daily")
    target_df = await db.get_prices([target_symbol], start_date, end_date, "daily")

    if len(base_df) < 20 or len(target_df) < 20:
        return {
            "base_symbol": base_symbol,
            "target_symbol": target_symbol,
            "error": "Insufficient price data",
        }

    # Build daily returns by date
    def get_returns(df):
        prices = []
        for row in df.iter_rows(named=True):
            timestamp = row["timestamp"]
            if hasattr(timestamp, "date"):
                p_date = timestamp.date()
            else:
                p_date = _date_from_str(str(timestamp)[:10])
            prices.append({"date": p_date, "close": float(row["close"])})

        returns = {}
        for i in range(1, len(prices)):
            if prices[i - 1]["close"] != 0:
                ret = ((prices[i]["close"] - prices[i - 1]["close"]) / prices[i - 1]["close"]) * 100
                returns[prices[i]["date"]] = ret
        return returns

    base_returns = get_returns(base_df)
    target_returns = get_returns(target_df)

    # Find days where base met the threshold
    occurrences = []
    common_dates = set(base_returns.keys()) & set(target_returns.keys())

    for d in sorted(common_dates):
        base_move = base_returns[d]

        # Check if base meets condition
        meets_threshold = abs(base_move) >= base_move_threshold

        if base_direction == "up":
            meets_direction = base_move > 0
        elif base_direction == "down":
            meets_direction = base_move < 0
        else:
            meets_direction = True

        if meets_threshold and meets_direction:
            occurrences.append({
                "date": str(d),
                "base_move": round(base_move, 2),
                "target_move": round(target_returns[d], 2),
            })

    if not occurrences:
        return {
            "base_symbol": base_symbol,
            "target_symbol": target_symbol,
            "condition": {
                "threshold": base_move_threshold,
                "direction": base_direction,
            },
            "n_occurrences": 0,
            "error": "No occurrences found matching condition",
        }

    # Calculate statistics
    target_moves = [o["target_move"] for o in occurrences]
    base_moves = [o["base_move"] for o in occurrences]

    # Calculate beta (regression coefficient)
    mean_base = statistics.mean(base_moves)
    mean_target = statistics.mean(target_moves)

    numerator = sum((base_moves[i] - mean_base) * (target_moves[i] - mean_target) for i in range(len(base_moves)))
    denominator = sum((b - mean_base) ** 2 for b in base_moves)

    beta = numerator / denominator if denominator != 0 else 0

    stats = {
        "mean_response": round(statistics.mean(target_moves), 2),
        "median_response": round(statistics.median(target_moves), 2),
        "std_dev": round(statistics.stdev(target_moves), 2) if len(target_moves) > 1 else 0,
        "beta": round(beta, 2),
        "positive_rate": round(sum(1 for m in target_moves if m > 0) / len(target_moves) * 100, 1),
        "min_response": round(min(target_moves), 2),
        "max_response": round(max(target_moves), 2),
    }

    return {
        "base_symbol": base_symbol,
        "target_symbol": target_symbol,
        "condition": {
            "threshold": base_move_threshold,
            "direction": base_direction,
        },
        "n_occurrences": len(occurrences),
        "lookback_days": lookback_days,
        "occurrences": occurrences[-10:],  # Last 10 for brevity
        "statistics": stats,
    }


@registry.register(
    name="generate_chart",
    description="""Generate a terminal-based ASCII chart.
Supports price charts, distribution histograms, and bar charts.
Returns the chart as a string that can be displayed in the terminal.""",
    parameters={
        "chart_type": {
            "type": "string",
            "enum": ["price", "distribution", "bar"],
            "description": "Type of chart to generate",
        },
        "symbol": {
            "type": "string",
            "description": "Symbol for price chart (required for price type)",
        },
        "days": {
            "type": "integer",
            "description": "Days of history for price chart (default: 30)",
        },
        "data": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Data for distribution/bar chart",
        },
        "labels": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Labels for bar chart",
        },
        "title": {
            "type": "string",
            "description": "Chart title",
        },
    },
    required=["chart_type"],
)
async def generate_chart(
    chart_type: str,
    symbol: str | None = None,
    days: int = 30,
    data: list[float] | None = None,
    labels: list[str] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Generate a terminal chart using plotext."""
    import io
    import sys

    try:
        import plotext as plt
    except ImportError:
        return {"error": "plotext not installed"}

    # Capture plotext output
    plt.clear_figure()
    plt.clear_data()

    if chart_type == "price":
        if not symbol:
            return {"error": "Symbol required for price chart"}

        db = get_database()
        end_date = today_et()
        start_date = end_date - timedelta(days=int(days * 1.5))

        df = await db.get_prices([symbol], start_date, end_date, "daily")

        if len(df) == 0:
            return {"error": f"No price data for {symbol}"}

        # Extract dates and closes
        dates = []
        closes = []
        for row in df.tail(days).iter_rows(named=True):
            timestamp = row["timestamp"]
            if hasattr(timestamp, "strftime"):
                dates.append(timestamp.strftime("%m/%d"))
            else:
                dates.append(str(timestamp)[:5])
            closes.append(float(row["close"]))

        plt.plot(closes, marker="braille")
        plt.title(title or f"{symbol} Price ({days}d)")
        plt.xlabel("Date")
        plt.ylabel("Price")

        # Set x-axis labels (sparse to avoid crowding)
        if len(dates) > 10:
            step = len(dates) // 5
            xticks = list(range(0, len(dates), step))
            xlabels = [dates[i] for i in xticks]
            plt.xticks(xticks, xlabels)

    elif chart_type == "distribution":
        if not data:
            return {"error": "Data required for distribution chart"}

        plt.hist(data, bins=15)
        plt.title(title or "Distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

    elif chart_type == "bar":
        if not data:
            return {"error": "Data required for bar chart"}

        bar_labels = labels or [str(i) for i in range(len(data))]
        plt.bar(bar_labels, data)
        plt.title(title or "Bar Chart")

    else:
        return {"error": f"Unknown chart type: {chart_type}"}

    # Configure terminal output
    plt.plotsize(60, 15)
    plt.theme("clear")

    # Build the chart string
    chart_str = plt.build()

    return {
        "chart_type": chart_type,
        "chart": chart_str,
    }
