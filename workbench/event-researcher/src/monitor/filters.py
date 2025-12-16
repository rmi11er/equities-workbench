"""Event filtering utilities."""

from datetime import date, timedelta
from typing import Any

from src.monitor.surfacer import InterestTier, SurfacedEvent


def filter_by_symbol(
    events: list[SurfacedEvent],
    symbols: list[str],
) -> list[SurfacedEvent]:
    """Filter events to specific symbols."""
    symbol_set = set(s.upper() for s in symbols)
    return [e for e in events if e.symbol.upper() in symbol_set]


def filter_by_sector(
    events: list[SurfacedEvent],
    sectors: list[str],
) -> list[SurfacedEvent]:
    """Filter events to specific sectors."""
    sector_set = set(s.lower() for s in sectors)
    return [e for e in events if e.sector and e.sector.lower() in sector_set]


def filter_by_event_type(
    events: list[SurfacedEvent],
    event_types: list[str],
) -> list[SurfacedEvent]:
    """Filter events to specific types."""
    type_set = set(t.lower() for t in event_types)
    return [e for e in events if e.event_type.lower() in type_set]


def filter_by_date_range(
    events: list[SurfacedEvent],
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[SurfacedEvent]:
    """Filter events to a date range."""
    result = events

    if start_date:
        result = [e for e in result if e.event_date >= start_date]

    if end_date:
        result = [e for e in result if e.event_date <= end_date]

    return result


def filter_by_tier(
    events: list[SurfacedEvent],
    min_tier: InterestTier,
) -> list[SurfacedEvent]:
    """Filter events by minimum interest tier."""
    tier_order = {InterestTier.HIGH: 3, InterestTier.STANDARD: 2, InterestTier.LOW: 1}
    min_order = tier_order[min_tier]
    return [e for e in events if tier_order[e.tier] >= min_order]


def filter_by_flag(
    events: list[SurfacedEvent],
    flag_name: str,
) -> list[SurfacedEvent]:
    """Filter to events that have a specific flag."""
    return [e for e in events if any(f.name == flag_name for f in e.flags)]


def sort_by_interest(
    events: list[SurfacedEvent],
    descending: bool = True,
) -> list[SurfacedEvent]:
    """Sort events by interest score."""
    return sorted(events, key=lambda e: e.interest_score, reverse=descending)


def sort_by_date(
    events: list[SurfacedEvent],
    descending: bool = False,
) -> list[SurfacedEvent]:
    """Sort events by date."""
    return sorted(events, key=lambda e: e.event_date, reverse=descending)


def group_by_date(
    events: list[SurfacedEvent],
) -> dict[date, list[SurfacedEvent]]:
    """Group events by date."""
    grouped: dict[date, list[SurfacedEvent]] = {}

    for event in events:
        if event.event_date not in grouped:
            grouped[event.event_date] = []
        grouped[event.event_date].append(event)

    return grouped


def group_by_symbol(
    events: list[SurfacedEvent],
) -> dict[str, list[SurfacedEvent]]:
    """Group events by symbol."""
    grouped: dict[str, list[SurfacedEvent]] = {}

    for event in events:
        if event.symbol not in grouped:
            grouped[event.symbol] = []
        grouped[event.symbol].append(event)

    return grouped
