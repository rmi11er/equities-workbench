"""Event monitoring and surfacing."""

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
from src.monitor.flags import (
    InterestFlag,
    calculate_all_flags,
    calculate_momentum_flag,
    calculate_sector_relative_flag,
    calculate_streak_flag,
    calculate_vix_flag,
)
from src.monitor.surfacer import EventSurfacer, InterestTier, SurfacedEvent

__all__ = [
    # Surfacer
    "EventSurfacer",
    "SurfacedEvent",
    "InterestTier",
    # Flags
    "InterestFlag",
    "calculate_all_flags",
    "calculate_momentum_flag",
    "calculate_sector_relative_flag",
    "calculate_vix_flag",
    "calculate_streak_flag",
    # Filters
    "filter_by_symbol",
    "filter_by_sector",
    "filter_by_event_type",
    "filter_by_date_range",
    "filter_by_tier",
    "filter_by_flag",
    "sort_by_interest",
    "sort_by_date",
    "group_by_date",
    "group_by_symbol",
]
