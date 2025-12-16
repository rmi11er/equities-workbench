"""Date and time utilities for market operations."""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def now_et() -> datetime:
    """Get current time in Eastern Time."""
    return datetime.now(ET)


def today_et() -> date:
    """Get today's date in Eastern Time."""
    return now_et().date()


def to_et(dt: datetime) -> datetime:
    """Convert a datetime to Eastern Time."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(ET)


def market_open_time() -> time:
    """Market open time (9:30 AM ET)."""
    return time(9, 30)


def market_close_time() -> time:
    """Market close time (4:00 PM ET)."""
    return time(16, 0)


def is_market_open(dt: datetime | None = None) -> bool:
    """Check if the market is currently open."""
    if dt is None:
        dt = now_et()
    else:
        dt = to_et(dt)

    # Check if it's a weekday
    if dt.weekday() >= 5:
        return False

    current_time = dt.time()
    return market_open_time() <= current_time < market_close_time()


def get_previous_trading_day(from_date: date | None = None) -> date:
    """Get the most recent trading day before the given date."""
    if from_date is None:
        from_date = today_et()

    current = from_date - timedelta(days=1)

    # Skip weekends
    while current.weekday() >= 5:
        current -= timedelta(days=1)

    return current


def get_trading_days_back(n: int, from_date: date | None = None) -> list[date]:
    """Get the last N trading days."""
    if from_date is None:
        from_date = today_et()

    days = []
    current = from_date

    while len(days) < n:
        if current.weekday() < 5:
            days.append(current)
        current -= timedelta(days=1)

    return days


def parse_event_time(event_time: str | None) -> str:
    """Parse event time string to standardized format.

    Accepts: 'BMO', 'AMC', 'DMH', or specific time strings.
    Returns standardized string.
    """
    if event_time is None:
        return "TBD"

    event_time = event_time.upper().strip()

    if event_time in ("BMO", "BEFORE MARKET OPEN", "PRE-MARKET"):
        return "BMO"
    elif event_time in ("AMC", "AFTER MARKET CLOSE", "AFTER-HOURS"):
        return "AMC"
    elif event_time in ("DMH", "DURING MARKET HOURS"):
        return "DMH"
    else:
        return event_time


def days_until(target_date: date, from_date: date | None = None) -> int:
    """Calculate days until a target date."""
    if from_date is None:
        from_date = today_et()
    return (target_date - from_date).days
