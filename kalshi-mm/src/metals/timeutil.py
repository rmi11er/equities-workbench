"""UTC nanosecond parsing helpers with no local-time fallbacks."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import re
import time
from typing import Any, Optional


def iso8601_to_ns(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        # Infer common epoch units without silently converting small identifiers.
        if value > 10**17:  # nanoseconds
            return value
        if value > 10**14:  # microseconds
            return value * 1_000
        if value > 10**11:  # milliseconds
            return value * 1_000_000
        if value > 10**8:  # seconds
            return value * 1_000_000_000
        return None
    if isinstance(value, float):
        # A float is already limited by its source precision, but Decimal(str)
        # avoids adding another binary rounding step during unit conversion.
        numeric = Decimal(str(value))
        if numeric > Decimal("1e17"):
            return int(numeric)
        if numeric > Decimal("1e14"):
            return int(numeric * 1_000)
        if numeric > Decimal("1e11"):
            return int(numeric * 1_000_000)
        if numeric > Decimal("1e8"):
            return int(numeric * 1_000_000_000)
        return None

    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # macOS system Python 3.9 accepts only selected fractional-second widths.
    # Normalize arbitrary source precision to microseconds for parsing; the
    # original timestamp remains preserved in raw storage.
    fractional = re.search(r"\.(\d+)(?=[+-]\d{2}:\d{2}$)", text)
    if fractional:
        digits = (fractional.group(1) + "000000")[:6]
        text = (
            text[: fractional.start()]
            + "."
            + digits
            + text[fractional.end() :]
        )
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include a timezone: {}".format(value))
    utc_value = parsed.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = utc_value - epoch
    return (
        delta.days * 86_400 * 1_000_000_000
        + delta.seconds * 1_000_000_000
        + delta.microseconds * 1_000
    )


def utc_now_ns() -> int:
    return time.time_ns()
