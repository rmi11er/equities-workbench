"""Pure functions for target/final rounding and settlement checks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP
from typing import Iterable, Mapping, Optional

from .types import SettlementCheck


ROUNDING_MODES = {
    "half_even": ROUND_HALF_EVEN,
    "half_up": ROUND_HALF_UP,
    "truncate": ROUND_DOWN,
}


def round_price(value: Decimal, digits: int, mode: str) -> Decimal:
    if mode not in ROUNDING_MODES:
        raise ValueError("Unsupported rounding mode: {}".format(mode))
    quantum = Decimal(1).scaleb(-digits)
    return value.quantize(quantum, rounding=ROUNDING_MODES[mode])


def evaluate_settlement(
    market_ticker: str,
    target_price: Decimal,
    final_value: Decimal,
    round_digits: int,
    rounding_mode: str,
    exchange_result: Optional[str] = None,
) -> SettlementCheck:
    rounded_final = round_price(final_value, round_digits, rounding_mode)
    predicted = "yes" if rounded_final >= target_price else "no"
    normalized_exchange = exchange_result.lower() if exchange_result else None
    agreement = (
        predicted == normalized_exchange if normalized_exchange is not None else None
    )
    return SettlementCheck(
        market_ticker=market_ticker,
        target_price=format(target_price, "f"),
        raw_final_value=format(final_value, "f"),
        rounded_final_value=format(rounded_final, "f"),
        round_digits=round_digits,
        tie_flag=rounded_final == target_price,
        predicted_result=predicted,
        exchange_result=normalized_exchange,
        agreement=agreement,
        discrepancy_classification=None if agreement in (None, True) else "unresolved",
    )


def candle_end_ns(window_boundary_ns: int) -> int:
    """Return the exact requested one-minute candle end boundary."""

    minute_ns = 60 * 1_000_000_000
    if window_boundary_ns % minute_ns != 0:
        raise ValueError("Window boundary is not aligned to an exact UTC minute")
    return window_boundary_ns


def select_candle_ending_at(
    candles: Iterable[Mapping[str, object]], boundary_ns: int
) -> Mapping[str, object]:
    matches = []
    for candle in candles:
        end_value = candle.get("end_ts_ns", candle.get("end_time_ns"))
        if end_value is not None and int(end_value) == candle_end_ns(boundary_ns):
            matches.append(candle)
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one candle ending at {}, found {}".format(
                boundary_ns, len(matches)
            )
        )
    return matches[0]
