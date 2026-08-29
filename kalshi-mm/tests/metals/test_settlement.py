from decimal import Decimal

import pytest

from src.metals.settlement import (
    candle_end_ns,
    evaluate_settlement,
    round_price,
    select_candle_ending_at,
)
from src.metals.timeutil import iso8601_to_ns


def test_silver_rounding_and_tie_resolve_yes():
    check = evaluate_settlement(
        "KXSILVER15M-TEST",
        target_price=Decimal("66.516"),
        final_value=Decimal("66.5156"),
        round_digits=3,
        rounding_mode="half_up",
        exchange_result="yes",
    )

    assert check.rounded_final_value == "66.516"
    assert check.tie_flag is True
    assert check.predicted_result == "yes"
    assert check.agreement is True


def test_rounding_mode_is_explicit_not_python_default():
    assert round_price(Decimal("1.2345"), 3, "half_up") == Decimal("1.235")
    assert round_price(Decimal("1.2345"), 3, "half_even") == Decimal("1.234")


def test_candle_boundary_requires_exact_utc_minute_and_unique_match():
    boundary = 1_788_000_000_000_000_000
    assert candle_end_ns(boundary) == boundary
    candle = select_candle_ending_at(
        [{"end_ts_ns": boundary, "close": "66.516"}], boundary
    )
    assert candle["close"] == "66.516"

    with pytest.raises(ValueError):
        candle_end_ns(boundary + 1)
    with pytest.raises(ValueError):
        select_candle_ending_at([], boundary)


def test_iso_timestamp_accepts_source_fractional_precision_on_python39():
    parsed = iso8601_to_ns("2026-08-29T03:30:21.09711+00:00")
    assert parsed is not None
    assert parsed % 1_000_000_000 == 97_110_000


def test_integer_epoch_nanoseconds_remain_bit_exact():
    timestamp = 1_777_123_456_789_012_345

    assert iso8601_to_ns(timestamp) == timestamp
