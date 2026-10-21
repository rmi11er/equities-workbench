from decimal import Decimal

from src.metals.combined_fast_feed_counterfactual import (
    _avoidance_overlap,
    _first_warning,
)
from src.metals.fast_feed_counterfactual import FillObservation


def _fill(timestamp):
    return FillObservation(
        logical_quote_id=str(timestamp),
        market_ticker="m",
        side="yes",
        placement_ts_ns=0,
        fill_ts_ns=timestamp,
        quantity=Decimal("1"),
        quote_price=Decimal("0.4"),
        frozen_charge=Decimal("0.04"),
        markout_1s=Decimal("-0.01"),
        markout_5s=Decimal("-0.02"),
    )


def test_first_warning_uses_earliest_available_feed():
    assert _first_warning(None, None) is None
    assert _first_warning(200, None) == 200
    assert _first_warning(200, 100) == 100


def test_avoidance_overlap_is_exhaustive():
    fills = [_fill(1_000_000_000) for _ in range(3)]
    result = _avoidance_overlap(
        fills,
        [100_000_000, 100_000_000, None],
        [100_000_000, None, 100_000_000],
        cancel_latency_ms=100,
    )

    assert result["pyth_avoided_events"] == 2
    assert result["si_avoided_events"] == 2
    assert result["both_avoided_events"] == 1
    assert result["union_avoided_events"] == 3
