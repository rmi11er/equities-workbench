from __future__ import annotations

import pytest

from src.metals.high_frequency_signal_research import (
    _diagnostic,
    _gap_summary,
    _trade_events,
    reconstruct_kalshi_book,
)


def _snapshot(side: str, price: str, quantity: str, sequence: int) -> dict:
    return {
        "market_ticker": "TEST",
        "outcome_side": side,
        "message_type": "snapshot",
        "recovery_snapshot_id": "connection:1:1",
        "price_dollars": price,
        "quantity_fp": quantity,
        "receive_ts_ns": sequence,
        "book_valid_after": True,
    }


def test_reconstruct_kalshi_book_uses_complementary_no_bid_as_yes_ask() -> None:
    rows = [
        _snapshot("yes", "0.40", "10", 1),
        _snapshot("yes", "0.41", "20", 1),
        _snapshot("no", "0.57", "30", 1),
        {
            "market_ticker": "TEST",
            "outcome_side": "yes",
            "message_type": "delta",
            "price_dollars": "0.41",
            "quantity_delta_fp": "-20",
            "receive_ts_ns": 2,
            "book_valid_after": True,
        },
    ]

    states, pressure, stats = reconstruct_kalshi_book(rows)

    assert states[-2].yes_bid == pytest.approx(0.41)
    assert states[-2].yes_ask == pytest.approx(0.43)
    assert states[-2].midpoint == pytest.approx(0.42)
    assert states[-1].yes_bid == pytest.approx(0.40)
    assert pressure[0].value < 0
    assert stats["negative_level_repairs"] == 0


def test_trade_flow_orients_yes_buy_and_no_sell_positive() -> None:
    events = _trade_events(
        [
            {
                "market_ticker": "TEST",
                "receive_ts_ns": 1,
                "quantity_fp": "2",
                "taker_outcome_side": "yes",
                "taker_book_side": "ask",
            },
            {
                "market_ticker": "TEST",
                "receive_ts_ns": 2,
                "quantity_fp": "3",
                "taker_outcome_side": "no",
                "taker_book_side": "bid",
            },
        ]
    )

    assert [row.value for row in events] == [2.0, 3.0]


def test_diagnostic_reports_quintile_spread_and_window_stability() -> None:
    rows = [(float(index), float(index * 2), "A") for index in range(20)]
    result = _diagnostic(rows, target_unit="units")

    assert result["pearson"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["top_minus_bottom_quintile"] > 0
    assert result["same_sign_window_fraction"] == 1.0


def test_gap_summary_uses_milliseconds() -> None:
    summary = _gap_summary([0, 200_000_000, 400_000_000, 1_000_000_000])
    assert summary == {
        "intervals": 3,
        "p50_ms": 200.0,
        "p95_ms": 600.0,
        "max_ms": 600.0,
    }
