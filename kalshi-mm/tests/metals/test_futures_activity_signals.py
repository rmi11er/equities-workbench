import pytest

from src.metals.futures_activity_signals import (
    FuturesActivitySignalConfig,
    FuturesActivitySignalError,
    select_active_trade_series,
)


def test_active_contract_selection_is_outcome_blind_and_volume_first():
    trades = [
        {
            "asset": "SI",
            "contract_symbol": "SIZ6",
            "source_event_ts_ns": 1,
            "price": "60.0",
            "size": 2,
        },
        {
            "asset": "SI",
            "contract_symbol": "SIZ6",
            "source_event_ts_ns": 2,
            "price": "60.1",
            "size": 2,
        },
        {
            "asset": "SI",
            "contract_symbol": "SIH7",
            "source_event_ts_ns": 3,
            "price": "61.0",
            "size": 3,
        },
    ]

    series, selected = select_active_trade_series(trades, "SI")

    assert series.symbol == "SIZ6"
    assert selected["volume"] == 4
    assert series.interval(1, 3) == (2, 4)


def test_activity_config_rejects_lookback_longer_than_reference():
    with pytest.raises(FuturesActivitySignalError, match="lookbacks"):
        FuturesActivitySignalConfig(
            baseline_study_id="a" * 64,
            volume_lookback_minutes=15,
            causal_reference_minutes=15,
        )
