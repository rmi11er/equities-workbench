import pytest

from src.metals.pyth import (
    PYTH_GENERIC_PROXY_SYMBOLS,
    PythAccessError,
    normalize_tradingview_history,
)


def test_normalize_pyth_history_preserves_decimal_text_and_marks_semantics_unconfirmed():
    rows = normalize_tradingview_history(
        {
            "s": "ok",
            "t": [100, 160],
            "o": ["66.50000", "66.51000"],
            "h": ["66.52000", "66.53000"],
            "l": ["66.49000", "66.50000"],
            "c": ["66.51560", "66.52010"],
            "v": ["0", "0"],
        },
        "Metal.Index.SILVER/USD",
        receive_ts_ns=999,
    )

    assert rows[0]["close"] == "66.51560"
    assert rows[0]["timestamp_semantics"] == "unconfirmed_tradingview_t"
    assert rows[1]["timestamp_seconds"] == 160


def test_normalize_pyth_history_rejects_misaligned_arrays():
    with pytest.raises(PythAccessError):
        normalize_tradingview_history(
            {"s": "ok", "t": [1], "o": [], "h": [], "l": [], "c": [], "v": []},
            "Metal.Index.SILVER/USD",
            receive_ts_ns=999,
        )


def test_generic_proxy_symbols_are_distinct_from_settlement_indices():
    assert PYTH_GENERIC_PROXY_SYMBOLS == {
        "silver": "Metal.XAG/USD",
        "gold": "Metal.XAU/USD",
    }
    assert all(".Index." not in value for value in PYTH_GENERIC_PROXY_SYMBOLS.values())
