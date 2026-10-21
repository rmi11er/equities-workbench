from decimal import Decimal

import pytest

from src.metals.databento_mbo import (
    F_LAST,
    MboReconstructionError,
    reconstruct_mbo_bbo,
    select_active_outright_contract,
)


def _row(
    action,
    side="N",
    order_id=0,
    price_raw=(1 << 63) - 1,
    size=0,
    *,
    flags=F_LAST,
    sequence=1,
):
    return {
        "instrument_id": 7,
        "contract_symbol": "SIZ6",
        "action": action,
        "side": side,
        "order_id": order_id,
        "price_raw": price_raw,
        "size": size,
        "flags": flags,
        "sequence": sequence,
        "ts_event": 1_000 + sequence,
        "ts_recv": 2_000 + sequence,
        "source_publish_ts_ns": 1_900 + sequence,
    }


def test_reconstructs_bbo_only_after_last_and_applies_cancel_modify():
    rows = [
        _row("R", flags=0),
        _row("A", "B", 1, 66_000_000_000, 3, flags=0),
        _row("A", "A", 2, 66_020_000_000, 4, sequence=2),
        _row("A", "B", 3, 66_010_000_000, 2, sequence=3),
        _row("C", "B", 3, 66_010_000_000, 1, sequence=4),
        _row("M", "A", 2, 66_015_000_000, 5, sequence=5),
        _row("T", "B", 0, 66_015_000_000, 2, sequence=6),
    ]

    result = reconstruct_mbo_bbo(rows, instrument_id=7)

    assert result.emitted_events == 5
    assert len(result.snapshots) == 4
    assert result.snapshots[0].bid_price == Decimal("66.000000000")
    assert result.snapshots[0].ask_price == Decimal("66.020000000")
    assert result.snapshots[-1].bid_price == Decimal("66.010000000")
    assert result.snapshots[-1].bid_size == 1
    assert result.snapshots[-1].ask_price == Decimal("66.015000000")
    assert result.snapshots[-1].ask_size == 5
    assert result.snapshots[-1].microprice < result.snapshots[-1].midpoint


def test_unknown_cancel_fails_closed():
    with pytest.raises(MboReconstructionError, match="unknown"):
        reconstruct_mbo_bbo(
            [_row("C", "B", 99, 66_000_000_000, 1)], instrument_id=7
        )


def test_selects_highest_volume_outright_si_contract():
    rows = [
        {
            "asset": "SI",
            "instrument_class": "F",
            "instrument_id": 1,
            "contract_symbol": "SIU6",
            "size": 5,
        },
        {
            "asset": "SI",
            "instrument_class": "F",
            "instrument_id": 2,
            "contract_symbol": "SIZ6",
            "size": 8,
        },
        {
            "asset": "SI",
            "instrument_class": "S",
            "instrument_id": 3,
            "contract_symbol": "SIU6-SIZ6",
            "size": 100,
        },
    ]

    result = select_active_outright_contract(rows)

    assert result["instrument_id"] == 2
    assert result["contract_symbol"] == "SIZ6"
