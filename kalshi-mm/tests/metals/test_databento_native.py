"""Compatibility checks against real databento-dbn extension record classes."""

import pytest

from src.metals.databento import normalize_databento_record


dbn = pytest.importorskip("databento_dbn")


def test_native_mbo_and_symbol_mapping_records_preserve_required_fields():
    mbo = dbn.MBOMsg(
        publisher_id=1,
        instrument_id=2,
        ts_event=3,
        order_id=4,
        price=5_000_000_001,
        size=6,
        action=dbn.Action.ADD,
        side=dbn.Side.BID,
        ts_recv=7,
        sequence=8,
    )
    mapping = dbn.SymbolMappingMsg(
        publisher_id=1,
        instrument_id=2,
        ts_event=3,
        stype_in=dbn.SType.PARENT,
        stype_in_symbol="SI.FUT",
        stype_out=dbn.SType.RAW_SYMBOL,
        stype_out_symbol="SIU6",
        start_ts=4,
        end_ts=5,
    )

    normalized_mbo = normalize_databento_record(mbo)
    normalized_mapping = normalize_databento_record(mapping)

    assert normalized_mbo["schema"] == "mbo"
    assert normalized_mbo["price"] == "5.000000001"
    assert normalized_mbo["price_raw"] == 5_000_000_001
    assert normalized_mbo["order_id"] == 4
    assert normalized_mbo["sequence"] == 8
    assert normalized_mapping["schema"] == "symbol_mapping"
    assert normalized_mapping["instrument_id"] == 2
    assert normalized_mapping["stype_in_symbol"] == "SI.FUT"
    assert normalized_mapping["stype_out_symbol"] == "SIU6"
    assert normalized_mapping["start_ts"] == 4
    assert normalized_mapping["end_ts"] == 5


def test_native_system_and_error_records_preserve_gateway_state():
    system = dbn.SystemMsg(
        ts_event=3,
        msg="heartbeat",
        code=dbn.SystemCode.HEARTBEAT,
    )
    error = dbn.ErrorMsg(
        ts_event=4,
        err="symbol failed",
        code=dbn.ErrorCode.SYMBOL_RESOLUTION_FAILED,
    )

    normalized_system = normalize_databento_record(system)
    normalized_error = normalize_databento_record(error)

    assert normalized_system["schema"] == "system"
    assert normalized_system["code"] == "heartbeat"
    assert normalized_error["schema"] == "error"
    assert normalized_error["code"] == "symbol_resolution_failed"
    assert normalized_error["err"] == "symbol failed"
