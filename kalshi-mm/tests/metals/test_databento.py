import asyncio
from dataclasses import dataclass
from enum import Enum

import pytest

from src.metals.databento import (
    DatabentoConfigurationError,
    DatabentoLiveCollector,
    DatabentoSubscription,
    DeferredDatabentoRecord,
    glbx_metals_subscriptions,
    normalize_databento_record,
    normalize_subscription_symbols,
)


@dataclass
class FakeHeader:
    length: int = 48
    rtype: int = 160
    publisher_id: int = 1
    instrument_id: int = 12345
    ts_event: int = 1_777_000_000_000_000_001


class Side(Enum):
    BID = "B"


class MboMsg:
    def __init__(self):
        self.hd = FakeHeader()
        self.order_id = 987654321
        self.price = 72_345_000_001
        self.size = 17
        self.action = "A"
        self.side = Side.BID
        self.flags = 128
        self.channel_id = 3
        self.ts_recv = 1_777_000_000_000_010_001
        self.ts_in_delta = 8_000
        self.ts_out = 1_777_000_000_000_020_001
        self.sequence = 456

    def to_dict(self):
        # Simulate a prettified representation. Native attributes must win so
        # exact fixed-point integers remain available.
        return {"price": "72.345000001", "vendor_extra": b"CME\x00"}


class TradeMsg:
    def __init__(self):
        self.hd = FakeHeader(rtype=0)
        self.price = -1_250_000_000
        self.size = 2
        self.action = "T"
        self.side = "A"
        self.depth = 0
        self.flags = 0
        self.ts_recv = 101
        self.ts_in_delta = 4
        self.sequence = 5


class FuturesInstrumentDefMsg:
    def __init__(self):
        self.hd = FakeHeader(rtype=19)
        self.ts_recv = 201
        self.raw_symbol = b"SIU6\x00"
        self.min_price_increment = 5_000_000
        self.display_factor = 1_000_000_000
        self.high_limit_price = 75_250_000_000
        self.unit_of_measure_qty = 5_000_000_000
        self.strike_price = (1 << 63) - 1
        self.lot_size = 1


class StatusMsg:
    def __init__(self):
        self.hd = FakeHeader(rtype=18)
        self.ts_recv = 301
        self.action = 1
        self.reason = 2
        self.trading_event = 3
        self.is_trading = 1
        self.is_quoting = 0
        self.is_short_sell_restricted = 0


class StatMsg:
    def __init__(self):
        self.hd = FakeHeader(rtype=24)
        self.ts_recv = 401
        self.ts_ref = 399
        self.price = 73_125_000_000
        self.quantity = 10
        self.sequence = 22
        self.ts_in_delta = 7
        self.stat_type = 3
        self.channel_id = 9
        self.update_action = 1
        self.stat_flags = 4


class SlotHeader:
    __slots__ = ("length", "rtype", "publisher_id", "instrument_id", "ts_event")

    def __init__(self):
        self.length = 80
        self.rtype = 22
        self.publisher_id = 1
        self.instrument_id = 987
        self.ts_event = 1_777_000_000_000_000_123


class SymbolMappingMsg:
    __slots__ = (
        "hd",
        "stype_in",
        "stype_in_symbol",
        "stype_out",
        "stype_out_symbol",
        "start_ts",
        "end_ts",
    )

    def __init__(self):
        self.hd = SlotHeader()
        self.stype_in = 2
        self.stype_in_symbol = b"SI.FUT\x00"
        self.stype_out = 0
        self.stype_out_symbol = b"SIU6\x00"
        self.start_ts = 1_777_000_000_000_000_000
        self.end_ts = 1_778_000_000_000_000_000


class SystemMsg:
    __slots__ = ("msg", "code")

    def __init__(self):
        self.msg = "heartbeat"
        self.code = 0


class ErrorMsg:
    __slots__ = ("err", "code", "is_last")

    def __init__(self):
        self.err = "symbol failed"
        self.code = 4
        self.is_last = True


def test_parent_and_raw_symbol_normalization_is_deterministic():
    assert normalize_subscription_symbols(
        ["si", "SIL.FUT", "sic", "SI"], "parent"
    ) == ("SI.FUT", "SIL.FUT", "SIC.FUT")
    assert normalize_subscription_symbols(
        ["siu6", "SIZ6"], "raw_symbol"
    ) == ("SIU6", "SIZ6")
    assert normalize_subscription_symbols([42, 42, 99], "instrument_id") == (
        42,
        99,
    )


def test_invalid_subscription_symbols_are_rejected():
    with pytest.raises(DatabentoConfigurationError):
        normalize_subscription_symbols([], "parent")
    with pytest.raises(DatabentoConfigurationError):
        normalize_subscription_symbols(["SI"], "not-a-stype")
    with pytest.raises(DatabentoConfigurationError):
        normalize_subscription_symbols([True], "instrument_id")


def test_standard_metals_subscriptions_cover_all_required_schemas():
    subscriptions = glbx_metals_subscriptions(["SI", "SIL", "SIC"])

    assert [subscription.schema for subscription in subscriptions] == [
        "mbo",
        "trades",
        "definition",
        "status",
        "statistics",
    ]
    assert subscriptions[0].symbols == ("SI.FUT", "SIL.FUT", "SIC.FUT")
    assert subscriptions[0].snapshot is True
    assert all(not item.snapshot for item in subscriptions[1:])


def test_raw_subscription_retains_contract_symbols_and_replay_start():
    subscription = DatabentoSubscription.for_raw_symbols(
        "trades", ["SIU6", "SIZ6"], start=123
    )

    assert subscription.subscribe_kwargs() == {
        "dataset": "GLBX.MDP3",
        "schema": "trades",
        "symbols": ["SIU6", "SIZ6"],
        "stype_in": "raw_symbol",
        "start": 123,
    }


def test_snapshot_validation_rejects_non_mbo_and_replay_combination():
    with pytest.raises(DatabentoConfigurationError):
        DatabentoSubscription("trades", ("SI",), snapshot=True)
    with pytest.raises(DatabentoConfigurationError):
        DatabentoSubscription("mbo", ("SI",), snapshot=True, start=0)


def test_mbo_normalization_preserves_timestamps_sequence_order_and_raw_price():
    result = normalize_databento_record(
        MboMsg(), local_receive_ts_ns=777, local_monotonic_ts_ns=888
    )

    assert result["source"] == "databento"
    assert result["dataset"] == "GLBX.MDP3"
    assert result["schema"] == "mbo"
    assert result["ts_event"] == 1_777_000_000_000_000_001
    assert result["ts_recv"] == 1_777_000_000_000_010_001
    assert result["ts_in_delta"] == 8_000
    assert result["ts_out"] == 1_777_000_000_000_020_001
    assert result["sequence"] == 456
    assert result["channel_id"] == 3
    assert result["order_id"] == 987654321
    assert result["price"] == "72.345000001"
    assert result["price_raw"] == 72_345_000_001
    assert result["side"] == "B"
    assert result["local_receive_ts_ns"] == 777
    assert result["local_monotonic_ts_ns"] == 888
    assert result["raw_record"]["price"] == 72_345_000_001
    assert result["raw_record"]["vendor_extra"] == "CME"


@pytest.mark.parametrize(
    "record,schema,expected",
    [
        (TradeMsg(), "trades", {"price": "-1.250000000", "sequence": 5}),
        (
            FuturesInstrumentDefMsg(),
            "definition",
            {
                "raw_symbol": "SIU6",
                "min_price_increment": "0.005000000",
                "display_factor": "1.000000000",
                "high_limit_price": "75.250000000",
                "unit_of_measure_qty": "5.000000000",
                "strike_price": None,
                "strike_price_raw": (1 << 63) - 1,
            },
        ),
        (StatusMsg(), "status", {"is_trading": 1, "trading_event": 3}),
        (
            StatMsg(),
            "statistics",
            {"price": "73.125000000", "channel_id": 9, "ts_ref": 399},
        ),
    ],
)
def test_supported_record_types_have_deterministic_schema_and_values(
    record, schema, expected
):
    first = normalize_databento_record(record)
    second = normalize_databento_record(record)

    assert first == second
    assert first["schema"] == schema
    for key, value in expected.items():
        assert first[key] == value


def test_slot_based_symbol_mapping_and_control_records_are_preserved():
    mapping = normalize_databento_record(SymbolMappingMsg())
    system = normalize_databento_record(SystemMsg())
    error = normalize_databento_record(ErrorMsg())

    assert mapping["schema"] == "symbol_mapping"
    assert mapping["instrument_id"] == 987
    assert mapping["stype_in_symbol"] == "SI.FUT"
    assert mapping["stype_out_symbol"] == "SIU6"
    assert mapping["start_ts"] == 1_777_000_000_000_000_000
    assert mapping["raw_record"]["end_ts"] == 1_778_000_000_000_000_000
    assert system["schema"] == "system"
    assert system["msg"] == "heartbeat"
    assert error["schema"] == "error"
    assert error["err"] == "symbol failed"


class FakeLiveClient:
    def __init__(self, records=()):
        self.records = list(records)
        self.subscriptions = []
        self.callback = None
        self.exception_callback = None
        self.reconnect_callback = None
        self.stream = None
        self.stream_exception_callback = None
        self.started = False
        self.stopped = False
        self.terminated = False
        self.block_timeout = "unset"
        self.waited = False

    def subscribe(self, **kwargs):
        self.subscriptions.append(kwargs)
        return len(self.subscriptions)

    def __aiter__(self):
        async def iterator():
            for record in self.records:
                yield record

        return iterator()

    def add_callback(self, record_callback, exception_callback=None):
        self.callback = record_callback
        self.exception_callback = exception_callback

    def add_reconnect_callback(self, reconnect_callback, exception_callback=None):
        self.reconnect_callback = reconnect_callback

    def add_stream(self, stream, exception_callback=None):
        self.stream = stream
        self.stream_exception_callback = exception_callback

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def terminate(self):
        self.terminated = True

    def block_for_close(self, timeout=None):
        self.block_timeout = timeout

    async def wait_for_close(self):
        self.waited = True


def test_client_factory_receives_key_without_exposing_it_in_repr():
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeLiveClient()

    key = "super-secret-test-key"
    collector = DatabentoLiveCollector(
        key,
        [DatabentoSubscription.for_product_roots("mbo", ["SI"])],
        client_factory=factory,
    )

    first_ids = collector.subscribe()
    second_ids = collector.subscribe()

    assert captured == {
        "key": key,
        "reconnect_policy": "reconnect",
        "ts_out": True,
    }
    assert key not in repr(collector)
    assert first_ids == second_ids == (1,)
    assert len(collector.client.subscriptions) == 1
    assert collector.client.subscriptions[0]["symbols"] == ["SI.FUT"]
    assert "snapshot" not in collector.client.subscriptions[0]


def test_mapping_record_schema_can_be_inferred_from_rtype():
    result = normalize_databento_record(
        {
            "rtype": 24,
            "price": 72_000_000_000,
            "sequence": 12,
            "channel_id": 4,
        }
    )

    assert result["schema"] == "statistics"
    assert result["price"] == "72.000000000"
    assert result["price_raw"] == 72_000_000_000


@pytest.mark.asyncio
async def test_async_collection_supports_awaitable_callback_and_max_records():
    client = FakeLiveClient([MboMsg(), TradeMsg(), StatMsg()])
    collector = DatabentoLiveCollector(
        "test-key",
        [DatabentoSubscription.for_product_roots("mbo", ["SI"])],
        client=client,
    )
    received = []

    async def callback(record):
        await asyncio.sleep(0)
        received.append(record)

    count = await collector.collect(callback, max_records=2)

    assert count == 2
    assert [record["schema"] for record in received] == ["mbo", "trades"]
    assert client.stopped is True
    assert len(client.subscriptions) == 1


def test_callback_collection_normalizes_before_dispatch_and_controls_session():
    client = FakeLiveClient()
    collector = DatabentoLiveCollector(
        "test-key",
        [DatabentoSubscription.for_product_roots("statistics", ["GC", "MGC"])],
        client=client,
    )
    received = []

    collector.start_callbacks(received.append)
    client.callback(StatMsg())
    collector.block_for_close(timeout=2.5)
    collector.terminate()

    assert client.started is True
    assert received[0]["schema"] == "statistics"
    assert received[0]["price"] == "73.125000000"
    assert client.block_timeout == 2.5
    assert client.terminated is True


def test_native_stream_and_reconnect_gap_are_delegated_and_normalized():
    client = FakeLiveClient()
    collector = DatabentoLiveCollector(
        "test-key",
        [DatabentoSubscription.for_product_roots("mbo", ["SI"])],
        client=client,
    )
    reconnects = []
    stream = object()

    collector.add_stream(stream)
    collector.start_callbacks(lambda record: None, reconnect_callback=reconnects.append)
    client.reconnect_callback(101, 202)

    assert client.stream is stream
    assert reconnects[0]["schema"] == "reconnect"
    assert reconnects[0]["start_ts_ns"] == 101
    assert reconnects[0]["end_ts_ns"] == 202


def test_callback_can_defer_record_normalization_off_network_loop():
    client = FakeLiveClient()
    collector = DatabentoLiveCollector(
        "test-key",
        [DatabentoSubscription.for_product_roots("mbo", ["SI"])],
        client=client,
    )
    received = []

    collector.start_callbacks(received.append, defer_normalization=True)
    client.callback(MboMsg())

    deferred = received[0]
    assert isinstance(deferred, DeferredDatabentoRecord)
    normalized = collector.normalize_record(
        deferred.record,
        deferred.local_receive_ts_ns,
        deferred.local_monotonic_ts_ns,
    )
    assert normalized["schema"] == "mbo"
    assert normalized["order_id"] == 987654321


@pytest.mark.asyncio
async def test_wait_for_close_delegates_to_live_client():
    client = FakeLiveClient()
    collector = DatabentoLiveCollector(
        "test-key",
        [DatabentoSubscription.for_product_roots("trades", ["SI"])],
        client=client,
    )

    await collector.wait_for_close()

    assert client.waited is True
