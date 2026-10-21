import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.metals.config import load_metals_config
from src.metals.ingestion import GlobalReceiveSequencer
from src.metals.provider_approval import ProviderApprovalError
from src.metals.pyth_lazer import (
    PYTH_LAZER_CHANNEL,
    PYTH_LAZER_MAX_QUEUE,
    PythLazerAccessError,
    PythLazerCollector,
    PythLazerProtocolError,
    build_pyth_lazer_subscription,
    normalize_pyth_lazer_update,
    parse_pyth_lazer_message,
)
from tests.metals.provider_approval_fixtures import (
    materialize_approved_provider_path,
)


def _update(
    *,
    timestamp_us="1788000000200000",
    feed_update_timestamp_us=1788000000000000,
):
    return {
        "type": "streamUpdated",
        "subscriptionId": 1,
        "parsed": {
            "timestampUs": timestamp_us,
            "priceFeeds": [
                {
                    "priceFeedId": 3154,
                    "price": "6651560",
                    "feedUpdateTimestamp": feed_update_timestamp_us,
                    "exponent": -5,
                    "confidence": "2000",
                    "publisherCount": 7,
                    "marketSession": "regular",
                }
            ],
        },
    }


def _normalize(message):
    return normalize_pyth_lazer_update(
        message,
        expected_feed_id=3154,
        pyth_symbol="Metal.Index.SILVER/USD",
        metal="silver",
        endpoint="wss://pyth-lazer-0.example/v1/stream",
        connection_id="connection-1",
        receive_ts_ns=1788000000300000000,
        local_monotonic_ts_ns=123456,
        ingestion_sequence=9,
        active_endpoint="wss://pyth-lazer-0.example/v1/stream",
    )


def _config(tmp_path: Path, endpoints):
    config = load_metals_config("configs/metals.example.toml")
    storage_root = tmp_path / "metals"
    return replace(
        config,
        storage=replace(config.storage, root=str(storage_root)),
        governance=replace(
            config.governance,
            provider_approval_path=materialize_approved_provider_path(storage_root),
        ),
        pyth=replace(
            config.pyth,
            enabled=True,
            api_key_env="TEST_PYTH_LAZER_API_KEY",
            lazer_ws_endpoints=list(endpoints),
        ),
    )


def _jsonl_rows(root: Path, tier: str, source: str):
    rows = []
    for path in sorted((root / tier / source).glob("**/*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    return rows


class _FakeWebSocket:
    def __init__(self, messages, stop_event=None):
        self.messages = list(messages)
        self.stop_event = stop_event
        self.sent = []
        self._blocked = asyncio.Event()

    async def send(self, value):
        self.sent.append(value)

    async def recv(self):
        if self.messages:
            value = self.messages.pop(0)
            if isinstance(value, BaseException):
                raise value
            if not self.messages and self.stop_event is not None:
                self.stop_event.set()
            return value
        await self._blocked.wait()


class _FakeConnection:
    def __init__(self, websocket=None, enter_error=None):
        self.websocket = websocket
        self.enter_error = enter_error

    async def __aenter__(self):
        if self.enter_error is not None:
            raise self.enter_error
        return self.websocket

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _ScriptedConnect:
    def __init__(self, scripts):
        self.scripts = {endpoint: list(values) for endpoint, values in scripts.items()}
        self.calls = []

    def __call__(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        return self.scripts[endpoint].pop(0)


def test_subscription_is_strict_json_offchain_and_read_only():
    subscription = build_pyth_lazer_subscription(3154)

    assert subscription == {
        "type": "subscribe",
        "subscriptionId": 1,
        "priceFeedIds": [3154],
        "properties": [
            "price",
            "feedUpdateTimestamp",
            "exponent",
            "confidence",
            "publisherCount",
            "marketSession",
        ],
        "formats": [],
        "channel": PYTH_LAZER_CHANNEL,
        "ignoreInvalidFeeds": False,
        "deliveryFormat": "json",
        "parsed": True,
    }
    assert not any(
        word in json.dumps(subscription).lower()
        for word in ("order", "trade", "execute")
    )


def test_normalize_preserves_mantissas_timestamps_and_carried_forward_semantics():
    row = _normalize(_update())[0]

    assert row["price_mantissa"] == "6651560"
    assert row["price"] == "66.51560"
    assert row["confidence_mantissa"] == "2000"
    assert row["confidence"] == "0.02000"
    assert row["source_event_ts_ns"] == 1788000000200000000
    assert row["source_publish_ts_ns"] == 1788000000000000000
    assert row["is_carried_forward"] is True
    assert row["receive_ts_ns"] == 1788000000300000000
    assert row["local_monotonic_ts_ns"] == 123456
    assert row["endpoint_role"] == "active"
    assert row["ingestion_sequence"] == 9


@pytest.mark.parametrize(
    "mutation",
    [
        lambda message: message.update(type="newUnknownType"),
        lambda message: message["parsed"]["priceFeeds"][0].update(price=66.5),
        lambda message: message["parsed"]["priceFeeds"][0].update(
            marketSession="auction"
        ),
        lambda message: message["parsed"]["priceFeeds"][0].update(priceFeedId=9999),
        lambda message: message["parsed"]["priceFeeds"][0].update(
            feedUpdateTimestamp=1788000000300000
        ),
    ],
)
def test_unknown_or_malformed_semantics_fail_closed(mutation):
    message = _update()
    mutation(message)

    with pytest.raises(PythLazerProtocolError):
        if message["type"] == "streamUpdated":
            _normalize(message)
        else:
            parse_pyth_lazer_message(json.dumps(message))


def test_binary_delivery_fails_closed():
    with pytest.raises(PythLazerProtocolError, match="binary"):
        parse_pyth_lazer_message(b"not accepted")


def test_disabled_live_collection_fails_before_key_storage_or_connection(tmp_path):
    endpoint = "wss://pyth-lazer-0.example/v1/stream"
    config = _config(tmp_path, [endpoint])
    config = replace(config, pyth=replace(config.pyth, enabled=False))
    connect = _ScriptedConnect({endpoint: []})

    with pytest.raises(PythLazerAccessError, match=r"\[pyth\]\.enabled"):
        PythLazerCollector(
            config,
            "silver",
            connect_factory=connect,
        )

    assert connect.calls == []
    assert not (tmp_path / "metals" / "raw").exists()
    assert not (tmp_path / "metals" / "normalized").exists()
    assert not (tmp_path / "metals" / "manifests").exists()


def test_live_collection_requires_frozen_provider_approval_before_connection(tmp_path):
    endpoint = "wss://pyth-lazer-0.example/v1/stream"
    config = _config(tmp_path, [endpoint])
    config = replace(
        config,
        governance=replace(config.governance, provider_approval_path=""),
    )
    connect = _ScriptedConnect({endpoint: []})

    with pytest.raises(ProviderApprovalError, match="approval path"):
        PythLazerCollector(config, "silver", connect_factory=connect)

    assert connect.calls == []
    assert not (tmp_path / "metals" / "raw").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_denied_or_expired_access_is_fatal_without_reconnect(tmp_path, monkeypatch, status):
    from types import SimpleNamespace
    from websockets.exceptions import InvalidStatus
    endpoint = "wss://pyth-lazer-0.example/v1/stream"
    denied = InvalidStatus(SimpleNamespace(status_code=status))
    connect = _ScriptedConnect({endpoint: [_FakeConnection(enter_error=denied)]})
    config = _config(tmp_path, [endpoint])
    monkeypatch.setenv(config.pyth.api_key_env, "test-credential-do-not-persist")
    collector = PythLazerCollector(config, "silver", connect_factory=connect)
    with pytest.raises(PythLazerAccessError, match=f"HTTP {status}"):
        await collector.run(duration_seconds=1)
    assert len(connect.calls) == 1
    assert collector._counts["reconnects"] == 0
    for path in (tmp_path / "metals").rglob("*.json*"):
        assert "test-credential-do-not-persist" not in path.read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf")])
async def test_collector_rejects_unbounded_duration_before_key_or_connection(
    tmp_path, duration
):
    endpoint = "wss://pyth-lazer-0.example/v1/stream"
    connect = _ScriptedConnect({endpoint: []})
    collector = PythLazerCollector(
        _config(tmp_path, [endpoint]),
        "silver",
        connect_factory=connect,
    )

    with pytest.raises(ValueError, match="finite and positive"):
        await collector.run(duration_seconds=duration)

    assert connect.calls == []
    assert not (tmp_path / "metals" / "raw").exists()
    assert not (tmp_path / "metals" / "normalized").exists()
    assert not (tmp_path / "metals" / "manifests").exists()


@pytest.mark.asyncio
async def test_collector_reconnects_records_exact_raw_and_never_persists_key(
    tmp_path, monkeypatch
):
    endpoint = "wss://pyth-lazer-0.example/v1/stream"
    ack_text = '{ "type": "subscribed", "subscriptionId": 1 }'
    update_text = json.dumps(_update(), separators=(",", ":"))
    websocket = _FakeWebSocket([ack_text, update_text])
    connect = _ScriptedConnect(
        {
            endpoint: [
                _FakeConnection(enter_error=OSError("temporary outage")),
                _FakeConnection(websocket=websocket),
            ]
        }
    )
    secret = "test-secret-that-must-never-be-written"
    monkeypatch.setenv("TEST_PYTH_LAZER_API_KEY", secret)
    collector = PythLazerCollector(
        _config(tmp_path, [endpoint]),
        "silver",
        connect_factory=connect,
        reconnect_base_seconds=0.001,
        reconnect_max_seconds=0.001,
        receive_timeout_seconds=1.0,
        sequencer=GlobalReceiveSequencer(start_sequence=100),
    )

    counts = await collector.run(duration_seconds=0.02)

    assert counts["reconnects"] == 1
    assert counts["raw_messages"] == 2
    assert counts["normalized_prices"] == 1
    assert len(connect.calls) == 2
    assert connect.calls[0][1]["additional_headers"] == {
        "Authorization": "Bearer {}".format(secret)
    }
    assert connect.calls[0][1]["max_queue"] == PYTH_LAZER_MAX_QUEUE
    assert isinstance(connect.calls[0][1]["max_queue"], int)
    assert connect.calls[0][1]["max_queue"] > 0
    sent_subscription = json.loads(websocket.sent[0])
    assert sent_subscription == build_pyth_lazer_subscription(3154)

    root = tmp_path / "metals"
    raw_rows = _jsonl_rows(root, "raw", "pyth_lazer")
    assert [row["raw_text"] for row in raw_rows] == [ack_text, update_text]
    normalized_rows = _jsonl_rows(root, "normalized", "pyth_lazer_price")
    assert normalized_rows[0]["price"] == "66.51560"
    assert normalized_rows[0]["ingestion_sequence"] == 104
    assert raw_rows[-1]["ingestion_sequence"] == 104
    assert raw_rows[-1]["receive_ts_ns"] == normalized_rows[0]["receive_ts_ns"]
    assert (
        raw_rows[-1]["local_monotonic_ts_ns"]
        == normalized_rows[0]["local_monotonic_ts_ns"]
    )
    gaps = _jsonl_rows(root, "normalized", "pyth_lazer_gap")
    assert "transport_connect_failure" in {row["gap_classification"] for row in gaps}
    assert "connection_recovered" in {row["gap_classification"] for row in gaps}
    assert all(row["missing_update_count"] is None for row in gaps)

    for path in root.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
    manifest = json.loads(
        next((root / "manifests").glob("run=*/manifest.json")).read_text()
    )
    assert manifest["status"] == "completed"
    assert manifest["config"]["collector"]["provider_approval_id"]
    approval_path = Path(collector.config.governance.provider_approval_path)
    assert {str(approval_path), str(approval_path.parent / "manifest.json")}.issubset(
        manifest["output_paths"]
    )


@pytest.mark.asyncio
async def test_collector_connects_every_configured_endpoint(tmp_path, monkeypatch):
    endpoints = [
        "wss://pyth-lazer-0.example/v1/stream",
        "wss://pyth-lazer-1.example/v1/stream",
        "wss://pyth-lazer-2.example/v1/stream",
    ]
    scripts = {
        endpoint: [
            _FakeConnection(
                websocket=_FakeWebSocket(
                    [json.dumps({"type": "subscribed", "subscriptionId": 1})]
                )
            )
        ]
        for endpoint in endpoints
    }
    connect = _ScriptedConnect(scripts)
    monkeypatch.setenv("TEST_PYTH_LAZER_API_KEY", "in-memory-only")
    collector = PythLazerCollector(
        _config(tmp_path, endpoints),
        "silver",
        connect_factory=connect,
        receive_timeout_seconds=1.0,
    )

    await collector.run(duration_seconds=0.02)

    assert {endpoint for endpoint, _ in connect.calls} == set(endpoints)


@pytest.mark.asyncio
async def test_protocol_violation_fails_run_and_manifest(tmp_path, monkeypatch):
    endpoint = "wss://pyth-lazer-0.example/v1/stream"
    unknown_text = json.dumps({"type": "futureUndocumentedMessage"})
    websocket = _FakeWebSocket(
        [
            json.dumps({"type": "subscribed", "subscriptionId": 1}),
            unknown_text,
        ]
    )
    connect = _ScriptedConnect({endpoint: [_FakeConnection(websocket=websocket)]})
    monkeypatch.setenv("TEST_PYTH_LAZER_API_KEY", "in-memory-only")
    collector = PythLazerCollector(
        _config(tmp_path, [endpoint]),
        "silver",
        connect_factory=connect,
        receive_timeout_seconds=1.0,
    )

    with pytest.raises(PythLazerProtocolError, match="unknown"):
        await collector.run(duration_seconds=1.0)

    manifest = json.loads(
        next(
            (tmp_path / "metals" / "manifests").glob("run=*/manifest.json")
        ).read_text()
    )
    assert manifest["status"] == "failed"
    assert manifest["errors"][0]["type"] == "PythLazerProtocolError"
    root = tmp_path / "metals"
    raw_rows = _jsonl_rows(root, "raw", "pyth_lazer")
    assert raw_rows[-1]["raw_text"] == unknown_text
    assert raw_rows[-1]["message_type"] == "protocol_error_unparsed"
    assert _jsonl_rows(root, "normalized", "pyth_lazer_price") == []
    gaps = _jsonl_rows(root, "normalized", "pyth_lazer_gap")
    assert gaps[-1]["gap_classification"] == "protocol_violation"
