import asyncio
import json

import pytest

from src.metals.config import load_metals_config
from src.metals.ingestion import GlobalReceiveSequencer
from src.metals.kalshi import KalshiWebSocketReader
from src.metals.sequence import SequenceTracker


class _Auth:
    def get_auth_headers(self, method, path):
        assert method == "GET"
        assert path == "/trade-api/ws/v2"
        return {"test": "header"}


class _Socket:
    def __init__(self):
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def recv(self):
        return json.dumps(
            {
                "type": "pyth_value",
                "sid": 2,
                "seq": 1,
                "msg": {
                    "underlying_ticker": "Metal.Index.SILVER/USD",
                    "value": "30.0",
                    "ts": 1,
                },
            }
        )


class _BlockingSocket(_Socket):
    def __init__(self):
        super().__init__()
        self.recv_started = asyncio.Event()
        self.recv_cancelled = False

    async def recv(self):
        self.recv_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.recv_cancelled = True
            raise


@pytest.mark.asyncio
async def test_pyth_only_connection_omits_empty_market_subscriptions(monkeypatch):
    config = load_metals_config("configs/metals.example.toml")
    socket = _Socket()
    monkeypatch.setattr(
        "src.metals.kalshi.websockets.connect", lambda *args, **kwargs: socket
    )
    reader = object.__new__(KalshiWebSocketReader)
    reader.config = config.kalshi
    reader._auth = _Auth()
    reader.sequencer = GlobalReceiveSequencer()
    reader.websocket_max_queue = 32
    reader._sequence = SequenceTracker()

    events = reader.events(
        [],
        pyth_underlyings=["Metal.Index.SILVER/USD"],
        stop_event=asyncio.Event(),
    )
    envelope = await anext(events)
    await events.aclose()

    assert envelope.stream == "pyth_value"
    channels = [message["params"]["channels"] for message in socket.sent]
    assert channels == [["market_lifecycle_v2"], ["pyth_value"]]
    assert all("market_tickers" not in message["params"] for message in socket.sent)


@pytest.mark.asyncio
async def test_websocket_reader_still_rejects_empty_scope(monkeypatch):
    reader = object.__new__(KalshiWebSocketReader)
    with pytest.raises(ValueError, match="ticker or Pyth"):
        await anext(reader.events([], pyth_underlyings=[]))


@pytest.mark.asyncio
async def test_cancelling_event_stream_retrieves_inflight_recv(monkeypatch):
    config = load_metals_config("configs/metals.example.toml")
    socket = _BlockingSocket()
    monkeypatch.setattr(
        "src.metals.kalshi.websockets.connect", lambda *args, **kwargs: socket
    )
    reader = object.__new__(KalshiWebSocketReader)
    reader.config = config.kalshi
    reader._auth = _Auth()
    reader.sequencer = GlobalReceiveSequencer()
    reader.websocket_max_queue = 32
    reader._sequence = SequenceTracker()

    events = reader.events(
        [],
        pyth_underlyings=["Metal.Index.SILVER/USD"],
        stop_event=asyncio.Event(),
    )
    pending = asyncio.create_task(anext(events))
    await socket.recv_started.wait()
    pending.cancel()
    await asyncio.gather(pending, return_exceptions=True)

    assert socket.recv_cancelled is True
