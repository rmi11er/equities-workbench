"""Read-only Kalshi REST and WebSocket clients for research collection."""

from __future__ import annotations

import asyncio
import json
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp
import websockets

from src.auth import KalshiAuth

from .config import KalshiConfig, ResolvedKalshiCredentials
from .ingestion import GlobalReceiveSequencer, ReceiveStamp
from .sequence import SequenceTracker
from .timeutil import iso8601_to_ns
from .types import SequenceStatus, SourceEnvelope


REST_SIGNING_PREFIX = "/trade-api/v2"
WS_SIGNING_PATH = "/trade-api/ws/v2"


class ReadOnlyViolation(RuntimeError):
    """Raised before any non-read-only Kalshi operation can reach the network."""


class KalshiReadError(RuntimeError):
    def __init__(self, status: int, payload: Any):
        self.status = status
        self.payload = payload
        super().__init__("Kalshi read failed with HTTP {}: {}".format(status, payload))


@dataclass(frozen=True)
class ReceivedResponse:
    path: str
    params: Dict[str, Any]
    receive_ts_ns: int
    local_monotonic_ts_ns: int
    payload: Dict[str, Any]
    raw_body: str
    receive_stamp: Optional[ReceiveStamp] = None


class KalshiReadClient:
    """Authenticated Kalshi client whose public surface exposes GET only."""

    def __init__(
        self,
        config: KalshiConfig,
        credentials: Optional[ResolvedKalshiCredentials],
        sequencer: Optional[GlobalReceiveSequencer] = None,
    ) -> None:
        self.config = config
        self._auth = (
            KalshiAuth(
                api_key_id=credentials.api_key_id,
                private_key_path=credentials.private_key_path,
            )
            if credentials is not None
            else None
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self._pace_lock = asyncio.Lock()
        self._next_request_monotonic = 0.0
        self.sequencer = sequencer

    async def _pace_read(self) -> None:
        rate = self.config.max_read_requests_per_second
        if rate <= 0:
            return
        async with self._pace_lock:
            now = time.monotonic()
            if now < self._next_request_monotonic:
                await asyncio.sleep(self._next_request_monotonic - now)
                now = time.monotonic()
            self._next_request_monotonic = now + (1.0 / rate)

    async def start(self) -> None:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> ReceivedResponse:
        if method.upper() != "GET":
            raise ReadOnlyViolation(
                "The metals collector refuses non-GET Kalshi requests: {}".format(method)
            )
        if self._session is None:
            raise RuntimeError("KalshiReadClient.start() must be called first")
        if not path.startswith("/") or ".." in path:
            raise ValueError("Kalshi API path must be an absolute, normalized path")

        query = urllib.parse.urlencode(params or {}, doseq=True)
        target = path + ("?" + query if query else "")
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_read_retries + 1):
            await self._pace_read()
            headers = (
                self._auth.get_auth_headers("GET", REST_SIGNING_PREFIX + target)
                if self._auth is not None
                else {}
            )
            try:
                async with self._session.get(
                    self.config.rest_url + target, headers=headers
                ) as response:
                    content_type = response.headers.get("Content-Type", "")
                    raw_body = await response.text()
                    if "json" in content_type:
                        payload = json.loads(raw_body, parse_float=Decimal)
                    else:
                        payload = {"body": raw_body}
                    receive_ts_ns = time.time_ns()
                    receive_monotonic_ns = time.monotonic_ns()
                    receive_stamp = (
                        self.sequencer.stamp(
                            "kalshi",
                            "rest:{}".format(path),
                            receive_ts_ns=receive_ts_ns,
                            local_monotonic_ts_ns=receive_monotonic_ns,
                        )
                        if self.sequencer is not None
                        else None
                    )
                    retryable = response.status == 429 or response.status in (
                        500,
                        502,
                        503,
                        504,
                    )
                    if response.status >= 400:
                        error = KalshiReadError(response.status, payload)
                        if retryable and attempt < self.config.max_read_retries:
                            last_error = error
                        else:
                            raise error
                    else:
                        if not isinstance(payload, dict):
                            raise KalshiReadError(
                                response.status, {"unexpected_payload": payload}
                            )
                        return ReceivedResponse(
                            path=path,
                            params=dict(params or {}),
                            receive_ts_ns=receive_ts_ns,
                            local_monotonic_ts_ns=receive_monotonic_ns,
                            payload=payload,
                            raw_body=raw_body,
                            receive_stamp=receive_stamp,
                        )
            except aiohttp.ClientError as error:
                if attempt >= self.config.max_read_retries:
                    raise
                last_error = error
            await asyncio.sleep(
                self.config.read_retry_base_seconds * (2 ** attempt)
            )
        if last_error is not None:
            raise last_error
        raise RuntimeError("Kalshi read retry loop exited unexpectedly")

    async def get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> ReceivedResponse:
        return await self.request("GET", path, params=params)

    async def get_all_pages(
        self,
        path: str,
        result_key: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[ReceivedResponse]:
        responses = []
        cursor: Optional[str] = None
        while True:
            page_params = dict(params or {})
            page_params["limit"] = self.config.page_limit
            if cursor:
                page_params["cursor"] = cursor
            response = await self.get(path, page_params)
            responses.append(response)
            batch = response.payload.get(result_key, [])
            cursor = response.payload.get("cursor")
            if not cursor or not batch:
                return responses

    async def __aenter__(self) -> "KalshiReadClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()


def _extract_source_timestamp(message: Dict[str, Any], names: List[str]) -> Optional[int]:
    for name in names:
        value = message.get(name)
        if value not in (None, ""):
            try:
                return iso8601_to_ns(value)
            except (TypeError, ValueError):
                continue
    return None


class KalshiWebSocketReader:
    """Read-only market-data subscriptions with strict sequence classification."""

    def __init__(
        self,
        config: KalshiConfig,
        credentials: ResolvedKalshiCredentials,
        sequencer: Optional[GlobalReceiveSequencer] = None,
        websocket_max_queue: int = 1_024,
    ) -> None:
        if websocket_max_queue < 1:
            raise ValueError("websocket_max_queue must be at least one")
        self.config = config
        self._auth = KalshiAuth(
            api_key_id=credentials.api_key_id,
            private_key_path=credentials.private_key_path,
        )
        self.sequencer = sequencer or GlobalReceiveSequencer()
        self.websocket_max_queue = websocket_max_queue
        self._sequence = SequenceTracker()

    async def events(
        self,
        tickers: List[str],
        pyth_underlyings: Optional[List[str]] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[SourceEnvelope]:
        if not tickers and not pyth_underlyings:
            raise ValueError(
                "At least one ticker or Pyth underlying is required for WebSocket collection"
            )

        headers = self._auth.get_auth_headers("GET", WS_SIGNING_PATH)
        connection_id = str(uuid.uuid4())
        async with websockets.connect(
            self.config.ws_url,
            additional_headers=headers,
            ping_interval=10,
            ping_timeout=20,
            max_queue=self.websocket_max_queue,
        ) as websocket:
            # One order-book subscription per market isolates sequence gaps and
            # snapshot recovery. Public trades can share one subscription.
            subscriptions = []
            next_command_id = 1
            for ticker in tickers:
                subscriptions.append(
                    {
                        "id": next_command_id,
                        "cmd": "subscribe",
                        "params": {
                            "channels": ["orderbook_delta"],
                            "market_ticker": ticker,
                        },
                    }
                )
                next_command_id += 1
            if tickers:
                subscriptions.append(
                    {
                        "id": next_command_id,
                        "cmd": "subscribe",
                        "params": {
                            "channels": ["trade"],
                            "market_tickers": tickers,
                        },
                    }
                )
                next_command_id += 1
            subscriptions.append(
                {
                    "id": next_command_id,
                    "cmd": "subscribe",
                    "params": {"channels": ["market_lifecycle_v2"]},
                }
            )
            next_command_id += 1
            if pyth_underlyings:
                subscriptions.append(
                    {
                        "id": next_command_id,
                        "cmd": "subscribe",
                        "params": {
                            "channels": ["pyth_value"],
                            "underlying_tickers": pyth_underlyings,
                        },
                    }
                )
                next_command_id += 1
            for subscription in subscriptions:
                await websocket.send(json.dumps(subscription))

            while stop_event is None or not stop_event.is_set():
                receive_task = asyncio.ensure_future(websocket.recv())
                waiters = [receive_task]
                stop_task = None
                if stop_event is not None:
                    stop_task = asyncio.ensure_future(stop_event.wait())
                    waiters.append(stop_task)
                try:
                    done, pending = await asyncio.wait(
                        waiters, return_when=asyncio.FIRST_COMPLETED
                    )
                except BaseException:
                    # Cancelling the async generator (ticker rollover or normal
                    # collector shutdown) also has to consume the child recv
                    # result.  Otherwise the socket context can close first and
                    # leave a noisy "Task exception was never retrieved" behind.
                    for waiter in waiters:
                        if not waiter.done():
                            waiter.cancel()
                    await asyncio.gather(*waiters, return_exceptions=True)
                    raise
                for pending_task in pending:
                    pending_task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                if stop_task is not None and stop_task in done and stop_task.result():
                    if not receive_task.done():
                        receive_task.cancel()
                    await asyncio.gather(*done, return_exceptions=True)
                    break

                raw_text = receive_task.result()
                local_monotonic_ts_ns = time.monotonic_ns()
                receive_ts_ns = time.time_ns()
                # Allocate the process-global receive key before parsing. The
                # storage layer must preserve this stamp rather than replacing it.
                stamp = self.sequencer.stamp(
                    "kalshi",
                    "websocket",
                    receive_ts_ns=receive_ts_ns,
                    local_monotonic_ts_ns=local_monotonic_ts_ns,
                )
                parsed = json.loads(raw_text, parse_float=Decimal)
                sid = parsed.get("sid")
                sequence = parsed.get("seq")
                sequence_key = "{}:{}".format(connection_id, sid)
                decision = self._sequence.observe(sequence_key, sequence)
                message = parsed.get("msg", {})
                if not isinstance(message, dict):
                    message = {"value": message}
                envelope = SourceEnvelope(
                    source="kalshi",
                    stream=str(parsed.get("type", "unknown")),
                    ingestion_sequence=stamp.ingestion_sequence,
                    local_monotonic_ts_ns=stamp.local_monotonic_ts_ns,
                    receive_ts_ns=stamp.receive_ts_ns,
                    raw=parsed,
                    raw_text=raw_text,
                    source_sequence=sequence,
                    source_event_ts_ns=_extract_source_timestamp(
                        message,
                        ["ts_ms", "ts", "timestamp", "created_time", "trade_time"],
                    ),
                    source_publish_ts_ns=_extract_source_timestamp(
                        message,
                        ["publish_time", "published_time"],
                    ),
                    connection_id=connection_id,
                    subscription_id=str(sid) if sid is not None else None,
                    sequence_status=decision.status.value,
                    gap_flag=decision.status == SequenceStatus.GAP,
                    previous_monotonic_ts_ns=stamp.previous_monotonic_ts_ns,
                    monotonic_regression=stamp.monotonic_regression,
                )
                yield envelope

                if decision.status == SequenceStatus.GAP and sid is not None:
                    ticker = message.get("market_ticker")
                    if ticker:
                        recovery = {
                            "id": next_command_id,
                            "cmd": "update_subscription",
                            "params": {
                                "sids": [sid],
                                "market_tickers": [ticker],
                                "action": "get_snapshot",
                            },
                        }
                        next_command_id += 1
                        await websocket.send(json.dumps(recovery))
