"""Read-only, redundant Pyth Pro (formerly Lazer) live collection.

The collector deliberately contains no order-entry or execution integration.  It
connects to every configured Pyth endpoint, records the exact JSON text received,
and writes a parsed representation separately.  Pyth stream messages don't carry
a source sequence number, so transport gaps are classified but never assigned a
made-up count of missing updates.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import websockets
from websockets.exceptions import InvalidStatus

from .config import MetalsConfig
from .ingestion import GlobalReceiveSequencer, ReceiveStamp
from .provider_approval import require_config_provider_operation
from .quality import QualityTracker
from .storage import PartitionedRawStore, update_run_manifest, write_run_manifest

PYTH_LAZER_SUBSCRIPTION_ID = 1
PYTH_LAZER_CHANNEL = "fixed_rate@200ms"
PYTH_LAZER_MAX_QUEUE = 4_096
PYTH_LAZER_PROPERTIES = (
    "price",
    "feedUpdateTimestamp",
    "exponent",
    "confidence",
    "publisherCount",
    "marketSession",
)
PYTH_LAZER_MARKET_SESSIONS = frozenset(
    ("regular", "preMarket", "postMarket", "overNight", "closed")
)
PYTH_LAZER_MESSAGE_TYPES = frozenset(
    (
        "error",
        "subscribed",
        "subscribedWithInvalidFeedIdsIgnored",
        "unsubscribed",
        "subscriptionError",
        "streamUpdated",
    )
)


class PythLazerError(RuntimeError):
    """Base class for collector failures."""


class PythLazerAccessError(PythLazerError):
    """Authentication or configuration prevents collection."""


class PythLazerProtocolError(PythLazerError):
    """A source message has malformed or unknown semantics."""


class PythLazerStorageError(PythLazerError):
    """A durable source or normalized write failed."""


class _TransportGap(PythLazerError):
    """A recoverable endpoint-local transport interruption."""

    def __init__(self, classification: str) -> None:
        super().__init__(classification)
        self.classification = classification


def _require_object(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise PythLazerProtocolError("{} must be a JSON object".format(field))
    return value


def _require_list(value: Any, field: str) -> List[Any]:
    if not isinstance(value, list):
        raise PythLazerProtocolError("{} must be a JSON array".format(field))
    return value


def _parse_integer(
    value: Any,
    field: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    # bool is an int subclass in Python but isn't an integer in this protocol.
    if isinstance(value, bool):
        raise PythLazerProtocolError("{} must be an integer".format(field))
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        if not value or value.strip() != value:
            raise PythLazerProtocolError("{} has invalid integer text".format(field))
        digits = value[1:] if value.startswith("-") else value
        if not digits.isdigit():
            raise PythLazerProtocolError("{} has invalid integer text".format(field))
        parsed = int(value)
    else:
        raise PythLazerProtocolError(
            "{} must be an integer or integer text".format(field)
        )
    if minimum is not None and parsed < minimum:
        raise PythLazerProtocolError("{} is below its protocol minimum".format(field))
    if maximum is not None and parsed > maximum:
        raise PythLazerProtocolError("{} is above its protocol maximum".format(field))
    return parsed


def _scaled_decimal_text(mantissa: int, exponent: int) -> str:
    try:
        value = Decimal(mantissa).scaleb(exponent)
    except (InvalidOperation, ValueError) as error:  # pragma: no cover - defensive
        raise PythLazerProtocolError("Pyth price cannot be scaled") from error
    return format(value, "f")


def parse_pyth_lazer_message(raw_text: str) -> Dict[str, Any]:
    """Parse a JSON-delivery message without losing the original source text."""

    if not isinstance(raw_text, str):
        raise PythLazerProtocolError(
            "Pyth sent binary data after JSON delivery was requested"
        )
    try:
        value = json.loads(raw_text, parse_float=Decimal)
    except (json.JSONDecodeError, InvalidOperation) as error:
        raise PythLazerProtocolError("Pyth sent malformed JSON") from error
    message = _require_object(value, "Pyth message")
    message_type = message.get("type")
    if (
        not isinstance(message_type, str)
        or message_type not in PYTH_LAZER_MESSAGE_TYPES
    ):
        raise PythLazerProtocolError("Pyth sent an unknown WebSocket message type")
    return message


def _validate_subscription_id(message: Mapping[str, Any]) -> int:
    subscription_id = _parse_integer(
        message.get("subscriptionId"), "subscriptionId", minimum=0
    )
    if subscription_id != PYTH_LAZER_SUBSCRIPTION_ID:
        raise PythLazerProtocolError("Pyth returned an unexpected subscriptionId")
    return subscription_id


def normalize_pyth_lazer_update(
    message: Mapping[str, Any],
    *,
    expected_feed_id: int,
    pyth_symbol: str,
    metal: str,
    endpoint: str,
    connection_id: str,
    receive_ts_ns: int,
    local_monotonic_ts_ns: int,
    ingestion_sequence: int,
    active_endpoint: Optional[str],
    previous_monotonic_ts_ns: Optional[int] = None,
    monotonic_regression: bool = False,
) -> List[Dict[str, Any]]:
    """Validate and normalize one documented ``streamUpdated`` response.

    The official examples serialize some 64-bit integers as strings while the
    OpenAPI schema calls them integers.  Both lossless representations are
    accepted; floats, booleans, and numeric text with whitespace are rejected.
    """

    if message.get("type") != "streamUpdated":
        raise PythLazerProtocolError("Expected a Pyth streamUpdated response")
    subscription_id = _validate_subscription_id(message)
    parsed = _require_object(message.get("parsed"), "parsed")
    update_timestamp_us = _parse_integer(
        parsed.get("timestampUs"), "parsed.timestampUs", minimum=0
    )
    feeds = _require_list(parsed.get("priceFeeds"), "parsed.priceFeeds")
    if not feeds:
        raise PythLazerProtocolError("Pyth stream update contains no price feeds")

    rows: List[Dict[str, Any]] = []
    seen_feed_ids = set()
    for index, raw_feed in enumerate(feeds):
        feed = _require_object(raw_feed, "parsed.priceFeeds[{}]".format(index))
        feed_id = _parse_integer(
            feed.get("priceFeedId"),
            "priceFeedId",
            minimum=0,
            maximum=2**32 - 1,
        )
        if feed_id in seen_feed_ids:
            raise PythLazerProtocolError("Pyth update contains a duplicate feed ID")
        seen_feed_ids.add(feed_id)
        if feed_id != expected_feed_id:
            raise PythLazerProtocolError("Pyth update contains an unexpected feed ID")

        price_mantissa = _parse_integer(
            feed.get("price"), "price", minimum=-(2**63), maximum=2**63 - 1
        )
        if price_mantissa == 0:
            raise PythLazerProtocolError("Pyth price must be non-zero when present")
        exponent = _parse_integer(
            feed.get("exponent"), "exponent", minimum=-(2**15), maximum=2**15 - 1
        )
        confidence_mantissa = _parse_integer(
            feed.get("confidence"),
            "confidence",
            minimum=0,
            maximum=2**63 - 1,
        )
        publisher_count = _parse_integer(
            feed.get("publisherCount"),
            "publisherCount",
            minimum=0,
            maximum=2**16 - 1,
        )
        market_session = feed.get("marketSession")
        if market_session not in PYTH_LAZER_MARKET_SESSIONS:
            raise PythLazerProtocolError("Pyth returned an unknown marketSession")
        feed_update_timestamp_us = _parse_integer(
            feed.get("feedUpdateTimestamp"),
            "feedUpdateTimestamp",
            minimum=0,
        )
        if feed_update_timestamp_us > update_timestamp_us:
            raise PythLazerProtocolError(
                "feedUpdateTimestamp is later than the stream timestamp"
            )

        rows.append(
            {
                "source": "pyth_lazer",
                "stream": "price",
                "metal": metal,
                "pyth_symbol": pyth_symbol,
                "pyth_lazer_id": feed_id,
                "subscription_id": subscription_id,
                "channel": PYTH_LAZER_CHANNEL,
                "price_mantissa": str(price_mantissa),
                "exponent": exponent,
                "price": _scaled_decimal_text(price_mantissa, exponent),
                "confidence_mantissa": str(confidence_mantissa),
                "confidence": _scaled_decimal_text(confidence_mantissa, exponent),
                "publisher_count": publisher_count,
                "market_session": market_session,
                "update_timestamp_us": update_timestamp_us,
                "source_event_ts_ns": update_timestamp_us * 1_000,
                "feed_update_timestamp_us": feed_update_timestamp_us,
                "source_publish_ts_ns": feed_update_timestamp_us * 1_000,
                "is_carried_forward": feed_update_timestamp_us < update_timestamp_us,
                "source_timestamp_semantics": "feedUpdateTimestamp_microseconds",
                "receive_ts_ns": receive_ts_ns,
                "local_monotonic_ts_ns": local_monotonic_ts_ns,
                "previous_monotonic_ts_ns": previous_monotonic_ts_ns,
                "monotonic_regression": monotonic_regression,
                "endpoint": endpoint,
                "connection_id": connection_id,
                "active_endpoint": active_endpoint,
                "endpoint_role": (
                    "active" if endpoint == active_endpoint else "standby"
                ),
                "ingestion_sequence": ingestion_sequence,
            }
        )

    if seen_feed_ids != {expected_feed_id}:
        raise PythLazerProtocolError("Pyth update omitted the requested feed ID")
    return rows


def build_pyth_lazer_subscription(feed_id: int) -> Dict[str, Any]:
    """Return the fixed, offchain-only subscription used by this collector."""

    if isinstance(feed_id, bool) or not isinstance(feed_id, int) or feed_id < 0:
        raise ValueError("Pyth Lazer feed ID must be a non-negative integer")
    return {
        "type": "subscribe",
        "subscriptionId": PYTH_LAZER_SUBSCRIPTION_ID,
        "priceFeedIds": [feed_id],
        "properties": list(PYTH_LAZER_PROPERTIES),
        "formats": [],
        "channel": PYTH_LAZER_CHANNEL,
        "ignoreInvalidFeeds": False,
        "deliveryFormat": "json",
        "parsed": True,
    }


class PythLazerCollector:
    """Collect one configured metal from every Pyth endpoint concurrently."""

    def __init__(
        self,
        config: MetalsConfig,
        metal: str,
        *,
        connect_factory: Optional[Callable[..., Any]] = None,
        reconnect_base_seconds: float = 0.25,
        reconnect_max_seconds: float = 5.0,
        receive_timeout_seconds: float = 5.0,
        sequencer: Optional[GlobalReceiveSequencer] = None,
    ) -> None:
        product = config.products.get(metal)
        if not config.pyth.enabled:
            raise PythLazerAccessError(
                "Pyth Lazer live collection requires [pyth].enabled = true"
            )
        if product is None:
            raise ValueError("Unknown metals product: {}".format(metal))
        if not product.enabled:
            raise ValueError("Metals product is disabled: {}".format(metal))
        if product.pyth_lazer_id is None or not product.pyth_symbol:
            raise ValueError("No Pyth Lazer feed configured for {}".format(metal))
        self._authorize_collection(config, metal)
        endpoints = list(config.pyth.lazer_ws_endpoints)
        if not endpoints or any(
            not endpoint.startswith("wss://") for endpoint in endpoints
        ):
            raise ValueError("Pyth Lazer requires at least one wss:// endpoint")
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("Pyth Lazer endpoints must be unique")
        if reconnect_base_seconds < 0 or reconnect_max_seconds < reconnect_base_seconds:
            raise ValueError("Invalid Pyth reconnect timing")
        if receive_timeout_seconds <= 0:
            raise ValueError("Pyth receive timeout must be positive")

        self.config = config
        self.metal = metal
        self.product = product
        self.endpoints = endpoints
        self._endpoint_order = {
            endpoint: index for index, endpoint in enumerate(endpoints)
        }
        self._connect_factory = connect_factory or websockets.connect
        self._reconnect_base_seconds = reconnect_base_seconds
        self._reconnect_max_seconds = reconnect_max_seconds
        self._receive_timeout_seconds = receive_timeout_seconds
        self.sequencer = sequencer or GlobalReceiveSequencer()
        self.quality = QualityTracker()
        self._stop = asyncio.Event()
        self._raw_store: Optional[PartitionedRawStore] = None
        self._normalized_store: Optional[PartitionedRawStore] = None
        self._healthy_connections: Dict[str, str] = {}
        self._active_endpoint: Optional[str] = None
        self._gap_started_monotonic_ns: Dict[str, int] = {}
        self._counts: Dict[str, int] = {
            "raw_messages": 0,
            "normalized_prices": 0,
            "control_messages": 0,
            "connections": 0,
            "reconnects": 0,
            "gap_events": 0,
            "transport_disconnects": 0,
            "ingestion_monotonic_regressions": 0,
        }

    def _authorize_collection(self, config: MetalsConfig, metal: str) -> None:
        """The general collector continues to require formal provider evidence.

        A specialized shadow adapter may supply a differently classified,
        explicitly scoped research authority. There is no public bypass flag.
        """
        self.provider_approval = require_config_provider_operation(
            config, "pyth", "live_lazer"
        )

    def _authorization_manifest(self) -> Tuple[Dict[str, Any], Tuple[Path, ...]]:
        return (
            {"provider_approval_id": self.provider_approval.approval_id},
            (self.provider_approval.approval_path, self.provider_approval.manifest_path),
        )

    @property
    def run_id(self) -> Optional[str]:
        return self._raw_store.run_id if self._raw_store is not None else None

    def stop(self) -> None:
        """Request a graceful stop without interrupting durable writes."""

        self._stop.set()

    def _select_active_endpoint(self) -> Optional[str]:
        if not self._healthy_connections:
            return None
        return min(
            self._healthy_connections,
            key=lambda endpoint: self._endpoint_order[endpoint],
        )

    def _write_gap(
        self,
        classification: str,
        endpoint: str,
        *,
        connection_id: Optional[str],
        receive_ts_ns: Optional[int] = None,
        local_monotonic_ts_ns: Optional[int] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self._normalized_store is None:
            raise RuntimeError("Pyth stores are not initialized")
        stamp = self.sequencer.stamp(
            "pyth_lazer",
            "gap",
            receive_ts_ns=receive_ts_ns,
            local_monotonic_ts_ns=local_monotonic_ts_ns,
        )
        if stamp.monotonic_regression:
            self._counts["ingestion_monotonic_regressions"] += 1
            self.quality.record("ingestion_monotonic_regressions")
            self.quality.record_missing_interval(
                "pyth",
                stamp.receive_ts_ns,
                stamp.receive_ts_ns,
                "global_monotonic_regression",
            )
        record = {
            "source": "pyth_lazer",
            "stream": "gap",
            "metal": self.metal,
            "gap_classification": classification,
            "potential_data_loss": classification
            not in ("connection_ready", "redundancy_restored"),
            "missing_update_count": None,
            "missing_update_count_semantics": "unavailable_no_source_sequence",
            "endpoint": endpoint,
            "connection_id": connection_id,
            "active_endpoint": self._active_endpoint,
            "receive_ts_ns": stamp.receive_ts_ns,
            "local_monotonic_ts_ns": stamp.local_monotonic_ts_ns,
            "previous_monotonic_ts_ns": stamp.previous_monotonic_ts_ns,
            "monotonic_regression": stamp.monotonic_regression,
            "ingestion_sequence": stamp.ingestion_sequence,
            "details": dict(details or {}),
        }
        self._normalized_store.write(
            source="pyth_lazer_gap",
            stream="pyth_lazer_gap",
            metal=self.metal,
            receive_ts_ns=stamp.receive_ts_ns,
            record=record,
        )
        self._counts["gap_events"] += 1
        self.quality.record("pyth_gap_events")
        self.quality.record_missing_interval(
            "pyth",
            stamp.receive_ts_ns,
            stamp.receive_ts_ns,
            "pyth_gap:{}".format(classification),
        )

    def _mark_ready(
        self,
        endpoint: str,
        connection_id: str,
        receive_ts_ns: int,
        local_monotonic_ts_ns: int,
    ) -> None:
        previous_active = self._active_endpoint
        was_gapped = endpoint in self._gap_started_monotonic_ns
        self._healthy_connections[endpoint] = connection_id
        self._active_endpoint = self._select_active_endpoint()
        details: Dict[str, Any] = {
            "previous_active_endpoint": previous_active,
            "new_active_endpoint": self._active_endpoint,
            "healthy_endpoint_count": len(self._healthy_connections),
        }
        if was_gapped:
            gap_start = self._gap_started_monotonic_ns.pop(endpoint)
            details["gap_duration_ns"] = max(0, local_monotonic_ts_ns - gap_start)
            classification = "connection_recovered"
        else:
            classification = "connection_ready"
        self._write_gap(
            classification,
            endpoint,
            connection_id=connection_id,
            receive_ts_ns=receive_ts_ns,
            local_monotonic_ts_ns=local_monotonic_ts_ns,
            details=details,
        )
        if previous_active is not None and self._active_endpoint != previous_active:
            self._write_gap(
                "active_endpoint_failover",
                endpoint,
                connection_id=connection_id,
                receive_ts_ns=receive_ts_ns,
                local_monotonic_ts_ns=local_monotonic_ts_ns,
                details=details,
            )

    def _mark_transport_gap(
        self,
        endpoint: str,
        connection_id: Optional[str],
        classification: str,
    ) -> None:
        self._counts["transport_disconnects"] += 1
        self.quality.record("pyth_transport_disconnects")
        transport_ts_ns = time.time_ns()
        self.quality.record_missing_interval(
            "pyth",
            transport_ts_ns,
            None,
            "pyth_transport:{}".format(classification),
        )
        now_monotonic_ns = time.monotonic_ns()
        self._gap_started_monotonic_ns.setdefault(endpoint, now_monotonic_ns)
        previous_active = self._active_endpoint
        self._healthy_connections.pop(endpoint, None)
        self._active_endpoint = self._select_active_endpoint()
        self._write_gap(
            classification,
            endpoint,
            connection_id=connection_id,
            local_monotonic_ts_ns=now_monotonic_ns,
            details={
                "previous_active_endpoint": previous_active,
                "failover_to_endpoint": self._active_endpoint,
                "healthy_endpoint_count": len(self._healthy_connections),
                "error_text_persisted": False,
            },
        )
        if previous_active == endpoint:
            self._write_gap(
                (
                    "all_endpoints_unavailable"
                    if self._active_endpoint is None
                    else "active_endpoint_failover"
                ),
                endpoint,
                connection_id=connection_id,
                local_monotonic_ts_ns=now_monotonic_ns,
                details={
                    "previous_active_endpoint": previous_active,
                    "failover_to_endpoint": self._active_endpoint,
                    "healthy_endpoint_count": len(self._healthy_connections),
                },
            )

    def _write_source_message(
        self,
        endpoint: str,
        connection_id: str,
        raw_text: str,
        message: Dict[str, Any],
        stamp: ReceiveStamp,
        *,
        normalize_update: bool = True,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        if self._raw_store is None or self._normalized_store is None:
            raise RuntimeError("Pyth stores are not initialized")
        ingestion_sequence = stamp.ingestion_sequence
        rows: List[Dict[str, Any]] = []
        source_event_ts_ns = None
        source_publish_ts_ns = None
        normalization_error: Optional[PythLazerProtocolError] = None
        if message["type"] == "streamUpdated" and normalize_update:
            try:
                rows = normalize_pyth_lazer_update(
                    message,
                    expected_feed_id=int(self.product.pyth_lazer_id),
                    pyth_symbol=self.product.pyth_symbol,
                    metal=self.metal,
                    endpoint=endpoint,
                    connection_id=connection_id,
                    receive_ts_ns=stamp.receive_ts_ns,
                    local_monotonic_ts_ns=stamp.local_monotonic_ts_ns,
                    ingestion_sequence=ingestion_sequence,
                    active_endpoint=self._active_endpoint,
                    previous_monotonic_ts_ns=stamp.previous_monotonic_ts_ns,
                    monotonic_regression=stamp.monotonic_regression,
                )
                source_event_ts_ns = rows[0]["source_event_ts_ns"]
                source_publish_ts_ns = rows[0]["source_publish_ts_ns"]
            except PythLazerProtocolError as error:
                # Preserve the exact offending source message before failing the
                # run, but never publish a partial normalized representation.
                normalization_error = error

        try:
            self._raw_store.write(
                source="pyth_lazer",
                stream="websocket",
                metal=self.metal,
                receive_ts_ns=stamp.receive_ts_ns,
                record={
                    "source": "pyth_lazer",
                    "stream": "websocket",
                    "message_type": message["type"],
                    "endpoint": endpoint,
                    "connection_id": connection_id,
                    "subscription_id": message.get("subscriptionId"),
                    "source_event_ts_ns": source_event_ts_ns,
                    "source_publish_ts_ns": source_publish_ts_ns,
                    "receive_ts_ns": stamp.receive_ts_ns,
                    "local_monotonic_ts_ns": stamp.local_monotonic_ts_ns,
                    "previous_monotonic_ts_ns": stamp.previous_monotonic_ts_ns,
                    "monotonic_regression": stamp.monotonic_regression,
                    "ingestion_sequence": ingestion_sequence,
                    "raw_text": raw_text,
                    "raw": message,
                },
            )
        except BaseException as error:
            raise PythLazerStorageError("Pyth raw source write failed") from error
        self._counts["raw_messages"] += 1
        self.quality.record("raw_records")
        if normalization_error is not None:
            raise normalization_error
        for row in rows:
            try:
                self._normalized_store.write(
                    source="pyth_lazer_price",
                    stream="pyth_lazer_price",
                    metal=self.metal,
                    receive_ts_ns=stamp.receive_ts_ns,
                    record=row,
                )
            except BaseException as error:
                raise PythLazerStorageError(
                    "Pyth normalized source write failed"
                ) from error
            self._counts["normalized_prices"] += 1
        return ingestion_sequence, rows

    def _write_unparsed_source_message(
        self,
        endpoint: str,
        connection_id: str,
        raw_value: Any,
        stamp: ReceiveStamp,
    ) -> None:
        """Persist an exact malformed/unknown callback before failing closed."""

        if self._raw_store is None:
            raise RuntimeError("Pyth stores are not initialized")
        exact_text = raw_value if isinstance(raw_value, str) else None
        exact_binary_hex = (
            bytes(raw_value).hex()
            if isinstance(raw_value, (bytes, bytearray, memoryview))
            else None
        )
        try:
            self._raw_store.write(
                source="pyth_lazer",
                stream="websocket",
                metal=self.metal,
                receive_ts_ns=stamp.receive_ts_ns,
                record={
                    "source": "pyth_lazer",
                    "stream": "websocket",
                    "message_type": "protocol_error_unparsed",
                    "endpoint": endpoint,
                    "connection_id": connection_id,
                    "subscription_id": None,
                    "source_event_ts_ns": None,
                    "source_publish_ts_ns": None,
                    "receive_ts_ns": stamp.receive_ts_ns,
                    "local_monotonic_ts_ns": stamp.local_monotonic_ts_ns,
                    "previous_monotonic_ts_ns": stamp.previous_monotonic_ts_ns,
                    "monotonic_regression": stamp.monotonic_regression,
                    "ingestion_sequence": stamp.ingestion_sequence,
                    "raw_text": exact_text,
                    "raw_binary_hex": exact_binary_hex,
                    "raw": None,
                },
            )
        except BaseException as error:
            raise PythLazerStorageError("Pyth raw source write failed") from error
        self._counts["raw_messages"] += 1
        self.quality.record("raw_records")

    async def _recv_or_stop(self, websocket: Any) -> Optional[Any]:
        receive_task = asyncio.ensure_future(websocket.recv())
        stop_task = asyncio.ensure_future(self._stop.wait())
        done, pending = await asyncio.wait(
            (receive_task, stop_task),
            timeout=self._receive_timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if not done:
            await asyncio.gather(*pending, return_exceptions=True)
            raise _TransportGap("transport_receive_timeout")
        if stop_task in done and stop_task.result():
            receive_task.cancel()
            await asyncio.gather(receive_task, return_exceptions=True)
            return None
        stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)
        try:
            return receive_task.result()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise _TransportGap("transport_disconnect") from error

    async def _connection_once(
        self, endpoint: str, api_key: str, connection_id: str
    ) -> None:
        headers = {"Authorization": "Bearer {}".format(api_key)}
        try:
            connection = self._connect_factory(
                endpoint,
                additional_headers=headers,
                ping_interval=10,
                ping_timeout=20,
                max_queue=PYTH_LAZER_MAX_QUEUE,
            )
            async with connection as websocket:
                self._counts["connections"] += 1
                await websocket.send(
                    json.dumps(
                        build_pyth_lazer_subscription(int(self.product.pyth_lazer_id)),
                        separators=(",", ":"),
                    )
                )
                subscribed = False
                while not self._stop.is_set():
                    raw_value = await self._recv_or_stop(websocket)
                    if raw_value is None:
                        return
                    # Stamp immediately at callback acceptance, before parsing or
                    # storage, so a shared sequencer defines causal replay order.
                    stamp = self.sequencer.stamp("pyth_lazer", "websocket")
                    if stamp.monotonic_regression:
                        self._counts["ingestion_monotonic_regressions"] += 1
                        self.quality.record("ingestion_monotonic_regressions")
                        self.quality.record_missing_interval(
                            "pyth",
                            stamp.receive_ts_ns,
                            stamp.receive_ts_ns,
                            "global_monotonic_regression",
                        )
                    try:
                        message = parse_pyth_lazer_message(raw_value)
                    except PythLazerProtocolError:
                        self._write_unparsed_source_message(
                            endpoint, connection_id, raw_value, stamp
                        )
                        raise
                    message_type = message["type"]
                    if message_type == "streamUpdated" and not subscribed:
                        self._write_source_message(
                            endpoint,
                            connection_id,
                            raw_value,
                            message,
                            stamp,
                            normalize_update=False,
                        )
                        raise PythLazerProtocolError(
                            "Pyth streamed data before confirming subscription"
                        )
                    self._write_source_message(
                        endpoint,
                        connection_id,
                        raw_value,
                        message,
                        stamp,
                    )
                    if message_type == "subscribed":
                        _validate_subscription_id(message)
                        if subscribed:
                            raise PythLazerProtocolError(
                                "Pyth sent duplicate subscribed responses"
                            )
                        subscribed = True
                        self._counts["control_messages"] += 1
                        self._mark_ready(
                            endpoint,
                            connection_id,
                            stamp.receive_ts_ns,
                            stamp.local_monotonic_ts_ns,
                        )
                    elif message_type == "streamUpdated":
                        pass
                    elif message_type == "subscribedWithInvalidFeedIdsIgnored":
                        raise PythLazerProtocolError(
                            "Pyth ignored a feed despite strict subscription semantics"
                        )
                    elif message_type in (
                        "error",
                        "subscriptionError",
                        "unsubscribed",
                    ):
                        raise PythLazerProtocolError(
                            "Pyth rejected or ended the live subscription"
                        )
                return
        except PythLazerProtocolError:
            self._write_gap(
                "protocol_violation",
                endpoint,
                connection_id=connection_id,
            )
            raise
        except PythLazerStorageError:
            raise
        except _TransportGap:
            raise
        except asyncio.CancelledError:
            raise
        except InvalidStatus as error:
            status = error.response.status_code
            if status in (401, 403):
                raise PythLazerAccessError(
                    f"Pyth live access denied (HTTP {status}); no automatic retry"
                ) from None
            raise _TransportGap("transport_connect_failure") from error
        except Exception as error:
            raise _TransportGap("transport_connect_failure") from error

    async def _endpoint_loop(self, endpoint: str, api_key: str) -> None:
        attempt = 0
        while not self._stop.is_set():
            connection_id = str(uuid.uuid4())
            try:
                await self._connection_once(endpoint, api_key, connection_id)
                if self._stop.is_set():
                    return
                raise _TransportGap("transport_disconnect")
            except PythLazerProtocolError:
                raise
            except _TransportGap as gap:
                if self._stop.is_set():
                    return
                self._mark_transport_gap(endpoint, connection_id, gap.classification)
                attempt += 1
                self._counts["reconnects"] += 1
                delay = min(
                    self._reconnect_max_seconds,
                    self._reconnect_base_seconds * (2 ** min(attempt - 1, 10)),
                )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    continue

    async def run(
        self,
        *,
        duration_seconds: Optional[float] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, int]:
        """Run until a positive duration elapses or a supplied stop is set."""

        if duration_seconds is None and stop_event is None:
            raise ValueError("Pyth collection requires a duration or stop_event")
        if duration_seconds is not None and (
            not math.isfinite(duration_seconds) or duration_seconds <= 0
        ):
            raise ValueError("Pyth collection duration must be finite and positive")
        if self._raw_store is not None:
            raise RuntimeError("A PythLazerCollector instance can run only once")
        api_key = os.environ.get(self.config.pyth.api_key_env, "").strip()
        if not api_key:
            raise PythLazerAccessError(
                "Set {} to a Pyth Pro API key".format(self.config.pyth.api_key_env)
            )

        authority_fields, authority_paths = self._authorization_manifest()
        storage_root = Path(self.config.storage.root)
        self._raw_store = PartitionedRawStore(
            storage_root,
            fsync_every_records=self.config.storage.fsync_every_records,
            tier="raw",
        )
        self._normalized_store = PartitionedRawStore(
            storage_root,
            run_id=self._raw_store.run_id,
            fsync_every_records=self.config.storage.fsync_every_records,
            tier="normalized",
        )
        manifest_config = self.config.public_snapshot()
        manifest_config["collector"] = {
            "component": "pyth_lazer_live",
            "metal": self.metal,
            "channel": PYTH_LAZER_CHANNEL,
            "properties": list(PYTH_LAZER_PROPERTIES),
            "formats": [],
            "duration_seconds": duration_seconds,
            **authority_fields,
        }
        write_run_manifest(
            storage_root,
            self._raw_store.run_id,
            manifest_config,
            output_paths=authority_paths,
        )

        async def stop_after_duration() -> None:
            if duration_seconds is not None:
                await asyncio.sleep(duration_seconds)
                self._stop.set()

        async def copy_external_stop() -> None:
            if stop_event is not None:
                await stop_event.wait()
                self._stop.set()

        workers = [
            asyncio.ensure_future(self._endpoint_loop(endpoint, api_key))
            for endpoint in self.endpoints
        ]
        bounds = [
            asyncio.ensure_future(stop_after_duration()),
            asyncio.ensure_future(copy_external_stop()),
        ]
        all_tasks = workers + bounds
        failure: Optional[BaseException] = None
        try:
            while workers and not self._stop.is_set():
                done, _ = await asyncio.wait(
                    all_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    if task in bounds:
                        all_tasks.remove(task)
                        continue
                    if task in workers:
                        workers.remove(task)
                        all_tasks.remove(task)
                    error = task.exception()
                    if error is not None:
                        raise error
                    if not self._stop.is_set():
                        raise PythLazerError(
                            "Pyth endpoint worker stopped unexpectedly"
                        )
            self._stop.set()
        except BaseException as error:
            failure = error
            self._stop.set()
            raise
        finally:
            for task in all_tasks:
                task.cancel()
            await asyncio.gather(*all_tasks, return_exceptions=True)
            close_failure: Optional[BaseException] = None
            for store in (self._raw_store, self._normalized_store):
                try:
                    store.close()
                except BaseException as error:
                    if close_failure is None:
                        close_failure = error
            update_run_manifest(
                storage_root,
                self._raw_store.run_id,
                status=(
                    "failed"
                    if failure is not None or close_failure is not None
                    else "completed"
                ),
                counts=self._counts,
                output_paths=self._raw_store.paths + self._normalized_store.paths,
                errors=[
                    error for error in (failure, close_failure) if error is not None
                ],
            )
            if failure is None and close_failure is not None:
                raise close_failure
        return dict(self._counts)


async def collect_pyth_lazer(
    config: MetalsConfig,
    metal: str,
    *,
    duration_seconds: Optional[float] = None,
    stop_event: Optional[asyncio.Event] = None,
) -> Dict[str, int]:
    """Convenience entry point for CLI or orchestration integration."""

    collector = PythLazerCollector(config, metal)
    return await collector.run(
        duration_seconds=duration_seconds,
        stop_event=stop_event,
    )
