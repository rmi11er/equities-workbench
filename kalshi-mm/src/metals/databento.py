"""Databento ``GLBX.MDP3`` live adapter for CME metals market data.

The Databento package is intentionally imported only when a real client is
created.  Normalization and injected-client tests therefore do not require the
optional dependency.  API keys are supplied by the caller and are never read,
serialized, or logged by this module.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


GLBX_DATASET = "GLBX.MDP3"
FIXED_PRICE_SCALE = 9
UNDEFINED_PRICE = (1 << 63) - 1
SUPPORTED_SCHEMAS = ("mbo", "trades", "definition", "status", "statistics")
CONTROL_SCHEMAS = ("symbol_mapping", "system", "error", "reconnect")
SUPPORTED_STYPES = ("raw_symbol", "instrument_id", "parent", "continuous")


class DatabentoUnavailableError(RuntimeError):
    """Raised when a real client is requested without the optional package."""


class DatabentoConfigurationError(ValueError):
    """Raised for an invalid local subscription or collector configuration."""


Symbol = Union[str, int]
NormalizedRecordCallback = Callable[
    [Dict[str, Any]], Optional[Awaitable[None]]
]


@dataclass(frozen=True)
class DeferredDatabentoRecord:
    """A native record plus callback-edge receipt clocks for off-thread work."""

    record: Any
    local_receive_ts_ns: int
    local_monotonic_ts_ns: int


def _deduplicate(values: Iterable[Symbol]) -> Tuple[Symbol, ...]:
    result: List[Symbol] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def normalize_subscription_symbols(
    symbols: Iterable[Symbol], stype_in: str
) -> Tuple[Symbol, ...]:
    """Validate symbols and expand CME product roots for parent symbology.

    Databento parent futures symbols use the ``ROOT.FUT`` convention.  Thus
    configured roots such as ``SI``, ``SIL``, ``SIC``, ``GC``, and ``MGC`` are
    accepted directly and expanded for callers.  Raw contract symbols such as
    ``SIU6`` remain unchanged when ``stype_in="raw_symbol"``.
    """

    if stype_in not in SUPPORTED_STYPES:
        raise DatabentoConfigurationError(
            "Unsupported Databento stype_in: {!r}".format(stype_in)
        )

    normalized: List[Symbol] = []
    for symbol in symbols:
        if stype_in == "instrument_id":
            if isinstance(symbol, bool) or not isinstance(symbol, int) or symbol < 0:
                raise DatabentoConfigurationError(
                    "instrument_id symbols must be non-negative integers"
                )
            normalized.append(symbol)
            continue

        if not isinstance(symbol, str) or not symbol.strip():
            raise DatabentoConfigurationError(
                "Databento symbols must be non-empty strings"
            )
        value = symbol.strip()
        if stype_in != "continuous":
            value = value.upper()
        if stype_in == "parent" and "." not in value:
            value = value + ".FUT"
        normalized.append(value)

    result = _deduplicate(normalized)
    if not result:
        raise DatabentoConfigurationError(
            "At least one Databento symbol is required"
        )
    return result


@dataclass(frozen=True)
class DatabentoSubscription:
    """One live schema subscription within a Databento session."""

    schema: str
    symbols: Tuple[Symbol, ...]
    stype_in: str = "parent"
    dataset: str = GLBX_DATASET
    start: Optional[Any] = None
    snapshot: bool = False

    def __post_init__(self) -> None:
        if self.schema not in SUPPORTED_SCHEMAS:
            raise DatabentoConfigurationError(
                "Unsupported Databento schema: {!r}".format(self.schema)
            )
        if not self.dataset:
            raise DatabentoConfigurationError("Databento dataset is empty")
        normalized = normalize_subscription_symbols(self.symbols, self.stype_in)
        object.__setattr__(self, "symbols", normalized)
        if self.snapshot and self.schema != "mbo":
            raise DatabentoConfigurationError(
                "Databento snapshots are only supported for the mbo schema"
            )
        if self.snapshot and self.start is not None:
            raise DatabentoConfigurationError(
                "Databento mbo snapshot conflicts with intraday replay start"
            )

    @classmethod
    def for_product_roots(
        cls,
        schema: str,
        roots: Iterable[str],
        dataset: str = GLBX_DATASET,
        start: Optional[Any] = None,
        snapshot: bool = False,
    ) -> "DatabentoSubscription":
        return cls(
            schema=schema,
            symbols=tuple(roots),
            stype_in="parent",
            dataset=dataset,
            start=start,
            snapshot=snapshot,
        )

    @classmethod
    def for_raw_symbols(
        cls,
        schema: str,
        symbols: Iterable[str],
        dataset: str = GLBX_DATASET,
        start: Optional[Any] = None,
        snapshot: bool = False,
    ) -> "DatabentoSubscription":
        return cls(
            schema=schema,
            symbols=tuple(symbols),
            stype_in="raw_symbol",
            dataset=dataset,
            start=start,
            snapshot=snapshot,
        )

    def subscribe_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "dataset": self.dataset,
            "schema": self.schema,
            "symbols": list(self.symbols),
            "stype_in": self.stype_in,
        }
        if self.start is not None:
            kwargs["start"] = self.start
        if self.snapshot:
            kwargs["snapshot"] = True
        return kwargs


def glbx_metals_subscriptions(
    symbols: Iterable[Symbol],
    stype_in: str = "parent",
    schemas: Sequence[str] = SUPPORTED_SCHEMAS,
    mbo_snapshot: bool = True,
    start: Optional[Any] = None,
) -> Tuple[DatabentoSubscription, ...]:
    """Build the standard MBO/trade/reference subscriptions for metal symbols."""

    normalized_symbols = normalize_subscription_symbols(symbols, stype_in)
    return tuple(
        DatabentoSubscription(
            schema=schema,
            symbols=normalized_symbols,
            stype_in=stype_in,
            start=start,
            snapshot=mbo_snapshot and schema == "mbo",
        )
        for schema in schemas
    )


_HEADER_FIELDS = (
    "length",
    "rtype",
    "publisher_id",
    "instrument_id",
    "ts_event",
)

_RECORD_FIELDS = (
    "ts_recv",
    "ts_in_delta",
    "ts_out",
    "sequence",
    "channel_id",
    "order_id",
    "action",
    "side",
    "price",
    "size",
    "depth",
    "flags",
    "ts_ref",
    "quantity",
    "stat_type",
    "update_action",
    "stat_flags",
    "reason",
    "trading_event",
    "is_trading",
    "is_quoting",
    "is_short_sell_restricted",
    "raw_symbol",
    "security_update_action",
    "instrument_class",
    "match_algorithm",
    "md_security_trading_status",
    "currency",
    "settl_currency",
    "secsubtype",
    "raw_instrument_id",
    "group",
    "exchange",
    "asset",
    "cfi",
    "security_type",
    "unit_of_measure",
    "underlying",
    "strike_price",
    "expiration",
    "activation",
    "maturity",
    "trading_reference_price",
    "min_price_increment",
    "min_price_increment_amount",
    "display_factor",
    "high_limit_price",
    "low_limit_price",
    "max_price_variation",
    "unit_of_measure_qty",
    "price_ratio",
    "inst_attrib_value",
    "underlying_id",
    "market_depth_implied",
    "market_depth",
    "market_segment_id",
    "expiration_style",
    "settlement_price_type",
    "lot_size",
    "min_trade_vol",
    "max_trade_vol",
    "min_lot_size",
    "min_lot_size_block",
    "min_lot_size_round_lot",
    "contract_multiplier",
    "decay_quantity",
    "original_contract_size",
    "appl_id",
    "maturity_year",
    "decay_start_date",
    "strike_price_currency",
    "main_fraction",
    "price_display_format",
    "sub_fraction",
    "underlying_product",
    "maturity_month",
    "maturity_day",
    "maturity_week",
    "user_defined_instrument",
    "contract_multiplier_unit",
    "flow_schedule_type",
    "tick_rule",
    "leg_count",
    "leg_index",
    "leg_instrument_id",
    "leg_ratio_price_numerator",
    "leg_ratio_price_denominator",
    "leg_ratio_qty_numerator",
    "leg_ratio_qty_denominator",
    "leg_underlying_id",
    "leg_raw_symbol",
    "leg_instrument_class",
    "leg_side",
    "leg_price",
    "leg_delta",
    # Live control and symbology records. Parent subscriptions cannot be
    # interpreted without retaining every mapping interval.
    "stype_in",
    "stype_in_symbol",
    "stype_out",
    "stype_out_symbol",
    "start_ts",
    "end_ts",
    "msg",
    "code",
    "err",
    "is_last",
)

_PRICE_FIELDS = frozenset(
    (
        "price",
        "strike_price",
        "trading_reference_price",
        "min_price_increment",
        "min_price_increment_amount",
        "display_factor",
        "high_limit_price",
        "low_limit_price",
        "max_price_variation",
        "unit_of_measure_qty",
        "price_ratio",
        "leg_price",
        "leg_delta",
    )
)

_RTYPE_SCHEMAS = {
    0: "trades",
    18: "status",
    19: "definition",
    24: "statistics",
    160: "mbo",
}


def _enum_value(value: Enum) -> Any:
    raw_value = value.value
    return raw_value if raw_value is not value else value.name


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Enum):
        return _json_safe(_enum_value(value))
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, float):
        return format(Decimal(str(value)), "f")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bytes):
        try:
            decoded = value.rstrip(b"\x00").decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
        return decoded if decoded.isprintable() else value.hex()
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            item = item_method()
        except (TypeError, ValueError):
            pass
        else:
            if item is not value:
                return _json_safe(item)
    return str(value)


def _object_mapping(record: Any) -> Dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
        except (TypeError, ValueError):
            result = None
        if isinstance(result, Mapping):
            return dict(result)
    if is_dataclass(record):
        return asdict(record)
    values = getattr(record, "__dict__", None)
    if isinstance(values, Mapping):
        return dict(values)

    # Current databento-dbn records are slot-based extension classes. Walk the
    # complete MRO rather than relying on __dict__ or a version-specific field
    # allowlist. Native DBN remains the authoritative lossless representation;
    # this path preserves every field exposed by the installed Python binding.
    result: Dict[str, Any] = {}
    for record_type in type(record).__mro__:
        slots = getattr(record_type, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for field in slots:
            if field in ("__dict__", "__weakref__") or field.startswith("_"):
                continue
            if hasattr(record, field):
                result[field] = getattr(record, field)
    return result


def _attribute(record: Any, field: str) -> Any:
    if isinstance(record, Mapping) and field in record:
        return record[field]
    if hasattr(record, field):
        return getattr(record, field)
    header = getattr(record, "hd", None)
    if header is not None:
        if isinstance(header, Mapping) and field in header:
            return header[field]
        if hasattr(header, field):
            return getattr(header, field)
    return None


def _raw_fields(record: Any) -> Dict[str, Any]:
    values = _object_mapping(record)
    nested_header = values.get("hd", getattr(record, "hd", None))
    header_values = _object_mapping(nested_header) if nested_header is not None else {}
    if header_values:
        for field in _HEADER_FIELDS:
            if field in header_values:
                values.setdefault(field, header_values[field])

    # Native DBN records expose fixed-point integers and header fields as
    # attributes even when their ``to_dict`` representation is prettified.
    for field in _HEADER_FIELDS + _RECORD_FIELDS:
        value = _attribute(record, field)
        if value is not None:
            values[field] = value
    values.pop("hd", None)
    return {
        str(key): _json_safe(value)
        for key, value in sorted(values.items(), key=lambda pair: str(pair[0]))
        if not str(key).startswith("_")
    }


def _schema_for_record(record: Any, raw: Mapping[str, Any]) -> str:
    explicit = raw.get("schema")
    if isinstance(explicit, str) and explicit in SUPPORTED_SCHEMAS + CONTROL_SCHEMAS:
        return explicit
    name = type(record).__name__.lower()
    if "symbolmapping" in name or "symbol_mapping" in name:
        return "symbol_mapping"
    if "errormsg" in name or name in ("errmsg", "error"):
        return "error"
    if "systemmsg" in name or name == "system":
        return "system"
    if "instrumentdef" in name or "definition" in name:
        return "definition"
    if "status" in name:
        return "status"
    if name.startswith("stat") or "statistics" in name:
        return "statistics"
    if "trade" in name:
        return "trades"
    if "mbo" in name:
        return "mbo"
    rtype = raw.get("rtype")
    if isinstance(rtype, int):
        return _RTYPE_SCHEMAS.get(rtype, "unknown")
    return "unknown"


def _fixed_price(value: Any) -> Tuple[Optional[str], Optional[int]]:
    if value is None:
        return None, None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a Databento fixed-point price")
    if isinstance(value, int):
        if value == UNDEFINED_PRICE:
            return None, value
        decimal_value = Decimal(value).scaleb(-FIXED_PRICE_SCALE)
        return format(decimal_value, "f"), value
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("Invalid Databento price: {!r}".format(value)) from error
    return format(decimal_value, "f"), None


def normalize_databento_record(
    record: Any,
    dataset: str = GLBX_DATASET,
    schema: Optional[str] = None,
    local_receive_ts_ns: Optional[int] = None,
    local_monotonic_ts_ns: Optional[int] = None,
) -> Dict[str, Any]:
    """Normalize one DBN record to a deterministic, JSON-safe dictionary.

    Native Databento price integers are retained as ``<field>_raw`` and also
    rendered as exact base-10 strings using Databento's 1e-9 price scale. Every
    field exposed by the installed Python binding is retained under
    ``raw_record``. Callers should also use :meth:`Live.add_stream` to preserve
    the authoritative native DBN bytes.
    """

    raw = _raw_fields(record)
    detected_schema = schema or _schema_for_record(record, raw)
    if (
        detected_schema not in SUPPORTED_SCHEMAS
        and detected_schema not in CONTROL_SCHEMAS
        and detected_schema != "unknown"
    ):
        raise DatabentoConfigurationError(
            "Unsupported Databento schema: {!r}".format(detected_schema)
        )

    payload = dict(raw)
    for field in _PRICE_FIELDS:
        if field not in raw:
            continue
        decimal_value, raw_value = _fixed_price(raw[field])
        payload[field] = decimal_value
        if raw_value is not None:
            payload[field + "_raw"] = raw_value

    normalized: Dict[str, Any] = {
        "source": "databento",
        "dataset": dataset,
        "schema": detected_schema,
        "record_type": type(record).__name__,
        "fixed_price_scale": FIXED_PRICE_SCALE,
        "local_receive_ts_ns": local_receive_ts_ns,
        "local_monotonic_ts_ns": local_monotonic_ts_ns,
    }
    for key in sorted(payload):
        if key not in normalized:
            normalized[key] = payload[key]
    normalized["raw_record"] = raw
    return normalized


def _import_databento() -> Any:
    try:
        import databento
    except ImportError as error:  # pragma: no cover - environment-dependent path
        raise DatabentoUnavailableError(
            "Databento live collection requires the optional 'databento' package"
        ) from error
    return databento


def _timestamp_ns(value: Any) -> Optional[int]:
    """Convert reconnect bounds without losing integer nanoseconds."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    native_value = getattr(value, "value", None)
    if isinstance(native_value, int):
        return native_value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return None
        utc_value = value.astimezone(timezone.utc)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        delta = utc_value - epoch
        return (
            delta.days * 86_400 * 1_000_000_000
            + delta.seconds * 1_000_000_000
            + delta.microseconds * 1_000
        )
    return None


class DatabentoLiveCollector:
    """Configure and consume a callback- or async-friendly live DBN session."""

    def __init__(
        self,
        api_key: str,
        subscriptions: Sequence[DatabentoSubscription],
        client: Optional[Any] = None,
        client_factory: Optional[Callable[..., Any]] = None,
        reconnect_policy: str = "reconnect",
        include_ts_out: bool = True,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise DatabentoConfigurationError(
                "A non-empty Databento API key must be supplied by the caller"
            )
        if not subscriptions:
            raise DatabentoConfigurationError(
                "At least one Databento subscription is required"
            )
        datasets = {subscription.dataset for subscription in subscriptions}
        if len(datasets) != 1:
            raise DatabentoConfigurationError(
                "A Databento live session may use only one dataset"
            )
        if client is not None and client_factory is not None:
            raise DatabentoConfigurationError(
                "Provide either client or client_factory, not both"
            )

        self._api_key = api_key.strip()
        self.subscriptions = tuple(subscriptions)
        self.dataset = next(iter(datasets))
        self._client = client
        self._client_factory = client_factory
        self.reconnect_policy = reconnect_policy
        self.include_ts_out = include_ts_out
        self._subscribed = False
        self._subscription_ids: Tuple[Any, ...] = tuple()

    def __repr__(self) -> str:
        return (
            "DatabentoLiveCollector(dataset={!r}, subscriptions={}, "
            "api_key=<redacted>)"
        ).format(self.dataset, len(self.subscriptions))

    @property
    def client(self) -> Any:
        if self._client is None:
            factory = self._client_factory
            if factory is None:
                factory = _import_databento().Live
            self._client = factory(
                key=self._api_key,
                reconnect_policy=self.reconnect_policy,
                ts_out=self.include_ts_out,
            )
        return self._client

    def subscribe(self) -> Tuple[Any, ...]:
        """Register every configured schema once and return subscription IDs."""

        if self._subscribed:
            return self._subscription_ids
        self._subscription_ids = tuple(
            self.client.subscribe(**subscription.subscribe_kwargs())
            for subscription in self.subscriptions
        )
        self._subscribed = True
        return self._subscription_ids

    def _normalize_received(self, record: Any) -> Dict[str, Any]:
        return self.normalize_record(
            record,
            local_receive_ts_ns=time.time_ns(),
            local_monotonic_ts_ns=time.monotonic_ns(),
        )

    def normalize_record(
        self,
        record: Any,
        local_receive_ts_ns: int,
        local_monotonic_ts_ns: int,
    ) -> Dict[str, Any]:
        return normalize_databento_record(
            record,
            dataset=self.dataset,
            local_receive_ts_ns=local_receive_ts_ns,
            local_monotonic_ts_ns=local_monotonic_ts_ns,
        )

    async def records(self) -> AsyncIterator[Dict[str, Any]]:
        """Asynchronously yield normalized records until the session closes."""

        self.subscribe()
        async for record in self.client:
            yield self._normalize_received(record)

    async def collect(
        self,
        callback: NormalizedRecordCallback,
        max_records: Optional[int] = None,
    ) -> int:
        """Feed normalized records to a synchronous or asynchronous callback."""

        if max_records is not None and max_records < 1:
            raise DatabentoConfigurationError("max_records must be at least one")
        count = 0
        async for record in self.records():
            result = callback(record)
            if inspect.isawaitable(result):
                await result
            count += 1
            if max_records is not None and count >= max_records:
                self.stop()
                break
        return count

    def start_callbacks(
        self,
        callback: Callable[[Dict[str, Any]], None],
        exception_callback: Optional[Callable[[Exception], None]] = None,
        reconnect_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        defer_normalization: bool = False,
    ) -> None:
        """Start a native callback session after installing normalization."""

        self.subscribe()

        def normalized_callback(record: Any) -> None:
            local_receive_ts_ns = time.time_ns()
            local_monotonic_ts_ns = time.monotonic_ns()
            if defer_normalization:
                callback(
                    DeferredDatabentoRecord(
                        record=record,
                        local_receive_ts_ns=local_receive_ts_ns,
                        local_monotonic_ts_ns=local_monotonic_ts_ns,
                    )
                )
            else:
                callback(
                    self.normalize_record(
                        record,
                        local_receive_ts_ns=local_receive_ts_ns,
                        local_monotonic_ts_ns=local_monotonic_ts_ns,
                    )
                )

        if exception_callback is None:
            self.client.add_callback(record_callback=normalized_callback)
        else:
            self.client.add_callback(
                record_callback=normalized_callback,
                exception_callback=exception_callback,
            )
        if reconnect_callback is not None:
            def normalized_reconnect(start: Any, end: Any) -> None:
                reconnect_callback(
                    {
                        "source": "databento",
                        "dataset": self.dataset,
                        "schema": "reconnect",
                        "record_type": "ReconnectGap",
                        "start_ts_ns": _timestamp_ns(start),
                        "end_ts_ns": _timestamp_ns(end),
                        "local_receive_ts_ns": time.time_ns(),
                        "local_monotonic_ts_ns": time.monotonic_ns(),
                        "raw_record": {
                            "start": _json_safe(start),
                            "end": _json_safe(end),
                        },
                    }
                )

            self.client.add_reconnect_callback(
                reconnect_callback=normalized_reconnect,
                exception_callback=exception_callback,
            )
        self.client.start()

    def add_stream(
        self,
        stream: Any,
        exception_callback: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """Install an authoritative native DBN output before starting."""

        self.client.add_stream(stream=stream, exception_callback=exception_callback)

    def stop(self) -> None:
        self.client.stop()

    def terminate(self) -> None:
        self.client.terminate()

    def block_for_close(self, timeout: Optional[float] = None) -> None:
        self.client.block_for_close(timeout=timeout)

    async def wait_for_close(self) -> None:
        await self.client.wait_for_close()
