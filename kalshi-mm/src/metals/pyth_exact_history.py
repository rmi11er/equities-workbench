"""Exact, resumable Pyth Pro price-range archives.

This module is intentionally independent from live collection and order entry.
It implements the documented Pyth Pro History request
``GET /fixed_rate@200ms/price/range`` using numeric feed IDs and inclusive
microsecond bounds.  Successful response bodies are retained byte-for-byte;
HTTP error bodies and third-party exception messages are never persisted or
included in raised errors.

Every request is content addressed.  A deterministic checkpoint binds each raw
page to its normalized, hash-chained projection.  Resume verifies the request,
all page hashes, normalization, timestamps, feed IDs, cursor chain, and any
checkpoint state before issuing another HTTP request.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Mapping, MutableSet, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from .historical_research_assumption import require_historical_operation
from .storage import update_run_manifest, write_run_manifest

PYTH_EXACT_HISTORY_BASE_URL = "https://pyth.dourolabs.app/v1"
PYTH_EXACT_HISTORY_CHANNEL = "fixed_rate@200ms"
PYTH_EXACT_HISTORY_MAX_FEEDS = 100
PYTH_EXACT_HISTORY_MAX_LIMIT = 1_000
PYTH_EXACT_HISTORY_MAX_WINDOW_US = 60_000_000
PYTH_EXACT_HISTORY_DEFAULT_MAX_PAGES = 1_000
PYTH_EXACT_HISTORY_MAX_CURSOR_BYTES = 8_192

_UINT8_MAX = 2**8 - 1
_UINT16_MAX = 2**16 - 1
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1
_INT16_MIN = -(2**15)
_INT16_MAX = 2**15 - 1
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_SAFE_METAL = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_RAW_PAGE = re.compile(r"^page-(\d{6})\.body\.json$")
_NORMALIZED_PAGE = re.compile(r"^page-(\d{6})\.jsonl$")


class PythExactHistoryError(RuntimeError):
    """Base class for exact-history failures."""


class PythExactHistoryAccessError(PythExactHistoryError):
    """Authentication, entitlement, rate-limit, or transport failure."""


class PythExactHistoryProtocolError(PythExactHistoryError):
    """A successful response contradicted the documented protocol."""


class PythExactHistoryIntegrityError(PythExactHistoryError):
    """A request, checkpoint, or archived artifact failed verification."""


def _require_integer(
    value: Any,
    field: str,
    *,
    minimum: int,
    maximum: int,
    error_type: type[Exception] = PythExactHistoryProtocolError,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error_type(f"{field} must be a numeric integer")
    if value < minimum or value > maximum:
        raise error_type(f"{field} is outside its documented integer range")
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Some filesystems do not implement directory fsync.  Files are still
        # individually fsynced and atomically published.
        pass


def _write_temporary(path: Path, value: bytes, prefix: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=prefix, suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return temporary_path


def _atomic_replace(path: Path, value: bytes) -> None:
    temporary_path = _write_temporary(path, value, ".pyth-exact-state-")
    try:
        os.replace(str(temporary_path), str(path))
        _fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_create_or_verify(path: Path, value: bytes, label: str) -> None:
    """Create an immutable artifact or verify an interrupted identical write."""

    temporary_path = _write_temporary(path, value, ".pyth-exact-page-")
    try:
        try:
            os.link(str(temporary_path), str(path))
            _fsync_directory(path.parent)
        except FileExistsError:
            if path.read_bytes() != value:
                raise PythExactHistoryIntegrityError(
                    f"Existing {label} failed immutable-content verification"
                ) from None
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PythExactHistoryProtocolError(
                "Pyth JSON contains a duplicate object field"
            )
        result[key] = value
    return result


def _reject_json_constant(_: str) -> None:
    raise PythExactHistoryProtocolError("Pyth JSON contains a non-finite number")


def _parse_json_bytes(
    value: bytes,
    *,
    error_type: type[Exception] = PythExactHistoryProtocolError,
) -> Any:
    if not isinstance(value, bytes):
        raise error_type("Archived Pyth body must be bytes")
    try:
        text = value.decode("utf-8")
        return json.loads(
            text,
            parse_float=Decimal,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_object,
        )
    except PythExactHistoryProtocolError as error:
        if error_type is PythExactHistoryProtocolError:
            raise
        raise error_type("Archived Pyth JSON failed strict parsing") from error
    except (UnicodeDecodeError, json.JSONDecodeError, InvalidOperation) as error:
        raise error_type("Pyth returned malformed UTF-8 JSON") from error


def _scaled_decimal_text(mantissa: int, exponent: int) -> str:
    try:
        return format(Decimal(mantissa).scaleb(exponent), "f")
    except (InvalidOperation, ValueError) as error:  # pragma: no cover - defensive
        raise PythExactHistoryProtocolError("Pyth price cannot be scaled") from error


def _validate_endpoint(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Pyth History base URL must be a non-empty HTTPS URL")
    endpoint = value.rstrip("/")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Pyth History base URL must be HTTPS and contain no credentials, query, or fragment"
        )
    return endpoint


@dataclass(frozen=True)
class PythExactHistoryRequest:
    """One explicit, bounded, content-addressed Pyth price-range request."""

    metal: str
    feed_ids: tuple[int, ...]
    start_timestamp_us: int
    end_timestamp_us: int
    limit: int = PYTH_EXACT_HISTORY_MAX_LIMIT
    max_pages: int = PYTH_EXACT_HISTORY_DEFAULT_MAX_PAGES
    base_url: str = PYTH_EXACT_HISTORY_BASE_URL
    channel: str = PYTH_EXACT_HISTORY_CHANNEL

    def __post_init__(self) -> None:
        metal = self.metal.strip().lower() if isinstance(self.metal, str) else ""
        if not _SAFE_METAL.fullmatch(metal):
            raise ValueError("metal must be a non-empty lowercase path-safe label")

        if isinstance(self.feed_ids, (str, bytes)):
            raise TypeError("feed_ids must be a sequence of numeric feed IDs")
        try:
            feed_ids = tuple(self.feed_ids)
        except TypeError as error:
            raise ValueError(
                "feed_ids must be a sequence of numeric feed IDs"
            ) from error
        if not feed_ids:
            raise ValueError("At least one numeric Pyth feed ID is required")
        if len(feed_ids) > PYTH_EXACT_HISTORY_MAX_FEEDS:
            raise ValueError(
                f"Pyth price/range accepts at most {PYTH_EXACT_HISTORY_MAX_FEEDS} feed IDs"
            )
        normalized_feed_ids = tuple(
            _require_integer(
                value,
                f"feed_ids[{index}]",
                minimum=0,
                maximum=_UINT32_MAX,
                error_type=ValueError,
            )
            for index, value in enumerate(feed_ids)
        )
        if len(set(normalized_feed_ids)) != len(normalized_feed_ids):
            raise ValueError("Pyth feed IDs must be unique")

        start = _require_integer(
            self.start_timestamp_us,
            "start_timestamp_us",
            minimum=0,
            maximum=_UINT64_MAX,
            error_type=ValueError,
        )
        end = _require_integer(
            self.end_timestamp_us,
            "end_timestamp_us",
            minimum=0,
            maximum=_UINT64_MAX,
            error_type=ValueError,
        )
        if end < start:
            raise ValueError("end_timestamp_us must be at or after start_timestamp_us")
        if end - start > PYTH_EXACT_HISTORY_MAX_WINDOW_US:
            raise ValueError("Pyth exact-history windows may not exceed 60 seconds")

        limit = _require_integer(
            self.limit,
            "limit",
            minimum=1,
            maximum=PYTH_EXACT_HISTORY_MAX_LIMIT,
            error_type=ValueError,
        )
        max_pages = _require_integer(
            self.max_pages,
            "max_pages",
            minimum=1,
            maximum=1_000_000,
            error_type=ValueError,
        )
        if self.channel != PYTH_EXACT_HISTORY_CHANNEL:
            raise ValueError(
                f"Exact settlement history is locked to {PYTH_EXACT_HISTORY_CHANNEL}"
            )

        object.__setattr__(self, "metal", metal)
        object.__setattr__(self, "feed_ids", tuple(sorted(normalized_feed_ids)))
        object.__setattr__(self, "start_timestamp_us", start)
        object.__setattr__(self, "end_timestamp_us", end)
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "max_pages", max_pages)
        object.__setattr__(self, "base_url", _validate_endpoint(self.base_url))

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.channel}/price/range"

    def public_dict(self) -> dict[str, Any]:
        return {
            "source": "pyth_pro_history",
            "endpoint": self.url,
            "method": "GET",
            "channel": self.channel,
            "metal": self.metal,
            "feed_ids": list(self.feed_ids),
            "start_timestamp_us": self.start_timestamp_us,
            "end_timestamp_us": self.end_timestamp_us,
            "timestamp_bounds": "inclusive_microseconds",
            "limit": self.limit,
            "max_pages": self.max_pages,
        }

    @property
    def request_id(self) -> str:
        canonical = json.dumps(
            self.public_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def query_params(self, after: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "ids": list(self.feed_ids),
            "start_timestamp": self.start_timestamp_us,
            "end_timestamp": self.end_timestamp_us,
            "limit": self.limit,
        }
        if after is not None:
            params["after"] = after
        return params


@dataclass(frozen=True)
class PythExactHistoryResult:
    request_id: str
    completed: bool
    page_count: int
    row_count: int
    requests_made: int
    resumed_pages: int
    raw_paths: tuple[Path, ...]
    normalized_paths: tuple[Path, ...]
    checkpoint_path: Path

    def public_dict(self) -> dict[str, Any]:
        """Return a CLI-ready JSON-serializable result without credentials."""

        return {
            "request_id": self.request_id,
            "completed": self.completed,
            "page_count": self.page_count,
            "row_count": self.row_count,
            "requests_made": self.requests_made,
            "resumed_pages": self.resumed_pages,
            "raw_paths": [str(path) for path in self.raw_paths],
            "normalized_paths": [str(path) for path in self.normalized_paths],
            "checkpoint_path": str(self.checkpoint_path),
        }


@dataclass(frozen=True)
class PythExactHistoryRunResult:
    archive: PythExactHistoryResult
    run_id: str
    manifest_path: Path

    def public_dict(self) -> dict[str, Any]:
        return {
            **self.archive.public_dict(),
            "run_id": self.run_id,
            "manifest_path": str(self.manifest_path),
        }


@dataclass(frozen=True)
class _ParsedPage:
    rows: tuple[dict[str, Any], ...]
    row_keys: tuple[tuple[int, int], ...]
    next_after: str | None


def _optional_integer(
    row: Mapping[str, Any],
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    value = row.get(field)
    if value is None:
        return None
    return _require_integer(value, field, minimum=minimum, maximum=maximum)


def normalize_pyth_exact_price(
    value: Mapping[str, Any],
    *,
    request: PythExactHistoryRequest,
) -> tuple[dict[str, Any], tuple[int, int]]:
    """Validate and normalize one documented ``PriceResponse`` object."""

    if not isinstance(value, Mapping):
        raise PythExactHistoryProtocolError("Pyth price entry must be an object")

    channel_number = _require_integer(
        value.get("channel"), "channel", minimum=0, maximum=_UINT8_MAX
    )
    channel_name = value.get("channel_name")
    if channel_name != request.channel:
        raise PythExactHistoryProtocolError(
            "Pyth price entry channel does not match the requested channel"
        )
    feed_id = _require_integer(
        value.get("price_feed_id"),
        "price_feed_id",
        minimum=0,
        maximum=_UINT32_MAX,
    )
    if feed_id not in request.feed_ids:
        raise PythExactHistoryProtocolError(
            "Pyth returned a non-requested price feed ID"
        )
    publish_time = _require_integer(
        value.get("publish_time"),
        "publish_time",
        minimum=0,
        maximum=_UINT64_MAX,
    )
    if not request.start_timestamp_us <= publish_time <= request.end_timestamp_us:
        raise PythExactHistoryProtocolError(
            "Pyth publish_time is outside the inclusive requested window"
        )
    price_mantissa = _require_integer(
        value.get("price"), "price", minimum=_INT64_MIN, maximum=_INT64_MAX
    )
    if price_mantissa == 0:
        raise PythExactHistoryProtocolError("Pyth price must be non-zero")
    exponent = _require_integer(
        value.get("exponent"),
        "exponent",
        minimum=_INT16_MIN,
        maximum=_INT16_MAX,
    )
    feed_update_timestamp = _require_integer(
        value.get("feed_update_timestamp"),
        "feed_update_timestamp",
        minimum=0,
        maximum=_UINT64_MAX,
    )
    if feed_update_timestamp > publish_time:
        raise PythExactHistoryProtocolError(
            "Pyth feed_update_timestamp is later than publish_time"
        )

    optional_signed = {
        field: _optional_integer(value, field, minimum=_INT64_MIN, maximum=_INT64_MAX)
        for field in (
            "best_bid_price",
            "best_ask_price",
            "confidence",
            "ema_price",
            "ema_confidence",
            "funding_rate",
        )
    }
    publisher_count = _optional_integer(
        value, "publisher_count", minimum=0, maximum=_UINT16_MAX
    )
    funding_timestamp = _optional_integer(
        value, "funding_timestamp", minimum=0, maximum=_UINT64_MAX
    )
    funding_rate_interval = _optional_integer(
        value, "funding_rate_interval", minimum=0, maximum=_UINT64_MAX
    )
    market_session = value.get("market_session")
    if market_session is not None and (
        not isinstance(market_session, str) or not market_session
    ):
        raise PythExactHistoryProtocolError(
            "market_session must be a non-empty string or null"
        )

    normalized: dict[str, Any] = {
        "source": "pyth_pro_history",
        "stream": "exact_price_range",
        "metal": request.metal,
        "historical_request_id": request.request_id,
        "channel": request.channel,
        "channel_number": channel_number,
        "price_feed_id": feed_id,
        "price_mantissa": str(price_mantissa),
        "exponent": exponent,
        "price": _scaled_decimal_text(price_mantissa, exponent),
        "publish_timestamp_us": publish_time,
        "feed_update_timestamp_us": feed_update_timestamp,
        "source_event_ts_ns": publish_time * 1_000,
        "source_publish_ts_ns": feed_update_timestamp * 1_000,
        "is_carried_forward": feed_update_timestamp < publish_time,
        "freshness_age_us": publish_time - feed_update_timestamp,
        "timestamp_semantics": "publish_time_and_feed_update_timestamp_microseconds",
        "request_bounds_semantics": "inclusive_microseconds",
        "publisher_count": publisher_count,
        "market_session": market_session,
        "funding_timestamp_us": funding_timestamp,
        "funding_rate_interval_us": funding_rate_interval,
    }
    for field, item in optional_signed.items():
        normalized[f"{field}_mantissa"] = str(item) if item is not None else None
        if item is not None and field != "funding_rate":
            normalized[field] = _scaled_decimal_text(item, exponent)
        elif item is None:
            normalized[field] = None
    return normalized, (feed_id, publish_time)


def _parse_page(
    raw_body: bytes,
    *,
    request: PythExactHistoryRequest,
    existing_row_keys: MutableSet[tuple[int, int]],
) -> _ParsedPage:
    payload = _parse_json_bytes(raw_body)
    if not isinstance(payload, Mapping):
        raise PythExactHistoryProtocolError("Pyth page must be a JSON object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise PythExactHistoryProtocolError("Pyth page data must be a JSON array")
    if len(data) > request.limit:
        raise PythExactHistoryProtocolError(
            "Pyth page contains more rows than the requested limit"
        )

    rows: list[dict[str, Any]] = []
    page_keys: MutableSet[tuple[int, int]] = set()
    ordered_keys: list[tuple[int, int]] = []
    for value in data:
        normalized, key = normalize_pyth_exact_price(value, request=request)
        if key in existing_row_keys or key in page_keys:
            raise PythExactHistoryProtocolError(
                "Pyth returned a duplicate feed ID/publish timestamp"
            )
        page_keys.add(key)
        ordered_keys.append(key)
        rows.append(normalized)

    next_after = payload.get("next")
    if next_after is not None:
        if not isinstance(next_after, str) or not next_after:
            raise PythExactHistoryProtocolError(
                "Pyth next cursor must be a non-empty opaque string or null"
            )
        if len(next_after.encode("utf-8")) > PYTH_EXACT_HISTORY_MAX_CURSOR_BYTES:
            raise PythExactHistoryProtocolError("Pyth next cursor is too large")
        if not rows:
            raise PythExactHistoryProtocolError(
                "Pyth returned an empty page with a continuation cursor"
            )
    return _ParsedPage(tuple(rows), tuple(ordered_keys), next_after)


def _encode_normalized_page(
    parsed: _ParsedPage,
    *,
    request: PythExactHistoryRequest,
    page_index: int,
    raw_sha256: str,
    previous_record_sha256: str | None,
) -> tuple[bytes, str | None]:
    lines: list[bytes] = []
    previous = previous_record_sha256
    for row_index, value in enumerate(parsed.rows):
        row = dict(value)
        row.update(
            {
                "page_index": page_index,
                "page_row_index": row_index,
                "raw_page_sha256": raw_sha256,
                "previous_record_sha256": previous,
            }
        )
        canonical = json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        record_hash = hashlib.sha256(canonical).hexdigest()
        row["record_sha256"] = record_hash
        lines.append(
            (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        )
        previous = record_hash
    return b"".join(lines), previous


def _safe_http_error(status: int) -> PythExactHistoryAccessError:
    if status == 401:
        detail = "authentication failed; verify the configured Pyth Pro API key"
    elif status == 403:
        detail = "the API key is not entitled to the requested Pyth Pro feed/channel"
    elif status == 404:
        detail = "a requested Pyth Pro feed ID was not found"
    elif status == 429:
        detail = "the Pyth Pro History rate limit was reached; retry later"
    elif status >= 500:
        detail = "the Pyth Pro History service is unavailable; retry later"
    elif status == 400:
        detail = "Pyth rejected the bounded price-range request"
    else:
        detail = f"Pyth Pro History returned HTTP {status}"
    return PythExactHistoryAccessError(
        f"Unable to retrieve exact Pyth history: {detail}."
    )


class PythExactHistoryArchive:
    """Download, verify, and resume one exact Pyth price-range request."""

    def __init__(
        self,
        root: Path,
        *,
        api_key: str,
        session: Any | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise PythExactHistoryAccessError(
                "A non-empty Pyth Pro API key is required"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0 < float(timeout_seconds) <= 60
        ):
            raise ValueError("timeout_seconds must be in (0, 60]")
        self.root = Path(root)
        self._api_key = api_key.strip()
        self._session = session
        self._timeout_seconds = float(timeout_seconds)
        self.network_requests_attempted = 0

    def _base_path(self, request: PythExactHistoryRequest, tier: str) -> Path:
        return (
            self.root
            / tier
            / "pyth_exact_history"
            / f"metal={request.metal}"
            / f"request={request.request_id[:24]}"
        )

    def _raw_path(self, request: PythExactHistoryRequest, index: int) -> Path:
        return self._base_path(request, "raw") / f"page-{index:06d}.body.json"

    def _normalized_path(self, request: PythExactHistoryRequest, index: int) -> Path:
        return self._base_path(request, "normalized") / f"page-{index:06d}.jsonl"

    def _state_dir(self, request: PythExactHistoryRequest) -> Path:
        return (
            self.root
            / "checkpoints"
            / "pyth_exact_history"
            / f"request={request.request_id}"
        )

    def _request_path(self, request: PythExactHistoryRequest) -> Path:
        return self._state_dir(request) / "request.json"

    def _checkpoint_path(self, request: PythExactHistoryRequest) -> Path:
        return self._state_dir(request) / "checkpoint.json"

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError as error:  # pragma: no cover - paths are internal
            raise PythExactHistoryIntegrityError(
                "Pyth archive path escaped its storage root"
            ) from error

    def _empty_state(
        self, request: PythExactHistoryRequest, request_sha256: str
    ) -> dict[str, Any]:
        return {
            "checkpoint_version": 1,
            "request_id": request.request_id,
            "request_sha256": request_sha256,
            "completed": False,
            "next_after": None,
            "page_count": 0,
            "row_count": 0,
            "last_record_sha256": None,
            "pages": [],
        }

    def _state_from_pages(
        self,
        request: PythExactHistoryRequest,
        request_sha256: str,
        pages: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not pages:
            return self._empty_state(request, request_sha256)
        last = pages[-1]
        return {
            "checkpoint_version": 1,
            "request_id": request.request_id,
            "request_sha256": request_sha256,
            "completed": last["next_after"] is None,
            "next_after": last["next_after"],
            "page_count": len(pages),
            "row_count": sum(int(page["row_count"]) for page in pages),
            "last_record_sha256": last["last_record_sha256"],
            "pages": list(pages),
        }

    def _page_summary(
        self,
        request: PythExactHistoryRequest,
        *,
        index: int,
        request_after: str | None,
        next_after: str | None,
        raw_path: Path,
        normalized_path: Path,
        row_count: int,
        last_record_sha256: str | None,
    ) -> dict[str, Any]:
        return {
            "page_index": index,
            "request_after": request_after,
            "next_after": next_after,
            "row_count": row_count,
            "last_record_sha256": last_record_sha256,
            "raw": {
                "path": self._relative(raw_path),
                "bytes": raw_path.stat().st_size,
                "sha256": _sha256_path(raw_path),
            },
            "normalized": {
                "path": self._relative(normalized_path),
                "bytes": normalized_path.stat().st_size,
                "sha256": _sha256_path(normalized_path),
            },
        }

    @staticmethod
    def _read_checkpoint(path: Path) -> dict[str, Any]:
        try:
            value = _parse_json_bytes(
                path.read_bytes(), error_type=PythExactHistoryIntegrityError
            )
        except OSError as error:
            raise PythExactHistoryIntegrityError(
                "Unable to read the Pyth exact-history checkpoint"
            ) from error
        if not isinstance(value, dict):
            raise PythExactHistoryIntegrityError(
                "Pyth exact-history checkpoint must be a JSON object"
            )
        return value

    @staticmethod
    def _artifact_indexes(path: Path, pattern: re.Pattern[str]) -> tuple[int, ...]:
        if not path.exists():
            return ()
        if not path.is_dir():
            raise PythExactHistoryIntegrityError(
                "Pyth exact-history artifact directory is not a directory"
            )
        indexes: list[int] = []
        for candidate in path.iterdir():
            if candidate.name.startswith("."):
                continue
            match = pattern.fullmatch(candidate.name)
            if match is None or not candidate.is_file():
                raise PythExactHistoryIntegrityError(
                    "Pyth exact-history archive contains an unexpected artifact"
                )
            indexes.append(int(match.group(1)))
        return tuple(sorted(indexes))

    def _prepare_and_verify(
        self,
        request: PythExactHistoryRequest,
        *,
        recover_interrupted: bool = True,
    ) -> tuple[dict[str, Any], MutableSet[tuple[int, int]]]:
        request_bytes = _canonical_json_bytes(
            {"request_id": request.request_id, "request": request.public_dict()}
        )
        request_sha256 = _sha256_bytes(request_bytes)
        request_path = self._request_path(request)
        checkpoint_path = self._checkpoint_path(request)
        raw_dir = self._base_path(request, "raw")
        normalized_dir = self._base_path(request, "normalized")

        existing_artifacts = raw_dir.exists() or normalized_dir.exists()
        if request_path.exists():
            if request_path.read_bytes() != request_bytes:
                raise PythExactHistoryIntegrityError(
                    "Pyth exact-history request fingerprint failed verification"
                )
        elif checkpoint_path.exists() or existing_artifacts:
            raise PythExactHistoryIntegrityError(
                "Pyth exact-history request metadata is missing"
            )
        else:
            _atomic_create_or_verify(request_path, request_bytes, "request metadata")

        initial = self._empty_state(request, request_sha256)
        if checkpoint_path.exists():
            checkpoint = self._read_checkpoint(checkpoint_path)
        elif existing_artifacts:
            raise PythExactHistoryIntegrityError(
                "Pyth exact-history checkpoint is missing"
            )
        else:
            checkpoint = initial
            _atomic_replace(checkpoint_path, _canonical_json_bytes(checkpoint))

        checkpoint_page_count = checkpoint.get("page_count")
        if (
            isinstance(checkpoint_page_count, bool)
            or not isinstance(checkpoint_page_count, int)
            or checkpoint_page_count < 0
            or checkpoint_page_count > request.max_pages
        ):
            raise PythExactHistoryIntegrityError(
                "Pyth exact-history checkpoint page count is invalid"
            )

        raw_indexes = self._artifact_indexes(raw_dir, _RAW_PAGE)
        normalized_indexes = self._artifact_indexes(normalized_dir, _NORMALIZED_PAGE)
        committed = set(range(checkpoint_page_count))
        raw_set = set(raw_indexes)
        normalized_set = set(normalized_indexes)
        if not committed.issubset(raw_set) or not committed.issubset(normalized_set):
            raise PythExactHistoryIntegrityError(
                "A checkpointed Pyth page artifact is missing"
            )
        extra_raw = raw_set - committed
        extra_normalized = normalized_set - committed
        extra_indexes = extra_raw | extra_normalized
        if (
            extra_normalized - extra_raw
            or len(extra_indexes) > 1
            or (
                extra_indexes
                and (
                    extra_indexes != {checkpoint_page_count}
                    or checkpoint.get("completed") is True
                )
            )
        ):
            raise PythExactHistoryIntegrityError(
                "Pyth raw and normalized page sets contradict the checkpoint"
            )
        if raw_set != set(range(len(raw_set))):
            raise PythExactHistoryIntegrityError(
                "Pyth exact-history raw page indexes are not contiguous"
            )

        pages: list[dict[str, Any]] = []
        row_keys: MutableSet[tuple[int, int]] = set()
        used_after: MutableSet[str] = set()
        previous_hash: str | None = None
        request_after: str | None = None
        for index in raw_indexes:
            if request_after is not None:
                if request_after in used_after:
                    raise PythExactHistoryIntegrityError(
                        "Pyth exact-history cursor cycle was found in the archive"
                    )
                used_after.add(request_after)
            raw_path = self._raw_path(request, index)
            normalized_path = self._normalized_path(request, index)
            raw_body = raw_path.read_bytes()
            try:
                parsed = _parse_page(
                    raw_body, request=request, existing_row_keys=row_keys
                )
            except PythExactHistoryProtocolError as error:
                raise PythExactHistoryIntegrityError(
                    "Archived Pyth page failed protocol verification"
                ) from error
            if parsed.next_after is not None and parsed.next_after in used_after:
                raise PythExactHistoryIntegrityError(
                    "Pyth exact-history cursor cycle was found in the archive"
                )
            if request_after is None and index > 0:
                raise PythExactHistoryIntegrityError(
                    "A terminal Pyth page is followed by another page"
                )
            raw_hash = _sha256_bytes(raw_body)
            expected_normalized, last_hash = _encode_normalized_page(
                parsed,
                request=request,
                page_index=index,
                raw_sha256=raw_hash,
                previous_record_sha256=previous_hash,
            )
            if not normalized_path.exists():
                # The raw body is the authoritative source.  If the process
                # exited after atomically publishing it but before publishing
                # its deterministic projection, finish that one uncheckpointed
                # page locally.  Missing artifacts for checkpointed pages were
                # rejected above as tampering.
                if not recover_interrupted:
                    raise PythExactHistoryIntegrityError(
                        "An uncheckpointed Pyth page artifact is incomplete"
                    )
                _atomic_create_or_verify(
                    normalized_path,
                    expected_normalized,
                    "normalized Pyth page",
                )
            elif normalized_path.read_bytes() != expected_normalized:
                raise PythExactHistoryIntegrityError(
                    "Archived Pyth normalized page failed content verification"
                )
            pages.append(
                self._page_summary(
                    request,
                    index=index,
                    request_after=request_after,
                    next_after=parsed.next_after,
                    raw_path=raw_path,
                    normalized_path=normalized_path,
                    row_count=len(parsed.rows),
                    last_record_sha256=last_hash,
                )
            )
            row_keys.update(parsed.row_keys)
            previous_hash = last_hash
            request_after = parsed.next_after

        checkpoint_prefix = self._state_from_pages(
            request, request_sha256, pages[:checkpoint_page_count]
        )
        if checkpoint != checkpoint_prefix:
            raise PythExactHistoryIntegrityError(
                "Pyth exact-history checkpoint failed deterministic verification"
            )
        state = self._state_from_pages(request, request_sha256, pages)
        if len(pages) > checkpoint_page_count:
            # A page can be durably published just before a process exits and
            # just before its checkpoint replace.  Recover only after fully
            # re-validating both artifacts and the entire cursor/hash chain.
            if not recover_interrupted:
                raise PythExactHistoryIntegrityError(
                    "Pyth page artifacts extend beyond the checkpoint"
                )
            _atomic_replace(checkpoint_path, _canonical_json_bytes(state))
        return state, row_keys

    def verify_completed(
        self, request: PythExactHistoryRequest
    ) -> PythExactHistoryResult:
        """Read-only verification of a completed local request archive.

        Unlike ``download``, this method never repairs interrupted artifacts and
        never issues an HTTP request.  It is intended for downstream evidence
        consumers that must fail closed on partial request coverage.
        """

        if not self._request_path(request).is_file():
            raise PythExactHistoryIntegrityError(
                "Pyth exact-history request metadata is missing"
            )
        if not self._checkpoint_path(request).is_file():
            raise PythExactHistoryIntegrityError(
                "Pyth exact-history checkpoint is missing"
            )
        state, _ = self._prepare_and_verify(request, recover_interrupted=False)
        if state.get("completed") is not True:
            raise PythExactHistoryIntegrityError(
                "Pyth exact-history request is not completed"
            )
        return self._result(
            request,
            state,
            requests_made=0,
            resumed_pages=int(state["page_count"]),
        )

    def prepare_resume(
        self, request: PythExactHistoryRequest
    ) -> PythExactHistoryResult | None:
        """Verify and safely repair local resume state without network access.

        ``None`` means no local request state exists. Existing state is verified in
        full; the only writes allowed are deterministic recovery of an interrupted
        raw-page projection or checkpoint publication.
        """

        paths = (
            self._request_path(request),
            self._checkpoint_path(request),
            self._base_path(request, "raw"),
            self._base_path(request, "normalized"),
        )
        if not any(path.exists() for path in paths):
            return None
        state, _ = self._prepare_and_verify(request, recover_interrupted=True)
        return self._result(
            request,
            state,
            requests_made=0,
            resumed_pages=int(state["page_count"]),
        )

    async def _fetch_page(
        self,
        session: Any,
        request: PythExactHistoryRequest,
        after: str | None,
    ) -> bytes:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        }
        try:
            context = session.get(
                request.url,
                params=request.query_params(after),
                headers=headers,
            )
            async with context as response:
                status = int(response.status)
                raw_body = await response.read()
        except PythExactHistoryAccessError:
            raise
        except Exception as error:  # noqa: BLE001 - provider clients vary
            # Exception text may contain URLs or other provider context.
            raise PythExactHistoryAccessError(
                f"Unable to retrieve exact Pyth history: transport failed ({type(error).__name__})."
            ) from None
        if status != 200:
            # The body was read only so the connection can be reused.  It is
            # deliberately neither parsed, persisted, nor included in errors.
            raise _safe_http_error(status)
        return raw_body

    async def download(
        self, request: PythExactHistoryRequest
    ) -> PythExactHistoryResult:
        """Resume and finish a request after verifying every local artifact."""

        state, row_keys = self._prepare_and_verify(request)
        resumed_pages = int(state["page_count"])
        requests_made = 0
        if state["completed"]:
            return self._result(
                request,
                state,
                requests_made=requests_made,
                resumed_pages=resumed_pages,
            )

        owns_session = self._session is None
        session = self._session
        if session is None:
            timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
            session = aiohttp.ClientSession(timeout=timeout, auto_decompress=False)
        try:
            used_after = {
                str(page["request_after"])
                for page in state["pages"]
                if page["request_after"] is not None
            }
            while not state["completed"]:
                page_index = int(state["page_count"])
                if page_index >= request.max_pages:
                    raise PythExactHistoryProtocolError(
                        "Pyth pagination exceeded the explicit max_pages bound"
                    )
                request_after = state["next_after"]
                if request_after is not None:
                    if request_after in used_after:
                        raise PythExactHistoryProtocolError(
                            "Pyth returned a pagination cursor cycle"
                        )
                    used_after.add(request_after)

                self.network_requests_attempted += 1
                requests_made += 1
                raw_body = await self._fetch_page(session, request, request_after)
                parsed = _parse_page(
                    raw_body, request=request, existing_row_keys=row_keys
                )
                if parsed.next_after is not None and parsed.next_after in used_after:
                    raise PythExactHistoryProtocolError(
                        "Pyth returned a pagination cursor cycle"
                    )

                raw_hash = _sha256_bytes(raw_body)
                normalized_body, last_hash = _encode_normalized_page(
                    parsed,
                    request=request,
                    page_index=page_index,
                    raw_sha256=raw_hash,
                    previous_record_sha256=state["last_record_sha256"],
                )
                raw_path = self._raw_path(request, page_index)
                normalized_path = self._normalized_path(request, page_index)
                _atomic_create_or_verify(raw_path, raw_body, "raw Pyth page")
                _atomic_create_or_verify(
                    normalized_path, normalized_body, "normalized Pyth page"
                )
                page = self._page_summary(
                    request,
                    index=page_index,
                    request_after=request_after,
                    next_after=parsed.next_after,
                    raw_path=raw_path,
                    normalized_path=normalized_path,
                    row_count=len(parsed.rows),
                    last_record_sha256=last_hash,
                )
                state = self._state_from_pages(
                    request,
                    str(state["request_sha256"]),
                    [*state["pages"], page],
                )
                _atomic_replace(
                    self._checkpoint_path(request), _canonical_json_bytes(state)
                )
                row_keys.update(parsed.row_keys)
        finally:
            if owns_session:
                await session.close()

        return self._result(
            request,
            state,
            requests_made=requests_made,
            resumed_pages=resumed_pages,
        )

    def _result(
        self,
        request: PythExactHistoryRequest,
        state: Mapping[str, Any],
        *,
        requests_made: int,
        resumed_pages: int,
    ) -> PythExactHistoryResult:
        page_count = int(state["page_count"])
        return PythExactHistoryResult(
            request_id=request.request_id,
            completed=bool(state["completed"]),
            page_count=page_count,
            row_count=int(state["row_count"]),
            requests_made=requests_made,
            resumed_pages=resumed_pages,
            raw_paths=tuple(
                self._raw_path(request, index) for index in range(page_count)
            ),
            normalized_paths=tuple(
                self._normalized_path(request, index) for index in range(page_count)
            ),
            checkpoint_path=self._checkpoint_path(request),
        )


async def backfill_pyth_exact_history(
    root: Path,
    request: PythExactHistoryRequest,
    *,
    api_key_env: str = "PYTH_API_KEY",
    environ: Mapping[str, str] | None = None,
    session: Any | None = None,
    timeout_seconds: float = 30.0,
    provider_approval_path: Path | str = "",
    historical_research_assumption_path: Path | str = "",
) -> PythExactHistoryResult:
    """CLI integration hook that resolves a secret only from its env name."""

    require_historical_operation(
        Path(root),
        provider_approval_path,
        historical_research_assumption_path,
        "pyth",
        "exact_history",
    )
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        raise ValueError("api_key_env must be a non-empty environment variable name")
    environment = os.environ if environ is None else environ
    api_key = environment.get(api_key_env, "")
    if not isinstance(api_key, str) or not api_key.strip():
        raise PythExactHistoryAccessError(
            f"Set {api_key_env} to an entitled Pyth Pro API key"
        )
    archive = PythExactHistoryArchive(
        root,
        api_key=api_key,
        session=session,
        timeout_seconds=timeout_seconds,
    )
    return await archive.download(request)


async def run_pyth_exact_history(
    root: Path,
    request: PythExactHistoryRequest,
    *,
    api_key_env: str = "PYTH_API_KEY",
    environ: Mapping[str, str] | None = None,
    session: Any | None = None,
    timeout_seconds: float = 30.0,
    provider_approval_path: Path | str = "",
    historical_research_assumption_path: Path | str = "",
) -> PythExactHistoryRunResult:
    """Run one bounded archive request with a terminal, secret-safe manifest."""

    authority = require_historical_operation(
        Path(root),
        provider_approval_path,
        historical_research_assumption_path,
        "pyth",
        "exact_history",
    )
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        raise ValueError("api_key_env must be a non-empty environment variable name")
    environment = os.environ if environ is None else environ
    api_key = environment.get(api_key_env, "")
    if not isinstance(api_key, str) or not api_key.strip():
        raise PythExactHistoryAccessError(
            f"Set {api_key_env} to an entitled Pyth Pro API key"
        )

    storage_root = Path(root)
    run_id = str(uuid.uuid4())
    public_config = {
        "collector": {
            "component": "pyth_exact_history",
            "read_only": True,
            "api_key_env": api_key_env,
            "timeout_seconds": timeout_seconds,
            **authority.manifest_fields(),
        },
        "request": request.public_dict(),
        "request_id": request.request_id,
    }
    manifest_path = write_run_manifest(
        storage_root,
        run_id,
        public_config,
        output_paths=authority.output_paths,
    )
    archive = PythExactHistoryArchive(
        storage_root,
        api_key=api_key,
        session=session,
        timeout_seconds=timeout_seconds,
    )
    try:
        result = await archive.download(request)
    except BaseException as error:
        candidate_paths = [
            archive._request_path(request),
            archive._checkpoint_path(request),
            *sorted(archive._base_path(request, "raw").glob("page-*.body.json")),
            *sorted(archive._base_path(request, "normalized").glob("page-*.jsonl")),
        ]
        update_run_manifest(
            storage_root,
            run_id,
            status="failed",
            counts={"completed": False},
            output_paths=(path for path in candidate_paths if path.is_file()),
            errors=(error,),
        )
        raise

    output_paths = (
        archive._request_path(request),
        result.checkpoint_path,
        *result.raw_paths,
        *result.normalized_paths,
    )
    update_run_manifest(
        storage_root,
        run_id,
        status="completed",
        counts={
            "completed": result.completed,
            "pages": result.page_count,
            "rows": result.row_count,
            "requests_made": result.requests_made,
            "resumed_pages": result.resumed_pages,
        },
        output_paths=output_paths,
    )
    return PythExactHistoryRunResult(
        archive=result,
        run_id=run_id,
        manifest_path=manifest_path,
    )
