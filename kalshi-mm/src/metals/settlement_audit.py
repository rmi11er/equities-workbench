"""Offline reconciliation of Kalshi metals settlements to exact Pyth history.

The audit consumes normalized ``market_dim`` and ``kalshi_settlement`` rows plus
completed, content-addressed ``pyth_exact_history`` price-range archives.  It
never consumes TradingView candles and never fetches data.  A window can be
*usable* only when both immediately preceding one-minute ranges have verified
request coverage and timestamp, correction, and rounding semantics have
explicit versions and confirmation evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .provider_approval import ProviderApprovalError, require_provider_operation
from .pyth_exact_history import (
    PythExactHistoryArchive,
    PythExactHistoryIntegrityError,
    PythExactHistoryRequest,
)
from .settlement import ROUNDING_MODES, round_price
from .storage import HashChainedJsonlWriter

AUDIT_SCHEMA_VERSION = "metals-settlement-audit/v2"
AUDIT_CODE_VERSION = "settlement-audit-2"
SEMANTICS_SCHEMA_VERSION = "metals-settlement-semantics/v2"
UNIVERSE_SCHEMA_VERSION = "metals-settlement-universe/v1"
_ONE_MINUTE_NS = 60 * 1_000_000_000
_ONE_MINUTE_US = 60 * 1_000_000
_STREAMS = ("market_dim", "kalshi_settlement")
_HEX_DIGITS = frozenset("0123456789abcdef")


class SettlementAuditError(RuntimeError):
    """Raised when local inputs or an immutable audit artifact are invalid."""


def _is_sha256(value: str | None) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value.lower())
    )


@dataclass(frozen=True)
class SemanticsConfirmation:
    """Versioned evidence for one externally confirmed interpretation."""

    version: str
    confirmed: bool
    evidence_sha256: str | None = None

    def validate(self, name: str) -> None:
        if not self.version.strip():
            raise ValueError(f"{name} semantics version is required")
        if self.confirmed and not _is_sha256(self.evidence_sha256):
            raise ValueError(f"confirmed {name} semantics require an evidence SHA-256")
        if self.evidence_sha256 is not None and not _is_sha256(self.evidence_sha256):
            raise ValueError(f"{name} evidence_sha256 is invalid")


@dataclass(frozen=True)
class SettlementAuditSemantics:
    """The only supported interpretation of local inputs for one audit.

    ``timestamp_source_label`` identifies the untouched timestamp-semantics label
    in normalized exact-history rows.  Confirmation remains external evidence;
    the exact-history schema does not by itself confirm Kalshi's rule meaning.
    """

    timestamp: SemanticsConfirmation
    correction: SemanticsConfirmation
    rounding: SemanticsConfirmation
    timestamp_field: str
    timestamp_meaning: str
    timestamp_source_label: str
    correction_policy: str
    rounding_modes_by_series: tuple[tuple[str, str], ...]
    schema_version: str = SEMANTICS_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SEMANTICS_SCHEMA_VERSION:
            raise ValueError("unsupported settlement semantics schema")
        self.timestamp.validate("timestamp")
        self.correction.validate("correction")
        self.rounding.validate("rounding")
        if self.timestamp_field != "publish_timestamp_us":
            raise ValueError("only exact Pyth publish_timestamp_us is supported")
        if self.timestamp_meaning != "last_publication_strictly_before_boundary":
            raise ValueError(
                "timestamp meaning must select the last publication strictly before the boundary"
            )
        if (
            self.timestamp_source_label
            != "publish_time_and_feed_update_timestamp_microseconds"
        ):
            raise ValueError("unsupported exact-history timestamp_source_label")
        if self.correction_policy != "last_publication_before_boundary_v1":
            raise ValueError("unsupported correction policy")
        series = set()
        for series_ticker, mode in self.rounding_modes_by_series:
            if not series_ticker or series_ticker in series:
                raise ValueError("rounding series tickers must be non-empty and unique")
            if mode not in ROUNDING_MODES:
                raise ValueError(f"unsupported rounding mode: {mode}")
            series.add(series_ticker)

    @property
    def all_confirmed(self) -> bool:
        return (
            self.timestamp.confirmed
            and self.correction.confirmed
            and self.rounding.confirmed
        )

    def rounding_mode(self, series_ticker: str) -> str | None:
        return dict(self.rounding_modes_by_series).get(series_ticker)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["rounding_modes_by_series"] = {
            key: mode for key, mode in self.rounding_modes_by_series
        }
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SettlementAuditSemantics:
        def confirmation(name: str) -> SemanticsConfirmation:
            item = value.get(name)
            if not isinstance(item, Mapping):
                raise ValueError(  # noqa: TRY004 - stable public parser contract
                    f"{name} semantics object is required"
                )
            return SemanticsConfirmation(
                version=str(item.get("version", "")),
                confirmed=item.get("confirmed") is True,
                evidence_sha256=(
                    None
                    if item.get("evidence_sha256") is None
                    else str(item.get("evidence_sha256"))
                ),
            )

        modes = value.get("rounding_modes_by_series", {})
        if not isinstance(modes, Mapping):
            raise ValueError(  # noqa: TRY004 - stable public parser contract
                "rounding_modes_by_series must be an object"
            )
        semantics = cls(
            timestamp=confirmation("timestamp"),
            correction=confirmation("correction"),
            rounding=confirmation("rounding"),
            timestamp_field=str(value.get("timestamp_field", "")),
            timestamp_meaning=str(value.get("timestamp_meaning", "")),
            timestamp_source_label=str(value.get("timestamp_source_label", "")),
            correction_policy=str(value.get("correction_policy", "")),
            rounding_modes_by_series=tuple(
                sorted((str(key), str(mode)) for key, mode in modes.items())
            ),
            schema_version=str(value.get("schema_version", "")),
        )
        semantics.validate()
        return semantics


@dataclass(frozen=True)
class SettlementUniverse:
    """Evidence that a sub-1,000-window local universe is complete."""

    version: str
    confirmed: bool
    evidence_sha256: str | None
    completed_market_tickers: tuple[str, ...]
    schema_version: str = UNIVERSE_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != UNIVERSE_SCHEMA_VERSION:
            raise ValueError("unsupported settlement universe schema")
        if not self.version.strip():
            raise ValueError("settlement universe version is required")
        if self.confirmed and not _is_sha256(self.evidence_sha256):
            raise ValueError(
                "confirmed settlement universe requires an evidence SHA-256"
            )
        if len(set(self.completed_market_tickers)) != len(
            self.completed_market_tickers
        ):
            raise ValueError("completed market tickers must be unique")
        if any(not ticker for ticker in self.completed_market_tickers):
            raise ValueError("completed market tickers must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SettlementUniverse:
        tickers = value.get("completed_market_tickers", ())
        if not isinstance(tickers, (list, tuple)):
            raise ValueError(  # noqa: TRY004 - stable public parser contract
                "completed_market_tickers must be an array"
            )
        universe = cls(
            version=str(value.get("version", "")),
            confirmed=value.get("confirmed") is True,
            evidence_sha256=(
                None
                if value.get("evidence_sha256") is None
                else str(value.get("evidence_sha256"))
            ),
            completed_market_tickers=tuple(sorted(str(item) for item in tickers)),
            schema_version=str(value.get("schema_version", "")),
        )
        universe.validate()
        return universe


@dataclass(frozen=True)
class SettlementAuditArtifact:
    audit_id: str
    directory: Path
    details_path: Path
    summary_path: Path
    manifest_path: Path
    summary: dict[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SettlementAuditError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise SettlementAuditError(f"JSON object required: {path}")
    return value


def _verify_and_load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_hash: str | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SettlementAuditError(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                record_hash = value.get("record_sha256")
                payload = dict(value)
                payload.pop("record_sha256", None)
                expected = hashlib.sha256(
                    _canonical_json(payload).encode("utf-8")
                ).hexdigest()
                if record_hash != expected:
                    raise SettlementAuditError(
                        f"record hash mismatch: {path}:{line_number}"
                    )
                if value.get("previous_record_sha256") != previous_hash:
                    raise SettlementAuditError(
                        f"record chain mismatch: {path}:{line_number}"
                    )
                previous_hash = str(record_hash)
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise SettlementAuditError(f"cannot read JSONL: {path}") from error
    return rows


def _discover_inputs(
    root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    streams: dict[str, list[dict[str, Any]]] = {name: [] for name in _STREAMS}
    descriptors: list[dict[str, Any]] = []
    for stream in _STREAMS:
        directory = root / "normalized" / stream
        paths = sorted(directory.glob("**/*.jsonl")) if directory.is_dir() else []
        for path in paths:
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(root)
            except ValueError as error:
                raise SettlementAuditError("input escapes storage root") from error
            rows = _verify_and_load_jsonl(resolved)
            streams[stream].extend(rows)
            descriptors.append(
                {
                    "stream": stream,
                    "path": str(relative),
                    "size_bytes": resolved.stat().st_size,
                    "sha256": _sha256_file(resolved),
                    "records": len(rows),
                }
            )
    return streams, descriptors


@dataclass(frozen=True)
class _VerifiedExactRequest:
    request: PythExactHistoryRequest
    rows: tuple[dict[str, Any], ...]


def _request_from_metadata(
    value: Mapping[str, Any], directory: Path
) -> PythExactHistoryRequest:
    public = value.get("request")
    if not isinstance(public, Mapping):
        raise SettlementAuditError(
            "Pyth exact-history request metadata is missing its request object"
        )
    endpoint = public.get("endpoint")
    channel = public.get("channel")
    suffix = f"/{channel}/price/range"
    if not isinstance(endpoint, str) or not endpoint.endswith(suffix):
        raise SettlementAuditError("Pyth exact-history request endpoint is invalid")
    feed_ids = public.get("feed_ids")
    if not isinstance(feed_ids, list):
        raise SettlementAuditError("Pyth exact-history request feed_ids are invalid")
    try:
        request = PythExactHistoryRequest(
            metal=public.get("metal"),
            feed_ids=tuple(feed_ids),
            start_timestamp_us=public.get("start_timestamp_us"),
            end_timestamp_us=public.get("end_timestamp_us"),
            limit=public.get("limit"),
            max_pages=public.get("max_pages"),
            base_url=endpoint[: -len(suffix)],
            channel=channel,
        )
    except (TypeError, ValueError) as error:
        raise SettlementAuditError(
            "Pyth exact-history request metadata is invalid"
        ) from error
    if dict(public) != request.public_dict():
        raise SettlementAuditError(
            "Pyth exact-history request metadata is not canonical"
        )
    request_id = value.get("request_id")
    if (
        request_id != request.request_id
        or directory.name != f"request={request.request_id}"
    ):
        raise SettlementAuditError("Pyth exact-history request identity mismatch")
    return request


def _descriptor(
    root: Path, path: Path, artifact_kind: str, **extra: Any
) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise SettlementAuditError(
            "Pyth exact-history artifact escapes storage root"
        ) from error
    return {
        "stream": "pyth_exact_history",
        "artifact_kind": artifact_kind,
        "path": str(relative),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
        **extra,
    }


@dataclass(frozen=True)
class _ExactHistoryProvenance:
    manifest_path: Path
    approval_path: Path
    approval_manifest_path: Path
    output_paths: frozenset[Path]


def _exact_history_provenance(root: Path) -> list[_ExactHistoryProvenance]:
    provenance: list[_ExactHistoryProvenance] = []
    manifest_root = root / "manifests"
    paths = (
        sorted(manifest_root.glob("run=*/manifest.json"))
        if manifest_root.is_dir()
        else []
    )
    for manifest_path in paths:
        manifest = _load_json(manifest_path)
        config = manifest.get("config")
        collector = config.get("collector") if isinstance(config, Mapping) else None
        component = (
            collector.get("component") if isinstance(collector, Mapping) else None
        )
        if component not in {
            "pyth_exact_history",
            "pyth_settlement_exact_backfill",
        }:
            continue
        if manifest.get("status") != "completed":
            continue
        if not isinstance(config, Mapping) or manifest.get(
            "config_sha256"
        ) != _canonical_sha256(config):
            raise SettlementAuditError(
                "Pyth exact-history run manifest config checksum is invalid"
            )
        raw_outputs = manifest.get("output_paths")
        if not isinstance(raw_outputs, list):
            raise SettlementAuditError(
                "Pyth exact-history run manifest outputs are invalid"
            )
        outputs: set[Path] = set()
        for value in raw_outputs:
            path = Path(str(value))
            if path.is_absolute():
                resolved = path.resolve()
            else:
                cwd_candidate = (Path.cwd() / path).resolve()
                root_candidate = (root / path).resolve()
                try:
                    cwd_candidate.relative_to(root)
                except ValueError:
                    resolved = root_candidate
                else:
                    resolved = cwd_candidate
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise SettlementAuditError(
                    "Pyth exact-history run manifest output escapes storage root"
                ) from error
            if not resolved.is_file():
                raise SettlementAuditError(
                    "Pyth exact-history run manifest output is missing"
                )
            outputs.add(resolved)
        approval_id = collector.get("provider_approval_id")
        approval_candidates = [
            path
            for path in outputs
            if path.name == "approval.json"
            and path.parent.name == f"approval={approval_id}"
            and path.parent.parent.name == "provider_approval"
        ]
        if len(approval_candidates) != 1:
            raise SettlementAuditError(
                "Pyth exact-history run lacks one manifest-listed provider approval"
            )
        started_ts = manifest.get("started_ts")
        if not isinstance(started_ts, str):
            raise SettlementAuditError(
                "Pyth exact-history run manifest start timestamp is missing"
            )
        try:
            started_at = datetime.fromisoformat(started_ts.replace("Z", "+00:00"))
        except ValueError as error:
            raise SettlementAuditError(
                "Pyth exact-history run manifest start timestamp is invalid"
            ) from error
        try:
            approval = require_provider_operation(
                root,
                approval_candidates[0],
                "pyth",
                "exact_history",
                as_of=started_at,
            )
        except (ProviderApprovalError, ValueError) as error:
            raise SettlementAuditError(
                "Pyth exact-history provider approval failed verification"
            ) from error
        if approval.approval_id != approval_id:
            raise SettlementAuditError(
                "Pyth exact-history provider approval identity is inconsistent"
            )
        required_approval_paths = {
            approval.approval_path.resolve(),
            approval.manifest_path.resolve(),
        }
        if not required_approval_paths.issubset(outputs):
            raise SettlementAuditError(
                "Pyth exact-history provider approval files are not manifest-listed"
            )
        provenance.append(
            _ExactHistoryProvenance(
                manifest_path=manifest_path.resolve(),
                approval_path=approval.approval_path.resolve(),
                approval_manifest_path=approval.manifest_path.resolve(),
                output_paths=frozenset(outputs),
            )
        )
    return provenance


def _discover_exact_history(
    root: Path, metal: str | None
) -> tuple[list[_VerifiedExactRequest], list[dict[str, Any]], int]:
    checkpoint_root = root / "checkpoints" / "pyth_exact_history"
    directories = (
        sorted(checkpoint_root.glob("request=*")) if checkpoint_root.is_dir() else []
    )
    verified: list[_VerifiedExactRequest] = []
    descriptors: list[dict[str, Any]] = []
    row_count = 0
    seen_request_ids = set()
    provenance = _exact_history_provenance(root)
    provenance_descriptors: dict[Path, dict[str, Any]] = {}
    for directory in directories:
        if not directory.is_dir():
            raise SettlementAuditError(
                "Pyth exact-history checkpoint entry is not a directory"
            )
        request_path = directory / "request.json"
        if not request_path.is_file():
            raise SettlementAuditError("Pyth exact-history request metadata is missing")
        request_metadata = _load_json(request_path)
        request = _request_from_metadata(request_metadata, directory)
        if metal is not None and request.metal != metal:
            continue
        if request.request_id in seen_request_ids:
            raise SettlementAuditError("duplicate Pyth exact-history request identity")
        seen_request_ids.add(request.request_id)
        archive = PythExactHistoryArchive(root, api_key="offline-verification")
        try:
            result = archive.verify_completed(request)
        except PythExactHistoryIntegrityError as error:
            raise SettlementAuditError(
                "Pyth exact-history archive failed completed-request verification"
            ) from error

        rows: list[dict[str, Any]] = []
        for normalized_path in result.normalized_paths:
            try:
                for line in normalized_path.read_text(encoding="utf-8").splitlines():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise SettlementAuditError(
                            "verified Pyth exact-history row is not an object"
                        )
                    rows.append(value)
            except (OSError, json.JSONDecodeError) as error:
                raise SettlementAuditError(
                    "cannot read verified Pyth exact-history projection"
                ) from error
        if len(rows) != result.row_count:
            raise SettlementAuditError(
                "Pyth exact-history normalized row count contradicts checkpoint"
            )
        request_artifacts = {
            request_path.resolve(),
            result.checkpoint_path.resolve(),
            *(path.resolve() for path in result.raw_paths),
            *(path.resolve() for path in result.normalized_paths),
        }
        owners = [
            item for item in provenance if request_artifacts.issubset(item.output_paths)
        ]
        if not owners:
            raise SettlementAuditError(
                "Pyth exact-history request is not bound to a completed approved run"
            )
        for owner in owners:
            for path, kind in (
                (owner.manifest_path, "run_manifest"),
                (owner.approval_path, "provider_approval"),
                (owner.approval_manifest_path, "provider_approval_manifest"),
            ):
                provenance_descriptors[path] = _descriptor(
                    root,
                    path,
                    kind,
                )
        verified.append(_VerifiedExactRequest(request=request, rows=tuple(rows)))
        row_count += len(rows)
        descriptors.append(
            _descriptor(root, request_path, "request", request_id=request.request_id)
        )
        descriptors.append(
            _descriptor(
                root,
                result.checkpoint_path,
                "checkpoint",
                request_id=request.request_id,
            )
        )
        for raw_path in result.raw_paths:
            descriptors.append(
                _descriptor(root, raw_path, "raw_page", request_id=request.request_id)
            )
        for normalized_path in result.normalized_paths:
            descriptors.append(
                _descriptor(
                    root,
                    normalized_path,
                    "normalized_page",
                    request_id=request.request_id,
                )
            )
    descriptors.extend(
        provenance_descriptors[path] for path in sorted(provenance_descriptors)
    )
    return verified, descriptors, row_count


def _required_text(row: Mapping[str, Any], field: str, stream: str) -> str:
    value = row.get(field)
    if value is None or not str(value).strip():
        raise SettlementAuditError(f"{stream} row is missing {field}")
    return str(value)


def _integer(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _latest(
    rows: Iterable[Mapping[str, Any]],
    key_field: str,
    order_fields: Sequence[str],
    stream: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_required_text(row, key_field, stream)].append(row)
    selected: dict[str, dict[str, Any]] = {}
    revisions: dict[str, int] = {}
    for key, candidates in grouped.items():

        def order(candidate: Mapping[str, Any]) -> tuple[int, ...]:
            values = tuple(
                _integer(candidate.get(field)) or -1 for field in order_fields
            )
            # The hash is a stable tie-break without using provider source time to
            # move a later local receipt ahead of an earlier one.
            return (*values, int(str(candidate["record_sha256"]), 16))

        selected[key] = dict(max(candidates, key=order))
        revisions[key] = len(candidates)
    return selected, revisions


@dataclass(frozen=True)
class _ExactBoundarySelection:
    row: dict[str, Any] | None
    request_ids: tuple[str, ...]
    duplicate_count: int
    coverage_complete: bool
    conflicting: bool


def _exact_row_fingerprint(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        row.get(field)
        for field in (
            "price_feed_id",
            "publish_timestamp_us",
            "feed_update_timestamp_us",
            "price_mantissa",
            "exponent",
            "price",
            "is_carried_forward",
            "freshness_age_us",
        )
    )


def _select_exact_boundary(
    archives: Sequence[_VerifiedExactRequest],
    *,
    metal: str,
    feed_id: int,
    boundary_ns: int,
) -> _ExactBoundarySelection:
    boundary_us = boundary_ns // 1_000
    start_us = boundary_us - _ONE_MINUTE_US
    end_us = boundary_us - 1
    candidates = [
        archive
        for archive in archives
        if archive.request.metal == metal
        and feed_id in archive.request.feed_ids
        and archive.request.start_timestamp_us <= end_us
        and archive.request.end_timestamp_us >= start_us
    ]

    # Inclusive request ranges must cover every microsecond in [B-60s, B).
    cursor = start_us
    for range_start, range_end in sorted(
        (
            max(start_us, archive.request.start_timestamp_us),
            min(end_us, archive.request.end_timestamp_us),
        )
        for archive in candidates
    ):
        if range_start > cursor:
            break
        cursor = max(cursor, range_end + 1)
        if cursor > end_us:
            break
    coverage_complete = cursor > end_us
    request_ids = tuple(sorted(archive.request.request_id for archive in candidates))
    if not coverage_complete:
        return _ExactBoundarySelection(None, request_ids, 0, False, False)

    by_publish: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for archive in candidates:
        for row in archive.rows:
            if _integer(row.get("price_feed_id")) != feed_id:
                continue
            publish_us = _integer(row.get("publish_timestamp_us"))
            if publish_us is not None and start_us <= publish_us < boundary_us:
                by_publish[publish_us].append(row)
    if not by_publish:
        return _ExactBoundarySelection(None, request_ids, 0, True, False)

    conflicting = any(
        len({_exact_row_fingerprint(row) for row in rows}) != 1
        for rows in by_publish.values()
    )
    if conflicting:
        return _ExactBoundarySelection(None, request_ids, 0, True, True)

    selected_rows = by_publish[max(by_publish)]
    selected = dict(
        min(
            selected_rows,
            key=lambda row: (
                str(row.get("historical_request_id") or ""),
                str(row.get("record_sha256") or ""),
            ),
        )
    )
    return _ExactBoundarySelection(
        selected,
        request_ids,
        len(selected_rows),
        True,
        False,
    )


def _normalized_result(value: Any) -> str | None:
    result = str(value).strip().lower() if value is not None else ""
    return result if result in ("yes", "no") else None


def _rate(numerator: int, denominator: int) -> str | None:
    if denominator == 0:
        return None
    return format(Decimal(numerator) / Decimal(denominator), ".12f")


def _row_for_window(
    settlement: Mapping[str, Any],
    market: Mapping[str, Any] | None,
    market_revision_count: int,
    settlement_revision_count: int,
    exact_archives: Sequence[_VerifiedExactRequest],
    semantics: SettlementAuditSemantics,
    exact_feed_id: int | None,
) -> dict[str, Any]:
    ticker = str(settlement["market_ticker"])
    exclusions: list[str] = []

    if not semantics.timestamp.confirmed:
        exclusions.append("timestamp_semantics_unconfirmed")
    if not semantics.correction.confirmed:
        exclusions.append("correction_semantics_unconfirmed")
    if not semantics.rounding.confirmed:
        exclusions.append("rounding_semantics_unconfirmed")
    if market is None:
        exclusions.append("missing_market_metadata")
        market = {}

    series = str(market.get("series_ticker") or settlement.get("series_ticker") or "")
    metal = str(market.get("metal") or "")
    symbol = str(market.get("pyth_symbol") or "")
    market_feed_id = market.get("pyth_feed_id")
    feed_id = exact_feed_id if exact_feed_id is not None else _integer(market_feed_id)
    open_ts_ns = _integer(market.get("open_ts_ns"))
    close_ts_ns = _integer(market.get("close_ts_ns"))
    round_digits = _integer(market.get("round_digits"))
    rounding_mode = semantics.rounding_mode(series)
    target = _decimal(market.get("target_price"))
    exchange_final = _decimal(settlement.get("expiration_value"))
    exchange_result = _normalized_result(settlement.get("result"))
    status = str(settlement.get("status") or "").lower()
    strike_type = str(market.get("strike_type") or "").lower()

    if status != "finalized":
        exclusions.append("settlement_not_finalized")
    if not series:
        exclusions.append("missing_series_ticker")
    if not str(market.get("rules_primary") or "").strip():
        exclusions.append("missing_rules_primary")
    if not str(market.get("rules_secondary") or "").strip():
        exclusions.append("missing_rules_secondary")
    if feed_id is None or feed_id < 0:
        exclusions.append("missing_numeric_pyth_feed_id")
    if open_ts_ns is None or open_ts_ns % _ONE_MINUTE_NS:
        exclusions.append("invalid_open_boundary")
    if close_ts_ns is None or close_ts_ns % _ONE_MINUTE_NS:
        exclusions.append("invalid_close_boundary")
    if open_ts_ns is not None and close_ts_ns is not None and close_ts_ns <= open_ts_ns:
        exclusions.append("invalid_window_order")
    if round_digits is None or round_digits < 0:
        exclusions.append("invalid_round_digits")
    if rounding_mode is None:
        exclusions.append("rounding_semantics_missing_for_series")
    if target is None:
        exclusions.append("missing_exchange_target")
    if exchange_final is None:
        exclusions.append("missing_exchange_settlement_value")
    if exchange_result is None:
        exclusions.append("invalid_exchange_result")
    if strike_type != "greater_or_equal":
        exclusions.append("unsupported_strike_type")

    empty_selection = _ExactBoundarySelection(None, (), 0, False, False)
    opening_selection = (
        _select_exact_boundary(
            exact_archives,
            metal=metal,
            feed_id=feed_id,
            boundary_ns=open_ts_ns,
        )
        if metal and feed_id is not None and open_ts_ns is not None
        else empty_selection
    )
    final_selection = (
        _select_exact_boundary(
            exact_archives,
            metal=metal,
            feed_id=feed_id,
            boundary_ns=close_ts_ns,
        )
        if metal and feed_id is not None and close_ts_ns is not None
        else empty_selection
    )
    opening = opening_selection.row
    final = final_selection.row
    for label, selection in (
        ("opening", opening_selection),
        ("settlement", final_selection),
    ):
        if selection.conflicting:
            exclusions.append(f"conflicting_{label}_exact_history")
        elif not selection.coverage_complete:
            exclusions.append(f"missing_{label}_exact_history_coverage")
        elif selection.row is None:
            exclusions.append(f"missing_{label}_exact_history_publication")

    for label, exact_row in (("opening", opening), ("settlement", final)):
        if (
            exact_row is not None
            and exact_row.get("timestamp_semantics") != semantics.timestamp_source_label
        ):
            exclusions.append(f"{label}_timestamp_label_mismatch")

    opening_raw = _decimal(opening.get("price")) if opening else None
    final_raw = _decimal(final.get("price")) if final else None
    if opening is not None and opening_raw is None:
        exclusions.append("invalid_opening_exact_history_price")
    if final is not None and final_raw is None:
        exclusions.append("invalid_settlement_exact_history_price")

    reconstructed_target: Decimal | None = None
    reconstructed_final: Decimal | None = None
    if round_digits is not None and round_digits >= 0 and rounding_mode is not None:
        if opening_raw is not None:
            reconstructed_target = round_price(opening_raw, round_digits, rounding_mode)
        if final_raw is not None:
            reconstructed_final = round_price(final_raw, round_digits, rounding_mode)

    target_agreement = (
        reconstructed_target == target
        if reconstructed_target is not None and target is not None
        else None
    )
    final_value_agreement = (
        reconstructed_final == exchange_final
        if reconstructed_final is not None and exchange_final is not None
        else None
    )
    predicted_result: str | None = None
    tie_flag: bool | None = None
    if reconstructed_final is not None and target is not None:
        predicted_result = "yes" if reconstructed_final >= target else "no"
        tie_flag = reconstructed_final == target
    result_agreement = (
        predicted_result == exchange_result
        if predicted_result is not None and exchange_result is not None
        else None
    )

    # Deduplicate while preserving deterministic diagnostic priority.
    exclusions = list(dict.fromkeys(exclusions))
    usable = not exclusions
    agreement = (
        target_agreement is True
        and final_value_agreement is True
        and result_agreement is True
    )
    discrepancy = "unresolved_discrepancy" if usable and not agreement else None

    opening_carried = bool(opening and opening.get("is_carried_forward") is True)
    final_carried = bool(final and final.get("is_carried_forward") is True)

    def source_age_ns(
        exact_row: Mapping[str, Any] | None, boundary: int | None
    ) -> int | None:
        update_us = _integer(
            exact_row.get("feed_update_timestamp_us") if exact_row else None
        )
        if update_us is None or boundary is None:
            return None
        age = boundary - update_us * 1_000
        return age if age >= 0 else None

    opening_age_ns = source_age_ns(opening, open_ts_ns)
    final_age_ns = source_age_ns(final, close_ts_ns)
    available_ages = [age for age in (opening_age_ns, final_age_ns) if age is not None]
    maximum_source_age_ns = max(available_ages) if available_ages else None
    rules_row: Mapping[str, Any] = market
    rules_secondary = market.get("rules_secondary")
    if rules_secondary is None and settlement.get("rules_secondary") is not None:
        rules_row = settlement
        rules_secondary = settlement.get("rules_secondary")
    rules_secondary_source_id = (
        rules_row.get("rules_secondary_source_id") or rules_row.get("record_sha256")
        if rules_secondary is not None
        else None
    )
    reconstructed_target_text = (
        None if reconstructed_target is None else format(reconstructed_target, "f")
    )
    opening_source_id = None if opening is None else opening.get("record_sha256")
    settlement_source_id = None if final is None else final.get("record_sha256")
    exclusion_reason = exclusions[0] if exclusions else None

    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_code_version": AUDIT_CODE_VERSION,
        "market_ticker": ticker,
        "event_ticker": market.get("event_ticker") or settlement.get("event_ticker"),
        "series_ticker": series or None,
        "metal": metal or None,
        "open_ts_ns": open_ts_ns,
        "close_ts_ns": close_ts_ns,
        "exchange_determination_ts_ns": _integer(settlement.get("settlement_ts_ns")),
        "exchange_target": None if target is None else format(target, "f"),
        "exchange_settlement_value": (
            None if exchange_final is None else format(exchange_final, "f")
        ),
        "exchange_result": exchange_result,
        "reconstructed_opening_target": reconstructed_target_text,
        "reconstructed_target": reconstructed_target_text,
        "opening_raw_pyth_value": (
            None if opening_raw is None else format(opening_raw, "f")
        ),
        "reconstructed_final_value": (
            None if reconstructed_final is None else format(reconstructed_final, "f")
        ),
        "final_raw_pyth_value": None if final_raw is None else format(final_raw, "f"),
        "opening_pyth_source_id": opening_source_id,
        "settlement_pyth_source_id": settlement_source_id,
        "opening_source_update_id": opening_source_id,
        "settlement_source_update_id": settlement_source_id,
        "opening_exact_history_request_id": (
            None if opening is None else opening.get("historical_request_id")
        ),
        "settlement_exact_history_request_id": (
            None if final is None else final.get("historical_request_id")
        ),
        "opening_exact_history_coverage_request_ids": list(
            opening_selection.request_ids
        ),
        "settlement_exact_history_coverage_request_ids": list(
            final_selection.request_ids
        ),
        "opening_publish_timestamp_us": (
            None if opening is None else opening.get("publish_timestamp_us")
        ),
        "settlement_publish_timestamp_us": (
            None if final is None else final.get("publish_timestamp_us")
        ),
        "opening_feed_update_timestamp_us": (
            None if opening is None else opening.get("feed_update_timestamp_us")
        ),
        "settlement_feed_update_timestamp_us": (
            None if final is None else final.get("feed_update_timestamp_us")
        ),
        "opening_source_age_ns": opening_age_ns,
        "settlement_source_age_ns": final_age_ns,
        "market_metadata_source_id": market.get("record_sha256"),
        "kalshi_settlement_source_id": settlement.get("record_sha256"),
        "settlement_source_name": market.get("settlement_source_name"),
        "settlement_source_url": market.get("settlement_source_url"),
        "rules_primary": market.get("rules_primary"),
        "rules_secondary": rules_secondary,
        "rules_secondary_source_id": rules_secondary_source_id,
        "kalshi_rules_secondary_text": rules_secondary,
        "kalshi_rules_secondary_source_id": rules_secondary_source_id,
        "pyth_symbol": symbol or None,
        "market_pyth_feed_id": market_feed_id,
        "pyth_feed_id": feed_id,
        "round_digits": round_digits,
        "rounding_mode": rounding_mode,
        "strike_type": strike_type or None,
        "tie_flag": tie_flag,
        "opening_carried_forward": opening_carried if opening is not None else None,
        "settlement_carried_forward": final_carried if final is not None else None,
        "stale_source_fallback_flag": opening_carried or final_carried,
        "stale_source_age_ns": maximum_source_age_ns,
        "stale_source_fallback": opening_carried or final_carried,
        "source_age_ns": maximum_source_age_ns,
        "predicted_result": predicted_result,
        "target_agreement": target_agreement,
        "settlement_value_agreement": final_value_agreement,
        "result_agreement": result_agreement,
        "agreement": agreement if usable else None,
        "timestamp_semantics_version": semantics.timestamp.version,
        "timestamp_semantics_confirmed": semantics.timestamp.confirmed,
        "timestamp_source_label": semantics.timestamp_source_label,
        "timestamp_field": semantics.timestamp_field,
        "timestamp_meaning": semantics.timestamp_meaning,
        "correction_semantics_version": semantics.correction.version,
        "correction_semantics_confirmed": semantics.correction.confirmed,
        "correction_policy": semantics.correction_policy,
        "rounding_semantics_version": semantics.rounding.version,
        "rounding_semantics_confirmed": semantics.rounding.confirmed,
        "market_revision_count": market_revision_count,
        "settlement_revision_count": settlement_revision_count,
        "opening_pyth_revision_count": opening_selection.duplicate_count,
        "settlement_pyth_revision_count": final_selection.duplicate_count,
        "usable": usable,
        "exclusion_reasons": exclusions,
        "exclusion_reason": exclusion_reason,
        "discrepancy_classification": discrepancy,
    }


def _summary(
    rows: Sequence[Mapping[str, Any]],
    input_counts: Mapping[str, int],
    semantics: SettlementAuditSemantics,
    universe: SettlementUniverse | None,
    observed_settlement_tickers: Sequence[str],
) -> dict[str, Any]:
    usable = [row for row in rows if row["usable"]]
    agreeing = [row for row in usable if row["agreement"] is True]
    target_exact = [
        row
        for row in rows
        if row["usable"]
        if row["reconstructed_opening_target"] is not None
        and row["timestamp_semantics_confirmed"]
        and row["rounding_semantics_confirmed"]
    ]
    final_exact = [
        row
        for row in rows
        if row["usable"]
        if row["reconstructed_final_value"] is not None
        and row["timestamp_semantics_confirmed"]
        and row["rounding_semantics_confirmed"]
    ]
    exclusions = Counter(reason for row in rows for reason in row["exclusion_reasons"])
    discrepancies = Counter(
        str(row["discrepancy_classification"])
        for row in rows
        if row["discrepancy_classification"] is not None
    )

    observed = set(observed_settlement_tickers)
    expected = set(universe.completed_market_tickers) if universe is not None else set()
    missing_expected = expected - observed
    rows_with_missing_evidence = [
        row
        for row in rows
        if any(
            str(reason).startswith("missing_") for reason in row["exclusion_reasons"]
        )
    ]
    conflicting_rows = [
        row
        for row in rows
        if any(
            str(reason).startswith("conflicting_")
            for reason in row["exclusion_reasons"]
        )
    ]
    unclassified = [
        row for row in rows if not row["usable"] and not row.get("exclusion_reason")
    ]
    expected_count = len(expected) if universe is not None else len(rows)
    universe_complete = bool(
        universe is not None
        and universe.confirmed
        and observed == expected
        and len(rows) == len(expected)
    )
    all_expected_usable = bool(
        universe_complete and {row["market_ticker"] for row in usable} == expected
    )
    sample_requirement_met = len(usable) >= 1000 or all_expected_usable
    agreement_gate = bool(usable) and len(agreeing) == len(usable)
    ready = bool(
        semantics.all_confirmed
        and agreement_gate
        and sample_requirement_met
        and not discrepancies
    )
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_code_version": AUDIT_CODE_VERSION,
        "input_record_counts": dict(input_counts),
        "expected_count": expected_count,
        "usable_count": len(usable),
        "excluded_count": max(expected_count - len(usable), 0),
        "missing_count": len(missing_expected) + len(rows_with_missing_evidence),
        "conflicting_count": len(conflicting_rows),
        "unclassified_count": len(unclassified),
        "completed_windows_observed": len(rows),
        "usable_windows": len(usable),
        "excluded_windows": len(rows) - len(usable),
        "exact_targets_reconstructed": len(target_exact),
        "exact_target_agreements": sum(
            row["target_agreement"] is True for row in target_exact
        ),
        "exact_settlements_reconstructed": len(final_exact),
        "exact_settlement_value_agreements": sum(
            row["settlement_value_agreement"] is True for row in final_exact
        ),
        "result_agreements": sum(row["result_agreement"] is True for row in usable),
        "fully_agreeing_usable_windows": len(agreeing),
        "target_reconstruction_rate": _rate(len(target_exact), len(rows)),
        "settlement_reconstruction_rate": _rate(len(final_exact), len(rows)),
        "usable_agreement_rate": _rate(len(agreeing), len(usable)),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "discrepancy_counts": dict(sorted(discrepancies.items())),
        "semantics_confirmed": semantics.all_confirmed,
        "universe_complete_and_confirmed": universe_complete,
        "all_confirmed_universe_windows_usable": all_expected_usable,
        "minimum_1000_or_all_available_met": sample_requirement_met,
        "usable_window_agreement_gate_passed": agreement_gate,
        "settlement_research_gate_passed": ready,
    }


def _filter_market_rows(
    rows: Iterable[Mapping[str, Any]], metal: str | None
) -> list[Mapping[str, Any]]:
    if metal is None:
        return list(rows)
    return [row for row in rows if str(row.get("metal") or "") == metal]


def run_local_settlement_audit(
    storage_root: Path,
    semantics: SettlementAuditSemantics,
    *,
    metal: str | None = None,
    universe: SettlementUniverse | None = None,
    exact_feed_id: int | None = None,
) -> SettlementAuditArtifact:
    """Audit every locally recorded completed settlement and freeze the result.

    The function performs no network access.  Inputs are discovered solely below
    ``storage_root/normalized`` and every hash chain is checked before selection.
    Canonical market and settlement revisions use latest local receive time.
    Pyth values come only from verified completed exact-history requests and use
    the last publication strictly before each boundary.
    """

    semantics.validate()
    if universe is not None:
        universe.validate()
    if metal is not None and (not metal or "/" in metal or ".." in metal):
        raise ValueError("metal is invalid")
    if exact_feed_id is not None and (
        isinstance(exact_feed_id, bool)
        or not isinstance(exact_feed_id, int)
        or exact_feed_id < 0
    ):
        raise ValueError("exact_feed_id must be a non-negative integer")
    root = Path(storage_root).resolve()
    streams, inputs = _discover_inputs(root)
    exact_archives, exact_inputs, exact_row_count = _discover_exact_history(root, metal)
    inputs.extend(exact_inputs)
    market_rows = _filter_market_rows(streams["market_dim"], metal)
    markets, market_revisions = _latest(
        market_rows,
        "market_ticker",
        ("metadata_receive_ts_ns", "metadata_version"),
        "market_dim",
    )
    settlements, settlement_revisions = _latest(
        streams["kalshi_settlement"],
        "market_ticker",
        ("settlement_ts_ns", "receive_ts_ns"),
        "kalshi_settlement",
    )
    if metal is not None:
        market_tickers = set(markets)
        settlements = {
            ticker: row
            for ticker, row in settlements.items()
            if ticker in market_tickers
        }
        settlement_revisions = {
            ticker: count
            for ticker, count in settlement_revisions.items()
            if ticker in market_tickers
        }
    rows = [
        _row_for_window(
            settlement=settlements[ticker],
            market=markets.get(ticker),
            market_revision_count=market_revisions.get(ticker, 0),
            settlement_revision_count=settlement_revisions[ticker],
            exact_archives=exact_archives,
            semantics=semantics,
            exact_feed_id=exact_feed_id,
        )
        for ticker in sorted(settlements)
    ]
    input_counts = {name: len(streams[name]) for name in _STREAMS}
    input_counts["pyth_exact_history"] = exact_row_count
    summary = _summary(
        rows,
        input_counts,
        semantics,
        universe,
        sorted(settlements),
    )
    identity = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_code_version": AUDIT_CODE_VERSION,
        "metal": metal,
        "exact_feed_id": exact_feed_id,
        "semantics": semantics.to_dict(),
        "universe": None if universe is None else universe.to_dict(),
        "inputs": inputs,
    }
    audit_id = _canonical_sha256(identity)
    summary = {"audit_id": audit_id, "metal": metal, **summary}
    destination = root / "derived" / "settlement_check" / f"audit={audit_id}"
    if destination.exists():
        verified = verify_settlement_audit(root, destination)
        return SettlementAuditArtifact(
            audit_id=audit_id,
            directory=destination,
            details_path=destination / "windows.jsonl",
            summary_path=destination / "summary.json",
            manifest_path=destination / "manifest.json",
            summary=verified["summary"],
        )

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".settlement-audit-", dir=str(parent)))
    try:
        details_path = temporary / "windows.jsonl"
        with HashChainedJsonlWriter(details_path) as writer:
            for row in rows:
                writer.write({"audit_id": audit_id, **row})
        summary_path = temporary / "summary.json"
        _write_json(summary_path, summary)
        manifest = {
            **identity,
            "audit_id": audit_id,
            "outputs": [
                {
                    "path": "windows.jsonl",
                    "size_bytes": details_path.stat().st_size,
                    "sha256": _sha256_file(details_path),
                    "records": len(rows),
                },
                {
                    "path": "summary.json",
                    "size_bytes": summary_path.stat().st_size,
                    "sha256": _sha256_file(summary_path),
                },
            ],
        }
        _write_json(temporary / "manifest.json", manifest)
        try:
            os.rename(str(temporary), str(destination))
        except FileExistsError:
            # A concurrent identical run won publication.  Its hashes must still
            # verify before it can be returned as the immutable artifact.
            shutil.rmtree(temporary)
        try:
            directory_fd = os.open(str(parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    verified = verify_settlement_audit(root, destination)
    return SettlementAuditArtifact(
        audit_id=audit_id,
        directory=destination,
        details_path=destination / "windows.jsonl",
        summary_path=destination / "summary.json",
        manifest_path=destination / "manifest.json",
        summary=verified["summary"],
    )


def verify_settlement_audit(storage_root: Path, artifact_path: Path) -> dict[str, Any]:
    """Verify content address, input hashes, output hashes, and row hash chain."""

    root = Path(storage_root).resolve()
    requested = Path(artifact_path).resolve()
    directory = requested.parent if requested.name == "manifest.json" else requested
    try:
        relative = directory.relative_to(root)
    except ValueError as error:
        raise SettlementAuditError("audit artifact escapes storage root") from error
    if len(relative.parts) < 3 or relative.parts[-2] != "settlement_check":
        raise SettlementAuditError("invalid settlement audit location")
    expected_name = relative.parts[-1]
    if not expected_name.startswith("audit="):
        raise SettlementAuditError("invalid settlement audit directory")
    manifest = _load_json(directory / "manifest.json")
    audit_id = str(manifest.get("audit_id") or "")
    if expected_name != f"audit={audit_id}":
        raise SettlementAuditError("settlement audit content address mismatch")
    identity = {
        key: manifest.get(key)
        for key in (
            "audit_schema_version",
            "audit_code_version",
            "metal",
            "exact_feed_id",
            "semantics",
            "universe",
            "inputs",
        )
    }
    if _canonical_sha256(identity) != audit_id:
        raise SettlementAuditError("settlement audit identity hash mismatch")
    for descriptor in manifest.get("inputs", []):
        path = (root / str(descriptor["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise SettlementAuditError("audit input escapes storage root") from error
        if (
            not path.is_file()
            or path.stat().st_size != descriptor.get("size_bytes")
            or _sha256_file(path) != descriptor.get("sha256")
        ):
            raise SettlementAuditError(f"settlement audit input changed: {path}")
    windows: list[dict[str, Any]] | None = None
    for descriptor in manifest.get("outputs", []):
        path = (directory / str(descriptor["path"])).resolve()
        try:
            path.relative_to(directory)
        except ValueError as error:
            raise SettlementAuditError("audit output escapes artifact") from error
        if (
            not path.is_file()
            or path.stat().st_size != descriptor.get("size_bytes")
            or _sha256_file(path) != descriptor.get("sha256")
        ):
            raise SettlementAuditError(f"settlement audit output changed: {path}")
        if path.name == "windows.jsonl":
            rows = _verify_and_load_jsonl(path)
            if len(rows) != descriptor.get("records"):
                raise SettlementAuditError("settlement audit row count changed")
            windows = rows
    if windows is None:
        raise SettlementAuditError("settlement audit windows output is missing")
    summary = _load_json(directory / "summary.json")
    if summary.get("audit_id") != audit_id:
        raise SettlementAuditError("settlement audit summary identity mismatch")
    return {"manifest": manifest, "summary": summary, "windows": windows}
