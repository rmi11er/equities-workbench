"""Evidence-backed Phase 1 daily-gate measurements.

The live collectors deliberately keep lightweight counters.  This module is a
separate, deterministic post-processing layer: it re-reads immutable normalized
outputs named by completed run manifests, verifies their record hash chains,
and derives the measurements used by the Phase 1 data gates.  Missing evidence
never becomes a successful measurement.

The daily policy is itself a manifest-listed JSON artifact.  It fixes the
measurement interval, expected market windows, staleness limits, clock policy,
and approved registry hashes before this module examines the day's records.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .provider_approval import ProviderApprovalError, require_provider_operation
from .quality import evaluate_data_gates
from .storage import update_run_manifest, write_run_manifest

EVIDENCE_VERSION = 1
POLICY_VERSION = 1
_SOURCES = ("kalshi", "pyth", "cme")
_COVERAGE_STREAMS = {
    "kalshi": frozenset(("kalshi_book_event",)),
    "pyth": frozenset(("pyth_lazer_price", "pyth_tick")),
    "cme": frozenset(("cme_mbo_event", "cme_quote", "cme_trade")),
}
_CLOCK_STREAMS = frozenset(("clock_telemetry", "infrastructure_telemetry"))
_SETTLEMENT_STREAMS = frozenset(("settlement_check", "kalshi_settlement"))
_REGISTRY_STREAMS = frozenset(
    ("market_dim", "fee_schedule_dim", "incentive_program_dim")
)
_RELEVANT_STREAMS = (
    frozenset(stream for streams in _COVERAGE_STREAMS.values() for stream in streams)
    | _CLOCK_STREAMS
    | _SETTLEMENT_STREAMS
    | _REGISTRY_STREAMS
)


class DailyGateMeasurementError(RuntimeError):
    """Persisted evidence is malformed, mutable, or internally inconsistent."""


@dataclass(frozen=True)
class ExpectedWindow:
    market_ticker: str
    series_ticker: str
    fee_group_id: str | None
    metal: str
    start_ts_ns: int
    end_ts_ns: int


@dataclass(frozen=True)
class DailyGatePolicy:
    date: str
    measurement_start_ts_ns: int
    measurement_end_ts_ns: int
    expected_windows: tuple[ExpectedWindow, ...]
    source_staleness_ns: Mapping[str, int]
    edge_half_life_ns: int
    max_clock_error_fraction: Decimal
    clock_sample_staleness_ns: int
    approved_metadata_hashes: Mapping[str, tuple[str, ...]]
    approved_fee_registry_hashes: tuple[str, ...]
    approved_incentive_registry_hashes: tuple[str, ...]


@dataclass(frozen=True)
class PersistedRecord:
    stream: str
    partition_metal: str | None
    path: str
    line_number: int
    value: Mapping[str, Any]


@dataclass(frozen=True)
class DailyGateArtifacts:
    evidence_path: Path
    quality_report_path: Path
    manifest_path: Path


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validated_date(value: str) -> str:
    try:
        parsed = date_type.fromisoformat(value)
    except ValueError as error:
        raise DailyGateMeasurementError("date must be an ISO calendar date") from error
    if parsed.isoformat() != value:
        raise DailyGateMeasurementError("date must use YYYY-MM-DD format")
    return value


def _strict_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DailyGateMeasurementError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise DailyGateMeasurementError(f"{field} must be at least {minimum}")
    return value


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DailyGateMeasurementError(f"{field} must be non-empty text")
    return value.strip()


def _hash_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise DailyGateMeasurementError(f"{field} must be a non-empty list")
    result = []
    for index, item in enumerate(value):
        text = _nonempty_text(item, f"{field}[{index}]").lower()
        if len(text) != 64 or any(
            character not in "0123456789abcdef" for character in text
        ):
            raise DailyGateMeasurementError(
                f"{field}[{index}] must be a SHA-256 hex digest"
            )
        result.append(text)
    if len(set(result)) != len(result):
        raise DailyGateMeasurementError(f"{field} contains duplicate hashes")
    return tuple(sorted(result))


def _load_policy(path: Path) -> tuple[DailyGatePolicy, str]:
    try:
        raw_bytes = path.read_bytes()
        value = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise DailyGateMeasurementError(
            "Daily gate policy is not valid JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise DailyGateMeasurementError("Daily gate policy must be a JSON object")
    if value.get("schema_version") != POLICY_VERSION:
        raise DailyGateMeasurementError("Unsupported daily gate policy version")

    date = _nonempty_text(value.get("date"), "date")
    start = _strict_int(
        value.get("measurement_start_ts_ns"), "measurement_start_ts_ns", minimum=0
    )
    end = _strict_int(
        value.get("measurement_end_ts_ns"), "measurement_end_ts_ns", minimum=1
    )
    if end <= start:
        raise DailyGateMeasurementError("measurement interval must be positive")

    raw_windows = value.get("expected_windows")
    if not isinstance(raw_windows, list) or not raw_windows:
        raise DailyGateMeasurementError("expected_windows must be a non-empty list")
    windows: list[ExpectedWindow] = []
    seen_tickers = set()
    for index, raw_window in enumerate(raw_windows):
        if not isinstance(raw_window, Mapping):
            raise DailyGateMeasurementError(
                f"expected_windows[{index}] must be an object"
            )
        prefix = f"expected_windows[{index}]"
        ticker = _nonempty_text(
            raw_window.get("market_ticker"), prefix + ".market_ticker"
        )
        if ticker in seen_tickers:
            raise DailyGateMeasurementError("expected market tickers must be unique")
        seen_tickers.add(ticker)
        window_start = _strict_int(
            raw_window.get("start_ts_ns"), prefix + ".start_ts_ns", minimum=0
        )
        window_end = _strict_int(
            raw_window.get("end_ts_ns"), prefix + ".end_ts_ns", minimum=1
        )
        if window_start < start or window_end > end or window_end <= window_start:
            raise DailyGateMeasurementError(
                f"{prefix} lies outside the positive measurement interval"
            )
        windows.append(
            ExpectedWindow(
                market_ticker=ticker,
                series_ticker=_nonempty_text(
                    raw_window.get("series_ticker"), prefix + ".series_ticker"
                ),
                fee_group_id=(
                    _nonempty_text(
                        raw_window.get("fee_group_id"), prefix + ".fee_group_id"
                    )
                    if raw_window.get("fee_group_id") is not None
                    else None
                ),
                metal=_nonempty_text(
                    raw_window.get("metal"), prefix + ".metal"
                ).lower(),
                start_ts_ns=window_start,
                end_ts_ns=window_end,
            )
        )

    raw_staleness = value.get("source_staleness_ns")
    if not isinstance(raw_staleness, Mapping):
        raise DailyGateMeasurementError("source_staleness_ns must be an object")
    staleness = {
        source: _strict_int(
            raw_staleness.get(source),
            f"source_staleness_ns.{source}",
            minimum=1,
        )
        for source in _SOURCES
    }
    edge_half_life = _strict_int(
        value.get("edge_half_life_ns"), "edge_half_life_ns", minimum=1
    )
    clock_staleness = _strict_int(
        value.get("clock_sample_staleness_ns"),
        "clock_sample_staleness_ns",
        minimum=1,
    )
    try:
        clock_fraction = Decimal(str(value.get("max_clock_error_fraction")))
    except (InvalidOperation, ValueError) as error:
        raise DailyGateMeasurementError(
            "max_clock_error_fraction is invalid"
        ) from error
    if not clock_fraction.is_finite() or not Decimal(0) < clock_fraction <= Decimal(1):
        raise DailyGateMeasurementError(
            "max_clock_error_fraction must be finite and in (0, 1]"
        )

    raw_metadata = value.get("approved_metadata_hashes")
    if not isinstance(raw_metadata, Mapping):
        raise DailyGateMeasurementError("approved_metadata_hashes must be an object")
    metadata_hashes = {
        ticker: _hash_tuple(
            raw_metadata.get(ticker), f"approved_metadata_hashes.{ticker}"
        )
        for ticker in sorted(seen_tickers)
    }

    policy = DailyGatePolicy(
        date=date,
        measurement_start_ts_ns=start,
        measurement_end_ts_ns=end,
        expected_windows=tuple(
            sorted(windows, key=lambda item: (item.start_ts_ns, item.market_ticker))
        ),
        source_staleness_ns=staleness,
        edge_half_life_ns=edge_half_life,
        max_clock_error_fraction=clock_fraction,
        clock_sample_staleness_ns=clock_staleness,
        approved_metadata_hashes=metadata_hashes,
        approved_fee_registry_hashes=_hash_tuple(
            value.get("approved_fee_registry_hashes"),
            "approved_fee_registry_hashes",
        ),
        approved_incentive_registry_hashes=_hash_tuple(
            value.get("approved_incentive_registry_hashes"),
            "approved_incentive_registry_hashes",
        ),
    )
    return policy, _sha256_bytes(raw_bytes)


def _within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_output_path(root: Path, value: Any) -> Path:
    text = _nonempty_text(
        str(value) if isinstance(value, Path) else value,
        "manifest output path",
    )
    candidate = Path(text)
    candidates = (
        [candidate]
        if candidate.is_absolute()
        else [Path.cwd() / candidate, root / candidate]
    )
    resolved_root = root.resolve()
    for item in candidates:
        resolved = item.resolve()
        if _within_root(resolved, resolved_root):
            return resolved
    raise DailyGateMeasurementError("Manifest output escapes the storage root")


def _resolve_existing_input(root: Path, value: Any, label: str) -> Path:
    path = _resolve_output_path(root, value)
    if not path.is_file():
        raise DailyGateMeasurementError(f"{label} is missing or is not a file")
    return path


def _stream_partition(path: Path, root: Path) -> tuple[str | None, str | None]:
    try:
        parts = path.relative_to(root.resolve()).parts
    except ValueError:
        return None, None
    stream = None
    if len(parts) >= 2 and parts[0] == "normalized":
        stream = parts[1]
    elif (
        len(parts) >= 2
        and parts[0] == "derived"
        and parts[1] == "settlement_check"
        and path.suffix == ".jsonl"
    ):
        stream = "settlement_check"
    metal = None
    for part in parts:
        if part.startswith("metal=") and len(part) > len("metal="):
            metal = part[len("metal=") :].lower()
            break
    return stream, metal


def _read_manifest(
    path: Path, root: Path
) -> tuple[Mapping[str, Any], dict[str, Any], list[Path]]:
    resolved = _resolve_existing_input(root, path, "Run manifest")
    try:
        raw_bytes = resolved.read_bytes()
        manifest = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise DailyGateMeasurementError("Run manifest is not valid JSON") from error
    if not isinstance(manifest, Mapping):
        raise DailyGateMeasurementError("Run manifest must be a JSON object")
    if manifest.get("status") != "completed":
        raise DailyGateMeasurementError("Run manifest is not completed")
    run_id = _nonempty_text(manifest.get("run_id"), "manifest run_id")
    config = manifest.get("config")
    if not isinstance(config, Mapping) or manifest.get(
        "config_sha256"
    ) != _sha256_bytes(_canonical_json(config)):
        raise DailyGateMeasurementError("Run manifest config checksum is invalid")
    raw_outputs = manifest.get("output_paths")
    if not isinstance(raw_outputs, list):
        raise DailyGateMeasurementError("Manifest output_paths must be a list")
    outputs = [_resolve_output_path(root, item) for item in raw_outputs]
    provenance = {
        "path": str(resolved.relative_to(root.resolve())),
        "run_id": run_id,
        "sha256": _sha256_bytes(raw_bytes),
    }
    return manifest, provenance, outputs


def _read_hash_chained_jsonl(
    path: Path,
    root: Path,
    stream: str,
    partition_metal: str | None,
) -> tuple[list[PersistedRecord], dict[str, Any]]:
    previous_hash: str | None = None
    rows: list[PersistedRecord] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise DailyGateMeasurementError(
            "Manifest-listed normalized output is missing"
        ) from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise DailyGateMeasurementError(
                    "Normalized JSONL contains a blank line"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise DailyGateMeasurementError(
                    "Normalized JSONL contains invalid JSON"
                ) from error
            if not isinstance(value, Mapping):
                raise DailyGateMeasurementError(
                    "Normalized JSONL row must be an object"
                )
            claimed_previous = value.get("previous_record_sha256")
            claimed_hash = value.get("record_sha256")
            if claimed_previous != previous_hash or not isinstance(claimed_hash, str):
                raise DailyGateMeasurementError(
                    "Normalized JSONL hash chain is invalid"
                )
            payload = dict(value)
            payload.pop("record_sha256", None)
            calculated = _sha256_bytes(_canonical_json(payload))
            if claimed_hash != calculated:
                raise DailyGateMeasurementError(
                    "Normalized JSONL record hash is invalid"
                )
            previous_hash = claimed_hash
            rows.append(
                PersistedRecord(
                    stream=stream,
                    partition_metal=partition_metal,
                    path=str(path.relative_to(root.resolve())),
                    line_number=line_number,
                    value=value,
                )
            )
    return rows, {
        "path": str(path.relative_to(root.resolve())),
        "sha256": _file_sha256(path),
        "records": len(rows),
        "stream": stream,
    }


def _verify_synchronized_provider_approval(
    root: Path,
    manifest: Mapping[str, Any],
    outputs: Sequence[Path],
) -> None:
    """Prove the synchronized run's provider policy was valid and preserved."""

    config = manifest.get("config")
    collector = config.get("collector") if isinstance(config, Mapping) else None
    governance = config.get("governance") if isinstance(config, Mapping) else None
    if not isinstance(collector, Mapping) or not isinstance(governance, Mapping):
        raise DailyGateMeasurementError(
            "Synchronized manifest is missing provider approval configuration"
        )
    approval_ids = collector.get("provider_approval_ids")
    if not isinstance(approval_ids, Mapping) or set(approval_ids) != {
        "pyth",
        "databento",
    }:
        raise DailyGateMeasurementError(
            "Synchronized manifest provider approval identities are invalid"
        )
    approval_path = governance.get("provider_approval_path")
    started_ts = manifest.get("started_ts")
    if not isinstance(started_ts, str):
        raise DailyGateMeasurementError(
            "Synchronized manifest is missing its start timestamp"
        )
    try:
        started_at = datetime.fromisoformat(started_ts.replace("Z", "+00:00"))
    except ValueError as error:
        raise DailyGateMeasurementError(
            "Synchronized manifest start timestamp is invalid"
        ) from error
    if started_at.tzinfo is None:
        raise DailyGateMeasurementError(
            "Synchronized manifest start timestamp must be timezone-aware"
        )
    try:
        pyth = require_provider_operation(
            root,
            approval_path,
            "pyth",
            "live_lazer",
            as_of=started_at,
        )
        databento = require_provider_operation(
            root,
            approval_path,
            "databento",
            "live",
            as_of=started_at,
        )
    except (ProviderApprovalError, ValueError) as error:
        raise DailyGateMeasurementError(
            "Synchronized provider approval failed verification"
        ) from error
    if (
        approval_ids.get("pyth") != pyth.approval_id
        or approval_ids.get("databento") != databento.approval_id
    ):
        raise DailyGateMeasurementError(
            "Synchronized provider approval identity does not match its artifact"
        )
    declared = {path.resolve() for path in outputs}
    required = {
        pyth.approval_path.resolve(),
        pyth.manifest_path.resolve(),
        databento.approval_path.resolve(),
        databento.manifest_path.resolve(),
    }
    if not required.issubset(declared):
        raise DailyGateMeasurementError(
            "Synchronized provider approval artifacts are not manifest-listed"
        )


def _load_inputs(
    root: Path,
    manifest_paths: Sequence[Path],
    policy_path: Path,
) -> tuple[
    DailyGatePolicy,
    str,
    list[PersistedRecord],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if not manifest_paths:
        raise DailyGateMeasurementError("At least one completed manifest is required")
    manifests = []
    outputs: dict[Path, None] = {}
    for manifest_path in sorted(
        (Path(path) for path in manifest_paths), key=lambda item: str(item)
    ):
        _, provenance, manifest_outputs = _read_manifest(manifest_path, root)
        manifests.append(provenance)
        for output in manifest_outputs:
            outputs[output] = None

    resolved_policy = _resolve_existing_input(root, policy_path, "Daily gate policy")
    if resolved_policy not in outputs:
        raise DailyGateMeasurementError(
            "Daily gate policy must be listed by a completed manifest"
        )
    policy, policy_hash = _load_policy(resolved_policy)

    records: list[PersistedRecord] = []
    files: list[dict[str, Any]] = []
    for output in sorted(outputs, key=lambda item: str(item)):
        stream, metal = _stream_partition(output, root)
        if stream not in _RELEVANT_STREAMS:
            continue
        if not output.is_file():
            raise DailyGateMeasurementError(
                f"Manifest-listed relevant output is missing: {output}"
            )
        if output.suffix != ".jsonl":
            raise DailyGateMeasurementError("Relevant normalized output must be JSONL")
        output_records, file_evidence = _read_hash_chained_jsonl(
            output, root, stream, metal
        )
        records.extend(output_records)
        files.append(file_evidence)
    return (
        policy,
        policy_hash,
        records,
        sorted(manifests, key=lambda item: (item["run_id"], item["path"])),
        sorted(files, key=lambda item: item["path"]),
    )


def _record_metal(record: PersistedRecord) -> str | None:
    value = record.value.get("metal")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return record.partition_metal


def _timestamp(value: Mapping[str, Any], *fields: str) -> int | None:
    for field in fields:
        candidate = value.get(field)
        if (
            isinstance(candidate, int)
            and not isinstance(candidate, bool)
            and candidate >= 0
        ):
            return candidate
    return None


def _coverage_event(
    record: PersistedRecord,
    source: str,
    staleness_ns: int,
) -> tuple[int, bool, int] | None:
    value = record.value
    receive_ts = _timestamp(value, "receive_ts_ns", "local_receive_ts_ns")
    if receive_ts is None:
        return None
    explicit_valid = value.get("feed_valid", value.get("record_valid"))
    if source == "kalshi":
        explicit_valid = value.get("book_valid_after", explicit_valid)
        if not isinstance(explicit_valid, bool):
            return None
        return receive_ts, explicit_valid, receive_ts + staleness_ns

    publish_ts = _timestamp(
        value, "source_publish_ts_ns", "source_event_ts_ns", "ts_event"
    )
    freshness_deadline = (
        publish_ts + staleness_ns if publish_ts is not None else receive_ts
    )
    structurally_valid = (
        publish_ts is not None
        and publish_ts <= receive_ts
        and freshness_deadline > receive_ts
    )
    if source == "pyth":
        structurally_valid = structurally_valid and value.get("price") not in (None, "")
    if isinstance(explicit_valid, bool):
        structurally_valid = structurally_valid and explicit_valid
    return receive_ts, structurally_valid, freshness_deadline


def _events_for_window(
    records: Iterable[PersistedRecord],
    source: str,
    window: ExpectedWindow,
    staleness_ns: int,
) -> list[tuple[int, bool, int]]:
    events = []
    for record in records:
        if record.stream not in _COVERAGE_STREAMS[source]:
            continue
        if source == "kalshi":
            if record.value.get("market_ticker") != window.market_ticker:
                continue
        elif _record_metal(record) != window.metal:
            continue
        event = _coverage_event(record, source, staleness_ns)
        if event is not None:
            events.append(event)
    # At a common receive timestamp, an explicit invalid state wins.  This is
    # conservative and removes manifest-order dependence.
    by_time: dict[int, tuple[bool, int]] = {}
    for timestamp, valid, freshness_deadline in events:
        previous_valid, previous_deadline = by_time.get(
            timestamp, (True, freshness_deadline)
        )
        by_time[timestamp] = (
            previous_valid and valid,
            min(previous_deadline, freshness_deadline),
        )
    return [
        (timestamp, valid, deadline)
        for timestamp, (valid, deadline) in sorted(by_time.items())
    ]


def _valid_intervals(
    events: Sequence[tuple[int, bool] | tuple[int, bool, int]],
    start_ts_ns: int,
    end_ts_ns: int,
    staleness_ns: int,
) -> list[tuple[int, int]]:
    relevant = [item for item in events if item[0] < end_ts_ns]
    intervals = []
    for index, event in enumerate(relevant):
        timestamp, valid = event[:2]
        if not valid:
            continue
        next_ts = relevant[index + 1][0] if index + 1 < len(relevant) else end_ts_ns
        freshness_deadline = event[2] if len(event) == 3 else timestamp + staleness_ns
        interval_start = max(start_ts_ns, timestamp)
        interval_end = min(end_ts_ns, next_ts, freshness_deadline)
        if interval_end > interval_start:
            intervals.append((interval_start, interval_end))
    return _merge_intervals(intervals)


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not result or start > result[-1][1]:
            result.append((start, end))
        else:
            result[-1] = (result[-1][0], max(result[-1][1], end))
    return result


def _complement(
    start: int, end: int, valid: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    missing = []
    cursor = start
    for valid_start, valid_end in valid:
        if valid_start > cursor:
            missing.append((cursor, valid_start))
        cursor = max(cursor, valid_end)
    if cursor < end:
        missing.append((cursor, end))
    return missing


def _measure_coverage(
    records: Sequence[PersistedRecord], policy: DailyGatePolicy
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coverage: dict[str, Any] = {}
    missing_intervals: list[dict[str, Any]] = []
    for source in _SOURCES:
        valid_ns = 0
        total_ns = 0
        stale_intervals = 0
        structured_events = 0
        for window in policy.expected_windows:
            total_ns += window.end_ts_ns - window.start_ts_ns
            events = _events_for_window(
                records,
                source,
                window,
                policy.source_staleness_ns[source],
            )
            structured_events += len(events)
            valid = _valid_intervals(
                events,
                window.start_ts_ns,
                window.end_ts_ns,
                policy.source_staleness_ns[source],
            )
            valid_ns += sum(end - start for start, end in valid)
            for missing_start, missing_end in _complement(
                window.start_ts_ns, window.end_ts_ns, valid
            ):
                stale_intervals += 1
                missing_intervals.append(
                    {
                        "source": source,
                        "ticker": window.market_ticker,
                        "start_ts_ns": missing_start,
                        "end_ts_ns": missing_end,
                        "reason": f"{source}_missing_or_stale",
                    }
                )
        fraction = valid_ns / total_ns if total_ns else None
        coverage[f"{source}_valid_ns"] = valid_ns
        coverage[f"{source}_total_ns"] = total_ns
        coverage[f"{source}_valid_fraction"] = fraction
        coverage[f"{source}_stale_ns"] = total_ns - valid_ns
        coverage[f"{source}_stale_intervals"] = stale_intervals
        coverage[f"{source}_structured_events"] = structured_events
        coverage[f"{source}_measurement_available"] = structured_events > 0
    missing_intervals.sort(
        key=lambda item: (
            item["start_ts_ns"],
            item["source"],
            item["ticker"],
            item["end_ts_ns"],
        )
    )
    return coverage, missing_intervals


def _records_by_stream(
    records: Sequence[PersistedRecord], stream: str
) -> list[PersistedRecord]:
    return [record for record in records if record.stream == stream]


def _latest_metadata_before(
    records: Sequence[PersistedRecord], ticker: str, timestamp_ns: int
) -> PersistedRecord | None:
    candidates = []
    for record in records:
        if record.value.get("market_ticker") != ticker:
            continue
        receive_ts = _timestamp(record.value, "metadata_receive_ts_ns", "receive_ts_ns")
        if receive_ts is not None and receive_ts <= timestamp_ns:
            version = record.value.get("metadata_version")
            candidates.append(
                (
                    receive_ts,
                    (
                        version
                        if isinstance(version, int) and not isinstance(version, bool)
                        else -1
                    ),
                    record.path,
                    record.line_number,
                    record,
                )
            )
    return max(candidates, default=(None, None, None, None, None))[-1]


_METADATA_REQUIRED = frozenset(
    (
        "market_ticker",
        "event_ticker",
        "series_ticker",
        "open_ts_ns",
        "close_ts_ns",
        "target_price",
        "round_digits",
        "strike_type",
        "settlement_source_name",
        "settlement_source_url",
        "rules_primary",
        "rules_secondary",
        "price_level_structure",
        "fractional_enabled",
        "metadata_hash",
        "metadata_version",
        "metadata_receive_ts_ns",
    )
)


def _has_fields(value: Mapping[str, Any], fields: Iterable[str]) -> bool:
    return all(
        field in value and value[field] is not None and value[field] != ""
        for field in fields
    )


def _measure_windows_and_metadata(
    records: Sequence[PersistedRecord], policy: DailyGatePolicy
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_records = _records_by_stream(records, "market_dim")
    observed = set()
    validated = []
    invalid = []
    changes = []
    for window in policy.expected_windows:
        ticker_records = [
            record
            for record in metadata_records
            if record.value.get("market_ticker") == window.market_ticker
        ]
        if ticker_records:
            observed.add(window.market_ticker)
        baseline = _latest_metadata_before(
            metadata_records, window.market_ticker, window.start_ts_ns
        )
        allowed = set(policy.approved_metadata_hashes[window.market_ticker])
        baseline_valid = bool(
            baseline is not None
            and _has_fields(baseline.value, _METADATA_REQUIRED)
            and not baseline.value.get("missing_required_fields")
            and baseline.value.get("series_ticker") == window.series_ticker
            and _record_metal(baseline) == window.metal
            and str(baseline.value.get("metadata_hash", "")).lower() in allowed
        )
        if baseline_valid:
            validated.append(window.market_ticker)
        else:
            invalid.append(window.market_ticker)

        for record in ticker_records:
            receive_ts = _timestamp(
                record.value, "metadata_receive_ts_ns", "receive_ts_ns"
            )
            if (
                receive_ts is None
                or not window.start_ts_ns < receive_ts < window.end_ts_ns
            ):
                continue
            if baseline is None or record.value.get(
                "metadata_hash"
            ) != baseline.value.get("metadata_hash"):
                changes.append(
                    {
                        "ticker": window.market_ticker,
                        "metadata_hash": record.value.get("metadata_hash"),
                        "metadata_receive_ts_ns": receive_ts,
                        "approved": str(record.value.get("metadata_hash", "")).lower()
                        in allowed,
                    }
                )
    expected_tickers = [window.market_ticker for window in policy.expected_windows]
    window_evidence = {
        "expected_windows": len(expected_tickers),
        "observed_windows": len(observed),
        "missing_markets": sorted(set(expected_tickers) - observed),
        "expected_market_tickers": expected_tickers,
        "observed_market_tickers": sorted(observed),
    }
    metadata_evidence = {
        "validated": not invalid and not changes,
        "validated_market_tickers": sorted(validated),
        "invalid_market_tickers": sorted(invalid),
        "changes": sorted(
            changes, key=lambda item: (item["metadata_receive_ts_ns"], item["ticker"])
        ),
        "unresolved_change": bool(invalid or changes),
    }
    return window_evidence, metadata_evidence


_SETTLEMENT_REQUIRED = frozenset(
    (
        "market_ticker",
        "exchange_result",
        "exchange_settlement_value",
        "reconstructed_target",
        "reconstructed_final_value",
        "opening_source_update_id",
        "settlement_source_update_id",
        "round_digits",
        "tie_flag",
        "stale_source_fallback",
        "source_age_ns",
        "predicted_result",
        "agreement",
        "audit_code_version",
    )
)


def _record_fingerprint(value: Mapping[str, Any]) -> str:
    payload = {
        key: item
        for key, item in value.items()
        if key not in ("record_sha256", "previous_record_sha256")
    }
    return _sha256_bytes(_canonical_json(payload))


def _measure_settlements(
    records: Sequence[PersistedRecord], policy: DailyGatePolicy
) -> dict[str, Any]:
    expected = {window.market_ticker for window in policy.expected_windows}
    candidates: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        if record.stream not in _SETTLEMENT_STREAMS:
            continue
        ticker = record.value.get("market_ticker")
        if ticker not in expected or "usable" not in record.value:
            continue
        candidates.setdefault(str(ticker), {})[_record_fingerprint(record.value)] = (
            record.value
        )

    checked = agreements = disagreements = excluded = 0
    missing = []
    invalid = []
    conflicting = []
    exclusions = []
    for ticker in sorted(expected):
        ticker_records = list(candidates.get(ticker, {}).values())
        if not ticker_records:
            missing.append(ticker)
            continue
        if len(ticker_records) != 1:
            conflicting.append(ticker)
            continue
        value = ticker_records[0]
        if value.get("usable") is False:
            raw_reasons = value.get("exclusion_reasons")
            reasons = (
                [str(reason).strip() for reason in raw_reasons]
                if isinstance(raw_reasons, list)
                else [str(value.get("exclusion_reason") or "").strip()]
            )
            reasons = sorted({reason for reason in reasons if reason})
            audit_version = value.get("audit_code_version")
            if not reasons or not isinstance(audit_version, str) or not audit_version:
                invalid.append(ticker)
                continue
            excluded += 1
            exclusions.append(
                {
                    "ticker": ticker,
                    "reason": ";".join(reasons),
                    "start_ts_ns": None,
                }
            )
            continue
        if value.get("usable") is not True:
            invalid.append(ticker)
            continue
        if not _has_fields(value, _SETTLEMENT_REQUIRED) or not isinstance(
            value.get("agreement"), bool
        ):
            invalid.append(ticker)
            continue
        checked += 1
        target_agreement = value.get("target_agreement", True)
        agreement = value["agreement"] is True and target_agreement is not False
        if agreement:
            agreements += 1
        else:
            disagreements += 1
    resolved = checked + excluded
    return {
        "expected": len(expected),
        "checked": checked,
        "agreements": agreements,
        "disagreements": disagreements,
        "excluded": excluded,
        "resolved": resolved,
        "complete": resolved == len(expected)
        and not missing
        and not invalid
        and not conflicting,
        "missing_market_tickers": missing,
        "invalid_market_tickers": invalid,
        "conflicting_market_tickers": conflicting,
        "exclusions": exclusions,
    }


def _clock_records(records: Sequence[PersistedRecord]) -> list[tuple[int, int, bool]]:
    samples = []
    for record in records:
        if record.stream not in _CLOCK_STREAMS:
            continue
        value = record.value
        if record.stream == "infrastructure_telemetry" and value.get("metric") not in (
            None,
            "clock_offset_ns",
        ):
            continue
        sample_ts = _timestamp(value, "sample_ts_ns", "receive_ts_ns")
        offset = value.get("clock_offset_ns")
        synchronized = value.get("clock_synchronized", value.get("synchronized"))
        if (
            sample_ts is None
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or not isinstance(synchronized, bool)
        ):
            continue
        samples.append((sample_ts, offset, synchronized))
    return sorted(set(samples))


def _measure_clock(
    records: Sequence[PersistedRecord], policy: DailyGatePolicy
) -> dict[str, Any]:
    samples = _clock_records(records)
    threshold = int(Decimal(policy.edge_half_life_ns) * policy.max_clock_error_fraction)
    health_events = [
        (timestamp, synchronized and abs(offset) <= threshold)
        for timestamp, offset, synchronized in samples
    ]
    valid = _valid_intervals(
        health_events,
        policy.measurement_start_ts_ns,
        policy.measurement_end_ts_ns,
        policy.clock_sample_staleness_ns,
    )
    total = policy.measurement_end_ts_ns - policy.measurement_start_ts_ns
    valid_ns = sum(end - start for start, end in valid)
    relevant_offsets = [
        offset
        for timestamp, offset, _ in samples
        if policy.measurement_start_ts_ns - policy.clock_sample_staleness_ns
        <= timestamp
        < policy.measurement_end_ts_ns
    ]
    maximum = max((abs(offset) for offset in relevant_offsets), default=None)
    return {
        "measured": bool(relevant_offsets),
        "samples": len(relevant_offsets),
        "offset_abs_max_ns": maximum,
        "allowed_offset_abs_max_ns": threshold,
        "healthy_coverage_ns": valid_ns,
        "total_ns": total,
        "healthy_coverage_fraction": valid_ns / total if total else None,
        "within_edge_half_life": bool(relevant_offsets)
        and valid_ns == total
        and maximum is not None
        and maximum <= threshold,
    }


def _effective_interval(
    value: Mapping[str, Any], prefix: str = "effective"
) -> tuple[int, int | None] | None:
    start = _timestamp(value, f"{prefix}_start_ts_ns", "start_ts_ns")
    raw_end = value.get(f"{prefix}_end_ts_ns", value.get("end_ts_ns"))
    if raw_end is None:
        end = None
    elif isinstance(raw_end, int) and not isinstance(raw_end, bool) and raw_end > 0:
        end = raw_end
    else:
        return None
    if start is None or (end is not None and end <= start):
        return None
    return start, end


def _registry_matches_window(
    value: Mapping[str, Any], window: ExpectedWindow, stream: str
) -> bool:
    direct = value.get("series_ticker", value.get("eligible_series_ticker"))
    if isinstance(direct, str) and direct == window.series_ticker:
        return True
    multiple = value.get("series_tickers", value.get("eligible_series"))
    if isinstance(multiple, list) and window.series_ticker in multiple:
        return True
    if stream == "fee_schedule_dim" and window.fee_group_id is not None:
        fee_group = value.get(
            "fee_group_id",
            value.get("standard_fee_group_identifier", value.get("fee_group")),
        )
        if fee_group == window.fee_group_id:
            return True
    if stream == "incentive_program_dim":
        direct_market = value.get("market_ticker", value.get("eligible_market_ticker"))
        if direct_market == window.market_ticker:
            return True
        eligible_markets = value.get("eligible_markets")
        if (
            isinstance(eligible_markets, list)
            and window.market_ticker in eligible_markets
        ):
            return True
    return False


_FEE_REQUIRED = frozenset(
    (
        "maker_fee_enabled",
        "taker_coefficient",
        "product_fee_multiplier",
        "account_type_precision_rule",
        "source_document",
        "retrieval_ts_ns",
        "manual_approval_id",
        "config_hash",
    )
)
_INCENTIVE_REQUIRED = frozenset(
    (
        "program_id",
        "program_type",
        "eligible_participant_class",
        "volume_treatment",
        "source_document",
        "retrieval_ts_ns",
        "config_hash",
    )
)


def _fee_fields_valid(value: Mapping[str, Any]) -> bool:
    maker_parameter = any(
        value.get(field) is not None
        for field in (
            "maker_coefficient_or_multiplier",
            "maker_coefficient",
            "maker_multiplier",
        )
    )
    return (
        _has_fields(value, _FEE_REQUIRED)
        and maker_parameter
        and _effective_interval(value) is not None
    )


def _incentive_fields_valid(value: Mapping[str, Any]) -> bool:
    if (
        not _has_fields(value, _INCENTIVE_REQUIRED)
        or _effective_interval(value, "program") is None
    ):
        return False
    if value.get("program_type") == "none":
        return True
    required_program = (
        "eligible_price_range",
        "reward_pool_amount",
        "per_contract_cap",
        "target_size",
        "discount_factor",
        "snapshot_rules",
        "maximum_spread",
        "minimum_size",
        "uptime_duty",
        "coverage_duty",
    )
    return _has_fields(value, required_program)


def _measure_registry(
    records: Sequence[PersistedRecord],
    policy: DailyGatePolicy,
    *,
    stream: str,
    approved_hashes: Sequence[str],
) -> dict[str, Any]:
    approved = set(approved_hashes)
    raw_records = _records_by_stream(records, stream)
    # Collapse replayed copies of the same registry version without hiding
    # distinct periods or scopes.
    unique: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for record in raw_records:
        value = record.value
        interval = _effective_interval(
            value, "program" if stream == "incentive_program_dim" else "effective"
        )
        unique[
            (
                value.get("config_hash"),
                interval,
                value.get("series_ticker"),
                value.get("fee_group_id"),
                value.get("eligible_market_ticker"),
                json.dumps(value.get("eligible_series"), sort_keys=True),
            )
        ] = value
    registry_records = list(unique.values())

    invalid_windows = []
    applied: dict[str, str] = {}
    change_boundaries = set()
    interval_prefix = "program" if stream == "incentive_program_dim" else "effective"
    for value in registry_records:
        if not any(
            _registry_matches_window(value, window, stream)
            for window in policy.expected_windows
        ):
            continue
        interval = _effective_interval(value, interval_prefix)
        if interval is None:
            continue
        start, end = interval
        scope = str(
            value.get("series_ticker")
            or value.get("fee_group_id")
            or value.get("eligible_series")
            or value.get("eligible_market_ticker")
            or "unknown"
        )
        if policy.measurement_start_ts_ns < start < policy.measurement_end_ts_ns:
            change_boundaries.add((scope, start))
        if (
            end is not None
            and policy.measurement_start_ts_ns < end < policy.measurement_end_ts_ns
        ):
            change_boundaries.add((scope, end))
    for window in policy.expected_windows:
        applicable = []
        for value in registry_records:
            if not _registry_matches_window(value, window, stream):
                continue
            interval = _effective_interval(value, interval_prefix)
            if interval is None:
                continue
            start, end = interval
            if start <= window.start_ts_ns and (end is None or end >= window.end_ts_ns):
                applicable.append(value)
        field_validator = (
            _incentive_fields_valid
            if stream == "incentive_program_dim"
            else _fee_fields_valid
        )
        valid = [
            value
            for value in applicable
            if field_validator(value)
            and str(value.get("config_hash", "")).lower() in approved
        ]
        hashes = {str(value.get("config_hash", "")).lower() for value in valid}
        if len(valid) != 1 or len(hashes) != 1:
            invalid_windows.append(window.market_ticker)
        else:
            applied[window.market_ticker] = next(iter(hashes))
    return {
        "validated": not invalid_windows and bool(policy.expected_windows),
        "records": len(registry_records),
        "applied_hash_by_market": dict(sorted(applied.items())),
        "invalid_market_tickers": sorted(invalid_windows),
        "changes": len(change_boundaries),
        "change_boundaries": [
            {"registry_scope": scope, "ts_ns": timestamp}
            for scope, timestamp in sorted(
                change_boundaries, key=lambda item: (item[1], item[0])
            )
        ],
    }


def measure_daily_gates(
    storage_root: Path,
    date: str,
    manifest_paths: Sequence[Path],
    policy_path: Path,
) -> dict[str, Any]:
    """Derive a deterministic Phase 1 evidence object from persisted inputs."""

    date = _validated_date(date)
    root = Path(storage_root).resolve()
    policy, policy_hash, records, manifests, files = _load_inputs(
        root, manifest_paths, Path(policy_path)
    )
    if policy.date != date:
        raise DailyGateMeasurementError("Policy date does not match requested date")

    coverage, missing_intervals = _measure_coverage(records, policy)
    windows, metadata = _measure_windows_and_metadata(records, policy)
    settlement = _measure_settlements(records, policy)
    clock = _measure_clock(records, policy)
    fee = _measure_registry(
        records,
        policy,
        stream="fee_schedule_dim",
        approved_hashes=policy.approved_fee_registry_hashes,
    )
    incentive = _measure_registry(
        records,
        policy,
        stream="incentive_program_dim",
        approved_hashes=policy.approved_incentive_registry_hashes,
    )
    evidence: dict[str, Any] = {
        "evidence_version": EVIDENCE_VERSION,
        "date": date,
        "policy": {
            "path": str(
                _resolve_existing_input(
                    root, policy_path, "Daily gate policy"
                ).relative_to(root)
            ),
            "sha256": policy_hash,
            "measurement_start_ts_ns": policy.measurement_start_ts_ns,
            "measurement_end_ts_ns": policy.measurement_end_ts_ns,
            "source_staleness_ns": dict(policy.source_staleness_ns),
            "edge_half_life_ns": policy.edge_half_life_ns,
            "max_clock_error_fraction": str(policy.max_clock_error_fraction),
            "clock_sample_staleness_ns": policy.clock_sample_staleness_ns,
        },
        "inputs": {"manifests": manifests, "files": files},
        "windows": windows,
        "coverage": coverage,
        "missing_intervals": missing_intervals,
        "clock": clock,
        "settlement_checksum": settlement,
        "metadata_registry": metadata,
        "fee_registry": fee,
        "incentive_registry": incentive,
    }
    evidence["evidence_sha256"] = _sha256_bytes(_canonical_json(evidence))
    return evidence


def verify_daily_gate_evidence(evidence: Mapping[str, Any]) -> None:
    """Reject edited or unsupported evidence before it is merged."""

    if evidence.get("evidence_version") != EVIDENCE_VERSION:
        raise DailyGateMeasurementError("Unsupported daily gate evidence version")
    claimed = evidence.get("evidence_sha256")
    payload = dict(evidence)
    payload.pop("evidence_sha256", None)
    if not isinstance(claimed, str) or claimed != _sha256_bytes(
        _canonical_json(payload)
    ):
        raise DailyGateMeasurementError("Daily gate evidence checksum is invalid")


def merge_daily_gate_evidence(
    quality_report: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Overlay measured fields onto a collector report and re-evaluate gates.

    Collector gap counters and classifications are preserved.  Evidence-derived
    coverage intervals are appended with deterministic de-duplication.
    """

    verify_daily_gate_evidence(evidence)
    if quality_report.get("date") != evidence.get("date"):
        raise DailyGateMeasurementError("Quality report and evidence dates differ")
    merged = copy.deepcopy(dict(quality_report))
    windows = evidence["windows"]
    settlement = evidence["settlement_checksum"]
    metadata = evidence["metadata_registry"]
    fee = evidence["fee_registry"]
    incentive = evidence["incentive_registry"]
    clock = evidence["clock"]
    merged["expected_windows"] = windows["expected_windows"]
    merged["observed_windows"] = windows["observed_windows"]
    merged["missing_markets"] = len(windows["missing_markets"])
    merged["settlement_checksum"] = {
        "expected": settlement["expected"],
        "checked": settlement["checked"],
        "agreements": settlement["agreements"],
        "disagreements": settlement["disagreements"],
        "excluded": settlement["excluded"],
        "resolved": settlement["resolved"],
        "complete": settlement["complete"],
    }
    merged["coverage"] = {
        key: value
        for key, value in evidence["coverage"].items()
        if key.endswith(("_valid_ns", "_total_ns", "_valid_fraction"))
    }
    existing_intervals = list(merged.get("missing_intervals", []))
    all_intervals = existing_intervals + list(evidence["missing_intervals"])
    by_canonical = {
        _canonical_json(item).decode("utf-8"): item for item in all_intervals
    }
    merged["missing_intervals"] = [by_canonical[key] for key in sorted(by_canonical)]
    existing_exclusions = list(merged.get("exclusions", []))
    merged["exclusions"] = existing_exclusions + list(settlement["exclusions"])
    merged["clock_samples"] = clock["samples"]
    merged["clock_offset_abs_max_ns"] = clock["offset_abs_max_ns"]
    merged["clock_error_within_edge_half_life"] = clock["within_edge_half_life"]
    merged["metadata_rules_validated"] = metadata["validated"]
    merged["unresolved_metadata_change"] = metadata["unresolved_change"]
    merged["metadata_changes"] = metadata["changes"]
    merged["fee_registry_validated"] = fee["validated"]
    merged["fee_changes"] = fee["changes"]
    merged["incentive_registry_validated"] = incentive["validated"]
    merged["incentive_changes"] = incentive["changes"]
    merged["daily_gate_measurement"] = copy.deepcopy(dict(evidence))
    merged["data_gates"] = evaluate_data_gates(merged)
    return merged


def _atomic_create_content(path: Path, value: Mapping[str, Any]) -> None:
    """Create an immutable JSON artifact, or verify an idempotent retry."""

    encoded = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise DailyGateMeasurementError(
                "Cannot verify existing daily gate artifact"
            ) from error
        if existing != encoded:
            raise DailyGateMeasurementError(
                "Existing daily gate artifact differs from its content address"
            )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".daily-gates-", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(str(temporary_path), str(path))
            temporary_path.unlink()
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise DailyGateMeasurementError(
                    "Concurrent daily gate artifact differs from its content address"
                )
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise DailyGateMeasurementError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise DailyGateMeasurementError(f"{label} must contain a JSON object")
    return value


def measure_and_persist_daily_gates(
    storage_root: Path,
    date: str,
    quality_report_path: Path,
    manifest_paths: Sequence[Path],
    policy_path: Path,
) -> DailyGateArtifacts:
    """Measure, merge, and immutably persist one day's gate evidence.

    Both the collector quality report and policy must already be outputs of one
    of the completed input manifests.  The resulting evidence and measured
    report receive content-addressed paths and their own completed run manifest,
    which can be supplied directly to ``freeze-daily-validation``.
    """

    date = _validated_date(date)
    root = Path(storage_root).resolve()
    if not manifest_paths:
        raise DailyGateMeasurementError("At least one completed manifest is required")

    source_quality_path = _resolve_existing_input(
        root, quality_report_path, "Collector quality report"
    )
    resolved_manifests = [
        _resolve_existing_input(root, path, "Run manifest") for path in manifest_paths
    ]
    manifest_outputs: set[Path] = set()
    quality_owners: list[tuple[Mapping[str, Any], list[Path]]] = []
    for manifest_path in resolved_manifests:
        manifest, _, outputs = _read_manifest(manifest_path, root)
        manifest_outputs.update(outputs)
        if source_quality_path in outputs:
            quality_owners.append((manifest, outputs))

    if source_quality_path not in manifest_outputs:
        raise DailyGateMeasurementError(
            "Collector quality report must be listed by a completed manifest"
        )
    synchronized_owners = []
    for manifest, outputs in quality_owners:
        config = manifest.get("config")
        collector = config.get("collector") if isinstance(config, Mapping) else None
        if not isinstance(collector, Mapping):
            continue
        required_sources = collector.get("required_sources")
        required_telemetry = collector.get("required_telemetry")
        if (
            collector.get("component") == "synchronized_metals_live"
            and collector.get("read_only") is True
            and required_sources == ["kalshi", "pyth_lazer", "databento"]
            and required_telemetry == ["clock"]
        ):
            synchronized_owners.append((manifest, outputs))
    if len(synchronized_owners) != 1:
        raise DailyGateMeasurementError(
            "Collector quality report must have exactly one synchronized-panel "
            "manifest owner"
        )
    synchronized_manifest, synchronized_outputs = synchronized_owners[0]
    _verify_synchronized_provider_approval(
        root,
        synchronized_manifest,
        synchronized_outputs,
    )
    source_quality = _load_json_object(source_quality_path, "Collector quality report")
    if source_quality.get("date") != date:
        raise DailyGateMeasurementError(
            "Collector quality report date does not match requested date"
        )
    if source_quality.get("run_id") != synchronized_manifest.get("run_id"):
        raise DailyGateMeasurementError(
            "Collector quality report run_id does not match its synchronized manifest"
        )

    evidence = measure_daily_gates(root, date, resolved_manifests, policy_path)
    measured_report = merge_daily_gate_evidence(source_quality, evidence)
    evidence_sha256 = str(evidence["evidence_sha256"])
    report_payload = copy.deepcopy(measured_report)
    report_sha256 = _sha256_bytes(_canonical_json(report_payload))
    measured_report["daily_quality_sha256"] = report_sha256

    evidence_path = (
        root
        / "derived"
        / "daily_gate_evidence"
        / f"date={date}"
        / f"evidence={evidence_sha256}"
        / "evidence.json"
    )
    measured_path = (
        root
        / "derived"
        / "daily_quality"
        / f"date={date}"
        / f"report={report_sha256}"
        / "report.json"
    )
    _atomic_create_content(evidence_path, evidence)
    _atomic_create_content(measured_path, measured_report)

    run_id = f"daily-gates-{report_sha256}"
    public_config = {
        "operation": "measure_daily_gates",
        "read_only_sources": True,
        "date": date,
        "evidence_sha256": evidence_sha256,
        "daily_quality_sha256": report_sha256,
        "source_quality_report": str(source_quality_path.relative_to(root)),
        "source_manifests": sorted(
            str(path.relative_to(root)) for path in resolved_manifests
        ),
    }
    manifest_path = root / "manifests" / f"run={run_id}" / "manifest.json"
    expected_counts = {
        "expected_windows": int(measured_report["expected_windows"]),
        "observed_windows": int(measured_report["observed_windows"]),
        "settlements_checked": int(measured_report["settlement_checksum"]["checked"]),
        "gates_passed": bool(measured_report["data_gates"]["passed"]),
    }
    expected_outputs = [str(evidence_path), str(measured_path)]
    if manifest_path.exists():
        manifest = _load_json_object(manifest_path, "Daily gate run manifest")
        if manifest.get("run_id") != run_id or manifest.get("config") != public_config:
            raise DailyGateMeasurementError(
                "Existing daily gate run manifest conflicts with content address"
            )
        if manifest.get("status") == "completed":
            if (
                manifest.get("counts") != expected_counts
                or manifest.get("output_paths") != expected_outputs
            ):
                raise DailyGateMeasurementError(
                    "Completed daily gate run manifest has inconsistent outputs"
                )
            return DailyGateArtifacts(
                evidence_path=evidence_path,
                quality_report_path=measured_path,
                manifest_path=manifest_path,
            )
        if manifest.get("status") != "running":
            raise DailyGateMeasurementError(
                "Existing daily gate run manifest is terminal but not completed"
            )
    else:
        try:
            write_run_manifest(root, run_id, public_config)
        except FileExistsError:
            manifest = _load_json_object(manifest_path, "Daily gate run manifest")
            if (
                manifest.get("run_id") != run_id
                or manifest.get("config") != public_config
            ):
                raise DailyGateMeasurementError(
                    "Concurrent daily gate run manifest conflicts with content address"
                )
            if manifest.get("status") == "completed":
                if (
                    manifest.get("counts") != expected_counts
                    or manifest.get("output_paths") != expected_outputs
                ):
                    raise DailyGateMeasurementError(
                        "Completed daily gate run manifest has inconsistent outputs"
                    )
                return DailyGateArtifacts(
                    evidence_path=evidence_path,
                    quality_report_path=measured_path,
                    manifest_path=manifest_path,
                )
            if manifest.get("status") != "running":
                raise DailyGateMeasurementError(
                    "Concurrent daily gate run manifest is terminal but not completed"
                )
    update_run_manifest(
        root,
        run_id,
        status="completed",
        counts=expected_counts,
        output_paths=(evidence_path, measured_path),
    )
    return DailyGateArtifacts(
        evidence_path=evidence_path,
        quality_report_path=measured_path,
        manifest_path=manifest_path,
    )
