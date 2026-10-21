"""Immutable, provisional Kalshi labels for historical plumbing tests.

This module deliberately does not verify settlement.  It projects exchange-
reported outcomes from completed local Kalshi runs and marks every included
row as unsuitable for research until an independent Pyth audit has passed.
There is no network path in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .storage import HashChainedJsonlWriter

ARTIFACT_SCHEMA_VERSION = "metals-provisional-kalshi-labels/v1"
LABEL_ROW_SCHEMA_VERSION = "metals-provisional-kalshi-label/v1"
EXCLUSION_ROW_SCHEMA_VERSION = "metals-provisional-kalshi-label-exclusion/v1"
BUILDER_CODE_VERSION = "provisional-kalshi-label-builder-1"
TARGET_SERIES = "KXSILVER15M"
LABEL_AUTHORITY = "kalshi_exchange_reported"
RESEARCH_USE = "plumbing_only"
_STREAMS = ("market_dim", "kalshi_settlement")
_WINDOW_NS = 15 * 60 * 1_000_000_000
_MINUTE_NS = 60 * 1_000_000_000
_HEX = frozenset("0123456789abcdef")


class ProvisionalKalshiLabelError(RuntimeError):
    """A source run or provisional-label artifact failed closed."""


@dataclass(frozen=True)
class ProvisionalKalshiLabelArtifact:
    """Paths belonging to one verified, content-addressed materialization."""

    artifact_id: str
    directory: Path
    labels_path: Path
    exclusions_path: Path
    summary_path: Path
    manifest_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class _SourceRecord:
    stream: str
    row: dict[str, Any]
    jsonl_path: str
    manifest_path: str
    run_id: str
    line_number: int

    @property
    def record_id(self) -> str:
        return str(self.row["record_sha256"])

    def provenance(self) -> dict[str, Any]:
        return {
            "stream": self.stream,
            "record_id": self.record_id,
            "record_sha256": self.record_id,
            "jsonl_path": self.jsonl_path,
            "manifest_path": self.manifest_path,
            "run_id": self.run_id,
            "line_number": self.line_number,
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _source_record_json(value: Any) -> str:
    """Match :class:`HashChainedJsonlWriter` record hashing exactly."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value.lower())
    )


def _inside(root: Path, path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ProvisionalKalshiLabelError(f"{label} is a symbolic link")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ProvisionalKalshiLabelError(
            f"{label} escapes the storage root"
        ) from error
    return resolved


def _relative(root: Path, path: Path) -> str:
    return str(_inside(root, path, "path").relative_to(root))


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProvisionalKalshiLabelError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ProvisionalKalshiLabelError(f"{label} must be a JSON object")
    return value


def _resolve_manifest_output(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ProvisionalKalshiLabelError("run manifest output path is invalid")
    path = Path(value)
    if path.is_absolute():
        return _inside(root, path, "run manifest output")
    cwd_candidate = Path.cwd() / path
    try:
        cwd_candidate.resolve().relative_to(root)
    except ValueError:
        return _inside(root, root / path, "run manifest output")
    return _inside(root, cwd_candidate, "run manifest output")


def _read_verified_jsonl(
    path: Path,
    *,
    stream: str,
    root: Path,
    manifest_path: str,
    run_id: str,
    allow_empty: bool = False,
) -> tuple[list[_SourceRecord], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ProvisionalKalshiLabelError(f"{stream} input is missing or unsafe")
    try:
        encoded = path.read_bytes()
        text = encoded.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ProvisionalKalshiLabelError(f"cannot read {stream} input") from error
    if not text:
        if allow_empty:
            return [], encoded
        raise ProvisionalKalshiLabelError(f"{stream} input is empty")
    if not text.endswith("\n"):
        raise ProvisionalKalshiLabelError(f"{stream} input is incomplete")

    previous_hash: str | None = None
    rows: list[_SourceRecord] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ProvisionalKalshiLabelError(f"{stream} contains a blank row")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ProvisionalKalshiLabelError(
                f"{stream} contains invalid JSON at line {line_number}"
            ) from error
        if not isinstance(value, dict):
            raise ProvisionalKalshiLabelError(
                f"{stream} row {line_number} is not an object"
            )
        record_hash = value.get("record_sha256")
        if not _is_sha256(record_hash):
            raise ProvisionalKalshiLabelError(
                f"{stream} row {line_number} has an invalid record hash"
            )
        if value.get("previous_record_sha256") != previous_hash:
            raise ProvisionalKalshiLabelError(
                f"{stream} row {line_number} has an invalid hash chain"
            )
        payload = dict(value)
        payload.pop("record_sha256", None)
        expected = _sha256_bytes(_source_record_json(payload).encode("utf-8"))
        if record_hash != expected:
            raise ProvisionalKalshiLabelError(
                f"{stream} row {line_number} has a record hash mismatch"
            )
        previous_hash = record_hash
        rows.append(
            _SourceRecord(
                stream=stream,
                row=value,
                jsonl_path=_relative(root, path),
                manifest_path=manifest_path,
                run_id=run_id,
                line_number=line_number,
            )
        )
    return rows, encoded


def _manifest_location(root: Path, requested: Path) -> tuple[Path, str]:
    path = _inside(root, requested, "source manifest")
    if path.is_symlink() or not path.is_file():
        raise ProvisionalKalshiLabelError("source manifest is missing or unsafe")
    relative = path.relative_to(root)
    if (
        len(relative.parts) != 3
        or relative.parts[0] != "manifests"
        or not relative.parts[1].startswith("run=")
        or relative.parts[2] != "manifest.json"
    ):
        raise ProvisionalKalshiLabelError("source manifest location is invalid")
    return path, relative.parts[1][4:]


def _stream_from_path(root: Path, path: Path, run_id: str) -> str | None:
    relative = path.relative_to(root)
    if len(relative.parts) != 6 or relative.parts[0] != "normalized":
        return None
    stream = relative.parts[1]
    if stream not in _STREAMS:
        return None
    if (
        not relative.parts[2].startswith("date=")
        or not relative.parts[3].startswith("metal=")
        or relative.parts[4] != f"run={run_id}"
        or path.suffix != ".jsonl"
    ):
        raise ProvisionalKalshiLabelError(
            f"manifest-listed {stream} path has invalid partitioning"
        )
    return stream


def _load_verified_inputs(
    root: Path, manifest_paths: Sequence[Path]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[_SourceRecord]]]:
    if not manifest_paths:
        raise ValueError("at least one completed source manifest is required")

    resolved_manifests: list[tuple[Path, str]] = []
    seen_manifests: set[Path] = set()
    for requested in manifest_paths:
        path, run_id = _manifest_location(root, Path(requested))
        if path in seen_manifests:
            raise ProvisionalKalshiLabelError("duplicate source manifest")
        seen_manifests.add(path)
        resolved_manifests.append((path, run_id))
    resolved_manifests.sort(key=lambda item: str(item[0]))

    manifest_descriptors: list[dict[str, Any]] = []
    input_descriptors: list[dict[str, Any]] = []
    records: dict[str, list[_SourceRecord]] = {stream: [] for stream in _STREAMS}
    owners: dict[Path, str] = {}
    record_ids: set[str] = set()

    for manifest_path, path_run_id in resolved_manifests:
        manifest = _load_object(manifest_path, "source run manifest")
        if manifest.get("manifest_version") != 2:
            raise ProvisionalKalshiLabelError("source manifest version is unsupported")
        run_id = manifest.get("run_id")
        if run_id != path_run_id or not isinstance(run_id, str) or not run_id:
            raise ProvisionalKalshiLabelError("source manifest run identity is invalid")
        if manifest.get("status") != "completed" or not manifest.get("completed_ts"):
            raise ProvisionalKalshiLabelError("source manifest is not completed")
        if manifest.get("errors") not in (None, []):
            raise ProvisionalKalshiLabelError("completed source manifest has errors")
        config = manifest.get("config")
        if not isinstance(config, Mapping):
            raise ProvisionalKalshiLabelError("source manifest config is invalid")
        if not isinstance(config.get("kalshi"), Mapping):
            raise ProvisionalKalshiLabelError(
                "source manifest is not a Kalshi collection manifest"
            )
        expected_config_hash = _sha256_bytes(
            json.dumps(
                config,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        if manifest.get("config_sha256") != expected_config_hash:
            raise ProvisionalKalshiLabelError(
                "source manifest config checksum is invalid"
            )
        raw_outputs = manifest.get("output_paths")
        if not isinstance(raw_outputs, list):
            raise ProvisionalKalshiLabelError("source manifest outputs are invalid")

        manifest_relative = _relative(root, manifest_path)
        manifest_descriptors.append(
            {
                "path": manifest_relative,
                "run_id": run_id,
                "size_bytes": manifest_path.stat().st_size,
                "sha256": _sha256_file(manifest_path),
            }
        )
        relevant = 0
        for raw_path in raw_outputs:
            output_path = _resolve_manifest_output(root, raw_path)
            stream = _stream_from_path(root, output_path, run_id)
            if stream is None:
                continue
            previous_owner = owners.get(output_path)
            if previous_owner is not None:
                raise ProvisionalKalshiLabelError(
                    "normalized input has ambiguous manifest ownership"
                )
            owners[output_path] = manifest_relative
            source_rows, encoded = _read_verified_jsonl(
                output_path,
                stream=stream,
                root=root,
                manifest_path=manifest_relative,
                run_id=run_id,
            )
            for source_row in source_rows:
                if source_row.record_id in record_ids:
                    raise ProvisionalKalshiLabelError(
                        "duplicate source record identity across inputs"
                    )
                record_ids.add(source_row.record_id)
            records[stream].extend(source_rows)
            input_descriptors.append(
                {
                    "stream": stream,
                    "path": _relative(root, output_path),
                    "manifest_path": manifest_relative,
                    "run_id": run_id,
                    "size_bytes": len(encoded),
                    "sha256": _sha256_bytes(encoded),
                    "records": len(source_rows),
                }
            )
            relevant += 1
        if relevant == 0:
            raise ProvisionalKalshiLabelError(
                "source manifest lists no supported normalized inputs"
            )

    missing = [stream for stream in _STREAMS if not records[stream]]
    if missing:
        raise ProvisionalKalshiLabelError(
            "source manifests have no verified rows for: " + ", ".join(missing)
        )
    manifest_descriptors.sort(key=lambda item: item["path"])
    input_descriptors.sort(key=lambda item: (item["stream"], item["path"]))
    return manifest_descriptors, input_descriptors, records


def _strict_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _finite_decimal(value: Any) -> bool:
    if value is None or value == "" or isinstance(value, bool):
        return False
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return False
    return parsed.is_finite()


def _candidate(record: _SourceRecord) -> bool:
    ticker = str(record.row.get("market_ticker") or "").strip()
    series = str(record.row.get("series_ticker") or "").strip()
    return series == TARGET_SERIES or ticker.startswith(TARGET_SERIES + "-")


def _semantic_fingerprint(record: _SourceRecord) -> str:
    value = {
        key: item
        for key, item in record.row.items()
        if key not in {"previous_record_sha256", "record_sha256"}
    }
    return _canonical_json(value)


def _latest(
    records: Sequence[_SourceRecord], order_fields: Sequence[str]
) -> tuple[_SourceRecord | None, bool]:
    if not records:
        return None, False

    def order(record: _SourceRecord) -> tuple[int, ...]:
        return tuple(
            value if (value := _strict_int(record.row.get(field))) is not None else -1
            for field in order_fields
        )

    latest_order = max(order(record) for record in records)
    latest = [record for record in records if order(record) == latest_order]
    ambiguous = len({_semantic_fingerprint(record) for record in latest}) != 1
    selected = min(latest, key=lambda record: record.record_id)
    return selected, ambiguous


def _source_detail(record: _SourceRecord, *, market: bool) -> dict[str, Any]:
    detail = record.provenance()
    if market:
        detail.update(
            {
                "metadata_hash": record.row.get("metadata_hash"),
                "metadata_version": record.row.get("metadata_version"),
                "metadata_receive_ts_ns": record.row.get("metadata_receive_ts_ns"),
            }
        )
    else:
        detail.update(
            {
                "raw_metadata_hash": record.row.get("raw_metadata_hash"),
                "receive_ts_ns": record.row.get("receive_ts_ns"),
            }
        )
    return detail


def _all_provenance(records: Sequence[_SourceRecord]) -> list[dict[str, Any]]:
    return [
        record.provenance()
        for record in sorted(records, key=lambda item: (item.stream, item.record_id))
    ]


def _exclusion(
    artifact_id: str,
    candidate_id: str,
    ticker: str | None,
    reasons: Sequence[str],
    records: Sequence[_SourceRecord],
) -> dict[str, Any]:
    return {
        "schema_version": EXCLUSION_ROW_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "candidate_id": candidate_id,
        "market_ticker": ticker,
        "classification": "excluded",
        "exclusion_reasons": sorted(set(reasons)),
        "source_records": _all_provenance(records),
    }


def _rows_for_inputs(
    artifact_id: str, records: Mapping[str, list[_SourceRecord]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    grouped: dict[str, dict[str, list[_SourceRecord]]] = defaultdict(
        lambda: {stream: [] for stream in _STREAMS}
    )
    exclusions: list[dict[str, Any]] = []
    ignored = 0
    for stream in _STREAMS:
        for record in records[stream]:
            if not _candidate(record):
                ignored += 1
                continue
            ticker = str(record.row.get("market_ticker") or "").strip()
            if not ticker:
                exclusions.append(
                    _exclusion(
                        artifact_id,
                        f"source-record:{record.record_id}",
                        None,
                        ["missing_market_ticker"],
                        [record],
                    )
                )
                continue
            grouped[ticker][stream].append(record)

    labels: list[dict[str, Any]] = []
    for ticker in sorted(grouped):
        group = grouped[ticker]
        settlement, settlement_ambiguous = _latest(
            group["kalshi_settlement"],
            ("settlement_ts_ns", "receive_ts_ns"),
        )
        linked_market_rows = group["market_dim"]
        raw_metadata_hash = (
            settlement.row.get("raw_metadata_hash") if settlement is not None else None
        )
        metadata_link_missing = False
        if raw_metadata_hash not in (None, ""):
            linked_market_rows = [
                record
                for record in linked_market_rows
                if record.row.get("metadata_hash") == raw_metadata_hash
            ]
            metadata_link_missing = not linked_market_rows
        market, market_ambiguous = _latest(
            linked_market_rows, ("metadata_receive_ts_ns", "metadata_version")
        )

        reasons: list[str] = []
        if not ticker.startswith(TARGET_SERIES + "-"):
            reasons.append("invalid_market_ticker")
        if not group["market_dim"]:
            reasons.append("missing_market_metadata")
        if not group["kalshi_settlement"]:
            reasons.append("missing_settlement")
        if metadata_link_missing:
            reasons.append("metadata_link_mismatch")
        if market_ambiguous:
            reasons.append("ambiguous_latest_market_revision")
        if settlement_ambiguous:
            reasons.append("ambiguous_latest_settlement_revision")

        market_row = market.row if market is not None else {}
        settlement_row = settlement.row if settlement is not None else {}
        market_series = str(market_row.get("series_ticker") or "").strip()
        settlement_series = str(settlement_row.get("series_ticker") or "").strip()
        if market is not None and market_series != TARGET_SERIES:
            reasons.append("series_ticker_mismatch")
        if settlement_series and settlement_series != TARGET_SERIES:
            reasons.append("series_ticker_mismatch")

        market_event = str(market_row.get("event_ticker") or "").strip()
        settlement_event = str(settlement_row.get("event_ticker") or "").strip()
        if market is not None and not market_event:
            reasons.append("missing_event_ticker")
        if market_event and settlement_event and market_event != settlement_event:
            reasons.append("event_ticker_mismatch")

        metal = str(market_row.get("metal") or "").strip().lower()
        if metal and metal != "silver":
            reasons.append("metal_mismatch")
        open_ts_ns = _strict_int(market_row.get("open_ts_ns"))
        close_ts_ns = _strict_int(market_row.get("close_ts_ns"))
        settlement_ts_ns = _strict_int(settlement_row.get("settlement_ts_ns"))
        if market is not None:
            if open_ts_ns is None or open_ts_ns < 0:
                reasons.append("invalid_open_ts_ns")
            if close_ts_ns is None or close_ts_ns < 1:
                reasons.append("invalid_close_ts_ns")
            if open_ts_ns is not None and close_ts_ns is not None:
                if close_ts_ns - open_ts_ns != _WINDOW_NS:
                    reasons.append("non_15_minute_window")
                if open_ts_ns % _MINUTE_NS or close_ts_ns % _MINUTE_NS:
                    reasons.append("unaligned_window_boundary")
        if settlement is not None:
            if settlement_ts_ns is None or settlement_ts_ns < 1:
                reasons.append("invalid_settlement_ts_ns")
            elif close_ts_ns is not None and settlement_ts_ns < close_ts_ns:
                reasons.append("settlement_precedes_close")

        status = str(settlement_row.get("status") or "").strip().lower()
        if settlement is not None and status != "finalized":
            reasons.append("settlement_not_finalized")
        result = str(settlement_row.get("result") or "").strip().lower()
        if settlement is not None and result not in {"yes", "no"}:
            reasons.append("invalid_exchange_result")
        if settlement is not None and not _finite_decimal(
            settlement_row.get("expiration_value")
        ):
            reasons.append("invalid_expiration_value")

        target_price = market_row.get("target_price")
        if target_price not in (None, "") and not _finite_decimal(target_price):
            reasons.append("invalid_target_price")
        round_digits = market_row.get("round_digits")
        if round_digits is not None and (
            _strict_int(round_digits) is None or round_digits < 0
        ):
            reasons.append("invalid_round_digits")

        all_records = [*group["market_dim"], *group["kalshi_settlement"]]
        if reasons:
            exclusions.append(
                _exclusion(artifact_id, ticker, ticker, reasons, all_records)
            )
            continue
        assert market is not None
        assert settlement is not None
        labels.append(
            {
                "schema_version": LABEL_ROW_SCHEMA_VERSION,
                "artifact_id": artifact_id,
                "market_ticker": ticker,
                "event_ticker": market_event,
                "series_ticker": TARGET_SERIES,
                "metal": market_row.get("metal"),
                "open_ts_ns": open_ts_ns,
                "close_ts_ns": close_ts_ns,
                "determination_ts_ns": market_row.get("determination_ts_ns"),
                "settlement_ts_ns": settlement_ts_ns,
                "target_price": target_price,
                "strike_type": market_row.get("strike_type"),
                "round_digits": round_digits,
                "price_level_structure": market_row.get("price_level_structure"),
                "price_ranges_json": market_row.get("price_ranges_json"),
                "fractional_enabled": market_row.get("fractional_enabled"),
                "expiration_value": settlement_row.get("expiration_value"),
                "settlement_value_dollars": settlement_row.get(
                    "settlement_value_dollars"
                ),
                "result": result,
                "exchange_status": status,
                "label_authority": LABEL_AUTHORITY,
                "independent_pyth_verified": False,
                "research_use": RESEARCH_USE,
                "market_revision_count": len(group["market_dim"]),
                "settlement_revision_count": len(group["kalshi_settlement"]),
                "market_source": _source_detail(market, market=True),
                "settlement_source": _source_detail(settlement, market=False),
            }
        )

    exclusions.sort(key=lambda row: row["candidate_id"])
    return labels, exclusions, ignored


def _identity(
    manifests: list[dict[str, Any]], inputs: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "builder_code_version": BUILDER_CODE_VERSION,
        "series_ticker": TARGET_SERIES,
        "label_policy": {
            "label_authority": LABEL_AUTHORITY,
            "independent_pyth_verified": False,
            "research_use": RESEARCH_USE,
            "network_access": False,
            "settlement_verification": False,
            "imputation": False,
        },
        "input_manifests": manifests,
        "inputs": inputs,
    }


def _summary(
    artifact_id: str,
    labels: Sequence[Mapping[str, Any]],
    exclusions: Sequence[Mapping[str, Any]],
    ignored: int,
    records: Mapping[str, list[_SourceRecord]],
) -> dict[str, Any]:
    reasons = Counter(
        reason for row in exclusions for reason in row["exclusion_reasons"]
    )
    return {
        "artifact_id": artifact_id,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "series_ticker": TARGET_SERIES,
        "label_authority": LABEL_AUTHORITY,
        "independent_pyth_verified": False,
        "research_use": RESEARCH_USE,
        "settlement_verification": False,
        "input_record_counts": {stream: len(records[stream]) for stream in _STREAMS},
        "candidate_windows": len(labels) + len(exclusions),
        "included_labels": len(labels),
        "excluded_candidates": len(exclusions),
        "ignored_non_target_records": ignored,
        "exclusion_counts": dict(sorted(reasons.items())),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with HashChainedJsonlWriter(path) as writer:
        for row in rows:
            writer.write(dict(row))


def _output_descriptor(path: Path, *, records: int | None = None) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        descriptor["records"] = records
    return descriptor


def materialize_provisional_kalshi_labels(
    storage_root: Path, manifest_paths: Sequence[Path]
) -> ProvisionalKalshiLabelArtifact:
    """Freeze provisional KXSILVER15M exchange labels from local inputs only."""

    root = Path(storage_root).resolve()
    manifests, inputs, records = _load_verified_inputs(root, manifest_paths)
    identity = _identity(manifests, inputs)
    artifact_id = _sha256_bytes(_canonical_json(identity).encode("utf-8"))
    labels, exclusions, ignored = _rows_for_inputs(artifact_id, records)
    summary = _summary(artifact_id, labels, exclusions, ignored, records)
    destination = (
        root
        / "derived"
        / "provisional_kalshi_window_labels"
        / f"artifact={artifact_id}"
    )
    if destination.exists():
        return verify_provisional_kalshi_labels(root, destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".provisional-kalshi-labels-", dir=destination.parent)
    )
    try:
        labels_path = temporary / "labels.jsonl"
        exclusions_path = temporary / "exclusions.jsonl"
        summary_path = temporary / "summary.json"
        _write_rows(labels_path, labels)
        _write_rows(exclusions_path, exclusions)
        _write_json(summary_path, summary)
        manifest = {
            **identity,
            "artifact_id": artifact_id,
            "outputs": [
                _output_descriptor(labels_path, records=len(labels)),
                _output_descriptor(exclusions_path, records=len(exclusions)),
                _output_descriptor(summary_path),
            ],
        }
        _write_json(temporary / "manifest.json", manifest)
        try:
            os.rename(temporary, destination)
        except FileExistsError:
            shutil.rmtree(temporary)
        try:
            directory_fd = os.open(str(destination.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verify_provisional_kalshi_labels(root, destination)


def _strip_chain(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: item
            for key, item in row.items()
            if key not in {"previous_record_sha256", "record_sha256"}
        }
        for row in rows
    ]


def verify_provisional_kalshi_labels(
    storage_root: Path, artifact_path: Path
) -> ProvisionalKalshiLabelArtifact:
    """Rebuild and verify inputs, content address, outputs, rows, and summary."""

    root = Path(storage_root).resolve()
    requested = _inside(root, Path(artifact_path), "provisional label artifact")
    directory = requested.parent if requested.name == "manifest.json" else requested
    relative = directory.relative_to(root)
    if (
        len(relative.parts) != 3
        or relative.parts[:2] != ("derived", "provisional_kalshi_window_labels")
        or not directory.name.startswith("artifact=")
    ):
        raise ProvisionalKalshiLabelError("provisional label location is invalid")
    manifest_path = directory / "manifest.json"
    manifest = _load_object(manifest_path, "provisional label manifest")
    required_keys = {
        "artifact_schema_version",
        "builder_code_version",
        "series_ticker",
        "label_policy",
        "input_manifests",
        "inputs",
        "artifact_id",
        "outputs",
    }
    if set(manifest) != required_keys:
        raise ProvisionalKalshiLabelError("provisional label manifest is malformed")
    if (
        manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("builder_code_version") != BUILDER_CODE_VERSION
        or manifest.get("series_ticker") != TARGET_SERIES
    ):
        raise ProvisionalKalshiLabelError("provisional label version is unsupported")

    manifest_inputs = manifest.get("input_manifests")
    if not isinstance(manifest_inputs, list):
        raise ProvisionalKalshiLabelError("input manifest descriptors are invalid")
    if any(not isinstance(item, Mapping) for item in manifest_inputs):
        raise ProvisionalKalshiLabelError("input manifest descriptor is invalid")
    source_paths = [root / str(item.get("path")) for item in manifest_inputs]
    actual_manifests, actual_inputs, records = _load_verified_inputs(root, source_paths)
    identity = _identity(actual_manifests, actual_inputs)
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise ProvisionalKalshiLabelError("provisional label inputs changed")
    artifact_id = _sha256_bytes(_canonical_json(identity).encode("utf-8"))
    if (
        manifest.get("artifact_id") != artifact_id
        or directory.name != f"artifact={artifact_id}"
    ):
        raise ProvisionalKalshiLabelError("provisional label content address mismatch")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 3:
        raise ProvisionalKalshiLabelError("provisional label outputs are invalid")
    by_name: dict[str, Mapping[str, Any]] = {}
    for descriptor in outputs:
        if not isinstance(descriptor, Mapping):
            raise ProvisionalKalshiLabelError("output descriptor is invalid")
        name = descriptor.get("path")
        if name not in {"labels.jsonl", "exclusions.jsonl", "summary.json"}:
            raise ProvisionalKalshiLabelError("output descriptor path is invalid")
        if name in by_name:
            raise ProvisionalKalshiLabelError("duplicate output descriptor")
        expected_descriptor_keys = (
            {"path", "size_bytes", "sha256", "records"}
            if name != "summary.json"
            else {"path", "size_bytes", "sha256"}
        )
        if set(descriptor) != expected_descriptor_keys:
            raise ProvisionalKalshiLabelError("output descriptor is malformed")
        by_name[str(name)] = descriptor
        path = _inside(directory, directory / str(name), "artifact output")
        if path.is_symlink() or not path.is_file():
            raise ProvisionalKalshiLabelError("artifact output is missing or unsafe")
        if path.stat().st_size != descriptor.get("size_bytes") or _sha256_file(
            path
        ) != descriptor.get("sha256"):
            raise ProvisionalKalshiLabelError("artifact output checksum mismatch")

    labels_source, _ = _read_verified_jsonl(
        directory / "labels.jsonl",
        stream="labels",
        root=root,
        manifest_path=_relative(root, manifest_path),
        run_id=artifact_id,
        allow_empty=True,
    )
    exclusions_source, _ = _read_verified_jsonl(
        directory / "exclusions.jsonl",
        stream="exclusions",
        root=root,
        manifest_path=_relative(root, manifest_path),
        run_id=artifact_id,
        allow_empty=True,
    )
    labels = [record.row for record in labels_source]
    exclusions = [record.row for record in exclusions_source]
    if by_name["labels.jsonl"].get("records") != len(labels):
        raise ProvisionalKalshiLabelError("label row count changed")
    if by_name["exclusions.jsonl"].get("records") != len(exclusions):
        raise ProvisionalKalshiLabelError("exclusion row count changed")
    expected_labels, expected_exclusions, ignored = _rows_for_inputs(
        artifact_id, records
    )
    if _strip_chain(labels) != expected_labels:
        raise ProvisionalKalshiLabelError("provisional labels do not match inputs")
    if _strip_chain(exclusions) != expected_exclusions:
        raise ProvisionalKalshiLabelError("label exclusions do not match inputs")
    summary = _load_object(directory / "summary.json", "provisional label summary")
    expected_summary = _summary(
        artifact_id, expected_labels, expected_exclusions, ignored, records
    )
    if summary != expected_summary:
        raise ProvisionalKalshiLabelError("provisional label summary is inconsistent")
    return ProvisionalKalshiLabelArtifact(
        artifact_id=artifact_id,
        directory=directory,
        labels_path=directory / "labels.jsonl",
        exclusions_path=directory / "exclusions.jsonl",
        summary_path=directory / "summary.json",
        manifest_path=manifest_path,
        summary=summary,
    )
