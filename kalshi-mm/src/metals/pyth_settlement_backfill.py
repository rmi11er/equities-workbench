"""Content-addressed planning and bounded execution for exact settlement history.

The planner is offline. It derives one inclusive Pyth 200 ms range request for the
minute immediately preceding each unique opening or settlement boundary in a verified
settlement audit. The executor is explicit, sequential, resumable, and bounded by a
maximum number of boundary ranges per invocation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from .historical_research_assumption import require_historical_operation
from .pyth_exact_history import (
    PYTH_EXACT_HISTORY_CHANNEL,
    PythExactHistoryArchive,
    PythExactHistoryError,
    PythExactHistoryRequest,
    PythExactHistoryResult,
)
from .settlement_audit import SettlementAuditError, verify_settlement_audit
from .storage import update_run_manifest, write_run_manifest

PLAN_SCHEMA_VERSION = "metals-pyth-settlement-backfill-plan/v1"
_ONE_MINUTE_NS = 60_000_000_000
_ONE_MINUTE_US = 60_000_000
_HEX_DIGITS = frozenset("0123456789abcdef")
_NON_BLOCKING_AUDIT_EXCLUSIONS = frozenset(
    {
        "timestamp_semantics_unconfirmed",
        "correction_semantics_unconfirmed",
        "rounding_semantics_unconfirmed",
        "missing_opening_exact_history_coverage",
        "missing_settlement_exact_history_coverage",
        "missing_opening_exact_history_publication",
        "missing_settlement_exact_history_publication",
    }
)


class PythSettlementBackfillError(RuntimeError):
    """A settlement-boundary plan or bounded run failed validation."""


@dataclass(frozen=True)
class PythSettlementBackfillPlan:
    plan_id: str
    directory: Path
    plan_path: Path
    manifest_path: Path
    value: dict[str, Any]

    def public_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_path": str(self.plan_path),
            "manifest_path": str(self.manifest_path),
            "window_count": self.value["window_count"],
            "planned_window_count": self.value["planned_window_count"],
            "invalid_window_count": len(self.value["invalid_windows"]),
            "boundary_count": self.value["boundary_count"],
            "maximum_provider_pages": self.value["maximum_provider_pages"],
        }


@dataclass(frozen=True)
class PythSettlementBackfillRunResult:
    run_id: str
    manifest_path: Path
    plan_id: str
    total_boundaries: int
    completed_before: int
    attempted_boundaries: int
    completed_this_run: int
    completed_after: int
    remaining_boundaries: int
    provider_requests_made: int
    rows_archived: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "manifest_path": str(self.manifest_path),
            "plan_id": self.plan_id,
            "total_boundaries": self.total_boundaries,
            "completed_before": self.completed_before,
            "attempted_boundaries": self.attempted_boundaries,
            "completed_this_run": self.completed_this_run,
            "completed_after": self.completed_after,
            "remaining_boundaries": self.remaining_boundaries,
            "provider_requests_made": self.provider_requests_made,
            "rows_archived": self.rows_archived,
            "plan_completed": self.remaining_boundaries == 0,
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PythSettlementBackfillError(f"Cannot read {label}") from error
    if not isinstance(value, dict):
        raise PythSettlementBackfillError(f"{label} must be a JSON object")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value.lower())
    )


def _resolve_inside(root: Path, path: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PythSettlementBackfillError(
            f"{label} escapes the storage root"
        ) from error
    return resolved


def _audit_descriptor(
    root: Path,
    artifact_path: Path,
    verified: Mapping[str, Any],
) -> dict[str, Any]:
    requested = _resolve_inside(root, artifact_path, "Settlement audit")
    directory = requested.parent if requested.name == "manifest.json" else requested
    manifest_path = directory / "manifest.json"
    manifest = verified["manifest"]
    windows = next(
        (
            item
            for item in manifest.get("outputs", [])
            if isinstance(item, Mapping) and item.get("path") == "windows.jsonl"
        ),
        None,
    )
    if windows is None or not _is_sha256(windows.get("sha256")):
        raise PythSettlementBackfillError(
            "Verified settlement audit has no valid windows descriptor"
        )
    return {
        "path": str(directory.relative_to(root)),
        "audit_id": manifest.get("audit_id"),
        "manifest_sha256": _sha256_path(manifest_path),
        "windows_sha256": windows["sha256"],
    }


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _request_from_entry(value: Mapping[str, Any]) -> PythExactHistoryRequest:
    request_value = value.get("request")
    if not isinstance(request_value, Mapping):
        raise PythSettlementBackfillError(
            "Planned request is missing its request object"
        )
    endpoint = request_value.get("endpoint")
    channel = request_value.get("channel")
    suffix = f"/{channel}/price/range"
    if not isinstance(endpoint, str) or not endpoint.endswith(suffix):
        raise PythSettlementBackfillError("Planned Pyth endpoint is invalid")
    feed_ids = request_value.get("feed_ids")
    if not isinstance(feed_ids, list):
        raise PythSettlementBackfillError("Planned Pyth feed IDs are invalid")
    try:
        request = PythExactHistoryRequest(
            metal=request_value.get("metal"),
            feed_ids=tuple(feed_ids),
            start_timestamp_us=request_value.get("start_timestamp_us"),
            end_timestamp_us=request_value.get("end_timestamp_us"),
            limit=request_value.get("limit"),
            max_pages=request_value.get("max_pages"),
            base_url=endpoint[: -len(suffix)],
            channel=channel,
        )
    except (TypeError, ValueError) as error:
        raise PythSettlementBackfillError("Planned Pyth request is invalid") from error
    if dict(request_value) != request.public_dict():
        raise PythSettlementBackfillError("Planned Pyth request is not canonical")
    if value.get("request_id") != request.request_id:
        raise PythSettlementBackfillError("Planned Pyth request identity is invalid")
    return request


def _build_plan_payload(
    *,
    source_audit: Mapping[str, Any],
    audit_manifest: Mapping[str, Any],
    audit_rows: Sequence[Mapping[str, Any]],
    metal: str,
    feed_id: int,
    base_url: str,
    limit: int,
    max_pages: int,
) -> dict[str, Any]:
    if audit_manifest.get("metal") != metal:
        raise PythSettlementBackfillError(
            "Settlement audit metal does not match the requested plan"
        )
    if audit_manifest.get("exact_feed_id") != feed_id:
        raise PythSettlementBackfillError(
            "Settlement audit exact feed ID does not match the requested plan"
        )
    audit_id = audit_manifest.get("audit_id")
    if not _is_sha256(audit_id):
        raise PythSettlementBackfillError("Settlement audit ID is invalid")

    boundaries: dict[int, list[dict[str, str]]] = defaultdict(list)
    invalid_windows: list[dict[str, Any]] = []
    seen_tickers: set[str] = set()
    planned_windows = 0
    for row in sorted(
        audit_rows, key=lambda item: str(item.get("market_ticker") or "")
    ):
        ticker = row.get("market_ticker")
        if not isinstance(ticker, str) or not ticker:
            raise PythSettlementBackfillError(
                "Settlement audit contains a missing market ticker"
            )
        if ticker in seen_tickers:
            raise PythSettlementBackfillError(
                "Settlement audit contains a duplicate market ticker"
            )
        seen_tickers.add(ticker)
        reasons: list[str] = []
        if row.get("audit_id") != audit_id:
            reasons.append("audit_identity_mismatch")
        if row.get("metal") != metal:
            reasons.append("metal_mismatch_or_missing")
        if _integer(row.get("pyth_feed_id")) != feed_id:
            reasons.append("pyth_feed_id_mismatch_or_missing")
        exclusions = row.get("exclusion_reasons")
        if not isinstance(exclusions, list) or any(
            not isinstance(reason, str) or not reason for reason in exclusions
        ):
            reasons.append("invalid_audit_exclusion_list")
        else:
            blocking = sorted(
                set(exclusions).difference(_NON_BLOCKING_AUDIT_EXCLUSIONS)
            )
            if blocking:
                reasons.append("non_exact_audit_exclusions:" + ",".join(blocking))

        open_ns = _integer(row.get("open_ts_ns"))
        close_ns = _integer(row.get("close_ts_ns"))
        if open_ns is None or open_ns <= _ONE_MINUTE_NS or open_ns % _ONE_MINUTE_NS:
            reasons.append("invalid_open_boundary")
        if close_ns is None or close_ns <= _ONE_MINUTE_NS or close_ns % _ONE_MINUTE_NS:
            reasons.append("invalid_close_boundary")
        if open_ns is not None and close_ns is not None and close_ns <= open_ns:
            reasons.append("invalid_window_order")
        if reasons:
            invalid_windows.append(
                {"market_ticker": ticker, "reasons": sorted(set(reasons))}
            )
            continue

        assert open_ns is not None and close_ns is not None
        boundaries[open_ns].append(
            {"market_ticker": ticker, "boundary_role": "opening"}
        )
        boundaries[close_ns].append(
            {"market_ticker": ticker, "boundary_role": "settlement"}
        )
        planned_windows += 1

    requests: list[dict[str, Any]] = []
    for boundary_ns in sorted(boundaries):
        boundary_us = boundary_ns // 1_000
        request = PythExactHistoryRequest(
            metal=metal,
            feed_ids=(feed_id,),
            start_timestamp_us=boundary_us - _ONE_MINUTE_US,
            end_timestamp_us=boundary_us - 1,
            limit=limit,
            max_pages=max_pages,
            base_url=base_url,
        )
        requests.append(
            {
                "boundary_ts_ns": boundary_ns,
                "request_id": request.request_id,
                "request": request.public_dict(),
                "uses": sorted(
                    boundaries[boundary_ns],
                    key=lambda item: (item["market_ticker"], item["boundary_role"]),
                ),
            }
        )
    if not requests:
        raise PythSettlementBackfillError(
            "No structurally valid settlement boundaries are available to plan"
        )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_audit": dict(source_audit),
        "metal": metal,
        "feed_id": feed_id,
        "request_parameters": {
            "base_url": requests[0]["request"]["endpoint"].removesuffix(
                f"/{PYTH_EXACT_HISTORY_CHANNEL}/price/range"
            ),
            "channel": PYTH_EXACT_HISTORY_CHANNEL,
            "limit": limit,
            "max_pages": max_pages,
            "bounds": "inclusive minute immediately preceding each boundary",
        },
        "window_count": len(audit_rows),
        "planned_window_count": planned_windows,
        "invalid_windows": invalid_windows,
        "boundary_count": len(requests),
        "maximum_provider_pages": len(requests) * max_pages,
        "requests": requests,
    }


def create_pyth_settlement_backfill_plan(
    storage_root: Path,
    settlement_audit_path: Path,
    *,
    metal: str,
    feed_id: int,
    base_url: str,
    limit: int = 1_000,
    max_pages: int = 2,
) -> PythSettlementBackfillPlan:
    """Create or verify one deterministic offline boundary-range plan."""

    normalized_metal = metal.strip().lower() if isinstance(metal, str) else ""
    if not normalized_metal or "/" in normalized_metal or ".." in normalized_metal:
        raise ValueError("metal must be a path-safe non-empty label")
    if isinstance(feed_id, bool) or not isinstance(feed_id, int) or feed_id < 0:
        raise ValueError("feed_id must be a non-negative numeric integer")
    root = Path(storage_root).resolve()
    try:
        verified = verify_settlement_audit(root, settlement_audit_path)
    except SettlementAuditError as error:
        raise PythSettlementBackfillError(
            "Settlement audit failed verification"
        ) from error
    source = _audit_descriptor(root, Path(settlement_audit_path), verified)
    payload = _build_plan_payload(
        source_audit=source,
        audit_manifest=verified["manifest"],
        audit_rows=verified["windows"],
        metal=normalized_metal,
        feed_id=feed_id,
        base_url=base_url,
        limit=limit,
        max_pages=max_pages,
    )
    plan_id = _sha256_bytes(_canonical_bytes(payload))
    plan = {"plan_id": plan_id, **payload}
    destination = root / "plans" / "pyth_settlement_exact" / f"plan={plan_id}"
    if destination.exists():
        return verify_pyth_settlement_backfill_plan(root, destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".pyth-settlement-plan-", dir=destination.parent)
    )
    try:
        plan_path = temporary / "plan.json"
        _write_json(plan_path, plan)
        manifest = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "plan_id": plan_id,
            "source_audit": source,
            "metal": normalized_metal,
            "feed_id": feed_id,
            "outputs": [
                {
                    "path": "plan.json",
                    "size_bytes": plan_path.stat().st_size,
                    "sha256": _sha256_path(plan_path),
                }
            ],
        }
        _write_json(temporary / "manifest.json", manifest)
        try:
            os.rename(temporary, destination)
        except FileExistsError:
            shutil.rmtree(temporary)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verify_pyth_settlement_backfill_plan(root, destination)


def verify_pyth_settlement_backfill_plan(
    storage_root: Path, plan_path: Path
) -> PythSettlementBackfillPlan:
    """Verify plan content address, source audit, and every derived request."""

    root = Path(storage_root).resolve()
    requested = _resolve_inside(root, plan_path, "Pyth settlement plan")
    if requested.name in {"plan.json", "manifest.json"}:
        directory = requested.parent
    else:
        directory = requested
    try:
        relative = directory.relative_to(root)
    except ValueError as error:  # pragma: no cover - guarded by _resolve_inside
        raise PythSettlementBackfillError("Plan escapes the storage root") from error
    if len(relative.parts) < 3 or relative.parts[-2] != "pyth_settlement_exact":
        raise PythSettlementBackfillError("Pyth settlement plan location is invalid")
    if not directory.name.startswith("plan="):
        raise PythSettlementBackfillError("Pyth settlement plan directory is invalid")

    plan_file = directory / "plan.json"
    manifest_file = directory / "manifest.json"
    plan = _read_object(plan_file, "Pyth settlement plan")
    manifest = _read_object(manifest_file, "Pyth settlement plan manifest")
    plan_id = plan.get("plan_id")
    if not _is_sha256(plan_id) or directory.name != f"plan={plan_id}":
        raise PythSettlementBackfillError("Pyth settlement plan identity is invalid")
    payload = dict(plan)
    payload.pop("plan_id", None)
    if _sha256_bytes(_canonical_bytes(payload)) != plan_id:
        raise PythSettlementBackfillError(
            "Pyth settlement plan content hash is invalid"
        )
    if (
        manifest.get("plan_id") != plan_id
        or manifest.get("schema_version") != PLAN_SCHEMA_VERSION
    ):
        raise PythSettlementBackfillError("Pyth settlement plan manifest is invalid")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise PythSettlementBackfillError(
            "Pyth settlement plan output descriptor is invalid"
        )
    output = outputs[0]
    if not isinstance(output, Mapping) or output.get("path") != "plan.json":
        raise PythSettlementBackfillError("Pyth settlement plan output path is invalid")
    if output.get("size_bytes") != plan_file.stat().st_size or output.get(
        "sha256"
    ) != _sha256_path(plan_file):
        raise PythSettlementBackfillError("Pyth settlement plan output changed")

    source = plan.get("source_audit")
    if not isinstance(source, Mapping) or manifest.get("source_audit") != source:
        raise PythSettlementBackfillError(
            "Pyth settlement plan source audit is invalid"
        )
    audit_relative = source.get("path")
    if not isinstance(audit_relative, str) or not audit_relative:
        raise PythSettlementBackfillError(
            "Pyth settlement source audit path is invalid"
        )
    audit_path = _resolve_inside(root, root / audit_relative, "Settlement audit")
    try:
        verified = verify_settlement_audit(root, audit_path)
    except SettlementAuditError as error:
        raise PythSettlementBackfillError(
            "Pyth settlement source audit failed verification"
        ) from error
    expected_source = _audit_descriptor(root, audit_path, verified)
    if dict(source) != expected_source:
        raise PythSettlementBackfillError("Pyth settlement source audit changed")

    params = plan.get("request_parameters")
    if not isinstance(params, Mapping):
        raise PythSettlementBackfillError(
            "Pyth settlement request parameters are invalid"
        )
    expected_payload = _build_plan_payload(
        source_audit=expected_source,
        audit_manifest=verified["manifest"],
        audit_rows=verified["windows"],
        metal=str(plan.get("metal") or ""),
        feed_id=plan.get("feed_id"),
        base_url=params.get("base_url"),
        limit=params.get("limit"),
        max_pages=params.get("max_pages"),
    )
    if _canonical_bytes(payload) != _canonical_bytes(expected_payload):
        raise PythSettlementBackfillError(
            "Pyth settlement plan does not match its verified source audit"
        )
    requests = plan.get("requests")
    if not isinstance(requests, list):
        raise PythSettlementBackfillError("Pyth settlement plan requests are invalid")
    seen_request_ids = set()
    seen_boundaries = set()
    for entry in requests:
        if not isinstance(entry, Mapping):
            raise PythSettlementBackfillError("Pyth settlement plan request is invalid")
        request = _request_from_entry(entry)
        boundary = _integer(entry.get("boundary_ts_ns"))
        if boundary is None or boundary in seen_boundaries:
            raise PythSettlementBackfillError(
                "Planned boundary is invalid or duplicated"
            )
        if request.request_id in seen_request_ids:
            raise PythSettlementBackfillError("Planned request identity is duplicated")
        seen_boundaries.add(boundary)
        seen_request_ids.add(request.request_id)
    return PythSettlementBackfillPlan(
        plan_id=plan_id,
        directory=directory,
        plan_path=plan_file,
        manifest_path=manifest_file,
        value=plan,
    )


def _archive_paths(
    request: PythExactHistoryRequest, result: PythExactHistoryResult
) -> set[Path]:
    paths = {
        result.checkpoint_path,
        result.checkpoint_path.parent / "request.json",
        *result.raw_paths,
        *result.normalized_paths,
    }
    return {path for path in paths if path.exists()}


async def run_pyth_settlement_backfill(
    storage_root: Path,
    plan_path: Path,
    *,
    api_key_env: str = "PYTH_API_KEY",
    environ: Mapping[str, str] | None = None,
    allow_network_download: bool = False,
    max_boundaries: int = 100,
    request_interval_seconds: float = 0.1,
    timeout_seconds: float = 30.0,
    session: Any | None = None,
    provider_approval_path: Path | str = "",
    historical_research_assumption_path: Path | str = "",
) -> PythSettlementBackfillRunResult:
    """Resume a bounded number of planned ranges with one sequential HTTP session."""

    if allow_network_download is not True:
        raise ValueError("Bulk Pyth history requires --allow-network-download")
    if (
        isinstance(max_boundaries, bool)
        or not isinstance(max_boundaries, int)
        or not 1 <= max_boundaries <= 10_000
    ):
        raise ValueError("max_boundaries must be an integer in [1, 10000]")
    if (
        isinstance(request_interval_seconds, bool)
        or not isinstance(request_interval_seconds, (int, float))
        or not math.isfinite(request_interval_seconds)
        or not 0 <= request_interval_seconds <= 60
    ):
        raise ValueError("request_interval_seconds must be finite and in [0, 60]")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 60
    ):
        raise ValueError("timeout_seconds must be finite and in (0, 60]")
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        raise ValueError("api_key_env must be a non-empty environment variable name")

    root = Path(storage_root).resolve()
    plan = verify_pyth_settlement_backfill_plan(root, plan_path)
    authority = require_historical_operation(
        root,
        provider_approval_path,
        historical_research_assumption_path,
        "pyth",
        "exact_history",
    )
    environment = os.environ if environ is None else environ
    api_key = environment.get(api_key_env, "")
    if not isinstance(api_key, str) or not api_key.strip():
        raise PythSettlementBackfillError(
            f"Set {api_key_env} to an entitled Pyth Pro API key"
        )

    entries = plan.value["requests"]
    requests = [_request_from_entry(entry) for entry in entries]
    run_id = str(uuid.uuid4())
    manifest_path = root / "manifests" / f"run={run_id}" / "manifest.json"
    write_run_manifest(
        root,
        run_id,
        {
            "collector": {
                "component": "pyth_settlement_exact_backfill",
                "read_only_provider_access": True,
                "submits_orders": False,
                "plan_id": plan.plan_id,
                "plan_path": str(plan.plan_path.relative_to(root)),
                "max_boundaries": max_boundaries,
                "maximum_provider_pages_this_run": max_boundaries
                * int(plan.value["request_parameters"]["max_pages"]),
                "request_interval_seconds": float(request_interval_seconds),
                "timeout_seconds": float(timeout_seconds),
                **authority.manifest_fields(),
            }
        },
        counts={"total_boundaries": len(requests)},
        output_paths=authority.output_paths,
    )

    completed_before = 0
    attempted = 0
    completed_this_run = 0
    provider_requests = 0
    rows_archived = 0
    completed_after = 0
    remaining = len(requests)
    output_paths: set[Path] = set()
    failure: BaseException | None = None
    archive: PythExactHistoryArchive | None = None
    active_request: PythExactHistoryRequest | None = None
    owns_session = session is None
    active_session = session
    try:
        if active_session is None:
            timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
            active_session = aiohttp.ClientSession(
                timeout=timeout, auto_decompress=False
            )
        archive = PythExactHistoryArchive(
            root,
            api_key=api_key,
            session=active_session,
            timeout_seconds=timeout_seconds,
        )
        pending: list[PythExactHistoryRequest] = []
        for request in requests:
            prepared = archive.prepare_resume(request)
            if prepared is not None:
                output_paths.update(_archive_paths(request, prepared))
            if prepared is not None and prepared.completed:
                completed_before += 1
            else:
                pending.append(request)

        selected = pending[:max_boundaries]
        for index, request in enumerate(selected):
            active_request = request
            attempted += 1
            result = await archive.download(request)
            if not result.completed:
                raise PythExactHistoryError(
                    "Pyth exact-history boundary request did not complete"
                )
            completed_this_run += 1
            provider_requests = archive.network_requests_attempted
            rows_archived += result.row_count
            output_paths.update(_archive_paths(request, result))
            completed_after = completed_before + completed_this_run
            remaining = len(requests) - completed_after
            if request_interval_seconds and index + 1 < len(selected):
                await asyncio.sleep(float(request_interval_seconds))
        active_request = None
        completed_after = completed_before + completed_this_run
        remaining = len(requests) - completed_after
    except BaseException as error:
        failure = error
        if archive is not None:
            provider_requests = archive.network_requests_attempted
        if archive is not None and active_request is not None:
            try:
                prepared = archive.prepare_resume(active_request)
                if prepared is not None:
                    output_paths.update(_archive_paths(active_request, prepared))
            except BaseException:
                # Preserve the primary network/protocol/integrity error. Any partial
                # state will be reverified on the next explicitly authorized run.
                pass
    finally:
        if owns_session and active_session is not None:
            try:
                await active_session.close()
            except BaseException as close_error:
                if failure is None:
                    failure = close_error
        counts = {
            "total_boundaries": len(requests),
            "completed_before": completed_before,
            "attempted_boundaries": attempted,
            "completed_this_run": completed_this_run,
            "completed_after": completed_after,
            "remaining_boundaries": remaining,
            "provider_requests_made": provider_requests,
            "rows_archived": rows_archived,
        }
        update_run_manifest(
            root,
            run_id,
            status="failed" if failure is not None else "completed",
            counts=counts,
            output_paths=sorted(output_paths),
            errors=[failure] if failure is not None else [],
        )
    if failure is not None:
        raise failure
    return PythSettlementBackfillRunResult(
        run_id=run_id,
        manifest_path=manifest_path,
        plan_id=plan.plan_id,
        total_boundaries=len(requests),
        completed_before=completed_before,
        attempted_boundaries=attempted,
        completed_this_run=completed_this_run,
        completed_after=completed_after,
        remaining_boundaries=remaining,
        provider_requests_made=provider_requests,
        rows_archived=rows_archived,
    )
