"""Cost-gated, resumable Databento historical archives for CME metals.

This module deliberately has no trading or order-entry dependency.  A request
is fully bounded, estimates its Databento cost before any new time-series or
batch request, and requires an explicit opt-in before an operation that can be
billable.  API keys are used only to construct a client; they are never placed
in paths, records, manifests, errors, or logs.

Native DBN is the authoritative artifact.  JSONL is a deterministic research
projection produced by :func:`src.metals.databento.normalize_databento_record`.
Parent symbology is resolved by Databento, while daily instrument definitions
are retained so instrument IDs can be joined to raw CME contract symbols
without imposing a locally guessed roll calendar.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from .databento import GLBX_DATASET, normalize_databento_record
from .historical_research_assumption import HistoricalOperationAuthority
from .provider_approval import ProviderApprovalArtifact
from .storage import HashChainedJsonlWriter, update_run_manifest, write_run_manifest

HISTORICAL_SCHEMAS = ("definition", "mbo", "trades", "status", "statistics")
METAL_PARENT_SYMBOLS: Mapping[str, Tuple[str, ...]] = {
    "silver": ("SI.FUT", "SIL.FUT", "SIC.FUT"),
    "gold": ("GC.FUT", "MGC.FUT", "1OZ.FUT", "QO.FUT"),
    "copper": ("HG.FUT", "MHG.FUT"),
}


class DatabentoHistoricalError(RuntimeError):
    """Base class for safe historical-ingestion errors."""


class DatabentoHistoricalDependencyError(DatabentoHistoricalError):
    """The optional Databento package is needed for a real client or DBN read."""


class DatabentoHistoricalAccessError(DatabentoHistoricalError):
    """Authentication, entitlement, billing, or vendor request failure."""


class BillableDownloadNotAuthorized(DatabentoHistoricalError):
    """A new billable request was stopped after its cost preflight."""

    def __init__(self, estimate: "HistoricalCostEstimate") -> None:
        self.estimate = estimate
        super().__init__(
            "Databento retrieval was not authorized; estimated cost is "
            "${:.6f}. Pass allow_billable_download=True only after reviewing "
            "the preflight estimate.".format(estimate.total_cost_usd)
        )


class DatabentoBatchPending(DatabentoHistoricalError):
    """Batch jobs exist but have not all reached the done state."""


def _timestamp_ns(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(
            "{} must be an integer nanosecond timestamp or aware datetime".format(field)
        )
    if isinstance(value, int):
        if value < 0:
            raise ValueError("{} must not be negative".format(field))
        return value
    if isinstance(value, str):
        parsed = value.strip()
        if parsed.endswith("Z"):
            parsed = parsed[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(parsed)
        except ValueError as error:
            raise ValueError(
                "{} must be an ISO 8601 timestamp with an offset".format(field)
            ) from error
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("{} datetime must include a timezone offset".format(field))
        utc_value = value.astimezone(timezone.utc)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        delta = utc_value - epoch
        return (
            delta.days * 86_400 * 1_000_000_000
            + delta.seconds * 1_000_000_000
            + delta.microseconds * 1_000
        )
    raise ValueError(
        "{} must be an integer nanosecond timestamp or aware datetime".format(field)
    )


def _iso_utc(timestamp_ns: int) -> str:
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    base = datetime.fromtimestamp(seconds, timezone.utc)
    return "{}.{:09d}Z".format(base.strftime("%Y-%m-%dT%H:%M:%S"), nanoseconds)


def _utc_date(timestamp_ns: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ns / 1_000_000_000, timezone.utc)
        .date()
        .isoformat()
    )


@dataclass(frozen=True)
class DatabentoHistoricalRequest:
    """One exact, bounded GLBX.MDP3 request across one metal's parent roots."""

    metal: str
    start: Any
    end: Any
    schemas: Tuple[str, ...] = HISTORICAL_SCHEMAS
    dataset: str = GLBX_DATASET

    def __post_init__(self) -> None:
        metal = self.metal.strip().lower()
        if metal not in METAL_PARENT_SYMBOLS:
            raise ValueError(
                "Unsupported metal {!r}; choose one of {}".format(
                    self.metal, ", ".join(sorted(METAL_PARENT_SYMBOLS))
                )
            )
        if self.dataset != GLBX_DATASET:
            raise ValueError("Historical metals ingestion is locked to GLBX.MDP3")
        start_ns = _timestamp_ns(self.start, "start")
        end_ns = _timestamp_ns(self.end, "end")
        if end_ns <= start_ns:
            raise ValueError("end must be strictly after start")

        schemas = tuple(str(value).strip().lower() for value in self.schemas)
        if not schemas or any(not schema for schema in schemas):
            raise ValueError("At least one historical schema is required")
        if len(set(schemas)) != len(schemas):
            raise ValueError("Historical schemas must be unique")
        unknown = sorted(set(schemas) - set(HISTORICAL_SCHEMAS))
        if unknown:
            raise ValueError(
                "Unsupported historical schemas: {}".format(", ".join(unknown))
            )
        ordered = tuple(schema for schema in HISTORICAL_SCHEMAS if schema in schemas)
        object.__setattr__(self, "metal", metal)
        object.__setattr__(self, "start", start_ns)
        object.__setattr__(self, "end", end_ns)
        object.__setattr__(self, "schemas", ordered)

    @property
    def symbols(self) -> Tuple[str, ...]:
        return METAL_PARENT_SYMBOLS[self.metal]

    def public_dict(self) -> Dict[str, Any]:
        return {
            "source": "databento_historical",
            "dataset": self.dataset,
            "metal": self.metal,
            "symbols": list(self.symbols),
            "stype_in": "parent",
            "stype_out": "instrument_id",
            "schemas": list(self.schemas),
            "start_ns": self.start,
            "end_ns": self.end,
            "start": _iso_utc(self.start),
            "end": _iso_utc(self.end),
        }

    @property
    def request_id(self) -> str:
        canonical = json.dumps(
            self.public_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @property
    def resume_scope_id(self) -> str:
        """Identify one bounded retrieval independently of its schema selection."""

        scope = self.public_dict()
        del scope["schemas"]
        canonical = json.dumps(scope, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class HistoricalCostEstimate:
    request_id: str
    schemas: Tuple[str, ...]
    per_schema_cost_usd: Mapping[str, float]
    total_cost_usd: float
    estimated_at: str
    caveat: str = (
        "Databento bills actual bytes sent; get_cost may over-report ranges not "
        "aligned to 10 minutes, and definition estimates not aligned to 24 hours."
    )
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class HistoricalArchiveResult:
    request_id: str
    transport: str
    completed: bool
    record_counts: Mapping[str, int]
    skipped_schemas: Tuple[str, ...]
    output_paths: Tuple[Path, ...]
    estimate: Optional[HistoricalCostEstimate] = None
    batch_states: Optional[Mapping[str, str]] = None


def _import_databento() -> Any:
    try:
        import databento
    except ImportError as error:  # pragma: no cover - depends on local extras
        raise DatabentoHistoricalDependencyError(
            "Databento historical ingestion requires the optional 'databento' "
            "package on Python 3.10 or newer; install the project requirements."
        ) from error
    return databento


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return format(Decimal(str(value)), "f")
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, bytes):
        try:
            return value.rstrip(b"\x00").decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).lower() not in {"api_key", "key", "authorization"}
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            mapped = to_dict()
        except (TypeError, ValueError):
            mapped = None
        if isinstance(mapped, Mapping):
            return _json_safe(mapped)
    values = getattr(value, "__dict__", None)
    if isinstance(values, Mapping):
        return _json_safe(values)
    return str(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".databento-history-", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
        _fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _error_status(error: BaseException) -> Optional[int]:
    for candidate in (
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        if callable(candidate):
            try:
                candidate = candidate()
            except TypeError:
                candidate = None
        if isinstance(candidate, int):
            return candidate
    return None


def _safe_access_error(
    action: str, error: BaseException
) -> DatabentoHistoricalAccessError:
    status = _error_status(error)
    if status == 401:
        detail = "authentication failed; verify DATABENTO_API_KEY"
    elif status == 402:
        detail = "billing authorization failed; review the Databento account"
    elif status == 403:
        detail = (
            "the key is not entitled to this GLBX.MDP3 schema/range; review CME "
            "historical licensing and Databento permissions"
        )
    elif status == 404:
        detail = "the requested dataset, symbols, or range were not found"
    elif status == 429:
        detail = "the Databento rate limit was reached; retry after the vendor delay"
    elif status is not None and status >= 500:
        detail = "Databento's historical service is unavailable; retry later"
    else:
        detail = "the vendor request failed ({})".format(type(error).__name__)
    return DatabentoHistoricalAccessError("Unable to {}: {}.".format(action, detail))


class DatabentoHistoricalArchive:
    """Archive and normalize bounded Databento streaming or batch requests."""

    def __init__(
        self,
        root: Path,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        dbn_loader: Optional[Callable[[Path], Any]] = None,
        provider_approval_id: Optional[str] = None,
        provider_approval: Optional[ProviderApprovalArtifact] = None,
        historical_authority: Optional[HistoricalOperationAuthority] = None,
    ) -> None:
        self.root = Path(root)
        if historical_authority is not None:
            if provider_approval is not None or provider_approval_id is not None:
                raise ValueError(
                    "historical authority cannot be combined with provider "
                    "approval arguments"
                )
            provider_approval = historical_authority.provider_approval
            if provider_approval is not None:
                provider_approval_id = provider_approval.approval_id
        if provider_approval is not None:
            if (
                provider_approval_id is not None
                and provider_approval_id != provider_approval.approval_id
            ):
                raise ValueError("provider approval identity does not match artifact")
            provider_approval_id = provider_approval.approval_id
        if provider_approval_id is not None and (
            not isinstance(provider_approval_id, str)
            or len(provider_approval_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in provider_approval_id.lower()
            )
        ):
            raise ValueError("provider_approval_id must be a SHA-256 or null")
        self.provider_approval_id = provider_approval_id
        self.provider_approval_paths: Tuple[Path, ...] = ()
        if provider_approval is not None:
            resolved_root = self.root.resolve()
            resolved_paths = (
                provider_approval.approval_path.resolve(),
                provider_approval.manifest_path.resolve(),
            )
            for path in resolved_paths:
                try:
                    path.relative_to(resolved_root)
                except ValueError as error:
                    raise ValueError(
                        "provider approval artifact escapes the storage root"
                    ) from error
                if not path.is_file():
                    raise ValueError("provider approval artifact is missing")
            self.provider_approval_paths = resolved_paths
        self.historical_research_assumption_id: Optional[str] = None
        self.historical_research_assumption_paths: Tuple[Path, ...] = ()
        if (
            historical_authority is not None
            and historical_authority.kind == "research_assumption"
        ):
            self.historical_research_assumption_id = historical_authority.authority_id
            resolved_root = self.root.resolve()
            resolved_paths = tuple(
                path.resolve() for path in historical_authority.output_paths
            )
            for path in resolved_paths:
                try:
                    path.relative_to(resolved_root)
                except ValueError as error:
                    raise ValueError(
                        "historical research assumption artifact escapes the "
                        "storage root"
                    ) from error
                if not path.is_file():
                    raise ValueError(
                        "historical research assumption artifact is missing"
                    )
            self.historical_research_assumption_paths = resolved_paths
        self.governance_paths = (
            self.provider_approval_paths or self.historical_research_assumption_paths
        )
        if client is None:
            if not isinstance(api_key, str) or not api_key.strip():
                raise DatabentoHistoricalAccessError(
                    "Set DATABENTO_API_KEY to a licensed Databento key."
                )
            databento = _import_databento()
            try:
                client = databento.Historical(api_key)
            except BaseException as error:
                raise _safe_access_error(
                    "create the Databento historical client", error
                ) from None
            dbn_loader = databento.DBNStore.from_file
        self.client = client
        self._dbn_loader = dbn_loader

    def preflight(
        self,
        request: DatabentoHistoricalRequest,
        schemas: Optional[Sequence[str]] = None,
    ) -> HistoricalCostEstimate:
        selected_values = (
            request.schemas
            if schemas is None
            else tuple(str(value).strip().lower() for value in schemas)
        )
        if not selected_values or any(not schema for schema in selected_values):
            raise ValueError("At least one cost-estimate schema is required")
        if len(set(selected_values)) != len(selected_values):
            raise ValueError("Cost-estimate schemas must be unique")
        outside_request = sorted(set(selected_values) - set(request.schemas))
        if outside_request:
            raise ValueError(
                "Cost-estimate schemas are outside the bounded request: {}".format(
                    ", ".join(outside_request)
                )
            )
        selected = tuple(
            schema for schema in request.schemas if schema in selected_values
        )
        costs: Dict[str, float] = {}
        for schema in selected:
            try:
                cost = self.client.metadata.get_cost(
                    dataset=request.dataset,
                    start=request.start,
                    end=request.end,
                    symbols=list(request.symbols),
                    schema=schema,
                    stype_in="parent",
                )
            except BaseException as error:
                raise _safe_access_error(
                    "estimate Databento cost for schema {!r}".format(schema), error
                ) from None
            costs[schema] = float(cost)
        return HistoricalCostEstimate(
            request_id=request.request_id,
            schemas=selected,
            per_schema_cost_usd=costs,
            total_cost_usd=sum(costs.values()),
            estimated_at=datetime.now(timezone.utc).isoformat(),
            warnings=tuple(
                warning
                for warning in (
                    "The requested range is inclusive at start and exclusive at end.",
                    (
                        "Definition records may include UTC-day reference-data "
                        "snapshots; snapshot emission is not an intraday instrument "
                        "change."
                        if "definition" in selected
                        else None
                    ),
                )
                if warning is not None
            ),
        )

    def download_stream(
        self,
        request: DatabentoHistoricalRequest,
        *,
        allow_billable_download: bool = False,
        max_cost_usd: Optional[float] = None,
    ) -> HistoricalArchiveResult:
        """Stream new DBN archives, then atomically publish normalized JSONL."""

        self._ensure_request_identity(request)

        valid = {
            schema: self._valid_completion_marker(request, schema, "stream")
            for schema in request.schemas
        }
        skipped = tuple(schema for schema, complete in valid.items() if complete)
        missing = tuple(schema for schema, complete in valid.items() if not complete)
        if not missing:
            result = self._completed_result(request, "stream", skipped, None)
            self._reconcile_completed_manifest(request, result)
            return result

        needs_download = tuple(
            schema
            for schema in missing
            if not self._native_path(request, schema).is_file()
        )
        estimate: Optional[HistoricalCostEstimate] = None
        if needs_download:
            estimate = self.preflight(request, needs_download)
            self._authorize(estimate, allow_billable_download, max_cost_usd)

        run_id = self._ensure_manifest(request, "stream")
        identities = self._load_identities(request)
        if "definition" in missing:
            identities = {}
        counts: Dict[str, int] = {}
        outputs: List[Path] = []
        if estimate is not None:
            update_run_manifest(
                self.root,
                run_id,
                counts={"estimated_cost_usd": estimate.total_cost_usd},
            )
        try:
            for schema in missing:
                native_path = self._native_path(request, schema)
                if native_path.is_file():
                    store = self._load_native(native_path)
                else:
                    store = self._stream_one(request, schema, native_path)
                metadata_path = self._metadata_path(request, schema, "stream")
                self._write_store_metadata(store, request, schema, metadata_path)
                count, normalized_path = self._normalize_stores(
                    request, schema, (store,), identities
                )
                counts[schema] = count
                if schema == "definition":
                    self._write_identities(request, identities)
                artifacts = [native_path, metadata_path, normalized_path]
                if schema == "definition":
                    artifacts.append(self._identity_path(request))
                self._write_completion_marker(
                    request, schema, "stream", artifacts, count
                )
                outputs.extend(artifacts)
                update_run_manifest(
                    self.root,
                    run_id,
                    counts={"{}_records".format(schema): count},
                    output_paths=artifacts,
                )
        except BaseException as error:
            update_run_manifest(
                self.root,
                run_id,
                status="failed",
                errors=[
                    {
                        "type": type(error).__name__,
                        "message": "Databento historical ingestion failed",
                    }
                ],
            )
            raise

        result = self._completed_result(request, "stream", skipped, estimate)
        update_run_manifest(
            self.root,
            run_id,
            status="completed",
            counts=result.record_counts,
            output_paths=result.output_paths,
        )
        return result

    def submit_batch(
        self,
        request: DatabentoHistoricalRequest,
        *,
        allow_billable_download: bool = False,
        max_cost_usd: Optional[float] = None,
    ) -> HistoricalArchiveResult:
        """Submit DBN/Zstd batch jobs, checkpointing each returned job ID."""

        self._ensure_request_identity(request)

        completed_schemas = tuple(
            schema
            for schema in request.schemas
            if self._valid_completion_marker(request, schema, "batch")
        )
        if len(completed_schemas) == len(request.schemas):
            result = self._completed_result(request, "batch", completed_schemas, None)
            self._reconcile_completed_manifest(request, result)
            return result
        jobs = self._load_batch_jobs(request)
        missing = tuple(schema for schema in request.schemas if schema not in jobs)
        estimate: Optional[HistoricalCostEstimate] = None
        if missing:
            estimate = self.preflight(request, missing)
            self._authorize(estimate, allow_billable_download, max_cost_usd)
        run_id = self._ensure_manifest(request, "batch")
        if estimate is not None:
            update_run_manifest(
                self.root,
                run_id,
                counts={"estimated_cost_usd": estimate.total_cost_usd},
            )
        for schema in missing:
            try:
                job = self.client.batch.submit_job(
                    dataset=request.dataset,
                    symbols=list(request.symbols),
                    schema=schema,
                    start=request.start,
                    end=request.end,
                    encoding="dbn",
                    compression="zstd",
                    split_duration="day",
                    delivery="download",
                    stype_in="parent",
                    stype_out="instrument_id",
                )
            except BaseException as error:
                safe_error = _safe_access_error(
                    "submit Databento batch schema {!r}".format(schema), error
                )
                update_run_manifest(
                    self.root,
                    run_id,
                    status="failed",
                    errors=[
                        {"type": type(safe_error).__name__, "message": str(safe_error)}
                    ],
                )
                raise safe_error from None
            job_id = job.get("id") if isinstance(job, Mapping) else None
            if not isinstance(job_id, str) or not job_id:
                raise DatabentoHistoricalAccessError(
                    "Databento batch submission returned no job ID."
                )
            jobs[schema] = {
                "job_id": job_id,
                "schema": schema,
                "state": str(job.get("state", "queued")),
                "cost_usd": job.get("cost_usd"),
            }
            self._write_batch_jobs(request, jobs)
        states = {
            schema: str(job.get("state", "queued")) for schema, job in jobs.items()
        }
        return HistoricalArchiveResult(
            request_id=request.request_id,
            transport="batch",
            completed=False,
            record_counts={},
            skipped_schemas=tuple(
                schema for schema in request.schemas if schema not in missing
            ),
            output_paths=(
                self._batch_jobs_path(request),
                self._request_identity_path(request),
            ),
            estimate=estimate,
            batch_states=states,
        )

    def ingest_submitted_batch(
        self, request: DatabentoHistoricalRequest
    ) -> HistoricalArchiveResult:
        """Resume authorized batch jobs and ingest them once every job is done."""

        self._ensure_request_identity(request)

        completed_schemas = tuple(
            schema
            for schema in request.schemas
            if self._valid_completion_marker(request, schema, "batch")
        )
        if len(completed_schemas) == len(request.schemas):
            result = self._completed_result(
                request,
                "batch",
                completed_schemas,
                None,
                {schema: "done" for schema in request.schemas},
            )
            self._reconcile_completed_manifest(request, result)
            return result
        jobs = self._load_batch_jobs(request)
        absent = [schema for schema in request.schemas if schema not in jobs]
        if absent:
            raise DatabentoHistoricalError(
                "Submit the missing batch schemas first: {}".format(", ".join(absent))
            )
        run_id = self._ensure_manifest(request, "batch")
        states: Dict[str, str] = {}
        for schema in request.schemas:
            job_id = str(jobs[schema]["job_id"])
            try:
                details = self.client.batch.get_job_details(job_id=job_id)
            except BaseException as error:
                raise _safe_access_error(
                    "check Databento batch job {!r}".format(job_id), error
                ) from None
            state = str(details.get("state", "unknown"))
            if state == "expired":
                raise DatabentoHistoricalAccessError(
                    "Databento batch job {!r} expired; submit a new authorized job.".format(
                        job_id
                    )
                )
            states[schema] = state
            jobs[schema].update(
                {
                    "state": state,
                    "cost_usd": details.get("cost_usd"),
                    "record_count": details.get("record_count"),
                }
            )
        self._write_batch_jobs(request, jobs)
        if any(state != "done" for state in states.values()):
            return HistoricalArchiveResult(
                request_id=request.request_id,
                transport="batch",
                completed=False,
                record_counts={},
                skipped_schemas=(),
                output_paths=(
                    self._batch_jobs_path(request),
                    self._request_identity_path(request),
                ),
                batch_states=states,
            )

        identities = self._load_identities(request)
        if not self._valid_completion_marker(request, "definition", "batch"):
            identities = {}
        for schema in request.schemas:
            if self._valid_completion_marker(request, schema, "batch"):
                continue
            native_paths = self._download_batch_schema(
                request, schema, str(jobs[schema]["job_id"])
            )
            stores = tuple(self._load_native(path) for path in native_paths)
            count, normalized_path = self._normalize_stores(
                request, schema, stores, identities
            )
            if schema == "definition":
                self._write_identities(request, identities)
            artifacts: List[Path] = [*native_paths, normalized_path]
            if schema == "definition":
                artifacts.append(self._identity_path(request))
            self._write_completion_marker(request, schema, "batch", artifacts, count)
            update_run_manifest(
                self.root,
                run_id,
                counts={"{}_records".format(schema): count},
                output_paths=artifacts,
            )
        result = self._completed_result(request, "batch", (), None, states)
        update_run_manifest(
            self.root,
            run_id,
            status="completed",
            counts=result.record_counts,
            output_paths=result.output_paths,
        )
        return result

    @staticmethod
    def _authorize(
        estimate: HistoricalCostEstimate,
        allowed: bool,
        max_cost_usd: Optional[float],
    ) -> None:
        if not allowed:
            raise BillableDownloadNotAuthorized(estimate)
        if max_cost_usd is not None:
            if max_cost_usd < 0:
                raise ValueError("max_cost_usd must not be negative")
            if estimate.total_cost_usd > max_cost_usd:
                raise BillableDownloadNotAuthorized(estimate)

    def _run_id(self, request: DatabentoHistoricalRequest, transport: str) -> str:
        return "databento-history-{}-{}".format(transport, request.request_id[:16])

    def _request_identity_path(self, request: DatabentoHistoricalRequest) -> Path:
        return (
            self.root
            / "checkpoints"
            / "databento_history"
            / "scope={}".format(request.resume_scope_id)
            / "request.json"
        )

    def _ensure_request_identity(self, request: DatabentoHistoricalRequest) -> Path:
        """Freeze schemas for a bounded scope before any provider request or resume."""

        path = self._request_identity_path(request)
        if not path.is_file():
            _atomic_json(
                path,
                {
                    "identity_version": 1,
                    "resume_scope_id": request.resume_scope_id,
                    "request_id": request.request_id,
                    "request": request.public_dict(),
                    "schemas": list(request.schemas),
                },
            )
        try:
            identity = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DatabentoHistoricalError(
                "Cannot verify the Databento bounded-request identity"
            ) from error
        if (
            identity.get("resume_scope_id") != request.resume_scope_id
            or identity.get("request_id") != request.request_id
            or identity.get("request") != request.public_dict()
            or identity.get("schemas") != list(request.schemas)
        ):
            raise DatabentoHistoricalError(
                "Databento bounded-request resume schema set does not match; rerun "
                "with exactly the original --schema selection"
            )
        return path

    def _ensure_manifest(
        self, request: DatabentoHistoricalRequest, transport: str
    ) -> str:
        run_id = self._run_id(request, transport)
        path = self.root / "manifests" / "run={}".format(run_id) / "manifest.json"
        config = request.public_dict()
        config["transport"] = transport
        config["provider_approval_id"] = self.provider_approval_id
        config["provider_approval_paths"] = [
            str(path) for path in self.provider_approval_paths
        ]
        config["historical_research_assumption_id"] = (
            self.historical_research_assumption_id
        )
        config["historical_research_assumption_paths"] = [
            str(path) for path in self.historical_research_assumption_paths
        ]
        if self.historical_research_assumption_id is not None:
            config["research_only"] = True
            config["no_live"] = True
            config["no_orders"] = True
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise DatabentoHistoricalError(
                    "Cannot verify the existing Databento run manifest"
                ) from error
            existing_config = existing.get("config")
            if not isinstance(existing_config, Mapping) or any(
                existing_config.get(key) != value for key, value in config.items()
            ):
                raise DatabentoHistoricalError(
                    "Databento run manifest authority/approval does not match its "
                    "request or schema set"
                )
            update_run_manifest(self.root, run_id, status="running")
        else:
            write_run_manifest(
                self.root,
                run_id,
                config,
                output_paths=(
                    *self.governance_paths,
                    self._request_identity_path(request),
                ),
            )
        return run_id

    def _reconcile_completed_manifest(
        self,
        request: DatabentoHistoricalRequest,
        result: HistoricalArchiveResult,
    ) -> None:
        """Make an interrupted final manifest converge to its completed markers."""

        run_id = self._ensure_manifest(request, result.transport)
        update_run_manifest(
            self.root,
            run_id,
            status="completed",
            counts=result.record_counts,
            output_paths=result.output_paths,
        )

    def _base_path(self, request: DatabentoHistoricalRequest, tier: str) -> Path:
        return (
            self.root
            / tier
            / "databento_history"
            / "date={}".format(_utc_date(request.start))
            / "metal={}".format(request.metal)
            / "request={}".format(request.request_id[:24])
        )

    def _native_path(self, request: DatabentoHistoricalRequest, schema: str) -> Path:
        return self._base_path(request, "raw") / "{}.dbn.zst".format(schema)

    def _normalized_path(
        self, request: DatabentoHistoricalRequest, schema: str
    ) -> Path:
        return self._base_path(request, "normalized") / "{}.jsonl".format(schema)

    def _metadata_path(
        self, request: DatabentoHistoricalRequest, schema: str, transport: str
    ) -> Path:
        return self._base_path(request, "raw") / "{}-{}-metadata.json".format(
            transport, schema
        )

    def _identity_path(self, request: DatabentoHistoricalRequest) -> Path:
        return self._base_path(request, "normalized") / "contract-identities.json"

    def _checkpoint_path(
        self, request: DatabentoHistoricalRequest, schema: str, transport: str
    ) -> Path:
        return (
            self.root
            / "checkpoints"
            / "databento_history"
            / "request={}".format(request.request_id)
            / "{}.{}.complete.json".format(schema, transport)
        )

    def _batch_jobs_path(self, request: DatabentoHistoricalRequest) -> Path:
        return (
            self.root
            / "checkpoints"
            / "databento_history"
            / "request={}".format(request.request_id)
            / "batch-jobs.json"
        )

    def _stream_one(
        self,
        request: DatabentoHistoricalRequest,
        schema: str,
        final_path: Path,
    ) -> Any:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = final_path.parent / ".{}-{}.dbn.zst".format(
            schema, uuid.uuid4().hex
        )
        try:
            try:
                store = self.client.timeseries.get_range(
                    dataset=request.dataset,
                    start=request.start,
                    end=request.end,
                    symbols=list(request.symbols),
                    schema=schema,
                    stype_in="parent",
                    stype_out="instrument_id",
                    path=temporary_path,
                )
            except BaseException as error:
                raise _safe_access_error(
                    "retrieve Databento schema {!r}".format(schema), error
                ) from None
            if not temporary_path.is_file():
                raise DatabentoHistoricalAccessError(
                    "Databento returned without writing the requested native DBN file."
                )
            with temporary_path.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(str(temporary_path), str(final_path))
            _fsync_directory(final_path.parent)
            return store
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _load_native(self, path: Path) -> Any:
        if self._dbn_loader is None:
            raise DatabentoHistoricalDependencyError(
                "A native DBN archive exists but no DBN loader was supplied; "
                "install 'databento' or inject dbn_loader=DBNStore.from_file."
            )
        try:
            return self._dbn_loader(path)
        except BaseException as error:
            raise DatabentoHistoricalError(
                "Unable to read archived native DBN {} ({}).".format(
                    path.name, type(error).__name__
                )
            ) from None

    def _write_store_metadata(
        self,
        store: Any,
        request: DatabentoHistoricalRequest,
        schema: str,
        path: Path,
    ) -> None:
        _atomic_json(
            path,
            {
                "request": request.public_dict(),
                "schema": schema,
                "metadata": _json_safe(getattr(store, "metadata", None)),
                "symbology": _json_safe(getattr(store, "symbology", None)),
            },
        )

    def _normalize_stores(
        self,
        request: DatabentoHistoricalRequest,
        schema: str,
        stores: Sequence[Any],
        identities: Dict[str, Dict[str, Any]],
    ) -> Tuple[int, Path]:
        final_path = self._normalized_path(request, schema)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = final_path.parent / ".{}-{}.jsonl".format(
            schema, uuid.uuid4().hex
        )
        count = 0
        writer = HashChainedJsonlWriter(temporary_path)

        def callback(record: Any) -> None:
            nonlocal count
            normalized = normalize_databento_record(
                record, dataset=request.dataset, schema=schema
            )
            normalized["metal"] = request.metal
            normalized["historical_request_id"] = request.request_id
            normalized["historical_request_schemas"] = list(request.schemas)
            normalized["requested_parent_symbols"] = list(request.symbols)
            normalized["stype_in"] = "parent"
            normalized["stype_out"] = "instrument_id"
            normalized["source_event_ts_ns"] = normalized.get("ts_event")
            ts_recv = normalized.get("ts_recv")
            ts_in_delta = normalized.get("ts_in_delta")
            normalized["source_publish_ts_ns"] = (
                ts_recv - ts_in_delta
                if isinstance(ts_recv, int) and isinstance(ts_in_delta, int)
                else None
            )
            normalized["vendor_timestamps_ns"] = {
                key: normalized.get(key)
                for key in ("ts_event", "ts_recv", "ts_in_delta", "ts_out", "ts_ref")
                if normalized.get(key) is not None
            }
            identity = self._record_identity(normalized, identities)
            normalized.update(identity)
            writer.write(normalized)
            count += 1

        try:
            for store in stores:
                replay = getattr(store, "replay", None)
                if not callable(replay):
                    raise DatabentoHistoricalDependencyError(
                        "DBNStore.replay(callback) is required to normalize native DBN."
                    )
                replay(callback)
            writer.close()
            os.replace(str(temporary_path), str(final_path))
            _fsync_directory(final_path.parent)
            return count, final_path
        except BaseException:
            writer.close()
            raise
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _record_identity(
        record: Mapping[str, Any], identities: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        instrument_id = record.get("instrument_id")
        timestamp = record.get("ts_recv", record.get("ts_event"))
        day: Optional[str] = None
        if isinstance(timestamp, int) and timestamp >= 0:
            day = _utc_date(timestamp)
        if record.get("schema") == "definition" and record.get("raw_symbol"):
            identity = {
                "instrument_id": instrument_id,
                "contract_symbol": record.get("raw_symbol"),
                "contract_expiration": record.get("expiration"),
                "instrument_class": record.get("instrument_class"),
                "asset": record.get("asset"),
                "underlying": record.get("underlying"),
            }
            if instrument_id is not None:
                identities["{}|{}".format(day or "unknown", instrument_id)] = identity
            return {**identity, "contract_identity_status": "definition"}
        if instrument_id is None:
            return {"contract_identity_status": "no_instrument_id"}
        exact = identities.get("{}|{}".format(day or "unknown", instrument_id))
        if exact is not None:
            return {**exact, "contract_identity_status": "resolved_daily_definition"}
        candidates = [
            value
            for key, value in identities.items()
            if key.endswith("|{}".format(instrument_id))
        ]
        unique_symbols = {value.get("contract_symbol") for value in candidates}
        if len(unique_symbols) == 1 and candidates:
            return {
                **candidates[-1],
                "contract_identity_status": "resolved_unique_definition",
            }
        return {
            "contract_symbol": None,
            "contract_identity_status": "unresolved_definition",
        }

    def _load_identities(
        self, request: DatabentoHistoricalRequest
    ) -> Dict[str, Dict[str, Any]]:
        path = self._identity_path(request)
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        identities = payload.get("identities", {})
        return dict(identities) if isinstance(identities, Mapping) else {}

    def _write_identities(
        self,
        request: DatabentoHistoricalRequest,
        identities: Mapping[str, Mapping[str, Any]],
    ) -> None:
        _atomic_json(
            self._identity_path(request),
            {
                "request_id": request.request_id,
                "identity_key": "UTC vendor date|instrument_id",
                "identities": identities,
            },
        )

    def _write_completion_marker(
        self,
        request: DatabentoHistoricalRequest,
        schema: str,
        transport: str,
        artifacts: Iterable[Path],
        record_count: int,
    ) -> None:
        values = []
        for artifact in artifacts:
            values.append(
                {
                    "path": str(artifact.relative_to(self.root)),
                    "bytes": artifact.stat().st_size,
                    "sha256": _sha256(artifact),
                }
            )
        _atomic_json(
            self._checkpoint_path(request, schema, transport),
            {
                "marker_version": 1,
                "request_id": request.request_id,
                "request": request.public_dict(),
                "schema": schema,
                "transport": transport,
                "record_count": record_count,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "artifacts": values,
            },
        )

    def _valid_completion_marker(
        self, request: DatabentoHistoricalRequest, schema: str, transport: str
    ) -> bool:
        path = self._checkpoint_path(request, schema, transport)
        if not path.is_file():
            return False
        try:
            with path.open("r", encoding="utf-8") as handle:
                marker = json.load(handle)
            if marker.get("request_id") != request.request_id:
                return False
            marker_request = marker.get("request")
            if marker_request is not None and marker_request != request.public_dict():
                raise DatabentoHistoricalError(
                    "Databento completion checkpoint request or schema set does not "
                    "match"
                )
            if marker.get("schema") != schema or marker.get("transport") != transport:
                return False
            for artifact in marker.get("artifacts", []):
                candidate = (self.root / str(artifact["path"])).resolve()
                candidate.relative_to(self.root.resolve())
                if not candidate.is_file() or _sha256(candidate) != artifact.get(
                    "sha256"
                ):
                    return False
            return bool(marker.get("artifacts"))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False

    def _completed_result(
        self,
        request: DatabentoHistoricalRequest,
        transport: str,
        skipped: Tuple[str, ...],
        estimate: Optional[HistoricalCostEstimate],
        states: Optional[Mapping[str, str]] = None,
    ) -> HistoricalArchiveResult:
        counts: Dict[str, int] = {}
        outputs: List[Path] = []
        for schema in request.schemas:
            marker_path = self._checkpoint_path(request, schema, transport)
            if not marker_path.is_file():
                continue
            with marker_path.open("r", encoding="utf-8") as handle:
                marker = json.load(handle)
            counts[schema] = int(marker.get("record_count", 0))
            outputs.extend(
                self.root / artifact["path"] for artifact in marker["artifacts"]
            )
            outputs.append(marker_path)
        return HistoricalArchiveResult(
            request_id=request.request_id,
            transport=transport,
            completed=len(counts) == len(request.schemas),
            record_counts=counts,
            skipped_schemas=skipped,
            output_paths=tuple(
                dict.fromkeys([self._request_identity_path(request), *outputs])
            ),
            estimate=estimate,
            batch_states=states,
        )

    def _load_batch_jobs(
        self, request: DatabentoHistoricalRequest
    ) -> Dict[str, Dict[str, Any]]:
        path = self._batch_jobs_path(request)
        if not path.is_file():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("request_id") != request.request_id:
            raise DatabentoHistoricalError("Batch checkpoint request ID mismatch.")
        if payload.get("request") != request.public_dict():
            raise DatabentoHistoricalError(
                "Batch checkpoint request or schema set does not match."
            )
        jobs = payload.get("jobs", {})
        selected = set(request.schemas)
        job_schemas = {str(key) for key in jobs}
        if not job_schemas.issubset(selected):
            raise DatabentoHistoricalError(
                "Batch checkpoint contains schemas outside the selected request."
            )
        return {str(key): dict(value) for key, value in jobs.items()}

    def _write_batch_jobs(
        self,
        request: DatabentoHistoricalRequest,
        jobs: Mapping[str, Mapping[str, Any]],
    ) -> None:
        # Only explicitly whitelisted fields are accepted, so vendor responses
        # containing an API-key name or other account metadata cannot leak.
        safe_jobs = {
            schema: {
                key: value
                for key, value in details.items()
                if key in {"job_id", "schema", "state", "cost_usd", "record_count"}
            }
            for schema, details in jobs.items()
        }
        _atomic_json(
            self._batch_jobs_path(request),
            {
                "request_id": request.request_id,
                "request": request.public_dict(),
                "jobs": safe_jobs,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _batch_native_dir(
        self, request: DatabentoHistoricalRequest, schema: str, job_id: str
    ) -> Path:
        return (
            self._base_path(request, "raw") / "batch" / schema / "job={}".format(job_id)
        )

    def _download_batch_schema(
        self,
        request: DatabentoHistoricalRequest,
        schema: str,
        job_id: str,
    ) -> Tuple[Path, ...]:
        final_dir = self._batch_native_dir(request, schema, job_id)
        final_dir.mkdir(parents=True, exist_ok=True)
        try:
            listed = self.client.batch.list_files(job_id=job_id)
        except BaseException as error:
            raise _safe_access_error(
                "list files for Databento batch job {!r}".format(job_id), error
            ) from None
        expected = {
            str(item.get("filename")): str(item.get("hash", "")).removeprefix("sha256:")
            for item in listed
            if isinstance(item, Mapping)
        }
        expected_native = sorted(name for name in expected if ".dbn" in name)
        existing = tuple(final_dir / name for name in expected_native)
        if expected_native and all(
            path.is_file()
            and (not expected[path.name] or _sha256(path) == expected[path.name])
            for path in existing
        ):
            return existing
        temporary_root = Path(
            tempfile.mkdtemp(prefix=".databento-batch-", dir=str(final_dir.parent))
        )
        try:
            try:
                downloaded = self.client.batch.download(
                    job_id=job_id, output_dir=temporary_root
                )
            except BaseException as error:
                raise _safe_access_error(
                    "download Databento batch job {!r}".format(job_id), error
                ) from None
            native_paths: List[Path] = []
            for downloaded_path in downloaded:
                source = Path(downloaded_path)
                if not source.is_file():
                    continue
                wanted_hash = expected.get(source.name)
                if wanted_hash and _sha256(source) != wanted_hash:
                    raise DatabentoHistoricalError(
                        "Databento batch SHA-256 mismatch for {}.".format(source.name)
                    )
                destination = final_dir / source.name
                os.replace(str(source), str(destination))
                if ".dbn" in destination.name:
                    native_paths.append(destination)
            _fsync_directory(final_dir)
            if not native_paths:
                raise DatabentoHistoricalError(
                    "Databento batch job contained no native DBN files."
                )
            final_names = {path.name for path in native_paths}
            if expected_native and final_names != set(expected_native):
                raise DatabentoHistoricalError(
                    "Databento batch download did not return every listed DBN file."
                )
            return tuple(sorted(native_paths))
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
