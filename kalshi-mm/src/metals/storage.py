"""Append-only, hash-chained storage for immutable source records."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, TextIO


def utc_date_from_ns(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, timezone.utc).date().isoformat()


class HashChainedJsonlWriter:
    """Write a new segment exclusively and chain each serialized record by SHA-256."""

    def __init__(self, path: Path, fsync_every_records: int = 1) -> None:
        if fsync_every_records < 1:
            raise ValueError("fsync_every_records must be at least one")
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = self.path.open("x", encoding="utf-8")
        self._fsync_every_records = fsync_every_records
        self._records_since_fsync = 0
        self._previous_hash: Optional[str] = None
        self.record_count = 0

    def write(self, record: Dict[str, Any]) -> str:
        payload = dict(record)
        payload["previous_record_sha256"] = self._previous_hash
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        record_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        payload["record_sha256"] = record_hash
        self._handle.write(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str) + "\n"
        )
        self._handle.flush()
        self._records_since_fsync += 1
        self.record_count += 1
        if self._records_since_fsync >= self._fsync_every_records:
            os.fsync(self._handle.fileno())
            self._records_since_fsync = 0
        self._previous_hash = record_hash
        return record_hash

    def close(self) -> None:
        if not self._handle.closed:
            try:
                self.sync()
            finally:
                self._handle.close()

    def sync(self) -> None:
        """Flush and fsync the current append-only segment."""

        if not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._records_since_fsync = 0

    def __enter__(self) -> "HashChainedJsonlWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class PartitionedRawStore:
    """Create one immutable segment per source/stream/date/metal/run."""

    def __init__(
        self,
        root: Path,
        run_id: Optional[str] = None,
        fsync_every_records: int = 1,
        tier: str = "raw",
    ) -> None:
        self.root = Path(root)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self._fsync_every_records = fsync_every_records
        if tier not in ("raw", "normalized", "derived"):
            raise ValueError("Unsupported storage tier: {}".format(tier))
        self.tier = tier
        self._writers: Dict[str, HashChainedJsonlWriter] = {}

    def _writer_key(self, source: str, stream: str, metal: str, date: str) -> str:
        return "|".join((source, stream, metal, date))

    def _path(self, source: str, stream: str, metal: str, date: str) -> Path:
        safe_values = (source, stream, metal, date, self.run_id)
        if any("/" in value or ".." in value for value in safe_values):
            raise ValueError("Partition values must not contain path separators")
        filename = "{}-{}.jsonl".format(stream, uuid.uuid4().hex[:12])
        return (
            self.root
            / self.tier
            / source
            / "date={}".format(date)
            / "metal={}".format(metal)
            / "run={}".format(self.run_id)
            / filename
        )

    def write(
        self,
        source: str,
        stream: str,
        metal: str,
        receive_ts_ns: int,
        record: Dict[str, Any],
    ) -> str:
        date = utc_date_from_ns(receive_ts_ns)
        key = self._writer_key(source, stream, metal, date)
        writer = self._writers.get(key)
        if writer is None:
            writer = HashChainedJsonlWriter(
                self._path(source, stream, metal, date),
                fsync_every_records=self._fsync_every_records,
            )
            self._writers[key] = writer
        return writer.write(record)

    @property
    def paths(self):
        return [writer.path for writer in self._writers.values()]

    def close(self) -> None:
        first_error: Optional[BaseException] = None
        for writer in self._writers.values():
            try:
                writer.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def sync(self) -> None:
        """Durably flush every currently open segment."""

        first_error: Optional[BaseException] = None
        for writer in self._writers.values():
            try:
                writer.sync()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "PartitionedRawStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _git_state() -> Dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


_MANIFEST_STATUSES = frozenset(("running", "completed", "failed", "cancelled"))
_SECRET_KEY = re.compile(
    r"(?:^|_)(?:api_?key|authorization|credentials?|password|private_?key|secret|token)"
    r"(?:$|_)",
    re.IGNORECASE,
)
_SECRET_TEXT = (
    re.compile(
        r"(?i)(\b(?:authorization|api[-_ ]?key|password|secret|token)\s*[:=]\s*)"
        r"(?:bearer\s+)?([^\s,;&]+)"
    ),
    re.compile(
        r"(?i)([?&](?:api[-_]?key|password|secret|token)=)([^&#\s]+)"
    ),
)


def _manifest_path(root: Path, run_id: str) -> Path:
    if "/" in run_id or ".." in run_id:
        raise ValueError("run_id must not contain path separators")
    return Path(root) / "manifests" / "run={}".format(run_id) / "manifest.json"


def _redact_text(value: str) -> str:
    redacted = value
    for pattern in _SECRET_TEXT:
        redacted = pattern.sub(r"\1[REDACTED]", redacted)
    return redacted


def _redact_secrets(value: Any, key: str = "") -> Any:
    """Return a JSON-safe copy with credential-shaped fields removed."""

    normalized_key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    normalized_key = normalized_key.lower().replace("-", "_")
    # Environment variable names and secret-file locations are configuration,
    # not credentials. Keeping them makes a run reproducible without exposing
    # the value stored at that location.
    if (
        normalized_key
        and not normalized_key.endswith(("_env", "_path"))
        and _SECRET_KEY.search(normalized_key)
    ):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {
            str(child_key): _redact_secrets(child_value, str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_redact_secrets(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": _redact_text(str(value)),
        }
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(str(value))


def _config_sha256(public_config: Any) -> str:
    canonical = json.dumps(
        public_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: Dict[str, Any], *, create: bool) -> None:
    """Durably publish JSON without exposing a partially-written manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".manifest-", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if create:
            # A hard link is an atomic create-if-absent operation. Preserve the
            # former write_run_manifest FileExistsError contract on collisions.
            os.link(str(temporary_path), str(path))
            temporary_path.unlink()
        else:
            os.replace(str(temporary_path), str(path))
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some filesystems do not support directory fsync. The file itself
            # is still fully written and atomically published.
            pass
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_run_manifest(
    root: Path,
    run_id: str,
    public_config: Dict[str, Any],
    *,
    counts: Optional[Mapping[str, Any]] = None,
    output_paths: Optional[Iterable[Path]] = None,
) -> Path:
    """Atomically create a running manifest.

    The original three-argument API and create-only behavior are retained.
    Call :func:`update_run_manifest` as work progresses or completes.
    """

    path = _manifest_path(root, run_id)
    now = datetime.now(timezone.utc).isoformat()
    sanitized_config = _redact_secrets(public_config)
    manifest = {
        "manifest_version": 2,
        "run_id": run_id,
        "created_ts": now,
        "started_ts": now,
        "updated_ts": now,
        "completed_ts": None,
        "status": "running",
        "git": _git_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": sanitized_config,
        "config_sha256": _config_sha256(sanitized_config),
        "counts": _redact_secrets(dict(counts or {})),
        "output_paths": _redact_secrets(
            [str(output_path) for output_path in (output_paths or ())]
        ),
        "errors": [],
    }
    _atomic_write_json(path, manifest, create=True)
    return path


def update_run_manifest(
    root: Path,
    run_id: str,
    *,
    status: Optional[str] = None,
    counts: Optional[Mapping[str, Any]] = None,
    output_paths: Optional[Iterable[Path]] = None,
    errors: Optional[Iterable[Any]] = None,
) -> Path:
    """Atomically merge progress or terminal state into an existing manifest."""

    if status is not None and status not in _MANIFEST_STATUSES:
        raise ValueError("Unsupported manifest status: {}".format(status))
    path = _manifest_path(root, run_id)
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest = _redact_secrets(manifest)

    now = datetime.now(timezone.utc).isoformat()
    manifest.setdefault("manifest_version", 1)
    manifest.setdefault("started_ts", manifest.get("created_ts"))
    manifest.setdefault("completed_ts", None)
    manifest.setdefault("status", "running")
    manifest["config_sha256"] = _config_sha256(manifest.get("config", {}))
    manifest.setdefault("counts", {})
    manifest.setdefault("output_paths", [])
    manifest.setdefault("errors", [])
    if counts is not None:
        manifest["counts"].update(_redact_secrets(dict(counts)))
    if output_paths is not None:
        combined_paths = [
            *(str(value) for value in manifest["output_paths"]),
            *(str(value) for value in output_paths),
        ]
        manifest["output_paths"] = list(dict.fromkeys(combined_paths))
    if errors is not None:
        manifest["errors"].extend(_redact_secrets(list(errors)))
    if status is not None:
        manifest["status"] = status
        manifest["completed_ts"] = (
            now if status in ("completed", "failed", "cancelled") else None
        )
    manifest["updated_ts"] = now
    # Also sanitize update payloads before they are persisted.
    manifest = _redact_secrets(manifest)
    _atomic_write_json(path, manifest, create=False)
    return path
