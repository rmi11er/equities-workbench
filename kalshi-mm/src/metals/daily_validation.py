"""Immutable daily validation snapshots for research data admission."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .daily_gate_measurement import DailyGateMeasurementError, measure_daily_gates
from .quality import evaluate_data_gates


class DailyValidationError(RuntimeError):
    """Raised when a daily snapshot cannot be frozen safely."""


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_measured_quality_provenance(
    root: Path,
    date: str,
    quality_path: Path,
    quality: Mapping[str, Any],
    manifest_records: Sequence[tuple[Path, Mapping[str, Any], Sequence[Path]]],
) -> None:
    claimed_quality_hash = quality.get("daily_quality_sha256")
    quality_payload = dict(quality)
    quality_payload.pop("daily_quality_sha256", None)
    if not isinstance(
        claimed_quality_hash, str
    ) or claimed_quality_hash != _canonical_sha256(quality_payload):
        raise DailyValidationError(
            "quality report is not a verified daily-gate measurement"
        )
    evidence = quality.get("daily_gate_measurement")
    if not isinstance(evidence, Mapping):
        raise DailyValidationError(
            "quality report is missing daily-gate measurement evidence"
        )
    claimed_evidence_hash = evidence.get("evidence_sha256")
    evidence_payload = dict(evidence)
    evidence_payload.pop("evidence_sha256", None)
    if (
        evidence.get("evidence_version") != 1
        or not isinstance(claimed_evidence_hash, str)
        or claimed_evidence_hash != _canonical_sha256(evidence_payload)
    ):
        raise DailyValidationError("daily-gate measurement evidence is invalid")

    owners = []
    for manifest_path, manifest, outputs in manifest_records:
        config = manifest.get("config")
        if not isinstance(config, Mapping):
            continue
        if (
            config.get("operation") == "measure_daily_gates"
            and config.get("date") == date
            and config.get("daily_quality_sha256") == claimed_quality_hash
            and manifest.get("run_id") == f"daily-gates-{claimed_quality_hash}"
            and quality_path in outputs
        ):
            owners.append((manifest_path, manifest, outputs, config))
    if len(owners) != 1:
        raise DailyValidationError(
            "quality report must have exactly one completed daily-gate manifest owner"
        )
    _, _, owner_outputs, owner_config = owners[0]

    evidence_path = (
        root
        / "derived"
        / "daily_gate_evidence"
        / f"date={date}"
        / f"evidence={claimed_evidence_hash}"
        / "evidence.json"
    ).resolve()
    if evidence_path not in owner_outputs:
        raise DailyValidationError(
            "daily-gate evidence is not listed by its measurement manifest"
        )
    external_evidence = _load_json(evidence_path)
    if external_evidence != dict(evidence):
        raise DailyValidationError(
            "daily-gate report evidence does not match its immutable artifact"
        )

    supplied_manifests = {path.resolve() for path, _, _ in manifest_records}
    source_manifests = owner_config.get("source_manifests")
    if not isinstance(source_manifests, list) or not source_manifests:
        raise DailyValidationError(
            "daily-gate manifest is missing source-manifest provenance"
        )
    resolved_sources = {
        _resolve_input_path(Path(str(path)), root) for path in source_manifests
    }
    if not resolved_sources.issubset(supplied_manifests):
        raise DailyValidationError(
            "all daily-gate source manifests must be supplied to the snapshot"
        )
    policy = evidence.get("policy")
    policy_path = policy.get("path") if isinstance(policy, Mapping) else None
    try:
        recomputed = measure_daily_gates(
            root,
            date,
            sorted(resolved_sources, key=str),
            _resolve_input_path(Path(str(policy_path)), root),
        )
    except (DailyGateMeasurementError, ValueError) as error:
        raise DailyValidationError(
            "daily-gate evidence cannot be reproduced from its frozen sources"
        ) from error
    if recomputed != dict(evidence):
        raise DailyValidationError(
            "daily-gate evidence does not match recomputed source evidence"
        )
    source_quality = owner_config.get("source_quality_report")
    source_quality_path = _resolve_input_path(Path(str(source_quality)), root)
    source_outputs = {
        output
        for manifest_path, _, outputs in manifest_records
        if manifest_path.resolve() in resolved_sources
        for output in outputs
    }
    if source_quality_path not in source_outputs:
        raise DailyValidationError(
            "source quality report is not listed by a supplied source manifest"
        )


def _validate_date(value: str) -> str:
    try:
        parsed = date_type.fromisoformat(value)
    except ValueError as error:
        raise ValueError("date must be an ISO calendar date") from error
    if parsed.isoformat() != value:
        raise ValueError("date must use YYYY-MM-DD format")
    return value


def _inside_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_input_path(path: Path, storage_root: Path) -> Path:
    """Resolve a manifest path without allowing it to escape storage_root."""

    root = storage_root.resolve()
    candidate = path.expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        # Current manifests commonly store paths such as ``data/metals/raw/...``.
        # Resolve those from the process cwd first; paths relative to the storage
        # root remain supported for portable test and archival manifests.
        cwd_candidate = (Path.cwd() / candidate).resolve()
        root_candidate = (root / candidate).resolve()
        resolved = (
            cwd_candidate if _inside_root(cwd_candidate, root) else root_candidate
        )
    if not _inside_root(resolved, root):
        raise DailyValidationError(
            "manifest output escapes storage root: {}".format(path)
        )
    if not resolved.is_file():
        raise DailyValidationError(
            "manifest output is missing or not a file: {}".format(path)
        )
    return resolved


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise DailyValidationError("cannot read JSON input: {}".format(path)) from error
    if not isinstance(value, dict):
        raise DailyValidationError("JSON input must contain an object: {}".format(path))
    return value


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".daily-validation-", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(str(temporary_path), str(path))
        temporary_path.unlink()
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def freeze_daily_validation(
    storage_root: Path,
    date: str,
    quality_report_path: Path,
    manifest_paths: Sequence[Path],
) -> Path:
    """Freeze one deterministic daily snapshot and all of its input checksums.

    A failed data gate is still frozen, but ``research_eligible`` remains false and
    the failed gates are retained.  Only completed run manifests are admitted.
    """

    date = _validate_date(date)
    root = Path(storage_root).resolve()
    if not manifest_paths:
        raise DailyValidationError("at least one completed run manifest is required")

    quality_path = _resolve_input_path(Path(quality_report_path), root)
    quality = _load_json(quality_path)
    if quality.get("date") != date:
        raise DailyValidationError(
            "quality report date {!r} does not match {!r}".format(
                quality.get("date"), date
            )
        )
    manifests: List[Dict[str, Any]] = []
    manifest_records: List[tuple[Path, Mapping[str, Any], Sequence[Path]]] = []
    input_paths = {quality_path}
    for requested_path in manifest_paths:
        manifest_path = _resolve_input_path(Path(requested_path), root)
        manifest = _load_json(manifest_path)
        if manifest.get("status") != "completed":
            raise DailyValidationError(
                "run {} is not completed".format(manifest.get("run_id", "unknown"))
            )
        run_id = str(manifest.get("run_id", "")).strip()
        if not run_id:
            raise DailyValidationError("run manifest is missing run_id")
        config = manifest.get("config")
        if not isinstance(config, Mapping) or manifest.get(
            "config_sha256"
        ) != _canonical_sha256(config):
            raise DailyValidationError("run manifest config checksum is invalid")
        raw_outputs = manifest.get("output_paths")
        if not isinstance(raw_outputs, list):
            raise DailyValidationError("run manifest output_paths must be a list")
        resolved_outputs = []
        resolved_output_paths = []
        for output in raw_outputs:
            resolved = _resolve_input_path(Path(str(output)), root)
            input_paths.add(resolved)
            resolved_outputs.append(str(resolved.relative_to(root)))
            resolved_output_paths.append(resolved)
        input_paths.add(manifest_path)
        manifest_records.append((manifest_path, manifest, tuple(resolved_output_paths)))
        manifests.append(
            {
                "run_id": run_id,
                "config_sha256": manifest.get("config_sha256"),
                "git": manifest.get("git"),
                "outputs": sorted(resolved_outputs),
            }
        )

    _verify_measured_quality_provenance(
        root,
        date,
        quality_path,
        quality,
        manifest_records,
    )
    gates = evaluate_data_gates(quality)

    checksums = [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(input_paths)
    ]
    immutable = {
        "snapshot_version": 2,
        "date": date,
        "research_eligible": bool(gates["passed"]),
        "failed_gates": list(gates["failed_gates"]),
        "data_gates": gates,
        "quality_summary": {
            "expected_windows": int(quality.get("expected_windows", 0)),
            "observed_windows": int(quality.get("observed_windows", 0)),
            "settlement_checksum": {
                "checked": int(
                    quality.get("settlement_checksum", {}).get("checked", 0)
                ),
                "agreements": int(
                    quality.get("settlement_checksum", {}).get("agreements", 0)
                ),
                "disagreements": int(
                    quality.get("settlement_checksum", {}).get("disagreements", 0)
                ),
            },
        },
        "quality_run_id": quality.get("run_id"),
        "manifests": sorted(manifests, key=lambda item: item["run_id"]),
        "inputs": checksums,
    }
    snapshot_sha256 = _canonical_sha256(immutable)
    snapshot = {
        **immutable,
        "snapshot_sha256": snapshot_sha256,
        "frozen_ts": datetime.now(timezone.utc).isoformat(),
    }
    destination = (
        root
        / "derived"
        / "daily_validation"
        / "date={}".format(date)
        / "snapshot={}".format(snapshot_sha256)
        / "snapshot.json"
    )
    if destination.exists():
        existing = _load_json(destination)
        existing_immutable = dict(existing)
        existing_immutable.pop("snapshot_sha256", None)
        existing_immutable.pop("frozen_ts", None)
        if _canonical_sha256(existing_immutable) != snapshot_sha256:
            raise DailyValidationError(
                "existing daily snapshot does not match its content hash"
            )
        return destination
    _atomic_create_json(destination, snapshot)
    return destination


def validated_research_dates(storage_root: Path) -> List[str]:
    """Return unique dates with at least one immutable passing snapshot."""

    root = Path(storage_root)
    dates = set()
    pattern = "derived/daily_validation/date=*/snapshot=*/snapshot.json"
    for path in root.glob(pattern):
        snapshot = verify_daily_snapshot(root, path)
        if snapshot.get("research_eligible") is True:
            dates.add(str(snapshot.get("date")))
    return sorted(dates)


def verify_daily_snapshot(storage_root: Path, snapshot_path: Path) -> Dict[str, Any]:
    """Verify the content address and every frozen input checksum."""

    root = Path(storage_root).resolve()
    path = _resolve_input_path(Path(snapshot_path), root)
    snapshot = _load_json(path)
    expected = str(snapshot.get("snapshot_sha256", ""))
    immutable = dict(snapshot)
    immutable.pop("snapshot_sha256", None)
    immutable.pop("frozen_ts", None)
    observed = _canonical_sha256(immutable)
    if not expected or observed != expected:
        raise DailyValidationError(
            "daily snapshot content hash mismatch: {}".format(path)
        )
    expected_directory = "snapshot={}".format(expected)
    if path.parent.name != expected_directory:
        raise DailyValidationError(
            "daily snapshot directory does not match content hash: {}".format(path)
        )
    for item in snapshot.get("inputs", []):
        input_path = _resolve_input_path(Path(str(item.get("path", ""))), root)
        size = input_path.stat().st_size
        digest = _sha256_file(input_path)
        if size != int(item.get("size_bytes", -1)) or digest != item.get("sha256"):
            raise DailyValidationError(
                "frozen input checksum mismatch: {}".format(input_path)
            )
    return snapshot


def _selected_passing_snapshots(storage_root: Path) -> Dict[str, Dict[str, Any]]:
    """Select at most one verified passing snapshot per UTC date.

    Multiple immutable reruns for one date must never multiply the settlement count.
    Prefer the candidate with the most checked settlements, then the lexicographically
    smallest content hash so selection is deterministic.
    """

    root = Path(storage_root)
    candidates: Dict[str, List[Dict[str, Any]]] = {}
    pattern = "derived/daily_validation/date=*/snapshot=*/snapshot.json"
    for path in root.glob(pattern):
        snapshot = verify_daily_snapshot(root, path)
        if snapshot.get("research_eligible") is not True:
            continue
        if snapshot.get("snapshot_version") != 2:
            continue
        summary = snapshot.get("quality_summary")
        checksum = (
            summary.get("settlement_checksum") if isinstance(summary, dict) else None
        )
        if not isinstance(checksum, dict):
            raise DailyValidationError(
                "passing v2 daily snapshot lacks settlement evidence"
            )
        for field in ("checked", "agreements", "disagreements"):
            value = checksum.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DailyValidationError(
                    "passing v2 daily snapshot has invalid settlement evidence"
                )
        if checksum["agreements"] + checksum["disagreements"] != checksum["checked"]:
            raise DailyValidationError(
                "passing v2 daily snapshot has inconsistent settlement evidence"
            )
        date = str(snapshot.get("date", ""))
        candidates.setdefault(date, []).append(snapshot)

    selected = {}
    for date, values in candidates.items():
        selected[date] = sorted(
            values,
            key=lambda item: (
                -int(
                    item.get("quality_summary", {})
                    .get("settlement_checksum", {})
                    .get("checked", -1)
                ),
                str(item.get("snapshot_sha256", "")),
            ),
        )[0]
    return selected


def synchronized_day_progress(
    storage_root: Path,
    minimum_complete_days: int = 10,
    minimum_settlements: int = 1000,
) -> Dict[str, Any]:
    """Report the independent synchronized-day and settlement-audit gates."""

    if minimum_complete_days < 1:
        raise ValueError("minimum_complete_days must be at least one")
    if minimum_settlements < 1:
        raise ValueError("minimum_settlements must be at least one")

    selected = _selected_passing_snapshots(storage_root)
    dates = sorted(selected)
    settlement_evidence_dates = []
    settlements_checked = 0
    settlement_agreements = 0
    settlement_disagreements = 0
    expected_windows = 0
    for date in dates:
        summary = selected[date].get("quality_summary")
        if not isinstance(summary, dict):
            continue
        checksum = summary.get("settlement_checksum")
        if not isinstance(checksum, dict):
            continue
        settlement_evidence_dates.append(date)
        settlements_checked += int(checksum.get("checked", 0))
        settlement_agreements += int(checksum.get("agreements", 0))
        settlement_disagreements += int(checksum.get("disagreements", 0))
        expected_windows += int(summary.get("expected_windows", 0))

    day_gate_passed = len(dates) >= minimum_complete_days
    settlement_gate_passed = (
        len(settlement_evidence_dates) == len(dates)
        and settlements_checked >= minimum_settlements
        and settlement_agreements == settlements_checked
        and settlement_disagreements == 0
    )
    return {
        "passed": day_gate_passed and settlement_gate_passed,
        "day_gate_passed": day_gate_passed,
        "complete_days": len(dates),
        "minimum_complete_days": minimum_complete_days,
        "dates": dates,
        "settlement_gate_passed": settlement_gate_passed,
        "settlement_evidence_days": len(settlement_evidence_dates),
        "minimum_settlements": minimum_settlements,
        "settlements_checked": settlements_checked,
        "settlement_agreements": settlement_agreements,
        "settlement_disagreements": settlement_disagreements,
        "expected_windows": expected_windows,
    }
