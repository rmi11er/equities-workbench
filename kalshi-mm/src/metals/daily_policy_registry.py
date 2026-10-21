"""Immutable materialization of reviewed daily gate policy and registries.

This module is deliberately a local persistence boundary.  It accepts JSON
objects which have already been reviewed; it does not fetch configuration,
contact an exchange, or place orders.

Registry ``config_hash`` values use one rule throughout this module:

* make a JSON object from the record;
* remove the top-level ``config_hash``, ``previous_record_sha256``, and
  ``record_sha256`` fields;
* encode it as UTF-8 JSON with sorted keys, compact separators, preserved
  Unicode, and no non-finite numbers; and
* take the lowercase SHA-256 hex digest of those bytes.

The chain fields are excluded so the same reviewed configuration has the same
identity before and after it is placed in a hash-chained JSONL segment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date as date_type
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .storage import update_run_manifest, write_run_manifest

MATERIALIZATION_VERSION = 1
POLICY_VERSION = 1
_HASH_EXCLUDED_FIELDS = frozenset(
    ("config_hash", "previous_record_sha256", "record_sha256")
)
_HASH_FIELDS = frozenset(("previous_record_sha256", "record_sha256"))
_SHA256_HEX = frozenset("0123456789abcdef")
_SOURCES = ("kalshi", "pyth", "cme")


class DailyPolicyRegistryError(RuntimeError):
    """Reviewed policy or registry input cannot be safely materialized."""


@dataclass(frozen=True)
class DailyPolicyRegistryArtifacts:
    """Paths belonging to one completed content-addressed materialization."""

    materialization_id: str
    policy_path: Path
    fee_schedule_path: Path
    incentive_program_path: Path
    manifest_path: Path


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalize_json(value: Any, field: str) -> Any:
    """Return a detached JSON value while rejecting Python-only extensions."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DailyPolicyRegistryError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [
            _normalize_json(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DailyPolicyRegistryError(f"{field} contains a non-string key")
            result[key] = _normalize_json(item, f"{field}.{key}")
        return result
    raise DailyPolicyRegistryError(f"{field} is not a JSON value")


def _json_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DailyPolicyRegistryError(f"{field} must be a JSON object")
    normalized = _normalize_json(value, field)
    assert isinstance(normalized, dict)
    return normalized


def _validated_date(value: Any, field: str = "date") -> str:
    if not isinstance(value, str):
        raise DailyPolicyRegistryError(f"{field} must use YYYY-MM-DD format")
    try:
        parsed = date_type.fromisoformat(value)
    except ValueError as error:
        raise DailyPolicyRegistryError(
            f"{field} must be an ISO calendar date"
        ) from error
    if parsed.isoformat() != value:
        raise DailyPolicyRegistryError(f"{field} must use YYYY-MM-DD format")
    return value


def _strict_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DailyPolicyRegistryError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise DailyPolicyRegistryError(f"{field} must be at least {minimum}")
    return value


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DailyPolicyRegistryError(f"{field} must be non-empty text")
    return value.strip()


def _digest(value: Any, field: str) -> str:
    text = _nonempty_text(value, field).lower()
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise DailyPolicyRegistryError(f"{field} must be a SHA-256 hex digest")
    return text


def _hash_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise DailyPolicyRegistryError(f"{field} must be a non-empty list")
    hashes = [_digest(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if len(set(hashes)) != len(hashes):
        raise DailyPolicyRegistryError(f"{field} contains duplicate hashes")
    return sorted(hashes)


def registry_config_hash(record: Mapping[str, Any]) -> str:
    """Return the canonical reviewed-configuration hash for one registry row.

    Only the three top-level identity/chain fields documented in the module
    docstring are removed.  Nested fields with the same names remain part of the
    reviewed configuration.
    """

    value = _json_object(record, "registry record")
    payload = {
        key: item for key, item in value.items() if key not in _HASH_EXCLUDED_FIELDS
    }
    return _sha256(_canonical_json(payload))


def _validate_policy(policy: Mapping[str, Any], requested_date: str) -> dict[str, Any]:
    value = _json_object(policy, "daily gate policy")
    if (
        isinstance(value.get("schema_version"), bool)
        or value.get("schema_version") != POLICY_VERSION
    ):
        raise DailyPolicyRegistryError("unsupported daily gate policy version")
    policy_date = _validated_date(value.get("date"), "daily gate policy date")
    if policy_date != requested_date:
        raise DailyPolicyRegistryError(
            "daily gate policy date does not match requested date"
        )

    start = _strict_int(
        value.get("measurement_start_ts_ns"),
        "measurement_start_ts_ns",
        minimum=0,
    )
    end = _strict_int(
        value.get("measurement_end_ts_ns"),
        "measurement_end_ts_ns",
        minimum=1,
    )
    if end <= start:
        raise DailyPolicyRegistryError("measurement interval must be positive")

    raw_windows = value.get("expected_windows")
    if not isinstance(raw_windows, list) or not raw_windows:
        raise DailyPolicyRegistryError("expected_windows must be a non-empty list")
    tickers: set[str] = set()
    for index, raw_window in enumerate(raw_windows):
        window = _json_object(raw_window, f"expected_windows[{index}]")
        prefix = f"expected_windows[{index}]"
        ticker = _nonempty_text(window.get("market_ticker"), prefix + ".market_ticker")
        if ticker in tickers:
            raise DailyPolicyRegistryError("expected market tickers must be unique")
        tickers.add(ticker)
        _nonempty_text(window.get("series_ticker"), prefix + ".series_ticker")
        _nonempty_text(window.get("metal"), prefix + ".metal")
        window_start = _strict_int(
            window.get("start_ts_ns"), prefix + ".start_ts_ns", minimum=0
        )
        window_end = _strict_int(
            window.get("end_ts_ns"), prefix + ".end_ts_ns", minimum=1
        )
        if window_start < start or window_end > end or window_end <= window_start:
            raise DailyPolicyRegistryError(
                f"{prefix} lies outside the positive measurement interval"
            )
        if window.get("fee_group_id") is not None:
            _nonempty_text(window.get("fee_group_id"), prefix + ".fee_group_id")

    staleness = value.get("source_staleness_ns")
    if not isinstance(staleness, Mapping):
        raise DailyPolicyRegistryError("source_staleness_ns must be an object")
    for source in _SOURCES:
        _strict_int(staleness.get(source), f"source_staleness_ns.{source}", minimum=1)
    _strict_int(value.get("edge_half_life_ns"), "edge_half_life_ns", minimum=1)
    _strict_int(
        value.get("clock_sample_staleness_ns"),
        "clock_sample_staleness_ns",
        minimum=1,
    )
    try:
        clock_fraction = Decimal(str(value.get("max_clock_error_fraction")))
    except (InvalidOperation, ValueError) as error:
        raise DailyPolicyRegistryError("max_clock_error_fraction is invalid") from error
    if not clock_fraction.is_finite() or not Decimal(0) < clock_fraction <= Decimal(1):
        raise DailyPolicyRegistryError(
            "max_clock_error_fraction must be finite and in (0, 1]"
        )

    metadata = value.get("approved_metadata_hashes")
    if not isinstance(metadata, Mapping):
        raise DailyPolicyRegistryError("approved_metadata_hashes must be an object")
    if set(metadata) != tickers:
        raise DailyPolicyRegistryError(
            "approved_metadata_hashes must exactly cover expected market tickers"
        )
    value["approved_metadata_hashes"] = {
        ticker: _hash_list(metadata[ticker], f"approved_metadata_hashes.{ticker}")
        for ticker in sorted(tickers)
    }
    value["approved_fee_registry_hashes"] = _hash_list(
        value.get("approved_fee_registry_hashes"),
        "approved_fee_registry_hashes",
    )
    value["approved_incentive_registry_hashes"] = _hash_list(
        value.get("approved_incentive_registry_hashes"),
        "approved_incentive_registry_hashes",
    )
    return value


def _interval(
    value: Mapping[str, Any], prefix: str, field: str
) -> tuple[int, int | None]:
    start = _strict_int(
        value.get(f"{prefix}_start_ts_ns"),
        f"{field}.{prefix}_start_ts_ns",
        minimum=0,
    )
    raw_end = value.get(f"{prefix}_end_ts_ns")
    if raw_end is None:
        end = None
    else:
        end = _strict_int(raw_end, f"{field}.{prefix}_end_ts_ns", minimum=1)
        if end <= start:
            raise DailyPolicyRegistryError(f"{field} has a non-positive interval")
    return start, end


def _validate_scope(value: Mapping[str, Any], stream: str, field: str) -> None:
    direct_fields = (
        (
            "series_ticker",
            "eligible_series_ticker",
            "fee_group_id",
            "standard_fee_group_identifier",
            "fee_group",
        )
        if stream == "fee_schedule_dim"
        else (
            "series_ticker",
            "eligible_series_ticker",
            "market_ticker",
            "eligible_market_ticker",
        )
    )
    has_scope = False
    for name in direct_fields:
        if value.get(name) is not None:
            _nonempty_text(value.get(name), f"{field}.{name}")
            has_scope = True
    list_fields = (
        ("series_tickers", "eligible_series")
        if stream == "fee_schedule_dim"
        else ("series_tickers", "eligible_series", "eligible_markets")
    )
    for name in list_fields:
        if value.get(name) is None:
            continue
        items = value.get(name)
        if not isinstance(items, list) or not items:
            raise DailyPolicyRegistryError(f"{field}.{name} must be a non-empty list")
        for index, item in enumerate(items):
            _nonempty_text(item, f"{field}.{name}[{index}]")
        if len(set(items)) != len(items):
            raise DailyPolicyRegistryError(f"{field}.{name} contains duplicates")
        has_scope = True
    if not has_scope:
        raise DailyPolicyRegistryError(f"{field} has no registry scope")


def _required_text(value: Mapping[str, Any], field: str, names: Sequence[str]) -> None:
    for name in names:
        _nonempty_text(value.get(name), f"{field}.{name}")


def _nonnegative_decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise DailyPolicyRegistryError(
            f"{field} must be an exact decimal string or integer"
        )
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise DailyPolicyRegistryError(f"{field} is not a decimal") from error
    if not parsed.is_finite() or parsed < 0:
        raise DailyPolicyRegistryError(f"{field} must be finite and non-negative")
    return parsed


def _validate_fee(value: Mapping[str, Any], field: str) -> None:
    _interval(value, "effective", field)
    _validate_scope(value, "fee_schedule_dim", field)
    if not isinstance(value.get("maker_fee_enabled"), bool):
        raise DailyPolicyRegistryError(f"{field}.maker_fee_enabled must be boolean")
    for name in ("taker_coefficient", "product_fee_multiplier"):
        _nonnegative_decimal(value.get(name), f"{field}.{name}")
    maker_fields = tuple(
        name
        for name in (
            "maker_coefficient_or_multiplier",
            "maker_coefficient",
            "maker_multiplier",
        )
        if value.get(name) is not None
    )
    if len(maker_fields) != 1:
        raise DailyPolicyRegistryError(
            f"{field} must contain exactly one maker fee parameter"
        )
    _nonnegative_decimal(value[maker_fields[0]], f"{field}.{maker_fields[0]}")
    _required_text(
        value,
        field,
        (
            "account_type_precision_rule",
            "source_document",
            "manual_approval_id",
        ),
    )
    _strict_int(value.get("retrieval_ts_ns"), f"{field}.retrieval_ts_ns", minimum=0)


def _validate_incentive(value: Mapping[str, Any], field: str) -> None:
    _interval(value, "program", field)
    _validate_scope(value, "incentive_program_dim", field)
    _required_text(
        value,
        field,
        (
            "program_id",
            "program_type",
            "eligible_participant_class",
            "volume_treatment",
            "source_document",
        ),
    )
    _strict_int(value.get("retrieval_ts_ns"), f"{field}.retrieval_ts_ns", minimum=0)
    if value.get("program_type") != "none":
        for name in (
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
        ):
            if value.get(name) is None:
                raise DailyPolicyRegistryError(f"{field}.{name} is required")


def _normalize_registry(
    records: Sequence[Mapping[str, Any]], stream: str, date: str
) -> tuple[dict[str, Any], ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise DailyPolicyRegistryError(f"{stream} must be a sequence of JSON objects")
    if not records:
        raise DailyPolicyRegistryError(f"{stream} must contain at least one record")
    normalized: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for index, raw_record in enumerate(records):
        field = f"{stream}[{index}]"
        value = _json_object(raw_record, field)
        for date_field in ("date", "utc_date"):
            if date_field in value:
                record_date = _validated_date(
                    value[date_field], f"{field}.{date_field}"
                )
                if record_date != date:
                    raise DailyPolicyRegistryError(
                        f"{field}.{date_field} does not match requested date"
                    )
        claimed = _digest(value.get("config_hash"), f"{field}.config_hash")
        calculated = registry_config_hash(value)
        if claimed != calculated:
            raise DailyPolicyRegistryError(f"{field}.config_hash mismatch")
        if claimed in seen_hashes:
            raise DailyPolicyRegistryError(
                f"{stream} contains duplicate config_hash values"
            )
        seen_hashes.add(claimed)
        for chain_field in _HASH_FIELDS:
            value.pop(chain_field, None)
        value["config_hash"] = claimed
        if stream == "fee_schedule_dim":
            _validate_fee(value, field)
        else:
            _validate_incentive(value, field)
        normalized.append(value)
    return tuple(
        sorted(
            normalized,
            key=lambda item: (item["config_hash"], _canonical_json(item)),
        )
    )


def _validate_registry_vintage(
    records: Sequence[Mapping[str, Any]],
    measurement_start_ts_ns: int,
    stream: str,
) -> None:
    for index, record in enumerate(records):
        retrieval = _strict_int(
            record.get("retrieval_ts_ns"),
            f"{stream}[{index}].retrieval_ts_ns",
            minimum=0,
        )
        if retrieval > measurement_start_ts_ns:
            raise DailyPolicyRegistryError(
                f"{stream}[{index}] was not retrieved by measurement start"
            )


def _encode_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _encode_hash_chained_jsonl(
    records: Sequence[Mapping[str, Any]],
) -> tuple[bytes, tuple[str, ...]]:
    previous_hash: str | None = None
    lines: list[bytes] = []
    record_hashes: list[str] = []
    for record in records:
        payload = dict(record)
        payload["previous_record_sha256"] = previous_hash
        record_hash = _sha256(_canonical_json(payload))
        payload["record_sha256"] = record_hash
        lines.append(
            json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        record_hashes.append(record_hash)
        previous_hash = record_hash
    return b"".join(lines), tuple(record_hashes)


def _materialization_id(
    date: str, policy_sha256: str, fee_sha256: str, incentive_sha256: str
) -> str:
    return _sha256(
        _canonical_json(
            {
                "materialization_version": MATERIALIZATION_VERSION,
                "date": date,
                "policy_sha256": policy_sha256,
                "fee_schedule_dim_sha256": fee_sha256,
                "incentive_program_dim_sha256": incentive_sha256,
            }
        )
    )


def _artifact_paths(
    root: Path,
    date: str,
    materialization_id: str,
    policy_sha256: str,
    fee_sha256: str,
    incentive_sha256: str,
) -> DailyPolicyRegistryArtifacts:
    run_id = f"daily-policy-registry-{materialization_id}"
    return DailyPolicyRegistryArtifacts(
        materialization_id=materialization_id,
        policy_path=(
            root
            / "policies"
            / "daily-gates"
            / f"date={date}"
            / f"policy={policy_sha256}"
            / "policy.json"
        ),
        fee_schedule_path=(
            root
            / "normalized"
            / "fee_schedule_dim"
            / f"date={date}"
            / f"run={run_id}"
            / f"segment={fee_sha256}"
            / "fee_schedule_dim.jsonl"
        ),
        incentive_program_path=(
            root
            / "normalized"
            / "incentive_program_dim"
            / f"date={date}"
            / f"run={run_id}"
            / f"segment={incentive_sha256}"
            / "incentive_program_dim.jsonl"
        ),
        manifest_path=root / "manifests" / f"run={run_id}" / "manifest.json",
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise DailyPolicyRegistryError("artifact path escapes storage root") from error


def _public_config(
    artifacts: DailyPolicyRegistryArtifacts,
    root: Path,
    date: str,
    policy_sha256: str,
    fee_sha256: str,
    incentive_sha256: str,
    fee_records: int,
    incentive_records: int,
) -> dict[str, Any]:
    return {
        "operation": "materialize_daily_policy_registries",
        "materialization_version": MATERIALIZATION_VERSION,
        "read_only_sources": True,
        "network_access": False,
        "live_orders": False,
        "date": date,
        "materialization_id": artifacts.materialization_id,
        "artifacts": {
            "daily_gate_policy": {
                "path": _relative(artifacts.policy_path, root),
                "sha256": policy_sha256,
                "records": 1,
            },
            "fee_schedule_dim": {
                "path": _relative(artifacts.fee_schedule_path, root),
                "sha256": fee_sha256,
                "records": fee_records,
            },
            "incentive_program_dim": {
                "path": _relative(artifacts.incentive_program_path, root),
                "sha256": incentive_sha256,
                "records": incentive_records,
            },
        },
    }


def _expected_counts(fee_records: int, incentive_records: int) -> dict[str, int]:
    return {
        "policies": 1,
        "fee_schedule_dim_records": fee_records,
        "incentive_program_dim_records": incentive_records,
        "registry_records": fee_records + incentive_records,
    }


def _atomic_create_bytes(root: Path, path: Path, encoded: bytes) -> None:
    """Create one immutable file or verify that an identical file won a race."""

    if path.is_symlink():
        raise DailyPolicyRegistryError("artifact path must not be a symbolic link")
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise DailyPolicyRegistryError("cannot verify existing artifact") from error
        if existing != encoded:
            raise DailyPolicyRegistryError(
                "existing artifact differs from its content address"
            )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    _relative(path.parent, root)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".daily-policy-registry-", suffix=".tmp"
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
            if path.is_symlink() or path.read_bytes() != encoded:
                raise DailyPolicyRegistryError(
                    "concurrent artifact differs from its content address"
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


def _load_json_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise DailyPolicyRegistryError(f"{field} is not valid JSON") from error
    return _json_object(value, field)


def _resolve_inside_root(root: Path, path: Path, field: str) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise DailyPolicyRegistryError(f"{field} escapes storage root") from error
    return resolved


def _read_hash_chained_jsonl(
    path: Path, stream: str, date: str
) -> tuple[tuple[dict[str, Any], ...], bytes]:
    try:
        encoded = path.read_bytes()
        text = encoded.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DailyPolicyRegistryError(
            f"{stream} artifact is not UTF-8 JSONL"
        ) from error
    if not text or not text.endswith("\n"):
        raise DailyPolicyRegistryError(f"{stream} artifact is incomplete")
    previous_hash: str | None = None
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise DailyPolicyRegistryError(f"{stream} contains a blank JSONL row")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise DailyPolicyRegistryError(f"{stream} contains invalid JSON") from error
        row = _json_object(raw, f"{stream} line {line_number}")
        if row.get("previous_record_sha256") != previous_hash:
            raise DailyPolicyRegistryError(f"{stream} hash chain is invalid")
        record_hash = _digest(
            row.get("record_sha256"), f"{stream} line {line_number}.record_sha256"
        )
        payload = dict(row)
        payload.pop("record_sha256", None)
        if record_hash != _sha256(_canonical_json(payload)):
            raise DailyPolicyRegistryError(f"{stream} record hash is invalid")
        previous_hash = record_hash
        rows.append(row)
    return _normalize_registry(rows, stream, date), encoded


def verify_daily_policy_registry_materialization(
    storage_root: Path, manifest_path: Path
) -> DailyPolicyRegistryArtifacts:
    """Verify a completed manifest, content addresses, registry hashes and chains."""

    root = Path(storage_root).resolve()
    requested_manifest = _resolve_inside_root(
        root, Path(manifest_path), "manifest path"
    )
    manifest = _load_json_object(requested_manifest, "daily policy registry manifest")
    if manifest.get("status") != "completed":
        raise DailyPolicyRegistryError(
            "daily policy registry manifest is not completed"
        )
    config = _json_object(manifest.get("config"), "manifest config")
    if config.get("operation") != "materialize_daily_policy_registries":
        raise DailyPolicyRegistryError(
            "manifest operation is not a policy materialization"
        )
    if config.get("read_only_sources") is not True:
        raise DailyPolicyRegistryError("manifest does not declare read-only sources")
    if (
        config.get("network_access") is not False
        or config.get("live_orders") is not False
    ):
        raise DailyPolicyRegistryError("manifest permits external side effects")
    date = _validated_date(config.get("date"), "manifest date")
    descriptors = config.get("artifacts")
    if not isinstance(descriptors, Mapping) or set(descriptors) != {
        "daily_gate_policy",
        "fee_schedule_dim",
        "incentive_program_dim",
    }:
        raise DailyPolicyRegistryError("manifest artifact descriptors are malformed")

    resolved: dict[str, Path] = {}
    observed_sha: dict[str, str] = {}
    for name in sorted(descriptors):
        descriptor = _json_object(descriptors[name], f"manifest artifacts.{name}")
        path = _resolve_inside_root(
            root, Path(_nonempty_text(descriptor.get("path"), f"{name}.path")), name
        )
        if not path.is_file() or path.is_symlink():
            raise DailyPolicyRegistryError(f"{name} artifact is missing or unsafe")
        digest = _sha256(path.read_bytes())
        if digest != _digest(descriptor.get("sha256"), f"{name}.sha256"):
            raise DailyPolicyRegistryError(f"{name} artifact checksum mismatch")
        resolved[name] = path
        observed_sha[name] = digest

    policy_raw = resolved["daily_gate_policy"].read_bytes()
    policy = _validate_policy(
        _load_json_object(resolved["daily_gate_policy"], "daily gate policy"), date
    )
    if _encode_json(policy) != policy_raw:
        raise DailyPolicyRegistryError("daily gate policy is not canonically encoded")
    fee, fee_raw = _read_hash_chained_jsonl(
        resolved["fee_schedule_dim"], "fee_schedule_dim", date
    )
    incentive, incentive_raw = _read_hash_chained_jsonl(
        resolved["incentive_program_dim"], "incentive_program_dim", date
    )
    measurement_start = _strict_int(
        policy.get("measurement_start_ts_ns"),
        "measurement_start_ts_ns",
        minimum=0,
    )
    _validate_registry_vintage(fee, measurement_start, "fee_schedule_dim")
    _validate_registry_vintage(incentive, measurement_start, "incentive_program_dim")
    if _encode_hash_chained_jsonl(fee)[0] != fee_raw:
        raise DailyPolicyRegistryError("fee_schedule_dim is not canonically encoded")
    if _encode_hash_chained_jsonl(incentive)[0] != incentive_raw:
        raise DailyPolicyRegistryError(
            "incentive_program_dim is not canonically encoded"
        )

    fee_hashes = sorted(record["config_hash"] for record in fee)
    incentive_hashes = sorted(record["config_hash"] for record in incentive)
    if policy["approved_fee_registry_hashes"] != fee_hashes:
        raise DailyPolicyRegistryError(
            "approved_fee_registry_hashes do not match fee_schedule_dim"
        )
    if policy["approved_incentive_registry_hashes"] != incentive_hashes:
        raise DailyPolicyRegistryError(
            "approved_incentive_registry_hashes do not match incentive_program_dim"
        )

    materialization_id = _materialization_id(
        date,
        observed_sha["daily_gate_policy"],
        observed_sha["fee_schedule_dim"],
        observed_sha["incentive_program_dim"],
    )
    artifacts = _artifact_paths(
        root,
        date,
        materialization_id,
        observed_sha["daily_gate_policy"],
        observed_sha["fee_schedule_dim"],
        observed_sha["incentive_program_dim"],
    )
    if requested_manifest != artifacts.manifest_path.resolve():
        raise DailyPolicyRegistryError("manifest path does not match content address")
    if resolved != {
        "daily_gate_policy": artifacts.policy_path,
        "fee_schedule_dim": artifacts.fee_schedule_path,
        "incentive_program_dim": artifacts.incentive_program_path,
    }:
        raise DailyPolicyRegistryError("artifact paths do not match content addresses")

    expected_config = _public_config(
        artifacts,
        root,
        date,
        observed_sha["daily_gate_policy"],
        observed_sha["fee_schedule_dim"],
        observed_sha["incentive_program_dim"],
        len(fee),
        len(incentive),
    )
    if config != expected_config:
        raise DailyPolicyRegistryError("manifest config does not match materialization")
    expected_run_id = f"daily-policy-registry-{materialization_id}"
    if manifest.get("run_id") != expected_run_id:
        raise DailyPolicyRegistryError("manifest run_id does not match content address")
    if manifest.get("config_sha256") != _sha256(_canonical_json(config)):
        raise DailyPolicyRegistryError("manifest config hash mismatch")
    if manifest.get("counts") != _expected_counts(len(fee), len(incentive)):
        raise DailyPolicyRegistryError("manifest counts do not match artifacts")
    expected_outputs = [
        str(artifacts.policy_path),
        str(artifacts.fee_schedule_path),
        str(artifacts.incentive_program_path),
    ]
    if manifest.get("output_paths") != expected_outputs:
        raise DailyPolicyRegistryError("manifest output paths do not match artifacts")
    return artifacts


def materialize_daily_policy_registries(
    storage_root: Path,
    date: str,
    policy: Mapping[str, Any],
    fee_schedule_dim: Sequence[Mapping[str, Any]],
    incentive_program_dim: Sequence[Mapping[str, Any]],
) -> DailyPolicyRegistryArtifacts:
    """Validate and immutably persist one reviewed UTC day's gate inputs.

    Registry input order is not significant.  An identical retry returns the
    existing verified paths.  In particular, a completed manifest is never
    passed to the mutable manifest update function.
    """

    requested_date = _validated_date(date)
    root = Path(storage_root).resolve()
    normalized_policy = _validate_policy(policy, requested_date)
    fee = _normalize_registry(fee_schedule_dim, "fee_schedule_dim", requested_date)
    incentive = _normalize_registry(
        incentive_program_dim, "incentive_program_dim", requested_date
    )
    measurement_start = _strict_int(
        normalized_policy.get("measurement_start_ts_ns"),
        "measurement_start_ts_ns",
        minimum=0,
    )
    _validate_registry_vintage(fee, measurement_start, "fee_schedule_dim")
    _validate_registry_vintage(incentive, measurement_start, "incentive_program_dim")

    fee_hashes = sorted(record["config_hash"] for record in fee)
    incentive_hashes = sorted(record["config_hash"] for record in incentive)
    if normalized_policy["approved_fee_registry_hashes"] != fee_hashes:
        raise DailyPolicyRegistryError(
            "approved_fee_registry_hashes do not match supplied fee_schedule_dim"
        )
    if normalized_policy["approved_incentive_registry_hashes"] != incentive_hashes:
        raise DailyPolicyRegistryError(
            "approved_incentive_registry_hashes do not match supplied "
            "incentive_program_dim"
        )

    policy_bytes = _encode_json(normalized_policy)
    fee_bytes, _ = _encode_hash_chained_jsonl(fee)
    incentive_bytes, _ = _encode_hash_chained_jsonl(incentive)
    policy_sha256 = _sha256(policy_bytes)
    fee_sha256 = _sha256(fee_bytes)
    incentive_sha256 = _sha256(incentive_bytes)
    materialization_id = _materialization_id(
        requested_date, policy_sha256, fee_sha256, incentive_sha256
    )
    artifacts = _artifact_paths(
        root,
        requested_date,
        materialization_id,
        policy_sha256,
        fee_sha256,
        incentive_sha256,
    )
    public_config = _public_config(
        artifacts,
        root,
        requested_date,
        policy_sha256,
        fee_sha256,
        incentive_sha256,
        len(fee),
        len(incentive),
    )
    counts = _expected_counts(len(fee), len(incentive))

    if artifacts.manifest_path.exists():
        existing = _load_json_object(
            artifacts.manifest_path, "daily policy registry manifest"
        )
        if existing.get("status") == "completed":
            return verify_daily_policy_registry_materialization(
                root, artifacts.manifest_path
            )
        if (
            existing.get("status") != "running"
            or existing.get("run_id") != f"daily-policy-registry-{materialization_id}"
            or existing.get("config") != public_config
        ):
            raise DailyPolicyRegistryError(
                "existing manifest conflicts with materialization content address"
            )

    _atomic_create_bytes(root, artifacts.policy_path, policy_bytes)
    _atomic_create_bytes(root, artifacts.fee_schedule_path, fee_bytes)
    _atomic_create_bytes(root, artifacts.incentive_program_path, incentive_bytes)

    if not artifacts.manifest_path.exists():
        try:
            write_run_manifest(
                root,
                f"daily-policy-registry-{materialization_id}",
                public_config,
            )
        except FileExistsError:
            existing = _load_json_object(
                artifacts.manifest_path, "daily policy registry manifest"
            )
            if existing.get("status") == "completed":
                return verify_daily_policy_registry_materialization(
                    root, artifacts.manifest_path
                )
            if (
                existing.get("status") != "running"
                or existing.get("config") != public_config
            ):
                raise DailyPolicyRegistryError(
                    "concurrent manifest conflicts with materialization content address"
                )

    update_run_manifest(
        root,
        f"daily-policy-registry-{materialization_id}",
        status="completed",
        counts=counts,
        output_paths=(
            artifacts.policy_path,
            artifacts.fee_schedule_path,
            artifacts.incentive_program_path,
        ),
    )
    return verify_daily_policy_registry_materialization(root, artifacts.manifest_path)
