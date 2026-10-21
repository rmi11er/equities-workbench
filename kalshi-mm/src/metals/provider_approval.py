"""Immutable provider-license, entitlement, and timestamp approval gates.

Credentials prove authentication, not permission. This module freezes reviewed,
secret-free evidence hashes and forces every approved provider operation to have
separate license, timestamp-semantic, and entitlement confirmations before collection.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

APPROVAL_SCHEMA_VERSION = "metals-provider-approval/v1"
PROVIDER_OPERATIONS = {
    "pyth": ("exact_history", "live_lazer"),
    "databento": ("historical", "live"),
}
_HEX_DIGITS = frozenset("0123456789abcdef")


class ProviderApprovalError(RuntimeError):
    """Provider governance evidence is absent, invalid, expired, or insufficient."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields do not match the frozen schema")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value.lower())
    )


def _parse_utc(value: str, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{label} must be UTC")
    return parsed


@dataclass(frozen=True)
class EvidenceConfirmation:
    version: str
    confirmed: bool
    evidence_sha256: str | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], label: str) -> EvidenceConfirmation:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be an object")
        _exact_keys(
            value,
            {"version", "confirmed", "evidence_sha256"},
            label,
        )
        version = value.get("version")
        confirmed = value.get("confirmed")
        evidence = value.get("evidence_sha256")
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"{label}.version must be non-empty")
        if not isinstance(confirmed, bool):
            raise ValueError(f"{label}.confirmed must be boolean")
        if evidence is not None and not _is_sha256(evidence):
            raise ValueError(f"{label}.evidence_sha256 is invalid")
        if confirmed and evidence is None:
            raise ValueError(f"{label} confirmation requires an evidence SHA-256")
        if confirmed and version.strip().lower() == "pending":
            raise ValueError(f"{label}.version cannot be pending when confirmed")
        return cls(version=version, confirmed=confirmed, evidence_sha256=evidence)


@dataclass(frozen=True)
class OperationApproval:
    approved: bool
    entitlement: EvidenceConfirmation

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], label: str) -> OperationApproval:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be an object")
        _exact_keys(value, {"approved", "entitlement"}, label)
        approved = value.get("approved")
        if not isinstance(approved, bool):
            raise ValueError(f"{label}.approved must be boolean")
        return cls(
            approved=approved,
            entitlement=EvidenceConfirmation.from_mapping(
                value.get("entitlement"), f"{label}.entitlement"
            ),
        )


@dataclass(frozen=True)
class ProviderApproval:
    license: EvidenceConfirmation
    timestamp_semantics: EvidenceConfirmation
    operations: dict[str, OperationApproval]

    @classmethod
    def from_mapping(cls, provider: str, value: Mapping[str, Any]) -> ProviderApproval:
        if not isinstance(value, Mapping):
            raise ValueError(f"providers.{provider} must be an object")
        _exact_keys(
            value,
            {"license", "timestamp_semantics", "operations"},
            f"providers.{provider}",
        )
        operations = value.get("operations")
        if not isinstance(operations, Mapping):
            raise ValueError(f"providers.{provider}.operations must be an object")
        expected = set(PROVIDER_OPERATIONS[provider])
        _exact_keys(operations, expected, f"providers.{provider}.operations")
        return cls(
            license=EvidenceConfirmation.from_mapping(
                value.get("license"), f"providers.{provider}.license"
            ),
            timestamp_semantics=EvidenceConfirmation.from_mapping(
                value.get("timestamp_semantics"),
                f"providers.{provider}.timestamp_semantics",
            ),
            operations={
                operation: OperationApproval.from_mapping(
                    operations[operation],
                    f"providers.{provider}.operations.{operation}",
                )
                for operation in PROVIDER_OPERATIONS[provider]
            },
        )


@dataclass(frozen=True)
class ProviderApprovalPolicy:
    review_version: str
    reviewed_at: str
    valid_until: str | None
    providers: dict[str, ProviderApproval]
    schema_version: str = APPROVAL_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProviderApprovalPolicy:
        if not isinstance(value, Mapping):
            raise ValueError("provider approval must be an object")
        _exact_keys(
            value,
            {
                "schema_version",
                "review_version",
                "reviewed_at",
                "valid_until",
                "providers",
            },
            "provider approval",
        )
        if value.get("schema_version") != APPROVAL_SCHEMA_VERSION:
            raise ValueError("unsupported provider approval schema")
        review_version = value.get("review_version")
        if not isinstance(review_version, str) or not review_version.strip():
            raise ValueError("review_version must be non-empty")
        reviewed_at = value.get("reviewed_at")
        reviewed = _parse_utc(reviewed_at, "reviewed_at")
        valid_until = value.get("valid_until")
        if valid_until is not None:
            expiry = _parse_utc(valid_until, "valid_until")
            if expiry <= reviewed:
                raise ValueError("valid_until must be later than reviewed_at")
        providers = value.get("providers")
        if not isinstance(providers, Mapping):
            raise ValueError("providers must be an object")
        _exact_keys(providers, set(PROVIDER_OPERATIONS), "providers")
        return cls(
            review_version=review_version,
            reviewed_at=reviewed_at,
            valid_until=valid_until,
            providers={
                provider: ProviderApproval.from_mapping(provider, providers[provider])
                for provider in PROVIDER_OPERATIONS
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderApprovalArtifact:
    approval_id: str
    directory: Path
    approval_path: Path
    manifest_path: Path
    policy: ProviderApprovalPolicy

    def public_dict(self, *, as_of: datetime | None = None) -> dict[str, Any]:
        now = datetime.now(UTC) if as_of is None else as_of
        if now.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        reviewed = _parse_utc(self.policy.reviewed_at, "reviewed_at")
        not_yet_valid = now < reviewed
        expired = self.policy.valid_until is not None and now >= _parse_utc(
            self.policy.valid_until, "valid_until"
        )
        active = not not_yet_valid and not expired
        return {
            "approval_id": self.approval_id,
            "approval_path": str(self.approval_path),
            "manifest_path": str(self.manifest_path),
            "review_version": self.policy.review_version,
            "reviewed_at": self.policy.reviewed_at,
            "valid_until": self.policy.valid_until,
            "not_yet_valid": not_yet_valid,
            "expired": expired,
            "operations": {
                provider: {
                    operation: active and _operation_passes(provider_policy, operation)
                    for operation in PROVIDER_OPERATIONS[provider]
                }
                for provider, provider_policy in self.policy.providers.items()
            },
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
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
        raise ProviderApprovalError(f"Cannot read {label}") from error
    if not isinstance(value, dict):
        raise ProviderApprovalError(f"{label} must be a JSON object")
    return value


def _resolve_inside(root: Path, path: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ProviderApprovalError(f"{label} escapes the storage root") from error
    return resolved


def materialize_provider_approval(
    storage_root: Path, value: Mapping[str, Any]
) -> ProviderApprovalArtifact:
    """Freeze a reviewed, secret-free approval record without provider access."""

    policy = ProviderApprovalPolicy.from_mapping(value)
    canonical = policy.to_dict()
    approval_id = _sha256(_canonical_bytes(canonical))
    root = Path(storage_root).resolve()
    destination = root / "policies" / "provider_approval" / f"approval={approval_id}"
    if destination.exists():
        return verify_provider_approval(root, destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".provider-approval-", dir=destination.parent)
    )
    try:
        approval_path = temporary / "approval.json"
        _write_json(approval_path, {"approval_id": approval_id, **canonical})
        manifest = {
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "approval_id": approval_id,
            "outputs": [
                {
                    "path": "approval.json",
                    "size_bytes": approval_path.stat().st_size,
                    "sha256": _sha256_path(approval_path),
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
    return verify_provider_approval(root, destination)


def verify_provider_approval(
    storage_root: Path, approval_path: Path
) -> ProviderApprovalArtifact:
    """Verify content address, schema, manifest, and immutable approval bytes."""

    root = Path(storage_root).resolve()
    requested = _resolve_inside(root, approval_path, "Provider approval")
    directory = (
        requested.parent
        if requested.name in {"approval.json", "manifest.json"}
        else requested
    )
    relative = directory.relative_to(root)
    if len(relative.parts) < 3 or relative.parts[-2] != "provider_approval":
        raise ProviderApprovalError("Provider approval location is invalid")
    if not directory.name.startswith("approval="):
        raise ProviderApprovalError("Provider approval directory is invalid")
    approval_file = directory / "approval.json"
    manifest_file = directory / "manifest.json"
    approval = _read_object(approval_file, "provider approval")
    manifest = _read_object(manifest_file, "provider approval manifest")
    approval_id = approval.get("approval_id")
    if not _is_sha256(approval_id) or directory.name != f"approval={approval_id}":
        raise ProviderApprovalError("Provider approval identity is invalid")
    policy_value = dict(approval)
    policy_value.pop("approval_id", None)
    try:
        policy = ProviderApprovalPolicy.from_mapping(policy_value)
    except ValueError as error:
        raise ProviderApprovalError("Provider approval schema is invalid") from error
    if _sha256(_canonical_bytes(policy.to_dict())) != approval_id:
        raise ProviderApprovalError("Provider approval content hash is invalid")
    outputs = manifest.get("outputs")
    if (
        set(manifest) != {"schema_version", "approval_id", "outputs"}
        or manifest.get("schema_version") != APPROVAL_SCHEMA_VERSION
        or manifest.get("approval_id") != approval_id
        or not isinstance(outputs, list)
        or len(outputs) != 1
        or not isinstance(outputs[0], Mapping)
        or set(outputs[0]) != {"path", "size_bytes", "sha256"}
        or outputs[0].get("path") != "approval.json"
        or outputs[0].get("size_bytes") != approval_file.stat().st_size
        or outputs[0].get("sha256") != _sha256_path(approval_file)
    ):
        raise ProviderApprovalError("Provider approval manifest is invalid")
    return ProviderApprovalArtifact(
        approval_id=approval_id,
        directory=directory,
        approval_path=approval_file,
        manifest_path=manifest_file,
        policy=policy,
    )


def _operation_passes(policy: ProviderApproval, operation: str) -> bool:
    approval = policy.operations[operation]
    return bool(
        policy.license.confirmed
        and policy.timestamp_semantics.confirmed
        and approval.approved
        and approval.entitlement.confirmed
    )


def require_provider_operation(
    storage_root: Path,
    approval_path: Path | str,
    provider: str,
    operation: str,
    *,
    as_of: datetime | None = None,
) -> ProviderApprovalArtifact:
    """Fail closed unless one frozen provider operation is currently approved."""

    if provider not in PROVIDER_OPERATIONS:
        raise ValueError(f"Unsupported provider: {provider}")
    if operation not in PROVIDER_OPERATIONS[provider]:
        raise ValueError(f"Unsupported {provider} operation: {operation}")
    if not isinstance(approval_path, (str, Path)) or not str(approval_path).strip():
        raise ProviderApprovalError(
            "A content-addressed provider approval path is required"
        )
    artifact = verify_provider_approval(storage_root, Path(approval_path))
    now = datetime.now(UTC) if as_of is None else as_of
    if now.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    if now < _parse_utc(artifact.policy.reviewed_at, "reviewed_at"):
        raise ProviderApprovalError("Provider approval is not yet valid")
    if artifact.policy.valid_until is not None and now >= _parse_utc(
        artifact.policy.valid_until, "valid_until"
    ):
        raise ProviderApprovalError("Provider approval has expired")
    policy = artifact.policy.providers[provider]
    if not policy.license.confirmed:
        raise ProviderApprovalError(f"{provider} license evidence is not confirmed")
    if not policy.timestamp_semantics.confirmed:
        raise ProviderApprovalError(f"{provider} timestamp semantics are not confirmed")
    approval = policy.operations[operation]
    if not approval.approved:
        raise ProviderApprovalError(f"{provider} operation {operation} is not approved")
    if not approval.entitlement.confirmed:
        raise ProviderApprovalError(
            f"{provider} operation {operation} entitlement is not confirmed"
        )
    return artifact


def require_config_provider_operation(
    config: Any, provider: str, operation: str
) -> ProviderApprovalArtifact:
    """Resolve and verify the frozen approval configured for one pipeline action."""

    return require_provider_operation(
        Path(config.storage.root),
        config.governance.provider_approval_path,
        provider,
        operation,
    )
