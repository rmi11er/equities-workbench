"""Immutable, research-only assumptions for historical provider exploration.

This is deliberately narrower than :mod:`provider_approval`.  It can authorize
only the two historical download operations listed below, cannot authorize live
access or orders, and is not evidence for production or daily-readiness gates.
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
from urllib.parse import urlsplit

from .provider_approval import (
    ProviderApprovalArtifact,
    require_provider_operation,
)

ASSUMPTION_SCHEMA_VERSION = "metals-historical-research-assumption/v1"
HISTORICAL_OPERATIONS = {
    "pyth": "exact_history",
    "databento": "historical",
}
PREFLIGHT_STATUSES = frozenset({"passed", "failed", "not_checked"})
_HEX_DIGITS = frozenset("0123456789abcdef")


class HistoricalResearchAssumptionError(RuntimeError):
    """A research-only historical assumption is absent, invalid, or inactive."""


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


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{label} must be UTC")
    return parsed


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{label} cannot contain NUL")
    return value.strip()


def _reviewed_version(value: Any, label: str) -> str:
    version = _nonempty_string(value, label)
    if version.lower() in {"pending", "todo", "unreviewed"}:
        raise ValueError(f"{label} must identify reviewed semantics or acceptance")
    return version


@dataclass(frozen=True)
class PrimarySourceEvidence:
    url: str
    sha256: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], label: str
    ) -> PrimarySourceEvidence:
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be an object")
        _exact_keys(value, {"url", "sha256"}, label)
        url = _nonempty_string(value.get("url"), f"{label}.url")
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(f"{label}.url must be a credential-free HTTPS URL")
        digest = value.get("sha256")
        if not _is_sha256(digest):
            raise ValueError(f"{label}.sha256 must be a SHA-256")
        return cls(url=url, sha256=digest.lower())


@dataclass(frozen=True)
class AccountPreflightEvidence:
    status: str
    observed_at: str
    evidence_sha256: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], label: str
    ) -> AccountPreflightEvidence:
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be an object")
        _exact_keys(value, {"status", "observed_at", "evidence_sha256"}, label)
        status = value.get("status")
        if status not in PREFLIGHT_STATUSES:
            raise ValueError(
                f"{label}.status must be one of {sorted(PREFLIGHT_STATUSES)}"
            )
        observed_at = value.get("observed_at")
        _parse_utc(observed_at, f"{label}.observed_at")
        evidence_sha256 = value.get("evidence_sha256")
        if not _is_sha256(evidence_sha256):
            raise ValueError(f"{label}.evidence_sha256 must be a SHA-256")
        return cls(
            status=status,
            observed_at=observed_at,
            evidence_sha256=evidence_sha256.lower(),
        )


@dataclass(frozen=True)
class OperatorAcceptance:
    accepted_by: str
    version: str
    accepted_at: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], label: str) -> OperatorAcceptance:
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be an object")
        _exact_keys(value, {"accepted_by", "version", "accepted_at"}, label)
        accepted_at = value.get("accepted_at")
        _parse_utc(accepted_at, f"{label}.accepted_at")
        return cls(
            accepted_by=_nonempty_string(
                value.get("accepted_by"), f"{label}.accepted_by"
            ),
            version=_reviewed_version(value.get("version"), f"{label}.version"),
            accepted_at=accepted_at,
        )


@dataclass(frozen=True)
class HistoricalResearchAssumptionPolicy:
    assumption_version: str
    provider: str
    operation: str
    primary_source_evidence: tuple[PrimarySourceEvidence, ...]
    technical_semantics_version: str
    account_preflight: AccountPreflightEvidence
    research_only: bool
    no_live: bool
    no_orders: bool
    operator_acceptance: OperatorAcceptance
    valid_until: str
    known_limitations: tuple[str, ...]
    schema_version: str = ASSUMPTION_SCHEMA_VERSION

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> HistoricalResearchAssumptionPolicy:
        if not isinstance(value, Mapping):
            raise TypeError("historical research assumption must be an object")
        _exact_keys(
            value,
            {
                "schema_version",
                "assumption_version",
                "provider",
                "operation",
                "primary_source_evidence",
                "technical_semantics_version",
                "account_preflight",
                "research_only",
                "no_live",
                "no_orders",
                "operator_acceptance",
                "valid_until",
                "known_limitations",
            },
            "historical research assumption",
        )
        if value.get("schema_version") != ASSUMPTION_SCHEMA_VERSION:
            raise ValueError("unsupported historical research assumption schema")
        provider = value.get("provider")
        if provider not in HISTORICAL_OPERATIONS:
            raise ValueError("provider must be pyth or databento")
        operation = value.get("operation")
        if operation != HISTORICAL_OPERATIONS[provider]:
            raise ValueError(
                "research assumptions cannot authorize "
                f"{provider} operation {operation}"
            )
        sources = value.get("primary_source_evidence")
        if not isinstance(sources, list) or not sources:
            raise ValueError("primary_source_evidence must be a non-empty array")
        parsed_sources = tuple(
            PrimarySourceEvidence.from_mapping(source, f"primary_source_evidence[{i}]")
            for i, source in enumerate(sources)
        )
        if len({source.url for source in parsed_sources}) != len(parsed_sources):
            raise ValueError("primary_source_evidence URLs must be unique")
        if len({source.sha256 for source in parsed_sources}) != len(parsed_sources):
            raise ValueError("primary_source_evidence hashes must be unique")

        for flag in ("research_only", "no_live", "no_orders"):
            if value.get(flag) is not True:
                raise ValueError(f"{flag} must be exactly true")

        account_preflight = AccountPreflightEvidence.from_mapping(
            value.get("account_preflight"), "account_preflight"
        )
        acceptance = OperatorAcceptance.from_mapping(
            value.get("operator_acceptance"), "operator_acceptance"
        )
        observed_at = _parse_utc(account_preflight.observed_at, "observed_at")
        accepted_at = _parse_utc(acceptance.accepted_at, "accepted_at")
        if observed_at > accepted_at:
            raise ValueError(
                "account preflight cannot be later than operator acceptance"
            )
        valid_until = value.get("valid_until")
        expiry = _parse_utc(valid_until, "valid_until")
        if expiry <= accepted_at:
            raise ValueError("valid_until must be later than operator acceptance")

        limitations = value.get("known_limitations")
        if not isinstance(limitations, list) or not limitations:
            raise ValueError("known_limitations must be a non-empty array")
        normalized_limitations = tuple(
            _nonempty_string(item, f"known_limitations[{i}]")
            for i, item in enumerate(limitations)
        )
        if len(set(normalized_limitations)) != len(normalized_limitations):
            raise ValueError("known_limitations must be unique")

        return cls(
            assumption_version=_reviewed_version(
                value.get("assumption_version"), "assumption_version"
            ),
            provider=provider,
            operation=operation,
            primary_source_evidence=parsed_sources,
            technical_semantics_version=_reviewed_version(
                value.get("technical_semantics_version"),
                "technical_semantics_version",
            ),
            account_preflight=account_preflight,
            research_only=True,
            no_live=True,
            no_orders=True,
            operator_acceptance=acceptance,
            valid_until=valid_until,
            known_limitations=normalized_limitations,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HistoricalResearchAssumptionArtifact:
    assumption_id: str
    directory: Path
    assumption_path: Path
    manifest_path: Path
    policy: HistoricalResearchAssumptionPolicy

    def public_dict(self, *, as_of: datetime | None = None) -> dict[str, Any]:
        now = datetime.now(UTC) if as_of is None else as_of
        if now.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        accepted = _parse_utc(
            self.policy.operator_acceptance.accepted_at, "accepted_at"
        )
        expired = now >= _parse_utc(self.policy.valid_until, "valid_until")
        active = (
            now >= accepted
            and not expired
            and self.policy.account_preflight.status == "passed"
        )
        return {
            "assumption_id": self.assumption_id,
            "assumption_path": str(self.assumption_path),
            "manifest_path": str(self.manifest_path),
            "provider": self.policy.provider,
            "operation": self.policy.operation,
            "research_only": True,
            "no_live": True,
            "no_orders": True,
            "account_preflight_status": self.policy.account_preflight.status,
            "not_yet_valid": now < accepted,
            "expired": expired,
            "active": active,
        }


@dataclass(frozen=True)
class HistoricalOperationAuthority:
    """One verified authority for a historical operation only."""

    kind: str
    authority_id: str
    output_paths: tuple[Path, Path]
    provider_approval: ProviderApprovalArtifact | None = None
    research_assumption: HistoricalResearchAssumptionArtifact | None = None

    def manifest_fields(self) -> dict[str, Any]:
        if self.kind == "provider_approval":
            return {"provider_approval_id": self.authority_id}
        return {
            "historical_research_assumption_id": self.authority_id,
            "research_only": True,
            "no_live": True,
            "no_orders": True,
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
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
        raise HistoricalResearchAssumptionError(f"Cannot read {label}") from error
    if not isinstance(value, dict):
        raise HistoricalResearchAssumptionError(f"{label} must be a JSON object")
    return value


def _resolve_inside(root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise HistoricalResearchAssumptionError(
            f"{label} escapes the storage root"
        ) from error
    return resolved


def materialize_historical_research_assumption(
    storage_root: Path, value: Mapping[str, Any]
) -> HistoricalResearchAssumptionArtifact:
    """Freeze a secret-free operator assumption without provider access."""

    policy = HistoricalResearchAssumptionPolicy.from_mapping(value)
    canonical = policy.to_dict()
    assumption_id = _sha256(_canonical_bytes(canonical))
    root = Path(storage_root).resolve()
    destination = (
        root
        / "policies"
        / "historical_research_assumption"
        / f"assumption={assumption_id}"
    )
    if destination.exists():
        return verify_historical_research_assumption(root, destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".historical-assumption-", dir=destination.parent)
    )
    try:
        assumption_path = temporary / "assumption.json"
        _write_json(assumption_path, {"assumption_id": assumption_id, **canonical})
        _write_json(
            temporary / "manifest.json",
            {
                "schema_version": ASSUMPTION_SCHEMA_VERSION,
                "assumption_id": assumption_id,
                "outputs": [
                    {
                        "path": "assumption.json",
                        "size_bytes": assumption_path.stat().st_size,
                        "sha256": _sha256_path(assumption_path),
                    }
                ],
            },
        )
        try:
            os.rename(temporary, destination)
        except FileExistsError:
            shutil.rmtree(temporary)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verify_historical_research_assumption(root, destination)


def verify_historical_research_assumption(
    storage_root: Path, assumption_path: Path
) -> HistoricalResearchAssumptionArtifact:
    """Verify the schema, content address, location, and immutable manifest."""

    root = Path(storage_root).resolve()
    requested = _resolve_inside(root, Path(assumption_path), "Research assumption")
    directory = (
        requested.parent
        if requested.name in {"assumption.json", "manifest.json"}
        else requested
    )
    relative = directory.relative_to(root)
    if (
        len(relative.parts) < 3
        or relative.parts[-2] != "historical_research_assumption"
        or not directory.name.startswith("assumption=")
    ):
        raise HistoricalResearchAssumptionError(
            "Historical research assumption location is invalid"
        )
    assumption_file = directory / "assumption.json"
    manifest_file = directory / "manifest.json"
    assumption = _read_object(assumption_file, "historical research assumption")
    manifest = _read_object(manifest_file, "historical research assumption manifest")
    assumption_id = assumption.get("assumption_id")
    if not _is_sha256(assumption_id) or directory.name != f"assumption={assumption_id}":
        raise HistoricalResearchAssumptionError(
            "Historical research assumption identity is invalid"
        )
    policy_value = dict(assumption)
    policy_value.pop("assumption_id", None)
    try:
        policy = HistoricalResearchAssumptionPolicy.from_mapping(policy_value)
    except (TypeError, ValueError) as error:
        raise HistoricalResearchAssumptionError(
            "Historical research assumption schema is invalid"
        ) from error
    if _sha256(_canonical_bytes(policy.to_dict())) != assumption_id:
        raise HistoricalResearchAssumptionError(
            "Historical research assumption content hash is invalid"
        )
    outputs = manifest.get("outputs")
    if (
        set(manifest) != {"schema_version", "assumption_id", "outputs"}
        or manifest.get("schema_version") != ASSUMPTION_SCHEMA_VERSION
        or manifest.get("assumption_id") != assumption_id
        or not isinstance(outputs, list)
        or len(outputs) != 1
        or not isinstance(outputs[0], Mapping)
        or set(outputs[0]) != {"path", "size_bytes", "sha256"}
        or outputs[0].get("path") != "assumption.json"
        or outputs[0].get("size_bytes") != assumption_file.stat().st_size
        or outputs[0].get("sha256") != _sha256_path(assumption_file)
    ):
        raise HistoricalResearchAssumptionError(
            "Historical research assumption manifest is invalid"
        )
    return HistoricalResearchAssumptionArtifact(
        assumption_id=assumption_id,
        directory=directory,
        assumption_path=assumption_file,
        manifest_path=manifest_file,
        policy=policy,
    )


def require_historical_research_assumption(
    storage_root: Path,
    assumption_path: Path | str,
    provider: str,
    operation: str,
    *,
    as_of: datetime | None = None,
) -> HistoricalResearchAssumptionArtifact:
    """Fail closed unless an active assumption exactly matches one historical action."""

    if provider not in HISTORICAL_OPERATIONS:
        raise ValueError(f"Unsupported historical provider: {provider}")
    if operation != HISTORICAL_OPERATIONS[provider]:
        raise ValueError(
            f"Research assumptions cannot authorize {provider} operation {operation}"
        )
    if not isinstance(assumption_path, (str, Path)) or not str(assumption_path).strip():
        raise HistoricalResearchAssumptionError(
            "A content-addressed historical research assumption path is required"
        )
    artifact = verify_historical_research_assumption(
        Path(storage_root), Path(assumption_path)
    )
    if artifact.policy.provider != provider or artifact.policy.operation != operation:
        raise HistoricalResearchAssumptionError(
            "Historical research assumption does not match the requested operation"
        )
    now = datetime.now(UTC) if as_of is None else as_of
    if now.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    accepted_at = _parse_utc(
        artifact.policy.operator_acceptance.accepted_at, "accepted_at"
    )
    if now < accepted_at:
        raise HistoricalResearchAssumptionError(
            "Historical research assumption is not yet valid"
        )
    if now >= _parse_utc(artifact.policy.valid_until, "valid_until"):
        raise HistoricalResearchAssumptionError(
            "Historical research assumption has expired"
        )
    if artifact.policy.account_preflight.status != "passed":
        raise HistoricalResearchAssumptionError(
            "Historical account preflight status is not passed"
        )
    return artifact


def require_historical_operation(
    storage_root: Path,
    provider_approval_path: Path | str,
    assumption_path: Path | str,
    provider: str,
    operation: str,
    *,
    as_of: datetime | None = None,
) -> HistoricalOperationAuthority:
    """Use verified production evidence, else an explicit research assumption.

    A configured production evidence path is authoritative and is never silently
    bypassed when malformed, inactive, or insufficient.
    """

    root = Path(storage_root)
    if (
        isinstance(provider_approval_path, (str, Path))
        and str(provider_approval_path).strip()
    ):
        artifact = require_provider_operation(
            root, provider_approval_path, provider, operation, as_of=as_of
        )
        return HistoricalOperationAuthority(
            kind="provider_approval",
            authority_id=artifact.approval_id,
            output_paths=(artifact.approval_path, artifact.manifest_path),
            provider_approval=artifact,
        )
    artifact = require_historical_research_assumption(
        root, assumption_path, provider, operation, as_of=as_of
    )
    return HistoricalOperationAuthority(
        kind="research_assumption",
        authority_id=artifact.assumption_id,
        output_paths=(artifact.assumption_path, artifact.manifest_path),
        research_assumption=artifact,
    )


def configured_historical_assumption_path(config: Any, provider: str) -> str:
    if provider == "pyth":
        return config.governance.pyth_historical_research_assumption_path
    if provider == "databento":
        return config.governance.databento_historical_research_assumption_path
    raise ValueError(f"Unsupported historical provider: {provider}")


def require_config_historical_operation(
    config: Any, provider: str, operation: str
) -> HistoricalOperationAuthority:
    return require_historical_operation(
        Path(config.storage.root),
        config.governance.provider_approval_path,
        configured_historical_assumption_path(config, provider),
        provider,
        operation,
    )
