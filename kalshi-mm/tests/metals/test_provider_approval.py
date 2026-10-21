import json
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from src.metals.provider_approval import (
    APPROVAL_SCHEMA_VERSION,
    ProviderApprovalError,
    ProviderApprovalPolicy,
    materialize_provider_approval,
    require_provider_operation,
    verify_provider_approval,
)

EVIDENCE = "a" * 64


def _confirmation(*, confirmed=True):
    return {
        "version": "review-v1",
        "confirmed": confirmed,
        "evidence_sha256": EVIDENCE if confirmed else None,
    }


def approval_value(*, confirmed=True, valid_until=None):
    return {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "review_version": "governance-v1",
        "reviewed_at": "2026-08-30T00:00:00Z",
        "valid_until": valid_until,
        "providers": {
            "pyth": {
                "license": _confirmation(confirmed=confirmed),
                "timestamp_semantics": _confirmation(confirmed=confirmed),
                "operations": {
                    "exact_history": {
                        "approved": confirmed,
                        "entitlement": _confirmation(confirmed=confirmed),
                    },
                    "live_lazer": {
                        "approved": confirmed,
                        "entitlement": _confirmation(confirmed=confirmed),
                    },
                },
            },
            "databento": {
                "license": _confirmation(confirmed=confirmed),
                "timestamp_semantics": _confirmation(confirmed=confirmed),
                "operations": {
                    "historical": {
                        "approved": confirmed,
                        "entitlement": _confirmation(confirmed=confirmed),
                    },
                    "live": {
                        "approved": confirmed,
                        "entitlement": _confirmation(confirmed=confirmed),
                    },
                },
            },
        },
    }


def test_materialization_is_content_addressed_idempotent_and_operation_specific(
    tmp_path,
):
    first = materialize_provider_approval(tmp_path, approval_value())
    second = materialize_provider_approval(tmp_path, approval_value())

    assert first == second
    assert first.directory.name == f"approval={first.approval_id}"
    assert verify_provider_approval(tmp_path, first.manifest_path) == first
    assert (
        require_provider_operation(
            tmp_path,
            first.approval_path,
            "pyth",
            "exact_history",
            as_of=datetime(2026, 8, 30, 1, tzinfo=UTC),
        )
        == first
    )
    report = first.public_dict(as_of=datetime(2026, 8, 30, 1, tzinfo=UTC))
    assert report["expired"] is False
    assert report["operations"]["pyth"]["live_lazer"] is True
    assert report["operations"]["databento"]["historical"] is True


def test_unconfirmed_license_semantics_and_entitlement_each_fail_closed(tmp_path):
    cases = []
    license_missing = approval_value()
    license_missing["providers"]["pyth"]["license"] = _confirmation(confirmed=False)
    cases.append((license_missing, "license evidence"))
    semantics_missing = approval_value()
    semantics_missing["providers"]["pyth"]["timestamp_semantics"] = _confirmation(
        confirmed=False
    )
    cases.append((semantics_missing, "timestamp semantics"))
    entitlement_missing = approval_value()
    entitlement_missing["providers"]["pyth"]["operations"]["exact_history"][
        "entitlement"
    ] = _confirmation(confirmed=False)
    cases.append((entitlement_missing, "entitlement"))

    for index, (value, message) in enumerate(cases):
        root = tmp_path / str(index)
        artifact = materialize_provider_approval(root, value)
        with pytest.raises(ProviderApprovalError, match=message):
            require_provider_operation(
                root,
                artifact.approval_path,
                "pyth",
                "exact_history",
                as_of=datetime(2026, 8, 30, 1, tzinfo=UTC),
            )


def test_expired_approval_fails_even_when_every_confirmation_is_true(tmp_path):
    artifact = materialize_provider_approval(
        tmp_path,
        approval_value(valid_until="2026-08-31T00:00:00Z"),
    )

    with pytest.raises(ProviderApprovalError, match="expired"):
        require_provider_operation(
            tmp_path,
            artifact.directory,
            "databento",
            "live",
            as_of=datetime(2026, 9, 1, tzinfo=UTC),
        )


def test_future_dated_approval_is_inactive_and_fails_closed(tmp_path):
    artifact = materialize_provider_approval(tmp_path, approval_value())
    as_of = datetime(2026, 8, 29, 23, 59, tzinfo=UTC)

    report = artifact.public_dict(as_of=as_of)
    assert report["not_yet_valid"] is True
    assert report["operations"]["pyth"]["exact_history"] is False
    with pytest.raises(ProviderApprovalError, match="not yet valid"):
        require_provider_operation(
            tmp_path,
            artifact.directory,
            "pyth",
            "exact_history",
            as_of=as_of,
        )


def test_schema_rejects_unknown_fields_and_boolean_without_evidence():
    extra = approval_value()
    extra["providers"]["pyth"]["license"]["note"] = "not frozen"
    with pytest.raises(ValueError, match="frozen schema"):
        ProviderApprovalPolicy.from_mapping(extra)

    missing_hash = approval_value()
    missing_hash["providers"]["databento"]["license"]["evidence_sha256"] = None
    with pytest.raises(ValueError, match="requires an evidence"):
        ProviderApprovalPolicy.from_mapping(missing_hash)

    pending_version = approval_value()
    pending_version["providers"]["pyth"]["license"]["version"] = "pending"
    with pytest.raises(ValueError, match="cannot be pending"):
        ProviderApprovalPolicy.from_mapping(pending_version)


def test_tampering_and_path_escape_are_rejected(tmp_path):
    artifact = materialize_provider_approval(tmp_path, approval_value())
    changed = json.loads(artifact.approval_path.read_text(encoding="utf-8"))
    changed["review_version"] = "edited"
    artifact.approval_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ProviderApprovalError, match="content hash|manifest"):
        verify_provider_approval(tmp_path, artifact.directory)

    outside = tmp_path.parent / "outside-approval"
    with pytest.raises(ProviderApprovalError, match="escapes"):
        verify_provider_approval(tmp_path, outside)


def test_manifest_rejects_unreviewed_extra_fields(tmp_path):
    artifact = materialize_provider_approval(tmp_path, approval_value())
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    manifest["note"] = "not part of the frozen schema"
    artifact.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProviderApprovalError, match="manifest is invalid"):
        verify_provider_approval(tmp_path, artifact.directory)


def test_one_operation_can_remain_unapproved_without_weakening_another(tmp_path):
    value = deepcopy(approval_value())
    value["providers"]["databento"]["operations"]["live"]["approved"] = False
    artifact = materialize_provider_approval(tmp_path, value)

    require_provider_operation(
        tmp_path,
        artifact.directory,
        "databento",
        "historical",
        as_of=datetime(2026, 8, 30, 1, tzinfo=UTC),
    )
    with pytest.raises(ProviderApprovalError, match="live is not approved"):
        require_provider_operation(
            tmp_path,
            artifact.directory,
            "databento",
            "live",
            as_of=datetime(2026, 8, 30, 1, tzinfo=UTC),
        )
