from pathlib import Path

from src.metals.provider_approval import materialize_provider_approval

_HASH = "b" * 64


def _confirmation():
    return {
        "version": "test-evidence-v1",
        "confirmed": True,
        "evidence_sha256": _HASH,
    }


def approved_provider_policy():
    return {
        "schema_version": "metals-provider-approval/v1",
        "review_version": "test-only-v1",
        "reviewed_at": "2026-08-30T00:00:00Z",
        "valid_until": None,
        "providers": {
            "pyth": {
                "license": _confirmation(),
                "timestamp_semantics": _confirmation(),
                "operations": {
                    "exact_history": {
                        "approved": True,
                        "entitlement": _confirmation(),
                    },
                    "live_lazer": {
                        "approved": True,
                        "entitlement": _confirmation(),
                    },
                },
            },
            "databento": {
                "license": _confirmation(),
                "timestamp_semantics": _confirmation(),
                "operations": {
                    "historical": {
                        "approved": True,
                        "entitlement": _confirmation(),
                    },
                    "live": {
                        "approved": True,
                        "entitlement": _confirmation(),
                    },
                },
            },
        },
    }


def materialize_approved_provider_path(storage_root: Path) -> str:
    artifact = materialize_provider_approval(storage_root, approved_provider_policy())
    return str(artifact.approval_path)
