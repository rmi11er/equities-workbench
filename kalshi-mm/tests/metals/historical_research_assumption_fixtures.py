from src.metals.historical_research_assumption import (
    ASSUMPTION_SCHEMA_VERSION,
    materialize_historical_research_assumption,
)


def historical_assumption_policy(*, provider="pyth", operation="exact_history"):
    return {
        "schema_version": ASSUMPTION_SCHEMA_VERSION,
        "assumption_version": "research-gate-v1",
        "provider": provider,
        "operation": operation,
        "primary_source_evidence": [
            {"url": "https://example.com/technical", "sha256": "a" * 64}
        ],
        "technical_semantics_version": "technical-v1",
        "account_preflight": {
            "status": "passed",
            "observed_at": "2026-08-29T23:59:00Z",
            "evidence_sha256": "b" * 64,
        },
        "research_only": True,
        "no_live": True,
        "no_orders": True,
        "operator_acceptance": {
            "accepted_by": "test-operator",
            "version": "test-acceptance-v1",
            "accepted_at": "2026-08-30T00:00:00Z",
        },
        "valid_until": "2099-01-01T00:00:00Z",
        "known_limitations": ["Historical exploratory research only."],
    }


def materialize_historical_assumption_path(
    storage_root, *, provider="pyth", operation="exact_history"
):
    artifact = materialize_historical_research_assumption(
        storage_root,
        historical_assumption_policy(provider=provider, operation=operation),
    )
    return artifact.assumption_path
