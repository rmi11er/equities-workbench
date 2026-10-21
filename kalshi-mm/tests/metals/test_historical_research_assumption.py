import json
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from src.metals.historical_research_assumption import (
    ASSUMPTION_SCHEMA_VERSION,
    HistoricalResearchAssumptionError,
    HistoricalResearchAssumptionPolicy,
    materialize_historical_research_assumption,
    require_historical_operation,
    require_historical_research_assumption,
    verify_historical_research_assumption,
)
from src.metals.provider_approval import ProviderApprovalError


def assumption_value(
    *,
    provider="pyth",
    operation="exact_history",
    preflight_status="passed",
    valid_until="2026-09-30T00:00:00Z",
):
    return {
        "schema_version": ASSUMPTION_SCHEMA_VERSION,
        "assumption_version": "research-gate-v1",
        "provider": provider,
        "operation": operation,
        "primary_source_evidence": [
            {
                "url": "https://example.com/provider-technical-document",
                "sha256": "a" * 64,
            }
        ],
        "technical_semantics_version": "provider-docs-2026-08-30",
        "account_preflight": {
            "status": preflight_status,
            "observed_at": "2026-08-30T00:00:00Z",
            "evidence_sha256": "b" * 64,
        },
        "research_only": True,
        "no_live": True,
        "no_orders": True,
        "operator_acceptance": {
            "accepted_by": "research-operator",
            "version": "operator-acceptance-v1",
            "accepted_at": "2026-08-30T00:01:00Z",
        },
        "valid_until": valid_until,
        "known_limitations": [
            "Provider contractual rights are assumed, not verified.",
            "This artifact is ineligible for production readiness.",
        ],
    }


def test_materializes_content_addressed_operation_specific_artifact(tmp_path):
    first = materialize_historical_research_assumption(tmp_path, assumption_value())
    second = materialize_historical_research_assumption(tmp_path, assumption_value())

    assert first == second
    assert first.directory.name == f"assumption={first.assumption_id}"
    assert verify_historical_research_assumption(tmp_path, first.manifest_path) == first
    assert (
        require_historical_research_assumption(
            tmp_path,
            first.assumption_path,
            "pyth",
            "exact_history",
            as_of=datetime(2026, 8, 31, tzinfo=UTC),
        )
        == first
    )
    report = first.public_dict(as_of=datetime(2026, 8, 31, tzinfo=UTC))
    assert report["active"] is True
    assert report["research_only"] is True
    assert report["no_live"] is True
    assert report["no_orders"] is True


@pytest.mark.parametrize("flag", ["research_only", "no_live", "no_orders"])
def test_every_safety_flag_must_be_exactly_true(flag):
    value = assumption_value()
    value[flag] = False
    with pytest.raises(ValueError, match=flag):
        HistoricalResearchAssumptionPolicy.from_mapping(value)


def test_only_exact_historical_provider_operation_pairs_are_allowed():
    invalid_pairs = [
        ("pyth", "live_lazer"),
        ("databento", "live"),
        ("databento", "exact_history"),
    ]
    for provider, operation in invalid_pairs:
        with pytest.raises(ValueError, match="cannot authorize"):
            HistoricalResearchAssumptionPolicy.from_mapping(
                assumption_value(provider=provider, operation=operation)
            )


def test_missing_or_unreviewed_evidence_and_limitations_fail_schema():
    cases = []
    no_sources = assumption_value()
    no_sources["primary_source_evidence"] = []
    cases.append(no_sources)
    bad_hash = assumption_value()
    bad_hash["account_preflight"]["evidence_sha256"] = "short"
    cases.append(bad_hash)
    pending_semantics = assumption_value()
    pending_semantics["technical_semantics_version"] = "pending"
    cases.append(pending_semantics)
    no_limitations = assumption_value()
    no_limitations["known_limitations"] = []
    cases.append(no_limitations)

    for value in cases:
        with pytest.raises(ValueError):
            HistoricalResearchAssumptionPolicy.from_mapping(value)


@pytest.mark.parametrize("status", ["failed", "not_checked"])
def test_nonpassing_account_preflight_is_frozen_but_cannot_authorize(tmp_path, status):
    artifact = materialize_historical_research_assumption(
        tmp_path, assumption_value(preflight_status=status)
    )
    with pytest.raises(HistoricalResearchAssumptionError, match="not passed"):
        require_historical_research_assumption(
            tmp_path,
            artifact.directory,
            "pyth",
            "exact_history",
            as_of=datetime(2026, 8, 31, tzinfo=UTC),
        )


def test_future_and_expired_assumptions_fail_closed(tmp_path):
    artifact = materialize_historical_research_assumption(
        tmp_path,
        assumption_value(valid_until="2026-08-31T00:00:00Z"),
    )
    with pytest.raises(HistoricalResearchAssumptionError, match="not yet valid"):
        require_historical_research_assumption(
            tmp_path,
            artifact.directory,
            "pyth",
            "exact_history",
            as_of=datetime(2026, 8, 30, 0, 0, 30, tzinfo=UTC),
        )
    with pytest.raises(HistoricalResearchAssumptionError, match="expired"):
        require_historical_research_assumption(
            tmp_path,
            artifact.directory,
            "pyth",
            "exact_history",
            as_of=datetime(2026, 8, 31, tzinfo=UTC),
        )


def test_tamper_path_escape_and_operation_mismatch_are_rejected(tmp_path):
    artifact = materialize_historical_research_assumption(tmp_path, assumption_value())
    changed = json.loads(artifact.assumption_path.read_text(encoding="utf-8"))
    changed["known_limitations"].append("edited after acceptance")
    artifact.assumption_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(
        HistoricalResearchAssumptionError, match="content hash|manifest"
    ):
        verify_historical_research_assumption(tmp_path, artifact.directory)

    with pytest.raises(HistoricalResearchAssumptionError, match="escapes"):
        verify_historical_research_assumption(tmp_path, tmp_path.parent / "outside")

    other_root = tmp_path / "other"
    databento = materialize_historical_research_assumption(
        other_root,
        assumption_value(provider="databento", operation="historical"),
    )
    with pytest.raises(HistoricalResearchAssumptionError, match="does not match"):
        require_historical_research_assumption(
            other_root,
            databento.directory,
            "pyth",
            "exact_history",
            as_of=datetime(2026, 8, 31, tzinfo=UTC),
        )


def test_research_artifact_cannot_satisfy_production_or_live_gate(tmp_path):
    artifact = materialize_historical_research_assumption(tmp_path, assumption_value())

    with pytest.raises(ValueError, match="cannot authorize"):
        require_historical_operation(
            tmp_path,
            "",
            artifact.assumption_path,
            "pyth",
            "live_lazer",
            as_of=datetime(2026, 8, 31, tzinfo=UTC),
        )
    with pytest.raises(ProviderApprovalError):
        require_historical_operation(
            tmp_path,
            artifact.assumption_path,
            artifact.assumption_path,
            "pyth",
            "exact_history",
            as_of=datetime(2026, 8, 31, tzinfo=UTC),
        )


def test_invalid_configured_production_path_is_never_bypassed(tmp_path):
    artifact = materialize_historical_research_assumption(tmp_path, assumption_value())
    with pytest.raises(ProviderApprovalError):
        require_historical_operation(
            tmp_path,
            tmp_path / "missing-production-artifact",
            artifact.assumption_path,
            "pyth",
            "exact_history",
            as_of=datetime(2026, 8, 31, tzinfo=UTC),
        )


def test_unknown_fields_are_rejected():
    value = deepcopy(assumption_value())
    value["unreviewed_extra"] = True
    with pytest.raises(ValueError, match="frozen schema"):
        HistoricalResearchAssumptionPolicy.from_mapping(value)
