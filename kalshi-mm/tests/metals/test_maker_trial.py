import json
from dataclasses import replace
from decimal import Decimal

import pytest

from src.metals.config import load_metals_config
from src.metals.maker_trial import (
    DAY_NS, NSEC, PUBLIC_SOURCES, SCHEMA_VERSION, TrialResearchPermit,
    materialize_trial_research,
)


NOW = 1788537600 * NSEC


def _payload(**changes):
    return {
        "schema_version": SCHEMA_VERSION,
        "accepted_at_ns": NOW,
        "collection_not_after_ns": NOW + DAY_NS,
        "retention_review_at_ns": NOW + 2 * DAY_NS,
        "accepted_by": "project_owner",
        "acceptance_text": "The trial use should be fine; proceed with the shadow research.",
        "provider": "pyth", "feed_id": 345, "symbol": "Metal.XAG/USD",
        "purpose": "frozen_model_maker_shadow", "no_orders": True,
        "research_only": True, "no_redistribution": True,
        "provider_permission_confirmed": False,
        "provider_trial_expires_at_ns": None,
        "sources": list(PUBLIC_SOURCES), **changes,
    }


def _config():
    return load_metals_config("configs/pyth_xag_trial.toml")


def test_materialized_record_is_immutable_addressed_and_not_provider_approval(tmp_path):
    path = tmp_path / "trial.json"
    permit = materialize_trial_research(path, _payload())
    assert permit.path == path.resolve()
    assert permit.sha256 == permit.trial_research_id
    assert len(permit.sha256) == 64
    assert TrialResearchPermit.load(path) == permit
    assert permit.public_dict()["provider_permission_confirmed"] is False
    assert permit.public_dict()["provider_trial_expires_at_ns"] is None
    assert not hasattr(permit, "approval_id")
    permit.require_current(NOW)
    permit.validate_scope(_config(), "silver-xag-proxy")
    permit.require_duration(900, NOW)
    with pytest.raises(FileExistsError):
        materialize_trial_research(path, _payload(acceptance_text="a different acceptance"))
    assert TrialResearchPermit.load(path) == permit


def test_mutation_of_source_or_public_dict_cannot_change_frozen_permit():
    source = _payload()
    permit = TrialResearchPermit.from_dict(source)
    digest = permit.sha256
    source["sources"].clear()
    output = permit.public_dict()
    output["sources"].clear()
    output["provider_permission_confirmed"] = True
    assert permit.sha256 == digest
    assert permit.public_dict()["sources"] == list(PUBLIC_SOURCES)
    assert permit.public_dict()["provider_permission_confirmed"] is False


def test_load_rejects_unhashed_or_tampered_artifact(tmp_path):
    path = tmp_path / "trial.json"
    permit = materialize_trial_research(path, _payload())
    changed = permit.public_dict() | {"acceptance_text": "changed after review"}
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        TrialResearchPermit.load(path)
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    with pytest.raises(ValueError, match="requires a content hash"):
        TrialResearchPermit.load(path)


@pytest.mark.parametrize("changes", [
    {"provider_permission_confirmed": True}, {"no_orders": False},
    {"research_only": 1}, {"no_redistribution": False}, {"feed_id": "345"},
    {"feed_id": 3154}, {"symbol": "Metal.Index.SILVER/USD"}, {"provider": "databento"},
    {"purpose": "live_trading"}, {"accepted_by": "provider"}, {"acceptance_text": " "},
    {"accepted_at_ns": True}, {"collection_not_after_ns": NOW},
    {"collection_not_after_ns": NOW + DAY_NS + 1},
    {"retention_review_at_ns": NOW + DAY_NS},
    {"retention_review_at_ns": NOW + 7 * DAY_NS + 1},
    {"provider_trial_expires_at_ns": NOW}, {"provider_trial_expires_at_ns": True},
    {"sources": ["https://example.com/"]}, {"sources": list(PUBLIC_SOURCES) * 2},
    {"sources": [None]}, {"approval_id": "pretend-provider-approval"},
])
def test_record_schema_does_not_allow_scope_or_authority_expansion(changes):
    with pytest.raises(ValueError):
        TrialResearchPermit.from_dict(_payload(**changes))


def test_future_acceptance_and_exact_expiry_boundaries_fail_closed():
    permit = TrialResearchPermit.from_dict(_payload())
    with pytest.raises(ValueError, match="not yet effective"):
        permit.require_current(NOW - 1)
    permit.require_current(NOW + DAY_NS - 1)
    with pytest.raises(ValueError, match="local.*deadline has expired"):
        permit.require_current(NOW + DAY_NS)
    early = TrialResearchPermit.from_dict(_payload(provider_trial_expires_at_ns=NOW + 100 * NSEC))
    early.require_current(NOW + 100 * NSEC - 1)
    with pytest.raises(ValueError, match="known provider trial has expired"):
        early.require_current(NOW + 100 * NSEC)


def test_duration_reserves_drain_and_uses_stricter_known_provider_expiry():
    permit = TrialResearchPermit.from_dict(_payload(provider_trial_expires_at_ns=NOW + 100 * NSEC))
    permit.require_duration(97, NOW)
    with pytest.raises(ValueError, match="3-second drain"):
        permit.require_duration(97.000000001, NOW)
    permit = TrialResearchPermit.from_dict(_payload(collection_not_after_ns=NOW + 10 * NSEC))
    permit.require_duration(Decimal("7"), NOW)
    with pytest.raises(ValueError, match="3-second drain"):
        permit.require_duration(Decimal("7.0000000001"), NOW)


@pytest.mark.parametrize("duration", [0, -1, 900.1, float("nan"), float("inf"), True, "60"])
def test_duration_rejects_unbounded_invalid_or_ambiguous_values(duration):
    with pytest.raises(ValueError):
        TrialResearchPermit.from_dict(_payload()).require_duration(duration, NOW)


def test_wall_clock_is_used_when_now_is_not_supplied(monkeypatch):
    permit = TrialResearchPermit.from_dict(_payload())
    monkeypatch.setattr("src.metals.maker_trial.time.time_ns", lambda: NOW)
    permit.require_current()
    permit.require_duration(30)
    monkeypatch.setattr("src.metals.maker_trial.time.time_ns", lambda: NOW + DAY_NS)
    with pytest.raises(ValueError, match="expired"):
        permit.require_current()


@pytest.mark.parametrize("change", [
    lambda c: replace(c, shadow=replace(c.shadow, execution_mode="live")),
    lambda c: replace(c, shadow=replace(c.shadow, allow_live_orders=True)),
    lambda c: replace(c, shadow=replace(c.shadow, enabled=False)),
    lambda c: replace(c, pyth=replace(c.pyth, enabled=False)),
    lambda c: replace(c, cme=replace(c.cme, enabled=True)),
    lambda c: replace(c, products={"silver-xag-proxy": replace(c.products["silver-xag-proxy"], pyth_lazer_id=3154)}),
    lambda c: replace(c, products={"silver-xag-proxy": replace(c.products["silver-xag-proxy"], pyth_symbol="Metal.XAU/USD")}),
    lambda c: replace(c, products={**c.products, "another": c.products["silver-xag-proxy"]}),
    lambda c: replace(c, pyth=replace(c.pyth, lazer_ws_endpoints=c.pyth.lazer_ws_endpoints[:2])),
    lambda c: replace(c, pyth=replace(c.pyth, lazer_ws_endpoints=[c.pyth.lazer_ws_endpoints[0]] * 3)),
    lambda c: replace(c, pyth=replace(c.pyth, lazer_ws_endpoints=["wss://example.com/stream", *c.pyth.lazer_ws_endpoints[:2]])),
    lambda c: replace(c, pyth=replace(c.pyth, endpoint="https://example.com/v1")),
])
def test_scope_rejects_other_feeds_sources_endpoints_and_live_execution(change):
    with pytest.raises(ValueError):
        TrialResearchPermit.from_dict(_payload()).validate_scope(change(_config()), "silver-xag-proxy")


def test_unknown_product_or_direct_constructor_cannot_skip_validation():
    permit = TrialResearchPermit.from_dict(_payload())
    with pytest.raises(ValueError, match="exactly one enabled"):
        permit.validate_scope(_config(), "gold")
    with pytest.raises(ValueError, match="hash mismatch"):
        TrialResearchPermit(permit._payload_json, "0" * 64)
    with pytest.raises(ValueError, match="required schema"):
        TrialResearchPermit("{}", "0" * 64)
