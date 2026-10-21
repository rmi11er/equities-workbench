import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import src.metals.research_design as research_design_module
from src.metals.research_design import (
    FinalHoldoutSealedError,
    ResearchDesignError,
    ResearchDesignPolicy,
    assign_complete_window,
    materialize_research_design,
    verify_research_design,
)

EXAMPLE = (
    Path(__file__).resolve().parents[2] / "configs" / "research_design.example.json"
)


def design_value():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value)


def assignment(policy, open_text, *, complete=True, label_delta=timedelta(minutes=15)):
    opened = utc(open_text)
    closed = opened + timedelta(minutes=15)
    return assign_complete_window(
        policy,
        window_open=opened,
        window_close=closed,
        maximum_forward_label_end=opened + label_delta,
        maximum_order_lifetime_end=closed,
        complete=complete,
    )


def test_example_is_strict_and_canonical():
    value = design_value()
    policy = ResearchDesignPolicy.from_mapping(value)

    assert policy.to_dict() == value
    assert policy.research_only is True
    assert policy.no_live is True
    assert policy.no_orders is True
    assert policy.provisional_label.eligible_as_training_label is False
    assert policy.provisional_label.value_field == "result"
    assert policy.observation_unit.required_terminal_streams == (
        "market_dim",
        "kalshi_settlement",
    )
    assert policy.exact_pyth_label.construction_uses_exchange_result is False
    assert (
        policy.exact_pyth_label.eligibility_gate
        == "audit_usable_and_target_value_result_agreements_all_true"
    )
    assert policy.splits.interval("final_holdout").start == "2026-11-16T00:00:00Z"


def test_materialize_verify_and_idempotence_are_content_addressed(tmp_path):
    first = materialize_research_design(tmp_path, design_value())
    second = materialize_research_design(tmp_path, design_value())

    assert first == second
    assert first.directory.name == f"design={first.design_id}"
    assert verify_research_design(tmp_path, first.design_path) == first
    assert verify_research_design(tmp_path, first.manifest_path) == first
    assert first.public_dict()["final_holdout_status"] == "sealed_unread"


def test_changed_design_gets_a_different_identity(tmp_path):
    first = materialize_research_design(tmp_path, design_value())
    changed = design_value()
    changed["limitations"]["items"].append(
        {"id": "L4", "statement": "A separately versioned limitation."}
    )
    second = materialize_research_design(tmp_path, changed)

    assert first.design_id != second.design_id
    assert first.directory != second.directory


@pytest.mark.parametrize("flag", ["research_only", "no_live", "no_orders"])
def test_safety_flags_are_literal_true(flag):
    value = design_value()
    value[flag] = False
    with pytest.raises(ValueError, match=flag):
        ResearchDesignPolicy.from_mapping(value)


def test_unknown_or_missing_split_fields_fail_closed():
    unknown = design_value()
    unknown["unreviewed_extra"] = True
    with pytest.raises(ValueError, match="frozen schema"):
        ResearchDesignPolicy.from_mapping(unknown)

    missing_boundary = design_value()
    del missing_boundary["splits"]["intervals"]["validation"]["end"]
    with pytest.raises(ValueError, match="fields do not match"):
        ResearchDesignPolicy.from_mapping(missing_boundary)


def test_splits_must_be_strictly_chronological_and_contiguous():
    value = design_value()
    value["splits"]["intervals"]["validation"]["start"] = "2026-10-25T00:00:00Z"
    with pytest.raises(ValueError, match="chronological and contiguous"):
        ResearchDesignPolicy.from_mapping(value)


def test_final_holdout_must_be_future_when_design_is_frozen(monkeypatch):
    class AfterPanelDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2027, 1, 1, tzinfo=tz)

    monkeypatch.setattr(research_design_module, "datetime", AfterPanelDatetime)
    value = design_value()
    value["frozen_at"] = "2026-11-16T00:00:00Z"
    with pytest.raises(ValueError, match="future and unread"):
        ResearchDesignPolicy.from_mapping(value)


def test_freeze_timestamp_cannot_be_in_the_future():
    value = design_value()
    value["frozen_at"] = "2999-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="frozen_at cannot be in the future"):
        ResearchDesignPolicy.from_mapping(value)


def test_embargo_covers_both_maximum_horizons():
    value = design_value()
    value["leakage_control"]["embargo_seconds"] = 899
    with pytest.raises(ValueError, match="at least the maximum"):
        ResearchDesignPolicy.from_mapping(value)

    mismatch = design_value()
    mismatch["leakage_control"]["max_forward_label_horizon_seconds"] = 300
    with pytest.raises(ValueError, match="largest horizon"):
        ResearchDesignPolicy.from_mapping(mismatch)


def test_provisional_and_exact_label_contracts_cannot_be_conflated():
    provisional = design_value()
    provisional["label_contract"]["provisional_exchange_reported"][
        "eligible_as_training_label"
    ] = True
    with pytest.raises(ValueError, match="eligible_as_training_label"):
        ResearchDesignPolicy.from_mapping(provisional)

    exact = design_value()
    exact["label_contract"]["independently_reconstructed_exact_pyth"][
        "construction_uses_exchange_result"
    ] = True
    with pytest.raises(ValueError, match="construction_uses_exchange_result"):
        ResearchDesignPolicy.from_mapping(exact)

    weak_gate = design_value()
    weak_gate["label_contract"]["independently_reconstructed_exact_pyth"][
        "eligibility_gate"
    ] = "exchange_result_present"
    with pytest.raises(ValueError, match="eligibility_gate"):
        ResearchDesignPolicy.from_mapping(weak_gate)


def test_complete_research_window_is_assigned_atomically():
    policy = ResearchDesignPolicy.from_mapping(design_value())
    result = assignment(policy, "2026-09-14T00:00:00Z")

    assert result.split == "research"
    assert result.eligible is True
    assert result.exclusion_reason is None
    assert result.utc_trading_day == "2026-09-14"


def test_incomplete_and_preexisting_windows_are_never_eligible():
    policy = ResearchDesignPolicy.from_mapping(design_value())
    incomplete = assignment(policy, "2026-09-14T00:00:00Z", complete=False)
    plumbing = assignment(policy, "2026-08-30T12:00:00Z")

    assert incomplete == incomplete.__class__(
        None, False, "incomplete_window", "2026-09-14"
    )
    assert plumbing.split is None
    assert plumbing.eligible is False
    assert plumbing.exclusion_reason == "preexisting_research_plumbing_only"


def test_forward_label_crossing_boundary_is_purged():
    policy = ResearchDesignPolicy.from_mapping(design_value())
    opened = utc("2026-10-25T23:45:00Z")
    result = assign_complete_window(
        policy,
        window_open=opened,
        window_close=opened + timedelta(minutes=15),
        maximum_forward_label_end=utc("2026-10-26T00:00:01Z"),
        maximum_order_lifetime_end=utc("2026-10-26T00:00:00Z"),
        complete=True,
    )

    assert result.split == "research"
    assert result.eligible is False
    assert result.exclusion_reason == "purged_at_split_end"


def test_validation_start_is_embargoed_for_frozen_maximum_lifetime():
    policy = ResearchDesignPolicy.from_mapping(design_value())
    excluded = assignment(policy, "2026-10-26T00:00:00Z")
    included = assignment(policy, "2026-10-26T00:15:00Z")

    assert excluded.exclusion_reason == "embargoed_at_split_start"
    assert included.split == "validation"
    assert included.eligible is True


def test_final_holdout_cannot_be_classified_or_unsealed_by_boolean():
    policy = ResearchDesignPolicy.from_mapping(design_value())
    with pytest.raises(FinalHoldoutSealedError, match="sealed"):
        assignment(policy, "2026-11-16T00:15:00Z")


def test_assignment_requires_utc_and_valid_window_order():
    policy = ResearchDesignPolicy.from_mapping(design_value())
    naive = datetime(2026, 9, 14)  # noqa: DTZ001 - intentional invalid input
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        assign_complete_window(
            policy,
            window_open=naive,
            window_close=naive + timedelta(minutes=15),
            maximum_forward_label_end=naive + timedelta(minutes=15),
            maximum_order_lifetime_end=naive + timedelta(minutes=15),
            complete=True,
        )


def test_tamper_manifest_and_path_escape_are_rejected(tmp_path):
    artifact = materialize_research_design(tmp_path, design_value())
    changed = json.loads(artifact.design_path.read_text(encoding="utf-8"))
    changed["limitations"]["items"][0]["statement"] += " Edited."
    artifact.design_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ResearchDesignError, match="content hash|manifest"):
        verify_research_design(tmp_path, artifact.directory)

    with pytest.raises(ResearchDesignError, match="escapes"):
        verify_research_design(tmp_path, tmp_path.parent / "outside")


def test_holdout_policy_requires_zero_prior_reads_and_full_freeze_sequence():
    read = design_value()
    read["final_holdout_policy"]["records_read_before_unseal"] = 1
    with pytest.raises(ValueError, match="records_read_before_unseal"):
        ResearchDesignPolicy.from_mapping(read)

    missing = design_value()
    missing["final_holdout_policy"]["unseal_prerequisites"].pop()
    with pytest.raises(ValueError, match="prerequisites"):
        ResearchDesignPolicy.from_mapping(missing)
