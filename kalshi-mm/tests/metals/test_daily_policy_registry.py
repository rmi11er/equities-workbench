import hashlib
import json
from pathlib import Path

import pytest

from src.metals.daily_gate_measurement import measure_daily_gates
from src.metals.daily_policy_registry import (
    DailyPolicyRegistryError,
    materialize_daily_policy_registries,
    registry_config_hash,
    verify_daily_policy_registry_materialization,
)

DATE = "2026-08-29"
METADATA_HASH = "a" * 64


def _fee(**updates):
    value = {
        "effective_start_ts_ns": 1,
        "effective_end_ts_ns": None,
        "series_ticker": "KXSILVER15M",
        "maker_fee_enabled": False,
        "taker_coefficient": "0.07",
        "maker_coefficient": "0",
        "product_fee_multiplier": "1",
        "account_type_precision_rule": "cent_accumulator_v1",
        "source_document": "fee-schedule-2026-08",
        "retrieval_ts_ns": 50,
        "manual_approval_id": "approval-fee-1",
    }
    value.update(updates)
    value["config_hash"] = registry_config_hash(value)
    return value


def _incentive(**updates):
    value = {
        "program_id": "no-active-program",
        "program_type": "none",
        "eligible_series": ["KXSILVER15M"],
        "eligible_participant_class": "all",
        "program_start_ts_ns": 1,
        "program_end_ts_ns": None,
        "volume_treatment": "none",
        "source_document": "incentive-review-2026-08",
        "retrieval_ts_ns": 50,
    }
    value.update(updates)
    value["config_hash"] = registry_config_hash(value)
    return value


def _policy(fees, incentives, **updates):
    value = {
        "schema_version": 1,
        "date": DATE,
        "measurement_start_ts_ns": 100,
        "measurement_end_ts_ns": 200,
        "expected_windows": [
            {
                "market_ticker": "SILVER-ONE",
                "series_ticker": "KXSILVER15M",
                "metal": "silver",
                "start_ts_ns": 100,
                "end_ts_ns": 200,
            }
        ],
        "source_staleness_ns": {"kalshi": 30, "pyth": 30, "cme": 30},
        "edge_half_life_ns": 100,
        "max_clock_error_fraction": "0.1",
        "clock_sample_staleness_ns": 30,
        "approved_metadata_hashes": {"SILVER-ONE": [METADATA_HASH]},
        "approved_fee_registry_hashes": [
            record["config_hash"] for record in fees
        ],
        "approved_incentive_registry_hashes": [
            record["config_hash"] for record in incentives
        ],
    }
    value.update(updates)
    return value


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_registry_config_hash_has_documented_chain_exclusions():
    record = {
        "series_ticker": "KXSILVER15M",
        "nested": {"config_hash": "this nested field remains covered"},
        "config_hash": "f" * 64,
        "previous_record_sha256": "e" * 64,
        "record_sha256": "d" * 64,
    }
    covered = {
        "nested": {"config_hash": "this nested field remains covered"},
        "series_ticker": "KXSILVER15M",
    }
    expected = hashlib.sha256(
        json.dumps(
            covered,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    assert registry_config_hash(record) == expected


def test_materializes_content_addressed_inputs_for_measurement_and_is_idempotent(
    tmp_path,
):
    root = tmp_path / "metals"
    fee = _fee(previous_record_sha256="ignored input chain")
    other_fee = _fee(
        series_ticker="KXGOLD15M",
        source_document="gold-fee-schedule-2026-08",
        manual_approval_id="approval-fee-2",
    )
    incentive = _incentive()
    policy = _policy([fee, other_fee], [incentive])

    first = materialize_daily_policy_registries(
        root, DATE, policy, [other_fee, fee], [incentive]
    )
    manifest_bytes = first.manifest_path.read_bytes()
    second = materialize_daily_policy_registries(
        root, DATE, policy, [fee, other_fee], [incentive]
    )

    assert second == first
    assert second.manifest_path.read_bytes() == manifest_bytes
    assert first.policy_path.parent.name.startswith("policy=")
    assert "normalized/fee_schedule_dim/date=" in str(first.fee_schedule_path)
    assert "normalized/incentive_program_dim/date=" in str(
        first.incentive_program_path
    )
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["config"]["read_only_sources"] is True
    assert manifest["config"]["network_access"] is False
    assert manifest["config"]["live_orders"] is False
    assert manifest["counts"] == {
        "policies": 1,
        "fee_schedule_dim_records": 2,
        "incentive_program_dim_records": 1,
        "registry_records": 3,
    }

    fee_rows = _rows(first.fee_schedule_path)
    incentive_row = _rows(first.incentive_program_path)[0]
    assert fee_rows[0]["previous_record_sha256"] is None
    assert fee_rows[1]["previous_record_sha256"] == fee_rows[0]["record_sha256"]
    assert incentive_row["previous_record_sha256"] is None
    assert [row["config_hash"] for row in fee_rows] == sorted(
        (fee["config_hash"], other_fee["config_hash"])
    )
    assert all(
        registry_config_hash(row) == row["config_hash"] for row in fee_rows
    )
    assert registry_config_hash(incentive_row) == incentive_row["config_hash"]
    assert (
        verify_daily_policy_registry_materialization(root, first.manifest_path)
        == first
    )

    evidence = measure_daily_gates(
        root, DATE, [first.manifest_path], first.policy_path
    )
    assert evidence["fee_registry"]["validated"] is True
    assert evidence["incentive_registry"]["validated"] is True


@pytest.mark.parametrize(
    ("requested_date", "policy_date", "record_date", "message"),
    (
        ("2026-8-29", DATE, None, "ISO calendar date"),
        (DATE, "2026-08-28", None, "policy date does not match"),
        (DATE, DATE, "2026-08-28", "does not match requested date"),
    ),
)
def test_malformed_or_mismatched_dates_fail_closed(
    tmp_path, requested_date, policy_date, record_date, message
):
    fee = _fee(**({"date": record_date} if record_date else {}))
    incentive = _incentive()
    policy = _policy([fee], [incentive], date=policy_date)

    with pytest.raises(DailyPolicyRegistryError, match=message):
        materialize_daily_policy_registries(
            tmp_path, requested_date, policy, [fee], [incentive]
        )

    assert not (tmp_path / "manifests").exists()


def test_bad_config_hash_and_policy_registry_mismatch_fail_before_writes(tmp_path):
    fee = _fee()
    incentive = _incentive()
    bad_fee = dict(fee, config_hash="0" * 64)

    with pytest.raises(DailyPolicyRegistryError, match="config_hash mismatch"):
        materialize_daily_policy_registries(
            tmp_path, DATE, _policy([bad_fee], [incentive]), [bad_fee], [incentive]
        )

    policy = _policy([fee], [incentive])
    policy["approved_fee_registry_hashes"] = ["f" * 64]
    with pytest.raises(DailyPolicyRegistryError, match="do not match supplied"):
        materialize_daily_policy_registries(
            tmp_path, DATE, policy, [fee], [incentive]
        )
    assert not (tmp_path / "manifests").exists()


def test_completed_retry_detects_tampered_artifact_without_rewriting_manifest(tmp_path):
    fee = _fee()
    incentive = _incentive()
    policy = _policy([fee], [incentive])
    artifacts = materialize_daily_policy_registries(
        tmp_path, DATE, policy, [fee], [incentive]
    )
    manifest_bytes = artifacts.manifest_path.read_bytes()
    artifacts.fee_schedule_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(DailyPolicyRegistryError, match="checksum mismatch"):
        materialize_daily_policy_registries(
            tmp_path, DATE, policy, [fee], [incentive]
        )

    assert artifacts.manifest_path.read_bytes() == manifest_bytes


def test_non_json_and_duplicate_registry_inputs_fail_closed(tmp_path):
    fee = _fee()
    incentive = _incentive()
    non_json_fee = dict(fee, review_score=float("nan"))
    non_json_fee["config_hash"] = "0" * 64

    with pytest.raises(DailyPolicyRegistryError, match="non-finite"):
        materialize_daily_policy_registries(
            tmp_path,
            DATE,
            _policy([non_json_fee], [incentive]),
            [non_json_fee],
            [incentive],
        )

    with pytest.raises(DailyPolicyRegistryError, match="duplicate config_hash"):
        materialize_daily_policy_registries(
            tmp_path,
            DATE,
            _policy([fee], [incentive]),
            [fee, fee],
            [incentive],
        )


def test_fee_decimals_and_registry_live_vintage_are_enforced(tmp_path):
    incentive = _incentive()
    invalid_fee = _fee(taker_coefficient="not-a-number")
    with pytest.raises(DailyPolicyRegistryError, match="not a decimal"):
        materialize_daily_policy_registries(
            tmp_path,
            DATE,
            _policy([invalid_fee], [incentive]),
            [invalid_fee],
            [incentive],
        )

    late_fee = _fee(retrieval_ts_ns=101)
    with pytest.raises(DailyPolicyRegistryError, match="measurement start"):
        materialize_daily_policy_registries(
            tmp_path,
            DATE,
            _policy([late_fee], [incentive]),
            [late_fee],
            [incentive],
        )
