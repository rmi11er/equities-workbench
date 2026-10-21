import json
from pathlib import Path

import pytest

from src.metals.daily_gate_measurement import (
    DailyGateMeasurementError,
    _stream_partition,
    measure_and_persist_daily_gates,
    measure_daily_gates,
    merge_daily_gate_evidence,
    verify_daily_gate_evidence,
)
from src.metals.daily_validation import (
    freeze_daily_validation,
    synchronized_day_progress,
    verify_daily_snapshot,
)
from src.metals.storage import (
    PartitionedRawStore,
    update_run_manifest,
    write_run_manifest,
)
from tests.metals.provider_approval_fixtures import (
    materialize_approved_provider_path,
)

DATE = "2026-08-29"
METADATA_HASH = "a" * 64
FEE_HASH = "b" * 64
INCENTIVE_HASH = "c" * 64


def test_settlement_audit_sidecars_are_not_parsed_as_jsonl(tmp_path):
    root = (tmp_path / "metals").resolve()
    directory = root / "derived" / "settlement_check" / ("audit=" + "a" * 64)

    assert _stream_partition(directory / "windows.jsonl", root)[0] == (
        "settlement_check"
    )
    assert _stream_partition(directory / "summary.json", root)[0] is None
    assert _stream_partition(directory / "manifest.json", root)[0] is None


def _policy():
    return {
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
        "approved_fee_registry_hashes": [FEE_HASH],
        "approved_incentive_registry_hashes": [INCENTIVE_HASH],
    }


def _metadata():
    return {
        "market_ticker": "SILVER-ONE",
        "event_ticker": "SILVER-EVENT",
        "series_ticker": "KXSILVER15M",
        "metal": "silver",
        "open_ts_ns": 100,
        "close_ts_ns": 200,
        "determination_ts_ns": 201,
        "target_price": "32.123",
        "round_digits": 3,
        "strike_type": "greater_or_equal",
        "settlement_source_name": "Pyth Silver",
        "settlement_source_url": "https://example.invalid/silver",
        "rules_primary": "The closing value is at least the opening value.",
        "rules_secondary": "Use the preceding minute close and round to 3 decimals.",
        "pyth_symbol": "Metal.Index.SILVER/USD",
        "pyth_feed_id": "feed",
        "price_level_structure": "standard",
        "price_ranges_json": "[]",
        "fractional_enabled": True,
        "exchange_index": 1,
        "metadata_version": 1,
        "metadata_hash": METADATA_HASH,
        "metadata_receive_ts_ns": 90,
        "missing_required_fields": [],
    }


def _settlement():
    return {
        "market_ticker": "SILVER-ONE",
        "exchange_result": "yes",
        "exchange_settlement_value": "32.124",
        "reconstructed_target": "32.123",
        "reconstructed_final_value": "32.124",
        "opening_source_update_id": "open-source-id",
        "settlement_source_update_id": "close-source-id",
        "round_digits": 3,
        "tie_flag": False,
        "stale_source_fallback": False,
        "source_age_ns": 1,
        "predicted_result": "yes",
        "agreement": True,
        "target_agreement": True,
        "usable": True,
        "audit_code_version": "audit-v1",
    }


def _fee_registry():
    return {
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
        "config_hash": FEE_HASH,
    }


def _incentive_registry():
    return {
        "program_id": "no-active-program",
        "program_type": "none",
        "eligible_series": ["KXSILVER15M"],
        "eligible_participant_class": "all",
        "program_start_ts_ns": 1,
        "program_end_ts_ns": None,
        "volume_treatment": "none",
        "source_document": "incentive-review-2026-08",
        "retrieval_ts_ns": 50,
        "config_hash": INCENTIVE_HASH,
    }


def _write_stream(root: Path, run_id: str, stream: str, records):
    store = PartitionedRawStore(root, run_id=run_id, tier="normalized")
    for record in records:
        receive_ts = record.get(
            "receive_ts_ns",
            record.get(
                "metadata_receive_ts_ns",
                record.get("sample_ts_ns", 150),
            ),
        )
        store.write(stream, stream, "silver", receive_ts, record)
    store.close()
    return store.paths


def _fixture(tmp_path: Path, *, omit_streams=(), stream_overrides=None, policy=None):
    root = tmp_path / "metals"
    run_id = "daily-evidence-run"
    outputs = []
    streams = {
        "market_dim": [_metadata()],
        "settlement_check": [_settlement()],
        "fee_schedule_dim": [_fee_registry()],
        "incentive_program_dim": [_incentive_registry()],
        "kalshi_book_event": [
            {
                "market_ticker": "SILVER-ONE",
                "receive_ts_ns": timestamp,
                "book_valid_after": True,
            }
            for timestamp in (90, 120, 150, 180)
        ],
        "pyth_lazer_price": [
            {
                "metal": "silver",
                "receive_ts_ns": timestamp,
                "source_publish_ts_ns": timestamp,
                "price": "32.1",
            }
            for timestamp in (90, 120, 150, 180)
        ],
        "cme_mbo_event": [
            {
                "metal": "silver",
                "receive_ts_ns": timestamp,
                "source_event_ts_ns": timestamp,
            }
            for timestamp in (90, 120, 150, 180)
        ],
        "clock_telemetry": [
            {
                "sample_ts_ns": timestamp,
                "clock_offset_ns": -1,
                "clock_synchronized": True,
            }
            for timestamp in (90, 120, 150, 180)
        ],
    }
    streams.update(stream_overrides or {})
    for stream, records in streams.items():
        if stream not in omit_streams:
            outputs.extend(_write_stream(root, run_id, stream, records))

    policy_path = root / "policies" / "daily-gates" / "policy.json"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(
        json.dumps(policy or _policy(), sort_keys=True), encoding="utf-8"
    )
    outputs.append(policy_path)
    manifest_path = write_run_manifest(root, run_id, {"read_only": True})
    update_run_manifest(
        root,
        run_id,
        status="completed",
        output_paths=outputs,
    )
    return root, manifest_path, policy_path


def _collector_report():
    return {
        "date": DATE,
        "run_id": "synchronized-run",
        "strategy_decisions_on_invalid_book": 0,
        "causal_ordering_checked": True,
        "timestamp_ordering_violations": 0,
        "missing_intervals": [],
        "exclusions": [],
    }


def _manifest_collector_report(root: Path):
    quality_path = root / "derived" / "collector_quality" / "report.json"
    quality_path.parent.mkdir(parents=True)
    quality_path.write_text(
        json.dumps(_collector_report(), sort_keys=True), encoding="utf-8"
    )
    run_id = "synchronized-run"
    approval_path = materialize_approved_provider_path(root)
    approval_directory = Path(approval_path).parent
    approval_id = approval_directory.name.removeprefix("approval=")
    manifest_path = write_run_manifest(
        root,
        run_id,
        {
            "governance": {"provider_approval_path": str(approval_path)},
            "collector": {
                "component": "synchronized_metals_live",
                "read_only": True,
                "required_sources": ["kalshi", "pyth_lazer", "databento"],
                "required_telemetry": ["clock"],
                "provider_approval_ids": {
                    "pyth": approval_id,
                    "databento": approval_id,
                },
            },
        },
        output_paths=(
            Path(approval_path),
            approval_directory / "manifest.json",
        ),
    )
    update_run_manifest(
        root,
        run_id,
        status="completed",
        output_paths=(quality_path,),
    )
    return manifest_path, quality_path


def test_complete_persisted_evidence_measures_and_passes_daily_gates(tmp_path):
    root, manifest, policy = _fixture(tmp_path)

    evidence = measure_daily_gates(root, DATE, [manifest], policy)
    merged = merge_daily_gate_evidence(_collector_report(), evidence)

    verify_daily_gate_evidence(evidence)
    assert evidence["windows"]["expected_windows"] == 1
    assert evidence["windows"]["observed_windows"] == 1
    assert evidence["coverage"]["kalshi_valid_fraction"] == 1.0
    assert evidence["coverage"]["pyth_valid_fraction"] == 1.0
    assert evidence["coverage"]["cme_valid_fraction"] == 1.0
    assert evidence["clock"]["within_edge_half_life"] is True
    assert evidence["settlement_checksum"]["agreements"] == 1
    assert evidence["metadata_registry"]["validated"] is True
    assert evidence["fee_registry"]["validated"] is True
    assert evidence["incentive_registry"]["validated"] is True
    assert merged["data_gates"]["passed"] is True


def test_measure_and_persist_creates_content_addressed_artifacts_and_manifest(tmp_path):
    root, source_manifest, policy = _fixture(tmp_path)
    quality_manifest, quality_path = _manifest_collector_report(root)

    first = measure_and_persist_daily_gates(
        root,
        DATE,
        quality_path,
        [source_manifest, quality_manifest],
        policy,
    )
    frozen_manifest_bytes = first.manifest_path.read_bytes()
    second = measure_and_persist_daily_gates(
        root,
        DATE,
        quality_path,
        [source_manifest, quality_manifest],
        policy,
    )

    assert first == second
    assert second.manifest_path.read_bytes() == frozen_manifest_bytes
    evidence = json.loads(first.evidence_path.read_text(encoding="utf-8"))
    report = json.loads(first.quality_report_path.read_text(encoding="utf-8"))
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert first.evidence_path.parent.name == (
        "evidence=" + evidence["evidence_sha256"]
    )
    assert first.quality_report_path.parent.name == (
        "report=" + report["daily_quality_sha256"]
    )
    assert report["daily_gate_measurement"] == evidence
    assert report["data_gates"]["passed"] is True
    assert manifest["status"] == "completed"
    assert set(manifest["output_paths"]) == {
        str(first.evidence_path),
        str(first.quality_report_path),
    }


def test_measure_and_persist_rejects_unmanifested_collector_report(tmp_path):
    root, source_manifest, policy = _fixture(tmp_path)
    quality_path = root / "quality.json"
    quality_path.write_text(
        json.dumps(_collector_report(), sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(DailyGateMeasurementError, match="listed"):
        measure_and_persist_daily_gates(
            root,
            DATE,
            quality_path,
            [source_manifest],
            policy,
        )


def test_measure_and_persist_requires_synchronized_quality_owner(tmp_path):
    root, source_manifest, policy = _fixture(tmp_path)
    quality_path = root / "quality.json"
    quality_path.write_text(
        json.dumps(_collector_report(), sort_keys=True), encoding="utf-8"
    )
    owner = write_run_manifest(
        root,
        "synchronized-run",
        {"collector": {"component": "single_source", "read_only": True}},
    )
    update_run_manifest(
        root,
        "synchronized-run",
        status="completed",
        output_paths=(quality_path,),
    )

    with pytest.raises(DailyGateMeasurementError, match="synchronized-panel"):
        measure_and_persist_daily_gates(
            root,
            DATE,
            quality_path,
            [source_manifest, owner],
            policy,
        )


def test_measurement_rejects_unlisted_provider_approval_artifacts(tmp_path):
    root, source_manifest, policy = _fixture(tmp_path)
    quality_manifest, quality_path = _manifest_collector_report(root)
    manifest = json.loads(quality_manifest.read_text(encoding="utf-8"))
    manifest["output_paths"] = [
        path
        for path in manifest["output_paths"]
        if "/policies/provider_approval/" not in path
    ]
    quality_manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(DailyGateMeasurementError, match="not manifest-listed"):
        measure_and_persist_daily_gates(
            root,
            DATE,
            quality_path,
            [source_manifest, quality_manifest],
            policy,
        )


def test_measured_artifacts_freeze_and_count_once_end_to_end(tmp_path):
    root, source_manifest, policy = _fixture(tmp_path)
    quality_manifest, quality_path = _manifest_collector_report(root)
    artifacts = measure_and_persist_daily_gates(
        root,
        DATE,
        quality_path,
        [source_manifest, quality_manifest],
        policy,
    )

    snapshot_path = freeze_daily_validation(
        root,
        DATE,
        artifacts.quality_report_path,
        [source_manifest, quality_manifest, artifacts.manifest_path],
    )
    snapshot = verify_daily_snapshot(root, snapshot_path)
    progress = synchronized_day_progress(
        root,
        minimum_complete_days=1,
        minimum_settlements=1,
    )

    assert snapshot["research_eligible"] is True
    assert progress["passed"] is True
    assert progress["complete_days"] == 1
    assert progress["settlements_checked"] == 1
    approval_inputs = [
        item
        for item in snapshot["inputs"]
        if item["path"].startswith("policies/provider_approval/")
    ]
    assert {Path(item["path"]).name for item in approval_inputs} == {
        "approval.json",
        "manifest.json",
    }


def test_missing_persisted_inputs_fail_closed_without_inventing_measurements(tmp_path):
    root, manifest, policy = _fixture(
        tmp_path,
        omit_streams=(
            "market_dim",
            "settlement_check",
            "fee_schedule_dim",
            "incentive_program_dim",
            "pyth_lazer_price",
            "clock_telemetry",
        ),
    )

    evidence = measure_daily_gates(root, DATE, [manifest], policy)
    merged = merge_daily_gate_evidence(_collector_report(), evidence)

    assert evidence["windows"]["observed_windows"] == 0
    assert evidence["coverage"]["pyth_valid_ns"] == 0
    assert evidence["coverage"]["pyth_measurement_available"] is False
    assert evidence["clock"]["measured"] is False
    assert evidence["settlement_checksum"]["checked"] == 0
    assert evidence["metadata_registry"]["validated"] is False
    assert evidence["fee_registry"]["validated"] is False
    assert evidence["incentive_registry"]["validated"] is False
    assert set(merged["data_gates"]["failed_gates"]) >= {
        "settlement_agreement",
        "pyth_coverage",
        "clock_health",
        "rules_unchanged",
    }


def test_fee_group_and_market_scoped_registries_are_supported(tmp_path):
    policy_value = _policy()
    policy_value["expected_windows"][0]["fee_group_id"] = "standard-metals"
    fee = _fee_registry()
    fee.pop("series_ticker")
    fee["fee_group_id"] = "standard-metals"
    incentive = _incentive_registry()
    incentive.pop("eligible_series")
    incentive["eligible_market_ticker"] = "SILVER-ONE"
    root, manifest, policy_path = _fixture(
        tmp_path,
        policy=policy_value,
        stream_overrides={
            "fee_schedule_dim": [fee],
            "incentive_program_dim": [incentive],
        },
    )

    evidence = measure_daily_gates(root, DATE, [manifest], policy_path)

    assert evidence["fee_registry"]["validated"] is True
    assert evidence["incentive_registry"]["validated"] is True


def test_intraday_registry_boundary_is_counted_and_fails_rule_gate(tmp_path):
    replacement_hash = "d" * 64
    policy_value = _policy()
    policy_value["approved_fee_registry_hashes"].append(replacement_hash)
    before = _fee_registry()
    before["effective_end_ts_ns"] = 150
    after = _fee_registry()
    after["effective_start_ts_ns"] = 150
    after["config_hash"] = replacement_hash
    root, manifest, policy_path = _fixture(
        tmp_path,
        policy=policy_value,
        stream_overrides={"fee_schedule_dim": [before, after]},
    )

    evidence = measure_daily_gates(root, DATE, [manifest], policy_path)
    merged = merge_daily_gate_evidence(_collector_report(), evidence)

    assert evidence["fee_registry"]["changes"] == 1
    assert evidence["fee_registry"]["validated"] is False
    assert merged["data_gates"]["gates"]["rules_unchanged"]["passed"] is False


def test_staleness_is_time_weighted_and_emits_classified_intervals(tmp_path):
    root, manifest, _policy_path = _fixture(tmp_path)
    pyth_path = next((root / "normalized" / "pyth_lazer_price").glob("**/*.jsonl"))
    # Create a second completed manifest with a policy that uses a shorter
    # staleness budget; persisted observations stay immutable.
    policy_value = _policy()
    policy_value["source_staleness_ns"]["pyth"] = 5
    policy_path = root / "policies" / "daily-gates" / "short-policy.json"
    policy_path.write_text(json.dumps(policy_value, sort_keys=True), encoding="utf-8")
    second_manifest = write_run_manifest(root, "short-policy-run", {"read_only": True})
    update_run_manifest(
        root,
        "short-policy-run",
        status="completed",
        output_paths=[pyth_path, policy_path],
    )

    evidence = measure_daily_gates(root, DATE, [manifest, second_manifest], policy_path)

    assert evidence["coverage"]["pyth_valid_ns"] == 15
    assert evidence["coverage"]["pyth_stale_ns"] == 85
    pyth_gaps = [
        interval
        for interval in evidence["missing_intervals"]
        if interval["source"] == "pyth"
    ]
    assert pyth_gaps == [
        {
            "source": "pyth",
            "ticker": "SILVER-ONE",
            "start_ts_ns": 100,
            "end_ts_ns": 120,
            "reason": "pyth_missing_or_stale",
        },
        {
            "source": "pyth",
            "ticker": "SILVER-ONE",
            "start_ts_ns": 125,
            "end_ts_ns": 150,
            "reason": "pyth_missing_or_stale",
        },
        {
            "source": "pyth",
            "ticker": "SILVER-ONE",
            "start_ts_ns": 155,
            "end_ts_ns": 180,
            "reason": "pyth_missing_or_stale",
        },
        {
            "source": "pyth",
            "ticker": "SILVER-ONE",
            "start_ts_ns": 185,
            "end_ts_ns": 200,
            "reason": "pyth_missing_or_stale",
        },
    ]


def test_carried_forward_pyth_updates_expire_from_feed_update_time(tmp_path):
    carried = [
        {
            "metal": "silver",
            "receive_ts_ns": timestamp,
            "source_publish_ts_ns": 99,
            "feed_update_timestamp_us": 0,
            "is_carried_forward": True,
            "price": "32.1",
        }
        for timestamp in (100, 120, 140, 160, 180)
    ]
    root, manifest, policy = _fixture(
        tmp_path,
        stream_overrides={"pyth_lazer_price": carried},
    )

    evidence = measure_daily_gates(root, DATE, [manifest], policy)

    # The repeated delivery timestamps cannot refresh a price generated at t=99.
    assert evidence["coverage"]["pyth_valid_ns"] == 29
    assert evidence["coverage"]["pyth_stale_ns"] == 71
    assert evidence["coverage"]["pyth_valid_fraction"] == 0.29


def test_tampered_hash_chain_and_unlisted_policy_are_rejected(tmp_path):
    root, manifest, policy = _fixture(tmp_path)
    metadata_path = next((root / "normalized" / "market_dim").glob("**/*.jsonl"))
    row = json.loads(metadata_path.read_text(encoding="utf-8"))
    row["target_price"] = "999"
    metadata_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(DailyGateMeasurementError, match="record hash"):
        measure_daily_gates(root, DATE, [manifest], policy)

    unlisted = root / "policies" / "daily-gates" / "unlisted.json"
    unlisted.write_text(json.dumps(_policy()), encoding="utf-8")
    with pytest.raises(DailyGateMeasurementError, match="listed"):
        measure_daily_gates(root, DATE, [manifest], unlisted)


def test_conflicting_settlement_audits_fail_completeness_without_counting(tmp_path):
    root, manifest, policy = _fixture(tmp_path)
    conflict = _settlement()
    conflict["agreement"] = False
    conflict["predicted_result"] = "no"
    paths = _write_stream(
        root, "conflicting-settlement-run", "settlement_check", [conflict]
    )
    conflict_manifest = write_run_manifest(
        root, "conflicting-settlement-run", {"read_only": True}
    )
    update_run_manifest(
        root,
        "conflicting-settlement-run",
        status="completed",
        output_paths=paths,
    )

    evidence = measure_daily_gates(root, DATE, [manifest, conflict_manifest], policy)

    assert evidence["settlement_checksum"]["checked"] == 0
    assert evidence["settlement_checksum"]["agreements"] == 0
    assert evidence["settlement_checksum"]["disagreements"] == 0
    assert evidence["settlement_checksum"]["complete"] is False
    assert evidence["settlement_checksum"]["conflicting_market_tickers"] == [
        "SILVER-ONE"
    ]


def test_classified_unusable_settlement_is_complete_but_not_counted(tmp_path):
    excluded = {
        "market_ticker": "SILVER-ONE",
        "usable": False,
        "agreement": None,
        "exclusion_reasons": ["exact_history_interval_incomplete"],
        "audit_code_version": "audit-v2",
    }
    root, manifest, policy = _fixture(
        tmp_path,
        stream_overrides={"settlement_check": [excluded]},
    )

    evidence = measure_daily_gates(root, DATE, [manifest], policy)
    merged = merge_daily_gate_evidence(_collector_report(), evidence)

    checksum = evidence["settlement_checksum"]
    assert checksum["expected"] == 1
    assert checksum["checked"] == 0
    assert checksum["excluded"] == 1
    assert checksum["resolved"] == 1
    assert checksum["complete"] is True
    assert merged["settlement_checksum"]["checked"] == 0
    assert merged["data_gates"]["gates"]["settlement_agreement"]["passed"] is True


def test_evidence_checksum_and_date_binding_are_enforced(tmp_path):
    root, manifest, policy = _fixture(tmp_path)
    evidence = measure_daily_gates(root, DATE, [manifest], policy)
    evidence["coverage"]["pyth_valid_ns"] = 0

    with pytest.raises(DailyGateMeasurementError, match="checksum"):
        merge_daily_gate_evidence(_collector_report(), evidence)

    fresh = measure_daily_gates(root, DATE, [manifest], policy)
    report = _collector_report()
    report["date"] = "2026-08-30"
    with pytest.raises(DailyGateMeasurementError, match="dates differ"):
        merge_daily_gate_evidence(report, fresh)
