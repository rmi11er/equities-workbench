import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.metals import daily_validation
from src.metals.daily_validation import (
    DailyValidationError,
    freeze_daily_validation,
    synchronized_day_progress,
    validated_research_dates,
    verify_daily_snapshot,
)
from src.metals.storage import update_run_manifest, write_run_manifest


@pytest.fixture(autouse=True)
def _recompute_synthetic_evidence(monkeypatch):
    def recompute(root, date, manifests, policy_path):
        policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
        for path in Path(root).glob(
            f"derived/daily_gate_evidence/date={date}/evidence=*/evidence.json"
        ):
            evidence = json.loads(path.read_text(encoding="utf-8"))
            if evidence.get("inputs", {}).get("source") == policy["source"]:
                return evidence
        raise AssertionError("synthetic evidence fixture is missing")

    monkeypatch.setattr(daily_validation, "measure_daily_gates", recompute)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _passing_quality(date="2026-08-29"):
    return {
        "date": date,
        "run_id": "unified-run",
        "expected_windows": 1000,
        "settlement_checksum": {
            "checked": 1000,
            "agreements": 1000,
            "disagreements": 0,
        },
        "strategy_decisions_on_invalid_book": 0,
        "coverage": {
            "kalshi_valid_fraction": 0.999,
            "pyth_valid_fraction": 0.995,
            "cme_valid_fraction": 0.995,
        },
        "missing_intervals": [],
        "clock_error_within_edge_half_life": True,
        "causal_ordering_checked": True,
        "metadata_rules_validated": True,
        "fee_registry_validated": True,
        "incentive_registry_validated": True,
    }


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _completed_manifest(output: Path):
    config = {"read_only": True}
    return {
        "run_id": "unified-run",
        "status": "completed",
        "config": config,
        "config_sha256": _canonical_sha256(config),
        "git": {"commit": "abc", "dirty": False},
        "output_paths": [str(output)],
    }


def _measured_inputs(root: Path, report, raw: Path, *, suffix="one"):
    report = copy.deepcopy(report)
    date = report["date"]
    source_quality = root / "derived" / "collector_quality" / f"{suffix}.json"
    policy_path = root / "policies" / "synthetic" / f"{suffix}.json"
    _write_json(source_quality, {"date": date, "run_id": report["run_id"]})
    _write_json(policy_path, {"source": suffix})
    source_manifest = write_run_manifest(
        root,
        f"source-{suffix}",
        {"collector": {"component": "synchronized_metals_live"}},
    )
    update_run_manifest(
        root,
        f"source-{suffix}",
        status="completed",
        output_paths=(raw, source_quality, policy_path),
    )

    evidence = {
        "evidence_version": 1,
        "date": date,
        "inputs": {"source": suffix},
        "policy": {"path": str(policy_path.relative_to(root))},
    }
    evidence["evidence_sha256"] = _canonical_sha256(evidence)
    report["daily_gate_measurement"] = evidence
    quality_hash = _canonical_sha256(report)
    report["daily_quality_sha256"] = quality_hash
    evidence_path = (
        root
        / "derived"
        / "daily_gate_evidence"
        / f"date={date}"
        / f"evidence={evidence['evidence_sha256']}"
        / "evidence.json"
    )
    quality_path = (
        root
        / "derived"
        / "daily_quality"
        / f"date={date}"
        / f"report={quality_hash}"
        / "report.json"
    )
    _write_json(evidence_path, evidence)
    _write_json(quality_path, report)
    measurement_run_id = f"daily-gates-{quality_hash}"
    measurement_manifest = write_run_manifest(
        root,
        measurement_run_id,
        {
            "operation": "measure_daily_gates",
            "read_only_sources": True,
            "date": date,
            "evidence_sha256": evidence["evidence_sha256"],
            "daily_quality_sha256": quality_hash,
            "source_quality_report": str(source_quality.relative_to(root)),
            "source_manifests": [str(source_manifest.relative_to(root))],
        },
    )
    update_run_manifest(
        root,
        measurement_run_id,
        status="completed",
        output_paths=(evidence_path, quality_path),
    )
    return quality_path, [source_manifest, measurement_manifest]


def test_freeze_daily_validation_is_immutable_and_idempotent(tmp_path):
    root = tmp_path / "metals"
    raw = root / "raw" / "events.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text("event\n", encoding="utf-8")
    quality, manifests = _measured_inputs(root, _passing_quality(), raw)

    first = freeze_daily_validation(root, "2026-08-29", quality, manifests)
    second = freeze_daily_validation(root, "2026-08-29", quality, manifests)
    snapshot = json.loads(first.read_text(encoding="utf-8"))

    assert first == second
    assert snapshot["research_eligible"] is True
    assert snapshot["failed_gates"] == []
    assert {item["path"] for item in snapshot["inputs"]} >= {
        "raw/events.jsonl",
        str(quality.relative_to(root)),
        str(manifests[0].relative_to(root)),
        str(manifests[1].relative_to(root)),
    }
    assert validated_research_dates(root) == ["2026-08-29"]
    assert (
        verify_daily_snapshot(root, first)["snapshot_sha256"]
        == snapshot["snapshot_sha256"]
    )
    assert synchronized_day_progress(root) == {
        "passed": False,
        "day_gate_passed": False,
        "complete_days": 1,
        "minimum_complete_days": 10,
        "dates": ["2026-08-29"],
        "settlement_gate_passed": True,
        "settlement_evidence_days": 1,
        "minimum_settlements": 1000,
        "settlements_checked": 1000,
        "settlement_agreements": 1000,
        "settlement_disagreements": 0,
        "expected_windows": 1000,
    }


def test_failed_gate_is_frozen_but_not_research_eligible(tmp_path):
    root = tmp_path / "metals"
    raw = root / "raw" / "events.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text("event\n", encoding="utf-8")
    report = _passing_quality()
    report["coverage"]["pyth_valid_fraction"] = 0.90
    quality, manifests = _measured_inputs(root, report, raw)

    path = freeze_daily_validation(root, "2026-08-29", quality, manifests)
    snapshot = json.loads(path.read_text(encoding="utf-8"))

    assert snapshot["research_eligible"] is False
    assert "pyth_coverage" in snapshot["failed_gates"]
    assert validated_research_dates(root) == []


def test_incomplete_manifest_and_escaping_output_fail_closed(tmp_path):
    root = tmp_path / "metals"
    root.mkdir()
    quality = root / "quality.json"
    manifest = root / "manifest.json"
    _write_json(quality, _passing_quality())
    incomplete = _completed_manifest(root / "missing.jsonl")
    incomplete["status"] = "failed"
    _write_json(manifest, incomplete)

    with pytest.raises(DailyValidationError, match="not completed"):
        freeze_daily_validation(root, "2026-08-29", quality, [manifest])

    outside = tmp_path / "outside.jsonl"
    outside.write_text("event\n", encoding="utf-8")
    _write_json(manifest, _completed_manifest(outside))
    with pytest.raises(DailyValidationError, match="escapes storage root"):
        freeze_daily_validation(root, "2026-08-29", quality, [manifest])


def test_modified_frozen_input_is_detected(tmp_path):
    root = tmp_path / "metals"
    raw = root / "raw" / "events.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text("event\n", encoding="utf-8")
    quality, manifests = _measured_inputs(root, _passing_quality(), raw)
    snapshot = freeze_daily_validation(root, "2026-08-29", quality, manifests)

    raw.write_text("changed\n", encoding="utf-8")

    with pytest.raises(DailyValidationError, match="checksum mismatch"):
        verify_daily_snapshot(root, snapshot)


def test_hand_authored_passing_report_cannot_enter_readiness(tmp_path):
    root = tmp_path / "metals"
    raw = root / "raw" / "events.jsonl"
    quality = root / "quality.json"
    manifest = root / "manifest.json"
    raw.parent.mkdir(parents=True)
    raw.write_text("event\n", encoding="utf-8")
    _write_json(quality, _passing_quality())
    _write_json(manifest, _completed_manifest(raw))

    with pytest.raises(DailyValidationError, match="verified daily-gate measurement"):
        freeze_daily_validation(root, "2026-08-29", quality, [manifest])


def test_snapshot_requires_every_measurement_source_manifest(tmp_path):
    root = tmp_path / "metals"
    raw = root / "raw" / "events.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text("event\n", encoding="utf-8")
    quality, manifests = _measured_inputs(root, _passing_quality(), raw)

    with pytest.raises(DailyValidationError, match="source manifests"):
        freeze_daily_validation(root, "2026-08-29", quality, manifests[1:])


def test_snapshot_recomputes_and_rejects_mismatched_source_evidence(
    monkeypatch, tmp_path
):
    root = tmp_path / "metals"
    raw = root / "raw" / "events.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text("event\n", encoding="utf-8")
    quality, manifests = _measured_inputs(root, _passing_quality(), raw)
    monkeypatch.setattr(
        daily_validation,
        "measure_daily_gates",
        lambda *args, **kwargs: {"evidence_version": 1},
    )

    with pytest.raises(DailyValidationError, match="does not match recomputed"):
        freeze_daily_validation(root, "2026-08-29", quality, manifests)


def test_readiness_requires_both_ten_days_and_one_thousand_settlements(tmp_path):
    root = tmp_path / "metals"
    for day in range(1, 11):
        date = "2026-08-{:02d}".format(day)
        run_id = "run-{:02d}".format(day)
        raw = root / "raw" / (run_id + ".jsonl")
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text("event\n", encoding="utf-8")
        report = _passing_quality(date)
        report["run_id"] = run_id
        report["expected_windows"] = 100
        report["observed_windows"] = 100
        report["settlement_checksum"] = {
            "checked": 100,
            "agreements": 100,
            "disagreements": 0,
        }
        quality, manifests = _measured_inputs(root, report, raw, suffix=run_id)
        freeze_daily_validation(root, date, quality, manifests)

    progress = synchronized_day_progress(root)

    assert progress["passed"] is True
    assert progress["day_gate_passed"] is True
    assert progress["settlement_gate_passed"] is True
    assert progress["complete_days"] == 10
    assert progress["settlements_checked"] == 1000


def test_duplicate_passing_snapshots_for_one_day_do_not_double_count(tmp_path):
    root = tmp_path / "metals"
    date = "2026-08-29"
    for suffix, checked in (("a", 1000), ("b", 1200)):
        raw = root / "raw" / (suffix + ".jsonl")
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(suffix + "\n", encoding="utf-8")
        report = _passing_quality(date)
        report["run_id"] = suffix
        report["expected_windows"] = checked
        report["observed_windows"] = checked
        report["settlement_checksum"] = {
            "checked": checked,
            "agreements": checked,
            "disagreements": 0,
        }
        quality, manifests = _measured_inputs(root, report, raw, suffix=suffix)
        freeze_daily_validation(root, date, quality, manifests)

    progress = synchronized_day_progress(
        root, minimum_complete_days=1, minimum_settlements=1
    )

    assert progress["complete_days"] == 1
    assert progress["settlements_checked"] == 1200
