import pytest

from src.metals.quality import QualityTracker, evaluate_data_gates


def test_quality_report_fails_closed_when_sources_are_unmeasured():
    report = QualityTracker().report("2026-08-29", "run-1")

    assert report["coverage"]["kalshi_valid_fraction"] is None
    assert report["data_gates"]["passed"] is False
    assert set(report["data_gates"]["failed_gates"]) >= {
        "settlement_agreement",
        "kalshi_book_coverage",
        "pyth_coverage",
        "cme_coverage",
        "clock_health",
    }


def test_quality_report_passes_only_with_all_specified_data_gates():
    quality = QualityTracker()
    quality.record("settlements_checked", count=1000)
    quality.record("settlement_agreements", count=1000)
    quality.record_source_coverage("kalshi", valid_ns=999, total_ns=1000)
    quality.record_source_coverage("pyth", valid_ns=995, total_ns=1000)
    quality.record_source_coverage("cme", valid_ns=995, total_ns=1000)
    quality.record_clock_offset(-250)
    quality.record("clock_error_within_edge_half_life")
    quality.record("metadata_rules_validated")
    quality.record("fee_registry_validated")
    quality.record("incentive_registry_validated")

    report = quality.report("2026-08-29", "run-2")

    assert report["data_gates"]["passed"] is True
    assert report["clock_offset_abs_max_ns"] == 250


def test_unclassified_missing_interval_and_rule_change_fail_gates():
    report = {
        "settlement_checksum": {"checked": 1, "agreements": 1, "disagreements": 0},
        "strategy_decisions_on_invalid_book": 0,
        "coverage": {
            "kalshi_valid_fraction": 1.0,
            "pyth_valid_fraction": 1.0,
            "cme_valid_fraction": 1.0,
        },
        "missing_intervals": [{"reason": ""}],
        "clock_error_within_edge_half_life": True,
        "unresolved_metadata_change": True,
        "fee_changes": 1,
        "incentive_changes": 0,
        "metadata_rules_validated": True,
        "fee_registry_validated": True,
        "incentive_registry_validated": True,
    }

    result = evaluate_data_gates(report)

    assert result["passed"] is False
    assert "missing_intervals_classified" in result["failed_gates"]
    assert "rules_unchanged" in result["failed_gates"]


def test_known_gap_cannot_pass_without_a_classified_interval():
    quality = QualityTracker()
    quality.record("cme_reconnects")

    report = quality.report("2026-08-29", "gap-run")

    gate = report["data_gates"]["gates"]["missing_intervals_classified"]
    assert gate["passed"] is False
    assert gate["observed"]["known_gap_events"] == 1


def test_invalid_coverage_and_empty_exclusion_reason_are_rejected():
    quality = QualityTracker()
    with pytest.raises(ValueError):
        quality.record_source_coverage("kalshi", valid_ns=2, total_ns=1)
    with pytest.raises(ValueError):
        quality.record_exclusion("ticker", "")
