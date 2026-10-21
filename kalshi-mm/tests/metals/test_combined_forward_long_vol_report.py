from __future__ import annotations

from src.metals.combined_forward_long_vol_report import (
    combine_forward_long_vol_reports,
)
from src.metals.forward_long_vol_report import SCHEMA_VERSION


def _report(report_id: str, day: str, pnl: str, won: bool) -> dict:
    entry = {
        "candidate_id": f"candidate-{report_id}",
        "decision_ts_ns": 1 if report_id == "one" else 2,
        "utc_day": day,
        "entry_ask": "0.20",
        "charged_external_edge": "0.04",
        "implied_vol_ratio_to_external": 0.7,
        "terminal_pnl": pnl,
        "won": won,
        "markouts": {
            "1s": None,
            "5s": None,
            "30s": None,
            "60s": None,
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "report_id": report_id,
        "policy_version": "fixed-v1",
        "trading_eligible": False,
        "candidate_funnel": {
            "all_rows": 10,
            "evaluated_rows": 9,
            "skipped_rows": 1,
            "screen_eligible_rows": 3,
            "screen_eligible_markets": 2,
            "shadow_policy_eligible_rows": 1,
            "shadow_policy_eligible_markets": 1,
            "selected_first_entries": 1,
            "rejection_reason_rows": {"persistence_not_met": 4},
            "rejection_reason_markets": {"persistence_not_met": 2},
        },
        "selected_entries": [entry],
    }


def test_combiner_pools_frozen_entries_by_day_without_refitting():
    combined = combine_forward_long_vol_reports(
        [
            _report("one", "2026-09-02", "0.80", True),
            _report("two", "2026-09-03", "-0.20", False),
        ]
    )

    assert combined["source_reports"] == 2
    assert combined["candidate_funnel_sums"]["all_rows"] == 20
    assert combined["candidate_funnel_sums"]["rejection_reason_rows"] == {
        "persistence_not_met": 8
    }
    assert combined["primary_results"]["entries"] == 2
    assert combined["primary_results"]["wins"] == 1
    assert combined["primary_results"]["mean_terminal_pnl"] == "0.30"
    assert combined["primary_results"]["total_terminal_pnl"] == "0.60"
    interval = combined["primary_results"]["trading_day_block_95pct_mean_pnl"]
    assert interval is not None
    assert interval[0] <= "-0.20"
    assert interval[1] >= "0.80"
