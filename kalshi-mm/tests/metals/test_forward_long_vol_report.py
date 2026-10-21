from __future__ import annotations

from src.metals.forward_long_vol_report import analyze_forward_candidates
from src.metals.forward_shadow import LONG_VOL_CANDIDATE_SCHEMA


def _candidate(
    candidate_id: str,
    decision_ts_ns: int,
    *,
    policy_eligible: bool,
    yes_bid: str = "0.19",
    yes_ask: str = "0.20",
    fair_yes: str = "0.30",
) -> dict:
    return {
        "schema_version": LONG_VOL_CANDIDATE_SCHEMA,
        "candidate_id": candidate_id,
        "policy_version": "long-vol-forward-shadow-v1",
        "status": "evaluated",
        "market_ticker": "SILVER-A",
        "decision_ts_ns": decision_ts_ns,
        "seconds_remaining": 300,
        "book": {
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": str(1 - float(yes_ask)),
            "no_ask": str(1 - float(yes_bid)),
        },
        "external_fair": {
            "yes_gaussian": fair_yes,
            "yes_empirical_cdf": fair_yes,
        },
        "underdog_side": "yes",
        "underdog_bid": yes_bid,
        "underdog_ask": yes_ask,
        "underdog_fair_conservative": "0.30",
        "taker_fee_per_contract": "0.0112",
        "net_external_edge_after_fee": "0.0888",
        "charged_external_edge": "0.0588",
        "displayed_edge_persistence_ms": "1200",
        "implied_vol_ratio_to_external": 0.8,
        "screen_eligible": True,
        "shadow_policy_eligible": policy_eligible,
        "shadow_policy_rejection_reasons": (
            [] if policy_eligible else ["persistence_not_met"]
        ),
        "execution_eligible": False,
        "trading_eligible": False,
        "long_vol_policy": {"taker_fee_coefficient": "0.07"},
    }


def test_forward_report_uses_one_strict_entry_and_exchange_outcome():
    rows = [
        _candidate("entry", 1_000_000_000, policy_eligible=True),
        _candidate(
            "future-1s",
            2_000_000_000,
            policy_eligible=True,
            yes_bid="0.24",
            yes_ask="0.26",
            fair_yes="0.32",
        ),
        _candidate(
            "future-5s",
            6_000_000_000,
            policy_eligible=False,
            yes_bid="0.22",
            yes_ask="0.24",
            fair_yes="0.31",
        ),
        _candidate(
            "future-30s",
            31_000_000_000,
            policy_eligible=False,
            yes_bid="0.28",
            yes_ask="0.30",
            fair_yes="0.35",
        ),
        _candidate(
            "future-60s",
            61_000_000_000,
            policy_eligible=False,
            yes_bid="0.33",
            yes_ask="0.35",
            fair_yes="0.40",
        ),
    ]
    settlements = [
        {
            "market_ticker": "SILVER-A",
            "result": "yes",
            "expiration_value": "30.123",
            "receive_ts_ns": 100_000_000_000,
        }
    ]

    report = analyze_forward_candidates(rows, settlements)

    assert report["candidate_funnel"]["shadow_policy_eligible_rows"] == 2
    assert report["candidate_funnel"]["shadow_policy_eligible_markets"] == 1
    assert report["candidate_funnel"]["selected_first_entries"] == 1
    assert report["primary_results"]["entries"] == 1
    assert report["primary_results"]["wins"] == 1
    assert report["primary_results"]["total_terminal_pnl"] == "0.7888"
    assert report["primary_results"]["trading_day_block_95pct_mean_pnl"] is None

    selected = report["selected_entries"][0]
    assert selected["candidate_id"] == "entry"
    assert selected["settlement_label_source"] == "kalshi_recorded_result"
    assert selected["markouts"]["1s"]["midpoint_pnl"] == "0.0388"
    assert selected["markouts"]["1s"]["executable_liquidation_pnl"] == "0.0160"
    assert selected["markouts"]["1s"]["external_fair_change"] == "0.02"


def test_forward_report_does_not_score_screen_only_rows():
    rows = [_candidate("screen-only", 1_000_000_000, policy_eligible=False)]

    report = analyze_forward_candidates(rows, [])

    assert report["candidate_funnel"]["screen_eligible_rows"] == 1
    assert report["candidate_funnel"]["shadow_policy_eligible_rows"] == 0
    assert report["primary_results"]["entries"] == 0
    assert report["primary_results"]["mean_terminal_pnl"] is None
