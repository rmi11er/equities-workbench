from __future__ import annotations

import pytest

from src.metals.opportunistic_long_vol import (
    LongVolPolicy,
    _fair_yes,
    evaluate_entry,
    executable_underdog,
    implied_vol_ratio,
    one_contract_taker_fee,
)


def test_executable_underdog_maps_yes_and_no_quotes() -> None:
    assert executable_underdog(yes_bid=0.20, yes_ask=0.22) == ("yes", 0.20, 0.22)
    assert executable_underdog(yes_bid=0.78, yes_ask=0.80) == (
        "no",
        pytest.approx(0.20),
        pytest.approx(0.22),
    )


def test_one_contract_taker_fee_rounds_up_to_centicent() -> None:
    assert one_contract_taker_fee(0.22) == pytest.approx(0.0121)
    assert one_contract_taker_fee(0.50) == pytest.approx(0.0175)
    assert one_contract_taker_fee(0.00) == 0
    assert one_contract_taker_fee(1.00) == 0


def test_entry_requires_fee_and_frozen_edge_hurdle() -> None:
    policy = LongVolPolicy(min_net_external_edge=0.03)
    accepted = evaluate_entry(
        market_ticker="TEST",
        split="test",
        utc_day="2026-01-01",
        decision_ts_ns=1,
        remaining_seconds=300,
        yes_bid=0.20,
        yes_ask=0.22,
        fair_yes=0.27,
        outcome_yes=True,
        policy=policy,
    )
    assert accepted is not None
    assert accepted.net_external_edge == pytest.approx(0.0379)
    assert accepted.terminal_pnl == pytest.approx(0.7679)

    rejected = evaluate_entry(
        market_ticker="TEST",
        split="test",
        utc_day="2026-01-01",
        decision_ts_ns=1,
        remaining_seconds=300,
        yes_bid=0.20,
        yes_ask=0.22,
        fair_yes=0.25,
        outcome_yes=True,
        policy=policy,
    )
    assert rejected is None


def test_no_side_payout_and_implied_vol_mapping() -> None:
    accepted = evaluate_entry(
        market_ticker="TEST",
        split="development",
        utc_day="2026-01-01",
        decision_ts_ns=1,
        remaining_seconds=600,
        yes_bid=0.78,
        yes_ask=0.80,
        fair_yes=0.73,
        outcome_yes=False,
        policy=LongVolPolicy(min_net_external_edge=0.03),
    )
    assert accepted is not None
    assert accepted.side == "no"
    assert accepted.underdog_ask == pytest.approx(0.22)
    assert accepted.terminal_pnl == pytest.approx(0.7679)
    assert implied_vol_ratio(
        external_underdog_fair=0.27,
        market_underdog_price=0.22,
    ) < 1


def test_non_core_price_and_time_are_ineligible() -> None:
    base = dict(
        market_ticker="TEST",
        split="test",
        utc_day="2026-01-01",
        decision_ts_ns=1,
        yes_bid=0.03,
        yes_ask=0.04,
        fair_yes=0.20,
        outcome_yes=False,
        policy=LongVolPolicy(),
    )
    assert evaluate_entry(remaining_seconds=300, **base) is None
    assert evaluate_entry(
        remaining_seconds=60,
        **{**base, "yes_bid": 0.20, "yes_ask": 0.22},
    ) is None


def test_robust_minimum_is_conservative_for_the_selected_side() -> None:
    row = {
        "fair_yes_frozen_tod_ewma_gaussian": 0.30,
        "fair_yes_frozen_empirical_cdf": 0.25,
    }
    assert _fair_yes(
        row,
        "robust_minimum",
        yes_bid=0.20,
        yes_ask=0.22,
    ) == 0.25
    # NO fairs are 0.70 and 0.75, so the conservative NO fair is 0.70,
    # represented as a synthetic YES fair of 0.30.
    assert _fair_yes(
        row,
        "robust_minimum",
        yes_bid=0.78,
        yes_ask=0.80,
    ) == 0.30
