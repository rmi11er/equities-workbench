from copy import deepcopy
from decimal import Decimal

import pytest

from src.metals.maker_edge_metrics import summarize_edge_events
from src.metals.maker_policy import MakerSettings

D = Decimal
MSEC = 1_000_000


def _fair(slow="0.30", fast="0.28"):
    return {"exact_gaussian_yes": slow, "fast_gaussian_yes": fast,
            "fast_trace": {"estimated_index": "66"}}


def _event(kind, ms, sequence, **payload):
    return {"kind": kind, "arm": "manual", "local_monotonic_ts_ns": ms*MSEC,
            "decision_ts_ns": (100000+ms)*MSEC, "input_ingestion_sequence": sequence,
            **payload}


def _decision(action, ms, sequence, **payload):
    return _event("decision", ms, sequence, action=action, market_ticker="M1",
                  side="yes", fair=_fair(), **payload)


def _transition(kind, ms, sequence, quote="q1"):
    return _event("arrival_transition", ms, sequence, transition={
        "event_type": kind, "logical_quote_id": quote, "market_ticker": "M1"})


def _fill(trade, qty, *, fair="0.27", terminal=None, cancel=False, mark=None):
    return {"logical_quote_id": "q1", "market_ticker": "M1", "side": "yes",
        "trade_id": trade, "quantity": qty, "price": "0.20",
        "cost": str(D(qty)*D("0.21")), "placement_fair": _fair(),
        "fair_at_fill": _fair(fair), "cancel_in_flight": cancel,
        "terminal_pnl": terminal, "markouts": {} if mark is None else {"1s": mark}}


def _summary(fills=None):
    return {"arms": {"manual": {"fills": fills or [], "outstanding_orders": {}},
                     "baseline": {"fills": []}},
            "counts": {"manual:touch_without_fill": 7}}


def _events():
    return [
        _decision("wait_persistence", 0, 1, quote_price="0.20"),
        _decision("wait_persistence", 500, 2, quote_price="0.20"),
        _decision("submit", 1000, 3, quote_price="0.20", logical_quote_id="q1"),
        _transition("submit_acknowledged", 1100, 4),
        _transition("order_resting", 1100, 4),
        _decision("retain", 1200, 5, logical_quote_id="q1"),
    ]


def test_modeled_cushion_haircut_and_native_touch_have_explicit_denominators():
    result = summarize_edge_events(_events(), _summary(), MakerSettings())
    metrics = result["modeled_edge"]["submit"]["metrics"]
    assert metrics["slow_fair_minus_price"]["weighted_mean"] == "0.10"
    assert metrics["fast_fair_minus_price"]["weighted_mean"] == "0.08"
    assert metrics["fast_protection_haircut"]["weighted_mean"] == "0.02"
    assert metrics["protected_fair_minus_price"]["weighted_mean"] == "0.08"
    assert metrics["cushion_after_all_charges"]["weighted_mean"] == "0.02"
    assert metrics["slow_fair_minus_price"]["covered_weight"] == "1"
    assert result["modeled_edge"]["retain"]["metrics"]["slow_fair_minus_price"]["weighted_mean"] == "0.10"
    assert result["decision_funnel"]["arrived_quotes"] == 1
    assert result["decision_funnel"]["native_touch_without_fill_count"] == 7
    assert result["independence"]["distinct_observed_markets"] == 1
    assert result["independence"]["effective_independent_sample_size"] is None
    assert result["trading_qualified"] is False


def test_fast_positive_move_cannot_increase_protected_fair():
    events = _events()
    events[2]["fair"] = _fair("0.30", "0.40")
    metrics = summarize_edge_events(events, _summary(), MakerSettings())["modeled_edge"]["submit"]["metrics"]
    assert D(metrics["fast_protection_haircut"]["weighted_mean"]) == 0
    assert metrics["protected_fair_minus_price"]["weighted_mean"] == "0.10"


def test_no_side_fair_uses_complement_and_missing_fast_is_not_zero_edge():
    events = [_decision("submit", 1000, 1, quote_price="0.20", logical_quote_id="q1")]
    events[0].update(side="no", fair=_fair("0.65", "0.70"))
    metrics = summarize_edge_events(events, _summary(), MakerSettings())["modeled_edge"]["submit"]["metrics"]
    assert metrics["slow_fair_minus_price"]["weighted_mean"] == "0.15"
    assert metrics["fast_protection_haircut"]["weighted_mean"] == "0.05"
    events[0]["fair"].pop("fast_gaussian_yes")
    missing = summarize_edge_events(events, _summary(), MakerSettings())["modeled_edge"]["submit"]["metrics"]
    assert missing["slow_fair_minus_price"]["available_observations"] == 1
    assert missing["protected_fair_minus_price"]["weighted_mean"] is None
    assert missing["protected_fair_minus_price"]["covered_weight"] == "0"
    exact = summarize_edge_events(events, _summary(), MakerSettings(require_fast=False))["modeled_edge"]["submit"]["metrics"]
    assert exact["protected_fair_minus_price"]["weighted_mean"] == "0.15"


def test_persistence_episodes_record_observed_bounds_and_right_censoring():
    events = _events()
    events += [
        _event("decision", 1300, 6, action="wait_persistence", market_ticker="M2", side="no", quote_price="0.20", fair=_fair()),
        _event("decision", 1400, 7, action="wait_persistence", market_ticker="M2", side="no", quote_price="0.20", fair=_fair()),
    ]
    result = summarize_edge_events(events, _summary(), MakerSettings())
    first, second = result["persistence_episodes"]["observations"]
    assert first["observed_eligible_span_ms"] == "500"
    assert first["elapsed_to_terminal_or_censor_ms"] == "1000"
    assert first["outcome"] == "submit"
    assert first["right_censored"] is False
    assert second["outcome"] == "last_emitted_event"
    assert second["right_censored"] is True
    assert second["observed_eligible_span_ms"] == "100"
    lifecycle = result["quote_lifecycles"][0]
    assert lifecycle["right_censored"] is True
    assert lifecycle["simulated_resting_duration_ms"] == "300"
    assert lifecycle["submit_to_observed_arrival_ms"] == "100"


def test_skip_gap_and_unobserved_wait_have_separate_episode_outcomes():
    events = [_decision("wait_persistence", 0, 1, quote_price="0.20"),
              _decision("skip", 50, 2, reason="fast_source_stale"),
              _decision("wait_persistence", 100, 3, quote_price="0.20"),
              _event("gap", 150, 4, reason="lost_feed"),
              _decision("risk_rejected", 200, 5, quote_price="0.20", logical_quote_id="q1", rejections=["cap"])]
    result = summarize_edge_events(events, _summary(), MakerSettings())
    assert result["persistence_episodes"]["outcomes"] == {"skip": 1, "gap": 1, "risk_rejected": 1}
    last = result["persistence_episodes"]["observations"][-1]
    assert last["left_censored"] is True
    assert last["elapsed_to_terminal_or_censor_ms"] is None
    assert result["decision_funnel"]["skip_reason_counts"] == {"fast_source_stale": 1}
    assert result["decision_funnel"]["risk_rejection_reason_counts"] == {"cap": 1}


def test_cancel_ack_and_full_fill_end_arrival_based_lifecycle():
    cancelled = _events()+[_event("cancel_request", 1300, 6, logical_quote_id="q1", market_ticker="M1"),
                           _transition("cancel_acknowledged", 1400, 7)]
    order = summarize_edge_events(cancelled, _summary(), MakerSettings())["quote_lifecycles"][0]
    assert order["simulated_resting_duration_ms"] == "300"
    assert order["cancel_request_to_observed_terminal_ms"] == "100"
    assert order["right_censored"] is False
    first, second = _fill("t1", "0.4"), _fill("t2", "0.6", cancel=True)
    filled = _events()+[_event("fill", 1300, 6, fill=first),
                        _event("cancel_request", 1300, 6, logical_quote_id="q1", market_ticker="M1"),
                        _event("fill", 1350, 7, fill=second)]
    result = summarize_edge_events(filled, _summary([first, second]), MakerSettings())
    order = result["quote_lifecycles"][0]
    assert order["terminal_state"] == "full_fill"
    assert order["simulated_resting_duration_ms"] == "250"
    assert order["cancel_request_to_observed_terminal_ms"] == "50"
    assert result["fills"]["cancel_in_flight_quantity"] == "0.6"


def test_post_only_rejection_has_no_invented_resting_duration():
    events = _events()[:3]+[_transition("post_only_rejected", 1100, 4)]
    order = summarize_edge_events(events, _summary(), MakerSettings())["quote_lifecycles"][0]
    assert order["terminal_state"] == "post_only_rejected"
    assert order["simulated_resting_duration_ms"] is None
    assert order["arrival_monotonic_ns"] is None


def test_final_summary_owns_marks_and_partial_fills_collapse_per_quote_horizon():
    first = _fill("t1", "0.4", fair="0.28", terminal="0.316", mark={
        "external_fair_change": "-0.01", "bid_mark_less_entry_cost_per_contract": "-0.03"})
    second = _fill("t2", "0.6", fair="0.26", terminal="0.474", cancel=True, mark={
        "external_fair_change": "-0.03", "bid_mark_less_entry_cost_per_contract": "0.02"})
    events = _events()+[_event("fill", 1300, 6, fill={**first, "markouts": {}}),
                        _event("fill", 1350, 7, fill={**second, "markouts": {}}),
                        _event("markout", 2400, 8, mark={"external_fair_change": "999"})]
    result = summarize_edge_events(events, _summary([first, second]), MakerSettings())["fills"]
    assert result["distinct_quotes"] == 1 and result["fill_legs"] == 2
    horizon = result["horizons"]["1s"]
    fair = horizon["external_fair_change"]
    assert D(fair["weighted_mean"]) == D("-0.022")
    assert fair["available_observations"] == 1
    assert fair["available_fill_legs"] == 2
    assert fair["covered_weight"] == "1.0"
    assert D(horizon["bid_mark_less_entry_cost_per_contract"]["weighted_mean"]) == 0
    assert D(result["external_fair_change_placement_to_fill"]["weighted_mean"]) == D("-0.032")
    assert D(result["terminal_settlement_pnl"]["weighted_mean"]) == D("0.79")
    assert D(result["terminal_settlement_pnl"]["covered_total_pnl_dollars"]) == D("0.79")
    assert result["horizons"]["5s"]["external_fair_change"]["weighted_mean"] is None
    assert result["liquidation_exit_fees_included"] is False


def test_missing_partial_fill_marks_and_unsettled_pnl_keep_quantity_coverage():
    first = _fill("t1", "0.4", terminal="-0.084", mark={"external_fair_change": "0.02"})
    second = _fill("t2", "0.6", fair=None)
    second["markouts"] = {"1s": {"missing": True}}
    result = summarize_edge_events(_events(), _summary([first, second]), MakerSettings())["fills"]
    mark = result["horizons"]["1s"]["external_fair_change"]
    assert mark["eligible_weight"] == "1.0"
    assert mark["covered_weight"] == "0.4"
    assert mark["weighted_mean"] == "0.02"
    assert result["terminal_settlement_pnl"]["covered_weight"] == "0.4"
    assert result["terminal_settlement_pnl"]["covered_total_pnl_dollars"] == "-0.084"
    assert result["fill_slow_fair_less_entry_cost"]["covered_weight"] == "0.4"


def test_no_opportunities_or_fills_returns_null_estimates_and_no_mutation():
    events = [_decision("skip", 0, 1, reason="closed")]
    summary = _summary()
    original = deepcopy((events, summary))
    result = summarize_edge_events(events, summary, MakerSettings())
    assert (events, summary) == original
    assert result["modeled_edge"]["submit"]["metrics"]["slow_fair_minus_price"]["weighted_mean"] is None
    assert result["fills"]["terminal_settlement_pnl"]["covered_total_pnl_dollars"] is None
    empty = summarize_edge_events([], summary, MakerSettings(), arm="manual")
    assert empty["decision_funnel"]["logged_decision_rows"] == 0
    with pytest.raises(ValueError, match="empty events require arm"):
        summarize_edge_events([], summary, MakerSettings())


def test_bad_sequence_mixed_arms_nonfinite_and_duplicate_fill_legs_rejected():
    events = _events()
    events[-1]["input_ingestion_sequence"] = 0
    with pytest.raises(ValueError, match="sequence"):
        summarize_edge_events(events, _summary(), MakerSettings())
    events = _events()
    events[-1]["arm"] = "baseline"
    with pytest.raises(ValueError, match="exactly one"):
        summarize_edge_events(events, _summary(), MakerSettings())
    events = _events()
    events[2]["fair"]["exact_gaussian_yes"] = "NaN"
    with pytest.raises(ValueError, match="finite"):
        summarize_edge_events(events, _summary(), MakerSettings())
    fill = _fill("t1", "0.4")
    with pytest.raises(ValueError, match="duplicate fill legs"):
        summarize_edge_events(_events(), _summary([fill, fill]), MakerSettings())


def test_clock_regression_makes_duration_unavailable_not_negative():
    events = _events()
    events[-1]["local_monotonic_ts_ns"] = 1050*MSEC
    result = summarize_edge_events(events, _summary(), MakerSettings())
    assert result["monotonic_regression_observed"] is True
    assert result["quote_lifecycles"][0]["simulated_resting_duration_ms"] is None
    assert result["persistence_episodes"]["observations"][0]["observed_eligible_span_ms"] is None


def test_horizon_matching_cohort_keeps_initial_cushion_drift_and_model_value_separate():
    first = _fill("t1", "0.4", fair="0.268", mark={
        "side_external_fair": "0.235", "external_fair_change": "-0.033",
        "observation_lag_ms": "25", "bid_mark_less_entry_cost_per_contract": "-0.01"})
    second = _fill("t2", "0.6", fair=None, mark={
        "side_external_fair": "0.30", "external_fair_change": None,
        "observation_lag_ms": "75"})
    result = summarize_edge_events(_events(), _summary([first, second]), MakerSettings())
    horizon = result["fills"]["horizons"]["1s"]
    matching = horizon["matching_cohort"]
    assert D(matching["initial_fill_slow_fair_less_entry_cost"]["weighted_mean"]) == D("0.058")
    assert D(matching["external_fair_change"]["weighted_mean"]) == D("-0.033")
    assert D(matching["model_fair_less_entry_cost_per_contract"]["weighted_mean"]) == D("0.025")
    assert {metric["covered_weight"] for metric in matching.values()} == {"0.4"}
    assert {metric["eligible_weight"] for metric in matching.values()} == {"1.0"}
    assert D(horizon["model_fair_less_entry_cost_per_contract"]["weighted_mean"]) == D("0.064")
    assert D(horizon["observation_lag_ms"]["weighted_mean"]) == 55
    assert horizon["observation_lag_ms"]["covered_weight"] == "1.0"
    assert result["fills"]["terminal_settlement_pnl"]["weighted_mean"] is None
    assert result["actual_fills_observed"] is False


def test_missing_mark_ignores_even_present_numeric_values_and_reports_absence():
    first = _fill("t1", "0.4", mark={"missing": True,
        "side_external_fair": "NaN", "external_fair_change": "0.10",
        "observation_lag_ms": "5", "bid_mark_less_entry_cost_per_contract": "0.10"})
    second = _fill("t2", "0.6")
    horizon = summarize_edge_events(_events(), _summary([first, second]), MakerSettings())["fills"]["horizons"]["1s"]
    assert horizon["missing_mark_fill_legs"] == 1
    assert horizon["absent_mark_fill_legs"] == 1
    for name in ("external_fair_change", "observation_lag_ms", "model_fair_less_entry_cost_per_contract",
                 "bid_mark_less_entry_cost_per_contract"):
        assert horizon[name]["weighted_mean"] is None
        assert horizon[name]["covered_weight"] == "0"
    assert all(metric["weighted_mean"] is None for metric in horizon["matching_cohort"].values())


def test_arrival_fair_is_first_observed_transition_not_later_ack_duplicate():
    events = _events()
    events[3]["fair_at_arrival"] = _fair("0.26", "0.25")
    events[4]["fair_at_arrival"] = _fair("0.29", "0.28")
    result = summarize_edge_events(events, _summary(), MakerSettings())
    order = result["quote_lifecycles"][0]
    assert order["quote_price"] == "0.20"
    assert order["fair_at_submit"]["slow_side_fair"] == "0.30"
    assert order["fair_at_arrival"] == {"slow_side_fair": "0.26", "fast_side_fair": "0.25"}
    assert order["slow_fair_change_submit_to_arrival"] == "-0.04"
    arrival = result["modeled_edge"]["arrival"]
    assert arrival["logged_rows"] == 1
    assert arrival["metrics"]["slow_fair_minus_price"]["weighted_mean"] == "0.06"
    missing = summarize_edge_events(_events(), _summary(), MakerSettings())["quote_lifecycles"][0]
    assert missing["fair_at_arrival"]["slow_side_fair"] is None
    assert missing["slow_fair_change_submit_to_arrival"] is None


@pytest.mark.parametrize("field,value,reason", [
    ("side_external_fair", "1.01", "probability"),
    ("observation_lag_ms", "-1", "nonnegative"),
])
def test_invalid_horizon_probability_and_lag_are_not_reported_as_metrics(field, value, reason):
    fill = _fill("t1", "1", mark={field: value})
    with pytest.raises(ValueError, match=reason):
        summarize_edge_events(_events(), _summary([fill]), MakerSettings())


def test_partial_legs_cannot_share_quote_identity_across_different_markets():
    first, second = _fill("t1", "0.4"), _fill("t2", "0.6")
    second["market_ticker"] = "M2"
    with pytest.raises(ValueError, match="one market and side"):
        summarize_edge_events(_events(), _summary([first, second]), MakerSettings())


def test_control_revisions_flag_manual_parameter_drift_not_baseline_regime():
    controls = [_event("control", ms, seq, control={"version": seq,
        "volatility_multiplier": multiplier, "paused": False}, error=error)
        for ms, seq, multiplier, error in ((0, 1, "1", ""), (100, 2, "1.5", ""),
                                          (200, 3, "NaN", "control_invalid"))]
    result = summarize_edge_events(controls, _summary(), MakerSettings())
    assert result["manual_regime_changed"] is True
    assert result["controls"]["accepted_control_multipliers"] == ["1", "1.5"]
    assert [r["version"] for r in result["controls"]["observations"]] == [1, 2, 3]
    assert result["controls"]["observations"][-1]["accepted"] is False
    assert result["controls"]["baseline_multiplier_fixed_at_one"] is False
    baseline = summarize_edge_events([{**event, "arm": "baseline"} for event in controls], _summary(), MakerSettings())
    assert baseline["manual_regime_changed"] is False
    assert baseline["controls"]["baseline_multiplier_fixed_at_one"] is True
    assert baseline["controls"]["operator_pause_and_health_shared_across_arms"] is True


def test_partial_legs_and_quotes_have_separate_denominators_across_markets():
    first = _fill("t1", "0.4", mark={"external_fair_change": "-0.01"})
    second = _fill("t2", "0.6", mark={"external_fair_change": "-0.03"})
    third = {**_fill("t3", "1", mark={"external_fair_change": "0.02"}),
             "logical_quote_id": "q2", "market_ticker": "M2"}
    result = summarize_edge_events(_events(), _summary([first, second, third]), MakerSettings())
    metric = result["fills"]["horizons"]["1s"]["external_fair_change"]
    assert D(metric["weighted_mean"]) == D("-0.001")
    assert metric["covered_distinct_quotes"] == 2
    assert metric["available_fill_legs"] == 3
    assert metric["available_observations"] == 2
    assert result["independence"]["distinct_filled_markets"] == 2
    assert result["independence"]["effective_independent_sample_size"] is None
