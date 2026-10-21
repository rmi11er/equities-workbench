"""Descriptive accessible-edge measurements from an unchanged maker replay.

These are observed decision episodes and declared simulated fills, never actual
queue measurements. Modeled value, changes in fair value, liquidation marks and
terminal settlement P&L have separate denominators and are not interchangeable.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from typing import Any

from .maker_policy import MakerSettings

D = Decimal
ZERO = D(0)
ONE = D(1)
MSEC = D(1_000_000)
SCHEMA_VERSION = "maker-accessible-edge-metrics/v1"


def _number(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("metric value cannot be boolean")
    try:
        number = D(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("invalid metric number") from error
    if not number.is_finite():
        raise ValueError("metric number must be finite")
    return number


def _text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _side_fair(fair: dict | None, side: str, *, fast: bool = False) -> Decimal | None:
    if fair is None:
        return None
    if side not in {"yes", "no"}:
        raise ValueError("metric side must be yes or no")
    key = "fast_side_fair" if fast else "side_fair"
    if fair.get(key) is not None:
        value = _number(fair[key])
    else:
        value = _number(fair.get("fast_gaussian_yes" if fast else "exact_gaussian_yes"))
        if value is not None and side == "no":
            value = ONE - value
    if value is not None and not ZERO <= value <= ONE:
        raise ValueError("metric fair must be a probability")
    return value


def _weighted(rows: list[tuple[str, str, Decimal, Decimal | None]], *, unit: str) -> dict:
    known = [(quote, market, quantity, value) for quote, market, quantity, value in rows if value is not None]
    total_quantity = sum((r[2] for r in rows), ZERO)
    covered = sum((r[2] for r in known), ZERO)
    return {
        "unit": unit, "input_observations": len(rows), "available_observations": len(known),
        "eligible_weight": _text(total_quantity), "covered_weight": _text(covered),
        "covered_distinct_observation_ids": len({r[0] for r in known}),
        "covered_distinct_markets": len({r[1] for r in known}),
        "weighted_mean": _text(sum((q * value for _, _, q, value in known), ZERO) / covered) if covered else None,
        "min": _text(min((r[3] for r in known), default=None)),
        "max": _text(max((r[3] for r in known), default=None)),
    }


def _edge_values(fair: dict | None, side: str, price: Decimal | None, settings: MakerSettings) -> dict:
    slow, fast = _side_fair(fair, side), _side_fair(fair, side, fast=True)
    if not settings.require_fast and fast is None:
        fast = slow
    protected = min(slow, fast) if slow is not None and fast is not None else None
    charges = (settings.uncertainty_charge + settings.adverse_selection_charge
               + settings.maker_fee_per_contract + settings.required_edge)
    difference = lambda value: None if value is None or price is None else value - price
    protected_edge = difference(protected)
    return {"slow_fair_minus_price": difference(slow), "fast_fair_minus_price": difference(fast),
            "fast_protection_haircut": None if protected is None else slow - protected,
            "protected_fair_minus_price": protected_edge,
            "cushion_after_all_charges": None if protected_edge is None else protected_edge - charges}


def summarize_edge_events(events: list[dict], summary: dict, settings: MakerSettings, *, arm: str | None = None) -> dict:
    """Summarize one verified arm; final summary, not fill emissions, owns marks.

    ``arm`` is optional when nonempty events identify it uniquely. Emitted rows
    are throttled: their counts/means are neither time-weighted nor independent
    opportunities. Equal input sequences are expected for multiple emissions.
    """
    event_arms = {row.get("arm") for row in events}
    if arm is None and len(event_arms) == 1:
        arm = next(iter(event_arms))
    if arm not in summary.get("arms", {}) or event_arms - {arm}:
        raise ValueError("provide exactly one known arm; empty events require arm")
    prior = -1
    for row in events:
        sequence = row.get("input_ingestion_sequence")
        mono = row.get("local_monotonic_ts_ns")
        if type(sequence) is not int or sequence < 0 or sequence < prior or type(mono) is not int or mono < 0:
            raise ValueError("invalid maker event sequence or monotonic timestamp")
        prior = sequence
    clock_regressed = any(b["local_monotonic_ts_ns"] < a["local_monotonic_ts_ns"] for a, b in zip(events, events[1:]))
    end_mono = events[-1]["local_monotonic_ts_ns"] if events else None
    duration = lambda start, end: None if clock_regressed or start is None or end is None or end < start else _text(D(end-start)/MSEC)
    decisions = [row for row in events if row["kind"] == "decision"]
    actions = Counter(row.get("action") for row in decisions)
    skips = Counter(row.get("reason") for row in decisions if row.get("action") == "skip")
    risk_rejections = Counter(reason for row in decisions if row.get("action") == "risk_rejected" for reason in row.get("rejections", []))
    orders: dict[str, dict] = {}
    episodes: list[dict] = []
    open_episodes: dict[str, dict] = {}
    stages: dict[str, list[dict]] = defaultdict(list)

    def fair_trace(fair, side):
        return {"slow_side_fair": _text(_side_fair(fair, side)),
                "fast_side_fair": _text(_side_fair(fair, side, fast=True))}

    def end_episode(ticker: str, now: int, outcome: str, censored: bool = False) -> None:
        episode = open_episodes.pop(ticker, None)
        if episode is None:
            return
        episode.update(outcome=outcome, end_monotonic_ns=now, right_censored=censored,
                       observed_eligible_span_ms=duration(episode["first_wait_monotonic_ns"], episode["last_wait_monotonic_ns"]),
                       elapsed_to_terminal_or_censor_ms=duration(episode["first_wait_monotonic_ns"], now))
        episodes.append(episode)

    for index, row in enumerate(events):
        kind, now = row["kind"], row["local_monotonic_ts_ns"]
        ticker, quote_id = row.get("market_ticker"), row.get("logical_quote_id")
        if kind in {"control", "gap", "cancel_request"}:
            for market in list(open_episodes):
                if ticker is None or ticker == market:
                    end_episode(market, now, kind)
        if kind == "decision":
            action, side = row.get("action"), row.get("side")
            if action == "wait_persistence":
                if ticker in open_episodes and open_episodes[ticker]["side"] != side:
                    end_episode(ticker, now, "side_changed")
                episode = open_episodes.setdefault(ticker, {"market_ticker": ticker, "side": side,
                    "first_wait_monotonic_ns": now, "last_wait_monotonic_ns": now,
                    "first_input_sequence": row["input_ingestion_sequence"], "wait_rows": 0, "left_censored": False})
                episode["last_wait_monotonic_ns"] = now
                episode["wait_rows"] += 1
            elif action in {"submit", "skip", "risk_rejected"}:
                if action in {"submit", "risk_rejected"} and ticker not in open_episodes:
                    open_episodes[ticker] = {"market_ticker": ticker, "side": side,
                        "first_wait_monotonic_ns": None, "last_wait_monotonic_ns": None,
                        "first_input_sequence": None, "wait_rows": 0, "left_censored": True}
                end_episode(ticker, now, action)
            if action == "submit":
                if not quote_id or quote_id in orders:
                    raise ValueError("submitted quote identity missing or duplicated")
                orders[quote_id] = {"logical_quote_id": quote_id, "market_ticker": ticker, "side": side,
                    "quote_price": _text(_number(row.get("quote_price"))), "submit_monotonic_ns": now,
                    "fair_at_submit": fair_trace(row.get("fair"), side),
                    "fair_at_arrival": fair_trace(None, side),
                    "slow_fair_change_submit_to_arrival": None,
                    "arrival_monotonic_ns": None, "cancel_request_monotonic_ns": None,
                    "end_monotonic_ns": None, "terminal_state": None, "filled_quantity": ZERO}
            if action in {"wait_persistence", "submit", "risk_rejected", "retain"}:
                price = _number(row.get("quote_price", orders.get(quote_id, {}).get("quote_price")))
                stages[action].append({"id": quote_id or f"emission:{index}", "market": ticker,
                    "metrics": _edge_values(row.get("fair"), side, price, settings)})
        if kind == "arrival_transition":
            transition = row["transition"]
            order = orders.get(transition["logical_quote_id"])
            if order is not None:
                state = transition["event_type"]
                if state in {"submit_acknowledged", "order_resting"} and order["arrival_monotonic_ns"] is None:
                    order["arrival_monotonic_ns"] = now
                    fair = row.get("fair_at_arrival")
                    order["fair_at_arrival"] = fair_trace(fair, order["side"])
                    initial = _number(order["fair_at_submit"]["slow_side_fair"])
                    current = _number(order["fair_at_arrival"]["slow_side_fair"])
                    order["slow_fair_change_submit_to_arrival"] = _text(None if initial is None or current is None else current-initial)
                    stages["arrival"].append({"id": transition["logical_quote_id"], "market": order["market_ticker"],
                        "metrics": _edge_values(fair, order["side"], _number(order["quote_price"]), settings)})
                if state in {"cancel_acknowledged", "post_only_rejected"}:
                    order.update(end_monotonic_ns=now, terminal_state=state)
        if kind == "cancel_request" and quote_id in orders:
            if orders[quote_id]["cancel_request_monotonic_ns"] is None:
                orders[quote_id]["cancel_request_monotonic_ns"] = now
        if kind == "fill":
            fill = row["fill"]
            order = orders.get(fill["logical_quote_id"])
            if order is not None:
                order["filled_quantity"] += _number(fill["quantity"])
                if order["filled_quantity"] >= ONE:
                    order.update(end_monotonic_ns=now, terminal_state="full_fill")
    for ticker in list(open_episodes):
        end_episode(ticker, end_mono, "last_emitted_event", censored=True)
    for order in orders.values():
        censored = order["end_monotonic_ns"] is None
        endpoint = end_mono if censored else order["end_monotonic_ns"]
        order.update(right_censored=censored, observed_end_monotonic_ns=endpoint,
            simulated_resting_duration_ms=duration(order["arrival_monotonic_ns"], endpoint),
            submit_to_observed_arrival_ms=duration(order["submit_monotonic_ns"], order["arrival_monotonic_ns"]),
            cancel_request_to_observed_terminal_ms=duration(order["cancel_request_monotonic_ns"], endpoint),
            filled_quantity=_text(order["filled_quantity"]))

    unit = "probability_dollars_per_contract"
    metric_names = list(_edge_values(None, "yes", None, settings))
    edge = {action: {"logged_rows": len(stages[action]),
        "weight_basis": "one_per_logged_row_not_independent_opportunities", "metrics": {
        metric: _weighted([(r["id"], r["market"], ONE, r["metrics"][metric]) for r in stages[action]], unit=unit)
        for metric in metric_names}} for action in ("wait_persistence", "submit", "arrival", "risk_rejected", "retain")}
    fill_rows = summary["arms"][arm].get("fills", [])
    fill_keys = [(f["logical_quote_id"], f["trade_id"]) for f in fill_rows]
    if len(set(fill_keys)) != len(fill_keys):
        raise ValueError("final summary has duplicate fill legs")
    groups: dict[str, list[dict]] = defaultdict(list)
    for fill in fill_rows:
        quantity = _number(fill["quantity"])
        if quantity is None or quantity <= ZERO:
            raise ValueError("fill quantity must be positive")
        groups[fill["logical_quote_id"]].append(fill)
    if any(len({(f["market_ticker"], f["side"]) for f in fills}) != 1 for fills in groups.values()):
        raise ValueError("fill quote must have one market and side")

    def fill_metric(function, metric_unit=unit) -> dict:
        # Aggregate partial legs once per logical quote before the cross-quote
        # weighted mean. Missing legs never borrow another leg's mark or horizon.
        collapsed = []
        eligible_quantity = ZERO
        available_legs = 0
        for quote, fills in groups.items():
            weighted, covered = ZERO, ZERO
            for fill in fills:
                quantity, value = _number(fill["quantity"]), function(fill)
                eligible_quantity += quantity
                if value is not None:
                    weighted += quantity * value
                    covered += quantity
                    available_legs += 1
            collapsed.append((quote, fills[0]["market_ticker"], covered, weighted/covered if covered else None))
        result = _weighted(collapsed, unit=metric_unit)
        result.update(eligible_weight=_text(eligible_quantity), total_fill_legs=len(fill_rows),
                      available_fill_legs=available_legs, weight_basis="covered_filled_contracts",
                      covered_distinct_quotes=result["covered_distinct_observation_ids"],
                      observation_basis="one_quantity_weighted_observation_per_quote")
        return result

    def at_fill(fill, field):
        fair = _side_fair(fill.get(field), fill["side"])
        return None if fair is None else fair - _number(fill["cost"])/_number(fill["quantity"])

    def fair_change(fill):
        placement = _side_fair(fill.get("placement_fair"), fill["side"])
        current = _side_fair(fill.get("fair_at_fill"), fill["side"])
        return None if placement is None or current is None else current-placement

    terminal = fill_metric(lambda fill: None if fill.get("terminal_pnl") is None else _number(fill["terminal_pnl"])/_number(fill["quantity"]), "settlement_pnl_dollars_per_contract")
    terminal["covered_total_pnl_dollars"] = _text(sum((_number(f["terminal_pnl"]) for f in fill_rows if f.get("terminal_pnl") is not None), ZERO)) if terminal["available_fill_legs"] else None
    def mark_value(fill, horizon, field):
        mark = fill.get("markouts", {}).get(horizon)
        if mark is None or mark.get("missing") is True:
            return None
        value = _number(mark.get(field))
        if field == "side_external_fair" and value is not None and not ZERO <= value <= ONE:
            raise ValueError("horizon fair must be a probability")
        if field == "observation_lag_ms" and value is not None and value < ZERO:
            raise ValueError("observation lag must be nonnegative")
        return value

    def horizon_value(fill, horizon, metric, *, matched=False):
        current = mark_value(fill, horizon, "side_external_fair")
        initial = _side_fair(fill.get("fair_at_fill"), fill["side"])
        if current is None or (matched and initial is None):
            return None
        cost = _number(fill["cost"])/_number(fill["quantity"])
        if metric == "initial_fill_slow_fair_less_entry_cost":
            return initial-cost
        if metric == "external_fair_change":
            return current-initial
        return current-cost

    horizons = {}
    for horizon in ("1s", "5s", "30s", "60s"):
        horizons[horizon] = {
            "external_fair_change": fill_metric(lambda f: mark_value(f, horizon, "external_fair_change"), "fair_change_probability_dollars"),
            "bid_mark_less_entry_cost_per_contract": fill_metric(lambda f: mark_value(f, horizon, "bid_mark_less_entry_cost_per_contract"), "bid_mark_less_entry_cost_dollars_per_contract"),
            "model_fair_less_entry_cost_per_contract": fill_metric(lambda f: horizon_value(f, horizon, "model")),
            "observation_lag_ms": fill_metric(lambda f: mark_value(f, horizon, "observation_lag_ms"), "milliseconds_after_nominal_horizon"),
            "missing_mark_fill_legs": sum(f.get("markouts", {}).get(horizon, {}).get("missing") is True for f in fill_rows),
            "absent_mark_fill_legs": sum(horizon not in f.get("markouts", {}) for f in fill_rows),
            "matching_cohort": {metric: fill_metric(lambda f, m=metric: horizon_value(f, horizon, m, matched=True),
                "fair_change_probability_dollars" if metric == "external_fair_change" else unit)
                for metric in ("initial_fill_slow_fair_less_entry_cost", "external_fair_change", "model_fair_less_entry_cost_per_contract")},
            "fair_basis": "exact_gaussian_side_probability_not_independently_calibrated",
        }
    controls, accepted_multipliers, fair_multipliers = [], set(), set()
    for row in events:
        if row["kind"] == "control":
            control = row.get("control") or {}
            accepted = not bool(row.get("error"))
            multiplier = control.get("volatility_multiplier")
            if accepted and multiplier is not None:
                accepted_multipliers.add(_number(multiplier))
            controls.append({"input_ingestion_sequence": row["input_ingestion_sequence"],
                "local_monotonic_ts_ns": row["local_monotonic_ts_ns"],
                "version": control.get("version"), "effective_ts_ns": control.get("effective_ts_ns"),
                "expires_ts_ns": control.get("expires_ts_ns"), "paused": control.get("paused"),
                "volatility_multiplier": None if multiplier is None else str(multiplier),
                "accepted": accepted, "error": row.get("error")})
        for field in ("fair", "fair_at_arrival"):
            multiplier = (row.get(field) or {}).get("volatility_multiplier")
            if multiplier is not None:
                fair_multipliers.add(_number(multiplier))
    filled_markets = {f["market_ticker"] for f in fill_rows}
    submitted_markets = {o["market_ticker"] for o in orders.values()}
    observed_markets = {r["market_ticker"] for r in decisions}
    return {"schema_version": SCHEMA_VERSION, "arm": arm,
        "decision_funnel": {"logged_decision_rows": len(decisions), "action_counts": dict(actions),
            "skip_reason_counts": dict(skips), "risk_rejection_reason_counts": dict(risk_rejections),
            "rows_with_slow_fair": sum((r.get("fair") or {}).get("exact_gaussian_yes") is not None for r in decisions),
            "rows_with_fast_trace": sum(bool((r.get("fair") or {}).get("fast_trace")) for r in decisions),
            "submitted_quotes": len(orders), "arrived_quotes": sum(o["arrival_monotonic_ns"] is not None for o in orders.values()),
            "native_touch_without_fill_count": summary.get("counts", {}).get(f"{arm}:touch_without_fill", 0)},
        "persistence_episodes": {"count": len(episodes), "outcomes": dict(Counter(e["outcome"] for e in episodes)),
            "configured_persistence_ms": settings.persistence_ms, "observations": episodes},
        "modeled_edge": edge, "quote_lifecycles": list(orders.values()),
        "modeled_charges_per_contract": {name: _text(getattr(settings, name)) for name in (
            "uncertainty_charge", "adverse_selection_charge", "maker_fee_per_contract", "required_edge")},
        "controls": {"observations": controls,
            "accepted_control_multipliers": [_text(v) for v in sorted(accepted_multipliers)],
            "observed_fair_multipliers": [_text(v) for v in sorted(fair_multipliers)],
            "baseline_multiplier_fixed_at_one": arm == "baseline",
            "operator_pause_and_health_shared_across_arms": True},
        "manual_regime_changed": arm == "manual" and len(accepted_multipliers | fair_multipliers) > 1,
        "fills": {"fill_legs": len(fill_rows), "distinct_quotes": len(groups), "distinct_markets": len(filled_markets),
            "total_quantity": _text(sum((_number(f["quantity"]) for f in fill_rows), ZERO)),
            "cancel_in_flight_fill_legs": sum(f.get("cancel_in_flight") is True for f in fill_rows),
            "cancel_in_flight_quantity": _text(sum((_number(f["quantity"]) for f in fill_rows if f.get("cancel_in_flight") is True), ZERO)),
            "placement_slow_fair_less_entry_cost": fill_metric(lambda f: at_fill(f, "placement_fair")),
            "fill_slow_fair_less_entry_cost": fill_metric(lambda f: at_fill(f, "fair_at_fill")),
            "external_fair_change_placement_to_fill": fill_metric(fair_change, "fair_change_probability_dollars"),
            "horizons": horizons, "terminal_settlement_pnl": terminal, "liquidation_exit_fees_included": False},
        "independence": {"observed_markets": sorted(observed_markets), "submitted_markets": sorted(submitted_markets),
            "filled_markets": sorted(filled_markets), "distinct_observed_markets": len(observed_markets),
            "distinct_submitted_markets": len(submitted_markets), "distinct_filled_markets": len(filled_markets),
            "effective_independent_sample_size": None},
        "monotonic_regression_observed": clock_regressed, "trading_qualified": False, "actual_fills_observed": False,
        "limitations": ["Decision rows are throttled, not time-weighted observations or independent opportunities.",
            "Persistence episodes bound observed eligibility; they do not prove continuous availability between observations.",
            "Censored durations end at the last emitted arm event, which can precede the capture end.",
            "Order durations use simulated observed arrival and acknowledgment; queue position is unknown.",
            "Strict-penetration fills are declared scenarios, not actual executions or guaranteed fills.",
            "Final-summary fill legs are collapsed by quote at each horizon; partial legs and horizons are not independent.",
            "Fair changes are probability changes, bid marks exclude exit fees, and only terminal P&L uses settlement outcomes.",
            "Horizon model value less entry cost is not realized P&L; matching-cohort initial cushion plus drift uses identical covered legs.",
            "Fair drift can include manual multiplier changes and elapsed-time effects; it is not pure adverse selection. Baseline fixes m=1 but shares pauses.",
            "Markets, trading sessions and matched arms can be dependent; no independent sample size is asserted.",
            "This function does not verify artifact integrity, capture completeness, or trading qualification."]}
