from dataclasses import asdict, replace
from decimal import Decimal

import pytest

from src.metals.config import load_metals_config
from src.metals.maker_policy import MakerEngine, MakerSettings
from src.metals.shadow_session import ShadowSession
from tests.metals.test_forward_shadow import _metadata, _model


D = Decimal
MSEC = 1_000_000
NSEC = 1_000_000_000
BASE = 1300 * NSEC
TICKER = "KXSILVER15M-TEST"


class Scenario:
    def __init__(self, tmp_path, settings):
        self.rows = {"manual": [], "baseline": []}
        self.events = []
        self.sessions = {}
        self.sequence = 0
        for arm in self.rows:
            config = load_metals_config("configs/metals.example.toml")
            config = replace(
                config,
                storage=replace(config.storage, root=str(tmp_path / arm)),
                shadow=replace(
                    config.shadow, bankroll_min_dollars="25", bankroll_max_dollars="25",
                    max_contracts_per_order="1", max_net_contracts_per_market="1",
                    max_aggregate_worst_case_loss_dollars="1",
                    max_loss_per_market_dollars="1", daily_loss_limit_dollars="5",
                ),
            )
            session = ShadowSession(config)
            original = session._write

            def record(stream, metal, timestamp, row, *, original=original, arm=arm):
                if stream == "maker_event":
                    self.rows[arm].append(row)
                original(stream, metal, timestamp, row)

            session._write = record
            self.sessions[arm] = session
        self.engine = MakerEngine(_model(), "fixture-study", settings, self.sessions)

    def send(self, kind, ms, *, wall_ms=None, mono_ms=None, **payload):
        self.sequence += 1
        event = {
            "kind": kind, "receive_ts_ns": BASE + (ms if wall_ms is None else wall_ms) * MSEC,
            "local_monotonic_ts_ns": 10 * NSEC + (ms if mono_ms is None else mono_ms) * MSEC,
            "ingestion_sequence": self.sequence, **payload,
        }
        self.events.append(event)
        self.engine.handle(event)
        return event

    def control(self, ms, *, version=1, multiplier="1", paused=False, expiry_ms=3_600_000):
        return self.send("control", ms, control={
            "version": version, "effective_ts_ns": BASE + ms * MSEC,
            "expires_ts_ns": BASE + expiry_ms * MSEC,
            "volatility_multiplier": multiplier, "paused": paused, "reason": "fixture",
        })

    def book(self, ms, *, yes="0.20", no="0.78", valid=True):
        return self.send("book", ms, market_ticker=TICKER,
                         yes_bids={yes: "10"}, no_bids={no: "10"}, valid=valid,
                         book_state_hash=f"book-{self.sequence + 1}")

    def boot(self, *, exact="66.95", multiplier="1", expiry_ms=3_600_000,
             yes="0.20", no="0.78"):
        self.control(0, multiplier=multiplier, expiry_ms=expiry_ms)
        self.send("metadata", 1, metadata=asdict(_metadata()))
        self.book(2, yes=yes, no=no)
        self.send("fast", 3, value="100", source_ts_ns=BASE)
        self.send("exact", 4, value=exact, source_ts_ns=BASE)
        return self

    def submit(self):
        self.send("timer", 104)
        return self

    def rest(self):
        self.submit()
        self.send("timer", 204)
        return self

    def orders(self, arm="manual"):
        return list(self.sessions[arm].engine.resting_orders.values())

    def trade(self, ms, *, source_ms=None, qty="1", price="0.20", trade_id=None):
        return self.send(
            "trade", ms, market_ticker=TICKER,
            trade_id=trade_id or f"trade-{ms}",
            source_event_ts_ns=BASE + (ms if source_ms is None else source_ms) * MSEC,
            taker_outcome_side="no", yes_price_dollars=price,
            no_price_dollars=str(D(1) - D(price)), quantity_fp=qty,
        )


@pytest.fixture
def scenario(tmp_path):
    scenarios = []

    def create(**kwargs):
        settings = MakerSettings(
            require_fast=False, persistence_ms=100, decision_interval_ms=1, **kwargs
        )
        result = Scenario(tmp_path / str(len(scenarios)), settings)
        scenarios.append(result)
        return result

    yield create
    for item in scenarios:
        for session in item.sessions.values():
            session.close()


def test_exact_only_diagnostic_uses_shared_risk_and_delayed_post_only_arrival(scenario):
    case = scenario().boot().submit()
    for arm in case.sessions:
        order = case.orders(arm)[0]
        assert order.state == "pending_submit"
        assert order.intent.quantity == D(1)
        assert order.intent.price == D("0.21")
        assert order.arrival_ts_ns is None
        snapshot = case.sessions[arm].engine.snapshot()
        assert D(snapshot.aggregate_worst_case_loss_dollars) == D("0.21")
        decision = [row for row in case.rows[arm] if row.get("action") == "submit"][-1]
        assert decision["fair"]["mode"] == "exact_only_diagnostic"
    case.send("timer", 204)
    assert case.orders()[0].state == "resting"
    assert case.orders()[0].arrival_ts_ns == BASE + 204 * MSEC


def test_gap_with_outstanding_exposure_freezes_arm_even_after_full_recovery(scenario):
    case = scenario().boot().rest()
    case.send("gap", 210, reason="test_gap")
    case.send("timer", 310)
    assert not case.orders()
    case.book(320)
    case.send("exact", 321, value="66.95", source_ts_ns=BASE + 300 * MSEC)
    case.send("timer", 700)
    assert not case.orders()
    summary = case.engine.summary()
    assert summary["execution_qualification_incomplete"] is True
    assert summary["uncertain_exposure"]["manual"] == "data_gap_while_order_outstanding"
    assert any(row.get("reason") == "unknown_exposure_restart_required"
               for row in case.rows["manual"])


@pytest.mark.parametrize("multiplier", ["1", "1.5"])
def test_manual_sigma_multiplier_has_fixed_baseline_comparator(scenario, multiplier):
    case = scenario().boot(multiplier=multiplier).rest()
    manual = case.engine._safe_fair("manual", TICKER, case.events[-1])
    baseline = case.engine._safe_fair("baseline", TICKER, case.events[-1])
    assert baseline["volatility_multiplier"] == D(1)
    assert manual["remaining_variance"] == baseline["remaining_variance"] * D(multiplier) ** 2
    if multiplier == "1":
        assert manual == baseline
    else:
        assert manual["exact_gaussian_yes"] > baseline["exact_gaussian_yes"]


def test_fast_input_can_cancel_but_cannot_raise_the_exact_bid_ceiling(scenario):
    case = scenario()
    case.engine.settings = replace(case.engine.settings, require_fast=True)
    case.boot().rest()
    order_id = case.orders()[0].intent.logical_quote_id
    placement = next(row for row in case.rows["manual"] if row.get("action") == "submit")
    case.send("fast", 310, value="100.1", source_ts_ns=BASE + 300 * MSEC)
    assert case.orders()[0].intent.logical_quote_id == order_id
    retained = [row for row in case.rows["manual"] if row.get("action") == "retain"][-1]
    assert retained["bid_ceiling"] == placement["bid_ceiling"]
    assert retained["fair"]["fast_side_fair"] > retained["fair"]["side_fair"]
    case.send("fast", 410, value="99.9", source_ts_ns=BASE + 400 * MSEC)
    assert case.orders()[0].state == "pending_cancel"
    assert any(row.get("reason") == "resting_edge_invalidated" for row in case.rows["manual"])


def test_complete_fast_health_and_edge_rule_must_persist_continuously(scenario):
    case = scenario()
    case.engine.settings = replace(case.engine.settings, require_fast=True)
    case.boot()
    case.send("fast", 50, value="99.9", source_ts_ns=BASE + 40 * MSEC)
    case.send("fast", 90, value="100", source_ts_ns=BASE + 80 * MSEC)
    case.send("timer", 104)
    assert not case.orders()
    case.send("timer", 190)
    assert case.orders()[0].state == "pending_submit"


def test_fast_freshness_timer_cancels_while_exact_index_remains_fresh(scenario):
    case = scenario()
    case.engine.settings = replace(case.engine.settings, require_fast=True)
    case.boot().rest()
    case.send("timer", 751)
    assert case.orders()[0].state == "pending_cancel"
    assert any(row.get("reason") == "fast_source_stale" for row in case.rows["manual"])


def test_post_only_order_is_rejected_when_book_crosses_before_arrival(scenario):
    case = scenario().boot().submit()
    quote_id = case.orders()[0].intent.logical_quote_id
    case.book(204, yes="0.18", no="0.80")
    assert all(order.intent.logical_quote_id != quote_id for order in case.orders())
    assert any(row.get("transition", {}).get("event_type") == "post_only_rejected"
               for row in case.rows["manual"])
    assert not case.engine.fills["manual"]


def test_partial_fill_keeps_cancel_risk_and_prevents_replenishment(scenario):
    case = scenario().boot().rest()
    case.trade(220, source_ms=210, qty="0.4")
    order = case.orders()[0]
    assert order.state == "pending_cancel"
    assert order.remaining_quantity == D("0.6")
    assert D(case.sessions["manual"].engine.snapshot().aggregate_worst_case_loss_dollars) == D("0.21")
    case.trade(240, source_ms=230, qty="0.6")
    assert not case.orders()
    assert len(case.engine.fills["manual"]) == 2
    assert case.engine.fills["manual"][1]["cancel_in_flight"] is True
    case.send("timer", 600)
    assert not case.orders()
    assert any(row.get("reason") == "no_replenishment_after_fill" for row in case.rows["manual"])


def test_trade_source_before_arrival_touch_and_duplicate_do_not_create_fills(scenario):
    case = scenario().boot().rest()
    case.trade(220, source_ms=203)
    assert case.engine.counts["manual:trade_precedes_arrival"] == 1
    case.trade(230, source_ms=210, price="0.21")
    assert case.engine.counts["manual:touch_without_fill"] == 1
    assert not case.engine.fills["manual"]
    case.trade(240, source_ms=230, qty="0.4", trade_id="once")
    case.trade(250, source_ms=240, qty="0.4", trade_id="once")
    assert len(case.engine.fills["manual"]) == 1


def test_pause_and_expiry_cancel_both_comparator_arms(scenario):
    paused = scenario().boot().rest()
    paused.control(300, version=2, paused=True)
    for arm in paused.sessions:
        assert paused.orders(arm)[0].state == "pending_cancel"
    paused.send("timer", 400)
    assert all(not paused.orders(arm) for arm in paused.sessions)
    expired = scenario().boot(expiry_ms=350).rest()
    expired.send("timer", 350)
    assert expired.orders()[0].state == "pending_cancel"
    expired.send("timer", 450)
    assert not expired.orders()
    assert any(row.get("reason") == "manual_control_expired" for row in expired.rows["manual"])


def test_bad_control_version_cancels_and_requires_a_new_valid_version(scenario):
    case = scenario().boot().rest()
    case.control(300, version=1)
    assert case.orders()[0].state == "pending_cancel"
    assert case.engine.control_error.startswith("control_invalid:")
    case.send("timer", 400)
    assert not case.orders()
    case.control(500, version=2)
    case.send("timer", 600)
    assert case.orders()[0].state == "pending_submit"


def test_quiet_timer_cancels_stale_quotes_without_any_feed_event(scenario):
    case = scenario().boot().rest()
    case.send("timer", 2001)
    assert case.orders()[0].state == "pending_cancel"
    case.send("timer", 2101)
    assert not case.orders()
    assert any(row.get("reason") == "exact_index_stale" for row in case.rows["manual"])


def test_book_gap_cancels_and_freezes_unknown_exposure(scenario):
    case = scenario().boot().rest()
    case.send("gap", 300, market_ticker=TICKER, reason="sequence_gap")
    assert case.orders()[0].state == "pending_cancel"
    assert case.engine.books[TICKER].valid is False
    case.send("timer", 400)
    case.send("timer", 600)
    assert not case.orders()
    case.book(700)
    case.send("timer", 800)
    assert not case.orders()
    assert "manual" in case.engine.uncertain_exposure


def test_book_gap_without_exposure_can_recover_after_fresh_snapshot(scenario):
    case = scenario().boot()
    case.send("gap", 50, market_ticker=TICKER, reason="sequence_gap")
    case.send("timer", 100)
    assert not case.orders()
    case.book(110)
    case.send("timer", 210)
    assert case.orders()[0].state == "pending_submit"


@pytest.mark.parametrize(
    "yes,no,expected,side",
    [("0.099", "0.89", "0.100", "yes"),
     ("0.20", "0.79", "0.20", "yes"),
     ("0.78", "0.20", "0.21", "no")],
)
def test_quote_improvement_uses_native_grid_and_never_crosses(scenario, yes, no, expected, side):
    case = scenario().boot(exact="67.05" if side == "no" else "66.95", yes=yes, no=no).submit()
    assert case.orders()[0].intent.price == D(expected)
    assert case.orders()[0].intent.outcome_side == side


def test_settlement_reconciles_fractional_fills_fees_and_position_release(scenario):
    case = scenario(maker_fee_per_contract=D("0.003")).boot().rest()
    case.trade(220, qty="0.4")
    case.trade(240, qty="0.6")
    case.send("metadata", 600100, metadata=asdict(_metadata()), result="yes")
    for arm, session in case.sessions.items():
        assert session.engine.positions == {}
        assert session.engine.daily_realized_pnl == D("0.787")
        assert sum(fill["terminal_pnl"] for fill in case.engine.fills[arm]) == D("0.787")
        assert TICKER in case.engine.settled[arm]


def test_book_markout_is_distinct_from_fair_change_and_survives_stale_index(scenario):
    case = scenario().boot().rest()
    case.trade(220)
    case.book(1220, yes="0.19")
    mark = case.engine.fills["manual"][0]["markouts"]["1s"]
    assert mark["bid_mark_less_entry_cost_per_contract"] == D("-0.02")
    assert mark["external_fair_change"] == 0
    assert mark["exit_fees_included"] is False
    case.book(5220, yes="0.20")
    later = case.engine.fills["manual"][0]["markouts"]["5s"]
    assert later["bid_mark_less_entry_cost_per_contract"] == D("-0.01")
    assert later["side_external_fair"] is None


def test_replaying_identical_events_produces_identical_decision_digest(scenario):
    first = scenario().boot(multiplier="1.25").rest()
    first.trade(220, qty="0.4")
    first.send("timer", 320)
    first.send("metadata", 600100, metadata=asdict(_metadata()), result="no")
    second = scenario()
    for event in first.events:
        second.engine.handle(event)
    assert first.engine.summary()["decision_digest"] == second.engine.summary()["decision_digest"]
    assert first.engine.summary()["arms"] == second.engine.summary()["arms"]


def test_wall_clock_jump_uses_monotonic_cancel_deadline_and_never_resumes(scenario):
    case = scenario().boot().rest()
    case.send("timer", 300, wall_ms=800)
    assert case.engine.clock_fault is True
    assert case.orders()[0].state == "pending_cancel"
    case.send("timer", 305, wall_ms=805)
    assert case.orders()[0].state == "pending_cancel"
    case.send("timer", 400, wall_ms=900)
    assert not case.orders()
    case.send("timer", 1000, wall_ms=1500)
    assert not case.orders()


def test_monotonic_backwards_jump_latches_clock_fault(scenario):
    case = scenario().boot().rest()
    case.send("timer", 205, mono_ms=190)
    assert case.engine.clock_fault is True
    assert case.orders()[0].state == "pending_cancel"


def test_future_price_source_is_rejected_before_becoming_a_fair(scenario):
    case = scenario().boot().rest()
    prior = case.engine.exact
    with pytest.raises(ValueError, match="source timestamp is after receipt"):
        case.send("exact", 300, value="67", source_ts_ns=BASE + 301 * MSEC)
    assert case.engine.exact is prior


def test_future_exchange_trade_latches_clock_alignment_fault_and_cancels(scenario):
    case = scenario().boot().rest()
    case.trade(250, source_ms=275)
    assert case.engine.clock_domain_fault is True
    assert case.orders()[0].state == "pending_cancel"
    assert not case.engine.fills["manual"]
    case.send("timer", 350)
    case.trade(400, source_ms=399)
    case.control(500, version=2)
    case.send("timer", 700)
    assert not case.orders()
    assert "exchange_source_clock_ahead_of_receive" in case.engine.summary()["execution_qualification_reasons"]


def test_delayed_trade_after_cancel_flags_unknown_fill_without_fabricating_position(scenario):
    case = scenario().boot().rest()
    case.control(300, version=2, paused=True)
    case.send("timer", 400)
    assert not case.orders()
    case.trade(450, source_ms=350)
    for arm, session in case.sessions.items():
        assert not case.engine.fills[arm]
        assert not session.engine.positions
        assert case.engine.counts[f"{arm}:unknown_fills_after_retirement"] == 1
        row = next(row for row in case.rows[arm] if row["kind"] == "late_trade_after_retirement")
        assert row["execution_qualification_incomplete"] is True
    case.trade(460, source_ms=410)
    assert case.engine.counts["manual:unknown_fills_after_retirement"] == 1


def test_trade_during_clock_fault_is_uncertain_and_cannot_fabricate_a_fill(scenario):
    case = scenario().boot().rest()
    case.send("timer", 250, wall_ms=750)
    case.send(
        "trade", 260, wall_ms=760, market_ticker=TICKER, trade_id="uncertain-clock",
        source_event_ts_ns=BASE + 210 * MSEC, taker_outcome_side="no",
        yes_price_dollars="0.20", no_price_dollars="0.80", quantity_fp="1",
    )
    assert not case.engine.fills["manual"]
    assert case.engine.counts["manual:trade_ignored_clock_fault"] == 1
    assert any(row["kind"] == "trade_time_uncertain" for row in case.rows["manual"])
