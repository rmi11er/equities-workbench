"""Exchange-isolated order timing and cancellation-race regressions."""

import json
from dataclasses import replace
from decimal import Decimal

import pytest

from src.metals.config import ShadowConfig, load_metals_config
from src.metals.shadow import ShadowOrderIntent, ShadowRiskEngine
from src.metals.shadow_session import ShadowSession


def _intent(quote_id="q1", **changes):
    intent = ShadowOrderIntent(
        logical_quote_id=quote_id,
        market_ticker="KXSILVER15M-TEST",
        metal="silver",
        outcome_side="yes",
        price_dollars="0.20",
        quantity_fp="1",
        decision_ts_ns=1000,
        strategy_module="opportunistic_maker",
        model_version="baseline-v1",
        config_version="metals-shadow-v1",
        state_snapshot_hash="state-hash",
        worst_case_fee_dollars="0.01",
    )
    return replace(intent, **changes)


def _types(events):
    return [event.event_type for event in events]


def test_pending_submission_reserves_full_risk_and_cannot_fill():
    engine = ShadowRiskEngine(ShadowConfig(max_net_contracts_per_market="1"))
    decision, events = engine.place(
        _intent(), monotonic_ts_ns=10, submit_latency_ns=100
    )

    assert decision.approved
    assert _types(events) == ["decision_created", "submit_sent"]
    order = engine.resting_orders["q1"]
    assert order.state == "pending_submit"
    assert order.arrival_ts_ns is None
    assert engine.snapshot().aggregate_worst_case_loss_dollars == "0.21"
    assert not engine.assess(_intent("q2")).approved
    with pytest.raises(ValueError, match="before simulated arrival"):
        engine.fill("q1", "1", 1001)

    assert engine.advance(5000, 109, lambda _: True) == []
    events = engine.advance(5001, 110, lambda _: True)
    assert _types(events) == ["submit_acknowledged", "order_resting"]
    assert order.state == "resting"
    assert order.arrival_ts_ns == 5001
    with pytest.raises(ValueError, match="before simulated arrival"):
        engine.fill("q1", "1", 5000)
    assert engine.fill("q1", "1", 5002).event_type == "full_fill"


def test_arrival_rechecks_current_book_and_releases_rejected_risk():
    engine = ShadowRiskEngine(ShadowConfig())
    engine.place(_intent(), monotonic_ts_ns=10, submit_latency_ns=100)
    seen = []

    def crossed_book(intent):
        seen.append(intent.logical_quote_id)
        return False

    events = engine.advance(1100, 110, crossed_book)

    assert seen == ["q1"]
    assert _types(events) == ["post_only_rejected"]
    assert engine.resting_orders == {}
    assert engine.snapshot().aggregate_worst_case_loss_dollars == "0"
    assert engine.cancel("q1", 1101) == []


def test_pending_cancel_keeps_risk_and_allows_partial_fills_until_ack():
    engine = ShadowRiskEngine(ShadowConfig())
    engine.place(_intent(quantity_fp="2"))
    events = engine.cancel(
        "q1", 1100, monotonic_ts_ns=20, cancel_latency_ns=100
    )
    assert _types(events) == ["cancel_decision", "cancel_sent"]
    assert engine.resting_orders["q1"].state == "pending_cancel"
    assert engine.snapshot().aggregate_worst_case_loss_dollars == "0.41"

    fill = engine.fill("q1", "0.5", 1150, fee_dollars="0.005")
    assert fill.event_type == "partial_fill"
    assert engine.resting_orders["q1"].remaining_quantity == Decimal("1.5")
    assert engine.snapshot().aggregate_worst_case_loss_dollars == "0.410"
    assert engine.advance(1199, 119, lambda _: True) == []
    assert _types(engine.advance(1200, 120, lambda _: True)) == [
        "cancel_acknowledged"
    ]
    assert engine.snapshot().aggregate_worst_case_loss_dollars == "0.105"
    with pytest.raises(KeyError):
        engine.fill("q1", "0.5", 1201)


def test_duplicate_cancellation_does_not_move_deadline_or_duplicate_events():
    engine = ShadowRiskEngine(ShadowConfig())
    engine.place(_intent())
    engine.cancel("q1", 1100, monotonic_ts_ns=20, cancel_latency_ns=100)

    assert engine.cancel(
        "q1", 1199, monotonic_ts_ns=119, cancel_latency_ns=100
    ) == []
    assert engine.resting_orders["q1"].cancel_deadline_monotonic_ns == 120
    assert _types(engine.advance(1200, 120, lambda _: True)) == [
        "cancel_acknowledged"
    ]
    assert engine.cancel("q1", 1300) == []
    with pytest.raises(KeyError):
        engine.cancel("never-created", 1300)


def test_cancel_before_arrival_retains_submission_and_waits_for_exchange_id():
    engine = ShadowRiskEngine(ShadowConfig())
    engine.place(_intent(), monotonic_ts_ns=10, submit_latency_ns=100)
    events = engine.cancel(
        "q1", 1010, monotonic_ts_ns=20, cancel_latency_ns=50
    )
    assert _types(events) == ["cancel_decision"]
    order = engine.resting_orders["q1"]
    assert order.state == "pending_cancel"
    assert order.submit_deadline_monotonic_ns == 110
    assert order.arrival_ts_ns is None

    # Even though the cancel request predates submission arrival, it cannot
    # eliminate the transport race after an order ID becomes available.
    assert engine.advance(1050, 100, lambda _: True) == []
    with pytest.raises(ValueError, match="before simulated arrival"):
        engine.fill("q1", "1", 1060)
    assert _types(engine.advance(1100, 110, lambda _: True)) == [
        "submit_acknowledged", "order_resting", "cancel_sent"
    ]
    assert order.state == "pending_cancel"
    assert order.arrival_ts_ns == 1100
    assert order.cancel_deadline_monotonic_ns == 160
    engine.fill("q1", "0.25", 1120)
    assert engine.advance(1149, 159, lambda _: True) == []
    assert _types(engine.advance(1150, 160, lambda _: True)) == [
        "cancel_acknowledged"
    ]


def test_overdue_arrival_does_not_backdate_fills_or_compress_cancel_transport():
    engine = ShadowRiskEngine(ShadowConfig())
    engine.place(_intent(), monotonic_ts_ns=10, submit_latency_ns=100)
    engine.cancel("q1", 1020, monotonic_ts_ns=20, cancel_latency_ns=50)

    events = engine.advance(5000, 500, lambda _: True)
    order = engine.resting_orders["q1"]
    assert [event.event_ts_ns for event in events] == [5000, 5000, 5000]
    assert order.arrival_ts_ns == 5000
    assert order.cancel_deadline_monotonic_ns == 550
    assert not engine.positions
    with pytest.raises(ValueError, match="before simulated arrival"):
        engine.fill("q1", "1", 4999)
    assert engine.advance(5049, 549, lambda _: True) == []
    assert _types(engine.advance(5050, 550, lambda _: True)) == [
        "cancel_acknowledged"
    ]


def test_zero_latency_cancel_of_pending_submission_still_checks_arrival():
    engine = ShadowRiskEngine(ShadowConfig())
    engine.place(_intent(), monotonic_ts_ns=10, submit_latency_ns=100)
    engine.cancel("q1", 1020, monotonic_ts_ns=20)

    assert _types(engine.advance(1100, 110, lambda _: True)) == [
        "submit_acknowledged", "order_resting", "cancel_sent", "cancel_acknowledged"
    ]
    assert engine.resting_orders == {}


def test_rejected_arrival_resolves_pending_cancel_without_phantom_ack():
    engine = ShadowRiskEngine(ShadowConfig())
    engine.place(_intent(), monotonic_ts_ns=10, submit_latency_ns=100)
    engine.cancel("q1", 1020, monotonic_ts_ns=20, cancel_latency_ns=50)
    assert _types(engine.advance(1100, 110, lambda _: False)) == [
        "post_only_rejected"
    ]
    assert engine.advance(2000, 200, lambda _: True) == []
    assert engine.resting_orders == {}


def test_full_fill_during_cancel_finishes_order_without_later_cancel_ack():
    engine = ShadowRiskEngine(ShadowConfig())
    engine.place(_intent())
    engine.cancel("q1", 1100, monotonic_ts_ns=20, cancel_latency_ns=100)
    engine.fill("q1", "1", 1150)

    assert engine.advance(1200, 120, lambda _: True) == []
    assert engine.cancel("q1", 1200) == []


def test_wall_clock_changes_cannot_accelerate_or_delay_monotonic_deadlines():
    engine = ShadowRiskEngine(ShadowConfig())
    engine.place(_intent(), monotonic_ts_ns=10, submit_latency_ns=100)

    assert engine.advance(1_000_000, 109, lambda _: True) == []
    assert _types(engine.advance(900, 110, lambda _: True)) == [
        "submit_acknowledged", "order_resting"
    ]
    assert engine.resting_orders["q1"].arrival_ts_ns == 900
    engine.cancel("q1", 910, monotonic_ts_ns=120, cancel_latency_ns=20)
    assert engine.advance(1_000_000, 139, lambda _: True) == []
    assert _types(engine.advance(800, 140, lambda _: True)) == [
        "cancel_acknowledged"
    ]


@pytest.mark.parametrize("latency", [-1, 0.5, "10"])
def test_invalid_latencies_do_not_mutate_engine(latency):
    engine = ShadowRiskEngine(ShadowConfig())
    with pytest.raises(ValueError, match="latency_ns"):
        engine.place(_intent(), monotonic_ts_ns=10, submit_latency_ns=latency)
    assert engine.resting_orders == {}


def test_delayed_calls_require_explicit_monotonic_clock():
    engine = ShadowRiskEngine(ShadowConfig())
    with pytest.raises(ValueError, match="monotonic_ts_ns"):
        engine.place(_intent(), submit_latency_ns=100)
    engine.place(_intent(), monotonic_ts_ns=10, submit_latency_ns=100)
    with pytest.raises(ValueError, match="monotonic_ts_ns"):
        engine.cancel("q1", 1010, cancel_latency_ns=100)
    with pytest.raises(ValueError, match="monotonic_ts_ns"):
        engine.cancel("q1", 1010)
    assert engine.resting_orders["q1"].state == "pending_submit"


def test_session_persists_delayed_arrivals_and_cancel_acknowledgements(tmp_path):
    config = load_metals_config("configs/metals.example.toml")
    config = replace(config, storage=replace(config.storage, root=str(tmp_path)))
    with ShadowSession(config) as session:
        session.place(_intent(), monotonic_ts_ns=10, submit_latency_ns=100)
        session.advance(1100, 110, lambda _: True)
        session.cancel("q1", 1150, monotonic_ts_ns=160, cancel_latency_ns=50)
        session.advance(1200, 210, lambda _: True)
        assert session.cancel("q1", 1210) == []

    rows = [
        json.loads(line)
        for path in (tmp_path / "normalized" / "order_lifecycle").glob("**/*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event_type"] for row in rows] == [
        "decision_created", "submit_sent", "submit_acknowledged", "order_resting",
        "cancel_decision", "cancel_sent", "cancel_acknowledged",
    ]
    assert [row["event_ts_ns"] for row in rows] == [
        1000, 1000, 1100, 1100, 1150, 1150, 1200
    ]
    assert session.counts["order_lifecycle_events"] == 7
    assert session.counts["shadow_fills"] == 0
