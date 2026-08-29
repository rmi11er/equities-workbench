from decimal import Decimal

import pytest

from src.metals.config import ShadowConfig
from src.metals.shadow import ShadowOrderIntent, ShadowRiskEngine


def _intent(
    quote_id="q1",
    ticker="KXSILVER15M-TEST",
    side="yes",
    price="0.40",
    quantity="250",
    config_version="metals-shadow-v1",
    worst_case_fee="0",
):
    return ShadowOrderIntent(
        logical_quote_id=quote_id,
        market_ticker=ticker,
        metal="silver",
        outcome_side=side,
        price_dollars=price,
        quantity_fp=quantity,
        decision_ts_ns=100,
        strategy_module="S2",
        model_version="model-v0",
        config_version=config_version,
        state_snapshot_hash="state-hash",
        worst_case_fee_dollars=worst_case_fee,
    )


def test_shadow_config_and_engine_refuse_live_orders():
    with pytest.raises(ValueError):
        ShadowRiskEngine(
            ShadowConfig(execution_mode="production", allow_live_orders=True)
        )


def test_order_limit_and_config_version_are_enforced():
    engine = ShadowRiskEngine(ShadowConfig())

    too_large = engine.assess(_intent(quantity="250.01"))
    wrong_config = engine.assess(_intent(config_version="stale"))

    assert too_large.approved is False
    assert "max_contracts_per_order" in too_large.rejection_reasons
    assert wrong_config.rejection_reasons == ("config_version_mismatch",)


def test_resting_orders_count_toward_directional_and_loss_limits():
    engine = ShadowRiskEngine(
        ShadowConfig(
            max_contracts_per_order="1000",
            max_net_contracts_per_market="1000",
            max_loss_per_market_dollars="1000",
        )
    )
    decision, _ = engine.place(_intent(quantity="900", price="0.90"))
    assert decision.approved

    second = engine.assess(_intent(quote_id="q2", quantity="101", price="0.90"))

    assert second.approved is False
    assert "max_net_contracts_per_market" in second.rejection_reasons


def test_two_sided_resting_orders_use_one_sided_adverse_fill_scenario():
    engine = ShadowRiskEngine(ShadowConfig())
    yes_decision, _ = engine.place(_intent(quantity="250", price="0.40"))
    no_decision, _ = engine.place(
        _intent(quote_id="q2", side="no", quantity="250", price="0.40")
    )

    assert yes_decision.approved and no_decision.approved
    snapshot = engine.snapshot()
    assert snapshot.market_max_abs_net_contracts["KXSILVER15M-TEST"] == "250"
    assert snapshot.market_worst_case_loss_dollars["KXSILVER15M-TEST"] == "100.00"


def test_fill_lifecycle_position_and_settlement_are_deterministic():
    engine = ShadowRiskEngine(ShadowConfig())
    decision, events = engine.place(
        _intent(quantity="10", price="0.40", worst_case_fee="0.01")
    )

    assert decision.approved
    assert [event.event_type for event in events] == [
        "decision_created",
        "submit_sent",
        "submit_acknowledged",
        "order_resting",
    ]
    partial = engine.fill("q1", "4", event_ts_ns=110, fee_dollars="0.01")
    full = engine.fill("q1", "6", event_ts_ns=120)
    assert partial.event_type == "partial_fill"
    assert partial.remaining_quantity_fp == "6"
    assert full.event_type == "full_fill"
    assert full.remaining_quantity_fp == "0"

    position = engine.positions["KXSILVER15M-TEST"]
    assert position.yes_quantity == Decimal("10")
    assert position.cash_spent == Decimal("4.01")
    assert engine.settle_market("KXSILVER15M-TEST", "yes") == Decimal("5.99")
    assert engine.daily_realized_pnl == Decimal("5.99")


def test_fee_reserve_counts_toward_limits_and_fill_cannot_exceed_it():
    engine = ShadowRiskEngine(
        ShadowConfig(
            max_contracts_per_order="1000",
            max_loss_per_market_dollars="100",
        )
    )
    at_limit_without_fee = engine.assess(
        _intent(quantity="250", price="0.40", worst_case_fee="0.01")
    )

    assert at_limit_without_fee.approved is False
    assert "max_loss_per_market" in at_limit_without_fee.rejection_reasons

    placed, _ = engine.place(
        _intent(quote_id="q2", quantity="10", worst_case_fee="0.01")
    )
    assert placed.approved
    with pytest.raises(ValueError, match="exceeds the reserved"):
        engine.fill("q2", "1", event_ts_ns=110, fee_dollars="0.02")


def test_kill_switch_blocks_new_shadow_orders_and_cancel_remains_available():
    engine = ShadowRiskEngine(ShadowConfig())
    decision, _ = engine.place(_intent(quantity="10"))
    assert decision.approved

    engine.set_kill_switch("invalid_kalshi_book")
    blocked = engine.assess(_intent(quote_id="q2", quantity="10"))
    cancel_events = engine.cancel("q1", event_ts_ns=130)

    assert blocked.approved is False
    assert "kill_switch:invalid_kalshi_book" in blocked.rejection_reasons
    assert [event.event_type for event in cancel_events] == [
        "cancel_decision",
        "cancel_sent",
        "cancel_acknowledged",
    ]


def test_aggregate_and_daily_worst_case_limits_span_markets():
    engine = ShadowRiskEngine(
        ShadowConfig(
            max_contracts_per_order="1000",
            max_loss_per_market_dollars="1000",
            max_aggregate_worst_case_loss_dollars="1500",
            daily_loss_limit_dollars="1400",
        )
    )
    first, _ = engine.place(
        _intent(ticker="SILVER-A", quantity="1000", price="0.71")
    )
    assert first.approved

    second = engine.assess(
        _intent(quote_id="q2", ticker="SILVER-B", quantity="1000", price="0.71")
    )

    assert second.approved is False
    assert "daily_loss_limit" in second.rejection_reasons


@pytest.mark.parametrize(
    "price,quantity",
    [("0", "1"), ("1", "1"), ("NaN", "1"), ("0.5", "0")],
)
def test_order_values_must_be_finite_and_inside_binary_bounds(price, quantity):
    with pytest.raises(ValueError):
        _intent(price=price, quantity=quantity)
