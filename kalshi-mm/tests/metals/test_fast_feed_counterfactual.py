from decimal import Decimal

import pytest

from src.metals.fast_feed_counterfactual import (
    ExactPrice,
    FillObservation,
    ProxyPrice,
    _scenario_summary,
    simulate_fast_snapshots,
)


def test_fast_snapshot_is_anchored_and_causal():
    exact = [
        ExactPrice(1_000_000_000, 1_300_000_000, Decimal("66.00")),
        ExactPrice(2_000_000_000, 2_300_000_000, Decimal("66.10")),
    ]
    proxy = [
        ProxyPrice(1_000_000_000, Decimal("66.02")),
        ProxyPrice(1_400_000_000, Decimal("66.07")),
        ProxyPrice(2_000_000_000, Decimal("66.12")),
    ]

    snapshots, predictions = simulate_fast_snapshots(
        exact, proxy, proxy_transport_latency_ms=100
    )

    proxy_delta = next(
        row for row in snapshots if row.available_ts_ns == 1_500_000_000
    )
    assert proxy_delta.estimated_index == Decimal("66.05")
    assert all(row.available_ts_ns >= row.source_ts_ns for row in snapshots)
    assert predictions[0].stale_absolute_error == Decimal("0.10")
    assert predictions[0].fast_absolute_error == Decimal("0.00")


def test_proxy_at_same_receive_time_is_available_before_exact_anchor():
    exact = [ExactPrice(1_000_000_000, 1_100_000_000, Decimal("10"))]
    proxy = [ProxyPrice(1_000_000_000, Decimal("20"))]

    snapshots, predictions = simulate_fast_snapshots(
        exact, proxy, proxy_transport_latency_ms=100
    )

    assert predictions == []
    assert snapshots[-1].source == "exact_anchor"
    assert snapshots[-1].estimated_index == Decimal("10")


@pytest.mark.parametrize("latency", [-1, -100])
def test_negative_proxy_latency_is_rejected(latency):
    with pytest.raises(ValueError, match="nonnegative"):
        simulate_fast_snapshots([], [], proxy_transport_latency_ms=latency)


def test_scenario_attributes_avoided_markout_and_reconciles_residual():
    fills = [
        FillObservation(
            logical_quote_id="q1",
            market_ticker="m",
            side="yes",
            placement_ts_ns=0,
            fill_ts_ns=1_000_000_000,
            quantity=Decimal("2"),
            quote_price=Decimal("0.4"),
            frozen_charge=Decimal("0.04"),
            markout_1s=Decimal("-0.06"),
            markout_5s=Decimal("-0.04"),
        ),
        FillObservation(
            logical_quote_id="q2",
            market_ticker="m",
            side="yes",
            placement_ts_ns=0,
            fill_ts_ns=1_000_000_000,
            quantity=Decimal("1"),
            quote_price=Decimal("0.4"),
            frozen_charge=Decimal("0.04"),
            markout_1s=Decimal("-0.03"),
            markout_5s=Decimal("0.01"),
        ),
    ]

    summary = _scenario_summary(
        fills, [500_000_000, None], cancel_latency_ms=100
    )
    attribution = summary["markout_1s_attribution"]

    assert summary["avoided_fill_events"] == 1
    assert attribution["original_markout_total_dollars"] == "-0.15"
    assert attribution["avoided_group_markout_total_dollars"] == "-0.12"
    assert attribution["kept_markout_total_dollars"] == "-0.03"
    assert attribution["slow_feed_avoidable_loss_fraction_of_original"] == pytest.approx(
        0.8
    )
