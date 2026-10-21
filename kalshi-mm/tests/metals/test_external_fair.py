from decimal import Decimal

import pytest

from src.metals.external_fair import (
    CausalityError,
    FairSnapshot,
    FillAttribution,
    PriceEvent,
    ReceiveKey,
    ReplicatorValidationLimits,
    asof_receive_event,
    build_replicator_rows,
    fit_replicator,
    replicator_metrics,
    run_replicator_lag_sweep,
    summarize_attribution,
    validate_replicator,
)


def _event(value, nanoseconds, sequence, day="2026-01-01", session="comex"):
    return PriceEvent(
        value=float(value),
        receive_key=ReceiveKey(nanoseconds, sequence),
        day=day,
        session=session,
        source_event_ts_ns=nanoseconds - 10,
        vendor_ts_ns=nanoseconds - 5,
    )


def test_receive_time_asof_uses_sequence_as_tie_break_and_blocks_positive_shift():
    events = [_event(31.0, 1_000, 1), _event(31.1, 1_000, 3)]

    observed = asof_receive_event(events, ReceiveKey(1_000, 2))

    assert observed.value == 31.0
    with pytest.raises(CausalityError, match="diagnostic only"):
        asof_receive_event(events, ReceiveKey(1_000, 2), shift_ms=100)


def test_build_replicator_rows_never_uses_later_receive_event_at_zero_shift():
    pyth = [_event(30.0, 1_000, 2), _event(30.2, 2_000, 3)]
    futures = [
        _event(31.0, 1_000, 1),
        _event(31.2, 2_000, 2),
        _event(99.0, 2_000, 4),
    ]

    rows = build_replicator_rows(pyth, futures)

    assert len(rows) == 1
    assert rows[0].anchor_futures == 31.0
    assert rows[0].current_futures == 31.2
    assert not rows[0].uses_future_input
    assert rows[0].maximum_input_key <= rows[0].decision_key


def test_positive_shift_is_labeled_as_future_input_when_it_reaches_a_later_event():
    pyth = [_event(30.0, 1_000_000_000, 2), _event(30.2, 2_000_000_000, 3)]
    futures = [
        _event(31.0, 1_000_000_000, 1),
        _event(31.2, 1_050_000_000, 2),
        _event(31.3, 2_050_000_000, 4),
    ]

    row = build_replicator_rows(
        pyth, futures, shift_ms=100, allow_noncausal=True
    )[0]

    assert row.uses_future_input
    assert row.anchor_futures == 31.2
    assert row.current_futures == 31.3


def test_replicator_fits_session_beta_and_beats_unchanged_pyth():
    pyth = []
    futures = []
    future_values = (30.0, 31.0, 33.0, 36.0, 40.0, 45.0)
    for index, future in enumerate(future_values):
        timestamp = (index + 1) * 1_000
        benchmark = 60.0 + 2.0 * (future - 30.0)
        futures.append(_event(future, timestamp, 1))
        pyth.append(_event(benchmark, timestamp, 2))

    rows = build_replicator_rows(pyth, futures)
    model = fit_replicator(rows[:3])
    metrics = replicator_metrics(model, rows[3:])

    assert model.coefficients["comex"].beta == pytest.approx(2.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["unchanged_pyth_mae"] == pytest.approx(9.0)


def test_validation_fails_closed_when_strike_or_daily_stability_is_unmeasured():
    pyth = [_event(30 + index, (index + 1) * 1_000, 2) for index in range(5)]
    futures = [_event(60 + index, (index + 1) * 1_000, 1) for index in range(5)]
    rows = build_replicator_rows(pyth, futures)
    model = fit_replicator(rows[:2])
    limits = ReplicatorValidationLimits(2, 0.01, 0.01, 0.01, 0.10)

    report = validate_replicator(
        model,
        rows[2:],
        limits,
        settlements_expected=0,
        settlements_reconstructed=0,
    )

    assert not report["passed"]
    assert not report["gates"]["settlements_exactly_reconstructed"]
    assert not report["gates"]["error_small_relative_to_strike_distance"]


def test_lag_sweep_marks_a_positive_best_shift_noncausal():
    pyth = []
    futures = []
    for day_index, day in enumerate(("2026-01-01", "2026-01-02")):
        base = day_index * 10_000_000_000
        future_values = (30.0, 31.0, 34.0, 35.0, 40.0)
        for index, future in enumerate(future_values):
            timestamp = base + (index + 1) * 1_000_000_000
            # The useful futures value arrives 100 ms after each Pyth timestamp.
            pyth.append(
                _event(60 + 2 * (future - 30), timestamp, 2, day=day)
            )
            futures.append(
                _event(future, timestamp + 100_000_000, 1, day=day)
            )

    report = run_replicator_lag_sweep(
        pyth,
        futures,
        training_days=["2026-01-01"],
        heldout_days=["2026-01-02"],
        shifts_ms=(0, 100),
    )

    assert report["best_shift_ms"] == 100
    assert report["best_shift_is_noncausal"]
    assert report["best_causal_shift_ms"] == 0


def _snapshot(probability, timestamp, sequence=1, maximum_input=None):
    decision = ReceiveKey(timestamp, sequence)
    return FairSnapshot(
        Decimal(probability),
        decision,
        maximum_input or decision,
    )


def test_fill_attribution_reconciles_one_second_and_settlement_identities():
    row = FillAttribution(
        fill_id="f1",
        day="2026-01-01",
        entry_cost=Decimal("0.15"),
        outcome=Decimal("1"),
        kalshi_mark_1s=Decimal("0.12"),
        fair_at_quote=_snapshot("0.18", 1_000),
        fair_at_fill=_snapshot("0.16", 2_000),
        fair_at_1s=_snapshot("0.14", 3_000),
    )

    components = row.components()

    assert components["quoted_edge"] == Decimal("0.03")
    assert components["external_fair_move_before_fill"] == Decimal("-0.02")
    assert components["external_fair_move_after_fill"] == Decimal("-0.02")
    assert components["residual_kalshi_flow_markout"] == Decimal("-0.02")
    assert components["kalshi_fill_to_1s_markout"] == Decimal("-0.03")
    assert components["settlement_pnl"] == Decimal("0.85")


def test_fair_snapshot_rejects_future_feature_input():
    with pytest.raises(CausalityError, match="future"):
        FairSnapshot(
            Decimal("0.5"),
            ReceiveKey(1_000, 1),
            ReceiveKey(1_001, 1),
        )


def test_attribution_summary_reports_cents_ci_and_signed_loss_shares():
    rows = [
        FillAttribution(
            fill_id="f1",
            day="2026-01-01",
            entry_cost=Decimal("0.15"),
            outcome=Decimal("0"),
            kalshi_mark_1s=Decimal("0.12"),
            fair_at_quote=_snapshot("0.18", 1_000),
            fair_at_fill=_snapshot("0.16", 2_000),
            fair_at_1s=_snapshot("0.14", 3_000),
        ),
        FillAttribution(
            fill_id="f2",
            day="2026-01-02",
            entry_cost=Decimal("0.15"),
            outcome=Decimal("1"),
            kalshi_mark_1s=Decimal("0.12"),
            fair_at_quote=_snapshot("0.18", 4_000),
            fair_at_fill=_snapshot("0.16", 5_000),
            fair_at_1s=_snapshot("0.14", 6_000),
        ),
    ]

    summary = summarize_attribution(rows, bootstrap_samples=50, bootstrap_seed=7)
    by_component = {row["component"]: row for row in summary}

    assert by_component["kalshi_fill_to_1s_markout"]["mean_cents"] == Decimal(
        "-3.00"
    )
    assert by_component["external_fair_move_before_fill"][
        "signed_share_of_total_markout_pct"
    ] == Decimal("66.66666666666666666666666667")
    component_share = sum(
        by_component[name]["signed_share_of_total_markout_pct"]
        for name in (
            "quoted_edge",
            "external_fair_move_before_fill",
            "external_fair_move_after_fill",
            "residual_kalshi_flow_markout",
        )
    )
    assert component_share == Decimal("100.0000000000000000000000000")
