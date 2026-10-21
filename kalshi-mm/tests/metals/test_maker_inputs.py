from decimal import Decimal

import pytest

from src.metals.maker_inputs import AnchoredXagUpdater, ManualControl, PriceObservation


MSEC = 1_000_000


def _control(**overrides):
    payload = {
        "version": 1,
        "effective_ts_ns": 1000 * MSEC,
        "expires_ts_ns": 2000 * MSEC,
        "volatility_multiplier": "1.20",
        "paused": False,
        "reason": "Operator assessment before the session",
    }
    return ManualControl.from_dict(payload | overrides)


def _price(value, source_ms, receive_ms, sequence):
    return PriceObservation(
        Decimal(value), source_ms * MSEC, receive_ms * MSEC, sequence
    )


def _delayed_anchor():
    updater = AnchoredXagUpdater()
    updater.observe_fast(_price("100", 1000, 1010, 1))
    updater.observe_fast(_price("100.1", 1200, 1210, 2))
    updater.observe_exact(_price("50", 1000, 1400, 3))
    return updater


def test_manual_control_is_explicit_roundtrippable_and_expiring():
    control = _control()
    assert control.volatility_multiplier == Decimal("1.20")
    assert ManualControl.from_dict(control.public_dict()) == control
    assert control.rejection_reason(999 * MSEC) == "manual_control_not_effective"
    assert control.rejection_reason(1000 * MSEC) is None
    assert control.rejection_reason(1999 * MSEC) is None
    assert control.rejection_reason(2000 * MSEC) == "manual_control_expired"
    assert _control(paused=True).rejection_reason(1500 * MSEC) == "manual_control_paused"


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": True},
        {"version": 0},
        {"version": "1"},
        {"effective_ts_ns": 1000.0},
        {"effective_ts_ns": -1},
        {"expires_ts_ns": 1000 * MSEC},
        {"expires_ts_ns": False},
        {"paused": "false"},
        {"paused": 0},
        {"reason": None},
        {"volatility_multiplier": True},
        {"volatility_multiplier": "NaN"},
        {"volatility_multiplier": "Infinity"},
        {"volatility_multiplier": "oops"},
        {"volatility_multiplier": "0.249"},
        {"volatility_multiplier": "4.01"},
        {"unknown_control": "ignored typo would be unsafe"},
    ],
)
def test_manual_control_rejects_ambiguous_or_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        _control(**overrides)


def test_manual_control_does_not_coerce_boolean_or_non_decimal_direct_inputs():
    with pytest.raises(ValueError, match="must be Decimal"):
        ManualControl(1, 1000, 2000, 1.2, False, "reason")
    with pytest.raises(ValueError, match="required schema"):
        ManualControl.from_dict({"version": 1})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value": Decimal("NaN")},
        {"value": Decimal("Infinity")},
        {"value": Decimal("0")},
        {"value": Decimal("-1")},
        {"value": 50.0},
        {"source_ts_ns": 0},
        {"source_ts_ns": True},
        {"receive_ts_ns": 1},
        {"receive_ts_ns": 200.0},
        {"ingestion_sequence": True},
        {"ingestion_sequence": -1},
    ],
)
def test_price_observation_rejects_invalid_or_acausal_values(kwargs):
    payload = {
        "value": Decimal("50"), "source_ts_ns": 100,
        "receive_ts_ns": 200, "ingestion_sequence": 0,
    }
    with pytest.raises(ValueError):
        PriceObservation(**(payload | kwargs))


def test_delayed_exact_arrival_retains_fast_move_since_source_not_receipt():
    updater = _delayed_anchor()
    estimate = updater.estimate(1450 * MSEC)
    assert estimate["estimated_index"] == Decimal("50.05")
    assert estimate["exact_anchor"]["source_ts_ns"] == 1000 * MSEC
    assert estimate["exact_anchor"]["receive_ts_ns"] == 1400 * MSEC
    assert estimate["anchor_fast"]["ingestion_sequence"] == 1
    assert estimate["current_fast"]["ingestion_sequence"] == 2
    assert estimate["assumed_return_beta"] == "1"
    assert estimate["fast_source_age_ms"] == 250
    assert estimate["fast_receive_age_ms"] == 240


def test_refreshing_exact_anchor_does_not_apply_prior_fast_move_twice():
    updater = _delayed_anchor()
    updater.observe_fast(_price("100.2", 1400, 1450, 4))
    previous = updater.estimate(1450 * MSEC)
    updater.observe_exact(_price("50.05", 1200, 1550, 5))
    refreshed = updater.estimate(1550 * MSEC)
    assert previous["estimated_index"] == Decimal("50.10")
    assert refreshed["estimated_index"] == pytest.approx(Decimal("50.10"))
    assert refreshed["anchor_fast"]["source_ts_ns"] == 1200 * MSEC
    assert refreshed["current_fast"]["source_ts_ns"] == 1400 * MSEC
    assert Decimal(refreshed["fast_move_bps"]) < Decimal(previous["fast_move_bps"])


def test_source_reordering_never_moves_latest_fast_or_exact_backward():
    updater = _delayed_anchor()
    updater.observe_fast(_price("99.9", 900, 1450, 4))
    updater.observe_exact(_price("49", 800, 1500, 5))
    estimate = updater.estimate(1500 * MSEC)
    assert estimate["estimated_index"] == Decimal("50.05")
    assert updater.latest_fast.source_ts_ns == 1200 * MSEC
    assert updater.latest_exact.source_ts_ns == 1000 * MSEC


def test_delayed_older_fast_sample_is_available_to_a_later_exact_anchor():
    updater = AnchoredXagUpdater()
    updater.observe_fast(_price("100.1", 1200, 1210, 1))
    updater.observe_fast(_price("100", 1000, 1250, 2))
    updater.observe_exact(_price("50", 1000, 1400, 3))
    estimate = updater.estimate(1450 * MSEC)
    assert estimate["estimated_index"] == Decimal("50.05")
    assert estimate["anchor_fast"]["ingestion_sequence"] == 2
    assert estimate["current_fast"]["ingestion_sequence"] == 1


def test_fast_sample_arriving_after_exact_does_not_retroactively_create_anchor():
    updater = AnchoredXagUpdater()
    updater.observe_exact(_price("50", 1000, 1100, 1))
    updater.observe_fast(_price("100", 1000, 1110, 2))
    with pytest.raises(ValueError, match="missing_source_aligned_fast_anchor"):
        updater.estimate(1120 * MSEC)
    updater.observe_fast(_price("100.1", 1200, 1210, 3))
    updater.observe_exact(_price("50.05", 1200, 1300, 4))
    assert updater.estimate(1300 * MSEC)["estimated_index"] == Decimal("50.05")


def test_duplicate_publications_cannot_refresh_receipt_or_source_freshness():
    updater = _delayed_anchor()
    updater.observe_fast(_price("100.1", 1200, 1900, 4))
    updater.observe_exact(_price("50", 1000, 1901, 5))
    assert updater.latest_fast.receive_ts_ns == 1210 * MSEC
    assert updater.latest_exact.receive_ts_ns == 1400 * MSEC
    with pytest.raises(ValueError, match="fast_source_stale"):
        updater.estimate(1951 * MSEC)
    with pytest.raises(ValueError, match="exact_source_stale"):
        updater.estimate(3001 * MSEC, max_fast_age_ms=5000)


def test_new_source_publication_with_unchanged_price_is_a_valid_refresh():
    updater = _delayed_anchor()
    updater.observe_fast(_price("100.1", 1800, 1810, 4))
    estimate = updater.estimate(1900 * MSEC)
    assert estimate["estimated_index"] == Decimal("50.05")
    assert estimate["current_fast"]["source_ts_ns"] == 1800 * MSEC


@pytest.mark.parametrize("stream", ["fast", "exact"])
def test_conflicting_duplicate_publications_are_rejected(stream):
    updater = _delayed_anchor()
    observe = getattr(updater, f"observe_{stream}")
    source = 1200 if stream == "fast" else 1000
    with pytest.raises(ValueError, match=f"conflicting_{stream}_publication"):
        observe(_price("999", source, 1500, 4))


def test_source_age_protects_against_freshly_received_but_delayed_fast_price():
    updater = AnchoredXagUpdater()
    updater.observe_fast(_price("100", 1000, 2000, 1))
    updater.observe_exact(_price("50", 1000, 2100, 2))
    with pytest.raises(ValueError, match="fast_source_stale"):
        updater.estimate(2100 * MSEC)


def test_source_age_protects_against_freshly_received_but_delayed_exact_price():
    updater = AnchoredXagUpdater()
    updater.observe_fast(_price("100", 1000, 1010, 1))
    updater.observe_fast(_price("100.1", 3000, 3010, 2))
    updater.observe_exact(_price("50", 1000, 3100, 3))
    with pytest.raises(ValueError, match="exact_source_stale"):
        updater.estimate(3100 * MSEC)


def test_excess_anchor_gap_and_large_move_fail_closed():
    updater = AnchoredXagUpdater()
    updater.observe_fast(_price("100", 700, 710, 1))
    updater.observe_fast(_price("100.1", 1200, 1210, 2))
    updater.observe_exact(_price("50", 1000, 1400, 3))
    with pytest.raises(ValueError, match="fast_anchor_gap_exceeded"):
        updater.estimate(1450 * MSEC)
    assert updater.estimate(1450 * MSEC, max_anchor_gap_ms=300)["estimated_index"] == Decimal("50.05")
    updater.observe_fast(_price("100.3", 1500, 1510, 4))
    with pytest.raises(ValueError, match="fast_move_limit_exceeded"):
        updater.estimate(1510 * MSEC, max_anchor_gap_ms=300)


def test_missing_future_and_reordered_arrivals_fail_closed():
    updater = AnchoredXagUpdater()
    with pytest.raises(ValueError, match="missing_exact_anchor"):
        updater.estimate(1000 * MSEC)
    updater.observe_exact(_price("50", 1000, 1100, 1))
    with pytest.raises(ValueError, match="missing_fast_price"):
        updater.estimate(1100 * MSEC)
    with pytest.raises(ValueError, match="decision_precedes_observation"):
        updater.estimate(1099 * MSEC)
    with pytest.raises(ValueError, match="ingestion_sequence_not_increasing"):
        updater.observe_fast(_price("100", 1000, 1200, 1))
    with pytest.raises(ValueError, match="receive_timestamp_reordered"):
        updater.observe_fast(_price("100", 1000, 1099, 2))


def test_equal_arrival_timestamps_respect_ingestion_order():
    updater = AnchoredXagUpdater()
    updater.observe_fast(_price("100", 1000, 1100, 1))
    updater.observe_exact(_price("50", 1000, 1100, 2))
    assert updater.estimate(1100 * MSEC)["estimated_index"] == Decimal("50")
    reverse = AnchoredXagUpdater()
    reverse.observe_exact(_price("50", 1000, 1100, 1))
    reverse.observe_fast(_price("100", 1000, 1100, 2))
    with pytest.raises(ValueError, match="missing_source_aligned_fast_anchor"):
        reverse.estimate(1100 * MSEC)


def test_bounded_history_cannot_recover_an_evicted_anchor():
    updater = AnchoredXagUpdater(history_window_ms=100)
    updater.observe_fast(_price("100", 1000, 1010, 1))
    updater.observe_fast(_price("100.1", 1200, 1210, 2))
    updater.observe_exact(_price("50", 1000, 1400, 3))
    with pytest.raises(ValueError, match="missing_source_aligned_fast_anchor"):
        updater.estimate(1400 * MSEC)


@pytest.mark.parametrize(
    "kwargs",
    [{"max_fast_age_ms": -1}, {"max_anchor_gap_ms": True}, {"max_move_bps": "NaN"}],
)
def test_estimation_limits_are_validated(kwargs):
    with pytest.raises(ValueError):
        _delayed_anchor().estimate(1450 * MSEC, **kwargs)
