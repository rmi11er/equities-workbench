from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from src.metals.external_fair import ReceiveKey
from src.metals.model_boundaries import (
    PROBABILITY_DOLLARS,
    FastBenchmarkUpdate,
    FillSelectionEstimate,
    FullRepriceRequired,
    ModelBoundaryError,
    ModelInput,
    NonCausalModelInput,
    QuoteSide,
    ResearchQuoteBounds,
    SlowDistributionMetadata,
    SlowTerminalDistribution,
    SourceKind,
    compose_research_quote_bounds,
)

D = Decimal


def _input(name, source, value, timestamp, sequence=1):
    return ModelInput(name, source, D(value), ReceiveKey(timestamp, sequence))


def _distribution(probability_below="0.40", benchmark="30"):
    return SlowDistributionMetadata(
        family="gaussian",
        location_log_return=D("0.0001"),
        scale_log_return=D("0.002"),
        jump_probability=D("0.01"),
        probability_below_strike=D(probability_below),
        benchmark_price=D(benchmark),
        strike_price=D("30.10"),
        seconds_remaining=300,
        strike_rounding_rule="half_up_to_cent",
    )


def _slow(probability="0.60", inputs=None, decision=None):
    return SlowTerminalDistribution(
        market_ticker="KXSILVER15M-TEST",
        yes_probability=D(probability),
        decision_key=decision or ReceiveKey(1_000, 4),
        inputs=inputs
        or (
            _input("pyth_spot", SourceKind.PYTH_BENCHMARK, "30", 900, 1),
            _input("tod_variance", SourceKind.HISTORICAL_RETURNS, "0.1", 800, 2),
        ),
        distribution=_distribution(str(ONE - D(probability))),
        model_version="slow-v1",
        research_design_id="design-v1",
    )


ONE = D("1")


def _selection(side, charge="0.02", market="KXSILVER15M-TEST", timestamp=1_100):
    source = SourceKind.KALSHI_BOOK if side is QuoteSide.BID else SourceKind.QUEUE_STATE
    return FillSelectionEstimate(
        market_ticker=market,
        side=side,
        expected_adverse_selection=D(charge),
        decision_key=ReceiveKey(timestamp, 5),
        inputs=(_input(f"{side.value}_state", source, "1", timestamp, 4),),
        horizon_ms=1_000,
        model_version="selection-v1",
    )


def _fast(current="31.01", limit="5"):
    slow = _slow()
    return FastBenchmarkUpdate(
        anchor=slow,
        decision_key=ReceiveKey(1_100, 5),
        anchor_si=_input("anchor_si", SourceKind.SI_FUTURES, "31", 950, 1),
        current_si=_input("current_si", SourceKind.SI_FUTURES, current, 1_100, 4),
        benchmark_dollars_per_si_dollar=D("0.9"),
        probability_delta_per_benchmark_dollar=D("0.1"),
        small_move_limit_bps=D(limit),
        model_version="fast-v1",
    )


def test_slow_distribution_is_immutable_coherent_and_research_only():
    slow = _slow()

    assert slow.usable_probability == D("0.60")
    assert slow.maximum_input_key == ReceiveKey(900, 1)
    assert slow.distribution.standardized_moneyness.is_finite()
    assert slow.trading_eligible is False
    assert slow.execution_mode == "research_only"
    with pytest.raises(FrozenInstanceError):
        slow.yes_probability = D("0.9")


def test_slow_distribution_rejects_execution_source_and_model_output_leakage():
    kalshi_input = _input("kalshi_midpoint", SourceKind.KALSHI_BOOK, "0.55", 900, 1)
    with pytest.raises(ModelBoundaryError, match="forbids source"):
        _slow(inputs=(kalshi_input,))

    selection = _selection(QuoteSide.BID, timestamp=900)
    with pytest.raises(ModelBoundaryError, match="smuggled"):
        _slow(inputs=(selection,))


def test_slow_distribution_rejects_future_inputs_and_incoherent_tail():
    future = _input("pyth", SourceKind.PYTH_BENCHMARK, "30", 1_001, 1)
    with pytest.raises(NonCausalModelInput, match="arrived after"):
        _slow(inputs=(future,))

    with pytest.raises(ModelBoundaryError, match="incoherent"):
        SlowTerminalDistribution(
            market_ticker="KXSILVER15M-TEST",
            yes_probability=D("0.60"),
            decision_key=ReceiveKey(1_000, 4),
            inputs=(_input("pyth", SourceKind.PYTH_BENCHMARK, "30", 900),),
            distribution=_distribution("0.50"),
            model_version="slow-v1",
            research_design_id="design-v1",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scale_log_return", "0", "scale_log_return"),
        ("jump_probability", "1.1", "jump_probability"),
        ("benchmark_price", "NaN", "benchmark_price"),
    ],
)
def test_distribution_metadata_rejects_invalid_values(field, value, message):
    kwargs = {
        "family": "gaussian",
        "location_log_return": D("0"),
        "scale_log_return": D("0.01"),
        "jump_probability": D("0.01"),
        "probability_below_strike": D("0.4"),
        "benchmark_price": D("30"),
        "strike_price": D("30.1"),
        "seconds_remaining": 60,
    }
    kwargs[field] = D(value)

    with pytest.raises(ModelBoundaryError, match=message):
        SlowDistributionMetadata(**kwargs)


def test_small_fast_update_uses_causal_si_delta_and_remains_research_only():
    fast = _fast()

    assert not fast.full_reprice_required
    assert fast.estimated_benchmark_price == D("30.009")
    assert fast.usable_probability == D("0.6009")
    assert fast.maximum_input_key == ReceiveKey(1_100, 4)
    assert fast.trading_eligible is False


def test_large_fast_update_fails_closed_until_full_reprice():
    fast = _fast(current="31.10", limit="5")

    assert fast.full_reprice_required
    assert fast.approximate_probability is None
    assert fast.full_reprice_reasons == ("benchmark_move_exceeds_delta_limit",)
    with pytest.raises(FullRepriceRequired, match="outside"):
        _ = fast.usable_probability


def test_fast_update_rejects_non_si_and_noncausal_observations():
    slow = _slow()
    kwargs = {
        "anchor": slow,
        "decision_key": ReceiveKey(1_100, 5),
        "anchor_si": _input("anchor", SourceKind.SI_FUTURES, "31", 950),
        "current_si": _input("current", SourceKind.KALSHI_BOOK, "31.01", 1_100),
        "benchmark_dollars_per_si_dollar": D("1"),
        "probability_delta_per_benchmark_dollar": D("0.1"),
        "small_move_limit_bps": D("5"),
        "model_version": "fast-v1",
    }
    with pytest.raises(ModelBoundaryError, match="SI futures"):
        FastBenchmarkUpdate(**kwargs)

    kwargs["current_si"] = _input("current", SourceKind.SI_FUTURES, "31.01", 1_101)
    with pytest.raises(NonCausalModelInput, match="after"):
        FastBenchmarkUpdate(**kwargs)


def test_fast_update_rejects_negative_yes_delta():
    slow = _slow()
    with pytest.raises(ModelBoundaryError, match="nonnegative"):
        FastBenchmarkUpdate(
            anchor=slow,
            decision_key=ReceiveKey(1_100, 5),
            anchor_si=_input("anchor", SourceKind.SI_FUTURES, "31", 950),
            current_si=_input("current", SourceKind.SI_FUTURES, "31.01", 1_100),
            benchmark_dollars_per_si_dollar=D("1"),
            probability_delta_per_benchmark_dollar=D("-0.1"),
            small_move_limit_bps=D("5"),
            model_version="fast-v1",
        )


def test_fill_selection_accepts_execution_and_causal_movement_features():
    estimate = _selection(QuoteSide.BID)

    assert estimate.unit == PROBABILITY_DOLLARS
    assert estimate.trading_eligible is False

    si_estimate = FillSelectionEstimate(
        market_ticker="KXSILVER15M-TEST",
        side=QuoteSide.BID,
        expected_adverse_selection=D("0.02"),
        decision_key=ReceiveKey(1_100, 5),
        inputs=(
            _input("si_momentum", SourceKind.SI_FUTURES, "0.1", 1_000),
            _input("fair_move", SourceKind.EXTERNAL_FAIR_STATE, "0.01", 1_050),
        ),
        horizon_ms=1_000,
        model_version="selection-v1",
    )
    assert si_estimate.maximum_input_key == ReceiveKey(1_050, 1)

    with pytest.raises(ModelBoundaryError, match="forbids source"):
        FillSelectionEstimate(
            market_ticker="KXSILVER15M-TEST",
            side=QuoteSide.BID,
            expected_adverse_selection=D("0.02"),
            decision_key=ReceiveKey(1_100, 5),
            inputs=(
                _input(
                    "option_skew",
                    SourceKind.EXTERNAL_OPTIONS,
                    "0.1",
                    1_000,
                ),
            ),
            horizon_ms=1_000,
            model_version="selection-v1",
        )
    with pytest.raises(ModelBoundaryError, match="between zero and one"):
        _selection(QuoteSide.BID, charge="-0.01")


def test_fill_selection_rejects_future_execution_feature():
    with pytest.raises(NonCausalModelInput, match="arrived after"):
        FillSelectionEstimate(
            market_ticker="KXSILVER15M-TEST",
            side=QuoteSide.BID,
            expected_adverse_selection=D("0.02"),
            decision_key=ReceiveKey(1_100, 5),
            inputs=(_input("book", SourceKind.KALSHI_BOOK, "1", 1_101),),
            horizon_ms=1_000,
            model_version="selection-v1",
        )


def test_quote_composition_preserves_side_and_inventory_sign_conventions():
    slow = _slow()
    bounds = compose_research_quote_bounds(
        slow,
        _selection(QuoteSide.BID, "0.02"),
        _selection(QuoteSide.OFFER, "0.03"),
        decision_key=ReceiveKey(1_100, 6),
        uncertainty_charge=D("0.01"),
        bid_fee_charge=D("0.005"),
        offer_fee_charge=D("0.006"),
        inventory_charge=D("0.04"),
    )

    assert bounds.raw_bid_ceiling == D("0.525")
    assert bounds.raw_offer_floor == D("0.606")
    assert bounds.bid_ceiling == D("0.525")
    assert bounds.offer_floor == D("0.606")
    assert bounds.unit == PROBABILITY_DOLLARS
    assert bounds.trading_eligible is False
    assert bounds.order_intents == ()

    short_inventory = compose_research_quote_bounds(
        slow,
        _selection(QuoteSide.BID, "0.02"),
        _selection(QuoteSide.OFFER, "0.03"),
        decision_key=ReceiveKey(1_100, 6),
        uncertainty_charge=D("0.01"),
        bid_fee_charge=D("0.005"),
        offer_fee_charge=D("0.006"),
        inventory_charge=D("-0.04"),
    )
    assert short_inventory.raw_bid_ceiling - bounds.raw_bid_ceiling == D("0.08")
    assert short_inventory.raw_offer_floor - bounds.raw_offer_floor == D("0.08")


def test_quote_composition_clips_and_suppresses_infeasible_sides():
    slow = _slow(probability="0.02")
    bounds = compose_research_quote_bounds(
        slow,
        _selection(QuoteSide.BID, "0.03"),
        _selection(QuoteSide.OFFER, "0.03"),
        decision_key=ReceiveKey(1_100, 6),
        uncertainty_charge=D("0.01"),
        inventory_charge=D("0"),
    )

    assert bounds.raw_bid_ceiling == D("-0.02")
    assert bounds.bid_ceiling == D("0")
    assert bounds.bid_suppressed
    assert not bounds.offer_suppressed


def test_quote_bounds_record_cannot_be_constructed_with_inconsistent_equations():
    with pytest.raises(ModelBoundaryError, match="quote equation"):
        ResearchQuoteBounds(
            market_ticker="KXSILVER15M-TEST",
            decision_key=ReceiveKey(1_100, 6),
            external_probability=D("0.60"),
            bid_adverse_selection=D("0.02"),
            offer_adverse_selection=D("0.03"),
            uncertainty_charge=D("0.01"),
            bid_fee_charge=D("0"),
            offer_fee_charge=D("0"),
            inventory_charge=D("0.04"),
            raw_bid_ceiling=D("0.54"),
            raw_offer_floor=D("0.60"),
            bid_ceiling=D("0.54"),
            offer_floor=D("0.60"),
            bid_suppressed=False,
            offer_suppressed=False,
        )


def test_quote_composition_rejects_wrong_side_market_and_fill_as_external_fair():
    slow = _slow()
    bid = _selection(QuoteSide.BID)
    offer = _selection(QuoteSide.OFFER)
    kwargs = {
        "decision_key": ReceiveKey(1_100, 6),
        "uncertainty_charge": D("0.01"),
        "inventory_charge": D("0"),
    }

    with pytest.raises(ModelBoundaryError, match="wrong quote side"):
        compose_research_quote_bounds(slow, offer, bid, **kwargs)
    with pytest.raises(ModelBoundaryError, match="market does not match"):
        compose_research_quote_bounds(
            slow,
            bid,
            _selection(QuoteSide.OFFER, market="OTHER"),
            **kwargs,
        )
    with pytest.raises(ModelBoundaryError, match="external-fair"):
        compose_research_quote_bounds(bid, bid, offer, **kwargs)


def test_quote_composition_refuses_fast_update_that_needs_full_reprice():
    fast = _fast(current="31.10", limit="5")

    with pytest.raises(FullRepriceRequired):
        compose_research_quote_bounds(
            fast,
            _selection(QuoteSide.BID),
            _selection(QuoteSide.OFFER),
            decision_key=ReceiveKey(1_100, 6),
            uncertainty_charge=D("0.01"),
            inventory_charge=D("0"),
        )


@pytest.mark.parametrize("value", ["-0.01", "1.01", "NaN"])
def test_quote_composition_rejects_invalid_uncertainty_cost(value):
    with pytest.raises(ModelBoundaryError, match="uncertainty_charge"):
        compose_research_quote_bounds(
            _slow(),
            _selection(QuoteSide.BID),
            _selection(QuoteSide.OFFER),
            decision_key=ReceiveKey(1_100, 6),
            uncertainty_charge=D(value),
            inventory_charge=D("0"),
        )
