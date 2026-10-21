"""Research-only boundaries for the three-model metals architecture.

The records in this module are deliberately small and immutable.  They make the
separation between external price discovery and Kalshi execution state executable:

* the slow terminal model accepts only external-market inputs;
* the fast updater accepts only causal SI futures observations; and
* the fill-selection model accepts execution state plus causal short-horizon
  external-movement predictors, but can output only an adverse-selection charge.

All quote adjustments use YES probability dollars per contract.  This module has
no exchange client, order type, or order-entry function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Protocol, runtime_checkable

from .external_fair import ReceiveKey

ZERO = Decimal(0)
ONE = Decimal(1)
TEN_THOUSAND = Decimal(10000)
PROBABILITY_TOLERANCE = Decimal("1e-12")
PROBABILITY_DOLLARS = "yes_probability_dollars_per_contract"


class ModelBoundaryError(ValueError):
    """Raised when information crosses a research-model boundary."""


class NonCausalModelInput(ModelBoundaryError):
    """Raised when a model input was not available by its decision receive key."""


class FullRepriceRequired(ModelBoundaryError):
    """Raised when the fast delta approximation cannot supply a usable fair."""


class SourceKind(str, Enum):
    """Closed source taxonomy used to enforce model ownership."""

    PYTH_BENCHMARK = "pyth_benchmark"
    SI_FUTURES = "si_futures"
    EXTERNAL_OPTIONS = "external_options"
    EXTERNAL_CALENDAR = "external_calendar"
    HISTORICAL_RETURNS = "historical_returns"
    CROSS_METAL = "cross_metal"
    EXTERNAL_FAIR_STATE = "external_fair_state"

    KALSHI_BOOK = "kalshi_book"
    KALSHI_TRADES = "kalshi_trades"
    QUEUE_STATE = "queue_state"
    LOCAL_QUOTE_STATE = "local_quote_state"


EXTERNAL_SOURCES = frozenset(
    {
        SourceKind.PYTH_BENCHMARK,
        SourceKind.SI_FUTURES,
        SourceKind.EXTERNAL_OPTIONS,
        SourceKind.EXTERNAL_CALENDAR,
        SourceKind.HISTORICAL_RETURNS,
        SourceKind.CROSS_METAL,
    }
)
EXECUTION_SOURCES = frozenset(
    {
        SourceKind.KALSHI_BOOK,
        SourceKind.KALSHI_TRADES,
        SourceKind.QUEUE_STATE,
        SourceKind.LOCAL_QUOTE_STATE,
    }
)
FILL_SELECTION_SOURCES = frozenset(
    {
        *EXECUTION_SOURCES,
        SourceKind.SI_FUTURES,
        SourceKind.PYTH_BENCHMARK,
        SourceKind.EXTERNAL_FAIR_STATE,
    }
)


def _require_finite(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ModelBoundaryError(f"{name} must be a finite Decimal")


def _require_probability(value: Decimal, name: str) -> None:
    _require_finite(value, name)
    if not ZERO <= value <= ONE:
        raise ModelBoundaryError(f"{name} must be between zero and one")


def _require_nonnegative_cost(value: Decimal, name: str) -> None:
    _require_probability(value, name)


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelBoundaryError(f"{name} is required")


@dataclass(frozen=True)
class ModelInput:
    """One scalar value with explicit source ownership and availability."""

    name: str
    source: SourceKind
    value: Decimal
    receive_key: ReceiveKey

    def __post_init__(self) -> None:
        _require_text(self.name, "input name")
        if not isinstance(self.source, SourceKind):
            raise ModelBoundaryError("input source must be a SourceKind")
        _require_finite(self.value, f"input {self.name!r} value")
        if not isinstance(self.receive_key, ReceiveKey):
            raise ModelBoundaryError("input receive_key must be a ReceiveKey")


def _validate_inputs(
    inputs: tuple[ModelInput, ...],
    *,
    allowed_sources: frozenset[SourceKind],
    decision_key: ReceiveKey,
    owner: str,
) -> None:
    if not isinstance(inputs, tuple) or not inputs:
        raise ModelBoundaryError(f"{owner} requires a nonempty input tuple")
    names = set()
    for model_input in inputs:
        if not isinstance(model_input, ModelInput):
            raise ModelBoundaryError(
                f"{owner} inputs must be ModelInput records; model outputs cannot "
                "be smuggled across the boundary"
            )
        if model_input.name in names:
            raise ModelBoundaryError(f"{owner} input names must be unique")
        names.add(model_input.name)
        if model_input.source not in allowed_sources:
            raise ModelBoundaryError(
                f"{owner} forbids source {model_input.source.value!r}"
            )
        if model_input.receive_key > decision_key:
            raise NonCausalModelInput(
                f"{owner} input {model_input.name!r} arrived after its decision"
            )


@dataclass(frozen=True)
class SlowDistributionMetadata:
    """Auditable terminal-return distribution metadata for one strike.

    ``probability_below_strike`` is the distribution mass for ``B_T < K``.
    Consequently it must complement the slow snapshot's inclusive YES tail,
    ``P(B_T >= K)``.  Location and scale are log-return units.
    """

    family: str
    location_log_return: Decimal
    scale_log_return: Decimal
    jump_probability: Decimal
    probability_below_strike: Decimal
    benchmark_price: Decimal
    strike_price: Decimal
    seconds_remaining: int
    settlement_operator: str = "greater_than_or_equal"
    strike_rounding_rule: str = "half_up_to_market_precision"

    def __post_init__(self) -> None:
        _require_text(self.family, "distribution family")
        _require_finite(self.location_log_return, "location_log_return")
        _require_finite(self.scale_log_return, "scale_log_return")
        if self.scale_log_return <= ZERO:
            raise ModelBoundaryError("scale_log_return must be positive")
        _require_probability(self.jump_probability, "jump_probability")
        _require_probability(self.probability_below_strike, "probability_below_strike")
        for name, value in (
            ("benchmark_price", self.benchmark_price),
            ("strike_price", self.strike_price),
        ):
            _require_finite(value, name)
            if value <= ZERO:
                raise ModelBoundaryError(f"{name} must be positive")
        if not isinstance(self.seconds_remaining, int) or self.seconds_remaining <= 0:
            raise ModelBoundaryError("seconds_remaining must be a positive integer")
        if self.settlement_operator != "greater_than_or_equal":
            raise ModelBoundaryError(
                "slow fair must represent the inclusive YES settlement event"
            )
        _require_text(self.strike_rounding_rule, "strike_rounding_rule")

    @property
    def standardized_moneyness(self) -> Decimal:
        return (
            (self.strike_price / self.benchmark_price).ln() - self.location_log_return
        ) / self.scale_log_return


@runtime_checkable
class ExternalProbabilitySnapshot(Protocol):
    """The minimal read-only interface accepted by quote composition."""

    market_ticker: str
    decision_key: ReceiveKey
    model_version: str
    trading_eligible: bool

    @property
    def maximum_input_key(self) -> ReceiveKey: ...

    @property
    def usable_probability(self) -> Decimal: ...


@dataclass(frozen=True)
class SlowTerminalDistribution:
    """Slow external fair, isolated from all Kalshi price and flow state."""

    market_ticker: str
    yes_probability: Decimal
    decision_key: ReceiveKey
    inputs: tuple[ModelInput, ...]
    distribution: SlowDistributionMetadata
    model_version: str
    research_design_id: str
    trading_eligible: bool = field(default=False, init=False)
    execution_mode: str = field(default="research_only", init=False)

    def __post_init__(self) -> None:
        _require_text(self.market_ticker, "market_ticker")
        _require_text(self.model_version, "model_version")
        _require_text(self.research_design_id, "research_design_id")
        _require_probability(self.yes_probability, "yes_probability")
        if not isinstance(self.decision_key, ReceiveKey):
            raise ModelBoundaryError("decision_key must be a ReceiveKey")
        if not isinstance(self.distribution, SlowDistributionMetadata):
            raise ModelBoundaryError("distribution must be SlowDistributionMetadata")
        complement_error = abs(
            self.yes_probability + self.distribution.probability_below_strike - ONE
        )
        if complement_error > PROBABILITY_TOLERANCE:
            raise ModelBoundaryError(
                "YES probability is incoherent with distribution tail metadata"
            )
        _validate_inputs(
            self.inputs,
            allowed_sources=EXTERNAL_SOURCES,
            decision_key=self.decision_key,
            owner="slow terminal distribution",
        )

    @property
    def maximum_input_key(self) -> ReceiveKey:
        return max(model_input.receive_key for model_input in self.inputs)

    @property
    def usable_probability(self) -> Decimal:
        return self.yes_probability


@dataclass(frozen=True)
class FastBenchmarkUpdate:
    """Causal SI-to-Pyth update using delta only inside a frozen move limit.

    For a move beyond ``small_move_limit_bps`` (or an approximation outside the
    probability simplex), ``usable_probability`` raises ``FullRepriceRequired``.
    Consumers must then obtain a new full terminal distribution rather than quote
    from an extrapolated delta.
    """

    anchor: SlowTerminalDistribution
    decision_key: ReceiveKey
    anchor_si: ModelInput
    current_si: ModelInput
    benchmark_dollars_per_si_dollar: Decimal
    probability_delta_per_benchmark_dollar: Decimal
    small_move_limit_bps: Decimal
    model_version: str
    market_ticker: str = field(init=False)
    estimated_benchmark_price: Decimal = field(init=False)
    approximate_probability: Decimal | None = field(init=False)
    full_reprice_required: bool = field(init=False)
    full_reprice_reasons: tuple[str, ...] = field(init=False)
    trading_eligible: bool = field(default=False, init=False)
    execution_mode: str = field(default="research_only", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.anchor, SlowTerminalDistribution):
            raise ModelBoundaryError(
                "fast updater anchor must be a slow external distribution"
            )
        if not isinstance(self.decision_key, ReceiveKey):
            raise ModelBoundaryError("decision_key must be a ReceiveKey")
        _require_text(self.model_version, "model_version")
        for name, model_input in (
            ("anchor_si", self.anchor_si),
            ("current_si", self.current_si),
        ):
            if not isinstance(model_input, ModelInput):
                raise ModelBoundaryError(f"{name} must be a ModelInput")
            if model_input.source is not SourceKind.SI_FUTURES:
                raise ModelBoundaryError(f"{name} must come from SI futures")
            if model_input.value <= ZERO:
                raise ModelBoundaryError(f"{name} price must be positive")
        if self.anchor_si.receive_key > self.anchor.decision_key:
            raise NonCausalModelInput(
                "anchor SI observation arrived after the anchor fair decision"
            )
        if self.current_si.receive_key < self.anchor_si.receive_key:
            raise NonCausalModelInput(
                "current SI observation predates its anchor observation"
            )
        if self.current_si.receive_key < self.anchor.decision_key:
            raise NonCausalModelInput(
                "current SI observation predates the anchored external fair"
            )
        if self.current_si.receive_key > self.decision_key:
            raise NonCausalModelInput(
                "current SI observation arrived after the update decision"
            )
        if self.decision_key < self.anchor.decision_key:
            raise NonCausalModelInput("fast update decision predates its anchor fair")

        for name, value in (
            (
                "benchmark_dollars_per_si_dollar",
                self.benchmark_dollars_per_si_dollar,
            ),
            (
                "probability_delta_per_benchmark_dollar",
                self.probability_delta_per_benchmark_dollar,
            ),
            ("small_move_limit_bps", self.small_move_limit_bps),
        ):
            _require_finite(value, name)
        if self.benchmark_dollars_per_si_dollar <= ZERO:
            raise ModelBoundaryError("benchmark_dollars_per_si_dollar must be positive")
        if self.probability_delta_per_benchmark_dollar < ZERO:
            raise ModelBoundaryError(
                "probability delta must be nonnegative for an inclusive YES event"
            )
        if self.small_move_limit_bps <= ZERO:
            raise ModelBoundaryError("small_move_limit_bps must be positive")

        benchmark_change = (
            self.current_si.value - self.anchor_si.value
        ) * self.benchmark_dollars_per_si_dollar
        estimate = self.anchor.distribution.benchmark_price + benchmark_change
        if estimate <= ZERO:
            raise ModelBoundaryError("estimated benchmark price must be positive")
        probability = (
            self.anchor.yes_probability
            + benchmark_change * self.probability_delta_per_benchmark_dollar
        )
        move_bps = (
            abs(benchmark_change)
            / self.anchor.distribution.benchmark_price
            * TEN_THOUSAND
        )
        reasons = []
        if move_bps > self.small_move_limit_bps:
            reasons.append("benchmark_move_exceeds_delta_limit")
        if not ZERO <= probability <= ONE:
            reasons.append("delta_probability_outside_simplex")

        object.__setattr__(self, "market_ticker", self.anchor.market_ticker)
        object.__setattr__(self, "estimated_benchmark_price", estimate)
        object.__setattr__(self, "full_reprice_reasons", tuple(reasons))
        object.__setattr__(self, "full_reprice_required", bool(reasons))
        object.__setattr__(
            self,
            "approximate_probability",
            None if reasons else probability,
        )

    @property
    def maximum_input_key(self) -> ReceiveKey:
        return max(self.anchor.maximum_input_key, self.current_si.receive_key)

    @property
    def usable_probability(self) -> Decimal:
        if self.full_reprice_required or self.approximate_probability is None:
            raise FullRepriceRequired(
                "fast delta update is outside its frozen validity region: "
                + ", ".join(self.full_reprice_reasons)
            )
        return self.approximate_probability


class QuoteSide(str, Enum):
    BID = "bid"
    OFFER = "offer"


@dataclass(frozen=True)
class FillSelectionEstimate:
    """Side-specific expected adverse selection from causal fill-state inputs.

    Expected adverse selection is a nonnegative charge in YES probability dollars
    per contract.  It is subtracted from a bid ceiling and added to an offer floor.
    SI/Pyth/fair movement may help predict selection, but this record cannot modify
    the external probability; it can only emit the side-specific charge.
    """

    market_ticker: str
    side: QuoteSide
    expected_adverse_selection: Decimal
    decision_key: ReceiveKey
    inputs: tuple[ModelInput, ...]
    horizon_ms: int
    model_version: str
    unit: str = field(default=PROBABILITY_DOLLARS, init=False)
    trading_eligible: bool = field(default=False, init=False)
    execution_mode: str = field(default="research_only", init=False)

    def __post_init__(self) -> None:
        _require_text(self.market_ticker, "market_ticker")
        _require_text(self.model_version, "model_version")
        if not isinstance(self.side, QuoteSide):
            raise ModelBoundaryError("side must be QuoteSide.BID or QuoteSide.OFFER")
        _require_nonnegative_cost(
            self.expected_adverse_selection, "expected_adverse_selection"
        )
        if not isinstance(self.decision_key, ReceiveKey):
            raise ModelBoundaryError("decision_key must be a ReceiveKey")
        if not isinstance(self.horizon_ms, int) or self.horizon_ms <= 0:
            raise ModelBoundaryError("horizon_ms must be a positive integer")
        _validate_inputs(
            self.inputs,
            allowed_sources=FILL_SELECTION_SOURCES,
            decision_key=self.decision_key,
            owner="fill-selection estimate",
        )

    @property
    def maximum_input_key(self) -> ReceiveKey:
        return max(model_input.receive_key for model_input in self.inputs)


@dataclass(frozen=True)
class ResearchQuoteBounds:
    """Research-only YES quote bounds; never an order instruction."""

    market_ticker: str
    decision_key: ReceiveKey
    external_probability: Decimal
    bid_adverse_selection: Decimal
    offer_adverse_selection: Decimal
    uncertainty_charge: Decimal
    bid_fee_charge: Decimal
    offer_fee_charge: Decimal
    inventory_charge: Decimal
    raw_bid_ceiling: Decimal
    raw_offer_floor: Decimal
    bid_ceiling: Decimal
    offer_floor: Decimal
    bid_suppressed: bool
    offer_suppressed: bool
    unit: str = field(default=PROBABILITY_DOLLARS, init=False)
    trading_eligible: bool = field(default=False, init=False)
    execution_mode: str = field(default="research_only", init=False)
    order_intents: tuple[()] = field(default=(), init=False)

    def __post_init__(self) -> None:
        _require_text(self.market_ticker, "market_ticker")
        if not isinstance(self.decision_key, ReceiveKey):
            raise ModelBoundaryError("decision_key must be a ReceiveKey")
        _require_probability(self.external_probability, "external_probability")
        _require_nonnegative_cost(self.bid_adverse_selection, "bid_adverse_selection")
        _require_nonnegative_cost(
            self.offer_adverse_selection, "offer_adverse_selection"
        )
        _require_nonnegative_cost(self.uncertainty_charge, "uncertainty_charge")
        _require_nonnegative_cost(self.bid_fee_charge, "bid_fee_charge")
        _require_nonnegative_cost(self.offer_fee_charge, "offer_fee_charge")
        _require_finite(self.inventory_charge, "inventory_charge")
        if not -ONE <= self.inventory_charge <= ONE:
            raise ModelBoundaryError(
                "inventory_charge must be within one YES probability dollar"
            )
        for name, value in (
            ("raw_bid_ceiling", self.raw_bid_ceiling),
            ("raw_offer_floor", self.raw_offer_floor),
        ):
            _require_finite(value, name)
        _require_probability(self.bid_ceiling, "bid_ceiling")
        _require_probability(self.offer_floor, "offer_floor")

        expected_raw_bid = (
            self.external_probability
            - self.bid_adverse_selection
            - self.uncertainty_charge
            - self.bid_fee_charge
            - self.inventory_charge
        )
        expected_raw_offer = (
            self.external_probability
            + self.offer_adverse_selection
            + self.uncertainty_charge
            + self.offer_fee_charge
            - self.inventory_charge
        )
        if self.raw_bid_ceiling != expected_raw_bid:
            raise ModelBoundaryError("raw_bid_ceiling violates the quote equation")
        if self.raw_offer_floor != expected_raw_offer:
            raise ModelBoundaryError("raw_offer_floor violates the quote equation")
        if self.raw_bid_ceiling > self.raw_offer_floor:
            raise ModelBoundaryError("research quote bounds cannot cross")
        if self.bid_ceiling != max(ZERO, min(ONE, self.raw_bid_ceiling)):
            raise ModelBoundaryError("bid_ceiling must be the clipped raw bid")
        if self.offer_floor != max(ZERO, min(ONE, self.raw_offer_floor)):
            raise ModelBoundaryError("offer_floor must be the clipped raw offer")
        if self.bid_suppressed is not (self.raw_bid_ceiling <= ZERO):
            raise ModelBoundaryError("bid_suppressed is inconsistent with raw bid")
        if self.offer_suppressed is not (self.raw_offer_floor >= ONE):
            raise ModelBoundaryError("offer_suppressed is inconsistent with raw offer")


def compose_research_quote_bounds(
    external_fair: ExternalProbabilitySnapshot,
    bid_selection: FillSelectionEstimate,
    offer_selection: FillSelectionEstimate,
    *,
    decision_key: ReceiveKey,
    uncertainty_charge: Decimal,
    bid_fee_charge: Decimal = ZERO,
    offer_fee_charge: Decimal = ZERO,
    inventory_charge: Decimal,
) -> ResearchQuoteBounds:
    """Compose YES-side research bounds with explicit signs and units.

    ``inventory_charge = gamma * q`` is signed: positive YES inventory lowers
    both bounds, while negative YES inventory raises both.  Raw bounds retain the
    exact research equation.  Clipped bounds stay in the probability domain, and a
    side is marked suppressed when its raw bound is outside the tradable interval.
    """

    if not isinstance(external_fair, ExternalProbabilitySnapshot):
        raise ModelBoundaryError(
            "quote composition requires an external-fair snapshot, not a "
            "fill-selection output"
        )
    if external_fair.trading_eligible:
        raise ModelBoundaryError("research boundary refuses trading-eligible fair")
    if not isinstance(decision_key, ReceiveKey):
        raise ModelBoundaryError("decision_key must be a ReceiveKey")
    if external_fair.decision_key > decision_key:
        raise NonCausalModelInput("external fair decision is later than quote decision")
    if external_fair.maximum_input_key > decision_key:
        raise NonCausalModelInput("external fair contains a future quote input")

    for name, estimate, side in (
        ("bid_selection", bid_selection, QuoteSide.BID),
        ("offer_selection", offer_selection, QuoteSide.OFFER),
    ):
        if not isinstance(estimate, FillSelectionEstimate):
            raise ModelBoundaryError(f"{name} must be FillSelectionEstimate")
        if estimate.side is not side:
            raise ModelBoundaryError(f"{name} has the wrong quote side")
        if estimate.market_ticker != external_fair.market_ticker:
            raise ModelBoundaryError(f"{name} market does not match external fair")
        if estimate.decision_key > decision_key:
            raise NonCausalModelInput(f"{name} decision is later than quote decision")
        if estimate.maximum_input_key > decision_key:
            raise NonCausalModelInput(f"{name} contains a future quote input")
        if estimate.trading_eligible:
            raise ModelBoundaryError(f"{name} must remain research-only")

    _require_nonnegative_cost(uncertainty_charge, "uncertainty_charge")
    _require_nonnegative_cost(bid_fee_charge, "bid_fee_charge")
    _require_nonnegative_cost(offer_fee_charge, "offer_fee_charge")
    _require_finite(inventory_charge, "inventory_charge")
    if not -ONE <= inventory_charge <= ONE:
        raise ModelBoundaryError(
            "inventory_charge must be within one YES probability dollar"
        )

    probability = external_fair.usable_probability
    _require_probability(probability, "external_probability")
    raw_bid = (
        probability
        - bid_selection.expected_adverse_selection
        - uncertainty_charge
        - bid_fee_charge
        - inventory_charge
    )
    raw_offer = (
        probability
        + offer_selection.expected_adverse_selection
        + uncertainty_charge
        + offer_fee_charge
        - inventory_charge
    )
    if raw_bid > raw_offer:
        raise AssertionError("quote charge signs produced a crossed research market")

    return ResearchQuoteBounds(
        market_ticker=external_fair.market_ticker,
        decision_key=decision_key,
        external_probability=probability,
        bid_adverse_selection=bid_selection.expected_adverse_selection,
        offer_adverse_selection=offer_selection.expected_adverse_selection,
        uncertainty_charge=uncertainty_charge,
        bid_fee_charge=bid_fee_charge,
        offer_fee_charge=offer_fee_charge,
        inventory_charge=inventory_charge,
        raw_bid_ceiling=raw_bid,
        raw_offer_floor=raw_offer,
        bid_ceiling=max(ZERO, min(ONE, raw_bid)),
        offer_floor=max(ZERO, min(ONE, raw_offer)),
        bid_suppressed=raw_bid <= ZERO,
        offer_suppressed=raw_offer >= ONE,
    )


__all__ = (
    "EXECUTION_SOURCES",
    "EXTERNAL_SOURCES",
    "FILL_SELECTION_SOURCES",
    "PROBABILITY_DOLLARS",
    "ExternalProbabilitySnapshot",
    "FastBenchmarkUpdate",
    "FillSelectionEstimate",
    "FullRepriceRequired",
    "ModelBoundaryError",
    "ModelInput",
    "NonCausalModelInput",
    "QuoteSide",
    "ResearchQuoteBounds",
    "SlowDistributionMetadata",
    "SlowTerminalDistribution",
    "SourceKind",
    "compose_research_quote_bounds",
)
