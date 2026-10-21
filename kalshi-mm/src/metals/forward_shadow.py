"""Live-data, exchange-isolated forward shadowing for Silver 15-minute markets.

The runner consumes Kalshi's book, trade, metadata, and exact Pyth-value streams,
computes the frozen external fair, and records counterfactual quote lifecycle.  It
has no order-entry dependency.  Fills require strict trade penetration; trades at
the quote are retained as non-fills because historical queue position is unknown.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from .baseline_fair import BaselineVolModel, load_baseline_vol_model
from .book import FixedPointOrderBook
from .config import MetalsConfig
from .opportunistic_long_vol import implied_vol_ratio
from .recorder import MetalsRecorder
from .shadow import ShadowOrderIntent
from .shadow_session import ShadowSession
from .timeutil import iso8601_to_ns
from .types import MarketMetadataVersion, SourceEnvelope

NSEC = 1_000_000_000
ONE = Decimal(1)
ZERO = Decimal(0)
FORWARD_SHADOW_SCHEMA = "metals-forward-shadow/v1"
FORWARD_SHADOW_MODEL_VERSION = "tod-gaussian-exact-kalshi-pyth-v1"
LONG_VOL_CANDIDATE_SCHEMA = "metals-long-vol-candidate/v1"
LONG_VOL_POLICY_VERSION = "long-vol-forward-shadow-v1"


class ForwardShadowError(ValueError):
    """Raised when a forward-shadow decision would be noncausal or ambiguous."""


@dataclass(frozen=True)
class LongVolLoggingPolicy:
    """Frozen candidate definition; it never grants live-order eligibility."""

    min_underdog_ask: Decimal = Decimal("0.07")
    max_underdog_ask: Decimal = Decimal("0.31")
    min_seconds_remaining: int = 120
    max_seconds_remaining: int = 900
    taker_fee_coefficient: Decimal = Decimal("0.07")
    min_net_external_edge: Decimal = Decimal("0.03")
    max_book_receive_age_ms: int = 250
    max_exact_receive_age_ms: int = 250
    persistence_ms: int = 1_000

    def __post_init__(self) -> None:
        if not ZERO < self.min_underdog_ask < self.max_underdog_ask < Decimal("0.5"):
            raise ForwardShadowError("invalid long-vol underdog ask bounds")
        if not 1 <= self.min_seconds_remaining <= self.max_seconds_remaining <= 900:
            raise ForwardShadowError("invalid long-vol remaining-time bounds")
        for name in (
            "taker_fee_coefficient",
            "min_net_external_edge",
        ):
            value = getattr(self, name)
            if not value.is_finite() or value < ZERO:
                raise ForwardShadowError(f"{name} must be finite and nonnegative")
        if min(
            self.max_book_receive_age_ms,
            self.max_exact_receive_age_ms,
            self.persistence_ms,
        ) < 0:
            raise ForwardShadowError("long-vol ages and persistence cannot be negative")

    def public_dict(self) -> dict[str, Any]:
        return {
            "policy_version": LONG_VOL_POLICY_VERSION,
            "min_underdog_ask": _decimal_text(self.min_underdog_ask),
            "max_underdog_ask": _decimal_text(self.max_underdog_ask),
            "min_seconds_remaining": self.min_seconds_remaining,
            "max_seconds_remaining": self.max_seconds_remaining,
            "taker_fee_coefficient": _decimal_text(self.taker_fee_coefficient),
            "min_net_external_edge": _decimal_text(self.min_net_external_edge),
            "max_book_receive_age_ms": self.max_book_receive_age_ms,
            "max_exact_receive_age_ms": self.max_exact_receive_age_ms,
            "persistence_ms": self.persistence_ms,
            "max_entries_per_market": 1,
        }


def _finite_decimal(value: Any, name: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except Exception as error:
        raise ForwardShadowError(f"{name} must be a finite decimal") from error
    if not parsed.is_finite():
        raise ForwardShadowError(f"{name} must be a finite decimal")
    return parsed


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _one_contract_taker_fee(
    price: Decimal, coefficient: Decimal
) -> Decimal:
    """Apply the declared one-contract fee and round up to a centicent."""

    if not ZERO <= price <= ONE:
        raise ForwardShadowError("taker fee price must be in [0, 1]")
    if coefficient < ZERO or not coefficient.is_finite():
        raise ForwardShadowError("taker fee coefficient must be nonnegative")
    raw = coefficient * price * (ONE - price)
    return raw.quantize(Decimal("0.0001"), rounding=ROUND_CEILING)


@dataclass(frozen=True)
class ForwardShadowSettings:
    """Frozen conservative policy parameters for one forward run."""

    decision_interval_ms: int = 250
    max_pyth_age_ms: int = 2_000
    min_seconds_remaining: int = 15
    quantity_fp: Decimal = Decimal(1)
    uncertainty_charge: Decimal = Decimal("0.02")
    adverse_selection_charge: Decimal = Decimal("0.01")
    fee_charge_per_contract: Decimal = Decimal("0.01")
    inventory_gamma: Decimal = Decimal("0.001")
    inventory_charge_cap: Decimal = Decimal("0.05")

    def __post_init__(self) -> None:
        if self.decision_interval_ms < 1:
            raise ForwardShadowError("decision_interval_ms must be positive")
        if self.max_pyth_age_ms < 1:
            raise ForwardShadowError("max_pyth_age_ms must be positive")
        if not 1 <= self.min_seconds_remaining < 900:
            raise ForwardShadowError("min_seconds_remaining must be in [1, 899]")
        if self.quantity_fp <= ZERO or not self.quantity_fp.is_finite():
            raise ForwardShadowError("quantity_fp must be finite and positive")
        for name in (
            "uncertainty_charge",
            "adverse_selection_charge",
            "fee_charge_per_contract",
            "inventory_gamma",
            "inventory_charge_cap",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value < ZERO:
                raise ForwardShadowError(f"{name} must be finite and nonnegative")
        if self.inventory_charge_cap > ONE:
            raise ForwardShadowError("inventory_charge_cap cannot exceed one dollar")

    def public_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "quantity_fp": _decimal_text(self.quantity_fp),
            "uncertainty_charge": _decimal_text(self.uncertainty_charge),
            "adverse_selection_charge": _decimal_text(
                self.adverse_selection_charge
            ),
            "fee_charge_per_contract": _decimal_text(
                self.fee_charge_per_contract
            ),
            "inventory_gamma": _decimal_text(self.inventory_gamma),
            "inventory_charge_cap": _decimal_text(self.inventory_charge_cap),
        }


@dataclass(frozen=True)
class ExactPythObservation:
    symbol: str
    value: Decimal
    source_ts_ns: int
    receive_ts_ns: int
    ingestion_sequence: int


@dataclass(frozen=True)
class OnlineFairDecision:
    market_ticker: str
    yes_probability: Decimal
    yes_probability_empirical: Decimal
    current_index: Decimal
    target_price: Decimal
    round_digits: int
    seconds_remaining: int
    utc_bucket: int
    time_of_day_variance_15m: Decimal
    forecast_variance_remaining: Decimal
    pyth_age_ms: Decimal
    pyth_receive_age_ms: Decimal
    pyth_source_ts_ns: int
    pyth_receive_ts_ns: int
    pyth_ingestion_sequence: int
    metadata_receive_ts_ns: int
    decision_ts_ns: int
    maximum_input_ingestion_sequence: int
    metadata_hash: str
    state_snapshot_hash: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FORWARD_SHADOW_SCHEMA,
            "market_ticker": self.market_ticker,
            "yes_probability": _decimal_text(self.yes_probability),
            "yes_probability_empirical": _decimal_text(
                self.yes_probability_empirical
            ),
            "current_index": _decimal_text(self.current_index),
            "target_price": _decimal_text(self.target_price),
            "round_digits": self.round_digits,
            "seconds_remaining": self.seconds_remaining,
            "utc_bucket": self.utc_bucket,
            "time_of_day_variance_15m": _decimal_text(
                self.time_of_day_variance_15m
            ),
            "forecast_variance_remaining": _decimal_text(
                self.forecast_variance_remaining
            ),
            "pyth_age_ms": _decimal_text(self.pyth_age_ms),
            "pyth_receive_age_ms": _decimal_text(self.pyth_receive_age_ms),
            "pyth_source_ts_ns": self.pyth_source_ts_ns,
            "pyth_receive_ts_ns": self.pyth_receive_ts_ns,
            "pyth_ingestion_sequence": self.pyth_ingestion_sequence,
            "metadata_receive_ts_ns": self.metadata_receive_ts_ns,
            "decision_ts_ns": self.decision_ts_ns,
            "maximum_input_ingestion_sequence": (
                self.maximum_input_ingestion_sequence
            ),
            "metadata_hash": self.metadata_hash,
            "state_snapshot_hash": self.state_snapshot_hash,
            "model_version": FORWARD_SHADOW_MODEL_VERSION,
            "variance_state": "time_of_day_only",
            "distribution": "gaussian",
            "distributions": ["gaussian", "empirical_cdf"],
            "pyth_transport": "kalshi_authenticated_pyth_value",
            "settlement_symbol_aligned": True,
            "execution_mode": "shadow",
            "trading_eligible": False,
        }


@dataclass(frozen=True)
class QuoteProposal:
    outcome_side: str
    fair_probability: Decimal
    bid_ceiling: Decimal
    quote_price: Decimal | None
    inventory_charge: Decimal

    def public_dict(self) -> dict[str, Any]:
        return {
            "outcome_side": self.outcome_side,
            "fair_probability": _decimal_text(self.fair_probability),
            "bid_ceiling": _decimal_text(self.bid_ceiling),
            "quote_price": (
                _decimal_text(self.quote_price)
                if self.quote_price is not None
                else None
            ),
            "inventory_charge": _decimal_text(self.inventory_charge),
        }


def compute_online_fair(
    model: BaselineVolModel,
    metadata: MarketMetadataVersion,
    pyth: ExactPythObservation,
    *,
    decision_ts_ns: int,
    max_pyth_age_ms: int,
) -> OnlineFairDecision:
    """Compute a causal TOD-only Gaussian fair from exact live index input."""

    if metadata.missing_required_fields:
        raise ForwardShadowError("market metadata is incomplete")
    if metadata.open_ts_ns is None or metadata.close_ts_ns is None:
        raise ForwardShadowError("market window timestamps are missing")
    if metadata.target_price is None or metadata.round_digits is None:
        raise ForwardShadowError("market target or rounding rule is missing")
    if metadata.pyth_symbol != pyth.symbol:
        raise ForwardShadowError("Pyth symbol does not match market settlement symbol")
    if metadata.metadata_receive_ts_ns > decision_ts_ns:
        raise ForwardShadowError("market metadata arrived after the decision")
    if pyth.receive_ts_ns > decision_ts_ns:
        raise ForwardShadowError("Pyth observation arrived after the decision")
    if pyth.source_ts_ns > pyth.receive_ts_ns:
        raise ForwardShadowError("Pyth source timestamp is later than local receipt")
    if pyth.source_ts_ns < metadata.open_ts_ns:
        raise ForwardShadowError("Pyth observation predates the market window")
    age_ns = decision_ts_ns - pyth.source_ts_ns
    if age_ns < 0 or age_ns > max_pyth_age_ms * 1_000_000:
        raise ForwardShadowError("Pyth observation is stale")
    remaining_ns = metadata.close_ts_ns - pyth.source_ts_ns
    seconds_remaining = math.ceil(remaining_ns / NSEC)
    if not 1 <= seconds_remaining <= 900:
        raise ForwardShadowError("decision is outside the 15-minute market window")
    target = _finite_decimal(metadata.target_price, "target_price")
    if target <= ZERO or pyth.value <= ZERO:
        raise ForwardShadowError("target and current index must be positive")
    bucket = int((metadata.open_ts_ns // NSEC % 86_400) // 900)
    probability = Decimal(
        str(
            model.probability_yes(
                current_index=float(pyth.value),
                strike=float(target),
                remaining_seconds=seconds_remaining,
                bucket=bucket,
                # The 21-day diagnostic supported TOD but not incremental EWMA.
                ewma_variance=model.global_variance_15m,
                round_digits=metadata.round_digits,
                distribution="gaussian",
            )
        )
    )
    empirical_probability = Decimal(
        str(
            model.probability_yes(
                current_index=float(pyth.value),
                strike=float(target),
                remaining_seconds=seconds_remaining,
                bucket=bucket,
                ewma_variance=model.global_variance_15m,
                round_digits=metadata.round_digits,
                distribution="empirical",
            )
        )
    )
    time_of_day_variance = model.time_variance(bucket)
    forecast_variance_remaining = model.full_window_variance(
        bucket, model.global_variance_15m
    ) * (seconds_remaining / 900)
    state_payload = {
        "market_ticker": metadata.market_ticker,
        "metadata_hash": metadata.metadata_hash,
        "pyth_symbol": pyth.symbol,
        "pyth_value": _decimal_text(pyth.value),
        "pyth_source_ts_ns": pyth.source_ts_ns,
        "pyth_receive_ts_ns": pyth.receive_ts_ns,
        "pyth_ingestion_sequence": pyth.ingestion_sequence,
        "decision_ts_ns": decision_ts_ns,
        "model_version": FORWARD_SHADOW_MODEL_VERSION,
    }
    state_hash = hashlib.sha256(
        json.dumps(state_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return OnlineFairDecision(
        market_ticker=metadata.market_ticker,
        yes_probability=probability,
        yes_probability_empirical=empirical_probability,
        current_index=pyth.value,
        target_price=target,
        round_digits=metadata.round_digits,
        seconds_remaining=seconds_remaining,
        utc_bucket=bucket,
        time_of_day_variance_15m=Decimal(str(time_of_day_variance)),
        forecast_variance_remaining=Decimal(
            str(forecast_variance_remaining)
        ),
        pyth_age_ms=Decimal(age_ns) / Decimal(1_000_000),
        pyth_receive_age_ms=Decimal(
            decision_ts_ns - pyth.receive_ts_ns
        )
        / Decimal(1_000_000),
        pyth_source_ts_ns=pyth.source_ts_ns,
        pyth_receive_ts_ns=pyth.receive_ts_ns,
        pyth_ingestion_sequence=pyth.ingestion_sequence,
        metadata_receive_ts_ns=metadata.metadata_receive_ts_ns,
        decision_ts_ns=decision_ts_ns,
        maximum_input_ingestion_sequence=pyth.ingestion_sequence,
        metadata_hash=metadata.metadata_hash,
        state_snapshot_hash=state_hash,
    )


def _price_ranges(metadata: MarketMetadataVersion) -> list[dict[str, Decimal]]:
    try:
        raw_ranges = json.loads(metadata.price_ranges_json)
        return [
            {
                "start": _finite_decimal(value["start"], "price range start"),
                "end": _finite_decimal(value["end"], "price range end"),
                "step": _finite_decimal(value["step"], "price range step"),
            }
            for value in raw_ranges
        ]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ForwardShadowError("market price ranges are malformed") from error


def _step_at(price: Decimal, metadata: MarketMetadataVersion) -> Decimal:
    for value in _price_ranges(metadata):
        if value["start"] <= price < value["end"] and value["step"] > ZERO:
            return value["step"]
    raise ForwardShadowError("book price is outside the market price ranges")


def _quote_at_or_improve(
    *,
    best_bid: Decimal,
    best_ask: Decimal,
    bid_ceiling: Decimal,
    metadata: MarketMetadataVersion,
) -> Decimal | None:
    if best_bid > bid_ceiling or not ZERO < best_bid < best_ask < ONE:
        return None
    step = _step_at(best_bid, metadata)
    improved = best_bid + step
    if improved <= bid_ceiling and improved < best_ask:
        return improved
    return best_bid


class ForwardShadowEngine:
    """Pure causal fair/quote state plus exchange-isolated shadow lifecycle."""

    def __init__(
        self,
        model: BaselineVolModel,
        study_id: str,
        settings: ForwardShadowSettings,
        session: ShadowSession,
        long_vol_policy: LongVolLoggingPolicy | None = None,
    ) -> None:
        self.model = model
        self.study_id = study_id
        self.settings = settings
        self.long_vol_policy = long_vol_policy or LongVolLoggingPolicy()
        self.session = session
        self.markets: dict[str, MarketMetadataVersion] = {}
        self.latest_pyth: dict[str, ExactPythObservation] = {}
        self.active_quotes: dict[tuple[str, str], str] = {}
        self.last_decision_ts_ns: dict[str, int] = {}
        self.long_vol_streak_start_ns: dict[tuple[str, str], int] = {}
        self.counts: dict[str, int] = {
            "fair_decisions": 0,
            "long_vol_candidates": 0,
            "long_vol_screen_eligible": 0,
            "long_vol_shadow_policy_eligible": 0,
            "quote_placements": 0,
            "quote_cancels": 0,
            "market_rollover_cancels": 0,
            "strict_penetration_fills": 0,
            "touches_not_filled": 0,
            "skipped_decisions": 0,
        }

    def observe_market(self, metadata: MarketMetadataVersion) -> None:
        self.markets[metadata.market_ticker] = metadata

    def observe_pyth(self, observation: ExactPythObservation) -> None:
        self.latest_pyth[observation.symbol] = observation

    def _inventory_charge(self, market_ticker: str) -> Decimal:
        position = self.session.engine.positions.get(market_ticker)
        net = position.net_outcome_exposure if position is not None else ZERO
        raw = self.settings.inventory_gamma * net
        cap = self.settings.inventory_charge_cap
        return max(-cap, min(cap, raw))

    def _proposal(
        self,
        fair: OnlineFairDecision,
        metadata: MarketMetadataVersion,
        book: FixedPointOrderBook,
        side: str,
    ) -> QuoteProposal:
        inventory = self._inventory_charge(metadata.market_ticker)
        if side == "yes":
            side_fair = fair.yes_probability
            side_inventory = inventory
            best_bid = book.best_yes_bid()
            best_ask = book.best_yes_ask()
        else:
            side_fair = ONE - fair.yes_probability
            side_inventory = -inventory
            best_bid = max(map(Decimal, book.no_bids)) if book.no_bids else None
            best_ask = ONE - book.best_yes_bid() if book.best_yes_bid() else None
        charge = (
            self.settings.uncertainty_charge
            + self.settings.adverse_selection_charge
            + self.settings.fee_charge_per_contract
            + side_inventory
        )
        ceiling = side_fair - charge
        quote = None
        if best_bid is not None and best_ask is not None and ceiling > ZERO:
            quote = _quote_at_or_improve(
                best_bid=best_bid,
                best_ask=best_ask,
                bid_ceiling=ceiling,
                metadata=metadata,
            )
        return QuoteProposal(side, side_fair, ceiling, quote, side_inventory)

    def _cancel(self, key: tuple[str, str], event_ts_ns: int) -> None:
        quote_id = self.active_quotes.pop(key, None)
        if quote_id is None or quote_id not in self.session.engine.resting_orders:
            return
        self.session.cancel(quote_id, event_ts_ns)
        self.counts["quote_cancels"] += 1

    def _sync_quote(
        self,
        proposal: QuoteProposal,
        fair: OnlineFairDecision,
        metadata: MarketMetadataVersion,
        book: FixedPointOrderBook,
        input_ingestion_sequence: int,
    ) -> str:
        key = (metadata.market_ticker, proposal.outcome_side)
        quote_id = self.active_quotes.get(key)
        resting = self.session.engine.resting_orders.get(quote_id or "")
        if proposal.quote_price is None:
            self._cancel(key, fair.decision_ts_ns)
            return "no_quote"
        if resting is not None and resting.intent.price == proposal.quote_price:
            return "unchanged"
        self._cancel(key, fair.decision_ts_ns)
        state_hash = hashlib.sha256(
            (
                fair.state_snapshot_hash
                + book.state_hash()
                + proposal.outcome_side
                + _decimal_text(proposal.quote_price)
            ).encode()
        ).hexdigest()
        new_id = (
            f"forward:{metadata.market_ticker}:{proposal.outcome_side}:"
            f"{input_ingestion_sequence}"
        )
        intent = ShadowOrderIntent(
            logical_quote_id=new_id,
            market_ticker=metadata.market_ticker,
            metal=metadata.metal,
            outcome_side=proposal.outcome_side,
            price_dollars=_decimal_text(proposal.quote_price),
            quantity_fp=_decimal_text(self.settings.quantity_fp),
            decision_ts_ns=fair.decision_ts_ns,
            strategy_module="S2_forward_shadow",
            model_version=FORWARD_SHADOW_MODEL_VERSION,
            config_version=self.session.config.shadow.config_version,
            state_snapshot_hash=state_hash,
            worst_case_fee_dollars=_decimal_text(
                self.settings.fee_charge_per_contract * self.settings.quantity_fp
            ),
        )
        decision, _ = self.session.place(intent)
        if not decision.approved:
            return "risk_rejected"
        self.active_quotes[key] = new_id
        self.counts["quote_placements"] += 1
        return "placed"

    def _reset_long_vol_streaks(
        self, market_ticker: str, *, keep_side: str | None = None
    ) -> None:
        for key in list(self.long_vol_streak_start_ns):
            if key[0] == market_ticker and key[1] != keep_side:
                self.long_vol_streak_start_ns.pop(key, None)

    def _record_long_vol_skip(
        self,
        metadata: MarketMetadataVersion,
        book: FixedPointOrderBook,
        *,
        decision_ts_ns: int,
        input_ingestion_sequence: int,
        trigger_source: str,
        reason: str,
        book_receive_ts_ns: int | None,
        book_local_monotonic_ts_ns: int | None,
        book_ingestion_sequence: int | None,
    ) -> None:
        self._reset_long_vol_streaks(metadata.market_ticker)
        candidate_id = hashlib.sha256(
            (
                self.session.run_id
                + metadata.market_ticker
                + str(input_ingestion_sequence)
                + reason
            ).encode()
        ).hexdigest()
        self.session.record_long_vol_candidate(
            metadata.metal,
            decision_ts_ns,
            {
                "schema_version": LONG_VOL_CANDIDATE_SCHEMA,
                "candidate_id": candidate_id,
                "policy_version": LONG_VOL_POLICY_VERSION,
                "status": "skipped",
                "market_ticker": metadata.market_ticker,
                "decision_ts_ns": decision_ts_ns,
                "trigger_source": trigger_source,
                "input_ingestion_sequence": input_ingestion_sequence,
                "book_receive_ts_ns": book_receive_ts_ns,
                "book_local_monotonic_ts_ns": book_local_monotonic_ts_ns,
                "book_ingestion_sequence": book_ingestion_sequence,
                "book_state_hash": book.state_hash(),
                "metadata_hash": metadata.metadata_hash,
                "metadata_receive_ts_ns": metadata.metadata_receive_ts_ns,
                "screen_eligible": False,
                "shadow_policy_eligible": False,
                "execution_eligible": False,
                "screen_rejection_reasons": [f"fair_unavailable:{reason}"],
                "shadow_policy_rejection_reasons": [f"fair_unavailable:{reason}"],
                "execution_rejection_reasons": [
                    "fair_unavailable",
                    "fast_external_proxy_not_in_online_fair",
                    "fill_selection_model_unfrozen",
                    "live_orders_disabled",
                ],
                "long_vol_policy": self.long_vol_policy.public_dict(),
                "execution_mode": "shadow",
                "trading_eligible": False,
                "derivation_is_replayable_from_raw": True,
            },
        )
        self.counts["long_vol_candidates"] += 1

    def _record_long_vol_candidate(
        self,
        fair: OnlineFairDecision,
        metadata: MarketMetadataVersion,
        book: FixedPointOrderBook,
        *,
        input_ingestion_sequence: int,
        trigger_source: str,
        book_receive_ts_ns: int | None,
        book_local_monotonic_ts_ns: int | None,
        book_ingestion_sequence: int | None,
    ) -> None:
        policy = self.long_vol_policy
        yes_key = max(book.yes_bids, key=Decimal)
        no_key = max(book.no_bids, key=Decimal)
        yes_bid = Decimal(yes_key)
        no_bid = Decimal(no_key)
        yes_ask = ONE - no_bid
        no_ask = ONE - yes_bid
        midpoint = (yes_bid + yes_ask) / 2

        screen_rejections: list[str] = []
        underdog_side: str | None
        if midpoint < Decimal("0.5"):
            underdog_side = "yes"
            underdog_bid = yes_bid
            underdog_ask = yes_ask
            gaussian_fair = fair.yes_probability
            empirical_fair = fair.yes_probability_empirical
        elif midpoint > Decimal("0.5"):
            underdog_side = "no"
            underdog_bid = no_bid
            underdog_ask = no_ask
            gaussian_fair = ONE - fair.yes_probability
            empirical_fair = ONE - fair.yes_probability_empirical
        else:
            underdog_side = None
            underdog_bid = None
            underdog_ask = None
            gaussian_fair = None
            empirical_fair = None
            screen_rejections.append("no_unique_underdog")

        conservative_fair = (
            min(gaussian_fair, empirical_fair)
            if gaussian_fair is not None and empirical_fair is not None
            else None
        )
        taker_fee = (
            _one_contract_taker_fee(
                underdog_ask, policy.taker_fee_coefficient
            )
            if underdog_ask is not None
            else None
        )
        gross_edge = (
            conservative_fair - underdog_ask
            if conservative_fair is not None and underdog_ask is not None
            else None
        )
        net_edge_after_fee = (
            gross_edge - taker_fee
            if gross_edge is not None and taker_fee is not None
            else None
        )
        charged_edge = (
            net_edge_after_fee
            - self.settings.uncertainty_charge
            - self.settings.adverse_selection_charge
            if net_edge_after_fee is not None
            else None
        )

        if underdog_ask is not None and not (
            policy.min_underdog_ask
            <= underdog_ask
            <= policy.max_underdog_ask
        ):
            screen_rejections.append("outside_underdog_ask_band")
        if not (
            policy.min_seconds_remaining
            <= fair.seconds_remaining
            <= policy.max_seconds_remaining
        ):
            screen_rejections.append("outside_remaining_time_band")
        if (
            net_edge_after_fee is None
            or net_edge_after_fee < policy.min_net_external_edge
        ):
            screen_rejections.append("net_external_edge_below_hurdle")
        screen_eligible = not screen_rejections

        persistence_ms = Decimal(0)
        if underdog_side is not None:
            self._reset_long_vol_streaks(
                metadata.market_ticker, keep_side=underdog_side
            )
            streak_key = (metadata.market_ticker, underdog_side)
            if screen_eligible:
                start = self.long_vol_streak_start_ns.setdefault(
                    streak_key, fair.decision_ts_ns
                )
                persistence_ms = Decimal(fair.decision_ts_ns - start) / Decimal(
                    1_000_000
                )
            else:
                self.long_vol_streak_start_ns.pop(streak_key, None)
        else:
            self._reset_long_vol_streaks(metadata.market_ticker)

        book_receive_age_ms = (
            Decimal(fair.decision_ts_ns - book_receive_ts_ns)
            / Decimal(1_000_000)
            if book_receive_ts_ns is not None
            else None
        )
        shadow_rejections = list(screen_rejections)
        if book_receive_age_ms is None:
            shadow_rejections.append("book_lineage_unavailable")
        elif not ZERO <= book_receive_age_ms <= policy.max_book_receive_age_ms:
            shadow_rejections.append("book_receive_age_exceeded")
        if not ZERO <= fair.pyth_receive_age_ms <= policy.max_exact_receive_age_ms:
            shadow_rejections.append("exact_index_receive_age_exceeded")
        if persistence_ms < policy.persistence_ms:
            shadow_rejections.append("persistence_not_met")
        if charged_edge is None or charged_edge < policy.min_net_external_edge:
            shadow_rejections.append("charged_edge_below_hurdle")
        shadow_policy_eligible = not shadow_rejections

        ratio = (
            implied_vol_ratio(
                external_underdog_fair=float(conservative_fair),
                market_underdog_price=float(underdog_ask),
            )
            if conservative_fair is not None and underdog_ask is not None
            else None
        )
        maximum_input_sequence = max(
            value
            for value in (
                input_ingestion_sequence,
                fair.pyth_ingestion_sequence,
                book_ingestion_sequence,
            )
            if value is not None
        )
        candidate_payload = {
            "market_ticker": metadata.market_ticker,
            "decision_ts_ns": fair.decision_ts_ns,
            "trigger_source": trigger_source,
            "input_ingestion_sequence": input_ingestion_sequence,
            "maximum_input_ingestion_sequence": maximum_input_sequence,
            "book_state_hash": book.state_hash(),
            "metadata_hash": metadata.metadata_hash,
            "pyth_ingestion_sequence": fair.pyth_ingestion_sequence,
            "policy_version": LONG_VOL_POLICY_VERSION,
        }
        candidate_id = hashlib.sha256(
            json.dumps(
                candidate_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.session.record_long_vol_candidate(
            metadata.metal,
            fair.decision_ts_ns,
            {
                "schema_version": LONG_VOL_CANDIDATE_SCHEMA,
                "candidate_id": candidate_id,
                "policy_version": LONG_VOL_POLICY_VERSION,
                "status": "evaluated",
                "market_ticker": metadata.market_ticker,
                "window_open_ts_ns": metadata.open_ts_ns,
                "window_close_ts_ns": metadata.close_ts_ns,
                "decision_ts_ns": fair.decision_ts_ns,
                "trigger_source": trigger_source,
                "input_ingestion_sequence": input_ingestion_sequence,
                "maximum_input_ingestion_sequence": maximum_input_sequence,
                "pyth_ingestion_sequence": fair.pyth_ingestion_sequence,
                "book_receive_ts_ns": book_receive_ts_ns,
                "book_local_monotonic_ts_ns": book_local_monotonic_ts_ns,
                "book_ingestion_sequence": book_ingestion_sequence,
                "book_receive_age_ms": (
                    _decimal_text(book_receive_age_ms)
                    if book_receive_age_ms is not None
                    else None
                ),
                "book_state_hash": book.state_hash(),
                "recovery_snapshot_id": book.recovery_snapshot_id,
                "metadata_hash": metadata.metadata_hash,
                "metadata_receive_ts_ns": metadata.metadata_receive_ts_ns,
                "target_price": _decimal_text(fair.target_price),
                "round_digits": fair.round_digits,
                "seconds_remaining": fair.seconds_remaining,
                "exact_index": {
                    "symbol": metadata.pyth_symbol,
                    "value": _decimal_text(fair.current_index),
                    "source_ts_ns": fair.pyth_source_ts_ns,
                    "receive_ts_ns": fair.pyth_receive_ts_ns,
                    "source_age_ms": _decimal_text(fair.pyth_age_ms),
                    "receive_age_ms": _decimal_text(fair.pyth_receive_age_ms),
                    "ingestion_sequence": fair.pyth_ingestion_sequence,
                },
                "book": {
                    "yes_bid": _decimal_text(yes_bid),
                    "yes_ask": _decimal_text(yes_ask),
                    "yes_bid_quantity_fp": book.yes_bids[yes_key],
                    "yes_ask_quantity_fp": book.no_bids[no_key],
                    "no_bid": _decimal_text(no_bid),
                    "no_ask": _decimal_text(no_ask),
                    "no_bid_quantity_fp": book.no_bids[no_key],
                    "no_ask_quantity_fp": book.yes_bids[yes_key],
                    "yes_midpoint": _decimal_text(midpoint),
                },
                "external_fair": {
                    "baseline_study_id": self.study_id,
                    "model_version": FORWARD_SHADOW_MODEL_VERSION,
                    "variance_state": "time_of_day_only",
                    "time_of_day_variance_15m": _decimal_text(
                        fair.time_of_day_variance_15m
                    ),
                    "forecast_variance_remaining": _decimal_text(
                        fair.forecast_variance_remaining
                    ),
                    "yes_gaussian": _decimal_text(fair.yes_probability),
                    "yes_empirical_cdf": _decimal_text(
                        fair.yes_probability_empirical
                    ),
                    "kalshi_price_or_flow_used": False,
                    "fast_external_proxy_available": False,
                },
                "underdog_side": underdog_side,
                "underdog_bid": (
                    _decimal_text(underdog_bid)
                    if underdog_bid is not None
                    else None
                ),
                "underdog_ask": (
                    _decimal_text(underdog_ask)
                    if underdog_ask is not None
                    else None
                ),
                "underdog_fair_gaussian": (
                    _decimal_text(gaussian_fair)
                    if gaussian_fair is not None
                    else None
                ),
                "underdog_fair_empirical_cdf": (
                    _decimal_text(empirical_fair)
                    if empirical_fair is not None
                    else None
                ),
                "underdog_fair_conservative": (
                    _decimal_text(conservative_fair)
                    if conservative_fair is not None
                    else None
                ),
                "taker_fee_per_contract": (
                    _decimal_text(taker_fee) if taker_fee is not None else None
                ),
                "gross_external_edge": (
                    _decimal_text(gross_edge) if gross_edge is not None else None
                ),
                "net_external_edge_after_fee": (
                    _decimal_text(net_edge_after_fee)
                    if net_edge_after_fee is not None
                    else None
                ),
                "uncertainty_charge": _decimal_text(
                    self.settings.uncertainty_charge
                ),
                "adverse_selection_charge": _decimal_text(
                    self.settings.adverse_selection_charge
                ),
                "charged_external_edge": (
                    _decimal_text(charged_edge) if charged_edge is not None else None
                ),
                "implied_vol_ratio_to_external": ratio,
                "displayed_edge_persistence_ms": _decimal_text(persistence_ms),
                "screen_eligible": screen_eligible,
                "shadow_policy_eligible": shadow_policy_eligible,
                "execution_eligible": False,
                "screen_rejection_reasons": screen_rejections,
                "shadow_policy_rejection_reasons": shadow_rejections,
                "execution_rejection_reasons": [
                    "fast_external_proxy_not_in_online_fair",
                    "fill_selection_model_unfrozen",
                    "live_orders_disabled",
                ],
                "long_vol_policy": policy.public_dict(),
                "execution_mode": "shadow",
                "trading_eligible": False,
                "derivation_is_replayable_from_raw": True,
            },
        )
        self.counts["long_vol_candidates"] += 1
        if screen_eligible:
            self.counts["long_vol_screen_eligible"] += 1
        if shadow_policy_eligible:
            self.counts["long_vol_shadow_policy_eligible"] += 1

    def evaluate(
        self,
        market_ticker: str,
        book: FixedPointOrderBook,
        *,
        decision_ts_ns: int,
        input_ingestion_sequence: int,
        trigger_source: str = "unknown",
        book_receive_ts_ns: int | None = None,
        book_local_monotonic_ts_ns: int | None = None,
        book_ingestion_sequence: int | None = None,
        force: bool = False,
    ) -> OnlineFairDecision | None:
        previous = self.last_decision_ts_ns.get(market_ticker)
        minimum_ns = self.settings.decision_interval_ms * 1_000_000
        if not force and previous is not None and decision_ts_ns - previous < minimum_ns:
            return None
        self.last_decision_ts_ns[market_ticker] = decision_ts_ns
        metadata = self.markets.get(market_ticker)
        if metadata is None or not book.valid:
            self._cancel((market_ticker, "yes"), decision_ts_ns)
            self._cancel((market_ticker, "no"), decision_ts_ns)
            if metadata is not None:
                self._record_long_vol_skip(
                    metadata,
                    book,
                    decision_ts_ns=decision_ts_ns,
                    input_ingestion_sequence=input_ingestion_sequence,
                    trigger_source=trigger_source,
                    reason="Kalshi book is invalid",
                    book_receive_ts_ns=book_receive_ts_ns,
                    book_local_monotonic_ts_ns=book_local_monotonic_ts_ns,
                    book_ingestion_sequence=book_ingestion_sequence,
                )
            self.counts["skipped_decisions"] += 1
            return None
        pyth = self.latest_pyth.get(metadata.pyth_symbol or "")
        try:
            if pyth is None:
                raise ForwardShadowError("no exact Pyth observation is available")
            fair = compute_online_fair(
                self.model,
                metadata,
                pyth,
                decision_ts_ns=decision_ts_ns,
                max_pyth_age_ms=self.settings.max_pyth_age_ms,
            )
            if fair.seconds_remaining < self.settings.min_seconds_remaining:
                raise ForwardShadowError("market is inside the terminal no-quote window")
            proposals = [
                self._proposal(fair, metadata, book, side) for side in ("yes", "no")
            ]
        except ForwardShadowError as error:
            self._cancel((market_ticker, "yes"), decision_ts_ns)
            self._cancel((market_ticker, "no"), decision_ts_ns)
            self.session.record_forward_fair_decision(
                metadata.metal,
                decision_ts_ns,
                {
                    "schema_version": FORWARD_SHADOW_SCHEMA,
                    "market_ticker": market_ticker,
                    "decision_ts_ns": decision_ts_ns,
                    "input_ingestion_sequence": input_ingestion_sequence,
                    "status": "skipped",
                    "reason": str(error),
                    "execution_mode": "shadow",
                    "trading_eligible": False,
                },
            )
            self._record_long_vol_skip(
                metadata,
                book,
                decision_ts_ns=decision_ts_ns,
                input_ingestion_sequence=input_ingestion_sequence,
                trigger_source=trigger_source,
                reason=str(error),
                book_receive_ts_ns=book_receive_ts_ns,
                book_local_monotonic_ts_ns=book_local_monotonic_ts_ns,
                book_ingestion_sequence=book_ingestion_sequence,
            )
            self.counts["skipped_decisions"] += 1
            return None

        self._record_long_vol_candidate(
            fair,
            metadata,
            book,
            input_ingestion_sequence=input_ingestion_sequence,
            trigger_source=trigger_source,
            book_receive_ts_ns=book_receive_ts_ns,
            book_local_monotonic_ts_ns=book_local_monotonic_ts_ns,
            book_ingestion_sequence=book_ingestion_sequence,
        )
        actions = {
            proposal.outcome_side: self._sync_quote(
                proposal,
                fair,
                metadata,
                book,
                input_ingestion_sequence,
            )
            for proposal in proposals
        }
        self.session.record_forward_fair_decision(
            metadata.metal,
            decision_ts_ns,
            {
                **fair.public_dict(),
                "baseline_study_id": self.study_id,
                "input_ingestion_sequence": input_ingestion_sequence,
                "book_state_hash": book.state_hash(),
                "settings": self.settings.public_dict(),
                "proposals": [proposal.public_dict() for proposal in proposals],
                "quote_actions": actions,
                "fill_rule": "strict_trade_penetration_no_touch_fill",
            },
        )
        self.counts["fair_decisions"] += 1
        return fair

    def observe_trade(self, metal: str, trade: dict[str, Any]) -> None:
        ticker = str(trade.get("market_ticker", ""))
        taker_side = str(trade.get("taker_outcome_side", "")).lower()
        if taker_side not in {"yes", "no"}:
            return
        passive_side = "no" if taker_side == "yes" else "yes"
        quote_id = self.active_quotes.get((ticker, passive_side))
        resting = self.session.engine.resting_orders.get(quote_id or "")
        if resting is None:
            return
        price_field = f"{passive_side}_price_dollars"
        trade_price = _finite_decimal(trade.get(price_field), price_field)
        if trade_price == resting.intent.price:
            self.counts["touches_not_filled"] += 1
            return
        if trade_price > resting.intent.price:
            return
        quantity = min(
            resting.remaining_quantity,
            _finite_decimal(trade.get("quantity_fp"), "trade quantity"),
        )
        if quantity <= ZERO:
            return
        event_ts_ns = int(trade["receive_ts_ns"])
        fee = self.settings.fee_charge_per_contract * quantity
        event = self.session.fill(
            resting.intent.logical_quote_id,
            quantity,
            event_ts_ns,
            fee_dollars=fee,
        )
        self.session.record_shadow_fill_attribution(
            metal,
            event_ts_ns,
            {
                "schema_version": FORWARD_SHADOW_SCHEMA,
                "market_ticker": ticker,
                "logical_quote_id": resting.intent.logical_quote_id,
                "trade_id": trade.get("trade_id"),
                "passive_outcome_side": passive_side,
                "quote_price_dollars": resting.intent.price_dollars,
                "trade_price_dollars": _decimal_text(trade_price),
                "fill_quantity_fp": _decimal_text(quantity),
                "fill_fee_dollars": _decimal_text(fee),
                "fill_rule": "strict_trade_penetration",
                "source_trade_ingestion_sequence": trade.get("ingestion_sequence"),
                "execution_mode": "shadow",
                "trading_eligible": False,
            },
        )
        self.counts["strict_penetration_fills"] += 1
        if event.event_type == "full_fill":
            self.active_quotes.pop((ticker, passive_side), None)

    def cancel_all(self, event_ts_ns: int) -> None:
        for key in list(self.active_quotes):
            self._cancel(key, event_ts_ns)

    def cancel_inactive(
        self, active_market_tickers: set[str], event_ts_ns: int
    ) -> None:
        """Retire shadow quotes immediately when discovery rolls to a new window."""
        for key in list(self.active_quotes):
            if key[0] in active_market_tickers:
                continue
            before = self.counts["quote_cancels"]
            self._cancel(key, event_ts_ns)
            if self.counts["quote_cancels"] > before:
                self.counts["market_rollover_cancels"] += 1


class KalshiForwardShadowRecorder(MetalsRecorder):
    """Record Kalshi live data and drive the counterfactual policy in-process."""

    def __init__(
        self,
        config: MetalsConfig,
        *,
        baseline_model_path: Path,
        settings: ForwardShadowSettings,
    ) -> None:
        super().__init__(config)
        model, study_id = load_baseline_vol_model(Path(baseline_model_path))
        long_vol_policy = LongVolLoggingPolicy()
        self._book_lineage: dict[str, tuple[int, int, int]] = {}
        self.shadow_session = ShadowSession(
            config,
            manifest_context={
                "schema_version": FORWARD_SHADOW_SCHEMA,
                "collector_run_id": self.run_id,
                "baseline_model_path": str(Path(baseline_model_path).resolve()),
                "baseline_study_id": study_id,
                "model_version": FORWARD_SHADOW_MODEL_VERSION,
                "settings": settings.public_dict(),
                "long_vol_candidate_schema": LONG_VOL_CANDIDATE_SCHEMA,
                "long_vol_logging_policy": long_vol_policy.public_dict(),
                "raw_inputs_are_authoritative": True,
                "derived_candidate_rows_are_replayable": True,
                "read_only": True,
                "live_orders": False,
            },
        )
        self.forward = ForwardShadowEngine(
            model,
            study_id,
            settings,
            self.shadow_session,
            long_vol_policy=long_vol_policy,
        )

    async def discover_once(self) -> list[dict[str, Any]]:
        markets = await super().discover_once()
        for metadata in self.market_versions.values():
            self.forward.observe_market(metadata)
        self.forward.cancel_inactive(set(self._desired_tickers), time.time_ns())
        return markets

    def _record_book_snapshot(
        self, envelope: SourceEnvelope, metal: str, message: dict[str, Any]
    ) -> None:
        super()._record_book_snapshot(envelope, metal, message)
        ticker = str(message.get("market_ticker", message.get("ticker", "")))
        self._book_lineage[ticker] = (
            envelope.receive_ts_ns,
            envelope.local_monotonic_ts_ns,
            envelope.ingestion_sequence,
        )
        if ticker not in self._desired_tickers:
            return
        book = self.books.get(ticker)
        if book is not None:
            self.forward.evaluate(
                ticker,
                book,
                decision_ts_ns=envelope.receive_ts_ns,
                input_ingestion_sequence=envelope.ingestion_sequence,
                trigger_source="kalshi_book_snapshot",
                book_receive_ts_ns=envelope.receive_ts_ns,
                book_local_monotonic_ts_ns=envelope.local_monotonic_ts_ns,
                book_ingestion_sequence=envelope.ingestion_sequence,
                force=True,
            )

    def _record_book_delta(
        self, envelope: SourceEnvelope, metal: str, message: dict[str, Any]
    ) -> None:
        super()._record_book_delta(envelope, metal, message)
        ticker = str(message.get("market_ticker", message.get("ticker", "")))
        self._book_lineage[ticker] = (
            envelope.receive_ts_ns,
            envelope.local_monotonic_ts_ns,
            envelope.ingestion_sequence,
        )
        if ticker not in self._desired_tickers:
            return
        book = self.books.get(ticker)
        if book is not None:
            self.forward.evaluate(
                ticker,
                book,
                decision_ts_ns=envelope.receive_ts_ns,
                input_ingestion_sequence=envelope.ingestion_sequence,
                trigger_source="kalshi_book_delta",
                book_receive_ts_ns=envelope.receive_ts_ns,
                book_local_monotonic_ts_ns=envelope.local_monotonic_ts_ns,
                book_ingestion_sequence=envelope.ingestion_sequence,
            )

    def _record_kalshi_pyth(
        self, envelope: SourceEnvelope, metal: str, message: dict[str, Any]
    ) -> None:
        super()._record_kalshi_pyth(envelope, metal, message)
        source_ts_ns = iso8601_to_ns(message.get("source_ts_ms"))
        if source_ts_ns is None:
            return
        observation = ExactPythObservation(
            symbol=str(message.get("underlying_ticker", "")),
            value=_finite_decimal(message.get("value_usd"), "Pyth value"),
            source_ts_ns=source_ts_ns,
            receive_ts_ns=envelope.receive_ts_ns,
            ingestion_sequence=envelope.ingestion_sequence,
        )
        self.forward.observe_pyth(observation)
        for ticker, book in self.books.items():
            if ticker not in self._desired_tickers:
                continue
            book_lineage = self._book_lineage.get(ticker)
            self.forward.evaluate(
                ticker,
                book,
                decision_ts_ns=envelope.receive_ts_ns,
                input_ingestion_sequence=envelope.ingestion_sequence,
                trigger_source="kalshi_exact_pyth",
                book_receive_ts_ns=(book_lineage[0] if book_lineage else None),
                book_local_monotonic_ts_ns=(
                    book_lineage[1] if book_lineage else None
                ),
                book_ingestion_sequence=(book_lineage[2] if book_lineage else None),
                force=True,
            )

    def _record_trade(
        self, envelope: SourceEnvelope, metal: str, message: dict[str, Any]
    ) -> None:
        super()._record_trade(envelope, metal, message)
        trade = {
            "trade_id": message.get("trade_id"),
            "market_ticker": str(
                message.get("market_ticker", message.get("ticker", ""))
            ),
            "yes_price_dollars": message.get(
                "yes_price_dollars", message.get("yes_price")
            ),
            "no_price_dollars": message.get(
                "no_price_dollars", message.get("no_price")
            ),
            "quantity_fp": message.get("count_fp", message.get("count")),
            "taker_outcome_side": message.get("taker_outcome_side"),
            "receive_ts_ns": envelope.receive_ts_ns,
            "ingestion_sequence": envelope.ingestion_sequence,
        }
        self.forward.observe_trade(metal, trade)

    async def run(self, duration_seconds: float) -> dict[str, Any]:
        status = "completed"
        failure: BaseException | None = None
        try:
            await super().run(duration_seconds)
        except asyncio.CancelledError as error:
            status = "cancelled"
            failure = error
            raise
        except BaseException as error:
            status = "failed"
            failure = error
            raise
        finally:
            self.forward.cancel_all(time.time_ns())
            self.shadow_session.close(status=status, error=failure)
        return {
            "collector_run_id": self.run_id,
            "shadow_run_id": self.shadow_session.run_id,
            "forward_counts": dict(self.forward.counts),
            "shadow_counts": dict(self.shadow_session.counts),
            "execution_mode": "shadow",
            "live_orders": False,
        }


__all__ = (
    "FORWARD_SHADOW_MODEL_VERSION",
    "FORWARD_SHADOW_SCHEMA",
    "LONG_VOL_CANDIDATE_SCHEMA",
    "LONG_VOL_POLICY_VERSION",
    "ExactPythObservation",
    "ForwardShadowEngine",
    "ForwardShadowError",
    "ForwardShadowSettings",
    "KalshiForwardShadowRecorder",
    "LongVolLoggingPolicy",
    "OnlineFairDecision",
    "QuoteProposal",
    "compute_online_fair",
)
