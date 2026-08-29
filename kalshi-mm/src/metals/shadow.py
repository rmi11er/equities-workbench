"""Deterministic, exchange-isolated shadow order and risk state.

This module intentionally has no connector or order-entry dependency. It records
counterfactual order lifecycle events and applies conservative risk checks before a
shadow order may become resting simulation state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import ShadowConfig


ZERO = Decimal("0")
ONE = Decimal("1")


def _decimal(value: Any, field_name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("{} must be a finite decimal".format(field_name)) from error
    if not result.is_finite():
        raise ValueError("{} must be a finite decimal".format(field_name))
    return result


def _decimal_string(value: Decimal) -> str:
    return format(value, "f")


@dataclass(frozen=True)
class ShadowOrderIntent:
    logical_quote_id: str
    market_ticker: str
    metal: str
    outcome_side: str
    price_dollars: str
    quantity_fp: str
    decision_ts_ns: int
    strategy_module: str
    model_version: str
    config_version: str
    state_snapshot_hash: str
    worst_case_fee_dollars: str = "0"

    def __post_init__(self) -> None:
        if not self.logical_quote_id or not self.market_ticker:
            raise ValueError("Shadow orders require quote and market identifiers")
        if self.outcome_side.lower() not in ("yes", "no"):
            raise ValueError("outcome_side must be 'yes' or 'no'")
        price = _decimal(self.price_dollars, "price_dollars")
        quantity = _decimal(self.quantity_fp, "quantity_fp")
        fee = _decimal(self.worst_case_fee_dollars, "worst_case_fee_dollars")
        if price <= ZERO or price >= ONE:
            raise ValueError("price_dollars must be strictly between zero and one")
        if quantity <= ZERO:
            raise ValueError("quantity_fp must be positive")
        if fee < ZERO:
            raise ValueError("worst_case_fee_dollars cannot be negative")
        if self.decision_ts_ns < 0:
            raise ValueError("decision_ts_ns cannot be negative")

    @property
    def price(self) -> Decimal:
        return _decimal(self.price_dollars, "price_dollars")

    @property
    def quantity(self) -> Decimal:
        return _decimal(self.quantity_fp, "quantity_fp")

    @property
    def worst_case_fee(self) -> Decimal:
        return _decimal(self.worst_case_fee_dollars, "worst_case_fee_dollars")


@dataclass
class ShadowPosition:
    yes_quantity: Decimal = ZERO
    no_quantity: Decimal = ZERO
    cash_spent: Decimal = ZERO

    @property
    def net_outcome_exposure(self) -> Decimal:
        return self.yes_quantity - self.no_quantity

    @property
    def pnl_if_yes(self) -> Decimal:
        return self.yes_quantity - self.cash_spent

    @property
    def pnl_if_no(self) -> Decimal:
        return self.no_quantity - self.cash_spent

    @property
    def worst_case_loss(self) -> Decimal:
        return max(ZERO, -min(self.pnl_if_yes, self.pnl_if_no))


@dataclass
class RestingShadowOrder:
    intent: ShadowOrderIntent
    remaining_quantity: Decimal
    remaining_fee_reserve: Decimal


@dataclass(frozen=True)
class ShadowLifecycleEvent:
    event_type: str
    event_ts_ns: int
    logical_quote_id: str
    client_order_id: str
    exchange_order_id: Optional[str]
    market_ticker: str
    strategy_module: str
    model_version: str
    config_version: str
    outcome_side: str
    price_dollars: str
    original_quantity_fp: str
    remaining_quantity_fp: str
    fee_dollars: str
    state_snapshot_hash: str
    execution_mode: str = "shadow"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowRiskSnapshot:
    market_worst_case_loss_dollars: Dict[str, str]
    market_max_abs_net_contracts: Dict[str, str]
    aggregate_worst_case_loss_dollars: str
    daily_realized_pnl_dollars: str
    realized_plus_worst_case_pnl_dollars: str
    active_kill_switches: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowRiskDecision:
    approved: bool
    rejection_reasons: Tuple[str, ...]
    snapshot: ShadowRiskSnapshot


class ShadowRiskEngine:
    """Track simulated exposure including conservative full resting-order fills."""

    def __init__(self, config: ShadowConfig) -> None:
        if config.execution_mode != "shadow" or config.allow_live_orders:
            raise ValueError("ShadowRiskEngine refuses live-order configuration")
        self.config = config
        self.positions: Dict[str, ShadowPosition] = {}
        self.resting_orders: Dict[str, RestingShadowOrder] = {}
        self.daily_realized_pnl = ZERO
        self._kill_switches: set = set()

        self._max_order = _decimal(
            config.max_contracts_per_order, "max_contracts_per_order"
        )
        self._max_net = _decimal(
            config.max_net_contracts_per_market,
            "max_net_contracts_per_market",
        )
        self._max_aggregate_loss = _decimal(
            config.max_aggregate_worst_case_loss_dollars,
            "max_aggregate_worst_case_loss_dollars",
        )
        self._max_market_loss = _decimal(
            config.max_loss_per_market_dollars,
            "max_loss_per_market_dollars",
        )
        self._daily_loss_limit = _decimal(
            config.daily_loss_limit_dollars, "daily_loss_limit_dollars"
        )
        self._bankroll_min = _decimal(
            config.bankroll_min_dollars, "bankroll_min_dollars"
        )
        self._bankroll_max = _decimal(
            config.bankroll_max_dollars, "bankroll_max_dollars"
        )
        limits = (
            self._max_order,
            self._max_net,
            self._max_aggregate_loss,
            self._max_market_loss,
            self._daily_loss_limit,
            self._bankroll_min,
            self._bankroll_max,
        )
        if any(value <= ZERO for value in limits):
            raise ValueError("Shadow risk limits must be positive")
        if self._bankroll_min > self._bankroll_max:
            raise ValueError("bankroll_min_dollars cannot exceed bankroll_max_dollars")
        if self._max_aggregate_loss > self._bankroll_min:
            raise ValueError(
                "aggregate worst-case loss cannot exceed the minimum shadow bankroll"
            )

    def set_kill_switch(self, name: str, active: bool = True) -> None:
        normalized = name.strip()
        if not normalized:
            raise ValueError("Kill-switch name cannot be empty")
        if active:
            self._kill_switches.add(normalized)
        else:
            self._kill_switches.discard(normalized)

    def set_daily_realized_pnl(self, value: Any) -> None:
        self.daily_realized_pnl = _decimal(value, "daily_realized_pnl")

    def _orders_for_market(
        self,
        market_ticker: str,
        candidate: Optional[ShadowOrderIntent] = None,
    ) -> Iterable[Tuple[str, Decimal, Decimal, Decimal]]:
        for order in self.resting_orders.values():
            if order.intent.market_ticker == market_ticker:
                yield (
                    order.intent.outcome_side.lower(),
                    order.intent.price,
                    order.remaining_quantity,
                    order.remaining_fee_reserve,
                )
        if candidate is not None and candidate.market_ticker == market_ticker:
            yield (
                candidate.outcome_side.lower(),
                candidate.price,
                candidate.quantity,
                candidate.worst_case_fee,
            )

    def _market_risk(
        self,
        market_ticker: str,
        candidate: Optional[ShadowOrderIntent] = None,
    ) -> Tuple[Decimal, Decimal]:
        position = self.positions.get(market_ticker, ShadowPosition())
        yes_open = ZERO
        no_open = ZERO
        yes_open_cost = ZERO
        no_open_cost = ZERO
        for side, price, quantity, fee_reserve in self._orders_for_market(
            market_ticker, candidate
        ):
            if side == "yes":
                yes_open += quantity
                yes_open_cost += price * quantity + fee_reserve
            else:
                no_open += quantity
                no_open_cost += price * quantity + fee_reserve

        # For each settlement outcome, assume only losing-side resting bids fill.
        pnl_if_yes = position.pnl_if_yes - no_open_cost
        pnl_if_no = position.pnl_if_no - yes_open_cost
        worst_case_loss = max(ZERO, -min(pnl_if_yes, pnl_if_no))

        current_net = position.net_outcome_exposure
        possible_net = (
            current_net,
            current_net + yes_open,
            current_net - no_open,
            current_net + yes_open - no_open,
        )
        max_abs_net = max(abs(value) for value in possible_net)
        return worst_case_loss, max_abs_net

    def snapshot(
        self, candidate: Optional[ShadowOrderIntent] = None
    ) -> ShadowRiskSnapshot:
        markets = set(self.positions)
        markets.update(
            order.intent.market_ticker for order in self.resting_orders.values()
        )
        if candidate is not None:
            markets.add(candidate.market_ticker)

        market_losses: Dict[str, str] = {}
        market_nets: Dict[str, str] = {}
        aggregate_loss = ZERO
        for ticker in sorted(markets):
            loss, max_abs_net = self._market_risk(ticker, candidate)
            market_losses[ticker] = _decimal_string(loss)
            market_nets[ticker] = _decimal_string(max_abs_net)
            aggregate_loss += loss

        realized_plus_worst = self.daily_realized_pnl - aggregate_loss
        return ShadowRiskSnapshot(
            market_worst_case_loss_dollars=market_losses,
            market_max_abs_net_contracts=market_nets,
            aggregate_worst_case_loss_dollars=_decimal_string(aggregate_loss),
            daily_realized_pnl_dollars=_decimal_string(self.daily_realized_pnl),
            realized_plus_worst_case_pnl_dollars=_decimal_string(
                realized_plus_worst
            ),
            active_kill_switches=tuple(sorted(self._kill_switches)),
        )

    def assess(self, intent: ShadowOrderIntent) -> ShadowRiskDecision:
        reasons: List[str] = []
        if intent.config_version != self.config.config_version:
            reasons.append("config_version_mismatch")
        if intent.quantity > self._max_order:
            reasons.append("max_contracts_per_order")
        reasons.extend(
            "kill_switch:{}".format(name) for name in sorted(self._kill_switches)
        )

        snapshot = self.snapshot(intent)
        market_loss = _decimal(
            snapshot.market_worst_case_loss_dollars[intent.market_ticker],
            "market_worst_case_loss",
        )
        market_net = _decimal(
            snapshot.market_max_abs_net_contracts[intent.market_ticker],
            "market_max_abs_net_contracts",
        )
        aggregate_loss = _decimal(
            snapshot.aggregate_worst_case_loss_dollars,
            "aggregate_worst_case_loss",
        )
        realized_plus_worst = _decimal(
            snapshot.realized_plus_worst_case_pnl_dollars,
            "realized_plus_worst_case_pnl",
        )
        if market_net > self._max_net:
            reasons.append("max_net_contracts_per_market")
        if market_loss > self._max_market_loss:
            reasons.append("max_loss_per_market")
        if aggregate_loss > self._max_aggregate_loss:
            reasons.append("max_aggregate_worst_case_loss")
        if realized_plus_worst < -self._daily_loss_limit:
            reasons.append("daily_loss_limit")
        return ShadowRiskDecision(
            approved=not reasons,
            rejection_reasons=tuple(reasons),
            snapshot=snapshot,
        )

    def place(
        self, intent: ShadowOrderIntent
    ) -> Tuple[ShadowRiskDecision, List[ShadowLifecycleEvent]]:
        if intent.logical_quote_id in self.resting_orders:
            raise ValueError("logical_quote_id is already resting")
        decision = self.assess(intent)
        event_types = ["decision_created"]
        if decision.approved:
            self.resting_orders[intent.logical_quote_id] = RestingShadowOrder(
                intent=intent,
                remaining_quantity=intent.quantity,
                remaining_fee_reserve=intent.worst_case_fee,
            )
            event_types.extend(("submit_sent", "submit_acknowledged", "order_resting"))
        else:
            event_types.append("risk_rejected")
        events = [self._event(intent, event_type, intent.quantity) for event_type in event_types]
        return decision, events

    def fill(
        self,
        logical_quote_id: str,
        quantity_fp: Any,
        event_ts_ns: int,
        fee_dollars: Any = "0",
    ) -> ShadowLifecycleEvent:
        order = self.resting_orders.get(logical_quote_id)
        if order is None:
            raise KeyError("Unknown resting shadow order: {}".format(logical_quote_id))
        quantity = _decimal(quantity_fp, "quantity_fp")
        fee = _decimal(fee_dollars, "fee_dollars")
        if quantity <= ZERO or quantity > order.remaining_quantity:
            raise ValueError("Fill quantity must be positive and no greater than remaining")
        if fee < ZERO:
            raise ValueError("fee_dollars cannot be negative")
        if fee > order.remaining_fee_reserve:
            raise ValueError("fee_dollars exceeds the reserved worst-case fee")

        position = self.positions.setdefault(
            order.intent.market_ticker, ShadowPosition()
        )
        if order.intent.outcome_side.lower() == "yes":
            position.yes_quantity += quantity
        else:
            position.no_quantity += quantity
        position.cash_spent += order.intent.price * quantity + fee
        order.remaining_quantity -= quantity
        order.remaining_fee_reserve -= fee
        event_type = "full_fill" if order.remaining_quantity == ZERO else "partial_fill"
        event = self._event(
            order.intent,
            event_type,
            order.remaining_quantity,
            event_ts_ns=event_ts_ns,
            fee=fee,
        )
        if order.remaining_quantity == ZERO:
            del self.resting_orders[logical_quote_id]
        return event

    def cancel(
        self, logical_quote_id: str, event_ts_ns: int
    ) -> List[ShadowLifecycleEvent]:
        order = self.resting_orders.pop(logical_quote_id, None)
        if order is None:
            raise KeyError("Unknown resting shadow order: {}".format(logical_quote_id))
        return [
            self._event(
                order.intent,
                event_type,
                order.remaining_quantity,
                event_ts_ns=event_ts_ns,
            )
            for event_type in (
                "cancel_decision",
                "cancel_sent",
                "cancel_acknowledged",
            )
        ]

    def settle_market(self, market_ticker: str, result: str) -> Decimal:
        if any(
            order.intent.market_ticker == market_ticker
            for order in self.resting_orders.values()
        ):
            raise ValueError("Cancel resting shadow orders before settlement")
        position = self.positions.pop(market_ticker, ShadowPosition())
        normalized = result.lower()
        if normalized == "yes":
            pnl = position.pnl_if_yes
        elif normalized == "no":
            pnl = position.pnl_if_no
        else:
            raise ValueError("Settlement result must be 'yes' or 'no'")
        self.daily_realized_pnl += pnl
        return pnl

    def _event(
        self,
        intent: ShadowOrderIntent,
        event_type: str,
        remaining_quantity: Decimal,
        event_ts_ns: Optional[int] = None,
        fee: Decimal = ZERO,
    ) -> ShadowLifecycleEvent:
        return ShadowLifecycleEvent(
            event_type=event_type,
            event_ts_ns=(intent.decision_ts_ns if event_ts_ns is None else event_ts_ns),
            logical_quote_id=intent.logical_quote_id,
            client_order_id="shadow:{}".format(intent.logical_quote_id),
            exchange_order_id=None,
            market_ticker=intent.market_ticker,
            strategy_module=intent.strategy_module,
            model_version=intent.model_version,
            config_version=intent.config_version,
            outcome_side=intent.outcome_side.lower(),
            price_dollars=_decimal_string(intent.price),
            original_quantity_fp=_decimal_string(intent.quantity),
            remaining_quantity_fp=_decimal_string(remaining_quantity),
            fee_dollars=_decimal_string(fee),
            state_snapshot_hash=intent.state_snapshot_hash,
        )
