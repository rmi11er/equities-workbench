"""Silver maker experiment using the shared shadow risk/lifecycle engine.

Inputs are immutable replay events. No exchange order API is imported. Fast XAG
can veto a quote; it cannot increase the exact-index bid ceiling in this version.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from typing import Any

from .baseline_fair import BaselineVolModel
from .book import FixedPointOrderBook
from .config import MetalsConfig
from .forward_shadow import _price_ranges
from .maker_inputs import AnchoredXagUpdater, ManualControl, PriceObservation
from .shadow import ShadowOrderIntent
from .shadow_session import ShadowSession
from .types import MarketMetadataVersion

D = Decimal
ONE = D(1)
ZERO = D(0)
NSEC = 1_000_000_000
MSEC = 1_000_000
VERSION = "silver-opportunistic-maker-v1"


def tiny_maker_config(config: MetalsConfig) -> MetalsConfig:
    """Separate each simulated arm from the older large stress profile."""
    if config.shadow.allow_live_orders or config.shadow.execution_mode != "shadow":
        raise ValueError("maker experiment refuses live execution configuration")
    return replace(config, shadow=replace(
        config.shadow, enabled=True, config_version=VERSION,
        bankroll_min_dollars="25", bankroll_max_dollars="25",
        max_contracts_per_order="1", max_net_contracts_per_market="1",
        max_aggregate_worst_case_loss_dollars="1",
        max_loss_per_market_dollars="1", daily_loss_limit_dollars="5",
    ))


@dataclass(frozen=True)
class MakerSettings:
    require_fast: bool = True
    maker_fee_per_contract: Decimal = ZERO
    uncertainty_charge: Decimal = D("0.02")
    adverse_selection_charge: Decimal = D("0.01")
    required_edge: Decimal = D("0.03")
    min_bid: Decimal = D("0.07")
    max_bid: Decimal = D("0.31")
    min_seconds_remaining: int = 120
    persistence_ms: int = 1000
    submit_latency_ms: int = 100
    cancel_latency_ms: int = 100
    max_exact_age_ms: int = 2000
    max_fast_age_ms: int = 750
    max_anchor_gap_ms: int = 250
    max_fast_move_bps: int = 25
    max_clock_jump_ms: int = 100
    decision_interval_ms: int = 250

    def __post_init__(self) -> None:
        if type(self.require_fast) is not bool:
            raise ValueError("require_fast must be boolean")
        for name in ("maker_fee_per_contract", "uncertainty_charge",
                     "adverse_selection_charge", "required_edge", "min_bid", "max_bid"):
            value = getattr(self, name)
            if not isinstance(value, D) or not value.is_finite() or not ZERO <= value < ONE:
                raise ValueError(f"{name} must be a finite probability-dollar amount")
        if not ZERO < self.min_bid < self.max_bid < D("0.5"):
            raise ValueError("maker bid band must be below 50 cents")
        if not 120 <= self.min_seconds_remaining < 900:
            raise ValueError("maker entry cutoff must be in [120, 900) seconds")
        for name in ("persistence_ms", "submit_latency_ms", "cancel_latency_ms",
                     "max_exact_age_ms", "max_fast_age_ms", "max_anchor_gap_ms",
                     "max_fast_move_bps", "max_clock_jump_ms"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.decision_interval_ms) is not int or self.decision_interval_ms <= 0:
            raise ValueError("decision_interval_ms must be a positive integer")

    def public_dict(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, D) else value
                for key, value in asdict(self).items()}


class MakerEngine:
    def __init__(self, model: BaselineVolModel, study_id: str,
                 settings: MakerSettings, sessions: dict[str, ShadowSession]) -> None:
        if set(sessions) != {"manual", "baseline"}:
            raise ValueError("maker experiment requires manual and baseline arms")
        self.model, self.study_id, self.settings, self.sessions = model, study_id, settings, sessions
        self.markets: dict[str, MarketMetadataVersion] = {}
        self.books: dict[str, FixedPointOrderBook] = {}
        self.book_lineage: dict[str, dict[str, Any]] = {}
        self.updater = AnchoredXagUpdater()
        self.exact: PriceObservation | None = None
        self.control: ManualControl | None = None
        self.control_version = -1
        self.control_error = "control_missing"
        self.clock_fault = False
        self.clock_domain_fault = False
        self.uncertain_exposure: dict[str, str] = {}
        self.previous: dict[str, Any] | None = None
        self.streaks: dict[tuple[str, str, str], int] = {}
        self.filled_markets: dict[str, set[str]] = {arm: set() for arm in sessions}
        self.order_context: dict[str, dict[str, Any]] = {}
        self.fills: dict[str, list[dict[str, Any]]] = {arm: [] for arm in sessions}
        self.results: dict[str, str] = {}
        self.settled: dict[str, set[str]] = {arm: set() for arm in sessions}
        self.seen_trades: set[str] = set()
        self.retired_orders: dict[str, list[dict[str, Any]]] = {
            arm: [] for arm in sessions
        }
        self.counts: Counter[str] = Counter()
        self.digest = hashlib.sha256()
        self.closed = False
        self.last_evaluation: dict[tuple[str, str], int] = {}
        self.last_decision_log: dict[tuple[str, str], tuple[str, int]] = {}
        self.grids: dict[tuple[str, str], tuple[Decimal, ...]] = {}

    def _emit(self, arm: str, event: dict[str, Any], kind: str, **payload: Any) -> None:
        row = {"schema_version": VERSION, "kind": kind, "arm": arm,
               "decision_ts_ns": event["receive_ts_ns"],
               "local_monotonic_ts_ns": event["local_monotonic_ts_ns"],
               "input_ingestion_sequence": event["ingestion_sequence"],
               "baseline_study_id": self.study_id, "trading_eligible": False,
               "live_orders": False, **payload}
        canonical = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"))
        self.digest.update(canonical.encode())
        self.sessions[arm]._write("maker_event", "silver", event["receive_ts_ns"], row)
        self.counts[f"{arm}:{kind}"] += 1

    def _cancel(self, arm: str, event: dict[str, Any], reason: str,
                ticker: str | None = None) -> None:
        session = self.sessions[arm]
        for order in list(session.engine.resting_orders.values()):
            if ticker is not None and ticker != order.intent.market_ticker:
                continue
            events = session.cancel(
                order.intent.logical_quote_id, event["receive_ts_ns"],
                monotonic_ts_ns=event["local_monotonic_ts_ns"],
                cancel_latency_ns=self.settings.cancel_latency_ms * MSEC,
            )
            if events:
                self._emit(arm, event, "cancel_request", reason=reason,
                           logical_quote_id=order.intent.logical_quote_id,
                           market_ticker=order.intent.market_ticker,
                           fair_at_cancel=self._safe_fair(arm, order.intent.market_ticker, event))
        for key in list(self.streaks):
            if key[0] == arm and (ticker is None or key[1] == ticker):
                self.streaks.pop(key)

    def _quotes(self, ticker: str, side: str) -> tuple[Decimal, Decimal]:
        book = self.books.get(ticker)
        if book is None or not book.valid or not book.yes_bids or not book.no_bids:
            raise ValueError("book_invalid")
        yes_bid, no_bid = max(map(D, book.yes_bids)), max(map(D, book.no_bids))
        bid, ask = (yes_bid, ONE - no_bid) if side == "yes" else (no_bid, ONE - yes_bid)
        if not ZERO < bid < ask < ONE:
            raise ValueError("book_locked_or_crossed")
        return bid, ask

    def _multiplier(self, arm: str) -> Decimal:
        return self.control.volatility_multiplier if arm == "manual" and self.control else ONE

    def _probabilities(self, ticker: str, value: Decimal, source_ns: int,
                       multiplier: Decimal) -> dict[str, Decimal]:
        meta = self.markets[ticker]
        if (meta.metal != "silver" or meta.pyth_symbol != "Metal.Index.SILVER/USD"
                or meta.missing_required_fields or meta.target_price is None
                or meta.round_digits is None or meta.open_ts_ns is None or meta.close_ts_ns is None
                or meta.strike_type != "greater_or_equal"):
            raise ValueError("metadata_incomplete_or_unsupported")
        remaining = math.ceil((meta.close_ts_ns - source_ns) / NSEC)
        if source_ns < meta.open_ts_ns or not 0 < remaining <= 900:
            raise ValueError("price_outside_window")
        bucket = int((meta.open_ts_ns // NSEC % 86400) // 900)
        arguments = dict(current_index=float(value), strike=float(meta.target_price),
                         remaining_seconds=remaining, bucket=bucket,
                         ewma_variance=self.model.global_variance_15m,
                         round_digits=meta.round_digits, volatility_multiplier=float(multiplier))
        return {
            "gaussian": D(str(self.model.probability_yes(**arguments, distribution="gaussian"))),
            "empirical": D(str(self.model.probability_yes(**arguments, distribution="empirical"))),
            "remaining_variance": D(str(self.model.full_window_variance(
                bucket, self.model.global_variance_15m))) * D(remaining) / D(900) * multiplier**2,
        }

    def _fair(self, arm: str, ticker: str, event: dict[str, Any]) -> dict[str, Any]:
        now = event["receive_ts_ns"]
        meta = self.markets[ticker]
        if meta.metadata_receive_ts_ns > now:
            raise ValueError("noncausal_metadata")
        if self.exact is None:
            raise ValueError("exact_index_missing")
        if not (0 <= now - self.exact.source_ts_ns <= self.settings.max_exact_age_ms * MSEC
                and 0 <= now - self.exact.receive_ts_ns <= self.settings.max_exact_age_ms * MSEC):
            raise ValueError("exact_index_stale")
        multiplier = self._multiplier(arm)
        base = self._probabilities(ticker, self.exact.value, self.exact.source_ts_ns, multiplier)
        fast = None
        if self.settings.require_fast:
            fast = self.updater.estimate(
                now, max_exact_age_ms=self.settings.max_exact_age_ms,
                max_fast_age_ms=self.settings.max_fast_age_ms,
                max_anchor_gap_ms=self.settings.max_anchor_gap_ms,
                max_move_bps=self.settings.max_fast_move_bps,
            )
            current = fast["current_fast"]
            fast_probability = self._probabilities(
                ticker, fast["estimated_index"], current["source_ts_ns"], multiplier)
        else:
            fast_probability = base
        return {"exact_gaussian_yes": base["gaussian"],
                "exact_empirical_yes": base["empirical"],
                "fast_gaussian_yes": fast_probability["gaussian"],
                "remaining_variance": base["remaining_variance"],
                "volatility_multiplier": multiplier,
                "exact_source_ts_ns": self.exact.source_ts_ns,
                "exact_receive_ts_ns": self.exact.receive_ts_ns,
                "exact_ingestion_sequence": self.exact.ingestion_sequence,
                "fast_trace": fast,
                "mode": "xag_cancel_only" if self.settings.require_fast else "exact_only_diagnostic"}

    def _safe_fair(self, arm: str, ticker: str, event: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return self._fair(arm, ticker, event)
        except (ValueError, KeyError):
            return None

    def _health_reason(self, event: dict[str, Any]) -> str | None:
        if self.closed:
            return "stopped"
        if self.clock_fault:
            return "clock_discontinuity_restart_required"
        if self.clock_domain_fault:
            return "exchange_clock_alignment_restart_required"
        if self.control_error:
            return self.control_error
        if self.control is None:
            return "control_missing"
        return self.control.rejection_reason(event["receive_ts_ns"])

    def _can_rest(self, intent: ShadowOrderIntent, event: dict[str, Any]) -> bool:
        if self._health_reason(event):
            return False
        try:
            meta = self.markets[intent.market_ticker]
            if meta.close_ts_ns - event["receive_ts_ns"] < self.settings.min_seconds_remaining * NSEC:
                return False
            return intent.price < self._quotes(intent.market_ticker, intent.outcome_side)[1]
        except (KeyError, ValueError, TypeError):
            return False

    def _advance(self, event: dict[str, Any]) -> None:
        for arm, session in self.sessions.items():
            before = dict(session.engine.resting_orders)
            events = session.advance(event["receive_ts_ns"], event["local_monotonic_ts_ns"],
                                     lambda intent: arm not in self.uncertain_exposure
                                     and self._can_rest(intent, event))
            for transition in events:
                if transition.event_type in {"cancel_acknowledged", "post_only_rejected"}:
                    order = before.get(transition.logical_quote_id)
                    if order is not None and order.arrival_ts_ns is not None:
                        self.retired_orders[arm].append({
                            "logical_quote_id": transition.logical_quote_id,
                            "market_ticker": order.intent.market_ticker,
                            "side": order.intent.outcome_side,
                            "price": order.intent.price,
                            "remaining_quantity": order.remaining_quantity,
                            "arrival_ts_ns": order.arrival_ts_ns,
                            "retired_ts_ns": event["receive_ts_ns"],
                            "retired_input_sequence": event["ingestion_sequence"],
                        })
                self._emit(arm, event, "arrival_transition",
                           transition=transition.to_dict(),
                           fair_at_arrival=self._safe_fair(arm, transition.market_ticker, event))
            for ticker, result in self.results.items():
                if ticker in self.settled[arm]:
                    continue
                if any(o.intent.market_ticker == ticker for o in session.engine.resting_orders.values()):
                    continue
                pnl = session.engine.settle_market(ticker, result)
                self.settled[arm].add(ticker)
                for fill in self.fills[arm]:
                    if fill["market_ticker"] == ticker:
                        fill["settlement_result"] = result
                        fill["terminal_pnl"] = (ONE if fill["side"] == result else ZERO) * D(fill["quantity"]) - D(fill["cost"])
                self._emit(arm, event, "settlement", market_ticker=ticker, result=result, pnl=pnl)

    def _select_price(self, ticker: str, bid: Decimal, ask: Decimal,
                      ceiling: Decimal, side: str) -> Decimal | None:
        # Enumerate the small native grid so improvements across tapered boundaries
        # cannot create an off-grid price. NO is converted through the same grid.
        grid_key = (self.markets[ticker].price_ranges_json, side)
        if grid_key not in self.grids:
            legal: set[Decimal] = set()
            for band in _price_ranges(self.markets[ticker]):
                start, end, step = band["start"], band["end"], band["step"]
                if step <= 0 or not ZERO <= start < end <= ONE:
                    raise ValueError("invalid_price_grid")
                if (end - start) / step > 10000:
                    raise ValueError("price_grid_too_large")
                value = start
                while value < end:
                    if ZERO < value < ONE:
                        legal.add(value if side == "yes" else ONE-value)
                    value += step
            self.grids[grid_key] = tuple(sorted(legal))
        legal = self.grids[grid_key]
        if bid not in legal or bid > ceiling:
            return None
        improvement = min((price for price in legal if price > bid), default=ONE)
        return improvement if improvement <= ceiling and improvement < ask else bid

    def _decision(self, arm: str, ticker: str, event: dict[str, Any], **payload: Any) -> None:
        # Risk and cancellation checks run on every event; repetitive logging does not.
        key = (arm, ticker)
        label = f"{payload.get('action')}:{payload.get('reason', '')}"
        prior = self.last_decision_log.get(key)
        now = event["local_monotonic_ts_ns"]
        if (prior and prior[0] == label and now-prior[1] < self.settings.decision_interval_ms*MSEC
                and payload.get("action") not in {"submit", "risk_rejected"}):
            return
        self.last_decision_log[key] = (label, now)
        self._emit(arm, event, "decision", market_ticker=ticker, **payload)

    def _evaluate(self, arm: str, ticker: str, event: dict[str, Any]) -> None:
        meta = self.markets[ticker]
        session = self.sessions[arm]
        reason = self._health_reason(event) or (
            "unknown_exposure_restart_required" if arm in self.uncertain_exposure else None)
        key = (arm, ticker)
        now_mono = event["local_monotonic_ts_ns"]
        outstanding = any(o.intent.market_ticker == ticker
                          for o in session.engine.resting_orders.values())
        observing_streak = any(k[:2] == key for k in self.streaks)
        forced = event["kind"] in {"control", "gap", "metadata"}
        previous = self.last_evaluation.get(key)
        if (not outstanding and not observing_streak and not reason and not forced and previous is not None
                and now_mono-previous < self.settings.decision_interval_ms*MSEC):
            return
        self.last_evaluation[key] = now_mono
        fair = None
        side, bid, ask, ceiling, quote = None, None, None, None, None
        try:
            if reason:
                raise ValueError(reason)
            if ticker in self.results:
                raise ValueError("settled")
            if meta.close_ts_ns is None or meta.close_ts_ns - event["receive_ts_ns"] < self.settings.min_seconds_remaining * NSEC:
                raise ValueError("terminal_entry_cutoff")
            yes_bid, yes_ask = self._quotes(ticker, "yes")
            mid = (yes_bid + yes_ask) / 2
            if mid == D("0.5"):
                raise ValueError("no_unique_underdog")
            side = "yes" if mid < D("0.5") else "no"
            bid, ask = self._quotes(ticker, side)
            fair = self._fair(arm, ticker, event)
            p = fair["exact_gaussian_yes"]
            fast_p = fair["fast_gaussian_yes"]
            side_p, fast_side_p = (p, fast_p) if side == "yes" else (ONE-p, ONE-fast_p)
            # Fast protection never authorizes a higher bid than the slow fair.
            charges = (self.settings.uncertainty_charge + self.settings.adverse_selection_charge
                       + self.settings.maker_fee_per_contract + self.settings.required_edge)
            ceiling = min(side_p, fast_side_p) - charges
            fair.update(side_fair=side_p, fast_side_fair=fast_side_p,
                        externally_otm=side_p < D("0.5"))
            if ticker in self.filled_markets[arm]:
                raise ValueError("no_replenishment_after_fill")
            if session.engine.positions:
                raise ValueError("unsettled_position_cap")
            own = [o for o in session.engine.resting_orders.values()
                   if o.intent.market_ticker == ticker]
            if any(o.intent.market_ticker != ticker for o in session.engine.resting_orders.values()):
                raise ValueError("outstanding_market_cap")
            for order in own:
                if (order.intent.outcome_side != side or order.intent.price > ceiling
                        or order.intent.price >= ask):
                    raise ValueError("resting_edge_invalidated")
            if own:
                # Queue is retained while its edge is valid. Never chase every BBO move.
                self._decision(arm, ticker, event, action="retain",
                           fair=fair, side=side, bid_ceiling=ceiling,
                           logical_quote_id=own[0].intent.logical_quote_id,
                           book_lineage=self.book_lineage.get(ticker))
                return
            quote = self._select_price(ticker, bid, ask, ceiling, side)
            if quote is None or not self.settings.min_bid <= quote <= self.settings.max_bid:
                raise ValueError("no_quote_in_price_band_and_edge")
            key = (arm, ticker, side)
            for other in list(self.streaks):
                if other[:2] == key[:2] and other != key:
                    self.streaks.pop(other)
            start = self.streaks.setdefault(key, event["local_monotonic_ts_ns"])
            if event["local_monotonic_ts_ns"] - start < self.settings.persistence_ms * MSEC:
                self._decision(arm, ticker, event, action="wait_persistence",
                           fair=fair, side=side, quote_price=quote, bid_ceiling=ceiling)
                return
            quote_id = f"maker:{arm}:{ticker}:{event['ingestion_sequence']}"
            state = {"fair": fair, "control": self.control.public_dict(), "quote": quote,
                     "metadata_hash": meta.metadata_hash, "settings": self.settings.public_dict(),
                     "book_lineage": self.book_lineage.get(ticker)}
            state_hash = hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()
            intent = ShadowOrderIntent(
                quote_id, ticker, "silver", side, str(quote), "1", event["receive_ts_ns"],
                VERSION, self.study_id, session.config.shadow.config_version, state_hash,
                str(self.settings.maker_fee_per_contract),
            )
            decision, _ = session.place(intent, monotonic_ts_ns=event["local_monotonic_ts_ns"],
                                         submit_latency_ns=self.settings.submit_latency_ms * MSEC)
            self.order_context[quote_id] = {"placement_fair": fair, "control": self.control.public_dict()}
            self._decision(arm, ticker, event,
                       action="submit" if decision.approved else "risk_rejected",
                       logical_quote_id=quote_id, quote_price=quote, side=side, fair=fair,
                       bid_ceiling=ceiling, rejections=decision.rejection_reasons,
                       book_lineage=self.book_lineage.get(ticker))
            return
        except (ValueError, KeyError) as error:
            reason = str(error)
        self._cancel(arm, event, reason, ticker)
        self._decision(arm, ticker, event, action="skip",
                   reason=reason, fair=fair, side=side, bid_ceiling=ceiling)

    def _trade(self, event: dict[str, Any]) -> None:
        trade_id = event.get("trade_id")
        if not trade_id or trade_id in self.seen_trades or event.get("is_block_trade"):
            self.counts["excluded_duplicate_unidentified_or_block_trade"] += 1
            return
        self.seen_trades.add(trade_id)
        if self.clock_fault or self.clock_domain_fault:
            for arm, session in self.sessions.items():
                self.counts[f"{arm}:trade_ignored_clock_fault"] += 1
                self._emit(
                    arm, event, "trade_time_uncertain", trade_id=trade_id,
                    source_event_ts_ns=event.get("source_event_ts_ns"),
                    outstanding_orders=list(session.engine.resting_orders),
                    reason="exchange_source_and_host_arrival_times_not_comparable",
                    execution_qualification_incomplete=True,
                )
            return
        if event.get("taker_outcome_side") not in {"yes", "no"}:
            return
        ticker = event["market_ticker"]
        source_ns = event.get("source_event_ts_ns")
        if type(source_ns) is not int or source_ns > event["receive_ts_ns"]:
            self.counts["trade_source_time_unusable"] += 1
            future = type(source_ns) is int and source_ns > event["receive_ts_ns"]
            if future:
                self.clock_domain_fault = True
            for arm, session in self.sessions.items():
                if any(o.intent.market_ticker == ticker for o in session.engine.resting_orders.values()):
                    self.uncertain_exposure[arm] = "trade_time_unusable_while_order_outstanding"
                self._cancel(arm, event, "trade_source_time_unusable", None if future else ticker)
                self._emit(arm, event, "trade_time_uncertain", trade_id=trade_id,
                           source_event_ts_ns=source_ns,
                           source_ahead_of_receive_ns=(source_ns-event["receive_ts_ns"]) if future else None,
                           reason="source_clock_ahead_of_receive" if future else "missing_trade_source_time",
                           execution_qualification_incomplete=True)
            return
        side = "no" if event["taker_outcome_side"] == "yes" else "yes"
        price = D(str(event[f"{side}_price_dollars"]))
        quantity = D(str(event["quantity_fp"]))
        if not price.is_finite() or not quantity.is_finite() or not ZERO <= price <= ONE or quantity <= ZERO:
            raise ValueError("invalid trade price or quantity")
        for arm, session in self.sessions.items():
            for retired in self.retired_orders[arm]:
                if (retired["market_ticker"] == ticker and retired["side"] == side
                        and retired["arrival_ts_ns"] <= source_ns <= retired["retired_ts_ns"]
                        and price < retired["price"] and retired["remaining_quantity"] > ZERO):
                    self.counts[f"{arm}:unknown_fills_after_retirement"] += 1
                    self.uncertain_exposure[arm] = "delayed_trade_after_retirement"
                    self._emit(
                        arm, event, "late_trade_after_retirement", trade_id=trade_id,
                        source_event_ts_ns=source_ns, trade_price=price,
                        trade_quantity=quantity, retired_order=retired,
                        reason="delayed_trade_source_overlaps_pre_cancel_exposure",
                        execution_qualification_incomplete=True,
                    )
                    self._cancel(arm, event, "unknown_exposure_restart_required")
            available = quantity
            for order in list(session.engine.resting_orders.values()):
                if order.intent.market_ticker != ticker or order.intent.outcome_side != side:
                    continue
                if order.arrival_ts_ns is None or source_ns < order.arrival_ts_ns:
                    self.counts[f"{arm}:trade_precedes_arrival"] += 1
                    continue
                if price == order.intent.price:
                    self.counts[f"{arm}:touch_without_fill"] += 1
                    continue
                if price > order.intent.price or available <= ZERO:
                    continue
                qty = min(available, order.remaining_quantity)
                available -= qty
                fee = qty * self.settings.maker_fee_per_contract
                pending_cancel = order.state == "pending_cancel"
                fair = self._safe_fair(arm, ticker, event)
                session.fill(order.intent.logical_quote_id, qty, event["receive_ts_ns"], fee)
                self.filled_markets[arm].add(ticker)
                fill = {"market_ticker": ticker, "side": side, "quantity": str(qty),
                        "price": str(order.intent.price), "cost": str(qty * order.intent.price + fee),
                        "fill_ts_ns": event["receive_ts_ns"], "source_trade_ts_ns": source_ns,
                        "fill_mono_ns": event["local_monotonic_ts_ns"],
                        "logical_quote_id": order.intent.logical_quote_id, "trade_id": trade_id,
                        "cancel_in_flight": pending_cancel, "fair_at_fill": fair,
                        **self.order_context[order.intent.logical_quote_id],
                        "markouts": {}, "terminal_pnl": None}
                self.fills[arm].append(fill)
                self._emit(arm, event, "fill", fill=dict(fill),
                           attribution="strict_penetration_scenario_not_guaranteed_fill")
                # Stop after any partial fill; remainder still exposed during cancel.
                self._cancel(arm, event, "first_fill", ticker)

    def _markouts(self, event: dict[str, Any]) -> None:
        for arm, fills in self.fills.items():
            for fill in fills:
                elapsed = event["local_monotonic_ts_ns"] - fill["fill_mono_ns"]
                for seconds in (1, 5, 30, 60):
                    key = f"{seconds}s"
                    if key in fill["markouts"] or elapsed < seconds * NSEC:
                        continue
                    if elapsed > (seconds + 1) * NSEC:
                        fill["markouts"][key] = {"missing": True}
                        continue
                    ticker, side = fill["market_ticker"], fill["side"]
                    try:
                        bid, ask = self._quotes(ticker, side)
                    except ValueError:
                        continue
                    fair = self._safe_fair(arm, ticker, event)
                    p = None if fair is None else fair["exact_gaussian_yes"]
                    side_p = p if side == "yes" or p is None else ONE-p
                    fill_fair = fill["fair_at_fill"]
                    initial_p = None if fill_fair is None else fill_fair["exact_gaussian_yes"]
                    initial_p = initial_p if side == "yes" or initial_p is None else ONE-initial_p
                    mark = {"side_bid": bid, "side_midpoint": (bid+ask)/2,
                            "bid_mark_less_entry_cost_per_contract": bid - D(fill["cost"])/D(fill["quantity"]),
                            "side_external_fair": side_p,
                            "external_fair_change": None if side_p is None or initial_p is None else side_p-initial_p,
                            "observation_lag_ms": D(elapsed-seconds*NSEC)/MSEC,
                            "exit_fees_included": False}
                    fill["markouts"][key] = mark
                    self._emit(arm, event, "markout", logical_quote_id=fill["logical_quote_id"],
                               trade_id=fill["trade_id"], horizon=key, mark=mark)

    def handle(self, event: dict[str, Any]) -> None:
        for key in ("receive_ts_ns", "local_monotonic_ts_ns", "ingestion_sequence"):
            if type(event.get(key)) is not int or event[key] < 0:
                raise ValueError(f"invalid event {key}")
        if self.previous is not None:
            if event["ingestion_sequence"] <= self.previous["ingestion_sequence"]:
                raise ValueError("non-increasing input sequence")
            wall_delta = event["receive_ts_ns"] - self.previous["receive_ts_ns"]
            mono_delta = event["local_monotonic_ts_ns"] - self.previous["local_monotonic_ts_ns"]
            if mono_delta < 0 or abs(wall_delta-mono_delta) > self.settings.max_clock_jump_ms*MSEC:
                self.clock_fault = True
        self.previous = event
        kind, ticker = event["kind"], event.get("market_ticker")
        if kind == "control":
            try:
                control = ManualControl.from_dict(event["control"])
                if control.version <= self.control_version:
                    raise ValueError("control_version_not_increasing")
                self.control, self.control_version, self.control_error = control, control.version, ""
            except (ValueError, TypeError, KeyError) as error:
                self.control_error = "control_invalid:" + str(error)
            for arm in self.sessions:
                self._cancel(arm, event, "control_changed")
                self._emit(arm, event, "control", control=event.get("control"), error=self.control_error)
        elif kind == "metadata":
            meta = MarketMetadataVersion(**event["metadata"])
            prior = self.markets.get(meta.market_ticker)
            self.markets[meta.market_ticker] = meta
            if prior is not None and prior.metadata_hash != meta.metadata_hash:
                for arm in self.sessions:
                    self._cancel(arm, event, "metadata_changed", meta.market_ticker)
            if event.get("result") in {"yes", "no"}:
                old_result = self.results.get(meta.market_ticker)
                if old_result and old_result != event["result"]:
                    self.clock_fault = True
                    raise ValueError("settlement_correction_requires_review")
                self.results[meta.market_ticker] = event["result"]
        elif kind == "book":
            book = FixedPointOrderBook(ticker)
            book.apply_snapshot({"yes_dollars_fp": list(event["yes_bids"].items()),
                                 "no_dollars_fp": list(event["no_bids"].items())},
                                event.get("book_state_hash", "maker-input"))
            if event.get("valid") is not True:
                book.invalidate()
            self.books[ticker] = book
            self.book_lineage[ticker] = {k: event.get(k) for k in (
                "book_state_hash", "receive_ts_ns", "local_monotonic_ts_ns", "ingestion_sequence")}
        elif kind in {"exact", "fast"}:
            obs = PriceObservation(D(str(event["value"])), event["source_ts_ns"],
                                   event["receive_ts_ns"], event["ingestion_sequence"])
            if kind == "exact":
                self.updater.observe_exact(obs)
                if self.exact is None or obs.source_ts_ns > self.exact.source_ts_ns:
                    self.exact = obs
            else:
                self.updater.observe_fast(obs)
        elif kind == "gap":
            affected = [ticker] if ticker else list(self.books)
            for name in affected:
                if name in self.books:
                    self.books[name].invalidate()
            if ticker is None:
                self.exact = None
                self.updater = AnchoredXagUpdater()
            for arm in self.sessions:
                if any(ticker is None or order.intent.market_ticker == ticker
                       for order in self.sessions[arm].engine.resting_orders.values()):
                    self.uncertain_exposure[arm] = "data_gap_while_order_outstanding"
                self._cancel(arm, event, event.get("reason", "gap"), ticker)
                self._emit(arm, event, "gap", reason=event.get("reason"), market_ticker=ticker,
                           unknown_fills_during_gap=True)
        elif kind not in {"timer", "trade"}:
            raise ValueError("unknown maker event kind")

        self._advance(event)
        if kind == "trade":
            self._trade(event)
        # Trade fills are processed before current-state cancels: a decision at
        # receipt cannot retroactively prevent that observed aggressive trade.
        for arm in self.sessions:
            for name in self.markets:
                if name not in self.settled[arm]:
                    self._evaluate(arm, name, event)
        self._markouts(event)

    def close(self, event: dict[str, Any]) -> None:
        self.closed = True
        for arm in self.sessions:
            self._cancel(arm, event, "runner_stopped")
        # No invented cancellation acknowledgments after observation stops.

    def summary(self) -> dict[str, Any]:
        qualification_reasons = []
        if self.clock_fault:
            qualification_reasons.append("clock_discontinuity")
        if self.clock_domain_fault:
            qualification_reasons.append("exchange_source_clock_ahead_of_receive")
        if self.counts["trade_source_time_unusable"]:
            qualification_reasons.append("unusable_trade_source_timestamps")
        for arm, session in self.sessions.items():
            if arm in self.uncertain_exposure:
                qualification_reasons.append(f"{arm}:unknown_exposure")
            if self.counts[f"{arm}:unknown_fills_after_retirement"]:
                qualification_reasons.append(f"{arm}:unknown_fills_after_retirement")
            if self.counts[f"{arm}:gap"]:
                qualification_reasons.append(f"{arm}:data_gaps")
            if session.engine.resting_orders:
                qualification_reasons.append(f"{arm}:unacknowledged_orders_at_end")
        return {"schema_version": VERSION, "settings": self.settings.public_dict(),
                "study_id": self.study_id, "counts": dict(self.counts),
                "decision_digest": self.digest.hexdigest(), "live_orders": False,
                "trading_eligible": False, "clock_fault": self.clock_fault,
                "clock_domain_fault": self.clock_domain_fault,
                "execution_qualification_incomplete": bool(qualification_reasons),
                "execution_qualification_reasons": qualification_reasons,
                "uncertain_exposure": dict(self.uncertain_exposure),
                "arms": {arm: {
                    "fills": rows,
                    "settled_pnl": str(self.sessions[arm].engine.daily_realized_pnl),
                    "unsettled_markets": sorted(self.sessions[arm].engine.positions),
                    "outstanding_orders": {key: value.state for key, value in
                                           self.sessions[arm].engine.resting_orders.items()},
                } for arm, rows in self.fills.items()},
                "limitations": [
                    "Strict penetration is a declared fill scenario; queue position is unknown.",
                    "Arrival is evaluated at the first observed event after its simulated deadline.",
                    "No fills are invented across data gaps; outstanding exposure is retained.",
                    "Fast XAG only reduces the exact-index ceiling; return beta is fixed at one.",
                    "Manual settings are part of the strategy; baseline shares pause/health rules.",
                    "Maker fees are an explicit per-contract reserve, not account fee reconciliation.",
                    "Session loss accounting is conservative across dates; it does not reset at midnight.",
                ]}
