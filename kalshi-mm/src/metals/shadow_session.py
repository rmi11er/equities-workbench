"""Persistent session wrapper for exchange-isolated shadow risk decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import MetalsConfig
from .shadow import (
    ShadowLifecycleEvent,
    ShadowOrderIntent,
    ShadowRiskDecision,
    ShadowRiskEngine,
)
from .storage import PartitionedRawStore, update_run_manifest, write_run_manifest


class ShadowSession:
    """Record every counterfactual decision and lifecycle transition."""

    def __init__(
        self,
        config: MetalsConfig,
        *,
        manifest_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not config.shadow.enabled:
            raise ValueError("Shadow session is disabled")
        self.config = config
        self.engine = ShadowRiskEngine(config.shadow)
        self.store = PartitionedRawStore(
            Path(config.storage.root),
            fsync_every_records=config.storage.fsync_every_records,
            tier="normalized",
        )
        self.run_id = self.store.run_id
        self.counts: Dict[str, int] = {
            "shadow_decisions": 0,
            "order_lifecycle_events": 0,
            "shadow_fills": 0,
            "forward_fair_decisions": 0,
            "long_vol_candidate_decisions": 0,
            "shadow_fill_attributions": 0,
        }
        self._closed = False
        manifest_config = config.public_snapshot()
        if manifest_context:
            manifest_config["forward_shadow"] = dict(manifest_context)
        write_run_manifest(
            Path(config.storage.root), self.run_id, manifest_config
        )

    def _write(
        self, stream: str, metal: str, event_ts_ns: int, record: Dict[str, Any]
    ) -> None:
        self.store.write(
            source=stream,
            stream=stream,
            metal=metal,
            receive_ts_ns=event_ts_ns,
            record={**record, "run_id": self.run_id},
        )

    def _write_event(self, metal: str, event: ShadowLifecycleEvent) -> None:
        self._write(
            "order_lifecycle", metal, event.event_ts_ns, event.to_dict()
        )
        self.counts["order_lifecycle_events"] += 1
        if event.event_type in ("partial_fill", "full_fill"):
            self.counts["shadow_fills"] += 1

    def place(
        self,
        intent: ShadowOrderIntent,
        *,
        monotonic_ts_ns: Optional[int] = None,
        submit_latency_ns: int = 0,
    ) -> Tuple[ShadowRiskDecision, List[ShadowLifecycleEvent]]:
        decision, events = self.engine.place(
            intent,
            monotonic_ts_ns=monotonic_ts_ns,
            submit_latency_ns=submit_latency_ns,
        )
        self._write(
            "shadow_decision",
            intent.metal,
            intent.decision_ts_ns,
            {
                "logical_quote_id": intent.logical_quote_id,
                "market_ticker": intent.market_ticker,
                "strategy_module": intent.strategy_module,
                "model_version": intent.model_version,
                "config_version": intent.config_version,
                "state_snapshot_hash": intent.state_snapshot_hash,
                "decision_ts_ns": intent.decision_ts_ns,
                "monotonic_ts_ns": monotonic_ts_ns,
                "submit_latency_ns": submit_latency_ns,
                "outcome_side": intent.outcome_side,
                "price_dollars": intent.price_dollars,
                "quantity_fp": intent.quantity_fp,
                "worst_case_fee_dollars": intent.worst_case_fee_dollars,
                "approved": decision.approved,
                "rejection_reasons": list(decision.rejection_reasons),
                "risk_snapshot": decision.snapshot.to_dict(),
                "execution_mode": "shadow",
            },
        )
        self.counts["shadow_decisions"] += 1
        for event in events:
            self._write_event(intent.metal, event)
        return decision, events

    def record_forward_fair_decision(
        self,
        metal: str,
        event_ts_ns: int,
        record: Dict[str, Any],
    ) -> None:
        """Persist one causal fair/quote evaluation alongside shadow lifecycle."""

        self._write("forward_fair_decision", metal, event_ts_ns, record)
        self.counts["forward_fair_decisions"] += 1

    def record_shadow_fill_attribution(
        self,
        metal: str,
        event_ts_ns: int,
        record: Dict[str, Any],
    ) -> None:
        """Bind a simulated fill to the observed trade and frozen fill rule."""

        self._write("shadow_fill_attribution", metal, event_ts_ns, record)
        self.counts["shadow_fill_attributions"] += 1

    def record_long_vol_candidate(
        self,
        metal: str,
        event_ts_ns: int,
        record: Dict[str, Any],
    ) -> None:
        """Persist a research-only long-vol evaluation, including rejections."""

        self._write("long_vol_candidate", metal, event_ts_ns, record)
        self.counts["long_vol_candidate_decisions"] += 1

    def fill(
        self,
        logical_quote_id: str,
        quantity_fp: Any,
        event_ts_ns: int,
        fee_dollars: Any = "0",
    ) -> ShadowLifecycleEvent:
        order = self.engine.resting_orders.get(logical_quote_id)
        if order is None:
            raise KeyError("Unknown resting shadow order: {}".format(logical_quote_id))
        metal = order.intent.metal
        event = self.engine.fill(
            logical_quote_id,
            quantity_fp,
            event_ts_ns,
            fee_dollars=fee_dollars,
        )
        self._write_event(metal, event)
        return event

    def cancel(
        self,
        logical_quote_id: str,
        event_ts_ns: int,
        *,
        monotonic_ts_ns: Optional[int] = None,
        cancel_latency_ns: int = 0,
    ) -> List[ShadowLifecycleEvent]:
        order = self.engine.resting_orders.get(logical_quote_id)
        events = self.engine.cancel(
            logical_quote_id,
            event_ts_ns,
            monotonic_ts_ns=monotonic_ts_ns,
            cancel_latency_ns=cancel_latency_ns,
        )
        for event in events:
            self._write_event(order.intent.metal, event)
        return events

    def advance(
        self,
        event_ts_ns: int,
        monotonic_ts_ns: int,
        can_rest: Callable[[ShadowOrderIntent], bool],
    ) -> List[ShadowLifecycleEvent]:
        """Persist simulated arrivals, post-only rejections, and cancel acks."""

        metals = {
            quote_id: order.intent.metal
            for quote_id, order in self.engine.resting_orders.items()
        }
        events = self.engine.advance(event_ts_ns, monotonic_ts_ns, can_rest)
        for event in events:
            self._write_event(metals[event.logical_quote_id], event)
        return events

    def close(
        self,
        status: str = "completed",
        error: Optional[BaseException] = None,
    ) -> None:
        if self._closed:
            return
        errors = [error] if error is not None else []
        close_error: Optional[BaseException] = None
        try:
            self.store.close()
        except BaseException as storage_error:
            close_error = storage_error
            errors.append(storage_error)
            status = "failed"
        update_run_manifest(
            Path(self.config.storage.root),
            self.run_id,
            status=status,
            counts=self.counts,
            output_paths=self.store.paths,
            errors=errors,
        )
        self._closed = True
        if close_error is not None:
            raise close_error

    def __enter__(self) -> "ShadowSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(status="failed" if exc is not None else "completed", error=exc)
