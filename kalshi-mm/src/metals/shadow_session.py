"""Persistent session wrapper for exchange-isolated shadow risk decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

    def __init__(self, config: MetalsConfig) -> None:
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
        }
        self._closed = False
        write_run_manifest(
            Path(config.storage.root), self.run_id, config.public_snapshot()
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
        self, intent: ShadowOrderIntent
    ) -> Tuple[ShadowRiskDecision, List[ShadowLifecycleEvent]]:
        decision, events = self.engine.place(intent)
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
        self, logical_quote_id: str, event_ts_ns: int
    ) -> List[ShadowLifecycleEvent]:
        order = self.engine.resting_orders.get(logical_quote_id)
        if order is None:
            raise KeyError("Unknown resting shadow order: {}".format(logical_quote_id))
        metal = order.intent.metal
        events = self.engine.cancel(logical_quote_id, event_ts_ns)
        for event in events:
            self._write_event(metal, event)
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
