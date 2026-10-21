"""Validated operator controls and a causal Silver/XAG quote-protection input.

This module performs no I/O.  XAG is a proxy for changes in the exact settlement
index, using an explicitly assumed unit return beta.  It is neither settlement
data nor a fitted futures-to-index model.  An exact publication is paired with a
fast publication at or before its *source* time, never its arrival time.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

MSEC = 1_000_000
ONE = Decimal("1")
BPS = Decimal("10000")


def _integer(value: Any, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _decimal(value: Any, label: str) -> Decimal:
    if type(value) not in (str, int, float, Decimal):
        raise ValueError(f"{label} must be a decimal number")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"{label} must be a decimal number") from error
    if not parsed.is_finite():
        raise ValueError(f"{label} must be finite")
    return parsed


@dataclass(frozen=True)
class ManualControl:
    version: int
    effective_ts_ns: int
    expires_ts_ns: int
    volatility_multiplier: Decimal
    paused: bool
    reason: str

    def __post_init__(self) -> None:
        _integer(self.version, "version")
        _integer(self.effective_ts_ns, "effective_ts_ns")
        _integer(self.expires_ts_ns, "expires_ts_ns")
        if self.expires_ts_ns <= self.effective_ts_ns:
            raise ValueError("expires_ts_ns must be after effective_ts_ns")
        if type(self.volatility_multiplier) is not Decimal:
            raise ValueError("volatility_multiplier must be Decimal")
        multiplier = _decimal(self.volatility_multiplier, "volatility_multiplier")
        if not Decimal("0.25") <= multiplier <= Decimal("4"):
            raise ValueError("volatility_multiplier must be in [0.25, 4]")
        if type(self.paused) is not bool:
            raise ValueError("paused must be a boolean")
        if type(self.reason) is not str:
            raise ValueError("reason must be a string")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ManualControl:
        if type(payload) is not dict:
            raise ValueError("manual control must be an object")
        required = {
            "version", "effective_ts_ns", "expires_ts_ns",
            "volatility_multiplier", "paused", "reason",
        }
        if set(payload) != required:
            raise ValueError("manual control fields do not match the required schema")
        return cls(
            version=payload["version"],
            effective_ts_ns=payload["effective_ts_ns"],
            expires_ts_ns=payload["expires_ts_ns"],
            volatility_multiplier=_decimal(
                payload["volatility_multiplier"], "volatility_multiplier"
            ),
            paused=payload["paused"],
            reason=payload["reason"],
        )

    def rejection_reason(self, now_ns: int) -> str | None:
        _integer(now_ns, "now_ns")
        if now_ns < self.effective_ts_ns:
            return "manual_control_not_effective"
        if now_ns >= self.expires_ts_ns:
            return "manual_control_expired"
        if self.paused:
            return "manual_control_paused"
        return None

    def public_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "effective_ts_ns": self.effective_ts_ns,
            "expires_ts_ns": self.expires_ts_ns,
            "volatility_multiplier": format(self.volatility_multiplier, "f"),
            "paused": self.paused,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PriceObservation:
    value: Decimal
    source_ts_ns: int
    receive_ts_ns: int
    ingestion_sequence: int

    def __post_init__(self) -> None:
        if type(self.value) is not Decimal:
            raise ValueError("price value must be Decimal")
        if not self.value.is_finite() or self.value <= 0:
            raise ValueError("price value must be positive and finite")
        _integer(self.source_ts_ns, "source_ts_ns")
        _integer(self.receive_ts_ns, "receive_ts_ns")
        _integer(self.ingestion_sequence, "ingestion_sequence", minimum=0)
        if self.source_ts_ns > self.receive_ts_ns:
            raise ValueError("price source timestamp is after receipt")

    def public_dict(self) -> dict[str, Any]:
        return {
            "value": format(self.value, "f"),
            "source_ts_ns": self.source_ts_ns,
            "receive_ts_ns": self.receive_ts_ns,
            "ingestion_sequence": self.ingestion_sequence,
        }


class AnchoredXagUpdater:
    """Source-aligned unit-return proxy, suitable initially for quote protection.

    Call observe methods in recorder ingestion order with a shared increasing
    sequence.  Source publications may arrive out of order.  A duplicate source
    publication cannot refresh either source or receipt freshness; conflicting
    duplicate values are rejected.  Each exact arrival freezes its fast anchor
    using only fast observations already received at that arrival.

    ``estimate`` is a current-state operation, not historical replay lookup: its
    decision time must be at or after all ingested arrivals.  Feed history is
    bounded by receive time; a missing anchor fails closed.
    """

    def __init__(self, *, history_window_ms: int = 60_000) -> None:
        self.history_window_ms = _integer(history_window_ms, "history_window_ms")
        self._fast_history: list[PriceObservation] = []
        self._fast_source_times: list[int] = []
        self._latest_fast: PriceObservation | None = None
        self._latest_exact: PriceObservation | None = None
        self._anchor_fast: PriceObservation | None = None
        self._last_ingestion_sequence = -1
        self._last_receive_ts_ns = 0

    @property
    def latest_fast(self) -> PriceObservation | None:
        return self._latest_fast

    @property
    def latest_exact(self) -> PriceObservation | None:
        return self._latest_exact

    def _accept_arrival(self, observation: PriceObservation) -> None:
        if not isinstance(observation, PriceObservation):
            raise ValueError("expected PriceObservation")
        if observation.ingestion_sequence <= self._last_ingestion_sequence:
            raise ValueError("ingestion_sequence_not_increasing")
        if observation.receive_ts_ns < self._last_receive_ts_ns:
            raise ValueError("receive_timestamp_reordered")
        self._last_ingestion_sequence = observation.ingestion_sequence
        self._last_receive_ts_ns = observation.receive_ts_ns
        cutoff = observation.receive_ts_ns - self.history_window_ms * MSEC
        self._fast_history = [
            row for row in self._fast_history if row.receive_ts_ns >= cutoff
        ]
        self._fast_source_times = [row.source_ts_ns for row in self._fast_history]

    def observe_fast(self, observation: PriceObservation) -> None:
        self._accept_arrival(observation)
        index = bisect_left(self._fast_source_times, observation.source_ts_ns)
        if (
            index < len(self._fast_history)
            and self._fast_history[index].source_ts_ns == observation.source_ts_ns
        ):
            if self._fast_history[index].value != observation.value:
                raise ValueError("conflicting_fast_publication")
            return
        if (
            self._latest_fast is not None
            and self._latest_fast.source_ts_ns == observation.source_ts_ns
        ):
            if self._latest_fast.value != observation.value:
                raise ValueError("conflicting_fast_publication")
            return
        self._fast_source_times.insert(index, observation.source_ts_ns)
        self._fast_history.insert(index, observation)
        if (
            self._latest_fast is None
            or observation.source_ts_ns > self._latest_fast.source_ts_ns
        ):
            self._latest_fast = observation

    def observe_exact(self, observation: PriceObservation) -> None:
        self._accept_arrival(observation)
        if self._latest_exact is not None:
            if observation.source_ts_ns < self._latest_exact.source_ts_ns:
                return
            if observation.source_ts_ns == self._latest_exact.source_ts_ns:
                if observation.value != self._latest_exact.value:
                    raise ValueError("conflicting_exact_publication")
                return
        self._latest_exact = observation
        index = bisect_right(self._fast_source_times, observation.source_ts_ns) - 1
        self._anchor_fast = self._fast_history[index] if index >= 0 else None

    def estimate(
        self,
        decision_ts_ns: int,
        *,
        max_exact_age_ms: int = 2000,
        max_fast_age_ms: int = 750,
        max_anchor_gap_ms: int = 250,
        max_move_bps: Decimal | int | float = 25,
    ) -> dict[str, Any]:
        _integer(decision_ts_ns, "decision_ts_ns")
        for label, value in (
            ("max_exact_age_ms", max_exact_age_ms),
            ("max_fast_age_ms", max_fast_age_ms),
            ("max_anchor_gap_ms", max_anchor_gap_ms),
        ):
            _integer(value, label, minimum=0)
        move_limit = _decimal(max_move_bps, "max_move_bps")
        if move_limit < 0:
            raise ValueError("max_move_bps must be nonnegative")
        if decision_ts_ns < self._last_receive_ts_ns:
            raise ValueError("decision_precedes_observation")
        exact, current, anchor = (
            self._latest_exact, self._latest_fast, self._anchor_fast
        )
        if exact is None:
            raise ValueError("missing_exact_anchor")
        if current is None:
            raise ValueError("missing_fast_price")
        if decision_ts_ns - exact.source_ts_ns > max_exact_age_ms * MSEC:
            raise ValueError("exact_source_stale")
        if decision_ts_ns - exact.receive_ts_ns > max_exact_age_ms * MSEC:
            raise ValueError("exact_receive_stale")
        if decision_ts_ns - current.source_ts_ns > max_fast_age_ms * MSEC:
            raise ValueError("fast_source_stale")
        if decision_ts_ns - current.receive_ts_ns > max_fast_age_ms * MSEC:
            raise ValueError("fast_receive_stale")
        if anchor is None:
            raise ValueError("missing_source_aligned_fast_anchor")
        anchor_gap = exact.source_ts_ns - anchor.source_ts_ns
        if anchor_gap > max_anchor_gap_ms * MSEC:
            raise ValueError("fast_anchor_gap_exceeded")
        fast_return = current.value / anchor.value - ONE
        fast_move_bps = fast_return * BPS
        if abs(fast_move_bps) > move_limit:
            raise ValueError("fast_move_limit_exceeded")
        return {
            "estimated_index": exact.value * (ONE + fast_return),
            "method": "source_time_anchored_xag_return",
            "assumed_return_beta": "1",
            "decision_ts_ns": decision_ts_ns,
            "source_ts_ns": max(exact.source_ts_ns, current.source_ts_ns),
            "receive_ts_ns": max(exact.receive_ts_ns, current.receive_ts_ns),
            "ingestion_sequence": max(
                exact.ingestion_sequence, current.ingestion_sequence
            ),
            "exact_anchor": exact.public_dict(),
            "anchor_fast": anchor.public_dict(),
            "current_fast": current.public_dict(),
            "exact_source_age_ms": (decision_ts_ns - exact.source_ts_ns) / MSEC,
            "exact_receive_age_ms": (decision_ts_ns - exact.receive_ts_ns) / MSEC,
            "fast_source_age_ms": (decision_ts_ns - current.source_ts_ns) / MSEC,
            "fast_receive_age_ms": (decision_ts_ns - current.receive_ts_ns) / MSEC,
            "anchor_gap_ms": anchor_gap / MSEC,
            "fast_move_bps": format(fast_move_bps, "f"),
        }
