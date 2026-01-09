"""Decision logging for post-session analysis.

Captures WHY the quoter made each decision, not just WHAT it did.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional, Any

logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
    """Types of logged decisions."""
    QUOTE_UPDATE = "quote_update"
    QUOTE_SKIP = "quote_skip"  # Debounced
    IMPULSE_CHECK = "impulse_check"
    IMPULSE_TRIGGER = "impulse_trigger"
    FILL = "fill"
    REGIME_CHANGE = "regime_change"
    ERROR = "error"
    TICK_SUMMARY = "tick_summary"


@dataclass
class LatencyBreakdown:
    """Component-level latency in microseconds."""
    orderbook_update_us: int = 0
    effective_quote_us: int = 0
    impulse_check_us: int = 0
    strategy_calc_us: int = 0
    lip_adjustment_us: int = 0
    execution_us: int = 0
    total_tick_us: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MarketState:
    """Snapshot of market state at decision time."""
    best_bid: Optional[int] = None
    best_ask: Optional[int] = None
    effective_bid: Optional[int] = None
    effective_ask: Optional[int] = None
    effective_mid: Optional[float] = None
    effective_spread: Optional[float] = None
    simple_mid: Optional[float] = None
    volatility: float = 0.0
    ofi: float = 0.0
    liquidity_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PositionState:
    """Snapshot of position at decision time."""
    inventory: int = 0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QuoteDecision:
    """Full context for a quote update decision."""
    # Timing
    timestamp: str = ""
    tick_number: int = 0

    # Regime
    regime: str = "STANDARD"  # STANDARD or PEGGED

    # Market state
    market: MarketState = field(default_factory=MarketState)

    # Position
    position: PositionState = field(default_factory=PositionState)

    # Strategy calculation
    reservation_price: float = 0.0
    raw_bid: int = 0
    raw_ask: int = 0
    raw_spread: float = 0.0
    inventory_skew: float = 0.0

    # Adjustments applied
    liquidity_spread_mult: float = 1.0
    liquidity_size_mult: float = 1.0
    lip_adjusted: bool = False
    lip_min_size: int = 0
    lip_max_distance: int = 0

    # Final output
    final_bid: int = 0
    final_ask: int = 0
    final_bid_size: int = 0
    final_ask_size: int = 0
    should_bid: bool = True
    should_ask: bool = True

    # Why this update happened
    update_reason: str = ""  # "price_change", "time_elapsed", "forced", "initial"

    # Latency
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["market"] = self.market.to_dict()
        d["position"] = self.position.to_dict()
        d["latency"] = self.latency.to_dict()
        return d


@dataclass
class ImpulseEvent:
    """Impulse check/trigger event."""
    timestamp: str = ""
    tick_number: int = 0
    triggered: bool = False
    reason: str = ""  # HARD_LIMIT, RESERVATION_CROSSING, TOXICITY_SPIKE
    inventory: int = 0
    reservation_price: float = 0.0
    best_bid: Optional[int] = None
    best_ask: Optional[int] = None
    ofi: float = 0.0
    hard_stop_threshold: int = 0
    bailout_side: str = ""
    bailout_quantity: int = 0
    bailout_price: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FillEvent:
    """Fill event with full context."""
    timestamp: str = ""
    order_id: str = ""
    side: str = ""
    price: int = 0
    size: int = 0
    # Position change
    inventory_before: int = 0
    inventory_after: int = 0
    # P&L impact
    realized_pnl_delta: float = 0.0
    total_realized_pnl: float = 0.0
    # Market context at fill
    mid_at_fill: float = 0.0
    our_bid_at_fill: Optional[int] = None
    our_ask_at_fill: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecisionLogEntry:
    """Generic log entry wrapper."""
    type: str
    data: dict
    seq: int = 0

    def to_json(self) -> str:
        return json.dumps({
            "seq": self.seq,
            "type": self.type,
            **self.data
        }, default=str)


class DecisionLogger:
    """
    Logs all trading decisions with full context for post-session analysis.

    Output: JSONL file (one JSON object per line) for easy parsing.
    """

    def __init__(self, log_path: str = "logs/decisions.jsonl"):
        self._log_path = log_path
        self._queue: Queue[Optional[DecisionLogEntry]] = Queue()
        self._thread: Optional[Thread] = None
        self._running = False
        self._seq = 0

        # Ensure directory exists
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start the background writer thread."""
        if self._running:
            return

        self._running = True
        self._seq = 0
        self._thread = Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        logger.info(f"Decision logger started: {self._log_path}")

    def stop(self) -> None:
        """Stop the background writer thread."""
        if not self._running:
            return

        self._running = False
        self._queue.put(None)

        if self._thread:
            self._thread.join(timeout=5.0)

        logger.info(f"Decision logger stopped. {self._seq} entries written.")

    def _writer_loop(self) -> None:
        """Background thread that writes entries to JSONL file."""
        with open(self._log_path, "a") as f:
            while self._running or not self._queue.empty():
                try:
                    entry = self._queue.get(timeout=1.0)
                    if entry is None:
                        break
                    f.write(entry.to_json() + "\n")
                    f.flush()
                except Exception:
                    pass

    def _log(self, entry_type: str, data: dict) -> None:
        """Queue a log entry."""
        if not self._running:
            return
        self._seq += 1
        entry = DecisionLogEntry(type=entry_type, data=data, seq=self._seq)
        self._queue.put(entry)

    def log_quote_update(self, decision: QuoteDecision) -> None:
        """Log a quote update decision."""
        self._log(DecisionType.QUOTE_UPDATE, decision.to_dict())

    def log_quote_skip(self, tick_number: int, reason: str,
                       last_bid: int, last_ask: int,
                       target_bid: int, target_ask: int) -> None:
        """Log a skipped quote update (debounced)."""
        self._log(DecisionType.QUOTE_SKIP, {
            "timestamp": datetime.now().isoformat(),
            "tick_number": tick_number,
            "reason": reason,
            "last_bid": last_bid,
            "last_ask": last_ask,
            "target_bid": target_bid,
            "target_ask": target_ask,
        })

    def log_impulse_check(self, event: ImpulseEvent) -> None:
        """Log an impulse check (triggered or not)."""
        entry_type = DecisionType.IMPULSE_TRIGGER if event.triggered else DecisionType.IMPULSE_CHECK
        self._log(entry_type, event.to_dict())

    def log_fill(self, event: FillEvent) -> None:
        """Log a fill event."""
        self._log(DecisionType.FILL, event.to_dict())

    def log_regime_change(self, old_regime: str, new_regime: str, reason: str) -> None:
        """Log a regime change."""
        self._log(DecisionType.REGIME_CHANGE, {
            "timestamp": datetime.now().isoformat(),
            "old_regime": old_regime,
            "new_regime": new_regime,
            "reason": reason,
        })

    def log_error(self, error: str, context: dict = None) -> None:
        """Log an error with context."""
        self._log(DecisionType.ERROR, {
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "context": context or {},
        })

    def log_tick_summary(self, tick_number: int, latency: LatencyBreakdown) -> None:
        """Log tick-level latency summary."""
        self._log(DecisionType.TICK_SUMMARY, {
            "timestamp": datetime.now().isoformat(),
            "tick_number": tick_number,
            "latency": latency.to_dict(),
        })


class LatencyTimer:
    """Context manager for measuring component latency."""

    def __init__(self):
        self._start_ns: int = 0
        self._elapsed_us: int = 0

    def __enter__(self) -> "LatencyTimer":
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *args) -> None:
        self._elapsed_us = (time.perf_counter_ns() - self._start_ns) // 1000

    @property
    def elapsed_us(self) -> int:
        return self._elapsed_us


class TickLatencyTracker:
    """Tracks latency breakdown for a single tick."""

    def __init__(self):
        self.breakdown = LatencyBreakdown()
        self._tick_start_ns = time.perf_counter_ns()

    def record_orderbook_update(self, us: int) -> None:
        self.breakdown.orderbook_update_us = us

    def record_effective_quote(self, us: int) -> None:
        self.breakdown.effective_quote_us = us

    def record_impulse_check(self, us: int) -> None:
        self.breakdown.impulse_check_us = us

    def record_strategy_calc(self, us: int) -> None:
        self.breakdown.strategy_calc_us = us

    def record_lip_adjustment(self, us: int) -> None:
        self.breakdown.lip_adjustment_us = us

    def record_execution(self, us: int) -> None:
        self.breakdown.execution_us = us

    def finalize(self) -> LatencyBreakdown:
        self.breakdown.total_tick_us = (time.perf_counter_ns() - self._tick_start_ns) // 1000
        return self.breakdown
