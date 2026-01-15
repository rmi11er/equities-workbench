"""RFQ Decision Logger - captures all RFQ decisions for post-session analysis.

Writes decisions to a JSONL file (one JSON object per line) that can be
analyzed later with analyze_rfqs.py to compare our theo vs actual execution.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class RFQLegRecord:
    """Record of a single leg in an RFQ."""
    event_ticker: str
    market_ticker: str
    side: str  # "yes" or "no"
    leg_price: Optional[float] = None  # Our BBO mid for this leg (probability)


@dataclass
class RFQDecisionRecord:
    """
    Complete record of an RFQ decision for post-session analysis.

    This captures everything needed to:
    1. Understand what RFQ we saw
    2. What we calculated as fair value
    3. What we quoted (if anything)
    4. Why we didn't quote (if filtered/skipped)
    """
    # Identifiers
    rfq_id: str
    timestamp: str  # ISO format

    # RFQ details
    contracts: int
    leg_count: int
    collection_ticker: Optional[str] = None
    legs: list[RFQLegRecord] = field(default_factory=list)

    # Our pricing
    theo_value: Optional[float] = None  # Our calculated fair value (probability)
    theo_dollars: Optional[float] = None  # theo * contracts

    # Our quote (if sent)
    action: str = "unknown"  # "quoted", "filtered", "skipped", "error"
    quote_id: Optional[str] = None
    yes_bid: Optional[str] = None  # Dollar string we quoted
    no_bid: Optional[str] = None
    edge: Optional[float] = None  # Expected edge (theo - yes_bid)

    # Filter/skip reason (if not quoted)
    filter_reason: Optional[str] = None
    filter_name: Optional[str] = None  # "pre_filter", "post_filter", "risk", "pricing"

    # Shadow mode indicator
    shadow_mode: bool = False
    spread_multiplier: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        d = asdict(self)
        # Convert legs to dicts
        d["legs"] = [asdict(leg) if hasattr(leg, "__dict__") else leg for leg in self.legs]
        return d


@dataclass
class QuoteOutcomeRecord:
    """
    Record of a quote outcome (acceptance, execution, expiry).

    Written when we receive WebSocket events about our quotes.
    """
    quote_id: str
    rfq_id: str
    timestamp: str

    event_type: str  # "accepted", "confirmed", "executed", "expired", "deleted"
    accepted_side: Optional[str] = None  # "yes" or "no" if accepted

    # Execution details (if executed)
    execution_price: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class RFQDecisionLogger:
    """
    Async-safe logger for RFQ decisions.

    Uses a background thread to write to JSONL file without blocking
    the main event loop.
    """

    def __init__(self, log_dir: str, session_id: Optional[str] = None):
        """
        Args:
            log_dir: Directory to write log files
            session_id: Optional session identifier (defaults to timestamp)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id

        # File paths
        self.decisions_path = self.log_dir / f"decisions_{session_id}.jsonl"
        self.outcomes_path = self.log_dir / f"outcomes_{session_id}.jsonl"

        # Write queue and background thread
        self._queue: Queue = Queue()
        self._running = False
        self._thread: Optional[Thread] = None

        # Stats
        self._decision_count = 0
        self._outcome_count = 0

    def start(self) -> None:
        """Start the background writer thread."""
        if self._running:
            return

        self._running = True
        self._thread = Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

        logger.info(f"RFQ Decision Logger started: {self.decisions_path}")

    def stop(self) -> None:
        """Stop the background writer thread."""
        if not self._running:
            return

        self._running = False
        self._queue.put(None)  # Sentinel to wake up thread

        if self._thread:
            self._thread.join(timeout=5.0)

        logger.info(
            f"RFQ Decision Logger stopped: "
            f"{self._decision_count} decisions, {self._outcome_count} outcomes"
        )

    def log_decision(self, record: RFQDecisionRecord) -> None:
        """Log an RFQ decision (async-safe)."""
        self._queue.put(("decision", record))
        self._decision_count += 1

    def log_outcome(self, record: QuoteOutcomeRecord) -> None:
        """Log a quote outcome (async-safe)."""
        self._queue.put(("outcome", record))
        self._outcome_count += 1

    def _writer_loop(self) -> None:
        """Background thread that writes to files."""
        decisions_file = open(self.decisions_path, "a")
        outcomes_file = open(self.outcomes_path, "a")

        try:
            while self._running or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=1.0)
                except:
                    continue

                if item is None:
                    break

                record_type, record = item

                try:
                    line = json.dumps(record.to_dict()) + "\n"

                    if record_type == "decision":
                        decisions_file.write(line)
                        decisions_file.flush()
                    elif record_type == "outcome":
                        outcomes_file.write(line)
                        outcomes_file.flush()

                except Exception as e:
                    logger.error(f"Failed to write log record: {e}")
        finally:
            decisions_file.close()
            outcomes_file.close()

    @property
    def decision_count(self) -> int:
        return self._decision_count

    @property
    def outcome_count(self) -> int:
        return self._outcome_count


def create_decision_record(
    rfq,  # RFQ object
    action: str,
    theo_value: Optional[float] = None,
    quote_id: Optional[str] = None,
    yes_bid: Optional[str] = None,
    no_bid: Optional[str] = None,
    edge: Optional[float] = None,
    filter_reason: Optional[str] = None,
    filter_name: Optional[str] = None,
    leg_prices: Optional[dict[str, float]] = None,
    shadow_mode: bool = False,
    spread_multiplier: float = 1.0,
) -> RFQDecisionRecord:
    """
    Helper to create an RFQDecisionRecord from an RFQ object.

    Args:
        rfq: The RFQ object
        action: "quoted", "filtered", "skipped", "error"
        theo_value: Our calculated fair value
        quote_id: ID of quote we sent (if any)
        yes_bid: YES bid we quoted (dollar string)
        no_bid: NO bid we quoted (dollar string)
        edge: Expected edge
        filter_reason: Why we filtered/skipped
        filter_name: Which filter triggered
        leg_prices: Dict of market_ticker -> probability for each leg
        shadow_mode: Whether running in shadow mode
        spread_multiplier: Spread multiplier used
    """
    legs = []
    for leg in rfq.mve_selected_legs:
        leg_price = None
        if leg_prices:
            leg_price = leg_prices.get(leg.market_ticker)

        legs.append(RFQLegRecord(
            event_ticker=leg.event_ticker,
            market_ticker=leg.market_ticker,
            side=leg.side,
            leg_price=leg_price,
        ))

    theo_dollars = None
    if theo_value is not None:
        theo_dollars = theo_value * rfq.contracts

    return RFQDecisionRecord(
        rfq_id=rfq.id,
        timestamp=datetime.now().isoformat(),
        contracts=rfq.contracts,
        leg_count=rfq.leg_count,
        collection_ticker=rfq.mve_collection_ticker,
        legs=legs,
        theo_value=theo_value,
        theo_dollars=theo_dollars,
        action=action,
        quote_id=quote_id,
        yes_bid=yes_bid,
        no_bid=no_bid,
        edge=edge,
        filter_reason=filter_reason,
        filter_name=filter_name,
        shadow_mode=shadow_mode,
        spread_multiplier=spread_multiplier,
    )


def create_outcome_record(
    quote_id: str,
    rfq_id: str,
    event_type: str,
    accepted_side: Optional[str] = None,
    execution_price: Optional[float] = None,
) -> QuoteOutcomeRecord:
    """Helper to create a QuoteOutcomeRecord."""
    return QuoteOutcomeRecord(
        quote_id=quote_id,
        rfq_id=rfq_id,
        timestamp=datetime.now().isoformat(),
        event_type=event_type,
        accepted_side=accepted_side,
        execution_price=execution_price,
    )
