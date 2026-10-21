"""RFQ type definitions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class QuoteStatus(Enum):
    """Status of an RFQ quote."""
    PENDING = "pending"        # Quote sent, awaiting response
    ACCEPTED = "accepted"      # User accepted, awaiting our confirmation
    CONFIRMED = "confirmed"    # We confirmed, awaiting execution
    EXECUTED = "executed"      # Trade completed
    EXPIRED = "expired"        # Quote expired (user didn't accept in time)
    REJECTED = "rejected"      # Quote was rejected
    FAILED = "failed"          # Confirmation or execution failed


class RFQStatus(Enum):
    """Status of an RFQ."""
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class RFQLeg:
    """A single leg in an RFQ (for parlays)."""
    event_ticker: str
    market_ticker: str
    side: str  # "yes" or "no"
    yes_settlement_value_dollars: str = "1.00"

    @property
    def is_yes(self) -> bool:
        """Check if this leg is betting on YES outcome."""
        return self.side.lower() == "yes"


@dataclass
class RFQ:
    """Incoming RFQ from Kalshi."""
    id: str
    contracts: int
    created_at: datetime
    status: RFQStatus = RFQStatus.OPEN

    # Single market RFQ
    market_ticker: Optional[str] = None

    # Parlay RFQ (multivariate event)
    mve_collection_ticker: Optional[str] = None
    mve_selected_legs: list[RFQLeg] = field(default_factory=list)

    # Optional fields
    target_cost_centi_cents: Optional[int] = None  # Requester's target price (in centi-cents)
    expires_at: Optional[datetime] = None
    creator_id: Optional[str] = None
    rest_remainder: bool = False

    @property
    def target_cost_dollars(self) -> Optional[float]:
        """Requester's target cost in dollars (CashOrderQty equivalent)."""
        if self.target_cost_centi_cents is None:
            return None
        return self.target_cost_centi_cents / 10000.0

    @property
    def is_parlay(self) -> bool:
        """Check if this is a multi-leg parlay RFQ."""
        return len(self.mve_selected_legs) > 1

    @property
    def leg_count(self) -> int:
        """Number of legs in this RFQ."""
        return max(1, len(self.mve_selected_legs))

    @property
    def all_market_tickers(self) -> list[str]:
        """Get all market tickers involved in this RFQ."""
        if self.mve_selected_legs:
            return [leg.market_ticker for leg in self.mve_selected_legs]
        elif self.market_ticker:
            return [self.market_ticker]
        return []


@dataclass
class QuoteResponse:
    """Our quote response to an RFQ."""
    rfq_id: str
    yes_bid: str  # Dollar string like "0.56"
    no_bid: str   # Dollar string like "0.44"
    theo_value: float  # Our calculated fair value (probability 0-1)
    yes_edge: float  # Edge if they sell us YES (theo - yes_bid)
    no_edge: float   # Edge if they sell us NO ((1-theo) - no_bid)

    @property
    def edge(self) -> float:
        """Best edge across both sides (for backward compatibility)."""
        return max(self.yes_edge, self.no_edge)

    def to_api_payload(self, one_sided: bool = False) -> dict:
        """Convert to API request payload.

        Args:
            one_sided: If True, only include the side with positive edge.
                      This prevents us from being picked off on the negative edge side.
        """
        payload = {
            "rfq_id": self.rfq_id,
            "rest_remainder": False,
        }

        if one_sided:
            # Only quote the side with positive edge
            if self.yes_edge >= 0:
                payload["yes_bid"] = self.yes_bid
            if self.no_edge >= 0:
                payload["no_bid"] = self.no_bid
        else:
            # Always include both sides
            payload["yes_bid"] = self.yes_bid
            payload["no_bid"] = self.no_bid

        return payload

    @property
    def is_one_sided(self) -> bool:
        """Check if this quote should be one-sided (one edge negative)."""
        return self.yes_edge < 0 or self.no_edge < 0

    @property
    def profitable_side(self) -> Optional[str]:
        """Return which side has positive edge, or None if both do."""
        if self.yes_edge >= 0 and self.no_edge >= 0:
            return None  # Both sides profitable
        elif self.yes_edge >= 0:
            return "yes"
        elif self.no_edge >= 0:
            return "no"
        return None  # Neither side profitable (shouldn't happen)


@dataclass
class ActiveQuote:
    """Tracks a quote we've sent, pending acceptance."""
    quote_id: str
    rfq_id: str
    rfq: RFQ
    response: QuoteResponse
    sent_at: datetime
    status: QuoteStatus = QuoteStatus.PENDING

    # Tracking fields
    accepted_at: Optional[datetime] = None
    confirmed_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    accepted_side: Optional[str] = None  # "yes" or "no" - which side user took

    @property
    def dollar_exposure(self) -> float:
        """Calculate dollar exposure for this quote."""
        return self.rfq.contracts * self.response.theo_value

    @property
    def age_seconds(self) -> float:
        """Time since quote was sent."""
        return (datetime.now() - self.sent_at).total_seconds()


@dataclass
class RFQDecision:
    """Record of a decision made on an RFQ (for logging)."""
    rfq_id: str
    timestamp: datetime
    action: str  # "quoted", "skipped", "filtered"
    reason: Optional[str] = None

    # Pricing context (if quoted)
    theo_value: Optional[float] = None
    yes_bid: Optional[str] = None
    no_bid: Optional[str] = None
    leg_prices: Optional[dict[str, float]] = None  # market_ticker -> probability

    # Filter context (if filtered/skipped)
    filter_name: Optional[str] = None
    rfq_dollars: Optional[float] = None
    leg_count: Optional[int] = None
