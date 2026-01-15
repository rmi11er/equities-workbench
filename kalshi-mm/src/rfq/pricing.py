"""RFQ Pricing Engine with pluggable price sources."""

import logging
from typing import Optional, Protocol

from ..orderbook import OrderBookManager
from .config import PricingConfig
from .types import RFQ, RFQLeg, QuoteResponse

logger = logging.getLogger(__name__)


class PriceSource(Protocol):
    """Interface for price sources (BBO, sportsbooks, etc.)."""

    def get_yes_price(self, market_ticker: str) -> Optional[float]:
        """
        Get YES probability (0-1) for a market.

        Returns:
            Probability in range [0, 1], or None if unavailable.
        """
        ...

    def get_no_price(self, market_ticker: str) -> Optional[float]:
        """
        Get NO probability (0-1) for a market.

        Returns:
            Probability in range [0, 1], or None if unavailable.
        """
        ...


class BBOPriceSource:
    """Price source using Kalshi BBO mid prices."""

    def __init__(self, orderbook_manager: OrderBookManager):
        self.orderbook_manager = orderbook_manager

    def get_yes_price(self, market_ticker: str) -> Optional[float]:
        """
        Get YES probability from BBO mid.

        Converts cents (1-99) to probability (0.01-0.99).
        """
        book = self.orderbook_manager.get(market_ticker)
        if book is None:
            logger.debug(f"No orderbook for {market_ticker}")
            return None

        mid = book.mid_price()
        if mid is None:
            logger.debug(f"No mid price for {market_ticker}")
            return None

        # Convert cents to probability (cents 1-99 -> probability 0.01-0.99)
        return mid / 100.0

    def get_no_price(self, market_ticker: str) -> Optional[float]:
        """Get NO probability (complement of YES)."""
        yes_price = self.get_yes_price(market_ticker)
        if yes_price is None:
            return None
        return 1.0 - yes_price


class PricingEngine:
    """
    Computes fair value for RFQs including N-leg parlays.

    Combo pricing uses independent probability assumption:
        P(parlay) = P1 * P2 * ... * Pn

    For each leg, we use the BBO mid (or external source) as the
    probability, then multiply across all legs.

    NOTE: This ignores correlation between legs. For correlated events
    (e.g., same-game parlays), the true probability differs from the
    product. Correlation adjustment is a planned future enhancement.
    """

    def __init__(
        self,
        config: PricingConfig,
        primary_source: PriceSource,
        fallback_source: Optional[PriceSource] = None,
    ):
        self.config = config
        self.primary_source = primary_source
        self.fallback_source = fallback_source

    def compute_fair_value(self, rfq: RFQ) -> Optional[float]:
        """
        Compute theoretical fair value for an RFQ.

        Returns:
            Fair value as probability (0-1), or None if pricing unavailable.
        """
        if not rfq.mve_selected_legs:
            # Single market RFQ - use direct pricing
            if rfq.market_ticker:
                return self._price_single_market(rfq.market_ticker)
            return None

        # Parlay: multiply leg probabilities
        return self._price_parlay(rfq.mve_selected_legs)

    def _price_single_market(self, ticker: str) -> Optional[float]:
        """Price a single market RFQ."""
        price = self.primary_source.get_yes_price(ticker)
        if price is None and self.fallback_source:
            price = self.fallback_source.get_yes_price(ticker)
        return price

    def _price_parlay(self, legs: list[RFQLeg]) -> Optional[float]:
        """
        Price an N-leg parlay by multiplying probabilities.

        P(all legs win) = P1 * P2 * ... * Pn

        NOTE: Assumes independent legs. Correlation adjustment is not
        yet implemented.
        """
        combo_prob = 1.0
        unpriceable_legs: list[str] = []

        for leg in legs:
            leg_prob = self._get_leg_probability(leg)

            if leg_prob is None:
                unpriceable_legs.append(leg.market_ticker)
                continue

            combo_prob *= leg_prob

        if unpriceable_legs:
            logger.info(
                f"Cannot price parlay: missing prices for {unpriceable_legs}"
            )
            return None

        return combo_prob

    def _get_leg_probability(self, leg: RFQLeg) -> Optional[float]:
        """Get the probability for a single leg."""
        if leg.is_yes:
            prob = self.primary_source.get_yes_price(leg.market_ticker)
            if prob is None and self.fallback_source:
                prob = self.fallback_source.get_yes_price(leg.market_ticker)
        else:
            prob = self.primary_source.get_no_price(leg.market_ticker)
            if prob is None and self.fallback_source:
                prob = self.fallback_source.get_no_price(leg.market_ticker)

        return prob

    def get_leg_prices(self, rfq: RFQ) -> dict[str, Optional[float]]:
        """
        Get individual leg prices for logging/debugging.

        Returns:
            Dict mapping market_ticker -> probability (or None if unavailable)
        """
        result: dict[str, Optional[float]] = {}

        for leg in rfq.mve_selected_legs:
            prob = self._get_leg_probability(leg)
            result[leg.market_ticker] = prob

        return result

    def compute_quote_prices(
        self,
        fair_value: float,
        contracts: int,
    ) -> tuple[str, str]:
        """
        Compute bid prices with spread around fair value.

        The spread is applied symmetrically around the theoretical value.
        We quote our YES bid and NO bid such that taking either side
        gives us expected edge.

        Returns:
            (yes_bid, no_bid) as dollar strings like ("0.56", "0.44")
        """
        # Calculate spread in probability terms
        spread_pct = self.config.default_spread_pct

        # Convert min/max spread from cents to probability
        min_spread = self.config.min_spread_cents / 100.0
        max_spread = self.config.max_spread_cents / 100.0

        # Effective spread (clamped)
        spread = max(min_spread, min(spread_pct, max_spread))

        # Apply spread symmetrically
        # We bid for YES at (fair - spread/2)
        # We bid for NO at (1 - fair - spread/2) = complement minus half spread
        yes_bid_prob = max(0.01, fair_value - spread / 2)
        no_bid_prob = max(0.01, (1.0 - fair_value) - spread / 2)

        # Ensure probabilities stay in valid range
        yes_bid_prob = min(0.99, yes_bid_prob)
        no_bid_prob = min(0.99, no_bid_prob)

        # Convert to dollar strings (Kalshi API expects dollar format)
        yes_bid = f"{yes_bid_prob:.4f}"
        no_bid = f"{no_bid_prob:.4f}"

        return yes_bid, no_bid

    def create_quote_response(
        self,
        rfq: RFQ,
        theo_value: float,
    ) -> QuoteResponse:
        """
        Create a full quote response for an RFQ.

        Args:
            rfq: The RFQ to quote
            theo_value: Computed theoretical fair value

        Returns:
            QuoteResponse ready to send to API
        """
        yes_bid, no_bid = self.compute_quote_prices(theo_value, rfq.contracts)

        # Calculate edge (spread/2 is our expected profit per side)
        edge = (theo_value - float(yes_bid))

        return QuoteResponse(
            rfq_id=rfq.id,
            yes_bid=yes_bid,
            no_bid=no_bid,
            theo_value=theo_value,
            edge=edge,
        )
