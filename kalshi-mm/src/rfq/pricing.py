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

    def get_effective_spread(self, market_ticker: str) -> Optional[float]:
        """
        Get the effective spread (in probability terms) for a market.

        This is the spread at the depth where sufficient liquidity exists.
        Used for computing combo spread from underlying leg liquidity.

        Returns:
            Spread as probability (e.g., 0.05 = 5 cents), or None if unavailable.
        """
        ...


class BBOPriceSource:
    """
    Price source using Kalshi BBO mid prices with liquidity filtering.

    A market is considered "liquid enough" to price if we can find
    min_liquidity_dollars on each side of the book within max_spread_cents.
    This ensures we're not pricing based on thin, unreliable orderbooks.
    """

    def __init__(
        self,
        orderbook_manager: OrderBookManager,
        min_liquidity_dollars: float = 10.0,
        max_spread_cents: int = 20,
    ):
        """
        Args:
            orderbook_manager: Manager for orderbook data
            min_liquidity_dollars: Minimum $ required on each side
            max_spread_cents: Maximum spread (at required depth) to consider liquid
                              0 means no spread limit
        """
        self.orderbook_manager = orderbook_manager
        self.min_liquidity_dollars = min_liquidity_dollars
        self.max_spread_cents = max_spread_cents

    def get_yes_price(self, market_ticker: str) -> Optional[float]:
        """
        Get YES probability from BBO mid, if market has sufficient liquidity.

        Returns None if:
        - No orderbook exists
        - Book is empty
        - Insufficient liquidity within spread limit

        Converts cents (1-99) to probability (0.01-0.99).
        """
        book = self.orderbook_manager.get(market_ticker)
        if book is None:
            logger.debug(f"No orderbook for {market_ticker}")
            return None

        # Check liquidity - returns (is_liquid, effective_spread, mid)
        is_liquid, eff_spread, mid = self._check_liquidity(book)

        if not is_liquid:
            logger.debug(
                f"{market_ticker} illiquid: spread={eff_spread}c at "
                f"${self.min_liquidity_dollars} depth (max={self.max_spread_cents}c)"
            )
            return None

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

    def get_effective_spread(self, market_ticker: str) -> Optional[float]:
        """
        Get the effective spread at liquidity depth for a market.

        Returns spread in probability terms (cents / 100).
        """
        book = self.orderbook_manager.get(market_ticker)
        if book is None:
            return None

        is_liquid, eff_spread_cents, _ = self._check_liquidity(book)
        if not is_liquid or eff_spread_cents is None:
            return None

        # Convert cents to probability
        return eff_spread_cents / 100.0

    def _check_liquidity(self, book) -> tuple[bool, Optional[float], Optional[float]]:
        """
        Check if book has sufficient liquidity within spread limit.

        Walks through bid and ask levels, accumulating dollar liquidity
        until we hit the target on each side. The "effective spread" is
        the spread at the depth where we found sufficient liquidity.

        Returns:
            (is_liquid, effective_spread_cents, mid_price)
            is_liquid is False if we can't find min_liquidity within max_spread
        """
        # If no liquidity requirement, just check for valid BBO
        if self.min_liquidity_dollars <= 0:
            mid = book.mid_price()
            spread = book.spread()
            return (mid is not None), spread, mid

        # Walk bids (descending price) to find where we accumulate target $
        worst_bid = None
        cumulative_bid_dollars = 0.0
        for price in sorted(book.yes_bids.keys(), reverse=True):
            size = book.yes_bids[price]
            # Dollar value = price_cents / 100 * num_contracts
            dollars = (price / 100.0) * size
            cumulative_bid_dollars += dollars
            worst_bid = price
            if cumulative_bid_dollars >= self.min_liquidity_dollars:
                break

        # Walk asks (ascending price) to find where we accumulate target $
        worst_ask = None
        cumulative_ask_dollars = 0.0
        for price in sorted(book.yes_asks.keys()):
            size = book.yes_asks[price]
            # Dollar value for asks: we pay the ask price
            dollars = (price / 100.0) * size
            cumulative_ask_dollars += dollars
            worst_ask = price
            if cumulative_ask_dollars >= self.min_liquidity_dollars:
                break

        # Check if we found enough liquidity
        if worst_bid is None or worst_ask is None:
            return False, None, None

        bid_sufficient = cumulative_bid_dollars >= self.min_liquidity_dollars
        ask_sufficient = cumulative_ask_dollars >= self.min_liquidity_dollars

        if not (bid_sufficient and ask_sufficient):
            # Not enough total liquidity in the book
            effective_spread = worst_ask - worst_bid if worst_ask and worst_bid else None
            return False, effective_spread, None

        # Compute effective spread at the depth where we found liquidity
        effective_spread = worst_ask - worst_bid

        # Check spread limit (0 means no limit)
        if self.max_spread_cents > 0 and effective_spread > self.max_spread_cents:
            return False, effective_spread, None

        # Market is liquid - return BBO mid as fair value
        mid = book.mid_price()
        return True, effective_spread, mid


class PricingEngine:
    """
    Computes fair value for RFQs including N-leg parlays.

    Combo pricing uses independent probability assumption:
        P(parlay) = P1 * P2 * ... * Pn

    For each leg, we use the BBO mid (or external source) as the
    probability, then multiply across all legs.

    Spread Calculation:
        combo_spread = liquidity_spread + correlation_premium

        - liquidity_spread: Derived from underlying leg effective spreads.
          We must charge at least as wide as the widest leg to avoid being
          arbed through the underlying markets.

        - correlation_premium: Additional spread for correlated legs.
          (NOT YET IMPLEMENTED - skeleton for future math engine update)

    NOTE: Correlation adjustment for theo is also a planned future enhancement.
    Currently assumes independent legs for both theo and spread.
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

        # Cache leg spreads from most recent pricing call
        self._last_leg_spreads: dict[str, float] = {}

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

        Also caches leg spreads for use in spread calculation.

        NOTE: Assumes independent legs. Correlation adjustment is not
        yet implemented.
        """
        combo_prob = 1.0
        unpriceable_legs: list[str] = []
        self._last_leg_spreads.clear()

        for leg in legs:
            leg_prob = self._get_leg_probability(leg)

            if leg_prob is None:
                unpriceable_legs.append(leg.market_ticker)
                continue

            combo_prob *= leg_prob

            # Cache leg spread for spread calculation
            leg_spread = self._get_leg_spread(leg.market_ticker)
            if leg_spread is not None:
                self._last_leg_spreads[leg.market_ticker] = leg_spread

        if unpriceable_legs:
            logger.info(
                f"Cannot price parlay: missing prices for {unpriceable_legs}"
            )
            return None

        return combo_prob

    def _get_leg_spread(self, market_ticker: str) -> Optional[float]:
        """Get effective spread for a leg market."""
        spread = None
        if hasattr(self.primary_source, 'get_effective_spread'):
            spread = self.primary_source.get_effective_spread(market_ticker)
        if spread is None and self.fallback_source and hasattr(self.fallback_source, 'get_effective_spread'):
            spread = self.fallback_source.get_effective_spread(market_ticker)
        return spread

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

    def compute_combo_spread(self, legs: list[RFQLeg]) -> float:
        """
        Compute the spread for a combo based on underlying leg liquidity.

        The combo spread has two components:
        1. Liquidity spread: Based on underlying leg effective spreads
        2. Correlation premium: Additional charge for correlated legs

        Returns:
            Spread in probability terms (e.g., 0.05 = 5 cents)
        """
        liquidity_spread = self._compute_liquidity_spread(legs)
        correlation_premium = self._compute_correlation_premium(legs)

        total_spread = liquidity_spread + correlation_premium

        # Clamp to config bounds
        min_spread = self.config.min_spread_cents / 100.0
        max_spread = self.config.max_spread_cents / 100.0

        return max(min_spread, min(total_spread, max_spread))

    def _compute_liquidity_spread(self, legs: list[RFQLeg]) -> float:
        """
        Compute spread component based on underlying leg liquidity.

        Strategy: Use the maximum leg spread as a baseline. We can't quote
        tighter than the widest underlying market, or we'd be arbed.

        For multi-leg parlays, we add a fraction of the other leg spreads
        to account for the increased complexity of hedging.

        Returns:
            Liquidity-based spread in probability terms
        """
        if not self._last_leg_spreads:
            # No spread data - fall back to config default
            return self.config.default_spread_pct

        leg_spreads = list(self._last_leg_spreads.values())

        if len(leg_spreads) == 1:
            # Single leg: just use that spread
            return leg_spreads[0]

        # Multi-leg: max spread + fraction of others
        # This accounts for hedging complexity without being too punitive
        sorted_spreads = sorted(leg_spreads, reverse=True)
        max_spread = sorted_spreads[0]
        other_spreads_sum = sum(sorted_spreads[1:])

        # Add 25% of other leg spreads to the max
        # (tunable parameter - could be in config)
        return max_spread + 0.25 * other_spreads_sum

    def _compute_correlation_premium(self, legs: list[RFQLeg]) -> float:
        """
        Compute additional spread for correlated legs.

        *** NOT YET IMPLEMENTED - SKELETON FOR FUTURE MATH ENGINE UPDATE ***

        When legs are correlated (e.g., same-game parlay, related events),
        we need to charge more spread because:
        1. Our theo is less accurate (independence assumption breaks down)
        2. Adverse selection is higher (informed traders exploit correlation)

        Future implementation should consider:
        - Same-event detection (legs from same game/match)
        - Historical correlation estimation
        - Sport-specific correlation models
        - Opposite-side legs (negatively correlated, e.g., both teams can't win)

        Returns:
            Correlation premium in probability terms (currently 0)
        """
        # TODO: Implement correlation detection and premium calculation
        #
        # Pseudocode for future implementation:
        #
        # 1. Detect same-event legs:
        #    same_event_groups = group_legs_by_event(legs)
        #    if any group has > 1 leg: add same_game_premium
        #
        # 2. Detect impossible combos (mutually exclusive legs):
        #    if legs_are_mutually_exclusive(legs):
        #        return INFINITY  # or reject the RFQ
        #
        # 3. Look up historical correlation:
        #    for each pair of legs:
        #        corr = get_correlation(leg1.event_ticker, leg2.event_ticker)
        #        premium += corr * CORRELATION_MULTIPLIER
        #
        # 4. Apply sport-specific adjustments:
        #    if is_player_prop(leg) and is_game_outcome(other_leg):
        #        premium += PLAYER_GAME_CORRELATION_PREMIUM

        return 0.0  # No correlation premium until implemented

    def compute_quote_prices(
        self,
        fair_value: float,
        contracts: int,
        legs: Optional[list[RFQLeg]] = None,
    ) -> tuple[str, str]:
        """
        Compute bid prices with spread around fair value.

        The spread is computed dynamically based on:
        1. Underlying leg liquidity (effective spreads)
        2. Correlation premium (not yet implemented)

        The spread is applied symmetrically around the theoretical value.
        We quote our YES bid and NO bid such that taking either side
        gives us expected edge.

        Args:
            fair_value: Theoretical fair value (probability)
            contracts: Number of contracts
            legs: Optional list of legs for spread calculation

        Returns:
            (yes_bid, no_bid) as dollar strings like ("0.56", "0.44")
        """
        # Calculate spread based on leg liquidity
        if legs:
            spread = self.compute_combo_spread(legs)
        else:
            # Fallback to config default for single-market RFQs
            spread = self.config.default_spread_pct
            min_spread = self.config.min_spread_cents / 100.0
            max_spread = self.config.max_spread_cents / 100.0
            spread = max(min_spread, min(spread, max_spread))

        # Apply spread symmetrically
        # We bid for YES at (fair - spread/2)
        # We bid for NO at (1 - fair - spread/2) = complement minus half spread
        yes_bid_prob = max(0.01, fair_value - spread / 2)
        no_bid_prob = max(0.01, (1.0 - fair_value) - spread / 2)

        # Ensure probabilities stay in valid range
        yes_bid_prob = min(0.99, yes_bid_prob)
        no_bid_prob = min(0.99, no_bid_prob)

        # Round to whole cents (Kalshi requires 2 decimal places)
        # Round DOWN for bids (be conservative)
        yes_bid_prob = int(yes_bid_prob * 100) / 100.0
        no_bid_prob = int(no_bid_prob * 100) / 100.0

        # Ensure minimum 1 cent after rounding
        yes_bid_prob = max(0.01, yes_bid_prob)
        no_bid_prob = max(0.01, no_bid_prob)

        # Convert to dollar strings (Kalshi API expects dollar format, 2 decimal places)
        yes_bid = f"{yes_bid_prob:.2f}"
        no_bid = f"{no_bid_prob:.2f}"

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
        yes_bid, no_bid = self.compute_quote_prices(
            theo_value, rfq.contracts, rfq.mve_selected_legs
        )

        # Calculate edge (spread/2 is our expected profit per side)
        edge = (theo_value - float(yes_bid))

        return QuoteResponse(
            rfq_id=rfq.id,
            yes_bid=yes_bid,
            no_bid=no_bid,
            theo_value=theo_value,
            edge=edge,
        )

    def get_last_leg_spreads(self) -> dict[str, float]:
        """Get the leg spreads from the most recent pricing call (for logging)."""
        return self._last_leg_spreads.copy()
