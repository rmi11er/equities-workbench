"""
v3 Strategy - Simple passive market making logic.

Core rules:
1. Only quote markets with spread >= min_market_width
2. Join depth levels, never be alone at front
3. Fade inventory gently toward neutral
4. Exception: can be alone on inventory-reducing side with sufficient edge
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from .config import ConfigV3


@dataclass
class QuoteDecision:
    """Output of strategy calculation."""
    bid_price: Optional[int] = None     # None = don't quote bid
    ask_price: Optional[int] = None     # None = don't quote ask
    bid_size: int = 0
    ask_size: int = 0
    reservation_price: float = 50.0
    reason: str = ""                    # Why we made this decision (for logging)


class PassiveStrategy:
    """
    Passive market making strategy.

    - Joins existing depth (never improves to be alone)
    - Fades inventory toward neutral
    - Only participates in markets with sufficient spread
    """

    def __init__(self, config: ConfigV3):
        self.config = config

    def compute_reservation_price(
        self,
        mid: float,
        inventory: int,
        volatility: float,
    ) -> float:
        """
        Compute inventory-adjusted reservation price.

        reservation = mid - inventory * gamma * sigma^2

        When long: reservation < mid (want to sell, ask is more competitive)
        When short: reservation > mid (want to buy, bid is more competitive)
        """
        gamma = self.config.inventory.gamma
        skew = inventory * gamma * (volatility ** 2)
        return mid - skew

    def find_joinable_level(
        self,
        book_side: dict,
        is_bid: bool,
        min_depth_dollars: float,
    ) -> Optional[int]:
        """
        Find the price level where cumulative depth >= threshold.

        For bids: walk from best (highest) down to worse prices
        For asks: walk from best (lowest) up to worse prices

        Args:
            book_side: dict of {price: size} for one side of the book
            is_bid: True for bid side, False for ask side
            min_depth_dollars: minimum cumulative depth in dollars to join

        Returns:
            Price level to join, or None if no suitable level exists
        """
        if not book_side:
            return None

        # Sort prices: bids descending (best=highest), asks ascending (best=lowest)
        prices = sorted(book_side.keys(), reverse=is_bid)

        cumulative_depth_dollars = 0.0

        for price in prices:
            size = book_side[price]
            # Convert to dollars: size * price_in_dollars
            level_dollars = size * (price / 100.0)
            cumulative_depth_dollars += level_dollars

            if cumulative_depth_dollars >= min_depth_dollars:
                return price

        # No level with sufficient depth
        return None

    def compute_quote_size(self, price: int) -> int:
        """
        Compute quote size in contracts for a given price.

        size = quote_size_dollars / price_in_dollars
        capped by max_order_size_dollars
        """
        if price <= 0:
            return 0

        price_dollars = price / 100.0
        base_size = int(self.config.sizing.quote_size_dollars / price_dollars)

        # Cap by max order size
        max_size = int(self.config.sizing.max_order_size_dollars / price_dollars)
        return max(1, min(base_size, max_size))

    def should_quote_at_price(
        self,
        price: int,
        best_price: int,
        is_bid: bool,
        inventory: int,
        mid: float,
    ) -> Tuple[bool, str]:
        """
        Determine if we should quote at this price level.

        Rules:
        1. If we'd be at a better price than BBO (alone at front) -> NO
           Exception: inventory-reducing side with sufficient edge -> YES
        2. Otherwise -> YES

        Args:
            price: Price we want to quote at
            best_price: Current BBO on this side
            is_bid: True for bid side
            inventory: Current position (+ = long, - = short)
            mid: Current mid price

        Returns:
            (should_quote, reason)
        """
        # Check if we'd be improving (alone at front)
        if is_bid:
            would_improve = price > best_price
            is_reducing = inventory > 0  # Long, selling reduces
            edge = mid - price
        else:
            would_improve = price < best_price
            is_reducing = inventory < 0  # Short, buying reduces
            edge = price - mid

        if not would_improve:
            # We're joining or behind existing depth - always OK
            return True, "joining depth"

        # We'd be alone at front - only OK if reducing with edge
        min_edge = self.config.quoting.min_edge_for_reduce

        if is_reducing and edge >= min_edge:
            return True, f"reducing inventory with {edge:.0f}c edge"

        return False, f"would be alone (edge={edge:.1f}c, reducing={is_reducing})"

    def generate_quotes(
        self,
        best_bid: Optional[int],
        best_ask: Optional[int],
        bid_book: dict,
        ask_book: dict,
        inventory: int,
        volatility: float,
    ) -> QuoteDecision:
        """
        Generate quote decision for a market.

        Args:
            best_bid: Current best bid price (or None)
            best_ask: Current best ask price (or None)
            bid_book: dict of {price: size} for bid side
            ask_book: dict of {price: size} for ask side
            inventory: Current position in contracts
            volatility: Current volatility estimate

        Returns:
            QuoteDecision with prices/sizes to quote
        """
        # Need valid BBO
        if best_bid is None or best_ask is None:
            return QuoteDecision(reason="no valid BBO")

        bbo_spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2.0

        # Gate: market too tight?
        if bbo_spread < self.config.quoting.min_market_width:
            return QuoteDecision(
                reservation_price=mid,
                reason=f"market too tight ({bbo_spread}c < {self.config.quoting.min_market_width}c)",
            )

        # Compute reservation price with inventory fade
        reservation = self.compute_reservation_price(mid, inventory, volatility)

        # Find joinable levels
        min_depth = self.config.quoting.min_join_depth_dollars
        join_bid = self.find_joinable_level(bid_book, is_bid=True, min_depth_dollars=min_depth)
        join_ask = self.find_joinable_level(ask_book, is_bid=False, min_depth_dollars=min_depth)

        # Default to BBO if no joinable level found
        bid_price = join_bid if join_bid is not None else best_bid
        ask_price = join_ask if join_ask is not None else best_ask

        # Apply inventory fade - shift both quotes toward reducing side
        # If long (reservation < mid): lower both quotes to encourage selling
        # If short (reservation > mid): raise both quotes to encourage buying
        fade_offset = int(mid - reservation)

        # Clamp fade to not cross mid aggressively
        bid_price = max(1, bid_price - fade_offset)
        ask_price = min(99, ask_price - fade_offset)

        # Ensure no cross
        if bid_price >= ask_price:
            bid_price = max(1, int(reservation) - 1)
            ask_price = min(99, int(reservation) + 1)

        # Check if we should quote each side
        should_bid, bid_reason = self.should_quote_at_price(
            bid_price, best_bid, is_bid=True, inventory=inventory, mid=mid
        )
        should_ask, ask_reason = self.should_quote_at_price(
            ask_price, best_ask, is_bid=False, inventory=inventory, mid=mid
        )

        # Check inventory limits
        max_inv_dollars = self.config.sizing.max_inventory_dollars

        # Long exposure uses YES price, short uses NO price (100 - YES)
        if inventory > 0:
            long_dollars = inventory * (mid / 100.0)
            if long_dollars >= max_inv_dollars:
                should_bid = False
                bid_reason = "at max long inventory"
        elif inventory < 0:
            short_dollars = abs(inventory) * ((100 - mid) / 100.0)
            if short_dollars >= max_inv_dollars:
                should_ask = False
                ask_reason = "at max short inventory"

        # Build result
        result = QuoteDecision(
            bid_price=bid_price if should_bid else None,
            ask_price=ask_price if should_ask else None,
            bid_size=self.compute_quote_size(bid_price) if should_bid else 0,
            ask_size=self.compute_quote_size(ask_price) if should_ask else 0,
            reservation_price=reservation,
            reason=f"bid: {bid_reason}, ask: {ask_reason}",
        )

        return result
