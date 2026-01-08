"""Avellaneda-Stoikov market making strategy."""

import logging
import math
from dataclasses import dataclass
from typing import Optional

from .config import StrategyConfig
from .constants import MIN_PRICE, MAX_PRICE
from .types import StrategyOutput

logger = logging.getLogger(__name__)


@dataclass
class StoikovParams:
    """Runtime parameters for the Stoikov model."""
    mid_price: float      # S - current mid price (cents)
    inventory: int        # q - current position (positive = long YES)
    volatility: float     # σ - realized volatility (cents)
    gamma: float          # γ - risk aversion
    time_horizon: float   # T-t - time remaining (constant = 1)
    base_spread: float    # δ - base spread (cents)


class StoikovStrategy:
    """
    Avellaneda-Stoikov market making strategy adapted for binary markets.

    The model computes a reservation price that skews based on inventory,
    then brackets quotes around that price.

    Reservation Price:
        r = S - q * γ * σ² * (T - t)

    Where:
        S = mid price
        q = inventory (positive = long)
        γ = risk aversion parameter
        σ = volatility
        T - t = time horizon

    Quote Spread:
        The spread δ is placed symmetrically around r:
        bid = floor(r - δ/2)
        ask = ceil(r + δ/2)
    """

    def __init__(self, config: StrategyConfig):
        self.config = config

    def compute_reservation_price(self, params: StoikovParams) -> float:
        """
        Compute the inventory-adjusted reservation price.

        r = S - q * γ * σ² * (T - t)

        When long (q > 0), r < S (want to sell, lower ask)
        When short (q < 0), r > S (want to buy, raise bid)
        """
        inventory_skew = (
            params.inventory *
            params.gamma *
            (params.volatility ** 2) *
            params.time_horizon
        )

        return params.mid_price - inventory_skew

    def compute_spread(self, params: StoikovParams) -> float:
        """
        Compute the optimal spread.

        In the full A-S model, optimal spread depends on volatility and gamma.
        For simplicity, we use a base spread that can be configured.

        A more sophisticated version could use:
        δ = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)

        where k is a fill rate parameter.
        """
        # Base spread, potentially adjusted for volatility
        # Higher volatility -> wider spread to compensate for adverse selection
        vol_adjusted = params.base_spread * (1 + params.volatility / 10.0)
        return max(params.base_spread, vol_adjusted)

    def generate_quotes(
        self,
        mid_price: float,
        inventory: int,
        volatility: float,
        external_skew: float = 0.0,
    ) -> StrategyOutput:
        """
        Generate bid and ask quotes.

        Args:
            mid_price: Current mid price (cents)
            inventory: Current position (positive = long YES)
            volatility: Realized volatility estimate (cents)
            external_skew: Additional skew from alpha engine (cents)

        Returns:
            StrategyOutput with bid/ask prices and sizes
        """
        params = StoikovParams(
            mid_price=mid_price,
            inventory=inventory,
            volatility=volatility,
            gamma=self.config.risk_aversion,
            time_horizon=self.config.time_horizon,
            base_spread=self.config.base_spread,
        )

        # Compute reservation price
        reservation = self.compute_reservation_price(params)

        # Apply external alpha skew
        reservation += external_skew

        # Compute spread
        spread = self.compute_spread(params)

        # Generate quotes bracketing reservation price
        bid_price = math.floor(reservation - spread / 2)
        ask_price = math.ceil(reservation + spread / 2)

        # Ensure quotes don't cross
        if bid_price >= ask_price:
            # Widen spread minimally
            bid_price = math.floor(reservation - 0.5)
            ask_price = math.ceil(reservation + 0.5)

        # Clamp to valid price range
        bid_price = max(MIN_PRICE, min(MAX_PRICE - 1, bid_price))
        ask_price = max(MIN_PRICE + 1, min(MAX_PRICE, ask_price))

        # Final sanity check
        if bid_price >= ask_price:
            logger.warning(f"Quote cross after clamping: bid={bid_price}, ask={ask_price}")
            # Force valid quotes
            mid_int = int(round(mid_price))
            bid_price = max(MIN_PRICE, mid_int - 1)
            ask_price = min(MAX_PRICE, mid_int + 1)

        # Compute quote sizes based on inventory
        bid_size, ask_size = self._compute_sizes(inventory)

        # Compute inventory skew for logging
        inventory_skew = mid_price - reservation + external_skew

        return StrategyOutput(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            reservation_price=reservation,
            spread=spread,
            inventory_skew=inventory_skew,
        )

    def _compute_sizes(self, inventory: int) -> tuple[int, int]:
        """
        Compute quote sizes based on inventory.

        When inventory is high, reduce bid size to slow accumulation.
        When inventory is low (negative), reduce ask size.
        """
        base_size = self.config.quote_size
        max_inv = self.config.max_inventory

        # Inventory ratio: -1 to 1
        inv_ratio = inventory / max_inv if max_inv > 0 else 0

        # Scale sizes inversely with inventory
        # High positive inventory -> smaller bids, normal asks
        # High negative inventory -> normal bids, smaller asks
        if inv_ratio > 0:
            bid_scale = max(0.1, 1 - inv_ratio)
            ask_scale = 1.0
        else:
            bid_scale = 1.0
            ask_scale = max(0.1, 1 + inv_ratio)

        bid_size = max(1, int(base_size * bid_scale))
        ask_size = max(1, int(base_size * ask_scale))

        # Apply fat finger limit
        bid_size = min(bid_size, self.config.max_order_size)
        ask_size = min(ask_size, self.config.max_order_size)

        return bid_size, ask_size

    def should_quote(self, inventory: int) -> tuple[bool, bool]:
        """
        Determine if we should quote bid/ask based on inventory limits.

        Returns:
            (should_bid, should_ask)
        """
        max_inv = self.config.max_inventory

        # Don't bid if at max long position
        should_bid = inventory < max_inv

        # Don't ask if at max short position
        should_ask = inventory > -max_inv

        return should_bid, should_ask


class AlphaEngine:
    """
    External signal integration (stub).

    Currently returns 0.0 (no external signal).
    Future: integrate DraftKings, Polymarket, etc.
    """

    def __init__(self):
        self._enabled = False

    def get_external_skew(self, ticker: str) -> float:
        """
        Get external skew signal for a ticker.

        Args:
            ticker: Market ticker

        Returns:
            Skew in cents to add to reservation price.
            Positive = bullish (raise quotes)
            Negative = bearish (lower quotes)
        """
        # TODO: Implement external signal integration
        return 0.0

    async def start(self) -> None:
        """Start the alpha engine (e.g., connect to external APIs)."""
        self._enabled = True
        logger.info("Alpha engine started (stub mode)")

    async def stop(self) -> None:
        """Stop the alpha engine."""
        self._enabled = False
        logger.info("Alpha engine stopped")
