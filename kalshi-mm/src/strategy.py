"""Avellaneda-Stoikov market making strategy."""

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from .config import StrategyConfig
from .constants import MIN_PRICE, MAX_PRICE
from .types import StrategyOutput

logger = logging.getLogger(__name__)

# Market impact parameter for A-S spread formula
MARKET_IMPACT_K = 1.5


def calculate_time_horizon(expiry_ts: Optional[float], normalization_sec: float = 86400.0) -> float:
    """
    Calculate dynamic time horizon that shrinks as market nears expiration.

    As expiry approaches, inventory risk increases (less time to scratch trades),
    requiring more aggressive skewing.

    Args:
        expiry_ts: Unix timestamp of market expiration (None = infinite horizon)
        normalization_sec: Seconds for T-t normalization (default 1 day)
            - If expiry > normalization_sec away, T-t = 1.0
            - If expiry < normalization_sec away, T-t decays linearly

    Returns:
        Time horizon clamped to [0.1, 1.0]
        - 1.0 = far from expiry, normal risk
        - 0.1 = near expiry, maximum urgency (but never zero)
    """
    if expiry_ts is None:
        return 1.0

    now = time.time()
    seconds_remaining = expiry_ts - now

    if seconds_remaining <= 0:
        return 0.1  # Expired or about to expire - max urgency

    fraction = seconds_remaining / normalization_sec

    # Clamp: upper 1.0, lower 0.1 (never let T-t hit 0)
    return max(0.1, min(1.0, fraction))


@dataclass
class StoikovParams:
    """Runtime parameters for the Stoikov model."""
    mid_price: float      # S - current mid price (cents)
    inventory: int        # q - current position (positive = long YES)
    volatility: float     # σ - realized volatility (cents)
    gamma: float          # γ - risk aversion
    time_horizon: float   # T-t - time remaining (dynamic, 0.1 to 1.0)
    min_spread: float     # minimum absolute spread floor (cents)


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
        Compute the optimal spread using full Avellaneda-Stoikov formula.

        δ = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)

        where:
            γ = risk aversion
            σ = volatility
            T-t = time horizon
            k = market impact parameter (MARKET_IMPACT_K = 1.5)

        The result is then floored by min_absolute_spread to ensure we're
        compensated in illiquid markets.
        """
        gamma = params.gamma
        sigma = params.volatility
        T_minus_t = params.time_horizon

        # Inventory risk component: wider spread when vol is high
        inventory_risk_term = gamma * (sigma ** 2) * T_minus_t

        # Market impact component: baseline spread from fill probability
        market_impact_term = (2 / gamma) * math.log(1 + gamma / MARKET_IMPACT_K)

        calculated_spread = inventory_risk_term + market_impact_term

        # Apply minimum absolute spread floor (safety net for illiquid markets)
        return max(calculated_spread, params.min_spread)

    def generate_quotes(
        self,
        mid_price: float,
        inventory: int,
        volatility: float,
        external_skew: float = 0.0,
        expiry_ts: Optional[float] = None,
        effective_spread: Optional[float] = None,
    ) -> StrategyOutput:
        """
        Generate bid and ask quotes.

        Args:
            mid_price: Current mid price (cents) - use effective mid for depth-based pricing
            inventory: Current position (positive = long YES)
            volatility: Realized volatility estimate (cents)
            external_skew: Additional skew from alpha engine (cents)
            expiry_ts: Market expiration Unix timestamp (None = infinite horizon)
            effective_spread: Spread based on depth-weighted prices (V2 depth-based pricing)

        Returns:
            StrategyOutput with bid/ask prices and sizes
        """
        # Calculate dynamic time horizon based on expiry
        time_horizon = calculate_time_horizon(
            expiry_ts,
            normalization_sec=self.config.time_normalization_sec,
        )

        params = StoikovParams(
            mid_price=mid_price,
            inventory=inventory,
            volatility=volatility,
            gamma=self.config.risk_aversion,
            time_horizon=time_horizon,
            min_spread=self.config.min_absolute_spread,
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
