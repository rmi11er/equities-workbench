"""Avellaneda-Stoikov market making strategy."""

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from .config import StrategyConfig, LayeringConfig
from .constants import MIN_PRICE, MAX_PRICE, OrderSide
from .types import StrategyOutput, QuoteLevel, QuoteLadder, LayerIndex

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
    max_spread: float     # maximum spread ceiling (0 = no ceiling)


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
        spread = max(calculated_spread, params.min_spread)

        # Apply maximum spread ceiling (for tight quoting in liquid markets)
        if params.max_spread > 0:
            spread = min(spread, params.max_spread)

        return spread

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
            max_spread=self.config.max_spread,
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

        # Compute quote sizes based on inventory and price (dollar-based sizing)
        bid_size, ask_size = self._compute_sizes(inventory, bid_price, ask_price)

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

    def _compute_sizes(
        self, inventory: int, bid_price: int, ask_price: int
    ) -> tuple[int, int]:
        """
        Compute quote sizes based on inventory and price.

        Uses dollar-based sizing: quote_size_dollars is converted to contracts
        based on the quote price. This ensures equal dollar risk on each side
        regardless of price level.

        Example with quote_size_dollars=50:
        - At 1c:  50 / 0.01 = 5000 contracts
        - At 50c: 50 / 0.50 = 100 contracts
        - At 99c: 50 / 0.99 = 50 contracts

        When inventory is high, reduce bid size to slow accumulation.
        When inventory is low (negative), reduce ask size.
        """
        quote_dollars = self.config.quote_size_dollars
        max_inv_dollars = self.config.max_inventory_dollars
        max_order_dollars = self.config.max_order_size_dollars

        # Convert dollars to contracts based on price
        # contracts = dollars / (price_cents / 100)
        bid_base = int(quote_dollars / (bid_price / 100)) if bid_price > 0 else 1
        ask_base = int(quote_dollars / (ask_price / 100)) if ask_price > 0 else 1

        # Convert max limits to contracts at mid price (average of bid/ask)
        mid_price = (bid_price + ask_price) / 2
        max_inv_contracts = int(max_inv_dollars / (mid_price / 100)) if mid_price > 0 else 1000
        max_order_contracts = int(max_order_dollars / (bid_price / 100)) if bid_price > 0 else 100

        # Ensure at least 1 contract
        bid_base = max(1, bid_base)
        ask_base = max(1, ask_base)

        # Inventory ratio: -1 to 1
        inv_ratio = inventory / max_inv_contracts if max_inv_contracts > 0 else 0

        # Scale sizes inversely with inventory
        # High positive inventory -> smaller bids, normal asks
        # High negative inventory -> normal bids, smaller asks
        if inv_ratio > 0:
            bid_scale = max(0.1, 1 - inv_ratio)
            ask_scale = 1.0
        else:
            bid_scale = 1.0
            ask_scale = max(0.1, 1 + inv_ratio)

        bid_size = max(1, int(bid_base * bid_scale))
        ask_size = max(1, int(ask_base * ask_scale))

        # Apply fat finger limit (converted to contracts)
        bid_size = min(bid_size, max_order_contracts)
        max_ask_contracts = int(max_order_dollars / (ask_price / 100)) if ask_price > 0 else 100
        ask_size = min(ask_size, max(1, max_ask_contracts))

        return bid_size, ask_size

    def generate_ladder(
        self,
        mid_price: float,
        inventory: int,
        volatility: float,
        layering_config: LayeringConfig,
        is_high_vol: bool = False,
        external_skew: float = 0.0,
        expiry_ts: Optional[float] = None,
    ) -> QuoteLadder:
        """
        Generate a multi-layer quote ladder.

        In Normal mode: full 3-layer ladder (Canary + Body + Wall)
        In High-Vol mode: Wall-only with wider spacing

        Args:
            mid_price: Current mid price (cents)
            inventory: Current position (positive = long YES)
            volatility: Realized volatility estimate (cents)
            layering_config: LayeringConfig with layer parameters
            is_high_vol: True if in high-vol regime (Wall-only mode)
            external_skew: Additional skew from alpha engine (cents)
            expiry_ts: Market expiration Unix timestamp

        Returns:
            QuoteLadder with bid and ask levels
        """
        if not layering_config.enabled:
            # Fallback to single quote mode
            output = self.generate_quotes(
                mid_price=mid_price,
                inventory=inventory,
                volatility=volatility,
                external_skew=external_skew,
                expiry_ts=expiry_ts,
            )
            # Convert StrategyOutput to QuoteLadder with single level
            bid = QuoteLevel(
                price=output.bid_price,
                size=output.bid_size,
                layer_index=LayerIndex.BODY,
                is_active=True,
                side=OrderSide.BUY,
            )
            ask = QuoteLevel(
                price=output.ask_price,
                size=output.ask_size,
                layer_index=LayerIndex.BODY,
                is_active=True,
                side=OrderSide.SELL,
            )
            return QuoteLadder(
                bids=[bid],
                asks=[ask],
                reservation_price=output.reservation_price,
            )

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
            max_spread=self.config.max_spread,
        )

        # Compute reservation price
        reservation = self.compute_reservation_price(params)
        reservation += external_skew

        # Compute base spread
        spread = self.compute_spread(params)

        # Anchor prices (Layer 0 = Canary at tightest spread)
        anchor_bid = math.floor(reservation - spread / 2)
        anchor_ask = math.ceil(reservation + spread / 2)

        # Ensure no cross
        if anchor_bid >= anchor_ask:
            anchor_bid = math.floor(reservation - 0.5)
            anchor_ask = math.ceil(reservation + 0.5)

        # Determine tick step and active layers based on regime
        if is_high_vol:
            tick_step = int(layering_config.base_tick_step * layering_config.high_vol_spread_multiplier)
            active_layers = {LayerIndex.WALL} if layering_config.wall_only_in_high_vol else {0, 1, 2}
        else:
            tick_step = layering_config.base_tick_step
            active_layers = {LayerIndex.CANARY, LayerIndex.BODY, LayerIndex.WALL}

        # Size ratios per layer
        size_ratios = {
            LayerIndex.CANARY: layering_config.canary_size_ratio,
            LayerIndex.BODY: layering_config.body_size_ratio,
            LayerIndex.WALL: layering_config.wall_size_ratio,
        }

        # Compute base sizes at anchor prices
        base_bid_size, base_ask_size = self._compute_sizes(inventory, anchor_bid, anchor_ask)

        bids = []
        asks = []

        for layer_idx in range(layering_config.layer_count):
            is_active = layer_idx in active_layers

            # Calculate price offset from anchor
            # Layer 0 = anchor, Layer 1 = anchor - step, Layer 2 = anchor - 2*step
            bid_price = anchor_bid - (layer_idx * tick_step)
            ask_price = anchor_ask + (layer_idx * tick_step)

            # Clamp to valid range
            bid_price = max(MIN_PRICE, min(MAX_PRICE - 1, bid_price))
            ask_price = max(MIN_PRICE + 1, min(MAX_PRICE, ask_price))

            # Compute size for this layer
            ratio = size_ratios.get(layer_idx, 1.0)
            bid_size = max(1, int(base_bid_size * ratio))
            ask_size = max(1, int(base_ask_size * ratio))

            bids.append(QuoteLevel(
                price=bid_price,
                size=bid_size,
                layer_index=layer_idx,
                is_active=is_active,
                side=OrderSide.BUY,
            ))

            asks.append(QuoteLevel(
                price=ask_price,
                size=ask_size,
                layer_index=layer_idx,
                is_active=is_active,
                side=OrderSide.SELL,
            ))

        return QuoteLadder(
            bids=bids,
            asks=asks,
            reservation_price=reservation,
        )

    def should_quote(self, inventory: int, mid_price: float = 50.0) -> tuple[bool, bool]:
        """
        Determine if we should quote bid/ask based on inventory limits.

        Args:
            inventory: Current position in contracts (+ = long YES, - = short YES/long NO)
            mid_price: Current YES mid price (used for dollar exposure calculation)

        Returns:
            (should_bid, should_ask)
        """
        # Calculate current dollar exposure at current price
        # Long YES: use YES price
        # Short YES (long NO): use NO price = 100 - YES price
        if inventory >= 0:
            price_for_exposure = mid_price
        else:
            price_for_exposure = 100.0 - mid_price  # NO price

        current_exposure_dollars = abs(inventory) * (price_for_exposure / 100)

        # Compare against dollar limit directly
        max_dollars = self.config.max_inventory_dollars

        # Don't bid if long exposure >= max
        # Don't ask if short exposure >= max
        if inventory >= 0:
            # Long YES position - check if we can add more longs
            should_bid = current_exposure_dollars < max_dollars
            should_ask = True  # Can always sell when long
        else:
            # Short YES position (long NO) - check if we can add more shorts
            should_bid = True  # Can always buy when short
            should_ask = current_exposure_dollars < max_dollars

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
