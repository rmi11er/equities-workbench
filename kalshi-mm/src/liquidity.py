"""Liquidity analysis and adaptive quoting."""

import math
from dataclasses import dataclass
from typing import Optional

from .orderbook import OrderBook


@dataclass
class LiquidityMetrics:
    """Metrics describing current market liquidity."""
    # Depth metrics
    total_bid_depth: int      # total contracts on bid side
    total_ask_depth: int      # total contracts on ask side
    depth_at_top: int         # depth at best bid + ask

    # Spread metrics
    spread: Optional[float]   # current bid-ask spread (None if no quotes)
    mid_price: Optional[float]

    # Derived
    liquidity_score: float    # 0 = empty book, 1 = very liquid
    is_empty: bool


def analyze_liquidity(book: OrderBook, levels: int = 5) -> LiquidityMetrics:
    """
    Analyze orderbook liquidity.

    Args:
        book: OrderBook to analyze
        levels: Number of price levels to consider

    Returns:
        LiquidityMetrics with current state
    """
    # Get depth on each side
    bid_prices = sorted(book.yes_bids.keys(), reverse=True)[:levels]
    ask_prices = sorted(book.yes_asks.keys())[:levels]

    total_bid = sum(book.yes_bids.get(p, 0) for p in bid_prices)
    total_ask = sum(book.yes_asks.get(p, 0) for p in ask_prices)

    # Depth at top of book
    best_bid = book.best_yes_bid()
    best_ask = book.best_yes_ask()

    depth_at_top = 0
    if best_bid is not None:
        depth_at_top += book.yes_bids.get(best_bid, 0)
    if best_ask is not None:
        depth_at_top += book.yes_asks.get(best_ask, 0)

    # Spread
    spread = None
    mid = book.mid_price()
    if best_bid is not None and best_ask is not None:
        spread = float(best_ask - best_bid)

    # Compute liquidity score (0 to 1)
    # Based on: depth, spread tightness
    is_empty = total_bid == 0 and total_ask == 0

    if is_empty:
        liquidity_score = 0.0
    else:
        # Depth component: log scale, saturates around 1000 contracts
        depth_score = min(1.0, math.log1p(total_bid + total_ask) / math.log1p(1000))

        # Spread component: tighter = more liquid
        # 1 cent spread = 1.0, 10+ cent spread = ~0.1
        if spread is not None and spread > 0:
            spread_score = min(1.0, 2.0 / spread)
        else:
            spread_score = 0.0

        # Combined score (weight depth more)
        liquidity_score = 0.7 * depth_score + 0.3 * spread_score

    return LiquidityMetrics(
        total_bid_depth=total_bid,
        total_ask_depth=total_ask,
        depth_at_top=depth_at_top,
        spread=spread,
        mid_price=mid,
        liquidity_score=liquidity_score,
        is_empty=is_empty,
    )


@dataclass
class AdaptiveParams:
    """Liquidity-adjusted quoting parameters."""
    spread_multiplier: float  # multiply base spread by this
    size_multiplier: float    # multiply base size by this
    urgency: float            # 0 = passive, 1 = aggressive (cross spread)


def compute_adaptive_params(metrics: LiquidityMetrics) -> AdaptiveParams:
    """
    Compute adaptive parameters based on liquidity.

    High liquidity (score ~1):
      - Tight spread (compete for queue)
      - Smaller size (less risk, more turns)
      - More aggressive (join best price)

    Low liquidity (score ~0):
      - Wide spread (pricing power)
      - Larger size (capture rare flow)
      - Passive (let them come to you)

    Args:
        metrics: Current liquidity metrics

    Returns:
        AdaptiveParams with multipliers
    """
    score = metrics.liquidity_score

    if metrics.is_empty:
        # Empty book: max width, max size
        return AdaptiveParams(
            spread_multiplier=50.0,  # Will result in 1/99 quotes
            size_multiplier=2.0,     # Double normal size
            urgency=0.0,
        )

    # Spread: inverse relationship with liquidity
    # High liquidity (1.0) -> multiplier 0.5 (tighter)
    # Low liquidity (0.0) -> multiplier 3.0 (wider)
    spread_multiplier = 0.5 + 2.5 * (1.0 - score)

    # Size: also inverse - in liquid markets, smaller size to get queue priority
    # High liquidity -> 0.5x size
    # Low liquidity -> 1.5x size
    size_multiplier = 0.5 + 1.0 * (1.0 - score)

    # Urgency: how aggressively to join/improve
    # High liquidity -> more urgent (need to compete)
    # Low liquidity -> passive (they come to you)
    urgency = score

    return AdaptiveParams(
        spread_multiplier=spread_multiplier,
        size_multiplier=size_multiplier,
        urgency=urgency,
    )
