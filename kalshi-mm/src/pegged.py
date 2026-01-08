"""Pegged strategy for solved markets with fixed fair value."""

import logging
from dataclasses import dataclass
from typing import Optional

from .config import PeggedModeConfig, StrategyConfig
from .constants import MIN_PRICE, MAX_PRICE
from .types import StrategyOutput

logger = logging.getLogger(__name__)


@dataclass
class PeggedQuoteState:
    """Tracks current pegged quote state for reload logic."""
    current_bid_size: int = 0
    current_ask_size: int = 0


class PeggedStrategy:
    """
    Strategy for solved markets with fixed fair value.

    Used when the outcome is known/predictable, allowing the bot to
    dominate volume by quoting aggressively around a fixed price.

    Key differences from Stoikov:
    - Ignores effective_mid and volatility
    - Uses fixed fair_value as center price
    - Quotes bid at FV-1, ask at FV+1 (minimum spread)
    - Uses larger position limits (max_exposure)
    - Ignores OFI toxicity triggers (high volume = opportunity)
    """

    def __init__(self, pegged_config: PeggedModeConfig, strategy_config: StrategyConfig):
        self.pegged_config = pegged_config
        self.strategy_config = strategy_config
        self._quote_state = PeggedQuoteState()

    @property
    def fair_value(self) -> int:
        """Get the fixed fair value."""
        return self.pegged_config.fair_value

    def generate_quotes(
        self,
        inventory: int,
        current_bid_size: int = 0,
        current_ask_size: int = 0,
    ) -> StrategyOutput:
        """
        Generate quotes for pegged mode.

        Args:
            inventory: Current position (positive = long YES)
            current_bid_size: Current open bid order size
            current_ask_size: Current open ask order size

        Returns:
            StrategyOutput with fixed prices around fair_value
        """
        fv = self.pegged_config.fair_value
        max_exposure = self.pegged_config.max_exposure
        max_size = self.strategy_config.max_order_size

        # Fixed pricing: bid at FV-1, ask at FV+1
        bid_price = max(MIN_PRICE, fv - 1)
        ask_price = min(MAX_PRICE, fv + 1)

        # Ensure valid spread
        if bid_price >= ask_price:
            bid_price = max(MIN_PRICE, fv - 1)
            ask_price = min(MAX_PRICE, fv + 1)
            if bid_price >= ask_price:
                # Edge case: FV at extreme prices
                bid_price = MIN_PRICE
                ask_price = MIN_PRICE + 2

        # Domination sizing: target max size limited by exposure
        target_size = min(max_size, max_exposure - abs(inventory))
        target_size = max(0, target_size)

        # Check reload threshold
        reload_threshold = self.pegged_config.reload_threshold
        bid_size = target_size
        ask_size = target_size

        # If current size is below reload threshold, signal full refresh
        needs_bid_refresh = current_bid_size < (target_size * reload_threshold)
        needs_ask_refresh = current_ask_size < (target_size * reload_threshold)

        if not needs_bid_refresh:
            bid_size = current_bid_size  # Keep current size

        if not needs_ask_refresh:
            ask_size = current_ask_size  # Keep current size

        # Inventory skew for size (not price - price stays fixed)
        # When long, reduce bid size; when short, reduce ask size
        inv_ratio = inventory / max_exposure if max_exposure > 0 else 0

        if inv_ratio > 0.5:
            # Very long - reduce bids significantly
            bid_size = max(1, int(bid_size * (1 - inv_ratio)))
        elif inv_ratio < -0.5:
            # Very short - reduce asks significantly
            ask_size = max(1, int(ask_size * (1 + inv_ratio)))

        return StrategyOutput(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            reservation_price=float(fv),
            spread=float(ask_price - bid_price),
            inventory_skew=0.0,  # No inventory skew on price in pegged mode
        )

    def should_quote(self, inventory: int) -> tuple[bool, bool]:
        """
        Determine if we should quote bid/ask based on inventory limits.

        Uses pegged_mode.max_exposure instead of strategy.max_inventory.

        Returns:
            (should_bid, should_ask)
        """
        max_exposure = self.pegged_config.max_exposure

        # Don't bid if at max long position
        should_bid = inventory < max_exposure

        # Don't ask if at max short position
        should_ask = inventory > -max_exposure

        return should_bid, should_ask

    def needs_refresh(
        self,
        current_bid_size: int,
        current_ask_size: int,
        inventory: int,
    ) -> tuple[bool, bool]:
        """
        Check if quotes need to be refreshed based on reload threshold.

        Returns:
            (needs_bid_refresh, needs_ask_refresh)
        """
        max_exposure = self.pegged_config.max_exposure
        max_size = self.strategy_config.max_order_size
        reload_threshold = self.pegged_config.reload_threshold

        target_size = min(max_size, max_exposure - abs(inventory))
        target_size = max(0, target_size)

        needs_bid = current_bid_size < (target_size * reload_threshold)
        needs_ask = current_ask_size < (target_size * reload_threshold)

        return needs_bid, needs_ask
