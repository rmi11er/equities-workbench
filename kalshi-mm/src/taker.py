"""Impulse Engine (Taker) for emergency position management."""

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from .config import ImpulseConfig, RiskConfig, StrategyConfig

logger = logging.getLogger(__name__)


class BailoutReason(Enum):
    """Reason for triggering a bailout."""
    HARD_LIMIT = auto()           # Inventory exceeded hard stop
    RESERVATION_CROSSING = auto()  # Reservation price crossed best bid/ask
    TOXICITY_SPIKE = auto()       # OFI threshold exceeded


@dataclass
class BailoutAction:
    """Action to take when bailout is triggered."""
    reason: BailoutReason
    side: str              # "yes" or "no" - what to dump
    quantity: int          # How many contracts to dump
    aggressive_price: int  # Price to use for IOC simulation

    def __str__(self) -> str:
        return f"BAILOUT({self.reason.name}): {self.side} {self.quantity}@{self.aggressive_price}"


class OFITracker:
    """
    Tracks Order Flow Imbalance over a rolling window.

    OFI measures the net contract imbalance from trade flow.
    Positive = more buying pressure, Negative = more selling pressure.
    """

    def __init__(self, window_sec: float = 10.0):
        self.window_sec = window_sec
        self._trades: deque = deque()  # (timestamp, signed_size)
        self._rolling_ofi: int = 0

    def record_trade(self, size: int, is_buy: bool) -> None:
        """
        Record a trade tick.

        Args:
            size: Number of contracts traded
            is_buy: True if buyer-initiated (uptick), False if seller-initiated
        """
        now = time.time()
        signed_size = size if is_buy else -size
        self._trades.append((now, signed_size))
        self._rolling_ofi += signed_size
        self._prune_old()

    def _prune_old(self) -> None:
        """Remove trades outside the rolling window."""
        now = time.time()
        cutoff = now - self.window_sec

        while self._trades and self._trades[0][0] < cutoff:
            _, old_size = self._trades.popleft()
            self._rolling_ofi -= old_size

    @property
    def rolling_ofi(self) -> int:
        """Get current rolling OFI (net contracts in window)."""
        self._prune_old()
        return self._rolling_ofi

    def reset(self) -> None:
        """Clear all tracked trades."""
        self._trades.clear()
        self._rolling_ofi = 0


class ImpulseEngine:
    """
    Impulse control engine for emergency taker actions.

    Monitors risk conditions and triggers market-taking bailouts when:
    1. Hard Limit: Inventory exceeds hard_stop_ratio * max_inventory
    2. Reservation Crossing: Reservation price crosses best bid/ask
    3. Toxicity Spike: OFI imbalance exceeds threshold and opposes inventory

    Note: Triggers 2 and 3 are disabled in PEGGED mode (solved markets).
    """

    def __init__(
        self,
        impulse_config: ImpulseConfig,
        risk_config: RiskConfig,
        strategy_config: StrategyConfig,
    ):
        self.impulse_config = impulse_config
        self.risk_config = risk_config
        self.strategy_config = strategy_config
        self.ofi_tracker = OFITracker(window_sec=impulse_config.ofi_window_sec)

        # Computed limits
        self._hard_stop_inventory = risk_config.get_hard_stop_inventory(
            strategy_config.max_inventory
        )

    @property
    def enabled(self) -> bool:
        """Check if impulse control is enabled."""
        return self.impulse_config.enabled

    def record_trade(self, size: int, is_buy: bool) -> None:
        """Record a trade for OFI tracking."""
        self.ofi_tracker.record_trade(size, is_buy)

    def check_bailout(
        self,
        inventory: int,
        reservation_price: float,
        best_bid: Optional[int],
        best_ask: Optional[int],
        regime: str = "STANDARD",
    ) -> Optional[BailoutAction]:
        """
        Check if any bailout condition is triggered.

        Args:
            inventory: Current position (positive = long YES)
            reservation_price: Current reservation price from strategy
            best_bid: Best bid price in the market
            best_ask: Best ask price in the market
            regime: "STANDARD" or "PEGGED" - affects which triggers are active

        Returns:
            BailoutAction if triggered, None otherwise
        """
        if not self.enabled:
            return None

        # 1. Hard Limit Bailout - always active
        action = self._check_hard_limit(inventory, best_bid, best_ask)
        if action:
            logger.warning(f"HARD LIMIT BAILOUT: {action}")
            return action

        # Skip other checks in PEGGED mode (solved markets ignore signals)
        if regime == "PEGGED":
            return None

        # 2. Reservation Crossing Bailout
        action = self._check_reservation_crossing(
            inventory, reservation_price, best_bid, best_ask
        )
        if action:
            logger.warning(f"RESERVATION CROSSING BAILOUT: {action}")
            return action

        # 3. Toxicity Spike Bailout
        action = self._check_toxicity_spike(inventory, best_bid, best_ask)
        if action:
            logger.warning(f"TOXICITY SPIKE BAILOUT: {action}")
            return action

        return None

    def _check_hard_limit(
        self,
        inventory: int,
        best_bid: Optional[int],
        best_ask: Optional[int],
    ) -> Optional[BailoutAction]:
        """
        Check for hard limit bailout.

        Triggers when: abs(inventory) > hard_stop_ratio * max_inventory
        Action: Dump excess inventory to get back to max_inventory.
        """
        abs_inv = abs(inventory)
        if abs_inv <= self._hard_stop_inventory:
            return None

        # Calculate how much to dump to get back to max_inventory
        max_inv = self.strategy_config.max_inventory
        excess = abs_inv - max_inv

        if inventory > 0:
            # Long position - need to sell YES
            if best_bid is None:
                return None  # Can't dump without a bid
            aggressive_price = max(1, best_bid - self.impulse_config.slippage_buffer)
            return BailoutAction(
                reason=BailoutReason.HARD_LIMIT,
                side="yes",  # Selling YES
                quantity=excess,
                aggressive_price=aggressive_price,
            )
        else:
            # Short position - need to buy YES (sell NO)
            if best_ask is None:
                return None
            aggressive_price = min(99, best_ask + self.impulse_config.slippage_buffer)
            return BailoutAction(
                reason=BailoutReason.HARD_LIMIT,
                side="no",  # Buying YES by selling NO at complement
                quantity=excess,
                aggressive_price=100 - aggressive_price,  # Convert to NO price
            )

    def _check_reservation_crossing(
        self,
        inventory: int,
        reservation_price: float,
        best_bid: Optional[int],
        best_ask: Optional[int],
    ) -> Optional[BailoutAction]:
        """
        Check for reservation crossing bailout.

        Triggers when reservation price crosses the spread significantly,
        indicating our fair value is far from the market.

        - Long trigger: reservation < (best_bid - threshold)
        - Short trigger: reservation > (best_ask + threshold)
        """
        threshold = self.risk_config.bailout_threshold

        if inventory > 0 and best_bid is not None:
            # Long: if reservation drops below best bid, we're overexposed
            if reservation_price < (best_bid - threshold):
                aggressive_price = max(1, best_bid - self.impulse_config.slippage_buffer)
                return BailoutAction(
                    reason=BailoutReason.RESERVATION_CROSSING,
                    side="yes",
                    quantity=abs(inventory),  # Flatten completely
                    aggressive_price=aggressive_price,
                )

        elif inventory < 0 and best_ask is not None:
            # Short: if reservation rises above best ask, we're overexposed
            if reservation_price > (best_ask + threshold):
                aggressive_price = min(99, best_ask + self.impulse_config.slippage_buffer)
                return BailoutAction(
                    reason=BailoutReason.RESERVATION_CROSSING,
                    side="no",
                    quantity=abs(inventory),  # Flatten completely
                    aggressive_price=100 - aggressive_price,
                )

        return None

    def _check_toxicity_spike(
        self,
        inventory: int,
        best_bid: Optional[int],
        best_ask: Optional[int],
    ) -> Optional[BailoutAction]:
        """
        Check for toxicity spike bailout.

        Triggers when OFI imbalance exceeds threshold AND the flow
        direction opposes our current inventory.

        - Long + negative OFI (selling pressure) = danger
        - Short + positive OFI (buying pressure) = danger
        """
        ofi = self.ofi_tracker.rolling_ofi
        threshold = self.impulse_config.ofi_threshold

        if abs(ofi) < threshold:
            return None

        # Check if OFI opposes our inventory
        if inventory > 0 and ofi < -threshold:
            # Long position and strong selling pressure
            if best_bid is None:
                return None
            aggressive_price = max(1, best_bid - self.impulse_config.slippage_buffer)
            return BailoutAction(
                reason=BailoutReason.TOXICITY_SPIKE,
                side="yes",
                quantity=abs(inventory),  # Flatten
                aggressive_price=aggressive_price,
            )

        elif inventory < 0 and ofi > threshold:
            # Short position and strong buying pressure
            if best_ask is None:
                return None
            aggressive_price = min(99, best_ask + self.impulse_config.slippage_buffer)
            return BailoutAction(
                reason=BailoutReason.TOXICITY_SPIKE,
                side="no",
                quantity=abs(inventory),  # Flatten
                aggressive_price=100 - aggressive_price,
            )

        return None

    def reset(self) -> None:
        """Reset OFI tracker."""
        self.ofi_tracker.reset()
