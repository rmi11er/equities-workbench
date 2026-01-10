"""Volatility regime detection and response coordination."""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VolatilityRegimeConfig:
    """Configuration for volatility regime detection."""
    # Spread-based trigger
    spread_threshold: int = 10  # Enter high vol if spread >= this (cents)

    # Fill-based trigger
    fills_threshold: int = 3      # Enter high vol if N fills in window
    fills_window_sec: float = 30.0  # Window for counting fills

    # Depth multiplier (require more liquidity in high vol)
    initial_depth_multiplier: float = 3.0  # Start at 3x min_depth when entering
    depth_decay_halflife_sec: float = 60.0  # Multiplier decays back to 1.0

    # Tick rate adjustment
    normal_tick_interval: float = 0.1    # 100ms normally
    high_vol_tick_interval: float = 0.03  # 30ms in high vol (faster updates)

    # Exit conditions
    calm_spread_threshold: int = 5  # Exit high vol when spread <= this
    calm_duration_sec: float = 30.0  # Must be calm for this long to exit


@dataclass
class MarketRegimeState:
    """Per-market volatility regime state."""
    ticker: str

    # Current state
    is_high_vol: bool = False
    entered_at: Optional[float] = None  # monotonic timestamp
    last_spread: int = 0

    # Calm tracking for exit
    calm_since: Optional[float] = None  # When spread dropped below calm threshold

    # Fill tracking
    recent_fill_times: list = field(default_factory=list)

    def record_fill(self, timestamp: float) -> None:
        """Record a fill timestamp."""
        self.recent_fill_times.append(timestamp)

    def count_recent_fills(self, window_sec: float) -> int:
        """Count fills within the window."""
        now = time.monotonic()
        cutoff = now - window_sec
        self.recent_fill_times = [t for t in self.recent_fill_times if t > cutoff]
        return len(self.recent_fill_times)


class VolatilityRegime:
    """
    Detects and manages volatility regime per market.

    Triggers high-vol state when:
    1. Spread blows out beyond threshold
    2. Multiple fills in short time window

    In high-vol state:
    1. Tick interval decreases (faster quote updates)
    2. min_join_depth_dollars multiplier increases (require more cover)
    3. Multiplier decays over time back to normal
    """

    def __init__(self, config: Optional[VolatilityRegimeConfig] = None):
        self.config = config or VolatilityRegimeConfig()
        self._states: dict[str, MarketRegimeState] = {}

    def get_state(self, ticker: str) -> MarketRegimeState:
        """Get or create state for a ticker."""
        if ticker not in self._states:
            self._states[ticker] = MarketRegimeState(ticker=ticker)
        return self._states[ticker]

    def update(
        self,
        ticker: str,
        spread: int,
        had_fill: bool = False,
    ) -> bool:
        """
        Update regime state based on current market conditions.

        Args:
            ticker: Market ticker
            spread: Current bid-ask spread in cents
            had_fill: Whether we just had a fill

        Returns:
            True if state changed (entered or exited high vol)
        """
        state = self.get_state(ticker)
        now = time.monotonic()
        old_state = state.is_high_vol

        # Record fill if we had one
        if had_fill:
            state.record_fill(now)

        state.last_spread = spread

        # Check entry conditions
        if not state.is_high_vol:
            should_enter = False
            reason = ""

            # Spread trigger
            if spread >= self.config.spread_threshold:
                should_enter = True
                reason = f"spread={spread}c >= {self.config.spread_threshold}c"

            # Fill frequency trigger
            recent_fills = state.count_recent_fills(self.config.fills_window_sec)
            if recent_fills >= self.config.fills_threshold:
                should_enter = True
                reason = f"{recent_fills} fills in {self.config.fills_window_sec}s"

            if should_enter:
                state.is_high_vol = True
                state.entered_at = now
                state.calm_since = None
                logger.warning(f"[{ticker}] ENTERING HIGH VOL: {reason}")

        # Check exit conditions
        else:
            # Track calm period
            if spread <= self.config.calm_spread_threshold:
                if state.calm_since is None:
                    state.calm_since = now
                elif now - state.calm_since >= self.config.calm_duration_sec:
                    # Been calm long enough, exit high vol
                    state.is_high_vol = False
                    state.entered_at = None
                    state.calm_since = None
                    logger.info(f"[{ticker}] EXITING HIGH VOL: calm for {self.config.calm_duration_sec}s")
            else:
                # Spread spiked again, reset calm timer
                state.calm_since = None

        return state.is_high_vol != old_state

    def get_depth_multiplier(self, ticker: str) -> float:
        """
        Get the current depth multiplier for a ticker.

        In high vol: starts at initial_depth_multiplier, decays toward 1.0
        In normal: returns 1.0

        Returns:
            Multiplier to apply to min_join_depth_dollars
        """
        state = self.get_state(ticker)

        if not state.is_high_vol or state.entered_at is None:
            return 1.0

        # Calculate decayed multiplier
        # multiplier(t) = 1 + (initial - 1) * exp(-ln(2) * t / halflife)
        elapsed = time.monotonic() - state.entered_at
        decay_factor = math.exp(-math.log(2) * elapsed / self.config.depth_decay_halflife_sec)

        multiplier = 1.0 + (self.config.initial_depth_multiplier - 1.0) * decay_factor
        return multiplier

    def get_tick_interval(self, ticker: str) -> float:
        """
        Get the tick interval for a ticker.

        Returns faster interval in high vol state.
        """
        state = self.get_state(ticker)

        if state.is_high_vol:
            return self.config.high_vol_tick_interval
        return self.config.normal_tick_interval

    def is_high_vol(self, ticker: str) -> bool:
        """Check if ticker is in high vol state."""
        return self.get_state(ticker).is_high_vol

    def get_status(self, ticker: str) -> dict:
        """Get detailed status for logging/debugging."""
        state = self.get_state(ticker)

        status = {
            "ticker": ticker,
            "is_high_vol": state.is_high_vol,
            "last_spread": state.last_spread,
            "recent_fills": state.count_recent_fills(self.config.fills_window_sec),
        }

        if state.is_high_vol and state.entered_at:
            elapsed = time.monotonic() - state.entered_at
            status["high_vol_duration_sec"] = round(elapsed, 1)
            status["depth_multiplier"] = round(self.get_depth_multiplier(ticker), 2)
            status["tick_interval"] = self.get_tick_interval(ticker)

            if state.calm_since:
                calm_duration = time.monotonic() - state.calm_since
                status["calm_duration_sec"] = round(calm_duration, 1)

        return status
