"""Volatility regime detection and response coordination."""

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def extract_market_root(ticker: str) -> str:
    """
    Extract market root from ticker for aggregating correlated markets.

    Examples:
        "KXNFLMVP-26-DMAY" -> "KXNFLMVP-26-"
        "KXNCAAFGAME-26JAN09OREIND-IND" -> "KXNCAAFGAME-26JAN09OREIND-"
        "NFL-KC-WIN" -> "NFL-KC-"

    For binary outcome markets (two-team games), fills on either team
    should count together since they represent the same directional bet.
    """
    # Find the last hyphen followed by letters/numbers (the outcome suffix)
    # and return everything before it (including the hyphen)
    match = re.match(r"^(.+-)[A-Z0-9]+$", ticker)
    if match:
        return match.group(1)
    # Fallback: return ticker as-is (no aggregation)
    return ticker + "-"


@dataclass
class VolatilityRegimeConfig:
    """Configuration for volatility regime detection."""
    # Spread-based trigger
    spread_threshold: int = 10  # Enter high vol if spread >= this (cents)

    # Fill-based trigger
    fills_threshold: int = 3      # Enter high vol if N fills in window
    fills_window_sec: float = 30.0  # Window for counting fills

    # Canary death trigger (v2.2)
    canary_timeout_ms: int = 50   # Fill faster than this = toxic sweep

    # Depth multiplier (require more liquidity in high vol)
    initial_depth_multiplier: float = 1.5  # Start at 1.5x min_depth when entering
    depth_decay_halflife_sec: float = 60.0  # Multiplier decays back to 1.0

    # Tick rate adjustment
    normal_tick_interval: float = 0.1    # 100ms normally
    high_vol_tick_interval: float = 0.03  # 30ms in high vol (faster updates)

    # Exit conditions
    calm_spread_threshold: int = 5  # Exit high vol when spread <= this
    calm_duration_sec: float = 15.0  # Must be calm for this long to exit

    # v2.2 high-vol timeout (auto-exit if no new triggers)
    high_vol_timeout_sec: float = 30.0  # Exit high vol after this duration


class HighVolReason:
    """Reasons for entering high-vol mode."""
    SPREAD = "spread"
    FILLS = "fills"
    CANARY_DEATH = "canary_death"


@dataclass
class MarketRegimeState:
    """Per-market volatility regime state."""
    ticker: str

    # Current state
    is_high_vol: bool = False
    entered_at: Optional[float] = None  # monotonic timestamp
    entry_reason: Optional[str] = None  # Why we entered high-vol
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
        # Track fills at market_root level for correlated markets
        # e.g., fills on GAME-TEAMA and GAME-TEAMB both count toward "GAME-" root
        self._root_fills: dict[str, list[float]] = {}

    def get_state(self, ticker: str) -> MarketRegimeState:
        """Get or create state for a ticker."""
        if ticker not in self._states:
            self._states[ticker] = MarketRegimeState(ticker=ticker)
        return self._states[ticker]

    def _record_root_fill(self, ticker: str, timestamp: float) -> None:
        """Record a fill at the market_root level for correlated market aggregation."""
        root = extract_market_root(ticker)
        if root not in self._root_fills:
            self._root_fills[root] = []
        self._root_fills[root].append(timestamp)

    def _count_root_fills(self, ticker: str, window_sec: float) -> int:
        """
        Count fills across all correlated markets (same market_root) within window.

        This aggregates fills from e.g. GAME-TEAMA and GAME-TEAMB together,
        since getting hit on either represents the same directional flow.
        """
        root = extract_market_root(ticker)
        if root not in self._root_fills:
            return 0

        now = time.monotonic()
        cutoff = now - window_sec
        # Clean up old fills while counting
        self._root_fills[root] = [t for t in self._root_fills[root] if t > cutoff]
        return len(self._root_fills[root])

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

        # Record fill if we had one - both at ticker level and root level
        if had_fill:
            state.record_fill(now)
            self._record_root_fill(ticker, now)

        state.last_spread = spread

        # Check entry conditions
        if not state.is_high_vol:
            should_enter = False
            reason = ""

            # Spread trigger
            if spread >= self.config.spread_threshold:
                should_enter = True
                reason = f"spread={spread}c >= {self.config.spread_threshold}c"

            # Fill frequency trigger - use ROOT level to catch correlated markets
            # e.g., 2 fills on TEAM-A + 2 fills on TEAM-B = 4 fills on root "TEAM-"
            root_fills = self._count_root_fills(ticker, self.config.fills_window_sec)
            if root_fills >= self.config.fills_threshold:
                should_enter = True
                root = extract_market_root(ticker)
                reason = f"{root_fills} fills on {root}* in {self.config.fills_window_sec}s"

            if should_enter:
                state.is_high_vol = True
                state.entered_at = now
                state.calm_since = None
                state.entry_reason = HighVolReason.SPREAD if "spread" in reason else HighVolReason.FILLS
                logger.warning(f"[{ticker}] ENTERING HIGH VOL: {reason}")

        # Check exit conditions
        else:
            should_exit = False
            exit_reason = ""

            # Timeout exit (v2.2) - auto-exit after high_vol_timeout_sec
            if state.entered_at is not None:
                elapsed = now - state.entered_at
                if elapsed >= self.config.high_vol_timeout_sec:
                    should_exit = True
                    exit_reason = f"timeout after {self.config.high_vol_timeout_sec}s"

            # Calm period exit - spread must stay below threshold
            if spread <= self.config.calm_spread_threshold:
                if state.calm_since is None:
                    state.calm_since = now
                elif now - state.calm_since >= self.config.calm_duration_sec:
                    should_exit = True
                    exit_reason = f"calm for {self.config.calm_duration_sec}s"
            else:
                # Spread spiked again, reset calm timer
                state.calm_since = None

            if should_exit:
                state.is_high_vol = False
                state.entered_at = None
                state.calm_since = None
                state.entry_reason = None
                logger.info(f"[{ticker}] EXITING HIGH VOL: {exit_reason}")

        return state.is_high_vol != old_state

    def activate_high_vol(
        self,
        ticker: str,
        reason: str = HighVolReason.CANARY_DEATH,
    ) -> bool:
        """
        Force-activate high-vol mode (v2.2 canary death trigger).

        Called when a canary order fills too quickly, indicating toxic flow.
        This bypasses normal spread/fill triggers.

        Args:
            ticker: Market ticker
            reason: Reason for activation (default: CANARY_DEATH)

        Returns:
            True if state changed (was not already in high-vol)
        """
        state = self.get_state(ticker)

        if state.is_high_vol:
            # Already in high-vol - just refresh the timer
            state.entered_at = time.monotonic()
            state.entry_reason = reason
            logger.warning(f"[{ticker}] HIGH VOL REFRESHED: {reason}")
            return False

        state.is_high_vol = True
        state.entered_at = time.monotonic()
        state.calm_since = None
        state.entry_reason = reason
        logger.critical(f"[{ticker}] ENTERING HIGH VOL: {reason}")

        return True

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

        root = extract_market_root(ticker)
        status = {
            "ticker": ticker,
            "market_root": root,
            "is_high_vol": state.is_high_vol,
            "last_spread": state.last_spread,
            "recent_fills_ticker": state.count_recent_fills(self.config.fills_window_sec),
            "recent_fills_root": self._count_root_fills(ticker, self.config.fills_window_sec),
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
