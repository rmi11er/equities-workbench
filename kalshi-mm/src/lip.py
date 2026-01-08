"""Liquidity Incentive Program (LIP) compliance and optimization."""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class LIPProgram:
    """Active LIP program specifications."""
    market_ticker: str
    period_reward: int          # Total reward pool (cents)
    target_size: int            # Min contracts to qualify
    discount_factor_bps: int    # Discount per tick away from best (basis points)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    @property
    def discount_factor(self) -> float:
        """Discount factor as decimal (5000 bps = 0.50)."""
        return self.discount_factor_bps / 10000.0

    def distance_multiplier(self, ticks_from_best: int) -> float:
        """
        Calculate score multiplier based on distance from best price.

        At best price (0 ticks): 1.0
        Each tick away: multiplied by (1 - discount_factor)

        Example with 50% discount (5000 bps):
          0 ticks: 1.0
          1 tick:  0.5
          2 ticks: 0.25
          3 ticks: 0.125
        """
        if ticks_from_best <= 0:
            return 1.0
        return (1.0 - self.discount_factor) ** ticks_from_best


@dataclass
class LIPConstraints:
    """Constraints derived from LIP program for quoting."""
    min_size: int               # Must quote at least this many contracts
    max_distance: int           # Max ticks from best before score becomes negligible
    is_active: bool = True

    @classmethod
    def from_program(cls, program: LIPProgram, min_score_threshold: float = 0.1) -> "LIPConstraints":
        """
        Derive quoting constraints from LIP program.

        Args:
            program: The LIP program
            min_score_threshold: Minimum multiplier to consider worthwhile (default 10%)
        """
        # Min size = target size (must meet this to qualify)
        min_size = program.target_size

        # Max distance: how many ticks until multiplier < threshold
        # (1 - df)^n < threshold
        # n > log(threshold) / log(1 - df)
        if program.discount_factor >= 1.0:
            max_distance = 0  # Must be at best price
        elif program.discount_factor <= 0:
            max_distance = 99  # No penalty, can be anywhere
        else:
            max_distance = int(
                math.log(min_score_threshold) / math.log(1 - program.discount_factor)
            )

        return cls(
            min_size=min_size,
            max_distance=max_distance,
            is_active=True,
        )


@dataclass
class LIPTracker:
    """Tracks our LIP performance metrics."""
    snapshots_total: int = 0
    snapshots_qualifying: int = 0  # Snapshots where we met target size
    estimated_score: float = 0.0
    last_snapshot: float = 0.0

    @property
    def uptime_pct(self) -> float:
        """Percentage of snapshots where we qualified."""
        if self.snapshots_total == 0:
            return 0.0
        return self.snapshots_qualifying / self.snapshots_total

    def record_snapshot(
        self,
        our_bid_size: int,
        our_ask_size: int,
        our_bid_distance: int,  # Ticks from best bid
        our_ask_distance: int,  # Ticks from best ask
        program: LIPProgram,
    ) -> float:
        """
        Record a snapshot and calculate our score contribution.

        Returns:
            Score contribution for this snapshot
        """
        self.snapshots_total += 1
        self.last_snapshot = time.time()

        # Check if we qualify (meet target size on at least one side)
        bid_qualifies = our_bid_size >= program.target_size
        ask_qualifies = our_ask_size >= program.target_size

        if not (bid_qualifies or ask_qualifies):
            return 0.0

        self.snapshots_qualifying += 1

        # Calculate score
        score = 0.0

        if bid_qualifies:
            bid_mult = program.distance_multiplier(our_bid_distance)
            score += our_bid_size * bid_mult

        if ask_qualifies:
            ask_mult = program.distance_multiplier(our_ask_distance)
            score += our_ask_size * ask_mult

        self.estimated_score += score
        return score


class LIPManager:
    """
    Manages LIP program data and compliance.

    Responsibilities:
    - Cache active LIP programs
    - Provide constraints for quoting
    - Track compliance metrics
    """

    def __init__(self, connector):
        self.connector = connector
        self._programs: Dict[str, LIPProgram] = {}
        self._trackers: Dict[str, LIPTracker] = {}
        self._last_refresh: float = 0
        self._refresh_interval: float = 3600 * 4  # Refresh every 4 hours

    async def refresh_programs(self, force: bool = False) -> None:
        """Fetch active LIP programs from API."""
        now = time.time()

        if not force and (now - self._last_refresh) < self._refresh_interval:
            return

        logger.info("Refreshing LIP programs from API...")

        try:
            resp = await self.connector._request(
                "GET",
                "/incentive_programs?status=active&type=liquidity&limit=200"
            )

            programs = resp.get("incentive_programs", [])
            self._programs.clear()

            for p in programs:
                ticker = p.get("market_ticker")
                if not ticker:
                    continue

                program = LIPProgram(
                    market_ticker=ticker,
                    period_reward=p.get("period_reward", 0),
                    target_size=p.get("target_size", 100),
                    discount_factor_bps=p.get("discount_factor_bps", 5000),
                )

                self._programs[ticker] = program

            self._last_refresh = now
            logger.info(f"Loaded {len(self._programs)} active LIP programs")

        except Exception as e:
            logger.warning(f"Failed to refresh LIP programs: {e}")

    def get_program(self, ticker: str) -> Optional[LIPProgram]:
        """Get LIP program for a ticker, if active."""
        return self._programs.get(ticker)

    def get_constraints(self, ticker: str) -> Optional[LIPConstraints]:
        """Get quoting constraints for LIP compliance."""
        program = self.get_program(ticker)
        if program is None:
            return None
        return LIPConstraints.from_program(program)

    def get_tracker(self, ticker: str) -> LIPTracker:
        """Get or create tracker for a ticker."""
        if ticker not in self._trackers:
            self._trackers[ticker] = LIPTracker()
        return self._trackers[ticker]

    def record_snapshot(
        self,
        ticker: str,
        our_bid_size: int,
        our_ask_size: int,
        our_bid_price: int,
        our_ask_price: int,
        best_bid: Optional[int],
        best_ask: Optional[int],
    ) -> Optional[float]:
        """
        Record a snapshot for LIP tracking.

        Returns:
            Score contribution, or None if no LIP program
        """
        program = self.get_program(ticker)
        if program is None:
            return None

        tracker = self.get_tracker(ticker)

        # Calculate distance from best
        if best_bid is not None and our_bid_price is not None:
            bid_distance = best_bid - our_bid_price  # Positive if we're below best
        else:
            bid_distance = 99  # No best bid, assume far

        if best_ask is not None and our_ask_price is not None:
            ask_distance = our_ask_price - best_ask  # Positive if we're above best
        else:
            ask_distance = 99

        return tracker.record_snapshot(
            our_bid_size=our_bid_size,
            our_ask_size=our_ask_size,
            our_bid_distance=max(0, bid_distance),
            our_ask_distance=max(0, ask_distance),
            program=program,
        )

    def get_status(self, ticker: str) -> dict:
        """Get LIP status for a ticker."""
        program = self.get_program(ticker)
        tracker = self.get_tracker(ticker)

        if program is None:
            return {"active": False}

        return {
            "active": True,
            "target_size": program.target_size,
            "discount_factor": program.discount_factor,
            "period_reward": program.period_reward,
            "uptime_pct": tracker.uptime_pct,
            "snapshots_total": tracker.snapshots_total,
            "snapshots_qualifying": tracker.snapshots_qualifying,
            "estimated_score": tracker.estimated_score,
        }
