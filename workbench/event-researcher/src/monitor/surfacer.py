"""Event surfacing logic - identifies and ranks upcoming events."""

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any

from src.data.database import Database
from src.monitor.flags import InterestFlag, calculate_all_flags
from src.utils.config import get_settings
from src.utils.datetime_utils import today_et
from src.utils.logging import get_logger

logger = get_logger(__name__)


class InterestTier(Enum):
    """Interest tier for surfaced events."""
    HIGH = "high"
    STANDARD = "standard"
    LOW = "low"


@dataclass
class SurfacedEvent:
    """An event surfaced for user attention."""
    event_id: str
    symbol: str
    event_type: str
    event_date: date
    event_time: str | None
    title: str | None
    description: str | None

    # Enrichment
    company_name: str | None = None
    sector: str | None = None
    market_cap: int | None = None

    # Scoring
    flags: list[InterestFlag] = field(default_factory=list)
    interest_score: int = 0
    tier: InterestTier = InterestTier.LOW

    # Additional context
    days_until: int = 0
    historical_avg_move: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "symbol": self.symbol,
            "event_type": self.event_type,
            "event_date": str(self.event_date),
            "event_time": self.event_time,
            "title": self.title,
            "company_name": self.company_name,
            "sector": self.sector,
            "tier": self.tier.value,
            "interest_score": self.interest_score,
            "flags": [{"name": f.name, "description": f.description} for f in self.flags],
            "days_until": self.days_until,
            "historical_avg_move": self.historical_avg_move,
        }


class EventSurfacer:
    """Surfaces and ranks upcoming events based on interest criteria."""

    def __init__(self, db: Database):
        self.db = db
        self.settings = get_settings()

    async def get_surfaced_events(
        self,
        lookahead_days: int | None = None,
        event_types: list[str] | None = None,
        symbols: list[str] | None = None,
        min_tier: InterestTier = InterestTier.LOW,
    ) -> list[SurfacedEvent]:
        """Get surfaced events for the upcoming period.

        Args:
            lookahead_days: Days ahead to look (default from config)
            event_types: Filter by event types
            symbols: Filter by symbols (default: watchlist)
            min_tier: Minimum interest tier to include

        Returns:
            List of surfaced events, sorted by date then interest score
        """
        if lookahead_days is None:
            lookahead_days = self.settings.filters.temporal.lookahead_days

        today = today_et()
        end_date = today + timedelta(days=lookahead_days)

        # Get watchlist for filtering and enrichment
        watchlist_df = await self.db.get_watchlist()
        watchlist_info = {}
        watchlist_symbols = set()

        for row in watchlist_df.iter_rows(named=True):
            sym = row["symbol"]
            watchlist_symbols.add(sym)
            watchlist_info[sym] = {
                "name": row["name"],
                "sector": row["sector"],
                "market_cap": row["market_cap"],
            }

        # Filter to watchlist if no specific symbols requested
        if symbols is None:
            symbols = list(watchlist_symbols)

        # Get events
        events_df = await self.db.get_events(
            symbols=symbols,
            event_types=event_types,
            start_date=today,
            end_date=end_date,
        )

        surfaced = []

        for row in events_df.iter_rows(named=True):
            symbol = row["symbol"]
            if not symbol:
                continue

            # Skip if not in watchlist
            if symbol not in watchlist_symbols:
                continue

            info = watchlist_info.get(symbol, {})

            event = SurfacedEvent(
                event_id=row["event_id"],
                symbol=symbol,
                event_type=row["event_type"],
                event_date=row["event_date"],
                event_time=row["event_time"],
                title=row["title"],
                description=row["description"],
                company_name=info.get("name"),
                sector=info.get("sector"),
                market_cap=info.get("market_cap"),
                days_until=(row["event_date"] - today).days,
            )

            # Calculate interest flags
            event.flags = await calculate_all_flags(
                self.db,
                symbol,
                sector=info.get("sector"),
            )

            # Calculate interest score
            event.interest_score = len(event.flags)

            # Add bonus for event type priority
            type_weights = self.settings.filters.event_types
            type_weight = getattr(type_weights, row["event_type"], 50)
            event.interest_score += type_weight // 50  # Normalize to ~1-2 points

            # Determine tier
            thresholds = self.settings.filters.thresholds
            if event.interest_score >= thresholds.high_interest_min_flags:
                event.tier = InterestTier.HIGH
            elif event.interest_score >= thresholds.standard_min_flags:
                event.tier = InterestTier.STANDARD
            else:
                event.tier = InterestTier.LOW

            # Calculate historical average move for earnings
            if row["event_type"] == "earnings":
                event.historical_avg_move = await self._get_historical_avg_move(symbol)

            # Filter by minimum tier
            tier_order = {InterestTier.HIGH: 3, InterestTier.STANDARD: 2, InterestTier.LOW: 1}
            if tier_order[event.tier] >= tier_order[min_tier]:
                surfaced.append(event)

        # Sort by date, then by interest score (descending)
        surfaced.sort(key=lambda e: (e.event_date, -e.interest_score))

        return surfaced

    async def _get_historical_avg_move(self, symbol: str) -> float | None:
        """Get historical average earnings move for a symbol."""
        # This is a simplified version - could be expanded
        df = await self.db.get_earnings([symbol], with_actuals_only=True)

        if len(df) < 4:
            return None

        # We'd need price data around earnings to calculate moves
        # For now, return None - this would need compute_event_response logic
        return None

    async def get_events_by_tier(
        self,
        lookahead_days: int | None = None,
    ) -> dict[InterestTier, list[SurfacedEvent]]:
        """Get events grouped by interest tier."""
        events = await self.get_surfaced_events(lookahead_days=lookahead_days)

        grouped = {
            InterestTier.HIGH: [],
            InterestTier.STANDARD: [],
            InterestTier.LOW: [],
        }

        for event in events:
            grouped[event.tier].append(event)

        return grouped

    async def get_event_summary(
        self,
        lookahead_days: int | None = None,
    ) -> dict[str, Any]:
        """Get a summary of upcoming events."""
        grouped = await self.get_events_by_tier(lookahead_days)

        return {
            "total": sum(len(events) for events in grouped.values()),
            "high_interest": len(grouped[InterestTier.HIGH]),
            "standard": len(grouped[InterestTier.STANDARD]),
            "low": len(grouped[InterestTier.LOW]),
            "events_by_tier": {
                tier.value: [e.to_dict() for e in events]
                for tier, events in grouped.items()
            },
        }
