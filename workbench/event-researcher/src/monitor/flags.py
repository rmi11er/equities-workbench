"""Interest flag calculations for event surfacing."""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from src.data.database import Database
from src.utils.config import get_settings
from src.utils.datetime_utils import today_et
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InterestFlag:
    """An interest flag with its value."""
    name: str
    description: str
    value: Any = None


async def calculate_momentum_flag(
    db: Database,
    symbol: str,
    threshold_pct: float = 30,
    window_days: int = 90,
) -> InterestFlag | None:
    """Calculate momentum flag if stock is up/down significantly."""
    end_date = today_et()
    start_date = end_date - timedelta(days=int(window_days * 1.5))

    df = await db.get_prices([symbol], start_date, end_date, "daily")

    if len(df) < 20:
        return None

    closes = df["close"].to_list()
    first_close = float(closes[0])
    last_close = float(closes[-1])

    if first_close == 0:
        return None

    change_pct = ((last_close - first_close) / first_close) * 100

    if abs(change_pct) >= threshold_pct:
        direction = "up" if change_pct > 0 else "down"
        return InterestFlag(
            name="momentum",
            description=f"{direction} {abs(change_pct):.0f}% in {window_days}d",
            value=round(change_pct, 1),
        )

    return None


async def calculate_sector_relative_flag(
    db: Database,
    symbol: str,
    sector: str | None,
    threshold_pct: float = 10,
    window_days: int = 30,
) -> InterestFlag | None:
    """Calculate if stock is outperforming/underperforming SPY."""
    if not sector:
        return None

    end_date = today_et()
    start_date = end_date - timedelta(days=int(window_days * 1.5))

    # Get stock and SPY prices
    stock_df = await db.get_prices([symbol], start_date, end_date, "daily")
    spy_df = await db.get_prices(["SPY"], start_date, end_date, "daily")

    if len(stock_df) < 10 or len(spy_df) < 10:
        return None

    stock_closes = stock_df["close"].to_list()
    spy_closes = spy_df["close"].to_list()

    stock_return = ((float(stock_closes[-1]) - float(stock_closes[0])) / float(stock_closes[0])) * 100
    spy_return = ((float(spy_closes[-1]) - float(spy_closes[0])) / float(spy_closes[0])) * 100

    relative = stock_return - spy_return

    if abs(relative) >= threshold_pct:
        if relative > 0:
            return InterestFlag(
                name="sector_relative",
                description=f"outperforming SPY by {relative:.0f}%",
                value=round(relative, 1),
            )
        else:
            return InterestFlag(
                name="sector_relative",
                description=f"underperforming SPY by {abs(relative):.0f}%",
                value=round(relative, 1),
            )

    return None


async def calculate_vix_flag(
    db: Database,
    vix_high: float = 25,
    vix_low: float = 15,
) -> InterestFlag | None:
    """Calculate VIX regime flag."""
    end_date = today_et()
    start_date = end_date - timedelta(days=5)

    # VIX is available as ^VIX in yfinance
    df = await db.get_prices(["^VIX"], start_date, end_date, "daily")

    if len(df) == 0:
        return None

    vix = float(df["close"].to_list()[-1])

    if vix >= vix_high:
        return InterestFlag(
            name="vix_regime",
            description=f"high VIX ({vix:.1f})",
            value=vix,
        )
    elif vix <= vix_low:
        return InterestFlag(
            name="vix_regime",
            description=f"low VIX ({vix:.1f})",
            value=vix,
        )

    return None


async def calculate_streak_flag(
    db: Database,
    symbol: str,
    threshold: int = 3,
) -> InterestFlag | None:
    """Calculate consecutive beats/misses flag."""
    df = await db.get_earnings([symbol], with_actuals_only=True)

    if len(df) < threshold:
        return None

    # Check for consecutive beats or misses
    surprises = []
    for row in df.head(threshold + 2).iter_rows(named=True):
        surprise = row.get("eps_surprise_pct")
        if surprise is not None:
            surprises.append(float(surprise))

    if len(surprises) < threshold:
        return None

    # Check for streak
    recent = surprises[:threshold]

    if all(s > 0 for s in recent):
        return InterestFlag(
            name="streak",
            description=f"{threshold} consecutive beats",
            value=threshold,
        )
    elif all(s < 0 for s in recent):
        return InterestFlag(
            name="streak",
            description=f"{threshold} consecutive misses",
            value=-threshold,
        )

    return None


async def calculate_all_flags(
    db: Database,
    symbol: str,
    sector: str | None = None,
) -> list[InterestFlag]:
    """Calculate all applicable interest flags for a symbol."""
    settings = get_settings()
    flags_config = settings.filters.flags

    flags = []

    # Momentum flag
    if flags_config.momentum_enabled:
        flag = await calculate_momentum_flag(
            db,
            symbol,
            threshold_pct=flags_config.momentum_threshold_pct,
            window_days=flags_config.momentum_window_days,
        )
        if flag:
            flags.append(flag)

    # Sector relative flag
    if flags_config.sector_relative_enabled:
        flag = await calculate_sector_relative_flag(
            db,
            symbol,
            sector,
            threshold_pct=flags_config.sector_relative_threshold_pct,
            window_days=flags_config.sector_relative_window_days,
        )
        if flag:
            flags.append(flag)

    # VIX regime flag
    if flags_config.vix_enabled:
        flag = await calculate_vix_flag(
            db,
            vix_high=flags_config.vix_high,
            vix_low=flags_config.vix_low,
        )
        if flag:
            flags.append(flag)

    # Streak flag
    if flags_config.streak_enabled:
        flag = await calculate_streak_flag(
            db,
            symbol,
            threshold=flags_config.streak_threshold,
        )
        if flag:
            flags.append(flag)

    return flags
