#!/usr/bin/env python3
"""Seed the watchlist from config and fetch company info."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.database import Database
from src.data.models import WatchlistItem
from src.data.sources.yfinance_source import YFinanceSource
from src.utils.config import get_settings
from src.utils.logging import setup_logging, get_logger


async def main():
    """Seed watchlist with symbols from config."""
    setup_logging()
    logger = get_logger(__name__)

    settings = get_settings()
    symbols = settings.watchlist.symbols

    if not symbols:
        logger.warning("No symbols in watchlist config")
        return

    logger.info(f"Seeding watchlist with {len(symbols)} symbols")

    db = Database()
    await db.initialize()

    # Fetch company info for all symbols
    source = YFinanceSource()

    logger.info("Fetching company info from Yahoo Finance...")
    company_info = await source.fetch_multiple_company_info(symbols)

    # Insert into watchlist
    added = 0
    for symbol in symbols:
        info = company_info.get(symbol) or {}

        item = WatchlistItem(
            symbol=symbol,
            name=info.get("name"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            market_cap=info.get("market_cap"),
        )

        await db.add_to_watchlist(item)
        added += 1

        if info.get("name"):
            logger.debug(f"Added {symbol}: {info.get('name')} ({info.get('sector')})")
        else:
            logger.debug(f"Added {symbol} (no info available)")

    logger.info(f"Added {added} symbols to watchlist")

    await source.close()
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
