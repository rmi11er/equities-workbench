#!/usr/bin/env python3
"""Backfill historical price data for watchlist symbols."""

import argparse
import asyncio
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.database import Database
from src.data.models import Price
from src.data.sources.yfinance_source import YFinanceSource
from src.utils.config import get_settings
from src.utils.logging import setup_logging, get_logger


async def backfill_symbol(
    db: Database,
    source: YFinanceSource,
    symbol: str,
    start_date: date,
    end_date: date,
    logger,
) -> int:
    """Backfill prices for a single symbol."""
    try:
        prices_data = await source.fetch_prices(symbol, start_date, end_date, "daily")

        if not prices_data:
            logger.warning(f"No price data returned for {symbol}")
            return 0

        prices = [
            Price(
                symbol=symbol,
                timestamp=p["timestamp"],
                timeframe="daily",
                open=Decimal(str(p["open"])),
                high=Decimal(str(p["high"])),
                low=Decimal(str(p["low"])),
                close=Decimal(str(p["close"])),
                volume=p["volume"],
                source="yfinance",
            )
            for p in prices_data
        ]

        count = await db.insert_prices(prices)
        logger.info(f"{symbol}: inserted {count} price records")
        return count

    except Exception as e:
        logger.error(f"Failed to backfill {symbol}: {e}")
        return 0


async def main():
    """Backfill historical prices for watchlist symbols."""
    parser = argparse.ArgumentParser(description="Backfill historical price data")
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Years of history to fetch (default: from settings)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="*",
        help="Specific symbols to backfill (default: all watchlist)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent fetches (default: 5)",
    )
    args = parser.parse_args()

    setup_logging()
    logger = get_logger(__name__)

    settings = get_settings()

    # Determine symbols to backfill
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = settings.watchlist.symbols

    if not symbols:
        logger.error("No symbols to backfill. Check watchlist config.")
        return

    # Determine date range
    years = args.years or settings.data.price_history_years
    end_date = date.today()
    start_date = end_date - timedelta(days=years * 365)

    logger.info(
        f"Backfilling {len(symbols)} symbols from {start_date} to {end_date} "
        f"({years} years)"
    )

    db = Database()
    await db.initialize()

    source = YFinanceSource()

    # Process symbols with limited concurrency
    semaphore = asyncio.Semaphore(args.concurrency)

    async def process_with_semaphore(symbol: str) -> int:
        async with semaphore:
            return await backfill_symbol(db, source, symbol, start_date, end_date, logger)

    tasks = [process_with_semaphore(s) for s in symbols]
    results = await asyncio.gather(*tasks)

    total = sum(results)
    successful = sum(1 for r in results if r > 0)

    logger.info(
        f"Backfill complete: {total} total records for {successful}/{len(symbols)} symbols"
    )

    await source.close()
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
