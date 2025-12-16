#!/usr/bin/env python3
"""Refresh data - fetch latest prices and earnings calendar."""

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
from src.data.models import Earnings, Event, Price
from src.data.sources.fmp import FMPSource
from src.data.sources.yfinance_source import YFinanceSource
from src.utils.config import get_settings
from src.utils.logging import setup_logging, get_logger


async def refresh_prices(db: Database, symbols: list[str], logger) -> int:
    """Refresh latest prices for all symbols."""
    source = YFinanceSource()

    # Get latest date in DB for each symbol and fetch from there
    end_date = date.today()
    total_inserted = 0

    for symbol in symbols:
        latest = await db.get_latest_price_date(symbol, "daily")
        if latest:
            start_date = latest + timedelta(days=1)
        else:
            # If no data, fetch last 30 days
            start_date = end_date - timedelta(days=30)

        if start_date > end_date:
            logger.debug(f"{symbol}: already up to date")
            continue

        try:
            prices_data = await source.fetch_prices(symbol, start_date, end_date, "daily")

            if prices_data:
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
                total_inserted += count
                logger.debug(f"{symbol}: inserted {count} new price records")

        except Exception as e:
            logger.warning(f"Failed to refresh prices for {symbol}: {e}")

    await source.close()
    return total_inserted


async def refresh_earnings_calendar(db: Database, symbols: set[str], logger) -> int:
    """Refresh earnings calendar from FMP, with yfinance fallback."""
    settings = get_settings()
    inserted = 0

    # Try FMP first if API key is available
    if settings.fmp_api_key:
        source = FMPSource()
        start_date = date.today()
        end_date = start_date + timedelta(days=30)

        try:
            earnings_data = await source.fetch_earnings_calendar(start_date, end_date)
            filtered = [e for e in earnings_data if e["symbol"] in symbols]
            logger.info(f"FMP: Found {len(filtered)} earnings events for watchlist symbols")

            for record in filtered:
                event = Event(
                    event_id=record["event_id"],
                    symbol=record["symbol"],
                    event_type="earnings",
                    event_date=record["event_date"],
                    event_time=record.get("event_time"),
                    title=f"{record['symbol']} Earnings",
                    description=record.get("fiscal_quarter"),
                    metadata={},
                )
                await db.insert_event(event)

                earnings = Earnings(
                    event_id=record["event_id"],
                    symbol=record["symbol"],
                    fiscal_quarter=record.get("fiscal_quarter"),
                    fiscal_year=record.get("fiscal_year"),
                    eps_estimate=Decimal(str(record["eps_estimate"])) if record.get("eps_estimate") else None,
                    eps_actual=Decimal(str(record["eps_actual"])) if record.get("eps_actual") else None,
                    revenue_estimate=record.get("revenue_estimate"),
                    revenue_actual=record.get("revenue_actual"),
                )
                await db.insert_earnings(earnings)
                inserted += 1

            await source.close()
            if inserted > 0:
                return inserted
        except Exception as e:
            logger.warning(f"FMP failed: {e}, falling back to yfinance")
            await source.close()

    # Fallback to yfinance
    logger.info("Using yfinance for earnings dates...")
    yf_source = YFinanceSource()

    for symbol in symbols:
        try:
            dates = await yf_source.fetch_earnings_dates(symbol)
            for record in dates:
                event_date = record["event_date"]
                # Only include dates in next 60 days
                if event_date > date.today() + timedelta(days=60):
                    continue

                event_id = f"{symbol}-earnings-{event_date.isoformat()}"

                event = Event(
                    event_id=event_id,
                    symbol=symbol,
                    event_type="earnings",
                    event_date=event_date,
                    event_time=record.get("event_time"),
                    title=f"{symbol} Earnings",
                    metadata={},
                )
                await db.insert_event(event)

                earnings = Earnings(
                    event_id=event_id,
                    symbol=symbol,
                )
                await db.insert_earnings(earnings)
                inserted += 1
                logger.debug(f"Added earnings event for {symbol} on {event_date}")
        except Exception as e:
            logger.warning(f"Failed to get earnings dates for {symbol}: {e}")

    await yf_source.close()
    return inserted


async def main():
    """Refresh all data."""
    parser = argparse.ArgumentParser(description="Refresh data")
    parser.add_argument("--prices-only", action="store_true", help="Only refresh prices")
    parser.add_argument(
        "--earnings-only", action="store_true", help="Only refresh earnings calendar"
    )
    args = parser.parse_args()

    setup_logging()
    logger = get_logger(__name__)

    settings = get_settings()
    symbols = settings.watchlist.symbols

    if not symbols:
        logger.error("No symbols in watchlist")
        return

    logger.info(f"Refreshing data for {len(symbols)} symbols")

    db = Database()
    await db.initialize()

    if not args.earnings_only:
        logger.info("Refreshing prices...")
        price_count = await refresh_prices(db, symbols, logger)
        logger.info(f"Inserted {price_count} price records")

    if not args.prices_only:
        logger.info("Refreshing earnings calendar...")
        earnings_count = await refresh_earnings_calendar(db, set(symbols), logger)
        logger.info(f"Inserted {earnings_count} earnings events")

    await db.close()
    logger.info("Data refresh complete")


if __name__ == "__main__":
    asyncio.run(main())
