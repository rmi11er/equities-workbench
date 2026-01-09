#!/usr/bin/env python3
"""
Kalshi Market Maker - Entry Point

Usage:
    python main.py --config config.toml
    python main.py --env demo
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config, Config
from src.constants import Environment
from src.market_maker import MarketMaker
from src.multi_market import MultiMarketOrchestrator
from src.run_context import create_run_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kalshi Market Maker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to configuration file (default: config.toml)",
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["demo", "production"],
        default=None,
        help="Override environment (demo or production)",
    )

    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Override market ticker",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without placing orders (logging only)",
    )

    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading config from {config_path}")
        config = load_config(config_path)
    else:
        print(f"Config file not found: {config_path}")
        print("Using default configuration")
        config = Config()

    # Apply command line overrides
    if args.env:
        config.environment = Environment[args.env.upper()]
        print(f"Environment override: {config.environment.name}")

    if args.ticker:
        config.market_ticker = args.ticker
        print(f"Ticker override: {config.market_ticker}")

    # Validate configuration
    tickers = config.tickers
    if not tickers:
        print("ERROR: No tickers configured")
        print("Set market_ticker or market_tickers in config.toml, or use --ticker flag")
        return 1

    if config.credentials.api_key_id == "TODO":
        print("ERROR: API credentials not configured")
        print("Set api_key_id and private_key_path in config.toml")
        return 1

    # Determine mode (single vs multi-market)
    is_multi_market = config.is_multi_market
    ticker_display = ", ".join(tickers) if is_multi_market else tickers[0]

    # Create run context for versioned logging
    run_context = create_run_context(
        base_log_dir=config.logging.base_log_dir,
        ticker=ticker_display,
        environment=config.environment.name,
        config=config,
    )

    # Print startup banner with version info
    print(run_context.get_startup_banner())

    # Create market maker (single or multi-market)
    if is_multi_market:
        print(f"\nMulti-market mode: {len(tickers)} markets")
        mm = MultiMarketOrchestrator(config, run_context=run_context)
    else:
        mm = MarketMaker(config, run_context=run_context)

    # Setup signal handlers
    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()

    def signal_handler():
        print("\nShutdown signal received...")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Run market maker
    print(f"\nStarting market maker:")
    print(f"  Environment: {config.environment.name}")
    if is_multi_market:
        print(f"  Markets ({len(tickers)}):")
        for t in tickers:
            print(f"    - {t}")
    else:
        print(f"  Ticker: {tickers[0]}")
    print(f"  Risk Aversion: {config.strategy.risk_aversion}")
    print(f"  Max Inventory: {config.strategy.max_inventory}")
    print()

    try:
        # Create tasks
        mm_task = asyncio.create_task(mm.run())
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        # Wait for either completion or shutdown
        done, pending = await asyncio.wait(
            [mm_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Check for errors
        for task in done:
            if task.exception():
                raise task.exception()

    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print("Market maker shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
