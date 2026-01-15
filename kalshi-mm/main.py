#!/usr/bin/env python3
"""
Kalshi Market Maker - Entry Point

Usage:
    python main.py --config config.toml      # Legacy v1/v2
    python main.py --config config_v3.toml   # Passive MM v3
    python main.py --config config_rfq.toml  # RFQ Responder
    python main.py --env demo
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.constants import Environment


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


def is_v3_config(config_path: Path) -> bool:
    """Check if config file is v3 format (has [quoting] section)."""
    if not config_path.exists():
        return False
    content = config_path.read_text()
    return "[quoting]" in content


def is_rfq_config(config_path: Path) -> bool:
    """Check if config file is RFQ format (has [filters] or [pricing] with RFQ settings)."""
    if not config_path.exists():
        return False
    content = config_path.read_text()
    # RFQ configs have [filters] or [pricing] with default_spread_pct
    # but not [quoting] (which is v3) or [strategy] (which is legacy)
    has_filters = "[filters]" in content
    has_rfq_pricing = "[pricing]" in content and "default_spread_pct" in content
    has_quoting = "[quoting]" in content
    has_strategy = "[strategy]" in content
    return (has_filters or has_rfq_pricing) and not has_quoting and not has_strategy


async def run_rfq(args) -> int:
    """Run RFQ Responder."""
    from src.rfq import load_rfq_config, RFQResponder

    config_path = Path(args.config)
    print(f"Loading RFQ config from {config_path}")
    config = load_rfq_config(str(config_path))

    # Apply environment override
    if args.env:
        config.environment = Environment[args.env.upper()]
        print(f"Environment override: {config.environment.name}")

    # Validate
    if not config.credentials.api_key_id:
        print("ERROR: API credentials not configured")
        return 1

    # Print startup info
    print(f"\n{'='*60}")
    print("  RFQ RESPONDER")
    print(f"{'='*60}")
    print(f"  Environment: {config.environment.name}")
    print(f"  Leg Tickers: {len(config.leg_tickers)}")
    print(f"  Spread: {config.pricing.default_spread_pct*100:.1f}%")
    print(f"  Max Exposure: ${config.risk.max_exposure_dollars:.0f}")
    print(f"  Max Active Quotes: {config.risk.max_active_quotes}")
    print(f"  Filters:")
    print(f"    - Dollar range: ${config.filters.min_dollars:.0f} - ${config.filters.max_dollars:.0f}")
    print(f"    - Leg range: {config.filters.min_legs} - {config.filters.max_legs}")
    print(f"{'='*60}\n")

    # Create responder
    responder = RFQResponder(config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()

    def signal_handler():
        print("\nShutdown signal received...")
        shutdown_event.set()
        asyncio.create_task(responder.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        responder_task = asyncio.create_task(responder.start())
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        done, pending = await asyncio.wait(
            [responder_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        for task in done:
            if task.exception():
                raise task.exception()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


async def run_v3(args) -> int:
    """Run Passive Market Maker v3."""
    from src.v3 import load_config_v3, PassiveQuoter

    config_path = Path(args.config)
    print(f"Loading v3 config from {config_path}")
    config = load_config_v3(str(config_path))

    # Apply environment override
    if args.env:
        config.environment = Environment[args.env.upper()]
        print(f"Environment override: {config.environment.name}")

    # Validate
    if not config.tickers:
        print("ERROR: No market_tickers configured")
        return 1

    if not config.credentials.api_key_id:
        print("ERROR: API credentials not configured")
        return 1

    # Print startup info
    print(f"\n{'='*60}")
    print("  PASSIVE MARKET MAKER v3")
    print(f"{'='*60}")
    print(f"  Environment: {config.environment.name}")
    print(f"  Markets: {len(config.tickers)}")
    print(f"  Min Market Width: {config.quoting.min_market_width}c")
    print(f"  Min Join Depth: ${config.quoting.min_join_depth_dollars:.0f}")
    print(f"  Inventory Gamma: {config.inventory.gamma}")
    print(f"{'='*60}\n")

    # Create quoter
    quoter = PassiveQuoter(config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()

    def signal_handler():
        print("\nShutdown signal received...")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        quoter_task = asyncio.create_task(quoter.start())
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        done, pending = await asyncio.wait(
            [quoter_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        for task in done:
            if task.exception():
                raise task.exception()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


async def run_legacy(args) -> int:
    """Run legacy v1/v2 market maker."""
    from src.config import load_config, Config
    from src.market_maker import MarketMaker
    from src.multi_market import MultiMarketOrchestrator
    from src.run_context import create_run_context

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading legacy config from {config_path}")
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
    print(f"  Max Inventory: ${config.strategy.max_inventory_dollars:.0f}")
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


async def main() -> int:
    args = parse_args()
    config_path = Path(args.config)

    # Detect config format: RFQ > v3 > legacy
    if is_rfq_config(config_path):
        print("Detected RFQ config format")
        return await run_rfq(args)
    elif is_v3_config(config_path):
        print("Detected v3 config format")
        return await run_v3(args)
    else:
        print("Detected legacy config format")
        return await run_legacy(args)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
