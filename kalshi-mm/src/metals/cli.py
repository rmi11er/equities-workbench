"""Command-line entry point for read-only metals research operations."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
from typing import Optional

from .config import load_metals_config, resolve_kalshi_credentials
from .cme import DatabentoMetalsRecorder
from .discovery import discover_open_markets
from .export import build_duckdb_catalog, export_normalized_parquet
from .kalshi import KalshiReadClient
from .recorder import MetalsRecorder, backfill_settled_markets
from .pyth import backfill_pyth_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Kalshi metals research")
    parser.add_argument(
        "--config", default="configs/metals.toml", help="Metals TOML configuration"
    )
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("discover", help="List current target markets")
    subparsers.add_parser(
        "check-access", help="Check authenticated read tier and endpoint costs"
    )

    collect = subparsers.add_parser("collect", help="Collect live read-only data")
    collect.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Stop after this many seconds; zero means run until signaled",
    )

    cme = subparsers.add_parser(
        "collect-databento",
        help="Collect read-only CME GLBX.MDP3 data through Databento",
    )
    cme.add_argument("--metal", default="silver")
    cme.add_argument(
        "--duration-seconds",
        type=float,
        default=60.0,
        help="Stop after this many seconds; zero runs until signaled",
    )
    cme.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional deterministic record limit; zero means unlimited",
    )

    backfill = subparsers.add_parser(
        "backfill-kalshi", help="Archive settled market results and trades"
    )
    backfill.add_argument("--metal", default="silver")
    backfill.add_argument("--max-markets", type=int, default=0)
    backfill.add_argument("--skip-trades", action="store_true")
    subparsers.add_parser(
        "export-parquet", help="Export normalized segments and refresh DuckDB views"
    )
    pyth = subparsers.add_parser(
        "backfill-pyth-history", help="Archive exact Pyth Pro one-minute candles"
    )
    pyth.add_argument("--metal", default="silver")
    pyth.add_argument("--from-seconds", type=int, required=True)
    pyth.add_argument("--to-seconds", type=int, required=True)
    return parser


async def _discover(config_path: str) -> int:
    config = load_metals_config(config_path)
    try:
        credentials = resolve_kalshi_credentials(config.kalshi)
    except (ValueError, FileNotFoundError):
        credentials = None
    async with KalshiReadClient(config.kalshi, credentials) as client:
        markets, _ = await discover_open_markets(client, config)
    for item in markets:
        market = item.raw
        print(
            json.dumps(
                {
                    "metal": item.metal,
                    "ticker": item.ticker,
                    "event_ticker": market.get("event_ticker"),
                    "series_ticker": market.get("series_ticker"),
                    "status": market.get("status"),
                    "open_time": market.get("open_time"),
                    "close_time": market.get("close_time"),
                    "floor_strike": market.get("floor_strike"),
                    "strike_type": market.get("strike_type"),
                    "price_level_structure": market.get("price_level_structure"),
                    "price_ranges": market.get("price_ranges"),
                },
                sort_keys=True,
                default=str,
            )
        )
    print("discovered={}".format(len(markets)))
    return 0


async def _check_access(config_path: str) -> int:
    config = load_metals_config(config_path)
    credentials = resolve_kalshi_credentials(config.kalshi)
    async with KalshiReadClient(config.kalshi, credentials) as client:
        limits = await client.get("/account/limits")
        costs = await client.get("/account/endpoint_costs")
    print(
        json.dumps(
            {"limits": limits.payload, "endpoint_costs": costs.payload},
            sort_keys=True,
            default=str,
        )
    )
    return 0


async def _collect(config_path: str, duration_seconds: float) -> int:
    config = load_metals_config(config_path)
    recorder = MetalsRecorder(config)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, recorder.stop)
        except NotImplementedError:  # pragma: no cover - Windows
            pass
    await recorder.run(duration_seconds if duration_seconds > 0 else None)
    print("run_id={}".format(recorder.run_id))
    for path in recorder.raw_store.paths + recorder.normalized_store.paths:
        print(path)
    return 0


async def _collect_databento(
    config_path: str,
    metal: str,
    duration_seconds: float,
    max_records: int,
) -> int:
    config = load_metals_config(config_path)
    recorder = DatabentoMetalsRecorder(
        config,
        metal=metal,
        max_records=max_records if max_records > 0 else None,
    )
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, recorder.stop)
        except NotImplementedError:  # pragma: no cover - Windows
            pass
    counts = await recorder.run(
        duration_seconds=duration_seconds if duration_seconds > 0 else None
    )
    print(json.dumps({**counts, "run_id": recorder.run_id}, sort_keys=True))
    return 0


async def async_main(args: argparse.Namespace) -> int:
    if args.command == "discover":
        return await _discover(args.config)
    if args.command == "check-access":
        return await _check_access(args.config)
    if args.command == "collect":
        return await _collect(args.config, args.duration_seconds)
    if args.command == "collect-databento":
        return await _collect_databento(
            args.config,
            args.metal,
            args.duration_seconds,
            args.max_records,
        )
    if args.command == "backfill-kalshi":
        config = load_metals_config(args.config)
        counts = await backfill_settled_markets(
            config,
            metal=args.metal,
            max_markets=args.max_markets or None,
            include_trades=not args.skip_trades,
        )
        print(json.dumps(counts, sort_keys=True))
        return 0
    if args.command == "export-parquet":
        config = load_metals_config(args.config)
        counts = export_normalized_parquet(config.storage.root)
        catalog = build_duckdb_catalog(config.storage.root)
        print(
            json.dumps(
                {**counts, "catalog": str(catalog)}, sort_keys=True
            )
        )
        return 0
    if args.command == "backfill-pyth-history":
        config = load_metals_config(args.config)
        counts = await backfill_pyth_history(
            config,
            metal=args.metal,
            start_seconds=args.from_seconds,
            end_seconds=args.to_seconds,
        )
        print(json.dumps(counts, sort_keys=True))
        return 0
    raise ValueError("Unknown command: {}".format(args.command))


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
