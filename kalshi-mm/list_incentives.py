#!/usr/bin/env python3
"""Fetch LIP (Liquidity Incentive Program) specifications from Kalshi API."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.connector import KalshiConnector, APIError


async def list_incentives(market_ticker: str = None):
    config = load_config()
    connector = KalshiConnector(config)

    try:
        await connector.start()

        print("=" * 80)
        print("KALSHI LIQUIDITY INCENTIVE PROGRAMS")
        print("=" * 80)

        # Fetch active liquidity incentives
        try:
            resp = await connector._request(
                "GET",
                "/incentive_programs?status=active&type=liquidity&limit=100"
            )

            programs = resp.get("incentive_programs", [])
            print(f"\nFound {len(programs)} active liquidity programs\n")

            if not programs:
                print("No active LIP programs found.")
                print("\nTry checking: https://kalshi.com/incentives")
                return

            # Filter by market if specified
            if market_ticker:
                programs = [p for p in programs if p.get("market_ticker") == market_ticker]
                print(f"Filtered to {len(programs)} programs for {market_ticker}\n")

            # Display
            print(f"{'MARKET TICKER':<35} {'TYPE':<10} {'REWARD':>10} {'TARGET':>10} {'DISCOUNT':>10}")
            print("-" * 80)

            for p in programs:
                ticker = p.get("market_ticker", "")[:34]
                itype = p.get("incentive_type", "")
                reward = p.get("period_reward", 0)
                target = p.get("target_size", 0)
                discount = p.get("discount_factor_bps", 0)

                print(f"{ticker:<35} {itype:<10} {reward:>10} {target:>10} {discount:>10}")

            # Show full details for first program
            if programs:
                print("\n" + "=" * 80)
                print("FULL DETAILS (first program)")
                print("=" * 80)
                import json
                print(json.dumps(programs[0], indent=2, default=str))

        except APIError as e:
            print(f"Error fetching incentives: {e}")

            if e.status == 404:
                print("\nIncentive endpoint may not be available on demo.")
                print("Check production or: https://kalshi.com/incentives")

    finally:
        await connector.stop()


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(list_incentives(ticker))
