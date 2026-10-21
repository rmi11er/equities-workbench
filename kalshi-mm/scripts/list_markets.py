#!/usr/bin/env python3
"""List active markets on Kalshi."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.connector import KalshiConnector


async def list_markets():
    config = load_config()
    connector = KalshiConnector(config)

    try:
        await connector.start()

        # Get markets with decent liquidity
        resp = await connector._request("GET", "/markets?status=open&limit=20")
        markets = resp.get("markets", [])

        print(f"{'TICKER':<30} {'TITLE':<50} {'YES BID':>8} {'YES ASK':>8}")
        print("-" * 100)

        for m in markets:
            ticker = m.get("ticker", "")
            title = m.get("title", "")[:48]
            yes_bid = m.get("yes_bid", 0)
            yes_ask = m.get("yes_ask", 0)

            # Only show markets with some liquidity
            if yes_bid > 0 or yes_ask > 0:
                print(f"{ticker:<30} {title:<50} {yes_bid:>8} {yes_ask:>8}")

    finally:
        await connector.stop()


if __name__ == "__main__":
    asyncio.run(list_markets())
