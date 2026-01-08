#!/usr/bin/env python3
"""List ALL available markets and events on demo."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.connector import KalshiConnector, APIError


async def list_all():
    config = load_config()
    connector = KalshiConnector(config)

    try:
        await connector.start()

        # Get ALL markets (no filters)
        print("=== ALL MARKETS ===\n")
        try:
            resp = await connector._request("GET", "/markets?limit=200")
            markets = resp.get("markets", [])
            print(f"Total markets found: {len(markets)}\n")

            for m in markets:
                ticker = m.get("ticker", "")
                title = m.get("title", "")[:45]
                status = m.get("status", "")
                yes_bid = m.get("yes_bid", 0) or 0
                yes_ask = m.get("yes_ask", 0) or 0
                volume = m.get("volume", 0) or 0

                print(f"{ticker:<35} {status:<8} bid={yes_bid:<3} ask={yes_ask:<3} vol={volume:<6} {title}")

        except APIError as e:
            print(f"Error: {e}")

        # Get ALL events
        print("\n\n=== ALL EVENTS ===\n")
        try:
            resp = await connector._request("GET", "/events?limit=200")
            events = resp.get("events", [])
            print(f"Total events found: {len(events)}\n")

            for e in events:
                ticker = e.get("event_ticker", "")
                title = e.get("title", "")[:50]
                status = e.get("status", "")
                print(f"{ticker:<30} {status:<10} {title}")

        except APIError as e:
            print(f"Error: {e}")

    finally:
        await connector.stop()


if __name__ == "__main__":
    asyncio.run(list_all())
