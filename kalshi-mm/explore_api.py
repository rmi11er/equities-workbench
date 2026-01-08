#!/usr/bin/env python3
"""Explore Kalshi API to find available markets."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.connector import KalshiConnector, APIError


async def explore():
    config = load_config()
    connector = KalshiConnector(config)

    try:
        await connector.start()

        # Try different endpoints
        endpoints = [
            "/markets",
            "/markets?status=open",
            "/markets?status=open&limit=100",
            "/events",
            "/events?status=open",
            "/series",
        ]

        for endpoint in endpoints:
            print(f"\n{'='*60}")
            print(f"Trying: {endpoint}")
            print('='*60)
            try:
                resp = await connector._request("GET", endpoint)

                # Print keys to understand structure
                print(f"Response keys: {list(resp.keys())}")

                # Try to find markets/events
                for key in ["markets", "events", "series"]:
                    if key in resp:
                        items = resp[key]
                        print(f"Found {len(items)} {key}")

                        # Show first few
                        for item in items[:5]:
                            if "ticker" in item:
                                title = item.get("title", item.get("event_ticker", ""))[:50]
                                print(f"  - {item['ticker']}: {title}")
                            elif "event_ticker" in item:
                                print(f"  - {item['event_ticker']}: {item.get('title', '')[:50]}")
                            elif "series_ticker" in item:
                                print(f"  - {item['series_ticker']}: {item.get('title', '')[:50]}")

                        if len(items) > 5:
                            print(f"  ... and {len(items) - 5} more")

            except APIError as e:
                print(f"Error: {e.status} - {e.response}")

        # Also try to get a specific market from the UI
        print(f"\n{'='*60}")
        print("Enter a ticker from the demo UI to test (or press Enter to skip):")
        print('='*60)

    finally:
        await connector.stop()


if __name__ == "__main__":
    asyncio.run(explore())
