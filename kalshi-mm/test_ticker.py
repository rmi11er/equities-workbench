#!/usr/bin/env python3
"""Test specific ticker access."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.connector import KalshiConnector, APIError


async def test_ticker(ticker: str):
    config = load_config()
    connector = KalshiConnector(config)

    try:
        await connector.start()

        # Try to get this specific market
        print(f"\n=== Testing ticker: {ticker} ===\n")

        # Try direct market lookup
        try:
            resp = await connector._request("GET", f"/markets/{ticker}")
            print(f"Direct lookup SUCCESS:")
            market = resp.get("market", resp)
            print(f"  Title: {market.get('title')}")
            print(f"  Status: {market.get('status')}")
            print(f"  Yes Bid: {market.get('yes_bid')}")
            print(f"  Yes Ask: {market.get('yes_ask')}")
            print(f"  Volume: {market.get('volume')}")
            return
        except APIError as e:
            print(f"Direct lookup failed: {e.status}")

        # Try via event
        event_ticker = ticker.rsplit("-", 1)[0]  # kxteamsinsb-26 -> kxteamsinsb
        print(f"\nTrying event: {event_ticker}")

        try:
            resp = await connector._request("GET", f"/events/{event_ticker}")
            print(f"Event lookup SUCCESS:")
            event = resp.get("event", resp)
            print(f"  Title: {event.get('title')}")
            print(f"  Status: {event.get('status')}")

            # Get markets for this event
            markets = event.get("markets", [])
            print(f"  Markets: {len(markets)}")
            for m in markets[:10]:
                print(f"    - {m.get('ticker')}: {m.get('title', '')[:40]}")

        except APIError as e:
            print(f"Event lookup failed: {e.status}")

        # Try listing markets with event filter
        print(f"\nTrying markets?event_ticker={event_ticker}")
        try:
            resp = await connector._request("GET", f"/markets?event_ticker={event_ticker}")
            markets = resp.get("markets", [])
            print(f"Found {len(markets)} markets")
            for m in markets[:10]:
                print(f"  - {m.get('ticker')}: bid={m.get('yes_bid')} ask={m.get('yes_ask')}")
        except APIError as e:
            print(f"Failed: {e.status}")

    finally:
        await connector.stop()


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "kxteamsinsb-26"
    asyncio.run(test_ticker(ticker))
