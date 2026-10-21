#!/usr/bin/env python3
"""
Generate config-compatible ticker list for upcoming NFL games.

For each game, includes:
- Both moneyline tickers (one per team)
- Top 2 spread tickers (by volume)
- Top 1 prop bet (by volume)

Usage:
    python nfl_tickers.py
    python nfl_tickers.py --output config  # Output as TOML array
"""

import asyncio
import argparse
import urllib.parse
from collections import defaultdict

from src.config import load_config
from src.connector import KalshiConnector

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
SERIES_FILTER = "KXNFL"
PAGE_LIMIT = 200


class NFLTickerGenerator:
    def __init__(self, connector):
        self.connector = connector

    async def _fetch_all_pages(self, endpoint, params):
        """Fetch all data from paginated endpoints."""
        all_results = []
        cursor = None
        params["limit"] = PAGE_LIMIT

        while True:
            current_params = dict(params)
            if cursor:
                current_params["cursor"] = cursor

            query = urllib.parse.urlencode(current_params)

            try:
                resp = await self.connector._request("GET", f"{endpoint}?{query}")
            except Exception as e:
                print(f"Error fetching from {endpoint}: {e}")
                break

            if "events" in endpoint:
                data_key = "events"
            elif "markets" in endpoint:
                data_key = "markets"
            else:
                data_key = list(resp.keys())[0] if resp else "data"

            batch = resp.get(data_key, [])
            all_results.extend(batch)

            cursor = resp.get("cursor")
            if not cursor or len(batch) == 0:
                break

        return all_results

    async def get_nfl_games(self):
        """Fetch NFL events and markets, grouped by game."""
        print(f"Scanning for NFL games ({SERIES_FILTER})...")

        # Fetch all open events
        all_events = await self._fetch_all_pages("/events", {"status": "open"})
        events = [e for e in all_events if e.get("event_ticker", "").startswith(SERIES_FILTER)]
        print(f"Found {len(events)} NFL events")

        if not events:
            return {}

        # Fetch markets for each event
        event_tickers = [e.get("event_ticker") for e in events if e.get("event_ticker")]
        all_markets = []

        for event_ticker in event_tickers:
            markets = await self._fetch_all_pages("/markets", {"event_ticker": event_ticker})
            all_markets.extend(markets)

        print(f"Found {len(all_markets)} total markets")

        # Index markets by event ticker
        markets_map = defaultdict(list)
        for m in all_markets:
            markets_map[m.get("event_ticker")].append(m)

        # Group by physical game (suffix join)
        games = defaultdict(lambda: {"title": "Unknown", "events": [], "markets": []})

        for event in events:
            ticker = event.get("event_ticker", "")
            game_id = ticker.split("-")[-1] if "-" in ticker else ticker

            if "GAME" in ticker:
                games[game_id]["title"] = event.get("title", "Unknown")

            games[game_id]["events"].append(ticker)
            games[game_id]["markets"].extend(markets_map.get(ticker, []))

        return games

    def select_tickers_for_game(self, markets):
        """
        Select tickers for a single game:
        - Both moneylines
        - Top 2 spreads by volume
        - Top 1 prop by volume
        """
        moneylines = []
        spreads = []
        totals = []
        props = []

        for m in markets:
            ticker = m.get("ticker", "")
            volume = m.get("volume", 0) or 0

            if "GAME" in ticker:
                moneylines.append((ticker, volume, m.get("subtitle", "")))
            elif "SPREAD" in ticker:
                spreads.append((ticker, volume, m.get("subtitle", "")))
            elif "TOTAL" in ticker:
                totals.append((ticker, volume, m.get("subtitle", "")))
            else:
                props.append((ticker, volume, m.get("subtitle", "")))

        # Sort each category by volume descending
        moneylines.sort(key=lambda x: x[1], reverse=True)
        spreads.sort(key=lambda x: x[1], reverse=True)
        totals.sort(key=lambda x: x[1], reverse=True)
        props.sort(key=lambda x: x[1], reverse=True)

        selected = []

        # Both moneylines
        for ticker, vol, subtitle in moneylines[:2]:
            selected.append({"ticker": ticker, "type": "Moneyline", "volume": vol, "subtitle": subtitle})

        # Top 2 spreads
        for ticker, vol, subtitle in spreads[:2]:
            selected.append({"ticker": ticker, "type": "Spread", "volume": vol, "subtitle": subtitle})

        # Top 1 prop (prefer totals, then other props)
        if totals:
            ticker, vol, subtitle = totals[0]
            selected.append({"ticker": ticker, "type": "Total", "volume": vol, "subtitle": subtitle})
        elif props:
            ticker, vol, subtitle = props[0]
            selected.append({"ticker": ticker, "type": "Prop", "volume": vol, "subtitle": subtitle})

        return selected

    def generate_ticker_list(self, games):
        """Generate the full ticker list for all games."""
        all_tickers = []
        game_details = []

        for game_id, data in sorted(games.items()):
            title = data["title"]
            markets = data["markets"]

            if not markets:
                continue

            selected = self.select_tickers_for_game(markets)
            if selected:
                game_details.append({
                    "game_id": game_id,
                    "title": title,
                    "tickers": selected,
                })
                all_tickers.extend([t["ticker"] for t in selected])

        return all_tickers, game_details


def print_results(all_tickers, game_details, output_format="table"):
    """Print results in requested format."""

    if output_format == "config":
        # Output as TOML-compatible array
        print("\n# Add this to your config.toml:")
        print("market_tickers = [")
        for ticker in all_tickers:
            print(f'    "{ticker}",')
        print("]")
        print(f"\n# Total: {len(all_tickers)} markets across {len(game_details)} games")
        return

    # Table format (default)
    print(f"\n{'='*80}")
    print(f"NFL TICKER SELECTION: {len(all_tickers)} markets across {len(game_details)} games")
    print("="*80)

    for game in game_details:
        print(f"\n{game['title']}")
        print(f"  Game ID: {game['game_id']}")
        print(f"  {'TYPE':<12} {'VOLUME':<10} {'TICKER'}")
        print("  " + "-"*70)

        for t in game["tickers"]:
            print(f"  {t['type']:<12} {t['volume']:<10} {t['ticker']}")

    # Summary
    print(f"\n{'='*80}")
    print("COMBINED TICKER LIST (copy-paste ready):")
    print("="*80)
    print("\nmarket_tickers = [")
    for ticker in all_tickers:
        print(f'    "{ticker}",')
    print("]")


async def main():
    parser = argparse.ArgumentParser(description="Generate NFL ticker list for config")
    parser.add_argument(
        "--output", "-o",
        choices=["table", "config"],
        default="table",
        help="Output format: table (detailed) or config (TOML only)"
    )
    args = parser.parse_args()

    config = load_config()
    connector = KalshiConnector(config)
    await connector.start()

    try:
        generator = NFLTickerGenerator(connector)
        games = await generator.get_nfl_games()
        all_tickers, game_details = generator.generate_ticker_list(games)
        print_results(all_tickers, game_details, args.output)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await connector.stop()


if __name__ == "__main__":
    asyncio.run(main())
