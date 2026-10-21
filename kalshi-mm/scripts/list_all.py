import asyncio
import urllib.parse
from collections import defaultdict

from src.config import load_config
config = load_config()
from src.connector import KalshiConnector

# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------
SERIES_FILTER = "KXNFL"  # Target College Football. Change to "KXNFL" for NFL.
PAGE_LIMIT    = 200        # Max allowed by Kalshi API

class MarketScanner:
    def __init__(self, connector):
        self.connector = connector

    async def _fetch_all_pages(self, endpoint, params):
        """
        Generic helper to fetch ALL data from paginated endpoints.
        Handles pagination via cursor.
        """
        all_results = []
        cursor = None
        page = 0

        # Ensure we use max limit for efficiency
        params["limit"] = PAGE_LIMIT

        while True:
            page += 1
            current_params = dict(params)  # Copy to avoid mutation
            if cursor:
                current_params["cursor"] = cursor

            # Construct query safely
            query = urllib.parse.urlencode(current_params)

            try:
                resp = await self.connector._request("GET", f"{endpoint}?{query}")
            except Exception as e:
                print(f"Error fetching page {page} from {endpoint}: {e}")
                break

            # Determine the data key based on endpoint
            if "events" in endpoint:
                data_key = "events"
            elif "markets" in endpoint:
                data_key = "markets"
            else:
                data_key = list(resp.keys())[0] if resp else "data"

            batch = resp.get(data_key, [])
            all_results.extend(batch)

            print(f"   Page {page}: fetched {len(batch)} items (total: {len(all_results)})")

            # Check for next page
            cursor = resp.get("cursor")
            if not cursor or len(batch) == 0:
                break

        return all_results

    async def get_football_state(self):
        """
        Fetches Events and Markets for college football.
        Returns a dictionary of games with their associated markets.
        """
        print(f"--- Scanning for Active {SERIES_FILTER} Games ---")

        # 1. Fetch Active Events - we need to search broader since individual games
        # have tickers like KXNCAAFGAME-26JAN08MIAMISS, KXNCAAFSPREAD-26JAN08MIAMISS, etc.
        # The series_ticker filter only returns the main series, not individual game events
        print("\n1. Fetching all open Events...")
        all_events = await self._fetch_all_pages("/events", {
            "status": "open",
        })

        # Filter to CFB events (tickers starting with KXNCAAF)
        events = [e for e in all_events if e.get("event_ticker", "").startswith(SERIES_FILTER)]
        print(f"   Found {len(events)} CFB events (filtered from {len(all_events)} total)")

        if not events:
            print("   No CFB events found!")
            return {}

        # Collect all event tickers to fetch their markets
        event_tickers = [e.get("event_ticker") for e in events if e.get("event_ticker")]
        print(f"   Event tickers: {event_tickers[:10]}..." if len(event_tickers) > 10 else f"   Event tickers: {event_tickers}")

        # 2. Fetch Markets for each event
        # The API supports filtering by event_ticker, which is more efficient
        print("\n2. Fetching Markets for each event...")
        all_markets = []

        for event_ticker in event_tickers:
            markets = await self._fetch_all_pages("/markets", {
                "event_ticker": event_ticker,
            })
            all_markets.extend(markets)

        print(f"   Total markets found: {len(all_markets)}")

        # 3. Index Markets by Event Ticker for O(1) lookup
        markets_map = defaultdict(list)
        for m in all_markets:
            markets_map[m.get("event_ticker")].append(m)

        # 4. Group by Physical Game (Suffix Join)
        # Logic: "KXNCAAFSPREAD-26JAN09OREIND" and "KXNCAAFGAME-26JAN09OREIND"
        # share the suffix "26JAN09OREIND".
        games_data = defaultdict(lambda: {"title": "Unknown", "events": [], "markets": []})

        for event in events:
            ticker = event.get("event_ticker", "")

            # Robust Suffix Extraction
            if "-" in ticker:
                game_id = ticker.split("-")[-1]
            else:
                game_id = ticker

            # Use the main GAME event for the human-readable title
            if "GAME" in ticker:
                games_data[game_id]["title"] = event.get("title")

            # Track event types
            games_data[game_id]["events"].append(ticker)

            # Link markets to this Game ID
            related_markets = markets_map.get(ticker, [])
            games_data[game_id]["markets"].extend(related_markets)

        return games_data

    def display_games(self, games_data):
        """
        Cleanly prints the game state.
        """
        if not games_data:
            print("\nNo active games found.")
            return

        print(f"\n{'='*80}")
        print(f"SUCCESS: Found {len(games_data)} Active Games")
        print("=" * 80)

        for game_id, data in sorted(games_data.items()):
            title = data["title"]
            events = data.get("events", [])
            markets = data.get("markets", [])

            # Fallback title if we missed the main event
            if title == "Unknown":
                title = f"Game ID: {game_id}"

            print(f"\n{'='*70}")
            print(f"GAME: {title}")
            print(f"   ID: {game_id}")
            print(f"   Events: {', '.join(events)}")
            print(f"   Markets: {len(markets)}")
            print(f"   {'TYPE':<15} {'TICKER':<35} {'BID':<5} {'ASK':<5} {'VOL':<8}")
            print("   " + "-" * 65)

            if not markets:
                print("   (No markets found for this event)")
                continue

            # Sort Markets by volume (descending) - most active markets first
            sorted_markets = sorted(markets, key=lambda m: m.get("volume", 0) or 0, reverse=True)

            for m in sorted_markets:
                ticker = m.get("ticker", "")
                subtitle = m.get("subtitle", "") or m.get("title", "")[:30]

                # Intelligent Labeling
                if "GAME" in ticker:
                    m_type = "Moneyline"
                elif "SPREAD" in ticker:
                    m_type = "Spread"
                elif "TOTAL" in ticker:
                    m_type = "Total"
                else:
                    m_type = "Prop"

                # Data
                bid = m.get("yes_bid", 0) or 0
                ask = m.get("yes_ask", 0) or 0
                vol = m.get("volume", 0) or 0

                # Truncate ticker for display
                display_ticker = ticker[-35:] if len(ticker) > 35 else ticker

                print(f"   {m_type:<15} {display_ticker:<35} {bid:<5} {ask:<5} {vol:<8}")

        print(f"\n{'='*80}")

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
async def main():
    connector = KalshiConnector(config)
    await connector.start()

    try:
        scanner = MarketScanner(connector)
        games = await scanner.get_football_state()
        scanner.display_games(games)
    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await connector.stop()

if __name__ == "__main__":
    asyncio.run(main())