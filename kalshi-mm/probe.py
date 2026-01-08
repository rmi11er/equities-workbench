#!/usr/bin/env python3
"""
Kalshi Connectivity Probe

Tests API connectivity by:
1. Connecting to Demo environment
2. Placing a 1 contract buy order at 1 cent
3. Confirming order ID received
4. Cancelling the order
5. Confirming cancellation
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.connector import KalshiConnector, APIError


async def run_probe() -> bool:
    """Run the connectivity probe."""
    print("=" * 60)
    print("Kalshi Connectivity Probe")
    print("=" * 60)

    # Load config
    config = load_config()

    if config.credentials.api_key_id == "TODO":
        print("\nERROR: API credentials not configured")
        print("Please configure config.toml with your credentials")
        return False

    if config.market_ticker == "TODO":
        print("\nERROR: market_ticker not configured")
        print("Please set a valid ticker in config.toml")
        return False

    connector = KalshiConnector(config)

    try:
        # Step 1: Start connector
        print("\n[1/5] Connecting to API...")
        await connector.start()
        print("      OK - HTTP session created")

        # Step 2: Test REST - Get balance
        print("\n[2/5] Testing REST API (get balance)...")
        try:
            balance = await connector.get_balance()
            print(f"      OK - Balance retrieved: {balance}")
        except APIError as e:
            print(f"      FAIL - API error: {e}")
            return False

        # Step 3: Place test order
        print(f"\n[3/5] Placing test order on {config.market_ticker}...")
        print("      (1 contract @ 1 cent)")
        try:
            order_resp = await connector.create_order(
                ticker=config.market_ticker,
                side="yes",
                price=1,
                count=1,
            )
            order_id = order_resp.get("order", {}).get("order_id")
            if not order_id:
                order_id = order_resp.get("order_id")

            if not order_id:
                print(f"      FAIL - No order_id in response: {order_resp}")
                return False

            print(f"      OK - Order placed: {order_id}")

        except APIError as e:
            print(f"      FAIL - API error: {e}")
            return False

        # Step 4: Cancel order
        print(f"\n[4/5] Cancelling order {order_id}...")
        try:
            cancel_resp = await connector.cancel_order(order_id)
            print(f"      OK - Order cancelled")
        except APIError as e:
            print(f"      FAIL - Cancel error: {e}")
            return False

        # Step 5: Verify cancellation (get orders)
        print("\n[5/5] Verifying cancellation...")
        try:
            orders = await connector.get_orders(config.market_ticker)
            order_list = orders.get("orders", [])

            # Check our order is not active
            active_ids = [o["order_id"] for o in order_list if o.get("status") == "resting"]

            if order_id in active_ids:
                print(f"      FAIL - Order still active")
                return False

            print("      OK - Order confirmed cancelled")

        except APIError as e:
            print(f"      FAIL - API error: {e}")
            return False

        print("\n" + "=" * 60)
        print("PROBE SUCCESSFUL - All connectivity tests passed!")
        print("=" * 60)
        return True

    finally:
        await connector.stop()


async def test_websocket() -> bool:
    """Test WebSocket connectivity."""
    print("\n" + "=" * 60)
    print("WebSocket Connectivity Test")
    print("=" * 60)

    config = load_config()

    if config.credentials.api_key_id == "TODO":
        print("\nERROR: API credentials not configured")
        return False

    connector = KalshiConnector(config)
    messages_received = []

    def on_message(msg):
        messages_received.append(msg)
        print(f"      Received: {msg.type}")

    try:
        await connector.start()

        print("\n[1/3] Connecting WebSocket...")
        await connector.connect_ws()
        print("      OK - WebSocket connected")

        print(f"\n[2/3] Subscribing to orderbook ({config.market_ticker})...")
        connector.on_message(on_message)
        await connector.subscribe_orderbook([config.market_ticker])

        print("\n[3/3] Waiting for messages (5 seconds)...")
        await asyncio.sleep(5)

        if not messages_received:
            print("      WARNING - No messages received")
        else:
            print(f"      OK - Received {len(messages_received)} messages")

        print("\n" + "=" * 60)
        print("WEBSOCKET TEST COMPLETE")
        print("=" * 60)
        return len(messages_received) > 0

    except Exception as e:
        print(f"      FAIL - {e}")
        return False

    finally:
        await connector.stop()


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Kalshi Connectivity Probe")
    parser.add_argument("--ws", action="store_true", help="Test WebSocket only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    success = True

    if args.ws or args.all:
        success = await test_websocket() and success

    if not args.ws or args.all:
        success = await run_probe() and success

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
