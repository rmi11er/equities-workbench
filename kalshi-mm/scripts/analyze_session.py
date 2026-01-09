#!/usr/bin/env python3
"""
Post-session analysis script for market maker decisions.

Reads logs/decisions.jsonl and generates a comprehensive report
explaining what happened and why during a trading session.

Usage:
    python scripts/analyze_session.py [--log-path logs/decisions.jsonl]
"""

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class SessionSummary:
    """Summary statistics for a session."""
    start_time: str = ""
    end_time: str = ""
    duration_sec: float = 0.0

    # Tick stats
    total_ticks: int = 0
    quote_updates: int = 0
    quote_skips: int = 0

    # Impulse stats
    impulse_checks: int = 0
    impulse_triggers: int = 0
    hard_limit_triggers: int = 0
    reservation_crossing_triggers: int = 0
    toxicity_triggers: int = 0

    # Fill stats
    total_fills: int = 0
    buy_fills: int = 0
    sell_fills: int = 0
    total_volume: int = 0
    realized_pnl: float = 0.0

    # Latency stats
    latency_mean_us: float = 0.0
    latency_p50_us: float = 0.0
    latency_p95_us: float = 0.0
    latency_p99_us: float = 0.0
    latency_max_us: float = 0.0

    # Component latency breakdown
    orderbook_mean_us: float = 0.0
    strategy_mean_us: float = 0.0
    execution_mean_us: float = 0.0


@dataclass
class QuoteChangeEvent:
    """Significant quote change with explanation."""
    timestamp: str
    tick: int
    old_bid: Optional[int]
    old_ask: Optional[int]
    new_bid: int
    new_ask: int
    reason: str
    details: dict


def load_decisions(log_path: str) -> list[dict]:
    """Load decision log entries from JSONL file."""
    entries = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def analyze_session(entries: list[dict]) -> SessionSummary:
    """Analyze session and compute summary statistics."""
    summary = SessionSummary()

    if not entries:
        return summary

    # Find time range
    timestamps = [e.get("timestamp", "") for e in entries if e.get("timestamp")]
    if timestamps:
        summary.start_time = min(timestamps)
        summary.end_time = max(timestamps)

        try:
            start_dt = datetime.fromisoformat(summary.start_time)
            end_dt = datetime.fromisoformat(summary.end_time)
            summary.duration_sec = (end_dt - start_dt).total_seconds()
        except (ValueError, TypeError):
            pass

    # Count by type
    latencies = []
    component_latencies = defaultdict(list)

    for entry in entries:
        entry_type = entry.get("type", "")

        if entry_type == "quote_update":
            summary.quote_updates += 1
            summary.total_ticks += 1

            # Collect latency data
            latency = entry.get("latency", {})
            if latency.get("total_tick_us"):
                latencies.append(latency["total_tick_us"])
                component_latencies["orderbook"].append(latency.get("orderbook_update_us", 0))
                component_latencies["strategy"].append(latency.get("strategy_calc_us", 0))
                component_latencies["execution"].append(latency.get("execution_us", 0))

        elif entry_type == "quote_skip":
            summary.quote_skips += 1

        elif entry_type == "impulse_check":
            summary.impulse_checks += 1

        elif entry_type == "impulse_trigger":
            summary.impulse_triggers += 1
            reason = entry.get("reason", "")
            if reason == "HARD_LIMIT":
                summary.hard_limit_triggers += 1
            elif reason == "RESERVATION_CROSSING":
                summary.reservation_crossing_triggers += 1
            elif reason == "TOXICITY_SPIKE":
                summary.toxicity_triggers += 1

        elif entry_type == "fill":
            summary.total_fills += 1
            summary.total_volume += entry.get("size", 0)
            if entry.get("side") == "buy":
                summary.buy_fills += 1
            else:
                summary.sell_fills += 1
            summary.realized_pnl = entry.get("total_realized_pnl", summary.realized_pnl)

    # Calculate latency statistics
    if latencies:
        latencies_sorted = sorted(latencies)
        summary.latency_mean_us = statistics.mean(latencies)
        summary.latency_p50_us = latencies_sorted[len(latencies) // 2]
        summary.latency_p95_us = latencies_sorted[int(0.95 * len(latencies))]
        summary.latency_p99_us = latencies_sorted[int(0.99 * len(latencies))]
        summary.latency_max_us = max(latencies)

        summary.orderbook_mean_us = statistics.mean(component_latencies["orderbook"])
        summary.strategy_mean_us = statistics.mean(component_latencies["strategy"])
        summary.execution_mean_us = statistics.mean(component_latencies["execution"])

    return summary


def find_significant_changes(entries: list[dict], threshold_cents: int = 3) -> list[QuoteChangeEvent]:
    """Find significant quote changes and explain why they happened."""
    changes = []
    prev_bid = None
    prev_ask = None

    for entry in entries:
        if entry.get("type") != "quote_update":
            continue

        curr_bid = entry.get("final_bid")
        curr_ask = entry.get("final_ask")

        if prev_bid is not None and prev_ask is not None:
            bid_change = abs(curr_bid - prev_bid) if curr_bid and prev_bid else 0
            ask_change = abs(curr_ask - prev_ask) if curr_ask and prev_ask else 0

            if bid_change >= threshold_cents or ask_change >= threshold_cents:
                # Analyze why the change happened
                market = entry.get("market", {})
                position = entry.get("position", {})

                reasons = []

                # Check mid price movement
                eff_mid = market.get("effective_mid", 50)
                simple_mid = market.get("simple_mid", 50)
                if eff_mid and simple_mid and abs(eff_mid - simple_mid) > 2:
                    reasons.append(f"effective_mid ({eff_mid:.1f}) diverged from simple_mid ({simple_mid:.1f})")

                # Check volatility
                vol = market.get("volatility", 0)
                if vol > 10:
                    reasons.append(f"high volatility ({vol:.1f})")

                # Check inventory skew
                inventory = position.get("inventory", 0)
                if abs(inventory) > 200:
                    reasons.append(f"inventory skew ({inventory})")

                # Check liquidity
                liq_score = market.get("liquidity_score", 1)
                spread_mult = entry.get("liquidity_spread_mult", 1)
                if spread_mult > 1.2:
                    reasons.append(f"liquidity adjustment (mult={spread_mult:.2f})")

                # Check LIP
                if entry.get("lip_adjusted"):
                    reasons.append("LIP constraint applied")

                reason_str = "; ".join(reasons) if reasons else "normal market movement"

                changes.append(QuoteChangeEvent(
                    timestamp=entry.get("timestamp", ""),
                    tick=entry.get("tick_number", 0),
                    old_bid=prev_bid,
                    old_ask=prev_ask,
                    new_bid=curr_bid,
                    new_ask=curr_ask,
                    reason=reason_str,
                    details={
                        "effective_mid": eff_mid,
                        "volatility": vol,
                        "inventory": inventory,
                        "ofi": market.get("ofi", 0),
                    }
                ))

        prev_bid = curr_bid
        prev_ask = curr_ask

    return changes


def find_fills_with_context(entries: list[dict]) -> list[dict]:
    """Extract fills with full context."""
    fills = []
    for entry in entries:
        if entry.get("type") == "fill":
            fills.append({
                "timestamp": entry.get("timestamp"),
                "side": entry.get("side"),
                "size": entry.get("size"),
                "price": entry.get("price"),
                "inventory_before": entry.get("inventory_before"),
                "inventory_after": entry.get("inventory_after"),
                "mid_at_fill": entry.get("mid_at_fill"),
                "our_bid": entry.get("our_bid_at_fill"),
                "our_ask": entry.get("our_ask_at_fill"),
                "pnl_delta": entry.get("realized_pnl_delta"),
                "edge": None,  # Calculated below
            })
            # Calculate edge: difference between fill price and mid
            if fills[-1]["mid_at_fill"]:
                mid = fills[-1]["mid_at_fill"]
                price = fills[-1]["price"]
                if fills[-1]["side"] == "buy":
                    fills[-1]["edge"] = mid - price  # Positive = bought below mid
                else:
                    fills[-1]["edge"] = price - mid  # Positive = sold above mid
    return fills


def print_report(summary: SessionSummary, changes: list[QuoteChangeEvent], fills: list[dict]):
    """Print formatted analysis report."""
    print("=" * 80)
    print("SESSION ANALYSIS REPORT")
    print("=" * 80)

    print(f"\n--- SESSION OVERVIEW ---")
    print(f"Start:      {summary.start_time}")
    print(f"End:        {summary.end_time}")
    print(f"Duration:   {summary.duration_sec:.1f} seconds")
    print(f"Total Ticks: {summary.total_ticks}")

    print(f"\n--- QUOTE ACTIVITY ---")
    print(f"Quote Updates:  {summary.quote_updates}")
    print(f"Quote Skips:    {summary.quote_skips} (debounced)")
    if summary.quote_updates + summary.quote_skips > 0:
        skip_rate = summary.quote_skips / (summary.quote_updates + summary.quote_skips) * 100
        print(f"Skip Rate:      {skip_rate:.1f}%")

    print(f"\n--- IMPULSE CONTROL ---")
    print(f"Impulse Checks:    {summary.impulse_checks}")
    print(f"Impulse Triggers:  {summary.impulse_triggers}")
    if summary.impulse_triggers > 0:
        print(f"  - Hard Limit:          {summary.hard_limit_triggers}")
        print(f"  - Reservation Crossing: {summary.reservation_crossing_triggers}")
        print(f"  - Toxicity Spike:       {summary.toxicity_triggers}")

    print(f"\n--- FILLS & P&L ---")
    print(f"Total Fills:    {summary.total_fills}")
    print(f"  - Buy Fills:  {summary.buy_fills}")
    print(f"  - Sell Fills: {summary.sell_fills}")
    print(f"Total Volume:   {summary.total_volume} contracts")
    print(f"Realized P&L:   {summary.realized_pnl:.2f} cents")

    print(f"\n--- LATENCY (microseconds) ---")
    print(f"Mean:    {summary.latency_mean_us:.0f}us")
    print(f"P50:     {summary.latency_p50_us:.0f}us")
    print(f"P95:     {summary.latency_p95_us:.0f}us")
    print(f"P99:     {summary.latency_p99_us:.0f}us")
    print(f"Max:     {summary.latency_max_us:.0f}us")

    print(f"\n--- COMPONENT LATENCY (mean) ---")
    print(f"Orderbook: {summary.orderbook_mean_us:.0f}us")
    print(f"Strategy:  {summary.strategy_mean_us:.0f}us")
    print(f"Execution: {summary.execution_mean_us:.0f}us")

    if changes:
        print(f"\n--- SIGNIFICANT QUOTE CHANGES (>3 cents) ---")
        for i, change in enumerate(changes[:20]):  # Show first 20
            print(f"\n[{change.tick}] {change.timestamp}")
            print(f"  Bid: {change.old_bid} -> {change.new_bid} ({change.new_bid - (change.old_bid or 0):+d}c)")
            print(f"  Ask: {change.old_ask} -> {change.new_ask} ({change.new_ask - (change.old_ask or 0):+d}c)")
            print(f"  Reason: {change.reason}")
        if len(changes) > 20:
            print(f"\n  ... and {len(changes) - 20} more significant changes")

    if fills:
        print(f"\n--- FILL DETAILS ---")
        for fill in fills[:20]:
            edge_str = f", edge={fill['edge']:+.1f}c" if fill['edge'] else ""
            print(f"[{fill['timestamp']}] {fill['side'].upper()} {fill['size']}@{fill['price']} "
                  f"(inv: {fill['inventory_before']}->{fill['inventory_after']}, "
                  f"pnl_delta={fill['pnl_delta']:.0f}c{edge_str})")
        if len(fills) > 20:
            print(f"  ... and {len(fills) - 20} more fills")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze market maker session")
    parser.add_argument(
        "--log-path",
        default="logs/decisions.jsonl",
        help="Path to decisions.jsonl file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable"
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return 1

    print(f"Loading {log_path}...")
    entries = load_decisions(str(log_path))
    print(f"Loaded {len(entries)} entries")

    summary = analyze_session(entries)
    changes = find_significant_changes(entries)
    fills = find_fills_with_context(entries)

    if args.json:
        import dataclasses
        output = {
            "summary": dataclasses.asdict(summary),
            "significant_changes": [
                {
                    "timestamp": c.timestamp,
                    "tick": c.tick,
                    "old_bid": c.old_bid,
                    "old_ask": c.old_ask,
                    "new_bid": c.new_bid,
                    "new_ask": c.new_ask,
                    "reason": c.reason,
                }
                for c in changes
            ],
            "fills": fills,
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(summary, changes, fills)

    return 0


if __name__ == "__main__":
    exit(main())
