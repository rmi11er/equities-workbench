# Future Enhancements

## CRITICAL: Trade Quality Analysis (Why Are We Getting Filled?)

**Status:** ROOT CAUSE IDENTIFIED

### Analysis from 2026-01-10 MVP Market Fills

**Key findings from KXNFLMVP fills (run_20260110_074134):**

1. **We were quoting ABOVE best bid (bid_distance = +1, +2, +4)**
   - This means we were ALONE at our price level, improving the market
   - We get picked off first when someone sells

2. **Root Cause: Market moved AFTER order placement (NOT a retreat logic bug)**
   - Retreat logic is working correctly at ORDER CREATION time
   - Example: we place bid at 55 when best_bid=55 (at best, safe)
   - Market drops: best_bid becomes 54, we're now EXPOSED at 55
   - The `bid_distance_from_best` in fill logs is measured AT FILL TIME
   - By fill time, we've been left behind by the moving market

3. **Market crashed 53% (mid 36.5 → 17.0) in ~1 minute**
   - We accumulated from 0 → 176 long inventory during the crash
   - Classic adverse selection - informed trader selling into us

4. **Accidental hedging across correlated markets (good luck!)**
   - MSTA: -143 (short)
   - DMAY: +176 (long)
   - Net exposure reduced, but this was not intentional

### Correlated Market Logging - NOW WORKING (run_20260110_080359)

**Example fill with correlated data:**
```json
{
  "ticker": "KXNFLMVP-26-DMAY",
  "fill_price": 11,
  "correlated_markets": [
    {"ticker": "KXNFLMVP-26-DMAY", "mid": 12.5, "arb_signal": -75.0},
    {"ticker": "KXNFLMVP-26-MSTA", "mid": 87.5, "arb_signal": 0.0}
  ],
  "market_root": "KXNFLMVP-26-"
}
```
- MSTA + DMAY mid sum = 87.5 + 12.5 = 100.0 (perfect pricing)
- arb_signal of 0.0 means no arb opportunity at this moment

### Real Problem: Quote Staleness in Fast Markets

**The issue is NOT retreat logic, it's REACTION TIME:**
1. Strategy computes target prices based on current market state
2. Order placed at "safe" price (at or behind best)
3. Market moves (in <1 second in volatile conditions)
4. Our order is now exposed
5. We get picked off before next strategy tick

**Solutions implemented:**
- [x] **Faster tick rate** - VolatilityRegime reduces tick interval from 100ms to 30ms in high-vol
- [x] **Dollar-based depth requirements** - min_join_depth_dollars scales with price (more contracts needed at low prices)
- [x] **Decaying depth multiplier** - In high-vol, require 3x more depth to join (decays over 60s)

**Solutions still needed:**
- [x] **Reactive quote pulling** - DONE: `check_and_pull_exposed_quotes()` in execution.py, called from orderbook handler
- [x] **Depth monitoring** - DONE: Same function now checks if cumulative depth drops below 50% of threshold

### Investigation Progress

**Completed:**
- [x] Analyze fills.jsonl to see our queue position at fill time (bid_distance tracked)
- [x] Check if fills correlate with being alone/exposed on a level (YES - many fills have positive bid_distance)
- [x] Compare our fill prices to correlated markets → now have `arb_signal` field

**Still needed:**
- [ ] Add queue_position_at_fill (requires API call during fill processing)
- [x] Implement reactive quote pulling when exposed - DONE
- [ ] Add pre-fill orderbook snapshot for aggressor size estimation

---

## FUTURE: Position-Aware Quote Floors (Puke Protection)

**Status:** PLANNED

**Problem:** During a big move (e.g., 40c → 10c crash), we accumulate position by buying at multiple levels (40c, 35c, 30c, 25c, 20c, 15c, 10c). If the market then reverts to 20c, our ask quotes would sell at 20c - locking in losses on all purchases above 20c.

**Scenario:**
```
Buy 25 @ 40c (entry)
Buy 30 @ 35c
Buy 35 @ 30c
Buy 35 @ 25c
Buy 25 @ 20c
Buy 25 @ 15c
Market now at 20c, we're +175 long with avg entry ~27c
Our ask quotes at 21c - if filled, we lock in ~6c loss per contract
```

**Desired behavior:**
1. Track weighted average entry price per market
2. In high-vol state (post-crash), set a "quote floor" on asks above avg entry
3. Gradually relax floor as market stabilizes or as time passes
4. Exception: still allow selling at loss if inventory becomes dangerous (risk override)

**Implementation notes:**
- Need to track per-ticker: `avg_entry_price`, `position_size`, `entry_timestamp`
- Quote floor = `max(strategy_ask, avg_entry_price + min_edge)`
- min_edge could decay over time (patience for reversion)
- Risk override: ignore floor if `inventory > max_inventory * 0.9`

**Integration with VolatilityRegime:**
- When entering high-vol due to fills, capture avg_entry_price
- Floor is active while in high-vol state
- Decay as regime normalizes

---

## CRITICAL: Post-Fill Quoting Behavior

**Status:** FIXED

**Root cause found:** Debouncing logic was blocking re-quotes after fills. After a fill:
1. Order cleared from state (`bid_order = None`)
2. Next tick: target price unchanged (market didn't move)
3. `should_update()` only checked price change, not whether we had an order
4. Returned False → skipped update entirely
5. We had NO order on that side until debounce_seconds passed!

**Fix applied in `src/execution.py`:**
- `should_update()` now checks if we're missing an order that we should have
- If `should_bid=True` but `bid_order=None`, immediately return True (force re-quote)
- Same for asks

**Remaining work for correlated markets:**
- If quoting Team A YES and Team B YES for same game, a fill on one should inform the other
- Consider: if we buy Team A YES, we implicitly want to sell Team B YES (or buy Team B NO)

---

## Fill Analysis: Aggressor Order Size Tracking

**Context:** The current fill logger captures orderbook state at fill time, but cannot determine the total size of the incoming order that traded against us. Kalshi's fill message only reports how many of *our* contracts were filled.

**Goal:** Enable analysis of "what was the order size conditioned on us trading against it?"

**Potential approaches:**
1. **Rolling orderbook snapshots** - Capture orderbook state on a rolling basis (e.g., last 5 seconds) so we can compare pre-fill vs post-fill depth to infer aggressor order size
2. **Orderbook delta correlation** - Track sequence of deltas and correlate depth removals with our fills
3. **Public trade feed** - Subscribe to Kalshi's public trade feed (if available) to see all market activity, not just our fills
4. **Depth tracking at our price levels** - Specifically track depth changes at prices where we have resting orders

**Implementation notes:**
- Would require a circular buffer of recent orderbook states keyed by timestamp
- Fill handler would need to look back ~100-500ms to find the "before" state
- Could add fields to FillSnapshot: `depth_at_price_before`, `depth_at_price_after`, `inferred_aggressor_size`
