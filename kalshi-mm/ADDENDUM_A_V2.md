# Addendum A: KMM-v2 Specification

## Microstructure & Impulse Control

This addendum extends KMM-v1 from a passive market maker to a hybrid trader capable of intelligent liquidity taking.

## Overview

**Goal:** Upgrade from "Passive Maker" (limit orders only) to "Hybrid Trader" with Impulse Control (dynamic switching between making and taking).

**Theoretical Basis:** Baron, Law & Viens (2019) — *"Market Making under a Weakly Consistent Limit Order Book"*

**Core Insight:** Taking liquidity (paying the spread) is not a failure—it's a calculated tool for risk management (Bailout) and alpha capture (Snipe).

## Theoretical Primitives

### Impulse Control

The "Panic Button" — knowing when crossing the spread is cheaper than holding.

| Version | Behavior |
|---------|----------|
| KMM-v1 (Stoikov) | High inventory → lower bid → wait for fill or hold to expiry |
| KMM-v2 (Law-Viens) | High inventory → evaluate Hamiltonian threshold → market order if optimal |

**Decision Logic:**

```
If Cost_of_Holding > Cost_of_Crossing_Spread:
    Execute Market Order immediately
```

*Rationale:* Losing 1¢ + fee now beats risking a 50¢ adverse move on a 500-lot position.

### Queue Value

The "Invisible Price" — in tight markets, your queue position *is* your edge.

In a 1-tick market (21¢/22¢), price is static. The real variable is queue position.

**State Variable:** `Pos_queue` (estimated contracts ahead of us)

| Queue Position | Fill Probability | Adverse Selection Risk |
|----------------|------------------|------------------------|
| Front (#5) | High | High |
| Back (#5,000) | Low | Low (noise buffer) |

Being #100 at 21¢ is fundamentally different from being #5,000 at 21¢.

## Module Specifications

### Queue Estimator (`microstructure.py`)

**New module.** Tracks virtual queue position since Kalshi doesn't expose `MyQueuePos`.

**Estimation Logic:**

| Event | Action |
|-------|--------|
| Order Placement (T₀) | Snapshot `Bid_Size` at price P; set `My_Pos = Bid_Size` |
| Trade at P | `My_Pos -= Trade_Size` |
| Cancel at P | Conservative: assume cancel came from behind us (no change to `My_Pos`) unless `Cancel_Size > Total_Behind_Us` |
| Our Cancel/Replace | `My_Pos` resets to `Current_Total_Size` (back of queue) |

**Data Requirement:** Connector must expose `msg_seq` or timestamp for every trade packet to prevent double-counting our own fills.

### Impulse Engine (`taker.py`)

**New module.** Parallel execution path that bypasses maker logic.

**Trigger Conditions:**

| Trigger | Condition | Action |
|---------|-----------|--------|
| Hard Inventory Breach | `\|q\| > Q_critical` (e.g., 600 lots) | `Market_Sell(q - Q_target)` |
| Toxicity Spike | Alpha score moves >5% against position | `Market_Close_All` |
| Arb Opportunity | `External_Oracle > Kalshi_Ask + Spread + Fees` | `Market_Buy` |

### Enhanced Strategy Engine (`strategy_v2.py`)

Replaces continuous Stoikov formula with state-machine logic approximating the Law-Viens control policy.

```python
def decide_action(market_state, inventory, queue_pos):
    # Calculate value of holding vs. liquidation cost
    hold_value = calculate_hold_utility(inventory, volatility)
    liquidation_cost = spread + taker_fee
    
    # 1. Check Bailout (Impulse Control)
    if hold_value < -liquidation_cost:
        return Action.MARKET_DUMP
        
    # 2. Check Queue Optimization
    if queue_pos > MAX_QUEUE_DEPTH and inventory < DESIRED_INVENTORY:
        # Too far back to get filled—move to front?
        # Only if it doesn't put us on wrong side of alpha
        return Action.CANCEL_REPLACE_FRONT
        
    # 3. Default Stoikov Behavior
    return Action.MAINTAIN_QUOTES
```

**Decision Hierarchy:**
1. Bailout check (impulse control)
2. Queue optimization
3. Default Stoikov behavior

## Implementation Changes

### The "Recycle" Loop

| Version | Cancel/Replace Purpose |
|---------|------------------------|
| V1 | Change price |
| V2 | Change queue position |

**Scenario:** Long 100, want to sell at 22¢.

| Situation | V1 Action | V2 Action |
|-----------|-----------|-----------|
| Ask is 22¢ | Do nothing | Check queue position |
| Ask is 22¢, we're #5,000, OFI indicates buyers | Do nothing | Evaluate refresh |
| Front of queue, high toxicity | Hold | Cancel and yield priority |

**Key Insight:** If price hasn't moved and we're deep in queue, we're stuck. Refreshing only helps if price moved. But if we're at the front and flow is toxic, yielding priority to others can be optimal.

### Sequence Handling Requirements

Accurate queue estimation requires strict sequence handling:

- Connector must expose `msg_seq` or timestamp on every trade packet
- Queue estimator must verify sequence to avoid double-counting
- Gap detection triggers state reconciliation

## Testing Strategy (V2)

Phase 2 testing shifts focus from pure P&L to **execution quality**.

### Metric: Effective Spread Paid

When executing a bailout (market order):
- Target: 1¢ spread
- Failure: 5¢ slippage due to thin book

### Metric: Queue Accuracy

Log `Estimated_Queue_Pos` at time of fill.

| Estimated Position | Fill Size | Interpretation |
|--------------------|-----------|----------------|
| 500 | 500+ | Estimator accurate |
| 500 | 50 | Estimator broken or adverse selection |

## Implementation Roadmap (Phase 2)

| Phase | Component | Task | Validation |
|-------|-----------|------|------------|
| 2.1 | Microstructure | Build `QueueEstimator` | Place order on Demo, send trades from second account, predict exact fill timing |
| 2.2 | The Taker | Implement `taker.py` | Unit test panic button: `Inventory=1000` → `Market_Sell(500)` |
| 2.3 | Hybrid Strategy | Merge Stoikov pricing with taker bailouts | Integration testing |
| 2.4 | Live Fire | Deploy to low-stakes market | Test queue yielding on toxic flow |

## Updated Project Structure

```
kalshi-mm-v2/
├── kalshi_connector.py    # Exchange I/O (enhanced sequence handling)
├── orderbook.py           # Market state maintenance
├── microstructure.py      # NEW: Queue position estimation
├── stoikov.py             # Base Avellaneda-Stoikov (V1)
├── strategy_v2.py         # NEW: State-machine hybrid strategy
├── taker.py               # NEW: Impulse execution engine
├── execution.py           # Order diffing and execution
├── alpha_engine.py        # External signal integration
├── logger.py              # Async logging infrastructure
├── main.py                # Entry point
└── tests/
    ├── unit/
    │   ├── test_queue_estimator.py
    │   └── test_taker.py
    └── integration/
        └── test_hybrid_strategy.py
```

## Dependencies (Additional)

No new external dependencies required. V2 modules build on existing async infrastructure.

## Migration from V1

V2 is backward-compatible with V1 behavior:
- Set `IMPULSE_CONTROL_ENABLED = False` to disable taker logic
- Set `QUEUE_OPTIMIZATION_ENABLED = False` to disable microstructure tracking
- Default Stoikov behavior remains unchanged when both disabled
