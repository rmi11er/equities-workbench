# KMM-v2.2 SPECIFICATION
## Layering & Volatility Defense System

---

## OVERVIEW

**Objective:** Replace single-point quoting with a dynamic Quote Ladder that adapts to market regimes.

**Core Concept:** Deploy a 3-layer "Net" structure that detects toxic flow and automatically retreats to defensive positioning.

| Regime | Behavior |
|--------|----------|
| Normal | Full ladder (Canary + Body + Wall) captures spread and detects threats |
| High-Vol | Triggered by Canary death; retreats to Fortified Wall only |

---

## CONFIGURATION

**File:** `config.toml`

```toml
[layering]
enabled = true
layer_count = 3                 # 0=Canary, 1=Body, 2=Wall
base_tick_step = 1              # Gap between layers (cents)

# Sizing
canary_size_ratio = 0.5         # Layer 0: Base * 0.5
wall_size_ratio = 2.0           # Layer 2: Base * 2.0

# Tripwire
canary_timeout_ms = 50          # Fill faster than this = toxic sweep

[volatility_response]
enabled = true
spread_multiplier = 2.0         # Tick step multiplier in High-Vol
wall_only_mode = true           # High-Vol disables Layers 0 & 1
regime_timeout_sec = 30         # Duration of High-Vol state
```

---

## DATA MODELS

**File:** `src/models.py`

```python
@dataclass
class QuoteLevel:
    price: int
    size: int
    layer_index: int    # 0=Canary, 1=Body, 2=Wall
    is_active: bool

@dataclass
class QuoteLadder:
    bids: List[QuoteLevel]
    asks: List[QuoteLevel]
```

---

## STRATEGY LOGIC

**File:** `src/strategy.py`

### Ladder Generation Algorithm

**Inputs:**
- `anchor_price` (Stoikov Mid)
- `base_size`
- `vol_regime` (Normal | High-Vol)

**Step 1: Determine Parameters**

```
IF regime == Normal:
    step = config.base_tick_step
    active_layers = [0, 1, 2]

IF regime == High-Vol:
    step = config.base_tick_step * config.spread_multiplier
    active_layers = [2]  # Wall only
```

**Step 2: Generate Layers**

```
FOR i IN [0, 1, 2]:
    IF i NOT IN active_layers:
        mark is_active = False
        CONTINUE
    
    bid_price = anchor_bid - (i * step)
    ask_price = anchor_ask + (i * step)
    
    size = base_size
    IF i == 0: size *= canary_size_ratio
    IF i == 2: size *= wall_size_ratio
```

**Output:**
- Normal: Tight 3-rung ladder
- High-Vol: Single deep Wall order

---

## EXECUTION LOGIC

**File:** `src/execution.py`

### Reconciliation Algorithm

```
1. GET open_orders grouped by layer tag

2. FOR each layer IN desired_ladder:
   
   IF layer.is_active:
       IF matching_order_exists:
           AMEND order (price, size)
       ELSE:
           POST new order
   
   IF layer.is_inactive:
       IF matching_order_exists:
           CANCEL order immediately
```

**Regime Transition (Normal → High-Vol):**
- Layers 0 & 1 become inactive → immediate CANCEL
- Layer 2 remains active → AMEND to deeper price

---

## TRIPWIRE LOGIC

**File:** `src/market_maker.py`

### Canary Death Detection

**Trigger:** WebSocket fill event

```
ON fill_event:
    IF order.layer_index == 0:  # Canary
        lifespan = now - order.created_at
        
        IF lifespan < config.canary_timeout_ms:
            LOG "CRITICAL: Toxic sweep detected"
            volatility_regime.activate_high_vol(reason="CANARY_DEATH")
```

**Immediate Effect:**
- Next tick loop generates Wall-Only ladder
- Execution engine cancels L0/L1
- L2 retreats to deeper price

---

## BEHAVIORAL MATRIX

| Component | Normal State | High-Vol State |
|-----------|--------------|----------------|
| Layer 0 (Canary) | Active, small size, tight price | DISABLED, cancelled |
| Layer 1 (Body) | Active, standard size | DISABLED, cancelled |
| Layer 2 (Wall) | Active, large size | FORTIFIED, deeper + larger |
| Spread Width | Tight (`tick_step=1`) | Wide (`tick_step * multiplier`) |
| Goal | Capture spread, detect sweeps | Capital preservation |

---

## IMPLEMENTATION CHECKLIST

```
[ ] CONFIG
    [ ] Add [layering] block to config.toml
    [ ] Add wall_only_mode param to [volatility_response]

[ ] MODELS
    [ ] Update Quote struct with layer_index field
    [ ] Update Quote struct with is_active field

[ ] REGIME MANAGER
    [ ] Ensure activate_high_vol() is thread-safe
    [ ] Can be called from WebSocket thread

[ ] STRATEGY
    [ ] Rewrite get_quotes() to loop through layers
    [ ] Apply active_layers filter based on regime
    [ ] Apply sizing ratios per layer

[ ] EXECUTION
    [ ] Update reconcile() to handle layer state changes
    [ ] Cancel orders when layer transitions Active → Inactive
    [ ] Tag orders with layer index for identification

[ ] TESTING
    [ ] Simulate Canary Death event
    [ ] Verify L0/L1 cancel within 1ms
    [ ] Verify L2 moves deeper on regime switch
```

---

## FILE CHANGE SUMMARY

| File | Changes |
|------|---------|
| `config.toml` | Add `[layering]` block, add `wall_only_mode` |
| `src/models.py` | Add `QuoteLevel`, `QuoteLadder` dataclasses |
| `src/strategy.py` | Rewrite `get_quotes()` with layer generation loop |
| `src/execution.py` | Update `reconcile()` for layer state handling |
| `src/market_maker.py` | Add Canary death detection on fill events |
