# Quoting Logic Specification

This document describes the complete quoting logic as implemented in `kalshi-mm`.

---

## 1. Overview

Quotes are generated every 100ms through the following pipeline:

```
Market State → Base Stoikov → Liquidity Adjustment → LIP Constraints → Execution
```

Each stage can modify price and/or size. The final output is a bid/ask pair with sizes.

---

## 2. Base Model: Avellaneda-Stoikov

**Source:** `src/strategy.py`

### 2.1 Reservation Price

The reservation price is the "fair value" adjusted for inventory risk:

```
r = S - q * γ * σ² * (T - t)
```

Where:
- `S` = mid price (cents, 1-99)
- `q` = current inventory (positive = long, negative = short)
- `γ` = risk aversion parameter (default: 0.05)
- `σ` = realized volatility (cents, EMA with 60s half-life, floor 0.1)
- `T - t` = time to expiry (fixed at 1.0)

**Interpretation:** When long (`q > 0`), reservation price drops below mid to encourage selling. When short (`q < 0`), it rises above mid to encourage buying.

### 2.2 Optimal Spread

The theoretical optimal spread around reservation:

```
δ_calculated = γ * σ² * (T - t) + (2 / γ) * ln(1 + γ / k)
δ = max(δ_calculated, min_absolute_spread)
```

Where:
- `k` = market impact parameter (hardcoded: 1.5)
- `min_absolute_spread` = configurable floor (default: 2 cents)

**The min_absolute_spread floor ensures we're compensated in illiquid markets** where the Stoikov formula might suggest spreads too tight to cover adverse selection risk.

**Simplified for our parameters (T-t = 1.0):**
```
δ_calculated = 0.05 * σ² * 1.0 + (2 / 0.05) * ln(1 + 0.05 / 1.5)
δ_calculated = 0.05 * σ² + 40 * ln(1.0333)
δ_calculated = 0.05 * σ² + 1.31
δ = max(δ_calculated, 2.0)  # Apply floor
```

For typical volatility σ = 1.0: `δ_calculated ≈ 1.36 cents`, but floor applies → `δ = 2 cents`

### 2.3 Dynamic Time Horizon (Expiry Urgency)

Instead of a fixed `T - t = 1.0`, we calculate a dynamic time horizon based on market expiration:

```python
def calculate_time_horizon(expiry_ts, normalization_sec=86400):
    seconds_remaining = expiry_ts - now
    fraction = seconds_remaining / normalization_sec
    return clamp(fraction, 0.1, 1.0)
```

**Effect on quoting:**
- Far from expiry (>1 day): `T-t = 1.0` → normal inventory skew
- Near expiry (<1 day): `T-t` shrinks linearly → more aggressive skew
- At expiry: `T-t = 0.1` → 10x more sensitive to inventory (urgency to flatten)

| Time to Expiry | T-t   | Inventory Skew Multiplier |
|----------------|-------|---------------------------|
| > 1 day        | 1.0   | 1.0x (normal)             |
| 12 hours       | 0.5   | 0.5x (reduced)            |
| 2.4 hours      | 0.1   | 0.1x (minimum)            |

**Why this matters:** As expiry approaches, you run out of time to scratch trades. The shrinking `T-t` makes the reservation price more sensitive to inventory, encouraging faster risk reduction.

### 2.4 External Skew

Alpha signals can shift the reservation price:

```
r_adjusted = r + external_skew
```

Where `external_skew` is in cents (positive = bullish, negative = bearish).

**Current state:** `AlphaEngine` returns 0.0 (stub implementation).

### 2.5 Quote Prices

Raw quote prices before clamping:

```
bid_raw = r_adjusted - δ / 2
ask_raw = r_adjusted + δ / 2
```

Clamped to valid range:
```
bid = clamp(round(bid_raw), 1, 99)
ask = clamp(round(ask_raw), 1, 99)
```

### 2.6 Quote Sizes

Base size from config, reduced as inventory approaches limits:

```
inventory_ratio = |q| / max_inventory
size_factor = max(0.1, 1.0 - inventory_ratio)

bid_size = round(base_order_size * size_factor)
ask_size = round(base_order_size * size_factor)
```

**Defaults:**
- `base_order_size` = 10
- `max_inventory` = 500

### 2.7 Quoting Gates

At inventory extremes, we stop quoting one side entirely:

```python
def should_quote(inventory):
    should_bid = inventory < max_inventory   # Don't buy if max long
    should_ask = inventory > -max_inventory  # Don't sell if max short
    return (should_bid, should_ask)
```

---

## 3. Liquidity-Adaptive Adjustments

**Source:** `src/liquidity.py`, applied in `src/market_maker.py`

### 3.1 Liquidity Score

Measures current orderbook health (0 = empty, 1 = very liquid):

```
liquidity_score = 0.7 * depth_score + 0.3 * spread_score
```

**Depth Score** (log-scaled, saturates at ~1000 contracts):
```
total_depth = sum of top 5 bid levels + sum of top 5 ask levels
depth_score = min(1.0, ln(1 + total_depth) / ln(1001))
```

| Total Depth | depth_score |
|-------------|-------------|
| 0           | 0.00        |
| 10          | 0.35        |
| 100         | 0.67        |
| 500         | 0.90        |
| 1000        | 1.00        |

**Spread Score** (tighter = better):
```
spread_score = min(1.0, 2.0 / spread)  # spread in cents
```

| Spread | spread_score |
|--------|--------------|
| 1      | 1.00         |
| 2      | 1.00         |
| 5      | 0.40         |
| 10     | 0.20         |
| 20     | 0.10         |

### 3.2 Adaptive Parameters

Derived from liquidity score:

```
spread_multiplier = 0.5 + 2.5 * (1 - liquidity_score)
size_multiplier = 0.5 + 1.0 * (1 - liquidity_score)
```

| liquidity_score | spread_mult | size_mult |
|-----------------|-------------|-----------|
| 0.0 (empty)     | 3.0         | 1.5       |
| 0.25            | 2.375       | 1.25      |
| 0.5             | 1.75        | 1.0       |
| 0.75            | 1.125       | 0.75      |
| 1.0 (liquid)    | 0.5         | 0.5       |

**Rationale:**
- Low liquidity → wider spreads (pricing power), larger size (capture rare flow)
- High liquidity → tighter spreads (compete for queue), smaller size (reduce risk)

### 3.3 Application to Quotes

The Stoikov spread is scaled, then quotes recalculated around reservation:

```python
base_spread = ask_price - bid_price  # From Stoikov
adjusted_spread = base_spread * spread_multiplier
half_spread = int(adjusted_spread / 2)

new_bid = max(1, int(reservation_price - half_spread))
new_ask = min(99, int(reservation_price + half_spread))

# Prevent crossed quotes
if new_bid >= new_ask:
    new_bid = max(1, int(reservation_price) - 1)
    new_ask = min(99, int(reservation_price) + 1)

# Scale sizes
new_bid_size = clamp(int(bid_size * size_multiplier), 1, max_order_size)
new_ask_size = clamp(int(ask_size * size_multiplier), 1, max_order_size)
```

### 3.4 Empty Book Special Case

When orderbook is completely empty (`total_depth = 0`):

```python
if is_empty:
    bid_price = 1
    ask_price = 99
    bid_size = max_order_size  # Default: 100
    ask_size = max_order_size
```

**Rationale:** Maximum width (98 cent spread) with maximum size. Worst-case loss is 1 cent per contract, but potential gain from mispriced flow is substantial.

---

## 4. LIP Constraints

**Source:** `src/lip.py`, applied in `src/market_maker.py`

Only applied when an active LIP program exists for the market.

### 4.1 LIP Program Parameters

Fetched from Kalshi API (`/incentive_programs?status=active&type=liquidity`):

- `target_size`: Minimum contracts to qualify for points
- `discount_factor_bps`: Score penalty per tick away from best (basis points)
- `period_reward`: Total reward pool (informational)

### 4.2 Constraint Derivation

```python
min_size = target_size

# Max distance: ticks until score multiplier < 10%
# (1 - df)^n < 0.1
# n = log(0.1) / log(1 - df)
calculated_distance = int(log(0.1) / log(1 - discount_factor))

# Apply max_tick_cap for capital efficiency
max_distance = min(calculated_distance, max_tick_cap)  # Default: 20
```

**Why cap max_distance?** If the LIP discount factor is low (e.g., 5%), the formula allows quoting 50+ ticks away. This locks up capital in orders that will never fill and provide minimal LIP value.

| discount_factor | calculated | capped (max_tick_cap=20) |
|-----------------|------------|--------------------------|
| 0.50 (50%)      | 3 ticks    | 3 ticks                  |
| 0.30 (30%)      | 6 ticks    | 6 ticks                  |
| 0.10 (10%)      | 21 ticks   | 20 ticks                 |
| 0.05 (5%)       | 44 ticks   | 20 ticks                 |

### 4.3 Constraint Application

```python
# Enforce minimum size (to qualify for points)
bid_size = max(bid_size, min_size)
ask_size = max(ask_size, min_size)

# Cap at max order size
bid_size = min(bid_size, max_order_size)
ask_size = min(ask_size, max_order_size)

# Enforce max distance from best (to get meaningful score)
if best_bid is not None:
    min_allowed_bid = best_bid - max_distance
    bid_price = max(bid_price, min_allowed_bid, 1)

if best_ask is not None:
    max_allowed_ask = best_ask + max_distance
    ask_price = min(ask_price, max_allowed_ask, 99)

# Prevent crossed quotes
if bid_price >= ask_price:
    mid = (bid_price + ask_price) // 2
    bid_price = max(1, mid - 1)
    ask_price = min(99, mid + 1)
```

### 4.4 LIP Score Calculation (Tracking Only)

For monitoring purposes, we calculate expected score contribution:

```
distance_multiplier = (1 - discount_factor) ^ ticks_from_best

snapshot_score = 0
if bid_size >= target_size:
    snapshot_score += bid_size * bid_distance_multiplier
if ask_size >= target_size:
    snapshot_score += ask_size * ask_distance_multiplier
```

Snapshots recorded every ~1 second (every 10 ticks).

---

## 5. Execution Layer

**Source:** `src/execution.py`

### 5.1 Debouncing

Prevents excessive order amendments:

```python
def should_update(current, target):
    price_delta = abs(target.price - current.price)
    time_since_last = now - last_update_time

    # Update if price moved enough OR enough time passed
    return price_delta >= min_price_delta or time_since_last >= min_time_delta
```

**Defaults:**
- `min_price_delta` = 2 cents
- `min_time_delta` = 5 seconds

### 5.2 Order Diffing

Determines action needed to reach target state:

| Current State | Target State | Action |
|---------------|--------------|--------|
| No order      | Want order   | CREATE |
| Have order    | Don't want   | CANCEL |
| Have order    | Want different price/size | AMEND |
| Have order    | Same price/size | NO-OP |

### 5.3 Amend vs Cancel-Replace

We use Kalshi's amend endpoint (`PATCH /orders/{order_id}`) which modifies in-place without losing queue priority (when price unchanged).

---

## 6. Volatility Estimation

**Source:** `src/orderbook.py`

### 6.1 Tick Volatility

Updated on each mid-price change:

```python
price_change = abs(new_mid - last_mid)
tick_variance = price_change ** 2
```

### 6.2 Exponential Moving Average

Time-weighted EMA with configurable half-life:

```python
dt = time_since_last_update
alpha = 1 - exp(-ln(2) * dt / half_life)  # half_life default: 60s

ema_variance = alpha * tick_variance + (1 - alpha) * ema_variance
volatility = max(sqrt(ema_variance), volatility_floor)  # floor: 0.1
```

---

## 7. Complete Flow Example

**Scenario:** Mid = 50, Inventory = +100, Volatility = 1.5, Liquidity Score = 0.3, No LIP

### Step 1: Base Stoikov

```
# Time horizon (assuming >1 day to expiry)
T_minus_t = 1.0

# Reservation price
r = 50 - 100 * 0.05 * 1.5² * 1.0 = 50 - 11.25 = 38.75

# Spread (with floor)
δ_calculated = 0.05 * 1.5² * 1.0 + 40 * ln(1.0333) = 0.1125 + 1.31 = 1.42
δ = max(1.42, 2.0) = 2.0  # min_absolute_spread floor applied

bid_raw = 38.75 - 1.0 = 37.75 → 37
ask_raw = 38.75 + 1.0 = 39.75 → 39

inventory_ratio = 100/500 = 0.2
size_factor = 1.0 - 0.2 = 0.8
bid_size = round(10 * 0.8) = 8
ask_size = round(10 * 0.8) = 8
```

### Step 2: Liquidity Adjustment

```
spread_mult = 0.5 + 2.5 * (1 - 0.3) = 2.25
size_mult = 0.5 + 1.0 * (1 - 0.3) = 1.2

# Base spread from Stoikov = 39 - 37 = 2
adjusted_spread = 2 * 2.25 = 4.5
half_spread = int(4.5 / 2) = 2

new_bid = max(1, int(38.75 - 2)) = 36
new_ask = min(99, int(38.75 + 2)) = 40

new_bid_size = clamp(int(8 * 1.2), 1, 100) = 9
new_ask_size = clamp(int(8 * 1.2), 1, 100) = 9
```

### Step 3: Final Quotes

```
BID: 36 @ 9 contracts
ASK: 40 @ 9 contracts
```

---

## 8. Configuration Reference

From `config.toml`:

```toml
[strategy]
risk_aversion = 0.05           # γ - Risk aversion parameter
max_inventory = 500            # Position limit
max_order_size = 100           # Size cap (also fat-finger protection)
base_spread = 2.0              # Base spread in cents
min_absolute_spread = 2        # Minimum spread floor (safety net below Stoikov math)
quote_size = 10                # Default quote size
time_normalization_sec = 86400 # T-t normalization (1 day - expiries beyond this = T-t=1.0)
debounce_cents = 2             # Debounce: min cents to trigger update
debounce_seconds = 5.0         # Debounce: min seconds between updates

[volatility]
ema_halflife_sec = 60.0        # EMA half-life in seconds
min_volatility = 0.1           # Minimum volatility floor

[lip]
max_tick_cap = 20              # Never quote for LIP more than this many cents away
```

---

## 9. Liquidity Joining (Retreat Logic)

**Source:** `src/execution.py`

### 9.1 Problem: Adverse Selection

Being alone on a price level means you're first to get picked off by informed traders. We want to "hide" behind existing liquidity.

### 9.2 Dollar-Based Depth Threshold

```python
min_join_depth_dollars = 200.0  # Default: $200 of cumulative liquidity required
required_contracts = min_join_depth_dollars / (price / 100)
```

| Price | Required Contracts |
|-------|-------------------|
| 1c    | 20,000           |
| 10c   | 2,000            |
| 50c   | 400              |

### 9.3 Retreat Logic

If target price doesn't have sufficient depth, we "retreat" to a worse price:

```python
for offset in range(1, max_retreat + 1):  # max_retreat default: 15
    check_price = target_price - offset  # for bids
    if cumulative_depth(check_price) >= threshold:
        return check_price
```

Exception: If edge to mid >= `allow_solo_if_edge` (default: 7c), we quote alone.

---

## 10. Volatility Regime

**Source:** `src/volatility_regime.py`

### 10.1 Triggers (Enter High-Vol)

- Spread blows out: `spread >= 10c`
- Multiple fills: `3+ fills in 30 seconds`

### 10.2 Effects in High-Vol Mode

| Parameter | Normal | High-Vol |
|-----------|--------|----------|
| Tick interval | 100ms | 30ms |
| Depth multiplier | 1.0x | 3.0x (decaying) |

### 10.3 Depth Multiplier Decay

```python
# Decays with 60s half-life
multiplier = 1.0 + (3.0 - 1.0) * exp(-ln(2) * elapsed / 60)
```

After 60s: 2.0x, after 120s: 1.5x, etc.

### 10.4 Exit Conditions

Spread must stay ≤ 5c for 30 seconds to exit high-vol.

---

## 11. Reactive Quote Protection

**Source:** `src/execution.py` - `check_and_pull_exposed_quotes()`

Called on every orderbook update (not just tick loop) for faster reaction.

### 11.1 Exposure Detection

```python
# Bid exposed: we're improving the market
if our_bid > best_bid:
    cancel_bid()

# Ask exposed: we're improving the market
if our_ask < best_ask:
    cancel_ask()
```

### 11.2 Depth Thinning Detection

```python
# If depth at our level dropped below 50% of threshold
if cumulative_depth < min_depth // 2:
    cancel_quote()
```

---

## 12. Edge Cases

| Condition | Behavior |
|-----------|----------|
| Empty orderbook | Quote 1/99 with max_order_size |
| Mid price unavailable | Default to 50 |
| Inventory at +max | Stop bidding, continue asking |
| Inventory at -max | Stop asking, continue bidding |
| Quotes would cross | Force 1-tick spread around reservation |
| LIP min_size > max_order_size | Capped at max_order_size (won't fully qualify) |
| Quote becomes exposed | Immediately cancelled (reactive) |
| Depth thins at our level | Quote cancelled if < 50% threshold |
| Spread blows out | Enter high-vol: faster ticks, stricter depth |
