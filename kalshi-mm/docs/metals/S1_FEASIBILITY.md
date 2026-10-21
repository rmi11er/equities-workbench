# Preliminary S1 candle feasibility study

This study answers a narrow question: does the passive favorite-fade hypothesis show
enough historical signal to justify continued shadow-data work? It can reject S1. It
cannot approve trading, because one-minute candles do not expose queue position,
displayed depth, side-specific aggressive volume, partial fills, or receive-time
latency.

## Inputs and selection

- Settled silver 15-minute markets in the local canonical DuckDB catalog.
- Kalshi one-minute Yes bid/ask OHLC candles and final Yes/No settlement.
- A reliable two-sided candle-close quote with `0 < bid <= ask < 1`.
- The underdog midpoint in the S1 core range, 7–31 cents.
- No new signal with less than 120 seconds remaining.
- At most the earliest eligible signal in each market and required time bucket:
  10–15, 5–10, and 2–5 minutes remaining.
- The next candle must begin exactly one minute later; gaps are not bridged.

Markets are split chronologically by market open time: the first 70% are train and the
last 30% are held out. There is no random row split.

## Fill sensitivities

Each signal buys one underdog contract and holds it to settlement.

- `aggressive_next_open_proxy`: buy at the next candle's derived underdog ask open and
  apply the configured unrounded taker-fee formula. This is a taker benchmark, not S1.
- `passive_touch_proxy`: rest at the current derived underdog bid and count a fill only
  if the next candle's derived underdog ask reaches the bid and aggregate volume is
  positive.
- `passive_penetration_proxy`: the same resting bid, but require the next derived ask
  to penetrate it by the configured amount, initially one cent.

The two passive rules are candle sensitivities, not Q0/Q1/Q2 queue models. Aggregate
candle volume does not prove that a specific resting order filled. The report also
reprices held-out results at 0, 0.5, and 1 cent maker fees and at 0, 0.5, and 1 cent of
additional aggressive-entry slippage.

## Trade-anchored markout smile

The supplemental markout layer uses historical trade timestamps to resolve 1-second,
5-second, and 30-second horizons that candles cannot identify. It anchors a passive
fill to the first qualifying opposing-taker trade within 60 seconds:

- A resting Yes bid requires a No taker trading Yes at or below the bid.
- A resting No bid requires a Yes taker trading No at or below the bid.
- The penetration proxy requires the trade to pass through the bid by the configured
  amount.

At 1 second, 5 seconds, 30 seconds, 1 minute, and 5 minutes after that anchor, the mark
is the latest quantity-weighted trade price at or before the horizon. If the market has
already closed, the settlement payout is used. `15m/end` is always the settlement
payout. These are last-trade markouts, not midpoint or microprice markouts. The report
includes average and 95th-percentile trade-mark age so stale carried-forward trades
remain visible.

The study does not label `p(1-p)` as vega. True digital-option vega requires a fair
value model with the underlying, target, time, and volatility inputs that are not yet
available. For context only, the report supplies the market-implied Bernoulli payout
variance `p(1-p)` and its one-contract terminal P&L standard deviation. The primary
risk representation remains terminal outcome exposure and stress loss.

## Run

```bash
PYTHONPATH=. python3 -m src.metals.cli --config configs/metals.toml \
  backfill-kalshi-candles --metal silver
PYTHONPATH=. python3 -m src.metals.cli --config configs/metals.toml export-parquet
PYTHONPATH=. python3 -m src.metals.cli --config configs/metals.toml \
  run-s1-feasibility --metal silver
```

The run directory contains a concise Markdown report, full JSON statistics, selected
signals, and all execution-proxy rows. The JSON includes calibration by price and time
bucket, chronological train/test results, fee and slippage sensitivity, UTC day-block
bootstrap intervals, drawdown, profit concentration, the trade-anchored markout smile,
and its fill and staleness diagnostics. `markouts.csv` contains every horizon row.

## Interpretation

The decision remains `NO_GO_FOR_TRADING` regardless of a positive proxy result. Final
acceptance still requires synchronized live/shadow data, a versioned effective fee
schedule, CME/Pyth fair value, latency scenarios measured from telemetry, and the
event-driven Q0/Q1/Q2 simulator required by the strategy specification. Candle data
cannot support a capacity estimate, so one-contract P&L is not scaled to the approved
shadow stress limits.

The broader `run-toxicity-diagnostics` command should be used for generic two-sided
BBO conditional-fill research. It includes the full quoted-price partition, four
Yes-side favorite/underdog action labels, paired-leg fills, exit-cost bounds,
concentration cuts, and the entry-to-settlement P&L waterfall. It remains separate
from fair-value strategy evaluation.
