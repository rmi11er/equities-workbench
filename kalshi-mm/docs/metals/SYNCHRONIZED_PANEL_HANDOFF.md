# Synchronized research-panel handoff

Status: collection specification; no live orders and no new data purchase authorized.

## Required panel

The next inferential panel must contain at least 20 complete trading days; 40 or more
is preferred. The existing ten-day gate remains only an engineering burn-in check and
is not enough for signal selection.

One host and one monotonic receive sequence must capture:

- exact Silver benchmark publications used by the Kalshi settlement contract;
- SI top-of-book/depth and trades, with source, vendor, and local receive timestamps;
- Kalshi book snapshots/deltas, trades, lifecycle events, metadata, and settlement;
- clock-health, reconnect, sequence-gap, exclusion, fee, and registry events;
- GC/XAU data when cross-metal research is enabled.

Every feature must satisfy `maximum_input_receive_key <= decision_receive_key`.
Historical event timestamps and later backfill receipt timestamps cannot substitute
for original live receive order.

## Current readiness

| Component | Current state | What prevents the panel |
|---|---|---|
| Kalshi | Authenticated read-only access works; public historical trades/candles exist | No complete synchronized book/event run with the other sources |
| Pyth generic | Trial `Metal.XAG/USD` and `Metal.XAU/USD` history works | Generic feeds cannot certify exact settlement and historical candles lack live receive latency |
| Pyth exact | Exact Silver index feed ID and guarded collectors are implemented | Current trial entitlement returns HTTP 403 for the exact index |
| Databento | Cost-capped historical SI/SIL/SIC retrieval works | A synchronized live SI book/trade run and its operating approval are not established |
| Clock | Fail-closed collector is implemented | The current Mac has no healthy `chronyc`; approved collection requires a synchronized host |

The project can prepare models and offline diagnostics now, but these four facts mean
that fast-update and fill-selection claims must remain unmeasured.

## Daily acceptance unit

A day enters research only when its immutable snapshot proves:

1. Expected Kalshi windows and complete lifecycle coverage.
2. Exact benchmark coverage and settlement reconstruction.
3. SI book/trade coverage, stable contract mapping, and classified sequence gaps.
4. Causal receive-key joins and healthy clock measurements.
5. Known fee, market-rule, and event-calendar vintages.
6. Explicit exclusion of disconnects, stale intervals, corrections, and ambiguous
   settlement records.

Confidence intervals and model comparisons are blocked by trading day. Repeated
minute forecasts and multiple windows from one day do not increase the effective day
count.

## Research routing after collection

- Slow terminal distribution: rolling/expanding day splits; signed returns, realized
  variance/range, tails, and jumps first; settlement probabilities second.
- Fast benchmark updater: consecutive Pyth publications and intervening SI moves;
  mandatory causal lag sweep and unchanged-Pyth benchmark.
- Fill selection: fill-conditioned external-fair markouts at 100 ms, 250 ms, 1 s,
  5 s, and settlement; Kalshi flow never enters the external fair.
- Pricing comparison: constant, moneyness-only, flat-volatility, TOD/EWMA,
  empirical-CDF, Kalshi midpoint, and executable bid/ask after fees on common rows.

The final holdout remains sealed until all three model definitions, feature eligibility,
quote composition, and execution replay policy are frozen.

## Operational handoff

No user action is needed for the existing-data implementation work. Starting the
actual panel requires external state that code cannot create: exact Pyth entitlement,
a usable synchronized SI book/trade service, and a Chrony-capable collection host.
Once those are present, run a bounded one-hour synchronized dry run, validate its
immutable daily evidence, then begin the 20-day minimum collection without changing
schemas or model definitions mid-panel.
