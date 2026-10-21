# Silver forward-shadow capture — 2026-08-31

## Decision

The collection path and exact settlement reconstruction are useful. The frozen
quote policy is **not eligible for live orders**. Three complete windows all lost
under strict-penetration shadow fills, and fills experienced negative external-fair
markouts at both one and five seconds.

Do not tune the slow distribution model to three outcomes. First fix Pyth continuity,
locked-book reconstruction, fill selection, and inventory accumulation; then replay
the frozen alternatives against this capture and collect a multi-day panel.

## Frozen run identity

- Collector: `20260901T002306.637377Z`
- Shadow session: `20260901T002306.683717Z`
- Settlement verification: `20260901T012034.434293Z`
- Baseline study: `ed1005f459a5beb2db07a99ada3e97feb17e03d9d669fae80a864b7da9412b89`
- Model: `tod-gaussian-exact-kalshi-pyth-v1`
- Verified report: `data/metals/derived/forward_shadow_report/report=2a54a139854ffeb597a345ba/report.json`
- Report ID: `2a54a139854ffeb597a345ba75f305c3a5224407af106936d8245b3ebb850a71`
- Report file SHA-256: `89b88e7de185c559bad619565efa815d69d95894c7fc89e1451dd4e8d4895145`

The operator stopped the originally two-hour bounded command after the third complete
window because the same failure repeated across opposite outcomes. The collector
manifest therefore says `cancelled`, and the shadow manifest says `failed`; both list
only `CancelledError`. Files were closed and flushed. The report preserves those
statuses, accepts no other error type, and verified every normalized record hash and
chain link before analysis.

## Coverage

The capture spans 3,171 seconds (52 minutes 51 seconds):

- 560 MiB raw Kalshi data.
- 410,396 normalized order-book rows from 409,843 deltas and nine snapshots.
- 4,533 trades.
- 3,156 exact `Metal.Index.SILVER/USD` updates.
- 9,868 causal fair decisions.
- 667 shadow quote placements and 567 cancellations.
- 105 strict-penetration fill events totaling 100.76 contracts.
- Five observed markets: three complete open-to-settlement windows, one opening
  partial window with settlement, and one unfinished closing partial window.

No exchange order ID and no non-shadow client order ID exists in the lifecycle data.
`live_orders=false` and `read_only=true` are frozen in the shadow manifest.

## Settlement verification

The correct live reconstruction uses the last Pyth update with source timestamp
**strictly less than** the contract close. The update stamped exactly at the boundary
belongs to the next interval for this purpose.

| Window | Full capture | Target | Strict-pre-close Pyth | Kalshi expiration | Result | Shadow P&L |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 7:15–7:30 PM CT | No | 66.719 | 66.921 | 66.921 | Yes | −$5.64800 |
| 7:30–7:45 PM CT | Yes | 66.921 | 66.725 | 66.725 | No | −$0.19000 |
| 7:45–8:00 PM CT | Yes | 66.725 | 66.828 | 66.828 | Yes | −$2.87866 |
| 8:00–8:15 PM CT | Yes | 66.828 | 66.643 | 66.643 | No | −$2.79408 |

All four reconstructed values and all four results agree exactly with Kalshi. The
three complete windows lost $5.86274 on 60.76 filled contracts, or about 9.65¢ per
contract. Including the settled opening partial window, the shadow result is
−$11.51074 on 98.76 contracts. This sample is far too small for an expected-P&L
estimate, but the execution failure is too large and consistent to ignore.

## What failed

### Fill selection

All 105 fills joined to their causal external fair and observed trade:

| Filled side | Quantity | Gross model edge | Net edge after fee | 1-second markout | 5-second markout |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 100.76 | 7.24¢ | 6.24¢ | −2.38¢ | −3.25¢ |
| Yes | 31.51 | 6.25¢ | 5.25¢ | −3.38¢ | −1.97¢ |
| No | 69.25 | 7.69¢ | 6.69¢ | −1.92¢ | −3.83¢ |

The frozen policy reserved only 1¢ for adverse selection. Observed mean loss was
already 2.38¢ after one second and 3.25¢ after five seconds. A static 1¢ charge is
not defensible. The apparent entry edge also depends on an uncalibrated three-window
fair, so it cannot be treated as real alpha.

### Directional inventory

Inventory repeatedly concentrated on the side that eventually lost:

- Opening partial, eventual Yes: 36 No versus 2 Yes.
- 7:30 window, eventual No: 1 Yes versus 0 No.
- 7:45 window, eventual Yes: 27.25 No versus 13.11 Yes.
- 8:00 window, eventual No: 15.40 Yes versus 4 No.

The configured inventory slope is only 0.001 dollars per contract (0.1¢), capped at
5¢. It did not stop repeated same-side acquisition. Inventory needs both a meaningful
dynamic price charge and a hard directional cap; price skew alone is insufficient.

### Feed and book operations

- Exact Pyth cadence was normally one second: p50 and p95 source gaps were both
  1,000 ms.
- Source-to-local receive lag was 381 ms p50 and 459 ms p95, uncorrected for the
  one-off NTP offset measurement. Clock telemetry was not active, so this is not yet
  a production latency claim.
- Two source gaps exceeded 1.5 seconds: a 16-second rollover gap when discovery
  briefly found no active market, and one two-second gap.
- There were zero Kalshi sequence-gap flags and zero decisions on an invalid book.
- 344 book rows were invalid, concentrated near settlement when complementary books
  locked. The engine failed closed, but the old reconstructor then ignored later
  deltas until a snapshot.
- Fair skips: 151 incomplete-metadata, 30 stale-Pyth, 29 terminal no-quote, and two
  initial no-Pyth observations.

Future-run fixes now keep a Pyth-only websocket alive when no contract is briefly
active, await canceled receive tasks, treat locked (but not crossed) books as
reconstructable, and retire quotes from rolled contracts. Focused and full metals
tests pass.

## Model interpretation

This was an execution and plumbing experiment, not a model-validation sample.

- Three complete binary outcomes cannot establish Brier improvement, calibration,
  signal alpha, or expected P&L.
- The online fair used the exact settlement index and the frozen time-of-day Gaussian
  variance curve. It did **not** yet include a causal slow EWMA update, SI futures
  benchmark updating, implied volatility, jumps, or signal-family additions.
- The negative one- and five-second markouts support the prior hypothesis that a
  faster benchmark updater and fill-selection model matter more immediately than
  adding more slow terminal signals.
- The fact that the displayed model edge remained positive while every settled
  window lost says to test calibration and execution separately. It does not justify
  fitting the distribution model to these three outcomes.

## Required next sequence

1. **Keep execution shadow-only.** No live-capital order path should be enabled from
   this result.
2. **Replay execution controls on this exact capture.** Compare the frozen policy
   with: a dynamic selection charge; a conservative initial 4¢ selection floor for
   research; same-side fill cooldown; hard per-side inventory caps; per-market loss
   stops; and earlier terminal quote exclusion. Treat 4¢ as an experiment, not a
   production constant.
3. **Measure quote age and cancellation latency.** Add queue-aware eligibility and
   latency scenarios before interpreting strict penetration as attainable fills.
4. **Run a fast-updater ablation.** The exact Kalshi-carried Pyth feed is suitable for
   settlement but only 1 Hz. The next valuable external input is SI trades/BBO from
   Databento, followed by the generic Pyth `Metal.XAG/USD` Pro feed if its retention
   approval is frozen. Databento live still requires the GLBX.MDP3 live license.
5. **Complete the slow baseline online.** Add a causal EWMA state and score it against
   time-of-day-only Gaussian, empirical CDF, Kalshi midpoint, and the frozen historical
   comparator panel.
6. **Collect days, not more signals.** Repeat fixed-duration captures after the
   operational fixes. Use at least 20 trading days before signal admission and keep
   whole Kalshi windows together in chronological splits.
7. **Then research signal families through natural targets.** Start with returns and
   path/range variance; keep SI microstructure in the fast updater/selection model;
   require out-of-sample probability improvement and queue-aware net EV before any
   signal enters quoting.

The immediate research question is no longer whether the plumbing works. It is how
much of the 2.38–3.25¢ short-horizon selection loss can be removed by faster external
updating, and how much remaining risk must be charged or excluded by the execution
policy.
