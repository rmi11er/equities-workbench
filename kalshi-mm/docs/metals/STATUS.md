# Metals implementation status

Updated 2026-09-05. The authoritative requirements are in
[STRATEGY_SPEC.md](STRATEGY_SPEC.md).

## 2026-09-05 accessible-edge evaluation

The current sparse Silver maker now has an offline accessible-edge evaluator and
a [prospective evaluation protocol](ACCESSIBLE_EDGE_PROTOCOL.md). It verifies the
unchanged capture before comparing a fixed cancellation-latency grid, reports
source/anchor coverage and candidate/quote/fill funnels, and separates model fair
movement from model-marked value and settlement P&L. Actual queue access, scalable
volume, and profit after operating costs remain unidentified.

The connected September 4 captures were weekend-closure engineering tests. In
the 60-second capture, 296 active XAG deliveries contained one source publication;
none of 1,123 active-window timer samples had fresh XAG or a usable fast anchor.
The bot submitted no quotes. Exact replay and successful stale-data rejection do
not establish strategy edge. No policy threshold or model was changed, no SI
subscription was purchased, and no live or scheduled order process was added.

Next: obtain fresh open-market coverage, evaluate the unchanged rule under the
frozen protocol, and review accessibility before expanding research spend. See
the [maker runbook](MAKER_FORWARD_RUNBOOK.md) for commands and current trial-control
requirements. Earlier paired-maker and taking results below remain historical
comparisons, not the track record of this new sparse maker.

## 2026-09-02 extended Silver forward capture

An 11.44-hour read-only run retained 1,528,455 book records, 13,796 trades,
10,058 exact-index updates, 32,695 fair decisions, and 33,314 long-vol candidate
rows across 45 observed windows. All hash chains and candidate lineage checks pass;
the controlled stop placed no live orders.

The settlement backfill identified and corrected a report bug: stale last-pre-close
index values had been promoted to labels after transport gaps. The corrected report
requires an update no more than 1.5 seconds before the boundary. All 11 eligible
labels exactly match Kalshi; only 10 belong to full captured windows. Stale values can
no longer affect provisional outcomes or settlement P&L.

The frozen strict long-vol policy selected one independent 15-cent YES entry. It won
84.10 cents after the entry fee, but executable liquidation was -3.70 cents at both
one and five seconds. This is one entry from one UTC day and has no identifiable
day-block interval. The decision remains **continue forward collection, no live
capital**. See [the extended capture report](FORWARD_CAPTURE_2026-09-02.md).

A post-fix 2026-09-03 smoke added one exchange-labeled strict entry: 11-cent YES lost
11.69 cents and again had negative 1/5-second liquidation marks. The frozen combined
sample is now two entries on two UTC dates, one win, +72.41 cents total, with a
two-point day-block interval from -11.69 to +84.10 cents. It remains radically
underpowered and does not change the no-go.

## 2026-08-31 live Silver shadow capture

A 52-minute read-only forward-shadow capture produced three complete Silver windows,
410,396 normalized book rows, 4,533 trades, 3,156 exact settlement-index updates, and
105 strict-penetration fill events. All four observed strict-pre-close settlement
values and results matched the subsequent Kalshi backfill. The three complete windows
lost $5.86274 in shadow settlement P&L, while filled-side external-fair markouts
averaged −2.38¢ at one second and −3.25¢ at five seconds. The quote policy remains
ineligible for live orders. See
[`LIVE_CAPTURE_2026-08-31.md`](LIVE_CAPTURE_2026-08-31.md) and the verified report at
`data/metals/derived/forward_shadow_report/report=2a54a139854ffeb597a345ba/report.json`.

## Current decision

The project is **NO-GO for trading**. Development, historical research, read-only
collection, deterministic replay, and shadow simulation are allowed. No metals module
may submit, amend, or cancel an exchange order.

The first offline historical screen also **preliminarily rejects the unfiltered S1
favorite-fade rule**. The public candle backfill completed 1,994 market requests with
zero failures and produced 28,888 one-minute candles. A chronological 70/30 study
retained 4,138 eligible signals across 1,887 markets. On the 567-market held-out set:

- 1,239 signals had an 18.40% underdog win rate versus a 21.26% average midpoint.
- The passive-touch proxy lost 3.76 cents per fill at an assumed zero maker fee.
- The stricter one-cent-penetration proxy lost 3.85 cents per fill at zero maker fee.
- The aggressive-next-open benchmark lost 4.82 cents per fill after the configured
  unrounded taker-fee formula.
- Every required time bucket and every S1 core price bucket was negative.

A trade-timestamped held-out markout follow-up reinforces the terminal result. For the
touch proxy, average one-contract markout P&L was -0.05 cents at 1 second, +0.10 cents
at 5 seconds, -0.36 cents at 30 seconds, -0.25 cents at 1 minute, -1.71 cents at 5
minutes, and -3.31 cents at settlement. The one-cent penetration proxy was immediately
negative: -1.25, -1.24, -1.69, -1.55, -2.69, and -4.10 cents at the same horizons.
These are last-trade proxies rather than midpoint marks. Held-out average passive entry
prices were 20.49 cents for touch and 20.82 cents for penetration; both had a median
entry of 22 cents.

Train and held-out signs agree. This is sufficient to lower the priority of the naive
S1 rule, but it does not test a future independently specified CME/Pyth fair-value
filter and cannot replace the required Q0/Q1/Q2 event replay.

## External access

| Source | Decision | Current blocker |
| --- | --- | --- |
| Kalshi | Production read-only API and WebSocket | Credentials are configured and working; a complete synchronized collection still depends on the remaining Pyth/live-clock choices |
| Pyth | Exact `Metal.Index.SILVER/USD` feed | Direct trial access still returns HTTP 403 for feed 3154, but the authenticated Kalshi WebSocket now demonstrably forwards the exact live index; generic direct `Metal.XAG/USD` feed 345 is available at 200 ms as a future fast predictor |
| CME | Databento `GLBX.MDP3` | Historical access and SI/SIL/SIC mappings passed; the live probe authenticated but explicitly requires a live license. Standard is currently listed at $199/month; no subscription was purchased |
| Clock | Local Chrony tracking | Collector and fail-closed gate are implemented; the current Mac host has no `chronyc`, so a synchronized collection host with an active, healthy Chrony daemon is still required |

The preflights wrote no credentials to disk. The generic Pyth checks establish that the
configured credential is valid, so the exact-index failure is now classified as an
entitlement gap rather than an unknown credential problem. Generic XAG/XAU may support
exploratory external-fair work, but cannot certify exact Kalshi settlement labels. CME
live collection remains disabled until the bounded live probe and written
non-display/retention/use confirmation are complete. Databento historical exploration
is enabled only through the narrow immutable research-assumption gate and remains
cost-capped per request; the ignored local config still has no full provider-approval
path.

The immutable-readiness check currently reports **0 of 10 required synchronized UTC
days and 0 of 1,000 required exact settlement agreements**. A refreshed public Kalshi
metadata/results pass now identifies 1,991 completed Silver windows and retains each
window's primary and secondary settlement rules; the archive also contains 1,713,517
trades and 28,888 one-minute contract candles. The offline exact audit classifies all
1,991 windows as excluded because exact Pyth coverage and reviewed semantics evidence
are absent. There is still no historical Pyth/CME/book/clock overlap, so none of those
legacy dates can be promoted into synchronized research days. There is now one day of
Kalshi-label/generic-Pyth/CME historical overlap, but it remains proxy-only and lacks
exact Pyth, Kalshi book, and clock evidence.

The verified audit now also drives an offline, content-addressed exact-history plan.
Plan `781818064d3e8336595b2f4018844c936e9fd236f7d6d88fec5565941d83c670`
classifies four malformed contracts, retains 1,987 structurally eligible windows, and
deduplicates their opening/settlement requirements into 1,998 one-minute Pyth ranges.
No range was downloaded: the guarded executor remains unused while Pyth returns 403.

Direct CME connectivity is deferred unless the measured S3 lead is too short for
Databento delivery and the incremental economics justify direct-feed cost.

## Exploratory settlement-proxy pilot

A research-only 2026-08-28 Silver pilot now compares 96 complete Kalshi windows with
the full-day Databento history and the trial `Metal.XAG/USD` one-minute feed. The
generic Pyth feed covers 83 windows before its Friday session ends. Its
preceding-minute candle alignment is strongly preferred over treating each timestamp
as a boundary label:

- Preceding-minute Pyth changes have 0.99794 correlation with Kalshi-reported changes,
  96.39% direction agreement, and 98.75% agreement outside half-cent Kalshi moves.
- Comparing the generic Pyth close directly with the Kalshi target reproduces 97.59%
  of the 83 covered outcomes. Median absolute level error is $0.003 and mean absolute
  level error is $0.0071, but the maximum is $0.1205 and only 27.71% of displayed
  three-decimal levels match after diagnostic half-up rounding.
- Independent SIZ6 last-trade changes cover 37 windows with 0.99746 correlation and
  97.30% direction agreement. SILZ6 covers 50 with 0.99681 correlation and 94.00%
  direction agreement; away from half-cent moves those rates are 100% and 97.92%.
- All 96 labels in this day are structurally consistent; no pilot window was
  quarantined.

These results pass the small-cohort exploratory expectation check and identify the
preceding-minute interpretation for baseline research. They do not prove the exact
settlement source or rounding rule: the generic feed is permanently marked
`settlement_eligible=false`, and the formal settlement gate remains closed pending
exact `Metal.Index.*` observations and reviewed semantics.

## Exploratory volatility fair and signal cards

A research-only basic volatility fair is now frozen for the proxy panel. It estimates
a shrunk UTC time-of-day variance curve, slow deseasonalized EWMA state, Gaussian
distribution, and standardized empirical CDF from 1,874 prior exchange-reported
Silver returns. On 1,245 decisions across the 83 covered 2026-08-28 windows, Gaussian
`p0` scored 0.160065 Brier and 0.473279 log loss, slightly better than the empirical
CDF's 0.161766 and 0.475334. The Gaussian fair therefore remains the unchanged
exploratory baseline.

Five families were then fitted one signal at a time, using 55 development windows, a
one-window embargo, and 27 test windows. None of 15 signals passed the frozen expected
sign plus lower-Brier plus lower-log-loss gate, and queue-aware net EV is not measurable
from this panel. This is underpowered, inconclusive evidence—not a negative finding:
the 405 test rows contain only 27 outcomes from one UTC day, and terminal Brier score
is not the correct first target for every feature. No signal was added and no joint
model was created. The tested families are spot returns; range volatility/path efficiency; COMEX volume,
intensity, and price impact; semivariance/jumps; and XAU confirmation/XAG residuals.
The XAU trial pull added 1,260 research-only candles without a Databento purchase.

The common-row comparator scorecard ranks the TOD/slow-EWMA Gaussian fair first on
the same one-day holdout: 0.157724 Brier versus 0.159509 for flat-volatility Gaussian,
0.160192 for the empirical CDF, 0.166035 for the historical Kalshi midpoint, 0.183910
for a current-side development rate, and 0.250000 for constant 50%. Kalshi midpoint
does better in the final 1--2 minutes. This is descriptive, not an alpha result:
historical candle publication latency is unknown, spread and fees are excluded, and
only one UTC-day block exists.

The natural-target runner keeps all 18 feature cards underpowered and inconclusive on
that synchronized day. Separately, a causal expanding-window variance diagnostic
scores 1,490 prior 15-minute returns over 21 UTC days. Time-of-day variance improves
on flat variance under both MSE and QLIKE with 95% day-block intervals below zero;
slow EWMA has no stable incremental improvement over time of day. This is provisional
component evidence, not terminal-probability or trading-model admission.

## Forward shadow

The first exchange-isolated live forward run is operational using Kalshi books,
trades, metadata, and the exact Pyth Silver index carried by Kalshi. A 30-second smoke
run produced 94 scored fair decisions, seven simulated placements, seven simulated
cancellations, four non-fill touches, and zero strict-penetration fills. Every exchange
order ID was null and both manifests completed with `live_orders=false`.

The forward policy loads the frozen baseline without refitting, uses time-of-day-only
Gaussian variance, and charges uncertainty, provisional adverse selection, fees, and
inventory before producing paired counterfactual bids. See the
[forward-shadow runbook](FORWARD_SHADOW_RUNBOOK.md). SI and direct 200 ms XAG remain
future fast-updater inputs; Databento live requires the separate subscription noted
above.

Implied-volatility level/term structure, skew/curvature, and the option-expiry event
study remain untested because the archive has no historical option panel or adequate
multi-day expiry controls. The complete results and artifact IDs are in the
[exploratory fair and signal ledger](EXPLORATORY_FAIR_SIGNAL_LEDGER.md). This work did
not access the formal final holdout and does not change the trading no-go.
The exact requirements and current blockers for the next 20--40-day panel are in the
[synchronized panel handoff](SYNCHRONIZED_PANEL_HANDOFF.md).
The research-only interfaces and source barriers for the slow terminal distribution,
fast SI updater, fill-selection model, and fee-aware quote composition are frozen in
the [three-model architecture](THREE_MODEL_ARCHITECTURE.md).

## Risk profiles

The owner-approved **shadow stress envelope** is:

- Hypothetical bankroll range: $5,000–$10,000.
- Maximum counterfactual order: 250 contracts.
- Maximum counterfactual net outcome exposure: 1,000 contracts per market.
- Maximum aggregate worst-case loss: $3,000 across unsettled positions and all
  potentially filled resting shadow orders.
- Maximum worst-case loss: $1,000 per market.
- Daily stop: $3,000, evaluated as realized P&L plus current worst-case loss.
- Execution mode: shadow; live orders disabled.

These are simulation boundaries, not pilot approval. The strategy specification's
controlled-pilot defaults—10 contracts per order, 100 net contracts per market, $250
aggregate worst-case loss, and a $250 daily stop—remain a physically separate future
profile and may only be reconsidered after all prior rollout gates pass.

## Implemented foundation

- A pinned Python 3.11 project environment (`.python-version`, `pyproject.toml`, and
  `uv.lock`) with a locked CLI/test invocation.
- Immutable hash-chained raw and normalized JSONL storage, plus exclusive native DBN
  archives as the authoritative field-lossless CME source.
- Atomic manifests with running/completed/failed state, sanitized errors, counts,
  output paths, Git state, and a deterministic public-config hash.
- An immutable, content-addressed provider-approval policy that independently binds
  reviewed license scope, timestamp semantics, and per-operation entitlement evidence.
  All Pyth and Databento collection paths verify it before credential use, download,
  billable submission, or output creation and record its approval ID in run manifests.
- Canonical deduplicated DuckDB views plus explicit all-run provenance views.
- Databento normalization for MBO, trades, definitions, status, and statistics,
  retaining vendor timestamps, sequence/channel/order identifiers, raw fixed-point
  values, and local receipt clocks.
- Databento recorder orchestration and CLI with no exchange execution path.
- Bounded, cost-preflighted Databento Historical/Batch ingestion with explicit
  billable-download acknowledgement, maximum-cost refusal, resumable checkpoints,
  native DBN authority, and output hash verification.
- A separate Databento live-entitlement preflight that opens only an explicitly
  acknowledged, at-most-30-second MBO snapshot subscription, discards every record,
  sanitizes provider failures, and reports success only after observing a record.
- Non-blocking Databento callback handoff, off-thread normalization, symbol/control
  record retention, feed-close monitoring, and classified reconnect gaps.
- Redundant direct Pyth Lazer WebSocket collection with secret-safe bearer auth,
  bounded queues, exact raw messages, normalized prices, hash chaining, endpoint
  failover, and classified transport gaps.
- A separate explicitly acknowledged Pyth Lazer entitlement preflight that probes all
  configured WebSocket endpoints concurrently, validates an update for the configured
  feed, discards every source message, and retains only sanitized per-endpoint status.
- Bounded exact Pyth `fixed_rate@200ms/price/range` archives with numeric feed IDs,
  inclusive microsecond bounds, opaque-cursor pagination, exact successful bodies,
  freshness/carry-forward fields, resumable content verification, and terminal
  secret-safe manifests.
- An offline settlement-audit-to-Pyth planner that freezes and deduplicates exact
  preceding-minute boundary ranges, classifies non-actionable windows, and a separate
  sequential executor with explicit network acknowledgement, per-run boundary caps,
  per-range page caps, resumable verification, and terminal manifests.
- A read-only synchronized coordinator that shares one receive sequencer across
  Kalshi, Pyth Lazer, Databento, and a required Chrony sidecar; supervises all children
  as one run; rejects missing component stamps and clock discontinuities; and emits a
  unified fail-closed daily quality report.
- A standalone, bounded Chrony collector and sanitized preflight that read local daemon
  state without changing the system clock, preserve exact probe evidence, and expose
  signed local-minus-reference offsets, synchronization state, stratum, and leap status.
- Content-addressed daily validation snapshots that re-evaluate every data gate and
  require one content-addressed measured-report owner, reproduce its evidence from all
  declared source manifests, and verify the quality report, provider-approval files,
  run manifests, and all declared outputs before admitting a UTC day to research. A
  separate deterministic measurement pass verifies relevant
  JSONL hash chains and derives time-weighted source staleness, settlement, clock,
  metadata, fee, and incentive evidence. The readiness CLI independently requires ten
  distinct passing dates and 1,000 exact agreeing settlements; repeated snapshots for
  one date cannot inflate either count.
- A local reviewed-policy materializer that verifies canonical registry hashes,
  requires fee/incentive vintages to be known by measurement start, persists
  hash-chained registry rows, and produces the completed manifest required by the
  daily measurement pass.
- An immutable exact-settlement audit that refuses TradingView candles, verifies
  completed exact-history requests, requires every request artifact to be bound to a
  completed collection manifest and acquisition-time provider approval, proves full
  preceding-minute coverage, selects the last publication strictly before each
  boundary, preserves Kalshi rule text and Pyth source age, and emits classified
  exclusions plus a standard daily-input manifest.
- A content-addressed settlement-proxy pilot that verifies every input JSONL hash
  chain, selects COMEX contracts without outcome information, tests both generic Pyth
  candle alignments, quarantines inconsistent labels, reports chronological fit and
  threshold metrics, and can never certify the exact settlement gate.
- A research-only volatility baseline with shrunk time-of-day variance, slow EWMA
  state, Gaussian and empirical probabilities, monotonicity checks, and immutable
  one-minute fair rows over a provisional proxy cohort.
- Independent, versioned signal-card runners for spot returns, range/path state,
  futures activity, semivariance/jumps, and cross-metal returns. Each uses an atomic
  window split, embargo, intercept-only control, and fail-closed probability and
  trading-EV admission rules; none creates a joint model.
- A common-row baseline scorecard covering constant, current-side, flat-volatility,
  TOD/EWMA, empirical-CDF, and descriptive Kalshi-midpoint forecasts, with horizon,
  moneyness, session, calibration, and day-block reporting.
- Natural-target signal diagnostics plus a 21-day causal variance walk-forward, and
  immutable research-only model boundaries separating the slow external fair, fast
  SI update, fill selection, uncertainty, inventory, and side-specific fee charges.
- A bounded live-data forward-shadow runner that consumes the exact Silver index from
  Kalshi, emits causal baseline fairs, manages counterfactual paired quote lifecycle,
  and attributes only strict-penetration fills without any exchange order route.
- Shadow order lifecycle, fills, cancels, binary terminal P&L, conservative resting
  order and fee exposure, risk limits, and kill-switch blocking.
- Fail-closed Phase 1 data-gate evaluation. Unmeasured coverage cannot pass.
- Resumable public one-minute Kalshi candle backfill with exact raw response storage,
  normalized bid/ask OHLC, request-level completion records, and current/historical
  endpoint routing.
- Preliminary S1 candle feasibility study with chronological 70/30 market splits,
  settlement calibration, aggressive-next-open/passive-touch/passive-penetration
  proxies, fee/slippage sensitivity, daily block bootstrap, and concentration and
  drawdown diagnostics. It remains explicitly non-Q0/Q1/Q2 and no-go for capital.

The locked metals test suite currently passes 525 tests.

## Required gate sequence

1. Complete source access, license, and timestamp-semantic confirmation.
2. Record at least 10 complete synchronized trading days.
3. Reconstruct opening targets and settlements for at least 1,000/all usable windows
   with 100% agreement.
4. Pass Kalshi, Pyth, CME, clock, gap-classification, metadata, and fee data gates.
5. Complete R1 capacity research and Phase 2 fair-value research.
6. Complete the deterministic Q0/Q1/Q2 event simulator and economic gates.
7. Run a frozen build in shadow for at least five complete trading days with replay
   agreement and tested kill switches.

Only a separate controlled-pilot review can authorize real money after those steps.
