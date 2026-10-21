# Post-data research queue

Status: queued behind the source-access and synchronized-data gates. This document
freezes the order of work; it does not authorize live orders or capital.

Related documents:

- [External-fair attribution protocol](EXTERNAL_FAIR_PROTOCOL.md)
- [Exploratory fair and signal ledger](EXPLORATORY_FAIR_SIGNAL_LEDGER.md)
- [Strategy specification](STRATEGY_SPEC.md)
- [Data-access requirements](DATA_ACCESS.md)
- [Implementation status](STATUS.md)

The one-day 2026-08-28 proxy work is a preexisting-data plumbing and hypothesis
generation exercise, documented in the exploratory ledger. It does not satisfy this
entry gate, does not consume the final holdout, and cannot admit a feature into a
trading model. Families 1--4 and 7 have exploratory cards; families 5, 6, and 8 remain
data-gated. The formal queue and ordering below are unchanged.

## Entry gate

No signal testing begins until all of the following are true:

1. The exact Pyth return labels, final observation, correction policy, and Kalshi
   settlement transformation are documented and reproduce every usable settlement.
2. Futures, direct Pyth, Kalshi-proxied Pyth, and Kalshi book/trade records retain
   source, vendor/publication, UTC receive, monotonic receive, and sequence fields.
3. Historical gaps and forward disconnects are classified rather than silently
   omitted.
4. Causal features satisfy `max(input_receive_key) <= decision_receive_key`.
5. At least ten complete synchronized trading days pass the engineering data gates.
   Ten days are a technical burn-in minimum, not enough evidence for model selection;
   use at least 20 complete days for initial research and prefer 40 or more.
6. The research, validation, and final-holdout dates are committed to a run manifest
   before labels or strategy results for those periods are inspected.

## Frozen sample design

- The independent unit is a complete Kalshi 15-minute market window. Every quote,
  hypothetical fill, and terminal outcome from one window remains in one split.
- Splits are chronological: research, validation, then final holdout.
- Purge any row whose forward label crosses a split boundary. Apply an embargo at
  least as long as the maximum forward label/order lifetime; record the exact embargo
  in the manifest.
- Block confidence intervals by UTC trading day and report effective day/window counts.
- Keep the final holdout unread until the external-fair model, signal set, execution
  model, eligibility rules, and risk policy are frozen.
- The preferred first forward panel is a ten-complete-day engineering burn-in followed
  by twelve calendar weeks: six research weeks, three validation weeks, and three final
  holdout weeks. Continue collection after the panel closes. A shorter panel may be
  used for plumbing checks, never for a trading claim.

## Label contract to freeze before research

The exact values depend on written Pyth/Kalshi timestamp and settlement confirmation.
Once confirmed, version and freeze:

- consecutive-publication Pyth level change and log return;
- future raw-index log return from decision time to the contract's authoritative final
  Pyth observation;
- integrated realized variance, positive/negative semivariance, maximum move, range,
  and jump labels over each forward horizon;
- the exact unrounded final benchmark observation(s);
- the contract-specific rounding grid and comparison operator;
- binary settlement `Y` reconstructed from raw benchmark values and market metadata;
- correction/revision handling and the information vintage used by replay.

Never infer live availability from a historical bar timestamp. Historical request time
is provenance for the backfill, not the original live receive time.

## Baseline external fair

Build only this first:

1. session-specific futures-to-Pyth benchmark estimator;
2. time-of-day variance curve;
3. slow EWMA volatility state;
4. empirical standardized-return CDF;
5. exact strike rounding and settlement rule.

This produces `p0(t)`. Every later signal must improve held-out probability quality
and net trading EV relative to this same baseline.

## Incremental signal queue

Research one family at a time, in this fixed order:

1. fast and slow spot returns;
2. fast/slow range volatility and path efficiency;
3. futures volume, trade intensity, and price impact;
4. realized semivariance and jump indicators;
5. implied-volatility level and term structure;
6. implied skew and curvature;
7. cross-metal confirmation and residual returns;
8. scheduled-event and option-expiration effects.

Before fitting each family, state its economic hypothesis, sign, live availability,
normal time-of-day adjustment, expected horizon, and failure conditions. Measure its
incremental effect on conditional mean, variance, skew, and jump probability after
controlling for `p0`. Test stability by metal, session, remaining time, and volatility
regime.

Each family receives a versioned signal card containing formula, source/latency,
mechanism, decay, correlations, out-of-sample probability change, out-of-sample net-EV
change, and failure conditions. It enters the model only if both probability quality
and trading EV improve out of sample.

## COMEX option-expiry event study

Load a versioned official exchange calendar in `America/Chicago`; resolve holidays and
daylight-saving changes through the calendar. Never encode a fixed UTC expiry time.

For silver and gold, create expiry-day, seconds-to/from-expiry, 30/15/5/1-minute pre,
and 1/5/15/30-minute post features. Add distance to large strikes only when licensed,
reliable open interest and contract definitions are available. Measure futures volume,
range, spread, depth, and Pyth-basis changes around expiry.

Match each expiry window to non-expiry controls on local time, implied volatility,
recent realized volatility, known-event state, and volume regime. Classify effects as
variance-only, directional, strike attraction, breakaway, post-expiry reversal, or
basis-risk increase. Unstable effects become exclusions or uncertainty charges, not
alpha.

## Distribution components

Estimate conditional mean, integrated variance, residual distribution, and jump
probability separately. Start with regularized models and preserve probability
monotonicity in strike distance. Compare Gaussian, Student-t, state-conditioned
empirical CDF, and continuous-plus-jump distributions. The output is a coherent full
return distribution across strikes and remaining times.

## Model separation

- External fair may use futures, direct Pyth, options, volume, and historical returns.
  It may not use Kalshi price or flow.
- The execution model may use Kalshi book, trades, taker flow, queue state, and
  fill-conditioned residual toxicity.

Eligibility remains:

`external edge > expected selection + model uncertainty + inventory charge + fees`.

## Acceptance report for every version

Report Brier score, log loss, calibration by probability/time, improvement versus
`p0`, fair-value changes at 15/85, 25/75, and 50 cents, fill-conditioned settlement
P&L, queue-aware net cents per contract, daily block intervals, strongest-day removal,
and signal ablations. Forecast improvement alone is insufficient; net trading EV is
the acceptance test.

## Freeze and replay

After validation selection, freeze features, parameters, eligibility rules, and risk
settings. Run chronological queue-aware replay, latency/cancellation scenarios, and
the neutral, opportunistic-underdog, paired, and futures-led policies. Then run a live
shadow period. Any change creates a new research version and requires a new untouched
holdout.

The final attribution must show, in cents, the incremental contribution from spot
updating, volatility state, distribution shape, event effects, execution filtering,
fees, and settlement P&L.

## Immediate data-collection queue

### A. Access and semantic confirmation

1. Resolve the current Pyth HTTP 403 as credential or product entitlement, then confirm
   the exact Silver/Gold proprietary index entitlements, history depth, Lazer
   publication fields, and retention rights.
2. Obtain Databento `GLBX.MDP3` entitlement and written permission for non-display
   research, raw DBN retention, model development, and later alternative-venue use.
3. Resolve authenticated Kalshi WebSocket access with Support. Public REST history is
   useful but cannot replace book snapshots/deltas and local receive timestamps.
4. Obtain written Pyth/Kalshi confirmation of final-observation, candle-boundary,
   rounding, correction, and stale-value semantics.

### B. Historical backfill

1. Close the existing Kalshi coverage gaps and add durable completed-trade markers.
   Retain metadata, series/event rules, settlements, public trades, and one-minute BBO
   candles for every usable `KXSILVER15M` market.
2. Backfill exact `Metal.Index.SILVER/USD` history over the entire Kalshi overlap and as
   far back as licensed. Use high-frequency publication history for lead/lag work;
   one-minute candles are adequate only for slower variance/settlement research.
3. Add a Databento Historical/Batch adapter and archive native DBN plus normalized SI,
   SIL, and SIC MBO, trades, definitions, status, and statistics over the Kalshi
   overlap. Retain contract/roll identity; never splice a continuous future without a
   versioned roll rule.
4. Backfill longer, cheaper L1/trade bars for time-of-day variance and event research.
   Add GC and HG history for later cross-metal tests after the silver mapping passes.
5. Acquire versioned CME option definitions, official expiry calendars, settlement,
   volume/open-interest statistics, and—when licensed—historical quotes needed for IV,
   skew, and expiry matching. Do not infer expiry from a fixed UTC clock.

Historical Databento `ts_recv` is vendor capture time, not this system's live local
receive time. Historical Kalshi has no reconstructible native queue. Those limitations
must remain explicit in every historical report.

### C. Forward synchronized collection

Run continuously on one host and one monotonic clock:

- direct Pyth Lazer from all required redundant endpoints;
- Kalshi-proxied Pyth values for delivery-path comparison;
- Databento SI/SIL/SIC MBO, trades, definitions, status, and statistics;
- Kalshi book snapshots/deltas, trades, lifecycle, metadata, and settlements;
- GC and HG top-of-book/trades for future cross-metal confirmation;
- clock-health, reconnect, sequence-gap, and exclusion events;
- a versioned Central-Time event/expiry calendar known before each event.

The direct Pyth Lazer recorder and unified three-source coordinator are implemented.
The coordinator shares one in-process receive sequencer, requires all sources, stops
the panel when one child fails, classifies known transport/queue/ordering gaps, and
writes one parent quality report. It remains config-disabled until the access,
licensing, and timestamp-semantic gates pass. Separate processes may share a host
monotonic clock, but they do not share this deterministic ingestion sequence and are
therefore not the approved burn-in path.

### D. Daily validation and immutable snapshots

Every UTC day, export Parquet, rebuild deduplicated views, produce a manifest/quality
report, and verify:

- expected versus observed Kalshi windows;
- valid-book coverage and snapshot recovery;
- Pyth/CME/Kalshi coverage and staleness;
- source/vendor/local timestamp offsets and lag-sweep stability;
- CME sequence/reconnect gaps and contract mappings;
- exact target/final-observation/settlement reconstruction;
- duplicates, corrections, metadata changes, and fee changes;
- no feature or simulated decision using a later receive key.

Freeze immutable daily inputs and checksums. Research consumes only validated days;
excluded days remain stored with an explicit reason.

The coordinator report, content-addressed daily freeze, checksum verifier, and
ten-distinct-day readiness check are implemented. Operational collection, settlement
reconstruction, registry validation, and 10/10 passing days remain incomplete; the
current readiness result is 0/10.
