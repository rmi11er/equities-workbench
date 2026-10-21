# Three-model research architecture

Status: **research only**. These interfaces cannot create orders and every output
has `trading_eligible = false`.

All adjustments use one unit: **YES probability dollars per contract**. Thus a
two-cent charge is represented as `Decimal("0.02")`, not `2`.

## Boundaries

1. **Slow terminal distribution.** Estimates `P(B_T >= K)` from external Pyth,
   SI futures, options, calendar, historical-return, and cross-metal inputs. It
   rejects Kalshi prices, flow, and queue state. Its output includes the complete
   distribution family, log-return location and scale, jump probability, strike,
   benchmark, remaining time, rounding rule, and the below-strike probability.
   The below-strike mass must complement the inclusive YES probability.

2. **Fast SI-to-Pyth updater.** Starts from a slow external snapshot and maps a
   causal SI price change into benchmark and probability changes. The delta
   approximation is usable only inside a frozen basis-point limit and while the
   result remains inside `[0, 1]`. Otherwise it emits a full-reprice requirement
   and supplies no usable probability.

3. **Fill-selection model.** Produces separate, nonnegative adverse-selection
   charges for a YES bid and YES offer. It accepts Kalshi book/trade and local
   queue/quote state, plus causally available SI/Pyth momentum and recent external-fair
   movement. These external predictors can change only the selection charge; the fill
   model cannot change the external fair itself. Slow options, calendar, and historical
   distribution inputs remain outside this boundary.

Every input carries a process-local receive key. A record fails closed if any input
arrived after its decision key.

## Relationship to `external_fair.py`

This boundary layer extends the existing external-fair research primitives; it
does not replace them:

- `external_fair.ReceiveKey` is the shared causal key used by all three records.
- `PriceEvent`, `build_replicator_rows`, `fit_replicator`, and
  `ReplicatorModel` estimate and validate the SI-to-Pyth relationship. A validated
  replicator beta supplies the fast updater's
  `benchmark_dollars_per_si_dollar`; the boundary record then limits when its
  probability delta may be used.
- `external_fair.FairSnapshot` is the compact causal probability used by existing
  markout diagnostics. `SlowTerminalDistribution` and `FastBenchmarkUpdate` add
  source ownership, distribution metadata, research-design provenance, and the
  full-reprice interlock before a snapshot reaches quote research.
- `FillAttribution` and `summarize_attribution` remain the post-fill decomposition
  of quoted edge, stale fair, post-fill external movement, and residual Kalshi
  flow. `FillSelectionEstimate` is the separate pre-quote prediction of that
  adverse-selection cost.

The remaining integration gaps are intentional: there is no adapter from these
records to the existing shadow-order simulator, no online model-fitting path, and
no order-entry path. A later research integration should convert validated
replicator output into fast-update parameters, convert usable external snapshots
into `FairSnapshot` for attribution, and retain the full-reprice interlock.

## Quote composition

For signed YES inventory charge `I = gamma * q`, nonnegative uncertainty `U`,
and side-specific all-in fee charges `F_bid` and `F_offer`:

```text
raw bid ceiling = external probability - bid adverse selection - U - F_bid - I
raw offer floor = external probability + offer adverse selection + U + F_offer - I
```

Positive YES inventory lowers both bounds; negative YES inventory raises both.
Fee inputs use the same YES-probability-dollar unit and must be computed from the
applicable Kalshi fee schedule before any executable replay comparison.
The research result retains the raw values, clips displayed bounds to `[0, 1]`,
and suppresses a side whose raw bound is infeasible. It contains an empty immutable
`order_intents` tuple and is not an execution instruction.
