# Silver forward-shadow runbook

Status: operational research path. It records live data and counterfactual orders but
cannot submit, amend, or cancel an exchange order.

## Confirmed live inputs

The authenticated Kalshi WebSocket delivers the active `KXSILVER15M` book, trades,
lifecycle events, and `pyth_value` updates for `Metal.Index.SILVER/USD`. A bounded
2026-08-31 capture observed 19 distinct exact-index values over 18 seconds with mean
source-to-local receipt lag of about 410 ms, plus 1,951 book deltas and six trades.
The market metadata confirmed the exact Pyth symbol, opening target, three-decimal
rounding, and preceding-minute settlement rule.

Pyth also publishes live `Metal.XAG/USD` as Pro feed ID 345 with a minimum
`fixed_rate@200ms` channel. It is a useful future fast predictor, but it is not the
Kalshi settlement index. Direct feed 3154 remains outside the current trial
entitlement; this no longer prevents forward shadowing because Kalshi supplies the
exact live index in its authenticated stream.

The Databento live probe authenticated but returned `A live data license is required
to access GLBX.MDP3`. Historical credits do not by themselves activate live CME data.
The current Standard subscription is listed at $199/month, and signup credits may be
applied toward the first subscription month.

## Frozen policy

`run-forward-shadow` loads the immutable baseline model without refitting and uses:

- exact live `Metal.Index.SILVER/USD` from Kalshi;
- the supported UTC time-of-day variance curve;
- a Gaussian terminal distribution and exact market target/rounding metadata;
- two cents of model uncertainty, one cent of provisional adverse selection, and one
  cent of fee reserve per contract by default;
- signed inventory skew capped at five cents;
- paired YES/NO bids only when the native price grid and current book permit positive
  conservative edge;
- strict trade penetration for simulated fills. A trade at the shadow price is a
  touch, not a fill, because queue position is unknown.

Incremental slow EWMA is excluded because the 21-day diagnostic did not support it.
SI and direct XAG are not yet used in the fair; they will enter only as fast causal
updaters after their live streams are available.

## Bounded command

Load the ignored credential environment and run a finite session:

```bash
set -a
source ~/.codex/.env
set +a
.venv/bin/python -m src.metals.cli --config configs/metals.toml \
  run-forward-shadow \
  --baseline-model data/metals/derived/baseline_fair/run=baseline-fair-ed1005f459a5beb2db07a99a/model.json \
  --duration-seconds 3600 \
  --allow-live-shadow-data
```

The acknowledgement enables live **read-only data**, not live trading. Configuration
loading refuses any execution mode other than `shadow` and refuses
`allow_live_orders=true`.

## Durable outputs

Each run creates a live collector manifest and a linked shadow manifest. The shadow
run persists:

- `forward_fair_decision`: exact fair inputs, staleness, probability, book hash,
  quote ceilings, actions, and frozen settings;
- `shadow_decision`: price, quantity, fee reserve, risk result, and state hash;
- `order_lifecycle`: counterfactual submit/rest/cancel/fill transitions, always with
  a null exchange order ID;
- `shadow_fill_attribution`: observed trade, strict penetration rule, quantity, and
  simulated fee for every attributed fill.

The first 30-second smoke run completed as collector
`20260901T001700.858879Z` and shadow run `20260901T001700.912690Z`. It produced 94
scored fair decisions, seven quote placements, seven cancellations, four touches not
filled, and zero strict-penetration fills. All manifests completed without errors and
reported `live_orders=false`.

## Next expansion

1. Continue full-session Kalshi-only shadows. The 2026-09-02 run validated storage
   and candidate lineage but exposed exact-index transport gaps; require explicit
   boundary freshness and exchange-recorded settlement labels in every report.
2. Activate Databento CME live access if the $199 Standard subscription is approved;
   rerun the discard-only entitlement probe before retaining data.
3. Add SI BBO/trades and direct Pyth XAG feed 345 under the same receive sequencer.
4. Fit the SI/XAG-to-exact-index fast updater and fill-selection charges from the
   accumulated event panel.
5. Replay the identical quote policy with measured latency and queue assumptions.

Build the exchange-verified reports after each run:

```bash
.venv/bin/python -m src.metals.forward_shadow_report \
  --root data/metals --collector-run-id <collector> --shadow-run-id <shadow> \
  --settlement-run-id <targeted-kalshi-backfill>
.venv/bin/python -m src.metals.forward_long_vol_report \
  --root data/metals --collector-run-id <collector> --shadow-run-id <shadow> \
  --settlement-run-id <targeted-kalshi-backfill>
.venv/bin/python -m src.metals.combined_forward_long_vol_report \
  --root data/metals --report <forward-long-vol-report> \
  --report <next-forward-long-vol-report>
```

No live-capital transition is implied by this runbook.
