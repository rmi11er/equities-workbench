# Fast-feed attribution and high-frequency signal memo — 2026-08-31 capture

## Bottom line

The best declared-latency estimate is that the slow external spot path explains
**51.3% of the observed one-second adverse markout** and **39.5% of the observed
five-second adverse markout** in this capture. The counterfactual uses direct Pyth
XAG at 100 ms transport, Databento SI at 25 ms provider-to-host latency, and a 100 ms
cancel path. It would have suppressed 32 of 105 fill events and 31.25 of 100.76
contracts.

This is a run-specific counterfactual association, not a population causal estimate.
There were only 105 strict-penetration shadow fills across three complete and two
partial Kalshi windows. Historical Pyth and Databento retain source/provider
timestamps, not this host's missing live receive timestamps, so those two transport
latencies are explicit scenarios. Kalshi book, trade, and exact-index receive times
were recorded live.

The result is nevertheless operationally decisive: the one-hertz exact settlement
feed was too slow to protect quotes by itself. A direct XAG updater plus SI is the
first execution improvement to implement. The residual loss remains large enough
that faster data alone does not make the policy eligible for live capital.

## Data used

- Live-recorded Kalshi: 410,396 book rows, 4,533 trades, and 3,156 exact
  `Metal.Index.SILVER/USD` updates over 3,171 seconds.
- Pyth Pro history: 15,853 unique overlapping `Metal.XAG/USD` feed-345 observations
  on the `fixed_rate@200ms` channel. The full bounded archive contains 15,903 rows.
- Databento `GLBX.MDP3`: definitions, MBO, and trades from midnight through the end
  of the shadow run. The selected outright was SIZ6 by request-period volume.
- Reconstructed SI: 42,897 overlap BBO changes and 1,611 SIZ6 trades.
- Shadow execution: 105 strict-penetration fill events totaling 100.76 contracts.

The Databento request cost **$0.056852879375**, below the operator-approved $0.25
ceiling. Request ID:
`fb4ba7b96e711bca2a693e0e0eac39af95d014cdc02963e349dc3062d85a43af`.
The MBO reconstruction processed 503,171 records with zero duplicate adds, unknown
cancels, unknown modifies, or crossed reconstructed events.

## Why the old feed bled

| Stream | Observed/source cadence | Availability used in the replay |
|---|---:|---:|
| Kalshi-carried exact Pyth index | 1,000 ms p50 and p95 | 381 ms p50, 459 ms p95 source-to-host |
| Direct Pyth XAG | 200 ms p50 and p95 | source time + 100 ms scenario |
| SI BBO changes | 0.204 ms p50, 416 ms p95 | Databento receive time + 25 ms scenario |
| Kalshi book changes | 2.54 ms p50, 195 ms p95 | recorded local receive time |
| Kalshi trades | 92 ms p50, 2.99 s p95 | recorded local receive time |

The exact index normally supplied only one new value per second and arrived about
0.38–0.46 seconds after its source timestamp. A quote could therefore rest through
several direct-XAG updates and many SI events before the pricing engine saw a new
settlement-index value. The direct feeds often invalidated the frozen quote before
the trade that generated the shadow fill.

The anchored updater never substitutes generic XAG or SI for settlement:

```text
estimated exact index = last causally received exact index
                      + (current fast proxy - proxy at the exact anchor)
```

Each candidate quote is cancelled only when the proxy-updated side fair, less the
quote's original frozen charges, falls below the resting price. It is not allowed to
create a quote or make a quote more aggressive.

## Loss attribution

Original filled-side markout was −2.3828¢ per contract after one second and
−3.2462¢ after five seconds, or −$2.3771 and −$3.2384 in total.

| Counterfactual, 100 ms cancel | Avoided fills | Avoided quantity | Share of 1 s loss | Share of 5 s loss |
|---|---:|---:|---:|---:|
| Direct Pyth XAG only, 100 ms transport | 21 | 21.00 | 45.2% | 26.6% |
| SI only, 25 ms provider-to-host | 21 | 20.25 | 27.9% | 20.6% |
| Cancel if either feed invalidates quote | 32 | 31.25 | **51.3%** | **39.5%** |

The rows overlap and must not be added. In the union replay, 10 fills were caught by
both feeds, 11 only by Pyth, and 11 only by SI. At one second, direct Pyth alone
accounts for 45.2 percentage points and SI adds 6.2 points beyond it. At five
seconds, Pyth alone accounts for 26.6 points and SI adds 12.9 points beyond it.

Under the primary union scenario:

- Slow-feed-avoidable one-second loss: $1.2198, or 1.2227¢ per original contract.
- Residual one-second loss: $1.1573, or −1.1600¢ per original contract.
- Slow-feed-avoidable five-second loss: $1.2783, or 1.2814¢ per original contract.
- Residual five-second loss: $1.9600, or −1.9647¢ per original contract.
- Conditional on the retained fills, markout remains −1.6649¢ at one second and
  −2.8198¢ at five seconds.

### Cancellation-latency sensitivity

Pyth is fixed at 100 ms and SI at 25 ms below.

| Cancel path | Avoided fills | Share of 1 s loss | Residual 1 s | Share of 5 s loss | Residual 5 s |
|---:|---:|---:|---:|---:|---:|
| 50 ms | 40 | 62.4% | −0.897¢ | 53.4% | −1.513¢ |
| 100 ms | 32 | 51.3% | −1.160¢ | 39.5% | −1.965¢ |
| 250 ms | 16 | 14.0% | −2.049¢ | 14.7% | −2.769¢ |

The median warning lead was only 141 ms in the primary scenario. Transport and
cancel/ack telemetry therefore matter as much as nominal source frequency. Scenario
cells are not perfectly monotone because a 200 ms Pyth grid, one-second exact
anchors, irregular quote placements, and discrete fills interact.

## Higher-frequency signal research

The synchronized study used 5,571 causal feature rows on the 200 ms Pyth grid where
the active Kalshi book was valid. Targets were sampled non-overlapping at each
horizon and correlations were also recomputed by observed 15-minute window. This is
natural-target research, not a terminal-probability or trading-P&L test.

### Results worth carrying forward

| Signal and natural target | Horizon | Pearson r | Top-minus-bottom target | Same-sign windows |
|---|---:|---:|---:|---:|
| 1 s Pyth XAG return → next exact-index return | 1 s | 0.770 | 2.115 bp | 5 / 5 |
| 1 s SI return → next exact-index return | 1 s | 0.320 | 0.928 bp | 5 / 5 |
| 1 s SI return → next exact-index return | 5 s | 0.505 | 3.163 bp | 4 / 4 eligible |
| 200 ms SI return → next XAG return | 1 s | 0.602 | 1.593 bp | 5 / 5 |
| 1 s SI return → next XAG return | 1 s | 0.546 | 1.681 bp | 5 / 5 |
| 1 s Pyth absolute path → next absolute XAG return | 1 s | 0.304 | 0.589 bp | 4 / 5 |
| 1 s SI trade intensity → next absolute XAG return | 1 s | 0.320 | 0.834 bp | 4 / 5 |

The first row is partly a mechanical catch-up relationship between two related Pyth
metal feeds, which is exactly why it is useful as a fast updater but not evidence of
standalone economic alpha. SI price momentum is the strongest independent candidate:
it leads XAG and the exact index with the expected sign in every observed window.
Short-run path/range and SI activity also forecast their correct variance targets.

At the observed fills, Pyth 200 ms movement oriented toward the filled passive side
had a 0.305 correlation with one-second markout and a 4.49¢ top-versus-bottom
quintile spread. SI 200 ms movement had a 0.329 correlation with five-second markout
and a 3.49¢ spread. These reinforce the cancellation result, but the fill sample is
only 104 feature-matched events.

### Results not ready to use

- SI top-book imbalance and microprice displacement were weak at one second
  (`r=0.067` and `r=0.055`) and reversed by five seconds.
- The one-second jump ratio had essentially no variance predictiveness
  (`r=-0.025` at the one-second target).
- Kalshi top imbalance had a modest relationship to its own next one-second midpoint
  (`r=0.128`). Raw Kalshi taker flow had `r=0.380`, but rank correlation was only
  0.024 and the sign held in only three of five windows, indicating an outlier-driven
  result.
- Fill-conditioned Kalshi flow showed a notable five-second reversal/toxicity
  pattern (`r=-0.459`, −5.76¢ quintile spread), but only across 104 fills. Preserve
  it as an execution-model candidate; do not put it into the external fair.
- No high-frequency IV, skew, gold cross-metal, or expiry study was possible because
  those synchronized inputs were not present in this capture.

Every generated signal card has null out-of-sample probability and net-EV fields and
the disposition `exploratory_only_no_model_admission`.

## Morning decisions and next actions

No action is required to preserve this capture; the raw, normalized, and derived
artifacts are already durable.

1. **Keep orders shadow-only.** The fast feeds explain a large fraction of the bleed,
   but the primary replay still leaves −1.16¢ per original contract at one second
   and −1.96¢ at five seconds. Strict-penetration replay also omits real queue and
   cancel races.
2. **Approve the fast-updater implementation order:** direct live Pyth XAG first,
   anchored to the exact Kalshi-carried index and used only for cancel/suppress;
   then SI when live CME entitlement exists. The first version should not use either
   feed to quote more aggressively.
3. **Choose the Databento operating mode.** Existing credits covered this historical
   request, but live `GLBX.MDP3` previously required the separate Standard live-data
   subscription. Without it, keep collecting Kalshi plus live Pyth and backfill SI
   after Databento's roughly eight-hour historical delay. With it, add SI to the
   real-time cancel path.
4. **Instrument measured latency.** Persist proxy publish, host receive, fair-change,
   cancel-decision, cancel-send, and cancel-ack timestamps. Set an initial shadow
   objective of p95 decision-to-cancel-send below 100 ms; the replay shows that 250
   ms gives away most of the benefit.
5. **Run a frozen multi-day shadow panel.** Record at least 20 trading days, retaining
   whole Kalshi windows and day blocks. Each morning, backfill the matching SI MBO
   and trades if live SI is unavailable.
6. **Evaluate in the same order:** anchored Pyth-only cancel, SI incremental cancel,
   price-momentum updater, range/activity variance state, then Kalshi
   fill-selection features. Require day-held-out probability calibration and
   queue-aware net EV before changing quote aggressiveness.
7. **Leave these deprioritized:** SI displayed imbalance, jump ratio, and omnibus
   signal fitting. Add synchronized options/IV, cross-metal gold, and exact expiry
   calendars only after the core event panel is accumulating reliably.

## Reproducible artifacts

- Combined attribution:
  `data/metals/derived/combined_fast_feed_counterfactual/report=e1769107c4eb6e190219016b/report.json`
- Direct-Pyth replay:
  `data/metals/derived/fast_feed_counterfactual/report=d24ed031bba42df248391a26/report.json`
- SI replay:
  `data/metals/derived/si_fast_feed_counterfactual/report=3e26c9603c4a53754de8e99a/report.json`
- High-frequency signal cards:
  `data/metals/derived/high_frequency_signal_research/report=fabddd4bfb31492df7c8d75c/report.json`
- Databento archive manifest:
  `data/metals/manifests/run=databento-history-stream-fb4ba7b96e711bca/manifest.json`

All reports are research-only and set `trading_eligible=false`.
