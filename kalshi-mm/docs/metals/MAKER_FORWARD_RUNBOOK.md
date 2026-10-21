# Silver maker forward experiment

This is a read-only, two-arm experiment. It records market data and simulates
post-only orders in the shared shadow risk engine. There is no exchange order
submission, funded account operation, or provider purchase in this runner.

The manual arm uses a prospectively chosen volatility multiplier; the comparator
uses 1.00. Both arms share pause, expiry, data-health, price-grid, fee, and execution
rules. A multiplier of 1.20 scales standard deviation by 1.20 and variance by 1.44.
Missing, invalid, expired, or non-increasing control revisions suppress entries
and request cancellation. Restore a faulty control with a new, higher revision.

Each arm simulates a $25 balance, one contract per order, one net contract per
market, $1 market/aggregate worst-case exposure, and a $5 session loss limit.
The policy further allows one unsettled position, stops replenishment after any
partial fill, and stops entries with fewer than two minutes remaining. The arms
are independent counterfactuals, not two orders competing for real liquidity.

## Inputs and execution assumptions

The frozen Silver model provides time-of-day remaining variance and Gaussian
digital probabilities. Empirical probabilities are recorded for comparison.
Direct Pyth `Metal.XAG/USD`, feed 345, is the initial fast protection input. It
can veto or reduce a slow-index bid ceiling; it cannot justify raising the bid.
The exact Kalshi-carried `Metal.Index.SILVER/USD` remains the settlement anchor.
The current XAG return mapping assumes beta one and is not an estimated SI/index
mapping. Databento live SI is a later separately tested integration.

Fast prices use `feedUpdateTimestamp`, not the newer stream delivery timestamp.
An exact index observation anchors to the most recent previously received XAG
observation at or before the exact observation's source time. The configured
maximum anchor gap and source-age limits must also pass.

The live XAG path retains the existing `pyth/live_lazer` provider gate. The
checked-in trial configuration has no approved live-provider artifact attached.
Preflight reports this explicitly. The formal route uses an existing, valid
approval artifact through the established provider configuration workflow.
An additional maker-only route accepts an explicit, expiring operator trial
research permit, described below. It preserves
`provider_permission_confirmed=false`; an operator assumption and a Pyth API key
are not represented as formal provider permission.

The initial latency scenario is 100 ms to submit and 100 ms to cancel. Pending
submissions reserve exposure. Post-only validity is checked against the observed
book at simulated arrival. A cancellation requested before an order arrives
waits for its simulated acknowledgment, then begins its full cancel delay.
Trades remain eligible during cancellation, and prints predating actual
simulated arrival cannot fill the quote. Arrival is never backdated across an
unobserved interval.

Strict trade penetration is a declared fill scenario. Touches are counted but
do not fill. Queue priority is unknown, and crossing a quote in the public tape
does not prove that our hypothetical order would fill. No missing trades or
fills are manufactured through connection gaps.

A 50 ms watchdog runs independently of market activity. It observes control
changes, checks staleness, and advances due cancellations. Source disconnects,
sequence gaps, and planned subscription reconnects immediately invalidate books
and request cancellations. Recovery requires new snapshots. At normal shutdown,
sources stay active for up to two seconds while pending cancellations drain;
unresolved exposure is retained in the report.

A gap while an order is pending or resting creates unknown execution exposure.
The affected arm is permanently frozen for that run; reconnecting or renewing a
control does not automatically resume quoting. Review the report's
`uncertain_exposure` and execution qualification fields before starting another
session. Successful deterministic replay cannot remove uncertainty about real
fills during a missing interval and never grants live trading eligibility.

Exchange source times and host receive times must also be comparable. A trade
timestamp ahead of local receipt latches `clock_domain_fault`, cancels orders,
and suppresses further quotes for that run. Missing trade times with an order
outstanding freeze the affected arm. Do not repair evidence by clamping source
times to receipt or shifting clocks based on observed trade delays. Check host
clock synchronization independently; the runner never changes the system clock.
Monotonic-versus-wall elapsed checks catch clock jumps, not an existing constant
offset. A local preflight pass alone does not certify source clock alignment.

## Prepare controls and check local prerequisites

From the repository root, create an explicitly bounded operator setting. Use a
higher `--version` whenever the destination already contains a revision.

```sh
uv run --locked python -m src.metals.maker_forward control \
  --control-file data/metals/controls/maker-silver.json \
  --version 1 --volatility-multiplier 1.00 \
  --ttl-seconds 3600 --reason 'Initial baseline engineering session'
```

The following example declares a 1-cent maker fee reserve per filled contract.
Choose the reserve explicitly for the account/series being studied. This is a
scenario charge, not automatic fee reconciliation or a claim that actual maker
fees equal one cent.

```sh
uv run --locked python -m src.metals.maker_forward preflight \
  --config configs/metals.toml \
  --baseline-model data/metals/derived/baseline_fair/run=baseline-fair-ed1005f459a5beb2db07a99a/model.json \
  --pyth-config configs/pyth_xag_trial.toml \
  --maker-fee-per-contract 0.01
```

Preflight checks model validity/hash, credentials presence, configuration, and
the existing provider gate. It makes no network requests and emits no credential
values. A non-ready result exits with status 2. Load credentials through the
existing local environment workflow before running; do not put them in command
arguments, controls, or artifacts.

## Explicit operator trial route

The project owner has accepted the bounded research use case under their trial.
The supplied content-hashed record preserves that statement, the exact XAG feed,
research-only/no-orders scope, a collection deadline, and a retention review
deadline. Its hash identifies the record; it is not a provider signature.
No command flag creates or silently upgrades provider approval.

Pass the existing permit path explicitly to both preflight and run. For example,
the owner-accepted September 4 artifact can be used while its recorded deadline
is current:

```sh
uv run --locked python -m src.metals.maker_forward preflight \
  --config configs/metals.toml \
  --baseline-model data/metals/derived/baseline_fair/run=baseline-fair-ed1005f459a5beb2db07a99a/model.json \
  --pyth-config configs/pyth_xag_trial.toml \
  --trial-research-permit data/metals/policies/maker_trial/trial-20260904-owner-accepted.json \
  --maker-fee-per-contract 0.01
```

Preflight can return `ready=true` with `trial_research_gate=true` and
`pyth_provider_gate=false`: the selected route is explicitly the operator trial.
The default formal route remains blocked when its evidence is absent. This
permit cannot be attached to the exact-only diagnostic, another feed, a general
Pyth collector, a CME source, or a live-order configuration.

```sh
uv run --locked python -m src.metals.maker_forward run \
  --config configs/metals.toml \
  --baseline-model data/metals/derived/baseline_fair/run=baseline-fair-ed1005f459a5beb2db07a99a/model.json \
  --pyth-config configs/pyth_xag_trial.toml \
  --trial-research-permit data/metals/policies/maker_trial/trial-20260904-owner-accepted.json \
  --maker-fee-per-contract 0.01 \
  --control-file data/metals/controls/maker-silver.json \
  --duration-seconds 60 --allow-live-shadow-data
```

Trial captures are capped at 900 seconds each and must fit before both the local
collection deadline and any known provider trial expiry, including three seconds
for shutdown. Before any network request, the runner validates the requested
duration. The watchdog and Pyth connection/message boundaries enforce expiry
during operation. Trial Pyth collection also requires an attached maker runner
and has its own hard duration bound. Expiry stops collection and preserves any
unresolved execution exposure in the failed run report.

The Kalshi collector, both arm manifests, and the Pyth source manifest preserve
typed `research_provenance` or collector trial fields with the permit path/hash
and `provider_permission_confirmed=false`. The report also carries the complete
record under `trial_research`; historical replay preserves that classification
without attempting new collection or requiring the old collection window to
remain open. The retention review deadline calls for review, not automatic
deletion or indefinite retention.

## Bounded capture

Once preflight passes, this starts a 15-minute read-only capture:

```sh
uv run --locked python -m src.metals.maker_forward run \
  --config configs/metals.toml \
  --baseline-model data/metals/derived/baseline_fair/run=baseline-fair-ed1005f459a5beb2db07a99a/model.json \
  --pyth-config configs/pyth_xag_trial.toml \
  --maker-fee-per-contract 0.01 \
  --control-file data/metals/controls/maker-silver.json \
  --duration-seconds 900 --allow-live-shadow-data
```

For a short Kalshi-only engineering smoke test, replace `--pyth-config ...` with
`--exact-only-diagnostic` and use `--duration-seconds 60`. Preflight accepts the
same diagnostic flag. This explicitly bypasses fast-input use for the diagnostic
run only; its results do not qualify the primary fast-protected experiment.

Pause an active session by atomically writing a higher revision:

```sh
uv run --locked python -m src.metals.maker_forward control \
  --control-file data/metals/controls/maker-silver.json \
  --version 2 --volatility-multiplier 1.00 --paused \
  --ttl-seconds 3600 --reason 'Operator pause'
```

Resuming requires another higher revision without `--paused`. Pausing cancels
quotes subject to simulated cancel latency; it does not erase positions or
unobserved execution risk. The baseline comparator shares operator pauses.

## Replay and inspect results

The runner prints its report path and writes
`data/metals/derived/maker_forward/run=<collector_run_id>/report.json`. The report
links the absolute frozen-model path/hash, input segment paths/counts, Kalshi and
Pyth source run IDs, arm run IDs, settings, lifecycle/fill diagnostics, markouts,
settled P&L, and unsettled or outstanding exposure.

Every `maker_input` is hash-chained and fsynced before the engine consumes it.
Source events retain their original receive timestamps and shared ingestion
sequence; watchdog/control/stop events use that same sequencer. Maker book
inputs retain top bids and the full source-book hash; the underlying recorder
retains full raw snapshots and deltas. Raw-source durability is unchanged.

```sh
uv run --locked python -m src.metals.maker_replay \
  --report data/metals/derived/maker_forward/run=COLLECTOR_RUN_ID/report.json \
  --output-root data/metals/maker-replay-verification
```

Replay checks model hash, every input segment's hash chain, input completeness,
decision digest, and final state without network or credentials. Include the
capture's stop event and drain observations. Closed markets are refreshed at a
bounded rate during capture; settlements arriving after capture remain unresolved
and must not be counted as losses or wins.

Before interpreting P&L, require successful replay, no unexplained clock fault,
adequate fast-anchor coverage, explicit cancellation-race/gap accounting, and
useful quote/fill counts across complete sessions. Compare manual settings as
actually chosen with the unchanged comparator. A tiny live maker pilot still
requires a separate order-routing implementation and the previously scoped pilot
authorization; this collector cannot place live orders.

## Accessible-edge scorecard

The [prospective evaluation protocol](ACCESSIBLE_EDGE_PROTOCOL.md) freezes the
next research milestone: persistent pricing disagreement, obtainable flow after
selection, and sufficient value after costs. It does not change the strategy,
open the formal holdout, authorize purchases/orders, or schedule a collector.

Run the offline evaluator on each completed maker capture:

```sh
uv run --locked python -m src.metals.maker_evaluation \
  --report data/metals/derived/maker_forward/run=COLLECTOR_RUN_ID/report.json \
  --output-root data/metals/derived \
  --session-label normal
```

Choose the observed session context explicitly: `normal`, `reopen`, `holiday`,
`closed`, `diagnostic`, or `unspecified`. This label does not override source
freshness. Keep reopening/holiday sessions separate from normal-session results.

The evaluator verifies the original model, input hashes/count, decision digest,
and complete final state before publishing a content-addressed report. It then
replays a fixed 100 ms submission / 100, 250, and 500 ms cancellation grid,
deduplicating a cell identical to the original capture settings. Primary and
sensitivity results remain separate; the evaluator never selects a winner.

Reports include active-window timer-sampled source/anchor coverage, source
publications versus repeated deliveries, decision/quote/fill funnels, observed
persistence episodes, simulated quote lifecycles, cancellation-in-flight fills,
and fill-cohort overlap across latency scenarios. Initial model edge, later fair
movement, model marks relative to execution cost, bid marks, and terminal P&L
are distinct fields with explicit missing-data coverage. The observation-rate
denominator includes shutdown drain because fills can occur during cancellation.
Actual queue access, scalable volume, measured exchange fees, and operating-cost
profit remain unknown. No report grants live trading eligibility.

The evaluator uses existing data only. It neither downloads SI nor substitutes
historical SI timestamps for live receipt evidence. Its output inherits the
source research classification and retention-review deadline; replay does not
renew a collection permit or extend retention rights.

For the next open-market capture, create a current manual control and use a
current bounded trial research record. The September 4 one-hour record has
expired for new collection and must not be reused for Sunday. Run a short
freshness check first, then a bounded 900-second capture if source/anchor/clock
health passes. No automatic future run is installed by this workflow.

## Engineering qualification, September 4, 2026 (Central)

Implemented and verified with 643 passing metals tests, including delayed
lifecycle, source-aligned XAG, manual controls, failure handling, and replay.
The final 30-second read-only exact-index diagnostic is recorded at
`data/metals/derived/maker_forward/run=20260905T001652.502170Z/report.json`
(run IDs use UTC). All 1,129 policy inputs and the final state replayed exactly.
It had no simulated fills or outstanding orders and activated the clock-alignment guard. This is plumbing
and guard verification, not evidence of profitable execution.

The earlier diagnostic observed 34 of 35 exchange trade timestamps ahead of
local receipt (median across all 35: about 25 ms). An independent, read-only
SNTP probe indicated a positive reference-minus-host offset; no system clock
change or timestamp compensation was made. The final run verifies that this
condition now suppresses quoting for the remainder of the session.

Before a meaningful primary forward session:

1. Independently verify and resolve host/exchange clock alignment, then repeat
   the short diagnostic. Do not loosen the guard to obtain fills.
2. Attach valid existing Pyth provider-approval evidence for retained live XAG
   collection. Local configuration and credentials passed; this gate did not.
3. Run the bounded XAG-protected experiment, verify deterministic replay, and
   inspect anchor coverage, opportunity persistence, cancel races, and unresolved
   exposure before interpreting simulated returns.

The earlier pre-guard diagnostic was replayed against its then-current code;
use the final run above with the current implementation. A failed sandbox-only
connection attempt was also retained as a failed run, not counted as a capture.

### Clock repair follow-up, September 4, 2026 (Central)

With the user's authorization, checked macOS network-time settings through the
system administrator prompt and resynchronized using Apple's SNTP client.
Automatic network time was already enabled with `time.apple.com`; it remains
enabled. Apple and Cloudflare probes initially estimated the host about 55 ms
slow. After resynchronization their point estimates were approximately -0.4 ms
and +2.1 ms, respectively, with network uncertainty of roughly 34-40 ms. These
probes do not certify sub-millisecond accuracy.

The subsequent 30-second exact-only diagnostic is
`data/metals/derived/maker_forward/run=20260905T002301.926295Z/report.json`.
All 2,650 policy inputs and the final state replayed exactly. All 33 trade
timestamps preceded local receipt: source-to-receive lag ranged from about
15 to 29 ms, with a 21 ms median. Both clock fault flags stayed false, and there
were no unusable trade timestamps, fills, or outstanding orders. The startup
subscription gap remains explicitly reported; this is not clean execution or
profitability qualification. No strategy timing guards were loosened and no
source timestamps were shifted. Clock alignment passed this follow-up smoke;
the Pyth account-evidence gate remains unresolved.

### First XAG-connected trial smoke, September 4, 2026 (Central)

The owner subsequently accepted proceeding under the trial. Implemented the
explicit, expiring research route described above; the general provider gate
remains unchanged. All 704 metals tests pass. The acceptance record permits
collection for one hour and calls for retention review after 24 hours; it does
not invent the provider's trial expiry or claim provider-confirmed permission.

Completed the first 60-second XAG-protected capture:

- Report: `data/metals/derived/maker_forward/run=20260905T010242.091091Z/report.json`
- Detailed source health: `health.json` beside that report.
- Retention inventory: `retention-review.json` beside that report.
- Replay: all 2,640 inputs and the final state reproduced exactly.
- Pyth: three connections, 887 normalized updates, no reconnects or transport
  disconnects. All updates were explicitly closed/carried-forward, repeating the
  source publication at `2026-09-04T20:59:59.800Z`, over four hours old.
- Each arm rejected 219 active-market evaluations for stale XAG. There were no
  quotes, fills, positions, or outstanding orders. Fast pricing and execution
  remain unqualified; stream-message frequency is not source-price freshness.
- The exact index remained fresh (about 340 ms median receive lag), and all 39
  trade timestamps preceded receipt. Both clock fault flags stayed false.

The Mac's automatic time service had applied another correction after the earlier
resync. Resynchronized before this capture, made no clock adjustments during it,
and checked afterward. Its time-server point estimates shifted from roughly
+4 ms before capture to -26 to -29 ms afterward, with substantial network
uncertainty. Short-term source ordering passed; continuous clock stability is
not established for unattended execution.

Do not extend this closed-market capture to obtain artificial fill statistics.
The next useful test is a fresh, bounded session during genuinely updating XAG
market hours, with a current manual control and research record. Verify source
freshness, aligned-anchor coverage, and complete-window quote/cancel behavior
before interpreting simulated P&L. No collector or exchange orders remain running.

### Accessible-edge evaluator verification, September 5, 2026

All 775 metals tests pass. Both saved XAG-connected captures reproduce their
original decision digest and final state before sensitivity evaluation: 2,640
inputs in the 60-second capture and 293 in the follow-up 10-second capture.

Final content-addressed scorecards:

- 60 seconds: `data/metals/derived/maker_evaluation/evaluation=2887f7c707925e099da219bf265ce3678be528c8eff92beb7718e6d3520bd309/report.json`
- 10 seconds: `data/metals/derived/maker_evaluation/evaluation=800217392e2ad470157a814117302f72b63ff6f6035462c0d76513b74a88aed1/report.json`

The 60-second capture has 1,123 active-window timer samples, 1,102 fresh exact
samples, and zero fresh XAG or causal fast-anchor samples. All three latency
cells have zero quotes/fills. The disposition is
`NO_USABLE_FAST_PRICING_EVIDENCE`, not an alpha finding or latency comparison with
statistical power. Supplemental tests demonstrate that slower cancellation can
produce a simulated cancellation-in-flight fill that the original timing avoids.

The evaluator records implementation hashes and preserves source provenance and
retention-review deadlines. These offline evaluations acquired no new market
data, purchased no subscription, changed no strategy parameters, and installed
no scheduled collector or live-order process.
