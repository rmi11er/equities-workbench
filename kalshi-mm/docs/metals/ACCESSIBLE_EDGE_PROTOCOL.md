# Accessible-edge evaluation protocol

Version: `accessible-edge-v1`, prospectively specified September 5, 2026.

The objective is to decide whether persistent mispricing is accessible to this
particular maker, after adverse selection, execution friction, and operating costs.
It is not to establish that every cheap underdog is cheap volatility, or that a
working collector establishes alpha. This protocol changes no quoting parameters,
authorizes no orders or purchases, and schedules no collection.

The operational commands and source-health safeguards remain in the
[maker forward runbook](MAKER_FORWARD_RUNBOOK.md). The first connected XAG captures
were closed-market engineering tests; repeated carried-forward publications are
not fresh-price observations or open-market strategy evidence.

## Three questions, in order

1. **Persistent disagreement or feed lag?** Does an external-model discount survive
   fresh, source-aligned XAG updates long enough for the unchanged policy to become
   eligible and for a post-only quote to arrive? A large instantaneous discrepancy
   that disappears before achievable arrival is not accessible edge.
2. **Obtainable flow or fill selection?** At the competitive quote prices we can
   reach, does the bid-to-fair cushion survive conditioning on a plausible fill and
   on cancellation latency? Favorable displayed prices without obtainable volume
   are not a business; fills occurring only after the underlying moves against us
   may consume the apparent discount.
3. **Enough value after costs?** Is the remaining opportunity frequent and large
   enough to justify data, infrastructure, trading costs, and operator time? A
   positive per-contract estimate does not establish positive monthly economics.

These questions distinguish a volatility/distribution advantage from a speed race.
They do not identify a counterparty as retail or professional: public prints alone
do not establish trader identity or motivation.

## Frozen strategy and experimental arms

The primary subject is `silver-opportunistic-maker-v1`, not the older continuous
paired maker or historical opportunistic-taking screen. Do not pool those results
as though they were observations of the current strategy.

The current strategy uses the frozen Silver time-of-day Gaussian model. Empirical
probabilities are recorded as a comparison, not silently substituted as the
minimum-fair policy from an earlier study. The manual arm applies the multiplier
recorded prospectively by the operator; the baseline arm uses `1.00`. Both share
pauses, source-health exclusions, and execution rules. A manual revision is part of
the strategy record, not a parameter to reconstruct after seeing an outcome.

Keep the existing maker settings unchanged:

- One contract per order, no replenishment after any partial fill, and at most one
  unsettled market per arm; each independent arm has a simulated $25 bankroll,
  $1 maximum worst-case market/aggregate exposure, and $5 session loss limit.
- A 7–31 cent bid band, at least 120 seconds remaining, and one second of
  continuous eligibility.
- The existing 2-cent uncertainty charge, 1-cent adverse-selection charge, and
  3-cent required edge, plus the explicitly declared maker-fee reserve.
- The recorded native price grid, post-only arrival checks, source-age and anchor
  limits, clock guards, and unknown-exposure handling.
- Direct XAG feed 345, causally anchored to the exact Kalshi-carried Silver index,
  can reduce or veto the slow-index ceiling; it cannot increase it. The XAG return
  mapping remains beta one.

The capture's serialized settings, model hash, and actual control history are
authoritative. Do not silently migrate an old capture to different policy defaults.
The fee reserve is a scenario charge, not a claim that the account's actual maker
fee equals that amount. Uncertainty and adverse-selection charges are entry
haircuts, not additional cash expenses to subtract again from realized P&L.

## Offline evaluation and latency sensitivity

Evaluate a saved maker report without collecting new data:

```sh
uv run --locked python -m src.metals.maker_evaluation \
  --report data/metals/derived/maker_forward/run=COLLECTOR_RUN_ID/report.json \
  --output-root data/metals/derived \
  --session-label normal
```

The evaluator first verifies the original deterministic replay and then evaluates
the fixed scenarios. Preserve the original capture result even if its recorded
latencies differ from these sensitivity settings:

- Original capture, with its original settings and control history.
- 100 ms simulated submission and 100 ms simulated cancellation.
- 100 ms simulated submission and 250 ms simulated cancellation.
- 100 ms simulated submission and 500 ms simulated cancellation.

Report all distinct scenario settings and both arms. A fixed-grid cell identical
to the original capture is represented once as the primary, not counted twice.
Do not select a latency winner, change the volatility model, or relabel a
retrospective scenario as the production policy.
Cancel delay begins according to the existing delayed lifecycle: requests made
before arrival cannot erase submission latency or remove pending exposure. Trades
during cancellation remain eligible. The source receive timeline is preserved;
these scenarios do not pretend to measure actual exchange acknowledgment latency.

Scenarios replay the same data and are dependent counterfactuals, not additional
days, independent experiments, or simultaneous orders competing for liquidity.
Replay verification establishes reproducibility, not correctness of the fair model
or real-world execution qualification.

Live SI is not enabled by this work. A later, separately specified comparison can
use an existing permitted historical SI backfill to ask whether CME movements
invalidate quotes before XAG does. Such a test must declare source-to-host latency
assumptions and cannot substitute source timestamps for missing live receipt
timestamps. The offline CLI above does not implement that SI comparison or acquire
new SI data.

## Scorecard and metric definitions

These are the required interpretation rules for the experiment. A field that
cannot be established from a capture remains unavailable; this protocol does not
claim that every research metric below is already implemented by the evaluator.

### Source health and the candidate funnel

Report observed duration, actively updating XAG coverage, valid exact-index anchor
coverage, clock faults, source gaps, outstanding exposure, and the reasons entries
were suppressed. Distinguish publication freshness from message-delivery cadence.
Do not call a closed-market run healthy for strategy evaluation because its socket
remained connected.

Separate evaluated rows, distinct eligible episodes, submitted quote intents,
acknowledged/resting quotes, touches, penetration-scenario fills, and distinct
markets with fills. Decision logs can be throttled, so their row count is neither
continuous eligible time nor an exact count of all underlying events. Where
available, report eligibility persistence, quote age, observed resting time,
post-only rejections, and fills during cancellation. Keep unavailable quantities
explicit rather than deriving unsupported rates from event counts.

For comparisons with costs and future capacity, report the sample's actual
observed-session opportunity frequency. Do not annualize an isolated burst, scale
one-contract simulated fills by displayed depth, or treat repeated quotes in one
15-minute market as independent outcomes.

### Fair drift, mark-to-price, and terminal P&L

For a contract bought at price `b`, let `p0` be the side's model fair at fill, `ph`
its fair at horizon `h`, `f` the entry fee per contract, and `Y` its final payout
in `{0, 1}`. Keep the following quantities separate:

- Initial model edge after entry fee: `p0 - b - f`.
- External-fair drift: `ph - p0`. This measures subsequent movement of the model's
  fair, not trading P&L. Negative drift does not, by itself, prove that the initial
  bid-to-fair cushion was exhausted.
- Model-marked value after entry fee: `ph - b - f`. This still depends on the
  model being well calibrated; it is not a liquidation price or realized profit.
- Executable liquidation P&L: the observed sell-side executable price less `b`,
  entry fee, and the applicable exit fee/slippage assumption. Missing executable
  depth or an unknown exit cost must be disclosed, not assigned a free exit.
- Terminal P&L per contract: `Y - b - f`, only when an authoritative result is
  available. Multiply by actual scenario-filled quantity to obtain dollars.

Use matching eligible rows and quantity weights when comparing initial edge with
subsequent drift; means from different horizon-coverage cohorts do not necessarily
add. Distinguish exact-index and fast-proxy model marks wherever both are reported.
The model's own favorable marks cannot independently validate its calibration.

Report 1/5/30/60-second marks when valid observations exist, alongside terminal
outcomes. Preserve missing horizons and unresolved settlements; do not carry a
stale price across a gap, invent an outcome, or treat an unsettled contract as a
loss. Disclose observation delay relative to the target horizon where available.
A positive one-second liquidation mark is not a universal admission requirement:
spread, fees, and holding tail risk can coexist with negative short-horizon
liquidation and positive terminal expectation. Equally, favorable hypothetical
terminal fair cannot excuse repeated, unexplained post-fill deterioration.

### Queue access, costs, and capacity

Strict penetration is a declared execution scenario, not a measured fill rate.
Trade-through events can overrepresent toxic fills relative to benign at-touch
flow, while public tape alone cannot prove that our hypothetical order would have
obtained either. Touches do not become assumed fills; penetration does not become
guaranteed execution. No point estimate of real fill probability or capacity is
justified without additional queue evidence.

Keep actual exchange fees, scenario fee reserves, data subscriptions,
infrastructure, and operator time separate. Unknown costs and unknown executable
volume remain `null`/unavailable, not zero. A possible economics worksheet is
`obtainable contracts per day * net expected value per contract - daily fixed
costs`, but every term needs an explicit basis and consistent units. Do not
produce a profit projection when the inputs are unidentified. The present
one-contract experiment cannot establish size scalability.

## Sampling and review schedule

Use the first 5–10 healthy trading days for a feasibility review: source coverage,
candidate frequency and persistence, obtainable quote prices, and the effect of
cancel-latency stress. This is a review budget, not an obligation to continue after
a clear failure and not a claim that ten days establish significance. Brief
connected smoke tests do not count as healthy trading days.

A broader panel of roughly 20–40 trading days is an aspiration if the feasibility
results justify it, not a magic admission threshold. Count independent market
windows and trading-day clusters, disclose partial coverage and serial dependence,
and retain all exclusions. Multiple arms, horizons, repeated fills, and latency
scenarios do not multiply independent sample size. Any uncertainty interval must
respect day-level clustering and the limited number of observed days.

Label Sunday reopening, holiday sessions, scheduled-event periods, and ordinary
sessions separately. The evaluator accepts `normal`, `reopen`, `holiday`, `closed`,
`diagnostic`, and `unspecified`; record more detailed event context separately.
A label is operator metadata, not automatic verification of market state. Reopening/holiday observations
are useful for gap handling but must not be pooled silently into ordinary-session
performance. Do not collect a stale weekend merely to increase elapsed runtime.

Keep the formal final holdout unopened. If evidence motivates a new model,
threshold, manual-setting rule, or feed integration, version that as a new
prospective experiment; do not optimize it on these observations and describe the
same observations as its independent test.

## Continue, stop, or pivot

Continue bounded shadow research when data quality is adequate and persistent,
competitively quoteable discounts remain after realistic timing and explicit
charges. Favorable flow access, calibration, and terminal evidence must ultimately
agree; no single counterfactual or lucky tail payout establishes that agreement.

Stop the run for the existing health, clock, or unknown-exposure faults. Repair
the cause and preserve the failed evidence; do not loosen a guard to obtain fills.
Repeated missing coverage is an engineering limitation, not evidence for or
against market alpha.

Stop or reassess the research thesis if apparent opportunities consistently
disappear before achievable arrival, competitive prices consume the modeled edge,
or plausible fills only occur after adverse movement consumes the cushion. A
conservatively estimated opportunity rate that cannot cover operating costs is
also a negative business result, even with positive per-contract gross value.

A result that works only under faster unmeasured cancellation is evidence of
latency dependence, not authorization to buy infrastructure. Likewise, evidence
that SI might improve protection motivates a separately scoped test, not an
automatic data subscription. Preserve the negative result rather than rescuing it
through threshold selection.

Real benign-flow access may remain unresolved after shadow evaluation. A tiny
real-order pilot would require separate order-routing implementation, operational
review, explicit authorization, and its own bounded risk/cost plan. This protocol
does not grant that gate, change provider permissions, or assert a probability of
eventual profitability.
