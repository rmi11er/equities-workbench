# Exploratory fair and signal ledger

Status: research-only hypothesis generation as of 2026-08-30. This ledger does not
open the formal research, final-holdout, replay, shadow, or trading gates.

## Scope and chronology

The frozen proxy baseline is
`ed1005f459a5beb2db07a99ada3e97feb17e03d9d669fae80a864b7da9412b89`.
It uses 1,874 structurally consistent, exchange-reported Silver 15-minute boundary
returns before 2026-08-28 to estimate a time-of-day variance curve, a slow
deseasonalized EWMA state, and a standardized empirical residual CDF. The validation
panel contains 83 complete 2026-08-28 Kalshi windows and 1,245 one-minute decisions.

The fair anchors generic `Metal.XAG/USD` to each exchange-reported opening target,
uses provisional half-up settlement rounding, and emits both Gaussian and empirical
probabilities. The Gaussian version is the frozen `p0` for these signal cards because
it was simpler and slightly better on this panel:

| Distribution | Brier | Log loss |
|---|---:|---:|
| Gaussian | 0.160065 | 0.473279 |
| Empirical CDF | 0.161766 | 0.475334 |

This is a basic volatility fair, not the final minimum external fair. A causally timed
futures-to-Pyth benchmark update is still open because historical Pyth candles do not
contain their original live receive times. The generic feed is permanently marked
`settlement_eligible=false`; exchange labels are provisional; only one validation UTC
day exists; and the final holdout was not accessed.

Every signal below uses the same chronological split: 55 complete development
windows, one complete embargoed window, and 27 complete test windows. Each feature is
fitted alone as an offset-logistic adjustment to Gaussian `p0` and compared with an
intercept-only recalibration fit on the same development rows. A negative delta means
the signal improved the metric. No joint signal model was created.

## Independent terminal-probability pilot results

| Signal | Coefficient | Δ Brier | Δ log loss | Frozen sign | Pilot probability gate | Current admission |
|---|---:|---:|---:|---|---|---|
| Spot return 1m | 0.007682 | -0.000017 | 0.000085 | pass | fail | no |
| Spot return 5m | -0.015584 | 0.000092 | -0.000045 | fail | fail | no |
| Spot return 10m | -0.176734 | 0.003759 | 0.006613 | fail | fail | no |
| Range volatility 3m | 0.174862 | 0.001137 | -0.000302 | pass | fail | no |
| Range volatility 10m | 0.188832 | 0.001458 | 0.000955 | pass | fail | no |
| Path efficiency 5m | -0.004005 | 0.000011 | -0.000024 | fail | fail | no |
| Path efficiency 10m | -0.075310 | 0.000954 | 0.001790 | fail | fail | no |
| COMEX trade intensity 1m | -0.059476 | -0.000202 | -0.000185 | fail | fail | no |
| COMEX volume state 5m | -0.255627 | 0.000120 | 0.000144 | fail | fail | no |
| COMEX price impact 5m | -0.035957 | 0.000343 | 0.000572 | fail | fail | no |
| Semivariance imbalance 10m | -0.368189 | 0.007904 | 0.017249 | fail | fail | no |
| Jump magnitude 10m | -0.029675 | -0.000044 | -0.000031 | fail | fail | no |
| Jump direction 10m | -0.263067 | 0.003043 | 0.005122 | fail | fail | no |
| Gold confirmation return 5m | 0.166687 | 0.002536 | 0.006873 | pass | fail | no |
| Silver residual return 5m | -0.213634 | 0.005560 | 0.011138 | fail | fail | no |

The cross-metal normalization was learned only on development observations. The
development silver-on-gold return beta was 2.222976. The XAU and XAG inputs each have
1,260 complete generic Pyth candles over the same day.

No feature passed the predeclared conjunction of expected sign, lower Brier, and lower
log loss. Therefore the signal-attributable fair-value improvement **admitted into this
version** is zero cents. This is a non-admission decision, not evidence that the
signals lack predictive content. Queue-aware net EV was not measurable from this
panel, so no card could have entered a trading model even if its probability gate had
passed.

The signal evidence is classified **underpowered and inconclusive**, not negative or
rejected. The 405 test rows contain only 27 distinct outcomes, all from one UTC day;
each coefficient pools predictions with 1--15 minutes remaining; and terminal binary
Brier score is a low-power first target for variance, tail, jump, and microstructure
features. Feature families must next be tested against their natural continuous or
tail targets before reconsideration in the terminal distribution.

The baseline result is more informative than the signal results. Gaussian `p0` has
about 36% Brier skill versus a constant 50% probability, but that is a weak comparator
and much of the gain can come from current spot relative to strike and time remaining.
The completed common-row scorecard compares every currently available benchmark on
378 historically boundary-aligned observations from the 27 held-out windows:

| Comparator | Brier | Log loss |
|---|---:|---:|
| Frozen TOD + slow-EWMA Gaussian `p0` | 0.157724 | 0.473211 |
| Flat-volatility Gaussian | 0.159509 | 0.477924 |
| Frozen empirical CDF | 0.160192 | 0.475016 |
| Historical Kalshi YES midpoint | 0.166035 | 0.490459 |
| Current-side development rate | 0.183910 | 0.553787 |
| Constant 50% | 0.250000 | 0.693147 |

`p0` ranks first overall on this one day, but the historical Kalshi midpoint has the
best 1--2-minute Brier score (0.114248 versus 0.124496 for `p0`). The midpoint is
descriptive and non-executable: original publication receipt latency is unknown and
the comparison excludes spread and fees. The scorecard has one effective day, so its
day-block interval is not estimable and it supports no pricing-alpha claim.

## Natural-target diagnostics

The 18 available cards now route momentum/path features to future signed return;
range, EWMA, intensity, and volume features to future realized variance; semivariance
to directional tails; jump features to occurrence and magnitude; and cross-metal
features to future Silver residual return. The synchronized test remains one day, so
every card stays `underpowered_inconclusive`; pooled coefficients are plumbing
diagnostics only, not model-admission evidence.

A separate causal expanding-window diagnostic can use the 1,874 prior boundary
returns for the slow variance components. After a 384-window burn-in it scores 1,490
windows over 21 UTC days. Time-of-day variance improves on flat variance in both MSE
and QLIKE, with 95% day-block paired-loss intervals wholly below zero. Adding the slow
EWMA to time of day has intervals spanning zero and worsens average QLIKE, so the
incremental EWMA evidence is mixed or absent.

| Variance forecast | MSE | QLIKE |
|---|---:|---:|
| Flat | 2.519294e-10 | -11.068967 |
| Time of day | 2.406930e-10 | -11.224461 |
| Time of day + slow EWMA | 2.406172e-10 | -11.218872 |

Detailed immutable reports:

- [Spot returns](../../data/metals/derived/signal_cards/run=spot-return-signals-be68845ad71547f77b464757/report.json)
- [Range volatility and path efficiency](../../data/metals/derived/signal_cards/run=path-variance-signals-066813c0f4ff52cd55d644f4/report.json)
- [Futures activity](../../data/metals/derived/signal_cards/run=futures-activity-signals-61df8fc2363218ad15f6f7d4/report.json)
- [Semivariance and jumps](../../data/metals/derived/signal_cards/run=semivariance-jump-signals-68f83fe293a3ed79857060e9/report.json)
- [Cross-metal returns](../../data/metals/derived/signal_cards/run=cross-metal-signals-a2e1021df62ebb16646578dc/report.json)
- [Baseline comparator scorecard](../../data/metals/derived/baseline_scorecard/run=baseline-scorecard-44de90abed88b1f2cd5317cd/report.json)
- [Natural-target signal study](../../data/metals/derived/natural_target_research/run=natural-targets-91df1b166cbaa1a46789f8a7/report.json)

## Untested, data-gated families

Implied-volatility level/term structure and implied skew/curvature were not skipped on
their merits. The local archive has no licensed historical option quotes or derived
implied volatilities. Testing them now would require inventing the feature.

The COMEX option-expiry study also remains untested. It needs a versioned official
Central-Time calendar, multiple expiry and matched non-expiry days, and preferably
option definitions plus reliable volume/open-interest data. One ordinary proxy day
cannot identify an expiry effect or support matched controls.

Execution alpha remains a separate future model. Historical Kalshi candles and public
trades cannot reconstruct native queue position, cancellation latency, fill-conditioned
toxicity, or a queue-aware net-EV panel.

## Next admissible work

1. Preserve Gaussian `p0` unchanged; do not combine currently unadmitted features.
2. Repeat the implemented natural-target study across the synchronized multi-day
   panel before another terminal-probability test.
3. Extend the proxy panel across multiple days for stability and day-block intervals.
4. Add a causally timed futures-to-Pyth benchmark estimator when the timestamp contract
   is supportable; do not infer live availability from historical bar labels.
5. Cost and acquire the smallest useful options/official-expiry sample before opening
   the implied-volatility, skew, and expiry cards.
6. Fit the fast updater and fill-selection model only from causal event data; keep
   Kalshi price and flow out of the external fair and apply fees explicitly in quote
   bounds.

The focused comparator, natural-target, model-boundary, and CLI test set passes 93
tests; the complete metals suite passes 490 tests. The repository-wide suite currently
has four unrelated failures in RFQ cent rounding and taker bailout expectations, with
605 tests passing.
