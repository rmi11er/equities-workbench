# Opportunistic long-vol screen — 2026-09-01

## Decision

The fair-filtered long-underdog hypothesis is promising enough to become the primary
forward-shadow strategy. It is not ready for live capital.

The existing study rejects a different strategy: buying every 7–31 cent underdog.
That unconditional rule lost about 3.8 cents per passive fill in its held-out candle
proxy. The relevant hypothesis is narrower:

> Buy the current underdog only when a fresh external terminal-distribution fair
> exceeds its executable ask by the taker fee, expected selection, model uncertainty,
> and a frozen entry hurdle.

Buying the underdog is not pure vega. It has positive variance sensitivity in the
core region, but also contrarian delta, skew, jump, rounding, and model-risk exposure.

## Frozen exploratory rule

The first reproducible screen uses:

- One entry at the first qualifying observation in each Kalshi window.
- Current underdog ask from 7 to 31 cents.
- Between 2 and 15 minutes remaining.
- The frozen time-of-day plus EWMA Gaussian external fair.
- A one-contract taker fee rounded up to a centicent using the declared `0.07 p(1-p)`
  coefficient.
- At least 3 cents of external edge after that fee.
- No Kalshi price or flow in the external fair.
- The formal final holdout remains unopened.

The implied-volatility interpretation is descriptive. Under a zero-mean Gaussian
digital with common spot, strike, and horizon:

$$
\frac{\sigma_{\text{Kalshi}}}{\sigma_{\text{external}}}
=
\frac{|\Phi^{-1}(p_{\text{external, underdog}})|}
{|\Phi^{-1}(p_{\text{Kalshi, underdog}})|}.
$$

A ratio below one is consistent with cheap Kalshi variance, but stale spot, drift,
skew, jumps, or a bad external variance forecast can produce the same gap.

## Historical screen

The input contains 55 development, one embargo, and 27 test windows, all from one UTC
day. Historical candle end times do not certify live receipt or execution.

| Fair definition | Split | Entries | Mean ask | Predicted net edge | Settlement P&L | Market-bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen Gaussian | Development | 28 | 21.93¢ | 4.75¢ | +5.47¢ | -9.86¢ to +22.18¢ |
| Frozen Gaussian | Test | 10 | 22.60¢ | 6.26¢ | +6.19¢ | -20.07¢ to +34.53¢ |
| Frozen empirical CDF | Development | 16 | 17.12¢ | 4.03¢ | +6.90¢ | -11.12¢ to +29.48¢ |
| Frozen empirical CDF | Test | 8 | 19.50¢ | 5.20¢ | +16.91¢ | -9.78¢ to +51.29¢ |
| Side-specific minimum of both fairs | Development | 16 | 17.12¢ | 4.03¢ | +6.90¢ | -11.12¢ to +29.48¢ |
| Side-specific minimum of both fairs | Test | 8 | 19.50¢ | 5.20¢ | +16.91¢ | -9.78¢ to +51.29¢ |

The Gaussian rule's test entries gained 5.66 cents at the next one-minute midpoint
and 3.47 cents under an immediate bid liquidation proxy after entry and exit fees.
The development equivalents were 3.75 and 1.98 cents. This is consistent with a
temporary pricing gap, but the timestamp limitation remains material.

The median implied-volatility ratio among selected Gaussian entries was 0.76 in
development and 0.69 in test. That is a useful monitoring variable, not a fitted
cutoff.

The confidence intervals all cross zero. With one effective day, a trading-day block
interval cannot be estimated. The historical result is therefore evidence for more
collection, not evidence of established alpha.

## Synchronized capture cross-check

The saved 2026-08-31 capture was replayed with the direct XAG feed anchored to the
exact settlement index and a declared 100 ms Pyth transport scenario. It contains
four completed outcomes.

At the 3-cent rule:

- The rule selected one first entry in each of four completed windows.
- All four selected underdogs lost; mean settlement P&L was -26.30 cents.
- Midpoint markout was -1.30 cents at 1 second, +0.20 at 5 seconds, +1.70 at 30
  seconds, and -5.05 at 60 seconds.
- Requiring one second of continuous eligibility reduced the sample to three but did
  not repair the result.
- Apparent gaps above 8 cents all disappeared before one second, identifying them as
  transient discrepancies rather than reliable entries.

The existing strict-penetration shadow fills provide a maker-selection warning. Of
the fills that bought the then-current underdog at at least 3 cents below the fast
external fair, 62 events totaling 58.65 contracts had mean external-fair markouts of
-0.52 cents at 1 second and -2.45 cents at 5 seconds. These are repeated fills across
only four windows, so they are not independent settlement evidence; they do show that
passive fills remain selected even when the displayed model edge looks large.

The historical and synchronized results conflict. The most plausible explanations
are small-sample outcome noise, candle timing optimism, residual external-feed
staleness, an overestimated nighttime variance state, and fill selection. None should
be resolved by tuning the test-day threshold.

## Forward-shadow policy

The strategy should now record, but not trade, two related actions.

### Opportunistic taker

For the current underdog side $u$:

$$
p_{\text{external},u}-a_u
>
\text{taker fee}+U+A_{\text{take}}+H,
$$

where $U$ is model uncertainty, $A_{\text{take}}$ is a fast selection/staleness
charge, and $H$ is the frozen entry hurdle.

Use the minimum of the frozen Gaussian and empirical-CDF fairs until their difference
is explicitly modeled as uncertainty. Require:

- Pyth-anchored fast fair and Kalshi book ages no greater than 250 ms.
- At least one second of continuous eligibility.
- Direct XAG and SI direction agreement when both feeds are healthy.
- One contract and one entry per market during shadow evaluation.
- No entries below 7 cents, above 31 cents, or inside the final two minutes.
- Scheduled-event and expiry states excluded until their variance multiplier is
  measured.

### Opportunistic maker

The passive bid ceiling is:

$$
b_u^{\max}
=
p_{\text{external},u}-U-A_{\text{maker}}-\text{maker fee}.
$$

It must also be capped below the current ask. A displayed edge is not sufficient:
the quote is eligible only if the fill-selection model's lower confidence bound still
leaves positive value. Do not replenish after a fill during the initial shadow stage.

## Admission gates

Collect at least 20 trading days, preferably 40, while logging every candidate and
every rejected reason. The pivot advances beyond shadow only if:

- The fixed rule is positive after effective fees by trading-day block confidence
  interval.
- It remains positive with the strongest days removed.
- Both terminal P&L and 1/5/30/60-second markouts are reported.
- Results are stable by time remaining, underdog price, session, volatility state,
  and event state.
- The robust minimum-fair rule is positive without threshold optimization.
- Queue-aware passive replay covers acknowledgments, cancellations, partial fills,
  and quote age.
- A newly frozen version passes a still-unseen final holdout and then a live shadow
  period.

The always-on paired market maker should remain a data collector and comparison arm.
The opportunistic long-vol overlay—not continuous two-sided quoting—is now the primary
alpha research path.

## Reproducibility

- Screen implementation: `src/metals/opportunistic_long_vol.py`
- Unit tests: `tests/metals/test_opportunistic_long_vol.py`
- Historical result: `data/metals/derived/opportunistic_long_vol_screen/run=20260901T183500Z/report.json`
- Baseline score rows: `data/metals/derived/baseline_scorecard/run=baseline-scorecard-44de90abed88b1f2cd5317cd/score_rows.jsonl`
- Synchronized capture identifiers: collector `20260901T002306.637377Z`, shadow
  `20260901T002306.683717Z`
