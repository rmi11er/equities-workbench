# Silver forward capture — 2026-09-02

## Decision

The 11.44-hour read-only run is usable for Kalshi tick, book, trade, candidate-funnel,
and fill-selection research. It is not a complete exact-index panel and does not
support live capital.

The frozen opportunistic long-underdog policy produced one independent entry. That
entry won at settlement, but its one- and five-second executable liquidation markouts
were both negative. One entry from one UTC day cannot identify alpha or a trading-day
confidence interval.

## Capture identity and integrity

- Collector: `20260902T003832.355240Z`
- Shadow: `20260902T003832.408452Z`
- Exchange settlement backfill: `20260903T121300.490998Z`
- Capture audit:
  `data/metals/derived/forward_collection_audit/collector=20260902T003832.355240Z/shadow=20260902T003832.408452Z/report.json`
- Corrected shadow report:
  `data/metals/derived/forward_shadow_report/report=c333ce27e497ced8b83ae7b0/report.json`
- Frozen-policy report:
  `data/metals/derived/forward_long_vol_report/report=eb39f986f8d1742f57a22d8f/report.json`
- Combined two-capture report:
  `data/metals/derived/combined_forward_long_vol_report/report=70e2c81a008b1b8c66932902/report.json`

The collector and shadow were intentionally interrupted after 41,175.6 seconds. The
audit accepts that single controlled `CancelledError`, verifies every JSONL hash chain
and candidate lineage edge, and confirms `live_orders=false` with no exchange order
IDs. The capture retained approximately 3.14 GiB of raw and normalized source data:

- 1,528,455 order-book records;
- 13,796 trades;
- 10,058 exact `Metal.Index.SILVER/USD` updates;
- 32,695 fair decisions;
- 33,314 long-vol candidate rows; and
- 45 observed market windows.

## Settlement-label correction

The first version of the report incorrectly treated any last pre-close Pyth value as
a settlement label whenever the overall capture continued past that boundary. This
allowed stale carry-forwards after WebSocket interruptions.

The exchange backfill made the failure observable:

- Updates exactly one second before the boundary matched the recorded displayed
  settlement value and direction in all 11 comparable cases.
- The other 33 displayed values were stale by 3 seconds to 14 minutes and all failed
  exact value agreement; nine also produced the wrong direction.

The report now requires a strict positive source age no greater than 1.5 seconds.
Stale values remain visible only in a diagnostic field and cannot populate the
provisional outcome, settlement value, completed-window count, or P&L label. Under the
corrected rule there are 11 boundary-eligible labels, all 11 exactly agree with
Kalshi, and 10 are full windows because collection began partway through the first.

Recorded Kalshi outcomes remain valid terminal labels for causal entries even when a
later exact-index transport gap prevents local boundary reconstruction.

## Frozen long-vol result

The causal funnel was:

| Stage | Rows | Markets |
| --- | ---: | ---: |
| All candidate evaluations | 33,314 | 45 |
| Evaluated with a fair | 30,375 | 45 |
| Loose historical-screen rule | 565 | 18 |
| Strict forward policy | 24 | 1 |
| First entry per market | 1 | 1 |

The strict policy includes the 7–31 cent price band, 2–15 minute horizon, conservative
minimum of Gaussian and empirical fairs, fee, 2-cent uncertainty charge, 1-cent
adverse-selection charge, 3-cent residual hurdle, 250 ms book/index receive ages, and
one second of continuous eligibility.

The sole entry bought YES at 15 cents with a 0.9-cent entry fee and 308 seconds left.
The conservative external fair was 23.44 cents and the fully charged edge was 4.54
cents. Kalshi recorded YES with expiration value 63.781, producing +84.10 cents for
one contract.

Short-horizon results point the other way:

| Horizon | Midpoint P&L | Immediate executable liquidation P&L |
| --- | ---: | ---: |
| 1 second | -1.40¢ | -3.70¢ |
| 5 seconds | -1.90¢ | -3.70¢ |
| 30 seconds | unavailable | unavailable |
| 60 seconds | unavailable | unavailable |

The missing 30/60-second marks are not imputed across the exact-index transport gap.
The looser 565 screen rows are retained as funnel diagnostics and are not scored as
if they were frozen-policy entries.

A post-fix 30-second smoke on 2026-09-03 caught a second strict entry in the 19:15
EDT window. It bought YES at 11 cents plus a 0.69-cent fee with 157 seconds left and
a 6.40-cent fully charged edge. Kalshi recorded NO, so the entry lost 11.69 cents.
Its executable liquidation marks were -3.17 cents at one second and -2.89 cents at
five seconds.

Across the two frozen entries on two UTC dates, terminal P&L is +72.41 cents, the win
rate is 1/2, and the trading-day block interval is [-11.69 cents, +84.10 cents]. Both
entries had negative immediate liquidation markouts: the mean was -3.44 cents at one
second and -3.30 cents at five seconds. This tiny mixed sample supports collection,
not admission.

## Paired-maker comparison arm

The always-on paired shadow produced 506 strict-penetration fill events totaling
486.40 contracts. Its external-fair markouts were -2.20 cents per contract at one
second and -3.31 cents at five seconds despite a displayed fee-net edge of +5.81
cents. This remains evidence of adverse fill selection, not an executable backtest.

## Transport health and next collection

The long run recorded 51 WebSocket disconnects and 66 exact-index source gaps over
1.5 seconds. A fresh three-minute reproduction on 2026-09-03 was healthy: 5,156 book
deltas, 121 trades, 179 consecutive 1 Hz index observations, no missing interval, and
270 ms median source-to-receive latency. It also exposed and motivated a cleanup fix
for an in-flight receive task during normal shutdown.

The next long capture must preserve the same frozen policy and add no tuned threshold.
For every window, report exact-index boundary freshness explicitly and obtain the
exchange settlement backfill before terminal scoring. Gold should be collected as a
separate raw comparison arm; the Silver-trained variance model must not be silently
applied to Gold.

The strategy remains `CONTINUE_FORWARD_COLLECTION_NO_LIVE_CAPITAL`.
