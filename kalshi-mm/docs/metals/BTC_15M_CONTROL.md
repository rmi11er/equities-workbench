# BTC 15-minute Kalshi control study

This is a read-only comparison market for the generic two-sided Kalshi BBO
toxicity study. It does not tune a BTC strategy and does not authorize live
orders. All collected and derived files use the isolated ignored root
`data/btc-control`; existing `data/metals` artifacts are not modified.
The shared canonical schema retains its historical column name `metal`; in
this isolated catalog the value `btc` is a product label, not a claim that BTC
is a metal.

## Public market identity and coverage

The public Kalshi series endpoint identifies the market as:

- Series ticker: `KXBTC15M`
- Title: `Bitcoin price up down`
- Frequency: `fifteen_min`
- Category: `Crypto`
- Settlement source: CF Benchmarks
- Published determination note: collect 60 corresponding Real Time Index
  observations during the final minute and use their average
- Fee metadata observed on 2026-08-29: `quadratic`, multiplier `1`
- Exchange index observed on 2026-08-29: `2`

The observed record is available from Kalshi's public
[`KXBTC15M` series endpoint](https://external-api.kalshi.com/trade-api/v2/series/KXBTC15M).

Market tickers must be discovered from Kalshi, not generated. For example,
`KXBTC15M-26AUG291700-00` opened at `2026-08-29T20:45:00Z` and closed at
`2026-08-29T21:00:00Z`. The embedded clock was New York local time during this
observation, not UTC. The observed current form is
`KXBTC15M-YYMONDDHHMM-MM`: an uppercase three-letter month, venue-local close
clock, and a final segment repeating the close minute. This is descriptive,
not a ticker-generation rule. The corresponding event ticker omits the final
minute segment (`KXBTC15M-26AUG291700` in the example). The code filters the
public catalog by the authoritative `open_time` and `close_time` fields and
retains venue-native tickers because historical clock conventions may differ.

At discovery time on 2026-08-29, Kalshi's public historical tier contained
18,629 `KXBTC15M` markets, from
`KXBTC15M-25DEC101700-00` (close `2025-12-10T22:00:00Z`) through
`KXBTC15M-26JUN291945-45` (close `2026-06-29T23:45:00Z`). The public historical
cutoff was `2026-06-30T00:00:00Z`; newer markets were in the current tier.
Catalog presence does not imply that every market has every candle or trade.

The control run is deliberately narrower and time-matched to the local metals
study. Its inclusive market-close bounds are:

- First close: `2026-07-31T18:15:00Z` (`1785521700`)
- Last close: `2026-08-29T04:00:00Z` (`1787976000`)
- Frozen train/test boundary: `2026-08-21T08:45:00Z`
  (`1787301900000000000` nanoseconds)

This returned 2,693 completed public BTC markets, and one-minute candles are
collected for the full bounded set. BTC trade pagination is materially larger
than metals, so the control uses a frozen outcome-blind sample: every eighth
15-minute close from the first bound (offset zero), or one market every two
hours. This selects 337 trade-complete markets spanning the same dates. The
diagnostic requires a durable completed-trade marker and excludes unsampled
markets rather than treating missing trade history as a no-fill observation.
The frozen grid contains 245 train and 92 held-out markets, covers all 30 UTC
dates in the bounded interval, and runs from the first `18:15Z` close through
the final sampled `2026-08-29T02:15:00Z` close.
The report's `data_coverage` object records the exact candle/trade coverage that
survived normalization and deduplication; it is the authoritative statement of
study coverage.

The endpoints and tier-routing behavior are documented by Kalshi's
[public market-data quick start](https://docs.kalshi.com/getting_started/quick_start_market_data),
[market list](https://docs.kalshi.com/api-reference/market/get-markets),
[market candlesticks](https://docs.kalshi.com/api-reference/market/get-market-candlesticks),
[public trades](https://docs.kalshi.com/api-reference/market/get-trades),
and [historical data guide](https://docs.kalshi.com/getting_started/historical_data).

## Frozen study contract

The BTC run calls the same `toxicity_diagnostics` implementation as silver,
with only `metal="btc"` and the isolated catalog changed. It preserves:

- the earliest valid two-sided one-minute candle close in each market/time
  bucket;
- the 10–15m, 5–10m, 2–5m, and 1–2m time buckets;
- all seven quoted-price buckets from 0–10 cents through 90–100 cents;
- 1s, 5s, 30s, 1m, 5m, and settlement (`15m/end`) horizons;
- touch and one-cent-penetration conditional-fill bounds over a 60-second
  quote lifetime;
- buying/selling market favorite/underdog Yes-side action cuts;
- paired Yes/No quote-window accounting for both, Yes-only, No-only, and
  neither-fill states, including settlement EV;
- UTC day-block uncertainty, 1% tail cuts, best/worst-day concentration,
  exit-frontier bounds, and entry-to-settlement waterfall definitions.

No policy, price bucket, horizon, fill threshold, or split boundary is selected
from the BTC results.

## Frozen-run comparison

The 2026-08-29 BTC control run is under
`data/btc-control/derived/toxicity_diagnostics/run=20260829T215059.672046Z`.
The existing silver comparator is
`data/metals/derived/toxicity_diagnostics/run=20260829T203628.520625Z`.

| Held-out metric | Silver | BTC control |
| --- | ---: | ---: |
| Touch settlement P&L | -0.005150 | -0.000055 |
| Touch 90% UTC-day CI | -0.006617 to -0.003480 | -0.004066 to 0.003620 |
| 1-cent penetration settlement P&L | -0.020648 | -0.013006 |
| 1-cent penetration 90% UTC-day CI | -0.023941 to -0.017672 | -0.028221 to 0.002413 |
| Touch paired-window EV | -0.009350 | -0.000109 |
| 1-cent penetration paired-window EV | -0.027637 | -0.018744 |
| Touch both-fill rate | 0.815581 | 0.964706 |
| 1-cent penetration both-fill rate | 0.470951 | 0.570588 |

Settlement P&L entries are dollars per filled contract; paired-window EV is
dollars per two-sided one-contract-per-leg quote decision. Rates are fractions.

The BTC control does not reproduce the silver action-side pattern. In BTC,
buying the underdog was positive and selling it negative at settlement; the
silver signs were the reverse. All four BTC action-side day-block intervals
cross zero under both fill bounds. The one-cent aggregate and paired-window
means remain negative, but their BTC day-block intervals also cross zero.
These results do not establish a portable favorite-fade edge.

## Reproduction

```bash
uv run python -m src.metals.cli \
  --config configs/btc_control.example.toml \
  backfill-kalshi --metal btc \
  --from-close-seconds 1785521700 \
  --to-close-seconds 1787976000 --skip-trades

uv run python -m src.metals.cli \
  --config configs/btc_control.example.toml export-parquet

uv run python -m src.metals.cli \
  --config configs/btc_control.example.toml \
  backfill-kalshi-candles --metal btc

# Re-run the bounded metadata command without --skip-trades. Completed
# per-market trade requests are skipped durably on subsequent runs.
uv run python -m src.metals.cli \
  --config configs/btc_control.example.toml \
  backfill-kalshi --metal btc \
  --from-close-seconds 1785521700 \
  --to-close-seconds 1787976000 \
  --market-stride 8 --market-stride-offset 0

uv run python -m src.metals.cli \
  --config configs/btc_control.example.toml export-parquet

uv run python -m src.metals.cli \
  --config configs/btc_control.example.toml \
  run-toxicity-diagnostics --metal btc \
  --test-start-ts-ns 1787301900000000000 \
  --require-complete-trades
```

## Comparability limits

This is a Kalshi-price toxicity control, not a BTC fair-value backtest.

- BTC resolves from CF Benchmarks. The metals pipeline's settlement reference
  is Pyth, so the underlying reference, averaging rule, and potential basis are
  not shared.
- There is no synchronized external BTC spot/futures fair, so the report cannot
  distinguish informed flow from a stale or wrong Kalshi quote.
- One-minute BBO candles do not reveal displayed depth, queue position,
  cancellations, partial fills, or sub-minute quote paths. Touch is optimistic;
  one-cent penetration is only a more conservative price-path bound.
- Public trades provide timestamps and taker orientation, but last-trade marks
  are not midpoint, microprice, or an executable exit.
- The candidate windows overlap within markets and across a 24/7 BTC session.
  UTC day-block intervals reduce but do not eliminate dependence.
- The BTC trade-complete sample retains one of every eight chronological
  15-minute markets. It spans the metals dates but is smaller, 24/7, and has a
  regular two-hour phase; compare signs and broad shapes, not raw counts.

The artifacts are suitable for a relative toxicity/control comparison only.
They cannot approve capital deployment, quote size, queue models, or a live BTC
or metals policy.
