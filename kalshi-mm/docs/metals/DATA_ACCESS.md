# Metals data access decision

Checked against official sources on 2026-08-28. Prices and entitlements can change;
obtain written confirmation for non-display research, raw retention, internal sharing,
cloud storage, derived models, and eventual automated trading on Kalshi.

## Pyth: what to acquire now

1. Create a free Pyth Terminal account and generate the self-service Pyth Pro trial
   API token. Pyth says no payment card is required; the trial duration and exact index
   entitlement are not published. See [API-key acquisition](https://docs.pyth.network/price-feeds/pro/acquire-api-key)
   and [Pyth Terminal](https://docs.pyth.network/price-feeds/pro/pyth-terminal).
2. Store the token as server-side `PYTH_API_KEY`; do not place it in TOML, source, logs,
   or chat.
3. Test or ask `data@dourolabs.xyz` to confirm entitlement to:
   - `Metal.Index.SILVER/USD`: Lazer ID `3154`, Core/Hermes ID
     `0x6afca47e1c79ecd0d2844327efb463478417a9d498dfb0915ce5e88ae6d952d3`.
   - Future Gold: `Metal.Index.GOLD/USD`: Lazer ID `3153`, Core/Hermes ID
     `0xfa0f57505be633c026896e15afef2c7ce2cf8ff9a45349d1da737f4f01266b01`.
4. Request the 200 ms Lazer stream, one-minute History, point/range history,
   publisher updates, Hermes/Benchmarks, and history from each index's launch.
5. Ask for the proprietary index methodology, correction/revision policy, stale
   carry-forward behavior, and exact one-minute candle boundary/close semantics used
   by Kalshi.
6. Obtain contractual rights for non-display research and automated trading, indefinite
   raw retention, internal derived research, and later private-cloud storage.

Do not substitute generic `Metal.XAG/USD` or `Metal.XAU/USD`. The contract settlement
sources identify the proprietary `Metal.Index.*` products.

Official Pyth Pro History request:

```text
GET https://pyth.dourolabs.app/v1/fixed_rate@200ms/history
  ?symbol=Metal.Index.SILVER%2FUSD
  &from=<inclusive Unix seconds>
  &to=<inclusive Unix seconds>
  &resolution=1
Authorization: Bearer $PYTH_API_KEY
```

See the [History API](https://docs.pyth.network/price-feeds/pro/api/history) and
[Lazer subscription guide](https://docs.pyth.network/price-feeds/pro/subscribe-to-prices).
Pyth requires redundant connections to all three Lazer endpoints. Since August 26,
2026, Hermes and Benchmarks also require bearer authentication; see the
[Core upgrade guide](https://docs.pyth.network/price-feeds/core/upgrade/preparing).

Published Pyth pricing currently starts at $2,500/month for Metals; FX + Metals is
$6,500/month and all asset classes are $10,000/month. Exact proprietary-index inclusion
is not guaranteed by the public table, so confirm before purchase. See
[Pyth Price Feeds](https://www.pyth.network/price-feeds?cluster=mainnet-beta) and the
[Core upgrade pricing announcement](https://www.pyth.network/blog/the-pyth-core-upgrade).

## CME provider comparison

### Recommendation: Databento first

Use Databento `GLBX.MDP3` MBO for SI, SIL, and SIC, plus definitions, status, and
statistics. It is the best fit for queue-aware replay because it provides every order
event and ID, venue sequence, matching-engine `ts_event`, Databento capture `ts_recv`,
and `ts_in_delta`, all at nanosecond encoding. History extends to 2010 generally and
MBO to May 2017. See [datasets](https://databento.com/docs/knowledge-base/datasets),
[MBO schema](https://databento.com/docs/schemas-and-data-formats/mbo), and
[timestamps/formats](https://databento.com/docs/schemas-and-data-formats).

Published tiers:

| Tier | Published price | Relevant included history | Main limitation |
| --- | ---: | --- | --- |
| Standard | $199/month, monthly | 1 month L2/L3; 1 year L1 | Described for personal use/up to two devices; automated alternative-venue use needs written classification |
| Plus | $1,750/month, annual | 16+ years L1 | L2/L3 beyond the included window remains separately priced |
| Unlimited | $4,500/month, annual | 16+ years all schemas | Higher fixed cost and annual commitment |

Older historical data is usage-priced through Databento Batch/API, so there is no
single accurate static total. See [Databento pricing](https://databento.com/pricing/).
Before relying on Standard, obtain written Databento/CME confirmation that research,
raw retention, internal access, later cloud storage, and signals driving Kalshi orders
are permitted. CME may classify the eventual use as alternative-venue non-display
trading rather than personal research.

### Interactive Brokers

Pros:

- Very inexpensive if eligible: the US Securities Snapshot and Futures Value Bundle is
  $10/month, and the PLUS depth add-on is $5/month. Individual non-professional COMEX
  L1 is $1.55/month and L2 is $12.10/month; professional COMEX L1/L2 is $145/month.
- Useful as a secondary live-price check and possible future hedge-execution route.

Cons:

- API depth lacks an authoritative exchange timestamp, IB receive timestamp, and native
  sequence.
- Historical time-and-sales is capped at 1,000 ticks per request and there is no
  historical L2 suitable for queue replay.
- IBKR explicitly does not position itself as a specialized historical data provider.

See [IBKR market-data pricing](https://www.interactivebrokers.com/en/pricing/market-data-pricing.php),
[market depth](https://ibkrcampus.com/docs/tws-api/doc/market-data-live/market-depth-l-2/introduction),
and [historical limitations](https://interactivebrokers.github.io/tws-api/historical_limitations.html).
Use IBKR only as a secondary check, not the canonical R0/R1 source.

### dxFeed

Pros:

- Direct CME MDP-derived futures coverage, live/historical replay, full order depth,
  candles, and a Python API.
- Credible alternative if its custom contract supplies needed timestamps and retention.

Cons:

- Retail named-application prices—$99/month for all-four-exchange depth or $39/month
  for top-of-book; $39/$19 for COMEX—do not license a custom Python collector.
- Custom Python/B2B live, history, retention, and automated-use rights are quote-only.
- Public documentation does not clearly guarantee a live vendor-receive timestamp or
  native CME packet/channel sequence.

See [CME futures coverage](https://dxfeed.com/market-data/futures/cme/),
[Python API](https://dxfeed.com/api/python-api/), and
[business pricing request](https://choose.dxfeed.com/business/).

### Direct CME MDP 3.0

Pros:

- Maximum control: native dual-feed UDP/SBE, packet/channel sequence, exchange time,
  unique order IDs, full depth, and true queue position.

Cons:

- Requires A/B reconciliation, gap recovery, snapshots, schema/channel operations,
  clock/NIC engineering, connectivity, hardware, and mandatory AutoCert+.
- Higher compliance and operating burden than a normalized vendor feed.

Current published per-exchange illustrative fees are $610/month for a real-time data
feed plus $363/month for one basic Research & Analysis non-display application, or
$1,820/month for one basic Alternative Venue trading application. That implies roughly
$973/month for research or $2,430/month once driving Kalshi trading, before connectivity,
hardware, and other charges; CME must determine the actual classification. See the
[June 2026 fee list](https://www.cmegroup.com/market-data/files/june-2026-market-data-fee-list.pdf),
[MDP platform](https://www.cmegroup.com/market-data/distributor/market-data-platform.html),
and [MBO FAQ](https://www.cmegroup.com/articles/faqs/market-by-order-mbo.html).

Direct CME is a future upgrade only if measured lead/latency economics justify it.

## Silver instruments

Start with SI, SIL, and the newer 100-ounce SIC contract. SI is 5,000 oz, SIL is
1,000 oz, and SIC is 100 oz. Gold-ready mappings include GC, MGC, 1OZ, and QO. See
[CME Micro Metals](https://www.cmegroup.com/markets/microsuite/metals.html) and
[100-ounce Silver](https://www.cmegroup.com/markets/metals/precious/100-ounce-silver.contractSpecs.html).
