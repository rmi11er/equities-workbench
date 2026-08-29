# Metals research pipeline

This directory implements Phase 0 and the R0/R1 foundation from the authoritative
Metals Strategy Spec. Silver is enabled first. Gold mappings are present but disabled.
Nothing in `src/metals/` can create, amend, cancel, or otherwise submit an order.

## Current state

- Public Kalshi discovery, settled-market metadata, results, and trades are collectible.
- Live Kalshi order-book, trade, lifecycle, and proxied-Pyth collection is implemented,
  but requires a currently valid Kalshi API key.
- Pyth Pro one-minute History collection is implemented and awaits `PYTH_API_KEY`.
- Databento `GLBX.MDP3` MBO/trade/definition/status/statistics collection and
  normalized storage are implemented; access and licensing are still pending.
- Settlement audit logic exists, but no result is usable until Pyth confirms candle
  timestamp semantics and the exact-source checksum passes.
- A shadow-only risk and order-lifecycle foundation enforces the approved research
  envelope. It has no exchange execution dependency and refuses live-order config.
- Phase 0/1 and every trading go/no-go gate remain incomplete. See
  [STATUS.md](STATUS.md).

Generated data is written below `data/metals/`, which is intentionally gitignored:

```text
data/metals/
  raw/databento_native/date=YYYY-MM-DD/metal=<metal>/run=<run-id>/*.dbn
  raw/<source>/date=YYYY-MM-DD/metal=<metal>/run=<run-id>/
  normalized/<table>/date=YYYY-MM-DD/metal=<metal>/run=<run-id>/
  normalized_parquet/<table>/date=YYYY-MM-DD/metal=<metal>/run=<run-id>/
  derived/daily_quality/date=YYYY-MM-DD/run=<run-id>/
  manifests/run=<run-id>/manifest.json
  metals.duckdb
```

Raw and normalized JSONL segments are created exclusively, never reopened for append,
and hash-chain every record. REST records retain the exact response body. WebSocket
records retain both exact source text and parsed fields. Databento additionally writes
an exclusive native DBN archive as the authoritative lossless CME source; extracted
JSON and normalized tables are written on a dedicated thread and are reproducible from
that DBN file. A queue overflow or unexpected feed close fails the run. Parquet is a
derived Zstandard representation and can be regenerated from normalized JSONL.

## Setup

1. Copy `configs/metals.example.toml` to the ignored `configs/metals.toml` and keep
   Silver enabled.
2. Create a fresh production Kalshi API key. The two pre-existing local keys currently
   return `401 authentication_error / NOT_FOUND`. Put the key ID and PEM path in an
   ignored credentials TOML or use `KALSHI_API_KEY_ID` and
   `KALSHI_PRIVATE_KEY_PATH`.
3. Obtain a Pyth Pro trial key as described in [DATA_ACCESS.md](DATA_ACCESS.md) and set
   `PYTH_API_KEY` only in the server environment.
4. Arrange CME data licensing before enabling automated collection or trading use.
5. Use Python 3.10+ and install `requirements.txt`; the Databento client dependency is
   environment-gated because the macOS system Python 3.9 cannot install current releases.

## Commands

All commands are read-only with respect to Kalshi:

```bash
PYTHONPATH=. python3 -m src.metals.cli --config configs/metals.toml discover
PYTHONPATH=. python3 -m src.metals.cli --config configs/metals.toml check-access
PYTHONPATH=. python3 -m src.metals.cli --config configs/metals.toml collect --duration-seconds 60
PYTHONPATH=. python3 -m src.metals.cli --config configs/metals.toml collect-databento --metal silver --duration-seconds 60
PYTHONPATH=. python3 -m src.metals.cli --config configs/metals.toml backfill-kalshi --metal silver
PYTHONPATH=. python3 -m src.metals.cli --config configs/metals.toml export-parquet
```

After obtaining Pyth access, use inclusive Unix-second bounds:

```bash
PYTH_API_KEY=... PYTHONPATH=. python3 -m src.metals.cli \
  --config configs/metals.toml backfill-pyth-history \
  --metal silver --from-seconds 0 --to-seconds 0
```

Replace the zero bounds with the earliest audited market open and latest close. The
collector chunks requests by day. Pyth's TradingView `t` candle label remains marked
`unconfirmed_tradingview_t`; do not run a production settlement checksum until Pyth
confirms whether it denotes candle open or end and its exact boundary inclusion rule.

## Safety and provenance

- The Kalshi client rejects every HTTP method except `GET` before network or key access.
- Order-book sequence is tracked per `(connection_id, sid)`, with one SID per market.
- Duplicates are recorded raw but never applied twice.
- A gap invalidates the affected book and requests a new WebSocket snapshot.
- REST snapshots are never stitched into an active WebSocket stream.
- Prices and quantities remain decimal strings; integer-cent fields are not used.
- `expected_expiration_time` is not misreported as determination time.
- Every run captures public configuration, git state, Python version, and platform.
- Databento symbol mappings, system/error records, and reconnect gap bounds are
  retained; reconnect gaps are classified in the daily quality report.

## Test and export

```bash
PYTHONPATH=. python3 -m pytest -q tests/metals
PYTHONPATH=. python3 -m src.metals.cli --config configs/metals.toml export-parquet
```

The DuckDB catalog exposes one view per normalized table, such as `market_dim`,
`kalshi_trade`, `kalshi_book_event`, `kalshi_settlement`, and `pyth_candle`.
Canonical analytical views deduplicate repeated collection runs. Each table also has
`<table>_all_runs` and `<table>_raw` views that retain full provenance.
