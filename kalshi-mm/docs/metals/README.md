# Metals research pipeline

Historical-only exploratory assumptions are governed separately from production
provider evidence; see
[`HISTORICAL_RESEARCH_ASSUMPTIONS.md`](HISTORICAL_RESEARCH_ASSUMPTIONS.md).

This directory implements Phase 0 and the R0/R1 foundation from the authoritative
Metals Strategy Spec. Silver is enabled first. Gold mappings are present but disabled.
Nothing in `src/metals/` can create, amend, cancel, or otherwise submit an order.

## Current state

- Public Kalshi discovery, settled-market metadata, results, and trades are collectible.
- Public Kalshi one-minute bid/ask candle backfill is implemented and resumable.
- Live Kalshi order-book, trade, lifecycle, and proxied-Pyth collection is implemented,
  but requires a currently valid Kalshi API key.
- Pyth Pro one-minute History, bounded exact 200 ms price-range archives, and
  redundant live Lazer collection are implemented and await licensed access. Exact
  ranges retain successful bodies byte-for-byte with resumable integrity checks; live
  Lazer collection remains config-disabled by default.
- Databento `GLBX.MDP3` MBO/trade/definition/status/statistics collection and
  normalized storage are implemented. Bounded Historical/Batch archives are
  cost-gated and resumable; production access and licensing remain external gates.
- Read-only Chrony clock telemetry is implemented as a required synchronized sidecar.
  Exact probe output and normalized health samples share the global receive sequencer;
  a missing sample, failed probe, unsynchronized daemon, or wall/monotonic-clock
  discontinuity fails closed. The collection host must provide a running Chrony daemon.
- Settlement audit logic consumes only completed, integrity-verified Pyth 200 ms range
  archives that are listed by a completed collection run with a provider approval
  valid at acquisition time. It requires complete preceding-minute coverage, selects the last
  publication strictly before each boundary, preserves Kalshi rule text and Pyth
  carry-forward age, and classifies incomplete evidence without counting it. No result
  is usable until the reviewed semantics confirmations and exact-source checksum pass.
- A shadow-only risk and order-lifecycle foundation enforces the approved research
  envelope. It has no exchange execution dependency and refuses live-order config.
- A preliminary offline S1 candle study provides chronological calibration and three
  explicit fill sensitivities. It is a rejection screen, not the required Q0/Q1/Q2
  event-driven backtest, and it cannot authorize capital.
- Phase 0/1 and every trading go/no-go gate remain incomplete. See
  [STATUS.md](STATUS.md).
- The incremental model-development sequence is frozen behind the data gates in
  [POST_DATA_RESEARCH_QUEUE.md](POST_DATA_RESEARCH_QUEUE.md).
- Existing-data comparator and natural-target results are recorded in
  [EXPLORATORY_FAIR_SIGNAL_LEDGER.md](EXPLORATORY_FAIR_SIGNAL_LEDGER.md); model source
  ownership and fee-aware quote equations are frozen in
  [THREE_MODEL_ARCHITECTURE.md](THREE_MODEL_ARCHITECTURE.md).
- The live-data, exchange-isolated operating path is in the
  [forward-shadow runbook](FORWARD_SHADOW_RUNBOOK.md).

Generated data is written below `data/metals/`, which is intentionally gitignored:

```text
data/metals/
  raw/databento_native/date=YYYY-MM-DD/metal=<metal>/run=<run-id>/*.dbn
  raw/databento_history/date=YYYY-MM-DD/metal=<metal>/request=<request-id>/*.dbn.zst
  raw/<source>/date=YYYY-MM-DD/metal=<metal>/run=<run-id>/
  normalized/databento_history/date=YYYY-MM-DD/metal=<metal>/request=<request-id>/*.jsonl
  normalized/pyth_exact_history/metal=<metal>/request=<request-id-prefix>/*.jsonl
  normalized/<table>/date=YYYY-MM-DD/metal=<metal>/run=<run-id>/
  normalized_parquet/<table>/date=YYYY-MM-DD/metal=<metal>/run=<run-id>/
  derived/daily_gate_evidence/date=YYYY-MM-DD/evidence=<sha256>/evidence.json
  derived/daily_quality/date=YYYY-MM-DD/report=<sha256>/report.json
  derived/daily_validation/date=YYYY-MM-DD/snapshot=<sha256>/snapshot.json
  derived/settlement_check/audit=<sha256>/{windows.jsonl,summary.json,manifest.json}
  derived/settlement_proxy_pilot/run=<run-id>/{rows.jsonl,anomalies.jsonl,report.json,REPORT.md}
  policies/provider_approval/approval=<sha256>/{approval.json,manifest.json}
  plans/pyth_settlement_exact/plan=<sha256>/{plan.json,manifest.json}
  derived/s1_feasibility/run=<run-id>/{report.json,REPORT.md,signals.csv,executions.csv,markouts.csv}
  derived/toxicity_diagnostics/run=<run-id>/{report.json,REPORT.md,*.csv,*.png}
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
5. Freeze reviewed Pyth and Databento license, entitlement, and timestamp evidence as
   described below. Live and production-readiness collection remains fail-closed
   while `[governance].provider_approval_path` is empty or any required confirmation
   is false, missing, expired, or fails integrity verification. Exploratory historical
   downloads may use the separate research-assumption gate linked above.
6. Install and synchronize Chrony on the eventual collection host. Set `[clock]`
   `enabled = true` only after `preflight-clock` passes; the probe reads daemon state
   and never changes the system clock.
7. Install `uv`; the checked `.python-version` and `uv.lock` select Python 3.11 and
   the exact tested dependency set.

## Commands

All commands are read-only with respect to Kalshi:

```bash
uv run --locked python -m src.metals.cli --config configs/metals.toml discover
uv run --locked python -m src.metals.cli --config configs/metals.toml preflight-kalshi
uv run --locked python -m src.metals.cli --config configs/metals.toml preflight-clock
uv run --locked python -m src.metals.cli --config configs/metals.toml preflight-pyth-lazer-live --metal silver --duration-seconds 5 --allow-live-entitlement-probe
uv run --locked python -m src.metals.cli --config configs/metals.toml materialize-provider-approval --approval reviewed-provider-approval.json
uv run --locked python -m src.metals.cli --config configs/metals.toml check-provider-approval --provider pyth --operation exact_history
uv run --locked python -m src.metals.cli --config configs/metals.toml backfill-pyth-exact-history --metal silver --start-timestamp-us <inclusive-us> --end-timestamp-us <inclusive-us>
uv run --locked python -m src.metals.cli --config configs/metals.toml plan-pyth-settlement-history --metal silver --audit data/metals/derived/settlement_check/audit=<sha256>
uv run --locked python -m src.metals.cli --config configs/metals.toml backfill-pyth-settlement-history --plan data/metals/plans/pyth_settlement_exact/plan=<sha256> --max-boundaries 100 --allow-network-download
uv run --locked python -m src.metals.cli --config configs/metals.toml audit-settlements --metal silver --semantics reviewed-settlement-semantics.json
uv run --locked python -m src.metals.cli --config configs/metals.toml preflight-databento --metal silver --probe-date 2026-08-28
uv run --locked python -m src.metals.cli --config configs/metals.toml preflight-databento-live --metal silver --duration-seconds 5 --allow-live-entitlement-probe
uv run --locked python -m src.metals.cli --config configs/metals.toml collect --duration-seconds 60
uv run --locked python -m src.metals.cli --config configs/metals.toml collect-databento --metal silver --duration-seconds 60
uv run --locked python -m src.metals.cli --config configs/metals.toml collect-pyth-lazer --metal silver --duration-seconds 60
uv run --locked python -m src.metals.cli --config configs/metals.toml collect-clock-telemetry --metal silver --duration-seconds 60
uv run --locked python -m src.metals.cli --config configs/metals.toml collect-synchronized --metal silver --duration-seconds 60
uv run --locked python -m src.metals.cli --config configs/metals.toml backfill-kalshi --metal silver
uv run --locked python -m src.metals.cli --config configs/metals.toml backfill-kalshi-candles --metal silver
uv run --locked python -m src.metals.cli --config configs/metals.toml export-parquet
uv run --locked python -m src.metals.cli --config configs/metals.toml run-s1-feasibility --metal silver
uv run --locked python -m src.metals.cli --config configs/metals.toml run-toxicity-diagnostics --metal silver
uv run --locked python -m src.metals.cli --config configs/metals.toml run-settlement-proxy-pilot --metal silver --labels <labels.jsonl> --databento-trades <trades.jsonl> --pyth-proxy-candles <pyth-proxy-candle.jsonl> --start-ts-ns <inclusive-ns> --end-ts-ns <exclusive-ns>
uv run --locked python -m src.metals.cli --config configs/metals.toml run-baseline-scorecard --baseline-study-id <sha256> --kalshi-candlesticks <kalshi-candles.jsonl>
uv run --locked python -m src.metals.cli --config configs/metals.toml run-natural-target-signals --baseline-study-id <sha256> --silver-pyth-proxy-candles <silver-pyth.jsonl> --gold-pyth-proxy-candles <gold-pyth.jsonl> --databento-trades <trades.jsonl>
uv run --locked python -m src.metals.cli --config configs/metals.toml run-forward-shadow --baseline-model <model.json> --duration-seconds 3600 --allow-live-shadow-data
.venv/bin/python -m src.metals.forward_shadow_report --root data/metals --collector-run-id <collector> --shadow-run-id <shadow> --settlement-run-id <targeted-kalshi-backfill>
.venv/bin/python -m src.metals.forward_long_vol_report --root data/metals --collector-run-id <collector> --shadow-run-id <shadow> --settlement-run-id <targeted-kalshi-backfill>
.venv/bin/python -m src.metals.combined_forward_long_vol_report --root data/metals --report <forward-long-vol-report> --report <next-forward-long-vol-report>
uv run --locked python -m src.metals.cli --config configs/metals.toml check-data-readiness
```

### Provider approval gate

Authentication does not establish permission to retain or use provider data. Copy
`configs/provider_approval.example.json` to an ignored review file. For each provider,
record versioned written confirmation of license scope and timestamp semantics; for
each operation, separately record entitlement evidence and set `approved` only after
review. The JSON contains SHA-256 hashes of those evidence documents, never credentials
or confidential document contents. Keep the source documents in the separately
controlled compliance record.

Materialize the reviewed JSON offline, then copy the returned content-addressed
`approval_path` into `[governance].provider_approval_path` in the ignored
`configs/metals.toml`:

```bash
uv run --locked python -m src.metals.cli --config configs/metals.toml \
  materialize-provider-approval --approval reviewed-provider-approval.json
uv run --locked python -m src.metals.cli --config configs/metals.toml \
  check-provider-approval --provider pyth --operation exact_history
uv run --locked python -m src.metals.cli --config configs/metals.toml \
  check-provider-approval --provider pyth --operation live_lazer
uv run --locked python -m src.metals.cli --config configs/metals.toml \
  check-provider-approval --provider databento --operation historical
uv run --locked python -m src.metals.cli --config configs/metals.toml \
  check-provider-approval --provider databento --operation live
```

Materialization only freezes what was reviewed; it does not turn false confirmations
into approvals. Source preflights and the Databento cost estimate remain available
without this artifact because they retain no provider market data. Every live or
production-readiness path verifies it before credential use, network collection, or
output creation. A historical exploratory collector can use the narrower assumption
artifact only when the provider path is empty; its manifest is permanently marked
research-only and is ineligible for production provenance. Collection manifests bind
the governing ID, and a resumed Databento request refuses any authority-ID mismatch.

Before the day starts, review the fee and incentive records, compute each record's
canonical `config_hash`, place those hashes in the policy's approved lists, and
materialize all three artifacts. Registry retrieval timestamps must be no later than
the measurement start, so a rule learned after the fact cannot pass as live-known:

```bash
uv run --locked python -m src.metals.cli hash-registry-config --record reviewed-fee.json
uv run --locked python -m src.metals.cli --config configs/metals.toml \
  materialize-daily-policy-registries --date YYYY-MM-DD \
  --policy reviewed-policy.json --fee-registry reviewed-fees.json \
  --incentive-registry reviewed-incentives.json
```

After a synchronized collector produces one unified collector report, derive the
evidence-backed report. The materialized policy and collector report must both be
listed outputs of completed manifests, and every contributing source manifest must be
supplied:

```bash
uv run --locked python -m src.metals.cli --config configs/metals.toml \
  measure-daily-gates --date YYYY-MM-DD \
  --quality-report data/metals/derived/daily_quality/date=YYYY-MM-DD/run=<run-id>/report.json \
  --policy data/metals/policies/daily-gates/date=YYYY-MM-DD/policy=<sha256>/policy.json \
  --manifest data/metals/manifests/run=<synchronized-run-id>/manifest.json \
  --manifest data/metals/manifests/run=daily-policy-registry-<sha256>/manifest.json
```

Then freeze the content-addressed measured report with the derived manifest and every
source manifest:

```bash
uv run --locked python -m src.metals.cli --config configs/metals.toml \
  freeze-daily-validation --date YYYY-MM-DD \
  --quality-report data/metals/derived/daily_quality/date=YYYY-MM-DD/report=<sha256>/report.json \
  --manifest data/metals/manifests/run=daily-gates-<sha256>/manifest.json \
  --manifest data/metals/manifests/run=<source-run-id>/manifest.json
```

Measurement verifies completed manifests and every relevant JSONL hash chain, derives
time-weighted source coverage, exact-settlement agreement, clock health, metadata, fee,
and incentive evidence, and fails closed when evidence is absent. The snapshot then
recomputes the measurement from its frozen source manifests, rejects hand-authored
passing reports, re-evaluates all gates, and hashes the measured report, manifests,
provider-approval artifacts, and every declared output. `check-data-readiness` verifies those checksums
again and passes only after ten distinct UTC dates have an eligible immutable snapshot
and the selected dates contain at least 1,000 exact, mutually agreeing settlement
checks. Multiple snapshots for one date never multiply the count. Failed days are
still frozen with their reasons and never enter research inputs. Every expected window
must resolve to either one usable exact audit or one classified exclusion; excluded,
missing, invalid, and conflicting audits never increase the 1,000-settlement count.

`collect-synchronized` is the required burn-in path. It refuses to start unless Pyth,
CME, and local Chrony telemetry are intentionally enabled and Kalshi credentials
resolve. It shares one receive sequencer across the three market-data collectors and
the clock sidecar, and fails the parent run if any child stops early or never
contributes to the sequence. Its bounded duration must be positive and finite. Keep it
disabled until the provider-access, license, clock, and timestamp-semantics gates are
resolved.

`preflight-clock` executes only `chronyc -n tracking` against the host's local Chrony
daemon. It writes nothing, changes no clock state, and emits only a sanitized status,
absolute offset, stratum, and leap status. The current Mac development host does not
have `chronyc`; its failed probe is not admissible clock evidence. A one-off SNTP query
is also not accepted as synchronized-run telemetry.

Pyth Lazer live collection requires an intentional local `enabled = true` under
`[pyth]`; the checked example remains disabled. The command is always time-bounded,
defaults to 60 seconds, and rejects non-positive or non-finite durations:

```bash
PYTH_API_KEY=... uv run --locked python -m src.metals.cli \
  --config configs/metals.toml collect-pyth-lazer \
  --metal silver --duration-seconds 60
```

Exact reconstruction uses the numeric `pyth_lazer_id` and the official
`fixed_rate@200ms/price/range` endpoint. Each request has inclusive microsecond bounds
no more than 60 seconds apart; the adapter follows opaque cursors, stores only
successful bodies, and verifies every artifact before resuming. Keep `[pyth].enabled`
false while using this historical command:

```bash
PYTH_API_KEY=... uv run --locked python -m src.metals.cli \
  --config configs/metals.toml backfill-pyth-exact-history --metal silver \
  --start-timestamp-us <inclusive-us> --end-timestamp-us <inclusive-us>
```

For the complete settlement universe, first build an offline plan from a verified,
content-addressed settlement audit. The planner performs no provider call. It removes
windows that cannot become usable from exact history alone, deduplicates shared opening
and settlement boundaries, and freezes every inclusive `[boundary-60s, boundary-1µs]`
request and its market uses:

```bash
uv run --locked python -m src.metals.cli --config configs/metals.toml \
  plan-pyth-settlement-history --metal silver \
  --audit data/metals/derived/settlement_check/audit=<sha256> --max-pages 2
```

The current verified Silver audit produces plan
`781818064d3e8336595b2f4018844c936e9fd236f7d6d88fec5565941d83c670`:
1,987 structurally eligible windows collapse to 1,998 unique boundary ranges. Four
malformed contracts are retained in the plan's classified-invalid list rather than
causing provider requests. At the two-page per-range ceiling, one complete pass is
bounded by 3,996 provider pages; the expected 200 ms observation count fits in one
1,000-row page per range.

Only after entitlement is confirmed, execute a bounded slice. The network command is
sequential, requires an explicit acknowledgement, re-verifies completed ranges before
skipping them, repairs only deterministic interrupted local state, and emits a
terminal run manifest. Rerun it until `remaining_boundaries` is zero:

```bash
PYTH_API_KEY=... uv run --locked python -m src.metals.cli \
  --config configs/metals.toml backfill-pyth-settlement-history \
  --plan data/metals/plans/pyth_settlement_exact/plan=<sha256> \
  --max-boundaries 100 --request-interval-seconds 0.1 \
  --allow-network-download
```

`--max-boundaries` bounds ranges per invocation; the frozen plan's `max_pages` bounds
each range. No bulk request has been sent while the current Pyth credential remains
403-denied.

After exact ranges cover the minute immediately preceding every opening and closing
boundary, copy `configs/settlement_semantics.example.json`, attach the SHA-256 hashes of
the reviewed timestamp/correction/rounding evidence, and change each confirmation only
after review. The example is intentionally unconfirmed and therefore cannot produce a
usable settlement. Run the offline audit with an optional reviewed universe file when
proving “all available” instead of the 1,000-window minimum:

```bash
uv run --locked python -m src.metals.cli --config configs/metals.toml \
  audit-settlements --metal silver \
  --semantics reviewed-settlement-semantics.json \
  --universe reviewed-completed-market-universe.json
```

The command has no network path and takes the numeric exact-history feed ID from the
reviewed product configuration. It emits a content-addressed audit plus a completed
standard run manifest; include that run manifest in `measure-daily-gates`. Legacy
TradingView `pyth_candle` files are permanently ineligible for this checksum.

### Databento historical runbook

Keep `[cme].enabled = false`; that switch controls live collection and is not needed for
bounded history. First estimate the exact inclusive-start/exclusive-end UTC request.
This command calls only Databento cost metadata and cannot download records:

```bash
uv run --locked python -m src.metals.cli --config configs/metals.toml \
  estimate-databento-history --metal silver \
  --start 2026-08-28T00:00:00Z --end 2026-08-29T00:00:00Z
```

The default estimate includes every schema in `[cme].schemas`, including MBO. For the
controlled baseline, estimate only the lower-cost schema family first by repeating
`--schema`:

```bash
uv run --locked python -m src.metals.cli --config configs/metals.toml \
  estimate-databento-history --metal silver \
  --start 2026-08-28T00:00:00Z --end 2026-08-29T00:00:00Z \
  --schema definition --schema trades --schema status --schema statistics
```

Selections must be a unique, non-empty subset of `[cme].schemas`. The canonical schema
set is included in the request fingerprint, manifest, native and normalized archive
paths, and resume checkpoints. Use exactly the same `--schema` values when resuming;
a checkpoint whose recorded schema set differs is rejected. Bounds remain inclusive at
`--start` and exclusive at `--end`.

Review the total and per-schema estimate, then start with a small streaming probe. A
backfill cannot run without both the acknowledgement and a dollar ceiling:

```bash
uv run --locked python -m src.metals.cli --config configs/metals.toml \
  backfill-databento-history --metal silver \
  --start 2026-08-28T00:00:00Z --end 2026-08-29T00:00:00Z \
  --schema definition --schema trades --schema status --schema statistics \
  --transport stream --max-cost-usd 1.00 --allow-billable-download
```

For larger ranges, use `--transport batch`. Batch mode submits DBN/Zstd jobs split by
UTC day, then checks their state. Rerun the identical bounded command until
`"completed": true`; request fingerprints and job checkpoints prevent resubmission of
already recorded schemas. Native DBN remains authoritative, daily instrument
definitions retain contract/roll identity, and normalized JSONL is reproducible.
Treat the initial definition records for each UTC day as reference-data snapshots, not
as evidence that a contract changed at that instant; retain later definition updates
in event-time order. Batch delivery is split on UTC-day boundaries, so requests that
cross a boundary may contain another daily definition snapshot.

The candle backfill uses only Kalshi GET endpoints, persists exact raw responses, and
can resume completed market windows. Run `export-parquet` after a backfill before
running the feasibility study. See [S1_FEASIBILITY.md](S1_FEASIBILITY.md) for the
method and interpretation boundaries.

`run-toxicity-diagnostics` is a Kalshi-only exploratory analysis, not a fair-value
backtest. It produces price-by-time heatmaps, action-side markout smiles, a settlement
waterfall, paired-leg diagnostics, uncertainty and concentration cuts, and explicit
unavailable-data notes.

The legacy TradingView history command remains available for exploratory candle work,
not settlement evidence. After obtaining Pyth access, it uses inclusive Unix-second
bounds:

```bash
PYTH_API_KEY=... uv run --locked python -m src.metals.cli \
  --config configs/metals.toml backfill-pyth-history \
  --metal silver --from-seconds 0 --to-seconds 0
```

Replace the zero bounds with the earliest audited market open and latest close. The
collector chunks requests by day. Pyth's TradingView `t` candle label remains marked
`unconfirmed_tradingview_t` and is never accepted by the exact settlement audit.

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
uv run --locked pytest -q tests/metals
uv run --locked python -m src.metals.cli --config configs/metals.toml export-parquet
```

The current locked metals suite passes 525 tests.

The DuckDB catalog exposes one view per normalized table, such as `market_dim`,
`kalshi_trade`, `kalshi_book_event`, `kalshi_settlement`, and `pyth_candle`.
Canonical analytical views deduplicate repeated collection runs. Each table also has
`<table>_all_runs` and `<table>_raw` views that retain full provenance.

The isolated public-data `KXBTC15M` comparison/control and its exact coverage and
comparability limits are documented in
[BTC_15M_CONTROL.md](BTC_15M_CONTROL.md).
