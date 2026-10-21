# Source-access preflights

These commands are opt-in, bounded, and read-only. They do not enable a source in
`configs/metals.toml`, collect market data, create a Databento batch job, open a live
stream, or write a provider response to disk. Output is a sanitized JSON status report;
credentials, response bodies, account objects, and third-party exception messages are
never emitted.

Use the project-pinned Python 3.11 environment:

```bash
uv sync
```

Export `PYTH_API_KEY` and `DATABENTO_API_KEY` into the process environment before
running the checks. Do not pass either secret as a command-line argument.

## Kalshi

The Kalshi check uses the configured ignored credential file or environment variables
and makes exactly two authenticated GET requests:

```bash
uv run metals-research --config configs/metals.toml preflight-kalshi
```

It checks `/account/limits` and `/account/endpoint_costs` but discards both account
payloads. Output contains only status classifications and exception class names; API
key IDs, private-key contents, account objects, provider bodies, and endpoint-cost
details are not emitted. The legacy `check-access` command is now a secret-safe alias
for this preflight.

## Pyth

Choose a Unix-second timestamp within an audited silver trading period:

```bash
uv run metals-research --config configs/metals.toml \
  preflight-pyth --metal silver --history-at-seconds <UNIX_SECONDS>
```

The command makes exactly two bounded requests:

1. A five-minute Pyth History request for `Metal.Index.SILVER/USD`.
2. A Pyth Pro Router `latest_price` request for feed `3154`.

It classifies HTTP `401` as a rejected credential and `403` as denied entitlement.
Other request, transport, rate-limit, and provider failures remain distinct. A successful
Router check validates the configured feed through Pyth Pro REST; it does not test all
three redundant WebSocket endpoints. The separate live probe does so without retaining
messages. It requires an explicit acknowledgement because it transiently accesses
licensed live data, runs each endpoint for at most 30 seconds, and passes only after a
valid update for the configured numeric feed is received from every endpoint:

```bash
uv run metals-research --config configs/metals.toml \
  preflight-pyth-lazer-live --metal silver --duration-seconds 5 \
  --allow-live-entitlement-probe
```

Do not pass the acknowledgement until the account owner approves the live-access
probe. No current Pyth WebSocket probe has been run.

The request shapes follow Pyth's official
[History API](https://docs.pyth.network/price-feeds/pro/api/history) and
[REST API](https://docs.pyth.network/price-feeds/pro/api/rest) documentation.

## Databento

Choose a UTC date on which the configured futures roots should resolve:

```bash
uv run metals-research --config configs/metals.toml \
  preflight-databento --metal silver --probe-date YYYY-MM-DD
```

The command checks, in order:

1. Whether the optional `databento` package is installed.
2. Whether the key can call the authenticated historical dataset catalog.
3. Whether `GLBX.MDP3` is listed.
4. Whether the account's historical dataset range exposes all configured schemas.
5. Whether `SI.FUT`, `SIL.FUT`, and `SIC.FUT` resolve on the chosen date.

Only Databento metadata and symbology methods are called. The report therefore labels
live entitlement `not_checked`; validate that later with the separately bounded live
stream probe. The method choices follow Databento's official
[Historical metadata API](https://databento.com/docs/api-reference-historical/metadata/metadata-list-datasets).

The live probe is a separate, explicit operation because it transiently accesses
licensed live data. It subscribes only to an MBO snapshot, runs for at most 30 seconds,
discards every callback record, writes nothing, and reports success only after at least
one record is observed. It never substitutes for written license confirmation:

```bash
uv run metals-research --config configs/metals.toml \
  preflight-databento-live --metal silver --duration-seconds 5 \
  --allow-live-entitlement-probe
```

Do not pass the acknowledgement until the account owner approves this live-access
probe. A timeout with no record remains `not_checked`, not a passing entitlement.

Both commands return exit code `0` only when every claim in their tested scope passes;
otherwise they return `2`. A `not_checked` live entitlement does not cause the
historical Databento preflight to fail.

## Clock

The synchronized collector requires a running local Chrony daemon. This preflight is
read-only, writes no artifact, and invokes only `chronyc -n tracking`:

```bash
uv run metals-research --config configs/metals.toml preflight-clock
```

The sanitized report contains synchronization status, absolute offset, stratum, and
leap status; raw Chrony output is not printed. Exit code `0` requires a synchronized
daemon. The production collector separately preserves exact probe output in immutable
raw storage, detects wall-clock versus monotonic-clock elapsed discontinuities, and
shares the market collectors' global receive sequencer. It never runs a Chrony command
that changes the system clock.

## Latest result

On 2026-08-30, the bounded silver checks produced:

- Kalshi authenticated read access: secret-safe preflight stopped before network I/O
  because the environment contains a key ID and private-key path but the referenced
  private-key file does not exist; account/API-key resolution with Kalshi Support
  remains pending.
- Pyth generic `Metal.XAG/USD` and `Metal.XAU/USD`: the configured trial key returned
  HTTP 200 from both bounded History and Pro Router requests, proving the credential is
  valid for the advertised demo feeds.
- Pyth exact `Metal.Index.SILVER/USD` History and Pro Router feed 3154: HTTP 403
  (`entitlement_denied`). The remaining failure is exact-index entitlement, not an
  unknown credential problem; generic XAG cannot certify settlement reconstruction.
- Pyth Lazer WebSockets: guarded no-retention probe implemented but not authorized or
  run.
- Databento Historical: credential accepted; `GLBX.MDP3`, every configured schema, and
  `SI.FUT`/`SIL.FUT`/`SIC.FUT` passed metadata and symbology checks.
- Databento live entitlement: not checked.
- The guarded no-retention live probe is implemented but has not been authorized or
  run, so it has incurred no live-data access or download.
- Clock: the current Mac development host has no `chronyc`, so the sanitized probe
  fails. A collection host with an active, healthy Chrony daemon is required; a one-off
  SNTP diagnostic is not accepted as gate evidence.
- A metadata-only estimate for the aligned 2026-08-28 00:00–00:10 UTC silver request
  across definition, MBO, trades, status, and statistics was $0.006635680794.
- After the owner confirmed $125 of remaining Databento credits, a separate
  cost-capped stream request for definition, trades, status, and statistics over the
  same ten-minute range completed against a $0.001750774681 estimate. It retained
  native DBN/Zstd plus 2,424 normalized records (1,266 definition, 319 trade, 659
  status, and 180 statistics records). The request's hard cost ceiling was $0.01.

These results do not establish contractual rights for non-display research, DBN
retention, model development, cloud storage, or later alternative-venue use. Preserve
written provider confirmation separately before enabling collection.

## Freezing provider approval

Preflights and Databento cost estimation are deliberately separate from collection:
they may run while provider approval is pending because they retain no provider market
data. Live and production-readiness paths require a verified immutable provider
artifact. Historical exploratory downloads may instead use the narrower immutable
assumption gate described in
[`HISTORICAL_RESEARCH_ASSUMPTIONS.md`](HISTORICAL_RESEARCH_ASSUMPTIONS.md); it cannot
authorize live access, orders, settlement certification, or daily readiness.

Copy `configs/provider_approval.example.json` to an ignored review file. Replace only
the version fields, confirmations, operation approvals, and SHA-256 evidence hashes
that written review supports. Do not embed secrets or evidence-document contents.
Materialize the result offline:

```bash
uv run metals-research --config configs/metals.toml \
  materialize-provider-approval --approval reviewed-provider-approval.json
```

Set `[governance].provider_approval_path` to the returned `approval_path`, then verify
each intended operation explicitly, for example:

```bash
uv run metals-research --config configs/metals.toml \
  check-provider-approval --provider databento --operation historical
```

The check returns exit code `2` when the path, content hash, manifest, expiry, license,
timestamp semantics, operation approval, or entitlement evidence is insufficient. The
current example is intentionally all-pending and the local config path remains empty.
