# Metals implementation status

Updated 2026-08-29. The authoritative requirements are in
[STRATEGY_SPEC.md](STRATEGY_SPEC.md).

## Current decision

The project is **NO-GO for trading**. Development, historical research, read-only
collection, deterministic replay, and shadow simulation are allowed. No metals module
may submit, amend, or cancel an exchange order.

## External access

| Source | Decision | Current blocker |
| --- | --- | --- |
| Kalshi | Production read-only API and WebSocket | Account/API-key issue pending Kalshi Support |
| Pyth | Exact `Metal.Index.SILVER/USD` feed | Exposed key must remain revoked; replacement access, license, and one-minute timestamp semantics pending |
| CME | Databento `GLBX.MDP3` | Account, license, API key, and observed timestamp validation pending |

Direct CME connectivity is deferred unless the measured S3 lead is too short for
Databento delivery and the incremental economics justify direct-feed cost.

## Risk profiles

The owner-approved **shadow stress envelope** is:

- Hypothetical bankroll range: $5,000–$10,000.
- Maximum counterfactual order: 250 contracts.
- Maximum counterfactual net outcome exposure: 1,000 contracts per market.
- Maximum aggregate worst-case loss: $3,000 across unsettled positions and all
  potentially filled resting shadow orders.
- Maximum worst-case loss: $1,000 per market.
- Daily stop: $3,000, evaluated as realized P&L plus current worst-case loss.
- Execution mode: shadow; live orders disabled.

These are simulation boundaries, not pilot approval. The strategy specification's
controlled-pilot defaults—10 contracts per order, 100 net contracts per market, $250
aggregate worst-case loss, and a $250 daily stop—remain a physically separate future
profile and may only be reconsidered after all prior rollout gates pass.

## Implemented foundation

- Immutable hash-chained raw and normalized JSONL storage, plus exclusive native DBN
  archives as the authoritative field-lossless CME source.
- Atomic manifests with running/completed/failed state, sanitized errors, counts,
  output paths, Git state, and a deterministic public-config hash.
- Canonical deduplicated DuckDB views plus explicit all-run provenance views.
- Databento normalization for MBO, trades, definitions, status, and statistics,
  retaining vendor timestamps, sequence/channel/order identifiers, raw fixed-point
  values, and local receipt clocks.
- Databento recorder orchestration and CLI with no exchange execution path.
- Non-blocking Databento callback handoff, off-thread normalization, symbol/control
  record retention, feed-close monitoring, and classified reconnect gaps.
- Shadow order lifecycle, fills, cancels, binary terminal P&L, conservative resting
  order and fee exposure, risk limits, and kill-switch blocking.
- Fail-closed Phase 1 data-gate evaluation. Unmeasured coverage cannot pass.

## Required gate sequence

1. Complete source access, license, and timestamp-semantic confirmation.
2. Record at least 10 complete synchronized trading days.
3. Reconstruct opening targets and settlements for at least 1,000/all usable windows
   with 100% agreement.
4. Pass Kalshi, Pyth, CME, clock, gap-classification, metadata, and fee data gates.
5. Complete R1 capacity research and Phase 2 fair-value research.
6. Complete the deterministic Q0/Q1/Q2 event simulator and economic gates.
7. Run a frozen build in shadow for at least five complete trading days with replay
   agreement and tested kill switches.

Only a separate controlled-pilot review can authorize real money after those steps.
