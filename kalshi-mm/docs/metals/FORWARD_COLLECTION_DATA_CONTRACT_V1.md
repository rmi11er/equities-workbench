# Forward collection data contract v1

## Purpose

This contract preserves model-independent source data so future fair-value models can
be regenerated without recollecting Kalshi observations. Derived fair and long-vol
rows are conveniences with explicit input lineage; they are never the authoritative
data source.

## Authoritative raw inputs

Every Kalshi WebSocket and REST response is retained append-only under `raw/kalshi`.
Each record contains the exact source text, parsed payload, local wall-clock receive
time, local monotonic receive time, process-global ingestion sequence, connection and
subscription identity, exchange sequence, and gap status. Files are hash-chained and
listed in a completed run manifest.

Normalized records retain:

- Every order-book snapshot level and ordered delta, including recovery snapshot ID,
  validity flag, source sequence, ingestion sequence, and complete book-state hash.
- Every trade with side, price, fractional quantity, source time when supplied, local
  receive clocks, and ingestion sequence.
- Every exact Kalshi-carried Pyth settlement-index update with symbol, value, source
  timestamp, Kalshi timestamp, local receive clocks, and ingestion sequence.
- Every metadata version with market window, target, settlement source, rounding,
  native price grid, receipt time, content hash, and version number.

These streams are sufficient to reconstruct the exact Kalshi book and baseline fair
on any later causal decision grid.

## Derived fair rows

`forward_fair_decision` records both Gaussian and empirical-CDF probabilities and the
inputs needed to reproduce them:

- Immutable baseline study and model version.
- Exact index value, source time, receive time, age, and ingestion sequence.
- Metadata hash and receipt time.
- Target, rounding digits, time bucket, time-of-day variance, remaining variance, and
  seconds remaining.
- Trigger ingestion sequence and full book-state hash.

The online baseline remains time-of-day-only. That is a versioned property, not an
assumption silently imposed on future models.

## Long-vol candidate rows

`long_vol_candidate` is emitted for every throttled fair evaluation, including
rejections. Each record contains:

- Deterministic candidate ID and frozen candidate-policy version.
- Trigger type and every exact/book/metadata lineage key.
- Window, strike, rounding, BBO prices, top quantities, recovery snapshot ID, and
  book age.
- Gaussian, empirical, and side-specific conservative external fairs.
- Executable underdog bid/ask, declared taker fee, gross edge, fee-net edge,
  uncertainty charge, adverse-selection charge, and fully charged edge.
- Descriptive implied-volatility ratio and continuous eligibility duration.
- Separate screen, shadow-policy, and execution eligibility flags with complete
  rejection-reason arrays.
- An unconditional `execution_eligible=false` and `trading_eligible=false` while the
  fast updater and fill-selection model remain unfrozen.

Kalshi price selects the side and executable entry but is never used inside the
external fair.

## Completeness audit

`python -m src.metals.forward_collection_audit` fails unless:

- Collector and shadow manifests completed without errors and are correctly linked.
- Every manifest output exists and every JSONL hash chain verifies.
- Raw, fair, and candidate counts agree with their manifests.
- Candidate IDs are unique and fair/candidate row counts agree.
- Every candidate joins to its trigger, metadata version, exact-index update, and
  exact book-state hash.
- Maximum input sequences are causal and no candidate enables execution.

The 60-second smoke capture passed this audit as collector
`20260902T003143.353510Z` and shadow `20260902T003143.431485Z`.

## Data that cannot be manufactured later

The current governed run records the exact settlement index and full Kalshi event
stream. Pyth `Metal.XAG/USD` feed 345 live entitlement passed a no-retention probe on
all three endpoints, but retained live use is not enabled without a reviewed provider
approval. Its 200 ms source-price history can be backfilled after each run, but that
does not recreate host receive latency.

Databento live CME data is also unavailable under the current subscription. Historical
SI can be joined by source time later; it cannot recreate the live host-to-exchange
transport path.

Therefore this contract prevents recollection for terminal-distribution, Kalshi-book,
settlement, opportunity, and fill-selection research. A later licensed synchronized
Pyth/SI collection is still required for measured cross-provider transport latency.

## Retention and labeling

- Never overwrite or mutate a completed raw run.
- Backfills receive separate run/request IDs and record the exact requested time
  bounds, feed, schema, and cost ceiling.
- Settlement and future-markout labels are appended as separate derived artifacts and
  joined by market ticker and causal timestamps.
- Model changes create new derived versions from the same raw capture.
- Whole Kalshi windows remain the independence unit; confidence intervals are blocked
  by trading day.
