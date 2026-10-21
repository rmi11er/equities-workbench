"""Deterministic diagnostics for one completed forward-shadow capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal
from itertools import pairwise
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "metals-forward-shadow-report/v1"
ZERO = Decimal(0)
ONE = Decimal(1)
MAX_SETTLEMENT_SOURCE_AGE_NS = 1_500_000_000


class ForwardShadowReportError(ValueError):
    """Raised when a capture cannot support a trustworthy report."""


def _json_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        expected_previous: str | None = None
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ForwardShadowReportError(
                        f"invalid JSON in {path}:{line_number}"
                    ) from error
                if not isinstance(row, dict):
                    raise ForwardShadowReportError(
                        f"non-object JSON in {path}:{line_number}"
                    )
                record_hash = row.get("record_sha256")
                payload = dict(row)
                payload.pop("record_sha256", None)
                canonical = json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    default=str,
                )
                computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                if record_hash != computed:
                    raise ForwardShadowReportError(
                        f"record hash mismatch in {path}:{line_number}"
                    )
                if row.get("previous_record_sha256") != expected_previous:
                    raise ForwardShadowReportError(
                        f"record chain mismatch in {path}:{line_number}"
                    )
                expected_previous = str(record_hash)
                rows.append(row)
    return rows


def _stream_rows(
    root: Path, stream: str, run_id: str, *, metal: str
) -> list[dict[str, Any]]:
    base = root / "normalized" / stream
    paths = base.glob(f"date=*/metal={metal}/run={run_id}/*.jsonl")
    return _json_rows(paths)


def _manifest(root: Path, run_id: str) -> dict[str, Any]:
    path = root / "manifests" / f"run={run_id}" / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ForwardShadowReportError(f"missing manifest: {path}") from error
    except json.JSONDecodeError as error:
        raise ForwardShadowReportError(f"invalid manifest JSON: {path}") from error
    if not isinstance(value, dict):
        raise ForwardShadowReportError(f"manifest is not an object: {path}")
    if value.get("run_id") != run_id:
        raise ForwardShadowReportError(f"manifest run identity mismatch: {path}")
    status = value.get("status")
    errors = value.get("errors") or []
    controlled_interruption = (
        status in {"cancelled", "failed"}
        and bool(errors)
        and all(error.get("type") == "CancelledError" for error in errors)
    )
    if status != "completed" and not controlled_interruption:
        raise ForwardShadowReportError(f"run is not reportable: {run_id}")
    if errors and not controlled_interruption:
        raise ForwardShadowReportError(f"run has manifest errors: {run_id}")
    return value


def _decimal(value: Any, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except Exception as error:
        raise ForwardShadowReportError(f"invalid decimal {field}: {value!r}") from error
    if not parsed.is_finite():
        raise ForwardShadowReportError(f"non-finite decimal {field}: {value!r}")
    return parsed


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _quantiles(values: Iterable[Decimal]) -> dict[str, str | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}

    def pick(fraction: Decimal) -> Decimal:
        index = int(
            (Decimal(len(ordered)) * fraction).to_integral_value(
                rounding="ROUND_CEILING"
            )
        )
        return ordered[max(0, min(len(ordered) - 1, index - 1))]

    return {
        "count": len(ordered),
        "mean": _decimal_text(sum(ordered, ZERO) / Decimal(len(ordered))),
        "p50": _decimal_text(pick(Decimal("0.50"))),
        "p95": _decimal_text(pick(Decimal("0.95"))),
        "max": _decimal_text(ordered[-1]),
    }


def _latest_metadata(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = str(row.get("market_ticker", ""))
        if not ticker:
            continue
        previous = latest.get(ticker)
        key = (
            int(row.get("metadata_version", 0)),
            int(row.get("metadata_receive_ts_ns", 0)),
        )
        previous_key = (
            (
                int(previous.get("metadata_version", 0)),
                int(previous.get("metadata_receive_ts_ns", 0)),
            )
            if previous is not None
            else (-1, -1)
        )
        if key >= previous_key:
            latest[ticker] = row
    return latest


def _skip_reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(row.get("reason", "unknown"))
                for row in rows
                if row.get("status") == "skipped"
            ).items()
        )
    )


def _action_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    actions: Counter[str] = Counter()
    for row in rows:
        value = row.get("quote_actions")
        if not isinstance(value, dict):
            continue
        for side, action in value.items():
            actions[f"{side}:{action}"] += 1
    return dict(sorted(actions.items()))


def _settlement_observation(
    pyth_rows: list[dict[str, Any]], symbol: str, close_ts_ns: int
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in pyth_rows
        if row.get("underlying_ticker") == symbol
        and int(row.get("source_ts_ns", -1)) < close_ts_ns
    ]
    return max(eligible, key=lambda row: int(row["source_ts_ns"])) if eligible else None


def _window_pyth_continuity(
    pyth_rows: list[dict[str, Any]],
    symbol: str,
    open_ts_ns: int,
    close_ts_ns: int,
) -> dict[str, Any]:
    symbol_source_ts = sorted(
        {
            int(row["source_ts_ns"])
            for row in pyth_rows
            if row.get("underlying_ticker") == symbol
            and row.get("source_ts_ns") is not None
        }
    )
    raw_updates_in_window = [
        int(row["source_ts_ns"])
        for row in pyth_rows
        if row.get("underlying_ticker") == symbol
        and row.get("source_ts_ns") is not None
        and open_ts_ns <= int(row["source_ts_ns"]) < close_ts_ns
    ]
    updates_in_window = [
        source_ts
        for source_ts in symbol_source_ts
        if open_ts_ns <= source_ts < close_ts_ns
    ]
    at_or_before_open = [
        source_ts for source_ts in symbol_source_ts if source_ts <= open_ts_ns
    ]
    before_close = [
        source_ts for source_ts in symbol_source_ts if source_ts < close_ts_ns
    ]
    start_source_ts_ns = max(at_or_before_open) if at_or_before_open else None
    end_source_ts_ns = max(before_close) if before_close else None
    continuity_points = (
        [start_source_ts_ns]
        + [source_ts for source_ts in updates_in_window if source_ts > open_ts_ns]
        if start_source_ts_ns is not None
        else updates_in_window
    )
    gaps_ms = [
        Decimal(current - previous) / Decimal(1_000_000)
        for previous, current in pairwise(continuity_points)
    ]
    gaps_over_threshold = sum(
        gap > Decimal(MAX_SETTLEMENT_SOURCE_AGE_NS) / Decimal(1_000_000)
        for gap in gaps_ms
    )
    start_age_ns = (
        open_ts_ns - start_source_ts_ns
        if start_source_ts_ns is not None
        else None
    )
    end_age_ns = (
        close_ts_ns - end_source_ts_ns if end_source_ts_ns is not None else None
    )
    capture_spans_window = bool(
        symbol_source_ts
        and symbol_source_ts[0] <= open_ts_ns
        and symbol_source_ts[-1] >= close_ts_ns
    )
    continuous = bool(
        capture_spans_window
        and start_age_ns is not None
        and start_age_ns <= MAX_SETTLEMENT_SOURCE_AGE_NS
        and end_age_ns is not None
        and 0 < end_age_ns <= MAX_SETTLEMENT_SOURCE_AGE_NS
        and gaps_over_threshold == 0
    )
    return {
        "continuous": continuous,
        "capture_spans_window": capture_spans_window,
        "maximum_allowed_gap_ms": _decimal_text(
            Decimal(MAX_SETTLEMENT_SOURCE_AGE_NS) / Decimal(1_000_000)
        ),
        "updates_in_window": len(raw_updates_in_window),
        "unique_source_timestamps_in_window": len(updates_in_window),
        "duplicate_source_timestamps_in_window": (
            len(raw_updates_in_window) - len(updates_in_window)
        ),
        "start_source_age_ms": (
            _decimal_text(Decimal(start_age_ns) / Decimal(1_000_000))
            if start_age_ns is not None
            else None
        ),
        "end_source_age_ms": (
            _decimal_text(Decimal(end_age_ns) / Decimal(1_000_000))
            if end_age_ns is not None
            else None
        ),
        "internal_source_gap_ms": _quantiles(gaps_ms),
        "gaps_over_1500ms": gaps_over_threshold,
    }


def _fill_summary(rows: list[dict[str, Any]], *, outcome: str | None) -> dict[str, Any]:
    quantities = {"yes": ZERO, "no": ZERO}
    costs = {"yes": ZERO, "no": ZERO}
    fees = {"yes": ZERO, "no": ZERO}
    for row in rows:
        side = str(row.get("passive_outcome_side", "")).lower()
        if side not in quantities:
            raise ForwardShadowReportError(f"invalid fill side: {side!r}")
        quantity = _decimal(row.get("fill_quantity_fp"), "fill_quantity_fp")
        price = _decimal(row.get("quote_price_dollars"), "quote_price_dollars")
        fee = _decimal(row.get("fill_fee_dollars"), "fill_fee_dollars")
        quantities[side] += quantity
        costs[side] += price * quantity
        fees[side] += fee
    total_quantity = quantities["yes"] + quantities["no"]
    concentration = (
        max(quantities.values()) / total_quantity if total_quantity > ZERO else ZERO
    )
    pnl: Decimal | None = None
    if outcome in {"yes", "no"}:
        payout = quantities[outcome]
        pnl = payout - sum(costs.values(), ZERO) - sum(fees.values(), ZERO)
    return {
        "fill_events": len(rows),
        "quantity": {side: _decimal_text(value) for side, value in quantities.items()},
        "notional_cost": {side: _decimal_text(value) for side, value in costs.items()},
        "fees": {side: _decimal_text(value) for side, value in fees.items()},
        "dominant_side_quantity_fraction": _decimal_text(concentration),
        "settlement_pnl_dollars": _decimal_text(pnl) if pnl is not None else None,
    }


def _fill_selection_summary(
    fill_rows: list[dict[str, Any]],
    trade_rows: list[dict[str, Any]],
    fair_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    trades = {
        int(row["ingestion_sequence"]): row
        for row in trade_rows
        if row.get("ingestion_sequence") is not None
    }
    fairs_by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fair_rows:
        if row.get("status") == "skipped" or row.get("yes_probability") is None:
            continue
        if row.get("decision_ts_ns") is None:
            continue
        fairs_by_ticker[str(row.get("market_ticker", ""))].append(row)
    fair_times: dict[str, list[int]] = {}
    for ticker, rows in fairs_by_ticker.items():
        rows.sort(key=lambda row: int(row["decision_ts_ns"]))
        fair_times[ticker] = [int(row["decision_ts_ns"]) for row in rows]

    observations: list[dict[str, Any]] = []
    for fill in fill_rows:
        sequence = fill.get("source_trade_ingestion_sequence")
        trade = trades.get(int(sequence)) if sequence is not None else None
        ticker = str(fill.get("market_ticker", ""))
        rows = fairs_by_ticker.get(ticker, [])
        times = fair_times.get(ticker, [])
        if trade is None or not rows or trade.get("receive_ts_ns") is None:
            continue
        fill_ts_ns = int(trade["receive_ts_ns"])
        current_index = bisect_right(times, fill_ts_ns) - 1
        if current_index < 0:
            continue
        current = rows[current_index]
        side = str(fill.get("passive_outcome_side", "")).lower()
        if side not in {"yes", "no"}:
            continue
        yes_probability = _decimal(current["yes_probability"], "yes_probability")
        side_fair = yes_probability if side == "yes" else ONE - yes_probability
        quantity = _decimal(fill.get("fill_quantity_fp"), "fill_quantity_fp")
        quote = _decimal(fill.get("quote_price_dollars"), "quote_price_dollars")
        fee = _decimal(fill.get("fill_fee_dollars"), "fill_fee_dollars")
        observation: dict[str, Any] = {
            "side": side,
            "quantity": quantity,
            "gross_edge": (side_fair - quote) * quantity,
            "net_edge": (side_fair - quote) * quantity - fee,
        }
        for seconds in (1, 5):
            future_index = bisect_left(times, fill_ts_ns + seconds * 1_000_000_000)
            if future_index >= len(rows):
                observation[f"markout_{seconds}s"] = None
                continue
            future_yes = _decimal(
                rows[future_index]["yes_probability"], "future yes_probability"
            )
            future_side = future_yes if side == "yes" else ONE - future_yes
            observation[f"markout_{seconds}s"] = (
                future_side - side_fair
            ) * quantity
        observations.append(observation)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        quantity = sum((row["quantity"] for row in rows), ZERO)

        def cents_per_contract(field: str) -> str | None:
            eligible = [row for row in rows if row.get(field) is not None]
            denominator = sum((row["quantity"] for row in eligible), ZERO)
            if denominator <= ZERO:
                return None
            value = sum((row[field] for row in eligible), ZERO) / denominator * 100
            return _decimal_text(value)

        return {
            "fill_events": len(rows),
            "quantity": _decimal_text(quantity),
            "gross_external_edge_cents_per_contract": cents_per_contract(
                "gross_edge"
            ),
            "net_external_edge_after_fee_cents_per_contract": cents_per_contract(
                "net_edge"
            ),
            "external_fair_markout_1s_cents_per_contract": cents_per_contract(
                "markout_1s"
            ),
            "external_fair_markout_5s_cents_per_contract": cents_per_contract(
                "markout_5s"
            ),
            "markout_1s_fill_events": sum(
                row.get("markout_1s") is not None for row in rows
            ),
            "markout_5s_fill_events": sum(
                row.get("markout_5s") is not None for row in rows
            ),
        }

    return {
        "source_fill_events": len(fill_rows),
        "joined_fill_events": len(observations),
        "all": summarize(observations),
        "yes": summarize([row for row in observations if row["side"] == "yes"]),
        "no": summarize([row for row in observations if row["side"] == "no"]),
    }


def build_forward_shadow_report(
    root: str | Path,
    collector_run_id: str,
    shadow_run_id: str,
    *,
    metal: str = "silver",
    settlement_run_id: str | None = None,
) -> dict[str, Any]:
    """Build one immutable-input diagnostic report from completed manifests."""

    storage_root = Path(root)
    collector_manifest = _manifest(storage_root, collector_run_id)
    shadow_manifest = _manifest(storage_root, shadow_run_id)
    context = shadow_manifest.get("config", {}).get("forward_shadow", {})
    if context.get("collector_run_id") != collector_run_id:
        raise ForwardShadowReportError("shadow manifest references another collector")
    if context.get("live_orders") is not False or context.get("read_only") is not True:
        raise ForwardShadowReportError(
            "capture is not manifest-declared read-only shadow"
        )

    market_rows = _stream_rows(
        storage_root, "market_dim", collector_run_id, metal=metal
    )
    pyth_rows = sorted(
        _stream_rows(storage_root, "kalshi_pyth_value", collector_run_id, metal=metal),
        key=lambda row: int(row.get("source_ts_ns", 0)),
    )
    book_rows = _stream_rows(
        storage_root, "kalshi_book_event", collector_run_id, metal=metal
    )
    trade_rows = _stream_rows(
        storage_root, "kalshi_trade", collector_run_id, metal=metal
    )
    transport_rows = _stream_rows(
        storage_root, "kalshi_transport_event", collector_run_id, metal=metal
    )
    fair_rows = _stream_rows(
        storage_root, "forward_fair_decision", shadow_run_id, metal=metal
    )
    fill_rows = _stream_rows(
        storage_root, "shadow_fill_attribution", shadow_run_id, metal=metal
    )
    lifecycle_rows = _stream_rows(
        storage_root, "order_lifecycle", shadow_run_id, metal=metal
    )
    settlement_rows: list[dict[str, Any]] = []
    if settlement_run_id is not None:
        settlement_manifest = _manifest(storage_root, settlement_run_id)
        if settlement_manifest.get("status") != "completed":
            raise ForwardShadowReportError(
                "settlement verification run must be normally completed"
            )
        settlement_rows = _stream_rows(
            storage_root, "kalshi_settlement", settlement_run_id, metal=metal
        )
    recorded_settlements = {
        str(row.get("market_ticker", "")): row for row in settlement_rows
    }
    if not pyth_rows or not fair_rows:
        raise ForwardShadowReportError("capture has no Pyth or fair-decision rows")

    metadata = _latest_metadata(market_rows)
    fair_tickers = sorted({str(row.get("market_ticker", "")) for row in fair_rows})
    max_source_ts_ns = max(int(row["source_ts_ns"]) for row in pyth_rows)
    windows: list[dict[str, Any]] = []
    for ticker in fair_tickers:
        market = metadata.get(ticker)
        if market is None:
            raise ForwardShadowReportError(f"missing metadata for fair market {ticker}")
        target_value = market.get("target_price")
        close_ts_ns = int(market["close_ts_ns"])
        symbol = str(market.get("pyth_symbol", ""))
        decisions = [row for row in fair_rows if row.get("market_ticker") == ticker]
        fills = [row for row in fill_rows if row.get("market_ticker") == ticker]
        invalid_book = [
            row
            for row in book_rows
            if row.get("market_ticker") == ticker
            and row.get("book_valid_after") is not True
        ]
        settlement_observed = max_source_ts_ns >= close_ts_ns
        pyth_continuity = _window_pyth_continuity(
            pyth_rows,
            symbol,
            int(market["open_ts_ns"]),
            close_ts_ns,
        )
        settlement = (
            _settlement_observation(pyth_rows, symbol, close_ts_ns)
            if settlement_observed
            else None
        )
        settlement_source_age_ns = (
            close_ts_ns - int(settlement["source_ts_ns"])
            if settlement is not None
            else None
        )
        settlement_label_eligible = (
            settlement_source_age_ns is not None
            and 0 < settlement_source_age_ns <= MAX_SETTLEMENT_SOURCE_AGE_NS
        )
        complete = (
            int(pyth_rows[0]["source_ts_ns"]) <= int(market["open_ts_ns"])
            and settlement_observed
            and settlement_label_eligible
        )
        rounded_value: Decimal | None = None
        outcome: str | None = None
        if (
            settlement is not None
            and target_value is not None
            and settlement_label_eligible
        ):
            digits = int(market.get("round_digits", 3))
            quantum = ONE.scaleb(-digits)
            rounded_value = _decimal(
                settlement.get("value_usd"), "settlement value"
            ).quantize(quantum, rounding=ROUND_HALF_UP)
            target = _decimal(target_value, "target_price")
            outcome = "yes" if rounded_value >= target else "no"
        recorded = recorded_settlements.get(ticker)
        recorded_result = (
            str(recorded.get("result", "")).lower() if recorded is not None else None
        )
        if recorded_result not in {"yes", "no"}:
            recorded_result = None
        recorded_value = recorded.get("expiration_value") if recorded else None
        pnl_outcome = recorded_result or outcome
        successful = [row for row in decisions if row.get("status") != "skipped"]
        pyth_ages = [
            _decimal(row["pyth_age_ms"], "pyth_age_ms")
            for row in successful
            if row.get("pyth_age_ms") is not None
        ]
        probabilities = [
            _decimal(row["yes_probability"], "yes_probability")
            for row in successful
            if row.get("yes_probability") is not None
        ]
        windows.append(
            {
                "market_ticker": ticker,
                "open_ts_ns": int(market["open_ts_ns"]),
                "close_ts_ns": close_ts_ns,
                "target_price": target_value,
                "exact_symbol": symbol,
                "complete_in_capture": complete,
                "exact_index_continuity": pyth_continuity,
                "settlement_observed": settlement_observed,
                "settlement_label_eligible": settlement_label_eligible,
                "settlement_max_source_age_ms": _decimal_text(
                    Decimal(MAX_SETTLEMENT_SOURCE_AGE_NS) / Decimal(1_000_000)
                ),
                "settlement_source_ts_ns": (
                    int(settlement["source_ts_ns"])
                    if settlement_label_eligible and settlement
                    else None
                ),
                "settlement_source_age_ms": (
                    _decimal_text(
                        Decimal(settlement_source_age_ns) / Decimal(1_000_000)
                    )
                    if settlement_label_eligible
                    and settlement_source_age_ns is not None
                    else None
                ),
                "settlement_value_unrounded": (
                    settlement.get("value_usd")
                    if settlement_label_eligible and settlement
                    else None
                ),
                "settlement_value_rounded": (
                    _decimal_text(rounded_value) if rounded_value is not None else None
                ),
                "last_preclose_pyth_diagnostic": (
                    {
                        "source_ts_ns": int(settlement["source_ts_ns"]),
                        "source_age_ms": _decimal_text(
                            Decimal(settlement_source_age_ns) / Decimal(1_000_000)
                        ),
                        "value_unrounded": settlement.get("value_usd"),
                    }
                    if settlement is not None and settlement_source_age_ns is not None
                    else None
                ),
                "provisional_outcome": outcome,
                "kalshi_recorded_result": recorded_result,
                "kalshi_expiration_value": recorded_value,
                "result_agreement": (
                    recorded_result == outcome
                    if recorded_result is not None and outcome is not None
                    else None
                ),
                "settlement_value_agreement": (
                    _decimal(recorded_value, "expiration_value") == rounded_value
                    if recorded_value is not None and rounded_value is not None
                    else None
                ),
                "decision_rows": len(decisions),
                "successful_decisions": len(successful),
                "skip_reasons": _skip_reason_counts(decisions),
                "quote_actions": _action_counts(decisions),
                "pyth_age_ms": _quantiles(pyth_ages),
                "yes_probability": _quantiles(probabilities),
                "invalid_book_rows": len(invalid_book),
                "first_invalid_book_receive_ts_ns": (
                    min(int(row["receive_ts_ns"]) for row in invalid_book)
                    if invalid_book
                    else None
                ),
                "fill_pnl_outcome_source": (
                    "kalshi_recorded_result"
                    if recorded_result is not None
                    else "strict_pre_close_pyth"
                    if outcome is not None
                    else None
                ),
                "fills": _fill_summary(fills, outcome=pnl_outcome),
            }
        )

    source_ts = [int(row["source_ts_ns"]) for row in pyth_rows]
    source_gaps_ms = [
        Decimal(current - previous) / Decimal(1_000_000)
        for previous, current in pairwise(source_ts)
    ]
    transport_ms = [
        Decimal(int(row["receive_ts_ns"]) - int(row["source_ts_ns"]))
        / Decimal(1_000_000)
        for row in pyth_rows
    ]
    completed = [row for row in windows if row["complete_in_capture"]]
    completed_pnl = sum(
        (
            _decimal(row["fills"]["settlement_pnl_dollars"], "settlement_pnl")
            for row in completed
            if row["fills"]["settlement_pnl_dollars"] is not None
        ),
        ZERO,
    )
    event_types = Counter(
        str(row.get("event_type", "unknown")) for row in lifecycle_rows
    )
    non_shadow_client_ids = sorted(
        {
            str(row["client_order_id"])
            for row in lifecycle_rows
            if row.get("client_order_id") is not None
            and not str(row["client_order_id"]).startswith("shadow:")
        }
    )
    exchange_order_ids = sorted(
        {
            str(row["exchange_order_id"])
            for row in lifecycle_rows
            if row.get("exchange_order_id") is not None
        }
    )
    transport_event_types = Counter(
        str(row.get("event_type", "unknown")) for row in transport_rows
    )
    disconnected_rows = [
        row for row in transport_rows if row.get("event_type") == "disconnected"
    ]
    reconnected_rows = [
        row for row in transport_rows if row.get("event_type") == "reconnected"
    ]
    outage_ms = [
        Decimal(int(row["outage_duration_ns"])) / Decimal(1_000_000)
        for row in reconnected_rows
        if row.get("outage_duration_ns") is not None
    ]
    recovered_disconnect_events = sum(
        int(row.get("disconnect_events_recovered", 0)) for row in reconnected_rows
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "collector_run_id": collector_run_id,
        "shadow_run_id": shadow_run_id,
        "metal": metal,
        "baseline_study_id": context.get("baseline_study_id"),
        "model_version": context.get("model_version"),
        "settings": context.get("settings"),
        "input_hash_chains_verified": True,
        "settlement_verification_run_id": settlement_run_id,
        "termination": {
            "collector_status": collector_manifest.get("status"),
            "collector_error_types": [
                error.get("type") for error in collector_manifest.get("errors", [])
            ],
            "shadow_status": shadow_manifest.get("status"),
            "shadow_error_types": [
                error.get("type") for error in shadow_manifest.get("errors", [])
            ],
            "controlled_interruption": (
                collector_manifest.get("status") != "completed"
                or shadow_manifest.get("status") != "completed"
            ),
        },
        "coverage": {
            "first_pyth_source_ts_ns": source_ts[0],
            "last_pyth_source_ts_ns": source_ts[-1],
            "pyth_updates": len(pyth_rows),
            "book_rows": len(book_rows),
            "trades": len(trade_rows),
            "fair_decisions": len(fair_rows),
            "windows_observed": len(windows),
            "completed_windows": len(completed),
            "continuous_completed_windows": sum(
                row["complete_in_capture"]
                and row["exact_index_continuity"]["continuous"]
                for row in windows
            ),
            "settlement_label_eligible_windows": sum(
                row["settlement_label_eligible"] for row in windows
            ),
        },
        "feed_health": {
            "exact_symbols": sorted(
                {str(row.get("underlying_ticker", "")) for row in pyth_rows}
            ),
            "pyth_source_gap_ms": _quantiles(source_gaps_ms),
            "pyth_gaps_over_1500ms": sum(gap > Decimal(1500) for gap in source_gaps_ms),
            "pyth_source_to_receive_ms": _quantiles(transport_ms),
            "book_gap_flags": sum(row.get("gap_flag") is True for row in book_rows),
            "book_invalid_rows": sum(
                row.get("book_valid_after") is not True for row in book_rows
            ),
            "fair_skip_reasons": _skip_reason_counts(fair_rows),
            "kalshi_transport_event_counts": dict(
                sorted(transport_event_types.items())
            ),
            "websocket_disconnect_error_types": dict(
                sorted(
                    Counter(
                        str(row.get("error_type", "unknown"))
                        for row in disconnected_rows
                    ).items()
                )
            ),
            "websocket_outage_ms": _quantiles(outage_ms),
            "websocket_disconnects_unrecovered": max(
                0, len(disconnected_rows) - recovered_disconnect_events
            ),
        },
        "execution_safety": {
            "manifest_read_only": True,
            "manifest_live_orders": False,
            "exchange_order_ids": exchange_order_ids,
            "non_shadow_client_order_ids": non_shadow_client_ids,
            "lifecycle_event_counts": dict(sorted(event_types.items())),
        },
        "fill_selection": _fill_selection_summary(fill_rows, trade_rows, fair_rows),
        "completed_window_settlement_pnl_dollars": _decimal_text(completed_pnl),
        "settlement_verification": {
            "recorded_results": sum(
                row["kalshi_recorded_result"] is not None for row in windows
            ),
            "comparable_labels": sum(
                row["kalshi_recorded_result"] is not None
                and row["provisional_outcome"] is not None
                for row in windows
            ),
            "result_agreements": sum(row["result_agreement"] is True for row in windows),
            "settlement_value_agreements": sum(
                row["settlement_value_agreement"] is True for row in windows
            ),
            "result_disagreements": sum(
                row["result_agreement"] is False for row in windows
            ),
            "settlement_value_disagreements": sum(
                row["settlement_value_agreement"] is False for row in windows
            ),
        },
        "windows": windows,
    }
    identity_payload = json.dumps(
        report, sort_keys=True, separators=(",", ":")
    ).encode()
    report["report_id"] = hashlib.sha256(identity_payload).hexdigest()
    return report


def write_forward_shadow_report(
    root: str | Path,
    collector_run_id: str,
    shadow_run_id: str,
    *,
    metal: str = "silver",
    settlement_run_id: str | None = None,
) -> Path:
    report = build_forward_shadow_report(
        root,
        collector_run_id,
        shadow_run_id,
        metal=metal,
        settlement_run_id=settlement_run_id,
    )
    destination = (
        Path(root)
        / "derived"
        / "forward_shadow_report"
        / f"report={report['report_id'][:24]}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "report.json"
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ForwardShadowReportError(
            f"refusing to overwrite different report: {path}"
        )
    path.write_text(payload, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/metals")
    parser.add_argument("--collector-run-id", required=True)
    parser.add_argument("--shadow-run-id", required=True)
    parser.add_argument("--settlement-run-id")
    parser.add_argument("--metal", default="silver")
    args = parser.parse_args(argv)
    path = write_forward_shadow_report(
        args.root,
        args.collector_run_id,
        args.shadow_run_id,
        metal=args.metal,
        settlement_run_id=args.settlement_run_id,
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
