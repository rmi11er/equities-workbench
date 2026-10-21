"""Historical fast-feed counterfactual for one forward-shadow capture.

The report asks a deliberately narrow execution question: if a faster external
silver observation had been available while a shadow quote was resting, would the
quote still have met the *original* frozen edge requirement?  It does not refit the
terminal distribution, infer fills at touch, or make an order eligible for trading.

Historical Pyth rows have source publication timestamps but no observation of this
host's live receive timestamp.  The replay therefore reports a grid of explicit
transport and cancellation latency assumptions.  The generic XAG feed is anchored
to each causally received exact Kalshi-carried Pyth index value and contributes only
its subsequent price change; it is never substituted for settlement.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from .baseline_fair import BaselineVolModel, load_baseline_vol_model
from .forward_shadow_report import (
    ForwardShadowReportError,
    _json_rows,
    _stream_rows,
)

NSEC = 1_000_000_000
MSEC = 1_000_000
ONE = Decimal("1")
ZERO = Decimal("0")
SCHEMA_VERSION = "metals-fast-feed-counterfactual/v1"
DEFAULT_TRANSPORT_LATENCIES_MS = (0, 100, 250, 500)
DEFAULT_CANCEL_LATENCIES_MS = (0, 50, 100, 250, 500)


@dataclass(frozen=True)
class ProxyPrice:
    publish_ts_ns: int
    value: Decimal


@dataclass(frozen=True)
class ExactPrice:
    source_ts_ns: int
    receive_ts_ns: int
    value: Decimal


@dataclass(frozen=True)
class FastSnapshot:
    available_ts_ns: int
    source_ts_ns: int
    estimated_index: Decimal
    source: str


@dataclass(frozen=True)
class PredictionObservation:
    exact_receive_ts_ns: int
    stale_absolute_error: Decimal
    fast_absolute_error: Decimal


@dataclass(frozen=True)
class FillObservation:
    logical_quote_id: str
    market_ticker: str
    side: str
    placement_ts_ns: int
    fill_ts_ns: int
    quantity: Decimal
    quote_price: Decimal
    frozen_charge: Decimal
    markout_1s: Decimal | None
    markout_5s: Decimal | None


def _decimal(value: Any, label: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except Exception as error:  # noqa: BLE001 - normalized inputs vary
        raise ForwardShadowReportError(f"invalid decimal {label}: {value!r}") from error
    if not parsed.is_finite():
        raise ForwardShadowReportError(f"non-finite decimal {label}: {value!r}")
    return parsed


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _quantiles(values: Iterable[Decimal]) -> dict[str, str | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}

    def pick(fraction: float) -> Decimal:
        index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
        return ordered[index]

    return {
        "count": len(ordered),
        "mean": _decimal_text(sum(ordered, ZERO) / len(ordered)),
        "p50": _decimal_text(pick(0.50)),
        "p95": _decimal_text(pick(0.95)),
        "max": _decimal_text(ordered[-1]),
    }


def simulate_fast_snapshots(
    exact_prices: Sequence[ExactPrice],
    proxy_prices: Sequence[ProxyPrice],
    *,
    proxy_transport_latency_ms: int,
    beta: Decimal = ONE,
) -> tuple[list[FastSnapshot], list[PredictionObservation]]:
    """Replay an anchored proxy using only values available by wall-clock time."""

    if proxy_transport_latency_ms < 0:
        raise ValueError("proxy transport latency must be nonnegative")
    if not beta.is_finite():
        raise ValueError("beta must be finite")
    events: list[tuple[int, int, str, ExactPrice | ProxyPrice]] = []
    for row in exact_prices:
        events.append((row.receive_ts_ns, 1, "exact", row))
    latency_ns = proxy_transport_latency_ms * MSEC
    for row in proxy_prices:
        events.append((row.publish_ts_ns + latency_ns, 0, "proxy", row))
    events.sort(key=lambda item: (item[0], item[1]))

    current_proxy: ProxyPrice | None = None
    exact_anchor: ExactPrice | None = None
    proxy_anchor: ProxyPrice | None = None
    snapshots: list[FastSnapshot] = []
    predictions: list[PredictionObservation] = []

    def estimate() -> Decimal | None:
        if exact_anchor is None or proxy_anchor is None or current_proxy is None:
            return None
        return exact_anchor.value + beta * (current_proxy.value - proxy_anchor.value)

    for available_ts_ns, _, kind, event in events:
        if kind == "proxy":
            assert isinstance(event, ProxyPrice)
            current_proxy = event
            current_estimate = estimate()
            if current_estimate is not None:
                snapshots.append(
                    FastSnapshot(
                        available_ts_ns=available_ts_ns,
                        source_ts_ns=event.publish_ts_ns,
                        estimated_index=current_estimate,
                        source="proxy_delta",
                    )
                )
            continue

        assert isinstance(event, ExactPrice)
        prior_estimate = estimate()
        if exact_anchor is not None and prior_estimate is not None:
            predictions.append(
                PredictionObservation(
                    exact_receive_ts_ns=event.receive_ts_ns,
                    stale_absolute_error=abs(event.value - exact_anchor.value),
                    fast_absolute_error=abs(event.value - prior_estimate),
                )
            )
        exact_anchor = event
        proxy_anchor = current_proxy
        snapshots.append(
            FastSnapshot(
                available_ts_ns=available_ts_ns,
                source_ts_ns=event.source_ts_ns,
                estimated_index=event.value,
                source="exact_anchor",
            )
        )

    snapshots.sort(key=lambda row: (row.available_ts_ns, row.source != "proxy_delta"))
    return snapshots, predictions


def _probability_yes(
    model: BaselineVolModel,
    market: dict[str, Any],
    snapshot: FastSnapshot,
) -> Decimal | None:
    close_ts_ns = int(market["close_ts_ns"])
    remaining_seconds = math.ceil((close_ts_ns - snapshot.source_ts_ns) / NSEC)
    if not 1 <= remaining_seconds <= 900:
        return None
    open_ts_ns = int(market["open_ts_ns"])
    bucket = int((open_ts_ns // NSEC % 86_400) // 900)
    value = model.probability_yes(
        current_index=float(snapshot.estimated_index),
        strike=float(market["target_price"]),
        remaining_seconds=remaining_seconds,
        bucket=bucket,
        ewma_variance=model.global_variance_15m,
        round_digits=int(market["round_digits"]),
        distribution="gaussian",
    )
    return Decimal(str(value))


def _successful_fairs(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("yes_probability") is None or row.get("decision_ts_ns") is None:
            continue
        grouped.setdefault(str(row["market_ticker"]), []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["decision_ts_ns"]))
    return grouped


def _side_fair(row: dict[str, Any], side: str) -> Decimal:
    yes = _decimal(row["yes_probability"], "yes_probability")
    return yes if side == "yes" else ONE - yes


def _markout(
    fair_rows: Sequence[dict[str, Any]],
    *,
    fill_ts_ns: int,
    side: str,
    seconds: int,
) -> Decimal | None:
    times = [int(row["decision_ts_ns"]) for row in fair_rows]
    current_index = bisect.bisect_right(times, fill_ts_ns) - 1
    future_index = bisect.bisect_left(times, fill_ts_ns + seconds * NSEC)
    if current_index < 0 or future_index >= len(fair_rows):
        return None
    return _side_fair(fair_rows[future_index], side) - _side_fair(
        fair_rows[current_index], side
    )


def _build_fill_observations(
    *,
    fill_rows: Sequence[dict[str, Any]],
    trade_rows: Sequence[dict[str, Any]],
    lifecycle_rows: Sequence[dict[str, Any]],
    fair_rows: Sequence[dict[str, Any]],
) -> list[FillObservation]:
    trades = {
        int(row["ingestion_sequence"]): row
        for row in trade_rows
        if row.get("ingestion_sequence") is not None
    }
    resting = {
        str(row["logical_quote_id"]): row
        for row in lifecycle_rows
        if row.get("event_type") == "order_resting" and row.get("logical_quote_id")
    }
    fairs = _successful_fairs(fair_rows)
    fair_by_key = {
        (str(row["market_ticker"]), int(row["decision_ts_ns"])): row
        for rows in fairs.values()
        for row in rows
    }
    observations: list[FillObservation] = []
    for fill in fill_rows:
        quote_id = str(fill.get("logical_quote_id", ""))
        placement = resting.get(quote_id)
        sequence = fill.get("source_trade_ingestion_sequence")
        trade = trades.get(int(sequence)) if sequence is not None else None
        if placement is None or trade is None:
            raise ForwardShadowReportError(f"cannot join fill {quote_id!r}")
        ticker = str(fill["market_ticker"])
        side = str(fill["passive_outcome_side"]).lower()
        placement_ts_ns = int(placement["event_ts_ns"])
        fair = fair_by_key.get((ticker, placement_ts_ns))
        if fair is None:
            raise ForwardShadowReportError(f"cannot join placement fair {quote_id!r}")
        proposal = next(
            (
                value
                for value in fair.get("proposals", [])
                if value.get("outcome_side") == side
            ),
            None,
        )
        if proposal is None:
            raise ForwardShadowReportError(f"cannot join placement proposal {quote_id!r}")
        frozen_charge = _decimal(
            proposal["fair_probability"], "proposal fair_probability"
        ) - _decimal(proposal["bid_ceiling"], "proposal bid_ceiling")
        fill_ts_ns = int(trade["receive_ts_ns"])
        ticker_fairs = fairs.get(ticker, [])
        observations.append(
            FillObservation(
                logical_quote_id=quote_id,
                market_ticker=ticker,
                side=side,
                placement_ts_ns=placement_ts_ns,
                fill_ts_ns=fill_ts_ns,
                quantity=_decimal(fill["fill_quantity_fp"], "fill quantity"),
                quote_price=_decimal(fill["quote_price_dollars"], "quote price"),
                frozen_charge=frozen_charge,
                markout_1s=_markout(
                    ticker_fairs, fill_ts_ns=fill_ts_ns, side=side, seconds=1
                ),
                markout_5s=_markout(
                    ticker_fairs, fill_ts_ns=fill_ts_ns, side=side, seconds=5
                ),
            )
        )
    return observations


def _warning_time(
    fill: FillObservation,
    snapshots: Sequence[FastSnapshot],
    snapshot_times: Sequence[int],
    market: dict[str, Any],
    model: BaselineVolModel,
) -> int | None:
    first_after_placement = bisect.bisect_right(snapshot_times, fill.placement_ts_ns)
    current_index = first_after_placement - 1
    candidates: list[FastSnapshot] = []
    if current_index >= 0:
        candidates.append(snapshots[current_index])
    end = bisect.bisect_right(snapshot_times, fill.fill_ts_ns)
    candidates.extend(snapshots[first_after_placement:end])
    for snapshot in candidates:
        yes = _probability_yes(model, market, snapshot)
        if yes is None:
            continue
        side_fair = yes if fill.side == "yes" else ONE - yes
        if side_fair - fill.frozen_charge < fill.quote_price:
            return max(fill.placement_ts_ns, snapshot.available_ts_ns)
    return None


def _weighted_cents(
    rows: Sequence[FillObservation], field: str
) -> str | None:
    eligible = [row for row in rows if getattr(row, field) is not None]
    quantity = sum((row.quantity for row in eligible), ZERO)
    if quantity <= ZERO:
        return None
    value = sum(
        (getattr(row, field) * row.quantity for row in eligible), ZERO
    ) / quantity
    return _decimal_text(value * 100)


def _markout_attribution(
    fills: Sequence[FillObservation],
    avoided: Sequence[FillObservation],
    kept: Sequence[FillObservation],
    field: str,
) -> dict[str, Any]:
    eligible = [row for row in fills if getattr(row, field) is not None]
    eligible_ids = {id(row) for row in eligible}
    avoided_eligible = [row for row in avoided if id(row) in eligible_ids]
    kept_eligible = [row for row in kept if id(row) in eligible_ids]
    quantity = sum((row.quantity for row in eligible), ZERO)

    def total(rows: Sequence[FillObservation]) -> Decimal:
        return sum(
            (getattr(row, field) * row.quantity for row in rows),
            ZERO,
        )

    original_total = total(eligible)
    avoided_total = total(avoided_eligible)
    kept_total = total(kept_eligible)
    if original_total != avoided_total + kept_total:
        raise ForwardShadowReportError("markout attribution does not reconcile")
    original_loss = max(ZERO, -original_total)
    avoidable_loss = max(ZERO, -avoided_total)
    return {
        "original_markout_total_dollars": _decimal_text(original_total),
        "avoided_group_markout_total_dollars": _decimal_text(avoided_total),
        "kept_markout_total_dollars": _decimal_text(kept_total),
        "slow_feed_avoidable_loss_dollars": _decimal_text(avoidable_loss),
        "slow_feed_avoidable_loss_cents_per_original_contract": (
            _decimal_text(avoidable_loss / quantity * 100) if quantity else None
        ),
        "slow_feed_avoidable_loss_fraction_of_original": (
            float(avoidable_loss / original_loss) if original_loss else None
        ),
        "residual_loss_dollars": _decimal_text(max(ZERO, -kept_total)),
        "residual_markout_cents_per_original_contract": (
            _decimal_text(kept_total / quantity * 100) if quantity else None
        ),
    }


def _scenario_summary(
    fills: Sequence[FillObservation],
    warnings: Sequence[int | None],
    *,
    cancel_latency_ms: int,
) -> dict[str, Any]:
    if cancel_latency_ms < 0:
        raise ValueError("cancel latency must be nonnegative")
    cancel_ns = cancel_latency_ms * MSEC
    avoided = [
        row
        for row, warning in zip(fills, warnings)
        if warning is not None and warning + cancel_ns < row.fill_ts_ns
    ]
    kept = [row for row in fills if row not in avoided]
    too_late = [
        row
        for row, warning in zip(fills, warnings)
        if warning is not None and warning + cancel_ns >= row.fill_ts_ns
    ]
    leads = [
        Decimal(row.fill_ts_ns - warning) / MSEC
        for row, warning in zip(fills, warnings)
        if warning is not None
    ]
    total_quantity = sum((row.quantity for row in fills), ZERO)
    avoided_quantity = sum((row.quantity for row in avoided), ZERO)

    def adverse(field: str, rows: Sequence[FillObservation]) -> list[FillObservation]:
        return [
            row
            for row in rows
            if getattr(row, field) is not None and getattr(row, field) < ZERO
        ]

    adverse_1s = adverse("markout_1s", fills)
    adverse_5s = adverse("markout_5s", fills)
    avoided_ids = {id(row) for row in avoided}
    return {
        "cancel_latency_ms": cancel_latency_ms,
        "warned_fill_events": sum(warning is not None for warning in warnings),
        "warnings_too_late_events": len(too_late),
        "avoided_fill_events": len(avoided),
        "avoided_fill_fraction": len(avoided) / len(fills) if fills else None,
        "avoided_quantity": _decimal_text(avoided_quantity),
        "avoided_quantity_fraction": (
            float(avoided_quantity / total_quantity) if total_quantity else None
        ),
        "warning_lead_ms": _quantiles(leads),
        "adverse_1s_fill_events": len(adverse_1s),
        "adverse_1s_avoided_events": sum(id(row) in avoided_ids for row in adverse_1s),
        "adverse_5s_fill_events": len(adverse_5s),
        "adverse_5s_avoided_events": sum(id(row) in avoided_ids for row in adverse_5s),
        "original_markout_1s_cents_per_contract": _weighted_cents(
            fills, "markout_1s"
        ),
        "kept_markout_1s_cents_per_contract": _weighted_cents(kept, "markout_1s"),
        "original_markout_5s_cents_per_contract": _weighted_cents(
            fills, "markout_5s"
        ),
        "kept_markout_5s_cents_per_contract": _weighted_cents(kept, "markout_5s"),
        "markout_1s_attribution": _markout_attribution(
            fills, avoided, kept, "markout_1s"
        ),
        "markout_5s_attribution": _markout_attribution(
            fills, avoided, kept, "markout_5s"
        ),
    }


def _prediction_summary(rows: Sequence[PredictionObservation]) -> dict[str, Any]:
    if not rows:
        return {
            "observations": 0,
            "stale_mae_usd": None,
            "fast_mae_usd": None,
            "mae_improvement_fraction": None,
            "updates_improved_fraction": None,
        }
    stale = [row.stale_absolute_error for row in rows]
    fast = [row.fast_absolute_error for row in rows]
    stale_mae = sum(stale, ZERO) / len(stale)
    fast_mae = sum(fast, ZERO) / len(fast)
    return {
        "observations": len(rows),
        "stale_mae_usd": _decimal_text(stale_mae),
        "fast_mae_usd": _decimal_text(fast_mae),
        "mae_improvement_fraction": (
            float((stale_mae - fast_mae) / stale_mae) if stale_mae else None
        ),
        "updates_improved_fraction": mean(
            row.fast_absolute_error < row.stale_absolute_error for row in rows
        ),
    }


def _latest_market_rows(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = str(row.get("market_ticker", ""))
        prior = latest.get(ticker)
        if prior is None or int(row.get("metadata_version", 0)) >= int(
            prior.get("metadata_version", 0)
        ):
            latest[ticker] = row
    return latest


def _load_proxy_rows(
    root: Path,
    *,
    proxy_metal: str,
    proxy_feed_id: int,
    start_ts_ns: int,
    end_ts_ns: int,
) -> tuple[list[ProxyPrice], list[str]]:
    paths = sorted(
        (root / "normalized" / "pyth_exact_history" / f"metal={proxy_metal}").glob(
            "request=*/page-*.jsonl"
        )
    )
    rows = _json_rows(paths)
    selected: dict[int, ProxyPrice] = {}
    input_paths: set[str] = set()
    for row in rows:
        if int(row.get("price_feed_id", -1)) != proxy_feed_id:
            continue
        timestamp = int(row["source_event_ts_ns"])
        if not start_ts_ns <= timestamp <= end_ts_ns:
            continue
        price = ProxyPrice(timestamp, _decimal(row["price"], "proxy price"))
        prior = selected.get(timestamp)
        if prior is not None and prior != price:
            raise ForwardShadowReportError("conflicting proxy prices at one timestamp")
        selected[timestamp] = price
        historical_request_id = str(row.get("historical_request_id", ""))
        if historical_request_id:
            input_paths.add(historical_request_id)
    return [selected[key] for key in sorted(selected)], sorted(input_paths)


def build_fast_feed_counterfactual_report(
    root: str | Path,
    collector_run_id: str,
    shadow_run_id: str,
    *,
    baseline_model_path: str | Path,
    metal: str = "silver",
    proxy_metal: str = "silver-xag-proxy",
    proxy_feed_id: int = 345,
    proxy_beta: Decimal = ONE,
    transport_latencies_ms: Sequence[int] = DEFAULT_TRANSPORT_LATENCIES_MS,
    cancel_latencies_ms: Sequence[int] = DEFAULT_CANCEL_LATENCIES_MS,
) -> dict[str, Any]:
    storage_root = Path(root)
    model, study_id = load_baseline_vol_model(Path(baseline_model_path))
    market_rows = _stream_rows(storage_root, "market_dim", collector_run_id, metal=metal)
    exact_rows = _stream_rows(
        storage_root, "kalshi_pyth_value", collector_run_id, metal=metal
    )
    trade_rows = _stream_rows(storage_root, "kalshi_trade", collector_run_id, metal=metal)
    fair_rows = _stream_rows(
        storage_root, "forward_fair_decision", shadow_run_id, metal=metal
    )
    fill_rows = _stream_rows(
        storage_root, "shadow_fill_attribution", shadow_run_id, metal=metal
    )
    lifecycle_rows = _stream_rows(
        storage_root, "order_lifecycle", shadow_run_id, metal=metal
    )
    if not exact_rows or not fill_rows:
        raise ForwardShadowReportError("capture has no exact Pyth rows or fills")
    exact = sorted(
        (
            ExactPrice(
                int(row["source_ts_ns"]),
                int(row["receive_ts_ns"]),
                _decimal(row["value_usd"], "exact Pyth value"),
            )
            for row in exact_rows
        ),
        key=lambda row: row.receive_ts_ns,
    )
    start_ts_ns = min(row.source_ts_ns for row in exact)
    end_ts_ns = max(row.source_ts_ns for row in exact)
    proxy, request_ids = _load_proxy_rows(
        storage_root,
        proxy_metal=proxy_metal,
        proxy_feed_id=proxy_feed_id,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
    )
    if not proxy:
        raise ForwardShadowReportError("no proxy rows cover the exact Pyth interval")
    markets = _latest_market_rows(market_rows)
    fills = _build_fill_observations(
        fill_rows=fill_rows,
        trade_rows=trade_rows,
        lifecycle_rows=lifecycle_rows,
        fair_rows=fair_rows,
    )
    missing_markets = sorted({row.market_ticker for row in fills} - set(markets))
    if missing_markets:
        raise ForwardShadowReportError(
            f"fill markets have no metadata: {', '.join(missing_markets)}"
        )

    gaps_ms = [
        Decimal(right.publish_ts_ns - left.publish_ts_ns) / MSEC
        for left, right in zip(proxy, proxy[1:])
    ]
    scenarios: list[dict[str, Any]] = []
    prediction_quality: list[dict[str, Any]] = []
    for transport_ms in transport_latencies_ms:
        snapshots, predictions = simulate_fast_snapshots(
            exact,
            proxy,
            proxy_transport_latency_ms=int(transport_ms),
            beta=proxy_beta,
        )
        snapshot_times = [row.available_ts_ns for row in snapshots]
        warnings = [
            _warning_time(
                fill,
                snapshots,
                snapshot_times,
                markets[fill.market_ticker],
                model,
            )
            for fill in fills
        ]
        prediction_quality.append(
            {
                "proxy_transport_latency_ms": int(transport_ms),
                **_prediction_summary(predictions),
            }
        )
        for cancel_ms in cancel_latencies_ms:
            scenarios.append(
                {
                    "proxy_transport_latency_ms": int(transport_ms),
                    **_scenario_summary(
                        fills, warnings, cancel_latency_ms=int(cancel_ms)
                    ),
                }
            )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "collector_run_id": collector_run_id,
        "shadow_run_id": shadow_run_id,
        "metal": metal,
        "baseline_study_id": study_id,
        "baseline_model_path": str(Path(baseline_model_path)),
        "execution_mode": "research_only",
        "trading_eligible": False,
        "method": {
            "proxy_symbol": "Metal.XAG/USD",
            "proxy_feed_id": proxy_feed_id,
            "proxy_archive_metal": proxy_metal,
            "proxy_beta": _decimal_text(proxy_beta),
            "anchor": "latest causally received exact Kalshi-carried Pyth index",
            "estimate": "exact_anchor + beta * (current_proxy - proxy_at_anchor)",
            "quote_test": "counterfactual side fair minus frozen original charge below resting quote price",
            "historical_receive_time": "proxy publish timestamp plus declared scenario latency",
        },
        "coverage": {
            "exact_updates": len(exact),
            "proxy_unique_updates": len(proxy),
            "proxy_first_source_ts_ns": proxy[0].publish_ts_ns,
            "proxy_last_source_ts_ns": proxy[-1].publish_ts_ns,
            "proxy_source_gap_ms": _quantiles(gaps_ms),
            "proxy_gaps_over_300ms": sum(value > Decimal("300") for value in gaps_ms),
            "proxy_max_gap_ms": _decimal_text(max(gaps_ms) if gaps_ms else None),
            "proxy_request_ids": request_ids,
            "fill_events": len(fills),
            "fill_quantity": _decimal_text(sum((row.quantity for row in fills), ZERO)),
        },
        "prediction_quality": prediction_quality,
        "scenarios": scenarios,
        "limitations": [
            "Historical Pyth proxy rows do not contain this host's original live receive timestamps.",
            "Latency results are counterfactual scenarios, not observed live transport performance.",
            "Generic Metal.XAG/USD is used only as an anchored delta proxy and cannot certify settlement.",
            "Beta is fixed before replay and is not fitted to this capture.",
            "The next-exact-update error diagnostic compares a current proxy estimate with an exact value stamped earlier; lower assumed proxy latency can therefore look worse when it moves beyond that older target.",
            "Strict-penetration shadow fills do not reproduce queue position or exchange cancellation races.",
            "The capture contains only three complete settlement windows.",
        ],
    }
    identity = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_id"] = hashlib.sha256(identity).hexdigest()
    return report


def write_fast_feed_counterfactual_report(
    root: str | Path,
    collector_run_id: str,
    shadow_run_id: str,
    **kwargs: Any,
) -> Path:
    report = build_fast_feed_counterfactual_report(
        root, collector_run_id, shadow_run_id, **kwargs
    )
    destination = (
        Path(root)
        / "derived"
        / "fast_feed_counterfactual"
        / f"report={report['report_id'][:24]}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "report.json"
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ForwardShadowReportError(f"refusing to overwrite different report: {path}")
    path.write_text(payload, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/metals")
    parser.add_argument("--collector-run-id", required=True)
    parser.add_argument("--shadow-run-id", required=True)
    parser.add_argument("--baseline-model-path", required=True)
    parser.add_argument("--metal", default="silver")
    parser.add_argument("--proxy-metal", default="silver-xag-proxy")
    parser.add_argument("--proxy-feed-id", type=int, default=345)
    args = parser.parse_args(argv)
    path = write_fast_feed_counterfactual_report(
        args.root,
        args.collector_run_id,
        args.shadow_run_id,
        baseline_model_path=args.baseline_model_path,
        metal=args.metal,
        proxy_metal=args.proxy_metal,
        proxy_feed_id=args.proxy_feed_id,
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
