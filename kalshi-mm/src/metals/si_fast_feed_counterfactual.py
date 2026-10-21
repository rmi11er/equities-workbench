"""SI MBO fast-feed counterfactual for one Kalshi forward-shadow capture."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

from .baseline_fair import load_baseline_vol_model
from .databento_mbo import (
    iter_hash_chained_jsonl,
    reconstruct_mbo_bbo,
    select_active_outright_contract,
)
from .fast_feed_counterfactual import (
    ExactPrice,
    ProxyPrice,
    _build_fill_observations,
    _latest_market_rows,
    _prediction_summary,
    _scenario_summary,
    _warning_time,
    simulate_fast_snapshots,
)
from .forward_shadow_report import ForwardShadowReportError, _stream_rows

SCHEMA_VERSION = "metals-si-fast-feed-counterfactual/v1"
DEFAULT_TRANSPORT_LATENCIES_MS = (0, 10, 25, 50, 100)
DEFAULT_CANCEL_LATENCIES_MS = (0, 50, 100, 250, 500)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _input_base(root: Path, request_prefix: str, metal: str) -> Path:
    matches = sorted(
        (
            root
            / "normalized"
            / "databento_history"
            / "date=2026-09-01"
            / f"metal={metal}"
        ).glob(f"request={request_prefix}*")
    )
    if len(matches) != 1:
        raise ForwardShadowReportError(
            f"expected one Databento request matching {request_prefix!r}, found {len(matches)}"
        )
    return matches[0]


def build_si_fast_feed_counterfactual_report(
    root: str | Path,
    collector_run_id: str,
    shadow_run_id: str,
    *,
    baseline_model_path: str | Path,
    databento_request_id: str,
    metal: str = "silver",
    proxy_beta: Decimal = Decimal("1"),
    transport_latencies_ms: Sequence[int] = DEFAULT_TRANSPORT_LATENCIES_MS,
    cancel_latencies_ms: Sequence[int] = DEFAULT_CANCEL_LATENCIES_MS,
) -> dict[str, Any]:
    storage_root = Path(root)
    model, study_id = load_baseline_vol_model(Path(baseline_model_path))
    input_base = _input_base(storage_root, databento_request_id[:24], metal)
    trade_path = input_base / "trades.jsonl"
    mbo_path = input_base / "mbo.jsonl"
    if not trade_path.is_file() or not mbo_path.is_file():
        raise ForwardShadowReportError("Databento trade or MBO projection is missing")

    selected = select_active_outright_contract(
        iter_hash_chained_jsonl([trade_path]), asset="SI"
    )
    reconstruction = reconstruct_mbo_bbo(
        iter_hash_chained_jsonl([mbo_path]),
        instrument_id=int(selected["instrument_id"]),
        contract_symbol=str(selected["contract_symbol"]),
    )
    if not reconstruction.snapshots:
        raise ForwardShadowReportError("selected SI contract produced no valid BBO")

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
    exact = sorted(
        (
            ExactPrice(
                int(row["source_ts_ns"]),
                int(row["receive_ts_ns"]),
                Decimal(str(row["value_usd"])),
            )
            for row in exact_rows
        ),
        key=lambda row: row.receive_ts_ns,
    )
    if not exact:
        raise ForwardShadowReportError("capture has no exact Pyth observations")
    start_ts_ns = min(row.source_ts_ns for row in exact)
    end_ts_ns = max(row.source_ts_ns for row in exact)
    bbo = list(reconstruction.snapshots)
    bbo_times = [row.ts_recv_ns for row in bbo]
    start_index = max(0, bisect.bisect_right(bbo_times, start_ts_ns) - 1)
    end_index = bisect.bisect_right(bbo_times, end_ts_ns)
    overlapping_bbo = bbo[start_index:end_index]
    proxy = [ProxyPrice(row.ts_recv_ns, row.midpoint) for row in overlapping_bbo]
    if not proxy:
        raise ForwardShadowReportError("SI BBO has no overlap with the capture")

    fills = _build_fill_observations(
        fill_rows=fill_rows,
        trade_rows=trade_rows,
        lifecycle_rows=lifecycle_rows,
        fair_rows=fair_rows,
    )
    markets = _latest_market_rows(market_rows)
    missing_markets = sorted({row.market_ticker for row in fills} - set(markets))
    if missing_markets:
        raise ForwardShadowReportError(
            f"fill markets have no metadata: {', '.join(missing_markets)}"
        )

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
                "databento_to_host_latency_ms": int(transport_ms),
                **_prediction_summary(predictions),
            }
        )
        for cancel_ms in cancel_latencies_ms:
            scenarios.append(
                {
                    "databento_to_host_latency_ms": int(transport_ms),
                    **_scenario_summary(
                        fills, warnings, cancel_latency_ms=int(cancel_ms)
                    ),
                }
            )

    spreads = [row.spread for row in overlapping_bbo]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "collector_run_id": collector_run_id,
        "shadow_run_id": shadow_run_id,
        "metal": metal,
        "baseline_study_id": study_id,
        "execution_mode": "research_only",
        "trading_eligible": False,
        "method": {
            "source": "Databento GLBX.MDP3 MBO",
            "contract_selection": selected,
            "proxy_beta": format(proxy_beta, "f"),
            "anchor": "latest causally received exact Kalshi-carried Pyth index",
            "estimate": "exact_anchor + beta * (current_SI_mid - SI_mid_at_anchor)",
            "availability": "Databento ts_recv plus declared host-transport latency",
            "quote_test": "counterfactual side fair minus frozen original charge below resting quote price",
        },
        "coverage": {
            "databento_request_id": databento_request_id,
            "trade_projection_sha256": _sha256_path(trade_path),
            "mbo_projection_sha256": _sha256_path(mbo_path),
            "mbo_input_records": reconstruction.input_records,
            "selected_mbo_records": reconstruction.selected_records,
            "selected_complete_events": reconstruction.emitted_events,
            "selected_bbo_changes_total": len(reconstruction.snapshots),
            "selected_bbo_changes_overlap": len(overlapping_bbo),
            "selected_first_bbo_ts_recv_ns": overlapping_bbo[0].ts_recv_ns,
            "selected_last_bbo_ts_recv_ns": overlapping_bbo[-1].ts_recv_ns,
            "minimum_spread_usd": format(min(spreads), "f"),
            "maximum_spread_usd": format(max(spreads), "f"),
            "duplicate_adds": reconstruction.duplicate_adds,
            "unknown_cancels": reconstruction.unknown_cancels,
            "unknown_modifies": reconstruction.unknown_modifies,
            "crossed_or_locked_events": reconstruction.crossed_or_locked_events,
            "exact_updates": len(exact),
            "fill_events": len(fills),
            "fill_quantity": format(sum((row.quantity for row in fills), Decimal(0)), "f"),
        },
        "prediction_quality": prediction_quality,
        "scenarios": scenarios,
        "limitations": [
            "Historical Databento ts_recv is the vendor capture time, not this host's unobserved live receive time.",
            "The Databento-to-host and cancellation latencies are explicit counterfactual scenarios.",
            "The active contract is selected by full-request trade quantity; live deployment must freeze a causal roll rule.",
            "SI is used only as an anchored delta proxy and never as the settlement value.",
            "Beta is fixed before replay and is not fitted to this capture.",
            "Strict-penetration shadow fills do not reproduce queue position or exchange cancellation races.",
            "The capture contains only three complete settlement windows.",
        ],
    }
    identity = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_id"] = hashlib.sha256(identity).hexdigest()
    return report


def write_si_fast_feed_counterfactual_report(
    root: str | Path,
    collector_run_id: str,
    shadow_run_id: str,
    **kwargs: Any,
) -> Path:
    report = build_si_fast_feed_counterfactual_report(
        root, collector_run_id, shadow_run_id, **kwargs
    )
    destination = (
        Path(root)
        / "derived"
        / "si_fast_feed_counterfactual"
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
    parser.add_argument("--databento-request-id", required=True)
    parser.add_argument("--metal", default="silver")
    args = parser.parse_args(argv)
    path = write_si_fast_feed_counterfactual_report(
        args.root,
        args.collector_run_id,
        args.shadow_run_id,
        baseline_model_path=args.baseline_model_path,
        databento_request_id=args.databento_request_id,
        metal=args.metal,
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
