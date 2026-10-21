"""Combined SI and direct-XAG cancellation counterfactual."""

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
    _load_proxy_rows,
    _scenario_summary,
    _warning_time,
    simulate_fast_snapshots,
)
from .forward_shadow_report import ForwardShadowReportError, _stream_rows
from .si_fast_feed_counterfactual import _input_base

SCHEMA_VERSION = "metals-combined-fast-feed-counterfactual/v1"
PYTH_LATENCIES_MS = (100, 250, 500)
SI_LATENCIES_MS = (10, 25, 50, 100)
CANCEL_LATENCIES_MS = (50, 100, 250)


def _first_warning(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def _avoidance_overlap(
    fills: Sequence[Any],
    pyth_warnings: Sequence[int | None],
    si_warnings: Sequence[int | None],
    *,
    cancel_latency_ms: int,
) -> dict[str, Any]:
    cancel_ns = cancel_latency_ms * 1_000_000

    def avoided(row: Any, warning: int | None) -> bool:
        return warning is not None and warning + cancel_ns < row.fill_ts_ns

    pyth = {index for index, row in enumerate(fills) if avoided(row, pyth_warnings[index])}
    si = {index for index, row in enumerate(fills) if avoided(row, si_warnings[index])}

    def quantity(indexes: set[int]) -> str:
        return format(sum((fills[index].quantity for index in indexes), Decimal(0)), "f")

    return {
        "pyth_avoided_events": len(pyth),
        "si_avoided_events": len(si),
        "both_avoided_events": len(pyth & si),
        "pyth_only_avoided_events": len(pyth - si),
        "si_only_avoided_events": len(si - pyth),
        "union_avoided_events": len(pyth | si),
        "pyth_only_avoided_quantity": quantity(pyth - si),
        "si_only_avoided_quantity": quantity(si - pyth),
        "both_avoided_quantity": quantity(pyth & si),
        "union_avoided_quantity": quantity(pyth | si),
    }


def build_combined_fast_feed_counterfactual_report(
    root: str | Path,
    collector_run_id: str,
    shadow_run_id: str,
    *,
    baseline_model_path: str | Path,
    databento_request_id: str,
    metal: str = "silver",
    pyth_proxy_metal: str = "silver-xag-proxy",
    pyth_feed_id: int = 345,
    pyth_latencies_ms: Sequence[int] = PYTH_LATENCIES_MS,
    si_latencies_ms: Sequence[int] = SI_LATENCIES_MS,
    cancel_latencies_ms: Sequence[int] = CANCEL_LATENCIES_MS,
) -> dict[str, Any]:
    storage_root = Path(root)
    model, study_id = load_baseline_vol_model(Path(baseline_model_path))
    market_rows = _stream_rows(storage_root, "market_dim", collector_run_id, metal=metal)
    exact_rows = _stream_rows(
        storage_root, "kalshi_pyth_value", collector_run_id, metal=metal
    )
    kalshi_trade_rows = _stream_rows(
        storage_root, "kalshi_trade", collector_run_id, metal=metal
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
    pyth_proxy, pyth_request_ids = _load_proxy_rows(
        storage_root,
        proxy_metal=pyth_proxy_metal,
        proxy_feed_id=pyth_feed_id,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
    )

    input_base = _input_base(storage_root, databento_request_id[:24], metal)
    selected = select_active_outright_contract(
        iter_hash_chained_jsonl([input_base / "trades.jsonl"]), asset="SI"
    )
    reconstruction = reconstruct_mbo_bbo(
        iter_hash_chained_jsonl([input_base / "mbo.jsonl"]),
        instrument_id=int(selected["instrument_id"]),
        contract_symbol=str(selected["contract_symbol"]),
    )
    bbo = list(reconstruction.snapshots)
    bbo_times = [row.ts_recv_ns for row in bbo]
    first = max(0, bisect.bisect_right(bbo_times, start_ts_ns) - 1)
    last = bisect.bisect_right(bbo_times, end_ts_ns)
    si_proxy = [ProxyPrice(row.ts_recv_ns, row.midpoint) for row in bbo[first:last]]
    if not pyth_proxy or not si_proxy:
        raise ForwardShadowReportError("combined replay is missing a fast proxy")

    fills = _build_fill_observations(
        fill_rows=fill_rows,
        trade_rows=kalshi_trade_rows,
        lifecycle_rows=lifecycle_rows,
        fair_rows=fair_rows,
    )
    markets = _latest_market_rows(market_rows)
    pyth_warning_by_latency: dict[int, list[int | None]] = {}
    si_warning_by_latency: dict[int, list[int | None]] = {}
    for latency in pyth_latencies_ms:
        snapshots, _ = simulate_fast_snapshots(
            exact, pyth_proxy, proxy_transport_latency_ms=int(latency)
        )
        times = [row.available_ts_ns for row in snapshots]
        pyth_warning_by_latency[int(latency)] = [
            _warning_time(fill, snapshots, times, markets[fill.market_ticker], model)
            for fill in fills
        ]
    for latency in si_latencies_ms:
        snapshots, _ = simulate_fast_snapshots(
            exact, si_proxy, proxy_transport_latency_ms=int(latency)
        )
        times = [row.available_ts_ns for row in snapshots]
        si_warning_by_latency[int(latency)] = [
            _warning_time(fill, snapshots, times, markets[fill.market_ticker], model)
            for fill in fills
        ]

    scenarios: list[dict[str, Any]] = []
    for pyth_latency, pyth_warnings in pyth_warning_by_latency.items():
        for si_latency, si_warnings in si_warning_by_latency.items():
            combined = [
                _first_warning(pyth, si)
                for pyth, si in zip(pyth_warnings, si_warnings)
            ]
            for cancel_latency in cancel_latencies_ms:
                scenarios.append(
                    {
                        "pyth_transport_latency_ms": pyth_latency,
                        "databento_to_host_latency_ms": si_latency,
                        **_scenario_summary(
                            fills, combined, cancel_latency_ms=int(cancel_latency)
                        ),
                        "avoidance_overlap": _avoidance_overlap(
                            fills,
                            pyth_warnings,
                            si_warnings,
                            cancel_latency_ms=int(cancel_latency),
                        ),
                    }
                )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "collector_run_id": collector_run_id,
        "shadow_run_id": shadow_run_id,
        "metal": metal,
        "baseline_study_id": study_id,
        "execution_mode": "research_only",
        "trading_eligible": False,
        "method": {
            "decision_rule": "cancel or suppress when either anchored direct XAG or anchored SI invalidates the frozen quote edge",
            "pyth_proxy": {"symbol": "Metal.XAG/USD", "feed_id": pyth_feed_id},
            "si_proxy": selected,
            "proxy_beta": "1",
        },
        "coverage": {
            "exact_updates": len(exact),
            "pyth_updates": len(pyth_proxy),
            "pyth_request_ids": pyth_request_ids,
            "si_bbo_changes": len(si_proxy),
            "databento_request_id": databento_request_id,
            "fill_events": len(fills),
            "fill_quantity": format(sum((row.quantity for row in fills), Decimal(0)), "f"),
        },
        "scenarios": scenarios,
        "limitations": [
            "Historical Pyth and Databento records do not recreate this host's live receive timestamps; all downstream transport and cancellation latencies are declared scenarios.",
            "The union rule is a cancel-only selection filter, not permission to move fair value or quote more aggressively.",
            "Both proxies use fixed unit beta and reset to every causally received exact settlement-index anchor.",
            "Strict-penetration shadow fills do not reproduce queue position or exchange cancellation races.",
            "The capture contains only three complete settlement windows, so attribution is specific to this run.",
        ],
    }
    identity = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_id"] = hashlib.sha256(identity).hexdigest()
    return report


def write_combined_fast_feed_counterfactual_report(
    root: str | Path,
    collector_run_id: str,
    shadow_run_id: str,
    **kwargs: Any,
) -> Path:
    report = build_combined_fast_feed_counterfactual_report(
        root, collector_run_id, shadow_run_id, **kwargs
    )
    destination = (
        Path(root)
        / "derived"
        / "combined_fast_feed_counterfactual"
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
    args = parser.parse_args(argv)
    path = write_combined_fast_feed_counterfactual_report(
        args.root,
        args.collector_run_id,
        args.shadow_run_id,
        baseline_model_path=args.baseline_model_path,
        databento_request_id=args.databento_request_id,
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
