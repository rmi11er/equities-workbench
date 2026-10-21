import hashlib
import json
from pathlib import Path

import pytest

from src.metals.forward_shadow_report import (
    ForwardShadowReportError,
    _fill_selection_summary,
    _window_pyth_continuity,
    build_forward_shadow_report,
    write_forward_shadow_report,
)


def _write_rows(root: Path, stream: str, run_id: str, rows: list[dict]) -> None:
    path = (
        root
        / "normalized"
        / stream
        / "date=2026-09-01"
        / "metal=silver"
        / f"run={run_id}"
        / f"{stream}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    chained = []
    previous = None
    for row in rows:
        payload = {**row, "previous_record_sha256": previous}
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        previous = hashlib.sha256(canonical.encode()).hexdigest()
        chained.append({**payload, "record_sha256": previous})
    path.write_text("".join(json.dumps(row) + "\n" for row in chained))


def _write_manifests(
    root: Path, *, status: str = "completed", errors: list[dict] | None = None
) -> None:
    collector = {
        "run_id": "collector",
        "status": status,
        "errors": errors or [],
        "config": {},
    }
    shadow = {
        "run_id": "shadow",
        "status": "completed",
        "errors": [],
        "config": {
            "forward_shadow": {
                "collector_run_id": "collector",
                "read_only": True,
                "live_orders": False,
                "baseline_study_id": "study",
                "model_version": "model",
                "settings": {"uncertainty_charge": "0.02"},
            }
        },
    }
    for run_id, manifest in (("collector", collector), ("shadow", shadow)):
        path = root / "manifests" / f"run={run_id}" / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest))


def _fixture(root: Path) -> None:
    _write_manifests(root)
    _write_rows(
        root,
        "market_dim",
        "collector",
        [
            {
                "market_ticker": "SILVER-A",
                "open_ts_ns": 1_000_000_000,
                "close_ts_ns": 2_000_000_000,
                "target_price": "30.000",
                "round_digits": 3,
                "pyth_symbol": "Metal.Index.SILVER/USD",
                "metadata_version": 1,
                "metadata_receive_ts_ns": 1,
            },
            {
                "market_ticker": "SILVER-B",
                "open_ts_ns": 2_000_000_000,
                "close_ts_ns": 4_000_000_000,
                "target_price": "30.100",
                "round_digits": 3,
                "pyth_symbol": "Metal.Index.SILVER/USD",
                "metadata_version": 1,
                "metadata_receive_ts_ns": 2,
            },
        ],
    )
    _write_rows(
        root,
        "kalshi_pyth_value",
        "collector",
        [
            {
                "underlying_ticker": "Metal.Index.SILVER/USD",
                "source_ts_ns": 1_000_000_000,
                "receive_ts_ns": 1_300_000_000,
                "value_usd": "30.0004",
            },
            {
                "underlying_ticker": "Metal.Index.SILVER/USD",
                "source_ts_ns": 2_000_000_000,
                "receive_ts_ns": 2_350_000_000,
                "value_usd": "29.0000",
            },
            {
                "underlying_ticker": "Metal.Index.SILVER/USD",
                "source_ts_ns": 3_000_000_000,
                "receive_ts_ns": 3_400_000_000,
                "value_usd": "30.2000",
            },
        ],
    )
    _write_rows(
        root,
        "kalshi_book_event",
        "collector",
        [
            {
                "market_ticker": "SILVER-A",
                "book_valid_after": True,
                "gap_flag": False,
                "receive_ts_ns": 1,
            },
            {
                "market_ticker": "SILVER-A",
                "book_valid_after": False,
                "gap_flag": False,
                "receive_ts_ns": 2,
            },
        ],
    )
    _write_rows(
        root,
        "kalshi_trade",
        "collector",
        [{"market_ticker": "SILVER-A"}],
    )
    _write_rows(
        root,
        "forward_fair_decision",
        "shadow",
        [
            {
                "market_ticker": "SILVER-A",
                "yes_probability": "0.55",
                "pyth_age_ms": "300",
                "quote_actions": {"yes": "placed", "no": "unchanged"},
            },
            {
                "market_ticker": "SILVER-A",
                "status": "skipped",
                "reason": "Pyth observation is stale",
            },
            {
                "market_ticker": "SILVER-B",
                "yes_probability": "0.60",
                "pyth_age_ms": "400",
                "quote_actions": {"yes": "placed", "no": "no_quote"},
            },
        ],
    )
    _write_rows(
        root,
        "shadow_fill_attribution",
        "shadow",
        [
            {
                "market_ticker": "SILVER-A",
                "passive_outcome_side": "yes",
                "fill_quantity_fp": "1",
                "quote_price_dollars": "0.40",
                "fill_fee_dollars": "0.01",
            },
            {
                "market_ticker": "SILVER-A",
                "passive_outcome_side": "no",
                "fill_quantity_fp": "2",
                "quote_price_dollars": "0.30",
                "fill_fee_dollars": "0.02",
            },
        ],
    )
    _write_rows(
        root,
        "order_lifecycle",
        "shadow",
        [
            {
                "event_type": "order_resting",
                "client_order_id": "shadow:one",
                "exchange_order_id": None,
            }
        ],
    )


def test_report_scores_only_complete_windows_and_proves_shadow_isolation(tmp_path):
    root = tmp_path / "metals"
    _fixture(root)

    report = build_forward_shadow_report(root, "collector", "shadow")

    assert report["coverage"]["windows_observed"] == 2
    assert report["coverage"]["completed_windows"] == 1
    assert report["feed_health"]["pyth_gaps_over_1500ms"] == 0
    assert report["feed_health"]["book_invalid_rows"] == 1
    assert report["execution_safety"]["exchange_order_ids"] == []
    assert report["execution_safety"]["non_shadow_client_order_ids"] == []
    first, second = report["windows"]
    assert first["settlement_label_eligible"] is True
    assert first["exact_index_continuity"]["continuous"] is True
    assert first["settlement_value_rounded"] == "30.000"
    assert first["provisional_outcome"] == "yes"
    assert first["fills"]["settlement_pnl_dollars"] == "-0.03"
    assert first["settlement_observed"] is True
    assert second["complete_in_capture"] is False
    assert second["exact_index_continuity"]["continuous"] is False
    assert second["fills"]["settlement_pnl_dollars"] is None
    path = write_forward_shadow_report(root, "collector", "shadow")
    assert json.loads(path.read_text()) == report
    assert write_forward_shadow_report(root, "collector", "shadow") == path


def test_window_continuity_detects_internal_exact_index_gap():
    rows = [
        {
            "underlying_ticker": "Metal.Index.SILVER/USD",
            "source_ts_ns": source_ts_ns,
        }
        for source_ts_ns in (
            1_000_000_000,
            2_000_000_000,
            4_000_000_000,
            5_000_000_000,
        )
    ]

    result = _window_pyth_continuity(
        rows,
        "Metal.Index.SILVER/USD",
        1_000_000_000,
        5_000_000_000,
    )

    assert result["capture_spans_window"] is True
    assert result["continuous"] is False
    assert result["gaps_over_1500ms"] == 1
    assert result["internal_source_gap_ms"]["max"] == "2000"


def test_report_never_promotes_a_stale_preclose_value_to_a_label(tmp_path):
    root = tmp_path / "metals"
    _fixture(root)
    _write_rows(
        root,
        "kalshi_pyth_value",
        "collector",
        [
            {
                "underlying_ticker": "Metal.Index.SILVER/USD",
                "source_ts_ns": 400_000_000,
                "receive_ts_ns": 500_000_000,
                "value_usd": "30.5000",
            },
            {
                "underlying_ticker": "Metal.Index.SILVER/USD",
                "source_ts_ns": 3_000_000_000,
                "receive_ts_ns": 3_100_000_000,
                "value_usd": "29.0000",
            },
        ],
    )

    report = build_forward_shadow_report(root, "collector", "shadow")

    first = report["windows"][0]
    assert first["settlement_observed"] is True
    assert first["settlement_label_eligible"] is False
    assert first["complete_in_capture"] is False
    assert first["provisional_outcome"] is None
    assert first["settlement_value_rounded"] is None
    assert first["last_preclose_pyth_diagnostic"] == {
        "source_ts_ns": 400_000_000,
        "source_age_ms": "1600",
        "value_unrounded": "30.5000",
    }


def test_report_rejects_running_or_non_shadow_manifest(tmp_path):
    root = tmp_path / "metals"
    _fixture(root)
    _write_manifests(root, status="running")
    with pytest.raises(ForwardShadowReportError, match="not reportable"):
        build_forward_shadow_report(root, "collector", "shadow")

    _write_manifests(root)
    path = root / "manifests" / "run=shadow" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["forward_shadow"]["live_orders"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ForwardShadowReportError, match="not manifest-declared"):
        build_forward_shadow_report(root, "collector", "shadow")


def test_report_preserves_controlled_interruption_status(tmp_path):
    root = tmp_path / "metals"
    _fixture(root)
    _write_manifests(
        root,
        status="cancelled",
        errors=[{"type": "CancelledError", "message": ""}],
    )

    report = build_forward_shadow_report(root, "collector", "shadow")

    assert report["termination"] == {
        "collector_status": "cancelled",
        "collector_error_types": ["CancelledError"],
        "shadow_status": "completed",
        "shadow_error_types": [],
        "controlled_interruption": True,
    }


def test_fill_selection_joins_trade_time_to_causal_fair_markouts():
    result = _fill_selection_summary(
        [
            {
                "market_ticker": "SILVER-A",
                "passive_outcome_side": "yes",
                "fill_quantity_fp": "2",
                "quote_price_dollars": "0.40",
                "fill_fee_dollars": "0.02",
                "source_trade_ingestion_sequence": 10,
            }
        ],
        [{"ingestion_sequence": 10, "receive_ts_ns": 1_000_000_000}],
        [
            {
                "market_ticker": "SILVER-A",
                "decision_ts_ns": 900_000_000,
                "yes_probability": "0.50",
            },
            {
                "market_ticker": "SILVER-A",
                "decision_ts_ns": 2_000_000_000,
                "yes_probability": "0.45",
            },
            {
                "market_ticker": "SILVER-A",
                "decision_ts_ns": 6_000_000_000,
                "yes_probability": "0.40",
            },
        ],
    )

    assert result["joined_fill_events"] == 1
    assert result["all"] == {
        "fill_events": 1,
        "quantity": "2",
        "gross_external_edge_cents_per_contract": "10.00",
        "net_external_edge_after_fee_cents_per_contract": "9.00",
        "external_fair_markout_1s_cents_per_contract": "-5.00",
        "external_fair_markout_5s_cents_per_contract": "-10.00",
        "markout_1s_fill_events": 1,
        "markout_5s_fill_events": 1,
    }
