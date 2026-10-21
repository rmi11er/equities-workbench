"""Combine immutable per-capture forward long-vol reports without refitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .forward_long_vol_report import (
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    _day_block_interval,
    _decimal,
    _decimal_text,
    _markout_summary,
    _mean,
)


SCHEMA_VERSION = "metals-combined-forward-long-vol-report/v1"
ZERO = Decimal(0)


class CombinedForwardLongVolReportError(ValueError):
    """Raised when per-capture reports cannot be pooled without ambiguity."""


def combine_forward_long_vol_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not reports:
        raise CombinedForwardLongVolReportError("at least one source report is required")
    policy_versions: set[str] = set()
    report_ids: set[str] = set()
    entries: list[Mapping[str, Any]] = []
    funnel_totals: Counter[str] = Counter()
    rejection_rows: Counter[str] = Counter()
    rejection_markets: Counter[str] = Counter()
    for report in reports:
        if report.get("schema_version") != SOURCE_SCHEMA_VERSION:
            raise CombinedForwardLongVolReportError("source report schema mismatch")
        if report.get("trading_eligible") is not False:
            raise CombinedForwardLongVolReportError("source report enabled trading")
        report_id = str(report.get("report_id", ""))
        if not report_id or report_id in report_ids:
            raise CombinedForwardLongVolReportError(
                "source report IDs are missing or duplicated"
            )
        report_ids.add(report_id)
        policy_versions.add(str(report.get("policy_version", "")))
        entries.extend(report.get("selected_entries") or [])
        funnel = report.get("candidate_funnel") or {}
        for field in (
            "all_rows",
            "evaluated_rows",
            "skipped_rows",
            "screen_eligible_rows",
            "screen_eligible_markets",
            "shadow_policy_eligible_rows",
            "shadow_policy_eligible_markets",
            "selected_first_entries",
        ):
            funnel_totals[field] += int(funnel.get(field, 0))
        rejection_rows.update(funnel.get("rejection_reason_rows") or {})
        rejection_markets.update(funnel.get("rejection_reason_markets") or {})
    if len(policy_versions) != 1 or "" in policy_versions:
        raise CombinedForwardLongVolReportError("source policy version is ambiguous")
    candidate_ids = [str(entry.get("candidate_id", "")) for entry in entries]
    if any(not value for value in candidate_ids) or len(set(candidate_ids)) != len(
        candidate_ids
    ):
        raise CombinedForwardLongVolReportError("selected entries overlap or lack IDs")

    labeled = [entry for entry in entries if entry.get("terminal_pnl") is not None]
    pnl = [_decimal(entry["terminal_pnl"], "terminal_pnl") for entry in labeled]
    asks = [_decimal(entry["entry_ask"], "entry_ask") for entry in labeled]
    charged_edges = [
        _decimal(entry["charged_external_edge"], "charged_external_edge")
        for entry in labeled
    ]
    ratios = [
        Decimal(str(entry["implied_vol_ratio_to_external"]))
        for entry in labeled
        if entry.get("implied_vol_ratio_to_external") is not None
    ]
    days = sorted({str(entry["utc_day"]) for entry in labeled})
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "combined_forward_shadow_research_only",
        "trading_eligible": False,
        "policy_version": next(iter(policy_versions)),
        "source_report_ids": sorted(report_ids),
        "source_reports": len(reports),
        "candidate_funnel_sums": {
            **dict(sorted(funnel_totals.items())),
            "rejection_reason_rows": dict(sorted(rejection_rows.items())),
            "rejection_reason_market_sums": dict(sorted(rejection_markets.items())),
        },
        "primary_results": {
            "entries": len(entries),
            "exchange_labeled_entries": len(labeled),
            "effective_utc_days": len(days),
            "utc_days": days,
            "wins": sum(entry.get("won") is True for entry in labeled),
            "win_rate": (
                _decimal_text(
                    Decimal(sum(entry.get("won") is True for entry in labeled))
                    / Decimal(len(labeled))
                )
                if labeled
                else None
            ),
            "mean_entry_ask": _decimal_text(_mean(asks)),
            "mean_charged_external_edge": _decimal_text(_mean(charged_edges)),
            "mean_terminal_pnl": _decimal_text(_mean(pnl)),
            "total_terminal_pnl": _decimal_text(sum(pnl, ZERO)),
            "trading_day_block_95pct_mean_pnl": _day_block_interval(labeled),
            "median_implied_vol_ratio_to_external": (
                _decimal_text(median(ratios)) if ratios else None
            ),
            "markouts": _markout_summary(entries),
        },
        "selected_entries": sorted(
            (dict(entry) for entry in entries),
            key=lambda entry: int(entry["decision_ts_ns"]),
        ),
        "limitations": [
            "The pooled sample remains too small for economic admission even when a numerical day-block interval is present.",
            "Each source report uses the same frozen rule; source reports are combined without threshold tuning or refitting.",
            "No live-order latency or fill guarantee is inferred from displayed taker quotes.",
        ],
        "decision": "CONTINUE_FORWARD_COLLECTION_NO_LIVE_CAPITAL",
    }


def _read_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CombinedForwardLongVolReportError(
            f"unreadable source report: {path}"
        ) from error
    if not isinstance(value, dict):
        raise CombinedForwardLongVolReportError(f"report is not an object: {path}")
    return value


def write_combined_report(paths: Sequence[Path], root: Path) -> Path:
    inputs = []
    reports = []
    for path in paths:
        resolved = path.resolve()
        payload = resolved.read_bytes()
        inputs.append(
            {"path": str(resolved), "sha256": hashlib.sha256(payload).hexdigest()}
        )
        reports.append(_read_report(resolved))
    report = {**combine_forward_long_vol_reports(reports), "inputs": inputs}
    identity = json.dumps(
        report, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    report["report_id"] = hashlib.sha256(identity).hexdigest()
    destination = (
        root
        / "derived"
        / "combined_forward_long_vol_report"
        / f"report={report['report_id'][:24]}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "report.json"
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != rendered:
        raise CombinedForwardLongVolReportError(
            f"refusing to overwrite report: {output}"
        )
    output.write_text(rendered, encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/metals"))
    parser.add_argument("--report", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    print(write_combined_report(args.report, args.root))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
