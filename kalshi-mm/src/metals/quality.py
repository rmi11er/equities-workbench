"""Collector counters and deterministic daily quality-report output."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DATA_GATE_THRESHOLDS = {
    "kalshi_valid_fraction": 0.999,
    "pyth_valid_fraction": 0.995,
    "cme_valid_fraction": 0.995,
}


def _coverage_fraction(valid_ns: int, total_ns: int) -> Optional[float]:
    if total_ns <= 0:
        return None
    return valid_ns / total_ns


def evaluate_data_gates(report: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the strategy specification's Phase 1 data gates.

    Unknown or unmeasured inputs fail closed. This evaluator is intentionally
    independent of trading code and can only produce research readiness status.
    """

    checksum = report.get("settlement_checksum", {})
    checked = int(checksum.get("checked", 0))
    agreements = int(checksum.get("agreements", 0))
    disagreements = int(checksum.get("disagreements", 0))
    coverage = report.get("coverage", {})
    missing_intervals = report.get("missing_intervals", [])
    known_gap_events = sum(
        int(report.get(counter, 0))
        for counter in (
            "kalshi_sequence_gaps",
            "websocket_disconnects",
            "pyth_missing_intervals",
            "pyth_gap_events",
            "pyth_transport_disconnects",
            "cme_sequence_gaps",
            "cme_reconnects",
            "cme_collection_errors",
            "cme_disconnects",
            "cme_normalization_queue_overflows",
            "ingestion_monotonic_regressions",
            "clock_probe_errors",
            "clock_elapsed_discontinuities",
        )
    )
    classified_intervals = sum(
        1 for item in missing_intervals if bool(str(item.get("reason", "")).strip())
    )
    expected_windows = int(report.get("expected_windows", 0))
    has_complete_settlement_evidence = all(
        field in checksum for field in ("expected", "excluded", "resolved", "complete")
    )
    if has_complete_settlement_evidence:
        expected_settlements = int(checksum.get("expected", 0))
        excluded_settlements = int(checksum.get("excluded", 0))
        resolved_settlements = int(checksum.get("resolved", 0))
        required_settlements = expected_settlements - excluded_settlements
        settlement_passed = (
            bool(checksum.get("complete"))
            and expected_settlements == expected_windows
            and 0 <= excluded_settlements <= expected_settlements
            and resolved_settlements == expected_settlements
            and checked == required_settlements
            and agreements == checked
            and disagreements == 0
        )
    else:
        required_settlements = (
            min(1000, expected_windows) if expected_windows > 0 else 1000
        )
        settlement_passed = (
            checked >= required_settlements
            and agreements == checked
            and disagreements == 0
        )

    gates = {
        "settlement_agreement": {
            "passed": settlement_passed,
            "observed": {
                "expected": checksum.get("expected"),
                "checked": checked,
                "agreements": agreements,
                "disagreements": disagreements,
                "excluded": checksum.get("excluded"),
                "resolved": checksum.get("resolved"),
                "complete": checksum.get("complete"),
            },
            "required": {
                "minimum_checked": required_settlements,
                "agreement": (
                    "every expected window is exactly audited or classified excluded; "
                    "100% agreement on usable audited windows"
                ),
            },
        },
        "no_invalid_book_decisions": {
            "passed": int(report.get("strategy_decisions_on_invalid_book", 0)) == 0,
            "observed": int(report.get("strategy_decisions_on_invalid_book", 0)),
            "required": 0,
        },
        "kalshi_book_coverage": {
            "passed": coverage.get("kalshi_valid_fraction") is not None
            and float(coverage["kalshi_valid_fraction"])
            >= DATA_GATE_THRESHOLDS["kalshi_valid_fraction"],
            "observed": coverage.get("kalshi_valid_fraction"),
            "required": DATA_GATE_THRESHOLDS["kalshi_valid_fraction"],
        },
        "pyth_coverage": {
            "passed": coverage.get("pyth_valid_fraction") is not None
            and float(coverage["pyth_valid_fraction"])
            >= DATA_GATE_THRESHOLDS["pyth_valid_fraction"],
            "observed": coverage.get("pyth_valid_fraction"),
            "required": DATA_GATE_THRESHOLDS["pyth_valid_fraction"],
        },
        "cme_coverage": {
            "passed": coverage.get("cme_valid_fraction") is not None
            and float(coverage["cme_valid_fraction"])
            >= DATA_GATE_THRESHOLDS["cme_valid_fraction"],
            "observed": coverage.get("cme_valid_fraction"),
            "required": DATA_GATE_THRESHOLDS["cme_valid_fraction"],
        },
        "missing_intervals_classified": {
            "passed": classified_intervals == len(missing_intervals)
            and classified_intervals >= known_gap_events,
            "observed": {
                "known_gap_events": known_gap_events,
                "classified_intervals": classified_intervals,
                "total_intervals": len(missing_intervals),
            },
            "required": "every known gap has a classified interval and reason",
        },
        "clock_health": {
            "passed": bool(report.get("clock_error_within_edge_half_life", False)),
            "observed": report.get("clock_offset_abs_max_ns"),
            "required": "clock error measured and small relative to edge half-life",
        },
        "causal_receive_order": {
            "passed": bool(report.get("causal_ordering_checked", False))
            and int(report.get("timestamp_ordering_violations", 0)) == 0,
            "observed": {
                "checked": bool(report.get("causal_ordering_checked", False)),
                "violations": int(report.get("timestamp_ordering_violations", 0)),
            },
            "required": (
                "global receive ordering checked and no feature/input timestamp "
                "later than its decision receive key"
            ),
        },
        "rules_unchanged": {
            "passed": bool(report.get("metadata_rules_validated", False))
            and bool(report.get("fee_registry_validated", False))
            and bool(report.get("incentive_registry_validated", False))
            and not report.get("unresolved_metadata_change", False)
            and int(report.get("fee_changes", 0)) == 0
            and int(report.get("incentive_changes", 0)) == 0,
            "observed": {
                "metadata_rules_validated": bool(
                    report.get("metadata_rules_validated", False)
                ),
                "fee_registry_validated": bool(
                    report.get("fee_registry_validated", False)
                ),
                "incentive_registry_validated": bool(
                    report.get("incentive_registry_validated", False)
                ),
                "unresolved_metadata_change": bool(
                    report.get("unresolved_metadata_change", False)
                ),
                "fee_changes": int(report.get("fee_changes", 0)),
                "incentive_changes": int(report.get("incentive_changes", 0)),
            },
            "required": "validated registries and no unresolved rule change",
        },
    }
    failed = [name for name, gate in gates.items() if not gate["passed"]]
    return {"passed": not failed, "failed_gates": failed, "gates": gates}


@dataclass
class QualityTracker:
    counters: Counter = field(default_factory=Counter)
    missing_intervals: list = field(default_factory=list)
    metadata_changes: list = field(default_factory=list)
    market_event_counts: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    source_coverage_ns: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    clock_offsets_ns: List[int] = field(default_factory=list)
    exclusions: list = field(default_factory=list)

    def record(self, event: str, ticker: Optional[str] = None, count: int = 1) -> None:
        self.counters[event] += count
        if ticker:
            self.market_event_counts[ticker][event] += count

    def record_missing_interval(
        self,
        source: str,
        start_ts_ns: int,
        end_ts_ns: Optional[int],
        reason: str,
        ticker: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        interval = {
            "source": source,
            "ticker": ticker,
            "start_ts_ns": start_ts_ns,
            "end_ts_ns": end_ts_ns,
            "reason": reason,
        }
        if details:
            interval["details"] = dict(details)
        self.missing_intervals.append(interval)
        return interval

    def record_source_coverage(self, source: str, valid_ns: int, total_ns: int) -> None:
        if valid_ns < 0 or total_ns < 0 or valid_ns > total_ns:
            raise ValueError("Coverage durations must satisfy 0 <= valid <= total")
        self.source_coverage_ns[source]["valid_ns"] += valid_ns
        self.source_coverage_ns[source]["total_ns"] += total_ns

    def record_clock_offset(self, offset_ns: int) -> None:
        self.clock_offsets_ns.append(int(offset_ns))

    def record_exclusion(
        self, ticker: str, reason: str, start_ts_ns: Optional[int] = None
    ) -> None:
        if not reason.strip():
            raise ValueError("Exclusion reason cannot be empty")
        self.exclusions.append(
            {
                "ticker": ticker,
                "reason": reason,
                "start_ts_ns": start_ts_ns,
            }
        )

    def report(self, date: str, run_id: str) -> Dict[str, Any]:
        coverage = {}
        for source in ("kalshi", "pyth", "cme"):
            durations = self.source_coverage_ns[source]
            coverage["{}_valid_fraction".format(source)] = _coverage_fraction(
                durations.get("valid_ns", 0), durations.get("total_ns", 0)
            )
            coverage["{}_valid_ns".format(source)] = durations.get("valid_ns", 0)
            coverage["{}_total_ns".format(source)] = durations.get("total_ns", 0)
        absolute_offsets = sorted(abs(value) for value in self.clock_offsets_ns)
        clock_abs_max = absolute_offsets[-1] if absolute_offsets else None
        report = {
            "date": date,
            "run_id": run_id,
            "expected_windows": self.counters.get("expected_windows", 0),
            "observed_windows": self.counters.get("observed_windows", 0),
            "missing_markets": self.counters.get("missing_markets", 0),
            "settlement_checksum": {
                "checked": self.counters.get("settlements_checked", 0),
                "agreements": self.counters.get("settlement_agreements", 0),
                "disagreements": self.counters.get("settlement_disagreements", 0),
            },
            "kalshi_sequence_gaps": self.counters.get("kalshi_sequence_gaps", 0),
            "websocket_disconnects": self.counters.get("websocket_disconnects", 0),
            "websocket_reconnects": self.counters.get("websocket_reconnects", 0),
            "pyth_missing_intervals": self.counters.get("pyth_missing_intervals", 0),
            "pyth_gap_events": self.counters.get("pyth_gap_events", 0),
            "pyth_transport_disconnects": self.counters.get(
                "pyth_transport_disconnects", 0
            ),
            "cme_sequence_gaps": self.counters.get("cme_sequence_gaps", 0),
            "cme_reconnects": self.counters.get("cme_reconnects", 0),
            "cme_collection_errors": self.counters.get("cme_collection_errors", 0),
            "cme_disconnects": self.counters.get("cme_disconnects", 0),
            "cme_normalization_queue_overflows": self.counters.get(
                "cme_normalization_queue_overflows", 0
            ),
            "ingestion_monotonic_regressions": self.counters.get(
                "ingestion_monotonic_regressions", 0
            ),
            "duplicate_messages": self.counters.get("duplicate_messages", 0),
            "strategy_decisions_on_invalid_book": self.counters.get(
                "strategy_decisions_on_invalid_book", 0
            ),
            "invalid_book_time_ns": self.counters.get("invalid_book_time_ns", 0),
            "timestamp_ordering_violations": self.counters.get(
                "timestamp_ordering_violations", 0
            ),
            "causal_ordering_checked": bool(
                self.counters.get("causal_ordering_checked", 0)
            ),
            "unknown_aggressor_side": self.counters.get("unknown_aggressor_side", 0),
            "fractional_quantity_parsing_failures": self.counters.get(
                "fractional_quantity_parsing_failures", 0
            ),
            "fee_changes": self.counters.get("fee_changes", 0),
            "incentive_changes": self.counters.get("incentive_changes", 0),
            "unresolved_metadata_change": bool(
                self.counters.get("unresolved_metadata_change", 0)
            ),
            "metadata_rules_validated": bool(
                self.counters.get("metadata_rules_validated", 0)
            ),
            "fee_registry_validated": bool(
                self.counters.get("fee_registry_validated", 0)
            ),
            "incentive_registry_validated": bool(
                self.counters.get("incentive_registry_validated", 0)
            ),
            "coverage": coverage,
            "clock_samples": len(self.clock_offsets_ns),
            "clock_offset_abs_max_ns": clock_abs_max,
            "clock_error_within_edge_half_life": bool(
                self.counters.get("clock_error_within_edge_half_life", 0)
            ),
            "clock_probe_errors": self.counters.get("clock_probe_errors", 0),
            "clock_elapsed_discontinuities": self.counters.get(
                "clock_elapsed_discontinuities", 0
            ),
            "raw_records": self.counters.get("raw_records", 0),
            "book_snapshots": self.counters.get("book_snapshots", 0),
            "book_deltas": self.counters.get("book_deltas", 0),
            "trades": self.counters.get("trades", 0),
            "metadata_versions": self.counters.get("metadata_versions", 0),
            "metadata_changes": list(self.metadata_changes),
            "missing_intervals": list(self.missing_intervals),
            "exclusions": list(self.exclusions),
            "by_market": {
                ticker: dict(counts)
                for ticker, counts in sorted(self.market_event_counts.items())
            },
        }
        report["data_gates"] = evaluate_data_gates(report)
        return report
