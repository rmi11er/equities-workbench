"""Post-session RFQ analysis script.

Analyzes RFQ decision logs to compare theo values vs actual execution prices.
Reads JSONL files written by the decision_logger and optionally queries the
Kalshi API for additional outcome data.

Usage:
    python -m src.rfq.analyze_rfqs logs/rfq/decisions_20250115_120000.jsonl
    python -m src.rfq.analyze_rfqs logs/rfq/ --all-sessions
    python -m src.rfq.analyze_rfqs logs/rfq/decisions_*.jsonl --query-api
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Statistics tracking
@dataclass
class SessionStats:
    """Aggregated statistics for a session."""
    total_rfqs: int = 0
    quoted: int = 0
    filtered: int = 0
    skipped: int = 0
    errors: int = 0

    # Outcome stats (from outcomes file)
    accepted: int = 0
    executed: int = 0
    expired: int = 0
    deleted: int = 0

    # P&L tracking
    total_edge_quoted: float = 0.0  # Sum of expected edge on quotes sent
    total_edge_executed: float = 0.0  # Sum of edge on executed trades

    # Theo accuracy
    theo_vs_execution: list = field(default_factory=list)  # (theo, execution_price) pairs

    # Filter breakdown
    filter_reasons: dict = field(default_factory=lambda: defaultdict(int))

    @property
    def quote_rate(self) -> float:
        return self.quoted / self.total_rfqs if self.total_rfqs > 0 else 0.0

    @property
    def fill_rate(self) -> float:
        return self.executed / self.quoted if self.quoted > 0 else 0.0


def load_decisions(path: Path) -> list[dict]:
    """Load decisions from a JSONL file."""
    decisions = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                decisions.append(json.loads(line))
    return decisions


def load_outcomes(path: Path) -> list[dict]:
    """Load outcomes from a JSONL file."""
    outcomes = []
    if not path.exists():
        return outcomes
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                outcomes.append(json.loads(line))
    return outcomes


def analyze_session(decisions_path: Path) -> SessionStats:
    """Analyze a single session's decision log."""
    stats = SessionStats()

    # Load decisions
    decisions = load_decisions(decisions_path)
    stats.total_rfqs = len(decisions)

    # Build decision lookup by RFQ ID
    decisions_by_rfq = {}
    decisions_by_quote = {}

    for d in decisions:
        rfq_id = d.get("rfq_id")
        decisions_by_rfq[rfq_id] = d

        action = d.get("action", "unknown")
        if action == "quoted":
            stats.quoted += 1
            if d.get("edge"):
                stats.total_edge_quoted += d["edge"]
            quote_id = d.get("quote_id")
            if quote_id:
                decisions_by_quote[quote_id] = d
        elif action == "filtered":
            stats.filtered += 1
            reason = d.get("filter_reason", "unknown")
            filter_name = d.get("filter_name", "unknown")
            stats.filter_reasons[f"{filter_name}: {reason}"] += 1
        elif action == "skipped":
            stats.skipped += 1
        elif action == "error":
            stats.errors += 1

    # Load outcomes (same session)
    outcomes_path = decisions_path.parent / decisions_path.name.replace("decisions_", "outcomes_")
    outcomes = load_outcomes(outcomes_path)

    for o in outcomes:
        event_type = o.get("event_type")
        if event_type == "accepted":
            stats.accepted += 1
        elif event_type == "executed":
            stats.executed += 1
            # Track theo vs execution
            quote_id = o.get("quote_id")
            execution_price = o.get("execution_price")
            if quote_id and quote_id in decisions_by_quote:
                d = decisions_by_quote[quote_id]
                theo = d.get("theo_value")
                if theo and execution_price:
                    stats.theo_vs_execution.append((theo, execution_price))
                    # Calculate realized edge
                    edge = d.get("edge", 0)
                    stats.total_edge_executed += edge
        elif event_type == "expired":
            stats.expired += 1
        elif event_type == "deleted":
            stats.deleted += 1

    return stats


def print_session_report(stats: SessionStats, session_name: str) -> None:
    """Print a formatted report for a session."""
    print(f"\n{'='*60}")
    print(f"Session: {session_name}")
    print(f"{'='*60}")

    print(f"\n--- RFQ Summary ---")
    print(f"Total RFQs seen:     {stats.total_rfqs}")
    print(f"Quoted:              {stats.quoted} ({stats.quote_rate*100:.1f}%)")
    print(f"Filtered:            {stats.filtered}")
    print(f"Skipped (no price):  {stats.skipped}")
    print(f"Errors:              {stats.errors}")

    print(f"\n--- Quote Outcomes ---")
    print(f"Accepted:            {stats.accepted}")
    print(f"Executed (filled):   {stats.executed} ({stats.fill_rate*100:.1f}% fill rate)")
    print(f"Expired:             {stats.expired}")
    print(f"Deleted:             {stats.deleted}")

    if stats.total_edge_quoted > 0:
        print(f"\n--- Edge Analysis ---")
        print(f"Total edge quoted:   ${stats.total_edge_quoted:.2f}")
        print(f"Total edge executed: ${stats.total_edge_executed:.2f}")

    if stats.theo_vs_execution:
        print(f"\n--- Theo Accuracy ({len(stats.theo_vs_execution)} fills) ---")
        errors = [(theo - exec_p) for theo, exec_p in stats.theo_vs_execution]
        avg_error = sum(errors) / len(errors)
        abs_errors = [abs(e) for e in errors]
        mae = sum(abs_errors) / len(abs_errors)
        print(f"Mean error (theo - execution): {avg_error:.4f}")
        print(f"Mean absolute error:           {mae:.4f}")

    if stats.filter_reasons:
        print(f"\n--- Filter Breakdown ---")
        sorted_reasons = sorted(stats.filter_reasons.items(), key=lambda x: -x[1])
        for reason, count in sorted_reasons[:10]:
            print(f"  {count:4d}  {reason}")


def find_session_files(log_dir: Path) -> list[Path]:
    """Find all decision files in a log directory."""
    return sorted(log_dir.glob("decisions_*.jsonl"))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RFQ decision logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to decisions JSONL file or log directory",
    )
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="Analyze all sessions in directory",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Only show aggregate summary across sessions",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    # Find files to analyze
    if args.path.is_dir():
        files = find_session_files(args.path)
        if not files:
            print(f"No decision files found in {args.path}", file=sys.stderr)
            return 1
    elif args.path.is_file():
        files = [args.path]
    else:
        print(f"Path not found: {args.path}", file=sys.stderr)
        return 1

    if not args.all_sessions and len(files) > 1:
        # Default: only analyze most recent
        files = [files[-1]]
        print(f"Analyzing most recent session: {files[0].name}")
        print("(Use --all-sessions to analyze all)")

    # Analyze each session
    all_stats = []
    for f in files:
        stats = analyze_session(f)
        all_stats.append((f.stem, stats))

        if not args.summary:
            print_session_report(stats, f.stem)

    # Aggregate summary
    if len(all_stats) > 1 or args.summary:
        agg = SessionStats()
        for _, s in all_stats:
            agg.total_rfqs += s.total_rfqs
            agg.quoted += s.quoted
            agg.filtered += s.filtered
            agg.skipped += s.skipped
            agg.errors += s.errors
            agg.accepted += s.accepted
            agg.executed += s.executed
            agg.expired += s.expired
            agg.deleted += s.deleted
            agg.total_edge_quoted += s.total_edge_quoted
            agg.total_edge_executed += s.total_edge_executed
            agg.theo_vs_execution.extend(s.theo_vs_execution)
            for k, v in s.filter_reasons.items():
                agg.filter_reasons[k] += v

        print_session_report(agg, f"AGGREGATE ({len(all_stats)} sessions)")

    if args.json:
        # JSON output for programmatic use
        result = {
            "sessions": len(all_stats),
            "total_rfqs": sum(s.total_rfqs for _, s in all_stats),
            "quoted": sum(s.quoted for _, s in all_stats),
            "executed": sum(s.executed for _, s in all_stats),
            "total_edge_quoted": sum(s.total_edge_quoted for _, s in all_stats),
            "total_edge_executed": sum(s.total_edge_executed for _, s in all_stats),
        }
        print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
