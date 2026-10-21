"""Outcome and markout report for the frozen forward long-vol policy.

The report selects the first ``shadow_policy_eligible`` candidate in each whole
Kalshi window.  It never substitutes the looser screen flag, and it only scores
terminal P&L from exchange-recorded settlements.  All rows remain research-only;
the candidate schema itself must declare execution and trading ineligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .forward_shadow import LONG_VOL_CANDIDATE_SCHEMA
from .forward_shadow_report import _manifest, _stream_rows


SCHEMA_VERSION = "metals-forward-long-vol-report/v1"
MARKOUT_HORIZONS_SECONDS = (1, 5, 30, 60)
MAX_MARKOUT_OBSERVATION_LAG_MS = Decimal("2000")
ZERO = Decimal(0)
ONE = Decimal(1)


class ForwardLongVolReportError(ValueError):
    """Raised when forward candidates cannot support a trustworthy report."""


def _decimal(value: Any, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except Exception as error:
        raise ForwardLongVolReportError(
            f"invalid decimal {field}: {value!r}"
        ) from error
    if not parsed.is_finite():
        raise ForwardLongVolReportError(f"non-finite decimal {field}: {value!r}")
    return parsed


def _decimal_text(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None


def _mean(values: Sequence[Decimal]) -> Decimal | None:
    return sum(values, ZERO) / Decimal(len(values)) if values else None


def _one_contract_taker_fee(price: Decimal, coefficient: Decimal) -> Decimal:
    if not ZERO <= price <= ONE:
        raise ForwardLongVolReportError("taker-fee price must be in [0, 1]")
    if coefficient < ZERO:
        raise ForwardLongVolReportError("taker-fee coefficient cannot be negative")
    raw = coefficient * price * (ONE - price)
    return (raw * Decimal(10_000)).to_integral_value(
        rounding=ROUND_CEILING
    ) / Decimal(10_000)


def _side_quote(row: Mapping[str, Any], side: str) -> tuple[Decimal, Decimal]:
    book = row.get("book")
    if not isinstance(book, Mapping):
        raise ForwardLongVolReportError("evaluated candidate has no book object")
    bid = _decimal(book.get(f"{side}_bid"), f"{side}_bid")
    ask = _decimal(book.get(f"{side}_ask"), f"{side}_ask")
    if not ZERO <= bid < ask <= ONE:
        raise ForwardLongVolReportError("candidate has an invalid two-sided quote")
    return bid, ask


def _conservative_side_fair(row: Mapping[str, Any], side: str) -> Decimal:
    fair = row.get("external_fair")
    if not isinstance(fair, Mapping):
        raise ForwardLongVolReportError("evaluated candidate has no external fair")
    gaussian_yes = _decimal(fair.get("yes_gaussian"), "yes_gaussian")
    empirical_yes = _decimal(fair.get("yes_empirical_cdf"), "yes_empirical_cdf")
    if side == "yes":
        return min(gaussian_yes, empirical_yes)
    if side == "no":
        return min(ONE - gaussian_yes, ONE - empirical_yes)
    raise ForwardLongVolReportError(f"invalid candidate side: {side!r}")


def _latest_settlements(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        ticker = str(row.get("market_ticker", ""))
        if not ticker:
            continue
        previous = latest.get(ticker)
        if previous is None or int(row.get("receive_ts_ns", 0)) >= int(
            previous.get("receive_ts_ns", 0)
        ):
            latest[ticker] = row
    return latest


def _first_policy_entries(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(candidate_rows, key=lambda value: int(value["decision_ts_ns"])):
        if row.get("shadow_policy_eligible") is not True:
            continue
        ticker = str(row.get("market_ticker", ""))
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        selected.append(row)
    return selected


def _day_block_interval(
    entries: Sequence[Mapping[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 91_103,
) -> list[str] | None:
    by_day: dict[str, list[Decimal]] = defaultdict(list)
    for entry in entries:
        pnl = entry.get("terminal_pnl")
        if pnl is not None:
            by_day[str(entry["utc_day"])].append(_decimal(pnl, "terminal_pnl"))
    days = sorted(by_day)
    if len(days) < 2:
        return None
    generator = random.Random(seed)
    bootstrapped: list[Decimal] = []
    for _ in range(samples):
        sampled = [generator.choice(days) for _ in days]
        pnl = [value for day in sampled for value in by_day[day]]
        bootstrapped.append(sum(pnl, ZERO) / Decimal(len(pnl)))
    bootstrapped.sort()
    lower = bootstrapped[math.floor((len(bootstrapped) - 1) * 0.025)]
    upper = bootstrapped[math.ceil((len(bootstrapped) - 1) * 0.975)]
    return [_decimal_text(lower), _decimal_text(upper)]  # type: ignore[list-item]


def _markout_summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for seconds in MARKOUT_HORIZONS_SECONDS:
        observations = [
            entry["markouts"][f"{seconds}s"]
            for entry in entries
            if entry["markouts"][f"{seconds}s"] is not None
        ]
        midpoint = [
            _decimal(row["midpoint_pnl"], "midpoint_pnl") for row in observations
        ]
        liquidation = [
            _decimal(row["executable_liquidation_pnl"], "liquidation_pnl")
            for row in observations
        ]
        fair_changes = [
            _decimal(row["external_fair_change"], "external_fair_change")
            for row in observations
        ]
        result[f"{seconds}s"] = {
            "observations": len(observations),
            "mean_observation_lag_ms": _decimal_text(
                _mean(
                    [
                        _decimal(row["observation_lag_ms"], "observation_lag_ms")
                        for row in observations
                    ]
                )
            ),
            "mean_midpoint_pnl": _decimal_text(_mean(midpoint)),
            "mean_executable_liquidation_pnl": _decimal_text(_mean(liquidation)),
            "mean_external_fair_change": _decimal_text(_mean(fair_changes)),
        }
    return result


def analyze_forward_candidates(
    candidate_rows: Sequence[Mapping[str, Any]],
    settlement_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select and score the frozen forward policy from already-verified rows."""

    if not candidate_rows:
        raise ForwardLongVolReportError("candidate input is empty")
    candidate_ids: set[str] = set()
    policy_versions: set[str] = set()
    for row in candidate_rows:
        if row.get("schema_version") != LONG_VOL_CANDIDATE_SCHEMA:
            raise ForwardLongVolReportError("candidate schema version mismatch")
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in candidate_ids:
            raise ForwardLongVolReportError("candidate IDs are missing or duplicated")
        candidate_ids.add(candidate_id)
        policy_versions.add(str(row.get("policy_version", "")))
        if row.get("execution_eligible") is not False:
            raise ForwardLongVolReportError("candidate enabled execution")
        if row.get("trading_eligible") is not False:
            raise ForwardLongVolReportError("candidate enabled trading")
    if len(policy_versions) != 1 or "" in policy_versions:
        raise ForwardLongVolReportError("candidate policy version is ambiguous")

    evaluated = [row for row in candidate_rows if row.get("status") == "evaluated"]
    future_by_ticker: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in evaluated:
        if isinstance(row.get("book"), Mapping):
            future_by_ticker[str(row.get("market_ticker", ""))].append(row)
    future_times: dict[str, list[int]] = {}
    for ticker, rows in future_by_ticker.items():
        rows.sort(key=lambda value: int(value["decision_ts_ns"]))
        future_times[ticker] = [int(row["decision_ts_ns"]) for row in rows]

    settlements = _latest_settlements(settlement_rows)
    selected = _first_policy_entries(candidate_rows)
    scored: list[dict[str, Any]] = []
    for row in selected:
        ticker = str(row["market_ticker"])
        side = str(row.get("underdog_side", "")).lower()
        if side not in {"yes", "no"}:
            raise ForwardLongVolReportError("eligible candidate has no valid side")
        decision_ts_ns = int(row["decision_ts_ns"])
        ask = _decimal(row.get("underdog_ask"), "underdog_ask")
        fee = _decimal(row.get("taker_fee_per_contract"), "taker_fee_per_contract")
        entry_fair = _decimal(
            row.get("underdog_fair_conservative"), "underdog_fair_conservative"
        )
        policy = row.get("long_vol_policy")
        if not isinstance(policy, Mapping):
            raise ForwardLongVolReportError("eligible candidate has no policy object")
        coefficient = _decimal(
            policy.get("taker_fee_coefficient"), "taker_fee_coefficient"
        )

        settlement = settlements.get(ticker)
        outcome = str(settlement.get("result", "")).lower() if settlement else ""
        if outcome not in {"yes", "no"}:
            outcome = ""
        won = side == outcome if outcome else None
        terminal_pnl = (
            (ONE if won else ZERO) - ask - fee if won is not None else None
        )

        rows = future_by_ticker.get(ticker, [])
        times = future_times.get(ticker, [])
        markouts: dict[str, Any] = {}
        for seconds in MARKOUT_HORIZONS_SECONDS:
            target_ts_ns = decision_ts_ns + seconds * 1_000_000_000
            index = bisect_left(times, target_ts_ns)
            if index >= len(rows):
                markouts[f"{seconds}s"] = None
                continue
            future = rows[index]
            future_ts_ns = int(future["decision_ts_ns"])
            lag_ms = Decimal(future_ts_ns - target_ts_ns) / Decimal(1_000_000)
            if lag_ms > MAX_MARKOUT_OBSERVATION_LAG_MS:
                markouts[f"{seconds}s"] = None
                continue
            future_bid, future_ask = _side_quote(future, side)
            midpoint = (future_bid + future_ask) / Decimal(2)
            exit_fee = _one_contract_taker_fee(future_bid, coefficient)
            future_fair = _conservative_side_fair(future, side)
            markouts[f"{seconds}s"] = {
                "observation_ts_ns": future_ts_ns,
                "observation_lag_ms": _decimal_text(lag_ms),
                "side_bid": _decimal_text(future_bid),
                "side_ask": _decimal_text(future_ask),
                "side_midpoint": _decimal_text(midpoint),
                "midpoint_pnl": _decimal_text(midpoint - ask - fee),
                "exit_taker_fee": _decimal_text(exit_fee),
                "executable_liquidation_pnl": _decimal_text(
                    future_bid - ask - fee - exit_fee
                ),
                "external_fair_change": _decimal_text(future_fair - entry_fair),
            }

        utc_day = datetime.fromtimestamp(
            decision_ts_ns / 1_000_000_000, timezone.utc
        ).date().isoformat()
        scored.append(
            {
                "candidate_id": row["candidate_id"],
                "market_ticker": ticker,
                "utc_day": utc_day,
                "decision_ts_ns": decision_ts_ns,
                "seconds_remaining": int(row["seconds_remaining"]),
                "side": side,
                "entry_bid": row.get("underdog_bid"),
                "entry_ask": _decimal_text(ask),
                "entry_taker_fee": _decimal_text(fee),
                "conservative_external_fair": _decimal_text(entry_fair),
                "net_external_edge_after_fee": row.get("net_external_edge_after_fee"),
                "charged_external_edge": row.get("charged_external_edge"),
                "displayed_edge_persistence_ms": row.get(
                    "displayed_edge_persistence_ms"
                ),
                "implied_vol_ratio_to_external": row.get(
                    "implied_vol_ratio_to_external"
                ),
                "settlement_label_source": (
                    "kalshi_recorded_result" if outcome else None
                ),
                "kalshi_recorded_result": outcome or None,
                "kalshi_expiration_value": (
                    settlement.get("expiration_value") if settlement else None
                ),
                "won": won,
                "terminal_pnl": _decimal_text(terminal_pnl),
                "markouts": markouts,
            }
        )

    labeled = [entry for entry in scored if entry["terminal_pnl"] is not None]
    pnl = [_decimal(entry["terminal_pnl"], "terminal_pnl") for entry in labeled]
    asks = [_decimal(entry["entry_ask"], "entry_ask") for entry in labeled]
    charged_edges = [
        _decimal(entry["charged_external_edge"], "charged_external_edge")
        for entry in labeled
    ]
    ratios = [
        Decimal(str(entry["implied_vol_ratio_to_external"]))
        for entry in labeled
        if entry["implied_vol_ratio_to_external"] is not None
    ]

    rejection_rows: Counter[str] = Counter()
    rejection_markets: dict[str, set[str]] = defaultdict(set)
    for row in candidate_rows:
        ticker = str(row.get("market_ticker", ""))
        for reason in row.get("shadow_policy_rejection_reasons") or []:
            rejection_rows[str(reason)] += 1
            if ticker:
                rejection_markets[str(reason)].add(ticker)

    screen_rows = [row for row in candidate_rows if row.get("screen_eligible") is True]
    strict_rows = [
        row for row in candidate_rows if row.get("shadow_policy_eligible") is True
    ]
    effective_days = sorted({entry["utc_day"] for entry in labeled})
    return {
        "policy_version": next(iter(policy_versions)),
        "classification": "forward_shadow_research_only",
        "trading_eligible": False,
        "selection_rule": (
            "Earliest shadow_policy_eligible candidate per market; one contract; "
            "no replacement from the looser screen flag."
        ),
        "candidate_funnel": {
            "all_rows": len(candidate_rows),
            "evaluated_rows": len(evaluated),
            "skipped_rows": len(candidate_rows) - len(evaluated),
            "screen_eligible_rows": len(screen_rows),
            "screen_eligible_markets": len(
                {str(row.get("market_ticker")) for row in screen_rows}
            ),
            "shadow_policy_eligible_rows": len(strict_rows),
            "shadow_policy_eligible_markets": len(
                {str(row.get("market_ticker")) for row in strict_rows}
            ),
            "selected_first_entries": len(selected),
            "rejection_reason_rows": dict(sorted(rejection_rows.items())),
            "rejection_reason_markets": {
                reason: len(markets)
                for reason, markets in sorted(rejection_markets.items())
            },
        },
        "primary_results": {
            "entries": len(selected),
            "exchange_labeled_entries": len(labeled),
            "effective_utc_days": len(effective_days),
            "utc_days": effective_days,
            "wins": sum(entry["won"] is True for entry in labeled),
            "win_rate": (
                _decimal_text(
                    Decimal(sum(entry["won"] is True for entry in labeled))
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
            "markouts": _markout_summary(scored),
        },
        "selected_entries": scored,
        "limitations": [
            "The strict policy has fewer than two trading days, so its day-block confidence interval is unidentified.",
            "No candidate can authorize execution; direct fast XAG/SI inputs and the fill-selection model remain unfrozen.",
            "Taker markouts use the first evaluated candidate no more than two seconds after each requested horizon, not an order acknowledgment simulation.",
            "The looser screen-eligible rows are funnel diagnostics and are not scored as if they were frozen-policy entries.",
        ],
        "decision": "CONTINUE_FORWARD_COLLECTION_NO_LIVE_CAPITAL",
    }


def build_forward_long_vol_report(
    root: str | Path,
    collector_run_id: str,
    shadow_run_id: str,
    settlement_run_id: str,
    *,
    metal: str = "silver",
) -> dict[str, Any]:
    storage_root = Path(root)
    collector_manifest = _manifest(storage_root, collector_run_id)
    shadow_manifest = _manifest(storage_root, shadow_run_id)
    settlement_manifest = _manifest(storage_root, settlement_run_id)
    context = shadow_manifest.get("config", {}).get("forward_shadow", {})
    if context.get("collector_run_id") != collector_run_id:
        raise ForwardLongVolReportError("shadow manifest references another collector")
    if context.get("live_orders") is not False or context.get("read_only") is not True:
        raise ForwardLongVolReportError("capture is not read-only shadow")
    if settlement_manifest.get("status") != "completed":
        raise ForwardLongVolReportError("settlement manifest did not complete normally")

    candidates = _stream_rows(
        storage_root, "long_vol_candidate", shadow_run_id, metal=metal
    )
    settlements = _stream_rows(
        storage_root, "kalshi_settlement", settlement_run_id, metal=metal
    )
    analysis = analyze_forward_candidates(candidates, settlements)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "collector_run_id": collector_run_id,
        "shadow_run_id": shadow_run_id,
        "settlement_run_id": settlement_run_id,
        "metal": metal,
        "capture_status": {
            "collector": collector_manifest.get("status"),
            "shadow": shadow_manifest.get("status"),
            "settlement": settlement_manifest.get("status"),
        },
        "input_hash_chains_verified": True,
        **analysis,
    }
    identity = json.dumps(
        report, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    report["report_id"] = hashlib.sha256(identity).hexdigest()
    return report


def write_forward_long_vol_report(
    root: str | Path,
    collector_run_id: str,
    shadow_run_id: str,
    settlement_run_id: str,
    *,
    metal: str = "silver",
) -> Path:
    report = build_forward_long_vol_report(
        root,
        collector_run_id,
        shadow_run_id,
        settlement_run_id,
        metal=metal,
    )
    destination = (
        Path(root)
        / "derived"
        / "forward_long_vol_report"
        / f"report={report['report_id'][:24]}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "report.json"
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ForwardLongVolReportError(f"refusing to overwrite report: {path}")
    path.write_text(payload, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/metals")
    parser.add_argument("--collector-run-id", required=True)
    parser.add_argument("--shadow-run-id", required=True)
    parser.add_argument("--settlement-run-id", required=True)
    parser.add_argument("--metal", default="silver")
    args = parser.parse_args(argv)
    path = write_forward_long_vol_report(
        args.root,
        args.collector_run_id,
        args.shadow_run_id,
        args.settlement_run_id,
        metal=args.metal,
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
