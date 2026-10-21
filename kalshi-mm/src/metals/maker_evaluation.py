"""Offline accessible-edge scorecard for the frozen Silver maker experiment.

Verifies the original capture before evaluating a fixed cancellation-latency
grid. No credentials, provider connection, exchange order, or model refit occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from .baseline_fair import load_baseline_vol_model
from .config import MetalsConfig
from .maker_edge_metrics import summarize_edge_events
from .maker_evidence_health import summarize_input_health
from .maker_policy import MakerEngine, MakerSettings, VERSION, tiny_maker_config
from .maker_replay import settings_from_dict, verified_inputs
from .shadow import ShadowRiskEngine
from .shadow_session import ShadowSession
from .storage import _atomic_write_json

SCHEMA_VERSION = "maker-accessible-edge-evaluation/v1"
SESSION_LABELS = ("normal", "reopen", "holiday", "closed", "diagnostic", "unspecified")
EVALUATION_SPEC = {
    "schema_version": SCHEMA_VERSION,
    "policy_version": VERSION,
    "primary": "original_capture_settings",
    "sensitivity_grid": [
        {"submit_latency_ms": 100, "cancel_latency_ms": delay}
        for delay in (100, 250, 500)
    ],
    "scenario_selection": "fixed_grid_no_best_scenario_selection",
    "latency_interpretation": "simulated_decision_to_arrival_not_measured_exchange_latency",
    "fill_rule": "original_strict_penetration_unknown_queue",
    "fair_horizons_seconds": [1, 5, 30, 60],
    "independent_sampling_unit": "market_window_with_trading_day_clustering",
    "model_refit": False,
    "live_orders": False,
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


class _ObservationSession(ShadowSession):
    """In-memory sink using the unchanged ShadowSession lifecycle methods.

Only persistence is replaced. Risk, place, advance, fill, and cancel remain the
same implementation used by capture; primary full-state/digest equality checks
this substitution on every evaluated tape. No production recorder is created.
"""

    def __init__(self, config: MetalsConfig) -> None:
        self.config = config
        self.engine = ShadowRiskEngine(config.shadow)
        self.counts: Counter[str] = Counter()
        self.rows: list[dict[str, Any]] = []
        self._closed = False

    def _write(self, stream: str, metal: str, event_ts_ns: int,
               record: dict[str, Any]) -> None:
        if stream == "maker_event":
            # Capture serializes at emission. Do not let later fill markouts or
            # mutable nested dictionaries retroactively alter earlier evidence.
            self.rows.append(json.loads(_canonical(record)))

    def close(self, status: str = "completed", error: BaseException | None = None) -> None:
        # No synthetic cancellation/settlement on analysis shutdown.
        self._closed = True


def _run_scenario(inputs: list[dict[str, Any]], model: Any, study_id: str,
                  settings: MakerSettings) -> tuple[dict[str, Any], dict[str, list[dict]]]:
    config = tiny_maker_config(MetalsConfig())
    sessions = {arm: _ObservationSession(config) for arm in ("manual", "baseline")}
    engine = MakerEngine(model, study_id, settings, sessions)
    for event in inputs:
        engine.close(event) if event["kind"] == "stop" else engine.handle(event)
    return json.loads(_canonical(engine.summary())), {arm: s.rows for arm, s in sessions.items()}


def _load_evidence(report_path: Path) -> tuple[dict, Any, str, MakerSettings, list[dict], dict]:
    raw_report = report_path.read_bytes()
    report = json.loads(raw_report)
    if report.get("summary", {}).get("schema_version") != VERSION:
        raise ValueError("evaluation requires the current frozen maker policy report")
    if report["summary"].get("live_orders") is not False:
        raise ValueError("evaluation accepts shadow-only captures")
    settings = settings_from_dict(report["settings"])
    if settings.public_dict() != report["summary"].get("settings"):
        raise ValueError("capture settings disagree with summary")
    model_path = Path(report["baseline_model_path"]).resolve()
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_hash != report["baseline_model_sha256"]:
        raise ValueError("maker baseline model hash changed")
    model, study_id = load_baseline_vol_model(model_path)
    paths = [Path(value).resolve() for value in report["maker_input_paths"]]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("maker input paths missing or duplicated")
    if any(not path.is_file() for path in paths):
        raise ValueError("maker input segment missing")
    before_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    inputs = list(heapq.merge(*(verified_inputs(path) for path in paths),
                             key=lambda row: row["ingestion_sequence"]))
    previous, stops = -1, 0
    for event in inputs:
        for key in ("ingestion_sequence", "receive_ts_ns", "local_monotonic_ts_ns"):
            if type(event.get(key)) is not int or event[key] < 0:
                raise ValueError(f"invalid replay event {key}")
        if event["ingestion_sequence"] <= previous:
            raise ValueError("non-increasing replay input sequence")
        previous = event["ingestion_sequence"]
        stops += event["kind"] == "stop"
    if stops != 1:
        raise ValueError("evaluation requires exactly one recorded stop; no inferred shutdown")
    if type(report["maker_input_counts"]) is not int or len(inputs) != report["maker_input_counts"]:
        raise ValueError("maker replay input count differs from capture")
    after_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if before_hashes != after_hashes or hashlib.sha256(model_path.read_bytes()).hexdigest() != model_hash:
        raise ValueError("evaluation evidence changed while reading")
    provenance = {
        "source_report": str(report_path.resolve()),
        "source_report_sha256": hashlib.sha256(raw_report).hexdigest(),
        "baseline_model_path": str(model_path), "baseline_model_sha256": model_hash,
        "input_segment_sha256": before_hashes, "input_records": len(inputs),
        "trial_research": report.get("trial_research"),
        "research_provenance": report.get("research_provenance"),
        "source_capture_status": report.get("status", "unknown"),
        "source_error_types": report.get("error_types", []),
    }
    return report, model, study_id, settings, inputs, provenance


def _capacity(summary: dict, elapsed_ns: int) -> dict:
    hours = Decimal(elapsed_ns) / Decimal(3_600_000_000_000)
    arms = {}
    for arm, values in summary["arms"].items():
        quantity = sum((Decimal(fill["quantity"]) for fill in values["fills"]), Decimal(0))
        arms[arm] = {
            "simulated_filled_contracts": str(quantity),
            "simulated_contracts_per_capture_hour": str(quantity / hours) if hours > 0 else None,
            "distinct_filled_markets": len({fill["market_ticker"] for fill in values["fills"]}),
            "actual_obtainable_contracts_per_hour": None,
            "expected_net_dollars_per_hour": None,
        }
    return {"observed_capture_hours": str(hours), "includes_shutdown_drain": True, "arms": arms,
            "fixed_data_and_hosting_costs": None, "measured_exchange_fees": None,
            "profit_after_fixed_costs": None,
            "interpretation": "Observed simulation intensity, not actual capacity or a forecast. Arms are not additive."}


def _fill_cohort_comparison(primary: dict, scenario: dict) -> dict:
    """Quantity overlap on the same quote/trade legs, not selected-fill averages."""
    def quantities(rows: list[dict]) -> dict:
        return {(f["logical_quote_id"], f["trade_id"]): Decimal(f["quantity"]) for f in rows}

    arms = {}
    for arm in ("manual", "baseline"):
        original = quantities(primary["arms"][arm]["fills"])
        other = quantities(scenario["arms"][arm]["fills"])
        overlap = sum((min(quantity, other.get(key, Decimal(0)))
                       for key, quantity in original.items()), Decimal(0))
        arms[arm] = {
            "matched_quote_trade_quantity": str(overlap),
            "primary_quantity_not_filled_in_scenario": str(sum(original.values(), Decimal(0)) - overlap),
            "scenario_quantity_not_filled_in_primary": str(sum(other.values(), Decimal(0)) - overlap),
        }
    return {"arms": arms, "matching_key": "logical_quote_id_and_trade_id",
            "interpretation": "Same-tape simulated fill overlap, not causal avoided losses or independent samples."}


def evaluate_report(report_path: Path, output_root: Path, *, session_label: str) -> dict[str, Any]:
    """Verify immutable evidence, then publish a content-addressed offline report."""
    if session_label not in SESSION_LABELS:
        raise ValueError("unknown session label")
    report, model, study_id, settings, inputs, provenance = _load_evidence(report_path)
    primary_summary, primary_rows = _run_scenario(inputs, model, study_id, settings)
    if primary_summary["decision_digest"] != report["summary"]["decision_digest"]:
        raise ValueError("maker replay decision digest differs from capture")
    if _canonical(primary_summary) != _canonical(report["summary"]):
        raise ValueError("maker replay final state differs from capture")

    health = summarize_input_health(inputs, settings)
    # Fills during cancellation drain remain in summary. Include those observed
    # drain seconds in the rate denominator rather than ending it at stop.
    elapsed_ns = max(0, inputs[-1]["local_monotonic_ts_ns"] - inputs[0]["local_monotonic_ts_ns"])
    scenarios = {}

    def summarize(name: str, scenario_settings: MakerSettings, summary: dict, rows: dict) -> None:
        scenarios[name] = {
            "settings": scenario_settings.public_dict(),
            "reproduces_original_capture": name == "primary",
            "decision_digest": summary["decision_digest"],
            "summary": summary,
            "arms": {arm: summarize_edge_events(rows[arm], summary, scenario_settings, arm=arm)
                     for arm in ("manual", "baseline")},
            "capacity": _capacity(summary, elapsed_ns),
            "fill_cohort_comparison_to_primary": _fill_cohort_comparison(primary_summary, summary),
            "live_orders": False,
        }

    summarize("primary", settings, primary_summary, primary_rows)
    for cell in EVALUATION_SPEC["sensitivity_grid"]:
        candidate = replace(settings, **cell)
        if candidate == settings:
            continue
        summary, rows = _run_scenario(inputs, model, study_id, candidate)
        name = f"submit_{candidate.submit_latency_ms}ms_cancel_{candidate.cancel_latency_ms}ms"
        summarize(name, candidate, summary, rows)

    reasons = ["shadow_fills_not_observed_exchange_fills", "actual_queue_access_unknown",
               "independent_live_SI_comparison_unavailable", "multi_day_edge_not_established"]
    if session_label != "normal":
        reasons.append("not_labeled_normal_session")
    if not settings.require_fast:
        reasons.append("exact_only_diagnostic")
    if report.get("status") != "completed" or report.get("error_types"):
        reasons.append("capture_not_cleanly_completed")
    reasons.extend(primary_summary.get("execution_qualification_reasons", []))
    if not any(values["fills"] for values in primary_summary["arms"].values()):
        reasons.append("no_simulated_fill_evidence")
    coverage = health["coverage"]
    if coverage["timer_samples"] == 0:
        reasons.append("no_active_window_timer_evidence")
    if coverage["exact_fresh_timer_samples"] == 0:
        reasons.append("no_fresh_exact_price_samples")
    if settings.require_fast:
        if coverage["fast_fresh_timer_samples"] == 0:
            reasons.append("no_fresh_fast_price_samples")
        if coverage["anchored_estimate_available_timer_samples"] == 0:
            reasons.append("no_causal_fast_anchor_samples")
        if not any(w["anchored_estimate_in_window_timer_samples"] for w in health["windows"]):
            reasons.append("no_in_window_fast_anchor_samples")
    if health["clock"]["discontinuity_detected"]:
        reasons.append("health_clock_discontinuity")
    disposition = "REVIEW_ACCESSIBILITY_NOT_LIVE_QUALIFIED"
    if (health["clock"]["discontinuity_detected"] or primary_summary["clock_fault"]
            or primary_summary["clock_domain_fault"] or primary_summary["uncertain_exposure"]):
        disposition = "REPAIR_CAPTURE_HEALTH"
    elif coverage["timer_samples"] == 0:
        disposition = "NO_ACTIVE_WINDOW_EVIDENCE"
    elif settings.require_fast and "no_in_window_fast_anchor_samples" in reasons:
        disposition = "NO_USABLE_FAST_PRICING_EVIDENCE"
    result = {
        "schema_version": SCHEMA_VERSION, "evaluation_spec": EVALUATION_SPEC,
        "evaluation_spec_sha256": _digest(EVALUATION_SPEC),
        "implementation_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("maker_evaluation.py", "maker_edge_metrics.py", "maker_evidence_health.py",
                         "maker_policy.py", "maker_inputs.py", "maker_replay.py",
                         "shadow.py", "shadow_session.py", "baseline_fair.py")
        },
        "session_label": session_label, "session_label_source": "operator_not_inferred_from_prices",
        "provenance": provenance, "replay_verified": True, "source_health": health,
        "scenarios": scenarios,
        "qualification": {"accessible_edge_demonstrated": False, "live_trading_eligible": False,
                          "disposition": disposition, "reasons": sorted(set(reasons))},
        "live_orders": False, "no_network": True, "model_refit": False,
        "retention": {"inherits_source_research_classification": True,
                      "retention_review_at_ns": (report.get("trial_research") or {}).get("retention_review_at_ns"),
                      "automatic_deletion": False, "retention_extension_authorized": False},
        "limitations": [
            "A negative fair change is not a loss from execution price; bid marks omit exit fees.",
            "Repeated decisions, partial fills, latency scenarios, and arms are not independent outcomes.",
            "Strict penetration omits potentially benign touch fills and cannot certify queue access.",
            "A positive simulated result does not demonstrate accessible edge or authorize live orders.",
            "Latency scenarios replay the observed tape; participant response to hypothetical orders is unknown.",
            "Current-source and anchor health are sampled, not a continuous availability certificate.",
            "No SI data is acquired or retrospectively treated as if it arrived live.",
        ],
    }
    evaluation_id = _digest(result)
    target = output_root / "maker_evaluation" / f"evaluation={evaluation_id}" / "report.json"
    published = {**result, "evaluation_id": evaluation_id}
    if target.exists():
        if _canonical(json.loads(target.read_bytes())) != _canonical(published):
            raise ValueError("existing evaluation artifact differs; refusing overwrite")
    else:
        _atomic_write_json(target, published, create=True)
    return {"report": str(target), "evaluation_id": evaluation_id, "replay_verified": True,
            "input_records": len(inputs), "scenario_count": len(scenarios),
            "accessible_edge_demonstrated": False, "live_orders": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--session-label", choices=SESSION_LABELS, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_report(args.report, args.output_root, session_label=args.session_label), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
