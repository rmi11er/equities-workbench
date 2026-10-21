"""Pure, timer-sampled health evidence for an already verified maker input stream.

This is not replay verification, execution qualification, or a market calendar.
Repeated source publications do not refresh prices. Coverage is a count of
observed timer instants, never an assertion of health between observations.
All durations are integer nanoseconds; ratios and means are decimal strings.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo

from .maker_inputs import AnchoredXagUpdater, PriceObservation
from .maker_policy import MakerSettings

MSEC = 1_000_000
NSEC = 1_000_000_000
KINDS = {"control", "metadata", "book", "trade", "exact", "fast", "gap", "timer", "stop"}
CHICAGO = ZoneInfo("America/Chicago")


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid health input {name}")
    return value


def _stats(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered), "min": ordered[0] if ordered else None,
        "p50": ordered[(len(ordered) - 1) // 2] if ordered else None,
        "p95": ordered[(len(ordered) * 95 + 99) // 100 - 1] if ordered else None,
        "max": ordered[-1] if ordered else None,
        "mean": str(Decimal(sum(ordered)) / len(ordered)) if ordered else None,
    }


def _fraction(numerator: int, denominator: int) -> str | None:
    return str(Decimal(numerator) / denominator) if denominator else None


def _dates(times: list[int], zone=timezone.utc) -> list[str]:
    return sorted({datetime.fromtimestamp(stamp // NSEC, zone).date().isoformat()
                   for stamp in times})


def _coverage() -> dict[str, Any]:
    return {"timer_samples": 0, "exact_fresh_timer_samples": 0,
            "fast_fresh_timer_samples": 0, "anchored_estimate_available_timer_samples": 0,
            "anchored_estimate_rejections": Counter()}


def _finish_coverage(value: dict[str, Any]) -> dict[str, Any]:
    total = value["timer_samples"]
    return {**value, "anchored_estimate_rejections": dict(sorted(
        value["anchored_estimate_rejections"].items())),
        "exact_fresh_fraction": _fraction(value["exact_fresh_timer_samples"], total),
        "fast_fresh_fraction": _fraction(value["fast_fresh_timer_samples"], total),
        "anchored_estimate_available_fraction": _fraction(
            value["anchored_estimate_available_timer_samples"], total)}


def summarize_input_health(inputs: list[dict], settings: MakerSettings) -> dict[str, Any]:
    """Summarize ingestion-ordered rows, excluding stop/drain from sample counts.

    The caller must verify row hashes and replay separately. Bad envelope/schema
    fields, future source timestamps, conflicting publications, or non-increasing
    ingestion sequences raise ValueError rather than producing reassuring data.
    Clock jumps are reported, latch sampling unhealthy, and do not certify the
    absolute relationship between a provider clock and the local clock.
    """
    if type(inputs) is not list or not isinstance(settings, MakerSettings):
        raise ValueError("health requires a list of inputs and MakerSettings")
    updater = AnchoredXagUpdater()
    feed_data = {name: {"seen": {}, "rows": [], "unique_rows": [],
                        "new_high_source_intervals": [], "out_of_order": 0,
                        "latest_high_source": None} for name in ("exact", "fast")}
    current_windows: dict[str, tuple[str, int, int] | None] = {}
    windows: dict[tuple[str, int, int], dict[str, Any]] = {}
    coverage = _coverage()
    ages = {name: {"source": [], "receive": [], "missing": 0,
                   "source_fresh": 0, "receive_fresh": 0} for name in feed_data}
    counts: Counter[str] = Counter()
    clock_events, all_gaps, timers, active_timers, pre_times = [], [], [], [], []
    mono_gaps, residuals = [], []
    previous = None
    previous_sequence = -1
    stop = None
    after_stop = 0
    clock_fault = False
    metadata_without_window = 0
    for row in inputs:
        if (type(row) is not dict or type(row.get("kind")) is not str
                or row["kind"] not in KINDS):
            raise ValueError("invalid health input kind")
        for key in ("receive_ts_ns", "local_monotonic_ts_ns", "ingestion_sequence"):
            _integer(row.get(key), key)
        if row["ingestion_sequence"] <= previous_sequence:
            raise ValueError("non-increasing health input sequence")
        previous_sequence = row["ingestion_sequence"]
        kind, now = row["kind"], row["receive_ts_ns"]
        observation = None
        if kind in feed_data:
            try:
                if type(row.get("value")) not in (str, int, float, Decimal):
                    raise ValueError("invalid price type")
                observation = PriceObservation(Decimal(str(row["value"])),
                    row["source_ts_ns"], now, row["ingestion_sequence"])
            except (KeyError, InvalidOperation, TypeError, ValueError) as error:
                raise ValueError("invalid health price observation") from error
        if kind == "metadata":
            meta = row.get("metadata")
            if type(meta) is not dict or type(meta.get("market_ticker")) is not str or not meta["market_ticker"]:
                raise ValueError("invalid health market metadata")
            _integer(meta.get("metadata_receive_ts_ns"), "metadata_receive_ts_ns")
            if meta["metadata_receive_ts_ns"] > now:
                raise ValueError("noncausal health metadata")
            if "open_ts_ns" not in meta or "close_ts_ns" not in meta:
                raise ValueError("missing health window fields")
            opened, closed = meta["open_ts_ns"], meta["close_ts_ns"]
            for name, value in (("open_ts_ns", opened), ("close_ts_ns", closed)):
                if value is not None:
                    _integer(value, name)
            if opened is not None and closed is not None and closed <= opened:
                raise ValueError("invalid health market window")
        if stop is not None:
            if kind == "stop":
                raise ValueError("multiple health stop events")
            after_stop += 1
            continue
        if previous is not None:
            wall_delta = now - previous["receive_ts_ns"]
            mono_delta = row["local_monotonic_ts_ns"] - previous["local_monotonic_ts_ns"]
            residuals.append(abs(wall_delta - mono_delta))
            if mono_delta >= 0:
                mono_gaps.append(mono_delta)
            if mono_delta < 0 or wall_delta < 0 or abs(wall_delta - mono_delta) > settings.max_clock_jump_ms * MSEC:
                clock_fault = True
                clock_events.append({"ingestion_sequence": row["ingestion_sequence"],
                    "wall_delta_ns": wall_delta, "monotonic_delta_ns": mono_delta,
                    "wall_minus_monotonic_delta_ns": wall_delta - mono_delta})
        previous = row
        if kind == "stop":
            stop = row
            continue
        pre_times.append(now)
        counts[kind] += 1
        if kind == "metadata":
            ticker = meta["market_ticker"]
            if opened is None or closed is None:
                current_windows[ticker] = None
                metadata_without_window += 1
            else:
                key = (ticker, opened, closed)
                current_windows[ticker] = key
                if key not in windows:
                    windows[key] = {"market_ticker": ticker, "open_ts_ns": opened,
                        "close_ts_ns": closed, "first_metadata_receive_ts_ns": now,
                        "samples": [], "coverage": _coverage(),
                        "exact_source_inside_window_timer_samples": 0,
                        "fast_source_inside_window_timer_samples": 0,
                        "exact_fresh_in_window_timer_samples": 0,
                        "anchored_estimate_in_window_timer_samples": 0}
        elif kind in feed_data:
            feed = feed_data[kind]
            source = observation.source_ts_ns
            feed["rows"].append(observation)
            if source in feed["seen"]:
                if feed["seen"][source] != observation.value:
                    raise ValueError(f"conflicting health {kind} publication")
            else:
                feed["seen"][source] = observation.value
                feed["unique_rows"].append(observation)
                high = feed["latest_high_source"]
                if high is not None and source < high:
                    feed["out_of_order"] += 1
                else:
                    if high is not None:
                        feed["new_high_source_intervals"].append(source - high)
                    feed["latest_high_source"] = source
            if not clock_fault:
                getattr(updater, f"observe_{kind}")(observation)
        elif kind == "gap":
            all_gaps.append({"receive_ts_ns": now, "market_ticker": row.get("market_ticker")})
            if row.get("market_ticker") is None:
                updater = AnchoredXagUpdater()
        elif kind == "timer":
            timers.append(now)
            active = [windows[key] for key in current_windows.values() if key is not None
                      and key[1] <= now < key[2]]
            if not active:
                continue
            active_timers.append(now)
            fresh = {}
            for name, maximum in (("exact", settings.max_exact_age_ms),
                                  ("fast", settings.max_fast_age_ms)):
                price = getattr(updater, f"latest_{name}")
                age = ages[name]
                if price is None:
                    age["missing"] += 1
                    fresh[name] = False
                else:
                    source_age, receive_age = now - price.source_ts_ns, now - price.receive_ts_ns
                    age["source"].append(source_age)
                    age["receive"].append(receive_age)
                    source_fresh = not clock_fault and 0 <= source_age <= maximum * MSEC
                    receive_fresh = not clock_fault and 0 <= receive_age <= maximum * MSEC
                    age["source_fresh"] += int(source_fresh)
                    age["receive_fresh"] += int(receive_fresh)
                    fresh[name] = source_fresh and receive_fresh
            rejection = "clock_discontinuity" if clock_fault else None
            if rejection is None:
                try:
                    updater.estimate(now, max_exact_age_ms=settings.max_exact_age_ms,
                        max_fast_age_ms=settings.max_fast_age_ms,
                        max_anchor_gap_ms=settings.max_anchor_gap_ms,
                        max_move_bps=settings.max_fast_move_bps)
                except ValueError as error:
                    rejection = str(error)
            # A valid source-aligned estimate can still be outside this market's
            # source-time window; expose that separately from generic feed health.
            for target in [coverage] + [window["coverage"] for window in active]:
                target["timer_samples"] += 1
                target["exact_fresh_timer_samples"] += int(fresh["exact"])
                target["fast_fresh_timer_samples"] += int(fresh["fast"])
                target["anchored_estimate_available_timer_samples"] += int(rejection is None)
                if rejection:
                    target["anchored_estimate_rejections"][rejection] += 1
            for window in active:
                window["samples"].append(now)
                inside = {}
                for name in feed_data:
                    price = getattr(updater, f"latest_{name}")
                    key = f"{name}_source_inside_window_timer_samples"
                    inside[name] = price is not None and window["open_ts_ns"] <= price.source_ts_ns < window["close_ts_ns"]
                    window[key] += int(inside[name])
                window["exact_fresh_in_window_timer_samples"] += int(fresh["exact"] and inside["exact"])
                window["anchored_estimate_in_window_timer_samples"] += int(
                    rejection is None and inside["exact"] and inside["fast"])

    feeds = {}
    for name, feed in feed_data.items():
        rows, unique = feed["rows"], feed["unique_rows"]
        age = ages[name]
        feeds[name] = {
            "message_count": len(rows), "unique_source_publications": len(unique),
            "repeated_source_publications": len(rows) - len(unique),
            "new_out_of_order_publications": feed["out_of_order"],
            "first_source_ts_ns": min(feed["seen"]) if unique else None,
            "last_source_ts_ns": max(feed["seen"]) if unique else None,
            "source_to_receive_lag_ns": _stats([row.receive_ts_ns - row.source_ts_ns for row in rows]),
            "first_seen_publication_lag_ns": _stats([row.receive_ts_ns - row.source_ts_ns for row in unique]),
            "message_receive_interval_ns": _stats([b.receive_ts_ns - a.receive_ts_ns for a, b in zip(rows, rows[1:])]),
            "new_high_source_interval_ns": _stats(feed["new_high_source_intervals"]),
            "active_timer_source_age_ns": _stats(age["source"]),
            "active_timer_receive_age_ns": _stats(age["receive"]),
            "active_timer_missing_price_samples": age["missing"],
            "source_fresh_active_timer_samples": age["source_fresh"],
            "receive_fresh_active_timer_samples": age["receive_fresh"],
            "market_session": None,
        }
    start = pre_times[0] if pre_times else None
    end = stop["receive_ts_ns"] if stop else (pre_times[-1] if pre_times else None)
    tolerance = settings.decision_interval_ms * MSEC
    window_rows = []
    for window in sorted(windows.values(), key=lambda value: (value["open_ts_ns"], value["market_ticker"])):
        samples = window.pop("samples")
        opened, closed = window["open_ts_ns"], window["close_ts_ns"]
        gaps = [b - a for a, b in zip(samples, samples[1:])]
        boundary_index = bisect_left(timers, closed) if not clock_fault else len(timers)
        close_sample = timers[boundary_index] if boundary_index < len(timers) else None
        spans = bool(not clock_fault and start is not None and start <= opened and end >= closed
                     and window["first_metadata_receive_ts_ns"] <= opened)
        boundary_samples = bool(spans and samples and samples[0] - opened <= tolerance
            and closed - samples[-1] <= tolerance and close_sample is not None
            and close_sample - closed <= tolerance)
        gap_count = sum(opened <= gap["receive_ts_ns"] < closed and
                        gap["market_ticker"] in (None, window["market_ticker"]) for gap in all_gaps)
        window_rows.append({**window, "coverage": _finish_coverage(window["coverage"]),
            "first_active_timer_ts_ns": samples[0] if samples else None,
            "last_active_timer_ts_ns": samples[-1] if samples else None,
            "max_active_timer_gap_ns": max(gaps) if gaps else None,
            "capture_spans_window_boundaries": spans,
            "timer_samples_span_window_boundaries": boundary_samples,
            "bounded_timer_sampling_across_window": bool(boundary_samples and not gap_count
                and (not gaps or max(gaps) <= tolerance)),
            "explicit_gap_events_in_window": gap_count,
            "full_window_health_established": False})
    return {
        "schema_version": "maker-evidence-input-health/v1",
        "capture": {"input_records": len(inputs), "pre_stop_input_records": len(pre_times),
            "pre_stop_kind_counts": dict(sorted(counts.items())), "stop_seen": stop is not None,
            "after_stop_records_excluded": after_stop, "first_receive_ts_ns": start,
            "observation_end_ts_ns": end,
            "observed_span_ns": end - start if start is not None else None,
            "max_inter_event_monotonic_gap_ns": max(mono_gaps) if mono_gaps else None,
            "utc_dates": _dates(pre_times), "chicago_dates": _dates(pre_times, CHICAGO),
            "active_timer_utc_dates": _dates(active_timers),
            "active_timer_chicago_dates": _dates(active_timers, CHICAGO),
            "metadata_versions_without_window": metadata_without_window,
            "market_windows_seen": len(windows),
            "distinct_window_intervals_seen": len({(key[1], key[2]) for key in windows}),
            "distinct_window_intervals_with_active_timer_samples": len({
                (window["open_ts_ns"], window["close_ts_ns"]) for window in window_rows
                if window["coverage"]["timer_samples"]}),
            "statistical_independence_established": False},
        "feeds": feeds,
        "coverage": {**_finish_coverage(coverage), "all_pre_stop_timer_samples": len(timers),
            "inactive_timer_samples_excluded": len(timers) - len(active_timers),
            "denominator": "timer events with at least one causally known open market window",
            "time_weighted_coverage": None, "full_window_health_established": False,
            "window_boundary_sampling_tolerance_ns": tolerance},
        "windows": window_rows,
        "clock": {"discontinuity_detected": clock_fault, "discontinuities": clock_events,
            "max_abs_wall_minus_monotonic_delta_ns": max(residuals) if residuals else None,
            "jump_limit_ns": settings.max_clock_jump_ms * MSEC,
            "absolute_clock_certified": False},
        "limitations": [
            "Input hashes and deterministic replay must be verified by the caller.",
            "Samples are timer instants, not message counts or time-weighted coverage; no health is extrapolated across gaps.",
            "Distinct market intervals/dates are replication counts, not proof of statistical independence.",
            "Out-of-order counts include only first-seen source timestamps below the prior high water; repeats are not out-of-order publications.",
            "Receive cadence is transport activity, not source publication freshness; timestamp lag is not certified network latency.",
            "Source ages use the updater's retained first arrival; duplicate publications do not refresh that arrival.",
            "Source health and anchored-estimate availability do not establish model, book, control, or execution eligibility.",
            "Market-session/closed flags are not persisted in maker inputs and cannot be inferred here.",
            "Stop and drain rows are excluded from source and timer denominators; stop is only an observation-end boundary.",
        ],
    }
