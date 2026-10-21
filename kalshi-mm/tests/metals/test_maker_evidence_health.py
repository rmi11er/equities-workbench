from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json

import pytest

from src.metals.maker_evidence_health import summarize_input_health
from src.metals.maker_policy import MakerSettings

NSEC = 1_000_000_000
MSEC = 1_000_000
BASE = int(datetime(2026, 9, 5, tzinfo=timezone.utc).timestamp()) * NSEC


def event(kind, ms, **payload):
    return {"kind": kind, "receive_ts_ns": BASE + ms * MSEC,
            "local_monotonic_ts_ns": NSEC + ms * MSEC, **payload}


def metadata(ms=0, *, ticker="SILVER", opened_ms=0, closed_ms=900_000):
    return event("metadata", ms, metadata={"market_ticker": ticker,
        "open_ts_ns": BASE + opened_ms * MSEC if opened_ms is not None else None,
        "close_ts_ns": BASE + closed_ms * MSEC if closed_ms is not None else None,
        "metadata_receive_ts_ns": BASE + ms * MSEC})


def price(kind, ms, *, source_ms=0, value="100"):
    return event(kind, ms, value=value, source_ts_ns=BASE + source_ms * MSEC)


def summarize(rows, settings=None):
    rows = [{**row, "ingestion_sequence": index} for index, row in enumerate(rows)]
    return summarize_input_health(rows, settings or MakerSettings())


def test_empty_is_unknown_not_healthy():
    result = summarize([])
    assert result["capture"]["stop_seen"] is False
    assert result["coverage"]["anchored_estimate_available_fraction"] is None
    assert result["feeds"]["fast"]["active_timer_source_age_ns"]["max"] is None
    assert result["clock"]["absolute_clock_certified"] is False
    assert result["windows"] == []
    json.dumps(result)


def test_timer_denominator_ignores_message_bursts_and_inactive_windows():
    rows = [metadata(), metadata(ticker="FUTURE", opened_ms=900_000, closed_ms=1_800_000),
            price("fast", 1), price("exact", 2)]
    rows += [price("fast", ms) for ms in range(3, 103)]
    rows += [event("timer", 110), event("timer", 200), event("stop", 201)]
    result = summarize(rows)
    assert result["feeds"]["fast"]["message_count"] == 101
    assert result["feeds"]["fast"]["unique_source_publications"] == 1
    assert result["coverage"]["timer_samples"] == 2
    assert result["coverage"]["anchored_estimate_available_timer_samples"] == 2
    assert result["windows"][1]["coverage"]["timer_samples"] == 0
    assert result["capture"]["distinct_window_intervals_with_active_timer_samples"] == 1
    assert result["capture"]["market_windows_seen"] == 2


def test_duplicates_do_not_refresh_source_or_first_receive_age():
    result = summarize([metadata(), price("fast", 1), price("exact", 2),
                        price("fast", 740), event("timer", 800)])
    fast = result["feeds"]["fast"]
    assert fast["active_timer_source_age_ns"]["max"] == 800 * MSEC
    assert fast["active_timer_receive_age_ns"]["max"] == 799 * MSEC
    assert fast["source_to_receive_lag_ns"]["max"] == 740 * MSEC
    assert fast["first_seen_publication_lag_ns"]["max"] == MSEC
    assert result["coverage"]["anchored_estimate_rejections"] == {"fast_source_stale": 1}
    assert fast["market_session"] is None


def test_new_out_of_order_is_not_duplicate_or_receive_reordering():
    result = summarize([metadata(), price("fast", 100, source_ms=90),
                        price("fast", 110, source_ms=80), price("fast", 120, source_ms=80),
                        price("fast", 130, source_ms=100), event("timer", 140)])
    fast = result["feeds"]["fast"]
    assert fast["unique_source_publications"] == 3
    assert fast["new_out_of_order_publications"] == 1
    assert fast["repeated_source_publications"] == 1
    assert fast["new_high_source_interval_ns"]["max"] == 10 * MSEC
    assert fast["active_timer_source_age_ns"]["max"] == 40 * MSEC
    assert result["clock"]["discontinuity_detected"] is False


def test_causal_anchor_missing_cannot_be_repaired_by_later_fast_or_duplicate_exact():
    result = summarize([metadata(), price("exact", 1), price("fast", 2),
                        price("exact", 3), event("timer", 4)])
    assert result["coverage"]["exact_fresh_timer_samples"] == 1
    assert result["coverage"]["fast_fresh_timer_samples"] == 1
    assert result["coverage"]["anchored_estimate_rejections"] == {
        "missing_source_aligned_fast_anchor": 1}


def test_global_gap_resets_sources_but_ticker_gap_does_not():
    boot = [metadata(), price("fast", 1), price("exact", 2)]
    scoped = summarize(boot + [event("gap", 3, market_ticker="SILVER"), event("timer", 4)])
    assert scoped["coverage"]["anchored_estimate_available_timer_samples"] == 1
    global_gap = summarize(boot + [event("gap", 3), event("timer", 4)])
    assert global_gap["coverage"]["anchored_estimate_rejections"] == {"missing_exact_anchor": 1}
    assert global_gap["feeds"]["exact"]["active_timer_source_age_ns"]["max"] is None
    assert global_gap["feeds"]["exact"]["active_timer_missing_price_samples"] == 1


def test_metadata_known_only_after_arrival_and_open_boundary_inclusive_close_exclusive():
    result = summarize([event("timer", 0), metadata(1, opened_ms=2, closed_ms=4),
                        event("timer", 1), event("timer", 2), event("timer", 4)])
    assert result["coverage"]["timer_samples"] == 1
    assert result["coverage"]["inactive_timer_samples_excluded"] == 3
    assert result["windows"][0]["first_active_timer_ts_ns"] == BASE + 2 * MSEC


def test_stop_and_drain_never_improve_coverage_or_source_counts():
    result = summarize([metadata(), price("fast", 1), event("timer", 2),
                        event("stop", 3), price("exact", 4), event("timer", 5)])
    assert result["capture"]["after_stop_records_excluded"] == 2
    assert result["capture"]["pre_stop_input_records"] == 3
    assert result["feeds"]["exact"]["message_count"] == 0
    assert result["coverage"]["timer_samples"] == 1
    assert result["capture"]["observation_end_ts_ns"] == BASE + 3 * MSEC


def test_clock_step_latches_unhealthy_without_absolute_clock_certification():
    shifted = event("timer", 500)
    shifted["local_monotonic_ts_ns"] = NSEC + 50 * MSEC
    result = summarize([metadata(), price("fast", 1), price("exact", 2), shifted])
    assert result["clock"]["discontinuity_detected"] is True
    assert result["clock"]["discontinuities"][0]["wall_minus_monotonic_delta_ns"] == 450 * MSEC
    assert result["coverage"]["anchored_estimate_rejections"] == {"clock_discontinuity": 1}
    assert result["coverage"]["exact_fresh_timer_samples"] == 0


def test_small_wall_regression_also_fails_closed():
    moved = event("timer", 1)
    moved["local_monotonic_ts_ns"] = NSEC + 3 * MSEC
    result = summarize([metadata(), price("exact", 2), moved])
    assert result["clock"]["discontinuity_detected"] is True


def test_long_silent_gap_does_not_count_as_healthy_duration():
    result = summarize([metadata(), price("fast", 1), price("exact", 2),
                        event("timer", 3), event("timer", 60_003), event("stop", 60_004)])
    assert result["coverage"]["anchored_estimate_available_fraction"] == "0.5"
    assert result["coverage"]["time_weighted_coverage"] is None
    assert result["capture"]["max_inter_event_monotonic_gap_ns"] == 60 * NSEC
    assert result["windows"][0]["max_active_timer_gap_ns"] == 60 * NSEC
    assert result["windows"][0]["full_window_health_established"] is False


def test_distinct_dates_and_intervals_not_message_volume_or_duplicate_tickers():
    result = summarize([metadata(-1), metadata(0, ticker="SAME_INTERVAL"),
                        event("timer", 0), event("timer", 1)])
    assert result["capture"]["utc_dates"] == ["2026-09-04", "2026-09-05"]
    assert result["capture"]["chicago_dates"] == ["2026-09-04"]
    assert result["capture"]["market_windows_seen"] == 2
    assert result["capture"]["distinct_window_intervals_seen"] == 1
    assert result["coverage"]["timer_samples"] == 2
    assert sum(window["coverage"]["timer_samples"] for window in result["windows"]) == 4
    assert result["capture"]["statistical_independence_established"] is False


def test_boundary_sampling_flags_never_claim_continuous_healthy_window():
    result = summarize([metadata(-1, closed_ms=500), event("timer", 0),
                        event("timer", 200), event("timer", 400), event("timer", 500),
                        event("stop", 501)])
    window = result["windows"][0]
    assert window["capture_spans_window_boundaries"] is True
    assert window["timer_samples_span_window_boundaries"] is True
    assert window["bounded_timer_sampling_across_window"] is True
    assert window["full_window_health_established"] is False
    assert window["coverage"]["anchored_estimate_available_timer_samples"] == 0


def test_post_stop_boundary_timer_cannot_fill_missing_window_end():
    result = summarize([metadata(-1, closed_ms=500), event("timer", 0),
                        event("timer", 200), event("timer", 400), event("stop", 500),
                        event("timer", 501)])
    window = result["windows"][0]
    assert window["capture_spans_window_boundaries"] is True
    assert window["timer_samples_span_window_boundaries"] is False


def test_price_fresh_but_previous_market_publication_is_not_window_eligible():
    result = summarize([metadata(), price("fast", 1, source_ms=-10),
                        price("exact", 2, source_ms=-10), event("timer", 3)])
    window = result["windows"][0]
    assert result["coverage"]["anchored_estimate_available_timer_samples"] == 1
    assert window["anchored_estimate_in_window_timer_samples"] == 0
    assert window["exact_fresh_in_window_timer_samples"] == 0


def test_missing_metadata_boundaries_excluded_and_counted():
    result = summarize([metadata(opened_ms=None), event("timer", 1)])
    assert result["capture"]["metadata_versions_without_window"] == 1
    assert result["capture"]["market_windows_seen"] == 0
    assert result["coverage"]["timer_samples"] == 0


def test_frozen_settings_threshold_applies_and_results_json_serializable_without_mutation():
    rows = [metadata(), price("fast", 1), price("exact", 2), event("timer", 20)]
    snapshot = deepcopy(rows)
    result = summarize(rows, replace(MakerSettings(), max_fast_age_ms=10))
    assert result["coverage"]["anchored_estimate_rejections"] == {"fast_source_stale": 1}
    json.dumps(result, allow_nan=False)
    assert rows == snapshot


@pytest.mark.parametrize("rows", [
    [event("unknown", 1)],
    [event("timer", 1, receive_ts_ns=True)],
    [event("timer", 1, local_monotonic_ts_ns=None)],
    [price("fast", 1, source_ms=2)],
    [price("fast", 1, value="NaN")],
    [price("fast", 1, value=True)],
    [price("fast", 1), price("fast", 2, value="101")],
    [price("exact", 1), price("exact", 2, source_ms=1), price("exact", 3, value="101")],
    [metadata(closed_ms=-1)],
    [event("metadata", 1, metadata={})],
    [event("stop", 1), event("stop", 2)],
])
def test_malformed_or_conflicting_inputs_fail_closed(rows):
    with pytest.raises(ValueError):
        summarize(rows)


def test_nonincreasing_sequences_fail_closed():
    rows = [event("timer", 1, ingestion_sequence=2), event("timer", 2, ingestion_sequence=2)]
    with pytest.raises(ValueError, match="sequence"):
        summarize_input_health(rows, MakerSettings())
