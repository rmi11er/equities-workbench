import hashlib
import json
import math
from pathlib import Path

import pytest

from src.metals.baseline_scorecard import (
    NSEC,
    BaselineScorecardConfig,
    BaselineScorecardError,
    brier_decomposition,
    run_baseline_scorecard,
)
from src.metals.storage import HashChainedJsonlWriter

STUDY_ID = "a" * 64


def _write_chain(path: Path, rows):
    with HashChainedJsonlWriter(path, fsync_every_records=100) as writer:
        for row in rows:
            writer.write(row)
    return path


def _fixture(tmp_path: Path):
    run_dir = (
        tmp_path / "derived" / "baseline_fair" / f"run=baseline-fair-{STUDY_ID[:24]}"
    )
    run_dir.mkdir(parents=True)
    probabilities_path = run_dir / "probabilities.jsonl"
    model_path = run_dir / "model.json"
    report_path = run_dir / "report.json"
    start = 1_787_875_200 * NSEC
    probability_rows = []
    candle_rows = []
    for window_index in range(83):
        open_ts = start + window_index * 900 * NSEC
        ticker = f"WINDOW-{window_index:03d}"
        outcome = window_index % 2
        for minute in range(15):
            decision_ts = open_ts + minute * 60 * NSEC
            remaining_seconds = 900 - minute * 60
            movement = (minute - 7) * 0.00008 + (0.0002 if outcome else -0.0002)
            current = 60.0 * math.exp(movement)
            gaussian = max(0.005, min(0.995, 0.5 + movement * 100))
            empirical = max(0.005, min(0.995, gaussian + 0.01))
            probability_rows.append(
                {
                    "study_id": STUDY_ID,
                    "market_ticker": ticker,
                    "utc_day": "2026-08-28",
                    "window_open_ts_ns": open_ts,
                    "window_close_ts_ns": open_ts + 900 * NSEC,
                    "decision_ts_ns": decision_ts,
                    "remaining_seconds": remaining_seconds,
                    "outcome": outcome,
                    "anchored_current_index": current,
                    "raw_yes_threshold_under_provisional_rounding": 59.9995,
                    "forecast_variance_remaining": 0.000006 * remaining_seconds / 900,
                    "fair_yes_gaussian": gaussian,
                    "fair_yes_empirical": empirical,
                    "final_holdout": False,
                    "trading_eligible": False,
                }
            )
            candle_rows.append(
                {
                    "market_ticker": ticker,
                    "period_interval_minutes": 1,
                    "start_period_ts_ns": open_ts + minute * 60 * NSEC,
                    "end_period_ts_ns": open_ts + (minute + 1) * 60 * NSEC,
                    "yes_bid_close_dollars": str(0.2 + 0.02 * minute),
                    "yes_ask_close_dollars": str(0.22 + 0.02 * minute),
                }
            )
    _write_chain(probabilities_path, probability_rows)
    candles_path = _write_chain(tmp_path / "inputs" / "candles.jsonl", candle_rows)
    model_path.write_text(
        json.dumps(
            {
                "schema_version": "test",
                "study_id": STUDY_ID,
                "model": {
                    "global_variance_15m": 0.000006,
                    "probability_floor": 0.005,
                },
            }
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(
            {
                "study_id": STUDY_ID,
                "research_only": True,
                "trading_eligible": False,
                "formal_final_holdout_accessed": False,
                "artifacts": {
                    "probabilities": str(probabilities_path),
                    "model": str(model_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return candles_path


def test_scorecard_freezes_split_and_scores_causal_common_panel(tmp_path):
    candles_path = _fixture(tmp_path)
    config = BaselineScorecardConfig(baseline_study_id=STUDY_ID)

    first = run_baseline_scorecard(tmp_path, candles_path, config)
    second = run_baseline_scorecard(tmp_path, candles_path, config)

    assert first["scorecard_id"] == second["scorecard_id"]
    assert first["split"] == {
        "unit": "whole_Kalshi_window",
        "development_windows": 55,
        "embargo_windows": 1,
        "test_windows": 27,
        "development_rows": 825,
        "test_rows": 405,
        "common_row_test_rows": 378,
        "common_row_exclusion": (
            "one market-open row per test window has no current-market candle "
            "ended by the decision timestamp"
        ),
    }
    common = first["scorecards"]["common_row_test"]
    assert set(common) == {
        "constant_50",
        "development_fitted_constant_base_rate",
        "current_side_development_rate",
        "flat_volatility_gaussian",
        "frozen_tod_ewma_gaussian_p0",
        "frozen_empirical_cdf",
        "kalshi_historical_yes_midpoint",
    }
    assert all(value["overall"]["observations"] == 378 for value in common.values())
    assert first["confidence_interval"]["status"] == (
        "not_estimable_fewer_than_two_days"
    )
    assert first["formal_final_holdout_accessed"] is False
    assert first["trading_eligible"] is False
    unavailable = {row["name"]: row for row in first["comparators"]}
    assert unavailable["full_signal_model"]["status"] == "unavailable_not_scored"

    output_rows = (
        Path(first["artifacts"]["rows"]).read_text(encoding="utf-8").splitlines()
    )
    parsed = [json.loads(line) for line in output_rows]
    test_open = [
        row
        for row in parsed
        if row["split"] == "test" and row["remaining_seconds"] == 900
    ]
    assert len(test_open) == 27
    assert all(row["kalshi_midpoint_available"] is False for row in test_open)
    assert all("kalshi_yes_midpoint" not in row for row in test_open)


def test_brier_decomposition_reports_named_components():
    rows = [
        {"probability": 0.1, "outcome": 0},
        {"probability": 0.2, "outcome": 0},
        {"probability": 0.8, "outcome": 1},
        {"probability": 0.9, "outcome": 1},
    ]

    result = brier_decomposition(rows, "probability")

    assert result["status"] == "descriptive_single_day"
    assert result["reliability"] >= 0
    assert result["resolution"] >= 0
    assert result["uncertainty"] == pytest.approx(0.25)
    assert result["component_identity_error"] == pytest.approx(0)


def test_scorecard_rejects_tampered_hash_chain(tmp_path):
    candles_path = _fixture(tmp_path)
    lines = candles_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["yes_bid_close_dollars"] = "0.99"
    lines[0] = json.dumps(row)
    candles_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(BaselineScorecardError, match="record hash mismatch"):
        run_baseline_scorecard(
            tmp_path,
            candles_path,
            BaselineScorecardConfig(baseline_study_id=STUDY_ID),
        )


def test_config_rejects_a_different_window_split():
    with pytest.raises(BaselineScorecardError, match="55 development"):
        BaselineScorecardConfig(
            baseline_study_id=hashlib.sha256(b"baseline").hexdigest(),
            development_windows=54,
        )
