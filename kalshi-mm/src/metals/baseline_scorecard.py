"""Research-only comparator scorecard for a frozen metals baseline fair.

The scorecard never refits the frozen volatility model and never opens a formal
holdout.  It applies the same whole-window 55 development / 1 embargo / 27 test
split used by the exploratory signal cards.  Historical Kalshi candles are aligned
strictly by candle end time; their boundary timestamps do not establish live
publication latency and their midpoint is not executable after fees.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .storage import HashChainedJsonlWriter, update_run_manifest, write_run_manifest

NSEC = 1_000_000_000
WINDOW_SECONDS = 15 * 60
SCHEMA_VERSION = "metals-baseline-comparator-scorecard/v3"
USAGE_CLASSIFICATION = "preexisting_research_and_hypothesis_generation_only"


class BaselineScorecardError(ValueError):
    """Raised when a scorecard input, split, or boundary alignment is invalid."""


@dataclass(frozen=True)
class BaselineScorecardConfig:
    baseline_study_id: str
    development_windows: int = 55
    embargo_windows: int = 1
    test_windows: int = 27
    expected_rows_per_window: int = 15
    scoring_probability_floor: float = 0.005
    session_timezone: str = "America/Chicago"

    def __post_init__(self) -> None:
        if len(self.baseline_study_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.baseline_study_id
        ):
            raise BaselineScorecardError(
                "baseline_study_id must be a lowercase SHA-256 identifier"
            )
        if (self.development_windows, self.embargo_windows, self.test_windows) != (
            55,
            1,
            27,
        ):
            raise BaselineScorecardError(
                "this scorecard freezes the 55 development / 1 embargo / 27 test split"
            )
        if self.expected_rows_per_window != 15:
            raise BaselineScorecardError("each 15-minute window must have 15 rows")
        if not 0 < self.scoring_probability_floor < 0.5:
            raise BaselineScorecardError(
                "scoring_probability_floor must be between zero and 0.5"
            )
        if self.session_timezone != "America/Chicago":
            raise BaselineScorecardError(
                "the exploratory session clock is frozen to America/Chicago"
            )


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_input(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise BaselineScorecardError(f"input escapes storage root: {path}") from error
    if not resolved.is_file():
        raise BaselineScorecardError(f"input is not a file: {path}")
    return resolved


def _read_hash_chained(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_previous: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise BaselineScorecardError(
                    f"invalid JSON at {path}:{line_number}"
                ) from error
            if not isinstance(row, dict):
                raise BaselineScorecardError(f"non-object at {path}:{line_number}")
            record_hash = row.get("record_sha256")
            payload = dict(row)
            payload.pop("record_sha256", None)
            computed = hashlib.sha256(_canonical(payload).encode()).hexdigest()
            if computed != record_hash:
                raise BaselineScorecardError(
                    f"record hash mismatch at {path}:{line_number}"
                )
            if row.get("previous_record_sha256") != expected_previous:
                raise BaselineScorecardError(
                    f"record chain mismatch at {path}:{line_number}"
                )
            expected_previous = str(record_hash)
            rows.append(row)
    if not rows:
        raise BaselineScorecardError(f"empty input: {path}")
    return rows


def _clip_probability(probability: float, floor: float) -> float:
    if not math.isfinite(probability):
        raise BaselineScorecardError("forecast probability is not finite")
    return max(floor, min(1 - floor, probability))


def _normal_survival(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def _metric(
    rows: Sequence[Mapping[str, Any]], probability_field: str, floor: float
) -> dict[str, Any]:
    if not rows:
        return {
            "observations": 0,
            "windows": 0,
            "effective_utc_days": 0,
            "brier": None,
            "log_loss": None,
            "mean_probability": None,
            "outcome_rate": None,
        }
    raw_probabilities = [float(row[probability_field]) for row in rows]
    outcomes = [int(row["outcome"]) for row in rows]
    probabilities = [_clip_probability(value, floor) for value in raw_probabilities]
    return {
        "observations": len(rows),
        "windows": len({str(row["market_ticker"]) for row in rows}),
        "effective_utc_days": len({str(row["utc_day"]) for row in rows}),
        "brier": sum(
            (probability - outcome) ** 2
            for probability, outcome in zip(raw_probabilities, outcomes)
        )
        / len(rows),
        "log_loss": -sum(
            outcome * math.log(probability) + (1 - outcome) * math.log(1 - probability)
            for probability, outcome in zip(probabilities, outcomes)
        )
        / len(rows),
        "mean_probability": sum(raw_probabilities) / len(rows),
        "outcome_rate": sum(outcomes) / len(rows),
    }


def _probability_bin(probability: float) -> int:
    return min(9, max(0, int(probability * 10)))


def _calibration(
    rows: Sequence[Mapping[str, Any]], probability_field: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index in range(10):
        selected = [
            row
            for row in rows
            if _probability_bin(float(row[probability_field])) == index
        ]
        if not selected:
            continue
        result.append(
            {
                "lower": index / 10,
                "upper": (index + 1) / 10,
                "observations": len(selected),
                "windows": len({str(row["market_ticker"]) for row in selected}),
                "mean_probability": sum(
                    float(row[probability_field]) for row in selected
                )
                / len(selected),
                "outcome_rate": sum(int(row["outcome"]) for row in selected)
                / len(selected),
            }
        )
    return result


def brier_decomposition(
    rows: Sequence[Mapping[str, Any]], probability_field: str
) -> dict[str, Any]:
    """Return a descriptive decile-binned Murphy decomposition.

    The component identity applies to forecasts replaced by their probability-bin
    means.  The raw Brier score remains separately reported because continuous
    forecasts vary within bins.
    """

    if not rows:
        return {
            "status": "unavailable_no_observations",
            "reliability": None,
            "resolution": None,
            "uncertainty": None,
        }
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_probability_bin(float(row[probability_field]))].append(row)
    count = len(rows)
    base_rate = sum(int(row["outcome"]) for row in rows) / count
    reliability = 0.0
    resolution = 0.0
    for selected in groups.values():
        weight = len(selected) / count
        mean_probability = sum(float(row[probability_field]) for row in selected) / len(
            selected
        )
        outcome_rate = sum(int(row["outcome"]) for row in selected) / len(selected)
        reliability += weight * (mean_probability - outcome_rate) ** 2
        resolution += weight * (outcome_rate - base_rate) ** 2
    uncertainty = base_rate * (1 - base_rate)
    reconstructed = reliability - resolution + uncertainty
    bin_mean_by_index = {
        index: sum(float(row[probability_field]) for row in selected) / len(selected)
        for index, selected in groups.items()
    }
    binned_brier = (
        sum(
            (
                bin_mean_by_index[_probability_bin(float(row[probability_field]))]
                - int(row["outcome"])
            )
            ** 2
            for row in rows
        )
        / count
    )
    return {
        "status": "descriptive_single_day",
        "method": "murphy_decomposition_over_fixed_probability_deciles",
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_of_bin_mean_forecasts": binned_brier,
        "reconstructed_brier_of_bin_mean_forecasts": reconstructed,
        "component_identity_error": binned_brier - reconstructed,
        "interpretation_caveat": (
            "Components describe forecasts grouped into fixed deciles; they are not "
            "day-stable estimates and need not reconstruct the raw continuous-forecast "
            "Brier score."
        ),
    }


def _remaining_time_group(row: Mapping[str, Any]) -> str:
    minutes = int(row["remaining_seconds"]) // 60
    if 11 <= minutes <= 15:
        return "11_to_15_minutes"
    if 7 <= minutes <= 10:
        return "7_to_10_minutes"
    if 3 <= minutes <= 6:
        return "3_to_6_minutes"
    if 1 <= minutes <= 2:
        return "1_to_2_minutes"
    raise BaselineScorecardError(f"unsupported remaining time: {minutes} minutes")


def _moneyness_group(row: Mapping[str, Any]) -> str:
    value = float(row["standardized_moneyness_z"])
    if value < -1.5:
        return "z_below_-1.5"
    if value < -0.5:
        return "z_-1.5_to_-0.5"
    if value <= 0.5:
        return "z_-0.5_to_0.5"
    if value <= 1.5:
        return "z_0.5_to_1.5"
    return "z_above_1.5"


def _session_group(timestamp_ns: int, timezone_name: str) -> str:
    hour = (
        datetime.fromtimestamp(timestamp_ns / NSEC, UTC)
        .astimezone(ZoneInfo(timezone_name))
        .hour
    )
    if hour < 7:
        return "ct_overnight_00_to_07"
    if hour < 12:
        return "ct_us_morning_07_to_12"
    if hour < 17:
        return "ct_us_afternoon_12_to_17"
    return "ct_evening_17_to_24"


def _group_metrics(
    rows: Sequence[Mapping[str, Any]],
    probability_field: str,
    floor: float,
    group_field: str,
    ordered_groups: Sequence[str],
) -> dict[str, Any]:
    return {
        group: _metric(
            [row for row in rows if row[group_field] == group],
            probability_field,
            floor,
        )
        for group in ordered_groups
    }


def _complete_metrics(
    rows: Sequence[Mapping[str, Any]], probability_field: str, floor: float
) -> dict[str, Any]:
    return {
        "overall": _metric(rows, probability_field, floor),
        "by_remaining_time": _group_metrics(
            rows,
            probability_field,
            floor,
            "remaining_time_group",
            (
                "11_to_15_minutes",
                "7_to_10_minutes",
                "3_to_6_minutes",
                "1_to_2_minutes",
            ),
        ),
        "by_standardized_moneyness": _group_metrics(
            rows,
            probability_field,
            floor,
            "moneyness_group",
            (
                "z_below_-1.5",
                "z_-1.5_to_-0.5",
                "z_-0.5_to_0.5",
                "z_0.5_to_1.5",
                "z_above_1.5",
            ),
        ),
        "by_session": _group_metrics(
            rows,
            probability_field,
            floor,
            "session",
            (
                "ct_overnight_00_to_07",
                "ct_us_morning_07_to_12",
                "ct_us_afternoon_12_to_17",
                "ct_evening_17_to_24",
            ),
        ),
        "calibration_deciles": _calibration(rows, probability_field),
        "brier_decomposition": brier_decomposition(rows, probability_field),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen baseline comparator scorecard",
        "",
        "Primary comparisons use the historically boundary-aligned common-row test panel shared with the Kalshi midpoint.",
        "",
        "| Comparator | Rows | Brier | Log loss | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for comparator in report["comparators"]:
        name = comparator["name"]
        status = comparator["status"]
        metrics = report["scorecards"]["common_row_test"].get(name)
        if metrics is None:
            lines.append(f"| {name} | — | — | — | {status} |")
            continue
        overall = metrics["overall"]
        lines.append(
            "| {} | {} | {:.6f} | {:.6f} | {} |".format(
                name,
                overall["observations"],
                overall["brier"],
                overall["log_loss"],
                status,
            )
        )
    lines.extend(
        [
            "",
            "Historical candle boundaries do not measure live publication latency. The midpoint is not an executable price after spread and fees.",
            "",
            "All confidence and calibration results are descriptive because the test panel contains one UTC day.",
            "",
        ]
    )
    return "\n".join(lines)


def run_baseline_scorecard(
    storage_root: Path,
    kalshi_candlesticks_path: Path,
    config: BaselineScorecardConfig,
) -> dict[str, Any]:
    """Score frozen baseline comparators without opening or creating a holdout."""

    root = storage_root.resolve()
    baseline_dir = (
        root
        / "derived"
        / "baseline_fair"
        / f"run=baseline-fair-{config.baseline_study_id[:24]}"
    )
    baseline_report_path = _resolve_input(root, baseline_dir / "report.json")
    baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    if (
        baseline.get("study_id") != config.baseline_study_id
        or baseline.get("formal_final_holdout_accessed") is not False
        or baseline.get("trading_eligible") is not False
        or baseline.get("research_only") is not True
    ):
        raise BaselineScorecardError(
            "baseline artifact is not research-scorecard eligible"
        )
    baseline_rows_path = _resolve_input(
        root, Path(str(baseline["artifacts"]["probabilities"]))
    )
    baseline_model_path = _resolve_input(
        root, Path(str(baseline["artifacts"]["model"]))
    )
    candles_path = _resolve_input(root, kalshi_candlesticks_path)
    inputs = {
        "baseline_report": baseline_report_path,
        "baseline_probabilities": baseline_rows_path,
        "baseline_model": baseline_model_path,
        "kalshi_historical_candlesticks": candles_path,
    }
    input_hashes = {name: _sha256_path(path) for name, path in inputs.items()}
    public_config = asdict(config)
    scorecard_id = hashlib.sha256(
        _canonical(
            {
                "schema_version": SCHEMA_VERSION,
                "usage_classification": USAGE_CLASSIFICATION,
                "config": public_config,
                "inputs": input_hashes,
            }
        ).encode()
    ).hexdigest()
    run_id = f"baseline-scorecard-{scorecard_id[:24]}"
    output_dir = root / "derived" / "baseline_scorecard" / f"run={run_id}"
    report_path = output_dir / "report.json"
    rows_path = output_dir / "score_rows.jsonl"
    markdown_path = output_dir / "REPORT.md"
    manifest_path = root / "manifests" / f"run={run_id}" / "manifest.json"
    if report_path.exists() and manifest_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))

    write_run_manifest(
        root,
        run_id,
        {
            "component": "baseline_comparator_scorecard",
            "schema_version": SCHEMA_VERSION,
            "usage_classification": USAGE_CLASSIFICATION,
            "research_only": True,
            "formal_final_holdout_accessed": False,
            "trading_eligible": False,
            "network_access": False,
            "live_orders": False,
            "config": public_config,
            "input_sha256": input_hashes,
        },
        output_paths=inputs.values(),
    )
    try:
        baseline_rows = _read_hash_chained(baseline_rows_path)
        model_artifact = json.loads(baseline_model_path.read_text(encoding="utf-8"))
        if model_artifact.get("study_id") != config.baseline_study_id:
            raise BaselineScorecardError("frozen model study ID does not match")
        model = model_artifact.get("model")
        if not isinstance(model, dict):
            raise BaselineScorecardError("frozen model payload is missing")
        global_variance = float(model["global_variance_15m"])
        if not math.isfinite(global_variance) or global_variance <= 0:
            raise BaselineScorecardError("frozen global variance is invalid")
        if float(model.get("probability_floor", -1)) != (
            config.scoring_probability_floor
        ):
            raise BaselineScorecardError(
                "scorecard probability floor must match the frozen baseline"
            )

        expected_remaining = set(range(60, WINDOW_SECONDS + 1, 60))
        grouped_baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in baseline_rows:
            if (
                row.get("study_id") != config.baseline_study_id
                or row.get("final_holdout") is not False
                or row.get("trading_eligible") is not False
            ):
                raise BaselineScorecardError(
                    "probability row provenance or eligibility is invalid"
                )
            grouped_baseline[str(row["market_ticker"])].append(row)
        windows: list[tuple[int, str]] = []
        for ticker, rows in grouped_baseline.items():
            if len(rows) != config.expected_rows_per_window:
                raise BaselineScorecardError(f"incomplete probability window: {ticker}")
            if {int(row["remaining_seconds"]) for row in rows} != expected_remaining:
                raise BaselineScorecardError(f"invalid decision grid: {ticker}")
            if len({int(row["outcome"]) for row in rows}) != 1:
                raise BaselineScorecardError(
                    f"inconsistent outcome in window: {ticker}"
                )
            windows.append((int(rows[0]["window_open_ts_ns"]), ticker))
        windows.sort()
        expected_windows = (
            config.development_windows + config.embargo_windows + config.test_windows
        )
        if len(windows) != expected_windows:
            raise BaselineScorecardError(
                f"expected exactly {expected_windows} complete windows; found {len(windows)}"
            )
        development_tickers = {
            ticker for _, ticker in windows[: config.development_windows]
        }
        embargo_end = config.development_windows + config.embargo_windows
        embargo_tickers = {
            ticker for _, ticker in windows[config.development_windows : embargo_end]
        }
        test_tickers = {ticker for _, ticker in windows[embargo_end:]}

        development_outcomes = [
            int(grouped_baseline[ticker][0]["outcome"])
            for _, ticker in windows[: config.development_windows]
        ]
        development_base_rate = sum(development_outcomes) / len(development_outcomes)

        candles = _read_hash_chained(candles_path)
        candle_times: dict[str, list[int]] = defaultdict(list)
        candle_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for candle in candles:
            ticker = str(candle.get("market_ticker", ""))
            if ticker not in grouped_baseline:
                continue
            if int(candle.get("period_interval_minutes", 0)) != 1:
                continue
            bid_value = candle.get("yes_bid_close_dollars")
            ask_value = candle.get("yes_ask_close_dollars")
            if bid_value is None or ask_value is None:
                continue
            bid = float(bid_value)
            ask = float(ask_value)
            if not (0 <= bid <= ask <= 1):
                raise BaselineScorecardError(f"invalid Kalshi candle quote: {ticker}")
            end_ts = int(candle["end_period_ts_ns"])
            candle_times[ticker].append(end_ts)
            candle_values[ticker].append((bid, ask))
        for ticker, timestamps in candle_times.items():
            combined = sorted(zip(timestamps, candle_values[ticker]))
            if len({timestamp for timestamp, _ in combined}) != len(combined):
                raise BaselineScorecardError(f"duplicate Kalshi candle end: {ticker}")
            candle_times[ticker] = [timestamp for timestamp, _ in combined]
            candle_values[ticker] = [value for _, value in combined]

        scored_rows: list[dict[str, Any]] = []
        for baseline_row in baseline_rows:
            ticker = str(baseline_row["market_ticker"])
            split = (
                "development"
                if ticker in development_tickers
                else "embargo"
                if ticker in embargo_tickers
                else "test"
                if ticker in test_tickers
                else "excluded"
            )
            current = float(baseline_row["anchored_current_index"])
            threshold = float(
                baseline_row["raw_yes_threshold_under_provisional_rounding"]
            )
            remaining_seconds = int(baseline_row["remaining_seconds"])
            frozen_variance = float(baseline_row["forecast_variance_remaining"])
            if current <= 0 or threshold <= 0 or frozen_variance <= 0:
                raise BaselineScorecardError("invalid moneyness inputs")
            flat_variance = global_variance * remaining_seconds / WINDOW_SECONDS
            flat_threshold_z = math.log(threshold / current) / math.sqrt(flat_variance)
            moneyness_z = math.log(current / threshold) / math.sqrt(frozen_variance)
            value: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "scorecard_id": scorecard_id,
                "market_ticker": ticker,
                "utc_day": str(baseline_row["utc_day"]),
                "window_open_ts_ns": int(baseline_row["window_open_ts_ns"]),
                "window_close_ts_ns": int(baseline_row["window_close_ts_ns"]),
                "decision_ts_ns": int(baseline_row["decision_ts_ns"]),
                "remaining_seconds": remaining_seconds,
                "outcome": int(baseline_row["outcome"]),
                "split": split,
                "standardized_moneyness_z": moneyness_z,
                "remaining_time_group": _remaining_time_group(baseline_row),
                "moneyness_group": _moneyness_group(
                    {"standardized_moneyness_z": moneyness_z}
                ),
                "session": _session_group(
                    int(baseline_row["decision_ts_ns"]), config.session_timezone
                ),
                "fair_yes_constant_50": 0.5,
                "fair_yes_development_base_rate": development_base_rate,
                "current_side_yes": current >= threshold,
                "fair_yes_flat_vol_gaussian": _clip_probability(
                    _normal_survival(flat_threshold_z),
                    config.scoring_probability_floor,
                ),
                "fair_yes_frozen_tod_ewma_gaussian": float(
                    baseline_row["fair_yes_gaussian"]
                ),
                "fair_yes_frozen_empirical_cdf": float(
                    baseline_row["fair_yes_empirical"]
                ),
                "kalshi_midpoint_available": False,
                "kalshi_midpoint_unavailable_reason": None,
                "usage_classification": USAGE_CLASSIFICATION,
                "final_holdout": False,
                "trading_eligible": False,
            }
            decision_ts = int(baseline_row["decision_ts_ns"])
            times = candle_times.get(ticker, [])
            candle_index = bisect.bisect_right(times, decision_ts) - 1
            if candle_index >= 0 and times[candle_index] > int(
                baseline_row["window_open_ts_ns"]
            ):
                bid, ask = candle_values[ticker][candle_index]
                value.update(
                    {
                        "kalshi_midpoint_available": True,
                        "kalshi_midpoint_unavailable_reason": None,
                        "kalshi_candle_end_ts_ns": times[candle_index],
                        "kalshi_yes_bid_close": bid,
                        "kalshi_yes_ask_close": ask,
                        "kalshi_yes_midpoint": (bid + ask) / 2,
                    }
                )
            else:
                value["kalshi_midpoint_unavailable_reason"] = (
                    "no_current_market_candle_ended_by_decision_time"
                )
            scored_rows.append(value)

        development_rows = [row for row in scored_rows if row["split"] == "development"]
        side_rates: dict[bool, float] = {}
        side_counts: dict[str, dict[str, Any]] = {}
        for side in (False, True):
            selected = [
                row for row in development_rows if row["current_side_yes"] is side
            ]
            if not selected:
                raise BaselineScorecardError(
                    "current-side forecast requires both sides in development"
                )
            rate = sum(int(row["outcome"]) for row in selected) / len(selected)
            side_rates[side] = _clip_probability(rate, config.scoring_probability_floor)
            side_counts["currently_yes" if side else "currently_no"] = {
                "observations": len(selected),
                "windows": len({row["market_ticker"] for row in selected}),
                "raw_outcome_rate": rate,
                "forecast_probability": side_rates[side],
            }
        for row in scored_rows:
            row["fair_yes_current_side_development_rate"] = side_rates[
                bool(row["current_side_yes"])
            ]

        test_rows = [row for row in scored_rows if row["split"] == "test"]
        common_test_rows = [
            row for row in test_rows if row["kalshi_midpoint_available"] is True
        ]
        expected_common = config.test_windows * (config.expected_rows_per_window - 1)
        if len(test_rows) != config.test_windows * config.expected_rows_per_window:
            raise BaselineScorecardError("test row count violates the frozen split")
        if len(common_test_rows) != expected_common:
            raise BaselineScorecardError(
                "Kalshi common panel must omit exactly the market-open row per test window"
            )

        available_fields = {
            "constant_50": "fair_yes_constant_50",
            "development_fitted_constant_base_rate": ("fair_yes_development_base_rate"),
            "current_side_development_rate": ("fair_yes_current_side_development_rate"),
            "flat_volatility_gaussian": "fair_yes_flat_vol_gaussian",
            "frozen_tod_ewma_gaussian_p0": ("fair_yes_frozen_tod_ewma_gaussian"),
            "frozen_empirical_cdf": "fair_yes_frozen_empirical_cdf",
            "kalshi_historical_yes_midpoint": "kalshi_yes_midpoint",
        }
        common_scorecard = {
            name: _complete_metrics(
                common_test_rows, field, config.scoring_probability_floor
            )
            for name, field in available_fields.items()
        }
        all_available_scorecard = {
            name: _complete_metrics(
                common_test_rows
                if name == "kalshi_historical_yes_midpoint"
                else test_rows,
                field,
                config.scoring_probability_floor,
            )
            for name, field in available_fields.items()
        }
        constant_brier = common_scorecard["constant_50"]["overall"]["brier"]
        rankings: list[dict[str, Any]] = []
        for name, metrics in common_scorecard.items():
            brier = metrics["overall"]["brier"]
            rankings.append(
                {
                    "name": name,
                    "brier": brier,
                    "log_loss": metrics["overall"]["log_loss"],
                    "delta_brier_vs_constant_50": brier - constant_brier,
                    "brier_skill_vs_constant_50": 1 - brier / constant_brier,
                }
            )
        rankings.sort(key=lambda row: (row["brier"], row["name"]))

        comparators = [
            {
                "name": "constant_50",
                "status": "scored",
                "definition": "50% YES at every decision",
            },
            {
                "name": "development_fitted_constant_base_rate",
                "status": "scored",
                "definition": (
                    "unweighted YES outcome rate across the 55 unique development windows"
                ),
                "fitted_probability": development_base_rate,
            },
            {
                "name": "current_side_development_rate",
                "status": "scored",
                "definition": (
                    "development-fitted YES rate conditional only on whether the anchored "
                    "current index is at or above the provisional rounded YES threshold; "
                    "it ignores distance and time remaining"
                ),
                "fit": side_counts,
            },
            {
                "name": "flat_volatility_gaussian",
                "status": "scored",
                "definition": (
                    "zero-mean Gaussian using frozen global 15-minute variance scaled "
                    "linearly by remaining seconds; no time-of-day or EWMA multiplier"
                ),
                "global_variance_15m": global_variance,
            },
            {
                "name": "frozen_tod_ewma_gaussian_p0",
                "status": "scored",
                "definition": "probability stored by the immutable frozen baseline artifact",
            },
            {
                "name": "frozen_empirical_cdf",
                "status": "scored",
                "definition": "empirical-CDF probability stored by the immutable baseline artifact",
            },
            {
                "name": "kalshi_historical_yes_midpoint",
                "status": "scored_descriptive_non_executable",
                "definition": (
                    "mean of historical YES bid-close and ask-close from the latest "
                    "one-minute candle whose end is no later than the decision"
                ),
                "historical_boundary_alignment": (
                    "end_period_ts_ns <= decision_ts_ns; not certified causal because "
                    "original publication receipt time is unavailable"
                ),
                "live_latency_status": "not_measured",
                "fees_included": False,
            },
            {
                "name": "full_signal_model",
                "status": "unavailable_not_scored",
                "reason": "no signal suite has been admitted and frozen",
            },
        ]
        test_spreads = [
            float(row["kalshi_yes_ask_close"]) - float(row["kalshi_yes_bid_close"])
            for row in common_test_rows
        ]
        report = {
            "schema_version": SCHEMA_VERSION,
            "scorecard_id": scorecard_id,
            "run_id": run_id,
            "usage_classification": USAGE_CLASSIFICATION,
            "research_only": True,
            "trading_eligible": False,
            "formal_final_holdout_accessed": False,
            "baseline_study_id": config.baseline_study_id,
            "config": public_config,
            "inputs": {
                name: {"path": str(path), "sha256": input_hashes[name]}
                for name, path in inputs.items()
            },
            "split": {
                "unit": "whole_Kalshi_window",
                "development_windows": config.development_windows,
                "embargo_windows": config.embargo_windows,
                "test_windows": config.test_windows,
                "development_rows": len(development_rows),
                "test_rows": len(test_rows),
                "common_row_test_rows": len(common_test_rows),
                "common_row_exclusion": (
                    "one market-open row per test window has no current-market candle "
                    "ended by the decision timestamp"
                ),
            },
            "strata_definitions": {
                "standardized_moneyness": (
                    "log(anchored_current_index / provisional_yes_threshold) divided "
                    "by frozen TOD+EWMA remaining-return standard deviation"
                ),
                "sessions": {
                    "timezone": config.session_timezone,
                    "ct_overnight_00_to_07": "00:00 <= local decision time < 07:00",
                    "ct_us_morning_07_to_12": "07:00 <= local decision time < 12:00",
                    "ct_us_afternoon_12_to_17": "12:00 <= local decision time < 17:00",
                    "ct_evening_17_to_24": "17:00 <= local decision time < 24:00",
                },
                "event_state": {
                    "status": "unavailable_not_scored",
                    "reason": (
                        "the proxy panel has no versioned event calendar known by "
                        "decision time"
                    ),
                },
            },
            "comparators": comparators,
            "scorecards": {
                "common_row_test": common_scorecard,
                "all_available_test": all_available_scorecard,
                "common_row_brier_ranking": rankings,
            },
            "kalshi_descriptive": {
                "test_rows_with_midpoint": len(common_test_rows),
                "mean_yes_spread_dollars": sum(test_spreads) / len(test_spreads),
                "midpoint_is_executable": False,
                "fees_included": False,
                "historical_candle_publication_latency_measured": False,
            },
            "confidence_interval": {
                "block_unit": "UTC_trading_day",
                "effective_blocks": len({row["utc_day"] for row in test_rows}),
                "status": "not_estimable_fewer_than_two_days",
            },
            "decision": {
                "baseline_scorecard_completed": True,
                "eligible_for_trading_claim": False,
                "full_signal_model_scored": False,
            },
            "limitations": [
                "The 27 held-out windows are 27 outcomes from one UTC day; minute rows are not independent outcomes.",
                "Exchange-reported labels and generic anchored Pyth remain provisional research inputs.",
                "Historical Kalshi candle boundaries do not establish live publication latency.",
                "Kalshi midpoint is not executable after spread and fees.",
                "Calibration and Brier decomposition are descriptive rather than day-stable estimates.",
                "No formal final holdout was accessed.",
            ],
            "artifacts": {
                "report": str(report_path),
                "rows": str(rows_path),
                "markdown": str(markdown_path),
                "manifest": str(manifest_path),
            },
        }

        output_dir.mkdir(parents=True, exist_ok=False)
        with HashChainedJsonlWriter(rows_path, fsync_every_records=100) as writer:
            for row in sorted(
                scored_rows,
                key=lambda item: (
                    int(item["window_open_ts_ns"]),
                    int(item["decision_ts_ns"]),
                    str(item["market_ticker"]),
                ),
            ):
                writer.write(row)
        _write_json(report_path, report)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        update_run_manifest(
            root,
            run_id,
            status="completed",
            counts={
                "development_windows": config.development_windows,
                "embargo_windows": config.embargo_windows,
                "test_windows": config.test_windows,
                "test_rows": len(test_rows),
                "common_row_test_rows": len(common_test_rows),
                "formal_final_holdout_accessed": False,
                "trading_eligible": False,
            },
            output_paths=(report_path, rows_path, markdown_path),
        )
        return report
    except BaseException as error:
        update_run_manifest(root, run_id, status="failed", errors=(error,))
        raise
