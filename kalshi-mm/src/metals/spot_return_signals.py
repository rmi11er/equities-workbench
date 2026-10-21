"""Independent fast/slow spot-return signal cards over a frozen baseline fair.

Each lookback is fitted and evaluated alone as an offset-logistic adjustment to p0.
An intercept-only recalibration is the incremental attribution control.  No joint
signal model is created, and no signal can pass without queue-aware net EV evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .storage import HashChainedJsonlWriter, update_run_manifest, write_run_manifest

NSEC = 1_000_000_000
SCHEMA_VERSION = "metals-spot-return-signal-family/v1"
FAMILY_ID = "fast_and_slow_spot_returns"


class SpotReturnSignalError(ValueError):
    """Raised when a signal-family input, chronology, or fit is invalid."""


@dataclass(frozen=True)
class SpotReturnSignalConfig:
    baseline_study_id: str
    lookback_minutes: tuple[int, ...] = (1, 5, 10)
    development_fraction: float = 0.67
    embargo_windows: int = 1
    ridge_penalty: float = 10.0
    minimum_development_windows: int = 40
    minimum_test_windows: int = 20
    expected_sign: str = "positive_momentum"
    baseline_probability_field: str = "fair_yes_gaussian"
    jump_threshold_standard_deviations: float = 2.0

    def __post_init__(self) -> None:
        if len(self.baseline_study_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.baseline_study_id
        ):
            raise SpotReturnSignalError("baseline_study_id must be lowercase SHA-256")
        if (
            not self.lookback_minutes
            or len(set(self.lookback_minutes)) != len(self.lookback_minutes)
            or any(value < 1 or value > 14 for value in self.lookback_minutes)
        ):
            raise SpotReturnSignalError("lookbacks must be unique integers in [1, 14]")
        if not 0.5 < self.development_fraction < 0.8:
            raise SpotReturnSignalError("development_fraction must be in (0.5, 0.8)")
        if self.embargo_windows < 1:
            raise SpotReturnSignalError(
                "at least one complete window must be embargoed"
            )
        if not math.isfinite(self.ridge_penalty) or self.ridge_penalty <= 0:
            raise SpotReturnSignalError("ridge_penalty must be finite and positive")
        if self.minimum_development_windows < 10 or self.minimum_test_windows < 10:
            raise SpotReturnSignalError("minimum split sizes must each be at least ten")
        if self.expected_sign != "positive_momentum":
            raise SpotReturnSignalError(
                "first-family expected sign must be positive momentum"
            )
        if self.baseline_probability_field != "fair_yes_gaussian":
            raise SpotReturnSignalError("the frozen first baseline is Gaussian p0")
        if self.jump_threshold_standard_deviations <= 0:
            raise SpotReturnSignalError("jump threshold must be positive")


@dataclass(frozen=True)
class OffsetLogisticModel:
    feature_mean: float
    feature_std: float
    intercept: float
    coefficient: float
    ridge_penalty: float
    observations: int

    def predict(self, baseline_probability: float, feature: float) -> float:
        clipped = max(0.005, min(0.995, baseline_probability))
        offset = math.log(clipped / (1 - clipped))
        standardized = (feature - self.feature_mean) / self.feature_std
        return max(
            0.005,
            min(
                0.995,
                _sigmoid(offset + self.intercept + self.coefficient * standardized),
            ),
        )

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def fit_offset_logistic(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_field: str,
    baseline_field: str,
    ridge_penalty: float,
    include_feature: bool,
) -> OffsetLogisticModel:
    if len(rows) < 20:
        raise SpotReturnSignalError("at least 20 development rows are required")
    values = [float(row[feature_field]) for row in rows]
    feature_mean = sum(values) / len(values)
    feature_std = math.sqrt(
        sum((value - feature_mean) ** 2 for value in values) / len(values)
    )
    if feature_std <= 0:
        raise SpotReturnSignalError("signal feature has zero development dispersion")
    intercept = 0.0
    coefficient = 0.0
    for _ in range(100):
        gradient_intercept = 0.0
        gradient_coefficient = 0.0
        hessian_intercept = 0.0
        hessian_cross = 0.0
        hessian_coefficient = 0.0
        for row, value in zip(rows, values):
            standardized = (
                (value - feature_mean) / feature_std if include_feature else 0.0
            )
            baseline = max(0.005, min(0.995, float(row[baseline_field])))
            offset = math.log(baseline / (1 - baseline))
            probability = _sigmoid(offset + intercept + coefficient * standardized)
            weight = probability * (1 - probability)
            residual = int(row["outcome"]) - probability
            gradient_intercept += residual
            gradient_coefficient += residual * standardized
            hessian_intercept += weight
            hessian_cross += weight * standardized
            hessian_coefficient += weight * standardized * standardized
        if include_feature:
            gradient_coefficient -= ridge_penalty * coefficient
            hessian_coefficient += ridge_penalty
        else:
            hessian_coefficient = 1.0
        determinant = (
            hessian_intercept * hessian_coefficient - hessian_cross * hessian_cross
        )
        if determinant <= 1e-15:
            raise SpotReturnSignalError("offset-logistic Hessian is singular")
        delta_intercept = (
            gradient_intercept * hessian_coefficient
            - gradient_coefficient * hessian_cross
        ) / determinant
        delta_coefficient = (
            gradient_coefficient * hessian_intercept
            - gradient_intercept * hessian_cross
        ) / determinant
        intercept += delta_intercept
        coefficient += delta_coefficient
        if abs(delta_intercept) + abs(delta_coefficient) < 1e-10:
            break
    return OffsetLogisticModel(
        feature_mean=feature_mean,
        feature_std=feature_std,
        intercept=intercept,
        coefficient=coefficient if include_feature else 0.0,
        ridge_penalty=ridge_penalty if include_feature else 0.0,
        observations=len(rows),
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
        raise SpotReturnSignalError(f"input escapes storage root: {path}") from error
    if not resolved.is_file():
        raise SpotReturnSignalError(f"input is not a file: {path}")
    return resolved


def _read_hash_chained(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_previous: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SpotReturnSignalError(
                    f"invalid JSON at {path}:{line_number}"
                ) from error
            if not isinstance(row, dict):
                raise SpotReturnSignalError(f"non-object at {path}:{line_number}")
            record_hash = row.get("record_sha256")
            payload = dict(row)
            payload.pop("record_sha256", None)
            computed = hashlib.sha256(_canonical(payload).encode()).hexdigest()
            if computed != record_hash:
                raise SpotReturnSignalError(
                    f"record hash mismatch at {path}:{line_number}"
                )
            if row.get("previous_record_sha256") != expected_previous:
                raise SpotReturnSignalError(
                    f"record chain mismatch at {path}:{line_number}"
                )
            expected_previous = str(record_hash)
            rows.append(row)
    if not rows:
        raise SpotReturnSignalError(f"empty input: {path}")
    return rows


def _metrics(
    rows: Sequence[Mapping[str, Any]], probability_field: str
) -> dict[str, Any]:
    if not rows:
        return {"observations": 0, "brier": None, "log_loss": None}
    probabilities = [float(row[probability_field]) for row in rows]
    outcomes = [int(row["outcome"]) for row in rows]
    return {
        "observations": len(rows),
        "windows": len({row["market_ticker"] for row in rows}),
        "effective_utc_days": len({row["utc_day"] for row in rows}),
        "brier": sum(
            (probability - outcome) ** 2
            for probability, outcome in zip(probabilities, outcomes)
        )
        / len(rows),
        "log_loss": -sum(
            outcome * math.log(probability) + (1 - outcome) * math.log(1 - probability)
            for probability, outcome in zip(probabilities, outcomes)
        )
        / len(rows),
        "mean_probability": sum(probabilities) / len(rows),
        "outcome_rate": sum(outcomes) / len(rows),
    }


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss <= 0 or right_ss <= 0:
        return None
    return sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    ) / math.sqrt(left_ss * right_ss)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * probability)]


def _distribution_effects(
    development_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    feature_field: str,
    jump_threshold: float,
) -> dict[str, Any]:
    development_values = [float(row[feature_field]) for row in development_rows]
    lower = _quantile(development_values, 1 / 3)
    upper = _quantile(development_values, 2 / 3)
    groups = {
        "low": [row for row in test_rows if float(row[feature_field]) <= lower],
        "middle": [
            row for row in test_rows if lower < float(row[feature_field]) < upper
        ],
        "high": [row for row in test_rows if float(row[feature_field]) >= upper],
    }
    result: dict[str, Any] = {
        "development_tercile_thresholds": {"lower": lower, "upper": upper},
        "test": {},
    }
    for name, rows in groups.items():
        future_returns = [
            float(row["future_log_return_to_reported_expiration"]) for row in rows
        ]
        standardized_future = [
            value / math.sqrt(max(float(row["forecast_variance_remaining"]), 1e-18))
            for value, row in zip(future_returns, rows)
        ]
        mean_return = sum(future_returns) / len(rows) if rows else None
        variance = (
            sum((value - mean_return) ** 2 for value in future_returns) / len(rows)
            if rows
            else None
        )
        standard_deviation = math.sqrt(variance) if variance and variance > 0 else None
        skew = (
            sum(
                ((value - mean_return) / standard_deviation) ** 3
                for value in future_returns
            )
            / len(rows)
            if rows and standard_deviation
            else None
        )
        result["test"][name] = {
            "observations": len(rows),
            "windows": len({row["market_ticker"] for row in rows}),
            "mean_future_log_return": mean_return,
            "future_log_return_variance": variance,
            "future_log_return_skew": skew,
            "jump_probability": (
                sum(abs(value) >= jump_threshold for value in standardized_future)
                / len(standardized_future)
                if standardized_future
                else None
            ),
        }
    return result


def _fair_change_buckets(
    rows: Sequence[Mapping[str, Any]], signal_probability_field: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for level in (0.15, 0.25, 0.5, 0.75, 0.85):
        selected = [
            row
            for row in rows
            if abs(float(row["baseline_probability"]) - level) <= 0.025
        ]
        changes = [
            (float(row[signal_probability_field]) - float(row["baseline_probability"]))
            * 100
            for row in selected
        ]
        result[str(level)] = {
            "observations": len(selected),
            "mean_change_cents": sum(changes) / len(changes) if changes else None,
            "mean_absolute_change_cents": (
                sum(abs(value) for value in changes) / len(changes) if changes else None
            ),
        }
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


# Shared, deterministic evaluation primitives for later one-family-at-a-time cards.
canonical_signal_json = _canonical
signal_sha256_path = _sha256_path
resolve_signal_input = _resolve_input
read_signal_hash_chain = _read_hash_chained
signal_metrics = _metrics
signal_correlation = _correlation
signal_distribution_effects = _distribution_effects
signal_fair_change_buckets = _fair_change_buckets
write_signal_json = _write_json


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Fast and slow spot-return signal cards",
        "",
        "Every lookback is fitted independently against the frozen Gaussian baseline and an intercept-only recalibration control.",
        "",
        "| Signal | Coefficient | Δ Brier vs recalibration | Δ log loss vs recalibration | Probability gate | Accepted |",
        "|---|---:|---:|---:|---|---|",
    ]
    for card in report["signal_cards"]:
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {} | {} |".format(
                card["signal_id"],
                card["fitted_model"]["coefficient"],
                card["out_of_sample_probability_improvement"][
                    "delta_brier_vs_recalibrated_baseline"
                ],
                card["out_of_sample_probability_improvement"][
                    "delta_log_loss_vs_recalibrated_baseline"
                ],
                "pass" if card["probability_gate_passed"] else "fail",
                "yes" if card["accepted_into_model"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "No signal is accepted without improvement in both probability metrics and positive queue-aware net EV. Trading EV is not measurable from this one-day proxy panel.",
            "",
        ]
    )
    return "\n".join(lines)


def run_spot_return_signal_study(
    storage_root: Path,
    pyth_proxy_candles_path: Path,
    config: SpotReturnSignalConfig,
) -> dict[str, Any]:
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
        or baseline.get("decision", {}).get(
            "eligible_for_incremental_signal_development"
        )
        is not True
        or baseline.get("trading_eligible") is not False
    ):
        raise SpotReturnSignalError(
            "baseline artifact is not eligible for signal development"
        )
    baseline_rows_path = _resolve_input(
        root, Path(baseline["artifacts"]["probabilities"])
    )
    pyth_path = _resolve_input(root, pyth_proxy_candles_path)
    inputs = {
        "baseline_report": baseline_report_path,
        "baseline_probabilities": baseline_rows_path,
        "pyth_generic_proxy": pyth_path,
    }
    input_hashes = {name: _sha256_path(path) for name, path in inputs.items()}
    public_config = asdict(config)
    family_study_id = hashlib.sha256(
        _canonical(
            {
                "schema_version": SCHEMA_VERSION,
                "family_id": FAMILY_ID,
                "config": public_config,
                "inputs": input_hashes,
            }
        ).encode()
    ).hexdigest()
    run_id = f"spot-return-signals-{family_study_id[:24]}"
    output_dir = root / "derived" / "signal_cards" / f"run={run_id}"
    report_path = output_dir / "report.json"
    cards_path = output_dir / "signal_cards.json"
    rows_path = output_dir / "rows.jsonl"
    markdown_path = output_dir / "REPORT.md"
    manifest_path = root / "manifests" / f"run={run_id}" / "manifest.json"
    if report_path.exists() and manifest_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))

    write_run_manifest(
        root,
        run_id,
        {
            "component": "spot_return_signal_cards",
            "schema_version": SCHEMA_VERSION,
            "family_id": FAMILY_ID,
            "research_only": True,
            "formal_final_holdout_accessed": False,
            "joint_signal_model_created": False,
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
        pyth_rows = _read_hash_chained(pyth_path)
        if any(row.get("settlement_eligible") is not False for row in pyth_rows):
            raise SpotReturnSignalError(
                "generic Pyth input must be settlement_eligible=false"
            )
        proxy = {
            int(row["timestamp_seconds"]): float(row["close"]) for row in pyth_rows
        }
        windows = sorted(
            {
                (int(row["window_open_ts_ns"]), str(row["market_ticker"]))
                for row in baseline_rows
            }
        )
        split_index = int(len(windows) * config.development_fraction)
        if split_index < config.minimum_development_windows:
            raise SpotReturnSignalError("too few development windows")
        test_start = split_index + config.embargo_windows
        if len(windows) - test_start < config.minimum_test_windows:
            raise SpotReturnSignalError("too few test windows after embargo")
        development_tickers = {ticker for _, ticker in windows[:split_index]}
        embargo_tickers = {ticker for _, ticker in windows[split_index:test_start]}
        test_tickers = {ticker for _, ticker in windows[test_start:]}

        augmented: list[dict[str, Any]] = []
        for row in baseline_rows:
            ticker = str(row["market_ticker"])
            split = (
                "development"
                if ticker in development_tickers
                else "embargo"
                if ticker in embargo_tickers
                else "test"
                if ticker in test_tickers
                else "excluded"
            )
            value = {
                key: row[key]
                for key in (
                    "market_ticker",
                    "utc_day",
                    "window_open_ts_ns",
                    "window_close_ts_ns",
                    "decision_ts_ns",
                    "remaining_seconds",
                    "outcome",
                    "future_log_return_to_reported_expiration",
                    "forecast_variance_remaining",
                )
            }
            value.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "family_study_id": family_study_id,
                    "split": split,
                    "baseline_probability": float(
                        row[config.baseline_probability_field]
                    ),
                    "final_holdout": False,
                    "trading_eligible": False,
                }
            )
            decision_seconds = int(row["decision_ts_ns"]) // NSEC
            current = proxy.get(decision_seconds - 60)
            if current is None:
                continue
            full_variance = (
                float(row["forecast_variance_remaining"])
                * 900
                / int(row["remaining_seconds"])
            )
            complete = True
            for lookback in config.lookback_minutes:
                previous = proxy.get(decision_seconds - 60 - lookback * 60)
                if previous is None or previous <= 0:
                    complete = False
                    break
                raw_return = math.log(current / previous)
                scale = math.sqrt(max(full_variance * lookback / 15, 1e-18))
                value[f"spot_return_{lookback}m"] = raw_return
                value[f"spot_return_{lookback}m_vol_adjusted"] = raw_return / scale
            if complete:
                augmented.append(value)

        signal_models: dict[int, OffsetLogisticModel] = {}
        recalibration_models: dict[int, OffsetLogisticModel] = {}
        for lookback in config.lookback_minutes:
            field = f"spot_return_{lookback}m_vol_adjusted"
            development = [row for row in augmented if row["split"] == "development"]
            recalibration_models[lookback] = fit_offset_logistic(
                development,
                feature_field=field,
                baseline_field="baseline_probability",
                ridge_penalty=config.ridge_penalty,
                include_feature=False,
            )
            signal_models[lookback] = fit_offset_logistic(
                development,
                feature_field=field,
                baseline_field="baseline_probability",
                ridge_penalty=config.ridge_penalty,
                include_feature=True,
            )
        for row in augmented:
            for lookback in config.lookback_minutes:
                field = f"spot_return_{lookback}m_vol_adjusted"
                row[f"fair_yes_recalibrated_{lookback}m"] = recalibration_models[
                    lookback
                ].predict(float(row["baseline_probability"]), float(row[field]))
                row[f"fair_yes_signal_{lookback}m"] = signal_models[lookback].predict(
                    float(row["baseline_probability"]), float(row[field])
                )

        test = [row for row in augmented if row["split"] == "test"]
        development = [row for row in augmented if row["split"] == "development"]
        baseline_metrics = _metrics(test, "baseline_probability")
        correlations: dict[str, Any] = {}
        for left_index, left in enumerate(config.lookback_minutes):
            for right in config.lookback_minutes[left_index + 1 :]:
                correlations[f"{left}m_vs_{right}m"] = _correlation(
                    [float(row[f"spot_return_{left}m_vol_adjusted"]) for row in test],
                    [float(row[f"spot_return_{right}m_vol_adjusted"]) for row in test],
                )

        cards: list[dict[str, Any]] = []
        for lookback in config.lookback_minutes:
            signal_id = f"spot_return_{lookback}m"
            feature_field = f"{signal_id}_vol_adjusted"
            recalibrated_field = f"fair_yes_recalibrated_{lookback}m"
            signal_field = f"fair_yes_signal_{lookback}m"
            recalibrated_metrics = _metrics(test, recalibrated_field)
            signal_metrics = _metrics(test, signal_field)
            delta_brier = signal_metrics["brier"] - recalibrated_metrics["brier"]
            delta_log_loss = (
                signal_metrics["log_loss"] - recalibrated_metrics["log_loss"]
            )
            expected_sign_passed = signal_models[lookback].coefficient > 0
            probability_gate = (
                delta_brier < 0 and delta_log_loss < 0 and expected_sign_passed
            )
            cards.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "family_id": FAMILY_ID,
                    "signal_id": signal_id,
                    "formula": f"ln(Pyth_close_t / Pyth_close_t_minus_{lookback}m) divided by baseline diffusion scale",
                    "data_source_and_latency": {
                        "source": "Pyth Metal.XAG/USD generic one-minute candle close",
                        "historical_alignment": "timestamp labels the preceding minute",
                        "live_availability_assumption": "available only after the minute close; not a measured live latency claim",
                    },
                    "expected_mechanism": "short-horizon momentum shifts the conditional mean in the same direction",
                    "expected_sign": "positive",
                    "effective_horizon_and_decay": f"{lookback}-minute trailing return; expected to decay within the remaining 15-minute window",
                    "time_of_day_adjustment": "divided by the baseline time-of-day and slow-EWMA diffusion scale; zero normal mean assumed because only one proxy day is available",
                    "fitted_model": signal_models[lookback].public_dict(),
                    "recalibration_control": recalibration_models[
                        lookback
                    ].public_dict(),
                    "correlation_with_existing_signals": {
                        key: value
                        for key, value in correlations.items()
                        if f"{lookback}m" in key
                    },
                    "conditional_distribution_effects": _distribution_effects(
                        development,
                        test,
                        feature_field,
                        config.jump_threshold_standard_deviations,
                    ),
                    "out_of_sample_probability_improvement": {
                        "baseline": baseline_metrics,
                        "intercept_only_recalibration": recalibrated_metrics,
                        "signal": signal_metrics,
                        "delta_brier_vs_raw_baseline": signal_metrics["brier"]
                        - baseline_metrics["brier"],
                        "delta_log_loss_vs_raw_baseline": signal_metrics["log_loss"]
                        - baseline_metrics["log_loss"],
                        "delta_brier_vs_recalibrated_baseline": delta_brier,
                        "delta_log_loss_vs_recalibrated_baseline": delta_log_loss,
                    },
                    "fair_value_change_cents": _fair_change_buckets(test, signal_field),
                    "out_of_sample_net_ev_improvement": {
                        "status": "not_measured_no_queue_aware_execution_panel",
                        "net_cents_per_contract": None,
                    },
                    "expected_sign_passed": expected_sign_passed,
                    "probability_gate_passed": probability_gate,
                    "trading_ev_gate_passed": False,
                    "accepted_into_model": False,
                    "failure_conditions": [
                        "coefficient sign is unstable or opposite the frozen momentum hypothesis",
                        "either Brier score or log loss fails to improve versus intercept-only recalibration",
                        "effect is concentrated in one time segment or volatility state",
                        "queue-aware net EV is not positive after fees and selection",
                        "live feed availability or latency differs from the historical candle assumption",
                    ],
                }
            )

        report = {
            "schema_version": SCHEMA_VERSION,
            "family_study_id": family_study_id,
            "run_id": run_id,
            "family_id": FAMILY_ID,
            "generated_ts": datetime.now(UTC).isoformat(),
            "research_only": True,
            "formal_final_holdout_accessed": False,
            "joint_signal_model_created": False,
            "trading_eligible": False,
            "baseline_study_id": config.baseline_study_id,
            "config": public_config,
            "inputs": {
                name: {"path": str(path), "sha256": input_hashes[name]}
                for name, path in inputs.items()
            },
            "split": {
                "unit": "complete_kalshi_window",
                "total_windows": len(windows),
                "development_windows": len(development_tickers),
                "embargo_windows": len(embargo_tickers),
                "test_windows": len(test_tickers),
                "development_rows": len(development),
                "test_rows": len(test),
                "effective_test_utc_days": len({row["utc_day"] for row in test}),
            },
            "signal_correlations": correlations,
            "signal_cards": cards,
            "decision": {
                "accepted_signal_ids": [
                    card["signal_id"] for card in cards if card["accepted_into_model"]
                ],
                "family_added_to_model": False,
                "reason": "No signal can enter without both incremental probability improvement and measured positive queue-aware net EV.",
                "next_signal_family": "fast_slow_range_volatility_and_path_efficiency",
            },
            "confidence_interval": {
                "block_unit": "UTC_trading_day",
                "effective_test_blocks": len({row["utc_day"] for row in test}),
                "status": "not_estimable_fewer_than_two_days",
            },
            "artifacts": {
                "report": str(report_path),
                "signal_cards": str(cards_path),
                "rows": str(rows_path),
                "markdown": str(markdown_path),
                "manifest": str(manifest_path),
            },
        }
        output_dir.mkdir(parents=True, exist_ok=False)
        with HashChainedJsonlWriter(rows_path, fsync_every_records=100) as writer:
            for row in augmented:
                writer.write(row)
        _write_json(cards_path, {"signal_cards": cards})
        _write_json(report_path, report)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        update_run_manifest(
            root,
            run_id,
            status="completed",
            counts={
                "development_windows": len(development_tickers),
                "embargo_windows": len(embargo_tickers),
                "test_windows": len(test_tickers),
                "signal_cards": len(cards),
                "accepted_signals": 0,
                "formal_final_holdout_accessed": False,
                "joint_signal_model_created": False,
            },
            output_paths=(report_path, cards_path, rows_path, markdown_path),
        )
        return report
    except BaseException as error:
        update_run_manifest(root, run_id, status="failed", errors=(error,))
        raise
