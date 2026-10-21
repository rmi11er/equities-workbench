"""Research-only natural-target diagnostics for the exploratory metals signals.

This module deliberately does not alter the frozen terminal-probability model.
Each signal is fitted alone against the economic quantity it is supposed to
forecast.  The one-day synchronized proxy panel remains underpowered, so every
result is reported as inconclusive and cannot be used for model admission or a
trading claim.

A separate walk-forward diagnostic evaluates the slow variance components on
the longer, pre-validation sequence of provisional Kalshi boundary returns.
Every forecast in that diagnostic uses only earlier observations.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from .futures_activity_signals import select_active_trade_series
from .spot_return_signals import (
    canonical_signal_json,
    read_signal_hash_chain,
    resolve_signal_input,
    signal_sha256_path,
    write_signal_json,
)
from .storage import HashChainedJsonlWriter, update_run_manifest, write_run_manifest

NSEC = 1_000_000_000
MINUTE_SECONDS = 60
WINDOW_SECONDS = 15 * MINUTE_SECONDS
SCHEMA_VERSION = "metals-natural-target-signal-research/v2"
FAMILY_ID = "independent_signal_natural_targets"
DISPOSITION = "underpowered_inconclusive"


class NaturalTargetSignalError(ValueError):
    """Raised when a natural-target study would violate its research contract."""


@dataclass(frozen=True)
class NaturalTargetSignalConfig:
    baseline_study_id: str
    development_fraction: float = 0.67
    embargo_windows: int = 1
    minimum_development_windows: int = 40
    minimum_test_windows: int = 20
    ridge_penalty: float = 10.0
    momentum_lookback_minutes: tuple[int, ...] = (1, 5, 10)
    range_lookback_minutes: tuple[int, ...] = (3, 10)
    efficiency_lookback_minutes: tuple[int, ...] = (5, 10)
    semivariance_jump_lookback_minutes: int = 10
    cross_metal_lookback_minutes: int = 5
    intensity_lookback_minutes: int = 1
    volume_lookback_minutes: int = 5
    price_impact_lookback_minutes: int = 5
    causal_activity_reference_minutes: int = 15
    max_trade_age_seconds: int = 30
    variance_floor: float = 1e-12
    jump_threshold_standard_deviations: float = 2.0
    tail_probability: float = 0.10
    walk_forward_initial_burnin_windows: int = 384
    walk_forward_tod_prior_observations: int = 20
    walk_forward_ewma_lambda: float = 0.97
    walk_forward_ewma_ratio_floor: float = 0.25
    walk_forward_ewma_ratio_ceiling: float = 4.0
    walk_forward_bootstrap_samples: int = 2_000
    walk_forward_bootstrap_seed: int = 17_290

    def __post_init__(self) -> None:
        if len(self.baseline_study_id) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.baseline_study_id
        ):
            raise NaturalTargetSignalError(
                "baseline_study_id must be lowercase SHA-256"
            )
        if not 0.5 < self.development_fraction < 0.8:
            raise NaturalTargetSignalError(
                "development_fraction must be in (0.5, 0.8)"
            )
        if self.embargo_windows < 1:
            raise NaturalTargetSignalError(
                "at least one complete window must be embargoed"
            )
        if self.minimum_development_windows < 10 or self.minimum_test_windows < 10:
            raise NaturalTargetSignalError("minimum split sizes must be at least ten")
        lookbacks = (
            *self.momentum_lookback_minutes,
            *self.range_lookback_minutes,
            *self.efficiency_lookback_minutes,
            self.semivariance_jump_lookback_minutes,
            self.cross_metal_lookback_minutes,
            self.intensity_lookback_minutes,
            self.volume_lookback_minutes,
            self.price_impact_lookback_minutes,
        )
        if any(value < 1 or value > 14 for value in lookbacks):
            raise NaturalTargetSignalError("signal lookbacks must be in [1, 14]")
        for values in (
            self.momentum_lookback_minutes,
            self.range_lookback_minutes,
            self.efficiency_lookback_minutes,
        ):
            if not values or len(set(values)) != len(values):
                raise NaturalTargetSignalError(
                    "lookback groups must be non-empty and unique"
                )
        if self.causal_activity_reference_minutes <= max(
            self.intensity_lookback_minutes, self.volume_lookback_minutes
        ):
            raise NaturalTargetSignalError(
                "activity reference must exceed activity lookbacks"
            )
        if self.max_trade_age_seconds < 1:
            raise NaturalTargetSignalError("max trade age must be positive")
        if not math.isfinite(self.ridge_penalty) or self.ridge_penalty <= 0:
            raise NaturalTargetSignalError("ridge penalty must be finite and positive")
        if self.variance_floor <= 0:
            raise NaturalTargetSignalError("variance floor must be positive")
        if self.jump_threshold_standard_deviations <= 0:
            raise NaturalTargetSignalError("jump threshold must be positive")
        if not 0 < self.tail_probability < 0.5:
            raise NaturalTargetSignalError("tail_probability must be in (0, 0.5)")
        if self.walk_forward_initial_burnin_windows < 96:
            raise NaturalTargetSignalError(
                "walk-forward burn-in must contain at least 96 windows"
            )
        if self.walk_forward_tod_prior_observations < 1:
            raise NaturalTargetSignalError("TOD prior must be positive")
        if not 0 < self.walk_forward_ewma_lambda < 1:
            raise NaturalTargetSignalError("EWMA lambda must be in (0, 1)")
        if not (
            0
            < self.walk_forward_ewma_ratio_floor
            <= 1
            <= self.walk_forward_ewma_ratio_ceiling
        ):
            raise NaturalTargetSignalError("EWMA ratio bounds must contain one")
        if self.walk_forward_bootstrap_samples < 500:
            raise NaturalTargetSignalError(
                "walk-forward bootstrap must contain at least 500 samples"
            )
        if self.walk_forward_bootstrap_seed < 0:
            raise NaturalTargetSignalError(
                "walk-forward bootstrap seed must be nonnegative"
            )


@dataclass(frozen=True)
class RidgeLinearModel:
    feature_mean: float
    feature_std: float
    target_mean: float
    coefficient: float
    ridge_penalty: float
    observations: int

    def predict(self, feature: float) -> float:
        return self.target_mean + self.coefficient * (
            (feature - self.feature_mean) / self.feature_std
        )

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RidgeBinaryModel:
    feature_mean: float
    feature_std: float
    intercept: float
    coefficient: float
    ridge_penalty: float
    observations: int

    def predict(self, feature: float) -> float:
        standardized = (feature - self.feature_mean) / self.feature_std
        return max(
            0.005,
            min(0.995, _sigmoid(self.intercept + self.coefficient * standardized)),
        )

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def fit_ridge_linear(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_field: str,
    target_field: str,
    ridge_penalty: float,
) -> RidgeLinearModel:
    """Fit one standardized feature to one continuous target."""

    if len(rows) < 20:
        raise NaturalTargetSignalError(
            "at least 20 development rows are required for a continuous fit"
        )
    features = [float(row[feature_field]) for row in rows]
    targets = [float(row[target_field]) for row in rows]
    feature_mean = sum(features) / len(features)
    target_mean = sum(targets) / len(targets)
    feature_variance = sum(
        (feature - feature_mean) ** 2 for feature in features
    ) / len(features)
    if feature_variance <= 0:
        raise NaturalTargetSignalError("signal feature has zero development dispersion")
    feature_std = math.sqrt(feature_variance)
    standardized = [(feature - feature_mean) / feature_std for feature in features]
    numerator = sum(
        feature * (target - target_mean)
        for feature, target in zip(standardized, targets)
    )
    denominator = sum(feature * feature for feature in standardized) + ridge_penalty
    return RidgeLinearModel(
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        coefficient=numerator / denominator,
        ridge_penalty=ridge_penalty,
        observations=len(rows),
    )


def fit_ridge_binary(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_field: str,
    target_field: str,
    ridge_penalty: float,
) -> RidgeBinaryModel:
    """Fit one standardized feature to one binary natural target."""

    if len(rows) < 20:
        raise NaturalTargetSignalError(
            "at least 20 development rows are required for a binary fit"
        )
    features = [float(row[feature_field]) for row in rows]
    targets = [int(row[target_field]) for row in rows]
    if any(target not in {0, 1} for target in targets):
        raise NaturalTargetSignalError("binary target must contain only zero and one")
    feature_mean = sum(features) / len(features)
    feature_variance = sum(
        (feature - feature_mean) ** 2 for feature in features
    ) / len(features)
    if feature_variance <= 0:
        raise NaturalTargetSignalError("signal feature has zero development dispersion")
    feature_std = math.sqrt(feature_variance)
    standardized = [(feature - feature_mean) / feature_std for feature in features]
    base_rate = max(0.005, min(0.995, sum(targets) / len(targets)))
    intercept = math.log(base_rate / (1 - base_rate))
    coefficient = 0.0
    for _ in range(100):
        gradient_intercept = 0.0
        gradient_coefficient = -ridge_penalty * coefficient
        hessian_intercept = 0.0
        hessian_cross = 0.0
        hessian_coefficient = ridge_penalty
        for feature, target in zip(standardized, targets):
            probability = _sigmoid(intercept + coefficient * feature)
            weight = probability * (1 - probability)
            residual = target - probability
            gradient_intercept += residual
            gradient_coefficient += residual * feature
            hessian_intercept += weight
            hessian_cross += weight * feature
            hessian_coefficient += weight * feature * feature
        determinant = (
            hessian_intercept * hessian_coefficient
            - hessian_cross * hessian_cross
        )
        if determinant <= 1e-15:
            raise NaturalTargetSignalError("binary fit Hessian is singular")
        delta_intercept = (
            gradient_intercept * hessian_coefficient
            - gradient_coefficient * hessian_cross
        ) / determinant
        delta_coefficient = (
            gradient_coefficient * hessian_intercept
            - gradient_intercept * hessian_cross
        ) / determinant
        delta_intercept = max(-5.0, min(5.0, delta_intercept))
        delta_coefficient = max(-5.0, min(5.0, delta_coefficient))
        intercept += delta_intercept
        coefficient += delta_coefficient
        if abs(delta_intercept) + abs(delta_coefficient) < 1e-10:
            break
    return RidgeBinaryModel(
        feature_mean=feature_mean,
        feature_std=feature_std,
        intercept=intercept,
        coefficient=coefficient,
        ridge_penalty=ridge_penalty,
        observations=len(rows),
    )


def chronological_window_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    development_fraction: float,
    embargo_windows: int,
    minimum_development_windows: int,
    minimum_test_windows: int,
) -> tuple[set[str], set[str], set[str]]:
    """Split complete windows atomically in chronological order."""

    windows = sorted(
        {
            (int(row["window_open_ts_ns"]), str(row["market_ticker"]))
            for row in rows
        }
    )
    split_index = int(len(windows) * development_fraction)
    test_start = split_index + embargo_windows
    if split_index < minimum_development_windows:
        raise NaturalTargetSignalError("too few development windows")
    if len(windows) - test_start < minimum_test_windows:
        raise NaturalTargetSignalError("too few test windows after embargo")
    development = {ticker for _, ticker in windows[:split_index]}
    embargo = {ticker for _, ticker in windows[split_index:test_start]}
    test = {ticker for _, ticker in windows[test_start:]}
    if development & embargo or development & test or embargo & test:
        raise NaturalTargetSignalError("window split is not disjoint")
    return development, embargo, test


def _candle_close(candle: Mapping[str, Any] | float) -> float:
    if isinstance(candle, Mapping):
        return float(candle["close"])
    return float(candle)


def build_future_path_targets(
    candles: Mapping[int, Mapping[str, Any] | float],
    *,
    decision_seconds: int,
    window_close_seconds: int,
    forecast_variance_remaining: float,
    jump_threshold_standard_deviations: float,
) -> dict[str, float | int] | None:
    """Build ex-post targets from closes strictly after the decision boundary.

    A Pyth candle timestamp labels the preceding minute.  Thus the last causal
    close at decision time is ``decision_seconds - 60``; target closes start at
    ``decision_seconds`` and end at ``window_close_seconds - 60``.
    """

    if window_close_seconds <= decision_seconds:
        raise NaturalTargetSignalError("window close must follow decision time")
    if (window_close_seconds - decision_seconds) % MINUTE_SECONDS:
        raise NaturalTargetSignalError("future horizon must contain whole minutes")
    timestamps = list(
        range(decision_seconds, window_close_seconds, MINUTE_SECONDS)
    )
    current = candles.get(decision_seconds - MINUTE_SECONDS)
    future = [candles.get(timestamp) for timestamp in timestamps]
    if current is None or any(candle is None for candle in future):
        return None
    levels = [_candle_close(current), *[_candle_close(candle) for candle in future]]
    if any(level <= 0 for level in levels):
        raise NaturalTargetSignalError("Pyth proxy levels must be positive")
    moves = [math.log(right / left) for left, right in pairwise(levels)]
    one_minute_variance = forecast_variance_remaining / len(moves)
    one_minute_sigma = math.sqrt(max(one_minute_variance, 1e-18))
    largest = max(moves, key=abs)
    realized_variance = sum(move * move for move in moves)
    return {
        "future_signed_log_return": math.log(levels[-1] / levels[0]),
        "future_realized_variance": realized_variance,
        "future_log_realized_variance": math.log(realized_variance + 1e-12),
        "future_close_range": math.log(max(levels) / min(levels)),
        "future_max_abs_minute_return": max(abs(move) for move in moves),
        "future_max_abs_minute_return_standardized": abs(largest)
        / one_minute_sigma,
        "future_signed_largest_minute_return_standardized": largest
        / one_minute_sigma,
        "future_jump_occurrence": int(
            any(
                abs(move) / one_minute_sigma
                >= jump_threshold_standard_deviations
                for move in moves
            )
        ),
        "future_standardized_return": math.log(levels[-1] / levels[0])
        / math.sqrt(max(forecast_variance_remaining, 1e-18)),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise NaturalTargetSignalError("cannot calculate a quantile of no values")
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * probability)]


def _continuous_metrics(
    targets: Sequence[float], predictions: Sequence[float]
) -> dict[str, float | int]:
    if len(targets) != len(predictions) or not targets:
        raise NaturalTargetSignalError("continuous metrics require paired values")
    errors = [prediction - target for target, prediction in zip(targets, predictions)]
    return {
        "observations": len(targets),
        "mse": sum(error * error for error in errors) / len(errors),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "mean_target": sum(targets) / len(targets),
        "mean_prediction": sum(predictions) / len(predictions),
    }


def _binary_metrics(
    targets: Sequence[int], predictions: Sequence[float]
) -> dict[str, float | int]:
    if len(targets) != len(predictions) or not targets:
        raise NaturalTargetSignalError("binary metrics require paired values")
    clipped = [max(0.005, min(0.995, probability)) for probability in predictions]
    return {
        "observations": len(targets),
        "brier": sum(
            (probability - target) ** 2
            for target, probability in zip(targets, clipped)
        )
        / len(targets),
        "log_loss": -sum(
            target * math.log(probability)
            + (1 - target) * math.log(1 - probability)
            for target, probability in zip(targets, clipped)
        )
        / len(targets),
        "target_rate": sum(targets) / len(targets),
        "mean_probability": sum(clipped) / len(clipped),
    }


def _horizon_bucket(remaining_seconds: int) -> str:
    minutes = remaining_seconds // MINUTE_SECONDS
    if minutes <= 2:
        return "1_to_2_minutes"
    if minutes <= 6:
        return "3_to_6_minutes"
    if minutes <= 10:
        return "7_to_10_minutes"
    return "11_to_15_minutes"


def evaluate_continuous_signal(
    development: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    *,
    feature_field: str,
    target_field: str,
    ridge_penalty: float,
) -> dict[str, Any]:
    """Evaluate a single feature against a mean-only continuous control."""

    model = fit_ridge_linear(
        development,
        feature_field=feature_field,
        target_field=target_field,
        ridge_penalty=ridge_penalty,
    )
    if not test:
        raise NaturalTargetSignalError("continuous evaluation has no test rows")

    def metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        targets = [float(row[target_field]) for row in rows]
        control = [model.target_mean] * len(rows)
        fitted = [model.predict(float(row[feature_field])) for row in rows]
        control_metrics = _continuous_metrics(targets, control)
        model_metrics = _continuous_metrics(targets, fitted)
        return {
            "mean_only_control": control_metrics,
            "single_signal_model": model_metrics,
            "delta_mse_vs_mean_control": model_metrics["mse"]
            - control_metrics["mse"],
            "delta_mae_vs_mean_control": model_metrics["mae"]
            - control_metrics["mae"],
            "mse_skill_vs_mean_control": (
                1 - model_metrics["mse"] / control_metrics["mse"]
                if control_metrics["mse"] > 0
                else None
            ),
        }

    by_horizon: dict[str, Any] = {}
    for bucket in (
        "11_to_15_minutes",
        "7_to_10_minutes",
        "3_to_6_minutes",
        "1_to_2_minutes",
    ):
        selected = [
            row
            for row in test
            if _horizon_bucket(int(row["remaining_seconds"])) == bucket
        ]
        by_horizon[bucket] = metrics(selected) if selected else None
    return {
        "target_type": "continuous",
        "development_only_model": model.public_dict(),
        "incremental_test_metrics": metrics(test),
        "test_metrics_by_remaining_time": by_horizon,
    }


def evaluate_binary_signal(
    development: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    *,
    feature_field: str,
    target_field: str,
    ridge_penalty: float,
) -> dict[str, Any]:
    """Evaluate a single feature against a development base-rate control."""

    model = fit_ridge_binary(
        development,
        feature_field=feature_field,
        target_field=target_field,
        ridge_penalty=ridge_penalty,
    )
    if not test:
        raise NaturalTargetSignalError("binary evaluation has no test rows")
    base_rate = max(
        0.005,
        min(
            0.995,
            sum(int(row[target_field]) for row in development) / len(development),
        ),
    )

    def metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        targets = [int(row[target_field]) for row in rows]
        control = [base_rate] * len(rows)
        fitted = [model.predict(float(row[feature_field])) for row in rows]
        control_metrics = _binary_metrics(targets, control)
        model_metrics = _binary_metrics(targets, fitted)
        return {
            "development_base_rate_control": control_metrics,
            "single_signal_model": model_metrics,
            "delta_brier_vs_base_rate": model_metrics["brier"]
            - control_metrics["brier"],
            "delta_log_loss_vs_base_rate": model_metrics["log_loss"]
            - control_metrics["log_loss"],
        }

    by_horizon: dict[str, Any] = {}
    for bucket in (
        "11_to_15_minutes",
        "7_to_10_minutes",
        "3_to_6_minutes",
        "1_to_2_minutes",
    ):
        selected = [
            row
            for row in test
            if _horizon_bucket(int(row["remaining_seconds"])) == bucket
        ]
        by_horizon[bucket] = metrics(selected) if selected else None
    return {
        "target_type": "binary",
        "development_base_rate": base_rate,
        "development_only_model": model.public_dict(),
        "incremental_test_metrics": metrics(test),
        "test_metrics_by_remaining_time": by_horizon,
    }


def unavailable_target_registry() -> list[dict[str, Any]]:
    """Return target families that the current historical inputs cannot support."""

    return [
        {
            "research_target": "subsecond_pyth_or_external_fair_movement",
            "signal_family": "si_order_book_microstructure",
            "status": "unavailable_current_data",
            "missing_inputs": [
                "synchronized SI BBO/depth with measured receive timestamps",
                "tick-level Pyth or exact settlement-index updates with measured receive timestamps",
            ],
            "reason": "trade-only historical vendor timestamps cannot identify sub-second live lead-lag",
        },
        {
            "research_target": "fill_conditioned_external_fair_markout",
            "signal_family": "kalshi_flow_and_queue_state",
            "status": "unavailable_current_data",
            "missing_inputs": [
                "causal Kalshi order-book events",
                "our quote acknowledgements, queue estimates, fills, cancels, and fees",
                "synchronized external-fair marks",
            ],
            "reason": "settled markets and historical trade summaries cannot reconstruct fill selection",
        },
    ]


def _day_block_paired_interval(
    forecasts: Sequence[Mapping[str, Any]],
    *,
    left_field: str,
    right_field: str,
    loss: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Bootstrap a paired loss difference by UTC day.

    A negative difference favors ``left_field``. Each UTC day receives equal weight
    regardless of its number of windows.
    """

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in forecasts:
        realized = float(row["realized_squared_log_return"])
        left = float(row[left_field])
        right = float(row[right_field])
        if loss == "mse":
            difference = (left - realized) ** 2 - (right - realized) ** 2
        elif loss == "qlike":
            difference = (math.log(left) + realized / left) - (
                math.log(right) + realized / right
            )
        else:
            raise NaturalTargetSignalError(f"unsupported block loss: {loss}")
        grouped[str(row["utc_day"])].append(difference)
    daily = {
        day: sum(values) / len(values) for day, values in sorted(grouped.items())
    }
    if len(daily) < 2:
        return {
            "status": "not_estimable_fewer_than_two_days",
            "effective_utc_days": len(daily),
            "mean_daily_paired_loss_difference": None,
            "lower_95": None,
            "upper_95": None,
        }
    values = list(daily.values())
    generator = random.Random(bootstrap_seed)
    bootstrap = sorted(
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(bootstrap_samples)
    )
    lower = bootstrap[int((bootstrap_samples - 1) * 0.025)]
    upper = bootstrap[int((bootstrap_samples - 1) * 0.975)]
    mean_daily = sum(values) / len(values)
    return {
        "status": "descriptive_day_block_bootstrap",
        "block_unit": "UTC_trading_day",
        "effective_utc_days": len(values),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "left_forecast": left_field,
        "right_forecast": right_field,
        "loss": loss,
        "sign_convention": "negative_favors_left",
        "mean_daily_paired_loss_difference": mean_daily,
        "lower_95": lower,
        "upper_95": upper,
        "interval_excludes_zero": lower > 0 or upper < 0,
        "daily_paired_loss_difference": daily,
    }


def walk_forward_variance_diagnostic(
    observations: Sequence[Mapping[str, Any]],
    *,
    initial_burnin_windows: int,
    tod_prior_observations: int,
    ewma_lambda: float,
    ewma_ratio_floor: float,
    ewma_ratio_ceiling: float,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 17_290,
) -> dict[str, Any]:
    """Compare flat, TOD, and TOD+EWMA forecasts of the next squared return."""

    ordered = sorted(
        observations,
        key=lambda row: (int(row["open_ts_ns"]), str(row["market_ticker"])),
    )
    if len(ordered) <= initial_burnin_windows:
        raise NaturalTargetSignalError(
            "walk-forward observations must exceed the initial burn-in"
        )
    if initial_burnin_windows < 96:
        raise NaturalTargetSignalError("walk-forward burn-in must be at least 96")
    global_sum = 0.0
    bucket_sum: dict[int, float] = defaultdict(float)
    bucket_count: dict[int, int] = defaultdict(int)
    ewma_state: float | None = None
    forecasts: list[dict[str, Any]] = []
    for index, row in enumerate(ordered):
        global_count = index
        log_return = float(row["log_return"])
        if not math.isfinite(log_return):
            raise NaturalTargetSignalError("walk-forward return must be finite")
        realized = log_return * log_return
        bucket = int(row["bucket"])
        if not 0 <= bucket < 96:
            raise NaturalTargetSignalError("time-of-day bucket must be in [0, 95]")
        if global_count:
            flat = max(global_sum / global_count, 1e-18)
            local_count = bucket_count[bucket]
            tod = max(
                (
                    bucket_sum[bucket]
                    + tod_prior_observations * flat
                )
                / (local_count + tod_prior_observations),
                1e-18,
            )
            state = ewma_state if ewma_state is not None else flat
            ratio = max(
                ewma_ratio_floor,
                min(ewma_ratio_ceiling, state / flat),
            )
            tod_ewma = max(tod * ratio, 1e-18)
            if index >= initial_burnin_windows:
                forecasts.append(
                    {
                        "market_ticker": str(row["market_ticker"]),
                        "open_ts_ns": int(row["open_ts_ns"]),
                        "utc_day": str(row["utc_day"]),
                        "bucket": bucket,
                        "realized_squared_log_return": realized,
                        "flat_variance_forecast": flat,
                        "time_of_day_variance_forecast": tod,
                        "time_of_day_slow_ewma_variance_forecast": tod_ewma,
                        "history_observations_available": global_count,
                        "final_holdout": False,
                        "trading_eligible": False,
                    }
                )
            seasonal_multiplier = tod / flat
            deseasonalized_realized = realized / max(
                seasonal_multiplier, 1e-12
            )
            ewma_state = (
                ewma_lambda * state
                + (1 - ewma_lambda) * deseasonalized_realized
            )
        else:
            ewma_state = realized
        global_sum += realized
        bucket_sum[bucket] += realized
        bucket_count[bucket] += 1
    if not forecasts:
        raise NaturalTargetSignalError("walk-forward diagnostic produced no forecasts")

    def score(field: str) -> dict[str, Any]:
        targets = [float(row["realized_squared_log_return"]) for row in forecasts]
        values = [float(row[field]) for row in forecasts]
        return {
            "observations": len(forecasts),
            "mse": sum(
                (forecast - realized) ** 2
                for realized, forecast in zip(targets, values)
            )
            / len(forecasts),
            "qlike": sum(
                math.log(forecast) + realized / forecast
                for realized, forecast in zip(targets, values)
            )
            / len(forecasts),
            "mean_forecast": sum(values) / len(values),
            "mean_realized_squared_return": sum(targets) / len(targets),
        }

    metrics = {
        "flat_variance": score("flat_variance_forecast"),
        "time_of_day_variance": score("time_of_day_variance_forecast"),
        "time_of_day_plus_slow_ewma": score(
            "time_of_day_slow_ewma_variance_forecast"
        ),
    }
    flat = metrics["flat_variance"]
    for value in metrics.values():
        value["delta_mse_vs_flat"] = value["mse"] - flat["mse"]
        value["delta_qlike_vs_flat"] = value["qlike"] - flat["qlike"]
    comparisons = {
        "time_of_day_vs_flat": (
            "time_of_day_variance_forecast",
            "flat_variance_forecast",
        ),
        "time_of_day_plus_slow_ewma_vs_flat": (
            "time_of_day_slow_ewma_variance_forecast",
            "flat_variance_forecast",
        ),
        "time_of_day_plus_slow_ewma_vs_time_of_day": (
            "time_of_day_slow_ewma_variance_forecast",
            "time_of_day_variance_forecast",
        ),
    }
    day_block_intervals = {
        name: {
            loss: _day_block_paired_interval(
                forecasts,
                left_field=left,
                right_field=right,
                loss=loss,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            for loss in ("mse", "qlike")
        }
        for name, (left, right) in comparisons.items()
    }
    tod = metrics["time_of_day_variance"]
    ewma = metrics["time_of_day_plus_slow_ewma"]
    return {
        "design": {
            "target": "next_observed_15m_squared_log_return",
            "forecast_timing": "strictly_before_each_scored_window",
            "estimation": "expanding causal moments and recursive slow EWMA",
            "initial_burnin_windows": initial_burnin_windows,
            "tod_prior_observations": tod_prior_observations,
            "ewma_lambda": ewma_lambda,
            "day_block_bootstrap_samples": bootstrap_samples,
            "day_block_bootstrap_seed": bootstrap_seed,
            "qlike_formula": "log(forecast_variance) + realized_squared_return / forecast_variance",
            "formal_final_holdout_accessed": False,
        },
        "cohort": {
            "input_windows": len(ordered),
            "scored_windows": len(forecasts),
            "effective_utc_days": len({row["utc_day"] for row in forecasts}),
        },
        "metrics": metrics,
        "day_block_confidence_intervals": day_block_intervals,
        "interpretation": {
            "time_of_day_vs_flat": (
                "provisional_support_on_both_losses"
                if tod["delta_mse_vs_flat"] < 0 and tod["delta_qlike_vs_flat"] < 0
                else "mixed_or_no_support"
            ),
            "slow_ewma_increment_vs_time_of_day": (
                "directionally_consistent_requires_confirmation"
                if (
                    ewma["mse"] <= tod["mse"]
                    and ewma["qlike"] <= tod["qlike"]
                )
                else "mixed_or_no_incremental_support"
            ),
            "scope": (
                "provisional opening-to-expiration labels; not a formal model "
                "admission or exact-settlement result"
            ),
        },
        "disposition": "provisional_component_evidence",
        "forecast_rows": forecasts,
    }


def _training_boundary_observations(
    labels: Sequence[Mapping[str, Any]],
    *,
    metal: str,
    training_end_ts_ns: int,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for row in labels:
        if str(row.get("metal", "")).lower() != metal.lower():
            continue
        open_ns = int(row.get("open_ts_ns") or -1)
        close_ns = int(row.get("close_ts_ns") or -1)
        if close_ns > training_end_ts_ns:
            continue
        if close_ns - open_ns != WINDOW_SECONDS * NSEC:
            continue
        try:
            target = float(row["target_price"])
            expiration = float(row["expiration_value"])
        except (KeyError, TypeError, ValueError):
            continue
        result = str(row.get("result", "")).lower()
        if target <= 0 or expiration <= 0 or result not in {"yes", "no"}:
            continue
        expected = "yes" if expiration >= target else "no"
        if expected != result:
            continue
        observations.append(
            {
                "market_ticker": str(row["market_ticker"]),
                "open_ts_ns": open_ns,
                "utc_day": datetime.fromtimestamp(open_ns / NSEC, UTC)
                .date()
                .isoformat(),
                "bucket": (open_ns // NSEC % 86_400) // WINDOW_SECONDS,
                "log_return": math.log(expiration / target),
            }
        )
    observations.sort(
        key=lambda value: (value["open_ts_ns"], value["market_ticker"])
    )
    return observations


def _split_label(
    ticker: str,
    development: set[str],
    embargo: set[str],
    test: set[str],
) -> str:
    if ticker in development:
        return "development"
    if ticker in embargo:
        return "embargo"
    if ticker in test:
        return "test"
    return "excluded"


def _trailing_levels(
    candles: Mapping[int, Mapping[str, Any]], current_seconds: int, lookback: int
) -> list[float] | None:
    values = [
        candles.get(current_seconds - offset * MINUTE_SECONDS)
        for offset in range(lookback, -1, -1)
    ]
    if any(value is None for value in values):
        return None
    levels = [_candle_close(value) for value in values]
    if any(level <= 0 for level in levels):
        raise NaturalTargetSignalError("Pyth proxy levels must be positive")
    return levels


def _signal_specs(config: NaturalTargetSignalConfig) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for lookback in config.momentum_lookback_minutes:
        specs.append(
            {
                "signal_id": f"spot_return_{lookback}m",
                "feature_field": f"spot_return_{lookback}m",
                "target_field": "future_signed_log_return",
                "target_type": "continuous",
                "family": "momentum",
                "formula": f"baseline-vol-adjusted XAG log return over {lookback} minutes",
                "mechanism": "short-horizon continuation in the same direction",
            }
        )
    for lookback in config.efficiency_lookback_minutes:
        specs.append(
            {
                "signal_id": f"path_efficiency_{lookback}m",
                "feature_field": f"path_efficiency_{lookback}m",
                "target_field": "future_signed_log_return",
                "target_type": "continuous",
                "family": "path_efficiency",
                "formula": f"standardized {lookback}m return times net/gross path efficiency",
                "mechanism": "efficient paths contain more directional information than choppy paths",
            }
        )
    for lookback in config.range_lookback_minutes:
        specs.append(
            {
                "signal_id": f"range_volatility_{lookback}m",
                "feature_field": f"range_volatility_{lookback}m",
                "target_field": "future_log_realized_variance",
                "target_type": "continuous",
                "family": "range_volatility",
                "formula": f"log Parkinson range variance over {lookback}m relative to baseline variance",
                "mechanism": "unusually wide recent range predicts elevated subsequent variance",
            }
        )
    specs.append(
        {
            "signal_id": "slow_ewma_state",
            "feature_field": "slow_ewma_state",
            "target_field": "future_log_realized_variance",
            "target_type": "continuous",
            "family": "slow_volatility",
            "formula": "log slow-EWMA variance ratio from the frozen baseline state",
            "mechanism": "persistent volatility state predicts subsequent realized variance",
        }
    )
    lookback = config.semivariance_jump_lookback_minutes
    specs.extend(
        [
            {
                "signal_id": f"semivariance_imbalance_{lookback}m_up_tail",
                "feature_field": f"semivariance_imbalance_{lookback}m",
                "target_field": "future_up_tail",
                "target_type": "binary",
                "family": "semivariance",
                "formula": "positive-minus-negative trailing semivariance divided by total semivariance",
                "mechanism": "upside-dominant paths raise the subsequent upper-tail probability",
            },
            {
                "signal_id": f"semivariance_imbalance_{lookback}m_down_tail",
                "feature_field": f"negative_semivariance_imbalance_{lookback}m",
                "target_field": "future_down_tail",
                "target_type": "binary",
                "family": "semivariance",
                "formula": "negative of positive-minus-negative trailing semivariance divided by total semivariance",
                "mechanism": "downside-dominant paths raise the subsequent lower-tail probability",
            },
            {
                "signal_id": f"jump_magnitude_{lookback}m_occurrence",
                "feature_field": f"jump_magnitude_{lookback}m",
                "target_field": "future_jump_occurrence",
                "target_type": "binary",
                "family": "jump",
                "formula": "log one plus largest trailing absolute one-minute return in baseline-sigma units",
                "mechanism": "recent jumps cluster with subsequent jump occurrence",
            },
            {
                "signal_id": f"jump_magnitude_{lookback}m_tail_magnitude",
                "feature_field": f"jump_magnitude_{lookback}m",
                "target_field": "future_max_abs_minute_return_standardized",
                "target_type": "continuous",
                "family": "jump",
                "formula": "log one plus largest trailing absolute one-minute return in baseline-sigma units",
                "mechanism": "recent jump magnitude predicts subsequent tail magnitude",
            },
            {
                "signal_id": f"jump_direction_{lookback}m",
                "feature_field": f"jump_direction_{lookback}m",
                "target_field": "future_signed_largest_minute_return_standardized",
                "target_type": "continuous",
                "family": "jump",
                "formula": "signed largest trailing one-minute return in baseline-sigma units",
                "mechanism": "jump direction predicts the direction of the subsequent largest move",
            },
        ]
    )
    return specs


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Natural-target signal research",
        "",
        "Every feature is fitted independently to its economic target. Results are exploratory and inconclusive because the synchronized test panel contains one effective UTC day.",
        "",
        "| Signal | Natural target | Type | Test observations | Disposition |",
        "|---|---|---|---:|---|",
    ]
    for card in report["signal_cards"]:
        metrics = card["evaluation"]["incremental_test_metrics"]
        if card["target_type"] == "continuous":
            observations = metrics["single_signal_model"]["observations"]
        else:
            observations = metrics["single_signal_model"]["observations"]
        lines.append(
            f"| {card['signal_id']} | {card['natural_target']} | "
            f"{card['target_type']} | {observations} | {card['disposition']} |"
        )
    walk = report["walk_forward_slow_variance_diagnostic"]
    lines.extend(
        [
            "",
            "## Separate walk-forward slow-variance diagnostic",
            "",
            f"Scored {walk['cohort']['scored_windows']} windows after a causal burn-in of {walk['design']['initial_burnin_windows']} windows. This uses provisional boundary returns and does not touch the formal holdout.",
            "",
            "The model-admission and trading-EV questions remain outside this artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def run_natural_target_signal_study(
    storage_root: Path,
    silver_pyth_proxy_candles_path: Path,
    config: NaturalTargetSignalConfig,
    *,
    databento_trades_path: Path | None = None,
    gold_pyth_proxy_candles_path: Path | None = None,
) -> dict[str, Any]:
    """Run independent one-day target tests and the longer variance diagnostic."""

    root = storage_root.resolve()
    baseline_dir = (
        root
        / "derived"
        / "baseline_fair"
        / f"run=baseline-fair-{config.baseline_study_id[:24]}"
    )
    baseline_report_path = resolve_signal_input(root, baseline_dir / "report.json")
    baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    if (
        baseline.get("study_id") != config.baseline_study_id
        or baseline.get("formal_final_holdout_accessed") is not False
        or baseline.get("trading_eligible") is not False
        or baseline.get("decision", {}).get(
            "eligible_for_incremental_signal_development"
        )
        is not True
    ):
        raise NaturalTargetSignalError(
            "baseline is not eligible for exploratory natural-target research"
        )
    baseline_rows_path = resolve_signal_input(
        root, Path(baseline["artifacts"]["probabilities"])
    )
    labels_path = resolve_signal_input(
        root, Path(baseline["inputs"]["provisional_labels"]["path"])
    )
    silver_path = resolve_signal_input(root, silver_pyth_proxy_candles_path)
    inputs: dict[str, Path] = {
        "baseline_report": baseline_report_path,
        "baseline_probabilities": baseline_rows_path,
        "provisional_boundary_labels": labels_path,
        "silver_pyth_generic_proxy": silver_path,
    }
    if databento_trades_path is not None:
        inputs["databento_trades"] = resolve_signal_input(
            root, databento_trades_path
        )
    if gold_pyth_proxy_candles_path is not None:
        inputs["gold_pyth_generic_proxy"] = resolve_signal_input(
            root, gold_pyth_proxy_candles_path
        )
    input_hashes = {name: signal_sha256_path(path) for name, path in inputs.items()}
    public_config = asdict(config)
    study_id = hashlib.sha256(
        canonical_signal_json(
            {
                "schema_version": SCHEMA_VERSION,
                "family_id": FAMILY_ID,
                "config": public_config,
                "inputs": input_hashes,
            }
        ).encode()
    ).hexdigest()
    run_id = f"natural-targets-{study_id[:24]}"
    output_dir = root / "derived" / "natural_target_research" / f"run={run_id}"
    report_path = output_dir / "report.json"
    cards_path = output_dir / "signal_cards.json"
    rows_path = output_dir / "one_day_rows.jsonl"
    variance_rows_path = output_dir / "walk_forward_variance_rows.jsonl"
    markdown_path = output_dir / "REPORT.md"
    manifest_path = root / "manifests" / f"run={run_id}" / "manifest.json"
    if report_path.exists() and manifest_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    write_run_manifest(
        root,
        run_id,
        {
            "component": "natural_target_signal_research",
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
        baseline_rows = read_signal_hash_chain(baseline_rows_path)
        silver_rows = read_signal_hash_chain(silver_path)
        if any(row.get("settlement_eligible") is not False for row in silver_rows):
            raise NaturalTargetSignalError(
                "generic silver Pyth input must be settlement_eligible=false"
            )
        silver = {int(row["timestamp_seconds"]): row for row in silver_rows}
        gold: dict[int, Mapping[str, Any]] | None = None
        if gold_pyth_proxy_candles_path is not None:
            gold_rows = read_signal_hash_chain(inputs["gold_pyth_generic_proxy"])
            if any(row.get("settlement_eligible") is not False for row in gold_rows):
                raise NaturalTargetSignalError(
                    "generic gold Pyth input must be settlement_eligible=false"
                )
            gold = {int(row["timestamp_seconds"]): row for row in gold_rows}
        series = None
        selected_contract = None
        if databento_trades_path is not None:
            trades = read_signal_hash_chain(inputs["databento_trades"])
            series, selected_contract = select_active_trade_series(trades, "SI")
        development_tickers, embargo_tickers, test_tickers = (
            chronological_window_split(
                baseline_rows,
                development_fraction=config.development_fraction,
                embargo_windows=config.embargo_windows,
                minimum_development_windows=config.minimum_development_windows,
                minimum_test_windows=config.minimum_test_windows,
            )
        )

        maximum_lookback = max(
            *config.momentum_lookback_minutes,
            *config.range_lookback_minutes,
            *config.efficiency_lookback_minutes,
            config.semivariance_jump_lookback_minutes,
            config.cross_metal_lookback_minutes,
        )
        augmented: list[dict[str, Any]] = []
        for source in baseline_rows:
            decision_ns = int(source["decision_ts_ns"])
            decision_seconds = decision_ns // NSEC
            close_seconds = int(source["window_close_ts_ns"]) // NSEC
            targets = build_future_path_targets(
                silver,
                decision_seconds=decision_seconds,
                window_close_seconds=close_seconds,
                forecast_variance_remaining=float(
                    source["forecast_variance_remaining"]
                ),
                jump_threshold_standard_deviations=(
                    config.jump_threshold_standard_deviations
                ),
            )
            if targets is None:
                continue
            current_seconds = decision_seconds - MINUTE_SECONDS
            history = _trailing_levels(silver, current_seconds, maximum_lookback)
            if history is None:
                continue
            ticker = str(source["market_ticker"])
            row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "study_id": study_id,
                "market_ticker": ticker,
                "utc_day": str(source["utc_day"]),
                "window_open_ts_ns": int(source["window_open_ts_ns"]),
                "window_close_ts_ns": int(source["window_close_ts_ns"]),
                "decision_ts_ns": decision_ns,
                "remaining_seconds": int(source["remaining_seconds"]),
                "split": _split_label(
                    ticker,
                    development_tickers,
                    embargo_tickers,
                    test_tickers,
                ),
                "forecast_variance_remaining": float(
                    source["forecast_variance_remaining"]
                ),
                "slow_ewma_state": math.log(float(source["slow_ewma_ratio"])),
                "final_holdout": False,
                "trading_eligible": False,
                **targets,
            }
            full_variance = (
                float(source["forecast_variance_remaining"])
                * WINDOW_SECONDS
                / int(source["remaining_seconds"])
            )
            for lookback in {
                *config.momentum_lookback_minutes,
                *config.efficiency_lookback_minutes,
            }:
                levels = _trailing_levels(silver, current_seconds, lookback)
                if levels is None:
                    continue
                moves = [math.log(right / left) for left, right in pairwise(levels)]
                net = math.log(levels[-1] / levels[0])
                scale = math.sqrt(max(full_variance * lookback / 15, 1e-18))
                if lookback in config.momentum_lookback_minutes:
                    row[f"spot_return_{lookback}m"] = net / scale
                if lookback in config.efficiency_lookback_minutes:
                    gross = sum(abs(move) for move in moves)
                    row[f"path_efficiency_{lookback}m"] = (
                        net / scale * abs(net) / gross if gross > 0 else 0.0
                    )
            for lookback in config.range_lookback_minutes:
                timestamps = [
                    current_seconds - offset * MINUTE_SECONDS
                    for offset in range(lookback - 1, -1, -1)
                ]
                selected = [silver.get(timestamp) for timestamp in timestamps]
                if any(candle is None for candle in selected):
                    continue
                high = max(float(candle["high"]) for candle in selected if candle)
                low = min(float(candle["low"]) for candle in selected if candle)
                parkinson = math.log(high / low) ** 2 / (4 * math.log(2))
                expected = full_variance * lookback / 15
                row[f"range_volatility_{lookback}m"] = math.log(
                    (parkinson + config.variance_floor)
                    / (expected + config.variance_floor)
                )
            lookback = config.semivariance_jump_lookback_minutes
            levels = _trailing_levels(silver, current_seconds, lookback)
            if levels is not None:
                moves = [math.log(right / left) for left, right in pairwise(levels)]
                positive = sum(move * move for move in moves if move > 0)
                negative = sum(move * move for move in moves if move < 0)
                total = positive + negative
                largest = max(moves, key=abs)
                sigma = math.sqrt(max(full_variance / 15, 1e-18))
                imbalance = (positive - negative) / (
                    total + config.variance_floor
                )
                row[f"semivariance_imbalance_{lookback}m"] = imbalance
                row[f"negative_semivariance_imbalance_{lookback}m"] = -imbalance
                row[f"jump_magnitude_{lookback}m"] = math.log1p(
                    abs(largest) / sigma
                )
                row[f"jump_direction_{lookback}m"] = largest / sigma
            if series is not None:
                for kind, lookback in (
                    ("trade_intensity", config.intensity_lookback_minutes),
                    ("volume_state", config.volume_lookback_minutes),
                ):
                    trailing_start = decision_ns - lookback * MINUTE_SECONDS * NSEC
                    count, volume = series.interval(trailing_start, decision_ns)
                    reference_start = (
                        decision_ns
                        - config.causal_activity_reference_minutes
                        * MINUTE_SECONDS
                        * NSEC
                    )
                    reference_count, reference_volume = series.interval(
                        reference_start, trailing_start
                    )
                    reference_minutes = (
                        config.causal_activity_reference_minutes - lookback
                    )
                    observed = count if kind == "trade_intensity" else volume
                    reference = (
                        reference_count
                        if kind == "trade_intensity"
                        else reference_volume
                    )
                    expected = reference / reference_minutes * lookback
                    row[f"{kind}_{lookback}m"] = math.log(
                        (observed + 1) / (expected + 1)
                    )
                lookback = config.price_impact_lookback_minutes
                maximum_age = config.max_trade_age_seconds * NSEC
                current_price = series.asof(decision_ns, maximum_age)
                previous_price = series.asof(
                    decision_ns - lookback * MINUTE_SECONDS * NSEC,
                    maximum_age,
                )
                _, volume = series.interval(
                    decision_ns - lookback * MINUTE_SECONDS * NSEC, decision_ns
                )
                if current_price is not None and previous_price is not None:
                    row[f"price_impact_{lookback}m"] = (
                        math.log(current_price / previous_price)
                        / math.sqrt(max(full_variance * lookback / 15, 1e-18))
                        / math.sqrt(1 + volume)
                    )
            if gold is not None:
                gold_targets = build_future_path_targets(
                    gold,
                    decision_seconds=decision_seconds,
                    window_close_seconds=close_seconds,
                    forecast_variance_remaining=float(
                        source["forecast_variance_remaining"]
                    ),
                    jump_threshold_standard_deviations=(
                        config.jump_threshold_standard_deviations
                    ),
                )
                gold_levels = _trailing_levels(
                    gold, current_seconds, config.cross_metal_lookback_minutes
                )
                silver_levels = _trailing_levels(
                    silver, current_seconds, config.cross_metal_lookback_minutes
                )
                if (
                    gold_targets is not None
                    and gold_levels is not None
                    and silver_levels is not None
                ):
                    row["future_gold_signed_log_return"] = float(
                        gold_targets["future_signed_log_return"]
                    )
                    row["trailing_gold_return"] = math.log(
                        gold_levels[-1] / gold_levels[0]
                    )
                    row["trailing_silver_return"] = math.log(
                        silver_levels[-1] / silver_levels[0]
                    )
            augmented.append(row)

        development = [row for row in augmented if row["split"] == "development"]
        test = [row for row in augmented if row["split"] == "test"]
        if len(development) < 20 or len(test) < 1:
            raise NaturalTargetSignalError(
                "natural-target panel has insufficient complete rows"
            )
        lower_tail = _quantile(
            [float(row["future_standardized_return"]) for row in development],
            config.tail_probability,
        )
        upper_tail = _quantile(
            [float(row["future_standardized_return"]) for row in development],
            1 - config.tail_probability,
        )
        for row in augmented:
            standardized = float(row["future_standardized_return"])
            row["future_up_tail"] = int(standardized >= upper_tail)
            row["future_down_tail"] = int(standardized <= lower_tail)

        specs = _signal_specs(config)
        if series is not None:
            specs.extend(
                [
                    {
                        "signal_id": f"trade_intensity_{config.intensity_lookback_minutes}m",
                        "feature_field": f"trade_intensity_{config.intensity_lookback_minutes}m",
                        "target_field": "future_log_realized_variance",
                        "target_type": "continuous",
                        "family": "futures_activity",
                        "formula": "trailing trade intensity relative to the immediately preceding causal activity window",
                        "mechanism": "high information-arrival intensity predicts subsequent realized variance",
                    },
                    {
                        "signal_id": f"volume_state_{config.volume_lookback_minutes}m",
                        "feature_field": f"volume_state_{config.volume_lookback_minutes}m",
                        "target_field": "future_log_realized_variance",
                        "target_type": "continuous",
                        "family": "futures_activity",
                        "formula": "trailing volume relative to the immediately preceding causal activity window",
                        "mechanism": "high volume state predicts subsequent realized variance",
                    },
                    {
                        "signal_id": f"price_impact_{config.price_impact_lookback_minutes}m",
                        "feature_field": f"price_impact_{config.price_impact_lookback_minutes}m",
                        "target_field": "future_signed_log_return",
                        "target_type": "continuous",
                        "family": "futures_activity",
                        "formula": "baseline-standardized futures return divided by square root of trailing volume",
                        "mechanism": "signed price impact contains directional information for subsequent XAG movement",
                    },
                ]
            )

        cross_beta = None
        if gold is not None:
            beta_rows = [
                row
                for row in development
                if row.get("future_gold_signed_log_return") is not None
            ]
            if len(beta_rows) >= 20:
                silver_mean = sum(
                    float(row["future_signed_log_return"]) for row in beta_rows
                ) / len(beta_rows)
                gold_mean = sum(
                    float(row["future_gold_signed_log_return"]) for row in beta_rows
                ) / len(beta_rows)
                gold_variance = sum(
                    (float(row["future_gold_signed_log_return"]) - gold_mean) ** 2
                    for row in beta_rows
                )
                if gold_variance > 0:
                    cross_beta = sum(
                        (float(row["future_signed_log_return"]) - silver_mean)
                        * (
                            float(row["future_gold_signed_log_return"])
                            - gold_mean
                        )
                        for row in beta_rows
                    ) / gold_variance
                    for row in augmented:
                        if row.get("future_gold_signed_log_return") is None:
                            continue
                        cross_lookback = config.cross_metal_lookback_minutes
                        row["future_silver_residual_return"] = float(
                            row["future_signed_log_return"]
                        ) - cross_beta * float(row["future_gold_signed_log_return"])
                        row[f"gold_confirmation_return_{cross_lookback}m"] = float(
                            row["trailing_gold_return"]
                        )
                        row[f"silver_residual_return_{cross_lookback}m"] = float(
                            row["trailing_silver_return"]
                        ) - cross_beta * float(row["trailing_gold_return"])
                    cross_lookback = config.cross_metal_lookback_minutes
                    specs.extend(
                        [
                            {
                                "signal_id": f"gold_confirmation_return_{cross_lookback}m",
                                "feature_field": f"gold_confirmation_return_{cross_lookback}m",
                                "target_field": "future_silver_residual_return",
                                "target_type": "continuous",
                                "family": "cross_metal",
                                "formula": f"trailing {cross_lookback}m XAU log return",
                                "mechanism": "gold may lead the silver component not captured contemporaneously",
                            },
                            {
                                "signal_id": f"silver_residual_return_{cross_lookback}m",
                                "feature_field": f"silver_residual_return_{cross_lookback}m",
                                "target_field": "future_silver_residual_return",
                                "target_type": "continuous",
                                "family": "cross_metal",
                                "formula": f"trailing {cross_lookback}m silver return minus development-only beta times gold return",
                                "mechanism": "silver-specific residual information may persist",
                            },
                        ]
                    )

        cards: list[dict[str, Any]] = []
        for spec in specs:
            feature = spec["feature_field"]
            target = spec["target_field"]
            development_rows = [
                row
                for row in augmented
                if row["split"] == "development"
                and row.get(feature) is not None
                and row.get(target) is not None
            ]
            test_rows = [
                row
                for row in augmented
                if row["split"] == "test"
                and row.get(feature) is not None
                and row.get(target) is not None
            ]
            if len(development_rows) < 20 or not test_rows:
                continue
            try:
                if spec["target_type"] == "binary":
                    evaluation = evaluate_binary_signal(
                        development_rows,
                        test_rows,
                        feature_field=feature,
                        target_field=target,
                        ridge_penalty=config.ridge_penalty,
                    )
                    control = "development_base_rate"
                else:
                    evaluation = evaluate_continuous_signal(
                        development_rows,
                        test_rows,
                        feature_field=feature,
                        target_field=target,
                        ridge_penalty=config.ridge_penalty,
                    )
                    control = "development_mean"
            except NaturalTargetSignalError as error:
                if "zero development dispersion" in str(error):
                    continue
                raise
            cards.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_id": study_id,
                    "signal_id": spec["signal_id"],
                    "signal_family": spec["family"],
                    "feature_formula": spec["formula"],
                    "expected_mechanism": spec["mechanism"],
                    "natural_target": target,
                    "target_type": spec["target_type"],
                    "incremental_control": control,
                    "fit_scope": "development_windows_only",
                    "test_scope": "chronological_complete_windows_after_one_window_embargo",
                    "terminal_probability_interactions": {
                        "status": "not_fitted_on_underpowered_one_day_panel",
                        "required_future_terms": [
                            "continuous_standardized_moneyness",
                            "continuous_time_remaining",
                            "signal_by_moneyness",
                            "signal_by_time_remaining",
                        ],
                    },
                    "evaluation": evaluation,
                    "disposition": DISPOSITION,
                    "model_admission_evaluated": False,
                    "trading_ev_evaluated": False,
                    "trading_eligible": False,
                }
            )

        labels = read_signal_hash_chain(labels_path)
        boundary_observations = _training_boundary_observations(
            labels,
            metal=str(baseline["config"]["metal"]),
            training_end_ts_ns=int(baseline["config"]["training_end_ts_ns"]),
        )
        walk_forward = walk_forward_variance_diagnostic(
            boundary_observations,
            initial_burnin_windows=config.walk_forward_initial_burnin_windows,
            tod_prior_observations=config.walk_forward_tod_prior_observations,
            ewma_lambda=config.walk_forward_ewma_lambda,
            ewma_ratio_floor=config.walk_forward_ewma_ratio_floor,
            ewma_ratio_ceiling=config.walk_forward_ewma_ratio_ceiling,
            bootstrap_samples=config.walk_forward_bootstrap_samples,
            bootstrap_seed=config.walk_forward_bootstrap_seed,
        )
        variance_forecasts = walk_forward.pop("forecast_rows")
        report = {
            "schema_version": SCHEMA_VERSION,
            "study_id": study_id,
            "run_id": run_id,
            "family_id": FAMILY_ID,
            "generated_ts": datetime.now(UTC).isoformat(),
            "research_only": True,
            "usage_classification": "preexisting_research_and_hypothesis_generation_only",
            "formal_final_holdout_accessed": False,
            "joint_signal_model_created": False,
            "trading_eligible": False,
            "baseline_study_id": config.baseline_study_id,
            "config": public_config,
            "inputs": {
                name: {"path": str(path), "sha256": input_hashes[name]}
                for name, path in inputs.items()
            },
            "one_day_natural_target_panel": {
                "source": "generic one-minute Pyth proxies and optional Databento SI trades",
                "unit_of_split": "complete Kalshi window",
                "development_windows": len(development_tickers),
                "embargo_windows": len(embargo_tickers),
                "test_windows": len(test_tickers),
                "development_rows": len(development),
                "test_rows": len(test),
                "effective_test_utc_days": len({row["utc_day"] for row in test}),
                "tail_thresholds_fitted_on_development": {
                    "lower_standardized_return": lower_tail,
                    "upper_standardized_return": upper_tail,
                },
                "selected_futures_contract": selected_contract,
                "development_only_future_silver_on_gold_beta": cross_beta,
                "disposition": DISPOSITION,
                "confidence_interval": {
                    "block_unit": "UTC_trading_day",
                    "effective_blocks": len({row["utc_day"] for row in test}),
                    "status": "not_estimable_fewer_than_two_days",
                },
            },
            "signal_cards": cards,
            "walk_forward_slow_variance_diagnostic": {
                **walk_forward,
                "source": "pre-validation provisional Kalshi-reported opening and expiration levels",
                "separate_from_one_day_signal_panel": True,
            },
            "unavailable_natural_targets": unavailable_target_registry(),
            "conclusion": {
                "disposition": DISPOSITION,
                "reason": "the synchronized signal test has one effective UTC day; metrics are diagnostics, not admission evidence",
                "pooled_coefficient_scope": (
                    "plumbing diagnostic only; horizon-bucket metrics are reported, "
                    "but a future terminal model must use continuous signal-by-time "
                    "and signal-by-moneyness interactions"
                ),
                "model_admission_performed": False,
                "trading_claim_permitted": False,
            },
            "artifacts": {
                "report": str(report_path),
                "signal_cards": str(cards_path),
                "one_day_rows": str(rows_path),
                "walk_forward_variance_rows": str(variance_rows_path),
                "markdown": str(markdown_path),
                "manifest": str(manifest_path),
            },
        }
        output_dir.mkdir(parents=True, exist_ok=False)
        with HashChainedJsonlWriter(rows_path, fsync_every_records=100) as writer:
            for row in augmented:
                writer.write(row)
        with HashChainedJsonlWriter(
            variance_rows_path, fsync_every_records=100
        ) as writer:
            for row in variance_forecasts:
                writer.write(row)
        write_signal_json(cards_path, {"signal_cards": cards})
        write_signal_json(report_path, report)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        update_run_manifest(
            root,
            run_id,
            status="completed",
            counts={
                "natural_target_signal_cards": len(cards),
                "development_windows": len(development_tickers),
                "embargo_windows": len(embargo_tickers),
                "test_windows": len(test_tickers),
                "walk_forward_scored_windows": walk_forward["cohort"][
                    "scored_windows"
                ],
                "formal_final_holdout_accessed": False,
                "joint_signal_model_created": False,
            },
            output_paths=(
                report_path,
                cards_path,
                rows_path,
                variance_rows_path,
                markdown_path,
            ),
        )
        return report
    except BaseException as error:
        update_run_manifest(root, run_id, status="failed", errors=(error,))
        raise
