"""Exploratory Gaussian and empirical baseline fair for 15-minute metals markets.

The model is intentionally small: an anchored external spot proxy, a shrunk UTC
quarter-hour variance curve, a slow deseasonalized EWMA state, and a standardized
empirical residual CDF.  Exchange-reported historical labels are permitted here only
for development and hypothesis generation.  This module cannot open the formal
holdout or produce a trading-eligible model.
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
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median
from typing import Any

from .research_design import verify_research_design
from .storage import HashChainedJsonlWriter, update_run_manifest, write_run_manifest

NSEC = 1_000_000_000
WINDOW_SECONDS = 15 * 60
SCHEMA_VERSION = "metals-baseline-fair/v1"
USAGE_CLASSIFICATION = "preexisting_research_and_hypothesis_generation_only"


class BaselineFairError(ValueError):
    """Raised when model inputs, chronology, or immutable provenance are invalid."""


@dataclass(frozen=True)
class BaselineFairConfig:
    metal: str
    training_end_ts_ns: int
    validation_start_ts_ns: int
    validation_end_ts_ns: int
    formal_research_design_id: str
    prerequisite_proxy_pilot_id: str
    ewma_lambda: float = 0.97
    tod_prior_observations: int = 20
    winsor_quantile: float = 0.995
    ewma_ratio_floor: float = 0.25
    ewma_ratio_ceiling: float = 4.0
    decision_interval_seconds: int = 60
    probability_floor: float = 0.005
    train_validation_embargo_seconds: int = 900
    pyth_alignment: str = "preceding_minute_close"
    spot_anchor_method: str = "multiplicative_window_open_anchor"
    conditional_mean: str = "zero"
    settlement_rounding: str = "provisional_half_up"

    def __post_init__(self) -> None:
        if self.metal.strip().lower() not in {"silver", "gold"}:
            raise BaselineFairError("metal must be silver or gold")
        if not (
            0
            <= self.training_end_ts_ns
            <= self.validation_start_ts_ns
            < self.validation_end_ts_ns
        ):
            raise BaselineFairError("training and validation timestamps are invalid")
        if self.training_end_ts_ns != self.validation_start_ts_ns:
            raise BaselineFairError("training must end exactly where validation starts")
        if not 0 < self.ewma_lambda < 1:
            raise BaselineFairError("ewma_lambda must be between zero and one")
        if self.tod_prior_observations < 1:
            raise BaselineFairError("tod_prior_observations must be positive")
        if not 0.9 <= self.winsor_quantile < 1:
            raise BaselineFairError("winsor_quantile must be in [0.9, 1)")
        if not 0 < self.ewma_ratio_floor <= 1 <= self.ewma_ratio_ceiling:
            raise BaselineFairError("EWMA ratio bounds must contain one")
        if self.decision_interval_seconds < 1:
            raise BaselineFairError("decision_interval_seconds must be positive")
        if WINDOW_SECONDS % self.decision_interval_seconds:
            raise BaselineFairError(
                "decision interval must divide the 15-minute window"
            )
        if not 0 < self.probability_floor < 0.5:
            raise BaselineFairError("probability_floor must be between zero and 0.5")
        if self.train_validation_embargo_seconds < WINDOW_SECONDS:
            raise BaselineFairError("embargo must cover the maximum return horizon")
        if self.pyth_alignment != "preceding_minute_close":
            raise BaselineFairError(
                "only the validated preceding-minute alignment is allowed"
            )
        if self.spot_anchor_method != "multiplicative_window_open_anchor":
            raise BaselineFairError("unsupported spot anchor method")
        if self.conditional_mean != "zero":
            raise BaselineFairError(
                "the minimum baseline must use zero conditional mean"
            )
        if self.settlement_rounding != "provisional_half_up":
            raise BaselineFairError("unsupported provisional settlement rule")
        for label, value in (
            ("formal_research_design_id", self.formal_research_design_id),
            ("prerequisite_proxy_pilot_id", self.prerequisite_proxy_pilot_id),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise BaselineFairError(
                    f"{label} must be a lowercase SHA-256 identifier"
                )


@dataclass(frozen=True)
class ReturnObservation:
    market_ticker: str
    open_ts_ns: int
    close_ts_ns: int
    target: float
    expiration: float
    outcome: int
    round_digits: int

    @property
    def bucket(self) -> int:
        return (self.open_ts_ns // NSEC % 86_400) // WINDOW_SECONDS

    @property
    def log_return(self) -> float:
        return math.log(self.expiration / self.target)


@dataclass(frozen=True)
class BaselineVolModel:
    global_variance_15m: float
    winsor_absolute_log_return: float
    time_of_day_variance: Mapping[int, float]
    time_of_day_observations: Mapping[int, int]
    ewma_variance_at_training_end: float
    ewma_lambda: float
    ewma_ratio_floor: float
    ewma_ratio_ceiling: float
    standardized_residual_scale: float
    standardized_residuals: tuple[float, ...]
    probability_floor: float

    def __post_init__(self) -> None:
        if self.global_variance_15m <= 0 or not math.isfinite(self.global_variance_15m):
            raise BaselineFairError("global variance must be finite and positive")
        if not self.standardized_residuals:
            raise BaselineFairError("empirical residual distribution cannot be empty")
        if tuple(sorted(self.standardized_residuals)) != self.standardized_residuals:
            raise BaselineFairError("empirical residuals must be sorted")

    def time_variance(self, bucket: int) -> float:
        return self.time_of_day_variance.get(bucket, self.global_variance_15m)

    def ewma_ratio(self, ewma_variance: float) -> float:
        return max(
            self.ewma_ratio_floor,
            min(self.ewma_ratio_ceiling, ewma_variance / self.global_variance_15m),
        )

    def full_window_variance(self, bucket: int, ewma_variance: float) -> float:
        return (
            self.time_variance(bucket)
            * self.ewma_ratio(ewma_variance)
            * self.standardized_residual_scale**2
        )

    def update_ewma(
        self, ewma_variance: float, log_return: float, bucket: int
    ) -> float:
        clipped = max(
            -self.winsor_absolute_log_return,
            min(self.winsor_absolute_log_return, log_return),
        )
        seasonal_multiplier = self.time_variance(bucket) / self.global_variance_15m
        deseasonalized_squared = clipped * clipped / max(seasonal_multiplier, 1e-12)
        return (
            self.ewma_lambda * ewma_variance
            + (1 - self.ewma_lambda) * deseasonalized_squared
        )

    def _threshold_z(
        self,
        current_index: float,
        strike: float,
        remaining_seconds: int,
        bucket: int,
        ewma_variance: float,
        round_digits: int,
        volatility_multiplier: float = 1.0,
    ) -> float:
        if not math.isfinite(volatility_multiplier) or volatility_multiplier <= 0:
            raise BaselineFairError("volatility multiplier must be positive and finite")
        if current_index <= 0 or strike <= 0:
            raise BaselineFairError("current index and strike must be positive")
        if not 0 < remaining_seconds <= WINDOW_SECONDS:
            raise BaselineFairError("remaining_seconds must be in (0, 900]")
        if not 0 <= round_digits <= 9:
            raise BaselineFairError("round_digits must be between zero and nine")
        half_grid = 0.5 * 10 ** (-round_digits)
        raw_yes_threshold = strike - half_grid
        if raw_yes_threshold <= 0:
            raise BaselineFairError("rounded settlement threshold must be positive")
        remaining_variance = self.full_window_variance(bucket, ewma_variance) * (
            remaining_seconds / WINDOW_SECONDS
        ) * volatility_multiplier**2
        return math.log(raw_yes_threshold / current_index) / math.sqrt(
            max(remaining_variance, 1e-18)
        )

    def probability_yes(
        self,
        current_index: float,
        strike: float,
        remaining_seconds: int,
        bucket: int,
        ewma_variance: float,
        round_digits: int,
        distribution: str,
        volatility_multiplier: float = 1.0,
    ) -> float:
        threshold = self._threshold_z(
            current_index,
            strike,
            remaining_seconds,
            bucket,
            ewma_variance,
            round_digits,
            volatility_multiplier,
        )
        if distribution == "gaussian":
            raw_probability = 0.5 * math.erfc(threshold / math.sqrt(2))
        elif distribution == "empirical":
            index = bisect.bisect_left(self.standardized_residuals, threshold)
            raw_probability = (len(self.standardized_residuals) - index + 0.5) / (
                len(self.standardized_residuals) + 1
            )
        else:
            raise BaselineFairError(f"unsupported distribution: {distribution}")
        return max(
            self.probability_floor,
            min(1 - self.probability_floor, raw_probability),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "global_variance_15m": self.global_variance_15m,
            "global_sigma_15m": math.sqrt(self.global_variance_15m),
            "winsor_absolute_log_return": self.winsor_absolute_log_return,
            "time_of_day_variance": {
                str(key): value
                for key, value in sorted(self.time_of_day_variance.items())
            },
            "time_of_day_observations": {
                str(key): value
                for key, value in sorted(self.time_of_day_observations.items())
            },
            "ewma_variance_at_training_end": self.ewma_variance_at_training_end,
            "ewma_ratio_at_training_end": self.ewma_ratio(
                self.ewma_variance_at_training_end
            ),
            "ewma_lambda": self.ewma_lambda,
            "ewma_ratio_floor": self.ewma_ratio_floor,
            "ewma_ratio_ceiling": self.ewma_ratio_ceiling,
            "standardized_residual_scale": self.standardized_residual_scale,
            "standardized_residuals": list(self.standardized_residuals),
            "probability_floor": self.probability_floor,
        }


def load_baseline_vol_model(
    path: Path,
    *,
    expected_study_id: str | None = None,
) -> tuple[BaselineVolModel, str]:
    """Load and validate one immutable baseline model artifact.

    The online shadow path consumes the frozen model directly instead of fitting
    against live observations.  Returning the study ID keeps every decision bound
    to the exact research artifact that produced it.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineFairError("baseline model artifact is unreadable") from error
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise BaselineFairError("baseline model schema is not supported")
    study_id = str(payload.get("study_id", "")).strip()
    if not study_id:
        raise BaselineFairError("baseline model artifact has no study ID")
    if expected_study_id is not None and study_id != expected_study_id:
        raise BaselineFairError("baseline model study ID does not match")
    if payload.get("usage_classification") != USAGE_CLASSIFICATION:
        raise BaselineFairError("baseline model usage classification is invalid")
    raw = payload.get("model")
    if not isinstance(raw, Mapping):
        raise BaselineFairError("baseline model payload is missing")
    try:
        model = BaselineVolModel(
            global_variance_15m=float(raw["global_variance_15m"]),
            winsor_absolute_log_return=float(raw["winsor_absolute_log_return"]),
            time_of_day_variance={
                int(key): float(value)
                for key, value in raw["time_of_day_variance"].items()
            },
            time_of_day_observations={
                int(key): int(value)
                for key, value in raw["time_of_day_observations"].items()
            },
            ewma_variance_at_training_end=float(
                raw["ewma_variance_at_training_end"]
            ),
            ewma_lambda=float(raw["ewma_lambda"]),
            ewma_ratio_floor=float(raw["ewma_ratio_floor"]),
            ewma_ratio_ceiling=float(raw["ewma_ratio_ceiling"]),
            standardized_residual_scale=float(raw["standardized_residual_scale"]),
            standardized_residuals=tuple(
                float(value) for value in raw["standardized_residuals"]
            ),
            probability_floor=float(raw["probability_floor"]),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise BaselineFairError("baseline model payload is malformed") from error
    if set(model.time_of_day_variance) != set(range(96)):
        raise BaselineFairError("baseline model must contain all 96 UTC buckets")
    return model, study_id


def fit_baseline_vol_model(
    observations: Sequence[ReturnObservation], config: BaselineFairConfig
) -> BaselineVolModel:
    if len(observations) < 100:
        raise BaselineFairError("at least 100 chronological returns are required")
    ordered = sorted(observations, key=lambda row: (row.open_ts_ns, row.market_ticker))
    absolute_returns = sorted(abs(row.log_return) for row in ordered)
    quantile_index = int((len(absolute_returns) - 1) * config.winsor_quantile)
    winsor_limit = absolute_returns[quantile_index]
    if winsor_limit <= 0:
        raise BaselineFairError("training returns have no positive dispersion")
    clipped_returns = [
        max(-winsor_limit, min(winsor_limit, row.log_return)) for row in ordered
    ]
    global_variance = sum(value * value for value in clipped_returns) / len(
        clipped_returns
    )
    grouped: dict[int, list[float]] = defaultdict(list)
    for row, clipped in zip(ordered, clipped_returns):
        grouped[row.bucket].append(clipped)
    bucket_variance: dict[int, float] = {}
    bucket_counts: dict[int, int] = {}
    for bucket in range(96):
        values = grouped.get(bucket, [])
        count = len(values)
        local = (
            sum(value * value for value in values) / count if count else global_variance
        )
        bucket_variance[bucket] = (
            count * local + config.tod_prior_observations * global_variance
        ) / (count + config.tod_prior_observations)
        bucket_counts[bucket] = count

    ewma = global_variance
    raw_standardized: list[float] = []
    for row, clipped in zip(ordered, clipped_returns):
        seasonal_multiplier = bucket_variance[row.bucket] / global_variance
        conditional_variance = ewma * seasonal_multiplier
        raw_standardized.append(clipped / math.sqrt(max(conditional_variance, 1e-18)))
        deseasonalized_squared = clipped * clipped / max(seasonal_multiplier, 1e-12)
        ewma = (
            config.ewma_lambda * ewma
            + (1 - config.ewma_lambda) * deseasonalized_squared
        )
    residual_scale = math.sqrt(
        sum(value * value for value in raw_standardized) / len(raw_standardized)
    )
    standardized = tuple(sorted(value / residual_scale for value in raw_standardized))
    return BaselineVolModel(
        global_variance_15m=global_variance,
        winsor_absolute_log_return=winsor_limit,
        time_of_day_variance=bucket_variance,
        time_of_day_observations=bucket_counts,
        ewma_variance_at_training_end=ewma,
        ewma_lambda=config.ewma_lambda,
        ewma_ratio_floor=config.ewma_ratio_floor,
        ewma_ratio_ceiling=config.ewma_ratio_ceiling,
        standardized_residual_scale=residual_scale,
        standardized_residuals=standardized,
        probability_floor=config.probability_floor,
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
        raise BaselineFairError(f"input escapes storage root: {path}") from error
    if not resolved.is_file():
        raise BaselineFairError(f"input is not a file: {path}")
    return resolved


def _read_hash_chained(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_previous: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise BaselineFairError(
                    f"invalid JSON at {path}:{line_number}"
                ) from error
            if not isinstance(row, dict):
                raise BaselineFairError(f"non-object at {path}:{line_number}")
            record_hash = row.get("record_sha256")
            payload = dict(row)
            payload.pop("record_sha256", None)
            computed = hashlib.sha256(_canonical(payload).encode()).hexdigest()
            if record_hash != computed:
                raise BaselineFairError(f"record hash mismatch at {path}:{line_number}")
            if row.get("previous_record_sha256") != expected_previous:
                raise BaselineFairError(
                    f"record chain mismatch at {path}:{line_number}"
                )
            expected_previous = str(record_hash)
            rows.append(row)
    if not rows:
        raise BaselineFairError(f"empty input: {path}")
    return rows


def _decimal_float(value: Any, label: str) -> float:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise BaselineFairError(f"invalid decimal {label}") from error
    if not parsed.is_finite():
        raise BaselineFairError(f"non-finite decimal {label}")
    return float(parsed)


def _label_observations(
    labels: Sequence[Mapping[str, Any]],
    config: BaselineFairConfig,
) -> tuple[list[ReturnObservation], list[ReturnObservation], list[dict[str, Any]]]:
    training: list[ReturnObservation] = []
    validation: list[ReturnObservation] = []
    exclusions: list[dict[str, Any]] = []
    for row in labels:
        if str(row.get("metal", "")).lower() != config.metal.lower():
            continue
        open_ts_ns = int(row.get("open_ts_ns") or -1)
        close_ts_ns = int(row.get("close_ts_ns") or -1)
        in_training = close_ts_ns <= config.training_end_ts_ns
        in_validation = (
            config.validation_start_ts_ns <= open_ts_ns
            and close_ts_ns <= config.validation_end_ts_ns
        )
        if not in_training and not in_validation:
            continue
        reasons: list[str] = []
        try:
            target = _decimal_float(row.get("target_price"), "target_price")
            expiration = _decimal_float(row.get("expiration_value"), "expiration_value")
        except BaselineFairError:
            target = expiration = math.nan
            reasons.append("missing_or_invalid_level")
        result = str(row.get("result", "")).lower()
        if close_ts_ns - open_ts_ns != WINDOW_SECONDS * NSEC:
            reasons.append("non_15_minute_window")
        if not (target > 0 and expiration > 0):
            reasons.append("nonpositive_level")
        if result not in {"yes", "no"}:
            reasons.append("invalid_result")
        elif math.isfinite(target) and math.isfinite(expiration):
            expected = "yes" if expiration >= target else "no"
            if expected != result:
                reasons.append("displayed_values_contradict_result")
        round_digits = row.get("round_digits")
        if isinstance(round_digits, bool) or not isinstance(round_digits, int):
            reasons.append("invalid_round_digits")
            round_digits = 3
        if reasons:
            exclusions.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "market_ticker": row.get("market_ticker"),
                    "split": "training" if in_training else "validation",
                    "reasons": sorted(set(reasons)),
                }
            )
            continue
        observation = ReturnObservation(
            market_ticker=str(row["market_ticker"]),
            open_ts_ns=open_ts_ns,
            close_ts_ns=close_ts_ns,
            target=target,
            expiration=expiration,
            outcome=1 if result == "yes" else 0,
            round_digits=round_digits,
        )
        (training if in_training else validation).append(observation)
    training.sort(key=lambda row: (row.open_ts_ns, row.market_ticker))
    validation.sort(key=lambda row: (row.open_ts_ns, row.market_ticker))
    if not training or not validation:
        raise BaselineFairError(
            "both training and validation observations are required"
        )
    if training[-1].close_ts_ns > config.validation_start_ts_ns:
        raise BaselineFairError("training labels overlap validation")
    return training, validation, exclusions


def _metric(
    rows: Sequence[Mapping[str, Any]], probability_field: str
) -> dict[str, Any]:
    if not rows:
        return {"observations": 0, "brier": None, "log_loss": None}
    probabilities = [float(row[probability_field]) for row in rows]
    outcomes = [int(row["outcome"]) for row in rows]
    brier = sum(
        (probability - outcome) ** 2
        for probability, outcome in zip(probabilities, outcomes)
    ) / len(rows)
    log_loss = -sum(
        outcome * math.log(probability) + (1 - outcome) * math.log(1 - probability)
        for probability, outcome in zip(probabilities, outcomes)
    ) / len(rows)
    return {
        "observations": len(rows),
        "windows": len({str(row["market_ticker"]) for row in rows}),
        "effective_utc_days": len({str(row["utc_day"]) for row in rows}),
        "brier": brier,
        "log_loss": log_loss,
        "mean_probability": sum(probabilities) / len(probabilities),
        "outcome_rate": sum(outcomes) / len(outcomes),
    }


def _calibration(
    rows: Sequence[Mapping[str, Any]], probability_field: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for lower_index in range(10):
        lower = lower_index / 10
        upper = (lower_index + 1) / 10
        selected = [
            row
            for row in rows
            if lower <= float(row[probability_field]) < upper
            or (upper == 1 and float(row[probability_field]) == 1)
        ]
        if not selected:
            continue
        result.append(
            {
                "lower": lower,
                "upper": upper,
                "observations": len(selected),
                "windows": len({row["market_ticker"] for row in selected}),
                "mean_probability": sum(
                    float(row[probability_field]) for row in selected
                )
                / len(selected),
                "outcome_rate": sum(int(row["outcome"]) for row in selected)
                / len(selected),
            }
        )
    return result


def _remaining_time_metrics(
    rows: Sequence[Mapping[str, Any]], probability_field: str
) -> dict[str, Any]:
    cuts = {
        "11_to_15_minutes": (11 * 60, 15 * 60),
        "7_to_10_minutes": (7 * 60, 10 * 60),
        "3_to_6_minutes": (3 * 60, 6 * 60),
        "1_to_2_minutes": (1 * 60, 2 * 60),
    }
    return {
        name: _metric(
            [row for row in rows if lower <= int(row["remaining_seconds"]) <= upper],
            probability_field,
        )
        for name, (lower, upper) in cuts.items()
    }


def _model_invariants(model: BaselineVolModel) -> dict[str, Any]:
    spots = (59.9, 60.0, 60.1)
    strikes = (59.9, 60.0, 60.1)
    failures: list[str] = []
    for distribution in ("gaussian", "empirical"):
        by_spot = [
            model.probability_yes(
                spot,
                60.0,
                300,
                48,
                model.ewma_variance_at_training_end,
                3,
                distribution,
            )
            for spot in spots
        ]
        by_strike = [
            model.probability_yes(
                60.0,
                strike,
                300,
                48,
                model.ewma_variance_at_training_end,
                3,
                distribution,
            )
            for strike in strikes
        ]
        if by_spot != sorted(by_spot):
            failures.append(f"{distribution}_not_monotone_in_spot")
        if by_strike != sorted(by_strike, reverse=True):
            failures.append(f"{distribution}_not_monotone_in_strike")
        if any(
            not model.probability_floor <= value <= 1 - model.probability_floor
            for value in (*by_spot, *by_strike)
        ):
            failures.append(f"{distribution}_outside_probability_bounds")
    return {
        "passed": not failures,
        "failures": failures,
        "yes_no_sum_rule": "no_probability_is_exactly_one_minus_yes_probability",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _markdown(report: Mapping[str, Any]) -> str:
    gaussian = report["metrics"]["gaussian"]
    empirical = report["metrics"]["empirical"]
    return "\n".join(
        [
            "# Exploratory baseline fair",
            "",
            f"Training windows: **{report['cohort']['training_windows']}**. Validation windows with complete proxy coverage: **{report['cohort']['validation_windows_with_complete_proxy']}**.",
            "",
            "| Distribution | Observations | Brier | Log loss |",
            "|---|---:|---:|---:|",
            f"| Gaussian | {gaussian['overall']['observations']} | {gaussian['overall']['brier']:.6f} | {gaussian['overall']['log_loss']:.6f} |",
            f"| Empirical CDF | {empirical['overall']['observations']} | {empirical['overall']['brier']:.6f} | {empirical['overall']['log_loss']:.6f} |",
            "",
            "The validation observations come from one UTC day, so a day-block confidence interval is not estimable. Repeated minute rows from one market are not treated as independent windows.",
            "",
            "This artifact uses exchange-reported historical levels and generic Pyth only for development. It does not access the sealed final holdout and is not eligible for a trading claim.",
            "",
        ]
    )


def run_baseline_fair_study(
    storage_root: Path,
    labels_path: Path,
    pyth_proxy_candles_path: Path,
    config: BaselineFairConfig,
) -> dict[str, Any]:
    root = storage_root.resolve()
    labels_file = _resolve_input(root, labels_path)
    pyth_file = _resolve_input(root, pyth_proxy_candles_path)
    design = verify_research_design(
        root,
        root
        / "policies"
        / "research_design"
        / f"design={config.formal_research_design_id}",
    )
    if design.design_id != config.formal_research_design_id:
        raise BaselineFairError("formal research design ID does not match")
    pilot_file = _resolve_input(
        root,
        root
        / "derived"
        / "settlement_proxy_pilot"
        / f"run=settlement-proxy-{config.prerequisite_proxy_pilot_id[:24]}"
        / "report.json",
    )
    pilot = json.loads(pilot_file.read_text(encoding="utf-8"))
    if (
        pilot.get("pilot_id") != config.prerequisite_proxy_pilot_id
        or pilot.get("decision", {}).get("exploratory_expectations_passed") is not True
        or pilot.get("formal_settlement_gate_passed") is not False
    ):
        raise BaselineFairError("settlement proxy prerequisite is invalid")
    inputs = {
        "provisional_labels": labels_file,
        "pyth_generic_proxy": pyth_file,
        "formal_research_design": design.design_path,
        "formal_research_design_manifest": design.manifest_path,
        "settlement_proxy_pilot": pilot_file,
    }
    input_hashes = {name: _sha256_path(path) for name, path in inputs.items()}
    public_config = {**asdict(config), "metal": config.metal.lower()}
    study_id = hashlib.sha256(
        _canonical(
            {
                "schema_version": SCHEMA_VERSION,
                "usage_classification": USAGE_CLASSIFICATION,
                "config": public_config,
                "inputs": input_hashes,
            }
        ).encode()
    ).hexdigest()
    run_id = f"baseline-fair-{study_id[:24]}"
    output_dir = root / "derived" / "baseline_fair" / f"run={run_id}"
    report_path = output_dir / "report.json"
    model_path = output_dir / "model.json"
    rows_path = output_dir / "probabilities.jsonl"
    exclusions_path = output_dir / "exclusions.jsonl"
    markdown_path = output_dir / "REPORT.md"
    manifest_path = root / "manifests" / f"run={run_id}" / "manifest.json"
    if report_path.exists() and manifest_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))

    manifest_config = {
        "component": "baseline_fair",
        "schema_version": SCHEMA_VERSION,
        "usage_classification": USAGE_CLASSIFICATION,
        "research_only": True,
        "final_holdout_accessed": False,
        "live_orders": False,
        "network_access": False,
        "config": public_config,
        "input_sha256": input_hashes,
    }
    write_run_manifest(
        root,
        run_id,
        manifest_config,
        output_paths=(labels_file, pyth_file),
    )
    try:
        labels = _read_hash_chained(labels_file)
        pyth_rows = _read_hash_chained(pyth_file)
        if any(row.get("settlement_eligible") is not False for row in pyth_rows):
            raise BaselineFairError(
                "generic Pyth input must be settlement_eligible=false"
            )
        training, validation, exclusions = _label_observations(labels, config)
        model = fit_baseline_vol_model(training, config)
        proxy = {
            int(row["timestamp_seconds"]): _decimal_float(
                row.get("close"), "Pyth close"
            )
            for row in pyth_rows
            if str(row.get("metal", "")).lower() == config.metal.lower()
        }

        probability_rows: list[dict[str, Any]] = []
        ewma = model.ewma_variance_at_training_end
        covered_windows = 0
        embargo_boundary = (
            config.validation_start_ts_ns
            + config.train_validation_embargo_seconds * NSEC
        )
        for window in validation:
            open_seconds = window.open_ts_ns // NSEC
            close_seconds = window.close_ts_ns // NSEC
            if window.open_ts_ns < embargo_boundary:
                exclusions.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "market_ticker": window.market_ticker,
                        "split": "validation",
                        "reasons": ["train_validation_embargo"],
                    }
                )
                ewma = model.update_ewma(ewma, window.log_return, window.bucket)
                continue
            proxy_open = proxy.get(open_seconds - 60)
            proxy_close = proxy.get(close_seconds - 60)
            if proxy_open is None or proxy_close is None:
                exclusions.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "market_ticker": window.market_ticker,
                        "split": "validation",
                        "reasons": ["incomplete_generic_pyth_window"],
                    }
                )
                ewma = model.update_ewma(ewma, window.log_return, window.bucket)
                continue
            covered_windows += 1
            window_ewma = ewma
            for elapsed in range(0, WINDOW_SECONDS, config.decision_interval_seconds):
                decision_seconds = open_seconds + elapsed
                proxy_current = proxy.get(decision_seconds - 60)
                if proxy_current is None:
                    raise BaselineFairError(
                        f"interior proxy gap in complete window {window.market_ticker}"
                    )
                current_index = window.target * proxy_current / proxy_open
                remaining_seconds = close_seconds - decision_seconds
                gaussian = model.probability_yes(
                    current_index,
                    window.target,
                    remaining_seconds,
                    window.bucket,
                    window_ewma,
                    window.round_digits,
                    "gaussian",
                )
                empirical = model.probability_yes(
                    current_index,
                    window.target,
                    remaining_seconds,
                    window.bucket,
                    window_ewma,
                    window.round_digits,
                    "empirical",
                )
                raw_threshold = window.target - 0.5 * 10 ** (-window.round_digits)
                probability_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "study_id": study_id,
                        "market_ticker": window.market_ticker,
                        "utc_day": datetime.fromtimestamp(open_seconds, UTC)
                        .date()
                        .isoformat(),
                        "window_open_ts_ns": window.open_ts_ns,
                        "window_close_ts_ns": window.close_ts_ns,
                        "decision_ts_ns": decision_seconds * NSEC,
                        "maximum_proxy_input_minute_ts": decision_seconds - 60,
                        "remaining_seconds": remaining_seconds,
                        "time_of_day_bucket": window.bucket,
                        "target": window.target,
                        "raw_yes_threshold_under_provisional_rounding": raw_threshold,
                        "expiration": window.expiration,
                        "outcome": window.outcome,
                        "pyth_proxy_open": proxy_open,
                        "pyth_proxy_current": proxy_current,
                        "anchored_current_index": current_index,
                        "anchored_log_return_from_open": math.log(
                            current_index / window.target
                        ),
                        "future_log_return_to_reported_expiration": math.log(
                            window.expiration / current_index
                        ),
                        "time_of_day_variance_15m": model.time_variance(window.bucket),
                        "slow_ewma_variance": window_ewma,
                        "slow_ewma_ratio": model.ewma_ratio(window_ewma),
                        "forecast_variance_remaining": model.full_window_variance(
                            window.bucket, window_ewma
                        )
                        * remaining_seconds
                        / WINDOW_SECONDS,
                        "fair_yes_gaussian": gaussian,
                        "fair_no_gaussian": 1 - gaussian,
                        "fair_yes_empirical": empirical,
                        "fair_no_empirical": 1 - empirical,
                        "usage_classification": USAGE_CLASSIFICATION,
                        "final_holdout": False,
                        "trading_eligible": False,
                    }
                )
            ewma = model.update_ewma(ewma, window.log_return, window.bucket)

        if not probability_rows:
            raise BaselineFairError(
                "no complete validation windows have proxy coverage"
            )
        metrics: dict[str, Any] = {}
        for name, field in (
            ("gaussian", "fair_yes_gaussian"),
            ("empirical", "fair_yes_empirical"),
        ):
            metrics[name] = {
                "overall": _metric(probability_rows, field),
                "by_remaining_time": _remaining_time_metrics(probability_rows, field),
                "calibration_deciles": _calibration(probability_rows, field),
            }
        invariants = _model_invariants(model)
        if not invariants["passed"]:
            raise BaselineFairError("baseline probability invariants failed")
        report = {
            "schema_version": SCHEMA_VERSION,
            "study_id": study_id,
            "run_id": run_id,
            "generated_ts": datetime.now(UTC).isoformat(),
            "usage_classification": USAGE_CLASSIFICATION,
            "research_only": True,
            "trading_eligible": False,
            "formal_final_holdout_accessed": False,
            "formal_research_design_id": config.formal_research_design_id,
            "prerequisite_proxy_pilot_id": config.prerequisite_proxy_pilot_id,
            "config": public_config,
            "inputs": {
                name: {"path": str(path), "sha256": input_hashes[name]}
                for name, path in inputs.items()
            },
            "cohort": {
                "training_windows": len(training),
                "training_start_utc": datetime.fromtimestamp(
                    training[0].open_ts_ns / NSEC, UTC
                ).isoformat(),
                "training_end_utc": datetime.fromtimestamp(
                    training[-1].close_ts_ns / NSEC, UTC
                ).isoformat(),
                "validation_candidate_windows": len(validation),
                "validation_windows_with_complete_proxy": covered_windows,
                "probability_rows": len(probability_rows),
                "effective_validation_utc_days": len(
                    {row["utc_day"] for row in probability_rows}
                ),
                "excluded_records": len(exclusions),
            },
            "model_summary": {
                "global_sigma_15m_log_return": math.sqrt(model.global_variance_15m),
                "global_sigma_15m_dollars_at_median_training_level": math.sqrt(
                    model.global_variance_15m
                )
                * median([row.target for row in training]),
                "winsor_absolute_log_return": model.winsor_absolute_log_return,
                "ewma_ratio_at_training_end": model.ewma_ratio(
                    model.ewma_variance_at_training_end
                ),
                "standardized_residual_count": len(model.standardized_residuals),
                "standardized_residual_quantiles": {
                    str(probability): model.standardized_residuals[
                        int((len(model.standardized_residuals) - 1) * probability)
                    ]
                    for probability in (0.01, 0.05, 0.5, 0.95, 0.99)
                },
            },
            "metrics": metrics,
            "model_invariants": invariants,
            "confidence_interval": {
                "block_unit": "UTC_trading_day",
                "effective_blocks": len({row["utc_day"] for row in probability_rows}),
                "status": "not_estimable_fewer_than_two_days",
            },
            "decision": {
                "baseline_artifact_completed": True,
                "eligible_for_incremental_signal_development": True,
                "eligible_for_trading_claim": False,
                "next_signal_family": "fast_and_slow_spot_returns",
            },
            "limitations": [
                "Exchange-reported target, expiration, and result fields are provisional development labels.",
                "Generic one-minute Pyth is anchored at each Kalshi opening target and cannot certify the exact settlement feed.",
                "The full 15-minute variance is scaled linearly with remaining time; this is the minimum diffusion benchmark.",
                "The validation panel contains one UTC day and cannot support a day-block confidence interval or trading claim.",
                "Historical candle timestamps do not establish live receive latency.",
                "No Kalshi price, order book, or flow enters the external fair.",
            ],
            "artifacts": {
                "report": str(report_path),
                "model": str(model_path),
                "probabilities": str(rows_path),
                "exclusions": str(exclusions_path),
                "markdown": str(markdown_path),
                "manifest": str(manifest_path),
            },
        }

        output_dir.mkdir(parents=True, exist_ok=False)
        with HashChainedJsonlWriter(rows_path, fsync_every_records=100) as writer:
            for row in probability_rows:
                writer.write(row)
        with HashChainedJsonlWriter(exclusions_path, fsync_every_records=100) as writer:
            for exclusion in exclusions:
                writer.write(exclusion)
        _write_json(
            model_path,
            {
                "schema_version": SCHEMA_VERSION,
                "study_id": study_id,
                "usage_classification": USAGE_CLASSIFICATION,
                "model": model.public_dict(),
            },
        )
        _write_json(report_path, report)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        update_run_manifest(
            root,
            run_id,
            status="completed",
            counts={
                "training_windows": len(training),
                "validation_windows": covered_windows,
                "probability_rows": len(probability_rows),
                "effective_validation_days": len(
                    {row["utc_day"] for row in probability_rows}
                ),
                "formal_final_holdout_accessed": False,
                "trading_eligible": False,
            },
            output_paths=(
                report_path,
                model_path,
                rows_path,
                exclusions_path,
                markdown_path,
            ),
        )
        return report
    except BaseException as error:
        update_run_manifest(root, run_id, status="failed", errors=(error,))
        raise
