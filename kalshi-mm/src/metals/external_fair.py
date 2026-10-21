"""Causal SI-to-Pyth replication and external-fair fill attribution.

This module is research-only. It deliberately has no exchange client or order path.
All joins used by a policy are keyed by local receive order; source timestamps are
retained only for clock diagnostics.
"""

from __future__ import annotations

import bisect
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


LAG_SWEEP_MS: Tuple[int, ...] = (-500, -250, -100, 0, 100, 250, 500)
ZERO = Decimal("0")


class CausalityError(ValueError):
    """Raised when a live-policy join requests information from the future."""


@dataclass(frozen=True, order=True)
class ReceiveKey:
    """Total local event order used for replay and feature availability."""

    local_monotonic_ts_ns: int
    ingestion_sequence: int

    def __post_init__(self) -> None:
        if self.local_monotonic_ts_ns < 0:
            raise ValueError("local_monotonic_ts_ns cannot be negative")
        if self.ingestion_sequence < 0:
            raise ValueError("ingestion_sequence cannot be negative")


@dataclass(frozen=True)
class PriceEvent:
    """One received Pyth or futures value with diagnostic timestamp provenance."""

    value: float
    receive_key: ReceiveKey
    day: str
    session: str
    source_event_ts_ns: Optional[int] = None
    vendor_ts_ns: Optional[int] = None
    publication_ts_ns: Optional[int] = None
    symbol: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("PriceEvent value must be finite")
        if not self.day:
            raise ValueError("PriceEvent day is required")
        if not self.session:
            raise ValueError("PriceEvent session is required")


@dataclass(frozen=True)
class ReplicatorRow:
    """One next-Pyth target and the SI innovation observable before it arrived."""

    day: str
    session: str
    anchor_pyth: float
    observed_pyth: float
    anchor_futures: float
    current_futures: float
    anchor_pyth_key: ReceiveKey
    decision_key: ReceiveKey
    anchor_futures_key: ReceiveKey
    current_futures_key: ReceiveKey
    shift_ms: int
    strike_distance: Optional[float] = None

    @property
    def futures_change(self) -> float:
        return self.current_futures - self.anchor_futures

    @property
    def pyth_change(self) -> float:
        return self.observed_pyth - self.anchor_pyth

    @property
    def stale_prediction_error(self) -> float:
        return self.observed_pyth - self.anchor_pyth

    @property
    def uses_future_input(self) -> bool:
        return (
            self.anchor_futures_key > self.anchor_pyth_key
            or self.current_futures_key > self.decision_key
        )

    @property
    def maximum_input_key(self) -> ReceiveKey:
        return max(
            self.anchor_pyth_key,
            self.anchor_futures_key,
            self.current_futures_key,
        )


@dataclass(frozen=True)
class ReplicatorCoefficient:
    session: str
    beta: float
    basis_correction: float
    observations: int


@dataclass(frozen=True)
class ReplicatorModel:
    coefficients: Mapping[str, ReplicatorCoefficient]

    def predict(self, row: ReplicatorRow) -> float:
        coefficient = self.coefficients.get(row.session)
        if coefficient is None:
            coefficient = self.coefficients["__all__"]
        return (
            row.anchor_pyth
            + coefficient.beta * row.futures_change
            + coefficient.basis_correction
        )


@dataclass(frozen=True)
class ReplicatorValidationLimits:
    """Predeclared gates; callers must freeze these before holdout inspection."""

    minimum_session_observations: int
    maximum_absolute_session_bias: float
    maximum_absolute_daily_beta_drift: float
    maximum_absolute_residual_futures_correlation: float
    maximum_p90_error_to_strike_distance: float

    def __post_init__(self) -> None:
        if self.minimum_session_observations < 2:
            raise ValueError("minimum_session_observations must be at least two")
        for name, value in (
            ("maximum_absolute_session_bias", self.maximum_absolute_session_bias),
            (
                "maximum_absolute_daily_beta_drift",
                self.maximum_absolute_daily_beta_drift,
            ),
            (
                "maximum_absolute_residual_futures_correlation",
                self.maximum_absolute_residual_futures_correlation,
            ),
            (
                "maximum_p90_error_to_strike_distance",
                self.maximum_p90_error_to_strike_distance,
            ),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("{} must be finite and nonnegative".format(name))


def _ordered_events(events: Iterable[PriceEvent]) -> List[PriceEvent]:
    ordered = sorted(events, key=lambda event: event.receive_key)
    if any(
        left.receive_key == right.receive_key
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError("PriceEvent receive keys must be unique within a source")
    return ordered


def asof_receive_event(
    ordered_events: Sequence[PriceEvent],
    decision_key: ReceiveKey,
    *,
    shift_ms: int = 0,
    allow_noncausal: bool = False,
) -> Optional[PriceEvent]:
    """Return the most recent event in local receive order.

    A negative shift deliberately makes the feed older. A positive shift is allowed
    only for explicit leakage/clock diagnostics and is never a policy join.
    """

    if shift_ms > 0 and not allow_noncausal:
        raise CausalityError("positive receive-time shifts are diagnostic only")
    cutoff_ns = decision_key.local_monotonic_ts_ns + shift_ms * 1_000_000
    cutoff_sequence = (
        decision_key.ingestion_sequence if shift_ms == 0 else 2**63 - 1
    )
    cutoff = ReceiveKey(max(cutoff_ns, 0), cutoff_sequence)
    keys = [event.receive_key for event in ordered_events]
    index = bisect.bisect_right(keys, cutoff) - 1
    return ordered_events[index] if index >= 0 else None


def build_replicator_rows(
    pyth_events: Iterable[PriceEvent],
    futures_events: Iterable[PriceEvent],
    *,
    shift_ms: int = 0,
    allow_noncausal: bool = False,
    strike_by_day: Optional[Mapping[str, float]] = None,
) -> List[ReplicatorRow]:
    """Build consecutive next-Pyth targets using receive-time as-of SI values."""

    pyth = _ordered_events(pyth_events)
    futures = _ordered_events(futures_events)
    rows: List[ReplicatorRow] = []
    for anchor, observed in zip(pyth, pyth[1:]):
        anchor_future = asof_receive_event(
            futures,
            anchor.receive_key,
            shift_ms=shift_ms,
            allow_noncausal=allow_noncausal,
        )
        current_future = asof_receive_event(
            futures,
            observed.receive_key,
            shift_ms=shift_ms,
            allow_noncausal=allow_noncausal,
        )
        if anchor_future is None or current_future is None:
            continue
        strike = (strike_by_day or {}).get(observed.day)
        rows.append(
            ReplicatorRow(
                day=observed.day,
                session=observed.session,
                anchor_pyth=anchor.value,
                observed_pyth=observed.value,
                anchor_futures=anchor_future.value,
                current_futures=current_future.value,
                anchor_pyth_key=anchor.receive_key,
                decision_key=observed.receive_key,
                anchor_futures_key=anchor_future.receive_key,
                current_futures_key=current_future.receive_key,
                shift_ms=shift_ms,
                strike_distance=(
                    abs(anchor.value - strike) if strike is not None else None
                ),
            )
        )
    return rows


def _fit_line(rows: Sequence[ReplicatorRow]) -> Tuple[float, float]:
    if not rows:
        raise ValueError("at least one row is required")
    xs = [row.futures_change for row in rows]
    ys = [row.pyth_change for row in rows]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    beta = (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        / denominator
        if denominator > 0
        else 0.0
    )
    return beta, mean_y - beta * mean_x


def fit_replicator(rows: Sequence[ReplicatorRow]) -> ReplicatorModel:
    """Fit OLS beta and basis correction globally and by declared session."""

    if not rows:
        raise ValueError("cannot fit a replicator without rows")
    grouped: Dict[str, List[ReplicatorRow]] = defaultdict(list)
    for row in rows:
        grouped[row.session].append(row)
    coefficients: Dict[str, ReplicatorCoefficient] = {}
    beta, correction = _fit_line(rows)
    coefficients["__all__"] = ReplicatorCoefficient(
        "__all__", beta, correction, len(rows)
    )
    for session, session_rows in grouped.items():
        session_beta, session_correction = _fit_line(session_rows)
        coefficients[session] = ReplicatorCoefficient(
            session, session_beta, session_correction, len(session_rows)
        )
    return ReplicatorModel(coefficients)


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = int((len(ordered) - 1) * probability)
    return ordered[index]


def _correlation(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if not left or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss <= 0 or right_ss <= 0:
        return 0.0
    covariance = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    return covariance / math.sqrt(left_ss * right_ss)


def replicator_metrics(
    model: ReplicatorModel, rows: Sequence[ReplicatorRow]
) -> Dict[str, Any]:
    predictions = [model.predict(row) for row in rows]
    residuals = [
        row.observed_pyth - prediction for row, prediction in zip(rows, predictions)
    ]
    baseline_errors = [row.stale_prediction_error for row in rows]
    by_session: Dict[str, List[float]] = defaultdict(list)
    for row, residual in zip(rows, residuals):
        by_session[row.session].append(residual)
    ratios = []
    for row, residual in zip(rows, residuals):
        if row.strike_distance is None:
            continue
        if row.strike_distance == 0:
            ratios.append(0.0 if residual == 0 else math.inf)
        else:
            ratios.append(abs(residual) / row.strike_distance)
    return {
        "observations": len(rows),
        "mae": _mean([abs(value) for value in residuals]),
        "rmse": (
            math.sqrt(sum(value * value for value in residuals) / len(residuals))
            if residuals
            else None
        ),
        "bias": _mean(residuals),
        "unchanged_pyth_mae": _mean([abs(value) for value in baseline_errors]),
        "session_bias": {
            session: _mean(values) for session, values in sorted(by_session.items())
        },
        "session_observations": {
            session: len(values) for session, values in sorted(by_session.items())
        },
        "residual_futures_change_correlation": _correlation(
            residuals, [row.futures_change for row in rows]
        ),
        "p90_absolute_error_to_strike_distance": _percentile(ratios, 0.90),
        "rows_using_future_input": sum(row.uses_future_input for row in rows),
    }


def validate_replicator(
    model: ReplicatorModel,
    heldout_rows: Sequence[ReplicatorRow],
    limits: ReplicatorValidationLimits,
    *,
    settlements_expected: int,
    settlements_reconstructed: int,
) -> Dict[str, Any]:
    """Evaluate every predeclared benchmark gate without silently dropping unknowns."""

    metrics = replicator_metrics(model, heldout_rows)
    session_biases = [
        abs(value)
        for session, value in metrics["session_bias"].items()
        if value is not None
        and metrics["session_observations"][session]
        >= limits.minimum_session_observations
    ]
    daily_beta_drift: Dict[str, float] = {}
    by_day_session: Dict[Tuple[str, str], List[ReplicatorRow]] = defaultdict(list)
    for row in heldout_rows:
        by_day_session[(row.day, row.session)].append(row)
    for (day, session), rows in sorted(by_day_session.items()):
        if len(rows) < limits.minimum_session_observations:
            continue
        daily_beta, _ = _fit_line(rows)
        fitted = model.coefficients.get(session, model.coefficients["__all__"])
        daily_beta_drift["{}|{}".format(day, session)] = abs(
            daily_beta - fitted.beta
        )
    correlation = metrics["residual_futures_change_correlation"]
    ratio = metrics["p90_absolute_error_to_strike_distance"]
    gates = {
        "better_than_unchanged_pyth": bool(
            metrics["mae"] is not None
            and metrics["unchanged_pyth_mae"] is not None
            and metrics["mae"] < metrics["unchanged_pyth_mae"]
        ),
        "near_zero_bias_by_session": bool(session_biases)
        and max(session_biases) <= limits.maximum_absolute_session_bias,
        "stable_beta_across_heldout_days": bool(daily_beta_drift)
        and max(daily_beta_drift.values())
        <= limits.maximum_absolute_daily_beta_drift,
        "no_material_signed_si_residual_relationship": correlation is not None
        and abs(correlation)
        <= limits.maximum_absolute_residual_futures_correlation,
        "settlements_exactly_reconstructed": settlements_expected > 0
        and settlements_reconstructed == settlements_expected,
        "no_future_receive_inputs": metrics["rows_using_future_input"] == 0
        and all(row.shift_ms <= 0 for row in heldout_rows),
        "error_small_relative_to_strike_distance": ratio is not None
        and ratio <= limits.maximum_p90_error_to_strike_distance,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": metrics,
        "daily_session_absolute_beta_drift": daily_beta_drift,
        "limits": {
            "minimum_session_observations": limits.minimum_session_observations,
            "maximum_absolute_session_bias": limits.maximum_absolute_session_bias,
            "maximum_absolute_daily_beta_drift": (
                limits.maximum_absolute_daily_beta_drift
            ),
            "maximum_absolute_residual_futures_correlation": (
                limits.maximum_absolute_residual_futures_correlation
            ),
            "maximum_p90_error_to_strike_distance": (
                limits.maximum_p90_error_to_strike_distance
            ),
        },
        "settlements": {
            "expected": settlements_expected,
            "reconstructed": settlements_reconstructed,
        },
    }


def run_replicator_lag_sweep(
    pyth_events: Iterable[PriceEvent],
    futures_events: Iterable[PriceEvent],
    *,
    training_days: Iterable[str],
    heldout_days: Iterable[str],
    shifts_ms: Sequence[int] = LAG_SWEEP_MS,
    strike_by_day: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Fit/evaluate every clock shift and label noncausal optima explicitly."""

    pyth = list(pyth_events)
    futures = list(futures_events)
    training = set(training_days)
    heldout = set(heldout_days)
    if not training or not heldout or training & heldout:
        raise ValueError("training and heldout days must be nonempty and disjoint")
    points = []
    for shift_ms in shifts_ms:
        rows = build_replicator_rows(
            pyth,
            futures,
            shift_ms=shift_ms,
            allow_noncausal=shift_ms > 0,
            strike_by_day=strike_by_day,
        )
        train_rows = [row for row in rows if row.day in training]
        test_rows = [row for row in rows if row.day in heldout]
        if not train_rows or not test_rows:
            points.append(
                {
                    "shift_ms": shift_ms,
                    "causal_shift": shift_ms <= 0,
                    "training_observations": len(train_rows),
                    "heldout_observations": len(test_rows),
                    "metrics": None,
                }
            )
            continue
        model = fit_replicator(train_rows)
        points.append(
            {
                "shift_ms": shift_ms,
                "causal_shift": shift_ms <= 0,
                "training_observations": len(train_rows),
                "heldout_observations": len(test_rows),
                "metrics": replicator_metrics(model, test_rows),
            }
        )
    valid = [point for point in points if point["metrics"] is not None]
    best = min(valid, key=lambda point: point["metrics"]["mae"]) if valid else None
    causal = [point for point in valid if point["causal_shift"]]
    best_causal = (
        min(causal, key=lambda point: point["metrics"]["mae"])
        if causal
        else None
    )
    return {
        "points": points,
        "best_shift_ms": best["shift_ms"] if best else None,
        "best_shift_is_noncausal": bool(best and not best["causal_shift"]),
        "best_causal_shift_ms": (
            best_causal["shift_ms"] if best_causal else None
        ),
    }


@dataclass(frozen=True)
class FairSnapshot:
    probability: Decimal
    decision_key: ReceiveKey
    maximum_input_key: ReceiveKey

    def __post_init__(self) -> None:
        if not ZERO <= self.probability <= Decimal("1"):
            raise ValueError("fair probability must be between zero and one")
        if self.maximum_input_key > self.decision_key:
            raise CausalityError("fair snapshot uses a future receive-time input")


@dataclass(frozen=True)
class FillAttribution:
    """Fair values and observed Kalshi mark for one acquired outcome."""

    fill_id: str
    day: str
    entry_cost: Decimal
    outcome: Decimal
    kalshi_mark_1s: Decimal
    fair_at_quote: FairSnapshot
    fair_at_fill: FairSnapshot
    fair_at_1s: FairSnapshot

    def __post_init__(self) -> None:
        for name, value in (
            ("entry_cost", self.entry_cost),
            ("outcome", self.outcome),
            ("kalshi_mark_1s", self.kalshi_mark_1s),
        ):
            if not ZERO <= value <= Decimal("1"):
                raise ValueError("{} must be between zero and one".format(name))
        if not self.fill_id or not self.day:
            raise ValueError("fill_id and day are required")
        if not (
            self.fair_at_quote.decision_key
            <= self.fair_at_fill.decision_key
            <= self.fair_at_1s.decision_key
        ):
            raise ValueError("quote, fill, and one-second decisions are out of order")

    def components(self) -> Dict[str, Decimal]:
        quote_edge = self.fair_at_quote.probability - self.entry_cost
        stale_quote = (
            self.fair_at_fill.probability - self.fair_at_quote.probability
        )
        post_fill_external = (
            self.fair_at_1s.probability - self.fair_at_fill.probability
        )
        residual_kalshi_flow = (
            self.kalshi_mark_1s - self.fair_at_1s.probability
        )
        forecast_residual = self.outcome - self.fair_at_1s.probability
        one_second_markout = self.kalshi_mark_1s - self.entry_cost
        settlement_pnl = self.outcome - self.entry_cost
        if (
            quote_edge + stale_quote + post_fill_external + residual_kalshi_flow
            != one_second_markout
        ):
            raise AssertionError("one-second attribution does not reconcile")
        if (
            quote_edge + stale_quote + post_fill_external + forecast_residual
            != settlement_pnl
        ):
            raise AssertionError("settlement attribution does not reconcile")
        return {
            "kalshi_fill_to_1s_markout": one_second_markout,
            "quoted_edge": quote_edge,
            "external_fair_move_before_fill": stale_quote,
            "external_fair_move_after_fill": post_fill_external,
            "residual_kalshi_flow_markout": residual_kalshi_flow,
            "forecast_residual": forecast_residual,
            "settlement_pnl": settlement_pnl,
        }


def _decimal_percentile(
    values: Sequence[Decimal], probability: Decimal
) -> Optional[Decimal]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[int(Decimal(len(ordered) - 1) * probability)]


def _day_block_interval(
    rows: Sequence[FillAttribution],
    component: str,
    *,
    samples: int,
    seed: int,
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    by_day: Dict[str, List[Decimal]] = defaultdict(list)
    for row in rows:
        by_day[row.day].append(row.components()[component])
    days = [by_day[day] for day in sorted(by_day)]
    if not days:
        return None, None
    generator = random.Random(seed)
    means = []
    for _ in range(samples):
        selected = [days[generator.randrange(len(days))] for _ in days]
        total = sum((sum(day, ZERO) for day in selected), ZERO)
        count = sum(len(day) for day in selected)
        means.append(total / Decimal(count))
    return (
        _decimal_percentile(means, Decimal("0.05")),
        _decimal_percentile(means, Decimal("0.95")),
    )


def summarize_attribution(
    rows: Sequence[FillAttribution],
    *,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 17_291,
) -> List[Dict[str, Any]]:
    """Return reconciled one-second attribution in cents and toxicity shares."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    component_order = (
        "kalshi_fill_to_1s_markout",
        "quoted_edge",
        "external_fair_move_before_fill",
        "external_fair_move_after_fill",
        "residual_kalshi_flow_markout",
    )
    totals = [row.components()["kalshi_fill_to_1s_markout"] for row in rows]
    total_mean = sum(totals, ZERO) / Decimal(len(totals)) if totals else None
    summary = []
    for component in component_order:
        values = [row.components()[component] for row in rows]
        mean = sum(values, ZERO) / Decimal(len(values)) if values else None
        lower, upper = _day_block_interval(
            rows,
            component,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        share = (
            mean / total_mean * Decimal("100")
            if mean is not None and total_mean not in (None, ZERO)
            else None
        )
        summary.append(
            {
                "component": component,
                "observations": len(values),
                "mean_dollars": mean,
                "mean_cents": mean * Decimal("100") if mean is not None else None,
                "day_block_90pct_ci_cents": (
                    lower * Decimal("100") if lower is not None else None,
                    upper * Decimal("100") if upper is not None else None,
                ),
                "signed_share_of_total_markout_pct": share,
            }
        )
    component_means = {
        row["component"]: row["mean_dollars"] for row in summary
    }
    reconciled = sum(
        (
            component_means[name]
            for name in component_order[1:]
            if component_means[name] is not None
        ),
        ZERO,
    )
    if total_mean is not None and reconciled != total_mean:
        raise AssertionError("aggregate one-second attribution does not reconcile")
    return summary
