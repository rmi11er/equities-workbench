"""Independent range-volatility and path-efficiency signal cards."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from .spot_return_signals import (
    canonical_signal_json,
    fit_offset_logistic,
    read_signal_hash_chain,
    resolve_signal_input,
    signal_correlation,
    signal_distribution_effects,
    signal_fair_change_buckets,
    signal_metrics,
    signal_sha256_path,
    write_signal_json,
)
from .storage import HashChainedJsonlWriter, update_run_manifest, write_run_manifest

NSEC = 1_000_000_000
SCHEMA_VERSION = "metals-path-variance-signal-family/v1"
FAMILY_ID = "fast_slow_range_volatility_and_path_efficiency"


class PathVarianceSignalError(ValueError):
    """Raised when this signal-family study cannot preserve its frozen contract."""


@dataclass(frozen=True)
class PathVarianceSignalConfig:
    baseline_study_id: str
    range_lookback_minutes: tuple[int, ...] = (3, 10)
    efficiency_lookback_minutes: tuple[int, ...] = (5, 10)
    development_fraction: float = 0.67
    embargo_windows: int = 1
    ridge_penalty: float = 10.0
    minimum_development_windows: int = 40
    minimum_test_windows: int = 20
    range_variance_constant: float = 2.772588722239781
    variance_floor: float = 1e-12
    jump_threshold_standard_deviations: float = 2.0

    def __post_init__(self) -> None:
        if len(self.baseline_study_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.baseline_study_id
        ):
            raise PathVarianceSignalError("baseline_study_id must be lowercase SHA-256")
        all_lookbacks = (
            *self.range_lookback_minutes,
            *self.efficiency_lookback_minutes,
        )
        if not all_lookbacks or any(value < 2 or value > 14 for value in all_lookbacks):
            raise PathVarianceSignalError("lookbacks must be integers in [2, 14]")
        if len(set(self.range_lookback_minutes)) != len(
            self.range_lookback_minutes
        ) or len(set(self.efficiency_lookback_minutes)) != len(
            self.efficiency_lookback_minutes
        ):
            raise PathVarianceSignalError(
                "lookbacks must be unique within a signal type"
            )
        if not 0.5 < self.development_fraction < 0.8:
            raise PathVarianceSignalError("development_fraction must be in (0.5, 0.8)")
        if self.embargo_windows < 1:
            raise PathVarianceSignalError("at least one window must be embargoed")
        if self.ridge_penalty <= 0 or not math.isfinite(self.ridge_penalty):
            raise PathVarianceSignalError("ridge_penalty must be finite and positive")
        if self.minimum_development_windows < 10 or self.minimum_test_windows < 10:
            raise PathVarianceSignalError("minimum split sizes must be at least ten")
        if self.range_variance_constant <= 0 or self.variance_floor <= 0:
            raise PathVarianceSignalError("range constant and floor must be positive")


def _logit(probability: float) -> float:
    clipped = max(0.005, min(0.995, probability))
    return math.log(clipped / (1 - clipped))


def _feature_specs(config: PathVarianceSignalConfig) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for lookback in config.range_lookback_minutes:
        specs.append(
            {
                "signal_id": f"range_volatility_{lookback}m",
                "lookback": lookback,
                "kind": "conditional_variance",
                "expected_sign": "positive",
                "mechanism": "higher trailing range increases remaining variance and moves a digital probability toward 50 cents",
                "formula": (
                    "-logit(p0) * log(((log(high/low)^2 / (4*ln(2))) + floor) "
                    f"/ (baseline_variance_{lookback}m + floor))"
                ),
            }
        )
    for lookback in config.efficiency_lookback_minutes:
        specs.append(
            {
                "signal_id": f"path_efficiency_{lookback}m",
                "lookback": lookback,
                "kind": "conditional_mean",
                "expected_sign": "positive",
                "mechanism": "an efficient signed path is more likely to continue than a choppy path with the same elapsed time",
                "formula": (
                    f"standardized_log_return_{lookback}m * "
                    "abs(net_log_return)/sum(abs(one_minute_log_returns))"
                ),
            }
        )
    return specs


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Range-volatility and path-efficiency signal cards",
        "",
        "Signals are fitted separately against the same frozen Gaussian p0 and intercept-only control.",
        "",
        "| Signal | Coefficient | Δ Brier vs recalibration | Δ log loss vs recalibration | Probability gate | Accepted |",
        "|---|---:|---:|---:|---|---|",
    ]
    for card in report["signal_cards"]:
        improvement = card["out_of_sample_probability_improvement"]
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {} | {} |".format(
                card["signal_id"],
                card["fitted_model"]["coefficient"],
                improvement["delta_brier_vs_recalibrated_baseline"],
                improvement["delta_log_loss_vs_recalibrated_baseline"],
                "pass" if card["probability_gate_passed"] else "fail",
                "yes" if card["accepted_into_model"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "No joint model was fitted. Queue-aware net EV remains unavailable, so no signal can be accepted from this panel.",
            "",
        ]
    )
    return "\n".join(lines)


def run_path_variance_signal_study(
    storage_root: Path,
    pyth_proxy_candles_path: Path,
    config: PathVarianceSignalConfig,
) -> dict[str, Any]:
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
        raise PathVarianceSignalError("baseline is not eligible for signal development")
    baseline_rows_path = resolve_signal_input(
        root, Path(baseline["artifacts"]["probabilities"])
    )
    pyth_path = resolve_signal_input(root, pyth_proxy_candles_path)
    inputs = {
        "baseline_report": baseline_report_path,
        "baseline_probabilities": baseline_rows_path,
        "pyth_generic_proxy": pyth_path,
    }
    input_hashes = {name: signal_sha256_path(path) for name, path in inputs.items()}
    public_config = asdict(config)
    family_study_id = hashlib.sha256(
        canonical_signal_json(
            {
                "schema_version": SCHEMA_VERSION,
                "family_id": FAMILY_ID,
                "config": public_config,
                "inputs": input_hashes,
            }
        ).encode()
    ).hexdigest()
    run_id = f"path-variance-signals-{family_study_id[:24]}"
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
            "component": "path_variance_signal_cards",
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
        pyth_rows = read_signal_hash_chain(pyth_path)
        if any(row.get("settlement_eligible") is not False for row in pyth_rows):
            raise PathVarianceSignalError(
                "generic Pyth input must be settlement_eligible=false"
            )
        candles = {int(row["timestamp_seconds"]): row for row in pyth_rows}
        windows = sorted(
            {
                (int(row["window_open_ts_ns"]), str(row["market_ticker"]))
                for row in baseline_rows
            }
        )
        split_index = int(len(windows) * config.development_fraction)
        test_start = split_index + config.embargo_windows
        if split_index < config.minimum_development_windows:
            raise PathVarianceSignalError("too few development windows")
        if len(windows) - test_start < config.minimum_test_windows:
            raise PathVarianceSignalError("too few test windows")
        development_tickers = {ticker for _, ticker in windows[:split_index]}
        embargo_tickers = {ticker for _, ticker in windows[split_index:test_start]}
        test_tickers = {ticker for _, ticker in windows[test_start:]}
        specs = _feature_specs(config)

        augmented: list[dict[str, Any]] = []
        maximum_lookback = max(spec["lookback"] for spec in specs)
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
            decision_seconds = int(row["decision_ts_ns"]) // NSEC
            history = [
                candles.get(decision_seconds - 60 - offset * 60)
                for offset in range(maximum_lookback)
            ]
            preceding = candles.get(decision_seconds - 60 - maximum_lookback * 60)
            if preceding is None or any(candle is None for candle in history):
                continue
            chronological = list(reversed(history))
            full_variance = (
                float(row["forecast_variance_remaining"])
                * 900
                / int(row["remaining_seconds"])
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
                    "baseline_probability": float(row["fair_yes_gaussian"]),
                    "final_holdout": False,
                    "trading_eligible": False,
                }
            )
            for spec in specs:
                lookback = int(spec["lookback"])
                selected = chronological[-lookback:]
                expected_variance = full_variance * lookback / 15
                if spec["kind"] == "conditional_variance":
                    high = max(float(candle["high"]) for candle in selected)
                    low = min(float(candle["low"]) for candle in selected)
                    range_variance = (
                        math.log(high / low) ** 2 / config.range_variance_constant
                    )
                    log_ratio = math.log(
                        (range_variance + config.variance_floor)
                        / (expected_variance + config.variance_floor)
                    )
                    feature = -_logit(value["baseline_probability"]) * log_ratio
                    value[f"{spec['signal_id']}_raw_log_variance_ratio"] = log_ratio
                else:
                    prior_timestamp = decision_seconds - 60 - lookback * 60
                    prior = candles.get(prior_timestamp)
                    if prior is None:
                        raise PathVarianceSignalError(
                            "path-efficiency prior is missing"
                        )
                    levels = [float(prior["close"])] + [
                        float(candle["close"]) for candle in selected
                    ]
                    moves = [math.log(right / left) for left, right in pairwise(levels)]
                    absolute_path = sum(abs(move) for move in moves)
                    net = sum(moves)
                    efficiency = abs(net) / absolute_path if absolute_path else 0.0
                    feature = (
                        net / math.sqrt(max(expected_variance, 1e-18)) * efficiency
                    )
                    value[f"{spec['signal_id']}_efficiency"] = efficiency
                value[spec["signal_id"]] = feature
            augmented.append(value)

        development = [row for row in augmented if row["split"] == "development"]
        test = [row for row in augmented if row["split"] == "test"]
        models: dict[str, Any] = {}
        controls: dict[str, Any] = {}
        for spec in specs:
            signal_id = spec["signal_id"]
            controls[signal_id] = fit_offset_logistic(
                development,
                feature_field=signal_id,
                baseline_field="baseline_probability",
                ridge_penalty=config.ridge_penalty,
                include_feature=False,
            )
            models[signal_id] = fit_offset_logistic(
                development,
                feature_field=signal_id,
                baseline_field="baseline_probability",
                ridge_penalty=config.ridge_penalty,
                include_feature=True,
            )
        for row in augmented:
            for spec in specs:
                signal_id = spec["signal_id"]
                row[f"fair_yes_recalibrated_{signal_id}"] = controls[signal_id].predict(
                    row["baseline_probability"], row[signal_id]
                )
                row[f"fair_yes_signal_{signal_id}"] = models[signal_id].predict(
                    row["baseline_probability"], row[signal_id]
                )

        correlations: dict[str, Any] = {}
        for left_index, left in enumerate(specs):
            for right in specs[left_index + 1 :]:
                correlations[f"{left['signal_id']}_vs_{right['signal_id']}"] = (
                    signal_correlation(
                        [float(row[left["signal_id"]]) for row in test],
                        [float(row[right["signal_id"]]) for row in test],
                    )
                )
        baseline_metrics = signal_metrics(test, "baseline_probability")
        cards: list[dict[str, Any]] = []
        for spec in specs:
            signal_id = spec["signal_id"]
            control_field = f"fair_yes_recalibrated_{signal_id}"
            signal_field = f"fair_yes_signal_{signal_id}"
            control_metrics = signal_metrics(test, control_field)
            model_metrics = signal_metrics(test, signal_field)
            delta_brier = model_metrics["brier"] - control_metrics["brier"]
            delta_log_loss = model_metrics["log_loss"] - control_metrics["log_loss"]
            expected_sign_passed = models[signal_id].coefficient > 0
            probability_gate = (
                expected_sign_passed and delta_brier < 0 and delta_log_loss < 0
            )
            cards.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "family_id": FAMILY_ID,
                    "signal_id": signal_id,
                    "formula": spec["formula"],
                    "data_source_and_latency": {
                        "source": "Pyth Metal.XAG/USD generic one-minute OHLC candles",
                        "historical_alignment": "timestamp labels the preceding minute",
                        "live_availability_assumption": "complete OHLC is available only after minute close; latency unmeasured",
                    },
                    "expected_mechanism": spec["mechanism"],
                    "expected_distribution_component": spec["kind"],
                    "expected_sign": spec["expected_sign"],
                    "effective_horizon_and_decay": f"{spec['lookback']}-minute trailing state; expected to decay within one Kalshi window",
                    "time_of_day_adjustment": "range variance is divided by the baseline TOD/EWMA expectation; path return is divided by the same diffusion scale",
                    "fitted_model": models[signal_id].public_dict(),
                    "recalibration_control": controls[signal_id].public_dict(),
                    "correlation_with_existing_signals": {
                        key: value
                        for key, value in correlations.items()
                        if signal_id in key
                    },
                    "conditional_distribution_effects": signal_distribution_effects(
                        development,
                        test,
                        signal_id,
                        config.jump_threshold_standard_deviations,
                    ),
                    "out_of_sample_probability_improvement": {
                        "baseline": baseline_metrics,
                        "intercept_only_recalibration": control_metrics,
                        "signal": model_metrics,
                        "delta_brier_vs_raw_baseline": model_metrics["brier"]
                        - baseline_metrics["brier"],
                        "delta_log_loss_vs_raw_baseline": model_metrics["log_loss"]
                        - baseline_metrics["log_loss"],
                        "delta_brier_vs_recalibrated_baseline": delta_brier,
                        "delta_log_loss_vs_recalibrated_baseline": delta_log_loss,
                    },
                    "fair_value_change_cents": signal_fair_change_buckets(
                        test, signal_field
                    ),
                    "out_of_sample_net_ev_improvement": {
                        "status": "not_measured_no_queue_aware_execution_panel",
                        "net_cents_per_contract": None,
                    },
                    "expected_sign_passed": expected_sign_passed,
                    "probability_gate_passed": probability_gate,
                    "trading_ev_gate_passed": False,
                    "accepted_into_model": False,
                    "failure_conditions": [
                        "coefficient sign contradicts the frozen mechanism",
                        "Brier or log loss fails to improve versus intercept-only recalibration",
                        "effect is unstable across time remaining or volatility state",
                        "OHLC range is distorted by the generic proxy or candle construction",
                        "queue-aware net EV is not positive after fees and selection",
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
                "accepted_signal_ids": [],
                "family_added_to_model": False,
                "reason": "No signal can enter without both incremental probability improvement and measured positive queue-aware net EV.",
                "next_signal_family": "futures_volume_trade_intensity_and_price_impact",
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
        write_signal_json(cards_path, {"signal_cards": cards})
        write_signal_json(report_path, report)
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
