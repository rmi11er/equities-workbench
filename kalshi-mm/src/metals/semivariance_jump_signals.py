"""Independent realized-semivariance and jump-indicator signal cards."""

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
SCHEMA_VERSION = "metals-semivariance-jump-signal-family/v1"
FAMILY_ID = "realized_semivariance_and_jump_indicators"


class SemivarianceJumpSignalError(ValueError):
    """Raised when this family cannot satisfy its frozen evaluation contract."""


@dataclass(frozen=True)
class SemivarianceJumpSignalConfig:
    baseline_study_id: str
    lookback_minutes: int = 10
    development_fraction: float = 0.67
    embargo_windows: int = 1
    ridge_penalty: float = 10.0
    minimum_development_windows: int = 40
    minimum_test_windows: int = 20
    variance_floor: float = 1e-12
    jump_threshold_standard_deviations: float = 2.0

    def __post_init__(self) -> None:
        if len(self.baseline_study_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.baseline_study_id
        ):
            raise SemivarianceJumpSignalError(
                "baseline_study_id must be lowercase SHA-256"
            )
        if not 3 <= self.lookback_minutes <= 14:
            raise SemivarianceJumpSignalError("lookback must be in [3, 14]")
        if not 0.5 < self.development_fraction < 0.8:
            raise SemivarianceJumpSignalError(
                "development_fraction must be in (0.5, 0.8)"
            )
        if self.embargo_windows < 1:
            raise SemivarianceJumpSignalError("at least one window must be embargoed")
        if self.ridge_penalty <= 0 or not math.isfinite(self.ridge_penalty):
            raise SemivarianceJumpSignalError(
                "ridge_penalty must be finite and positive"
            )
        if self.minimum_development_windows < 10 or self.minimum_test_windows < 10:
            raise SemivarianceJumpSignalError(
                "minimum split sizes must be at least ten"
            )
        if self.variance_floor <= 0 or self.jump_threshold_standard_deviations <= 0:
            raise SemivarianceJumpSignalError(
                "variance floor and jump threshold must be positive"
            )


def _logit(probability: float) -> float:
    clipped = max(0.005, min(0.995, probability))
    return math.log(clipped / (1 - clipped))


def _specs(config: SemivarianceJumpSignalConfig) -> list[dict[str, Any]]:
    lookback = config.lookback_minutes
    return [
        {
            "signal_id": f"semivariance_imbalance_{lookback}m",
            "kind": "conditional_mean_and_skew",
            "expected_sign": "positive",
            "formula": "(positive realized semivariance - negative realized semivariance)/(total realized semivariance + floor)",
            "mechanism": "dominant upside semivariance represents directional information that continues into settlement",
        },
        {
            "signal_id": f"jump_magnitude_{lookback}m",
            "kind": "conditional_variance_and_jump_probability",
            "expected_sign": "positive",
            "formula": "-logit(p0) * log(1 + maximum absolute one-minute return / baseline one-minute sigma)",
            "mechanism": "a recent large move increases remaining jump risk and pulls a digital fair toward 50 cents",
        },
        {
            "signal_id": f"jump_direction_{lookback}m",
            "kind": "conditional_mean_and_skew",
            "expected_sign": "positive",
            "formula": "signed largest absolute one-minute return / baseline one-minute sigma",
            "mechanism": "the direction of a recent jump carries information that continues into the settlement index",
        },
    ]


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Semivariance and jump signal cards",
        "",
        "Each indicator is fitted independently against the frozen Gaussian p0 and intercept-only control.",
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
            "No signal is accepted without both probability improvements, the frozen sign, and positive queue-aware net EV.",
            "",
        ]
    )
    return "\n".join(lines)


def run_semivariance_jump_signal_study(
    storage_root: Path,
    pyth_proxy_candles_path: Path,
    config: SemivarianceJumpSignalConfig,
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
        raise SemivarianceJumpSignalError(
            "baseline is not eligible for signal development"
        )
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
    run_id = f"semivariance-jump-signals-{family_study_id[:24]}"
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
            "component": "semivariance_jump_signal_cards",
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
            raise SemivarianceJumpSignalError(
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
        test_start = split_index + config.embargo_windows
        if split_index < config.minimum_development_windows:
            raise SemivarianceJumpSignalError("too few development windows")
        if len(windows) - test_start < config.minimum_test_windows:
            raise SemivarianceJumpSignalError("too few test windows")
        development_tickers = {ticker for _, ticker in windows[:split_index]}
        embargo_tickers = {ticker for _, ticker in windows[split_index:test_start]}
        test_tickers = {ticker for _, ticker in windows[test_start:]}
        specs = _specs(config)

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
            decision_seconds = int(row["decision_ts_ns"]) // NSEC
            levels = [
                proxy.get(decision_seconds - 60 - offset * 60)
                for offset in range(config.lookback_minutes + 1)
            ]
            if any(level is None for level in levels):
                continue
            levels = list(reversed(levels))
            moves = [math.log(right / left) for left, right in pairwise(levels)]
            positive = sum(move * move for move in moves if move > 0)
            negative = sum(move * move for move in moves if move < 0)
            total = positive + negative
            largest = max(moves, key=abs)
            full_variance = (
                float(row["forecast_variance_remaining"])
                * 900
                / int(row["remaining_seconds"])
            )
            one_minute_sigma = math.sqrt(max(full_variance / 15, 1e-18))
            semivariance_id = f"semivariance_imbalance_{config.lookback_minutes}m"
            magnitude_id = f"jump_magnitude_{config.lookback_minutes}m"
            direction_id = f"jump_direction_{config.lookback_minutes}m"
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
                    semivariance_id: (positive - negative)
                    / (total + config.variance_floor),
                    magnitude_id: -_logit(float(row["fair_yes_gaussian"]))
                    * math.log1p(abs(largest) / one_minute_sigma),
                    direction_id: largest / one_minute_sigma,
                    "positive_semivariance": positive,
                    "negative_semivariance": negative,
                    "largest_signed_one_minute_return": largest,
                    "final_holdout": False,
                    "trading_eligible": False,
                }
            )
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
                        "source": "Pyth Metal.XAG/USD generic one-minute candle closes",
                        "historical_alignment": "timestamp labels the preceding minute",
                        "live_availability_assumption": "complete close is available only after minute end; latency unmeasured",
                    },
                    "expected_mechanism": spec["mechanism"],
                    "expected_distribution_component": spec["kind"],
                    "expected_sign": spec["expected_sign"],
                    "effective_horizon_and_decay": f"{config.lookback_minutes}-minute state expected to decay within one Kalshi window",
                    "time_of_day_adjustment": "jump sizes use the baseline TOD/EWMA one-minute sigma; semivariance is normalized by its own total",
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
                        "coefficient sign contradicts the frozen continuation or variance mechanism",
                        "Brier or log loss fails to improve versus intercept-only recalibration",
                        "effect is unstable across session, time remaining, or volatility state",
                        "generic one-minute candles conceal intraminute jumps",
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
                "next_signal_family": "implied_volatility_level_and_term_structure",
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
