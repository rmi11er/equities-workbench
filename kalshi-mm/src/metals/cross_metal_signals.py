"""Independent cross-metal confirmation and residual-return signal cards.

The family deliberately fixes one lookback.  Gold confirmation and the
silver-on-gold residual are each fitted alone as offset-logistic adjustments to
the frozen Gaussian fair.  Cross-metal centering, scaling, and beta are learned
from development observations only.
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
SCHEMA_VERSION = "metals-cross-metal-signal-family/v1"
FAMILY_ID = "cross_metal_confirmation_and_residual_returns"


class CrossMetalSignalError(ValueError):
    """Raised when cross-metal inputs or chronology violate the frozen study."""


@dataclass(frozen=True)
class CrossMetalSignalConfig:
    baseline_study_id: str
    lookback_minutes: int = 5
    development_fraction: float = 0.67
    embargo_windows: int = 1
    ridge_penalty: float = 10.0
    minimum_development_windows: int = 40
    minimum_test_windows: int = 20
    jump_threshold_standard_deviations: float = 2.0

    def __post_init__(self) -> None:
        if len(self.baseline_study_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.baseline_study_id
        ):
            raise CrossMetalSignalError(
                "baseline_study_id must be lowercase SHA-256"
            )
        if not 1 <= self.lookback_minutes <= 14:
            raise CrossMetalSignalError("lookback must be in [1, 14]")
        if not 0.5 < self.development_fraction < 0.8:
            raise CrossMetalSignalError(
                "development_fraction must be in (0.5, 0.8)"
            )
        if self.embargo_windows < 1:
            raise CrossMetalSignalError("at least one window must be embargoed")
        if self.ridge_penalty <= 0 or not math.isfinite(self.ridge_penalty):
            raise CrossMetalSignalError(
                "ridge_penalty must be finite and positive"
            )
        if self.minimum_development_windows < 10 or self.minimum_test_windows < 10:
            raise CrossMetalSignalError("minimum split sizes must be at least ten")
        if self.jump_threshold_standard_deviations <= 0:
            raise CrossMetalSignalError("jump threshold must be positive")


@dataclass(frozen=True)
class CrossMetalNormalization:
    silver_mean: float
    gold_mean: float
    silver_std: float
    gold_std: float
    silver_on_gold_beta: float
    residual_std: float
    observations: int

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_cross_metal_normalization(
    pairs: Sequence[tuple[float, float]],
) -> CrossMetalNormalization:
    """Fit development-only centering, scales, and silver-on-gold beta."""

    if len(pairs) < 20:
        raise CrossMetalSignalError("at least 20 development return pairs are required")
    silver_mean = sum(pair[0] for pair in pairs) / len(pairs)
    gold_mean = sum(pair[1] for pair in pairs) / len(pairs)
    centered = [
        (silver - silver_mean, gold - gold_mean) for silver, gold in pairs
    ]
    silver_variance = sum(silver * silver for silver, _ in centered) / len(centered)
    gold_variance = sum(gold * gold for _, gold in centered) / len(centered)
    if silver_variance <= 0 or gold_variance <= 0:
        raise CrossMetalSignalError("development return variance must be positive")
    covariance = sum(silver * gold for silver, gold in centered) / len(centered)
    beta = covariance / gold_variance
    residuals = [silver - beta * gold for silver, gold in centered]
    residual_variance = sum(value * value for value in residuals) / len(residuals)
    if residual_variance <= 0:
        raise CrossMetalSignalError("development residual variance must be positive")
    return CrossMetalNormalization(
        silver_mean=silver_mean,
        gold_mean=gold_mean,
        silver_std=math.sqrt(silver_variance),
        gold_std=math.sqrt(gold_variance),
        silver_on_gold_beta=beta,
        residual_std=math.sqrt(residual_variance),
        observations=len(pairs),
    )


def _specs(config: CrossMetalSignalConfig) -> list[dict[str, str]]:
    lookback = config.lookback_minutes
    return [
        {
            "signal_id": f"gold_confirmation_return_{lookback}m",
            "formula": f"centered ln(XAU_t/XAU_t-{lookback}m) divided by development-only XAU scale",
            "mechanism": "a same-complex gold move confirms information relevant to the conditional mean of silver",
            "expected_sign": "positive",
        },
        {
            "signal_id": f"silver_residual_return_{lookback}m",
            "formula": f"centered XAG {lookback}m return minus development-only beta times centered XAU return, divided by residual scale",
            "mechanism": "silver-specific information not explained by the common precious-metals move continues into settlement",
            "expected_sign": "positive",
        },
    ]


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Cross-metal signal cards",
        "",
        "Gold confirmation and the silver-on-gold residual are fitted independently against the frozen Gaussian p0 and intercept-only control.",
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
            "This is a one-day proxy study. Time-of-day and multi-session stability cannot be established, and no signal is accepted without queue-aware net EV.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_proxy(rows: Sequence[Mapping[str, Any]], expected_metal: str) -> None:
    if any(row.get("settlement_eligible") is not False for row in rows):
        raise CrossMetalSignalError(
            "generic Pyth inputs must be settlement_eligible=false"
        )
    metals = {str(row.get("metal", "")).lower() for row in rows}
    if metals != {expected_metal}:
        raise CrossMetalSignalError(
            f"expected only {expected_metal} Pyth proxy rows, found {sorted(metals)}"
        )


def run_cross_metal_signal_study(
    storage_root: Path,
    silver_pyth_proxy_candles_path: Path,
    gold_pyth_proxy_candles_path: Path,
    config: CrossMetalSignalConfig,
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
        raise CrossMetalSignalError(
            "baseline is not eligible for incremental signal development"
        )
    baseline_rows_path = resolve_signal_input(
        root, Path(baseline["artifacts"]["probabilities"])
    )
    silver_path = resolve_signal_input(root, silver_pyth_proxy_candles_path)
    gold_path = resolve_signal_input(root, gold_pyth_proxy_candles_path)
    inputs = {
        "baseline_report": baseline_report_path,
        "baseline_probabilities": baseline_rows_path,
        "silver_pyth_generic_proxy": silver_path,
        "gold_pyth_generic_proxy": gold_path,
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
    run_id = f"cross-metal-signals-{family_study_id[:24]}"
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
            "component": "cross_metal_signal_cards",
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
        gold_rows = read_signal_hash_chain(gold_path)
        _validate_proxy(silver_rows, "silver")
        _validate_proxy(gold_rows, "gold")
        silver_proxy = {
            int(row["timestamp_seconds"]): float(row["close"]) for row in silver_rows
        }
        gold_proxy = {
            int(row["timestamp_seconds"]): float(row["close"]) for row in gold_rows
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
            raise CrossMetalSignalError("too few development windows")
        if len(windows) - test_start < config.minimum_test_windows:
            raise CrossMetalSignalError("too few test windows")
        development_tickers = {ticker for _, ticker in windows[:split_index]}
        embargo_tickers = {ticker for _, ticker in windows[split_index:test_start]}
        test_tickers = {ticker for _, ticker in windows[test_start:]}

        raw_rows: list[dict[str, Any]] = []
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
            current_seconds = decision_seconds - 60
            previous_seconds = current_seconds - config.lookback_minutes * 60
            silver_current = silver_proxy.get(current_seconds)
            silver_previous = silver_proxy.get(previous_seconds)
            gold_current = gold_proxy.get(current_seconds)
            gold_previous = gold_proxy.get(previous_seconds)
            if None in (silver_current, silver_previous, gold_current, gold_previous):
                continue
            if min(
                float(silver_current),
                float(silver_previous),
                float(gold_current),
                float(gold_previous),
            ) <= 0:
                raise CrossMetalSignalError("Pyth proxy levels must be positive")
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
                    "silver_return_raw": math.log(
                        float(silver_current) / float(silver_previous)
                    ),
                    "gold_return_raw": math.log(
                        float(gold_current) / float(gold_previous)
                    ),
                    "final_holdout": False,
                    "trading_eligible": False,
                }
            )
            raw_rows.append(value)

        development_raw = [row for row in raw_rows if row["split"] == "development"]
        normalization = fit_cross_metal_normalization(
            [
                (float(row["silver_return_raw"]), float(row["gold_return_raw"]))
                for row in development_raw
            ]
        )
        specs = _specs(config)
        gold_signal_id = specs[0]["signal_id"]
        residual_signal_id = specs[1]["signal_id"]
        augmented: list[dict[str, Any]] = []
        for row in raw_rows:
            centered_silver = (
                float(row["silver_return_raw"]) - normalization.silver_mean
            )
            centered_gold = float(row["gold_return_raw"]) - normalization.gold_mean
            row[gold_signal_id] = centered_gold / normalization.gold_std
            row[residual_signal_id] = (
                centered_silver
                - normalization.silver_on_gold_beta * centered_gold
            ) / normalization.residual_std
            augmented.append(row)

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
                row[f"fair_yes_recalibrated_{signal_id}"] = controls[
                    signal_id
                ].predict(row["baseline_probability"], row[signal_id])
                row[f"fair_yes_signal_{signal_id}"] = models[signal_id].predict(
                    row["baseline_probability"], row[signal_id]
                )

        correlation = signal_correlation(
            [float(row[gold_signal_id]) for row in test],
            [float(row[residual_signal_id]) for row in test],
        )
        correlations = {f"{gold_signal_id}_vs_{residual_signal_id}": correlation}
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
                        "source": "Pyth Metal.XAG/USD and Metal.XAU/USD generic one-minute candle closes",
                        "historical_alignment": "timestamp labels the preceding minute; strict one-minute availability lag is imposed",
                        "live_availability_assumption": "complete closes are available only after minute end; latency unmeasured",
                    },
                    "expected_mechanism": spec["mechanism"],
                    "expected_distribution_component": "conditional_mean",
                    "expected_sign": spec["expected_sign"],
                    "effective_horizon_and_decay": f"{config.lookback_minutes}-minute state expected to decay within one Kalshi window",
                    "time_of_day_adjustment": "development-only global centering and scaling; a stable intraday curve is not estimable from one proxy day",
                    "development_only_cross_metal_normalization": normalization.public_dict(),
                    "fitted_model": models[signal_id].public_dict(),
                    "recalibration_control": controls[signal_id].public_dict(),
                    "correlation_with_existing_signals": correlations,
                    "conditional_distribution_effects": signal_distribution_effects(
                        development,
                        test,
                        signal_id,
                        config.jump_threshold_standard_deviations,
                    ),
                    "stability": {
                        "metal": "not_estimable_single_silver_target",
                        "session": "not_estimable_single_utc_day",
                        "time_remaining": "descriptive_rows_available_but_not_inferentially_reliable",
                        "volatility_regime": "not_estimable_single_proxy_day",
                    },
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
                        "coefficient sign contradicts the frozen continuation mechanism",
                        "Brier or log loss fails to improve versus intercept-only recalibration",
                        "the development-only beta or effect is unstable across days and sessions",
                        "generic candle latency differs from the historical availability assumption",
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
            "development_only_cross_metal_normalization": normalization.public_dict(),
            "signal_correlations": correlations,
            "signal_cards": cards,
            "decision": {
                "accepted_signal_ids": [],
                "family_added_to_model": False,
                "reason": "No signal can enter without both incremental probability improvement and measured positive queue-aware net EV.",
                "next_signal_family": "scheduled_event_and_option_expiration_effects_when_data_available",
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
