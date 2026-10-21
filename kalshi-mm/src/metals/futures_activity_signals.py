"""Independent COMEX volume, intensity, and price-impact signal cards."""

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
SCHEMA_VERSION = "metals-futures-activity-signal-family/v1"
FAMILY_ID = "futures_volume_trade_intensity_and_price_impact"


class FuturesActivitySignalError(ValueError):
    """Raised when COMEX signal research cannot satisfy its frozen contract."""


@dataclass(frozen=True)
class FuturesActivitySignalConfig:
    baseline_study_id: str
    asset: str = "SI"
    intensity_lookback_minutes: int = 1
    volume_lookback_minutes: int = 5
    price_impact_lookback_minutes: int = 5
    causal_reference_minutes: int = 15
    max_trade_age_seconds: int = 30
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
            raise FuturesActivitySignalError(
                "baseline_study_id must be lowercase SHA-256"
            )
        if self.asset.strip().upper() not in {"SI", "SIL", "SIC"}:
            raise FuturesActivitySignalError("asset must be SI, SIL, or SIC")
        lookbacks = (
            self.intensity_lookback_minutes,
            self.volume_lookback_minutes,
            self.price_impact_lookback_minutes,
        )
        if any(
            value < 1 or value >= self.causal_reference_minutes for value in lookbacks
        ):
            raise FuturesActivitySignalError(
                "lookbacks must be positive and shorter than the causal reference"
            )
        if self.causal_reference_minutes < 10:
            raise FuturesActivitySignalError(
                "causal reference must be at least ten minutes"
            )
        if self.max_trade_age_seconds < 1:
            raise FuturesActivitySignalError("max trade age must be positive")
        if not 0.5 < self.development_fraction < 0.8:
            raise FuturesActivitySignalError(
                "development_fraction must be in (0.5, 0.8)"
            )
        if self.embargo_windows < 1:
            raise FuturesActivitySignalError("at least one window must be embargoed")
        if not math.isfinite(self.ridge_penalty) or self.ridge_penalty <= 0:
            raise FuturesActivitySignalError(
                "ridge penalty must be finite and positive"
            )
        if self.minimum_development_windows < 10 or self.minimum_test_windows < 10:
            raise FuturesActivitySignalError("minimum split sizes must be at least ten")


@dataclass(frozen=True)
class TradeSeries:
    symbol: str
    timestamps_ns: tuple[int, ...]
    prices: tuple[float, ...]
    sizes: tuple[int, ...]
    prefix_count: tuple[int, ...]
    prefix_volume: tuple[int, ...]

    def interval(self, start_ns: int, end_ns: int) -> tuple[int, int]:
        start = bisect.bisect_left(self.timestamps_ns, start_ns)
        end = bisect.bisect_left(self.timestamps_ns, end_ns)
        return (
            self.prefix_count[end] - self.prefix_count[start],
            self.prefix_volume[end] - self.prefix_volume[start],
        )

    def asof(self, boundary_ns: int, max_age_ns: int) -> float | None:
        index = bisect.bisect_left(self.timestamps_ns, boundary_ns) - 1
        if index < 0 or boundary_ns - self.timestamps_ns[index] > max_age_ns:
            return None
        return self.prices[index]


def select_active_trade_series(
    trades: Sequence[Mapping[str, Any]], asset: str
) -> tuple[TradeSeries, dict[str, Any]]:
    normalized_asset = asset.strip().upper()
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"trades": 0, "volume": 0})
    for row in trades:
        if str(row.get("asset", "")).upper() != normalized_asset:
            continue
        symbol = str(row.get("contract_symbol", "")).strip()
        if not symbol:
            continue
        stats[symbol]["trades"] += 1
        stats[symbol]["volume"] += max(0, int(row.get("size") or 0))
    if not stats:
        raise FuturesActivitySignalError(f"no trades for asset {normalized_asset}")
    symbol = max(
        stats,
        key=lambda candidate: (
            stats[candidate]["volume"],
            stats[candidate]["trades"],
            candidate,
        ),
    )
    values: list[tuple[int, float, int]] = []
    for row in trades:
        if row.get("contract_symbol") != symbol:
            continue
        timestamp = int(row.get("source_event_ts_ns") or row.get("ts_event") or 0)
        price = float(row.get("price") or 0)
        size = max(0, int(row.get("size") or 0))
        if timestamp > 0 and price > 0:
            values.append((timestamp, price, size))
    values.sort(key=lambda value: value[0])
    if not values:
        raise FuturesActivitySignalError(
            f"selected contract has no valid trades: {symbol}"
        )
    prefix_count = [0]
    prefix_volume = [0]
    for _, _, size in values:
        prefix_count.append(prefix_count[-1] + 1)
        prefix_volume.append(prefix_volume[-1] + size)
    return (
        TradeSeries(
            symbol=symbol,
            timestamps_ns=tuple(value[0] for value in values),
            prices=tuple(value[1] for value in values),
            sizes=tuple(value[2] for value in values),
            prefix_count=tuple(prefix_count),
            prefix_volume=tuple(prefix_volume),
        ),
        {"asset": normalized_asset, "symbol": symbol, **stats[symbol]},
    )


def _logit(probability: float) -> float:
    clipped = max(0.005, min(0.995, probability))
    return math.log(clipped / (1 - clipped))


def _specs(config: FuturesActivitySignalConfig) -> list[dict[str, Any]]:
    return [
        {
            "signal_id": f"trade_intensity_{config.intensity_lookback_minutes}m",
            "kind": "conditional_variance",
            "expected_sign": "positive",
            "formula": "-logit(p0) * log((trailing trade count + 1)/(causal reference count scaled to the trailing horizon + 1))",
            "mechanism": "unusually intense futures trading raises near-term information arrival and variance",
        },
        {
            "signal_id": f"volume_state_{config.volume_lookback_minutes}m",
            "kind": "conditional_variance",
            "expected_sign": "positive",
            "formula": "-logit(p0) * log((trailing futures volume + 1)/(causal reference volume scaled to the trailing horizon + 1))",
            "mechanism": "unusually high futures volume raises the remaining variance of the settlement index",
        },
        {
            "signal_id": f"price_impact_{config.price_impact_lookback_minutes}m",
            "kind": "conditional_mean",
            "expected_sign": "positive",
            "formula": "baseline-standardized futures log return divided by sqrt(1 + trailing futures volume)",
            "mechanism": "a large signed price move per unit volume represents information rather than uninformed turnover and should continue into Pyth",
        },
    ]


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# COMEX activity signal cards",
        "",
        f"Outcome-blind active contract: **{report['selected_contract']['symbol']}**.",
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
            "Historical vendor timestamps support exploratory sequencing, not measured local live latency. No joint signal model was fitted.",
            "",
        ]
    )
    return "\n".join(lines)


def run_futures_activity_signal_study(
    storage_root: Path,
    databento_trades_path: Path,
    config: FuturesActivitySignalConfig,
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
        raise FuturesActivitySignalError(
            "baseline is not eligible for signal development"
        )
    baseline_rows_path = resolve_signal_input(
        root, Path(baseline["artifacts"]["probabilities"])
    )
    trades_path = resolve_signal_input(root, databento_trades_path)
    inputs = {
        "baseline_report": baseline_report_path,
        "baseline_probabilities": baseline_rows_path,
        "databento_trades": trades_path,
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
    run_id = f"futures-activity-signals-{family_study_id[:24]}"
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
            "component": "futures_activity_signal_cards",
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
        trades = read_signal_hash_chain(trades_path)
        series, selected_contract = select_active_trade_series(trades, config.asset)
        windows = sorted(
            {
                (int(row["window_open_ts_ns"]), str(row["market_ticker"]))
                for row in baseline_rows
            }
        )
        split_index = int(len(windows) * config.development_fraction)
        test_start = split_index + config.embargo_windows
        if split_index < config.minimum_development_windows:
            raise FuturesActivitySignalError("too few development windows")
        if len(windows) - test_start < config.minimum_test_windows:
            raise FuturesActivitySignalError("too few test windows")
        development_tickers = {ticker for _, ticker in windows[:split_index]}
        embargo_tickers = {ticker for _, ticker in windows[split_index:test_start]}
        test_tickers = {ticker for _, ticker in windows[test_start:]}
        specs = _specs(config)
        max_age_ns = config.max_trade_age_seconds * NSEC

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
            decision_ns = int(row["decision_ts_ns"])
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
                    "contract_symbol": series.symbol,
                    "baseline_probability": float(row["fair_yes_gaussian"]),
                    "final_holdout": False,
                    "trading_eligible": False,
                }
            )
            for kind, lookback in (
                ("intensity", config.intensity_lookback_minutes),
                ("volume", config.volume_lookback_minutes),
            ):
                trailing_start = decision_ns - lookback * 60 * NSEC
                count, volume = series.interval(trailing_start, decision_ns)
                reference_start = (
                    decision_ns - config.causal_reference_minutes * 60 * NSEC
                )
                reference_count, reference_volume = series.interval(
                    reference_start, trailing_start
                )
                reference_minutes = config.causal_reference_minutes - lookback
                observed = count if kind == "intensity" else volume
                reference = reference_count if kind == "intensity" else reference_volume
                expected = reference / reference_minutes * lookback
                log_ratio = math.log((observed + 1) / (expected + 1))
                signal_id = (
                    f"trade_intensity_{lookback}m"
                    if kind == "intensity"
                    else f"volume_state_{lookback}m"
                )
                value[signal_id] = -_logit(value["baseline_probability"]) * log_ratio
                value[f"{signal_id}_raw_log_ratio"] = log_ratio
            lookback = config.price_impact_lookback_minutes
            current_price = series.asof(decision_ns, max_age_ns)
            prior_price = series.asof(decision_ns - lookback * 60 * NSEC, max_age_ns)
            _, impact_volume = series.interval(
                decision_ns - lookback * 60 * NSEC, decision_ns
            )
            if current_price is not None and prior_price is not None:
                normalized_return = math.log(current_price / prior_price) / math.sqrt(
                    max(full_variance * lookback / 15, 1e-18)
                )
                value[f"price_impact_{lookback}m"] = normalized_return / math.sqrt(
                    1 + impact_volume
                )
            else:
                value[f"price_impact_{lookback}m"] = None
            augmented.append(value)

        models: dict[str, Any] = {}
        controls: dict[str, Any] = {}
        datasets: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for spec in specs:
            signal_id = spec["signal_id"]
            development = [
                row
                for row in augmented
                if row["split"] == "development" and row.get(signal_id) is not None
            ]
            test = [
                row
                for row in augmented
                if row["split"] == "test" and row.get(signal_id) is not None
            ]
            datasets[signal_id] = {"development": development, "test": test}
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
                if row.get(signal_id) is None:
                    row[f"fair_yes_recalibrated_{signal_id}"] = None
                    row[f"fair_yes_signal_{signal_id}"] = None
                    continue
                row[f"fair_yes_recalibrated_{signal_id}"] = controls[signal_id].predict(
                    row["baseline_probability"], row[signal_id]
                )
                row[f"fair_yes_signal_{signal_id}"] = models[signal_id].predict(
                    row["baseline_probability"], row[signal_id]
                )

        correlations: dict[str, Any] = {}
        for left_index, left in enumerate(specs):
            for right in specs[left_index + 1 :]:
                common = [
                    row
                    for row in augmented
                    if row["split"] == "test"
                    and row.get(left["signal_id"]) is not None
                    and row.get(right["signal_id"]) is not None
                ]
                correlations[f"{left['signal_id']}_vs_{right['signal_id']}"] = (
                    signal_correlation(
                        [float(row[left["signal_id"]]) for row in common],
                        [float(row[right["signal_id"]]) for row in common],
                    )
                )

        cards: list[dict[str, Any]] = []
        for spec in specs:
            signal_id = spec["signal_id"]
            development = datasets[signal_id]["development"]
            test = datasets[signal_id]["test"]
            control_field = f"fair_yes_recalibrated_{signal_id}"
            signal_field = f"fair_yes_signal_{signal_id}"
            baseline_metrics = signal_metrics(test, "baseline_probability")
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
                        "source": f"Databento GLBX.MDP3 trades for {series.symbol}",
                        "timestamp": "vendor source_event_ts_ns with strict-before-decision joins",
                        "live_availability_assumption": "historical vendor time is not local live receive time; latency remains unmeasured",
                    },
                    "expected_mechanism": spec["mechanism"],
                    "expected_distribution_component": spec["kind"],
                    "expected_sign": spec["expected_sign"],
                    "effective_horizon_and_decay": "one-to-five-minute trailing state; expected to decay within one Kalshi window",
                    "time_of_day_adjustment": f"ratio to the immediately preceding causal {config.causal_reference_minutes}-minute activity window; this is not a multi-day TOD curve",
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
                        "coefficient sign contradicts the frozen economic mechanism",
                        "Brier or log loss fails to improve versus intercept-only recalibration",
                        "contract roll or session choice changes the effect",
                        "historical vendor-time results do not survive measured live receive-time replay",
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
            "selected_contract": {
                **selected_contract,
                "selection_rule": "highest full-day volume within the predeclared asset family; outcome-blind",
            },
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
                "effective_test_utc_days": len(
                    {
                        row["utc_day"]
                        for dataset in datasets.values()
                        for row in dataset["test"]
                    }
                ),
            },
            "signal_correlations": correlations,
            "signal_cards": cards,
            "decision": {
                "accepted_signal_ids": [],
                "family_added_to_model": False,
                "reason": "No signal can enter without both incremental probability improvement and measured positive queue-aware net EV.",
                "next_signal_family": "realized_semivariance_and_jump_indicators",
            },
            "confidence_interval": {
                "block_unit": "UTC_trading_day",
                "effective_test_blocks": 1,
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
