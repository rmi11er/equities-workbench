"""Command-line entry point for read-only metals research operations."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import signal
from collections.abc import Sequence
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

from .access import (
    preflight_databento,
    preflight_databento_live,
    preflight_kalshi,
    preflight_pyth,
    preflight_pyth_lazer_live,
)
from .baseline_fair import BaselineFairConfig, run_baseline_fair_study
from .baseline_scorecard import BaselineScorecardConfig, run_baseline_scorecard
from .candles import backfill_kalshi_candlesticks
from .clock import ClockTelemetryCollector, preflight_clock
from .cme import DatabentoMetalsRecorder
from .config import load_metals_config, resolve_kalshi_credentials
from .cross_metal_signals import (
    CrossMetalSignalConfig,
    run_cross_metal_signal_study,
)
from .daily_gate_measurement import measure_and_persist_daily_gates
from .daily_policy_registry import (
    materialize_daily_policy_registries,
    registry_config_hash,
)
from .daily_validation import freeze_daily_validation, synchronized_day_progress
from .databento_history import (
    HISTORICAL_SCHEMAS,
    DatabentoHistoricalArchive,
    DatabentoHistoricalRequest,
    HistoricalArchiveResult,
    HistoricalCostEstimate,
)
from .discovery import discover_open_markets
from .export import build_duckdb_catalog, export_normalized_parquet
from .futures_activity_signals import (
    FuturesActivitySignalConfig,
    run_futures_activity_signal_study,
)
from .forward_shadow import (
    ForwardShadowSettings,
    KalshiForwardShadowRecorder,
)
from .historical_research_assumption import (
    HistoricalOperationAuthority,
    HistoricalResearchAssumptionError,
    configured_historical_assumption_path,
    materialize_historical_research_assumption,
    require_config_historical_operation,
    require_historical_research_assumption,
)
from .kalshi import KalshiReadClient
from .natural_target_signals import (
    NaturalTargetSignalConfig,
    run_natural_target_signal_study,
)
from .path_variance_signals import (
    PathVarianceSignalConfig,
    run_path_variance_signal_study,
)
from .provider_approval import (
    ProviderApprovalError,
    materialize_provider_approval,
    require_config_provider_operation,
)
from .pyth import backfill_pyth_history, backfill_pyth_proxy_history
from .pyth_exact_history import PythExactHistoryRequest, run_pyth_exact_history
from .pyth_lazer import PythLazerCollector
from .pyth_settlement_backfill import (
    create_pyth_settlement_backfill_plan,
    run_pyth_settlement_backfill,
    verify_pyth_settlement_backfill_plan,
)
from .recorder import MetalsRecorder, backfill_settled_markets
from .s1_feasibility import S1FeasibilityConfig, run_s1_feasibility
from .semivariance_jump_signals import (
    SemivarianceJumpSignalConfig,
    run_semivariance_jump_signal_study,
)
from .settlement_audit import (
    SettlementAuditSemantics,
    SettlementUniverse,
    run_local_settlement_audit,
    verify_settlement_audit,
)
from .settlement_proxy_pilot import (
    SettlementProxyPilotConfig,
    run_settlement_proxy_pilot,
)
from .spot_return_signals import (
    SpotReturnSignalConfig,
    run_spot_return_signal_study,
)
from .storage import update_run_manifest, write_run_manifest
from .synchronized import SynchronizedMetalsRecorder
from .toxicity_diagnostics import ToxicityConfig, run_toxicity_diagnostics


def _nonnegative_finite_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except Exception as error:
        raise argparse.ArgumentTypeError("must be a decimal amount") from error
    if not parsed.is_finite() or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative amount")
    return parsed


def _positive_finite_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a number of seconds") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and strictly positive")
    return parsed


def _positive_finite_decimal(value: str) -> Decimal:
    parsed = _nonnegative_finite_decimal(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be strictly positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Kalshi metals research")
    parser.add_argument(
        "--config", default="configs/metals.toml", help="Metals TOML configuration"
    )
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("discover", help="List current target markets")
    subparsers.add_parser(
        "check-access",
        help="Secret-safe alias for the authenticated Kalshi read preflight",
    )
    subparsers.add_parser(
        "preflight-kalshi",
        help="Check bounded authenticated Kalshi reads without account payload output",
    )
    subparsers.add_parser(
        "preflight-clock",
        help="Check local Chrony synchronization without changing the system clock",
    )
    provider_materialize = subparsers.add_parser(
        "materialize-provider-approval",
        help="Freeze reviewed provider license, entitlement, and timestamp evidence",
    )
    provider_materialize.add_argument(
        "--approval",
        type=Path,
        required=True,
        help="Reviewed provider approval JSON; evidence documents remain external",
    )
    provider_check = subparsers.add_parser(
        "check-provider-approval",
        help="Verify one configured provider operation without network access",
    )
    provider_check.add_argument(
        "--provider", choices=("pyth", "databento"), required=True
    )
    provider_check.add_argument("--operation", required=True)
    assumption_materialize = subparsers.add_parser(
        "materialize-historical-research-assumption",
        help="Freeze a research-only assumption for one historical operation",
    )
    assumption_materialize.add_argument(
        "--assumption",
        type=Path,
        required=True,
        help="Reviewed secret-free research assumption JSON",
    )
    assumption_check = subparsers.add_parser(
        "check-historical-research-assumption",
        help="Verify one configured research-only historical assumption offline",
    )
    assumption_check.add_argument(
        "--provider", choices=("pyth", "databento"), required=True
    )
    assumption_check.add_argument("--operation", required=True)

    pyth_preflight = subparsers.add_parser(
        "preflight-pyth",
        help="Check Pyth History and Pro feed access without collecting data",
    )
    pyth_preflight.add_argument("--metal", default="silver")
    pyth_preflight.add_argument(
        "--history-at-seconds",
        type=int,
        required=True,
        help="Start of the bounded five-minute History entitlement probe",
    )
    pyth_preflight.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="Per-request timeout, capped at 60 seconds",
    )
    pyth_live_preflight = subparsers.add_parser(
        "preflight-pyth-lazer-live",
        help="Bounded no-retention probe of every configured Lazer endpoint",
    )
    pyth_live_preflight.add_argument("--metal", default="silver")
    pyth_live_preflight.add_argument(
        "--duration-seconds",
        type=float,
        default=5.0,
        help="Per-endpoint probe duration, finite and capped at 30 seconds",
    )
    pyth_live_preflight.add_argument(
        "--allow-live-entitlement-probe",
        action="store_true",
        required=True,
        help="Acknowledge transient licensed live access; no messages are retained",
    )
    pyth_exact = subparsers.add_parser(
        "backfill-pyth-exact-history",
        help="Archive one bounded 200 ms Pyth price-range window",
    )
    pyth_exact.add_argument("--metal", default="silver")
    pyth_exact.add_argument(
        "--start-timestamp-us",
        type=int,
        required=True,
        help="Inclusive Unix-microsecond window start",
    )
    pyth_exact.add_argument(
        "--end-timestamp-us",
        type=int,
        required=True,
        help="Inclusive Unix-microsecond window end; at most 60 seconds later",
    )
    pyth_exact.add_argument("--limit", type=int, default=1000)
    pyth_exact.add_argument("--max-pages", type=int, default=1000)
    pyth_exact.add_argument("--timeout-seconds", type=float, default=30.0)

    pyth_plan = subparsers.add_parser(
        "plan-pyth-settlement-history",
        help="Offline plan of exact Pyth ranges for a verified settlement audit",
    )
    pyth_plan.add_argument("--metal", default="silver")
    pyth_plan.add_argument(
        "--audit",
        required=True,
        help="Verified content-addressed settlement audit directory or manifest",
    )
    pyth_plan.add_argument("--limit", type=int, default=1000)
    pyth_plan.add_argument(
        "--max-pages",
        type=int,
        default=2,
        help="Per-boundary page ceiling; 60 seconds at 200 ms normally needs one page",
    )

    pyth_bulk = subparsers.add_parser(
        "backfill-pyth-settlement-history",
        help="Sequentially resume a bounded number of planned exact Pyth ranges",
    )
    pyth_bulk.add_argument("--plan", required=True)
    pyth_bulk.add_argument("--max-boundaries", type=int, default=100)
    pyth_bulk.add_argument("--request-interval-seconds", type=float, default=0.1)
    pyth_bulk.add_argument("--timeout-seconds", type=float, default=30.0)
    pyth_bulk.add_argument(
        "--allow-network-download",
        action="store_true",
        help="Required acknowledgement that the command will call Pyth History",
    )

    databento_preflight = subparsers.add_parser(
        "preflight-databento",
        help="Check Databento historical entitlements using metadata only",
    )
    databento_preflight.add_argument("--metal", default="silver")
    databento_preflight.add_argument(
        "--probe-date",
        required=True,
        help="UTC date used only for configured-symbol resolution (YYYY-MM-DD)",
    )
    databento_live_preflight = subparsers.add_parser(
        "preflight-databento-live",
        help="Bounded no-retention MBO snapshot entitlement probe",
    )
    databento_live_preflight.add_argument("--metal", default="silver")
    databento_live_preflight.add_argument(
        "--duration-seconds",
        type=float,
        default=5.0,
        help="Probe duration, finite and capped at 30 seconds",
    )
    databento_live_preflight.add_argument(
        "--allow-live-entitlement-probe",
        action="store_true",
        required=True,
        help="Acknowledge transient licensed live access; no records are retained",
    )

    databento_estimate = subparsers.add_parser(
        "estimate-databento-history",
        help="Estimate a bounded CME metals history request without downloading",
    )
    databento_estimate.add_argument("--metal", default="silver")
    databento_estimate.add_argument(
        "--start",
        required=True,
        help="Inclusive ISO 8601 timestamp with offset, or Unix nanoseconds",
    )
    databento_estimate.add_argument(
        "--end",
        required=True,
        help="Exclusive ISO 8601 timestamp with offset, or Unix nanoseconds",
    )
    databento_estimate.add_argument(
        "--schema",
        action="append",
        default=None,
        help=(
            "Historical schema to estimate; repeat for a unique non-empty subset "
            "of configured schemas (default: every configured schema)"
        ),
    )

    databento_history = subparsers.add_parser(
        "backfill-databento-history",
        help="Cost-gated native DBN backfill for configured CME metal roots",
    )
    databento_history.add_argument("--metal", default="silver")
    databento_history.add_argument(
        "--start",
        required=True,
        help="Inclusive ISO 8601 timestamp with offset, or Unix nanoseconds",
    )
    databento_history.add_argument(
        "--end",
        required=True,
        help="Exclusive ISO 8601 timestamp with offset, or Unix nanoseconds",
    )
    databento_history.add_argument(
        "--schema",
        action="append",
        default=None,
        help=(
            "Historical schema to retrieve; repeat for a unique non-empty subset "
            "of configured schemas (default: every configured schema)"
        ),
    )
    databento_history.add_argument(
        "--transport",
        choices=("stream", "batch"),
        default="stream",
        help="Stream immediately, or submit/resume daily-split batch DBN jobs",
    )
    databento_history.add_argument(
        "--max-cost-usd",
        type=_nonnegative_finite_decimal,
        required=True,
        help="Refuse new retrieval when the preflight estimate exceeds this amount",
    )
    databento_history.add_argument(
        "--allow-billable-download",
        action="store_true",
        required=True,
        help="Explicit acknowledgement that a new provider request may be billable",
    )

    collect = subparsers.add_parser("collect", help="Collect live read-only data")
    collect.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Stop after this many seconds; zero means run until signaled",
    )

    cme = subparsers.add_parser(
        "collect-databento",
        help="Collect read-only CME GLBX.MDP3 data through Databento",
    )
    cme.add_argument("--metal", default="silver")
    cme.add_argument(
        "--duration-seconds",
        type=float,
        default=60.0,
        help="Stop after this many seconds; zero runs until signaled",
    )
    cme.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional deterministic record limit; zero means unlimited",
    )

    pyth_lazer = subparsers.add_parser(
        "collect-pyth-lazer",
        help="Collect bounded read-only Pyth Pro live prices",
    )
    pyth_lazer.add_argument("--metal", default="silver")
    pyth_lazer.add_argument(
        "--duration-seconds",
        type=_positive_finite_seconds,
        default=60.0,
        help="Strictly positive collection duration (default: 60 seconds)",
    )

    synchronized = subparsers.add_parser(
        "collect-synchronized",
        help="Collect the read-only Kalshi/Pyth/Databento research panel",
    )
    synchronized.add_argument("--metal", default="silver")
    synchronized.add_argument(
        "--duration-seconds",
        type=_positive_finite_seconds,
        default=60.0,
        help="Strictly positive coordinated duration (default: 60 seconds)",
    )
    forward_shadow = subparsers.add_parser(
        "run-forward-shadow",
        help=(
            "Run the exchange-isolated Kalshi/Pyth fair and counterfactual quote "
            "loop while retaining live research data"
        ),
    )
    forward_shadow.add_argument(
        "--baseline-model",
        type=Path,
        required=True,
        help="Frozen baseline model.json artifact",
    )
    forward_shadow.add_argument(
        "--duration-seconds",
        type=_positive_finite_seconds,
        default=60.0,
    )
    forward_shadow.add_argument(
        "--decision-interval-ms", type=int, default=250
    )
    forward_shadow.add_argument("--max-pyth-age-ms", type=int, default=2_000)
    forward_shadow.add_argument("--min-seconds-remaining", type=int, default=15)
    forward_shadow.add_argument(
        "--quantity", type=_positive_finite_decimal, default=Decimal("1")
    )
    forward_shadow.add_argument(
        "--uncertainty-charge",
        type=_nonnegative_finite_decimal,
        default=Decimal("0.02"),
    )
    forward_shadow.add_argument(
        "--adverse-selection-charge",
        type=_nonnegative_finite_decimal,
        default=Decimal("0.01"),
    )
    forward_shadow.add_argument(
        "--fee-charge-per-contract",
        type=_nonnegative_finite_decimal,
        default=Decimal("0.01"),
    )
    forward_shadow.add_argument(
        "--inventory-gamma",
        type=_nonnegative_finite_decimal,
        default=Decimal("0.001"),
    )
    forward_shadow.add_argument(
        "--inventory-charge-cap",
        type=_nonnegative_finite_decimal,
        default=Decimal("0.05"),
    )
    forward_shadow.add_argument(
        "--allow-live-shadow-data",
        action="store_true",
        help="Acknowledge bounded live read-only collection; never enables orders",
    )
    clock = subparsers.add_parser(
        "collect-clock-telemetry",
        help="Collect bounded read-only local Chrony clock evidence",
    )
    clock.add_argument("--metal", default="silver")
    clock.add_argument(
        "--duration-seconds",
        type=_positive_finite_seconds,
        default=60.0,
        help="Strictly positive collection duration (default: 60 seconds)",
    )

    daily_freeze = subparsers.add_parser(
        "freeze-daily-validation",
        help="Checksum and freeze one UTC day's unified quality inputs",
    )
    daily_freeze.add_argument("--date", required=True, help="UTC date (YYYY-MM-DD)")
    daily_freeze.add_argument(
        "--quality-report",
        type=Path,
        required=True,
        help="Unified daily quality report below the configured storage root",
    )
    daily_freeze.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Completed run manifest to freeze; repeat for every source/run",
    )
    daily_measure = subparsers.add_parser(
        "measure-daily-gates",
        help="Derive and persist evidence-backed daily quality measurements",
    )
    daily_measure.add_argument("--date", required=True, help="UTC date (YYYY-MM-DD)")
    daily_measure.add_argument(
        "--quality-report",
        type=Path,
        required=True,
        help="Collector quality report listed by a completed input manifest",
    )
    daily_measure.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="Frozen daily gate policy listed by a completed input manifest",
    )
    daily_measure.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Completed input manifest; repeat for every contributing run",
    )
    registry_hash = subparsers.add_parser(
        "hash-registry-config",
        help="Print the canonical config_hash for one reviewed registry JSON object",
    )
    registry_hash.add_argument("--record", type=Path, required=True)
    daily_policy = subparsers.add_parser(
        "materialize-daily-policy-registries",
        help="Persist one reviewed daily policy and its fee/incentive registries",
    )
    daily_policy.add_argument("--date", required=True, help="UTC date (YYYY-MM-DD)")
    daily_policy.add_argument("--policy", type=Path, required=True)
    daily_policy.add_argument(
        "--fee-registry",
        type=Path,
        required=True,
        help="JSON array of reviewed fee_schedule_dim records",
    )
    daily_policy.add_argument(
        "--incentive-registry",
        type=Path,
        required=True,
        help="JSON array of reviewed incentive_program_dim records",
    )
    settlement_audit = subparsers.add_parser(
        "audit-settlements",
        help="Reconstruct local Kalshi settlements from verified exact Pyth history",
    )
    settlement_audit.add_argument("--metal", default="silver")
    settlement_audit.add_argument(
        "--semantics",
        type=Path,
        required=True,
        help="Reviewed v2 settlement-semantics JSON object",
    )
    settlement_audit.add_argument(
        "--universe",
        type=Path,
        help="Optional reviewed completed-market universe JSON object",
    )
    daily_progress = subparsers.add_parser(
        "check-data-readiness",
        help="Verify immutable daily snapshots and report burn-in progress",
    )
    daily_progress.add_argument(
        "--minimum-complete-days",
        type=int,
        default=10,
        help="Required count of passing synchronized UTC days (default: 10)",
    )
    daily_progress.add_argument(
        "--minimum-settlements",
        type=int,
        default=1000,
        help="Required aggregate exact settlement checks (default: 1000)",
    )

    backfill = subparsers.add_parser(
        "backfill-kalshi", help="Archive settled market results and trades"
    )
    backfill.add_argument("--metal", default="silver")
    backfill.add_argument("--max-markets", type=int, default=0)
    backfill.add_argument("--skip-trades", action="store_true")
    backfill.add_argument("--from-close-seconds", type=int)
    backfill.add_argument("--to-close-seconds", type=int)
    backfill.add_argument("--market-stride", type=int, default=1)
    backfill.add_argument("--market-stride-offset", type=int, default=0)
    candle_backfill = subparsers.add_parser(
        "backfill-kalshi-candles",
        help="Archive one-minute Kalshi bid/ask candlesticks for local markets",
    )
    candle_backfill.add_argument("--metal", default="silver")
    candle_backfill.add_argument("--max-markets", type=int, default=0)
    candle_backfill.add_argument("--no-resume", action="store_true")
    subparsers.add_parser(
        "export-parquet", help="Export normalized segments and refresh DuckDB views"
    )
    pyth = subparsers.add_parser(
        "backfill-pyth-history", help="Archive exact Pyth Pro one-minute candles"
    )
    pyth.add_argument("--metal", default="silver")
    pyth.add_argument("--from-seconds", type=int, required=True)
    pyth.add_argument("--to-seconds", type=int, required=True)
    pyth_proxy = subparsers.add_parser(
        "backfill-pyth-proxy-history",
        help="Archive generic XAG/XAU one-minute candles for proxy research only",
    )
    pyth_proxy.add_argument("--metal", choices=("silver", "gold"), default="silver")
    pyth_proxy.add_argument("--from-seconds", type=int, required=True)
    pyth_proxy.add_argument("--to-seconds", type=int, required=True)
    s1 = subparsers.add_parser(
        "run-s1-feasibility",
        help="Run the preliminary offline S1 one-minute candle study",
    )
    s1.add_argument("--metal", default="silver")
    s1.add_argument("--train-fraction", type=Decimal, default=Decimal("0.70"))
    s1.add_argument("--min-underdog-mid", type=Decimal, default=Decimal("0.07"))
    s1.add_argument("--max-underdog-mid", type=Decimal, default=Decimal("0.31"))
    s1.add_argument("--min-seconds-to-close", type=int, default=120)
    s1.add_argument("--penetration-dollars", type=Decimal, default=Decimal("0.01"))
    s1.add_argument("--maker-fee-per-contract", type=Decimal, default=Decimal("0"))
    s1.add_argument("--taker-fee-multiplier", type=Decimal, default=Decimal("0.07"))
    toxicity = subparsers.add_parser(
        "run-toxicity-diagnostics",
        help="Run exploratory Kalshi-only two-sided BBO toxicity cuts",
    )
    toxicity.add_argument("--metal", default="silver")
    toxicity.add_argument(
        "--test-start-ts-ns", type=int, default=1_787_301_900_000_000_000
    )
    toxicity.add_argument(
        "--penetration-dollars", type=Decimal, default=Decimal("0.01")
    )
    toxicity.add_argument(
        "--maker-fee-per-contract", type=Decimal, default=Decimal("0")
    )
    toxicity.add_argument(
        "--require-complete-trades",
        action="store_true",
        help="Restrict candidates to markets with a completed trade backfill marker",
    )
    proxy_pilot = subparsers.add_parser(
        "run-settlement-proxy-pilot",
        help="Compare a bounded Kalshi cohort with COMEX and generic Pyth proxies",
    )
    proxy_pilot.add_argument("--metal", choices=("silver", "gold"), default="silver")
    proxy_pilot.add_argument("--labels", type=Path, required=True)
    proxy_pilot.add_argument("--databento-trades", type=Path, required=True)
    proxy_pilot.add_argument("--pyth-proxy-candles", type=Path, required=True)
    proxy_pilot.add_argument("--start-ts-ns", type=int, required=True)
    proxy_pilot.add_argument("--end-ts-ns", type=int, required=True)
    proxy_pilot.add_argument("--primary-asset", default="SI")
    proxy_pilot.add_argument("--secondary-asset", default="SIL")
    proxy_pilot.add_argument("--max-trade-age-seconds", type=int, default=5)
    proxy_pilot.add_argument("--vwap-lookback-seconds", type=int, default=1)
    proxy_pilot.add_argument(
        "--ambiguity-dollars", type=Decimal, default=Decimal("0.005")
    )
    proxy_pilot.add_argument(
        "--train-fraction", type=Decimal, default=Decimal("0.67")
    )
    baseline_fair = subparsers.add_parser(
        "run-baseline-fair",
        help="Fit the research-only basic volatility fair and score a proxy day",
    )
    baseline_fair.add_argument("--metal", choices=("silver", "gold"), default="silver")
    baseline_fair.add_argument("--labels", type=Path, required=True)
    baseline_fair.add_argument("--pyth-proxy-candles", type=Path, required=True)
    baseline_fair.add_argument("--training-end-ts-ns", type=int, required=True)
    baseline_fair.add_argument("--validation-start-ts-ns", type=int, required=True)
    baseline_fair.add_argument("--validation-end-ts-ns", type=int, required=True)
    baseline_fair.add_argument("--formal-research-design-id", required=True)
    baseline_fair.add_argument("--prerequisite-proxy-pilot-id", required=True)
    baseline_fair.add_argument("--ewma-lambda", type=float, default=0.97)
    baseline_fair.add_argument("--tod-prior-observations", type=int, default=20)
    baseline_fair.add_argument("--winsor-quantile", type=float, default=0.995)
    spot_signals = subparsers.add_parser(
        "run-spot-return-signals",
        help="Test each fast/slow spot-return signal independently against p0",
    )
    spot_signals.add_argument("--baseline-study-id", required=True)
    spot_signals.add_argument("--pyth-proxy-candles", type=Path, required=True)
    spot_signals.add_argument(
        "--lookback-minutes", type=int, action="append", default=None
    )
    spot_signals.add_argument("--development-fraction", type=float, default=0.67)
    spot_signals.add_argument("--embargo-windows", type=int, default=1)
    spot_signals.add_argument("--ridge-penalty", type=float, default=10.0)
    path_variance = subparsers.add_parser(
        "run-path-variance-signals",
        help="Test range-volatility and path-efficiency signals independently",
    )
    path_variance.add_argument("--baseline-study-id", required=True)
    path_variance.add_argument("--pyth-proxy-candles", type=Path, required=True)
    path_variance.add_argument(
        "--range-lookback-minutes", type=int, action="append", default=None
    )
    path_variance.add_argument(
        "--efficiency-lookback-minutes", type=int, action="append", default=None
    )
    path_variance.add_argument("--development-fraction", type=float, default=0.67)
    path_variance.add_argument("--embargo-windows", type=int, default=1)
    path_variance.add_argument("--ridge-penalty", type=float, default=10.0)
    futures_activity = subparsers.add_parser(
        "run-futures-activity-signals",
        help="Test COMEX volume, intensity, and price-impact signals independently",
    )
    futures_activity.add_argument("--baseline-study-id", required=True)
    futures_activity.add_argument("--databento-trades", type=Path, required=True)
    futures_activity.add_argument("--asset", choices=("SI", "SIL", "SIC"), default="SI")
    futures_activity.add_argument("--development-fraction", type=float, default=0.67)
    futures_activity.add_argument("--embargo-windows", type=int, default=1)
    futures_activity.add_argument("--ridge-penalty", type=float, default=10.0)
    futures_activity.add_argument("--max-trade-age-seconds", type=int, default=30)
    semivariance_jump = subparsers.add_parser(
        "run-semivariance-jump-signals",
        help="Test realized-semivariance and jump indicators independently",
    )
    semivariance_jump.add_argument("--baseline-study-id", required=True)
    semivariance_jump.add_argument("--pyth-proxy-candles", type=Path, required=True)
    semivariance_jump.add_argument("--lookback-minutes", type=int, default=10)
    semivariance_jump.add_argument("--development-fraction", type=float, default=0.67)
    semivariance_jump.add_argument("--embargo-windows", type=int, default=1)
    semivariance_jump.add_argument("--ridge-penalty", type=float, default=10.0)
    cross_metal = subparsers.add_parser(
        "run-cross-metal-signals",
        help="Test XAU confirmation and XAG-on-XAU residual signals independently",
    )
    cross_metal.add_argument("--baseline-study-id", required=True)
    cross_metal.add_argument("--silver-pyth-proxy-candles", type=Path, required=True)
    cross_metal.add_argument("--gold-pyth-proxy-candles", type=Path, required=True)
    cross_metal.add_argument("--lookback-minutes", type=int, default=5)
    cross_metal.add_argument("--development-fraction", type=float, default=0.67)
    cross_metal.add_argument("--embargo-windows", type=int, default=1)
    cross_metal.add_argument("--ridge-penalty", type=float, default=10.0)
    baseline_scorecard = subparsers.add_parser(
        "run-baseline-scorecard",
        help="Score frozen fair comparators, including historical Kalshi midpoint",
    )
    baseline_scorecard.add_argument("--baseline-study-id", required=True)
    baseline_scorecard.add_argument(
        "--kalshi-candlesticks", type=Path, required=True
    )
    natural_targets = subparsers.add_parser(
        "run-natural-target-signals",
        help="Test each exploratory signal against its natural research target",
    )
    natural_targets.add_argument("--baseline-study-id", required=True)
    natural_targets.add_argument(
        "--silver-pyth-proxy-candles", type=Path, required=True
    )
    natural_targets.add_argument("--gold-pyth-proxy-candles", type=Path)
    natural_targets.add_argument("--databento-trades", type=Path)
    natural_targets.add_argument("--development-fraction", type=float, default=0.67)
    natural_targets.add_argument("--embargo-windows", type=int, default=1)
    natural_targets.add_argument("--ridge-penalty", type=float, default=10.0)
    natural_targets.add_argument(
        "--walk-forward-bootstrap-samples", type=int, default=2000
    )
    natural_targets.add_argument(
        "--walk-forward-bootstrap-seed", type=int, default=17290
    )
    return parser


async def _discover(config_path: str) -> int:
    config = load_metals_config(config_path)
    try:
        credentials = resolve_kalshi_credentials(config.kalshi)
    except (ValueError, FileNotFoundError):
        credentials = None
    async with KalshiReadClient(config.kalshi, credentials) as client:
        markets, _ = await discover_open_markets(client, config)
    for item in markets:
        market = item.raw
        print(
            json.dumps(
                {
                    "metal": item.metal,
                    "ticker": item.ticker,
                    "event_ticker": market.get("event_ticker"),
                    "series_ticker": market.get("series_ticker"),
                    "status": market.get("status"),
                    "open_time": market.get("open_time"),
                    "close_time": market.get("close_time"),
                    "floor_strike": market.get("floor_strike"),
                    "strike_type": market.get("strike_type"),
                    "price_level_structure": market.get("price_level_structure"),
                    "price_ranges": market.get("price_ranges"),
                },
                sort_keys=True,
                default=str,
            )
        )
    print("discovered={}".format(len(markets)))
    return 0


async def _check_access(config_path: str) -> int:
    config = load_metals_config(config_path)
    report = await preflight_kalshi(config)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 2


async def _collect(config_path: str, duration_seconds: float) -> int:
    config = load_metals_config(config_path)
    recorder = MetalsRecorder(config)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, recorder.stop)
        except NotImplementedError:  # pragma: no cover - Windows
            pass
    await recorder.run(duration_seconds if duration_seconds > 0 else None)
    print("run_id={}".format(recorder.run_id))
    for path in recorder.raw_store.paths + recorder.normalized_store.paths:
        print(path)
    return 0


async def _collect_databento(
    config_path: str,
    metal: str,
    duration_seconds: float,
    max_records: int,
) -> int:
    config = load_metals_config(config_path)
    recorder = DatabentoMetalsRecorder(
        config,
        metal=metal,
        max_records=max_records if max_records > 0 else None,
    )
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, recorder.stop)
        except NotImplementedError:  # pragma: no cover - Windows
            pass
    counts = await recorder.run(
        duration_seconds=duration_seconds if duration_seconds > 0 else None
    )
    print(json.dumps({**counts, "run_id": recorder.run_id}, sort_keys=True))
    return 0


async def _collect_pyth_lazer(
    config_path: str,
    metal: str,
    duration_seconds: float,
) -> int:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Pyth Lazer duration must be finite and strictly positive")
    config = load_metals_config(config_path)
    if not config.pyth.enabled:
        raise ValueError("Pyth Lazer live collection requires [pyth].enabled = true")
    collector = PythLazerCollector(config, metal)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, collector.stop)
        except NotImplementedError:  # pragma: no cover - Windows
            pass
    counts = await collector.run(duration_seconds=duration_seconds)
    print(json.dumps({**counts, "run_id": collector.run_id}, sort_keys=True))
    return 0


async def _collect_clock_telemetry(
    config_path: str,
    metal: str,
    duration_seconds: float,
) -> int:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError(
            "Clock telemetry duration must be finite and strictly positive"
        )
    config = load_metals_config(config_path)
    collector = ClockTelemetryCollector(config, metal)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, collector.stop)
        except NotImplementedError:  # pragma: no cover - Windows
            pass
    counts = await collector.run(duration_seconds=duration_seconds)
    print(json.dumps({**counts, "run_id": collector.run_id}, sort_keys=True))
    return 0


async def _backfill_pyth_exact_history(
    config_path: str,
    metal: str,
    start_timestamp_us: int,
    end_timestamp_us: int,
    limit: int,
    max_pages: int,
    timeout_seconds: float,
) -> int:
    config = load_metals_config(config_path)
    normalized_metal = metal.strip().lower()
    product = config.products.get(normalized_metal)
    if product is None or not product.enabled:
        raise ValueError("Exact Pyth history requires an enabled metals product")
    if product.pyth_lazer_id is None:
        raise ValueError("Exact Pyth history requires a numeric pyth_lazer_id")
    request = PythExactHistoryRequest(
        metal=normalized_metal,
        feed_ids=(product.pyth_lazer_id,),
        start_timestamp_us=start_timestamp_us,
        end_timestamp_us=end_timestamp_us,
        limit=limit,
        max_pages=max_pages,
        base_url=config.pyth.endpoint,
    )
    result = await run_pyth_exact_history(
        Path(config.storage.root),
        request,
        api_key_env=config.pyth.api_key_env,
        timeout_seconds=timeout_seconds,
        provider_approval_path=config.governance.provider_approval_path,
        historical_research_assumption_path=(
            config.governance.pyth_historical_research_assumption_path
        ),
    )
    print(json.dumps(result.public_dict(), sort_keys=True))
    return 0


def _plan_pyth_settlement_history(
    config_path: str,
    metal: str,
    audit_path: str,
    limit: int,
    max_pages: int,
) -> int:
    config = load_metals_config(config_path)
    normalized_metal = metal.strip().lower()
    product = config.products.get(normalized_metal)
    if product is None or not product.enabled:
        raise ValueError("Exact Pyth planning requires an enabled metals product")
    if product.pyth_lazer_id is None:
        raise ValueError("Exact Pyth planning requires a numeric pyth_lazer_id")
    artifact = create_pyth_settlement_backfill_plan(
        Path(config.storage.root),
        Path(audit_path),
        metal=normalized_metal,
        feed_id=product.pyth_lazer_id,
        base_url=config.pyth.endpoint,
        limit=limit,
        max_pages=max_pages,
    )
    print(json.dumps(artifact.public_dict(), sort_keys=True))
    return 0


async def _backfill_pyth_settlement_history(
    config_path: str,
    plan_path: str,
    max_boundaries: int,
    request_interval_seconds: float,
    timeout_seconds: float,
    allow_network_download: bool,
) -> int:
    config = load_metals_config(config_path)
    root = Path(config.storage.root)
    plan = verify_pyth_settlement_backfill_plan(root, Path(plan_path))
    metal = str(plan.value["metal"])
    product = config.products.get(metal)
    if product is None or not product.enabled:
        raise ValueError("Exact Pyth backfill requires an enabled plan product")
    if product.pyth_lazer_id != plan.value["feed_id"]:
        raise ValueError(
            "Configured numeric Pyth feed ID does not match the frozen plan"
        )
    if config.pyth.endpoint != plan.value["request_parameters"]["base_url"]:
        raise ValueError("Configured Pyth endpoint does not match the frozen plan")
    result = await run_pyth_settlement_backfill(
        root,
        plan.plan_path,
        api_key_env=config.pyth.api_key_env,
        allow_network_download=allow_network_download,
        max_boundaries=max_boundaries,
        request_interval_seconds=request_interval_seconds,
        timeout_seconds=timeout_seconds,
        provider_approval_path=config.governance.provider_approval_path,
        historical_research_assumption_path=(
            config.governance.pyth_historical_research_assumption_path
        ),
    )
    print(json.dumps(result.public_dict(), sort_keys=True))
    return 0


async def _collect_synchronized(
    config_path: str,
    metal: str,
    duration_seconds: float,
) -> int:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError(
            "Synchronized collection duration must be finite and strictly positive"
        )
    config = load_metals_config(config_path)
    recorder = SynchronizedMetalsRecorder(config, metal)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, recorder.stop)
        except NotImplementedError:  # pragma: no cover - Windows
            pass
    result = await recorder.run(duration_seconds=duration_seconds)
    print(json.dumps(result, sort_keys=True))
    return 0


async def _run_forward_shadow(args: argparse.Namespace) -> int:
    if args.allow_live_shadow_data is not True:
        raise ValueError("--allow-live-shadow-data is required")
    config = load_metals_config(args.config)
    if config.shadow.execution_mode != "shadow" or config.shadow.allow_live_orders:
        raise ValueError("forward runner requires exchange-isolated shadow mode")
    settings = ForwardShadowSettings(
        decision_interval_ms=args.decision_interval_ms,
        max_pyth_age_ms=args.max_pyth_age_ms,
        min_seconds_remaining=args.min_seconds_remaining,
        quantity_fp=args.quantity,
        uncertainty_charge=args.uncertainty_charge,
        adverse_selection_charge=args.adverse_selection_charge,
        fee_charge_per_contract=args.fee_charge_per_contract,
        inventory_gamma=args.inventory_gamma,
        inventory_charge_cap=args.inventory_charge_cap,
    )
    recorder = KalshiForwardShadowRecorder(
        config,
        baseline_model_path=args.baseline_model,
        settings=settings,
    )
    result = await recorder.run(args.duration_seconds)
    print(json.dumps(result, sort_keys=True))
    return 0


def _freeze_daily_validation(
    config_path: str,
    date: str,
    quality_report: Path,
    manifests: list[Path],
) -> int:
    config = load_metals_config(config_path)
    path = freeze_daily_validation(
        Path(config.storage.root),
        date,
        quality_report,
        manifests,
    )
    print(json.dumps({"snapshot": str(path)}, sort_keys=True))
    return 0


def _measure_daily_gates(
    config_path: str,
    date: str,
    quality_report: Path,
    policy: Path,
    manifests: list[Path],
) -> int:
    config = load_metals_config(config_path)
    artifacts = measure_and_persist_daily_gates(
        Path(config.storage.root),
        date,
        quality_report,
        manifests,
        policy,
    )
    print(
        json.dumps(
            {
                "evidence": str(artifacts.evidence_path),
                "quality_report": str(artifacts.quality_report_path),
                "manifest": str(artifacts.manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


def _load_cli_json(path: Path, label: str) -> Any:
    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains a duplicate JSON field")
            value[key] = item
        return value

    def reject_constant(_: str) -> None:
        raise ValueError(f"{label} contains a non-finite JSON number")

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=strict_object,
                parse_constant=reject_constant,
            )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid readable JSON") from error


def _audit_settlements(
    config_path: str,
    metal: str,
    semantics_path: Path,
    universe_path: Path | None,
) -> int:
    config = load_metals_config(config_path)
    normalized_metal = metal.strip().lower()
    product = config.products.get(normalized_metal)
    if product is None or not product.enabled:
        raise ValueError("Exact settlement audit requires an enabled metals product")
    if product.pyth_lazer_id is None:
        raise ValueError("Exact settlement audit requires a numeric pyth_lazer_id")

    semantics_value = _load_cli_json(semantics_path, "settlement semantics")
    if not isinstance(semantics_value, dict):
        raise ValueError("settlement semantics must be a JSON object")
    semantics = SettlementAuditSemantics.from_mapping(semantics_value)
    universe = None
    if universe_path is not None:
        universe_value = _load_cli_json(universe_path, "settlement universe")
        if not isinstance(universe_value, dict):
            raise ValueError("settlement universe must be a JSON object")
        universe = SettlementUniverse.from_mapping(universe_value)

    root = Path(config.storage.root).resolve()
    artifact = run_local_settlement_audit(
        root,
        semantics,
        metal=normalized_metal,
        universe=universe,
        exact_feed_id=int(product.pyth_lazer_id),
    )
    verify_settlement_audit(root, artifact.directory)

    run_id = f"settlement-audit-{artifact.audit_id}"
    run_config = {
        "component": "exact_settlement_audit",
        "operation": "audit_settlements",
        "read_only": True,
        "network_access": False,
        "live_orders": False,
        "metal": normalized_metal,
        "exact_feed_id": int(product.pyth_lazer_id),
        "audit_id": artifact.audit_id,
    }
    counts = {
        key: artifact.summary[key]
        for key in (
            "expected_count",
            "usable_count",
            "excluded_count",
            "missing_count",
            "conflicting_count",
            "fully_agreeing_usable_windows",
            "settlement_research_gate_passed",
        )
    }
    outputs = (
        artifact.details_path,
        artifact.summary_path,
        artifact.manifest_path,
    )
    run_manifest = root / "manifests" / f"run={run_id}" / "manifest.json"
    expected_outputs = [str(path) for path in outputs]
    if run_manifest.exists():
        existing = _load_cli_json(run_manifest, "settlement audit run manifest")
        if not isinstance(existing, dict):
            raise ValueError("settlement audit run manifest must be a JSON object")
        if existing.get("run_id") != run_id or existing.get("config") != run_config:
            raise ValueError("existing settlement audit run manifest conflicts")
        if existing.get("status") == "completed":
            if (
                existing.get("counts") != counts
                or existing.get("output_paths") != expected_outputs
            ):
                raise ValueError(
                    "completed settlement audit run manifest is inconsistent"
                )
        elif existing.get("status") != "running":
            raise ValueError("settlement audit run manifest is terminal but incomplete")
        else:
            update_run_manifest(
                root,
                run_id,
                status="completed",
                counts=counts,
                output_paths=outputs,
            )
    else:
        try:
            write_run_manifest(root, run_id, run_config)
        except FileExistsError:
            return _audit_settlements(config_path, metal, semantics_path, universe_path)
        update_run_manifest(
            root,
            run_id,
            status="completed",
            counts=counts,
            output_paths=outputs,
        )

    print(
        json.dumps(
            {
                "audit_id": artifact.audit_id,
                "artifact_manifest": str(artifact.manifest_path),
                "details": str(artifact.details_path),
                "summary": str(artifact.summary_path),
                "run_manifest": str(run_manifest),
                **counts,
            },
            sort_keys=True,
        )
    )
    return 0


def _hash_registry_config(record_path: Path) -> int:
    record = _load_cli_json(record_path, "registry record")
    if not isinstance(record, dict):
        raise ValueError("registry record must be a JSON object")
    print(json.dumps({"config_hash": registry_config_hash(record)}, sort_keys=True))
    return 0


def _materialize_daily_policy_registries(
    config_path: str,
    date: str,
    policy_path: Path,
    fee_registry_path: Path,
    incentive_registry_path: Path,
) -> int:
    config = load_metals_config(config_path)
    policy = _load_cli_json(policy_path, "daily gate policy")
    fees = _load_cli_json(fee_registry_path, "fee registry")
    incentives = _load_cli_json(incentive_registry_path, "incentive registry")
    if not isinstance(policy, dict):
        raise ValueError("daily gate policy must be a JSON object")
    if not isinstance(fees, list):
        raise ValueError("fee registry must be a JSON array")
    if not isinstance(incentives, list):
        raise ValueError("incentive registry must be a JSON array")
    artifacts = materialize_daily_policy_registries(
        Path(config.storage.root),
        date,
        policy,
        fees,
        incentives,
    )
    print(
        json.dumps(
            {
                "materialization_id": artifacts.materialization_id,
                "policy": str(artifacts.policy_path),
                "fee_schedule_dim": str(artifacts.fee_schedule_path),
                "incentive_program_dim": str(artifacts.incentive_program_path),
                "manifest": str(artifacts.manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


def _check_data_readiness(
    config_path: str,
    minimum_complete_days: int,
    minimum_settlements: int,
) -> int:
    if minimum_complete_days < 1:
        raise ValueError("minimum_complete_days must be at least one")
    if minimum_settlements < 1:
        raise ValueError("minimum_settlements must be at least one")
    config = load_metals_config(config_path)
    progress = synchronized_day_progress(
        Path(config.storage.root),
        minimum_complete_days=minimum_complete_days,
        minimum_settlements=minimum_settlements,
    )
    print(json.dumps(progress, sort_keys=True))
    return 0 if progress["passed"] else 2


def _history_bound(value: str) -> Any:
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def _databento_history_request(
    config: Any,
    metal: str,
    start: str,
    end: str,
    schemas: Sequence[str] | None = None,
) -> DatabentoHistoricalRequest:
    normalized_metal = metal.strip().lower()
    product = config.products.get(normalized_metal)
    if product is None:
        raise ValueError("Unknown metals product: {}".format(metal))
    if not product.enabled:
        raise ValueError(
            "Refusing historical retrieval for disabled product: {}".format(metal)
        )
    if config.cme.provider.lower() != "databento":
        raise ValueError(
            "Databento history requires cme.provider='databento', not {!r}".format(
                config.cme.provider
            )
        )
    configured_schemas = tuple(
        str(value).strip().lower() for value in config.cme.schemas
    )
    if not configured_schemas or any(not schema for schema in configured_schemas):
        raise ValueError("cme.schemas must contain at least one non-empty schema")
    if len(set(configured_schemas)) != len(configured_schemas):
        raise ValueError("Configured CME schemas must be unique")
    unsupported_configured = sorted(set(configured_schemas) - set(HISTORICAL_SCHEMAS))
    if unsupported_configured:
        raise ValueError(
            "Unsupported configured Databento schemas: {}".format(
                ", ".join(unsupported_configured)
            )
        )
    selected_schemas = (
        configured_schemas
        if schemas is None
        else tuple(str(value).strip().lower() for value in schemas)
    )
    if not selected_schemas or any(not schema for schema in selected_schemas):
        raise ValueError("--schema must select at least one non-empty schema")
    if len(set(selected_schemas)) != len(selected_schemas):
        raise ValueError("--schema values must be unique")
    unavailable_schemas = sorted(set(selected_schemas) - set(configured_schemas))
    if unavailable_schemas:
        raise ValueError(
            "Selected Databento schemas are not configured: {}".format(
                ", ".join(unavailable_schemas)
            )
        )
    request = DatabentoHistoricalRequest(
        metal=normalized_metal,
        start=_history_bound(start),
        end=_history_bound(end),
        schemas=selected_schemas,
        dataset=config.cme.dataset,
    )
    configured_symbols = tuple(
        dict.fromkeys(
            value if "." in value else value + ".FUT"
            for value in (str(symbol).strip().upper() for symbol in product.cme_symbols)
            if value
        )
    )
    if set(configured_symbols) != set(request.symbols):
        raise ValueError(
            "Configured CME roots {} do not match the frozen {} history mapping {}".format(
                configured_symbols, normalized_metal, request.symbols
            )
        )
    return request


def _databento_history_archive(
    config: Any,
    *,
    provider_approval: Any | None = None,
    authority: HistoricalOperationAuthority | None = None,
) -> DatabentoHistoricalArchive:
    api_key = os.environ.get(config.cme.api_key_env, "").strip()
    return DatabentoHistoricalArchive(
        root=Path(config.storage.root),
        api_key=api_key,
        provider_approval=provider_approval,
        historical_authority=authority,
    )


def _cost_payload(estimate: HistoricalCostEstimate) -> Dict[str, Any]:
    payload = asdict(estimate)
    payload["per_schema_cost_usd"] = dict(estimate.per_schema_cost_usd)
    return payload


def _archive_payload(result: HistoricalArchiveResult) -> Dict[str, Any]:
    return {
        "request_id": result.request_id,
        "transport": result.transport,
        "completed": result.completed,
        "record_counts": dict(result.record_counts),
        "skipped_schemas": list(result.skipped_schemas),
        "output_paths": [str(path) for path in result.output_paths],
        "estimate": _cost_payload(result.estimate) if result.estimate else None,
        "batch_states": (
            dict(result.batch_states) if result.batch_states is not None else None
        ),
    }


async def _estimate_databento_history(
    config_path: str,
    metal: str,
    start: str,
    end: str,
    schemas: Sequence[str] | None = None,
) -> int:
    config = load_metals_config(config_path)
    request = _databento_history_request(config, metal, start, end, schemas)
    archive = _databento_history_archive(config)
    estimate = await asyncio.to_thread(archive.preflight, request)
    print(json.dumps(_cost_payload(estimate), sort_keys=True))
    return 0


async def _backfill_databento_history(
    config_path: str,
    metal: str,
    start: str,
    end: str,
    transport: str,
    max_cost_usd: Decimal,
    allow_billable_download: bool,
    schemas: Sequence[str] | None = None,
) -> int:
    if not max_cost_usd.is_finite() or max_cost_usd < 0:
        raise ValueError("max_cost_usd must be finite and non-negative")
    if not allow_billable_download:
        raise ValueError("--allow-billable-download is required for Databento history")
    config = load_metals_config(config_path)
    request = _databento_history_request(config, metal, start, end, schemas)
    authority = require_config_historical_operation(config, "databento", "historical")
    if authority.provider_approval is not None:
        archive = _databento_history_archive(
            config, provider_approval=authority.provider_approval
        )
    else:
        archive = _databento_history_archive(config, authority=authority)
    max_cost = float(max_cost_usd)
    if transport == "stream":
        result = await asyncio.to_thread(
            archive.download_stream,
            request,
            allow_billable_download=True,
            max_cost_usd=max_cost,
        )
    elif transport == "batch":
        submitted = await asyncio.to_thread(
            archive.submit_batch,
            request,
            allow_billable_download=True,
            max_cost_usd=max_cost,
        )
        result = (
            submitted
            if submitted.completed
            else await asyncio.to_thread(archive.ingest_submitted_batch, request)
        )
    else:  # pragma: no cover - argparse enforces the transport choices
        raise ValueError(
            "Unsupported Databento history transport: {}".format(transport)
        )
    print(json.dumps(_archive_payload(result), sort_keys=True))
    return 0


def _materialize_provider_approval(config_path: str, approval_path: Path) -> int:
    config = load_metals_config(config_path)
    value = _load_cli_json(approval_path, "provider approval")
    if not isinstance(value, dict):
        raise ValueError("provider approval must be a JSON object")
    artifact = materialize_provider_approval(Path(config.storage.root), value)
    print(json.dumps(artifact.public_dict(), sort_keys=True))
    return 0


def _check_provider_approval(config_path: str, provider: str, operation: str) -> int:
    config = load_metals_config(config_path)
    try:
        artifact = require_config_provider_operation(config, provider, operation)
    except (ProviderApprovalError, ValueError) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "provider": provider,
                    "operation": operation,
                    "status": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "provider": provider,
                "operation": operation,
                "approval_id": artifact.approval_id,
                "status": "approved",
            },
            sort_keys=True,
        )
    )
    return 0


def _materialize_historical_research_assumption(
    config_path: str, assumption_path: Path
) -> int:
    config = load_metals_config(config_path)
    value = _load_cli_json(assumption_path, "historical research assumption")
    if not isinstance(value, dict):
        raise ValueError("historical research assumption must be a JSON object")
    artifact = materialize_historical_research_assumption(
        Path(config.storage.root), value
    )
    print(json.dumps(artifact.public_dict(), sort_keys=True))
    return 0


def _check_historical_research_assumption(
    config_path: str, provider: str, operation: str
) -> int:
    config = load_metals_config(config_path)
    try:
        artifact = require_historical_research_assumption(
            Path(config.storage.root),
            configured_historical_assumption_path(config, provider),
            provider,
            operation,
        )
    except (HistoricalResearchAssumptionError, ValueError) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "provider": provider,
                    "operation": operation,
                    "status": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "provider": provider,
                "operation": operation,
                "assumption_id": artifact.assumption_id,
                "research_only": True,
                "no_live": True,
                "no_orders": True,
                "status": "accepted_research_assumption",
            },
            sort_keys=True,
        )
    )
    return 0


async def async_main(args: argparse.Namespace) -> int:
    if args.command == "discover":
        return await _discover(args.config)
    if args.command in ("check-access", "preflight-kalshi"):
        return await _check_access(args.config)
    if args.command == "preflight-clock":
        report = await asyncio.to_thread(
            preflight_clock, load_metals_config(args.config)
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if report["ok"] else 2
    if args.command == "materialize-provider-approval":
        return _materialize_provider_approval(args.config, args.approval)
    if args.command == "check-provider-approval":
        return _check_provider_approval(args.config, args.provider, args.operation)
    if args.command == "materialize-historical-research-assumption":
        return _materialize_historical_research_assumption(args.config, args.assumption)
    if args.command == "check-historical-research-assumption":
        return _check_historical_research_assumption(
            args.config, args.provider, args.operation
        )
    if args.command == "preflight-pyth":
        config = load_metals_config(args.config)
        report = await preflight_pyth(
            config,
            metal=args.metal,
            history_at_seconds=args.history_at_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if report["ok"] else 2
    if args.command == "preflight-pyth-lazer-live":
        config = load_metals_config(args.config)
        report = await preflight_pyth_lazer_live(
            config,
            metal=args.metal,
            duration_seconds=args.duration_seconds,
            allow_live_probe=args.allow_live_entitlement_probe,
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if report["ok"] else 2
    if args.command == "preflight-databento":
        config = load_metals_config(args.config)
        report = await asyncio.to_thread(
            preflight_databento,
            config,
            metal=args.metal,
            probe_date=args.probe_date,
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if report["ok"] else 2
    if args.command == "preflight-databento-live":
        config = load_metals_config(args.config)
        report = await asyncio.to_thread(
            preflight_databento_live,
            config,
            metal=args.metal,
            duration_seconds=args.duration_seconds,
            allow_live_probe=args.allow_live_entitlement_probe,
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if report["ok"] else 2
    if args.command == "backfill-pyth-exact-history":
        return await _backfill_pyth_exact_history(
            args.config,
            args.metal,
            args.start_timestamp_us,
            args.end_timestamp_us,
            args.limit,
            args.max_pages,
            args.timeout_seconds,
        )
    if args.command == "plan-pyth-settlement-history":
        return _plan_pyth_settlement_history(
            args.config,
            args.metal,
            args.audit,
            args.limit,
            args.max_pages,
        )
    if args.command == "backfill-pyth-settlement-history":
        return await _backfill_pyth_settlement_history(
            args.config,
            args.plan,
            args.max_boundaries,
            args.request_interval_seconds,
            args.timeout_seconds,
            args.allow_network_download,
        )
    if args.command == "estimate-databento-history":
        return await _estimate_databento_history(
            args.config,
            args.metal,
            args.start,
            args.end,
            args.schema,
        )
    if args.command == "backfill-databento-history":
        return await _backfill_databento_history(
            args.config,
            args.metal,
            args.start,
            args.end,
            args.transport,
            args.max_cost_usd,
            args.allow_billable_download,
            args.schema,
        )
    if args.command == "collect":
        return await _collect(args.config, args.duration_seconds)
    if args.command == "collect-databento":
        return await _collect_databento(
            args.config,
            args.metal,
            args.duration_seconds,
            args.max_records,
        )
    if args.command == "collect-pyth-lazer":
        return await _collect_pyth_lazer(
            args.config,
            args.metal,
            args.duration_seconds,
        )
    if args.command == "collect-clock-telemetry":
        return await _collect_clock_telemetry(
            args.config,
            args.metal,
            args.duration_seconds,
        )
    if args.command == "collect-synchronized":
        return await _collect_synchronized(
            args.config,
            args.metal,
            args.duration_seconds,
        )
    if args.command == "run-forward-shadow":
        return await _run_forward_shadow(args)
    if args.command == "freeze-daily-validation":
        return _freeze_daily_validation(
            args.config,
            args.date,
            args.quality_report,
            args.manifest,
        )
    if args.command == "measure-daily-gates":
        return _measure_daily_gates(
            args.config,
            args.date,
            args.quality_report,
            args.policy,
            args.manifest,
        )
    if args.command == "hash-registry-config":
        return _hash_registry_config(args.record)
    if args.command == "materialize-daily-policy-registries":
        return _materialize_daily_policy_registries(
            args.config,
            args.date,
            args.policy,
            args.fee_registry,
            args.incentive_registry,
        )
    if args.command == "audit-settlements":
        return _audit_settlements(
            args.config,
            args.metal,
            args.semantics,
            args.universe,
        )
    if args.command == "check-data-readiness":
        return _check_data_readiness(
            args.config,
            args.minimum_complete_days,
            args.minimum_settlements,
        )
    if args.command == "backfill-kalshi":
        config = load_metals_config(args.config)
        counts = await backfill_settled_markets(
            config,
            metal=args.metal,
            max_markets=args.max_markets or None,
            include_trades=not args.skip_trades,
            min_close_seconds=args.from_close_seconds,
            max_close_seconds=args.to_close_seconds,
            market_stride=args.market_stride,
            market_stride_offset=args.market_stride_offset,
        )
        print(json.dumps(counts, sort_keys=True))
        return 0
    if args.command == "backfill-kalshi-candles":
        config = load_metals_config(args.config)
        counts = await backfill_kalshi_candlesticks(
            config,
            metal=args.metal,
            max_markets=args.max_markets or None,
            resume=not args.no_resume,
        )
        print(json.dumps(counts, sort_keys=True))
        return 0
    if args.command == "export-parquet":
        config = load_metals_config(args.config)
        counts = export_normalized_parquet(config.storage.root)
        catalog = build_duckdb_catalog(config.storage.root)
        print(json.dumps({**counts, "catalog": str(catalog)}, sort_keys=True))
        return 0
    if args.command == "backfill-pyth-history":
        config = load_metals_config(args.config)
        counts = await backfill_pyth_history(
            config,
            metal=args.metal,
            start_seconds=args.from_seconds,
            end_seconds=args.to_seconds,
        )
        print(json.dumps(counts, sort_keys=True))
        return 0
    if args.command == "backfill-pyth-proxy-history":
        config = load_metals_config(args.config)
        counts = await backfill_pyth_proxy_history(
            config,
            metal=args.metal,
            start_seconds=args.from_seconds,
            end_seconds=args.to_seconds,
        )
        print(json.dumps(counts, sort_keys=True))
        return 0
    if args.command == "run-s1-feasibility":
        config = load_metals_config(args.config)
        study_config = S1FeasibilityConfig(
            metal=args.metal,
            train_fraction=args.train_fraction,
            min_underdog_mid=args.min_underdog_mid,
            max_underdog_mid=args.max_underdog_mid,
            min_seconds_to_close=args.min_seconds_to_close,
            penetration_dollars=args.penetration_dollars,
            maker_fee_per_contract=args.maker_fee_per_contract,
            taker_fee_multiplier=args.taker_fee_multiplier,
        )
        result = run_s1_feasibility(
            catalog_path=Path(config.storage.root) / "metals.duckdb",
            output_root=Path(config.storage.root),
            config=study_config,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "run-toxicity-diagnostics":
        config = load_metals_config(args.config)
        result = run_toxicity_diagnostics(
            catalog_path=Path(config.storage.root) / "metals.duckdb",
            output_root=Path(config.storage.root),
            config=ToxicityConfig(
                metal=args.metal,
                test_start_ts_ns=args.test_start_ts_ns,
                penetration_dollars=args.penetration_dollars,
                maker_fee_per_contract=args.maker_fee_per_contract,
                require_completed_trade_request=args.require_complete_trades,
            ),
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "run-settlement-proxy-pilot":
        config = load_metals_config(args.config)
        result = run_settlement_proxy_pilot(
            Path(config.storage.root),
            args.labels,
            args.databento_trades,
            args.pyth_proxy_candles,
            SettlementProxyPilotConfig(
                metal=args.metal,
                start_ts_ns=args.start_ts_ns,
                end_ts_ns=args.end_ts_ns,
                primary_asset=args.primary_asset,
                secondary_asset=args.secondary_asset,
                max_trade_age_seconds=args.max_trade_age_seconds,
                vwap_lookback_seconds=args.vwap_lookback_seconds,
                ambiguity_dollars=args.ambiguity_dollars,
                train_fraction=args.train_fraction,
            ),
        )
        print(
            json.dumps(
                {
                    "pilot_id": result["pilot_id"],
                    "report": result["artifacts"]["report"],
                    "windows": result["cohort"]["included_windows"],
                    "quarantined_windows": result["cohort"]["quarantined_windows"],
                    "exploratory_expectations_passed": result["decision"][
                        "exploratory_expectations_passed"
                    ],
                    "formal_settlement_gate_passed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-baseline-fair":
        config = load_metals_config(args.config)
        result = run_baseline_fair_study(
            Path(config.storage.root),
            args.labels,
            args.pyth_proxy_candles,
            BaselineFairConfig(
                metal=args.metal,
                training_end_ts_ns=args.training_end_ts_ns,
                validation_start_ts_ns=args.validation_start_ts_ns,
                validation_end_ts_ns=args.validation_end_ts_ns,
                formal_research_design_id=args.formal_research_design_id,
                prerequisite_proxy_pilot_id=args.prerequisite_proxy_pilot_id,
                ewma_lambda=args.ewma_lambda,
                tod_prior_observations=args.tod_prior_observations,
                winsor_quantile=args.winsor_quantile,
            ),
        )
        print(
            json.dumps(
                {
                    "study_id": result["study_id"],
                    "report": result["artifacts"]["report"],
                    "training_windows": result["cohort"]["training_windows"],
                    "validation_windows": result["cohort"][
                        "validation_windows_with_complete_proxy"
                    ],
                    "eligible_for_incremental_signal_development": result[
                        "decision"
                    ]["eligible_for_incremental_signal_development"],
                    "trading_eligible": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-spot-return-signals":
        config = load_metals_config(args.config)
        result = run_spot_return_signal_study(
            Path(config.storage.root),
            args.pyth_proxy_candles,
            SpotReturnSignalConfig(
                baseline_study_id=args.baseline_study_id,
                lookback_minutes=tuple(args.lookback_minutes or (1, 5, 10)),
                development_fraction=args.development_fraction,
                embargo_windows=args.embargo_windows,
                ridge_penalty=args.ridge_penalty,
            ),
        )
        print(
            json.dumps(
                {
                    "family_study_id": result["family_study_id"],
                    "report": result["artifacts"]["report"],
                    "accepted_signal_ids": result["decision"][
                        "accepted_signal_ids"
                    ],
                    "family_added_to_model": result["decision"][
                        "family_added_to_model"
                    ],
                    "formal_final_holdout_accessed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-path-variance-signals":
        config = load_metals_config(args.config)
        result = run_path_variance_signal_study(
            Path(config.storage.root),
            args.pyth_proxy_candles,
            PathVarianceSignalConfig(
                baseline_study_id=args.baseline_study_id,
                range_lookback_minutes=tuple(
                    args.range_lookback_minutes or (3, 10)
                ),
                efficiency_lookback_minutes=tuple(
                    args.efficiency_lookback_minutes or (5, 10)
                ),
                development_fraction=args.development_fraction,
                embargo_windows=args.embargo_windows,
                ridge_penalty=args.ridge_penalty,
            ),
        )
        print(
            json.dumps(
                {
                    "family_study_id": result["family_study_id"],
                    "report": result["artifacts"]["report"],
                    "accepted_signal_ids": result["decision"][
                        "accepted_signal_ids"
                    ],
                    "family_added_to_model": result["decision"][
                        "family_added_to_model"
                    ],
                    "formal_final_holdout_accessed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-futures-activity-signals":
        config = load_metals_config(args.config)
        result = run_futures_activity_signal_study(
            Path(config.storage.root),
            args.databento_trades,
            FuturesActivitySignalConfig(
                baseline_study_id=args.baseline_study_id,
                asset=args.asset,
                development_fraction=args.development_fraction,
                embargo_windows=args.embargo_windows,
                ridge_penalty=args.ridge_penalty,
                max_trade_age_seconds=args.max_trade_age_seconds,
            ),
        )
        print(
            json.dumps(
                {
                    "family_study_id": result["family_study_id"],
                    "report": result["artifacts"]["report"],
                    "selected_contract": result["selected_contract"]["symbol"],
                    "accepted_signal_ids": result["decision"][
                        "accepted_signal_ids"
                    ],
                    "family_added_to_model": result["decision"][
                        "family_added_to_model"
                    ],
                    "formal_final_holdout_accessed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-semivariance-jump-signals":
        config = load_metals_config(args.config)
        result = run_semivariance_jump_signal_study(
            Path(config.storage.root),
            args.pyth_proxy_candles,
            SemivarianceJumpSignalConfig(
                baseline_study_id=args.baseline_study_id,
                lookback_minutes=args.lookback_minutes,
                development_fraction=args.development_fraction,
                embargo_windows=args.embargo_windows,
                ridge_penalty=args.ridge_penalty,
            ),
        )
        print(
            json.dumps(
                {
                    "family_study_id": result["family_study_id"],
                    "report": result["artifacts"]["report"],
                    "accepted_signal_ids": result["decision"][
                        "accepted_signal_ids"
                    ],
                    "family_added_to_model": result["decision"][
                        "family_added_to_model"
                    ],
                    "formal_final_holdout_accessed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-cross-metal-signals":
        config = load_metals_config(args.config)
        result = run_cross_metal_signal_study(
            Path(config.storage.root),
            args.silver_pyth_proxy_candles,
            args.gold_pyth_proxy_candles,
            CrossMetalSignalConfig(
                baseline_study_id=args.baseline_study_id,
                lookback_minutes=args.lookback_minutes,
                development_fraction=args.development_fraction,
                embargo_windows=args.embargo_windows,
                ridge_penalty=args.ridge_penalty,
            ),
        )
        print(
            json.dumps(
                {
                    "family_study_id": result["family_study_id"],
                    "report": result["artifacts"]["report"],
                    "accepted_signal_ids": result["decision"][
                        "accepted_signal_ids"
                    ],
                    "family_added_to_model": result["decision"][
                        "family_added_to_model"
                    ],
                    "formal_final_holdout_accessed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-baseline-scorecard":
        config = load_metals_config(args.config)
        result = run_baseline_scorecard(
            Path(config.storage.root),
            args.kalshi_candlesticks,
            BaselineScorecardConfig(baseline_study_id=args.baseline_study_id),
        )
        print(
            json.dumps(
                {
                    "scorecard_id": result["scorecard_id"],
                    "report": result["artifacts"]["report"],
                    "common_row_test_rows": result["split"][
                        "common_row_test_rows"
                    ],
                    "eligible_for_trading_claim": False,
                    "formal_final_holdout_accessed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-natural-target-signals":
        config = load_metals_config(args.config)
        result = run_natural_target_signal_study(
            Path(config.storage.root),
            args.silver_pyth_proxy_candles,
            NaturalTargetSignalConfig(
                baseline_study_id=args.baseline_study_id,
                development_fraction=args.development_fraction,
                embargo_windows=args.embargo_windows,
                ridge_penalty=args.ridge_penalty,
                walk_forward_bootstrap_samples=(
                    args.walk_forward_bootstrap_samples
                ),
                walk_forward_bootstrap_seed=args.walk_forward_bootstrap_seed,
            ),
            databento_trades_path=args.databento_trades,
            gold_pyth_proxy_candles_path=args.gold_pyth_proxy_candles,
        )
        print(
            json.dumps(
                {
                    "study_id": result["study_id"],
                    "report": result["artifacts"]["report"],
                    "natural_target_signal_cards": len(result["signal_cards"]),
                    "disposition": result["conclusion"]["disposition"],
                    "formal_final_holdout_accessed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    raise ValueError("Unknown command: {}".format(args.command))


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
