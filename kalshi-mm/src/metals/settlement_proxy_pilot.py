"""Small, research-only comparison of Kalshi labels with external metals proxies.

This module deliberately does not certify the Kalshi settlement feed.  It measures
whether independently collected COMEX trades and generic Pyth spot candles behave
as expected around the same 15-minute boundaries, while preserving the exact-feed
settlement audit as a separate formal gate.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from statistics import median
from typing import Any

from .storage import HashChainedJsonlWriter, update_run_manifest, write_run_manifest

NSEC = 1_000_000_000
SCHEMA_VERSION = "metals-settlement-proxy-pilot/v3"


class SettlementProxyPilotError(ValueError):
    """Raised when a pilot input is invalid or its provenance cannot be verified."""


@dataclass(frozen=True)
class SettlementProxyPilotConfig:
    metal: str = "silver"
    start_ts_ns: int = 0
    end_ts_ns: int = 0
    primary_asset: str = "SI"
    secondary_asset: str = "SIL"
    max_trade_age_seconds: int = 5
    vwap_lookback_seconds: int = 1
    ambiguity_dollars: Decimal = Decimal("0.005")
    train_fraction: Decimal = Decimal("0.67")

    def __post_init__(self) -> None:
        if self.metal.strip().lower() not in {"silver", "gold"}:
            raise SettlementProxyPilotError("metal must be silver or gold")
        if self.start_ts_ns < 0 or self.end_ts_ns <= self.start_ts_ns:
            raise SettlementProxyPilotError(
                "pilot timestamps must define a positive interval"
            )
        if self.max_trade_age_seconds < 1:
            raise SettlementProxyPilotError("max_trade_age_seconds must be positive")
        if self.vwap_lookback_seconds < 1:
            raise SettlementProxyPilotError("vwap_lookback_seconds must be positive")
        if not self.ambiguity_dollars.is_finite() or self.ambiguity_dollars < 0:
            raise SettlementProxyPilotError(
                "ambiguity_dollars must be finite and non-negative"
            )
        if not self.train_fraction.is_finite() or not Decimal(
            "0.5"
        ) < self.train_fraction < Decimal("0.9"):
            raise SettlementProxyPilotError(
                "train_fraction must be between 0.5 and 0.9"
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
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise SettlementProxyPilotError(
            f"input must be below storage root: {path}"
        ) from error
    if not resolved.is_file():
        raise SettlementProxyPilotError(f"input is not a readable file: {path}")
    return resolved


def _read_hash_chained(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_previous: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SettlementProxyPilotError(
                    f"invalid JSON in {path} at line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise SettlementProxyPilotError(
                    f"non-object record in {path} at line {line_number}"
                )
            record_hash = row.get("record_sha256")
            payload = dict(row)
            payload.pop("record_sha256", None)
            computed = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
            if record_hash != computed:
                raise SettlementProxyPilotError(
                    f"record hash mismatch in {path} at line {line_number}"
                )
            if row.get("previous_record_sha256") != expected_previous:
                raise SettlementProxyPilotError(
                    f"record chain mismatch in {path} at line {line_number}"
                )
            expected_previous = str(record_hash)
            rows.append(row)
    if not rows:
        raise SettlementProxyPilotError(f"input is empty: {path}")
    return rows


def _decimal(value: Any, label: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise SettlementProxyPilotError(
            f"invalid decimal {label}: {value!r}"
        ) from error
    if not parsed.is_finite():
        raise SettlementProxyPilotError(f"non-finite decimal {label}: {value!r}")
    return parsed


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    if left_ss <= 0 or right_ss <= 0:
        return None
    return covariance / math.sqrt(left_ss * right_ss)


def _fit_linear(pairs: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    if len(pairs) < 3:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0:
        return None
    beta = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denominator
    return y_mean - beta * x_mean, beta


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    feature: str,
    *,
    ambiguity: float,
    train_fraction: float,
) -> dict[str, Any]:
    usable = [
        row for row in rows if not row["quarantined"] and row.get(feature) is not None
    ]
    actual = [float(row["kalshi_reported_delta"]) for row in usable]
    proxy = [float(row[feature]) for row in usable]
    direct_matches = [
        (value >= 0) == (str(row["result"]).lower() == "yes")
        for row, value in zip(usable, proxy)
    ]
    non_ambiguous = [
        (row, value)
        for row, value in zip(usable, proxy)
        if abs(float(row["kalshi_reported_delta"])) > ambiguity
    ]
    summary: dict[str, Any] = {
        "eligible_windows": sum(not row["quarantined"] for row in rows),
        "observations": len(usable),
        "coverage": len(usable) / max(1, sum(not row["quarantined"] for row in rows)),
        "delta_correlation": _correlation(proxy, actual),
        "direct_result_agreement": _mean([float(value) for value in direct_matches]),
        "non_ambiguous_observations": len(non_ambiguous),
        "non_ambiguous_result_agreement": _mean(
            [
                float((value >= 0) == (str(row["result"]).lower() == "yes"))
                for row, value in non_ambiguous
            ]
        ),
        "mean_proxy_delta": _mean(proxy),
        "mean_kalshi_reported_delta": _mean(actual),
    }

    pairs = sorted(
        (
            (float(row[feature]), float(row["kalshi_reported_delta"]), row)
            for row in usable
        ),
        key=lambda item: int(item[2]["open_ts_ns"]),
    )
    split = max(3, min(len(pairs) - 1, int(len(pairs) * train_fraction)))
    if len(pairs) >= 5 and split < len(pairs):
        fit = _fit_linear([(x, y) for x, y, _ in pairs[:split]])
        if fit is not None:
            intercept, beta = fit
            errors: list[float] = []
            squared: list[float] = []
            classification: list[float] = []
            unchanged_errors: list[float] = []
            for x, y, row in pairs[split:]:
                prediction = intercept + beta * x
                error = prediction - y
                errors.append(error)
                squared.append(error * error)
                unchanged_errors.append(abs(y))
                classification.append(
                    float((prediction >= 0) == (str(row["result"]).lower() == "yes"))
                )
            summary["chronological_fit"] = {
                "train_observations": split,
                "test_observations": len(pairs) - split,
                "intercept": intercept,
                "beta": beta,
                "test_mae": _mean([abs(value) for value in errors]),
                "test_rmse": math.sqrt(sum(squared) / len(squared)),
                "test_bias": _mean(errors),
                "test_result_agreement": _mean(classification),
                "unchanged_delta_test_mae": _mean(unchanged_errors),
            }
        else:
            summary["chronological_fit"] = None
    else:
        summary["chronological_fit"] = None
    return summary


def _pyth_level_summary(
    rows: Sequence[Mapping[str, Any]], prefix: str
) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if not row["quarantined"]
        and row.get(f"{prefix}_open") is not None
        and row.get(f"{prefix}_close") is not None
    ]
    open_errors = [
        float(Decimal(row[f"{prefix}_open"]) - Decimal(row["target_price"]))
        for row in usable
    ]
    close_errors = [
        float(Decimal(row[f"{prefix}_close"]) - Decimal(row["expiration_value"]))
        for row in usable
    ]
    quantum = Decimal("0.001")

    def error_summary(values: Sequence[float]) -> dict[str, Any]:
        return {
            "bias": _mean(values),
            "mae": _mean([abs(value) for value in values]),
            "median_absolute_error": median([abs(value) for value in values])
            if values
            else None,
            "max_absolute_error": max((abs(value) for value in values), default=None),
            "within_half_cent": _mean([float(abs(value) <= 0.005) for value in values]),
            "within_one_cent": _mean([float(abs(value) <= 0.01) for value in values]),
        }

    raw_outcomes: list[float] = []
    non_ambiguous_raw_outcomes: list[float] = []
    rounded_outcomes: list[float] = []
    rounded_open_levels: list[float] = []
    rounded_close_levels: list[float] = []
    raw_mismatch_tickers: list[str] = []
    for row in usable:
        open_value = Decimal(row[f"{prefix}_open"])
        close_value = Decimal(row[f"{prefix}_close"])
        target = Decimal(row["target_price"])
        expiration = Decimal(row["expiration_value"])
        actual_yes = row["result"] == "yes"
        raw_match = (close_value >= target) == actual_yes
        raw_outcomes.append(float(raw_match))
        if abs(expiration - target) > Decimal("0.005"):
            non_ambiguous_raw_outcomes.append(float(raw_match))
        if not raw_match:
            raw_mismatch_tickers.append(str(row["market_ticker"]))
        rounded_outcomes.append(
            float(
                (close_value.quantize(quantum, rounding=ROUND_HALF_UP) >= target)
                == actual_yes
            )
        )
        rounded_open_levels.append(
            float(open_value.quantize(quantum, rounding=ROUND_HALF_UP) == target)
        )
        rounded_close_levels.append(
            float(close_value.quantize(quantum, rounding=ROUND_HALF_UP) == expiration)
        )
    return {
        "observations": len(usable),
        "open_level_error": error_summary(open_errors),
        "close_level_error": error_summary(close_errors),
        "threshold_result_agreement_raw_close_vs_kalshi_target": _mean(raw_outcomes),
        "non_ambiguous_threshold_result_agreement": _mean(non_ambiguous_raw_outcomes),
        "threshold_result_mismatch_tickers": raw_mismatch_tickers,
        "threshold_result_agreement_after_half_up_3dp": _mean(rounded_outcomes),
        "displayed_open_level_agreement_after_half_up_3dp": _mean(rounded_open_levels),
        "displayed_close_level_agreement_after_half_up_3dp": _mean(
            rounded_close_levels
        ),
        "rounding_note": "Half-up at three decimals is diagnostic only, not a certified settlement rule.",
    }


def _select_contract(
    trades: Iterable[Mapping[str, Any]], asset: str
) -> tuple[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"trades": 0, "volume": 0})
    normalized_asset = asset.strip().upper()
    for row in trades:
        if str(row.get("asset", "")).upper() != normalized_asset:
            continue
        symbol = str(row.get("contract_symbol", "")).strip()
        if not symbol:
            continue
        size = int(row.get("size") or 0)
        stats[symbol]["trades"] += 1
        stats[symbol]["volume"] += max(0, size)
    if not stats:
        raise SettlementProxyPilotError(f"no trades found for asset {normalized_asset}")
    symbol = max(
        stats, key=lambda item: (stats[item]["volume"], stats[item]["trades"], item)
    )
    return symbol, stats[symbol]


def _series(
    trades: Iterable[Mapping[str, Any]], symbol: str
) -> tuple[list[int], list[Decimal], list[int]]:
    values: list[tuple[int, Decimal, int]] = []
    for row in trades:
        if row.get("contract_symbol") != symbol:
            continue
        timestamp = int(row.get("source_event_ts_ns") or row.get("ts_event") or 0)
        if timestamp <= 0:
            continue
        values.append(
            (
                timestamp,
                _decimal(row.get("price"), f"{symbol} price"),
                int(row.get("size") or 0),
            )
        )
    values.sort(key=lambda item: item[0])
    return (
        [item[0] for item in values],
        [item[1] for item in values],
        [item[2] for item in values],
    )


def _asof(
    timestamps: Sequence[int],
    prices: Sequence[Decimal],
    boundary_ns: int,
    max_age_ns: int,
) -> Decimal | None:
    index = bisect.bisect_left(timestamps, boundary_ns) - 1
    if index < 0 or boundary_ns - timestamps[index] > max_age_ns:
        return None
    return prices[index]


def _vwap(
    timestamps: Sequence[int],
    prices: Sequence[Decimal],
    sizes: Sequence[int],
    boundary_ns: int,
    lookback_ns: int,
) -> Decimal | None:
    start = bisect.bisect_left(timestamps, boundary_ns - lookback_ns)
    end = bisect.bisect_left(timestamps, boundary_ns)
    total_size = sum(max(0, size) for size in sizes[start:end])
    if total_size <= 0:
        return None
    total = sum(
        (
            price * max(0, size)
            for price, size in zip(prices[start:end], sizes[start:end])
        ),
        Decimal(0),
    )
    return total / Decimal(total_size)


def _delta(start: Decimal | None, end: Decimal | None) -> str | None:
    return str(end - start) if start is not None and end is not None else None


def _trade_boundary_values(
    series: tuple[list[int], list[Decimal], list[int]],
    open_ns: int,
    close_ns: int,
    max_age_ns: int,
    vwap_ns: int,
) -> tuple[Decimal | None, Decimal | None, Decimal | None, Decimal | None]:
    timestamps, prices, sizes = series
    return (
        _asof(timestamps, prices, open_ns, max_age_ns),
        _asof(timestamps, prices, close_ns, max_age_ns),
        _vwap(timestamps, prices, sizes, open_ns, vwap_ns),
        _vwap(timestamps, prices, sizes, close_ns, vwap_ns),
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Settlement proxy pilot",
        "",
        f"Pilot interval: `{report['interval']['start_utc']}` to `{report['interval']['end_utc']}`",
        "",
        f"Included windows: **{report['cohort']['included_windows']}**; quarantined: **{report['cohort']['quarantined_windows']}**.",
        "",
        "This is an exploratory proxy check. It does **not** certify the exact Pyth settlement feed.",
        "",
        "| Feature | N | Coverage | Delta correlation | Direction agreement | Non-ambiguous agreement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metric in report["metrics"].items():

        def fmt(value: Any) -> str:
            return "n/a" if value is None else f"{float(value):.3f}"

        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                name,
                metric["observations"],
                fmt(metric["coverage"]),
                fmt(metric["delta_correlation"]),
                fmt(metric["direct_result_agreement"]),
                fmt(metric["non_ambiguous_result_agreement"]),
            )
        )
    lines.extend(
        [
            "",
            "## Generic Pyth absolute-level check",
            "",
            (
                "Using the better preceding-minute alignment, the raw proxy "
                "close-versus-Kalshi-target outcome agreement is "
                f"**{report['pyth_level_checks']['preceding_minute']['threshold_result_agreement_raw_close_vs_kalshi_target']:.3f}**. "
                "The generic feed is still not settlement-eligible."
            ),
            "",
            "## Decision",
            "",
            report["decision"]["summary"],
            "",
            "The formal settlement gate remains closed until exact `Metal.Index.*` observations and the reviewed Kalshi sampling/rounding semantics are available.",
            "",
        ]
    )
    return "\n".join(lines)


def run_settlement_proxy_pilot(
    storage_root: Path,
    labels_path: Path,
    databento_trades_path: Path,
    pyth_proxy_candles_path: Path,
    config: SettlementProxyPilotConfig,
) -> dict[str, Any]:
    """Run a bounded proxy comparison and persist a reproducible derived artifact."""

    root = storage_root.resolve()
    labels_file = _resolve_input(root, labels_path)
    trades_file = _resolve_input(root, databento_trades_path)
    proxy_file = _resolve_input(root, pyth_proxy_candles_path)
    input_files = {
        "kalshi_labels": labels_file,
        "databento_trades": trades_file,
        "pyth_generic_proxy_candles": proxy_file,
    }
    input_hashes = {name: _sha256_path(path) for name, path in input_files.items()}
    public_config = {
        **asdict(config),
        "metal": config.metal.strip().lower(),
        "primary_asset": config.primary_asset.strip().upper(),
        "secondary_asset": config.secondary_asset.strip().upper(),
    }
    pilot_id = hashlib.sha256(
        _canonical(
            {
                "schema_version": SCHEMA_VERSION,
                "config": public_config,
                "inputs": input_hashes,
            }
        ).encode("utf-8")
    ).hexdigest()
    run_id = f"settlement-proxy-{pilot_id[:24]}"
    output_dir = root / "derived" / "settlement_proxy_pilot" / f"run={run_id}"
    report_path = output_dir / "report.json"
    rows_path = output_dir / "rows.jsonl"
    anomalies_path = output_dir / "anomalies.jsonl"
    markdown_path = output_dir / "REPORT.md"
    manifest_path = root / "manifests" / f"run={run_id}" / "manifest.json"

    if report_path.exists() and manifest_path.exists():
        with report_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    manifest_config = {
        "component": "settlement_proxy_pilot",
        "schema_version": SCHEMA_VERSION,
        "research_only": True,
        "settlement_eligible": False,
        "network_access": False,
        "live_orders": False,
        "config": public_config,
        "input_sha256": input_hashes,
    }
    write_run_manifest(
        root,
        run_id,
        manifest_config,
        output_paths=(labels_file, trades_file, proxy_file),
    )
    try:
        labels = _read_hash_chained(labels_file)
        trades = _read_hash_chained(trades_file)
        proxy_rows = _read_hash_chained(proxy_file)
        if any(row.get("settlement_eligible") is not False for row in proxy_rows):
            raise SettlementProxyPilotError(
                "generic Pyth proxy input must explicitly be settlement_eligible=false"
            )

        metal = config.metal.strip().lower()
        candidates = [
            row
            for row in labels
            if row.get("metal") == metal
            and config.start_ts_ns <= int(row.get("open_ts_ns") or -1)
            and int(row.get("close_ts_ns") or -1) <= config.end_ts_ns
        ]
        candidates.sort(key=lambda row: int(row["open_ts_ns"]))
        if not candidates:
            raise SettlementProxyPilotError(
                "no Kalshi windows fall within the pilot interval"
            )

        primary_symbol, primary_stats = _select_contract(trades, config.primary_asset)
        secondary_symbol, secondary_stats = _select_contract(
            trades, config.secondary_asset
        )
        primary_series = _series(trades, primary_symbol)
        secondary_series = _series(trades, secondary_symbol)
        max_age_ns = config.max_trade_age_seconds * NSEC
        vwap_ns = config.vwap_lookback_seconds * NSEC
        proxy_by_second = {
            int(row["timestamp_seconds"]): _decimal(row["close"], "Pyth proxy close")
            for row in proxy_rows
            if row.get("metal") == metal
        }

        rows: list[dict[str, Any]] = []
        anomalies: list[dict[str, Any]] = []
        previous: Mapping[str, Any] | None = None
        for label in candidates:
            open_ns = int(label["open_ts_ns"])
            close_ns = int(label["close_ts_ns"])
            reasons: list[str] = []
            if close_ns - open_ns != 15 * 60 * NSEC:
                reasons.append("non_15_minute_window")
            if str(label.get("strike_type")) != "greater_or_equal":
                reasons.append("unexpected_strike_type")
            if str(label.get("result", "")).lower() not in {"yes", "no"}:
                reasons.append("invalid_result")
            target = _decimal(label.get("target_price"), "Kalshi target")
            expiration = _decimal(label.get("expiration_value"), "Kalshi expiration")
            expected_result = "yes" if expiration >= target else "no"
            if str(label.get("result", "")).lower() != expected_result:
                reasons.append("displayed_values_contradict_result")
            if previous is not None and int(previous["close_ts_ns"]) == open_ns:
                prior_expiration = _decimal(
                    previous["expiration_value"], "prior expiration"
                )
                if target != prior_expiration:
                    anomalies.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "type": "adjacent_boundary_value_mismatch",
                            "market_ticker": label.get("market_ticker"),
                            "prior_market_ticker": previous.get("market_ticker"),
                            "difference": str(target - prior_expiration),
                        }
                    )

            p_open, p_close, p_open_vwap, p_close_vwap = _trade_boundary_values(
                primary_series, open_ns, close_ns, max_age_ns, vwap_ns
            )
            s_open, s_close, s_open_vwap, s_close_vwap = _trade_boundary_values(
                secondary_series, open_ns, close_ns, max_age_ns, vwap_ns
            )
            open_second = open_ns // NSEC
            close_second = close_ns // NSEC
            pyth_pre_open = proxy_by_second.get(open_second - 60)
            pyth_pre_close = proxy_by_second.get(close_second - 60)
            pyth_boundary_open = proxy_by_second.get(open_second)
            pyth_boundary_close = proxy_by_second.get(close_second)
            row = {
                "schema_version": SCHEMA_VERSION,
                "pilot_id": pilot_id,
                "market_ticker": label.get("market_ticker"),
                "open_ts_ns": open_ns,
                "close_ts_ns": close_ns,
                "target_price": str(target),
                "expiration_value": str(expiration),
                "result": str(label.get("result", "")).lower(),
                "kalshi_reported_delta": str(expiration - target),
                "quarantined": bool(reasons),
                "quarantine_reasons": reasons,
                "primary_contract": primary_symbol,
                "secondary_contract": secondary_symbol,
                "comex_primary_last_delta": _delta(p_open, p_close),
                "comex_primary_vwap_delta": _delta(p_open_vwap, p_close_vwap),
                "comex_secondary_last_delta": _delta(s_open, s_close),
                "comex_secondary_vwap_delta": _delta(s_open_vwap, s_close_vwap),
                "pyth_preceding_minute_delta": _delta(pyth_pre_open, pyth_pre_close),
                "pyth_boundary_labeled_delta": _delta(
                    pyth_boundary_open, pyth_boundary_close
                ),
                "pyth_preceding_minute_open": (
                    str(pyth_pre_open) if pyth_pre_open is not None else None
                ),
                "pyth_preceding_minute_close": (
                    str(pyth_pre_close) if pyth_pre_close is not None else None
                ),
                "pyth_boundary_labeled_open": (
                    str(pyth_boundary_open) if pyth_boundary_open is not None else None
                ),
                "pyth_boundary_labeled_close": (
                    str(pyth_boundary_close)
                    if pyth_boundary_close is not None
                    else None
                ),
                "research_only": True,
                "settlement_eligible": False,
            }
            rows.append(row)
            for reason in reasons:
                anomalies.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "type": reason,
                        "market_ticker": label.get("market_ticker"),
                        "target_price": str(target),
                        "expiration_value": str(expiration),
                        "result": label.get("result"),
                    }
                )
            previous = label

        features = (
            "comex_primary_last_delta",
            "comex_primary_vwap_delta",
            "comex_secondary_last_delta",
            "comex_secondary_vwap_delta",
            "pyth_preceding_minute_delta",
            "pyth_boundary_labeled_delta",
        )
        metrics = {
            feature: _metric_summary(
                rows,
                feature,
                ambiguity=float(config.ambiguity_dollars),
                train_fraction=float(config.train_fraction),
            )
            for feature in features
        }
        pyth_level_checks = {
            "preceding_minute": _pyth_level_summary(rows, "pyth_preceding_minute"),
            "boundary_labeled": _pyth_level_summary(rows, "pyth_boundary_labeled"),
        }
        primary_metric = metrics["comex_primary_last_delta"]
        secondary_metric = metrics["comex_secondary_last_delta"]
        comex_metric = max(
            (primary_metric, secondary_metric),
            key=lambda value: (
                value["observations"],
                value["delta_correlation"]
                if value["delta_correlation"] is not None
                else -2,
            ),
        )
        pyth_metric = max(
            (
                metrics["pyth_preceding_minute_delta"],
                metrics["pyth_boundary_labeled_delta"],
            ),
            key=lambda value: (
                value["delta_correlation"]
                if value["delta_correlation"] is not None
                else -2,
                value["observations"],
            ),
        )
        expectation_checks = {
            "comex_family_minimum_50_observations": comex_metric["observations"] >= 50,
            "comex_family_delta_correlation_at_least_0_8": (
                comex_metric["delta_correlation"] is not None
                and comex_metric["delta_correlation"] >= 0.8
            ),
            "comex_family_non_ambiguous_agreement_at_least_0_8": (
                comex_metric["non_ambiguous_result_agreement"] is not None
                and comex_metric["non_ambiguous_result_agreement"] >= 0.8
            ),
            "generic_pyth_minimum_50_observations": pyth_metric["observations"] >= 50,
            "generic_pyth_delta_correlation_at_least_0_8": (
                pyth_metric["delta_correlation"] is not None
                and pyth_metric["delta_correlation"] >= 0.8
            ),
        }
        exploratory_pass = all(expectation_checks.values())
        report = {
            "schema_version": SCHEMA_VERSION,
            "pilot_id": pilot_id,
            "run_id": run_id,
            "generated_ts": datetime.now(UTC).isoformat(),
            "research_only": True,
            "settlement_eligible": False,
            "exact_settlement_certified": False,
            "formal_settlement_gate_passed": False,
            "interval": {
                "start_ts_ns": config.start_ts_ns,
                "end_ts_ns": config.end_ts_ns,
                "start_utc": datetime.fromtimestamp(
                    config.start_ts_ns / NSEC, UTC
                ).isoformat(),
                "end_utc": datetime.fromtimestamp(
                    config.end_ts_ns / NSEC, UTC
                ).isoformat(),
            },
            "config": public_config,
            "inputs": {
                name: {"path": str(path), "sha256": input_hashes[name]}
                for name, path in input_files.items()
            },
            "selected_contracts": {
                "primary": {
                    "asset": config.primary_asset.upper(),
                    "symbol": primary_symbol,
                    **primary_stats,
                },
                "secondary": {
                    "asset": config.secondary_asset.upper(),
                    "symbol": secondary_symbol,
                    **secondary_stats,
                },
                "selection_rule": "highest full-pilot-volume contract within the predeclared asset family",
            },
            "cohort": {
                "input_candidates": len(candidates),
                "included_windows": len(rows),
                "quarantined_windows": sum(row["quarantined"] for row in rows),
                "anomaly_records": len(anomalies),
            },
            "metrics": metrics,
            "pyth_level_checks": pyth_level_checks,
            "decision": {
                "exploratory_expectations_passed": exploratory_pass,
                "checks": expectation_checks,
                "summary": (
                    "The small proxy cohort behaves consistently enough to proceed with baseline research, subject to the explicit proxy limitations."
                    if exploratory_pass
                    else "At least one predeclared exploratory expectation failed; inspect alignment, coverage, and anomalies before using this cohort for baseline research."
                ),
                "formal_gate": "closed_exact_pyth_and_reviewed_settlement_semantics_required",
                "diagnostics": {
                    "primary_si_observations": primary_metric["observations"],
                    "secondary_sil_observations": secondary_metric["observations"],
                    "comex_evidence_metric": (
                        "primary_si"
                        if comex_metric is primary_metric
                        else "secondary_sil"
                    ),
                },
            },
            "limitations": [
                "Kalshi target and expiration values are exchange-reported labels, not independently reconstructed exact Pyth observations.",
                "Metal.XAG/USD is a generic one-minute research proxy and is not the Metal.Index.XAG/USD settlement feed.",
                "Generic Pyth candle timestamp semantics are unconfirmed, so both plausible minute alignments are reported.",
                "COMEX contract selection is outcome-blind but uses full-pilot volume and is not a live roll policy.",
                "Last-trade and one-second VWAP comparisons can differ from Kalshi sampling, averaging, and rounding rules.",
                "This one-day pilot checks expectations; it is not statistical proof over regimes.",
            ],
            "artifacts": {
                "rows": str(rows_path),
                "anomalies": str(anomalies_path),
                "report": str(report_path),
                "markdown": str(markdown_path),
                "manifest": str(manifest_path),
            },
        }

        output_dir.mkdir(parents=True, exist_ok=False)
        with HashChainedJsonlWriter(rows_path, fsync_every_records=100) as writer:
            for row in rows:
                writer.write(row)
        with HashChainedJsonlWriter(anomalies_path, fsync_every_records=100) as writer:
            for anomaly in anomalies:
                writer.write(anomaly)
        _write_json(report_path, report)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        update_run_manifest(
            root,
            run_id,
            status="completed",
            counts={
                "windows": len(rows),
                "quarantined_windows": sum(row["quarantined"] for row in rows),
                "anomalies": len(anomalies),
                "exploratory_expectations_passed": exploratory_pass,
                "formal_settlement_gate_passed": False,
            },
            output_paths=(rows_path, anomalies_path, report_path, markdown_path),
        )
        return report
    except BaseException as error:
        update_run_manifest(
            root,
            run_id,
            status="failed",
            errors=(error,),
        )
        raise
