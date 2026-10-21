"""Exploratory high-frequency signal diagnostics for one synchronized capture.

This module keeps external-fair and execution features separate.  It evaluates
one signal at a time against its natural short-horizon target and deliberately
does not modify the frozen fair model.  A single capture cannot support model
admission, probability claims, or a trading decision; the output is a set of
reproducible diagnostics and signal cards for prioritizing future collection.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .databento_mbo import (
    MboBboSnapshot,
    iter_hash_chained_jsonl,
    reconstruct_mbo_bbo,
    select_active_outright_contract,
)
from .fast_feed_counterfactual import (
    ExactPrice,
    FillObservation,
    _build_fill_observations,
    _load_proxy_rows,
)
from .forward_shadow_report import _stream_rows
from .si_fast_feed_counterfactual import _input_base

NSEC = 1_000_000_000
MSEC = 1_000_000
SCHEMA_VERSION = "metals-high-frequency-signal-research/v1"
DISPOSITION = "exploratory_only_no_model_admission"


@dataclass(frozen=True)
class KalshiBookState:
    ts_ns: int
    market_ticker: str
    midpoint: float
    spread: float
    top_imbalance: float
    yes_bid: float
    yes_ask: float


@dataclass(frozen=True)
class SignedEvent:
    ts_ns: int
    market_ticker: str
    value: float


@dataclass(frozen=True)
class SignalSpec:
    signal_id: str
    family: str
    target_kind: str
    expected_sign: str
    formula: str
    data_source_and_latency: str
    expected_mechanism: str
    effective_horizon: str
    failure_conditions: tuple[str, ...]


class HighFrequencySignalError(ValueError):
    """Raised when a high-frequency research artifact is not reproducible."""


def _gap_summary(timestamps: Sequence[int]) -> dict[str, float | int | None]:
    gaps = sorted(
        (right - left) / MSEC
        for left, right in zip(timestamps, timestamps[1:])
        if right > left
    )
    if not gaps:
        return {"intervals": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}

    def pick(fraction: float) -> float:
        index = max(0, min(len(gaps) - 1, math.ceil(len(gaps) * fraction) - 1))
        return gaps[index]

    return {
        "intervals": len(gaps),
        "p50_ms": pick(0.50),
        "p95_ms": pick(0.95),
        "max_ms": gaps[-1],
    }


def _latency_summary(latencies_ms: Sequence[float]) -> dict[str, float | int | None]:
    values = sorted(value for value in latencies_ms if math.isfinite(value))
    if not values:
        return {"observations": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}

    def pick(fraction: float) -> float:
        index = max(0, min(len(values) - 1, math.ceil(len(values) * fraction) - 1))
        return values[index]

    return {
        "observations": len(values),
        "p50_ms": pick(0.50),
        "p95_ms": pick(0.95),
        "max_ms": values[-1],
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_ss = sum((value - x_mean) ** 2 for value in xs)
    y_ss = sum((value - y_mean) ** 2 for value in ys)
    if x_ss <= 0 or y_ss <= 0:
        return None
    return sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / math.sqrt(x_ss * y_ss)


def _rank(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        tied_rank = (start + end - 1) / 2
        for position in range(start, end):
            ranks[ordered[position]] = tied_rank
        start = end
    return ranks


def _diagnostic(
    observations: Sequence[tuple[float, float, str]],
    *,
    target_unit: str,
) -> dict[str, Any]:
    clean = [row for row in observations if all(math.isfinite(v) for v in row[:2])]
    if len(clean) < 20:
        return {
            "observations": len(clean),
            "pearson": None,
            "spearman": None,
            "top_minus_bottom_quintile": None,
            "target_unit": target_unit,
            "window_correlations": [],
            "same_sign_window_fraction": None,
        }
    xs = [row[0] for row in clean]
    ys = [row[1] for row in clean]
    order = sorted(range(len(xs)), key=xs.__getitem__)
    quintile = max(1, len(order) // 5)
    bottom = mean(ys[index] for index in order[:quintile])
    top = mean(ys[index] for index in order[-quintile:])
    correlation = _pearson(xs, ys)
    by_window: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for x, y, window in clean:
        by_window[window].append((x, y))
    window_correlations: list[dict[str, Any]] = []
    signs: list[int] = []
    for window, rows in sorted(by_window.items()):
        if len(rows) < 10:
            continue
        value = _pearson([row[0] for row in rows], [row[1] for row in rows])
        window_correlations.append(
            {"market_ticker": window, "observations": len(rows), "pearson": value}
        )
        if value is not None and value != 0:
            signs.append(1 if value > 0 else -1)
    same_sign = None
    if correlation not in (None, 0) and signs:
        expected = 1 if correlation > 0 else -1
        same_sign = sum(sign == expected for sign in signs) / len(signs)
    return {
        "observations": len(clean),
        "pearson": correlation,
        "spearman": _pearson(_rank(xs), _rank(ys)),
        "bottom_quintile_target_mean": bottom,
        "top_quintile_target_mean": top,
        "top_minus_bottom_quintile": top - bottom,
        "target_unit": target_unit,
        "window_correlations": window_correlations,
        "same_sign_window_fraction": same_sign,
    }


def _asof_index(timestamps: Sequence[int], timestamp_ns: int) -> int | None:
    index = bisect.bisect_right(timestamps, timestamp_ns) - 1
    return index if index >= 0 else None


def _asof_value(
    timestamps: Sequence[int], values: Sequence[float], timestamp_ns: int
) -> float | None:
    index = _asof_index(timestamps, timestamp_ns)
    return None if index is None else values[index]


def _future_change_bps(
    timestamps: Sequence[int],
    values: Sequence[float],
    timestamp_ns: int,
    horizon_ns: int,
) -> float | None:
    start = _asof_index(timestamps, timestamp_ns)
    end = _asof_index(timestamps, timestamp_ns + horizon_ns)
    if start is None or end is None or end <= start or values[start] <= 0:
        return None
    return 10_000 * math.log(values[end] / values[start])


def _past_change_bps(
    timestamps: Sequence[int],
    values: Sequence[float],
    timestamp_ns: int,
    lookback_ns: int,
) -> float | None:
    end = _asof_index(timestamps, timestamp_ns)
    start = _asof_index(timestamps, timestamp_ns - lookback_ns)
    if start is None or end is None or end <= start or values[start] <= 0:
        return None
    return 10_000 * math.log(values[end] / values[start])


def _path_features(
    timestamps: Sequence[int], values: Sequence[float], timestamp_ns: int, lookback_ns: int
) -> dict[str, float] | None:
    start = bisect.bisect_left(timestamps, timestamp_ns - lookback_ns)
    end = bisect.bisect_right(timestamps, timestamp_ns)
    section = values[start:end]
    if len(section) < 3 or section[0] <= 0:
        return None
    increments = [10_000 * math.log(right / left) for left, right in zip(section, section[1:])]
    absolute_path = sum(abs(value) for value in increments)
    range_bps = 10_000 * math.log(max(section) / min(section))
    net = sum(increments)
    positive = sum(value * value for value in increments if value > 0)
    negative = sum(value * value for value in increments if value < 0)
    total_semivariance = positive + negative
    rms = math.sqrt(sum(value * value for value in increments) / len(increments))
    return {
        "pyth_abs_path_1s": absolute_path,
        "pyth_range_1s": range_bps,
        "pyth_path_efficiency_1s": abs(net) / absolute_path if absolute_path else 0.0,
        "pyth_down_semivariance_share_1s": negative / total_semivariance if total_semivariance else 0.5,
        "pyth_jump_ratio_1s": max(abs(value) for value in increments) / rms if rms else 0.0,
    }


def reconstruct_kalshi_book(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[KalshiBookState], list[SignedEvent], dict[str, Any]]:
    """Reconstruct valid per-market BBO states and signed displayed-size changes."""

    books: dict[str, dict[str, dict[float, float]]] = defaultdict(
        lambda: {"yes": {}, "no": {}}
    )
    snapshot_keys: set[tuple[str, str]] = set()
    states: list[KalshiBookState] = []
    pressure: list[SignedEvent] = []
    last_state: dict[str, tuple[float, float, float, float, float]] = {}
    rows_read = 0
    invalid_rows = 0
    negative_levels = 0
    for row in rows:
        rows_read += 1
        ticker = str(row.get("market_ticker", ""))
        side = str(row.get("outcome_side", ""))
        message_type = str(row.get("message_type", ""))
        if not ticker or side not in {"yes", "no"}:
            continue
        book = books[ticker]
        if message_type == "snapshot":
            recovery_id = str(row.get("recovery_snapshot_id", ""))
            key = (ticker, recovery_id)
            if key not in snapshot_keys:
                book["yes"].clear()
                book["no"].clear()
                snapshot_keys.add(key)
            quantity = float(row.get("quantity_fp") or 0)
            delta = None
        elif message_type == "delta":
            quantity = None
            delta = float(row.get("quantity_delta_fp") or 0)
        else:
            continue
        price = float(row.get("price_dollars") or 0)
        if not 0 < price < 1:
            continue
        if quantity is not None:
            if quantity > 0:
                book[side][price] = quantity
            else:
                book[side].pop(price, None)
        else:
            next_quantity = book[side].get(price, 0.0) + float(delta)
            if next_quantity < -1e-7:
                negative_levels += 1
                next_quantity = 0.0
            if next_quantity > 1e-7:
                book[side][price] = next_quantity
            else:
                book[side].pop(price, None)
            timestamp_ns = int(row.get("receive_ts_ns") or 0)
            if timestamp_ns:
                direction = 1.0 if side == "yes" else -1.0
                signed = direction * math.copysign(math.log1p(abs(float(delta))), float(delta))
                pressure.append(SignedEvent(timestamp_ns, ticker, signed))
        if not bool(row.get("book_valid_after", False)):
            invalid_rows += 1
            continue
        if not book["yes"] or not book["no"]:
            continue
        yes_bid = max(book["yes"])
        no_bid = max(book["no"])
        yes_ask = 1 - no_bid
        if yes_bid >= yes_ask:
            continue
        yes_size = book["yes"][yes_bid]
        ask_size = book["no"][no_bid]
        total = yes_size + ask_size
        imbalance = (yes_size - ask_size) / total if total else 0.0
        state = (
            yes_bid,
            yes_ask,
            yes_size,
            ask_size,
            imbalance,
        )
        if state == last_state.get(ticker):
            continue
        last_state[ticker] = state
        states.append(
            KalshiBookState(
                ts_ns=int(row.get("receive_ts_ns") or 0),
                market_ticker=ticker,
                midpoint=(yes_bid + yes_ask) / 2,
                spread=yes_ask - yes_bid,
                top_imbalance=imbalance,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
            )
        )
    states.sort(key=lambda row: row.ts_ns)
    pressure.sort(key=lambda row: row.ts_ns)
    return states, pressure, {
        "input_rows": rows_read,
        "valid_bbo_changes": len(states),
        "signed_delta_events": len(pressure),
        "invalid_rows": invalid_rows,
        "negative_level_repairs": negative_levels,
    }


def _active_market(markets: Sequence[Mapping[str, Any]], timestamp_ns: int) -> str | None:
    candidates = [
        row
        for row in markets
        if int(row.get("open_ts_ns") or 0) <= timestamp_ns
        < int(row.get("close_ts_ns") or 0)
    ]
    if not candidates:
        return None
    return str(max(candidates, key=lambda row: int(row["open_ts_ns"]))["market_ticker"])


def _group_timed_values(
    rows: Sequence[KalshiBookState], field: str
) -> dict[str, tuple[list[int], list[float]]]:
    grouped: dict[str, tuple[list[int], list[float]]] = {}
    temporary: dict[str, list[KalshiBookState]] = defaultdict(list)
    for row in rows:
        temporary[row.market_ticker].append(row)
    for ticker, values in temporary.items():
        grouped[ticker] = (
            [row.ts_ns for row in values],
            [float(getattr(row, field)) for row in values],
        )
    return grouped


def _group_signed_events(
    rows: Sequence[SignedEvent],
) -> dict[str, tuple[list[int], list[float], list[float]]]:
    grouped_rows: dict[str, list[SignedEvent]] = defaultdict(list)
    for row in rows:
        grouped_rows[row.market_ticker].append(row)
    grouped: dict[str, tuple[list[int], list[float], list[float]]] = {}
    for ticker, values in grouped_rows.items():
        times: list[int] = []
        event_values: list[float] = []
        prefix = [0.0]
        for row in values:
            times.append(row.ts_ns)
            event_values.append(row.value)
            prefix.append(prefix[-1] + row.value)
        grouped[ticker] = times, event_values, prefix
    return grouped


def _interval_sum(
    grouped: Mapping[str, tuple[list[int], list[float], list[float]]],
    ticker: str,
    start_ns: int,
    end_ns: int,
) -> float | None:
    series = grouped.get(ticker)
    if series is None:
        return None
    times, _, prefix = series
    left = bisect.bisect_right(times, start_ns)
    right = bisect.bisect_right(times, end_ns)
    return prefix[right] - prefix[left]


def _trade_events(rows: Sequence[Mapping[str, Any]]) -> list[SignedEvent]:
    events: list[SignedEvent] = []
    for row in rows:
        timestamp_ns = int(row.get("receive_ts_ns") or 0)
        quantity = float(row.get("quantity_fp") or 0)
        outcome = str(row.get("taker_outcome_side", ""))
        book_side = str(row.get("taker_book_side", ""))
        if not timestamp_ns or quantity <= 0 or outcome not in {"yes", "no"}:
            continue
        outcome_direction = 1.0 if outcome == "yes" else -1.0
        aggressor_direction = 1.0 if book_side == "ask" else -1.0
        events.append(
            SignedEvent(
                timestamp_ns,
                str(row.get("market_ticker", "")),
                outcome_direction * aggressor_direction * quantity,
            )
        )
    return sorted(events, key=lambda row: row.ts_ns)


def _si_trade_series(
    rows: Sequence[Mapping[str, Any]], symbol: str, availability_latency_ns: int
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    selected: list[tuple[int, float, float, float]] = []
    for row in rows:
        if str(row.get("contract_symbol", "")) != symbol:
            continue
        timestamp_ns = int(row.get("ts_recv") or row.get("source_publish_ts_ns") or 0)
        price = float(row.get("price") or 0)
        size = float(row.get("size") or 0)
        side = str(row.get("side", ""))
        if timestamp_ns and price > 0 and size > 0:
            signed = size if side == "B" else -size if side == "A" else 0.0
            selected.append((timestamp_ns + availability_latency_ns, price, size, signed))
    selected.sort()
    times: list[int] = []
    prices: list[float] = []
    volume_prefix = [0.0]
    signed_prefix = [0.0]
    for timestamp_ns, price, size, signed in selected:
        times.append(timestamp_ns)
        prices.append(price)
        volume_prefix.append(volume_prefix[-1] + size)
        signed_prefix.append(signed_prefix[-1] + signed)
    return times, prices, volume_prefix, signed_prefix, [1.0] * len(times)


def _prefix_interval(
    timestamps: Sequence[int], prefix: Sequence[float], start_ns: int, end_ns: int
) -> float:
    left = bisect.bisect_right(timestamps, start_ns)
    right = bisect.bisect_right(timestamps, end_ns)
    return prefix[right] - prefix[left]


def _signal_specs() -> list[SignalSpec]:
    common = (
        "One synchronized night cannot establish stability across days or regimes.",
        "Overlapping market windows and asynchronous vendor timestamps can inflate apparent precision.",
    )
    return [
        SignalSpec("pyth_return_200ms", "fast_spot_returns", "signed", "positive", "10,000 * log(XAG_t / XAG_{t-200ms})", "Pyth Metal.XAG/USD fixed_rate@200ms; 100 ms transport scenario", "very recent spot movement may continue before the settlement-index update", "0.2-1 seconds", common),
        SignalSpec("pyth_return_1s", "fast_spot_returns", "signed", "positive", "10,000 * log(XAG_t / XAG_{t-1s})", "Pyth Metal.XAG/USD fixed_rate@200ms; 100 ms transport scenario", "one-second spot momentum may update fair before the slow exact feed", "0.2-5 seconds", common),
        SignalSpec("si_return_200ms", "fast_spot_returns", "signed", "positive", "10,000 * log(SI_mid_t / SI_mid_{t-200ms})", "Databento GLBX.MDP3 SI MBO; 25 ms Databento-to-host scenario", "centralized futures price discovery may lead composite spot and the exact index", "0.2-1 seconds", common + ("Futures-to-index basis can change around thin liquidity or contract rolls.",)),
        SignalSpec("si_return_1s", "fast_spot_returns", "signed", "positive", "10,000 * log(SI_mid_t / SI_mid_{t-1s})", "Databento GLBX.MDP3 SI MBO; 25 ms Databento-to-host scenario", "one-second futures momentum may carry into Pyth", "0.2-5 seconds", common + ("Futures-to-index basis can change around thin liquidity or contract rolls.",)),
        SignalSpec("si_xag_residual_200ms", "fast_spot_returns", "signed", "positive", "SI 200ms return minus XAG 200ms return", "Databento MBO plus Pyth XAG under declared latency scenarios", "a futures move not yet reflected in XAG may predict the next XAG update", "0.2-1 seconds", common + ("Clock or transport misspecification can manufacture a lead-lag residual.",)),
        SignalSpec("pyth_abs_path_1s", "range_volatility_path_efficiency", "absolute", "positive", "sum of absolute 200ms XAG log returns over 1s", "Pyth Metal.XAG/USD fixed_rate@200ms; 100 ms transport scenario", "recent path variation may persist as near-term variance", "1-5 seconds", common),
        SignalSpec("pyth_range_1s", "range_volatility_path_efficiency", "absolute", "positive", "10,000 * log(max XAG / min XAG) over 1s", "Pyth Metal.XAG/USD fixed_rate@200ms; 100 ms transport scenario", "a wide recent range indicates elevated information arrival", "1-5 seconds", common),
        SignalSpec("pyth_path_efficiency_1s", "range_volatility_path_efficiency", "absolute", "positive", "absolute net XAG move divided by absolute XAG path over 1s", "Pyth Metal.XAG/USD fixed_rate@200ms; 100 ms transport scenario", "efficient directional paths may be more persistent than oscillation", "0.2-2 seconds", common),
        SignalSpec("si_book_imbalance", "futures_activity_and_impact", "signed", "positive", "(SI best-bid size - best-ask size)/(sum of top sizes)", "Databento GLBX.MDP3 SI MBO; 25 ms Databento-to-host scenario", "top-of-book pressure may precede a futures and spot move", "0.2-1 seconds", common + ("Displayed size can be cancelled and may not represent executable intent.",)),
        SignalSpec("si_microprice_basis_bps", "futures_activity_and_impact", "signed", "positive", "10,000 * (SI microprice - midpoint)/midpoint", "Databento GLBX.MDP3 SI MBO; 25 ms Databento-to-host scenario", "size-weighted microprice summarizes near-touch pressure", "0.2-1 seconds", common + ("Displayed size can be cancelled and may not represent executable intent.",)),
        SignalSpec("si_trade_intensity_1s", "futures_activity_and_impact", "absolute", "positive", "log(1 + SI trades during trailing 1s)", "Databento GLBX.MDP3 SI trades; 25 ms Databento-to-host scenario", "rapid information arrival raises immediate variance", "1-5 seconds", common),
        SignalSpec("si_volume_1s", "futures_activity_and_impact", "absolute", "positive", "log(1 + SI contracts traded during trailing 1s)", "Databento GLBX.MDP3 SI trades; 25 ms Databento-to-host scenario", "heavy futures turnover may indicate persistent high variance", "1-5 seconds", common),
        SignalSpec("si_signed_trade_imbalance_1s", "futures_activity_and_impact", "signed", "positive", "signed SI aggressor volume / total SI volume over 1s", "Databento GLBX.MDP3 SI trades; 25 ms Databento-to-host scenario", "aggressive futures flow may precede the external spot mark", "0.2-2 seconds", common + ("Aggressor-side coding and contract selection must remain stable.",)),
        SignalSpec("pyth_down_semivariance_share_1s", "semivariance_and_jumps", "down_tail", "positive", "negative squared XAG returns divided by total squared XAG returns over 1s", "Pyth Metal.XAG/USD fixed_rate@200ms; 100 ms transport scenario", "recent downside-dominated variation may raise the next downside-tail probability", "1-5 seconds", common),
        SignalSpec("pyth_jump_ratio_1s", "semivariance_and_jumps", "absolute", "positive", "largest absolute 200ms return / RMS 200ms return over 1s", "Pyth Metal.XAG/USD fixed_rate@200ms; 100 ms transport scenario", "a recent discontinuity may indicate a short-lived jump regime", "1-5 seconds", common),
        SignalSpec("kalshi_top_imbalance", "kalshi_execution_only", "kalshi_signed", "positive", "(YES best-bid size - YES best-ask size)/(sum of top sizes)", "Recorded Kalshi WebSocket book at local receive time", "displayed binary-book pressure may predict short markout and fill selection", "0.2-5 seconds", common + ("This feature is prohibited from the external fair model.",)),
        SignalSpec("kalshi_trade_flow_1s", "kalshi_execution_only", "kalshi_signed", "positive", "signed Kalshi taker quantity over trailing 1s", "Recorded Kalshi WebSocket trades at local receive time", "aggressive binary flow may continue and select stale passive quotes", "0.2-5 seconds", common + ("This feature is prohibited from the external fair model.",)),
        SignalSpec("kalshi_book_pressure_1s", "kalshi_execution_only", "kalshi_signed", "positive", "signed log displayed-size changes over trailing 1s", "Recorded Kalshi WebSocket book deltas at local receive time", "near-real-time additions and cancellations may reveal quote pressure", "0.2-2 seconds", common + ("This feature is prohibited from the external fair model.", "Deep-level updates can dilute top-of-book pressure.")),
    ]


def _latest_bbo(
    bbo_times: Sequence[int], bbo: Sequence[MboBboSnapshot], timestamp_ns: int
) -> MboBboSnapshot | None:
    index = _asof_index(bbo_times, timestamp_ns)
    return None if index is None else bbo[index]


def _fill_conditioned_diagnostics(
    fills: Sequence[FillObservation],
    *,
    feature_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    feature_times = [int(row["ts_ns"]) for row in feature_rows]
    fields = (
        "pyth_return_200ms",
        "si_return_200ms",
        "si_book_imbalance",
        "kalshi_top_imbalance",
        "kalshi_trade_flow_1s",
        "kalshi_book_pressure_1s",
    )
    results: list[dict[str, Any]] = []
    for field in fields:
        for horizon in (1, 5):
            observations: list[tuple[float, float, str]] = []
            for fill in fills:
                index = _asof_index(feature_times, fill.fill_ts_ns)
                if index is None:
                    continue
                row = feature_rows[index]
                if row.get("market_ticker") != fill.market_ticker:
                    continue
                value = row.get(field)
                markout = fill.markout_1s if horizon == 1 else fill.markout_5s
                if value is None or markout is None:
                    continue
                side_direction = 1.0 if fill.side == "yes" else -1.0
                observations.append(
                    (side_direction * float(value), float(markout) * 100, fill.market_ticker)
                )
            results.append(
                {
                    "signal_id": field,
                    "orientation": "positive means movement/pressure in favor of the filled passive outcome",
                    "target": f"fill_conditioned_{horizon}s_markout",
                    **_diagnostic(observations, target_unit="cents_per_contract"),
                }
            )
    return results


def build_high_frequency_signal_report(
    root: str | Path,
    collector_run_id: str,
    shadow_run_id: str,
    *,
    databento_request_id: str,
    metal: str = "silver",
    pyth_proxy_metal: str = "silver-xag-proxy",
    pyth_feed_id: int = 345,
    pyth_transport_latency_ms: int = 100,
    databento_to_host_latency_ms: int = 25,
) -> dict[str, Any]:
    storage_root = Path(root)
    market_rows = _stream_rows(storage_root, "market_dim", collector_run_id, metal=metal)
    exact_rows = _stream_rows(storage_root, "kalshi_pyth_value", collector_run_id, metal=metal)
    book_rows = _stream_rows(storage_root, "kalshi_book_event", collector_run_id, metal=metal)
    kalshi_trade_rows = _stream_rows(storage_root, "kalshi_trade", collector_run_id, metal=metal)
    fair_rows = _stream_rows(storage_root, "forward_fair_decision", shadow_run_id, metal=metal)
    fill_rows = _stream_rows(storage_root, "shadow_fill_attribution", shadow_run_id, metal=metal)
    lifecycle_rows = _stream_rows(storage_root, "order_lifecycle", shadow_run_id, metal=metal)
    exact = sorted(
        (
            ExactPrice(
                int(row["source_ts_ns"]),
                int(row["receive_ts_ns"]),
                Decimal(str(row["value_usd"])),
            )
            for row in exact_rows
        ),
        key=lambda row: row.receive_ts_ns,
    )
    if not exact:
        raise HighFrequencySignalError("capture has no exact index observations")
    start_ts_ns = min(row.source_ts_ns for row in exact)
    end_ts_ns = max(row.source_ts_ns for row in exact)
    proxy_rows, pyth_request_ids = _load_proxy_rows(
        storage_root,
        proxy_metal=pyth_proxy_metal,
        proxy_feed_id=pyth_feed_id,
        start_ts_ns=start_ts_ns,
        end_ts_ns=end_ts_ns,
    )
    pyth_latency_ns = pyth_transport_latency_ms * MSEC
    pyth_times = [row.publish_ts_ns + pyth_latency_ns for row in proxy_rows]
    pyth_values = [float(row.value) for row in proxy_rows]
    exact_times = [row.receive_ts_ns for row in exact]
    exact_values = [float(row.value) for row in exact]

    input_base = _input_base(storage_root, databento_request_id[:24], metal)
    databento_trade_rows = list(iter_hash_chained_jsonl([input_base / "trades.jsonl"]))
    selected = select_active_outright_contract(databento_trade_rows, asset="SI")
    reconstruction = reconstruct_mbo_bbo(
        iter_hash_chained_jsonl([input_base / "mbo.jsonl"]),
        instrument_id=int(selected["instrument_id"]),
        contract_symbol=str(selected["contract_symbol"]),
    )
    si_latency_ns = databento_to_host_latency_ms * MSEC
    si_bbo = [
        row
        for row in reconstruction.snapshots
        if start_ts_ns <= row.ts_recv_ns <= end_ts_ns
    ]
    si_bbo_times = [row.ts_recv_ns + si_latency_ns for row in si_bbo]
    si_midpoints = [float(row.midpoint) for row in si_bbo]
    si_trade_times, _, si_volume_prefix, si_signed_prefix, _ = _si_trade_series(
        databento_trade_rows, str(selected["contract_symbol"]), si_latency_ns
    )

    kalshi_states, kalshi_pressure, book_stats = reconstruct_kalshi_book(book_rows)
    kalshi_mid = _group_timed_values(kalshi_states, "midpoint")
    kalshi_imbalance = _group_timed_values(kalshi_states, "top_imbalance")
    kalshi_pressure_series = _group_signed_events(kalshi_pressure)
    kalshi_trade_events = _trade_events(kalshi_trade_rows)
    kalshi_trade_series = _group_signed_events(kalshi_trade_events)

    markets_by_ticker: dict[str, Mapping[str, Any]] = {}
    for row in market_rows:
        markets_by_ticker[str(row["market_ticker"])] = row
    markets = sorted(markets_by_ticker.values(), key=lambda row: int(row["open_ts_ns"]))
    feature_rows: list[dict[str, Any]] = []
    for timestamp_ns, pyth_price in zip(pyth_times, pyth_values):
        ticker = _active_market(markets, timestamp_ns)
        if ticker is None:
            continue
        kalshi_mid_series = kalshi_mid.get(ticker)
        kalshi_imbalance_series = kalshi_imbalance.get(ticker)
        if kalshi_mid_series is None or kalshi_imbalance_series is None:
            continue
        midpoint = _asof_value(*kalshi_mid_series, timestamp_ns)
        top_imbalance = _asof_value(*kalshi_imbalance_series, timestamp_ns)
        si_snapshot = _latest_bbo(si_bbo_times, si_bbo, timestamp_ns)
        if midpoint is None or top_imbalance is None or si_snapshot is None:
            continue
        pyth_200 = _past_change_bps(pyth_times, pyth_values, timestamp_ns, 200 * MSEC)
        pyth_1s = _past_change_bps(pyth_times, pyth_values, timestamp_ns, NSEC)
        si_200 = _past_change_bps(si_bbo_times, si_midpoints, timestamp_ns, 200 * MSEC)
        si_1s = _past_change_bps(si_bbo_times, si_midpoints, timestamp_ns, NSEC)
        path = _path_features(pyth_times, pyth_values, timestamp_ns, NSEC)
        if None in (pyth_200, pyth_1s, si_200, si_1s) or path is None:
            continue
        trade_count_1s = bisect.bisect_right(si_trade_times, timestamp_ns) - bisect.bisect_right(si_trade_times, timestamp_ns - NSEC)
        si_volume = _prefix_interval(si_trade_times, si_volume_prefix, timestamp_ns - NSEC, timestamp_ns)
        si_signed = _prefix_interval(si_trade_times, si_signed_prefix, timestamp_ns - NSEC, timestamp_ns)
        current_mid = float(si_snapshot.midpoint)
        feature_rows.append(
            {
                "ts_ns": timestamp_ns,
                "market_ticker": ticker,
                "pyth_price": pyth_price,
                "pyth_return_200ms": pyth_200,
                "pyth_return_1s": pyth_1s,
                "si_return_200ms": si_200,
                "si_return_1s": si_1s,
                "si_xag_residual_200ms": float(si_200) - float(pyth_200),
                **path,
                "si_book_imbalance": float(si_snapshot.imbalance),
                "si_microprice_basis_bps": 10_000 * (float(si_snapshot.microprice) - current_mid) / current_mid,
                "si_trade_intensity_1s": math.log1p(trade_count_1s),
                "si_volume_1s": math.log1p(si_volume),
                "si_signed_trade_imbalance_1s": si_signed / si_volume if si_volume else 0.0,
                "kalshi_midpoint": midpoint,
                "kalshi_top_imbalance": top_imbalance,
                "kalshi_trade_flow_1s": _interval_sum(kalshi_trade_series, ticker, timestamp_ns - NSEC, timestamp_ns) or 0.0,
                "kalshi_book_pressure_1s": _interval_sum(kalshi_pressure_series, ticker, timestamp_ns - NSEC, timestamp_ns) or 0.0,
            }
        )

    if len(feature_rows) < 100:
        raise HighFrequencySignalError("fewer than 100 synchronized feature rows")
    specs = _signal_specs()
    diagnostics: list[dict[str, Any]] = []
    horizons_ms = (200, 1000, 5000)
    for spec in specs:
        for horizon_ms in horizons_ms:
            # Non-overlapping sampling avoids counting the same target move many times.
            stride = max(1, math.ceil(horizon_ms / 200))
            observations: list[tuple[float, float, str]] = []
            for row in feature_rows[::stride]:
                timestamp_ns = int(row["ts_ns"])
                ticker = str(row["market_ticker"])
                x = float(row[spec.signal_id])
                if spec.target_kind == "kalshi_signed":
                    series = kalshi_mid.get(ticker)
                    if series is None:
                        continue
                    current = _asof_value(*series, timestamp_ns)
                    future = _asof_value(*series, timestamp_ns + horizon_ms * MSEC)
                    if current is None or future is None:
                        continue
                    y = 100 * (future - current)
                    unit = "cents"
                    target = "future_kalshi_mid_change"
                else:
                    future_return = _future_change_bps(
                        pyth_times, pyth_values, timestamp_ns, horizon_ms * MSEC
                    )
                    if future_return is None:
                        continue
                    if spec.target_kind == "signed":
                        y = future_return
                        unit = "basis_points"
                        target = "future_xag_return"
                    elif spec.target_kind == "absolute":
                        y = abs(future_return)
                        unit = "absolute_basis_points"
                        target = "future_xag_absolute_return"
                    elif spec.target_kind == "down_tail":
                        # A fixed move threshold is descriptive and declared, not fitted.
                        y = 1.0 if future_return <= -0.5 else 0.0
                        unit = "down_tail_frequency"
                        target = "future_xag_down_move_at_least_0.5bp"
                    else:
                        raise HighFrequencySignalError(f"unknown target kind {spec.target_kind}")
                observations.append((x, y, ticker))
            diagnostics.append(
                {
                    "signal_id": spec.signal_id,
                    "family": spec.family,
                    "target": target,
                    "horizon_ms": horizon_ms,
                    **_diagnostic(observations, target_unit=unit),
                }
            )

    exact_diagnostics: list[dict[str, Any]] = []
    for spec in specs:
        if spec.target_kind == "kalshi_signed":
            continue
        for horizon_ms in (1000, 5000):
            stride = max(1, math.ceil(horizon_ms / 200))
            observations = []
            for row in feature_rows[::stride]:
                future_return = _future_change_bps(
                    exact_times,
                    exact_values,
                    int(row["ts_ns"]),
                    horizon_ms * MSEC,
                )
                if future_return is None:
                    continue
                if spec.target_kind == "signed":
                    target_value = future_return
                    unit = "basis_points"
                    target = "future_exact_index_return"
                elif spec.target_kind == "absolute":
                    target_value = abs(future_return)
                    unit = "absolute_basis_points"
                    target = "future_exact_index_absolute_return"
                else:
                    target_value = 1.0 if future_return <= -0.5 else 0.0
                    unit = "down_tail_frequency"
                    target = "future_exact_index_down_move_at_least_0.5bp"
                observations.append(
                    (
                        float(row[spec.signal_id]),
                        target_value,
                        str(row["market_ticker"]),
                    )
                )
            exact_diagnostics.append(
                {
                    "signal_id": spec.signal_id,
                    "family": spec.family,
                    "target": target,
                    "horizon_ms": horizon_ms,
                    **_diagnostic(observations, target_unit=unit),
                }
            )

    fills = _build_fill_observations(
        fill_rows=fill_rows,
        trade_rows=kalshi_trade_rows,
        lifecycle_rows=lifecycle_rows,
        fair_rows=fair_rows,
    )
    fill_diagnostics = _fill_conditioned_diagnostics(fills, feature_rows=feature_rows)
    cards: list[dict[str, Any]] = []
    for spec in specs:
        results = [row for row in diagnostics if row["signal_id"] == spec.signal_id]
        exact_results = [
            row for row in exact_diagnostics if row["signal_id"] == spec.signal_id
        ]
        cards.append(
            {
                "signal_id": spec.signal_id,
                "family": spec.family,
                "formula": spec.formula,
                "data_source_and_latency": spec.data_source_and_latency,
                "expected_sign": spec.expected_sign,
                "expected_mechanism": spec.expected_mechanism,
                "effective_horizon_and_decay": spec.effective_horizon,
                "correlation_with_existing_signals": "not estimated robustly from one capture; inspect per-signal diagnostics only",
                "natural_target_diagnostics": results + exact_results,
                "out_of_sample_probability_improvement": None,
                "out_of_sample_net_ev_improvement": None,
                "failure_conditions": list(spec.failure_conditions),
                "disposition": DISPOSITION,
            }
        )

    book_times = [row.ts_ns for row in kalshi_states if start_ts_ns <= row.ts_ns <= end_ts_ns]
    exact_latencies = [(row.receive_ts_ns - row.source_ts_ns) / MSEC for row in exact]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "collector_run_id": collector_run_id,
        "shadow_run_id": shadow_run_id,
        "metal": metal,
        "execution_mode": "research_only",
        "trading_eligible": False,
        "disposition": DISPOSITION,
        "declared_latency_scenario": {
            "pyth_transport_latency_ms": pyth_transport_latency_ms,
            "databento_to_host_latency_ms": databento_to_host_latency_ms,
            "kalshi_book": "recorded local receive timestamp",
            "kalshi_exact_index": "recorded local receive timestamp",
        },
        "coverage": {
            "start_ts_ns": start_ts_ns,
            "end_ts_ns": end_ts_ns,
            "duration_seconds": (end_ts_ns - start_ts_ns) / NSEC,
            "market_windows_observed": len({row["market_ticker"] for row in feature_rows}),
            "synchronized_feature_rows": len(feature_rows),
            "pyth_xag_updates": len(proxy_rows),
            "pyth_request_ids": pyth_request_ids,
            "exact_index_updates": len(exact),
            "si_bbo_changes": len(si_bbo),
            "si_trade_records": len(si_trade_times),
            "kalshi_book": book_stats,
            "kalshi_trade_records": len(kalshi_trade_rows),
            "shadow_fill_events": len(fills),
        },
        "observed_cadence_and_latency": {
            "pyth_xag_publish_gap": _gap_summary([row.publish_ts_ns for row in proxy_rows]),
            "exact_index_source_gap": _gap_summary([row.source_ts_ns for row in exact]),
            "exact_index_source_to_host": _latency_summary(exact_latencies),
            "si_bbo_receive_gap": _gap_summary([row.ts_recv_ns for row in si_bbo]),
            "si_trade_receive_gap": _gap_summary(si_trade_times),
            "kalshi_book_change_receive_gap": _gap_summary(book_times),
            "kalshi_trade_receive_gap": _gap_summary([int(row["receive_ts_ns"]) for row in kalshi_trade_rows]),
        },
        "sampling_contract": {
            "feature_grid": "each available 200ms Pyth XAG observation",
            "causality": "features use as-of values under the declared availability latencies",
            "targets": "future XAG or Kalshi midpoint movement, sampled non-overlapping at each target horizon",
            "window_grouping": "diagnostic correlations are also computed separately by active Kalshi 15-minute window",
            "confidence": "no p-values or confidence intervals; one capture is not an independent multi-day sample",
        },
        "signal_diagnostics": diagnostics,
        "exact_index_signal_diagnostics": exact_diagnostics,
        "fill_conditioned_execution_diagnostics": fill_diagnostics,
        "signal_cards": cards,
        "limitations": [
            "Only one night and a handful of Kalshi windows are present; no result is out of sample across trading days.",
            "Historical Pyth and Databento transport timestamps are scenario assumptions, while Kalshi receive timestamps were recorded live.",
            "Natural targets are short-horizon XAG or Kalshi changes, not binary settlement outcomes.",
            "The active SI contract is selected by total request-period volume, which is acceptable for this post-run diagnostic but is not a causal live roll rule.",
            "Fill-conditioned diagnostics contain only the observed shadow fills and cannot recreate queue position or cancellation races.",
            "Kalshi features are execution-only and are prohibited from entering the external fair model.",
        ],
    }
    identity = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_id"] = hashlib.sha256(identity).hexdigest()
    return report


def write_high_frequency_signal_report(
    root: str | Path,
    collector_run_id: str,
    shadow_run_id: str,
    **kwargs: Any,
) -> Path:
    report = build_high_frequency_signal_report(root, collector_run_id, shadow_run_id, **kwargs)
    destination = (
        Path(root)
        / "derived"
        / "high_frequency_signal_research"
        / f"report={report['report_id'][:24]}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "report.json"
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise HighFrequencySignalError(f"refusing to overwrite different report: {path}")
    path.write_text(payload, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/metals")
    parser.add_argument("--collector-run-id", required=True)
    parser.add_argument("--shadow-run-id", required=True)
    parser.add_argument("--databento-request-id", required=True)
    parser.add_argument("--pyth-transport-latency-ms", type=int, default=100)
    parser.add_argument("--databento-to-host-latency-ms", type=int, default=25)
    args = parser.parse_args(argv)
    path = write_high_frequency_signal_report(
        args.root,
        args.collector_run_id,
        args.shadow_run_id,
        databento_request_id=args.databento_request_id,
        pyth_transport_latency_ms=args.pyth_transport_latency_ms,
        databento_to_host_latency_ms=args.databento_to_host_latency_ms,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
