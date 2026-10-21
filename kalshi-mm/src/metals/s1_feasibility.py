"""Preliminary S1 feasibility study using one-minute Kalshi candles.

This module deliberately does not call its fill assumptions Q0/Q1/Q2. Candles do
not contain displayed depth, queue position, side-specific aggressive volume, or
receive-time latency. The result is an offline screen that can reject a weak idea;
it cannot approve trading or replace the event-driven simulator in the strategy
specification.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ONE = Decimal("1")
ZERO = Decimal("0")


@dataclass(frozen=True)
class S1FeasibilityConfig:
    metal: str = "silver"
    train_fraction: Decimal = Decimal("0.70")
    min_underdog_mid: Decimal = Decimal("0.07")
    max_underdog_mid: Decimal = Decimal("0.31")
    min_seconds_to_close: int = 120
    max_seconds_to_close: int = 900
    candle_interval_seconds: int = 60
    penetration_dollars: Decimal = Decimal("0.01")
    maker_fee_per_contract: Decimal = Decimal("0")
    taker_fee_multiplier: Decimal = Decimal("0.07")
    bootstrap_samples: int = 2_000
    bootstrap_seed: int = 17_291

    def validate(self) -> None:
        if not ZERO < self.train_fraction < ONE:
            raise ValueError("train_fraction must be between zero and one")
        if not ZERO <= self.min_underdog_mid < self.max_underdog_mid < ONE:
            raise ValueError("invalid underdog midpoint range")
        if self.min_seconds_to_close < 1:
            raise ValueError("min_seconds_to_close must be positive")
        if self.max_seconds_to_close < self.min_seconds_to_close:
            raise ValueError("max_seconds_to_close must not precede the minimum")
        if self.candle_interval_seconds < 1:
            raise ValueError("candle_interval_seconds must be positive")
        if self.penetration_dollars < ZERO:
            raise ValueError("penetration_dollars cannot be negative")
        if self.maker_fee_per_contract < ZERO:
            raise ValueError("maker_fee_per_contract cannot be negative")
        if self.taker_fee_multiplier < ZERO:
            raise ValueError("taker_fee_multiplier cannot be negative")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")


@dataclass(frozen=True)
class CandleRow:
    market_ticker: str
    market_open_ts_ns: int
    market_close_ts_ns: int
    end_period_ts_ns: int
    yes_bid_open: Optional[Decimal]
    yes_bid_low: Optional[Decimal]
    yes_bid_high: Optional[Decimal]
    yes_bid_close: Optional[Decimal]
    yes_ask_open: Optional[Decimal]
    yes_ask_low: Optional[Decimal]
    yes_ask_high: Optional[Decimal]
    yes_ask_close: Optional[Decimal]
    volume: Decimal
    result: str


@dataclass(frozen=True)
class Signal:
    market_ticker: str
    market_open_ts_ns: int
    market_close_ts_ns: int
    decision_ts_ns: int
    seconds_to_close: int
    time_bucket: str
    price_bucket: str
    underdog_side: str
    underdog_mid: Decimal
    passive_bid: Decimal
    underdog_won: bool
    next_underdog_ask_open: Optional[Decimal]
    next_underdog_ask_low: Optional[Decimal]
    next_volume: Decimal
    split: str = ""


@dataclass(frozen=True)
class Execution:
    market_ticker: str
    decision_ts_ns: int
    market_close_ts_ns: int
    seconds_to_close: int
    time_bucket: str
    price_bucket: str
    underdog_side: str
    underdog_mid: Decimal
    underdog_won: bool
    split: str
    model: str
    filled: bool
    entry_price: Optional[Decimal]
    fee: Decimal
    gross_pnl: Optional[Decimal]
    net_pnl: Optional[Decimal]


@dataclass(frozen=True)
class MarkoutRow:
    market_ticker: str
    split: str
    model: str
    underdog_side: str
    market_close_ts_ns: int
    decision_ts_ns: int
    fill_ts_ns: int
    fill_delay_seconds: Decimal
    horizon: str
    horizon_seconds: int
    target_ts_ns: int
    entry_price: Decimal
    underdog_mid: Decimal
    trigger_trade_price: Decimal
    trigger_trade_quantity: Decimal
    mark_price: Decimal
    mark_ts_ns: Optional[int]
    mark_age_seconds: Optional[Decimal]
    mark_source: str
    net_markout_pnl: Decimal


def _duckdb():
    try:
        import duckdb
    except ImportError as error:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "S1 feasibility analysis requires duckdb; install requirements.txt"
        ) from error
    return duckdb


def _decimal(value: Any) -> Optional[Decimal]:
    if value in (None, ""):
        return None
    return Decimal(str(value))


def _valid_price(value: Optional[Decimal]) -> bool:
    return value is not None and ZERO < value < ONE


def _time_bucket(seconds_to_close: int) -> Optional[str]:
    if 600 <= seconds_to_close <= 900:
        return "10-15m"
    if 300 <= seconds_to_close < 600:
        return "5-10m"
    if 120 <= seconds_to_close < 300:
        return "2-5m"
    if 60 <= seconds_to_close < 120:
        return "1-2m"
    if 30 <= seconds_to_close < 60:
        return "30-60s"
    if 0 <= seconds_to_close < 30:
        return "lt-30s"
    return None


def _price_bucket(price: Decimal) -> str:
    if price < Decimal("0.15"):
        return "0.07-0.15"
    if price < Decimal("0.25"):
        return "0.15-0.25"
    return "0.25-0.31"


def load_candles(catalog_path: Path, metal: str) -> List[CandleRow]:
    """Read only the columns required by the candle proxy."""

    catalog_path = Path(catalog_path)
    if not catalog_path.is_file():
        raise FileNotFoundError(
            "Metals DuckDB catalog not found: {}".format(catalog_path)
        )
    connection = _duckdb().connect(str(catalog_path), read_only=True)
    try:
        tables = {
            row[0]
            for row in connection.execute("SHOW TABLES").fetchall()
        }
        required = {"kalshi_candlestick", "market_dim", "kalshi_settlement"}
        missing = sorted(required - tables)
        if missing:
            raise RuntimeError(
                "Catalog is missing required view(s): {}. Backfill candles and "
                "run export-parquet first.".format(", ".join(missing))
            )
        rows = connection.execute(
            "SELECT c.market_ticker, m.open_ts_ns, m.close_ts_ns, "
            "c.end_period_ts_ns, c.yes_bid_open_dollars, "
            "c.yes_bid_low_dollars, c.yes_bid_high_dollars, "
            "c.yes_bid_close_dollars, c.yes_ask_open_dollars, "
            "c.yes_ask_low_dollars, c.yes_ask_high_dollars, "
            "c.yes_ask_close_dollars, c.volume_fp, s.result "
            "FROM kalshi_candlestick c "
            "JOIN market_dim m USING (market_ticker) "
            "JOIN kalshi_settlement s USING (market_ticker) "
            "WHERE c.metal = ? AND m.metal = ? "
            "AND c.period_interval_minutes = 1 "
            "AND lower(s.result) IN ('yes', 'no') "
            "ORDER BY m.open_ts_ns, c.market_ticker, c.end_period_ts_ns",
            [metal, metal],
        ).fetchall()
    finally:
        connection.close()
    return [
        CandleRow(
            market_ticker=str(row[0]),
            market_open_ts_ns=int(row[1]),
            market_close_ts_ns=int(row[2]),
            end_period_ts_ns=int(row[3]),
            yes_bid_open=_decimal(row[4]),
            yes_bid_low=_decimal(row[5]),
            yes_bid_high=_decimal(row[6]),
            yes_bid_close=_decimal(row[7]),
            yes_ask_open=_decimal(row[8]),
            yes_ask_low=_decimal(row[9]),
            yes_ask_high=_decimal(row[10]),
            yes_ask_close=_decimal(row[11]),
            volume=_decimal(row[12]) or ZERO,
            result=str(row[13]).lower(),
        )
        for row in rows
    ]


def _sql_literal(value: Path) -> str:
    return str(value).replace("'", "''")


def load_trade_markouts(
    catalog_path: Path,
    signals: Sequence[Signal],
    config: S1FeasibilityConfig,
) -> List[MarkoutRow]:
    """Anchor passive fills to trades and calculate last-trade markouts.

    A resting Yes bid is eligible when a No taker trades Yes at or below the
    bid. A resting No bid is eligible when a Yes taker trades No at or below
    the bid. The first qualifying trade within one minute anchors the fill.
    """

    if not signals:
        return []
    connection = _duckdb().connect()
    try:
        connection.execute(
            "ATTACH '{}' AS metals (READ_ONLY)".format(
                _sql_literal(Path(catalog_path))
            )
        )
        connection.execute(
            "CREATE TEMP TABLE signal_input ("
            "market_ticker VARCHAR, market_open_ts_ns BIGINT, "
            "market_close_ts_ns BIGINT, decision_ts_ns BIGINT, split VARCHAR, "
            "underdog_side VARCHAR, passive_bid DECIMAL(9, 4), "
            "underdog_mid DECIMAL(9, 4), underdog_won BOOLEAN)"
        )
        connection.executemany(
            "INSERT INTO signal_input VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row.market_ticker,
                    row.market_open_ts_ns,
                    row.market_close_ts_ns,
                    row.decision_ts_ns,
                    row.split,
                    row.underdog_side,
                    row.passive_bid,
                    row.underdog_mid,
                    row.underdog_won,
                )
                for row in signals
            ],
        )
        connection.execute(
            "CREATE TEMP TABLE anchored_fill AS "
            "WITH signal_models AS ("
            "SELECT *, 'passive_touch_proxy' AS model, "
            "0::DECIMAL(9, 4) AS penetration FROM signal_input UNION ALL "
            "SELECT *, 'passive_penetration_proxy', ?::DECIMAL(9, 4) "
            "FROM signal_input) "
            "SELECT s.market_ticker, s.market_open_ts_ns, "
            "s.market_close_ts_ns, s.decision_ts_ns, s.split, s.model, "
            "s.underdog_side, s.passive_bid AS entry_price, "
            "s.underdog_mid, s.underdog_won, "
            "t.source_event_ts_ns AS fill_ts_ns, t.trade_id, "
            "CASE WHEN s.underdog_side = 'yes' "
            "THEN CAST(t.yes_price_dollars AS DECIMAL(9, 4)) "
            "ELSE CAST(t.no_price_dollars AS DECIMAL(9, 4)) END "
            "AS trigger_trade_price, "
            "CAST(t.quantity_fp AS DECIMAL(18, 4)) AS trigger_trade_quantity "
            "FROM signal_models s JOIN metals.kalshi_trade t "
            "ON t.market_ticker = s.market_ticker "
            "AND t.source_event_ts_ns > s.decision_ts_ns "
            "AND t.source_event_ts_ns <= s.decision_ts_ns + ? "
            "AND t.taker_outcome_side = CASE "
            "WHEN s.underdog_side = 'yes' THEN 'no' ELSE 'yes' END "
            "AND CASE WHEN s.underdog_side = 'yes' "
            "THEN CAST(t.yes_price_dollars AS DECIMAL(9, 4)) "
            "ELSE CAST(t.no_price_dollars AS DECIMAL(9, 4)) END "
            "<= s.passive_bid - s.penetration "
            "QUALIFY row_number() OVER (PARTITION BY s.market_ticker, "
            "s.decision_ts_ns, s.model ORDER BY t.source_event_ts_ns, "
            "t.trade_id) = 1",
            [
                config.penetration_dollars,
                config.candle_interval_seconds * 1_000_000_000,
            ],
        )
        connection.execute(
            "CREATE TEMP TABLE trade_mark AS "
            "SELECT t.market_ticker, t.source_event_ts_ns, "
            "CAST(sum(CAST(t.yes_price_dollars AS DECIMAL(9, 4)) * "
            "CAST(t.quantity_fp AS DECIMAL(18, 4))) / "
            "sum(CAST(t.quantity_fp AS DECIMAL(18, 4))) "
            "AS DECIMAL(9, 4)) AS yes_price "
            "FROM metals.kalshi_trade t SEMI JOIN "
            "(SELECT DISTINCT market_ticker FROM anchored_fill) a "
            "USING (market_ticker) GROUP BY 1, 2"
        )
        connection.execute(
            "CREATE TEMP TABLE markout_horizon "
            "(horizon VARCHAR, horizon_seconds BIGINT)"
        )
        horizons = (
            ("1s", 1),
            ("5s", 5),
            ("30s", 30),
            ("1m", 60),
            ("5m", 300),
            ("15m/end", 900),
        )
        connection.executemany(
            "INSERT INTO markout_horizon VALUES (?, ?)", horizons
        )
        rows = connection.execute(
            "WITH desired AS ("
            "SELECT a.*, h.horizon, h.horizon_seconds, "
            "least(a.fill_ts_ns + h.horizon_seconds * 1000000000, "
            "a.market_close_ts_ns) AS target_ts_ns, "
            "(h.horizon = '15m/end' OR a.fill_ts_ns + "
            "h.horizon_seconds * 1000000000 >= a.market_close_ts_ns) "
            "AS terminal FROM anchored_fill a CROSS JOIN markout_horizon h), "
            "trade_desired AS (SELECT * FROM desired WHERE NOT terminal), "
            "trade_joined AS ("
            "SELECT d.*, m.source_event_ts_ns AS mark_ts_ns, "
            "CASE WHEN d.underdog_side = 'yes' THEN m.yes_price "
            "ELSE 1 - m.yes_price END AS mark_price, 'trade' AS mark_source "
            "FROM trade_desired d ASOF LEFT JOIN trade_mark m "
            "ON d.market_ticker = m.market_ticker "
            "AND d.target_ts_ns >= m.source_event_ts_ns), "
            "terminal_rows AS ("
            "SELECT d.*, NULL::BIGINT AS mark_ts_ns, "
            "CASE WHEN d.underdog_won THEN 1::DECIMAL(9, 4) "
            "ELSE 0::DECIMAL(9, 4) END AS mark_price, "
            "'settlement' AS mark_source FROM desired d WHERE terminal), "
            "combined AS (SELECT * FROM trade_joined UNION ALL BY NAME "
            "SELECT * FROM terminal_rows) "
            "SELECT market_ticker, split, model, underdog_side, "
            "market_close_ts_ns, decision_ts_ns, fill_ts_ns, horizon, "
            "horizon_seconds, target_ts_ns, entry_price, underdog_mid, "
            "trigger_trade_price, trigger_trade_quantity, mark_price, "
            "mark_ts_ns, mark_source FROM combined "
            "WHERE mark_price IS NOT NULL "
            "ORDER BY split, model, fill_ts_ns, horizon_seconds"
        ).fetchall()
    finally:
        connection.close()

    result = []
    for row in rows:
        entry_price = _decimal(row[10])
        mark_price = _decimal(row[14])
        if entry_price is None or mark_price is None:  # pragma: no cover
            continue
        fill_ts_ns = int(row[6])
        target_ts_ns = int(row[9])
        mark_ts_ns = int(row[15]) if row[15] is not None else None
        result.append(
            MarkoutRow(
                market_ticker=str(row[0]),
                split=str(row[1]),
                model=str(row[2]),
                underdog_side=str(row[3]),
                market_close_ts_ns=int(row[4]),
                decision_ts_ns=int(row[5]),
                fill_ts_ns=fill_ts_ns,
                fill_delay_seconds=Decimal(
                    fill_ts_ns - int(row[5])
                ) / Decimal(1_000_000_000),
                horizon=str(row[7]),
                horizon_seconds=int(row[8]),
                target_ts_ns=target_ts_ns,
                entry_price=entry_price,
                underdog_mid=_decimal(row[11]) or ZERO,
                trigger_trade_price=_decimal(row[12]) or ZERO,
                trigger_trade_quantity=_decimal(row[13]) or ZERO,
                mark_price=mark_price,
                mark_ts_ns=mark_ts_ns,
                mark_age_seconds=(
                    Decimal(target_ts_ns - mark_ts_ns)
                    / Decimal(1_000_000_000)
                    if mark_ts_ns is not None
                    else None
                ),
                mark_source=str(row[16]),
                net_markout_pnl=(
                    mark_price
                    - entry_price
                    - config.maker_fee_per_contract
                ),
            )
        )
    return result


def build_signals(
    candles: Sequence[CandleRow], config: S1FeasibilityConfig
) -> List[Signal]:
    """Select the earliest eligible observation in each market/time bucket."""

    config.validate()
    by_market: Dict[str, List[CandleRow]] = defaultdict(list)
    for candle in candles:
        by_market[candle.market_ticker].append(candle)

    signals: List[Signal] = []
    for market_ticker, market_candles in by_market.items():
        ordered = sorted(market_candles, key=lambda row: row.end_period_ts_ns)
        by_end = {row.end_period_ts_ns: row for row in ordered}
        selected_buckets = set()
        for candle in ordered:
            seconds_to_close = (
                candle.market_close_ts_ns - candle.end_period_ts_ns
            ) // 1_000_000_000
            if not (
                config.min_seconds_to_close
                <= seconds_to_close
                <= config.max_seconds_to_close
            ):
                continue
            bucket = _time_bucket(seconds_to_close)
            if bucket is None or bucket in selected_buckets:
                continue
            bid = candle.yes_bid_close
            ask = candle.yes_ask_close
            if not (_valid_price(bid) and _valid_price(ask) and bid <= ask):
                continue
            midpoint = (bid + ask) / Decimal("2")
            if midpoint == Decimal("0.5"):
                continue
            underdog_side = "no" if midpoint > Decimal("0.5") else "yes"
            underdog_mid = min(midpoint, ONE - midpoint)
            if not (
                config.min_underdog_mid
                <= underdog_mid
                <= config.max_underdog_mid
            ):
                continue
            passive_bid = bid if underdog_side == "yes" else ONE - ask
            if not _valid_price(passive_bid):
                continue

            next_candle = by_end.get(
                candle.end_period_ts_ns
                + config.candle_interval_seconds * 1_000_000_000
            )
            if next_candle is None:
                continue
            if underdog_side == "yes":
                next_ask_open = next_candle.yes_ask_open
                next_ask_low = next_candle.yes_ask_low
            else:
                next_ask_open = (
                    ONE - next_candle.yes_bid_open
                    if next_candle.yes_bid_open is not None
                    else None
                )
                next_ask_low = (
                    ONE - next_candle.yes_bid_high
                    if next_candle.yes_bid_high is not None
                    else None
                )
            underdog_won = candle.result == underdog_side
            signals.append(
                Signal(
                    market_ticker=market_ticker,
                    market_open_ts_ns=candle.market_open_ts_ns,
                    market_close_ts_ns=candle.market_close_ts_ns,
                    decision_ts_ns=candle.end_period_ts_ns,
                    seconds_to_close=int(seconds_to_close),
                    time_bucket=bucket,
                    price_bucket=_price_bucket(underdog_mid),
                    underdog_side=underdog_side,
                    underdog_mid=underdog_mid,
                    passive_bid=passive_bid,
                    underdog_won=underdog_won,
                    next_underdog_ask_open=next_ask_open,
                    next_underdog_ask_low=next_ask_low,
                    next_volume=next_candle.volume,
                )
            )
            selected_buckets.add(bucket)
    return sorted(
        signals,
        key=lambda row: (
            row.market_open_ts_ns,
            row.market_ticker,
            row.decision_ts_ns,
        ),
    )


def assign_chronological_split(
    signals: Sequence[Signal], train_fraction: Decimal
) -> Tuple[List[Signal], Dict[str, Any]]:
    markets = sorted(
        {(signal.market_open_ts_ns, signal.market_ticker) for signal in signals}
    )
    if len(markets) < 2:
        raise RuntimeError("At least two eligible markets are required for a split")
    train_count = int(Decimal(len(markets)) * train_fraction)
    train_count = max(1, min(len(markets) - 1, train_count))
    train_tickers = {ticker for _, ticker in markets[:train_count]}
    assigned = [
        Signal(**{**asdict(signal), "split": (
            "train" if signal.market_ticker in train_tickers else "test"
        )})
        for signal in signals
    ]
    first_test_open, first_test_ticker = markets[train_count]
    return assigned, {
        "market_count": len(markets),
        "train_market_count": train_count,
        "test_market_count": len(markets) - train_count,
        "first_test_market_ticker": first_test_ticker,
        "first_test_market_open_ts_ns": first_test_open,
    }


def simulate_executions(
    signals: Sequence[Signal], config: S1FeasibilityConfig
) -> List[Execution]:
    executions: List[Execution] = []
    for signal in signals:
        definitions = (
            (
                "aggressive_next_open_proxy",
                _valid_price(signal.next_underdog_ask_open),
                signal.next_underdog_ask_open,
                True,
            ),
            (
                "passive_touch_proxy",
                signal.next_volume > ZERO
                and _valid_price(signal.next_underdog_ask_low)
                and signal.next_underdog_ask_low <= signal.passive_bid,
                signal.passive_bid,
                False,
            ),
            (
                "passive_penetration_proxy",
                signal.next_volume > ZERO
                and _valid_price(signal.next_underdog_ask_low)
                and signal.next_underdog_ask_low
                <= signal.passive_bid - config.penetration_dollars,
                signal.passive_bid,
                False,
            ),
        )
        for model, filled, entry, is_taker in definitions:
            entry_price = entry if filled else None
            fee = ZERO
            gross_pnl = None
            net_pnl = None
            if entry_price is not None:
                fee = (
                    config.taker_fee_multiplier
                    * entry_price
                    * (ONE - entry_price)
                    if is_taker
                    else config.maker_fee_per_contract
                )
                gross_pnl = (ONE if signal.underdog_won else ZERO) - entry_price
                net_pnl = gross_pnl - fee
            executions.append(
                Execution(
                    market_ticker=signal.market_ticker,
                    decision_ts_ns=signal.decision_ts_ns,
                    market_close_ts_ns=signal.market_close_ts_ns,
                    seconds_to_close=signal.seconds_to_close,
                    time_bucket=signal.time_bucket,
                    price_bucket=signal.price_bucket,
                    underdog_side=signal.underdog_side,
                    underdog_mid=signal.underdog_mid,
                    underdog_won=signal.underdog_won,
                    split=signal.split,
                    model=model,
                    filled=bool(filled),
                    entry_price=entry_price,
                    fee=fee,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                )
            )
    return executions


def _ratio(numerator: Decimal, denominator: Decimal) -> Optional[Decimal]:
    return numerator / denominator if denominator else None


def _mean(values: Sequence[Decimal]) -> Optional[Decimal]:
    return sum(values, ZERO) / Decimal(len(values)) if values else None


def _format_decimal(value: Optional[Decimal]) -> Optional[str]:
    if value is None:
        return None
    return format(value.quantize(Decimal("0.000001")), "f")


def _max_drawdown(pnls: Sequence[Decimal]) -> Decimal:
    cumulative = ZERO
    peak = ZERO
    drawdown = ZERO
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return drawdown


def _percentile(values: Sequence[Decimal], probability: Decimal) -> Decimal:
    ordered = sorted(values)
    index = int(Decimal(len(ordered) - 1) * probability)
    return ordered[index]


def _daily_bootstrap(
    rows: Sequence[Execution], samples: int, seed: int
) -> Dict[str, Any]:
    by_day: Dict[str, Decimal] = defaultdict(lambda: ZERO)
    for row in rows:
        day = datetime.fromtimestamp(
            row.market_close_ts_ns / 1_000_000_000, timezone.utc
        ).date().isoformat()
        by_day[day] += row.net_pnl or ZERO
    daily = [by_day[day] for day in sorted(by_day)]
    if not daily:
        return {
            "active_days": 0,
            "mean_daily_pnl_dollars": None,
            "mean_daily_pnl_90pct_ci_dollars": [None, None],
        }
    generator = random.Random(seed)
    bootstrap_means = []
    for _ in range(samples):
        total = sum(
            (daily[generator.randrange(len(daily))] for _ in daily), ZERO
        )
        bootstrap_means.append(total / Decimal(len(daily)))
    return {
        "active_days": len(daily),
        "mean_daily_pnl_dollars": _format_decimal(_mean(daily)),
        "mean_daily_pnl_90pct_ci_dollars": [
            _format_decimal(_percentile(bootstrap_means, Decimal("0.05"))),
            _format_decimal(_percentile(bootstrap_means, Decimal("0.95"))),
        ],
    }


def summarize_executions(
    rows: Sequence[Execution], bootstrap_samples: int, bootstrap_seed: int
) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (row.decision_ts_ns, row.market_ticker))
    fills = [row for row in ordered if row.filled and row.net_pnl is not None]
    pnls = [row.net_pnl for row in fills if row.net_pnl is not None]
    gross = [row.gross_pnl for row in fills if row.gross_pnl is not None]
    fees = [row.fee for row in fills]
    entries = [row.entry_price for row in fills if row.entry_price is not None]
    wins = sum(1 for row in fills if row.underdog_won)
    gross_positive = sum((max(pnl, ZERO) for pnl in pnls), ZERO)
    best_count = max(1, math.ceil(len(pnls) * 0.01)) if pnls else 0
    best_pnl = sum(sorted(pnls, reverse=True)[:best_count], ZERO)
    total_pnl = sum(pnls, ZERO)
    summary = {
        "signals": len(rows),
        "fills": len(fills),
        "fill_rate": _format_decimal(
            _ratio(Decimal(len(fills)), Decimal(len(rows)))
        ),
        "wins": wins,
        "empirical_win_rate": _format_decimal(
            _ratio(Decimal(wins), Decimal(len(fills)))
        ),
        "average_entry_price_dollars": _format_decimal(_mean(entries)),
        "average_fee_dollars": _format_decimal(_mean(fees)),
        "average_gross_pnl_dollars": _format_decimal(_mean(gross)),
        "average_net_pnl_dollars": _format_decimal(_mean(pnls)),
        "total_net_pnl_dollars": _format_decimal(total_pnl),
        "max_drawdown_dollars": _format_decimal(_max_drawdown(pnls)),
        "best_1pct_window_count": best_count,
        "best_1pct_share_of_gross_positive_pnl": _format_decimal(
            _ratio(best_pnl, gross_positive)
        ),
        "best_1pct_net_pnl_share_of_total_net_pnl": _format_decimal(
            _ratio(best_pnl, total_pnl) if total_pnl > ZERO else None
        ),
    }
    summary.update(_daily_bootstrap(ordered, bootstrap_samples, bootstrap_seed))
    return summary


def summarize_calibration(signals: Sequence[Signal]) -> Dict[str, Any]:
    probabilities = [row.underdog_mid for row in signals]
    outcomes = [ONE if row.underdog_won else ZERO for row in signals]
    average_probability = _mean(probabilities)
    win_rate = _mean(outcomes)
    return {
        "signals": len(signals),
        "average_underdog_midpoint": _format_decimal(average_probability),
        "empirical_underdog_win_rate": _format_decimal(win_rate),
        "win_rate_minus_midpoint": _format_decimal(
            win_rate - average_probability
            if win_rate is not None and average_probability is not None
            else None
        ),
        "brier_score": _format_decimal(
            _mean(
                [
                    (probability - outcome) ** 2
                    for probability, outcome in zip(probabilities, outcomes)
                ]
            )
        ),
    }


def _markout_day_bootstrap(
    rows: Sequence[MarkoutRow], samples: int, seed: int
) -> List[Optional[str]]:
    by_day: Dict[str, List[Decimal]] = defaultdict(list)
    for row in rows:
        day = datetime.fromtimestamp(
            row.market_close_ts_ns / 1_000_000_000, timezone.utc
        ).date().isoformat()
        by_day[day].append(row.net_markout_pnl)
    days = [by_day[day] for day in sorted(by_day)]
    if not days:
        return [None, None]
    generator = random.Random(seed)
    sample_means = []
    for _ in range(samples):
        sampled = [days[generator.randrange(len(days))] for _ in days]
        total = sum(
            (sum(day_rows, ZERO) for day_rows in sampled), ZERO
        )
        count = sum(len(day_rows) for day_rows in sampled)
        sample_means.append(total / Decimal(count))
    return [
        _format_decimal(_percentile(sample_means, Decimal("0.05"))),
        _format_decimal(_percentile(sample_means, Decimal("0.95"))),
    ]


def summarize_markouts(
    rows: Sequence[MarkoutRow], bootstrap_samples: int, bootstrap_seed: int
) -> Dict[str, Any]:
    pnls = [row.net_markout_pnl for row in rows]
    entries = [row.entry_price for row in rows]
    marks = [row.mark_price for row in rows]
    fill_delays = [row.fill_delay_seconds for row in rows]
    mark_ages = [
        row.mark_age_seconds
        for row in rows
        if row.mark_age_seconds is not None
    ]
    binary_variances = [price * (ONE - price) for price in entries]
    binary_stddevs = [variance.sqrt() for variance in binary_variances]
    return {
        "fills": len(rows),
        "average_entry_price_dollars": _format_decimal(_mean(entries)),
        "median_entry_price_dollars": _format_decimal(
            _percentile(entries, Decimal("0.50")) if entries else None
        ),
        "average_mark_price_dollars": _format_decimal(_mean(marks)),
        "average_net_markout_pnl_dollars": _format_decimal(_mean(pnls)),
        "median_net_markout_pnl_dollars": _format_decimal(
            _percentile(pnls, Decimal("0.50")) if pnls else None
        ),
        "p10_net_markout_pnl_dollars": _format_decimal(
            _percentile(pnls, Decimal("0.10")) if pnls else None
        ),
        "p90_net_markout_pnl_dollars": _format_decimal(
            _percentile(pnls, Decimal("0.90")) if pnls else None
        ),
        "positive_markout_rate": _format_decimal(
            _ratio(
                Decimal(sum(1 for pnl in pnls if pnl > ZERO)),
                Decimal(len(pnls)),
            )
        ),
        "day_block_bootstrap_mean_markout_90pct_ci_dollars": (
            _markout_day_bootstrap(rows, bootstrap_samples, bootstrap_seed)
        ),
        "average_fill_delay_seconds": _format_decimal(_mean(fill_delays)),
        "median_fill_delay_seconds": _format_decimal(
            _percentile(fill_delays, Decimal("0.50"))
            if fill_delays
            else None
        ),
        "average_trade_mark_age_seconds": _format_decimal(_mean(mark_ages)),
        "p95_trade_mark_age_seconds": _format_decimal(
            _percentile(mark_ages, Decimal("0.95")) if mark_ages else None
        ),
        "settlement_mark_count": sum(
            1 for row in rows if row.mark_source == "settlement"
        ),
        "trade_mark_count": sum(
            1 for row in rows if row.mark_source == "trade"
        ),
        "average_market_implied_binary_variance_dollars_squared": (
            _format_decimal(_mean(binary_variances))
        ),
        "average_market_implied_terminal_pnl_stddev_dollars": (
            _format_decimal(_mean(binary_stddevs))
        ),
    }


def summarize_trade_markout_study(
    markouts: Sequence[MarkoutRow],
    signals: Sequence[Signal],
    config: S1FeasibilityConfig,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for split_name in ("train", "test"):
        results[split_name] = {}
        for model in (
            "passive_touch_proxy",
            "passive_penetration_proxy",
        ):
            model_rows = [
                row
                for row in markouts
                if row.split == split_name and row.model == model
            ]
            results[split_name][model] = {
                horizon: summarize_markouts(
                    [row for row in model_rows if row.horizon == horizon],
                    config.bootstrap_samples,
                    config.bootstrap_seed,
                )
                for horizon in ("1s", "5s", "30s", "1m", "5m", "15m/end")
            }

    coverage: Dict[str, Any] = {}
    for split_name in ("train", "test"):
        signal_count = sum(1 for row in signals if row.split == split_name)
        coverage[split_name] = {"signals": signal_count}
        for model in (
            "passive_touch_proxy",
            "passive_penetration_proxy",
        ):
            fills = {
                (row.market_ticker, row.decision_ts_ns, row.fill_ts_ns)
                for row in markouts
                if row.split == split_name and row.model == model
            }
            coverage[split_name][model] = {
                "trade_anchored_fills": len(fills),
                "fill_rate": _format_decimal(
                    _ratio(Decimal(len(fills)), Decimal(signal_count))
                ),
            }
    return {
        "classification": "last_trade_price_proxy_not_midpoint_markout",
        "fill_anchor": (
            "First qualifying opposing-taker trade at or through the resting "
            "underdog bid within 60 seconds of the decision."
        ),
        "horizons": ["1s", "5s", "30s", "1m", "5m", "15m/end"],
        "mark_rule": (
            "Most recent quantity-weighted trade price at or before the horizon; "
            "settlement payout when the horizon reaches market close."
        ),
        "coverage": coverage,
        "results": results,
        "vega": {
            "true_digital_option_vega": "not_estimable_without_fair_value_model",
            "reported_proxy": (
                "Market-implied Bernoulli payout variance p*(1-p) and its "
                "one-contract terminal P&L standard deviation."
            ),
            "risk_warning": (
                "The strategy specification requires terminal outcome exposure and "
                "stress loss as primary limits; this proxy is not a risk limit."
            ),
        },
        "limitations": [
            "Sub-minute marks are trades, not two-sided midpoints or microprices.",
            "A carried-forward trade can be stale; mark-age statistics are reported.",
            "Trade-through does not reconstruct queue ahead or guarantee our fill.",
            "Trigger trade quantity is not an estimate of scalable order capacity.",
            "Five-minute marks use settlement when the market closes first.",
        ],
    }


def _grouped_execution_summary(
    executions: Sequence[Execution],
    field: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    groups: Dict[str, List[Execution]] = defaultdict(list)
    for row in executions:
        groups[str(getattr(row, field))].append(row)
    return {
        key: summarize_executions(rows, bootstrap_samples, bootstrap_seed)
        for key, rows in sorted(groups.items())
    }


def _nested_execution_summary(
    executions: Sequence[Execution],
    outer_field: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    outer_groups: Dict[str, List[Execution]] = defaultdict(list)
    for row in executions:
        outer_groups[str(getattr(row, outer_field))].append(row)
    return {
        key: _grouped_execution_summary(
            rows, "model", bootstrap_samples, bootstrap_seed
        )
        for key, rows in sorted(outer_groups.items())
    }


def _execution_sensitivity(
    test_executions: Sequence[Execution], config: S1FeasibilityConfig
) -> Dict[str, Any]:
    passive_rows = [
        row
        for row in test_executions
        if row.model in ("passive_touch_proxy", "passive_penetration_proxy")
    ]
    maker_fee_results: Dict[str, Any] = {}
    for fee in (ZERO, Decimal("0.005"), Decimal("0.01")):
        repriced = [
            replace(
                row,
                fee=fee if row.filled else ZERO,
                net_pnl=(
                    row.gross_pnl - fee
                    if row.filled and row.gross_pnl is not None
                    else None
                ),
            )
            for row in passive_rows
        ]
        maker_fee_results[str(fee)] = _grouped_execution_summary(
            repriced,
            "model",
            config.bootstrap_samples,
            config.bootstrap_seed,
        )

    aggressive_rows = [
        row
        for row in test_executions
        if row.model == "aggressive_next_open_proxy"
    ]
    slippage_results: Dict[str, Any] = {}
    for slippage in (ZERO, Decimal("0.005"), Decimal("0.01")):
        repriced = []
        for row in aggressive_rows:
            if not row.filled or row.entry_price is None:
                repriced.append(row)
                continue
            entry = row.entry_price + slippage
            if not _valid_price(entry):
                repriced.append(
                    replace(
                        row,
                        filled=False,
                        entry_price=None,
                        fee=ZERO,
                        gross_pnl=None,
                        net_pnl=None,
                    )
                )
                continue
            fee = config.taker_fee_multiplier * entry * (ONE - entry)
            gross_pnl = (ONE if row.underdog_won else ZERO) - entry
            repriced.append(
                replace(
                    row,
                    entry_price=entry,
                    fee=fee,
                    gross_pnl=gross_pnl,
                    net_pnl=gross_pnl - fee,
                )
            )
        slippage_results[str(slippage)] = summarize_executions(
            repriced,
            config.bootstrap_samples,
            config.bootstrap_seed,
        )
    return {
        "test_passive_maker_fee_per_contract_dollars": maker_fee_results,
        "test_aggressive_additional_slippage_dollars": slippage_results,
        "fill_strictness_models": [
            "passive_touch_proxy",
            "passive_penetration_proxy",
        ],
        "capacity": {
            "status": "not_estimable_from_candles",
            "reason": (
                "Candles omit displayed depth, queue ahead, and side-specific "
                "aggressive quantity. Scaling the one-contract P&L would imply "
                "unsupported fill capacity."
            ),
        },
        "risk": {
            "status": "one_contract_path_only",
            "reason": (
                "Drawdown is reported for chronological one-contract proxy fills; "
                "overlapping inventory and live-scale risk require event replay."
            ),
        },
    }


def analyze_s1_feasibility(
    candles: Sequence[CandleRow], config: S1FeasibilityConfig
) -> Tuple[Dict[str, Any], List[Signal], List[Execution]]:
    config.validate()
    signals = build_signals(candles, config)
    if not signals:
        raise RuntimeError("No eligible S1 candle observations were found")
    signals, split = assign_chronological_split(signals, config.train_fraction)
    executions = simulate_executions(signals, config)

    execution_results: Dict[str, Any] = {}
    for split_name in ("all", "train", "test"):
        split_rows = (
            executions
            if split_name == "all"
            else [row for row in executions if row.split == split_name]
        )
        execution_results[split_name] = _grouped_execution_summary(
            split_rows,
            "model",
            config.bootstrap_samples,
            config.bootstrap_seed,
        )

    calibration: Dict[str, Any] = {}
    for split_name in ("all", "train", "test"):
        split_signals = (
            signals
            if split_name == "all"
            else [row for row in signals if row.split == split_name]
        )
        calibration[split_name] = {
            "overall": summarize_calibration(split_signals),
            "by_price_bucket": {
                bucket: summarize_calibration(
                    [row for row in split_signals if row.price_bucket == bucket]
                )
                for bucket in ("0.07-0.15", "0.15-0.25", "0.25-0.31")
            },
            "by_time_bucket": {
                bucket: summarize_calibration(
                    [row for row in split_signals if row.time_bucket == bucket]
                )
                for bucket in ("10-15m", "5-10m", "2-5m")
            },
        }

    test_executions = [row for row in executions if row.split == "test"]
    test_calibration_edge = calibration["test"]["overall"][
        "win_rate_minus_midpoint"
    ]
    passive_test_edges = [
        execution_results["test"][model]["average_net_pnl_dollars"]
        for model in ("passive_touch_proxy", "passive_penetration_proxy")
    ]
    preliminary_reject = (
        test_calibration_edge is not None
        and Decimal(test_calibration_edge) < ZERO
        and all(
            edge is not None and Decimal(edge) < ZERO
            for edge in passive_test_edges
        )
    )
    research_conclusion = {
        "status": (
            "PRELIMINARY_REJECT_UNFILTERED_S1"
            if preliminary_reject
            else "INCONCLUSIVE_CONTINUE_SHADOW_ONLY"
        ),
        "basis": (
            "Held-out underdog calibration and both passive fill proxies are negative."
            if preliminary_reject
            else "The held-out calibration and passive proxy signs do not all reject S1."
        ),
        "scope": (
            "This applies to the unfiltered 7-31 cent favorite-fade rule only; "
            "it does not test a future independently specified fair-value filter."
        ),
    }
    report = {
        "study_version": "s1-candle-proxy-v1",
        "classification": "preliminary_candle_proxy_not_final_event_backtest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": "NO_GO_FOR_TRADING",
        "research_conclusion": research_conclusion,
        "parameters": {
            "metal": config.metal,
            "train_fraction": str(config.train_fraction),
            "min_underdog_mid_dollars": str(config.min_underdog_mid),
            "max_underdog_mid_dollars": str(config.max_underdog_mid),
            "min_seconds_to_close": config.min_seconds_to_close,
            "max_seconds_to_close": config.max_seconds_to_close,
            "candle_interval_seconds": config.candle_interval_seconds,
            "penetration_dollars": str(config.penetration_dollars),
            "maker_fee_per_contract_dollars_assumption": str(
                config.maker_fee_per_contract
            ),
            "unrounded_taker_fee_multiplier": str(config.taker_fee_multiplier),
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_seed": config.bootstrap_seed,
        },
        "data": {
            "candle_rows": len(candles),
            "eligible_signals": len(signals),
            "markets_with_eligible_signals": len(
                {row.market_ticker for row in signals}
            ),
            "first_market_open_ts_ns": min(row.market_open_ts_ns for row in signals),
            "last_market_close_ts_ns": max(row.market_close_ts_ns for row in signals),
        },
        "chronological_split": split,
        "calibration": calibration,
        "execution_results": execution_results,
        "sensitivity": _execution_sensitivity(test_executions, config),
        "test_execution_by_time_bucket": _nested_execution_summary(
            test_executions,
            "time_bucket",
            config.bootstrap_samples,
            config.bootstrap_seed,
        ),
        "test_execution_by_price_bucket": _nested_execution_summary(
            test_executions,
            "price_bucket",
            config.bootstrap_samples,
            config.bootstrap_seed,
        ),
        "gate_assessment": {
            "final_q0_q1_q2_available": False,
            "maker_fee_schedule_validated_by_effective_date_and_series": False,
            "latency_from_shadow_telemetry_available": False,
            "side_specific_aggressive_volume_and_queue_depth_available": False,
            "can_reject_s1_for_weak_preliminary_economics": True,
            "can_approve_shadow_or_live_capital": False,
            "conclusion": "NO_GO_FOR_TRADING",
        },
        "limitations": [
            "One-minute OHLC candles cannot reconstruct order priority or partial fills.",
            "Candle volume is not side-specific and does not prove volume traded "
            "at the simulated bid.",
            "The passive touch and one-cent penetration rules are sensitivity "
            "proxies, not Q0/Q1/Q2 queue models.",
            "The aggressive next-open rule is a taker benchmark, not the passive S1 strategy.",
            "Zero maker fee is an explicit unvalidated assumption unless overridden; "
            "the effective fee registry is absent.",
            "No CME/Pyth fair-value, latency, incentive, capacity, or clock-quality "
            "model is present.",
            "At most one signal per market/time bucket is retained to reduce "
            "repeated-observation dependence.",
        ],
    }
    return report, signals, executions


def _atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".s1-", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _csv_text(rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _serialized_dataclass(instance: Any) -> Dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Decimal) else value
        for key, value in asdict(instance).items()
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# S1 preliminary candle feasibility",
        "",
        "Decision: **NO-GO for trading**. This is a screening proxy, not the "
        "required event-driven Q0/Q1/Q2 backtest.",
        "",
        "Research conclusion: **{}**. {}".format(
            report["research_conclusion"]["status"],
            report["research_conclusion"]["basis"],
        ),
        "",
        "Held-out underdog win-rate minus midpoint: **{} dollars per contract**.".format(
            report["calibration"]["test"]["overall"][
                "win_rate_minus_midpoint"
            ]
        ),
        "",
        "Chronological out-of-sample results (one contract per filled signal):",
        "",
        "| Fill proxy | Signals | Fills | Fill rate | Avg net P&L | Total net P&L |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    test_results = report["execution_results"]["test"]
    for model in (
        "aggressive_next_open_proxy",
        "passive_touch_proxy",
        "passive_penetration_proxy",
    ):
        result = test_results.get(model, {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                model,
                result.get("signals", 0),
                result.get("fills", 0),
                result.get("fill_rate"),
                result.get("average_net_pnl_dollars"),
                result.get("total_net_pnl_dollars"),
            )
        )
    lines.extend(
        [
            "",
            "The passive results assume the configured maker fee and use next-minute "
            "OHLC/aggregate volume only. They do not model queue position, latency, "
            "partial fills, or side-specific aggressive volume.",
            "",
            "The JSON report contains calibration, time/price buckets, daily block "
            "bootstrap intervals, concentration, drawdown, parameters, and limitations.",
            "",
        ]
    )
    markout = report.get("markout_study")
    if markout:
        lines.extend(
            [
                "## Trade-anchored markout smile",
                "",
                "These held-out marks use the most recent trade, not a midpoint.",
                "",
                "| Fill proxy | Horizon | Fills | Avg entry | Avg markout P&L | 90% day-block CI |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        results = markout["results"]["test"]
        for model in (
            "passive_touch_proxy",
            "passive_penetration_proxy",
        ):
            for horizon in ("1s", "5s", "30s", "1m", "5m", "15m/end"):
                item = results[model][horizon]
                interval = item[
                    "day_block_bootstrap_mean_markout_90pct_ci_dollars"
                ]
                lines.append(
                    "| {} | {} | {} | {} | {} | {} to {} |".format(
                        model,
                        horizon,
                        item["fills"],
                        item["average_entry_price_dollars"],
                        item["average_net_markout_pnl_dollars"],
                        interval[0],
                        interval[1],
                    )
                )
        lines.extend(
            [
                "",
                "True digital-option vega is not identifiable without the fair-value "
                "model. The JSON report instead includes market-implied binary payout "
                "variance and one-contract terminal P&L standard deviation.",
                "",
            ]
        )
    return "\n".join(lines)


def run_s1_feasibility(
    catalog_path: Path,
    output_root: Path,
    config: S1FeasibilityConfig,
) -> Dict[str, Any]:
    candles = load_candles(catalog_path, config.metal)
    report, signals, executions = analyze_s1_feasibility(candles, config)
    markouts = load_trade_markouts(catalog_path, signals, config)
    report["markout_study"] = summarize_trade_markout_study(
        markouts, signals, config
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_path = Path(output_root) / "derived" / "s1_feasibility" / "run={}".format(run_id)
    report_path = run_path / "report.json"
    signals_path = run_path / "signals.csv"
    executions_path = run_path / "executions.csv"
    markouts_path = run_path / "markouts.csv"
    markdown_path = run_path / "REPORT.md"
    _atomic_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    signal_rows = [_serialized_dataclass(row) for row in signals]
    execution_rows = [_serialized_dataclass(row) for row in executions]
    markout_rows = [_serialized_dataclass(row) for row in markouts]
    _atomic_text(
        signals_path,
        _csv_text(signal_rows, list(signal_rows[0])) if signal_rows else "",
    )
    _atomic_text(
        executions_path,
        _csv_text(execution_rows, list(execution_rows[0])) if execution_rows else "",
    )
    _atomic_text(
        markouts_path,
        _csv_text(markout_rows, list(markout_rows[0])) if markout_rows else "",
    )
    _atomic_text(markdown_path, _markdown_report(report))
    return {
        "run_id": run_id,
        "report_path": str(report_path),
        "signals_path": str(signals_path),
        "executions_path": str(executions_path),
        "markouts_path": str(markouts_path),
        "markdown_path": str(markdown_path),
        "decision": report["decision"],
        "eligible_signals": report["data"]["eligible_signals"],
    }
