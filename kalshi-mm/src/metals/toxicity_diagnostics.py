"""Exploratory Kalshi BBO conditional-fill toxicity diagnostics.

This is not a fair-value backtest. It studies hypothetical resting Yes bids and
Yes offers using Kalshi candles and trades only. Sub-minute marks are last-trade
proxies, and the touch/penetration fill rules do not reconstruct queue priority.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ZERO = Decimal("0")
ONE = Decimal("1")
HORIZONS = (("1s", 1), ("5s", 5), ("30s", 30), ("1m", 60), ("5m", 300), ("15m/end", 900))
HORIZON_ORDER = tuple(name for name, _ in HORIZONS)
TIME_BUCKET_ORDER = ("10-15m", "5-10m", "2-5m", "1-2m")
PRICE_BUCKET_ORDER = (
    "0-10c",
    "10-20c",
    "20-35c",
    "35-65c",
    "65-80c",
    "80-90c",
    "90-100c",
)
ACTION_ORDER = (
    "buying_market_favorite",
    "selling_market_favorite",
    "buying_market_underdog",
    "selling_market_underdog",
)


@dataclass(frozen=True)
class ToxicityConfig:
    metal: str = "silver"
    test_start_ts_ns: int = 1_787_301_900_000_000_000
    quote_lifetime_seconds: int = 60
    penetration_dollars: Decimal = Decimal("0.01")
    maker_fee_per_contract: Decimal = Decimal("0")
    taker_fee_multiplier: Decimal = Decimal("0.07")
    bootstrap_samples: int = 2_000
    bootstrap_seed: int = 17_291
    trim_fraction: Decimal = Decimal("0.01")
    require_completed_trade_request: bool = False


@dataclass(frozen=True)
class ToxicityMarkout:
    market_ticker: str
    split: str
    model: str
    yes_action: str
    action_category: str
    position_outcome_side: str
    market_open_ts_ns: int
    market_close_ts_ns: int
    decision_ts_ns: int
    seconds_to_close: int
    time_bucket: str
    quote_yes_price: Decimal
    quote_price_bucket: str
    economic_entry_price: Decimal
    decision_midpoint: Decimal
    entry_concession: Decimal
    fill_ts_ns: int
    fill_delay_seconds: Decimal
    trigger_yes_price: Decimal
    trigger_quantity: Decimal
    horizon: str
    horizon_seconds: int
    target_ts_ns: int
    mark_yes_price: Decimal
    mark_ts_ns: Optional[int]
    mark_age_seconds: Optional[Decimal]
    mark_source: str
    gross_markout_pnl: Decimal
    net_markout_pnl: Decimal


@dataclass(frozen=True)
class PairedLeg:
    market_ticker: str
    split: str
    model: str
    market_close_ts_ns: int
    decision_ts_ns: int
    seconds_to_close: int
    time_bucket: str
    market_result: str
    yes_bid: Decimal
    yes_offer: Decimal
    yes_fill_ts_ns: Optional[int]
    no_fill_ts_ns: Optional[int]
    yes_filled: bool
    no_filled: bool
    both_filled: bool
    fill_state: str
    fill_sequence: str
    fill_gap_seconds: Optional[Decimal]
    total_entry_cost: Decimal
    locked_gross_pnl: Decimal
    window_gross_pnl: Decimal
    window_net_pnl: Decimal


def _duckdb():
    try:
        import duckdb
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Toxicity diagnostics require duckdb") from error
    return duckdb


def price_bucket(price: Decimal) -> str:
    if price < Decimal("0.10"):
        return "0-10c"
    if price < Decimal("0.20"):
        return "10-20c"
    if price < Decimal("0.35"):
        return "20-35c"
    if price < Decimal("0.65"):
        return "35-65c"
    if price < Decimal("0.80"):
        return "65-80c"
    if price < Decimal("0.90"):
        return "80-90c"
    return "90-100c"


def action_category(yes_action: str, midpoint: Decimal) -> Optional[str]:
    if midpoint == Decimal("0.5"):
        return None
    yes_is_favorite = midpoint > Decimal("0.5")
    if yes_action == "buy_yes":
        return (
            "buying_market_favorite"
            if yes_is_favorite
            else "buying_market_underdog"
        )
    if yes_action == "sell_yes":
        return (
            "selling_market_favorite"
            if yes_is_favorite
            else "selling_market_underdog"
        )
    raise ValueError("Unsupported Yes action: {}".format(yes_action))


def _paired_window_pnl(
    result: str,
    yes_bid: Decimal,
    yes_offer: Decimal,
    yes_filled: bool,
    no_filled: bool,
    maker_fee_per_contract: Decimal,
) -> Tuple[str, Decimal, Decimal]:
    """Return exhaustive fill state and settlement P&L for one quote window."""

    normalized_result = result.lower()
    if normalized_result not in ("yes", "no"):
        raise ValueError("Paired window requires a binary Yes/No result")
    if yes_filled and no_filled:
        state = "both"
        gross = yes_offer - yes_bid
    elif yes_filled:
        state = "yes_only"
        gross = (ONE if normalized_result == "yes" else ZERO) - yes_bid
    elif no_filled:
        state = "no_only"
        gross = (
            (ONE if normalized_result == "no" else ZERO)
            - (ONE - yes_offer)
        )
    else:
        state = "neither"
        gross = ZERO
    filled_legs = int(yes_filled) + int(no_filled)
    net = gross - maker_fee_per_contract * Decimal(filled_legs)
    return state, gross, net


def _decimal(value: Any) -> Optional[Decimal]:
    if value in (None, ""):
        return None
    return Decimal(str(value))


def _sql_literal(value: Path) -> str:
    return str(value).replace("'", "''")


def _format_decimal(value: Optional[Decimal]) -> Optional[str]:
    if value is None:
        return None
    return format(value.quantize(Decimal("0.000001")), "f")


def _mean(values: Sequence[Decimal]) -> Optional[Decimal]:
    return sum(values, ZERO) / Decimal(len(values)) if values else None


def _percentile(values: Sequence[Decimal], probability: Decimal) -> Optional[Decimal]:
    if not values:
        return None
    ordered = sorted(values)
    index = int(Decimal(len(ordered) - 1) * probability)
    return ordered[index]


def _trimmed(values: Sequence[Decimal], fraction: Decimal) -> List[Decimal]:
    ordered = sorted(values)
    count = int(Decimal(len(ordered)) * fraction)
    if count == 0:
        return ordered
    if count * 2 >= len(ordered):
        return []
    return ordered[count:-count]


def _day_block_interval(
    rows: Sequence[ToxicityMarkout], samples: int, seed: int
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
    means = []
    for _ in range(samples):
        selected = [days[generator.randrange(len(days))] for _ in days]
        total = sum((sum(day, ZERO) for day in selected), ZERO)
        count = sum(len(day) for day in selected)
        means.append(total / Decimal(count))
    return [
        _format_decimal(_percentile(means, Decimal("0.05"))),
        _format_decimal(_percentile(means, Decimal("0.95"))),
    ]


def summarize_rows(
    rows: Sequence[ToxicityMarkout], config: ToxicityConfig
) -> Dict[str, Any]:
    pnls = [row.net_markout_pnl for row in rows]
    entries = [row.economic_entry_price for row in rows]
    quotes = [row.quote_yes_price for row in rows]
    marks = [row.mark_yes_price for row in rows]
    ages = [row.mark_age_seconds for row in rows if row.mark_age_seconds is not None]
    trimmed = _trimmed(pnls, config.trim_fraction)
    return {
        "observations": len(rows),
        "raw_mean_pnl_dollars": _format_decimal(_mean(pnls)),
        "median_pnl_dollars": _format_decimal(
            _percentile(pnls, Decimal("0.50"))
        ),
        "trimmed_mean_pnl_dollars": _format_decimal(_mean(trimmed)),
        "day_block_90pct_ci_dollars": _day_block_interval(
            rows, config.bootstrap_samples, config.bootstrap_seed
        ),
        "average_quote_yes_price_dollars": _format_decimal(_mean(quotes)),
        "average_economic_entry_price_dollars": _format_decimal(_mean(entries)),
        "average_yes_mark_price_dollars": _format_decimal(_mean(marks)),
        "average_mark_age_seconds": _format_decimal(_mean(ages)),
        "p95_mark_age_seconds": _format_decimal(
            _percentile(ages, Decimal("0.95"))
        ),
    }


def _candidate_sql(require_completed_trade_request: bool = False) -> str:
    trade_coverage_join = (
        "JOIN metals.kalshi_trade_request r USING (market_ticker) "
        if require_completed_trade_request
        else ""
    )
    trade_coverage_filter = (
        "AND r.status = 'completed' "
        if require_completed_trade_request
        else ""
    )
    return (
        "CREATE TEMP TABLE quote_candidate AS WITH base AS ("
        "SELECT c.market_ticker, m.open_ts_ns AS market_open_ts_ns, "
        "m.close_ts_ns AS market_close_ts_ns, c.end_period_ts_ns AS decision_ts_ns, "
        "CAST((m.close_ts_ns - c.end_period_ts_ns) / 1000000000 AS BIGINT) "
        "AS seconds_to_close, CAST(c.yes_bid_close_dollars AS DECIMAL(9, 4)) "
        "AS yes_bid, CAST(c.yes_ask_close_dollars AS DECIMAL(9, 4)) AS yes_ask, "
        "s.result, CASE "
        "WHEN (m.close_ts_ns - c.end_period_ts_ns) / 1000000000 BETWEEN 600 AND 900 "
        "THEN '10-15m' WHEN (m.close_ts_ns - c.end_period_ts_ns) / 1000000000 "
        ">= 300 THEN '5-10m' WHEN (m.close_ts_ns - c.end_period_ts_ns) / "
        "1000000000 >= 120 THEN '2-5m' WHEN (m.close_ts_ns - "
        "c.end_period_ts_ns) / 1000000000 >= 60 THEN '1-2m' END AS time_bucket "
        "FROM metals.kalshi_candlestick c JOIN metals.market_dim m "
        "USING (market_ticker) JOIN metals.kalshi_settlement s USING (market_ticker) "
        + trade_coverage_join
        + "WHERE c.metal = ? AND m.metal = ? AND c.period_interval_minutes = 1 "
        + trade_coverage_filter
        + "AND lower(s.result) IN ('yes', 'no') "
        "AND CAST(c.yes_bid_close_dollars AS DECIMAL(9, 4)) > 0 "
        "AND CAST(c.yes_ask_close_dollars AS DECIMAL(9, 4)) < 1 "
        "AND CAST(c.yes_bid_close_dollars AS DECIMAL(9, 4)) <= "
        "CAST(c.yes_ask_close_dollars AS DECIMAL(9, 4)) "
        "AND (m.close_ts_ns - c.end_period_ts_ns) / 1000000000 BETWEEN 60 AND 900), "
        "selected AS (SELECT *, (yes_bid + yes_ask) / 2 AS midpoint FROM base "
        "WHERE time_bucket IS NOT NULL QUALIFY row_number() OVER (PARTITION BY "
        "market_ticker, time_bucket ORDER BY decision_ts_ns) = 1), actions AS ("
        "SELECT *, 'buy_yes' AS yes_action, yes_bid AS quote_yes_price FROM selected "
        "UNION ALL SELECT *, 'sell_yes', yes_ask FROM selected) "
        "SELECT *, CASE WHEN market_open_ts_ns >= ? THEN 'test' ELSE 'train' END "
        "AS split, CASE WHEN quote_yes_price < 0.10 THEN '0-10c' "
        "WHEN quote_yes_price < 0.20 THEN '10-20c' "
        "WHEN quote_yes_price < 0.35 THEN '20-35c' "
        "WHEN quote_yes_price < 0.65 THEN '35-65c' "
        "WHEN quote_yes_price < 0.80 THEN '65-80c' "
        "WHEN quote_yes_price < 0.90 THEN '80-90c' ELSE '90-100c' END "
        "AS quote_price_bucket, CASE WHEN midpoint = 0.5 THEN 'neutral' "
        "WHEN yes_action = 'buy_yes' AND midpoint > 0.5 THEN 'buying_market_favorite' "
        "WHEN yes_action = 'sell_yes' AND midpoint > 0.5 THEN 'selling_market_favorite' "
        "WHEN yes_action = 'buy_yes' THEN 'buying_market_underdog' "
        "ELSE 'selling_market_underdog' END AS action_category, "
        "CASE WHEN yes_action = 'buy_yes' THEN 'yes' ELSE 'no' END "
        "AS position_outcome_side, CASE WHEN yes_action = 'buy_yes' "
        "THEN quote_yes_price ELSE 1 - quote_yes_price END AS economic_entry_price, "
        "CASE WHEN yes_action = 'buy_yes' THEN midpoint - quote_yes_price "
        "ELSE quote_yes_price - midpoint END AS entry_concession FROM actions"
    )


def load_diagnostics(
    catalog_path: Path, config: ToxicityConfig
) -> Tuple[List[ToxicityMarkout], List[PairedLeg], Dict[str, int]]:
    """Build generic two-sided quote candidates and trade-anchored markouts."""

    connection = _duckdb().connect()
    try:
        connection.execute(
            "ATTACH '{}' AS metals (READ_ONLY)".format(
                _sql_literal(Path(catalog_path))
            )
        )
        connection.execute(
            _candidate_sql(config.require_completed_trade_request),
            [config.metal, config.metal, config.test_start_ts_ns],
        )
        candidate_count = connection.execute(
            "SELECT count(*) FROM quote_candidate"
        ).fetchone()[0]
        candidate_market_count = connection.execute(
            "SELECT count(DISTINCT market_ticker) FROM quote_candidate"
        ).fetchone()[0]
        decision_count = connection.execute(
            "SELECT count(*) FROM quote_candidate WHERE yes_action = 'buy_yes'"
        ).fetchone()[0]
        neutral_count = connection.execute(
            "SELECT count(*) FROM quote_candidate WHERE action_category = 'neutral'"
        ).fetchone()[0]
        connection.execute(
            "CREATE TEMP TABLE anchored_fill AS WITH models AS ("
            "SELECT *, 'touch' AS model, 0::DECIMAL(9, 4) AS penetration "
            "FROM quote_candidate UNION ALL SELECT *, 'penetration_1c', "
            "?::DECIMAL(9, 4) FROM quote_candidate) "
            "SELECT q.*, t.source_event_ts_ns AS fill_ts_ns, t.trade_id, "
            "CAST(t.yes_price_dollars AS DECIMAL(9, 4)) AS trigger_yes_price, "
            "CAST(t.quantity_fp AS DECIMAL(18, 4)) AS trigger_quantity "
            "FROM models q JOIN metals.kalshi_trade t "
            "ON t.market_ticker = q.market_ticker "
            "AND t.source_event_ts_ns > q.decision_ts_ns "
            "AND t.source_event_ts_ns <= q.decision_ts_ns + ? "
            "AND ((q.yes_action = 'buy_yes' AND t.taker_outcome_side = 'no' "
            "AND CAST(t.yes_price_dollars AS DECIMAL(9, 4)) <= "
            "q.quote_yes_price - q.penetration) OR "
            "(q.yes_action = 'sell_yes' AND t.taker_outcome_side = 'yes' "
            "AND CAST(t.yes_price_dollars AS DECIMAL(9, 4)) >= "
            "q.quote_yes_price + q.penetration)) "
            "QUALIFY row_number() OVER (PARTITION BY q.market_ticker, "
            "q.decision_ts_ns, q.yes_action, q.model ORDER BY "
            "t.source_event_ts_ns, t.trade_id) = 1",
            [
                config.penetration_dollars,
                config.quote_lifetime_seconds * 1_000_000_000,
            ],
        )
        fill_count = connection.execute(
            "SELECT count(*) FROM anchored_fill"
        ).fetchone()[0]
        connection.execute(
            "CREATE TEMP TABLE trade_mark AS SELECT t.market_ticker, "
            "t.source_event_ts_ns, CAST(sum(CAST(t.yes_price_dollars "
            "AS DECIMAL(9, 4)) * CAST(t.quantity_fp AS DECIMAL(18, 4))) / "
            "sum(CAST(t.quantity_fp AS DECIMAL(18, 4))) AS DECIMAL(9, 4)) "
            "AS yes_price FROM metals.kalshi_trade t SEMI JOIN "
            "(SELECT DISTINCT market_ticker FROM anchored_fill) f "
            "USING (market_ticker) GROUP BY 1, 2"
        )
        connection.execute(
            "CREATE TEMP TABLE horizon (horizon VARCHAR, horizon_seconds BIGINT)"
        )
        connection.executemany("INSERT INTO horizon VALUES (?, ?)", HORIZONS)
        raw_rows = connection.execute(
            "WITH desired AS (SELECT f.*, h.horizon, h.horizon_seconds, "
            "least(f.fill_ts_ns + h.horizon_seconds * 1000000000, "
            "f.market_close_ts_ns) AS target_ts_ns, "
            "(h.horizon = '15m/end' OR f.fill_ts_ns + h.horizon_seconds * "
            "1000000000 >= f.market_close_ts_ns) AS terminal FROM anchored_fill f "
            "CROSS JOIN horizon h), trade_desired AS (SELECT * FROM desired "
            "WHERE NOT terminal), trade_joined AS (SELECT d.*, "
            "m.source_event_ts_ns AS mark_ts_ns, m.yes_price AS mark_yes_price, "
            "'trade' AS mark_source FROM trade_desired d ASOF LEFT JOIN "
            "trade_mark m ON d.market_ticker = m.market_ticker AND "
            "d.target_ts_ns >= m.source_event_ts_ns), terminal_rows AS ("
            "SELECT d.*, NULL::BIGINT AS mark_ts_ns, CASE WHEN lower(d.result) = "
            "'yes' THEN 1::DECIMAL(9, 4) ELSE 0::DECIMAL(9, 4) END "
            "AS mark_yes_price, 'settlement' AS mark_source FROM desired d "
            "WHERE terminal), combined AS (SELECT * FROM trade_joined UNION ALL "
            "BY NAME SELECT * FROM terminal_rows) SELECT market_ticker, split, "
            "model, yes_action, action_category, position_outcome_side, "
            "market_open_ts_ns, market_close_ts_ns, decision_ts_ns, "
            "seconds_to_close, time_bucket, quote_yes_price, quote_price_bucket, "
            "economic_entry_price, midpoint, entry_concession, fill_ts_ns, "
            "trigger_yes_price, trigger_quantity, horizon, horizon_seconds, "
            "target_ts_ns, mark_yes_price, mark_ts_ns, mark_source FROM combined "
            "WHERE mark_yes_price IS NOT NULL ORDER BY split, model, "
            "market_ticker, decision_ts_ns, yes_action, horizon_seconds"
        ).fetchall()

        pair_rows = connection.execute(
            "WITH models AS (SELECT *, 'touch' AS model FROM quote_candidate "
            "WHERE yes_action = 'buy_yes' UNION ALL SELECT *, 'penetration_1c' "
            "FROM quote_candidate WHERE yes_action = 'buy_yes'), "
            "base AS (SELECT market_ticker, split, model, market_close_ts_ns, "
            "decision_ts_ns, seconds_to_close, time_bucket, result, yes_bid, yes_ask "
            "FROM models), yes_fill AS (SELECT "
            "market_ticker, decision_ts_ns, model, fill_ts_ns FROM anchored_fill "
            "WHERE yes_action = 'buy_yes'), no_fill AS (SELECT market_ticker, "
            "decision_ts_ns, model, fill_ts_ns FROM anchored_fill WHERE "
            "yes_action = 'sell_yes') SELECT b.market_ticker, b.split, b.model, "
            "b.market_close_ts_ns, b.decision_ts_ns, b.seconds_to_close, "
            "b.time_bucket, b.result, b.yes_bid, b.yes_ask, y.fill_ts_ns, "
            "n.fill_ts_ns "
            "FROM base b LEFT JOIN yes_fill y USING (market_ticker, "
            "decision_ts_ns, model) LEFT JOIN no_fill n USING (market_ticker, "
            "decision_ts_ns, model) ORDER BY b.split, b.model, "
            "b.market_ticker, b.decision_ts_ns"
        ).fetchall()
    finally:
        connection.close()

    markouts: List[ToxicityMarkout] = []
    for row in raw_rows:
        quote = _decimal(row[11]) or ZERO
        entry = _decimal(row[13]) or ZERO
        midpoint = _decimal(row[14]) or ZERO
        concession = _decimal(row[15]) or ZERO
        fill_ts = int(row[16])
        target_ts = int(row[21])
        mark = _decimal(row[22]) or ZERO
        mark_ts = int(row[23]) if row[23] is not None else None
        yes_action = str(row[3])
        gross = mark - quote if yes_action == "buy_yes" else quote - mark
        markouts.append(
            ToxicityMarkout(
                market_ticker=str(row[0]),
                split=str(row[1]),
                model=str(row[2]),
                yes_action=yes_action,
                action_category=str(row[4]),
                position_outcome_side=str(row[5]),
                market_open_ts_ns=int(row[6]),
                market_close_ts_ns=int(row[7]),
                decision_ts_ns=int(row[8]),
                seconds_to_close=int(row[9]),
                time_bucket=str(row[10]),
                quote_yes_price=quote,
                quote_price_bucket=str(row[12]),
                economic_entry_price=entry,
                decision_midpoint=midpoint,
                entry_concession=concession,
                fill_ts_ns=fill_ts,
                fill_delay_seconds=Decimal(fill_ts - int(row[8]))
                / Decimal(1_000_000_000),
                trigger_yes_price=_decimal(row[17]) or ZERO,
                trigger_quantity=_decimal(row[18]) or ZERO,
                horizon=str(row[19]),
                horizon_seconds=int(row[20]),
                target_ts_ns=target_ts,
                mark_yes_price=mark,
                mark_ts_ns=mark_ts,
                mark_age_seconds=(
                    Decimal(target_ts - mark_ts) / Decimal(1_000_000_000)
                    if mark_ts is not None
                    else None
                ),
                mark_source=str(row[24]),
                gross_markout_pnl=gross,
                net_markout_pnl=gross - config.maker_fee_per_contract,
            )
        )

    pairs: List[PairedLeg] = []
    for row in pair_rows:
        yes_fill = int(row[10]) if row[10] is not None else None
        no_fill = int(row[11]) if row[11] is not None else None
        both = yes_fill is not None and no_fill is not None
        if both and yes_fill < no_fill:
            sequence = "yes_then_no"
        elif both and no_fill < yes_fill:
            sequence = "no_then_yes"
        elif both:
            sequence = "same_timestamp"
        elif yes_fill is not None:
            sequence = "yes_only"
        elif no_fill is not None:
            sequence = "no_only"
        else:
            sequence = "neither"
        result = str(row[7]).lower()
        bid = _decimal(row[8]) or ZERO
        offer = _decimal(row[9]) or ZERO
        fill_state, window_gross, window_net = _paired_window_pnl(
            result=result,
            yes_bid=bid,
            yes_offer=offer,
            yes_filled=yes_fill is not None,
            no_filled=no_fill is not None,
            maker_fee_per_contract=config.maker_fee_per_contract,
        )
        pairs.append(
            PairedLeg(
                market_ticker=str(row[0]),
                split=str(row[1]),
                model=str(row[2]),
                market_close_ts_ns=int(row[3]),
                decision_ts_ns=int(row[4]),
                seconds_to_close=int(row[5]),
                time_bucket=str(row[6]),
                market_result=result,
                yes_bid=bid,
                yes_offer=offer,
                yes_fill_ts_ns=yes_fill,
                no_fill_ts_ns=no_fill,
                yes_filled=yes_fill is not None,
                no_filled=no_fill is not None,
                both_filled=both,
                fill_state=fill_state,
                fill_sequence=sequence,
                fill_gap_seconds=(
                    Decimal(abs(yes_fill - no_fill)) / Decimal(1_000_000_000)
                    if both
                    else None
                ),
                total_entry_cost=bid + (ONE - offer),
                locked_gross_pnl=offer - bid,
                window_gross_pnl=window_gross,
                window_net_pnl=window_net,
            )
        )
    return markouts, pairs, {
        "quote_candidates": int(candidate_count),
        "analytical_markets": int(candidate_market_count),
        "two_sided_quote_decisions": int(decision_count),
        "neutral_candidates": int(neutral_count),
        "anchored_fills": int(fill_count),
        "markout_rows": len(markouts),
    }


def _summary_record(
    rows: Sequence[ToxicityMarkout], config: ToxicityConfig, **dimensions: Any
) -> Dict[str, Any]:
    return {**dimensions, **summarize_rows(rows, config)}


def build_action_smile(
    markouts: Sequence[ToxicityMarkout], config: ToxicityConfig
) -> List[Dict[str, Any]]:
    records = []
    for split in ("train", "test"):
        for model in ("touch", "penetration_1c"):
            for category in ("all", *ACTION_ORDER):
                for horizon in HORIZON_ORDER:
                    rows = [
                        row
                        for row in markouts
                        if row.split == split
                        and row.model == model
                        and row.horizon == horizon
                        and (
                            category == "all"
                            or row.action_category == category
                        )
                    ]
                    records.append(
                        _summary_record(
                            rows,
                            config,
                            split=split,
                            model=model,
                            action_category=category,
                            horizon=horizon,
                        )
                    )
    return records


def build_price_time_cut(
    markouts: Sequence[ToxicityMarkout], config: ToxicityConfig
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], List[ToxicityMarkout]] = defaultdict(list)
    for row in markouts:
        if row.split != "test":
            continue
        grouped[
            (
                row.model,
                row.quote_price_bucket,
                row.time_bucket,
                row.horizon,
            )
        ].append(row)
    records = []
    for model in ("touch", "penetration_1c"):
        for price in PRICE_BUCKET_ORDER:
            for time_bucket in TIME_BUCKET_ORDER:
                for horizon in HORIZON_ORDER:
                    rows = grouped[(model, price, time_bucket, horizon)]
                    records.append(
                        _summary_record(
                            rows,
                            config,
                            split="test",
                            model=model,
                            yes_action="all",
                            quote_price_bucket=price,
                            time_bucket=time_bucket,
                            horizon=horizon,
                        )
                    )
    return records


def build_exit_frontier(
    markouts: Sequence[ToxicityMarkout], config: ToxicityConfig
) -> List[Dict[str, Any]]:
    records = []
    for model in ("touch", "penetration_1c"):
        for category in ("all", *ACTION_ORDER):
            for horizon in HORIZON_ORDER:
                rows = [
                    row
                    for row in markouts
                    if row.split == "test"
                    and row.model == model
                    and row.horizon == horizon
                    and (
                        category == "all"
                        or row.action_category == category
                    )
                ]
                gross = [row.gross_markout_pnl for row in rows]
                fees = []
                for row in rows:
                    if row.mark_source == "settlement":
                        fees.append(ZERO)
                    else:
                        economic_mark = (
                            row.mark_yes_price
                            if row.yes_action == "buy_yes"
                            else ONE - row.mark_yes_price
                        )
                        fees.append(
                            config.taker_fee_multiplier
                            * economic_mark
                            * (ONE - economic_mark)
                        )
                gross_mean = _mean(gross)
                fee_mean = _mean(fees)
                after_fee = (
                    gross_mean - fee_mean
                    if gross_mean is not None and fee_mean is not None
                    else None
                )
                records.append(
                    {
                        "model": model,
                        "action_category": category,
                        "horizon": horizon,
                        "observations": len(rows),
                        "gross_markout_before_exit_cost_dollars": (
                            _format_decimal(gross_mean)
                        ),
                        "indicative_unrounded_taker_exit_fee_dollars": (
                            _format_decimal(fee_mean)
                        ),
                        "pnl_after_fee_only_dollars": _format_decimal(after_fee),
                        "maximum_break_even_additional_exit_cost_dollars": (
                            _format_decimal(max(after_fee, ZERO))
                            if after_fee is not None
                            else None
                        ),
                        "executable_exit_ev_available": False,
                    }
                )
    return records


def build_waterfall(markouts: Sequence[ToxicityMarkout]) -> List[Dict[str, Any]]:
    records = []
    for model in ("touch", "penetration_1c"):
        for category in ("all", *ACTION_ORDER):
            rows = [
                row
                for row in markouts
                if row.split == "test"
                and row.model == model
                and (
                    category == "all"
                    or row.action_category == category
                )
            ]
            by_fill: Dict[Tuple[Any, ...], Dict[str, ToxicityMarkout]] = defaultdict(dict)
            for row in rows:
                key = (
                    row.market_ticker,
                    row.decision_ts_ns,
                    row.yes_action,
                    row.fill_ts_ns,
                )
                by_fill[key][row.horizon] = row
            components: Dict[str, List[Decimal]] = defaultdict(list)
            for horizons in by_fill.values():
                if not {"1s", "1m", "15m/end"}.issubset(horizons):
                    continue
                one_second = horizons["1s"]
                one_minute = horizons["1m"]
                terminal = horizons["15m/end"]
                concession = one_second.entry_concession
                components["entry_concession"].append(concession)
                components["fill_to_1s_selection"].append(
                    one_second.gross_markout_pnl - concession
                )
                components["1s_to_1m_drift"].append(
                    one_minute.gross_markout_pnl
                    - one_second.gross_markout_pnl
                )
                components["1m_to_settlement_drift"].append(
                    terminal.gross_markout_pnl
                    - one_minute.gross_markout_pnl
                )
                components["fees"].append(
                    terminal.net_markout_pnl - terminal.gross_markout_pnl
                )
            for component in (
                "entry_concession",
                "fill_to_1s_selection",
                "1s_to_1m_drift",
                "1m_to_settlement_drift",
                "fees",
            ):
                records.append(
                    {
                        "model": model,
                        "action_category": category,
                        "component": component,
                        "observations": len(components[component]),
                        "average_contribution_dollars": _format_decimal(
                            _mean(components[component])
                        ),
                    }
                )
            terminal_values = [
                horizons["15m/end"].net_markout_pnl
                for horizons in by_fill.values()
                if "15m/end" in horizons
            ]
            records.append(
                {
                    "model": model,
                    "action_category": category,
                    "component": "settlement_pnl",
                    "observations": len(terminal_values),
                    "average_contribution_dollars": _format_decimal(
                        _mean(terminal_values)
                    ),
                }
            )
    return records


def build_concentration(
    markouts: Sequence[ToxicityMarkout], config: ToxicityConfig
) -> List[Dict[str, Any]]:
    records = []
    for model in ("touch", "penetration_1c"):
        for category in ("all", *ACTION_ORDER):
            rows = [
                row
                for row in markouts
                if row.split == "test"
                and row.model == model
                and row.horizon == "15m/end"
                and (
                    category == "all"
                    or row.action_category == category
                )
            ]
            values = sorted(row.net_markout_pnl for row in rows)
            trim_count = int(Decimal(len(values)) * config.trim_fraction)
            without_worst = values[trim_count:] if trim_count else values
            without_best = values[:-trim_count] if trim_count else values
            without_both = (
                values[trim_count:-trim_count] if trim_count else values
            )
            by_day: Dict[str, List[Decimal]] = defaultdict(list)
            for row in rows:
                day = datetime.fromtimestamp(
                    row.market_close_ts_ns / 1_000_000_000,
                    timezone.utc,
                ).date().isoformat()
                by_day[day].append(row.net_markout_pnl)
            day_means = {
                day: _mean(day_values)
                for day, day_values in by_day.items()
            }
            ordered_days = sorted(
                day_means,
                key=lambda day: day_means[day]
                if day_means[day] is not None
                else ZERO,
            )
            excluded_days = set()
            if ordered_days:
                excluded_days.update((ordered_days[0], ordered_days[-1]))
            excluding_extreme_days = [
                row.net_markout_pnl
                for row in rows
                if datetime.fromtimestamp(
                    row.market_close_ts_ns / 1_000_000_000,
                    timezone.utc,
                ).date().isoformat()
                not in excluded_days
            ]
            records.append(
                {
                    "model": model,
                    "action_category": category,
                    "observations": len(values),
                    "trim_count_each_tail": trim_count,
                    "raw_mean_pnl_dollars": _format_decimal(_mean(values)),
                    "mean_without_worst_1pct_dollars": _format_decimal(
                        _mean(without_worst)
                    ),
                    "mean_without_best_1pct_dollars": _format_decimal(
                        _mean(without_best)
                    ),
                    "mean_without_best_and_worst_1pct_dollars": (
                        _format_decimal(_mean(without_both))
                    ),
                    "utc_days": len(day_means),
                    "negative_mean_days": sum(
                        value is not None and value < ZERO
                        for value in day_means.values()
                    ),
                    "worst_day": ordered_days[0] if ordered_days else None,
                    "worst_day_mean_pnl_dollars": _format_decimal(
                        day_means[ordered_days[0]] if ordered_days else None
                    ),
                    "best_day": ordered_days[-1] if ordered_days else None,
                    "best_day_mean_pnl_dollars": _format_decimal(
                        day_means[ordered_days[-1]] if ordered_days else None
                    ),
                    "pooled_mean_excluding_best_and_worst_day_dollars": (
                        _format_decimal(_mean(excluding_extreme_days))
                    ),
                }
            )
    return records


def _pair_day_block_interval(
    pairs: Sequence[PairedLeg], samples: int, seed: int
) -> List[Optional[str]]:
    by_day: Dict[str, List[Decimal]] = defaultdict(list)
    for row in pairs:
        day = datetime.fromtimestamp(
            row.market_close_ts_ns / 1_000_000_000, timezone.utc
        ).date().isoformat()
        by_day[day].append(row.window_net_pnl)
    days = [by_day[day] for day in sorted(by_day)]
    if not days:
        return [None, None]
    generator = random.Random(seed)
    means = []
    for _ in range(samples):
        selected = [days[generator.randrange(len(days))] for _ in days]
        total = sum((sum(day, ZERO) for day in selected), ZERO)
        count = sum(len(day) for day in selected)
        means.append(total / Decimal(count))
    return [
        _format_decimal(_percentile(means, Decimal("0.05"))),
        _format_decimal(_percentile(means, Decimal("0.95"))),
    ]


def summarize_pairs(
    pairs: Sequence[PairedLeg], config: ToxicityConfig
) -> List[Dict[str, Any]]:
    records = []
    for split in ("train", "test"):
        for model in ("touch", "penetration_1c"):
            for time_bucket in ("all", *TIME_BUCKET_ORDER):
                rows = [
                    row
                    for row in pairs
                    if row.split == split
                    and row.model == model
                    and (
                        time_bucket == "all"
                        or row.time_bucket == time_bucket
                    )
                ]
                both = [row for row in rows if row.both_filled]
                gap = [row.fill_gap_seconds for row in both if row.fill_gap_seconds is not None]
                locked = [row.locked_gross_pnl for row in both]
                window_gross = [row.window_gross_pnl for row in rows]
                window_net = [row.window_net_pnl for row in rows]
                trimmed_window_net = _trimmed(window_net, config.trim_fraction)
                by_day: Dict[str, List[Decimal]] = defaultdict(list)
                for row in rows:
                    day = datetime.fromtimestamp(
                        row.market_close_ts_ns / 1_000_000_000,
                        timezone.utc,
                    ).date().isoformat()
                    by_day[day].append(row.window_net_pnl)
                day_means = {
                    day: _mean(values) for day, values in by_day.items()
                }
                ordered_days = sorted(
                    day_means,
                    key=lambda day: (
                        day_means[day]
                        if day_means[day] is not None
                        else ZERO
                    ),
                )
                excluded_days = set()
                if ordered_days:
                    excluded_days.update((ordered_days[0], ordered_days[-1]))
                excluding_extreme_days = [
                    row.window_net_pnl
                    for row in rows
                    if datetime.fromtimestamp(
                        row.market_close_ts_ns / 1_000_000_000,
                        timezone.utc,
                    ).date().isoformat()
                    not in excluded_days
                ]
                records.append(
                    {
                        "split": split,
                        "model": model,
                        "time_bucket": time_bucket,
                        "two_sided_quote_decisions": len(rows),
                        "yes_leg_fills": sum(row.yes_filled for row in rows),
                        "no_leg_fills": sum(row.no_filled for row in rows),
                        "both_legs_fill": len(both),
                        "yes_only": sum(row.fill_state == "yes_only" for row in rows),
                        "no_only": sum(row.fill_state == "no_only" for row in rows),
                        "single_leg_fill": sum(
                            row.fill_state in ("yes_only", "no_only") for row in rows
                        ),
                        "neither_fills": sum(
                            row.fill_state == "neither" for row in rows
                        ),
                        "both_fill_rate": _format_decimal(
                            Decimal(len(both)) / Decimal(len(rows))
                            if rows
                            else None
                        ),
                        "yes_then_no": sum(
                            row.fill_sequence == "yes_then_no" for row in both
                        ),
                        "no_then_yes": sum(
                            row.fill_sequence == "no_then_yes" for row in both
                        ),
                        "same_timestamp": sum(
                            row.fill_sequence == "same_timestamp" for row in both
                        ),
                        "average_fill_gap_seconds": _format_decimal(_mean(gap)),
                        "average_locked_gross_pnl_dollars": _format_decimal(
                            _mean(locked)
                        ),
                        "median_locked_gross_pnl_dollars": _format_decimal(
                            _percentile(locked, Decimal("0.50"))
                        ),
                        "average_window_gross_pnl_dollars": _format_decimal(
                            _mean(window_gross)
                        ),
                        "average_window_net_pnl_dollars": _format_decimal(
                            _mean(window_net)
                        ),
                        "trimmed_mean_window_net_pnl_dollars": _format_decimal(
                            _mean(trimmed_window_net)
                        ),
                        "window_net_day_block_90pct_ci_dollars": (
                            _pair_day_block_interval(
                                rows,
                                config.bootstrap_samples,
                                config.bootstrap_seed,
                            )
                        ),
                        "utc_days": len(day_means),
                        "negative_mean_days": sum(
                            value is not None and value < ZERO
                            for value in day_means.values()
                        ),
                        "worst_day": ordered_days[0] if ordered_days else None,
                        "worst_day_mean_window_net_pnl_dollars": _format_decimal(
                            day_means[ordered_days[0]] if ordered_days else None
                        ),
                        "best_day": ordered_days[-1] if ordered_days else None,
                        "best_day_mean_window_net_pnl_dollars": _format_decimal(
                            day_means[ordered_days[-1]] if ordered_days else None
                        ),
                        "pooled_mean_excluding_best_and_worst_day_dollars": (
                            _format_decimal(_mean(excluding_extreme_days))
                        ),
                    }
                )
    return records


def _atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".toxicity-", suffix=".tmp"
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


def _serialized(instance: Any) -> Dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Decimal) else value
        for key, value in asdict(instance).items()
    }


def _iso_from_ns(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    return datetime.fromtimestamp(
        value / 1_000_000_000, timezone.utc
    ).isoformat()


def load_catalog_coverage(
    catalog_path: Path, product: str
) -> Dict[str, Any]:
    """Summarize the exact local inputs used by a diagnostics run."""

    connection = _duckdb().connect(str(catalog_path), read_only=True)
    try:
        available_tables = {
            row[0] for row in connection.execute("SHOW TABLES").fetchall()
        }
        market = connection.execute(
            "SELECT count(*), min(open_ts_ns), max(open_ts_ns), "
            "min(close_ts_ns), max(close_ts_ns) FROM market_dim "
            "WHERE metal = ?",
            [product],
        ).fetchone()
        candles = connection.execute(
            "SELECT count(*), count(DISTINCT market_ticker), "
            "min(end_period_ts_ns), max(end_period_ts_ns) "
            "FROM kalshi_candlestick WHERE metal = ? "
            "AND period_interval_minutes = 1",
            [product],
        ).fetchone()
        trades = connection.execute(
            "SELECT count(*), count(DISTINCT t.market_ticker), "
            "min(t.source_event_ts_ns), max(t.source_event_ts_ns) "
            "FROM kalshi_trade t WHERE t.market_ticker IN "
            "(SELECT market_ticker FROM market_dim WHERE metal = ?)",
            [product],
        ).fetchone()
        settlements = connection.execute(
            "SELECT count(*) FROM kalshi_settlement s "
            "WHERE s.market_ticker IN "
            "(SELECT market_ticker FROM market_dim WHERE metal = ?) "
            "AND lower(s.result) IN ('yes', 'no')",
            [product],
        ).fetchone()
        completed_trade_markets = None
        completed_trade_utc_days = None
        if "kalshi_trade_request" in available_tables:
            completed_trade_markets = connection.execute(
                "SELECT count(*) FROM kalshi_trade_request r JOIN "
                "market_dim m USING (market_ticker) "
                "WHERE r.status = 'completed' AND m.metal = ?",
                [product],
            ).fetchone()[0]
            completed_trade_utc_days = connection.execute(
                "SELECT count(DISTINCT CAST(timezone('UTC', "
                "to_timestamp(m.close_ts_ns / 1000000000.0)) AS DATE)) "
                "FROM market_dim m JOIN "
                "kalshi_trade_request r USING (market_ticker) "
                "WHERE r.status = 'completed' AND m.metal = ?",
                [product],
            ).fetchone()[0]
    finally:
        connection.close()
    return {
        "product": product,
        "markets": int(market[0]),
        "market_open_first": _iso_from_ns(market[1]),
        "market_open_last": _iso_from_ns(market[2]),
        "market_close_first": _iso_from_ns(market[3]),
        "market_close_last": _iso_from_ns(market[4]),
        "binary_settlements": int(settlements[0]),
        "one_minute_candles": int(candles[0]),
        "candle_markets": int(candles[1]),
        "candle_end_first": _iso_from_ns(candles[2]),
        "candle_end_last": _iso_from_ns(candles[3]),
        "trades": int(trades[0]),
        "trade_markets": int(trades[1]),
        "trade_first": _iso_from_ns(trades[2]),
        "trade_last": _iso_from_ns(trades[3]),
        "completed_trade_request_markets": (
            int(completed_trade_markets)
            if completed_trade_markets is not None
            else None
        ),
        "completed_trade_request_utc_days": (
            int(completed_trade_utc_days)
            if completed_trade_utc_days is not None
            else None
        ),
    }


def _record_value(
    rows: Sequence[Mapping[str, Any]], field: str, **filters: str
) -> Optional[float]:
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            value = row.get(field)
            return float(value) if value is not None else None
    return None


def _plot_action_smile(
    action_rows: Sequence[Mapping[str, Any]], path: Path
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/kalshi-mm-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "buying_market_favorite": "Buy favorite",
        "selling_market_favorite": "Sell favorite",
        "buying_market_underdog": "Buy underdog",
        "selling_market_underdog": "Sell underdog",
    }
    colors = ("#3366cc", "#dc3912", "#109618", "#990099")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x = list(range(len(HORIZON_ORDER)))
    for axis, model in zip(axes, ("touch", "penetration_1c")):
        for category, color in zip(ACTION_ORDER, colors):
            values = [
                _record_value(
                    action_rows,
                    "raw_mean_pnl_dollars",
                    split="test",
                    model=model,
                    action_category=category,
                    horizon=horizon,
                )
                for horizon in HORIZON_ORDER
            ]
            axis.plot(
                x,
                [value * 100 if value is not None else math.nan for value in values],
                marker="o",
                linewidth=1.8,
                label=labels[category],
                color=color,
            )
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.set_xticks(x, HORIZON_ORDER)
        axis.set_title("Touch" if model == "touch" else "1¢ penetration")
        axis.set_xlabel("Horizon")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Average signed markout P&L (¢/contract)")
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Held-out conditional-fill toxicity by Yes-side action")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_price_time(
    price_rows: Sequence[Mapping[str, Any]], path: Path
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/kalshi-mm-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    matrices = []
    counts = []
    for model in ("touch", "penetration_1c"):
        matrix = np.full((len(PRICE_BUCKET_ORDER), len(TIME_BUCKET_ORDER)), np.nan)
        count = np.zeros_like(matrix)
        for row_index, price in enumerate(PRICE_BUCKET_ORDER):
            for column_index, time_bucket in enumerate(TIME_BUCKET_ORDER):
                value = _record_value(
                    price_rows,
                    "raw_mean_pnl_dollars",
                    split="test",
                    model=model,
                    yes_action="all",
                    quote_price_bucket=price,
                    time_bucket=time_bucket,
                    horizon="15m/end",
                )
                observations = _record_value(
                    price_rows,
                    "observations",
                    split="test",
                    model=model,
                    yes_action="all",
                    quote_price_bucket=price,
                    time_bucket=time_bucket,
                    horizon="15m/end",
                )
                if value is not None:
                    matrix[row_index, column_index] = value * 100
                count[row_index, column_index] = observations or 0
        matrices.append(matrix)
        counts.append(count)
    maximum = max(
        abs(float(np.nanmin(matrix))) for matrix in matrices
    )
    maximum = max(
        maximum,
        max(abs(float(np.nanmax(matrix))) for matrix in matrices),
        0.01,
    )
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for axis, model, matrix, count in zip(
        axes, ("touch", "penetration_1c"), matrices, counts
    ):
        image = axis.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
        axis.set_xticks(range(len(TIME_BUCKET_ORDER)), TIME_BUCKET_ORDER)
        axis.set_yticks(range(len(PRICE_BUCKET_ORDER)), PRICE_BUCKET_ORDER)
        axis.set_xlabel("Time remaining")
        axis.set_title("Touch" if model == "touch" else "1¢ penetration")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                if math.isnan(matrix[row_index, column_index]):
                    label = "—"
                else:
                    label = "{:+.1f}¢\nn={}".format(
                        matrix[row_index, column_index],
                        int(count[row_index, column_index]),
                    )
                axis.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    axes[0].set_ylabel("Quoted Yes price")
    figure.subplots_adjust(
        top=0.86, bottom=0.12, left=0.10, right=0.86, wspace=0.10
    )
    color_axis = figure.add_axes([0.89, 0.20, 0.015, 0.60])
    figure.colorbar(
        image,
        cax=color_axis,
        label="Settlement P&L (¢/contract)",
    )
    figure.suptitle("Held-out settlement P&L by quoted Yes price × time")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_waterfall(
    waterfall_rows: Sequence[Mapping[str, Any]], path: Path
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/kalshi-mm-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    components = (
        "entry_concession",
        "fill_to_1s_selection",
        "1s_to_1m_drift",
        "1m_to_settlement_drift",
        "fees",
    )
    labels = (
        "Entry\nconcession",
        "Fill→1s\nselection",
        "1s→1m\ndrift",
        "1m→end\ndrift",
        "Fees",
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, model in zip(axes, ("touch", "penetration_1c")):
        values = [
            (_record_value(
                waterfall_rows,
                "average_contribution_dollars",
                model=model,
                action_category="all",
                component=component,
            ) or 0) * 100
            for component in components
        ]
        starts = []
        cumulative = 0.0
        for value in values:
            starts.append(cumulative)
            cumulative += value
        for index, (value, start) in enumerate(zip(values, starts)):
            axis.bar(
                index,
                abs(value),
                bottom=min(start, start + value),
                color="#109618" if value >= 0 else "#dc3912",
                alpha=0.85,
            )
            if index < len(values) - 1:
                axis.plot(
                    [index + 0.4, index + 0.6],
                    [start + value, start + value],
                    color="#777777",
                    linewidth=0.8,
                )
        axis.bar(
            len(values),
            abs(cumulative),
            bottom=min(0, cumulative),
            color="#666666",
            alpha=0.8,
        )
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.set_xticks(
            range(len(values) + 1), (*labels, "Settlement\nP&L")
        )
        axis.set_title("Touch" if model == "touch" else "1¢ penetration")
        axis.grid(axis="y", alpha=0.25)
        for index, (value, start) in enumerate(zip(values, starts)):
            endpoint = start + value
            axis.text(
                index,
                endpoint,
                "{:+.2f}¢".format(value),
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
        axis.text(
            len(values),
            cumulative,
            "{:+.2f}¢".format(cumulative),
            ha="center",
            va="bottom" if cumulative >= 0 else "top",
            fontsize=8,
        )
    axes[0].set_ylabel("Average contribution (¢/contract)")
    figure.suptitle("Held-out settlement P&L waterfall")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown_report(report: Mapping[str, Any]) -> str:
    action_rows = report["action_smile"]
    pair_rows = report["paired_leg_economics"]
    concentration_rows = report["concentration"]
    exit_rows = report["exit_frontier"]
    coverage = report["data_coverage"]
    counts = report["counts"]
    lines = [
        "# BBO conditional-fill toxicity diagnostics",
        "",
        "Classification: **Kalshi-only exploratory diagnostic; not a fair-value backtest.**",
        "",
        "## Local input coverage",
        "",
        "- Product label: `{}`".format(coverage["product"]),
        "- Markets / binary settlements: {} / {}".format(
            coverage["markets"], coverage["binary_settlements"]
        ),
        "- Market opens: {} through {}".format(
            coverage["market_open_first"], coverage["market_open_last"]
        ),
        "- One-minute candles: {} across {} markets, ending {} through {}".format(
            coverage["one_minute_candles"],
            coverage["candle_markets"],
            coverage["candle_end_first"],
            coverage["candle_end_last"],
        ),
        "- Public trades: {} across {} markets, {} through {}".format(
            coverage["trades"],
            coverage["trade_markets"],
            coverage["trade_first"],
            coverage["trade_last"],
        ),
        "- Completed per-market trade requests / UTC days: {} / {}".format(
            coverage["completed_trade_request_markets"],
            coverage["completed_trade_request_utc_days"],
        ),
        "- Analytical markets / two-sided quote decisions: {} / {}".format(
            counts["analytical_markets"],
            counts["two_sided_quote_decisions"],
        ),
        "",
        "## Held-out action-side settlement results",
        "",
        "| Fill model | Yes-side action | Fills | Mean | Median | 1% trimmed mean | 90% day-block CI |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in ("touch", "penetration_1c"):
        for category in ACTION_ORDER:
            item = next(
                row
                for row in action_rows
                if row["split"] == "test"
                and row["model"] == model
                and row["action_category"] == category
                and row["horizon"] == "15m/end"
            )
            interval = item["day_block_90pct_ci_dollars"]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} to {} |".format(
                    model,
                    category,
                    item["observations"],
                    item["raw_mean_pnl_dollars"],
                    item["median_pnl_dollars"],
                    item["trimmed_mean_pnl_dollars"],
                    interval[0],
                    interval[1],
                )
            )
    lines.extend(
        [
            "",
            "![Action markout smile](action-markout-smile.png)",
            "",
            "![Price by time settlement heatmaps](price-time-settlement.png)",
            "",
            "![Settlement P&L waterfall](settlement-waterfall.png)",
            "",
            "## Concentration and session persistence",
            "",
            "| Fill model | Raw mean | Both-tail 1% trimmed | Negative UTC days | Excluding best/worst day |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in ("touch", "penetration_1c"):
        item = next(
            row
            for row in concentration_rows
            if row["model"] == model and row["action_category"] == "all"
        )
        lines.append(
            "| {} | {} | {} | {}/{} | {} |".format(
                model,
                item["raw_mean_pnl_dollars"],
                item["mean_without_best_and_worst_1pct_dollars"],
                item["negative_mean_days"],
                item["utc_days"],
                item["pooled_mean_excluding_best_and_worst_day_dollars"],
            )
        )
    lines.extend(
        [
            "",
            "## Exit frontier bounds",
            "",
            "| Fill model | Horizon | Gross last-trade P&L | Indicative taker fee | After fee only |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for model in ("touch", "penetration_1c"):
        for horizon in HORIZON_ORDER:
            item = next(
                row
                for row in exit_rows
                if row["model"] == model
                and row["action_category"] == "all"
                and row["horizon"] == horizon
            )
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    model,
                    horizon,
                    item["gross_markout_before_exit_cost_dollars"],
                    item["indicative_unrounded_taker_exit_fee_dollars"],
                    item["pnl_after_fee_only_dollars"],
                )
            )
    lines.extend(
        [
            "",
            "## Paired-leg economics",
            "",
            "| Fill model | Decisions | Both | Yes only | No only | Neither | Window EV | 90% day-block CI |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for model in ("touch", "penetration_1c"):
        item = next(
            row
            for row in pair_rows
            if row["split"] == "test"
            and row["model"] == model
            and row["time_bucket"] == "all"
        )
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} to {} |".format(
                model,
                item["two_sided_quote_decisions"],
                item["both_legs_fill"],
                item["yes_only"],
                item["no_only"],
                item["neither_fills"],
                item["average_window_net_pnl_dollars"],
                item["window_net_day_block_90pct_ci_dollars"][0],
                item["window_net_day_block_90pct_ci_dollars"][1],
            )
        )
    lines.extend(
        [
            "",
            "## Unsupported cuts",
            "",
            "- Pyth/CME pre-fill momentum: unavailable because synchronized external history is absent.",
            "- Actual queue simulation: unavailable because historical displayed depth, order priority, and cancellation flow are absent.",
            "- Executable exit frontier: incomplete because future opposite BBO and maker-exit fill probability are absent; the CSV reports last-trade gross P&L and an indicative taker fee only.",
            "- Macro-event separation: unavailable because no frozen event calendar or event labels are present.",
            "- 30–60 second and under-30-second time buckets: unavailable from one-minute BBO candles.",
            "",
        ]
    )
    return "\n".join(lines)


def run_toxicity_diagnostics(
    catalog_path: Path, output_root: Path, config: ToxicityConfig
) -> Dict[str, Any]:
    markouts, pairs, counts = load_diagnostics(catalog_path, config)
    coverage = load_catalog_coverage(catalog_path, config.metal)
    action_rows = build_action_smile(markouts, config)
    price_rows = build_price_time_cut(markouts, config)
    exit_rows = build_exit_frontier(markouts, config)
    waterfall_rows = build_waterfall(markouts)
    concentration_rows = build_concentration(markouts, config)
    paired_rows = summarize_pairs(pairs, config)
    report = {
        "study_version": "bbo-toxicity-v1",
        "classification": "kalshi_only_exploratory_not_fair_value_backtest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            **asdict(config),
            "penetration_dollars": str(config.penetration_dollars),
            "maker_fee_per_contract": str(config.maker_fee_per_contract),
            "taker_fee_multiplier": str(config.taker_fee_multiplier),
            "trim_fraction": str(config.trim_fraction),
            "price_buckets": list(PRICE_BUCKET_ORDER),
            "supported_time_buckets": list(TIME_BUCKET_ORDER),
            "horizons": list(HORIZON_ORDER),
        },
        "counts": counts,
        "data_coverage": coverage,
        "action_smile": action_rows,
        "price_time_cut": price_rows,
        "exit_frontier": exit_rows,
        "waterfall": waterfall_rows,
        "concentration": concentration_rows,
        "paired_leg_economics": paired_rows,
        "unavailable": {
            "pyth_cme_prefill_momentum": "no synchronized external history",
            "actual_queue_simulation": "no historical depth/order-priority/cancel flow",
            "executable_exit_frontier": "no future opposite BBO or maker-exit fill probability",
            "macro_event_split": "no frozen macro-event calendar labels",
            "subminute_time_to_expiry_buckets": "one-minute BBO candles cannot observe them",
        },
        "interpretation_limits": [
            "Touch is optimistic and one-cent penetration is a conservative price-path proxy.",
            "Neither fill model reconstructs queue position or guarantees execution.",
            "Last-trade marks are not midpoint, microprice, or fair-value marks.",
            "The earliest valid quote in each market/time bucket is retained to reduce repeated-window dependence.",
            "Selling Yes is payoff-equivalent to buying No but remains separately labeled to preserve quote orientation.",
            "Repeated quote windows are correlated; day-block intervals are primary.",
        ],
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_path = Path(output_root) / "derived" / "toxicity_diagnostics" / "run={}".format(run_id)
    report_path = run_path / "report.json"
    markdown_path = run_path / "REPORT.md"
    markouts_path = run_path / "markouts.csv"
    pairs_path = run_path / "paired_legs.csv"
    action_path = run_path / "action_smile.csv"
    price_path = run_path / "price_time.csv"
    exit_path = run_path / "exit_frontier.csv"
    waterfall_path = run_path / "waterfall.csv"
    concentration_path = run_path / "concentration.csv"
    _atomic_text(report_path, json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    _atomic_text(markdown_path, _markdown_report(report))
    tables = (
        (markouts_path, [_serialized(row) for row in markouts]),
        (pairs_path, [_serialized(row) for row in pairs]),
        (action_path, action_rows),
        (price_path, price_rows),
        (exit_path, exit_rows),
        (waterfall_path, waterfall_rows),
        (concentration_path, concentration_rows),
    )
    for path, rows in tables:
        _atomic_text(
            path,
            _csv_text(rows, list(rows[0])) if rows else "",
        )
    action_plot = run_path / "action-markout-smile.png"
    price_plot = run_path / "price-time-settlement.png"
    waterfall_plot = run_path / "settlement-waterfall.png"
    _plot_action_smile(action_rows, action_plot)
    _plot_price_time(price_rows, price_plot)
    _plot_waterfall(waterfall_rows, waterfall_plot)
    return {
        "run_id": run_id,
        "run_path": str(run_path),
        "report_path": str(report_path),
        "markdown_path": str(markdown_path),
        "markouts_path": str(markouts_path),
        "price_time_path": str(price_path),
        "action_smile_plot": str(action_plot),
        "price_time_plot": str(price_plot),
        "waterfall_plot": str(waterfall_plot),
        **counts,
    }
