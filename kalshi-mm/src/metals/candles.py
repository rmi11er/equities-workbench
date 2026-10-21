"""Resumable Kalshi one-minute candlestick backfill for offline research."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .config import MetalsConfig, resolve_kalshi_credentials
from .kalshi import KalshiReadClient, KalshiReadError, ReceivedResponse
from .storage import PartitionedRawStore, update_run_manifest, write_run_manifest
from .timeutil import iso8601_to_ns


class CandleBackfillError(RuntimeError):
    """Raised after a partial candle run has been durably finalized as failed."""


@dataclass(frozen=True)
class CandleMarket:
    market_ticker: str
    series_ticker: str
    metal: str
    open_ts_ns: int
    close_ts_ns: int
    settlement_ts_ns: Optional[int]

    @property
    def start_seconds(self) -> int:
        return self.open_ts_ns // 1_000_000_000

    @property
    def end_seconds(self) -> int:
        return (self.close_ts_ns + 999_999_999) // 1_000_000_000


def _duckdb():
    try:
        import duckdb
    except ImportError as error:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "Candlestick backfill requires duckdb; install requirements.txt"
        ) from error
    return duckdb


def load_candle_markets(root: Path, metal: str) -> List[CandleMarket]:
    """Load canonical settled market windows from the local DuckDB catalog."""

    catalog = Path(root) / "metals.duckdb"
    if not catalog.is_file():
        raise FileNotFoundError(
            "Build the metals DuckDB catalog before candle backfill: {}".format(
                catalog
            )
        )
    connection = _duckdb().connect(str(catalog), read_only=True)
    try:
        rows = connection.execute(
            "SELECT m.market_ticker, m.series_ticker, m.metal, "
            "m.open_ts_ns, m.close_ts_ns, s.settlement_ts_ns "
            "FROM market_dim m "
            "LEFT JOIN kalshi_settlement s USING (market_ticker) "
            "WHERE m.metal = ? AND m.market_ticker IS NOT NULL "
            "AND m.series_ticker IS NOT NULL AND m.open_ts_ns IS NOT NULL "
            "AND m.close_ts_ns IS NOT NULL "
            "ORDER BY m.open_ts_ns, m.market_ticker",
            [metal],
        ).fetchall()
    finally:
        connection.close()
    return [
        CandleMarket(
            market_ticker=str(row[0]),
            series_ticker=str(row[1]),
            metal=str(row[2]),
            open_ts_ns=int(row[3]),
            close_ts_ns=int(row[4]),
            settlement_ts_ns=int(row[5]) if row[5] is not None else None,
        )
        for row in rows
    ]


def _decimal_field(container: Any, field: str) -> Optional[str]:
    if not isinstance(container, dict):
        return None
    value = container.get("{}_dollars".format(field), container.get(field))
    return None if value is None else str(value)


def normalize_kalshi_candlestick(
    candle: Dict[str, Any],
    market: CandleMarket,
    period_interval_minutes: int,
    response: ReceivedResponse,
    ingestion_sequence: int,
) -> Dict[str, Any]:
    """Normalize one current- or historical-tier Kalshi candlestick."""

    if period_interval_minutes < 1:
        raise ValueError("period_interval_minutes must be positive")
    end_seconds = int(candle["end_period_ts"])
    if end_seconds <= 0:
        raise ValueError("Candlestick end_period_ts must be positive")
    yes_bid = candle.get("yes_bid")
    yes_ask = candle.get("yes_ask")
    price = candle.get("price")
    return {
        "market_ticker": market.market_ticker,
        "series_ticker": market.series_ticker,
        "metal": market.metal,
        "period_interval_minutes": period_interval_minutes,
        "start_period_ts_ns": (
            end_seconds - period_interval_minutes * 60
        )
        * 1_000_000_000,
        "end_period_ts_ns": end_seconds * 1_000_000_000,
        "source_event_ts_ns": end_seconds * 1_000_000_000,
        "source_publish_ts_ns": None,
        "receive_ts_ns": response.receive_ts_ns,
        "local_monotonic_ts_ns": response.local_monotonic_ts_ns,
        "ingestion_sequence": ingestion_sequence,
        "yes_bid_open_dollars": _decimal_field(yes_bid, "open"),
        "yes_bid_low_dollars": _decimal_field(yes_bid, "low"),
        "yes_bid_high_dollars": _decimal_field(yes_bid, "high"),
        "yes_bid_close_dollars": _decimal_field(yes_bid, "close"),
        "yes_ask_open_dollars": _decimal_field(yes_ask, "open"),
        "yes_ask_low_dollars": _decimal_field(yes_ask, "low"),
        "yes_ask_high_dollars": _decimal_field(yes_ask, "high"),
        "yes_ask_close_dollars": _decimal_field(yes_ask, "close"),
        "trade_open_dollars": _decimal_field(price, "open"),
        "trade_low_dollars": _decimal_field(price, "low"),
        "trade_high_dollars": _decimal_field(price, "high"),
        "trade_close_dollars": _decimal_field(price, "close"),
        "trade_mean_dollars": _decimal_field(price, "mean"),
        "trade_previous_dollars": _decimal_field(price, "previous"),
        "trade_min_dollars": _decimal_field(price, "min"),
        "trade_max_dollars": _decimal_field(price, "max"),
        "volume_fp": str(candle.get("volume_fp", candle.get("volume", "0"))),
        "open_interest_fp": str(
            candle.get("open_interest_fp", candle.get("open_interest", "0"))
        ),
        "source_path": response.path,
    }


def _completed_requests(root: Path, period_interval_minutes: int) -> Set[Tuple]:
    completed: Set[Tuple] = set()
    request_root = Path(root) / "normalized" / "kalshi_candlestick_request"
    for path in sorted(request_root.glob("**/*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") != "completed":
                    continue
                if int(row.get("period_interval_minutes", 0)) != period_interval_minutes:
                    continue
                completed.add(
                    (
                        str(row.get("market_ticker", "")),
                        int(row.get("start_seconds", 0)),
                        int(row.get("end_seconds", 0)),
                        period_interval_minutes,
                    )
                )
    return completed


def _request_key(market: CandleMarket, period_interval_minutes: int) -> Tuple:
    return (
        market.market_ticker,
        market.start_seconds,
        market.end_seconds,
        period_interval_minutes,
    )


def _raw_response_record(
    response: ReceivedResponse, stream: str
) -> Dict[str, Any]:
    return {
        "source": "kalshi",
        "stream": stream,
        "receive_ts_ns": response.receive_ts_ns,
        "local_monotonic_ts_ns": response.local_monotonic_ts_ns,
        "path": response.path,
        "params": response.params,
        "raw_body": response.raw_body,
    }


async def backfill_kalshi_candlesticks(
    config: MetalsConfig,
    metal: str,
    max_markets: Optional[int] = None,
    period_interval_minutes: int = 1,
    resume: bool = True,
    markets: Optional[Sequence[CandleMarket]] = None,
    client: Optional[KalshiReadClient] = None,
    max_errors: int = 20,
) -> Dict[str, int]:
    """Archive one-minute bid/ask candles without any order-entry capability."""

    if period_interval_minutes not in (1, 60, 1440):
        raise ValueError("Kalshi candle interval must be 1, 60, or 1440 minutes")
    if max_markets is not None and max_markets < 1:
        raise ValueError("max_markets must be positive")
    if max_errors < 1:
        raise ValueError("max_errors must be positive")
    market_rows = list(
        markets
        if markets is not None
        else load_candle_markets(Path(config.storage.root), metal)
    )

    completed = (
        _completed_requests(Path(config.storage.root), period_interval_minutes)
        if resume
        else set()
    )
    pending_market_rows = [
        market
        for market in market_rows
        if _request_key(market, period_interval_minutes) not in completed
    ]
    if max_markets is not None:
        pending_market_rows = pending_market_rows[:max_markets]
    raw_store = PartitionedRawStore(
        Path(config.storage.root),
        fsync_every_records=config.storage.fsync_every_records,
        tier="raw",
    )
    normalized_store = PartitionedRawStore(
        Path(config.storage.root),
        run_id=raw_store.run_id,
        fsync_every_records=config.storage.fsync_every_records,
        tier="normalized",
    )
    write_run_manifest(
        Path(config.storage.root), raw_store.run_id, config.public_snapshot()
    )

    if client is None:
        try:
            credentials = resolve_kalshi_credentials(config.kalshi)
        except (ValueError, FileNotFoundError):
            credentials = None
        client = KalshiReadClient(config.kalshi, credentials)

    counts = {
        "markets_considered": len(market_rows),
        "markets_requested": 0,
        "markets_skipped": len(market_rows) - len(
            [
                market
                for market in market_rows
                if _request_key(market, period_interval_minutes) not in completed
            ]
        ),
        "markets_failed": 0,
        "candles": 0,
    }
    errors: List[BaseException] = []
    ingestion_sequence = 0
    failure: Optional[BaseException] = None
    close_failure: Optional[BaseException] = None
    try:
        if not pending_market_rows:
            return counts
        await client.start()
        cutoff_response = await client.get("/historical/cutoff")
        raw_store.write(
            source="kalshi",
            stream="rest_historical_cutoff",
            metal=metal,
            receive_ts_ns=cutoff_response.receive_ts_ns,
            record=_raw_response_record(
                cutoff_response, "rest_historical_cutoff"
            ),
        )
        cutoff_ns = iso8601_to_ns(
            cutoff_response.payload.get("market_settled_ts")
        )
        if cutoff_ns is None:
            raise CandleBackfillError(
                "Kalshi historical cutoff omitted market_settled_ts"
            )

        for market in pending_market_rows:
            is_historical = (
                market.settlement_ts_ns is not None
                and market.settlement_ts_ns < cutoff_ns
            )
            path = (
                "/historical/markets/{}/candlesticks".format(
                    market.market_ticker
                )
                if is_historical
                else "/series/{}/markets/{}/candlesticks".format(
                    market.series_ticker, market.market_ticker
                )
            )
            params = {
                "start_ts": market.start_seconds,
                "end_ts": market.end_seconds,
                "period_interval": period_interval_minutes,
            }
            counts["markets_requested"] += 1
            try:
                response = await client.get(path, params)
                raw_store.write(
                    source="kalshi",
                    stream="rest_candlesticks",
                    metal=metal,
                    receive_ts_ns=response.receive_ts_ns,
                    record=_raw_response_record(
                        response, "rest_candlesticks"
                    ),
                )
                candles = response.payload.get("candlesticks", [])
                if not isinstance(candles, list):
                    raise CandleBackfillError(
                        "Kalshi candlesticks payload is not a list"
                    )
                for candle in candles:
                    ingestion_sequence += 1
                    normalized_store.write(
                        source="kalshi_candlestick",
                        stream="kalshi_candlestick",
                        metal=metal,
                        receive_ts_ns=response.receive_ts_ns,
                        record=normalize_kalshi_candlestick(
                            candle,
                            market,
                            period_interval_minutes,
                            response,
                            ingestion_sequence,
                        ),
                    )
                counts["candles"] += len(candles)
                normalized_store.write(
                    source="kalshi_candlestick_request",
                    stream="kalshi_candlestick_request",
                    metal=metal,
                    receive_ts_ns=response.receive_ts_ns,
                    record={
                        "market_ticker": market.market_ticker,
                        "series_ticker": market.series_ticker,
                        "start_seconds": market.start_seconds,
                        "end_seconds": market.end_seconds,
                        "period_interval_minutes": period_interval_minutes,
                        "endpoint_tier": (
                            "historical" if is_historical else "current"
                        ),
                        "candle_count": len(candles),
                        "receive_ts_ns": response.receive_ts_ns,
                        "status": "completed",
                    },
                )
            except Exception as error:
                errors.append(error)
                counts["markets_failed"] += 1
                normalized_store.write(
                    source="kalshi_candlestick_request",
                    stream="kalshi_candlestick_request",
                    metal=metal,
                    receive_ts_ns=cutoff_response.receive_ts_ns,
                    record={
                        "market_ticker": market.market_ticker,
                        "series_ticker": market.series_ticker,
                        "start_seconds": market.start_seconds,
                        "end_seconds": market.end_seconds,
                        "period_interval_minutes": period_interval_minutes,
                        "endpoint_tier": (
                            "historical" if is_historical else "current"
                        ),
                        "candle_count": 0,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "http_status": (
                            error.status
                            if isinstance(error, KalshiReadError)
                            else None
                        ),
                    },
                )
                if (
                    isinstance(error, KalshiReadError)
                    and error.status in (401, 403)
                ) or len(errors) >= max_errors:
                    break
        if errors:
            raise CandleBackfillError(
                "{} Kalshi candlestick request(s) failed; rerun resumes completed "
                "markets".format(len(errors))
            ) from errors[0]
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            await client.close()
        except BaseException as error:
            close_failure = error
        for store in (raw_store, normalized_store):
            try:
                store.close()
            except BaseException as error:
                if close_failure is None:
                    close_failure = error
        terminal_error = failure or close_failure
        update_run_manifest(
            Path(config.storage.root),
            raw_store.run_id,
            status="failed" if terminal_error is not None else "completed",
            counts=counts,
            output_paths=raw_store.paths + normalized_store.paths,
            errors=[*errors, *([terminal_error] if terminal_error else [])],
        )
        if failure is None and close_failure is not None:
            raise close_failure
    return counts
