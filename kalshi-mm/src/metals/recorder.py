"""Silver-first Kalshi recorder and historical backfill orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .book import FixedPointOrderBook, decimal_string
from .config import MetalsConfig, resolve_kalshi_credentials
from .discovery import DiscoveredMarket, discover_open_markets, market_matches_product
from .kalshi import (
    KalshiReadClient,
    KalshiReadError,
    KalshiWebSocketReader,
    ReceivedResponse,
)
from .metadata import MetadataRegistry
from .quality import QualityTracker
from .storage import PartitionedRawStore, update_run_manifest, write_run_manifest
from .timeutil import iso8601_to_ns
from .types import SequenceStatus, SourceEnvelope


logger = logging.getLogger(__name__)


def _message_ticker(message: Dict[str, Any]) -> str:
    return str(message.get("market_ticker", message.get("ticker", "")))


def _message_type(envelope: SourceEnvelope) -> str:
    return str(envelope.raw.get("type", envelope.stream))


def _extract_series_source(series: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    sources = series.get("settlement_sources", series.get("settlement_source"))
    if isinstance(sources, dict):
        sources = [sources]
    if isinstance(sources, list) and sources:
        source = sources[0]
        if isinstance(source, dict):
            return (
                str(source.get("name", source.get("source_name", ""))) or None,
                str(source.get("url", source.get("source_url", ""))) or None,
            )
        return str(source), None
    return None, None


class MetalsRecorder:
    """Record target-market data without exposing any order-entry operation."""

    def __init__(self, config: MetalsConfig, require_credentials: bool = True) -> None:
        self.config = config
        try:
            self.credentials = resolve_kalshi_credentials(config.kalshi)
        except (ValueError, FileNotFoundError):
            if require_credentials:
                raise
            self.credentials = None
        self.raw_store = PartitionedRawStore(
            Path(config.storage.root),
            fsync_every_records=config.storage.fsync_every_records,
            tier="raw",
        )
        self.normalized_store = PartitionedRawStore(
            Path(config.storage.root),
            run_id=self.raw_store.run_id,
            fsync_every_records=config.storage.fsync_every_records,
            tier="normalized",
        )
        self.run_id = self.raw_store.run_id
        self.quality = QualityTracker()
        self.metadata = MetadataRegistry(
            fractional_capability=config.kalshi.fractional_contracts_enabled
        )
        self.books: Dict[str, FixedPointOrderBook] = {}
        self._market_metal: Dict[str, str] = {}
        self._market_raw: Dict[str, Dict[str, Any]] = {}
        self._desired_tickers: Set[str] = set()
        self._ticker_change = asyncio.Event()
        self._stop = asyncio.Event()
        self._ingestion_sequence = 0
        self._series_cache: Dict[str, Dict[str, Any]] = {}
        self._event_metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._finalized = False
        self._client = KalshiReadClient(config.kalshi, self.credentials)
        self._ws = (
            KalshiWebSocketReader(config.kalshi, self.credentials)
            if self.credentials is not None
            else None
        )
        write_run_manifest(
            Path(config.storage.root), self.run_id, config.public_snapshot()
        )

    def _next_ingestion_sequence(self) -> int:
        self._ingestion_sequence += 1
        return self._ingestion_sequence

    def _write_raw_envelope(self, envelope: SourceEnvelope, metal: str) -> None:
        envelope.ingestion_sequence = self._next_ingestion_sequence()
        self.raw_store.write(
            source=envelope.source,
            stream=envelope.stream,
            metal=metal,
            receive_ts_ns=envelope.receive_ts_ns,
            record=envelope.to_dict(),
        )
        self.quality.record("raw_records", _message_ticker(envelope.raw.get("msg", {})))

    def _write_normalized(
        self,
        stream: str,
        metal: str,
        receive_ts_ns: int,
        record: Dict[str, Any],
    ) -> None:
        self.normalized_store.write(
            source=stream,
            stream=stream,
            metal=metal,
            receive_ts_ns=receive_ts_ns,
            record=record,
        )

    def _metal_for_message(self, message: Dict[str, Any]) -> Optional[str]:
        ticker = _message_ticker(message)
        if ticker in self._market_metal:
            return self._market_metal[ticker]
        series = str(message.get("series_ticker", ""))
        candidate = dict(message)
        if series:
            candidate["series_ticker"] = series
        for metal, product in self.config.enabled_products.items():
            if market_matches_product(candidate, product):
                return metal
        underlying = str(message.get("underlying_ticker", ""))
        for metal, product in self.config.enabled_products.items():
            if product.pyth_symbol and underlying == product.pyth_symbol:
                return metal
        return None

    def _rest_envelope(
        self, response: ReceivedResponse, stream: str
    ) -> SourceEnvelope:
        return SourceEnvelope(
            source="kalshi",
            stream=stream,
            ingestion_sequence=0,
            local_monotonic_ts_ns=response.local_monotonic_ts_ns,
            receive_ts_ns=response.receive_ts_ns,
            raw={
                "path": response.path,
                "params": response.params,
            },
            raw_text=response.raw_body,
        )

    async def _get_series(self, series_ticker: str, metal: str) -> Dict[str, Any]:
        cached = self._series_cache.get(series_ticker)
        if cached is not None:
            return cached
        response = await self._client.get("/series/{}".format(series_ticker))
        self._write_raw_envelope(self._rest_envelope(response, "rest_series"), metal)
        series = response.payload.get("series", response.payload)
        self._series_cache[series_ticker] = series
        return series

    async def _get_event_metadata(
        self, event_ticker: str, metal: str
    ) -> Dict[str, Any]:
        cached = self._event_metadata_cache.get(event_ticker)
        if cached is not None:
            return cached
        response = await self._client.get(
            "/events/{}/metadata".format(event_ticker)
        )
        self._write_raw_envelope(
            self._rest_envelope(response, "rest_event_metadata"), metal
        )
        metadata = response.payload.get("event_metadata", response.payload)
        self._event_metadata_cache[event_ticker] = metadata
        return metadata

    async def _hydrate_market(self, discovered: DiscoveredMarket) -> Dict[str, Any]:
        ticker = discovered.ticker
        response = await self._client.get("/markets/{}".format(ticker))
        self._write_raw_envelope(
            self._rest_envelope(response, "rest_market"), discovered.metal
        )
        market = dict(response.payload.get("market", discovered.raw))
        if discovered.series_ticker:
            market.setdefault("series_ticker", discovered.series_ticker)
        series_ticker = str(market.get("series_ticker", ""))
        event_ticker = str(market.get("event_ticker", ""))
        if series_ticker:
            series = await self._get_series(series_ticker, discovered.metal)
            source_name, source_url = _extract_series_source(series)
            if source_name:
                market["settlement_source_name"] = source_name
            if source_url:
                market["settlement_source_url"] = source_url
            product = self.config.products.get(discovered.metal)
            decoded_source_url = urllib.parse.unquote(source_url or "")
            if product and product.pyth_symbol in decoded_source_url:
                market["pyth_symbol"] = product.pyth_symbol
        if event_ticker:
            try:
                await self._get_event_metadata(event_ticker, discovered.metal)
            except Exception as error:
                logger.warning("Event metadata unavailable for %s: %s", event_ticker, error)
        return market

    async def discover_once(self) -> List[Dict[str, Any]]:
        discovered, responses = await discover_open_markets(self._client, self.config)
        for response in responses:
            series_ticker = str(response.params.get("series_ticker", ""))
            metal = next(
                (
                    name
                    for name, product in self.config.enabled_products.items()
                    if series_ticker in product.series_tickers
                ),
                "unclassified",
            )
            self._write_raw_envelope(
                self._rest_envelope(response, "rest_market_discovery"), metal
            )

        # Keep every discovery response raw, but only hydrate active markets and
        # the nearest configured upcoming windows. This avoids unnecessary API
        # load from distant weekend schedules.
        selected: List[DiscoveredMarket] = []
        for metal in self.config.enabled_products:
            product_markets = [item for item in discovered if item.metal == metal]
            active = [
                item
                for item in product_markets
                if str(item.raw.get("status", "")).lower() in ("active", "open")
            ]
            upcoming = [
                item
                for item in product_markets
                if str(item.raw.get("status", "")).lower()
                not in ("active", "open")
            ]
            upcoming.sort(key=lambda item: str(item.raw.get("open_time", "")))
            selected.extend(active)
            selected.extend(
                upcoming[: self.config.kalshi.upcoming_markets_per_product]
            )

        markets: List[Dict[str, Any]] = []
        active_tickers: Set[str] = set()
        for item in selected:
            try:
                market = await self._hydrate_market(item)
            except Exception as error:
                logger.warning("Failed to hydrate %s: %s", item.ticker, error)
                market = dict(item.raw)
            markets.append(market)
            ticker = item.ticker
            self._market_metal[ticker] = item.metal
            self._market_raw[ticker] = market
            if str(market.get("status", "")).lower() in ("active", "open"):
                active_tickers.add(ticker)
            version, changed = self.metadata.observe(
                market, metal=item.metal, receive_ts_ns=time.time_ns()
            )
            if changed:
                self._write_normalized(
                    "market_dim",
                    item.metal,
                    version.metadata_receive_ts_ns,
                    version.to_dict(),
                )
                self.quality.record("metadata_versions", ticker)
                if version.metadata_version > 1:
                    self.quality.metadata_changes.append(
                        {
                            "ticker": ticker,
                            "version": version.metadata_version,
                            "hash": version.metadata_hash,
                        }
                    )

        if active_tickers != self._desired_tickers:
            self._desired_tickers = active_tickers
            self._ticker_change.set()
        return markets

    async def _discovery_loop(self) -> None:
        while not self._stop.is_set():
            try:
                markets = await self.discover_once()
                logger.info(
                    "Discovered %d target markets (%d active)",
                    len(markets),
                    len(self._desired_tickers),
                )
                if not markets:
                    self.quality.record("missing_markets")
            except Exception as error:
                logger.exception("Market discovery failed: %s", error)
                self.quality.record("discovery_errors")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self.config.kalshi.discovery_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass

    def _record_book_snapshot(
        self, envelope: SourceEnvelope, metal: str, message: Dict[str, Any]
    ) -> None:
        ticker = _message_ticker(message)
        book = self.books.setdefault(ticker, FixedPointOrderBook(ticker))
        snapshot_id = "{}:{}:{}".format(
            envelope.connection_id,
            envelope.subscription_id,
            envelope.source_sequence,
        )
        book.apply_snapshot(message, snapshot_id=snapshot_id)
        common = self._normalized_common(envelope, ticker, "snapshot")
        for side, field_name in (
            ("yes", "yes_dollars_fp"),
            ("no", "no_dollars_fp"),
        ):
            levels = message.get(field_name, message.get(side, []))
            for level in levels:
                price, quantity = level[0], level[1]
                self._write_normalized(
                    "kalshi_book_event",
                    metal,
                    envelope.receive_ts_ns,
                    {
                        **common,
                        "outcome_side": side,
                        "price_dollars": decimal_string(price),
                        "quantity_fp": decimal_string(quantity),
                        "recovery_snapshot_id": snapshot_id,
                        "book_valid_after": book.valid,
                        "book_state_hash": book.state_hash(),
                    },
                )
        self.quality.record("book_snapshots", ticker)

    def _record_book_delta(
        self, envelope: SourceEnvelope, metal: str, message: Dict[str, Any]
    ) -> None:
        ticker = _message_ticker(message)
        book = self.books.setdefault(ticker, FixedPointOrderBook(ticker))
        book.apply_delta(message)
        record = {
            **self._normalized_common(envelope, ticker, "delta"),
            "outcome_side": message.get("side"),
            "price_dollars": decimal_string(
                message.get("price_dollars", message.get("price"))
            ),
            "quantity_delta_fp": decimal_string(
                message.get("delta_fp", message.get("delta"))
            ),
            "recovery_snapshot_id": book.recovery_snapshot_id,
            "book_valid_after": book.valid,
            "book_state_hash": book.state_hash(),
        }
        self._write_normalized(
            "kalshi_book_event", metal, envelope.receive_ts_ns, record
        )
        self.quality.record("book_deltas", ticker)

    def _normalized_common(
        self, envelope: SourceEnvelope, ticker: str, message_type: str
    ) -> Dict[str, Any]:
        return {
            "market_ticker": ticker,
            "connection_id": envelope.connection_id,
            "subscription_id": envelope.subscription_id,
            "source_sequence": envelope.source_sequence,
            "message_type": message_type,
            "source_event_ts_ns": envelope.source_event_ts_ns,
            "source_publish_ts_ns": envelope.source_publish_ts_ns,
            "local_monotonic_ts_ns": envelope.local_monotonic_ts_ns,
            "receive_ts_ns": envelope.receive_ts_ns,
            "ingestion_sequence": envelope.ingestion_sequence,
            "gap_flag": envelope.gap_flag,
        }

    def _record_trade(
        self, envelope: SourceEnvelope, metal: str, message: Dict[str, Any]
    ) -> None:
        ticker = _message_ticker(message)
        record = {
            "trade_id": message.get("trade_id"),
            "market_ticker": ticker,
            "yes_price_dollars": message.get(
                "yes_price_dollars", message.get("yes_price")
            ),
            "no_price_dollars": message.get(
                "no_price_dollars", message.get("no_price")
            ),
            "quantity_fp": message.get("count_fp", message.get("count")),
            "taker_outcome_side": message.get("taker_outcome_side"),
            "taker_book_side": message.get("taker_book_side"),
            "is_block_trade": bool(message.get("is_block_trade", False)),
            "source_event_ts_ns": envelope.source_event_ts_ns,
            "receive_ts_ns": envelope.receive_ts_ns,
            "local_monotonic_ts_ns": envelope.local_monotonic_ts_ns,
            "ingestion_sequence": envelope.ingestion_sequence,
        }
        self._write_normalized("kalshi_trade", metal, envelope.receive_ts_ns, record)
        self.quality.record("trades", ticker)

    def _record_kalshi_pyth(
        self, envelope: SourceEnvelope, metal: str, message: Dict[str, Any]
    ) -> None:
        self._write_normalized(
            "kalshi_pyth_value",
            metal,
            envelope.receive_ts_ns,
            {
                "underlying_ticker": message.get("underlying_ticker"),
                "value_usd": message.get("value_usd"),
                "source_ts_ns": iso8601_to_ns(message.get("source_ts_ms")),
                "kalshi_received_ts_ns": iso8601_to_ns(message.get("received_at")),
                "receive_ts_ns": envelope.receive_ts_ns,
                "local_monotonic_ts_ns": envelope.local_monotonic_ts_ns,
                "source_sequence": envelope.source_sequence,
                "ingestion_sequence": envelope.ingestion_sequence,
            },
        )
        self.quality.record("kalshi_pyth_values")

    async def _process_ws(self, envelope: SourceEnvelope) -> None:
        message = envelope.raw.get("msg", {})
        if not isinstance(message, dict):
            message = {"value": message}
        message_type = _message_type(envelope)
        metal = self._metal_for_message(message)

        # Lifecycle is global; retain only target-product events plus protocol
        # acknowledgements/errors. Exact Pyth subscriptions are target-filtered.
        if metal is None and message_type not in ("subscribed", "error"):
            return
        storage_metal = metal or "unclassified"
        self._write_raw_envelope(envelope, storage_metal)

        ticker = _message_ticker(message)
        if envelope.sequence_status == SequenceStatus.DUPLICATE.value:
            self.quality.record("duplicate_messages", ticker)
            return
        if envelope.gap_flag:
            self.quality.record("kalshi_sequence_gaps", ticker)
            if ticker and ticker in self.books:
                self.books[ticker].invalidate()
            return

        if message_type == "orderbook_snapshot":
            self._record_book_snapshot(envelope, storage_metal, message)
        elif message_type == "orderbook_delta":
            self._record_book_delta(envelope, storage_metal, message)
        elif message_type == "trade":
            self._record_trade(envelope, storage_metal, message)
        elif message_type == "pyth_value":
            self._record_kalshi_pyth(envelope, storage_metal, message)
        elif message_type == "market_lifecycle_v2":
            self.quality.record("lifecycle_events", ticker)
            # Force the next discovery pass to re-read target/rules/price ranges.
            self._event_metadata_cache.pop(str(message.get("event_ticker", "")), None)

    async def _consume_ws(self, tickers: List[str], connection_stop: asyncio.Event) -> None:
        if self._ws is None:
            raise RuntimeError("Live collection requires Kalshi WebSocket credentials")
        underlyings = sorted(
            {
                product.pyth_symbol
                for product in self.config.enabled_products.values()
                if product.pyth_symbol
            }
        )
        async for envelope in self._ws.events(
            tickers,
            pyth_underlyings=underlyings,
            stop_event=connection_stop,
        ):
            await self._process_ws(envelope)

    async def _websocket_supervisor(self) -> None:
        reconnect_backoff = 1.0
        while not self._stop.is_set():
            if not self._desired_tickers:
                self._ticker_change.clear()
                try:
                    await asyncio.wait_for(self._ticker_change.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                continue

            tickers = sorted(self._desired_tickers)
            self._ticker_change.clear()
            connection_stop = asyncio.Event()
            consume = asyncio.create_task(self._consume_ws(tickers, connection_stop))
            changed = asyncio.create_task(self._ticker_change.wait())
            stopped = asyncio.create_task(self._stop.wait())
            done, pending = await asyncio.wait(
                [consume, changed, stopped], return_when=asyncio.FIRST_COMPLETED
            )
            connection_stop.set()
            for task in pending:
                task.cancel()
            if consume in done:
                error = consume.exception()
                if error is not None and not self._stop.is_set():
                    logger.warning("Kalshi WebSocket disconnected: %s", error)
                    self.quality.record("websocket_disconnects")
                    await asyncio.sleep(reconnect_backoff)
                    reconnect_backoff = min(reconnect_backoff * 2, 30.0)
                    continue
            reconnect_backoff = 1.0

    async def run(self, duration_seconds: Optional[float] = None) -> None:
        timer = None
        discovery = None
        websocket = None
        failure: Optional[BaseException] = None
        close_failure: Optional[BaseException] = None
        try:
            await self._client.start()
            if self._ws is None:
                raise RuntimeError("Live collection requires a valid Kalshi API key")
            await self._client.get("/account/limits")
            if duration_seconds is not None and duration_seconds > 0:
                async def stop_later() -> None:
                    await asyncio.sleep(duration_seconds)
                    self._stop.set()
                timer = asyncio.create_task(stop_later())

            discovery = asyncio.create_task(self._discovery_loop())
            websocket = asyncio.create_task(self._websocket_supervisor())
            await self._stop.wait()
        except KalshiReadError as error:
            failure = error
            if error.status == 401:
                raise RuntimeError(
                    "Kalshi rejected the configured API key; create a fresh key "
                    "before live WebSocket collection"
                ) from error
            raise
        except BaseException as error:
            failure = error
            raise
        finally:
            for task in (discovery, websocket, timer):
                if task is not None:
                    task.cancel()
            tasks = [
                task for task in (discovery, websocket, timer) if task is not None
            ]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await self._client.close()
            except BaseException as close_error:
                if failure is None:
                    failure = close_error
                    close_failure = close_error
            status = "completed"
            if isinstance(failure, asyncio.CancelledError):
                status = "cancelled"
            elif failure is not None:
                status = "failed"
            self._finalize(status=status, error=failure)
            if close_failure is not None:
                raise close_failure

    def stop(self) -> None:
        self._stop.set()

    def _write_quality_report(self) -> Path:
        now = datetime.now(timezone.utc)
        path = (
            Path(self.config.storage.root)
            / "derived"
            / "daily_quality"
            / "date={}".format(now.date().isoformat())
            / "run={}".format(self.run_id)
            / "report.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(
                self.quality.report(now.date().isoformat(), self.run_id),
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        return path

    def _finalize(
        self,
        status: str,
        counts: Optional[Dict[str, int]] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        if self._finalized:
            return
        errors: List[BaseException] = []
        finalize_failure: Optional[BaseException] = None
        if error is not None:
            errors.append(error)
        quality_path: Optional[Path] = None
        try:
            candidate_path = self._quality_report_path_for_retry()
            quality_path = (
                candidate_path
                if candidate_path.exists()
                else self._write_quality_report()
            )
        except BaseException as report_error:
            errors.append(report_error)
            finalize_failure = report_error
            status = "failed"
        for store in (self.raw_store, self.normalized_store):
            try:
                store.close()
            except BaseException as close_error:
                errors.append(close_error)
                if finalize_failure is None:
                    finalize_failure = close_error
                status = "failed"

        output_paths = self.raw_store.paths + self.normalized_store.paths
        if quality_path is not None:
            output_paths.append(quality_path)
        update_run_manifest(
            Path(self.config.storage.root),
            self.run_id,
            status=status,
            counts={
                **dict(self.quality.counters),
                **dict(counts or {}),
            },
            output_paths=output_paths,
            errors=errors,
        )
        self._finalized = True
        if error is None and finalize_failure is not None:
            raise finalize_failure

    def _quality_report_path_for_retry(self) -> Path:
        now = datetime.now(timezone.utc)
        return (
            Path(self.config.storage.root)
            / "derived"
            / "daily_quality"
            / "date={}".format(now.date().isoformat())
            / "run={}".format(self.run_id)
            / "report.json"
        )


async def backfill_settled_markets(
    config: MetalsConfig,
    metal: str,
    max_markets: Optional[int] = None,
    include_trades: bool = True,
) -> Dict[str, int]:
    """Archive settled market metadata/results and optionally public trades."""

    product = config.products.get(metal)
    if product is None or not product.series_tickers:
        raise ValueError("No configured series ticker for {}".format(metal))

    recorder = MetalsRecorder(config, require_credentials=False)
    counts = {"markets": 0, "trades": 0, "pages": 0}
    failure: Optional[BaseException] = None
    close_failure: Optional[BaseException] = None
    try:
        await recorder._client.start()
        for series_ticker in product.series_tickers:
            series_metadata = await recorder._get_series(series_ticker, metal)
            source_name, source_url = _extract_series_source(series_metadata)
            pages = await recorder._client.get_all_pages(
                "/markets",
                "markets",
                {"series_ticker": series_ticker, "status": "settled"},
            )
            for response in pages:
                counts["pages"] += 1
                recorder._write_raw_envelope(
                    recorder._rest_envelope(response, "rest_settled_markets"), metal
                )
                for market in response.payload.get("markets", []):
                    if max_markets is not None and counts["markets"] >= max_markets:
                        break
                    market = dict(market)
                    market.setdefault("series_ticker", series_ticker)
                    if source_name:
                        market["settlement_source_name"] = source_name
                    if source_url:
                        market["settlement_source_url"] = source_url
                    decoded_source_url = urllib.parse.unquote(source_url or "")
                    if product.pyth_symbol in decoded_source_url:
                        market["pyth_symbol"] = product.pyth_symbol
                    ticker = str(market.get("ticker", ""))
                    recorder._market_metal[ticker] = metal
                    version, changed = recorder.metadata.observe(
                        market, metal=metal, receive_ts_ns=response.receive_ts_ns
                    )
                    if changed:
                        recorder._write_normalized(
                            "market_dim", metal, response.receive_ts_ns, version.to_dict()
                        )
                        recorder.quality.record("metadata_versions", ticker)
                    recorder._write_normalized(
                        "kalshi_settlement",
                        metal,
                        response.receive_ts_ns,
                        {
                            "market_ticker": ticker,
                            "event_ticker": market.get("event_ticker"),
                            "series_ticker": market.get("series_ticker"),
                            "status": market.get("status"),
                            "result": market.get("result"),
                            "expiration_value": market.get("expiration_value"),
                            "settlement_value_dollars": market.get(
                                "settlement_value_dollars"
                            ),
                            "settlement_ts_ns": iso8601_to_ns(
                                market.get("settlement_ts")
                            ),
                            "raw_metadata_hash": version.metadata_hash,
                            "receive_ts_ns": response.receive_ts_ns,
                        },
                    )
                    counts["markets"] += 1
                    recorder.quality.record("observed_windows", ticker)

                    if include_trades and ticker:
                        trade_pages = await recorder._client.get_all_pages(
                            "/markets/trades", "trades", {"ticker": ticker}
                        )
                        for trade_response in trade_pages:
                            recorder._write_raw_envelope(
                                recorder._rest_envelope(
                                    trade_response, "rest_historical_trades"
                                ),
                                metal,
                            )
                            for trade in trade_response.payload.get("trades", []):
                                envelope = SourceEnvelope(
                                    source="kalshi",
                                    stream="rest_trade",
                                    ingestion_sequence=recorder._next_ingestion_sequence(),
                                    local_monotonic_ts_ns=trade_response.local_monotonic_ts_ns,
                                    receive_ts_ns=trade_response.receive_ts_ns,
                                    raw={"type": "trade", "msg": trade},
                                    source_event_ts_ns=iso8601_to_ns(
                                        trade.get("ts_ms", trade.get("created_time"))
                                    ),
                                )
                                recorder._record_trade(envelope, metal, trade)
                                counts["trades"] += 1
                if max_markets is not None and counts["markets"] >= max_markets:
                    break
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            await recorder._client.close()
        except BaseException as close_error:
            if failure is None:
                failure = close_error
                close_failure = close_error
        recorder._finalize(
            status="failed" if failure is not None else "completed",
            counts=counts,
            error=failure,
        )
        if close_failure is not None:
            raise close_failure
    return counts
