"""Databento-backed CME recorder orchestration for configured metals products."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional

from .config import MetalsConfig
from .databento import (
    DatabentoLiveCollector,
    DeferredDatabentoRecord,
    glbx_metals_subscriptions,
)
from .quality import QualityTracker
from .storage import PartitionedRawStore, update_run_manifest, write_run_manifest


_NORMALIZED_STREAMS = {
    "mbo": "cme_mbo_event",
    "trades": "cme_trade",
    "definition": "cme_instrument_definition",
    "status": "cme_status",
    "statistics": "cme_statistics",
    "symbol_mapping": "cme_symbol_mapping",
    "system": "cme_feed_event",
    "error": "cme_feed_event",
    "reconnect": "cme_feed_event",
}


class DatabentoAccessError(RuntimeError):
    pass


class DatabentoMetalsRecorder:
    """Persist native field-lossless DBN and normalized CME tables.

    This collector has no order-entry dependency. The requested metal is known
    from the subscription configuration, so records can be partitioned before
    instrument-definition messages have established an instrument-ID mapping.
    """

    def __init__(
        self,
        config: MetalsConfig,
        metal: str,
        collector: Optional[DatabentoLiveCollector] = None,
        max_records: Optional[int] = None,
    ) -> None:
        if not config.cme.enabled:
            raise DatabentoAccessError(
                "CME collection is disabled; enable [cme] only after access approval"
            )
        if config.cme.provider.lower() != "databento":
            raise DatabentoAccessError(
                "Unsupported CME provider: {}".format(config.cme.provider)
            )
        product = config.products.get(metal)
        if product is None or not product.enabled or not product.cme_symbols:
            raise DatabentoAccessError(
                "No enabled CME symbol mapping configured for {}".format(metal)
            )
        if max_records is not None and max_records < 1:
            raise ValueError("max_records must be at least one")

        self.config = config
        self.metal = metal
        self.max_records = max_records
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
        self.counts: Dict[str, int] = {"records": 0}
        self._ingestion_sequence = 0
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._errors: List[BaseException] = []
        self._accepted_records = 0
        self._record_queue: queue.Queue = queue.Queue(maxsize=100_000)
        self._writer_thread: Optional[threading.Thread] = None
        self._native_stream_handle: Optional[BinaryIO] = None
        self._native_stream_path: Optional[Path] = None
        self._finalized = False

        if collector is None:
            api_key = os.environ.get(config.cme.api_key_env, "").strip()
            if not api_key:
                raise DatabentoAccessError(
                    "Set {} to a licensed Databento key".format(
                        config.cme.api_key_env
                    )
                )
            subscriptions = glbx_metals_subscriptions(
                product.cme_symbols,
                stype_in=config.cme.stype_in,
                schemas=config.cme.schemas,
                mbo_snapshot=config.cme.mbo_snapshot,
            )
            collector = DatabentoLiveCollector(
                api_key,
                subscriptions,
                reconnect_policy=config.cme.reconnect_policy,
                include_ts_out=config.cme.include_ts_out,
            )
        self.collector = collector
        write_run_manifest(
            Path(config.storage.root), self.run_id, config.public_snapshot()
        )

    def _next_ingestion_sequence(self) -> int:
        self._ingestion_sequence += 1
        return self._ingestion_sequence

    def _enqueue_record(self, record: Any) -> None:
        """Accept a record without blocking Databento's networking loop."""

        with self._lock:
            if (
                self.max_records is not None
                and self._accepted_records >= self.max_records
            ):
                return
            try:
                self._record_queue.put_nowait(record)
            except queue.Full:
                self._errors.append(
                    DatabentoAccessError(
                        "Databento normalization queue overflow; native DBN is retained"
                    )
                )
                self.quality.record("cme_collection_errors")
                self.quality.record("cme_normalization_queue_overflows")
                self._done.set()
                return
            self._accepted_records += 1
            if (
                self.max_records is not None
                and self._accepted_records >= self.max_records
            ):
                self._done.set()

    def _writer_loop(self) -> None:
        while True:
            record = self._record_queue.get()
            try:
                if record is None:
                    return
                self._persist_record(record)
            except Exception as error:
                self._record_error(error)
            finally:
                self._record_queue.task_done()

    def _persist_record(self, record: Any) -> None:
        """Write extracted JSON and normalized rows on the writer thread."""

        if isinstance(record, DeferredDatabentoRecord):
            record = self.collector.normalize_record(
                record.record,
                local_receive_ts_ns=record.local_receive_ts_ns,
                local_monotonic_ts_ns=record.local_monotonic_ts_ns,
            )
        receive_ts_ns = int(record.get("local_receive_ts_ns") or time.time_ns())
        schema = str(record.get("schema", "unknown"))
        enriched = dict(record)
        enriched["metal"] = self.metal
        enriched["ingestion_sequence"] = self._next_ingestion_sequence()
        enriched["receive_ts_ns"] = receive_ts_ns
        enriched["source_event_ts_ns"] = record.get("ts_event")
        ts_recv = record.get("ts_recv")
        ts_in_delta = record.get("ts_in_delta")
        if isinstance(ts_recv, int) and isinstance(ts_in_delta, int):
            enriched["source_publish_ts_ns"] = ts_recv - ts_in_delta
        else:
            enriched["source_publish_ts_ns"] = None

        self.raw_store.write(
            source="databento_extracted",
            stream="dbn_{}".format(schema),
            metal=self.metal,
            receive_ts_ns=receive_ts_ns,
            record=enriched,
        )
        self.quality.record("raw_records")
        self.quality.record("cme_{}_records".format(schema))
        self.counts["records"] += 1
        self.counts[schema] = self.counts.get(schema, 0) + 1

        normalized_stream = _NORMALIZED_STREAMS.get(schema)
        if normalized_stream is None:
            self.quality.record("cme_unknown_schema")
        else:
            normalized = dict(enriched)
            normalized.pop("raw_record", None)
            self.normalized_store.write(
                source=normalized_stream,
                stream=normalized_stream,
                metal=self.metal,
                receive_ts_ns=receive_ts_ns,
                record=normalized,
            )

        if schema == "reconnect":
            self.quality.record("cme_reconnects")
            self.quality.record_missing_interval(
                "cme",
                int(record.get("start_ts_ns") or receive_ts_ns),
                (
                    int(record["end_ts_ns"])
                    if record.get("end_ts_ns") is not None
                    else None
                ),
                "databento_reconnect",
            )
        elif schema == "error":
            self._record_error(
                DatabentoAccessError(
                    "Databento gateway error: {}".format(
                        record.get("err", record.get("msg", "unknown"))
                    )
                )
            )

    def _record_error(self, error: Exception) -> None:
        with self._lock:
            self._errors.append(error)
            self.quality.record("cme_collection_errors")
            self._done.set()

    def _native_path(self) -> Path:
        date = datetime.now(timezone.utc).date().isoformat()
        return (
            Path(self.config.storage.root)
            / "raw"
            / "databento_native"
            / "date={}".format(date)
            / "metal={}".format(self.metal)
            / "run={}".format(self.run_id)
            / "glbx-mdp3.dbn"
        )

    def _start_writer(self) -> None:
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="databento-normalizer-{}".format(self.run_id),
            daemon=True,
        )
        self._writer_thread.start()

    async def _finish_writer(self) -> None:
        if self._writer_thread is None:
            return
        await asyncio.to_thread(self._record_queue.put, None)
        await asyncio.to_thread(self._record_queue.join)
        await asyncio.to_thread(self._writer_thread.join)

    async def run(self, duration_seconds: Optional[float] = None) -> Dict[str, int]:
        if duration_seconds is not None and duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive when supplied")
        failure: Optional[BaseException] = None
        close_failure: Optional[BaseException] = None
        late_failure: Optional[BaseException] = None
        started = False
        close_task: Optional[asyncio.Task] = None
        try:
            self._start_writer()
            self._native_stream_path = self._native_path()
            self._native_stream_path.parent.mkdir(parents=True, exist_ok=True)
            self._native_stream_handle = self._native_stream_path.open("xb")
            self.collector.add_stream(
                self._native_stream_handle,
                exception_callback=self._record_error,
            )
            self.collector.start_callbacks(
                self._enqueue_record,
                self._record_error,
                self._enqueue_record,
                defer_normalization=True,
            )
            started = True
            close_task = asyncio.create_task(self.collector.wait_for_close())
            deadline = (
                asyncio.get_running_loop().time() + duration_seconds
                if duration_seconds is not None
                else None
            )
            while not self._done.is_set():
                if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                    break
                closed, _ = await asyncio.wait((close_task,), timeout=0.05)
                if close_task in closed:
                    try:
                        await close_task
                    except BaseException as disconnect_error:
                        raise DatabentoAccessError(
                            "Databento feed disconnected unexpectedly: {}".format(
                                disconnect_error
                            )
                        ) from disconnect_error
                    raise DatabentoAccessError(
                        "Databento feed closed unexpectedly"
                    )
            if self._errors:
                raise DatabentoAccessError(
                    "Databento collection failed: {}".format(self._errors[0])
                ) from self._errors[0]
        except BaseException as error:
            failure = error
            raise
        finally:
            if started:
                try:
                    if close_task is None or not close_task.done():
                        self.collector.stop()
                    if close_task is None:
                        close_task = asyncio.create_task(
                            self.collector.wait_for_close()
                        )
                    await asyncio.wait_for(close_task, timeout=10.0)
                except BaseException as close_error:
                    if failure is None:
                        failure = close_error
                        close_failure = close_error
            if self._native_stream_handle is not None:
                native_failure: Optional[BaseException] = None
                try:
                    self._native_stream_handle.flush()
                    os.fsync(self._native_stream_handle.fileno())
                except BaseException as native_error:
                    native_failure = native_error
                finally:
                    try:
                        self._native_stream_handle.close()
                    except BaseException as native_close_error:
                        if native_failure is None:
                            native_failure = native_close_error
                if (
                    native_failure is not None
                    and failure is None
                    and close_failure is None
                ):
                    failure = native_failure
                    close_failure = native_failure
                if native_failure is not None:
                    with self._lock:
                        self._errors.append(native_failure)
                        self.quality.record("cme_collection_errors")
            try:
                await self._finish_writer()
            except BaseException as writer_error:
                if failure is None and close_failure is None:
                    failure = writer_error
                    late_failure = writer_error
            if failure is None and self._errors:
                late_failure = DatabentoAccessError(
                    "Databento collection failed: {}".format(self._errors[0])
                )
                failure = late_failure
            status = "completed"
            if isinstance(failure, asyncio.CancelledError):
                status = "cancelled"
            elif failure is not None:
                status = "failed"
            self._finalize(status, failure)
            if close_failure is not None:
                raise close_failure
            if late_failure is not None:
                raise late_failure
        return dict(self.counts)

    def stop(self) -> None:
        self._done.set()

    def _quality_report_path(self) -> Path:
        now = datetime.now(timezone.utc)
        return (
            Path(self.config.storage.root)
            / "derived"
            / "daily_quality"
            / "date={}".format(now.date().isoformat())
            / "run={}".format(self.run_id)
            / "report.json"
        )

    def _finalize(
        self, status: str, error: Optional[BaseException] = None
    ) -> None:
        if self._finalized:
            return
        errors: List[BaseException] = list(self._errors)
        finalize_failure: Optional[BaseException] = None
        if error is not None and error not in errors:
            errors.append(error)
        quality_path = self._quality_report_path()
        try:
            if not quality_path.exists():
                quality_path.parent.mkdir(parents=True, exist_ok=True)
                with quality_path.open("x", encoding="utf-8") as handle:
                    json.dump(
                        self.quality.report(
                            datetime.now(timezone.utc).date().isoformat(), self.run_id
                        ),
                        handle,
                        indent=2,
                        sort_keys=True,
                    )
                    handle.write("\n")
        except BaseException as report_error:
            errors.append(report_error)
            finalize_failure = report_error
            status = "failed"
            quality_path = None
        for store in (self.raw_store, self.normalized_store):
            try:
                store.close()
            except BaseException as close_error:
                errors.append(close_error)
                if finalize_failure is None:
                    finalize_failure = close_error
                status = "failed"

        output_paths = self.raw_store.paths + self.normalized_store.paths
        if (
            self._native_stream_path is not None
            and self._native_stream_path.exists()
        ):
            output_paths.append(self._native_stream_path)
        if quality_path is not None:
            output_paths.append(quality_path)
        update_run_manifest(
            Path(self.config.storage.root),
            self.run_id,
            status=status,
            counts=self.counts,
            output_paths=output_paths,
            errors=errors,
        )
        self._finalized = True
        if error is None and finalize_failure is not None:
            raise finalize_failure
