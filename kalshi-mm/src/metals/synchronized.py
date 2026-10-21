"""Fail-closed synchronized collection for the complete research panel.

This module coordinates read-only collectors only.  It deliberately has no
dependency on execution, order, or strategy modules.
"""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .clock import ClockTelemetryCollector
from .cme import DatabentoMetalsRecorder
from .config import MetalsConfig
from .ingestion import GlobalReceiveSequencer
from .provider_approval import require_config_provider_operation
from .pyth_lazer import PythLazerCollector
from .quality import QualityTracker
from .recorder import MetalsRecorder
from .storage import update_run_manifest, write_run_manifest

SYNCHRONIZED_SOURCES = ("kalshi", "pyth_lazer", "databento")
SYNCHRONIZED_TELEMETRY = ("clock",)
SYNCHRONIZED_COMPONENTS = (*SYNCHRONIZED_SOURCES, *SYNCHRONIZED_TELEMETRY)


class SynchronizedCollectionError(RuntimeError):
    """The complete synchronized research panel could not be collected."""


def merge_quality_trackers(children: Mapping[str, Any]) -> QualityTracker:
    """Merge child quality state in stable source and market order."""

    merged = QualityTracker()
    for source in SYNCHRONIZED_COMPONENTS:
        child = children.get(source)
        tracker = getattr(child, "quality", None)
        if tracker is None:
            continue
        merged.counters.update(dict(tracker.counters))
        for coverage_source in sorted(tracker.source_coverage_ns):
            merged.source_coverage_ns[coverage_source].update(
                dict(tracker.source_coverage_ns[coverage_source])
            )
        merged.clock_offsets_ns.extend(int(value) for value in tracker.clock_offsets_ns)
        merged.missing_intervals.extend(
            dict(value) for value in tracker.missing_intervals
        )
        merged.metadata_changes.extend(
            dict(value) for value in tracker.metadata_changes
        )
        merged.exclusions.extend(dict(value) for value in tracker.exclusions)
        for ticker in sorted(tracker.market_event_counts):
            merged.market_event_counts[ticker].update(
                dict(tracker.market_event_counts[ticker])
            )
    return merged


class SynchronizedMetalsRecorder:
    """Run Kalshi, Pyth Lazer, and Databento under one receive sequencer.

    Individual source collectors remain usable on their own.  This coordinator
    is intentionally stricter: its research panel is valid only when all three
    market sources and required clock telemetry are configured, start, remain
    supervised, and stop cleanly.
    """

    def __init__(
        self,
        config: MetalsConfig,
        metal: str,
        *,
        sequencer: Optional[GlobalReceiveSequencer] = None,
        kalshi_recorder: Optional[Any] = None,
        pyth_collector: Optional[Any] = None,
        databento_recorder: Optional[Any] = None,
        clock_collector: Optional[Any] = None,
        shutdown_timeout_seconds: float = 15.0,
    ) -> None:
        if shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown_timeout_seconds must be positive")
        product = config.products.get(metal)
        if product is None or not product.enabled:
            raise SynchronizedCollectionError(
                "Synchronized collection requires an enabled product"
            )
        missing = []
        if not config.pyth.enabled:
            missing.append("pyth_lazer")
        if not config.cme.enabled:
            missing.append("databento")
        if not config.clock.enabled:
            missing.append("clock")
        if missing:
            raise SynchronizedCollectionError(
                "Synchronized research-panel collection requires all market "
                "sources and clock telemetry; disabled: {}".format(", ".join(missing))
            )
        if config.shadow.execution_mode != "shadow" or config.shadow.allow_live_orders:
            raise SynchronizedCollectionError(
                "Synchronized collection is restricted to read-only shadow mode"
            )
        self.pyth_provider_approval = require_config_provider_operation(
            config, "pyth", "live_lazer"
        )
        self.databento_provider_approval = require_config_provider_operation(
            config, "databento", "live"
        )

        self.config = config
        self.metal = metal
        self.sequencer = sequencer or GlobalReceiveSequencer()
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.kalshi_recorder = kalshi_recorder or MetalsRecorder(
            config,
            sequencer=self.sequencer,
        )
        self.pyth_collector = pyth_collector or PythLazerCollector(
            config,
            metal,
            sequencer=self.sequencer,
        )
        self.databento_recorder = databento_recorder or DatabentoMetalsRecorder(
            config,
            metal,
            sequencer=self.sequencer,
        )
        self.clock_collector = clock_collector or ClockTelemetryCollector(
            config,
            metal,
            sequencer=self.sequencer,
        )
        self._children = {
            "kalshi": self.kalshi_recorder,
            "pyth_lazer": self.pyth_collector,
            "databento": self.databento_recorder,
            "clock": self.clock_collector,
        }
        for source, child in self._children.items():
            if getattr(child, "sequencer", None) is not self.sequencer:
                raise SynchronizedCollectionError(
                    "{} collector does not use the coordinator's global sequencer".format(
                        source
                    )
                )

        self.run_id = str(uuid.uuid4())
        self._stop_event: Optional[asyncio.Event] = None
        self._run_started = False

    def stop(self) -> None:
        """Request one coordinated, graceful stop for every child source."""

        if self._stop_event is not None:
            self._stop_event.set()
        for child in self._children.values():
            child.stop()

    async def _run_kalshi(self) -> Any:
        return await self.kalshi_recorder.run()

    async def _run_pyth(self) -> Any:
        assert self._stop_event is not None
        return await self.pyth_collector.run(stop_event=self._stop_event)

    async def _run_databento(self) -> Any:
        return await self.databento_recorder.run()

    async def _run_clock(self) -> Any:
        assert self._stop_event is not None
        return await self.clock_collector.run(stop_event=self._stop_event)

    @staticmethod
    def _result_counts(child: Any, result: Any) -> Dict[str, Any]:
        if isinstance(result, Mapping):
            return dict(result)
        child_counts = getattr(child, "counts", None)
        if isinstance(child_counts, Mapping):
            return dict(child_counts)
        quality = getattr(child, "quality", None)
        quality_counts = getattr(quality, "counters", None)
        if isinstance(quality_counts, Mapping):
            return dict(quality_counts)
        private_counts = getattr(child, "_counts", None)
        if isinstance(private_counts, Mapping):
            return dict(private_counts)
        return {}

    def _child_runs(
        self,
        results: Mapping[str, Any],
        source_errors: Mapping[str, BaseException],
    ) -> Dict[str, Any]:
        children: Dict[str, Any] = {}
        for source, child in self._children.items():
            error = source_errors.get(source)
            children[source] = {
                "run_id": getattr(child, "run_id", None),
                "status": "failed" if error is not None else "completed",
                "counts": self._result_counts(child, results.get(source)),
            }
        return children

    def _child_manifest_paths(self) -> list[Path]:
        root = Path(self.config.storage.root)
        paths = []
        for child in self._children.values():
            child_run_id = getattr(child, "run_id", None)
            if not child_run_id:
                continue
            path = root / "manifests" / "run={}".format(child_run_id) / "manifest.json"
            if path.exists():
                paths.append(path)
        return paths

    def _write_unified_quality_report(
        self,
        quality: QualityTracker,
        sequence_audit: Mapping[str, Any],
    ) -> Path:
        now = datetime.now(timezone.utc)
        path = (
            Path(self.config.storage.root)
            / "derived"
            / "daily_quality"
            / "date={}".format(now.date().isoformat())
            / "run={}".format(self.run_id)
            / "report.json"
        )
        report = quality.report(now.date().isoformat(), self.run_id)
        report["global_receive_sequence_audit"] = dict(sequence_audit)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    async def run(
        self,
        *,
        duration_seconds: Optional[float] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> Dict[str, Any]:
        """Collect the complete panel until duration, external stop, or failure."""

        if self._run_started:
            raise RuntimeError("A synchronized recorder instance can run only once")
        if duration_seconds is not None:
            try:
                finite_duration = math.isfinite(duration_seconds)
            except TypeError as error:
                raise ValueError("duration_seconds must be a finite number") from error
            if not finite_duration or duration_seconds <= 0:
                raise ValueError(
                    "duration_seconds must be positive and finite when supplied"
                )
        self._run_started = True
        self._stop_event = asyncio.Event()

        storage_root = Path(self.config.storage.root)
        manifest_config = self.config.public_snapshot()
        manifest_config["collector"] = {
            "component": "synchronized_metals_live",
            "metal": self.metal,
            "required_sources": list(SYNCHRONIZED_SOURCES),
            "required_telemetry": list(SYNCHRONIZED_TELEMETRY),
            "read_only": True,
            "duration_seconds": duration_seconds,
            "provider_approval_ids": {
                "pyth": self.pyth_provider_approval.approval_id,
                "databento": self.databento_provider_approval.approval_id,
            },
        }
        write_run_manifest(
            storage_root,
            self.run_id,
            manifest_config,
            counts={"required_component_count": len(SYNCHRONIZED_COMPONENTS)},
            output_paths=(
                self.pyth_provider_approval.approval_path,
                self.pyth_provider_approval.manifest_path,
                self.databento_provider_approval.approval_path,
                self.databento_provider_approval.manifest_path,
            ),
        )

        source_tasks = {
            "kalshi": asyncio.create_task(
                self._run_kalshi(), name="metals-sync-kalshi"
            ),
            "pyth_lazer": asyncio.create_task(
                self._run_pyth(), name="metals-sync-pyth"
            ),
            "databento": asyncio.create_task(
                self._run_databento(), name="metals-sync-databento"
            ),
            "clock": asyncio.create_task(self._run_clock(), name="metals-sync-clock"),
        }
        task_sources = {task: source for source, task in source_tasks.items()}
        boundary_tasks = [
            asyncio.create_task(self._stop_event.wait(), name="metals-sync-stop")
        ]
        if duration_seconds is not None:
            boundary_tasks.append(
                asyncio.create_task(
                    asyncio.sleep(duration_seconds), name="metals-sync-duration"
                )
            )
        if stop_event is not None:
            boundary_tasks.append(
                asyncio.create_task(stop_event.wait(), name="metals-sync-external-stop")
            )

        results: Dict[str, Any] = {}
        source_errors: Dict[str, BaseException] = {}
        run_error: Optional[BaseException] = None
        quality_path: Optional[Path] = None
        try:
            done, _ = await asyncio.wait(
                [*source_tasks.values(), *boundary_tasks],
                return_when=asyncio.FIRST_COMPLETED,
            )
            early_sources = [task for task in done if task in task_sources]
            if early_sources:
                first = early_sources[0]
                source = task_sources[first]
                if first.cancelled():
                    cause: BaseException = asyncio.CancelledError()
                else:
                    cause = first.exception() or SynchronizedCollectionError(
                        "{} collector stopped before the coordinated boundary".format(
                            source
                        )
                    )
                source_errors[source] = cause
                run_error = SynchronizedCollectionError(
                    "Synchronized collection failed because {} stopped".format(source)
                )

            self.stop()
            try:
                gathered = await asyncio.wait_for(
                    asyncio.gather(*source_tasks.values(), return_exceptions=True),
                    timeout=self.shutdown_timeout_seconds,
                )
            except asyncio.TimeoutError as error:
                for task in source_tasks.values():
                    task.cancel()
                await asyncio.gather(*source_tasks.values(), return_exceptions=True)
                run_error = SynchronizedCollectionError(
                    "Synchronized collectors did not stop within {:.1f}s".format(
                        self.shutdown_timeout_seconds
                    )
                )
                source_errors.setdefault("coordinator", error)
                gathered = []

            for (source, _), result in zip(source_tasks.items(), gathered):
                if isinstance(result, BaseException):
                    source_errors.setdefault(source, result)
                    if run_error is None:
                        run_error = SynchronizedCollectionError(
                            "Synchronized collection failed because {} failed".format(
                                source
                            )
                        )
                else:
                    results[source] = result
        except BaseException as error:
            run_error = error
            self.stop()
            for task in source_tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*source_tasks.values(), return_exceptions=True)
        finally:
            for task in boundary_tasks:
                task.cancel()
            await asyncio.gather(*boundary_tasks, return_exceptions=True)

            child_runs = self._child_runs(results, source_errors)
            quality = merge_quality_trackers(self._children)
            sequence_audit = self.sequencer.audit(SYNCHRONIZED_COMPONENTS)
            quality.record("causal_ordering_checked")
            ordering_violations = len(sequence_audit["missing_sources"])
            if not sequence_audit["contiguous"]:
                ordering_violations += 1
            ordering_violations += int(
                quality.counters.get("ingestion_monotonic_regressions", 0)
            )
            if ordering_violations:
                quality.record(
                    "timestamp_ordering_violations", count=ordering_violations
                )
                if run_error is None:
                    run_error = SynchronizedCollectionError(
                        "Synchronized collection failed global receive-order audit"
                    )
            try:
                quality_path = self._write_unified_quality_report(
                    quality, sequence_audit
                )
            except BaseException as quality_error:
                source_errors["coordinator_quality"] = quality_error
                if run_error is None:
                    run_error = SynchronizedCollectionError(
                        "Synchronized collection could not write unified quality"
                    )
            parent_counts: Dict[str, Any] = {
                "required_component_count": len(SYNCHRONIZED_COMPONENTS),
                "global_receive_sequence_last": self.sequencer.last_sequence,
                "global_receive_sequence_audit": dict(sequence_audit),
                "child_runs": child_runs,
            }
            update_run_manifest(
                storage_root,
                self.run_id,
                status="failed" if run_error is not None else "completed",
                counts=parent_counts,
                output_paths=[
                    *self._child_manifest_paths(),
                    *([quality_path] if quality_path is not None else []),
                ],
                errors=[run_error] if run_error is not None else [],
            )

        if run_error is not None:
            if isinstance(run_error, asyncio.CancelledError):
                raise run_error
            raise run_error from next(iter(source_errors.values()), None)
        return {
            "run_id": self.run_id,
            "global_receive_sequence_last": self.sequencer.last_sequence,
            "child_runs": self._child_runs(results, source_errors),
            "quality_report_path": str(quality_path),
        }


async def collect_synchronized_metals(
    config: MetalsConfig,
    metal: str,
    *,
    duration_seconds: Optional[float] = None,
    stop_event: Optional[asyncio.Event] = None,
) -> Dict[str, Any]:
    """Convenience entry point for future CLI integration."""

    recorder = SynchronizedMetalsRecorder(config, metal)
    return await recorder.run(
        duration_seconds=duration_seconds,
        stop_event=stop_event,
    )
