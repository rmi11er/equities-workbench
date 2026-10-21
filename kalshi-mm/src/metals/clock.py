"""Fail-closed local clock telemetry for synchronized metals research.

The collector reads the host's existing Chrony daemon state.  It never changes the
clock and never contacts a time server itself.  Exact command output and a normalized
projection are written to immutable hash-chained storage under the same global receive
sequencer as the market-data collectors.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

from .config import ClockConfig, MetalsConfig
from .ingestion import GlobalReceiveSequencer
from .quality import QualityTracker
from .storage import PartitionedRawStore, update_run_manifest, write_run_manifest

_SYSTEM_TIME = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)) seconds (fast|slow) of NTP time$"
)
_MAX_PROBE_OUTPUT_BYTES = 64 * 1024


class ClockTelemetryError(RuntimeError):
    """Clock evidence could not be collected or validated."""


@dataclass(frozen=True)
class ChronyProbeResult:
    raw_text: str
    clock_offset_ns: int
    synchronized: bool
    stratum: int
    leap_status: str


def parse_chrony_tracking(raw_text: str) -> ChronyProbeResult:
    """Parse ``chronyc -n tracking`` using local-minus-reference offset sign."""

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ClockTelemetryError("Chrony tracking output is empty")
    if len(raw_text.encode("utf-8")) > _MAX_PROBE_OUTPUT_BYTES:
        raise ClockTelemetryError("Chrony tracking output exceeds the size limit")
    fields: dict[str, str] = {}
    for line in raw_text.splitlines():
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        normalized_name = " ".join(name.strip().lower().split())
        if not normalized_name:
            continue
        if normalized_name in fields:
            raise ClockTelemetryError("Chrony tracking output repeats a field")
        fields[normalized_name] = value.strip()

    system_time = fields.get("system time")
    stratum_text = fields.get("stratum")
    leap_status = fields.get("leap status")
    if system_time is None or stratum_text is None or leap_status is None:
        raise ClockTelemetryError("Chrony tracking output is missing required fields")
    match = _SYSTEM_TIME.fullmatch(system_time)
    if match is None:
        raise ClockTelemetryError("Chrony system-time offset has an unknown format")
    try:
        seconds = Decimal(match.group(1))
        scaled = (seconds * Decimal(1_000_000_000)).to_integral_value(
            rounding=ROUND_HALF_EVEN
        )
        offset_ns = int(scaled)
        stratum = int(stratum_text)
    except (InvalidOperation, ValueError, OverflowError) as error:
        raise ClockTelemetryError("Chrony tracking values are invalid") from error
    if not seconds.is_finite() or seconds < 0:
        raise ClockTelemetryError("Chrony system-time magnitude must be finite")
    if match.group(2) == "slow":
        offset_ns = -offset_ns
    if not 0 <= stratum <= 16:
        raise ClockTelemetryError("Chrony stratum is outside the valid range")
    synchronized = leap_status.strip().lower() == "normal" and 1 <= stratum <= 15
    return ChronyProbeResult(
        raw_text=raw_text,
        clock_offset_ns=offset_ns,
        synchronized=synchronized,
        stratum=stratum,
        leap_status=leap_status,
    )


def probe_chrony(config: ClockConfig) -> ChronyProbeResult:
    """Read one local Chrony tracking report without changing clock state."""

    executable = config.chronyc_path
    resolved = executable if os.path.sep in executable else shutil.which(executable)
    if not resolved or not Path(resolved).is_file() or not os.access(resolved, os.X_OK):
        raise ClockTelemetryError("The configured chronyc executable was not found")
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            [resolved, "-n", "tracking"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=config.probe_timeout_seconds,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise ClockTelemetryError(
            f"Chrony tracking probe failed ({type(error).__name__})"
        ) from None
    if completed.returncode != 0:
        raise ClockTelemetryError(
            f"Chrony tracking probe exited unsuccessfully ({completed.returncode})"
        )
    return parse_chrony_tracking(completed.stdout)


def preflight_clock(config: MetalsConfig) -> dict[str, Any]:
    """Return one sanitized, non-persisting local clock-health probe."""

    try:
        result = probe_chrony(config.clock)
    except ClockTelemetryError as error:
        return {
            "ok": False,
            "provider": config.clock.provider,
            "status": "probe_failed",
            "error_type": type(error).__name__,
        }
    return {
        "ok": result.synchronized,
        "provider": config.clock.provider,
        "status": "synchronized" if result.synchronized else "not_synchronized",
        "clock_offset_abs_ns": abs(result.clock_offset_ns),
        "stratum": result.stratum,
        "leap_status": result.leap_status,
    }


class ClockTelemetryCollector:
    """Persist periodic Chrony samples as a required synchronized sidecar."""

    def __init__(
        self,
        config: MetalsConfig,
        metal: str,
        *,
        sequencer: GlobalReceiveSequencer | None = None,
        probe: Callable[[], ChronyProbeResult] | None = None,
    ) -> None:
        product = config.products.get(metal)
        if product is None or not product.enabled:
            raise ClockTelemetryError("Clock telemetry requires an enabled product")
        if not config.clock.enabled:
            raise ClockTelemetryError(
                "Synchronized collection requires [clock].enabled = true"
            )
        if config.clock.provider != "chrony":
            raise ClockTelemetryError("Unsupported clock telemetry provider")
        self.config = config
        self.metal = metal
        self.sequencer = sequencer or GlobalReceiveSequencer()
        self.quality = QualityTracker()
        self._probe = probe or (lambda: probe_chrony(config.clock))
        self._stop = asyncio.Event()
        self._run_started = False
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
        self.counts: dict[str, int] = {
            "clock_samples": 0,
            "clock_probe_errors": 0,
            "clock_elapsed_discontinuities": 0,
            "ingestion_monotonic_regressions": 0,
        }
        self._finalized = False
        write_run_manifest(
            Path(config.storage.root),
            self.run_id,
            {
                **config.public_snapshot(),
                "collector": {
                    "component": "clock_telemetry",
                    "provider": config.clock.provider,
                    "metal": metal,
                    "read_only": True,
                    "changes_system_clock": False,
                },
            },
        )

    def stop(self) -> None:
        self._stop.set()

    async def _sample(self) -> None:
        start_wall_ns = time.time_ns()
        start_monotonic_ns = time.monotonic_ns()
        try:
            result = await asyncio.to_thread(self._probe)
        except BaseException as error:
            self.counts["clock_probe_errors"] += 1
            self.quality.record("clock_probe_errors")
            self.quality.record_missing_interval(
                "clock",
                start_wall_ns,
                None,
                f"clock_probe_error:{type(error).__name__}",
            )
            if isinstance(error, ClockTelemetryError):
                raise
            raise ClockTelemetryError(
                f"Clock telemetry probe failed ({type(error).__name__})"
            ) from None
        if not isinstance(result, ChronyProbeResult):
            raise ClockTelemetryError("Clock telemetry probe returned an invalid type")
        end_wall_ns = time.time_ns()
        end_monotonic_ns = time.monotonic_ns()
        stamp = self.sequencer.stamp(
            "clock",
            "clock_telemetry",
            receive_ts_ns=end_wall_ns,
            local_monotonic_ts_ns=end_monotonic_ns,
        )
        if stamp.monotonic_regression:
            self.counts["ingestion_monotonic_regressions"] += 1
            self.quality.record("ingestion_monotonic_regressions")
            self.quality.record_missing_interval(
                "clock",
                end_wall_ns,
                end_wall_ns,
                "global_monotonic_regression",
            )
        probe_sha256 = hashlib.sha256(result.raw_text.encode("utf-8")).hexdigest()
        sample_ts_ns = start_wall_ns + (end_wall_ns - start_wall_ns) // 2
        elapsed_delta_ns = (end_wall_ns - start_wall_ns) - (
            end_monotonic_ns - start_monotonic_ns
        )
        elapsed_consistent = (
            abs(elapsed_delta_ns) <= self.config.clock.max_probe_elapsed_delta_ns
        )
        if not elapsed_consistent:
            self.counts["clock_elapsed_discontinuities"] += 1
            self.quality.record("clock_elapsed_discontinuities")
            self.quality.record_missing_interval(
                "clock",
                start_wall_ns,
                end_wall_ns,
                "clock_wall_monotonic_elapsed_discontinuity",
            )
        common = {
            "source": "local_chrony",
            "stream": "clock_telemetry",
            "metal": self.metal,
            "provider": self.config.clock.provider,
            "sample_ts_ns": sample_ts_ns,
            "probe_start_ts_ns": start_wall_ns,
            "receive_ts_ns": stamp.receive_ts_ns,
            "local_monotonic_ts_ns": stamp.local_monotonic_ts_ns,
            "previous_monotonic_ts_ns": stamp.previous_monotonic_ts_ns,
            "monotonic_regression": stamp.monotonic_regression,
            "ingestion_sequence": stamp.ingestion_sequence,
            "probe_duration_ns": max(0, end_monotonic_ns - start_monotonic_ns),
            "wall_monotonic_elapsed_delta_ns": elapsed_delta_ns,
            "allowed_wall_monotonic_elapsed_delta_abs_ns": (
                self.config.clock.max_probe_elapsed_delta_ns
            ),
            "wall_monotonic_elapsed_consistent": elapsed_consistent,
            "clock_offset_ns": result.clock_offset_ns,
            "clock_synchronized": result.synchronized
            and not stamp.monotonic_regression
            and elapsed_consistent,
            "stratum": result.stratum,
            "leap_status": result.leap_status,
            "probe_output_sha256": probe_sha256,
            "offset_sign_semantics": "local_clock_minus_ntp_reference",
        }
        self.raw_store.write(
            source="clock",
            stream="clock_probe",
            metal=self.metal,
            receive_ts_ns=stamp.receive_ts_ns,
            record={**common, "raw_text": result.raw_text},
        )
        self.normalized_store.write(
            source="clock_telemetry",
            stream="clock_telemetry",
            metal=self.metal,
            receive_ts_ns=stamp.receive_ts_ns,
            record=common,
        )
        self.counts["clock_samples"] += 1
        self.quality.record("clock_samples")
        self.quality.record_clock_offset(result.clock_offset_ns)

    async def _wait(self, timeout: float, external: asyncio.Event | None) -> None:
        tasks = [asyncio.create_task(self._stop.wait())]
        if external is not None:
            tasks.append(asyncio.create_task(external.wait()))
        try:
            await asyncio.wait(
                tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _finalize(self, status: str, error: BaseException | None = None) -> None:
        if self._finalized:
            return
        errors: list[BaseException] = []
        if error is not None:
            errors.append(error)
        try:
            self.raw_store.close()
        except BaseException as close_error:
            errors.append(close_error)
            status = "failed"
        try:
            self.normalized_store.close()
        except BaseException as close_error:
            errors.append(close_error)
            status = "failed"
        update_run_manifest(
            Path(self.config.storage.root),
            self.run_id,
            status=status,
            counts=self.counts,
            output_paths=[*self.raw_store.paths, *self.normalized_store.paths],
            errors=errors,
        )
        self._finalized = True

    async def run(
        self,
        *,
        duration_seconds: float | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> dict[str, int]:
        if self._run_started:
            raise RuntimeError("A clock telemetry collector can run only once")
        if duration_seconds is not None and (
            not math.isfinite(duration_seconds) or duration_seconds <= 0
        ):
            raise ValueError("Clock telemetry duration must be positive and finite")
        self._run_started = True
        loop = asyncio.get_running_loop()
        deadline = None if duration_seconds is None else loop.time() + duration_seconds
        failure: BaseException | None = None
        try:
            while not self._stop.is_set() and not (
                stop_event is not None and stop_event.is_set()
            ):
                if deadline is not None and loop.time() >= deadline:
                    break
                await self._sample()
                timeout = self.config.clock.sample_interval_seconds
                if deadline is not None:
                    timeout = min(timeout, max(0.0, deadline - loop.time()))
                if timeout <= 0:
                    break
                await self._wait(timeout, stop_event)
        except BaseException as error:
            failure = error
            raise
        finally:
            status = "completed"
            if isinstance(failure, asyncio.CancelledError):
                status = "cancelled"
            elif failure is not None:
                status = "failed"
            self._finalize(status, failure)
        return dict(self.counts)
