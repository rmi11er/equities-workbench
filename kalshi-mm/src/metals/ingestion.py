"""Global receive ordering primitives for synchronized source collection.

The sequencer in this module is deliberately source-agnostic.  Live collectors call
it at their callback boundary, before handing work to parser or storage threads.  A
single sequencer instance shared by every collector in one process therefore defines
the deterministic receive order used by causal replay.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Generic, Iterable, Optional, TypeVar


class IngestionError(RuntimeError):
    """Base class for fail-closed synchronized-ingestion errors."""


class IngestionQueueOverflow(IngestionError):
    """Raised when an accepted source callback cannot enter the bounded queue."""


class IngestionQueueClosed(IngestionError):
    """Raised when a producer attempts to publish after queue closure."""


@dataclass(frozen=True)
class ReceiveStamp:
    """One process-global receive-order assignment."""

    ingestion_sequence: int
    source: str
    stream: str
    receive_ts_ns: int
    local_monotonic_ts_ns: int
    previous_monotonic_ts_ns: Optional[int]
    monotonic_regression: bool


class GlobalReceiveSequencer:
    """Assign a strict process-global order to concurrent source callbacks.

    Supplied timestamps are useful when a transport captures its clocks immediately
    before entering this object.  They are never used to reorder callbacks.  A
    supplied monotonic timestamp older than the previous accepted timestamp is
    retained and flagged so the daily data gate can fail closed.
    """

    def __init__(self, start_sequence: int = 0) -> None:
        if start_sequence < 0:
            raise ValueError("start_sequence cannot be negative")
        self._sequence = start_sequence
        self._start_sequence = start_sequence
        self._stamp_count = 0
        self._source_counts: Counter[str] = Counter()
        self._last_monotonic_ts_ns: Optional[int] = None
        self._lock = threading.Lock()

    def stamp(
        self,
        source: str,
        stream: str,
        *,
        receive_ts_ns: Optional[int] = None,
        local_monotonic_ts_ns: Optional[int] = None,
    ) -> ReceiveStamp:
        if not source.strip():
            raise ValueError("source cannot be empty")
        if not stream.strip():
            raise ValueError("stream cannot be empty")

        # Capture both clocks at the producer boundary.  The global sequence is
        # allocated under the same lock that checks monotonic ordering.
        wall_ns = time.time_ns() if receive_ts_ns is None else int(receive_ts_ns)
        monotonic_ns = (
            time.monotonic_ns()
            if local_monotonic_ts_ns is None
            else int(local_monotonic_ts_ns)
        )
        if wall_ns < 0 or monotonic_ns < 0:
            raise ValueError("receive clocks cannot be negative")

        with self._lock:
            previous = self._last_monotonic_ts_ns
            regression = previous is not None and monotonic_ns < previous
            self._sequence += 1
            self._stamp_count += 1
            self._source_counts[source] += 1
            self._last_monotonic_ts_ns = monotonic_ns
            return ReceiveStamp(
                ingestion_sequence=self._sequence,
                source=source,
                stream=stream,
                receive_ts_ns=wall_ns,
                local_monotonic_ts_ns=monotonic_ns,
                previous_monotonic_ts_ns=previous,
                monotonic_regression=regression,
            )

    @property
    def last_sequence(self) -> int:
        with self._lock:
            return self._sequence

    def audit(self, required_sources: Iterable[str] = ()) -> dict:
        """Return a locked integrity snapshot of all allocated receive stamps."""

        required = tuple(dict.fromkeys(str(value) for value in required_sources))
        with self._lock:
            expected_last = self._start_sequence + self._stamp_count
            missing_sources = [
                source for source in required if self._source_counts[source] == 0
            ]
            contiguous = self._sequence == expected_last
            return {
                "checked": True,
                "passed": contiguous and not missing_sources,
                "start_sequence": self._start_sequence,
                "last_sequence": self._sequence,
                "stamp_count": self._stamp_count,
                "contiguous": contiguous,
                "source_counts": dict(sorted(self._source_counts.items())),
                "required_sources": list(required),
                "missing_sources": missing_sources,
            }


PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class IngestionItem(Generic[PayloadT]):
    stamp: ReceiveStamp
    metal: str
    payload: PayloadT


class UnifiedIngestionQueue(Generic[PayloadT]):
    """Bounded multi-producer queue stamped at the acceptance boundary.

    Overflow is raised synchronously to the producer.  Callers must treat it as a
    failed collection run and record a classified missing interval; continuing after
    an overflow would make the replay stream incomplete and non-causal.
    """

    def __init__(
        self,
        sequencer: Optional[GlobalReceiveSequencer] = None,
        *,
        maxsize: int = 100_000,
    ) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be at least one")
        self.sequencer = sequencer or GlobalReceiveSequencer()
        self._queue: queue.Queue[IngestionItem[PayloadT]] = queue.Queue(
            maxsize=maxsize
        )
        self._closed = False
        self._state_lock = threading.Lock()

    def publish(
        self,
        source: str,
        stream: str,
        metal: str,
        payload: PayloadT,
        *,
        receive_ts_ns: Optional[int] = None,
        local_monotonic_ts_ns: Optional[int] = None,
    ) -> ReceiveStamp:
        if not metal.strip():
            raise ValueError("metal cannot be empty")
        with self._state_lock:
            if self._closed:
                raise IngestionQueueClosed("unified ingestion queue is closed")
            stamp = self.sequencer.stamp(
                source,
                stream,
                receive_ts_ns=receive_ts_ns,
                local_monotonic_ts_ns=local_monotonic_ts_ns,
            )
            try:
                self._queue.put_nowait(
                    IngestionItem(stamp=stamp, metal=metal, payload=payload)
                )
            except queue.Full as error:
                raise IngestionQueueOverflow(
                    "unified ingestion queue overflow at sequence {}".format(
                        stamp.ingestion_sequence
                    )
                ) from error
        return stamp

    def get(self, timeout: Optional[float] = None) -> IngestionItem[PayloadT]:
        return self._queue.get(timeout=timeout)

    def task_done(self) -> None:
        self._queue.task_done()

    def join(self) -> None:
        self._queue.join()

    def close(self) -> None:
        with self._state_lock:
            self._closed = True

    @property
    def closed(self) -> bool:
        with self._state_lock:
            return self._closed

    @property
    def size(self) -> int:
        return self._queue.qsize()
