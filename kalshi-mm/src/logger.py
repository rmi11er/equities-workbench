"""Async logging infrastructure with queue-based file writing."""

import asyncio
import csv
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional

from .config import LoggingConfig
from .types import TapeEntry


class AsyncFileHandler(logging.Handler):
    """
    Async-safe logging handler using a background thread.

    Uses a queue to decouple the main event loop from file I/O.
    """

    def __init__(self, filename: str, level: int = logging.INFO):
        super().__init__(level)
        self._queue: Queue[Optional[logging.LogRecord]] = Queue()
        self._filename = filename
        self._thread: Optional[Thread] = None
        self._running = False

        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start the background writer thread."""
        if self._running:
            return

        self._running = True
        self._thread = Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background writer thread."""
        if not self._running:
            return

        self._running = False
        self._queue.put(None)  # Sentinel to stop the loop

        if self._thread:
            self._thread.join(timeout=5.0)

    def emit(self, record: logging.LogRecord) -> None:
        """Queue a log record for writing."""
        if self._running:
            self._queue.put(record)

    def _writer_loop(self) -> None:
        """Background thread that writes records to file."""
        with open(self._filename, "a", buffering=1) as f:  # Line buffered
            while self._running or not self._queue.empty():
                try:
                    record = self._queue.get(timeout=1.0)
                    if record is None:
                        break
                    msg = self.format(record)
                    f.write(msg + "\n")
                except Exception:
                    pass  # Don't let logging errors crash the thread


class TapeWriter:
    """
    Async-safe CSV tape writer using a background thread.

    Writes TapeEntry records to CSV for post-trade analysis.
    """

    COLUMNS = [
        "ts", "ticker", "mid", "my_bid", "my_ask",
        "inventory", "unrealized_pnl", "realized_pnl",
        "latency_ms", "volatility"
    ]

    def __init__(self, filename: str):
        self._filename = filename
        self._queue: Queue[Optional[TapeEntry]] = Queue()
        self._thread: Optional[Thread] = None
        self._running = False

        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start the background writer thread."""
        if self._running:
            return

        self._running = True
        self._thread = Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background writer thread."""
        if not self._running:
            return

        self._running = False
        self._queue.put(None)

        if self._thread:
            self._thread.join(timeout=5.0)

    def write(self, entry: TapeEntry) -> None:
        """Queue a tape entry for writing."""
        if self._running:
            self._queue.put(entry)

    def _writer_loop(self) -> None:
        """Background thread that writes entries to CSV."""
        file_exists = os.path.exists(self._filename)

        with open(self._filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)

            # Write header if new file
            if not file_exists:
                writer.writeheader()

            while self._running or not self._queue.empty():
                try:
                    entry = self._queue.get(timeout=1.0)
                    if entry is None:
                        break

                    row = asdict(entry)
                    # Format timestamp
                    row["ts"] = entry.ts.isoformat()
                    writer.writerow(row)
                    f.flush()

                except Exception:
                    pass


class LogManager:
    """
    Manages all logging infrastructure.

    Provides:
    - Structured ops logging (ops.log)
    - Data tape recording (tape.csv)

    When initialized with a RunContext, uses run-specific paths.
    Otherwise falls back to config paths.
    """

    def __init__(self, config: LoggingConfig, run_context: Optional["RunContext"] = None):
        from .run_context import RunContext  # Import here to avoid circular
        self.config = config
        self.run_context = run_context

        # Determine paths: RunContext overrides config
        if run_context and run_context.ops_log_path:
            ops_log_path = run_context.ops_log_path
            tape_csv_path = run_context.tape_path
        else:
            ops_log_path = config.ops_log_path
            tape_csv_path = config.tape_csv_path

        # Setup ops logging
        self._ops_handler = AsyncFileHandler(
            ops_log_path,
            level=getattr(logging, config.log_level),
        )
        self._ops_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        # Setup tape writer
        self._tape_writer = TapeWriter(tape_csv_path)

        # Console handler for development
        self._console_handler = logging.StreamHandler()
        self._console_handler.setFormatter(
            logging.Formatter(
                "[%(levelname)s] %(message)s",
            )
        )
        self._console_handler.setLevel(getattr(logging, config.log_level))

    def start(self) -> None:
        """Start all logging infrastructure."""
        self._ops_handler.start()
        self._tape_writer.start()

        # Configure root logger
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(self._ops_handler)
        root.addHandler(self._console_handler)

        logging.info("Logging started")

        # Log run context if available
        if self.run_context:
            logging.info(f"Run ID: {self.run_context.run_id}")
            logging.info(f"Git: {self.run_context.git_info.commit_short} ({self.run_context.git_info.branch})")
            if self.run_context.git_info.dirty:
                logging.warning("Git working directory is dirty (uncommitted changes)")

    def stop(self) -> None:
        """Stop all logging infrastructure."""
        logging.info("Logging stopped")

        self._ops_handler.stop()
        self._tape_writer.stop()

        # Remove handlers
        root = logging.getLogger()
        root.removeHandler(self._ops_handler)
        root.removeHandler(self._console_handler)

    def record_tick(
        self,
        ticker: str,
        mid: float,
        my_bid: Optional[int],
        my_ask: Optional[int],
        inventory: int,
        unrealized_pnl: float,
        realized_pnl: float,
        latency_ms: float,
        volatility: float,
    ) -> None:
        """Record a tick to the data tape."""
        entry = TapeEntry(
            ts=datetime.now(),
            ticker=ticker,
            mid=mid,
            my_bid=my_bid,
            my_ask=my_ask,
            inventory=inventory,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            latency_ms=latency_ms,
            volatility=volatility,
        )
        self._tape_writer.write(entry)

    async def __aenter__(self) -> "LogManager":
        self.start()
        return self

    async def __aexit__(self, *args) -> None:
        self.stop()
