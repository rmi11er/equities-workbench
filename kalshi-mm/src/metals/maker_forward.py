"""Bounded, read-only Silver maker collection and operator controls.

Run with ``python -m src.metals.maker_forward --help``. This module only uses
Kalshi's read client; shadow orders never reach an exchange order endpoint.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import math
import os
import time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from .baseline_fair import BaselineVolModel, load_baseline_vol_model
from .config import MetalsConfig, load_metals_config, resolve_kalshi_credentials
from .maker_inputs import ManualControl
from .maker_policy import MakerEngine, MakerSettings, VERSION, tiny_maker_config as tiny_config
from .maker_trial import TrialResearchPermit
from .pyth_lazer import PythLazerCollector
from .recorder import MetalsRecorder
from .shadow_session import ShadowSession
from .storage import PartitionedRawStore, _atomic_write_json, update_run_manifest
from .timeutil import iso8601_to_ns
from .types import SourceEnvelope

NSEC = 1_000_000_000
WATCHDOG_SECONDS = 0.05


def _silver_config(config: MetalsConfig) -> MetalsConfig:
    product = config.products.get("silver")
    if product is None or not product.enabled or product.pyth_symbol != "Metal.Index.SILVER/USD":
        raise ValueError("maker requires an enabled exact Silver index product")
    return replace(tiny_config(config), products={"silver": product})


def _xag_product(config: MetalsConfig) -> str:
    matches = [name for name, product in config.enabled_products.items()
               if product.pyth_symbol == "Metal.XAG/USD" and product.pyth_lazer_id == 345]
    if len(matches) != 1:
        raise ValueError("Pyth configuration requires one enabled Metal.XAG/USD feed 345")
    return matches[0]


def _data_authorization(settings: MakerSettings,
                        trial_permit: TrialResearchPermit | None) -> dict[str, Any]:
    if trial_permit is not None:
        return {"mode": "user_directed_trial_research", "provider_permission_confirmed": False,
                "trial_research_id": trial_permit.sha256,
                "trial_research_permit_path": str(trial_permit.path.resolve()) if trial_permit.path else None,
                "trial_research_permit": trial_permit.public_dict()}
    return {"mode": "formal_provider_approval" if settings.require_fast else "exact_only_diagnostic"}


def preflight(config: MetalsConfig, model_path: Path, *,
              pyth_config: MetalsConfig | None, settings: MakerSettings,
              trial_permit: TrialResearchPermit | None = None) -> dict[str, Any]:
    """Inspect local configuration and gates without contacting any provider."""
    if trial_permit is not None and not settings.require_fast:
        raise ValueError("trial research permit is only valid for the fast XAG maker path")
    model, study_id = load_baseline_vol_model(model_path)
    del model
    config = _silver_config(config)
    checks: dict[str, Any] = {"shadow_only": True, "silver_only": True,
                             "model_valid": True, "kalshi_credentials_available": False}
    try:
        resolve_kalshi_credentials(config.kalshi)
        checks["kalshi_credentials_available"] = True
    except (ValueError, OSError):
        pass
    ready = all(checks.values())
    if settings.require_fast:
        checks.update(pyth_configuration_valid=False, pyth_credentials_available=False,
                      pyth_provider_gate=False)
        if trial_permit is not None:
            checks["trial_research_gate"] = False
        if pyth_config is not None:
            try:
                product = _xag_product(pyth_config)
                checks["pyth_configuration_valid"] = True
                checks["pyth_credentials_available"] = bool(
                    os.environ.get(pyth_config.pyth.api_key_env, "").strip())
                # Construction checks the existing provider gate and endpoints;
                # it neither initializes stores nor connects to the provider.
                PythLazerCollector(pyth_config, product)
                checks["pyth_provider_gate"] = True
            except (ValueError, OSError, RuntimeError):
                pass
            if trial_permit is not None:
                try:
                    MakerXagCollector(pyth_config, _xag_product(pyth_config),
                                      recorder=None, trial_permit=trial_permit)
                    checks["trial_research_gate"] = True
                except (ValueError, OSError, RuntimeError):
                    pass
        selected_gate = "trial_research_gate" if trial_permit is not None else "pyth_provider_gate"
        ready = (ready and checks["pyth_configuration_valid"]
                 and checks["pyth_credentials_available"] and checks[selected_gate])
    return {"ready": ready, "checks": checks,
            "settings": settings.public_dict(), "baseline_study_id": study_id,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "risk": config.public_snapshot()["shadow"], "live_orders": False,
            "research_provenance": _data_authorization(settings, trial_permit),
            "mode": "xag_cancel_only" if settings.require_fast else "exact_only_diagnostic"}


def write_control(path: Path, *, version: int, volatility_multiplier: Decimal,
                  paused: bool, reason: str, ttl_seconds: int,
                  now_ns: int | None = None) -> dict[str, Any]:
    """Serialize cooperating writers and atomically replace a newer revision."""
    if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= 86400:
        raise ValueError("control TTL must be between 1 and 86400 seconds")
    now = time.time_ns() if now_ns is None else now_ns
    control = ManualControl(version, now, now + ttl_seconds * NSEC,
                            volatility_multiplier, paused, reason)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            old = ManualControl.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if control.version <= old.version:
                raise ValueError("control version must strictly increase")
            if control.effective_ts_ns < old.effective_ts_ns:
                raise ValueError("control effective time cannot move backwards")
        _atomic_write_json(path, control.public_dict(), create=not path.exists())
    return control.public_dict()


class MakerRecorder(MetalsRecorder):
    """Adapt persisted source observations into one deterministic input stream."""

    def __init__(self, config: MetalsConfig, model: BaselineVolModel, study_id: str,
                 settings: MakerSettings, control_path: Path, model_sha256: str,
                 *, require_credentials: bool = True, baseline_model_path: Path | None = None,
                 trial_permit: TrialResearchPermit | None = None) -> None:
        if trial_permit is not None:
            if not settings.require_fast:
                raise ValueError("trial research permit requires fast XAG")
            trial_permit.require_current()
        self.trial_permit = trial_permit
        self.trial_duration_seconds: float | None = None
        self.data_authorization = _data_authorization(settings, trial_permit)
        super().__init__(_silver_config(config), require_credentials=require_credentials)
        self.control_path, self.model_sha256 = control_path, model_sha256
        self.baseline_model_path = baseline_model_path
        self._control_digest: str | None = None
        self._metadata_fingerprints: dict[str, str] = {}
        self._settlement_last_refresh: dict[str, int] = {}
        self._fatal = asyncio.Event()
        self._fatal_error: BaseException | None = None
        self._source_tasks: set[asyncio.Task] = set()
        self.input_store = PartitionedRawStore(Path(config.storage.root), run_id=self.run_id,
                                               fsync_every_records=1, tier="normalized")
        context = {"component": VERSION, "collector_run_id": self.run_id,
                   "model_sha256": model_sha256, "settings": settings.public_dict(),
                   "research_provenance": self.data_authorization}
        self.sessions = {arm: ShadowSession(self.config, manifest_context={**context, "arm": arm})
                         for arm in ("manual", "baseline")}
        self.engine = MakerEngine(model, study_id, settings, self.sessions)
        self.input_count = 0
        manifest_path = Path(config.storage.root) / "manifests" / f"run={self.run_id}" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["research_provenance"] = self.data_authorization
        _atomic_write_json(manifest_path, manifest, create=False)

    def deliver(self, event: dict[str, Any]) -> None:
        # Append and fsync before changing counterfactual state. A failed input
        # write must not produce an unrepeatable order or fill.
        self.input_store.write("maker_input", "maker_input", "silver",
                               event["receive_ts_ns"], {**event, "collector_run_id": self.run_id})
        self.input_count += 1
        if event["kind"] == "stop":
            self.engine.close(event)
        else:
            self.engine.handle(event)

    def synthetic(self, kind: str, **payload: Any) -> dict[str, Any]:
        stamp = self.sequencer.stamp("maker", kind)
        event = {"kind": kind, "receive_ts_ns": stamp.receive_ts_ns,
                 "local_monotonic_ts_ns": stamp.local_monotonic_ts_ns,
                 "ingestion_sequence": stamp.ingestion_sequence, **payload}
        self.deliver(event)
        return event

    @staticmethod
    def _source_event(envelope: SourceEnvelope, kind: str, **payload: Any) -> dict[str, Any]:
        return {"kind": kind, "receive_ts_ns": envelope.receive_ts_ns,
                "local_monotonic_ts_ns": envelope.local_monotonic_ts_ns,
                "ingestion_sequence": envelope.ingestion_sequence, **payload}

    def check_control(self) -> None:
        try:
            raw = self.control_path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
        except OSError:
            digest, control = "missing-or-unreadable", None
        else:
            try:
                control = json.loads(raw)
                ManualControl.from_dict(control)
            except (ValueError, TypeError, KeyError):
                control = None
        if digest != self._control_digest:
            self.synthetic("control", control=control, control_file_sha256=digest)
            self._control_digest = digest

    def _publish_metadata(self, ticker: str, *, result: Any = None) -> None:
        meta = self.market_versions[ticker]
        payload = {"metadata": meta.to_dict(), "result": result if result in ("yes", "no") else None}
        fingerprint = json.dumps(payload, sort_keys=True, default=str)
        if fingerprint != self._metadata_fingerprints.get(ticker):
            self.synthetic("metadata", **payload)
            self._metadata_fingerprints[ticker] = fingerprint

    async def discover_once(self) -> list[dict[str, Any]]:
        markets = await super().discover_once()
        for ticker in self.market_versions:
            self._publish_metadata(ticker, result=self._market_raw.get(ticker, {}).get("result"))
        return markets

    async def _discovery_loop(self) -> None:
        # The base recorder's research loop tolerates errors. This experimental
        # runner propagates processing/storage errors to its task supervisor.
        while not self._stop.is_set():
            await self.discover_once()
            await asyncio.sleep(self.config.kalshi.discovery_interval_seconds)

    def _book_input(self, envelope: SourceEnvelope, ticker: str) -> None:
        book = self.books[ticker]
        def top(levels):
            if not levels:
                return {}
            price = max(levels, key=Decimal)
            return {price: levels[price]}
        self.deliver(self._source_event(
            envelope, "book", market_ticker=ticker, valid=book.valid,
            yes_bids=top(book.yes_bids), no_bids=top(book.no_bids),
            book_state_hash=book.state_hash()))

    def _record_book_snapshot(self, envelope, metal, message):
        super()._record_book_snapshot(envelope, metal, message)
        self._book_input(envelope, str(message.get("market_ticker", message.get("ticker", ""))))

    def _record_book_delta(self, envelope, metal, message):
        super()._record_book_delta(envelope, metal, message)
        self._book_input(envelope, str(message.get("market_ticker", message.get("ticker", ""))))

    def _record_trade(self, envelope, metal, message):
        super()._record_trade(envelope, metal, message)
        self.deliver(self._source_event(
            envelope, "trade", market_ticker=str(message.get("market_ticker", message.get("ticker", ""))),
            trade_id=message.get("trade_id"), quantity_fp=message.get("count_fp", message.get("count")),
            yes_price_dollars=message.get("yes_price_dollars", message.get("yes_price")),
            no_price_dollars=message.get("no_price_dollars", message.get("no_price")),
            taker_outcome_side=message.get("taker_outcome_side"),
            is_block_trade=bool(message.get("is_block_trade", False)),
            source_event_ts_ns=envelope.source_event_ts_ns))

    def _record_kalshi_pyth(self, envelope, metal, message):
        super()._record_kalshi_pyth(envelope, metal, message)
        if message.get("underlying_ticker") != "Metal.Index.SILVER/USD":
            return
        self.deliver(self._source_event(envelope, "exact", value=message.get("value_usd"),
                                       source_ts_ns=iso8601_to_ns(message.get("source_ts_ms"))))

    async def _process_ws(self, envelope: SourceEnvelope) -> None:
        try:
            await super()._process_ws(envelope)
            if envelope.gap_flag:
                message = envelope.raw.get("msg", {})
                ticker = message.get("market_ticker", message.get("ticker"))
                self.deliver(self._source_event(envelope, "gap", market_ticker=ticker,
                                               reason="kalshi_sequence_gap"))
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._fatal_error = error
            self._fatal.set()
            raise

    async def _consume_ws(self, tickers, connection_stop):
        task = asyncio.current_task()
        self._source_tasks.add(task)
        try:
            await super()._consume_ws(tickers, connection_stop)
        finally:
            self._source_tasks.discard(task)

    def _record_websocket_disconnect(self, error):
        for book in self.books.values():
            book.invalidate()
        event = super()._record_websocket_disconnect(error)
        self.deliver({"kind": "gap", "reason": "kalshi_disconnect",
                      **{key: event[key] for key in (
                          "receive_ts_ns", "local_monotonic_ts_ns", "ingestion_sequence")}})
        return event

    async def _websocket_supervisor(self) -> None:
        """Treat planned reconnects as gaps too, and own every child task."""
        backoff = 1.0
        while not self._stop.is_set():
            tickers = sorted(self._desired_tickers)
            self._ticker_change.clear()
            connection_stop = asyncio.Event()
            self._ws_attempt_number += 1
            self._write_transport_event("connection_attempt_started", connection_id=None,
                                        attempt_number=self._ws_attempt_number,
                                        subscribed_market_tickers=tickers)
            consume = asyncio.create_task(self._consume_ws(tickers, connection_stop))
            changed = asyncio.create_task(self._ticker_change.wait())
            stopped = asyncio.create_task(self._stop.wait())
            children = [consume, changed, stopped]
            try:
                done, _ = await asyncio.wait(children, return_when=asyncio.FIRST_COMPLETED)
            finally:
                connection_stop.set()
                for task in children:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*children, return_exceptions=True)
            if self._stop.is_set():
                return
            if self._fatal.is_set():
                raise self._fatal_error or RuntimeError("maker processing failed")
            error = consume.exception() if consume in done and not consume.cancelled() else None
            self._record_websocket_disconnect(error)
            if consume in done:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
            else:
                backoff = 1.0

    async def _watchdog(self) -> None:
        while not self._stop.is_set():
            if self.trial_permit is not None:
                self.trial_permit.require_current()
            self.check_control()
            self.synthetic("timer")
            await asyncio.sleep(WATCHDOG_SECONDS)

    async def _settlements(self) -> None:
        while not self._stop.is_set():
            now = time.time_ns()
            # At most one refresh per five seconds. Repeat until resolved;
            # unresolved contracts remain explicit in the experiment summary.
            candidates = [ticker for ticker, meta in self.market_versions.items()
                          if meta.close_ts_ns and meta.close_ts_ns <= now
                          and ticker not in self.engine.results]
            candidates.sort(key=lambda ticker: self._settlement_last_refresh.get(ticker, 0))
            for ticker in candidates[:1]:
                self._settlement_last_refresh[ticker] = now
                response = await self._client.get(f"/markets/{ticker}")
                self._write_raw_envelope(self._rest_envelope(response, "rest_maker_settlement"), "silver")
                raw = response.payload.get("market", {})
                merged = {**self._market_raw.get(ticker, {}), **raw}
                self._market_raw[ticker] = merged
                version, _ = self.metadata.observe(merged, metal="silver", receive_ts_ns=response.receive_ts_ns)
                self.market_versions[ticker] = version
                self._write_normalized("market_dim", "silver", version.metadata_receive_ts_ns, version.to_dict())
                self._publish_metadata(ticker, result=raw.get("result"))
            await asyncio.sleep(5)

    async def _raise_fatal(self) -> None:
        await self._fatal.wait()
        raise self._fatal_error or RuntimeError("maker source processing failed")

    async def run_maker(self, duration_seconds: float,
                        pyth_config: MetalsConfig | None = None) -> dict[str, Any]:
        if not math.isfinite(duration_seconds) or not 0 < duration_seconds <= 86400:
            raise ValueError("run duration must be finite, positive, and at most 86400 seconds")
        if self.engine.settings.require_fast and pyth_config is None:
            raise ValueError("fast XAG requires --pyth-config")
        pyth = None
        pyth_task = None
        tasks: list[asyncio.Task] = []
        failure: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        try:
            if self.trial_permit is not None:
                self.trial_permit.require_duration(duration_seconds)
                self.trial_duration_seconds = duration_seconds
            if pyth_config is not None and self.engine.settings.require_fast:
                pyth = MakerXagCollector(pyth_config, _xag_product(pyth_config), recorder=self,
                                        trial_permit=self.trial_permit)
            await self._client.start()
            if self._ws is None:
                raise RuntimeError("Kalshi credentials are required for read-only WebSocket data")
            await self._client.get("/account/limits")
            self.check_control()
            tasks = [asyncio.create_task(coro) for coro in (
                self._discovery_loop(), self._websocket_supervisor(), self._watchdog(),
                self._settlements(), self._raise_fatal())]
            if pyth is not None:
                pyth_task = asyncio.create_task(pyth.run(stop_event=self._stop))
                tasks.append(pyth_task)
            bound = asyncio.create_task(asyncio.sleep(duration_seconds))
            tasks.append(bound)
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task is not bound:
                    error = task.exception()
                    raise error or RuntimeError("maker child stopped unexpectedly")
            self.synthetic("stop")
            deadline = time.monotonic() + 2
            while any(s.engine.resting_orders for s in self.sessions.values()) and time.monotonic() < deadline:
                done, _ = await asyncio.wait([t for t in tasks if t is not bound],
                                             timeout=WATCHDOG_SECONDS,
                                             return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    error = task.exception()
                    raise error or RuntimeError("maker child stopped during cancellation drain")
        except BaseException as error:
            failure = error
        finally:
            if not self.engine.closed:
                try:
                    self.synthetic("stop")
                except BaseException as error:
                    cleanup_errors.append(error)
            self.stop()
            self._ticker_change.set()
            if pyth is not None:
                pyth.stop()
            if pyth_task is not None and not pyth_task.done() and failure is None:
                try:
                    await asyncio.wait_for(asyncio.shield(pyth_task), timeout=1)
                except asyncio.TimeoutError:
                    pass
                except BaseException as error:
                    cleanup_errors.append(error)
            for task in tasks:
                task.cancel()
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                        if result is not failure and all(result is not prior for prior in cleanup_errors):
                            cleanup_errors.append(result)
            source_tasks = list(self._source_tasks)
            for task in source_tasks:
                task.cancel()
            if source_tasks:
                results = await asyncio.gather(*source_tasks, return_exceptions=True)
                cleanup_errors.extend(result for result in results if isinstance(result, BaseException)
                                      and not isinstance(result, asyncio.CancelledError) and result is not failure)
            try:
                await self._client.close()
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                self.input_store.close()
            except BaseException as error:
                cleanup_errors.append(error)
            status = "failed" if failure or cleanup_errors else "completed"
            if isinstance(failure, asyncio.CancelledError):
                status = "cancelled"
            for session in self.sessions.values():
                try:
                    session.close(status=status, error=failure)
                except BaseException as error:
                    cleanup_errors.append(error)
            if cleanup_errors:
                status = "failed"
                for session in self.sessions.values():
                    try:
                        update_run_manifest(Path(self.config.storage.root), session.run_id,
                                            status=status, errors=cleanup_errors)
                    except BaseException as error:
                        cleanup_errors.append(error)
            try:
                self._finalize(status=status, counts={"maker_inputs": self.input_count},
                               error=failure or (cleanup_errors[0] if cleanup_errors else None))
            except BaseException as error:
                cleanup_errors.append(error)
        summary = {"summary": self.engine.summary(), "status": "failed" if cleanup_errors else status,
                   "settings": self.engine.settings.public_dict(),
                   "research_provenance": self.data_authorization,
                   "trial_research": self.trial_permit.public_dict() if self.trial_permit else None,
                   "collector_run_id": self.run_id,
                   "pyth_collector_run_id": None if pyth is None else pyth.run_id,
                   "shadow_run_ids": {arm: session.run_id for arm, session in self.sessions.items()},
                   "baseline_model_sha256": self.model_sha256,
                   "baseline_model_path": str(self.baseline_model_path.resolve()) if self.baseline_model_path else None,
                   "maker_input_counts": self.input_count,
                   "maker_input_paths": [str(path.resolve()) for path in self.input_store.paths],
                   "duration_seconds": duration_seconds,
                   "error_types": [type(error).__name__ for error in [failure, *cleanup_errors] if error]}
        path = Path(self.config.storage.root) / "derived" / "maker_forward" / f"run={self.run_id}" / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, json.loads(json.dumps(summary, default=str)), create=True)
        summary["report_path"] = str(path)
        if failure is not None:
            raise failure
        if cleanup_errors:
            raise cleanup_errors[0]
        return summary


class MakerXagCollector(PythLazerCollector):
    def __init__(self, config, metal, *, recorder: MakerRecorder | None,
                 trial_permit: TrialResearchPermit | None = None, **kwargs):
        self.recorder = recorder
        self.trial_permit = trial_permit
        super().__init__(config, metal,
                         sequencer=recorder.sequencer if recorder is not None else None, **kwargs)

    def _authorize_collection(self, config: MetalsConfig, metal: str) -> None:
        if self.trial_permit is None:
            super()._authorize_collection(config, metal)
            return
        if self.trial_permit.path is None:
            raise ValueError("trial research requires a persisted permit artifact")
        self.trial_permit.require_current()
        self.trial_permit.validate_scope(config, metal)

    def _authorization_manifest(self):
        if self.trial_permit is None:
            return super()._authorization_manifest()
        self._require_trial_current()
        return (_data_authorization(MakerSettings(), self.trial_permit),
                (self.trial_permit.path,))

    async def run(self, *, duration_seconds=None, stop_event=None):
        if self.trial_permit is None:
            return await super().run(duration_seconds=duration_seconds, stop_event=stop_event)
        if (self.recorder is None or self.recorder.trial_permit is None
                or self.recorder.trial_permit.sha256 != self.trial_permit.sha256
                or not self.recorder.engine.settings.require_fast):
            raise ValueError("trial XAG collection requires its attached maker recorder")
        planned = self.recorder.trial_duration_seconds
        self.trial_permit.require_duration(planned)
        if duration_seconds is not None or stop_event is not self.recorder._stop:
            raise ValueError("trial XAG lifetime must be controlled by its maker runner")
        # Preserve a hard source bound even if a future caller forgets to set
        # the shared stop event. Three seconds include the runner's drain.
        return await super().run(duration_seconds=float(planned) + 3, stop_event=stop_event)

    def _require_trial_current(self) -> None:
        trial_permit = getattr(self, "trial_permit", None)
        if trial_permit is not None:
            try:
                trial_permit.require_current()
            except BaseException as error:
                if self.recorder is not None:
                    self.recorder._fatal_error = error
                    self.recorder._fatal.set()
                raise

    async def _connection_once(self, endpoint, api_key, connection_id):
        self._require_trial_current()
        return await super()._connection_once(endpoint, api_key, connection_id)

    def _write_unparsed_source_message(self, *args, **kwargs):
        self._require_trial_current()
        return super()._write_unparsed_source_message(*args, **kwargs)

    def _write_source_message(self, *args, **kwargs):
        self._require_trial_current()
        sequence, rows = super()._write_source_message(*args, **kwargs)
        try:
            for row in rows:
                if row["endpoint_role"] != "active":
                    continue
                self.recorder.deliver({"kind": "fast", "value": row["price"],
                    "source_ts_ns": row["source_publish_ts_ns"],
                    **{key: row[key] for key in (
                        "receive_ts_ns", "local_monotonic_ts_ns", "ingestion_sequence")}})
        except BaseException as error:
            # The provider reconnect loop categorizes generic exceptions as
            # transport failures. Tell our supervisor before it can retry a
            # storage or strategy failure as if it were just a connection loss.
            self.recorder._fatal_error = error
            self.recorder._fatal.set()
            raise
        return sequence, rows

    def _mark_transport_gap(self, endpoint, connection_id, classification):
        active_was_lost = endpoint == self._active_endpoint
        super()._mark_transport_gap(endpoint, connection_id, classification)
        if active_was_lost:
            self.recorder.synthetic("gap", reason="active_xag_transport_gap")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/metals.toml")
        command.add_argument("--baseline-model", type=Path, required=True)
        command.add_argument("--pyth-config", type=Path)
        command.add_argument("--trial-research-permit", type=Path)
        command.add_argument("--maker-fee-per-contract", type=Decimal, required=True)
        command.add_argument("--exact-only-diagnostic", action="store_true")
        if name == "run":
            command.add_argument("--control-file", type=Path, required=True)
            command.add_argument("--duration-seconds", type=float, required=True)
            command.add_argument("--allow-live-shadow-data", action="store_true")
    control = commands.add_parser("control")
    control.add_argument("--control-file", type=Path, required=True)
    control.add_argument("--version", type=int, required=True)
    control.add_argument("--volatility-multiplier", type=Decimal, default=Decimal("1"))
    control.add_argument("--paused", action="store_true")
    control.add_argument("--reason", required=True)
    control.add_argument("--ttl-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    if args.command == "control":
        result = write_control(args.control_file, version=args.version,
                               volatility_multiplier=args.volatility_multiplier,
                               paused=args.paused, reason=args.reason, ttl_seconds=args.ttl_seconds)
    else:
        config = load_metals_config(args.config)
        pyth_config = load_metals_config(args.pyth_config) if args.pyth_config else None
        settings = MakerSettings(require_fast=not args.exact_only_diagnostic,
                                 maker_fee_per_contract=args.maker_fee_per_contract)
        trial_permit = TrialResearchPermit.load(args.trial_research_permit) if args.trial_research_permit else None
        result = preflight(config, args.baseline_model, pyth_config=pyth_config, settings=settings,
                           trial_permit=trial_permit)
        if args.command == "run":
            if not args.allow_live_shadow_data:
                parser.error("run requires --allow-live-shadow-data (read-only recording only)")
            if not result["ready"]:
                print(json.dumps(result, indent=2, default=str))
                return 2
            if not math.isfinite(args.duration_seconds) or not 0 < args.duration_seconds <= 86400:
                parser.error("run duration must be finite, positive, and at most 86400 seconds")
            if trial_permit is not None:
                trial_permit.require_duration(args.duration_seconds)
            model, study_id = load_baseline_vol_model(args.baseline_model)
            recorder = MakerRecorder(config, model, study_id, settings, args.control_file,
                                     result["model_sha256"], baseline_model_path=args.baseline_model,
                                     trial_permit=trial_permit)
            result = asyncio.run(recorder.run_maker(args.duration_seconds, pyth_config))
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ready", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
