import asyncio
import hashlib
import json
import time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from src.metals import maker_forward as module
from src.metals.baseline_fair import SCHEMA_VERSION, USAGE_CLASSIFICATION
from src.metals.config import load_metals_config
from src.metals.maker_forward import MakerRecorder, MakerXagCollector, preflight, write_control
from src.metals.maker_policy import MakerSettings
from src.metals.maker_replay import replay_report
from src.metals.maker_trial import PUBLIC_SOURCES, TrialResearchPermit, materialize_trial_research
from src.metals.pyth_lazer import PythLazerCollector
from src.metals.types import SourceEnvelope
from tests.metals.test_forward_shadow import _model


def _config(tmp_path):
    config = load_metals_config("configs/metals.example.toml")
    return replace(config, storage=replace(config.storage, root=str(tmp_path / "data")))


def _model_path(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "study_id": "maker-test",
                               "usage_classification": USAGE_CLASSIFICATION,
                               "model": _model().public_dict()}), encoding="utf-8")
    return path


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    def absent(_):
        raise ValueError("test has no credentials")
    monkeypatch.setattr("src.metals.recorder.resolve_kalshi_credentials", absent)
    path = _model_path(tmp_path)
    instance = MakerRecorder(_config(tmp_path), _model(), "maker-test",
                             MakerSettings(require_fast=False), tmp_path / "control.json",
                             hashlib.sha256(path.read_bytes()).hexdigest(),
                             require_credentials=False, baseline_model_path=path)
    yield instance
    for session in instance.sessions.values():
        session.close()
    instance.input_store.close()
    instance._finalize("completed")


def _inputs(recorder):
    return [json.loads(line) for path in recorder.input_store.paths
            for line in path.read_text(encoding="utf-8").splitlines()]


def test_control_write_is_atomic_versioned_and_has_explicit_expiry(tmp_path):
    path = tmp_path / "operator.json"
    first = write_control(path, version=1, volatility_multiplier=Decimal("1.2"),
                          paused=False, reason="manual test", ttl_seconds=10, now_ns=1000)
    assert first["expires_ts_ns"] == 10_000_001_000
    assert json.loads(path.read_text()) == first
    with pytest.raises(ValueError, match="strictly increase"):
        write_control(path, version=1, volatility_multiplier=Decimal("2"),
                      paused=True, reason="old revision", ttl_seconds=10, now_ns=1001)
    assert json.loads(path.read_text()) == first
    second = write_control(path, version=2, volatility_multiplier=Decimal("1"),
                           paused=True, reason="pause", ttl_seconds=20, now_ns=1002)
    assert second["paused"] is True
    assert json.loads(path.read_text()) == second
    assert not list(tmp_path.glob(".manifest-*.tmp"))


def test_invalid_missing_and_changed_controls_are_recorded_once_per_content(recorder):
    recorder.check_control()
    recorder.check_control()
    assert len(_inputs(recorder)) == 1
    assert _inputs(recorder)[0]["control"] is None
    recorder.control_path.write_text("invalid", encoding="utf-8")
    recorder.check_control()
    recorder.control_path.write_text("differently invalid", encoding="utf-8")
    recorder.check_control()
    assert len(_inputs(recorder)) == 3
    assert len({row["control_file_sha256"] for row in _inputs(recorder)}) == 3
    write_control(recorder.control_path.with_name("valid.json"), version=1,
                  volatility_multiplier=Decimal("1.2"), paused=False,
                  reason="prospective", ttl_seconds=60)
    recorder.control_path.write_bytes(recorder.control_path.with_name("valid.json").read_bytes())
    recorder.check_control()
    assert recorder.engine.control.volatility_multiplier == Decimal("1.2")
    recorder.control_path.unlink()
    recorder.check_control()
    assert recorder.engine.control_error.startswith("control_invalid")


def test_input_is_durable_before_engine_mutation_and_write_failure_stops(recorder, monkeypatch):
    seen = []
    monkeypatch.setattr(recorder.engine, "handle", lambda event: seen.append(_inputs(recorder)[-1]))
    event = recorder.synthetic("timer")
    assert seen[0]["ingestion_sequence"] == event["ingestion_sequence"]
    assert seen[0]["record_sha256"]
    def failed(*args, **kwargs):
        raise OSError("disk failure")
    monkeypatch.setattr(recorder.input_store, "write", failed)
    with pytest.raises(OSError, match="disk failure"):
        recorder.synthetic("timer")
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_book_and_gap_hooks_preserve_original_receive_order(recorder, monkeypatch):
    seen = []
    monkeypatch.setattr(recorder.engine, "handle", seen.append)
    stamp = recorder.sequencer.stamp("kalshi", "websocket")
    envelope = SourceEnvelope(
        source="kalshi", stream="orderbook_snapshot", ingestion_sequence=stamp.ingestion_sequence,
        receive_ts_ns=stamp.receive_ts_ns, local_monotonic_ts_ns=stamp.local_monotonic_ts_ns,
        raw={"type": "orderbook_snapshot", "msg": {
            "market_ticker": "KXSILVER15M-TEST", "yes_dollars_fp": [["0.20", "5"], ["0.19", "10"]],
            "no_dollars_fp": [["0.77", "8"]]}}, connection_id="test")
    recorder._market_metal["KXSILVER15M-TEST"] = "silver"
    await recorder._process_ws(envelope)
    assert seen[0]["ingestion_sequence"] == stamp.ingestion_sequence
    assert seen[0]["receive_ts_ns"] == stamp.receive_ts_ns
    assert seen[0]["yes_bids"] == {"0.20": "5"}
    assert len(recorder.books["KXSILVER15M-TEST"].yes_bids) == 2
    next_stamp = recorder.sequencer.stamp("kalshi", "websocket")
    gap = replace(envelope, ingestion_sequence=next_stamp.ingestion_sequence,
                  receive_ts_ns=next_stamp.receive_ts_ns,
                  local_monotonic_ts_ns=next_stamp.local_monotonic_ts_ns, gap_flag=True)
    await recorder._process_ws(gap)
    assert seen[-1]["kind"] == "gap"
    assert seen[-1]["ingestion_sequence"] == next_stamp.ingestion_sequence
    assert not recorder.books["KXSILVER15M-TEST"].valid


@pytest.mark.asyncio
async def test_planned_subscription_reconnect_invalidates_recorder_book_before_new_deltas(recorder, monkeypatch):
    from src.metals.book import FixedPointOrderBook
    book = FixedPointOrderBook("KXSILVER15M-TEST")
    book.apply_snapshot({"yes_dollars_fp": [["0.20", "1"]],
                         "no_dollars_fp": [["0.77", "1"]]}, "initial")
    recorder.books["KXSILVER15M-TEST"] = book
    seen, connections = [], []
    monkeypatch.setattr(recorder.engine, "handle", seen.append)
    async def consume(tickers, stop):
        connections.append(book.valid)
        if len(connections) == 1:
            recorder._ticker_change.set()
            await asyncio.Event().wait()
        recorder.stop()
    monkeypatch.setattr(recorder, "_consume_ws", consume)
    await asyncio.wait_for(recorder._websocket_supervisor(), timeout=1)
    assert connections == [True, False]
    assert [event["kind"] for event in seen] == ["gap"]
    book.apply_delta({"side": "yes", "price_dollars": "0.21", "delta_fp": "5"})
    assert not book.valid
    assert "0.21" not in book.yes_bids


def test_active_xag_uses_feed_publication_time_after_source_persistence(recorder, monkeypatch):
    calls = []
    stamp = recorder.sequencer.stamp("pyth", "price")
    row = {"endpoint_role": "active", "price": "67", "source_publish_ts_ns": 1000,
           "source_event_ts_ns": 2000, "receive_ts_ns": stamp.receive_ts_ns,
           "local_monotonic_ts_ns": stamp.local_monotonic_ts_ns,
           "ingestion_sequence": stamp.ingestion_sequence}
    def persisted(*args, **kwargs):
        calls.append("source_persisted")
        return stamp.ingestion_sequence, [row]
    monkeypatch.setattr(PythLazerCollector, "_write_source_message", persisted)
    monkeypatch.setattr(recorder.engine, "handle", lambda event: calls.append(event))
    collector = object.__new__(MakerXagCollector)
    collector.recorder = recorder
    collector._write_source_message()
    assert calls[0] == "source_persisted"
    assert calls[1]["source_ts_ns"] == 1000
    assert calls[1]["ingestion_sequence"] == stamp.ingestion_sequence
    row["endpoint_role"] = "standby"
    collector._write_source_message()
    assert len(_inputs(recorder)) == 1


def test_xag_processing_error_sets_fatal_signal_before_provider_reconnect(recorder, monkeypatch):
    row = {"endpoint_role": "active", "price": "67", "source_publish_ts_ns": 1000,
           "receive_ts_ns": 2000, "local_monotonic_ts_ns": 2000, "ingestion_sequence": 1}
    monkeypatch.setattr(PythLazerCollector, "_write_source_message", lambda *a, **k: (1, [row]))
    def failed(event):
        raise OSError("maker disk failure")
    monkeypatch.setattr(recorder, "deliver", failed)
    collector = object.__new__(MakerXagCollector)
    collector.recorder = recorder
    with pytest.raises(OSError):
        collector._write_source_message()
    assert recorder._fatal.is_set()
    assert isinstance(recorder._fatal_error, OSError)


def test_preflight_uses_local_checks_and_preserves_missing_provider_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "resolve_kalshi_credentials", lambda _: object())
    config, model_path = _config(tmp_path), _model_path(tmp_path)
    diagnostic = preflight(config, model_path, pyth_config=None,
                           settings=MakerSettings(require_fast=False))
    assert diagnostic["ready"] is True
    assert diagnostic["risk"]["max_contracts_per_order"] == "1"
    fast = preflight(config, model_path, pyth_config=None, settings=MakerSettings())
    assert fast["ready"] is False
    assert fast["checks"]["pyth_provider_gate"] is False
    assert "api_key" not in json.dumps(fast)


class FakeClient:
    def __init__(self):
        self.closed = False
        self.paths = []
    async def start(self):
        pass
    async def get(self, path):
        self.paths.append(path)
        return None
    async def close(self):
        self.closed = True


async def _idle():
    await asyncio.Event().wait()


def _mock_sources(recorder, monkeypatch):
    client = FakeClient()
    recorder._client, recorder._ws = client, object()
    monkeypatch.setattr(recorder, "_discovery_loop", _idle)
    monkeypatch.setattr(recorder, "_websocket_supervisor", _idle)
    monkeypatch.setattr(recorder, "_settlements", _idle)
    return client


@pytest.mark.asyncio
async def test_bounded_run_has_independent_watchdog_and_replays_exactly(recorder, monkeypatch, tmp_path):
    client = _mock_sources(recorder, monkeypatch)
    report = await recorder.run_maker(0.12)
    rows = _inputs(recorder)
    assert sum(row["kind"] == "timer" for row in rows) >= 2
    assert rows[-1]["kind"] == "stop"
    assert report["maker_input_counts"] == len(rows)
    assert report["status"] == "completed"
    assert client.closed and client.paths == ["/account/limits"]
    assert replay_report(Path(report["report_path"]), tmp_path / "replay")["replay_verified"]


@pytest.mark.asyncio
async def test_child_failure_stops_runner_and_closes_every_store(recorder, monkeypatch):
    client = _mock_sources(recorder, monkeypatch)
    async def failed():
        await asyncio.sleep(0)
        raise OSError("source processing failure")
    monkeypatch.setattr(recorder, "_discovery_loop", failed)
    with pytest.raises(OSError, match="source processing failure"):
        await asyncio.wait_for(recorder.run_maker(60), timeout=2)
    assert client.closed
    assert all(session._closed for session in recorder.sessions.values())
    reports = list((Path(recorder.config.storage.root) / "derived" / "maker_forward").glob("**/report.json"))
    assert json.loads(reports[0].read_text())["status"] == "failed"
    assert _inputs(recorder)[-1]["kind"] == "stop"


@pytest.mark.asyncio
async def test_provider_failure_during_normal_shutdown_is_not_silenced(recorder, monkeypatch):
    client = _mock_sources(recorder, monkeypatch)
    recorder.engine.settings = replace(recorder.engine.settings, require_fast=True)
    class FailedPythFinalizer:
        run_id = "pyth-test"
        def __init__(self, *args, **kwargs):
            pass
        async def run(self, *, stop_event):
            await stop_event.wait()
            raise OSError("provider close failed")
        def stop(self):
            pass
    monkeypatch.setattr(module, "MakerXagCollector", FailedPythFinalizer)
    monkeypatch.setattr(module, "_xag_product", lambda _: "test")
    with pytest.raises(OSError, match="provider close failed"):
        await recorder.run_maker(0.03, pyth_config=recorder.config)
    assert client.closed
    reports = list((Path(recorder.config.storage.root) / "derived" / "maker_forward").glob("**/report.json"))
    assert json.loads(reports[0].read_text())["status"] == "failed"


@pytest.fixture
def trial_permit(tmp_path):
    now = time.time_ns()
    return materialize_trial_research(tmp_path / "trial-research.json", {
        "schema_version": "maker-trial-research/v1", "accepted_by": "project_owner",
        "acceptance_text": "Test-only operator acceptance for the scoped maker trial.",
        "accepted_at_ns": now - 1_000_000_000,
        "collection_not_after_ns": now + 3600_000_000_000,
        "retention_review_at_ns": now + 86400_000_000_000,
        "provider": "pyth", "feed_id": 345, "symbol": "Metal.XAG/USD",
        "purpose": "frozen_model_maker_shadow", "no_orders": True,
        "research_only": True, "no_redistribution": True,
        "provider_permission_confirmed": False, "provider_trial_expires_at_ns": None,
        "sources": list(PUBLIC_SOURCES),
    })


def _trial_config(tmp_path):
    config = load_metals_config("configs/pyth_xag_trial.toml")
    return replace(config, storage=replace(config.storage, root=str(tmp_path / "data")))


def test_explicit_trial_route_does_not_claim_formal_provider_approval(tmp_path, monkeypatch, trial_permit):
    monkeypatch.setattr(module, "resolve_kalshi_credentials", lambda _: object())
    monkeypatch.setenv("PYTH_API_KEY", "test-key-never-printed")
    config, pyth_config, model = _config(tmp_path), _trial_config(tmp_path), _model_path(tmp_path)
    formal = preflight(config, model, pyth_config=pyth_config, settings=MakerSettings())
    assert formal["ready"] is False
    assert formal["checks"]["pyth_provider_gate"] is False
    trial = preflight(config, model, pyth_config=pyth_config, settings=MakerSettings(),
                      trial_permit=trial_permit)
    assert trial["ready"] is True
    assert trial["checks"]["pyth_provider_gate"] is False
    assert trial["checks"]["trial_research_gate"] is True
    assert trial["research_provenance"]["provider_permission_confirmed"] is False
    assert "test-key-never-printed" not in json.dumps(trial)
    with pytest.raises(RuntimeError):
        PythLazerCollector(pyth_config, "silver-xag-proxy")
    collector = MakerXagCollector(pyth_config, "silver-xag-proxy", recorder=None,
                                 trial_permit=trial_permit)
    fields, paths = collector._authorization_manifest()
    assert "provider_approval_id" not in fields
    assert fields["trial_research_id"] == trial_permit.sha256
    assert fields["provider_permission_confirmed"] is False
    assert paths == (trial_permit.path,)


@pytest.mark.parametrize("problem", ["wrong_feed", "live_orders", "expired"])
def test_trial_route_rejects_wrong_scope_or_expiry(tmp_path, monkeypatch, trial_permit, problem):
    monkeypatch.setattr(module, "resolve_kalshi_credentials", lambda _: object())
    monkeypatch.setenv("PYTH_API_KEY", "test-key")
    pyth_config = _trial_config(tmp_path)
    if problem == "wrong_feed":
        product = replace(pyth_config.products["silver-xag-proxy"], pyth_lazer_id=346)
        pyth_config = replace(pyth_config, products={"silver-xag-proxy": product})
    elif problem == "live_orders":
        pyth_config = replace(pyth_config, shadow=replace(pyth_config.shadow, allow_live_orders=True))
    else:
        payload = trial_permit.public_dict()
        payload.pop("record_sha256")
        payload.update(accepted_at_ns=time.time_ns() - 10_000_000_000,
                       collection_not_after_ns=time.time_ns() - 1_000_000_000)
        trial_permit = materialize_trial_research(tmp_path / "expired.json", payload)
    result = preflight(_config(tmp_path), _model_path(tmp_path), pyth_config=pyth_config,
                       settings=MakerSettings(), trial_permit=trial_permit)
    assert result["ready"] is False
    assert result["checks"]["trial_research_gate"] is False


def test_trial_permit_cannot_be_attached_to_exact_only_diagnostic(tmp_path, trial_permit):
    with pytest.raises(ValueError, match="only valid for the fast XAG"):
        preflight(_config(tmp_path), _model_path(tmp_path), pyth_config=None,
                  settings=MakerSettings(require_fast=False), trial_permit=trial_permit)


def _new_trial_recorder(tmp_path, monkeypatch, trial_permit):
    def absent(_):
        raise ValueError("test has no credentials")
    monkeypatch.setattr("src.metals.recorder.resolve_kalshi_credentials", absent)
    path = _model_path(tmp_path)
    return MakerRecorder(_config(tmp_path), _model(), "maker-test", MakerSettings(),
                         tmp_path / "control.json", hashlib.sha256(path.read_bytes()).hexdigest(),
                         require_credentials=False, baseline_model_path=path,
                         trial_permit=trial_permit)


class IdlePyth:
    run_id = "test-trial-source"
    def __init__(self, *args, **kwargs):
        pass
    async def run(self, *, stop_event):
        await stop_event.wait()
        return {}
    def stop(self):
        pass


@pytest.mark.asyncio
async def test_trial_provenance_is_retained_in_run_and_arm_manifests(tmp_path, monkeypatch, trial_permit):
    recorder = _new_trial_recorder(tmp_path, monkeypatch, trial_permit)
    _mock_sources(recorder, monkeypatch)
    monkeypatch.setattr(module, "MakerXagCollector", IdlePyth)
    report = await recorder.run_maker(0.03, pyth_config=_trial_config(tmp_path))
    assert report["trial_research"] == trial_permit.public_dict()
    root = Path(recorder.config.storage.root)
    collector = json.loads((root / "manifests" / f"run={recorder.run_id}" / "manifest.json").read_text())
    assert collector["research_provenance"]["provider_permission_confirmed"] is False
    assert collector["research_provenance"]["trial_research_id"] == trial_permit.sha256
    for session in recorder.sessions.values():
        manifest = json.loads((root / "manifests" / f"run={session.run_id}" / "manifest.json").read_text())
        authority = manifest["config"]["forward_shadow"]["research_provenance"]
        assert authority["trial_research_permit"]["record_sha256"] == trial_permit.sha256
        assert authority["provider_permission_confirmed"] is False


@pytest.mark.asyncio
async def test_runtime_trial_expiry_fail_stops_and_cleans_up(tmp_path, monkeypatch, trial_permit):
    recorder = _new_trial_recorder(tmp_path, monkeypatch, trial_permit)
    client = _mock_sources(recorder, monkeypatch)
    monkeypatch.setattr(module, "MakerXagCollector", IdlePyth)
    expired = asyncio.Event()
    original = TrialResearchPermit.require_current
    def current(self, now_ns=None):
        if expired.is_set():
            raise ValueError("test local trial deadline expired")
        return original(self, now_ns)
    monkeypatch.setattr(TrialResearchPermit, "require_current", current)
    async def expire():
        await asyncio.sleep(0.01)
        expired.set()
    expiry_task = asyncio.create_task(expire())
    with pytest.raises(ValueError, match="deadline expired"):
        await asyncio.wait_for(recorder.run_maker(60, _trial_config(tmp_path)), timeout=2)
    await expiry_task
    assert client.closed
    assert all(session._closed for session in recorder.sessions.values())
    assert _inputs(recorder)[-1]["kind"] == "stop"
    reports = list((Path(recorder.config.storage.root) / "derived" / "maker_forward").glob("**/report.json"))
    assert json.loads(reports[0].read_text())["status"] == "failed"


@pytest.mark.asyncio
async def test_trial_duration_cap_is_checked_before_network(tmp_path, monkeypatch, trial_permit):
    recorder = _new_trial_recorder(tmp_path, monkeypatch, trial_permit)
    client = _mock_sources(recorder, monkeypatch)
    with pytest.raises(ValueError, match="900"):
        await recorder.run_maker(901, _trial_config(tmp_path))
    assert client.paths == [] and client.closed


@pytest.mark.asyncio
async def test_expired_trial_blocks_new_source_writes_and_connections(recorder, monkeypatch, trial_permit):
    collector = object.__new__(MakerXagCollector)
    collector.recorder, collector.trial_permit = recorder, trial_permit
    def expired(*args, **kwargs):
        raise ValueError("trial expired")
    monkeypatch.setattr(TrialResearchPermit, "require_current", expired)
    def forbidden(*args, **kwargs):
        pytest.fail("expired trial reached source operation")
    monkeypatch.setattr(PythLazerCollector, "_write_source_message", forbidden)
    monkeypatch.setattr(PythLazerCollector, "_write_unparsed_source_message", forbidden)
    monkeypatch.setattr(PythLazerCollector, "_connection_once", forbidden)
    with pytest.raises(ValueError, match="trial expired"):
        collector._write_source_message()
    with pytest.raises(ValueError, match="trial expired"):
        collector._write_unparsed_source_message()
    with pytest.raises(ValueError, match="trial expired"):
        await collector._connection_once("unused", "unused", "unused")
    assert recorder._fatal.is_set()


@pytest.mark.asyncio
async def test_preflight_trial_collector_cannot_run_without_maker(tmp_path, trial_permit):
    collector = MakerXagCollector(_trial_config(tmp_path), "silver-xag-proxy",
                                 recorder=None, trial_permit=trial_permit)
    with pytest.raises(ValueError, match="attached maker recorder"):
        await collector.run(duration_seconds=60)
    assert collector._raw_store is None


@pytest.mark.asyncio
async def test_trial_source_has_hard_duration_bound_and_typed_source_manifest(tmp_path, monkeypatch, trial_permit):
    recorder = _new_trial_recorder(tmp_path, monkeypatch, trial_permit)
    recorder.trial_duration_seconds = 60
    collector = MakerXagCollector(_trial_config(tmp_path), "silver-xag-proxy",
                                 recorder=recorder, trial_permit=trial_permit)
    monkeypatch.setenv("PYTH_API_KEY", "test-value-not-emitted")
    durations = []
    original_run = PythLazerCollector.run
    async def bounded(self, *, duration_seconds, stop_event):
        durations.append(duration_seconds)
        return await original_run(self, duration_seconds=duration_seconds, stop_event=stop_event)
    async def endpoint(endpoint, api_key):
        await collector._stop.wait()
    monkeypatch.setattr(PythLazerCollector, "run", bounded)
    monkeypatch.setattr(collector, "_endpoint_loop", endpoint)
    async def stop():
        await asyncio.sleep(0.01)
        recorder.stop()
    stopper = asyncio.create_task(stop())
    try:
        await collector.run(stop_event=recorder._stop)
        await stopper
        assert durations == [63]
        manifest_path = Path(collector.config.storage.root) / "manifests" / f"run={collector.run_id}" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        facts = manifest["config"]["collector"]
        assert facts["provider_permission_confirmed"] is False
        assert facts["trial_research_permit"]["record_sha256"] == trial_permit.sha256
        assert "provider_approval_id" not in facts
        assert str(trial_permit.path) in manifest["output_paths"]
    finally:
        stopper.cancel()
        await asyncio.gather(stopper, return_exceptions=True)
        for session in recorder.sessions.values():
            session.close()
        recorder.input_store.close()
        recorder._finalize("completed")
