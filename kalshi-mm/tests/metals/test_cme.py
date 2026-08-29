import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.metals.cme import DatabentoAccessError, DatabentoMetalsRecorder
from src.metals.config import load_metals_config


class FakeCollector:
    def __init__(
        self,
        records=(),
        error=None,
        close_error=None,
        reconnect_gap=None,
        closes_unexpectedly=False,
    ):
        self.records = list(records)
        self.error = error
        self.close_error = close_error
        self.reconnect_gap = reconnect_gap
        self.closes_unexpectedly = closes_unexpectedly
        self.stopped = False
        self.waited = False
        self.stream = None
        self._closed = None

    def add_stream(self, stream, exception_callback=None):
        self.stream = stream
        stream.write(b"fake-dbn")

    def start_callbacks(
        self,
        callback,
        exception_callback,
        reconnect_callback=None,
        defer_normalization=False,
    ):
        self._closed = asyncio.Event()
        for record in self.records:
            callback(record)
        if self.reconnect_gap is not None and reconnect_callback is not None:
            reconnect_callback(self.reconnect_gap)
        if self.error is not None:
            exception_callback(self.error)
        if self.closes_unexpectedly:
            self._closed.set()

    def stop(self):
        self.stopped = True
        self._closed.set()

    async def wait_for_close(self):
        self.waited = True
        await self._closed.wait()
        if self.close_error is not None:
            raise self.close_error


def _config(tmp_path: Path, enabled=True):
    config = load_metals_config("configs/metals.example.toml")
    return replace(
        config,
        storage=replace(config.storage, root=str(tmp_path / "metals")),
        cme=replace(config.cme, enabled=enabled),
    )


def _record(schema="mbo", sequence=1):
    return {
        "source": "databento",
        "dataset": "GLBX.MDP3",
        "schema": schema,
        "record_type": "FakeMsg",
        "instrument_id": 123,
        "ts_event": 1_000,
        "ts_recv": 1_020,
        "ts_in_delta": 5,
        "ts_out": 1_030,
        "sequence": sequence,
        "local_receive_ts_ns": 2_000 + sequence,
        "local_monotonic_ts_ns": 3_000 + sequence,
        "price": "72.100000000",
        "price_raw": 72_100_000_000,
        "raw_record": {"price": 72_100_000_000},
    }


@pytest.mark.asyncio
async def test_recorder_persists_raw_normalized_and_completed_manifest(tmp_path):
    collector = FakeCollector([_record("mbo", 1), _record("trades", 2)])
    recorder = DatabentoMetalsRecorder(
        _config(tmp_path), "silver", collector=collector, max_records=2
    )

    counts = await recorder.run()

    assert counts == {"records": 2, "mbo": 1, "trades": 1}
    assert collector.stopped and collector.waited
    raw_paths = list(
        (tmp_path / "metals" / "raw" / "databento_extracted").glob("**/*.jsonl")
    )
    native_paths = list(
        (tmp_path / "metals" / "raw" / "databento_native").glob("**/*.dbn")
    )
    normalized_paths = list(
        (tmp_path / "metals" / "normalized").glob("**/*.jsonl")
    )
    assert len(raw_paths) == 2
    assert len(native_paths) == 1
    assert native_paths[0].read_bytes() == b"fake-dbn"
    assert len(normalized_paths) == 2
    raw_record = json.loads(raw_paths[0].read_text(encoding="utf-8").splitlines()[0])
    assert raw_record["source_publish_ts_ns"] == 1_015
    assert raw_record["raw_record"]["price"] == 72_100_000_000
    manifest = json.loads(
        (
            tmp_path
            / "metals"
            / "manifests"
            / "run={}".format(recorder.run_id)
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["counts"]["records"] == 2
    assert manifest["errors"] == []


@pytest.mark.asyncio
async def test_recorder_marks_manifest_failed_on_callback_error(tmp_path):
    collector = FakeCollector(error=RuntimeError("feed disconnected"))
    recorder = DatabentoMetalsRecorder(
        _config(tmp_path), "silver", collector=collector
    )

    with pytest.raises(DatabentoAccessError):
        await recorder.run(duration_seconds=0.1)

    manifest_path = (
        tmp_path
        / "metals"
        / "manifests"
        / "run={}".format(recorder.run_id)
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["errors"]
    assert "feed disconnected" in manifest["errors"][0]["message"]


@pytest.mark.asyncio
async def test_recorder_enforces_exact_record_cap_for_callback_batch(tmp_path):
    collector = FakeCollector([_record("mbo", index) for index in range(1, 5)])
    recorder = DatabentoMetalsRecorder(
        _config(tmp_path), "silver", collector=collector, max_records=2
    )

    counts = await recorder.run()

    assert counts == {"records": 2, "mbo": 2}


@pytest.mark.asyncio
async def test_recorder_surfaces_close_failure_and_marks_manifest_failed(tmp_path):
    collector = FakeCollector(
        [_record("mbo", 1)], close_error=RuntimeError("close failed")
    )
    recorder = DatabentoMetalsRecorder(
        _config(tmp_path), "silver", collector=collector, max_records=1
    )

    with pytest.raises(RuntimeError, match="close failed"):
        await recorder.run()

    manifest_path = (
        tmp_path
        / "metals"
        / "manifests"
        / "run={}".format(recorder.run_id)
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "close failed" in manifest["errors"][0]["message"]


@pytest.mark.asyncio
async def test_recorder_records_reconnect_gap_and_classification(tmp_path):
    reconnect = {
        "schema": "reconnect",
        "local_receive_ts_ns": 2_500,
        "start_ts_ns": 1_000,
        "end_ts_ns": 2_000,
    }
    collector = FakeCollector(reconnect_gap=reconnect)
    recorder = DatabentoMetalsRecorder(
        _config(tmp_path), "silver", collector=collector, max_records=1
    )

    counts = await recorder.run()

    assert counts == {"records": 1, "reconnect": 1}
    report_path = next(
        (tmp_path / "metals" / "derived" / "daily_quality").glob("**/report.json")
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["cme_reconnects"] == 1
    assert report["missing_intervals"][0]["reason"] == "databento_reconnect"


@pytest.mark.asyncio
async def test_recorder_fails_if_feed_closes_without_local_stop(tmp_path):
    collector = FakeCollector(closes_unexpectedly=True)
    recorder = DatabentoMetalsRecorder(
        _config(tmp_path), "silver", collector=collector
    )

    with pytest.raises(DatabentoAccessError, match="closed unexpectedly"):
        await recorder.run()


def test_recorder_requires_explicit_cme_enablement(tmp_path):
    with pytest.raises(DatabentoAccessError):
        DatabentoMetalsRecorder(
            _config(tmp_path, enabled=False), "silver", collector=FakeCollector()
        )
