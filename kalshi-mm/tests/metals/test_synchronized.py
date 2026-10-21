import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.metals.config import load_metals_config
from src.metals.daily_validation import DailyValidationError, freeze_daily_validation
from src.metals.ingestion import GlobalReceiveSequencer
from src.metals.provider_approval import ProviderApprovalError
from src.metals.quality import QualityTracker
from src.metals.synchronized import (
    SynchronizedCollectionError,
    SynchronizedMetalsRecorder,
)
from tests.metals.provider_approval_fixtures import (
    materialize_approved_provider_path,
)


def _config(tmp_path: Path, *, enabled=True):
    config = load_metals_config("configs/metals.example.toml")
    storage_root = tmp_path / "metals"
    return replace(
        config,
        storage=replace(config.storage, root=str(storage_root)),
        pyth=replace(config.pyth, enabled=enabled),
        cme=replace(config.cme, enabled=enabled),
        clock=replace(config.clock, enabled=enabled),
        governance=replace(
            config.governance,
            provider_approval_path=(
                materialize_approved_provider_path(storage_root) if enabled else ""
            ),
        ),
    )


class _FakeSource:
    def __init__(self, source, sequencer, *, failure=None, emit_stamp=True):
        self.source = source
        self.sequencer = sequencer
        self.failure = failure
        self.emit_stamp = emit_stamp
        self.quality = QualityTracker()
        self.run_id = "{}-child-run".format(source)
        self.stamps = []
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()
        self.stop_calls = 0

    async def run(self, **kwargs):
        self.started.set()
        if self.emit_stamp:
            self.stamps.append(self.sequencer.stamp(self.source, "fake_callback"))
        if self.failure is not None:
            # Give sibling tasks one scheduling turn to start before this source
            # fails, mirroring an injected transport failure after startup.
            await asyncio.sleep(0)
            raise self.failure
        await self.stopped.wait()
        return {"records": len(self.stamps)}

    def stop(self):
        self.stop_calls += 1
        self.stopped.set()


def _fakes(sequencer, *, pyth_failure=None):
    return {
        "kalshi": _FakeSource("kalshi", sequencer),
        "pyth_lazer": _FakeSource("pyth_lazer", sequencer, failure=pyth_failure),
        "databento": _FakeSource("databento", sequencer),
        "clock": _FakeSource("clock", sequencer),
    }


def _parent_manifest(tmp_path, run_id):
    path = tmp_path / "metals" / "manifests" / "run={}".format(run_id) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_complete_panel_shares_one_sequence_and_links_child_runs(tmp_path):
    sequencer = GlobalReceiveSequencer()
    children = _fakes(sequencer)
    recorder = SynchronizedMetalsRecorder(
        _config(tmp_path),
        "silver",
        sequencer=sequencer,
        kalshi_recorder=children["kalshi"],
        pyth_collector=children["pyth_lazer"],
        databento_recorder=children["databento"],
        clock_collector=children["clock"],
    )

    result = await recorder.run(duration_seconds=0.01)

    sequences = sorted(
        stamp.ingestion_sequence
        for child in children.values()
        for stamp in child.stamps
    )
    assert sequences == [1, 2, 3, 4]
    assert result["global_receive_sequence_last"] == 4
    assert all(child.stop_calls == 1 for child in children.values())
    manifest = _parent_manifest(tmp_path, recorder.run_id)
    assert manifest["status"] == "completed"
    assert manifest["config"]["collector"]["read_only"] is True
    assert manifest["config"]["collector"]["required_telemetry"] == ["clock"]
    assert set(manifest["config"]["collector"]["provider_approval_ids"]) == {
        "pyth",
        "databento",
    }
    approval_path = Path(recorder.config.governance.provider_approval_path)
    assert {str(approval_path), str(approval_path.parent / "manifest.json")}.issubset(
        manifest["output_paths"]
    )
    assert set(manifest["counts"]["child_runs"]) == set(children)
    assert manifest["counts"]["child_runs"]["kalshi"] == {
        "counts": {"records": 1},
        "run_id": "kalshi-child-run",
        "status": "completed",
    }
    assert manifest["counts"]["global_receive_sequence_audit"]["passed"] is True
    quality_path = Path(result["quality_report_path"])
    assert str(quality_path) in manifest["output_paths"]
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert quality["causal_ordering_checked"] is True
    assert quality["timestamp_ordering_violations"] == 0
    assert quality["global_receive_sequence_audit"]["passed"] is True
    assert quality["data_gates"]["passed"] is False
    assert set(quality["data_gates"]["failed_gates"]) >= {
        "settlement_agreement",
        "kalshi_book_coverage",
        "pyth_coverage",
        "cme_coverage",
        "rules_unchanged",
    }

    parent_manifest_path = (
        tmp_path
        / "metals"
        / "manifests"
        / "run={}".format(recorder.run_id)
        / "manifest.json"
    )
    with pytest.raises(DailyValidationError, match="verified daily-gate measurement"):
        freeze_daily_validation(
            tmp_path / "metals",
            quality["date"],
            quality_path,
            [parent_manifest_path],
        )


@pytest.mark.asyncio
async def test_one_source_failure_stops_all_and_fails_parent_manifest(tmp_path):
    sequencer = GlobalReceiveSequencer()
    children = _fakes(sequencer, pyth_failure=RuntimeError("injected disconnect"))
    recorder = SynchronizedMetalsRecorder(
        _config(tmp_path),
        "silver",
        sequencer=sequencer,
        kalshi_recorder=children["kalshi"],
        pyth_collector=children["pyth_lazer"],
        databento_recorder=children["databento"],
        clock_collector=children["clock"],
    )

    with pytest.raises(SynchronizedCollectionError, match="pyth_lazer stopped"):
        await recorder.run(duration_seconds=1.0)

    assert all(child.stopped.is_set() for child in children.values())
    manifest = _parent_manifest(tmp_path, recorder.run_id)
    assert manifest["status"] == "failed"
    assert manifest["counts"]["child_runs"]["pyth_lazer"]["status"] == "failed"
    assert "pyth_lazer stopped" in manifest["errors"][0]["message"]
    quality_paths = [
        Path(path) for path in manifest["output_paths"] if path.endswith("report.json")
    ]
    assert len(quality_paths) == 1
    assert quality_paths[0].is_file()


@pytest.mark.asyncio
async def test_missing_required_component_stamp_fails_parent_after_boundary(tmp_path):
    sequencer = GlobalReceiveSequencer()
    children = _fakes(sequencer)
    children["clock"] = _FakeSource("clock", sequencer, emit_stamp=False)
    recorder = SynchronizedMetalsRecorder(
        _config(tmp_path),
        "silver",
        sequencer=sequencer,
        kalshi_recorder=children["kalshi"],
        pyth_collector=children["pyth_lazer"],
        databento_recorder=children["databento"],
        clock_collector=children["clock"],
    )

    with pytest.raises(SynchronizedCollectionError, match="receive-order audit"):
        await recorder.run(duration_seconds=0.01)

    manifest = _parent_manifest(tmp_path, recorder.run_id)
    assert manifest["status"] == "failed"
    sequence = manifest["counts"]["global_receive_sequence_audit"]
    assert sequence["passed"] is False
    assert sequence["missing_sources"] == ["clock"]


def test_complete_panel_fails_closed_when_a_required_source_is_disabled(tmp_path):
    with pytest.raises(SynchronizedCollectionError, match="clock telemetry"):
        SynchronizedMetalsRecorder(_config(tmp_path, enabled=False), "silver")


def test_complete_panel_fails_before_children_without_provider_approval(tmp_path):
    config = _config(tmp_path)
    config = replace(
        config,
        governance=replace(config.governance, provider_approval_path=""),
    )

    with pytest.raises(ProviderApprovalError, match="approval path"):
        SynchronizedMetalsRecorder(config, "silver")

    assert not (tmp_path / "metals" / "manifests").exists()


def test_injected_child_must_use_the_exact_shared_sequencer(tmp_path):
    sequencer = GlobalReceiveSequencer()
    children = _fakes(sequencer)
    children["kalshi"].sequencer = GlobalReceiveSequencer()

    with pytest.raises(SynchronizedCollectionError, match="global sequencer"):
        SynchronizedMetalsRecorder(
            _config(tmp_path),
            "silver",
            sequencer=sequencer,
            kalshi_recorder=children["kalshi"],
            pyth_collector=children["pyth_lazer"],
            databento_recorder=children["databento"],
            clock_collector=children["clock"],
        )


@pytest.mark.asyncio
async def test_parent_quality_merges_every_child_tracker_component(tmp_path):
    sequencer = GlobalReceiveSequencer()
    children = _fakes(sequencer)
    kalshi = children["kalshi"].quality
    kalshi.record("raw_records", "TICKER", count=2)
    kalshi.record_source_coverage("kalshi", valid_ns=90, total_ns=100)
    kalshi.record_clock_offset(-25)
    kalshi.record_missing_interval("kalshi", 10, 20, "injected_gap", "TICKER")
    kalshi.metadata_changes.append({"ticker": "TICKER", "version": 2})
    children["pyth_lazer"].quality.record_source_coverage(
        "pyth", valid_ns=80, total_ns=100
    )
    children["clock"].quality.record_clock_offset(40)
    cme = children["databento"].quality
    cme.record_source_coverage("cme", valid_ns=70, total_ns=100)
    cme.record_exclusion("TICKER", "injected_exclusion", 30)
    cme.record("trades", "TICKER", count=3)
    recorder = SynchronizedMetalsRecorder(
        _config(tmp_path),
        "silver",
        sequencer=sequencer,
        kalshi_recorder=children["kalshi"],
        pyth_collector=children["pyth_lazer"],
        databento_recorder=children["databento"],
        clock_collector=children["clock"],
    )

    result = await recorder.run(duration_seconds=0.01)
    report = json.loads(Path(result["quality_report_path"]).read_text(encoding="utf-8"))

    assert report["raw_records"] == 2
    assert report["trades"] == 3
    assert report["coverage"]["kalshi_valid_ns"] == 90
    assert report["coverage"]["pyth_valid_ns"] == 80
    assert report["coverage"]["cme_valid_ns"] == 70
    assert report["clock_samples"] == 2
    assert report["clock_offset_abs_max_ns"] == 40
    assert report["missing_intervals"] == [
        {
            "end_ts_ns": 20,
            "reason": "injected_gap",
            "source": "kalshi",
            "start_ts_ns": 10,
            "ticker": "TICKER",
        }
    ]
    assert report["metadata_changes"] == [{"ticker": "TICKER", "version": 2}]
    assert report["exclusions"] == [
        {
            "reason": "injected_exclusion",
            "start_ts_ns": 30,
            "ticker": "TICKER",
        }
    ]
    assert report["by_market"]["TICKER"] == {"raw_records": 2, "trades": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [float("nan"), float("inf"), float("-inf")])
async def test_non_finite_duration_is_rejected_before_collection(tmp_path, duration):
    sequencer = GlobalReceiveSequencer()
    children = _fakes(sequencer)
    recorder = SynchronizedMetalsRecorder(
        _config(tmp_path),
        "silver",
        sequencer=sequencer,
        kalshi_recorder=children["kalshi"],
        pyth_collector=children["pyth_lazer"],
        databento_recorder=children["databento"],
        clock_collector=children["clock"],
    )

    with pytest.raises(ValueError, match="positive and finite"):
        await recorder.run(duration_seconds=duration)

    assert all(not child.started.is_set() for child in children.values())
