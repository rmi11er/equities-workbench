import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.metals import clock as clock_module
from src.metals.clock import (
    ChronyProbeResult,
    ClockTelemetryCollector,
    ClockTelemetryError,
    parse_chrony_tracking,
    preflight_clock,
)
from src.metals.config import load_metals_config
from src.metals.ingestion import GlobalReceiveSequencer

TRACKING_FAST = """Reference ID    : C0A80101 (192.168.1.1)
Stratum         : 3
Ref time (UTC)  : Sun Aug 30 12:00:00 2026
System time     : 0.000000250 seconds fast of NTP time
Last offset     : +0.000000100 seconds
RMS offset      : 0.000000200 seconds
Leap status     : Normal
"""


def _config(tmp_path: Path):
    config = load_metals_config("configs/metals.example.toml")
    return replace(
        config,
        storage=replace(config.storage, root=str(tmp_path / "metals")),
        clock=replace(
            config.clock,
            enabled=True,
            sample_interval_seconds=0.001,
        ),
    )


def test_chrony_parser_preserves_signed_local_minus_reference_offset():
    fast = parse_chrony_tracking(TRACKING_FAST)
    slow = parse_chrony_tracking(
        TRACKING_FAST.replace("0.000000250 seconds fast", "0.000000125 seconds slow")
    )

    assert fast.clock_offset_ns == 250
    assert fast.synchronized is True
    assert fast.stratum == 3
    assert slow.clock_offset_ns == -125


def test_chrony_parser_marks_valid_unsynchronized_state_without_guessing():
    result = parse_chrony_tracking(
        TRACKING_FAST.replace("Stratum         : 3", "Stratum         : 0").replace(
            "Leap status     : Normal", "Leap status     : Not synchronised"
        )
    )

    assert result.synchronized is False
    assert result.stratum == 0


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "Stratum: 3\nLeap status: Normal\n",
        TRACKING_FAST + "Stratum: 4\n",
        TRACKING_FAST.replace("0.000000250", "nan"),
    ],
)
def test_chrony_parser_rejects_missing_duplicate_or_nonfinite_fields(raw):
    with pytest.raises(ClockTelemetryError):
        parse_chrony_tracking(raw)


@pytest.mark.asyncio
async def test_clock_collector_persists_raw_and_normalized_samples(tmp_path):
    sequencer = GlobalReceiveSequencer()
    probe = ChronyProbeResult(
        raw_text=TRACKING_FAST,
        clock_offset_ns=250,
        synchronized=True,
        stratum=3,
        leap_status="Normal",
    )
    collector = ClockTelemetryCollector(
        _config(tmp_path),
        "silver",
        sequencer=sequencer,
        probe=lambda: probe,
    )

    counts = await collector.run(duration_seconds=0.005)

    assert counts["clock_samples"] >= 1
    manifest_path = (
        tmp_path / "metals" / "manifests" / f"run={collector.run_id}" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["config"]["collector"]["read_only"] is True
    assert manifest["config"]["collector"]["changes_system_clock"] is False
    normalized_path = next(
        (tmp_path / "metals" / "normalized" / "clock_telemetry").glob("**/*.jsonl")
    )
    row = json.loads(normalized_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["clock_offset_ns"] == 250
    assert row["clock_synchronized"] is True
    assert row["offset_sign_semantics"] == "local_clock_minus_ntp_reference"
    assert (
        row["probe_output_sha256"]
        == hashlib.sha256(TRACKING_FAST.encode("utf-8")).hexdigest()
    )
    assert sequencer.audit(("clock",))["passed"] is True


@pytest.mark.asyncio
async def test_clock_probe_failure_fails_manifest_without_secret_output(tmp_path):
    def fail():
        raise ClockTelemetryError("synthetic failure")

    collector = ClockTelemetryCollector(
        _config(tmp_path),
        "silver",
        probe=fail,
    )

    with pytest.raises(ClockTelemetryError, match="synthetic failure"):
        await collector.run(duration_seconds=0.01)

    manifest_path = (
        tmp_path / "metals" / "manifests" / f"run={collector.run_id}" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["counts"]["clock_probe_errors"] == 1
    assert "synthetic failure" in manifest["errors"][0]["message"]


@pytest.mark.asyncio
async def test_wall_monotonic_elapsed_discontinuity_marks_sample_unhealthy(
    monkeypatch, tmp_path
):
    wall = iter((1_000_000_000, 1_002_000_000))
    monotonic = iter((500_000_000, 500_000_000))
    monkeypatch.setattr(clock_module.time, "time_ns", lambda: next(wall))
    monkeypatch.setattr(clock_module.time, "monotonic_ns", lambda: next(monotonic))
    probe = ChronyProbeResult(
        raw_text=TRACKING_FAST,
        clock_offset_ns=250,
        synchronized=True,
        stratum=3,
        leap_status="Normal",
    )
    collector = ClockTelemetryCollector(
        _config(tmp_path),
        "silver",
        probe=lambda: probe,
    )

    await collector._sample()
    collector._finalize("completed")

    normalized_path = next(
        (tmp_path / "metals" / "normalized" / "clock_telemetry").glob("**/*.jsonl")
    )
    row = json.loads(normalized_path.read_text(encoding="utf-8"))
    assert row["wall_monotonic_elapsed_consistent"] is False
    assert row["clock_synchronized"] is False
    assert collector.counts["clock_elapsed_discontinuities"] == 1
    assert collector.quality.counters["clock_elapsed_discontinuities"] == 1
    manifest_path = (
        tmp_path / "metals" / "manifests" / f"run={collector.run_id}" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["clock_elapsed_discontinuities"] == 1


def test_clock_preflight_is_sanitized(monkeypatch, tmp_path):
    monkeypatch.setattr(
        clock_module,
        "probe_chrony",
        lambda config: ChronyProbeResult(
            raw_text="private-reference-details",
            clock_offset_ns=-125,
            synchronized=True,
            stratum=2,
            leap_status="Normal",
        ),
    )

    report = preflight_clock(_config(tmp_path))

    assert report == {
        "ok": True,
        "provider": "chrony",
        "status": "synchronized",
        "clock_offset_abs_ns": 125,
        "stratum": 2,
        "leap_status": "Normal",
    }
    assert "private-reference-details" not in json.dumps(report)
