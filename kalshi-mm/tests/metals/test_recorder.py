import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.metals.config import load_metals_config
from src.metals.ingestion import GlobalReceiveSequencer
from src.metals.recorder import MetalsRecorder, _completed_trade_markets
from src.metals.types import SourceEnvelope


def _config(tmp_path: Path):
    config = load_metals_config("configs/metals.example.toml")
    return replace(
        config,
        storage=replace(config.storage, root=str(tmp_path / "metals")),
        kalshi=replace(
            config.kalshi,
            credentials_config=str(tmp_path / "missing-credentials.toml"),
            api_key_id_env="TEST_MISSING_KALSHI_KEY_ID",
            private_key_path_env="TEST_MISSING_KALSHI_KEY_PATH",
        ),
    )


def _manifest(tmp_path: Path, run_id: str):
    path = (
        tmp_path
        / "metals"
        / "manifests"
        / "run={}".format(run_id)
        / "manifest.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_manual_recorder_finalize_is_idempotent_and_records_outputs(tmp_path):
    recorder = MetalsRecorder(_config(tmp_path), require_credentials=False)
    recorder.quality.record("raw_records", count=3)

    recorder._finalize("completed", counts={"markets": 2})
    recorder._finalize("failed", error=RuntimeError("must be ignored"))

    manifest = _manifest(tmp_path, recorder.run_id)
    assert manifest["status"] == "completed"
    assert manifest["counts"]["raw_records"] == 3
    assert manifest["counts"]["markets"] == 2
    assert manifest["errors"] == []
    assert any(path.endswith("report.json") for path in manifest["output_paths"])


def test_completed_trade_markets_only_resumes_durable_successes(tmp_path):
    request_path = (
        tmp_path
        / "normalized"
        / "kalshi_trade_request"
        / "date=2026-08-29"
        / "metal=btc"
        / "run=one"
        / "requests.jsonl"
    )
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {"market_ticker": "DONE", "status": "completed"}
                ),
                json.dumps(
                    {"market_ticker": "RETRY", "status": "failed"}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert _completed_trade_markets(tmp_path) == {"DONE"}


def test_receive_edge_sequence_is_not_overwritten_by_storage(tmp_path):
    sequencer = GlobalReceiveSequencer()
    recorder = MetalsRecorder(
        _config(tmp_path), require_credentials=False, sequencer=sequencer
    )
    stamp = sequencer.stamp(
        "kalshi", "websocket", receive_ts_ns=100, local_monotonic_ts_ns=200
    )
    envelope = SourceEnvelope(
        source="kalshi",
        stream="trade",
        ingestion_sequence=stamp.ingestion_sequence,
        local_monotonic_ts_ns=stamp.local_monotonic_ts_ns,
        receive_ts_ns=stamp.receive_ts_ns,
        raw={"type": "trade", "msg": {"market_ticker": "TEST"}},
        previous_monotonic_ts_ns=stamp.previous_monotonic_ts_ns,
        monotonic_regression=stamp.monotonic_regression,
    )

    recorder._write_raw_envelope(envelope, "silver")
    recorder._finalize("completed")

    raw_path = next(
        (tmp_path / "metals" / "raw" / "kalshi").glob("**/*.jsonl")
    )
    row = json.loads(raw_path.read_text(encoding="utf-8"))
    assert row["ingestion_sequence"] == stamp.ingestion_sequence
    assert sequencer.last_sequence == stamp.ingestion_sequence


def test_websocket_disconnect_diagnostics_close_on_first_reconnect_message(tmp_path):
    recorder = MetalsRecorder(_config(tmp_path), require_credentials=False)
    recorder._ws_attempt_number = 3
    recorder._active_ws_connection_id = "connection-old"

    class _CloseFrame:
        code = 1011
        reason = "remote\nmaintenance"

    class _Disconnect(RuntimeError):
        rcvd = _CloseFrame()
        sent = None

    disconnected = recorder._record_websocket_disconnect(_Disconnect())
    envelope = SourceEnvelope(
        source="kalshi",
        stream="pyth_value",
        ingestion_sequence=42,
        local_monotonic_ts_ns=200,
        receive_ts_ns=int(disconnected["receive_ts_ns"]) + 250_000_000,
        raw={"type": "pyth_value", "msg": {}},
        connection_id="connection-new",
    )
    recorder._observe_ws_connection(envelope)
    recorder._finalize("completed")

    interval = recorder.quality.missing_intervals[0]
    assert interval["end_ts_ns"] == envelope.receive_ts_ns
    assert interval["duration_ns"] == 250_000_000
    assert interval["details"] == {
        "connection_id": "connection-old",
        "attempt_number": 3,
        "error_type": "_Disconnect",
        "error_module": __name__,
        "received_close_code": 1011,
        "received_close_reason": "remotemaintenance",
        "sent_close_code": None,
        "sent_close_reason": None,
    }
    assert recorder.quality.counters["websocket_disconnects"] == 1
    assert recorder.quality.counters["websocket_reconnects"] == 1

    paths = list(
        (tmp_path / "metals" / "normalized" / "kalshi_transport_event").glob(
            "**/*.jsonl"
        )
    )
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    assert [row["event_type"] for row in rows] == ["disconnected", "reconnected"]
    assert rows[1]["outage_duration_ns"] == 250_000_000
    assert rows[1]["disconnect_events_recovered"] == 1


@pytest.mark.asyncio
async def test_missing_credentials_fail_before_network_and_finalize_manifest(tmp_path):
    recorder = MetalsRecorder(_config(tmp_path), require_credentials=False)

    with pytest.raises(RuntimeError, match="valid Kalshi API key"):
        await recorder.run(duration_seconds=0.01)

    manifest = _manifest(tmp_path, recorder.run_id)
    assert manifest["status"] == "failed"
    assert manifest["errors"]
    assert "valid Kalshi API key" in manifest["errors"][0]["message"]
