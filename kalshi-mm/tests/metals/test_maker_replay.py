import hashlib
import json
from dataclasses import asdict, replace

import pytest

from src.metals.baseline_fair import SCHEMA_VERSION, USAGE_CLASSIFICATION
from src.metals.config import MetalsConfig
from src.metals.maker_policy import MakerEngine, MakerSettings, tiny_maker_config
from src.metals.maker_replay import replay_report, settings_from_dict, verified_inputs
from src.metals.shadow_session import ShadowSession
from src.metals.storage import HashChainedJsonlWriter
from tests.metals.test_forward_shadow import _metadata, _model


@pytest.fixture
def captured_report(tmp_path):
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "usage_classification": USAGE_CLASSIFICATION,
        "study_id": "replay-fixture", "model": _model().public_dict(),
    }))
    settings = MakerSettings(require_fast=False, decision_interval_ms=1)
    config = tiny_maker_config(MetalsConfig())
    config = replace(config, storage=replace(config.storage, root=str(tmp_path / "capture")))
    sessions = {arm: ShadowSession(config) for arm in ("manual", "baseline")}
    engine = MakerEngine(_model(), "replay-fixture", settings, sessions)
    ticker, base, ms = _metadata().market_ticker, 1300_000_000_000, 1_000_000
    inputs = tmp_path / "inputs.jsonl"
    payloads = [
        (0, "control", {"control": {
            "version": 1, "effective_ts_ns": base, "expires_ts_ns": base + 3600_000_000_000,
            "volatility_multiplier": "1.5", "paused": False, "reason": "replay fixture",
        }}),
        (1, "metadata", {"metadata": asdict(_metadata())}),
        (2, "book", {"market_ticker": ticker, "yes_bids": {"0.20": "10"},
                     "no_bids": {"0.78": "10"}, "valid": True, "book_state_hash": "book"}),
        (4, "exact", {"value": "66.95", "source_ts_ns": base}),
        (1004, "timer", {}),
        (1104, "timer", {}),
        (1120, "trade", {"market_ticker": ticker, "trade_id": "fill-1",
                         "source_event_ts_ns": base + 1110 * ms,
                         "quantity_fp": "1", "yes_price_dollars": "0.20",
                         "no_price_dollars": "0.80", "taker_outcome_side": "no"}),
        (2120, "timer", {}),
        (601000, "metadata", {"metadata": asdict(_metadata()), "result": "yes"}),
        (601001, "stop", {}),
    ]
    try:
        with HashChainedJsonlWriter(inputs) as writer:
            for sequence, (offset, kind, payload) in enumerate(payloads, 1):
                event = {"kind": kind, "receive_ts_ns": base + offset * ms,
                         "local_monotonic_ts_ns": 10_000_000_000 + offset * ms,
                         "ingestion_sequence": sequence, **payload}
                writer.write(event)
                engine.close(event) if kind == "stop" else engine.handle(event)
        summary = engine.summary()
        assert all(len(arm["fills"]) == 1 for arm in summary["arms"].values())
        assert all(arm["fills"][0]["terminal_pnl"] is not None
                   for arm in summary["arms"].values())
        report = {"status": "completed", "baseline_model_path": str(model_path),
                  "baseline_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  "settings": settings.public_dict(), "maker_input_paths": [str(inputs)],
                  "maker_input_counts": len(payloads), "summary": summary}
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report, default=str))
        return path, report, inputs
    finally:
        for session in sessions.values():
            session.close()


def test_replay_verifies_decisions_fills_marks_and_terminal_state(captured_report, tmp_path):
    path, report, _ = captured_report
    result = replay_report(path, tmp_path / "replay")
    assert result["replay_verified"] is True
    assert result["input_records"] == report["maker_input_counts"]


@pytest.mark.parametrize("mutation,match", [
    ("model", "model hash changed"), ("input", "input hash mismatch"),
    ("count", "input count differs"), ("digest", "decision digest differs"),
    ("state", "final state differs"), ("duplicate", "paths missing or duplicated"),
])
def test_replay_rejects_changed_evidence(captured_report, tmp_path, mutation, match):
    path, report, inputs = captured_report
    if mutation == "model":
        from pathlib import Path
        model = Path(report["baseline_model_path"])
        model.write_text(model.read_text() + "\n")
    elif mutation == "input":
        inputs.write_text(inputs.read_text().replace('"volatility_multiplier": "1.5"',
                                                     '"volatility_multiplier": "2"'))
    elif mutation == "count":
        report["maker_input_counts"] += 1
    elif mutation == "digest":
        report["summary"]["decision_digest"] = "wrong"
    elif mutation == "state":
        report["summary"]["arms"]["manual"]["settled_pnl"] = "999"
    else:
        report["maker_input_paths"] *= 2
    path.write_text(json.dumps(report, default=str))
    with pytest.raises(ValueError, match=match):
        replay_report(path, tmp_path / "replay")


def test_replay_checks_chain_even_when_individual_hash_is_valid(tmp_path):
    path = tmp_path / "inputs.jsonl"
    with HashChainedJsonlWriter(path) as writer:
        writer.write({"ingestion_sequence": 1})
        writer.write({"ingestion_sequence": 2})
    rows = path.read_text().splitlines()
    path.write_text(rows[1] + "\n")
    with pytest.raises(ValueError, match="input chain mismatch"):
        list(verified_inputs(path))


def test_replay_requires_complete_settings():
    value = MakerSettings().public_dict()
    assert settings_from_dict(value) == MakerSettings()
    value.pop("require_fast")
    with pytest.raises(ValueError, match="complete frozen settings"):
        settings_from_dict(value)
