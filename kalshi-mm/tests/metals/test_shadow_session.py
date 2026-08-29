import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.metals.config import load_metals_config
from src.metals.shadow import ShadowOrderIntent
from src.metals import shadow_session as shadow_session_module
from src.metals.shadow_session import ShadowSession


def _config(tmp_path: Path):
    config = load_metals_config("configs/metals.example.toml")
    return replace(
        config,
        storage=replace(config.storage, root=str(tmp_path / "metals")),
    )


def _intent():
    return ShadowOrderIntent(
        logical_quote_id="q1",
        market_ticker="KXSILVER15M-TEST",
        metal="silver",
        outcome_side="yes",
        price_dollars="0.40",
        quantity_fp="10",
        decision_ts_ns=100,
        strategy_module="S2",
        model_version="model-v0",
        config_version="metals-shadow-v1",
        state_snapshot_hash="state-hash",
    )


def test_session_persists_decision_lifecycle_fill_and_manifest(tmp_path):
    session = ShadowSession(_config(tmp_path))

    decision, events = session.place(_intent())
    session.fill("q1", "10", event_ts_ns=110)
    session.close()

    assert decision.approved
    assert len(events) == 4
    decision_paths = list(
        (tmp_path / "metals" / "normalized" / "shadow_decision").glob("**/*.jsonl")
    )
    lifecycle_paths = list(
        (tmp_path / "metals" / "normalized" / "order_lifecycle").glob("**/*.jsonl")
    )
    assert len(decision_paths) == 1
    assert len(lifecycle_paths) == 1
    decision_row = json.loads(
        decision_paths[0].read_text(encoding="utf-8").splitlines()[0]
    )
    assert decision_row["approved"] is True
    assert decision_row["execution_mode"] == "shadow"
    lifecycle_rows = [
        json.loads(line)
        for line in lifecycle_paths[0].read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event_type"] for row in lifecycle_rows] == [
        "decision_created",
        "submit_sent",
        "submit_acknowledged",
        "order_resting",
        "full_fill",
    ]
    manifest = json.loads(
        (
            tmp_path
            / "metals"
            / "manifests"
            / "run={}".format(session.run_id)
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["counts"]["shadow_decisions"] == 1
    assert manifest["counts"]["order_lifecycle_events"] == 5


def test_session_records_risk_rejection_without_resting_order(tmp_path):
    session = ShadowSession(_config(tmp_path))
    session.engine.set_kill_switch("invalid_kalshi_book")

    decision, events = session.place(_intent())
    session.close()

    assert decision.approved is False
    assert [event.event_type for event in events] == [
        "decision_created",
        "risk_rejected",
    ]
    assert session.engine.resting_orders == {}


def test_manifest_update_failure_leaves_close_retryable(tmp_path, monkeypatch):
    session = ShadowSession(_config(tmp_path))
    real_update = shadow_session_module.update_run_manifest
    calls = {"count": 0}

    def fail_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("manifest unavailable")
        return real_update(*args, **kwargs)

    monkeypatch.setattr(shadow_session_module, "update_run_manifest", fail_once)

    with pytest.raises(OSError, match="manifest unavailable"):
        session.close()
    assert session._closed is False

    session.close()
    assert session._closed is True
