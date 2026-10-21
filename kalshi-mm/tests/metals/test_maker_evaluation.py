import hashlib
import json
from pathlib import Path

import pytest

from src.metals.baseline_fair import SCHEMA_VERSION, USAGE_CLASSIFICATION
from src.metals.maker_evaluation import EVALUATION_SPEC, _capacity, evaluate_report, main
from src.metals.maker_policy import MakerSettings, tiny_maker_config
from src.metals.shadow import ShadowRiskEngine
from src.metals.storage import HashChainedJsonlWriter
from tests.metals.test_forward_shadow import _model
from tests.metals.test_maker_policy import Scenario
from tests.metals.test_maker_replay import captured_report  # noqa: F401


def test_evaluation_verifies_primary_and_does_not_promote_scenarios(captured_report, tmp_path):
    path, capture, inputs = captured_report
    before = inputs.read_bytes(), path.read_bytes()
    result = evaluate_report(path, tmp_path / "output", session_label="diagnostic")
    report = json.loads(Path(result["report"]).read_bytes())
    assert result["replay_verified"] is True
    assert result["input_records"] == capture["maker_input_counts"]
    assert report["scenarios"]["primary"]["summary"] == json.loads(json.dumps(capture["summary"], default=str))
    assert len(report["scenarios"]) == 3
    assert report["evaluation_spec"] == EVALUATION_SPEC
    assert report["qualification"]["accessible_edge_demonstrated"] is False
    assert report["qualification"]["live_trading_eligible"] is False
    assert report["live_orders"] is False
    for name, cell in report["scenarios"].items():
        assert cell["reproduces_original_capture"] == (name == "primary")
        for field in capture["settings"]:
            if field not in {"submit_latency_ms", "cancel_latency_ms"}:
                assert cell["settings"][field] == capture["settings"][field]
        assert cell["capacity"]["profit_after_fixed_costs"] is None
        assert cell["capacity"]["arms"]["manual"]["actual_obtainable_contracts_per_hour"] is None
    assert (inputs.read_bytes(), path.read_bytes()) == before


@pytest.mark.parametrize("mutation,match", [
    ("model", "model hash changed"), ("input", "input hash mismatch"),
    ("count", "input count differs"), ("digest", "decision digest differs"),
    ("state", "final state differs"), ("duplicate", "paths missing or duplicated"),
    ("settings", "settings disagree"), ("version", "frozen maker policy"),
    ("live", "shadow-only"),
])
def test_invalid_evidence_never_publishes_success(captured_report, tmp_path, mutation, match):
    path, report, inputs = captured_report
    if mutation == "model":
        model = Path(report["baseline_model_path"])
        model.write_bytes(model.read_bytes() + b"\n")
    elif mutation == "input":
        inputs.write_text(inputs.read_text().replace('"volatility_multiplier": "1.5"',
                                                     '"volatility_multiplier": "2"'))
    elif mutation == "count":
        report["maker_input_counts"] += 1
    elif mutation == "digest":
        report["summary"]["decision_digest"] = "bad"
    elif mutation == "state":
        report["summary"]["arms"]["manual"]["settled_pnl"] = "999"
    elif mutation == "duplicate":
        report["maker_input_paths"] *= 2
    elif mutation == "settings":
        report["settings"]["cancel_latency_ms"] = 500
    elif mutation == "version":
        report["summary"]["schema_version"] = "old-paired-policy"
    else:
        report["summary"]["live_orders"] = True
    path.write_text(json.dumps(report, default=str))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match=match):
        evaluate_report(path, output, session_label="normal")
    assert not output.exists()


def test_historical_trial_expiry_is_not_new_collection(captured_report, tmp_path):
    path, report, _ = captured_report
    report["trial_research"] = {
        "provider_permission_confirmed": False, "collection_not_after_ns": 1,
        "retention_review_at_ns": 2,
    }
    path.write_text(json.dumps(report, default=str))
    result = evaluate_report(path, tmp_path / "output", session_label="closed")
    evaluation = json.loads(Path(result["report"]).read_bytes())
    assert evaluation["provenance"]["trial_research"] == report["trial_research"]
    assert evaluation["retention"]["retention_extension_authorized"] is False
    assert evaluation["retention"]["retention_review_at_ns"] == 2


def test_evaluation_is_content_addressed_and_refuses_overwrite(captured_report, tmp_path):
    path, _, _ = captured_report
    first = evaluate_report(path, tmp_path / "output", session_label="closed")
    second = evaluate_report(path, tmp_path / "output", session_label="closed")
    assert first == second
    output = Path(first["report"])
    output.write_text("{}")
    with pytest.raises(ValueError, match="refusing overwrite"):
        evaluate_report(path, tmp_path / "output", session_label="closed")


def test_label_is_explicit_and_cannot_certify_edge(captured_report, tmp_path):
    path, _, _ = captured_report
    with pytest.raises(ValueError, match="session label"):
        evaluate_report(path, tmp_path / "bad", session_label="profitable")
    result = evaluate_report(path, tmp_path / "normal", session_label="normal")
    assert result["accessible_edge_demonstrated"] is False


@pytest.mark.parametrize("stop_count", [0, 2])
def test_missing_or_duplicate_stop_fails_before_evaluation(captured_report, tmp_path, stop_count):
    path, report, inputs = captured_report
    rows = [json.loads(line) for line in inputs.read_text().splitlines()]
    stop = rows.pop()
    if stop_count:
        rows.extend([stop, {**stop, "ingestion_sequence": stop["ingestion_sequence"] + 1}])
    altered = tmp_path / "altered.jsonl"
    with HashChainedJsonlWriter(altered) as writer:
        for row in rows:
            row.pop("record_sha256")
            row.pop("previous_record_sha256")
            writer.write(row)
    report["maker_input_paths"] = [str(altered)]
    report["maker_input_counts"] = len(rows)
    path.write_text(json.dumps(report, default=str))
    with pytest.raises(ValueError, match="exactly one recorded stop"):
        evaluate_report(path, tmp_path / "output", session_label="diagnostic")


@pytest.mark.parametrize("primary_cancel", [100, 175, 250])
def test_counterfactual_slow_cancel_can_fill_without_changing_primary(tmp_path, primary_cancel):
    scenario = Scenario(tmp_path / "capture", MakerSettings(
        require_fast=False, decision_interval_ms=1, cancel_latency_ms=primary_cancel))
    try:
        for session in scenario.sessions.values():
            session.config = tiny_maker_config(session.config)
            session.engine = ShadowRiskEngine(session.config.shadow)
        scenario.boot()
        scenario.send("timer", 1004)
        scenario.send("timer", 1104)
        scenario.control(1120, version=2, paused=True)
        scenario.send("timer", 1220)
        scenario.trade(1250, source_ms=1240)
        scenario.send("timer", 1700)
        stop = {**scenario.events[-1], "kind": "stop",
                "ingestion_sequence": scenario.sequence + 1}
        scenario.events.append(stop)
        scenario.engine.close(stop)
        inputs = tmp_path / "inputs.jsonl"
        with HashChainedJsonlWriter(inputs) as writer:
            for event in scenario.events:
                writer.write(event)
        model = tmp_path / "model.json"
        model.write_text(json.dumps({
            "schema_version": SCHEMA_VERSION, "usage_classification": USAGE_CLASSIFICATION,
            "study_id": "fixture-study", "model": _model().public_dict(),
        }))
        report = {"status": "completed", "baseline_model_path": str(model),
                  "baseline_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                  "settings": scenario.engine.settings.public_dict(),
                  "maker_input_paths": [str(inputs)], "maker_input_counts": len(scenario.events),
                  "summary": scenario.engine.summary()}
        path = tmp_path / "report.json"
        path.write_text(json.dumps(report, default=str))
        result = evaluate_report(path, tmp_path / "output", session_label="diagnostic")
        evaluation = json.loads(Path(result["report"]).read_bytes())
        primary = evaluation["scenarios"]["primary"]
        slow = (evaluation["scenarios"]["submit_100ms_cancel_250ms"]
                if primary_cancel != 250 else primary)
        fast = (evaluation["scenarios"]["submit_100ms_cancel_100ms"]
                if primary_cancel != 100 else primary)
        assert fast["summary"]["arms"]["manual"]["fills"] == []
        assert len(slow["summary"]["arms"]["manual"]["fills"]) == 1
        assert slow["summary"]["arms"]["manual"]["fills"][0]["cancel_in_flight"] is True
        assert primary["reproduces_original_capture"] is True
        assert slow["reproduces_original_capture"] == (primary_cancel == 250)
        assert primary["settings"]["cancel_latency_ms"] == primary_cancel
        assert {cell["settings"]["cancel_latency_ms"] for cell in evaluation["scenarios"].values()} == {
            100, 250, 500, primary_cancel}
        if primary_cancel == 100:
            assert slow["fill_cohort_comparison_to_primary"]["arms"]["manual"][
                "scenario_quantity_not_filled_in_primary"] == "1"
    finally:
        for session in scenario.sessions.values():
            session.close()


def test_cli(captured_report, tmp_path, capsys):
    path, _, _ = captured_report
    assert main(["--report", str(path), "--output-root", str(tmp_path / "output"),
                 "--session-label", "diagnostic"]) == 0
    assert json.loads(capsys.readouterr().out)["replay_verified"] is True


def test_failed_source_status_is_preserved(captured_report, tmp_path):
    path, report, _ = captured_report
    report.update(status="failed", error_types=["ConnectionError"])
    path.write_text(json.dumps(report, default=str))
    result = evaluate_report(path, tmp_path / "output", session_label="normal")
    evaluation = json.loads(Path(result["report"]).read_bytes())
    assert evaluation["provenance"]["source_capture_status"] == "failed"
    assert "capture_not_cleanly_completed" in evaluation["qualification"]["reasons"]
    assert evaluation["qualification"]["accessible_edge_demonstrated"] is False


def test_capacity_counts_drain_observation_time():
    # One late fill over 120s including drain is 30/h, not 60/h over a 60s pre-stop window.
    summary = {"arms": {arm: {"fills": [{"quantity": "1", "market_ticker": "M"}]}
                        for arm in ("manual", "baseline")}}
    result = _capacity(summary, 120_000_000_000)
    assert float(result["arms"]["manual"]["simulated_contracts_per_capture_hour"]) == 30
    assert result["includes_shutdown_drain"] is True


def test_no_network_needed(captured_report, tmp_path, monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation must not connect")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    path, _, _ = captured_report
    assert evaluate_report(path, tmp_path / "output", session_label="diagnostic")["replay_verified"]
