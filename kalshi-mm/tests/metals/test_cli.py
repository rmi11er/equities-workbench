import json
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from src.metals import cli
from src.metals.cli import build_parser
from src.metals.config import (
    CmeConfig,
    MetalsConfig,
    ProductConfig,
    PythConfig,
    StorageConfig,
)
from src.metals.databento_history import (
    HistoricalArchiveResult,
    HistoricalCostEstimate,
)
from src.metals.historical_research_assumption import (
    HistoricalResearchAssumptionError,
)
from tests.metals.historical_research_assumption_fixtures import (
    historical_assumption_policy,
)
from tests.metals.provider_approval_fixtures import (
    approved_provider_policy,
    materialize_approved_provider_path,
)


def test_databento_cli_defaults_to_bounded_silver_collection():
    args = build_parser().parse_args(["collect-databento"])

    assert args.command == "collect-databento"
    assert args.metal == "silver"
    assert args.duration_seconds == 60.0
    assert args.max_records == 0


def test_pyth_lazer_cli_defaults_to_bounded_silver_collection():
    args = build_parser().parse_args(["collect-pyth-lazer"])

    assert args.command == "collect-pyth-lazer"
    assert args.metal == "silver"
    assert args.duration_seconds == 60.0


def test_pyth_exact_history_cli_requires_bounded_microsecond_window():
    args = build_parser().parse_args(
        [
            "backfill-pyth-exact-history",
            "--start-timestamp-us",
            "1788000000000000",
            "--end-timestamp-us",
            "1788000060000000",
        ]
    )

    assert args.metal == "silver"
    assert args.limit == 1000
    assert args.max_pages == 1000
    assert args.timeout_seconds == 30.0


def test_pyth_proxy_history_cli_is_explicitly_bounded_and_metal_limited():
    args = build_parser().parse_args(
        [
            "backfill-pyth-proxy-history",
            "--metal",
            "silver",
            "--from-seconds",
            "1787875200",
            "--to-seconds",
            "1787961599",
        ]
    )

    assert args.metal == "silver"
    assert args.from_seconds == 1787875200
    assert args.to_seconds == 1787961599


def test_settlement_proxy_pilot_cli_requires_explicit_inputs_and_bounds():
    args = build_parser().parse_args(
        [
            "run-settlement-proxy-pilot",
            "--labels",
            "labels.jsonl",
            "--databento-trades",
            "trades.jsonl",
            "--pyth-proxy-candles",
            "proxy.jsonl",
            "--start-ts-ns",
            "1787875200000000000",
            "--end-ts-ns",
            "1787961600000000000",
        ]
    )

    assert args.metal == "silver"
    assert args.primary_asset == "SI"
    assert args.secondary_asset == "SIL"
    assert args.max_trade_age_seconds == 5
    assert args.ambiguity_dollars == Decimal("0.005")


def test_baseline_fair_cli_freezes_chronology_and_prerequisites():
    args = build_parser().parse_args(
        [
            "run-baseline-fair",
            "--labels",
            "labels.jsonl",
            "--pyth-proxy-candles",
            "proxy.jsonl",
            "--training-end-ts-ns",
            "1787875200000000000",
            "--validation-start-ts-ns",
            "1787875200000000000",
            "--validation-end-ts-ns",
            "1787961600000000000",
            "--formal-research-design-id",
            "a" * 64,
            "--prerequisite-proxy-pilot-id",
            "b" * 64,
        ]
    )

    assert args.ewma_lambda == 0.97
    assert args.tod_prior_observations == 20
    assert args.winsor_quantile == 0.995


def test_spot_return_signal_cli_keeps_lookbacks_explicit_and_independent():
    args = build_parser().parse_args(
        [
            "run-spot-return-signals",
            "--baseline-study-id",
            "a" * 64,
            "--pyth-proxy-candles",
            "proxy.jsonl",
            "--lookback-minutes",
            "1",
            "--lookback-minutes",
            "5",
        ]
    )

    assert args.lookback_minutes == [1, 5]
    assert args.development_fraction == 0.67
    assert args.embargo_windows == 1
    assert args.ridge_penalty == 10.0


def test_path_variance_signal_cli_keeps_range_and_efficiency_separate():
    args = build_parser().parse_args(
        [
            "run-path-variance-signals",
            "--baseline-study-id",
            "a" * 64,
            "--pyth-proxy-candles",
            "proxy.jsonl",
            "--range-lookback-minutes",
            "3",
            "--efficiency-lookback-minutes",
            "5",
        ]
    )

    assert args.range_lookback_minutes == [3]
    assert args.efficiency_lookback_minutes == [5]
    assert args.embargo_windows == 1


def test_futures_activity_signal_cli_is_explicitly_contract_family_bounded():
    args = build_parser().parse_args(
        [
            "run-futures-activity-signals",
            "--baseline-study-id",
            "a" * 64,
            "--databento-trades",
            "trades.jsonl",
            "--asset",
            "SI",
        ]
    )

    assert args.asset == "SI"
    assert args.max_trade_age_seconds == 30
    assert args.embargo_windows == 1


def test_semivariance_jump_signal_cli_freezes_the_lookback():
    args = build_parser().parse_args(
        [
            "run-semivariance-jump-signals",
            "--baseline-study-id",
            "a" * 64,
            "--pyth-proxy-candles",
            "proxy.jsonl",
        ]
    )

    assert args.lookback_minutes == 10
    assert args.embargo_windows == 1


def test_cross_metal_signal_cli_keeps_sources_and_lookback_explicit():
    args = build_parser().parse_args(
        [
            "run-cross-metal-signals",
            "--baseline-study-id",
            "a" * 64,
            "--silver-pyth-proxy-candles",
            "silver.jsonl",
            "--gold-pyth-proxy-candles",
            "gold.jsonl",
        ]
    )

    assert args.lookback_minutes == 5
    assert args.silver_pyth_proxy_candles == Path("silver.jsonl")
    assert args.gold_pyth_proxy_candles == Path("gold.jsonl")
    assert args.embargo_windows == 1


def test_baseline_scorecard_cli_requires_the_historical_kalshi_panel():
    args = build_parser().parse_args(
        [
            "run-baseline-scorecard",
            "--baseline-study-id",
            "a" * 64,
            "--kalshi-candlesticks",
            "candles.jsonl",
        ]
    )

    assert args.kalshi_candlesticks == Path("candles.jsonl")


def test_natural_target_cli_keeps_optional_sources_and_bootstrap_explicit():
    args = build_parser().parse_args(
        [
            "run-natural-target-signals",
            "--baseline-study-id",
            "a" * 64,
            "--silver-pyth-proxy-candles",
            "silver.jsonl",
            "--gold-pyth-proxy-candles",
            "gold.jsonl",
            "--databento-trades",
            "trades.jsonl",
        ]
    )

    assert args.gold_pyth_proxy_candles == Path("gold.jsonl")
    assert args.databento_trades == Path("trades.jsonl")
    assert args.walk_forward_bootstrap_samples == 2000
    assert args.walk_forward_bootstrap_seed == 17290


def test_pyth_settlement_history_cli_separates_offline_plan_from_network_execution():
    planned = build_parser().parse_args(
        ["plan-pyth-settlement-history", "--audit", "audit-dir"]
    )
    executed = build_parser().parse_args(
        ["backfill-pyth-settlement-history", "--plan", "plan-dir"]
    )

    assert planned.metal == "silver"
    assert planned.max_pages == 2
    assert executed.max_boundaries == 100
    assert executed.request_interval_seconds == 0.1
    assert executed.timeout_seconds == 30.0
    assert executed.allow_network_download is False


def test_synchronized_cli_defaults_to_bounded_silver_collection():
    args = build_parser().parse_args(["collect-synchronized"])

    assert args.command == "collect-synchronized"
    assert args.metal == "silver"
    assert args.duration_seconds == 60.0


def test_forward_shadow_cli_requires_explicit_model_and_live_data_acknowledgement():
    args = build_parser().parse_args(
        [
            "run-forward-shadow",
            "--baseline-model",
            "model.json",
            "--duration-seconds",
            "30",
            "--allow-live-shadow-data",
        ]
    )

    assert args.command == "run-forward-shadow"
    assert args.baseline_model == Path("model.json")
    assert args.duration_seconds == 30
    assert args.allow_live_shadow_data is True
    assert args.quantity == Decimal("1")


def test_clock_cli_is_bounded_and_preflight_has_no_duration():
    collect = build_parser().parse_args(["collect-clock-telemetry"])
    preflight = build_parser().parse_args(["preflight-clock"])

    assert collect.metal == "silver"
    assert collect.duration_seconds == 60.0
    assert preflight.command == "preflight-clock"
    assert not hasattr(preflight, "duration_seconds")


def test_provider_approval_cli_has_separate_offline_materialize_and_check_commands():
    materialize = build_parser().parse_args(
        ["materialize-provider-approval", "--approval", "reviewed.json"]
    )
    check = build_parser().parse_args(
        [
            "check-provider-approval",
            "--provider",
            "pyth",
            "--operation",
            "exact_history",
        ]
    )

    assert materialize.approval == Path("reviewed.json")
    assert check.provider == "pyth"
    assert check.operation == "exact_history"


def test_historical_assumption_cli_has_offline_materialize_and_check_commands():
    materialize = build_parser().parse_args(
        [
            "materialize-historical-research-assumption",
            "--assumption",
            "reviewed.json",
        ]
    )
    check = build_parser().parse_args(
        [
            "check-historical-research-assumption",
            "--provider",
            "databento",
            "--operation",
            "historical",
        ]
    )

    assert materialize.assumption == Path("reviewed.json")
    assert check.provider == "databento"
    assert check.operation == "historical"


def test_historical_assumption_cli_materializes_and_checks_without_network(
    monkeypatch, tmp_path, capsys
):
    config = _pyth_lazer_config(tmp_path, enabled=False)
    reviewed = tmp_path / "reviewed-assumption.json"
    reviewed.write_text(json.dumps(historical_assumption_policy()), encoding="utf-8")
    monkeypatch.setattr(cli, "load_metals_config", lambda path: config)

    assert cli._materialize_historical_research_assumption("unused.toml", reviewed) == 0
    materialized = json.loads(capsys.readouterr().out)
    config = replace(
        config,
        governance=replace(
            config.governance,
            pyth_historical_research_assumption_path=materialized["assumption_path"],
        ),
    )
    monkeypatch.setattr(cli, "load_metals_config", lambda path: config)

    assert (
        cli._check_historical_research_assumption(
            "unused.toml", "pyth", "exact_history"
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["research_only"] is True
    assert report["no_live"] is True
    assert report["no_orders"] is True


def test_provider_approval_cli_materializes_and_checks_without_network(
    monkeypatch, tmp_path, capsys
):
    config = _pyth_lazer_config(tmp_path, enabled=False)
    reviewed = tmp_path / "reviewed-provider.json"
    reviewed.write_text(json.dumps(approved_provider_policy()), encoding="utf-8")
    monkeypatch.setattr(cli, "load_metals_config", lambda path: config)

    assert cli._materialize_provider_approval("unused.toml", reviewed) == 0
    materialized = json.loads(capsys.readouterr().out)
    config = replace(
        config,
        governance=replace(
            config.governance,
            provider_approval_path=materialized["approval_path"],
        ),
    )
    monkeypatch.setattr(cli, "load_metals_config", lambda path: config)

    assert cli._check_provider_approval("unused.toml", "pyth", "exact_history") == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_provider_approval_cli_fails_closed_when_config_path_is_absent(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        cli,
        "load_metals_config",
        lambda path: _pyth_lazer_config(tmp_path, enabled=False),
    )

    assert cli._check_provider_approval("unused.toml", "pyth", "live_lazer") == 2
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert "required" in report["status"]


def test_daily_validation_cli_requires_explicit_inputs_and_ten_days_by_default():
    frozen = build_parser().parse_args(
        [
            "freeze-daily-validation",
            "--date",
            "2026-08-29",
            "--quality-report",
            "quality.json",
            "--manifest",
            "run-1.json",
            "--manifest",
            "run-2.json",
        ]
    )
    progress = build_parser().parse_args(["check-data-readiness"])
    measured = build_parser().parse_args(
        [
            "measure-daily-gates",
            "--date",
            "2026-08-29",
            "--quality-report",
            "quality.json",
            "--policy",
            "policy.json",
            "--manifest",
            "run-1.json",
        ]
    )
    policy = build_parser().parse_args(
        [
            "materialize-daily-policy-registries",
            "--date",
            "2026-08-29",
            "--policy",
            "policy.json",
            "--fee-registry",
            "fees.json",
            "--incentive-registry",
            "incentives.json",
        ]
    )
    settlement = build_parser().parse_args(
        [
            "audit-settlements",
            "--semantics",
            "settlement-semantics.json",
            "--universe",
            "settlement-universe.json",
        ]
    )

    assert frozen.date == "2026-08-29"
    assert frozen.quality_report == Path("quality.json")
    assert frozen.manifest == [Path("run-1.json"), Path("run-2.json")]
    assert measured.date == "2026-08-29"
    assert measured.quality_report == Path("quality.json")
    assert measured.policy == Path("policy.json")
    assert measured.manifest == [Path("run-1.json")]
    assert policy.policy == Path("policy.json")
    assert policy.fee_registry == Path("fees.json")
    assert policy.incentive_registry == Path("incentives.json")
    assert settlement.metal == "silver"
    assert settlement.semantics == Path("settlement-semantics.json")
    assert settlement.universe == Path("settlement-universe.json")
    assert progress.minimum_complete_days == 10
    assert progress.minimum_settlements == 1000


def test_hash_registry_config_cli_reads_one_json_object(tmp_path, capsys):
    path = tmp_path / "fee.json"
    path.write_text(json.dumps({"value": "exact"}), encoding="utf-8")

    assert cli._hash_registry_config(path) == 0
    digest = json.loads(capsys.readouterr().out)["config_hash"]
    assert len(digest) == 64


def test_materialize_policy_cli_loads_reviewed_json_arrays(
    monkeypatch, tmp_path, capsys
):
    policy_path = tmp_path / "policy.json"
    fee_path = tmp_path / "fees.json"
    incentive_path = tmp_path / "incentives.json"
    policy_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    fee_path.write_text(json.dumps([{"config_hash": "a" * 64}]), encoding="utf-8")
    incentive_path.write_text(json.dumps([{"config_hash": "b" * 64}]), encoding="utf-8")
    storage_root = tmp_path / "metals"
    captured = {}

    class Artifacts:
        materialization_id = "materialization-id"
        policy_path = storage_root / "policy.json"
        fee_schedule_path = storage_root / "fees.jsonl"
        incentive_program_path = storage_root / "incentives.jsonl"
        manifest_path = storage_root / "manifest.json"

    monkeypatch.setattr(
        cli,
        "load_metals_config",
        lambda path: MetalsConfig(
            storage=StorageConfig(root=str(storage_root)),
            products={"silver": ProductConfig(enabled=True)},
        ),
    )

    def fake_materialize(root, date, policy, fees, incentives):
        captured.update(
            root=root,
            date=date,
            policy=policy,
            fees=fees,
            incentives=incentives,
        )
        return Artifacts()

    monkeypatch.setattr(cli, "materialize_daily_policy_registries", fake_materialize)

    assert (
        cli._materialize_daily_policy_registries(
            "unused.toml",
            "2026-08-29",
            policy_path,
            fee_path,
            incentive_path,
        )
        == 0
    )
    assert captured["root"] == storage_root
    assert captured["date"] == "2026-08-29"
    assert captured["fees"] == [{"config_hash": "a" * 64}]
    assert json.loads(capsys.readouterr().out)["materialization_id"] == (
        "materialization-id"
    )


def test_settlement_audit_cli_uses_reviewed_numeric_feed_and_persists_manifest(
    monkeypatch, tmp_path, capsys
):
    storage_root = tmp_path / "metals"
    semantics_path = tmp_path / "semantics.json"
    semantics_path.write_text(
        json.dumps(
            {
                "schema_version": "metals-settlement-semantics/v2",
                "timestamp": {
                    "version": "reviewed-v1",
                    "confirmed": True,
                    "evidence_sha256": "a" * 64,
                },
                "correction": {
                    "version": "reviewed-v1",
                    "confirmed": True,
                    "evidence_sha256": "b" * 64,
                },
                "rounding": {
                    "version": "reviewed-v1",
                    "confirmed": True,
                    "evidence_sha256": "c" * 64,
                },
                "timestamp_field": "publish_timestamp_us",
                "timestamp_meaning": ("last_publication_strictly_before_boundary"),
                "timestamp_source_label": (
                    "publish_time_and_feed_update_timestamp_microseconds"
                ),
                "correction_policy": "last_publication_before_boundary_v1",
                "rounding_modes_by_series": {"KXSILVER15M": "half_up"},
            }
        ),
        encoding="utf-8",
    )
    artifact_dir = storage_root / "derived" / "settlement_check" / ("audit=" + "d" * 64)
    artifact_dir.mkdir(parents=True)
    for name in ("windows.jsonl", "summary.json", "manifest.json"):
        (artifact_dir / name).write_text("{}\n", encoding="utf-8")

    class Artifact:
        audit_id = "d" * 64
        directory = artifact_dir
        details_path = artifact_dir / "windows.jsonl"
        summary_path = artifact_dir / "summary.json"
        manifest_path = artifact_dir / "manifest.json"
        summary = {
            "expected_count": 1,
            "usable_count": 1,
            "excluded_count": 0,
            "missing_count": 0,
            "conflicting_count": 0,
            "fully_agreeing_usable_windows": 1,
            "settlement_research_gate_passed": True,
        }

    captured = {}

    def fake_audit(root, semantics, **kwargs):
        captured.update(root=root, semantics=semantics, kwargs=kwargs)
        return Artifact()

    monkeypatch.setattr(
        cli,
        "load_metals_config",
        lambda path: MetalsConfig(
            storage=StorageConfig(root=str(storage_root)),
            products={
                "silver": ProductConfig(
                    enabled=True,
                    pyth_symbol="Metal.Index.SILVER/USD",
                    pyth_lazer_id=3154,
                )
            },
        ),
    )
    monkeypatch.setattr(cli, "run_local_settlement_audit", fake_audit)
    monkeypatch.setattr(cli, "verify_settlement_audit", lambda *args: {})

    assert cli._audit_settlements("unused.toml", "silver", semantics_path, None) == 0
    assert captured["root"] == storage_root
    assert captured["kwargs"]["exact_feed_id"] == 3154
    assert captured["kwargs"]["universe"] is None
    run_manifest = (
        storage_root
        / "manifests"
        / ("run=settlement-audit-" + "d" * 64)
        / "manifest.json"
    )
    first_bytes = run_manifest.read_bytes()
    manifest = json.loads(first_bytes)
    assert manifest["status"] == "completed"
    assert manifest["config"]["live_orders"] is False
    assert manifest["config"]["network_access"] is False
    assert manifest["counts"]["usable_count"] == 1
    assert len(manifest["output_paths"]) == 3

    assert cli._audit_settlements("unused.toml", "silver", semantics_path, None) == 0
    assert run_manifest.read_bytes() == first_bytes
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["usable_count"] == 1


def test_cli_json_loader_rejects_duplicate_semantics_fields(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON field"):
        cli._load_cli_json(path, "settlement semantics")


def test_data_readiness_rejects_nonpositive_required_day_count(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli,
        "load_metals_config",
        lambda path: MetalsConfig(
            storage=StorageConfig(root=str(tmp_path)),
            products={"silver": ProductConfig(enabled=True)},
        ),
    )

    with pytest.raises(ValueError, match="at least one"):
        cli._check_data_readiness("unused.toml", 0, 1000)

    with pytest.raises(ValueError, match="minimum_settlements"):
        cli._check_data_readiness("unused.toml", 10, 0)


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf", "-inf"])
def test_pyth_lazer_cli_rejects_unbounded_or_non_finite_duration(duration):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["collect-pyth-lazer", "--duration-seconds", duration]
        )


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf", "-inf"])
def test_synchronized_cli_rejects_unbounded_or_non_finite_duration(duration):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["collect-synchronized", "--duration-seconds", duration]
        )


def _pyth_lazer_config(tmp_path, *, enabled):
    return MetalsConfig(
        storage=StorageConfig(root=str(tmp_path / "metals")),
        pyth=PythConfig(enabled=enabled, api_key_env="TEST_PYTH_LAZER_API_KEY"),
        products={
            "silver": ProductConfig(
                enabled=True,
                pyth_symbol="Metal.Index.SILVER/USD",
                pyth_lazer_id=3154,
            )
        },
    )


def _clock_config(tmp_path):
    config = _pyth_lazer_config(tmp_path, enabled=False)
    return replace(config, clock=replace(config.clock, enabled=True))


@pytest.mark.asyncio
async def test_clock_preflight_returns_fail_closed_status(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cli, "load_metals_config", lambda path: _clock_config(tmp_path))
    monkeypatch.setattr(
        cli,
        "preflight_clock",
        lambda config: {
            "ok": False,
            "provider": "chrony",
            "status": "not_synchronized",
        },
    )
    args = build_parser().parse_args(["preflight-clock"])

    assert await cli.async_main(args) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "not_synchronized"


@pytest.mark.asyncio
async def test_clock_collection_cli_is_bounded_and_read_only(
    monkeypatch, tmp_path, capsys
):
    created = []

    class FakeClockCollector:
        run_id = "clock-run"

        def __init__(self, config, metal):
            self.config = config
            self.metal = metal
            self.duration = None
            created.append(self)

        def stop(self):
            pass

        async def run(self, *, duration_seconds):
            self.duration = duration_seconds
            return {"clock_samples": 2}

    monkeypatch.setattr(cli, "load_metals_config", lambda path: _clock_config(tmp_path))
    monkeypatch.setattr(cli, "ClockTelemetryCollector", FakeClockCollector)
    args = build_parser().parse_args(
        ["collect-clock-telemetry", "--duration-seconds", "2.5"]
    )

    assert await cli.async_main(args) == 0
    assert len(created) == 1
    assert created[0].duration == 2.5
    assert json.loads(capsys.readouterr().out) == {
        "clock_samples": 2,
        "run_id": "clock-run",
    }


@pytest.mark.asyncio
async def test_pyth_lazer_cli_refuses_disabled_config_before_collector_construction(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        cli,
        "load_metals_config",
        lambda path: _pyth_lazer_config(tmp_path, enabled=False),
    )

    def must_not_construct(*args, **kwargs):
        raise AssertionError("disabled Pyth must fail before collector construction")

    monkeypatch.setattr(cli, "PythLazerCollector", must_not_construct)
    args = build_parser().parse_args(["collect-pyth-lazer"])

    with pytest.raises(ValueError, match=r"\[pyth\]\.enabled"):
        await cli.async_main(args)

    assert not (tmp_path / "metals").exists()


@pytest.mark.asyncio
async def test_pyth_exact_history_cli_uses_numeric_feed_while_live_is_disabled(
    monkeypatch, tmp_path, capsys
):
    config = _pyth_lazer_config(tmp_path, enabled=False)
    approval_path = materialize_approved_provider_path(tmp_path / "metals")
    config = replace(
        config,
        governance=replace(config.governance, provider_approval_path=approval_path),
    )
    captured = {}

    class Result:
        @staticmethod
        def public_dict():
            return {"completed": True, "request_id": "request-id"}

    async def fake_run(root, request, **kwargs):
        captured["root"] = root
        captured["request"] = request
        captured["kwargs"] = kwargs
        return Result()

    monkeypatch.setattr(cli, "load_metals_config", lambda path: config)
    monkeypatch.setattr(cli, "run_pyth_exact_history", fake_run)

    code = await cli._backfill_pyth_exact_history(
        "unused.toml",
        "silver",
        1_788_000_000_000_000,
        1_788_000_060_000_000,
        1000,
        1000,
        30.0,
    )

    assert code == 0
    assert captured["root"] == tmp_path / "metals"
    assert captured["request"].feed_ids == (3154,)
    assert captured["request"].channel == "fixed_rate@200ms"
    assert captured["kwargs"]["api_key_env"] == "TEST_PYTH_LAZER_API_KEY"
    assert captured["kwargs"]["provider_approval_path"] == approval_path
    assert json.loads(capsys.readouterr().out)["completed"] is True


def test_pyth_settlement_plan_cli_is_offline_and_uses_configured_feed(
    monkeypatch, tmp_path, capsys
):
    config = _pyth_lazer_config(tmp_path, enabled=False)
    captured = {}

    class Artifact:
        @staticmethod
        def public_dict():
            return {"plan_id": "plan-id", "boundary_count": 3}

    def fake_create(root, audit, **kwargs):
        captured["root"] = root
        captured["audit"] = audit
        captured["kwargs"] = kwargs
        return Artifact()

    monkeypatch.setattr(cli, "load_metals_config", lambda path: config)
    monkeypatch.setattr(cli, "create_pyth_settlement_backfill_plan", fake_create)

    code = cli._plan_pyth_settlement_history(
        "unused.toml", "silver", "audit-dir", 1000, 2
    )

    assert code == 0
    assert captured["root"] == tmp_path / "metals"
    assert captured["audit"] == Path("audit-dir")
    assert captured["kwargs"]["feed_id"] == 3154
    assert captured["kwargs"]["max_pages"] == 2
    assert json.loads(capsys.readouterr().out)["boundary_count"] == 3


@pytest.mark.asyncio
async def test_pyth_settlement_backfill_cli_forwards_explicit_network_guard(
    monkeypatch, tmp_path, capsys
):
    config = _pyth_lazer_config(tmp_path, enabled=False)
    approval_path = materialize_approved_provider_path(tmp_path / "metals")
    config = replace(
        config,
        governance=replace(config.governance, provider_approval_path=approval_path),
    )
    captured = {}

    class Plan:
        plan_path = tmp_path / "metals" / "plans" / "plan.json"
        value = {
            "metal": "silver",
            "feed_id": 3154,
            "request_parameters": {"base_url": config.pyth.endpoint},
        }

    class Result:
        @staticmethod
        def public_dict():
            return {"plan_completed": False, "remaining_boundaries": 2}

    async def fake_run(root, plan, **kwargs):
        captured["root"] = root
        captured["plan"] = plan
        captured["kwargs"] = kwargs
        return Result()

    monkeypatch.setattr(cli, "load_metals_config", lambda path: config)
    monkeypatch.setattr(
        cli, "verify_pyth_settlement_backfill_plan", lambda root, path: Plan()
    )
    monkeypatch.setattr(cli, "run_pyth_settlement_backfill", fake_run)

    code = await cli._backfill_pyth_settlement_history(
        "unused.toml", "plan-dir", 7, 0.25, 12.0, True
    )

    assert code == 0
    assert captured["root"] == tmp_path / "metals"
    assert captured["plan"] == Plan.plan_path
    assert captured["kwargs"] == {
        "api_key_env": "TEST_PYTH_LAZER_API_KEY",
        "allow_network_download": True,
        "max_boundaries": 7,
        "request_interval_seconds": 0.25,
        "timeout_seconds": 12.0,
        "provider_approval_path": approval_path,
        "historical_research_assumption_path": "",
    }
    assert json.loads(capsys.readouterr().out)["remaining_boundaries"] == 2


@pytest.mark.asyncio
async def test_pyth_lazer_cli_forwards_strict_duration_without_network(
    monkeypatch, tmp_path, capsys
):
    class FakeCollector:
        def __init__(self, config, metal):
            self.config = config
            self.metal = metal
            self.run_id = "test-pyth-run"
            self.duration_seconds = None

        def stop(self):
            pass

        async def run(self, *, duration_seconds):
            self.duration_seconds = duration_seconds
            return {"raw_messages": 2, "normalized_prices": 1}

    created = []

    def make_collector(config, metal):
        collector = FakeCollector(config, metal)
        created.append(collector)
        return collector

    monkeypatch.setattr(
        cli,
        "load_metals_config",
        lambda path: _pyth_lazer_config(tmp_path, enabled=True),
    )
    monkeypatch.setattr(cli, "PythLazerCollector", make_collector)
    args = build_parser().parse_args(
        ["collect-pyth-lazer", "--metal", "silver", "--duration-seconds", "2.5"]
    )

    assert await cli.async_main(args) == 0
    assert len(created) == 1
    assert created[0].metal == "silver"
    assert created[0].duration_seconds == 2.5
    output = capsys.readouterr().out
    assert '"run_id": "test-pyth-run"' in output
    assert '"normalized_prices": 1' in output


@pytest.mark.asyncio
async def test_synchronized_cli_forwards_duration_without_network(
    monkeypatch, tmp_path, capsys
):
    class FakeRecorder:
        def __init__(self, config, metal):
            self.config = config
            self.metal = metal
            self.duration_seconds = None

        def stop(self):
            pass

        async def run(self, *, duration_seconds):
            self.duration_seconds = duration_seconds
            return {
                "run_id": "test-synchronized-run",
                "quality_report_path": str(tmp_path / "report.json"),
            }

    created = []

    def make_recorder(config, metal):
        recorder = FakeRecorder(config, metal)
        created.append(recorder)
        return recorder

    monkeypatch.setattr(
        cli,
        "load_metals_config",
        lambda path: _pyth_lazer_config(tmp_path, enabled=True),
    )
    monkeypatch.setattr(cli, "SynchronizedMetalsRecorder", make_recorder)
    args = build_parser().parse_args(
        ["collect-synchronized", "--metal", "silver", "--duration-seconds", "2.5"]
    )

    assert await cli.async_main(args) == 0
    assert len(created) == 1
    assert created[0].metal == "silver"
    assert created[0].duration_seconds == 2.5
    assert '"run_id": "test-synchronized-run"' in capsys.readouterr().out


def test_source_preflights_require_explicit_bounded_probe_times():
    kalshi = build_parser().parse_args(["preflight-kalshi"])
    pyth = build_parser().parse_args(
        ["preflight-pyth", "--history-at-seconds", "1780000000"]
    )
    with pytest.raises(SystemExit):
        build_parser().parse_args(["preflight-pyth-lazer-live"])
    pyth_live = build_parser().parse_args(
        ["preflight-pyth-lazer-live", "--allow-live-entitlement-probe"]
    )
    databento = build_parser().parse_args(
        ["preflight-databento", "--probe-date", "2026-08-28"]
    )
    with pytest.raises(SystemExit):
        build_parser().parse_args(["preflight-databento-live"])
    databento_live = build_parser().parse_args(
        ["preflight-databento-live", "--allow-live-entitlement-probe"]
    )

    assert kalshi.command == "preflight-kalshi"
    assert pyth.command == "preflight-pyth"
    assert pyth.metal == "silver"
    assert pyth.history_at_seconds == 1_780_000_000
    assert pyth.timeout_seconds == 10.0
    assert pyth_live.metal == "silver"
    assert pyth_live.duration_seconds == 5.0
    assert databento.command == "preflight-databento"
    assert databento.metal == "silver"
    assert databento.probe_date == "2026-08-28"
    assert databento_live.metal == "silver"
    assert databento_live.duration_seconds == 5.0


@pytest.mark.asyncio
async def test_databento_live_preflight_cli_forwards_bound_and_reports_status(
    monkeypatch, tmp_path, capsys
):
    captured = {}

    def preflight(config, *, metal, duration_seconds, allow_live_probe):
        captured.update(
            metal=metal,
            duration_seconds=duration_seconds,
            allow_live_probe=allow_live_probe,
        )
        return {"source": "databento", "scope": "live", "ok": True}

    monkeypatch.setattr(
        cli, "load_metals_config", lambda path: _history_config(tmp_path)
    )
    monkeypatch.setattr(cli, "preflight_databento_live", preflight)
    args = build_parser().parse_args(
        [
            "preflight-databento-live",
            "--metal",
            "silver",
            "--duration-seconds",
            "2",
            "--allow-live-entitlement-probe",
        ]
    )

    assert await cli.async_main(args) == 0
    assert captured == {
        "metal": "silver",
        "duration_seconds": 2.0,
        "allow_live_probe": True,
    }
    assert '"ok": true' in capsys.readouterr().out


@pytest.mark.asyncio
async def test_pyth_live_preflight_cli_forwards_bound_and_reports_status(
    monkeypatch, tmp_path, capsys
):
    captured = {}

    async def preflight(config, *, metal, duration_seconds, allow_live_probe):
        captured.update(
            metal=metal,
            duration_seconds=duration_seconds,
            allow_live_probe=allow_live_probe,
        )
        return {"source": "pyth", "scope": "live", "ok": False}

    monkeypatch.setattr(
        cli,
        "load_metals_config",
        lambda path: _pyth_lazer_config(tmp_path, enabled=False),
    )
    monkeypatch.setattr(cli, "preflight_pyth_lazer_live", preflight)
    args = build_parser().parse_args(
        [
            "preflight-pyth-lazer-live",
            "--metal",
            "silver",
            "--duration-seconds",
            "2",
            "--allow-live-entitlement-probe",
        ]
    )

    assert await cli.async_main(args) == 2
    assert captured == {
        "metal": "silver",
        "duration_seconds": 2.0,
        "allow_live_probe": True,
    }
    assert '"ok": false' in capsys.readouterr().out


def test_databento_history_estimate_is_bounded_and_has_no_download_switch():
    args = build_parser().parse_args(
        [
            "estimate-databento-history",
            "--metal",
            "silver",
            "--start",
            "2026-08-28T00:00:00Z",
            "--end",
            "2026-08-29T00:00:00Z",
        ]
    )

    assert args.command == "estimate-databento-history"
    assert args.start == "2026-08-28T00:00:00Z"
    assert args.end == "2026-08-29T00:00:00Z"
    assert args.schema is None
    assert not hasattr(args, "allow_billable_download")


def test_databento_history_schema_is_repeatable_for_estimate_and_backfill():
    bounds = [
        "--start",
        "2026-08-28T00:00:00Z",
        "--end",
        "2026-08-29T00:00:00Z",
        "--schema",
        "definition",
        "--schema",
        "trades",
        "--schema",
        "status",
        "--schema",
        "statistics",
    ]
    estimated = build_parser().parse_args(["estimate-databento-history", *bounds])
    backfilled = build_parser().parse_args(
        [
            "backfill-databento-history",
            *bounds,
            "--max-cost-usd",
            "1.00",
            "--allow-billable-download",
        ]
    )

    expected = ["definition", "trades", "status", "statistics"]
    assert estimated.schema == expected
    assert backfilled.schema == expected


def test_databento_history_backfill_requires_cost_cap_and_explicit_opt_in():
    base = [
        "backfill-databento-history",
        "--start",
        "2026-08-28T00:00:00Z",
        "--end",
        "2026-08-29T00:00:00Z",
    ]
    with pytest.raises(SystemExit):
        build_parser().parse_args(base)
    with pytest.raises(SystemExit):
        build_parser().parse_args([*base, "--max-cost-usd", "1.00"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                *base,
                "--max-cost-usd",
                "NaN",
                "--allow-billable-download",
            ]
        )

    args = build_parser().parse_args(
        [
            *base,
            "--transport",
            "batch",
            "--max-cost-usd",
            "1.00",
            "--allow-billable-download",
        ]
    )
    assert args.transport == "batch"
    assert args.max_cost_usd == Decimal("1.00")
    assert args.allow_billable_download is True


def test_candlestick_backfill_defaults_to_resumable_silver():
    args = build_parser().parse_args(["backfill-kalshi-candles"])

    assert args.command == "backfill-kalshi-candles"
    assert args.metal == "silver"
    assert args.max_markets == 0
    assert args.no_resume is False


def test_settled_backfill_accepts_an_inclusive_close_window():
    args = build_parser().parse_args(
        [
            "backfill-kalshi",
            "--metal",
            "btc",
            "--from-close-seconds",
            "1785521700",
            "--to-close-seconds",
            "1787976000",
            "--market-stride",
            "8",
        ]
    )

    assert args.metal == "btc"
    assert args.from_close_seconds == 1_785_521_700
    assert args.to_close_seconds == 1_787_976_000
    assert args.market_stride == 8
    assert args.market_stride_offset == 0


def test_s1_feasibility_defaults_are_conservative_and_offline():
    args = build_parser().parse_args(["run-s1-feasibility"])

    assert args.command == "run-s1-feasibility"
    assert args.metal == "silver"
    assert args.train_fraction == Decimal("0.70")
    assert args.min_seconds_to_close == 120
    assert args.penetration_dollars == Decimal("0.01")
    assert args.maker_fee_per_contract == Decimal("0")


def test_toxicity_diagnostic_uses_frozen_test_cutoff():
    args = build_parser().parse_args(["run-toxicity-diagnostics"])

    assert args.command == "run-toxicity-diagnostics"
    assert args.metal == "silver"
    assert args.test_start_ts_ns == 1_787_301_900_000_000_000
    assert args.penetration_dollars == Decimal("0.01")
    assert args.require_complete_trades is False


def _history_config(tmp_path):
    approval_path = materialize_approved_provider_path(tmp_path)
    return MetalsConfig(
        storage=StorageConfig(root=str(tmp_path)),
        cme=CmeConfig(api_key_env="TEST_DATABENTO_KEY"),
        governance=replace(
            MetalsConfig().governance,
            provider_approval_path=approval_path,
        ),
        products={
            "silver": ProductConfig(
                enabled=True,
                cme_symbols=["SI", "SIL", "SIC"],
            )
        },
    )


def _estimate(request_id):
    return HistoricalCostEstimate(
        request_id=request_id,
        schemas=("definition", "mbo", "trades", "status", "statistics"),
        per_schema_cost_usd={"definition": 0.01, "mbo": 0.49},
        total_cost_usd=0.5,
        estimated_at=datetime.now(timezone.utc).isoformat(),
    )


def test_databento_history_request_defaults_to_config_and_rejects_bad_subsets(
    tmp_path,
):
    config = _history_config(tmp_path)
    defaulted = cli._databento_history_request(
        config,
        "silver",
        "2026-08-28T00:00:00Z",
        "2026-08-29T00:00:00Z",
    )
    selected = cli._databento_history_request(
        config,
        "silver",
        "2026-08-28T00:00:00Z",
        "2026-08-29T00:00:00Z",
        ["definition", "trades", "status", "statistics"],
    )

    assert set(defaulted.schemas) == set(config.cme.schemas)
    assert selected.schemas == ("definition", "trades", "status", "statistics")
    with pytest.raises(ValueError, match="at least one"):
        cli._databento_history_request(
            config,
            "silver",
            "2026-08-28T00:00:00Z",
            "2026-08-29T00:00:00Z",
            [],
        )
    with pytest.raises(ValueError, match="unique"):
        cli._databento_history_request(
            config,
            "silver",
            "2026-08-28T00:00:00Z",
            "2026-08-29T00:00:00Z",
            ["trades", "TRADES"],
        )
    with pytest.raises(ValueError, match="not configured"):
        cli._databento_history_request(
            replace(config, cme=replace(config.cme, schemas=["definition", "trades"])),
            "silver",
            "2026-08-28T00:00:00Z",
            "2026-08-29T00:00:00Z",
            ["mbo"],
        )


@pytest.mark.asyncio
async def test_estimate_command_calls_only_cost_preflight(
    monkeypatch, tmp_path, capsys
):
    class EstimateOnlyArchive:
        def __init__(self):
            self.requests = []

        def preflight(self, request):
            self.requests.append(request)
            return _estimate(request.request_id)

        def download_stream(self, *args, **kwargs):
            raise AssertionError("estimate command must not download")

        def submit_batch(self, *args, **kwargs):
            raise AssertionError("estimate command must not submit a batch job")

    archive = EstimateOnlyArchive()
    monkeypatch.setattr(
        cli, "load_metals_config", lambda path: _history_config(tmp_path)
    )
    monkeypatch.setattr(
        cli, "_databento_history_archive", lambda config, **kwargs: archive
    )
    args = build_parser().parse_args(
        [
            "estimate-databento-history",
            "--start",
            "2026-08-28T00:00:00Z",
            "--end",
            "2026-08-29T00:00:00Z",
            "--schema",
            "definition",
            "--schema",
            "trades",
            "--schema",
            "status",
            "--schema",
            "statistics",
        ]
    )

    assert await cli.async_main(args) == 0
    output = capsys.readouterr().out
    assert '"total_cost_usd": 0.5' in output
    assert len(archive.requests) == 1
    assert archive.requests[0].schemas == (
        "definition",
        "trades",
        "status",
        "statistics",
    )
    assert "mbo" not in archive.requests[0].schemas


@pytest.mark.asyncio
async def test_databento_backfill_requires_approval_before_archive_or_billable_call(
    monkeypatch, tmp_path
):
    config = _history_config(tmp_path)
    config = replace(
        config,
        governance=replace(config.governance, provider_approval_path=""),
    )
    monkeypatch.setattr(cli, "load_metals_config", lambda path: config)

    def unexpected_archive(*args, **kwargs):
        raise AssertionError("approval gate must run before archive construction")

    monkeypatch.setattr(cli, "_databento_history_archive", unexpected_archive)

    with pytest.raises(HistoricalResearchAssumptionError, match="path is required"):
        await cli._backfill_databento_history(
            "unused.toml",
            "silver",
            "2026-08-28T00:00:00Z",
            "2026-08-29T00:00:00Z",
            "stream",
            Decimal("0.75"),
            True,
        )


@pytest.mark.asyncio
async def test_stream_backfill_forwards_explicit_cost_guard(
    monkeypatch, tmp_path, capsys
):
    class StreamArchive:
        def __init__(self):
            self.call = None

        def download_stream(self, request, **kwargs):
            self.call = (request, kwargs)
            return HistoricalArchiveResult(
                request_id=request.request_id,
                transport="stream",
                completed=True,
                record_counts={"definition": 1},
                skipped_schemas=(),
                output_paths=(Path(tmp_path) / "definition.dbn.zst",),
                estimate=_estimate(request.request_id),
            )

    archive = StreamArchive()
    factory_kwargs = {}
    monkeypatch.setattr(
        cli, "load_metals_config", lambda path: _history_config(tmp_path)
    )

    def archive_factory(config, **kwargs):
        factory_kwargs.update(kwargs)
        return archive

    monkeypatch.setattr(cli, "_databento_history_archive", archive_factory)
    args = build_parser().parse_args(
        [
            "backfill-databento-history",
            "--start",
            "2026-08-28T00:00:00Z",
            "--end",
            "2026-08-29T00:00:00Z",
            "--max-cost-usd",
            "0.75",
            "--allow-billable-download",
        ]
    )

    assert await cli.async_main(args) == 0
    _, kwargs = archive.call
    assert kwargs == {
        "allow_billable_download": True,
        "max_cost_usd": 0.75,
    }
    assert factory_kwargs["provider_approval"].approval_id
    assert '"transport": "stream"' in capsys.readouterr().out


@pytest.mark.asyncio
async def test_batch_backfill_submits_then_resumes_same_bounded_request(
    monkeypatch, tmp_path
):
    class BatchArchive:
        def __init__(self):
            self.submitted = None
            self.ingested = None

        def submit_batch(self, request, **kwargs):
            self.submitted = (request, kwargs)
            return HistoricalArchiveResult(
                request_id=request.request_id,
                transport="batch",
                completed=False,
                record_counts={},
                skipped_schemas=(),
                output_paths=(),
                batch_states={"definition": "queued"},
            )

        def ingest_submitted_batch(self, request):
            self.ingested = request
            return HistoricalArchiveResult(
                request_id=request.request_id,
                transport="batch",
                completed=False,
                record_counts={},
                skipped_schemas=(),
                output_paths=(),
                batch_states={"definition": "queued"},
            )

    archive = BatchArchive()
    monkeypatch.setattr(
        cli, "load_metals_config", lambda path: _history_config(tmp_path)
    )
    monkeypatch.setattr(
        cli, "_databento_history_archive", lambda config, **kwargs: archive
    )
    args = build_parser().parse_args(
        [
            "backfill-databento-history",
            "--start",
            "1787875200000000000",
            "--end",
            "1787961600000000000",
            "--transport",
            "batch",
            "--max-cost-usd",
            "2.00",
            "--allow-billable-download",
        ]
    )

    assert await cli.async_main(args) == 0
    request, kwargs = archive.submitted
    assert request.start == 1_787_875_200_000_000_000
    assert kwargs == {
        "allow_billable_download": True,
        "max_cost_usd": 2.0,
    }
    assert archive.ingested.request_id == request.request_id
