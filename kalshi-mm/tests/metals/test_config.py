from pathlib import Path

import pytest

from src.metals.config import load_metals_config


def test_example_config_is_silver_only_and_gold_ready():
    config = load_metals_config("configs/metals.example.toml")

    assert set(config.enabled_products) == {"silver"}
    assert config.products["silver"].pyth_symbol == "Metal.Index.SILVER/USD"
    assert config.products["silver"].cme_symbols == ["SI", "SIL", "SIC"]
    assert config.products["gold"].enabled is False
    assert config.products["gold"].pyth_symbol == "Metal.Index.GOLD/USD"
    assert config.pyth.enabled is False
    assert len(config.pyth.lazer_ws_endpoints) == 3
    assert config.kalshi.fractional_contract_quantum == "0.01"
    assert config.shadow.execution_mode == "shadow"
    assert config.shadow.allow_live_orders is False
    assert config.shadow.max_contracts_per_order == "250"
    assert config.shadow.max_net_contracts_per_market == "1000"
    assert config.shadow.max_aggregate_worst_case_loss_dollars == "3000"
    assert config.cme.dataset == "GLBX.MDP3"
    assert config.cme.stype_in == "parent"
    assert config.cme.schemas == [
        "mbo",
        "trades",
        "definition",
        "status",
        "statistics",
    ]
    assert config.cme.include_ts_out is True
    assert config.clock.enabled is False
    assert config.clock.provider == "chrony"
    assert config.clock.sample_interval_seconds == 1.0
    assert config.governance.provider_approval_path == ""
    assert config.governance.pyth_historical_research_assumption_path == ""
    assert config.governance.databento_historical_research_assumption_path == ""


def test_btc_control_config_is_isolated_read_only_and_public_data_only():
    config = load_metals_config("configs/btc_control.example.toml")

    assert set(config.enabled_products) == {"btc"}
    assert config.storage.root == "data/btc-control"
    assert config.products["btc"].series_tickers == ["KXBTC15M"]
    assert config.pyth.enabled is False
    assert config.cme.enabled is False
    assert config.shadow.enabled is False
    assert config.shadow.execution_mode == "shadow"
    assert config.shadow.allow_live_orders is False


def test_public_snapshot_contains_secret_names_not_secret_values():
    config = load_metals_config("configs/metals.example.toml")
    snapshot = config.public_snapshot()

    assert snapshot["kalshi"]["api_key_id_env"] == "KALSHI_API_KEY_ID"
    assert "api_key_id" not in snapshot["kalshi"]
    assert "private_key" not in snapshot["kalshi"]
    assert snapshot["shadow"]["config_version"] == "metals-shadow-v1"
    assert snapshot["clock"]["provider"] == "chrony"
    assert snapshot["clock"]["max_probe_elapsed_delta_ns"] == 1_000_000
    assert snapshot["governance"]["provider_approval_path"] == ""
    assert snapshot["governance"]["pyth_historical_research_assumption_path"] == ""
    assert snapshot["governance"]["databento_historical_research_assumption_path"] == ""


@pytest.mark.parametrize(
    "field",
    [
        "provider_approval_path",
        "pyth_historical_research_assumption_path",
        "databento_historical_research_assumption_path",
    ],
)
def test_config_rejects_nul_in_governance_artifact_paths(tmp_path: Path, field: str):
    config_path = tmp_path / "metals.toml"
    config_path.write_bytes(
        b"[products.silver]\nenabled = true\n[governance]\n"
        + f'{field} = "bad\\u0000path"\n'.encode()
    )

    with pytest.raises(ValueError, match="governance"):
        load_metals_config(str(config_path))


@pytest.mark.parametrize(
    "shadow_lines",
    [
        'execution_mode = "production"\nallow_live_orders = false',
        'execution_mode = "shadow"\nallow_live_orders = true',
    ],
)
def test_config_refuses_live_order_modes(tmp_path: Path, shadow_lines: str):
    config_path = tmp_path / "metals.toml"
    config_path.write_text(
        "\n".join(
            [
                "[products.silver]",
                "enabled = true",
                "",
                "[shadow]",
                shadow_lines,
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_metals_config(str(config_path))


@pytest.mark.parametrize(
    "clock_lines",
    [
        'provider = "unknown"',
        "sample_interval_seconds = 0",
        'sample_interval_seconds = "nan"',
        "probe_timeout_seconds = 31",
        "max_probe_elapsed_delta_ns = -1",
        'chronyc_path = ""',
    ],
)
def test_config_rejects_unbounded_or_unknown_clock_probes(
    tmp_path: Path, clock_lines: str
):
    config_path = tmp_path / "metals.toml"
    config_path.write_text(
        "\n".join(
            [
                "[products.silver]",
                "enabled = true",
                "",
                "[clock]",
                clock_lines,
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="clock"):
        load_metals_config(str(config_path))
