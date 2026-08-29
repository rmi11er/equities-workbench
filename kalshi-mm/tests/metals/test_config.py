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


def test_public_snapshot_contains_secret_names_not_secret_values():
    config = load_metals_config("configs/metals.example.toml")
    snapshot = config.public_snapshot()

    assert snapshot["kalshi"]["api_key_id_env"] == "KALSHI_API_KEY_ID"
    assert "api_key_id" not in snapshot["kalshi"]
    assert "private_key" not in snapshot["kalshi"]
    assert snapshot["shadow"]["config_version"] == "metals-shadow-v1"


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
                '[products.silver]',
                'enabled = true',
                '',
                '[shadow]',
                shadow_lines,
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_metals_config(str(config_path))
