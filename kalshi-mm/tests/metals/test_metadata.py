from decimal import Decimal

from src.metals.metadata import MetadataRegistry, normalize_market_metadata


def _market():
    return {
        "ticker": "KXSILVER15M-TEST",
        "event_ticker": "KXSILVER15M-EVENT",
        "series_ticker": "KXSILVER15M",
        "open_time": "2026-08-29T03:30:00Z",
        "close_time": "2026-08-29T03:45:00Z",
        "expected_expiration_time": "2026-08-29T09:30:00Z",
        "floor_strike": Decimal("66.516"),
        "custom_strike": {"round_digits": "3"},
        "strike_type": "greater_or_equal",
        "settlement_source_name": "Pyth - Silver",
        "settlement_source_url": "https://app.pyth.com/explore/Metal.Index.SILVER%2FUSD",
        "pyth_symbol": "Metal.Index.SILVER/USD",
        "price_level_structure": "tapered_deci_cent",
        "price_ranges": [{"start": "0.0000", "end": "0.1000", "step": "0.0010"}],
        "exchange_index": 0,
    }


def test_metadata_uses_exact_target_nested_round_digits_and_no_fake_determination():
    version = normalize_market_metadata(
        _market(), "silver", receive_ts_ns=123, metadata_version=1, fractional_capability=True
    )

    assert version.target_price == "66.516"
    assert version.round_digits == 3
    assert version.strike_type == "greater_or_equal"
    assert version.determination_ts_ns is None
    assert version.fractional_enabled is True
    assert "determination_ts_ns" not in version.missing_required_fields


def test_registry_versions_only_rule_relevant_changes():
    registry = MetadataRegistry(fractional_capability=True)
    first, changed = registry.observe(_market(), "silver", 100)
    assert changed and first.metadata_version == 1

    duplicate, changed = registry.observe(_market(), "silver", 200)
    assert changed is False
    assert duplicate.metadata_version == 1

    updated = _market()
    updated["floor_strike"] = Decimal("66.517")
    second, changed = registry.observe(updated, "silver", 300)
    assert changed and second.metadata_version == 2
    assert second.metadata_hash != first.metadata_hash
