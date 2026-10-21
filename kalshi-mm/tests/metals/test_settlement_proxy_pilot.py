import json
from decimal import Decimal

import pytest

from src.metals.settlement_proxy_pilot import (
    SettlementProxyPilotConfig,
    SettlementProxyPilotError,
    run_settlement_proxy_pilot,
)
from src.metals.storage import HashChainedJsonlWriter

NSEC = 1_000_000_000
START = 1_800_000_000 * NSEC


def _write(root, relative, rows):
    path = root / relative
    with HashChainedJsonlWriter(path, fsync_every_records=100) as writer:
        for row in rows:
            writer.write(row)
    return path


def test_small_proxy_pilot_is_reproducible_and_never_certifies_settlement(tmp_path):
    labels = []
    trades = []
    proxy = []
    levels = [
        Decimal("30.000"),
        Decimal("30.010"),
        Decimal("29.990"),
        Decimal("30.020"),
        Decimal("30.040"),
    ]
    for index in range(4):
        open_ns = START + index * 900 * NSEC
        close_ns = open_ns + 900 * NSEC
        labels.append(
            {
                "metal": "silver",
                "market_ticker": f"SYNTHETIC-{index}",
                "open_ts_ns": open_ns,
                "close_ts_ns": close_ns,
                "target_price": str(levels[index]),
                "expiration_value": str(levels[index + 1]),
                "strike_type": "greater_or_equal",
                "result": "yes" if levels[index + 1] >= levels[index] else "no",
            }
        )
        for boundary, level in (
            (open_ns, levels[index]),
            (close_ns, levels[index + 1]),
        ):
            for asset, symbol, scale in (("SI", "SIZ6", 1), ("SIL", "SILZ6", 2)):
                trades.append(
                    {
                        "asset": asset,
                        "contract_symbol": symbol,
                        "source_event_ts_ns": boundary - NSEC,
                        "price": str(level * scale),
                        "size": 2,
                    }
                )
        open_second = open_ns // NSEC
        close_second = close_ns // NSEC
        proxy.extend(
            [
                {
                    "metal": "silver",
                    "timestamp_seconds": open_second - 60,
                    "close": str(levels[index]),
                    "settlement_eligible": False,
                },
                {
                    "metal": "silver",
                    "timestamp_seconds": close_second - 60,
                    "close": str(levels[index + 1]),
                    "settlement_eligible": False,
                },
            ]
        )

    labels_path = _write(tmp_path, "inputs/labels.jsonl", labels)
    trades_path = _write(tmp_path, "inputs/trades.jsonl", trades)
    proxy_path = _write(tmp_path, "inputs/proxy.jsonl", proxy)
    config = SettlementProxyPilotConfig(
        start_ts_ns=START,
        end_ts_ns=START + 4 * 900 * NSEC,
    )

    first = run_settlement_proxy_pilot(
        tmp_path, labels_path, trades_path, proxy_path, config
    )
    second = run_settlement_proxy_pilot(
        tmp_path, labels_path, trades_path, proxy_path, config
    )

    assert first["pilot_id"] == second["pilot_id"]
    assert first["cohort"]["included_windows"] == 4
    assert first["cohort"]["quarantined_windows"] == 0
    assert first["selected_contracts"]["primary"]["symbol"] == "SIZ6"
    assert first["metrics"]["comex_primary_last_delta"]["observations"] == 4
    assert first["exact_settlement_certified"] is False
    assert first["formal_settlement_gate_passed"] is False
    assert (
        json.loads(
            (
                tmp_path / "manifests" / f"run={first['run_id']}" / "manifest.json"
            ).read_text()
        )["status"]
        == "completed"
    )


def test_generic_proxy_must_be_explicitly_ineligible_for_settlement(tmp_path):
    labels_path = _write(
        tmp_path,
        "inputs/labels.jsonl",
        [
            {
                "metal": "silver",
                "market_ticker": "SYNTHETIC",
                "open_ts_ns": START,
                "close_ts_ns": START + 900 * NSEC,
                "target_price": "30.000",
                "expiration_value": "30.001",
                "strike_type": "greater_or_equal",
                "result": "yes",
            }
        ],
    )
    trades_path = _write(
        tmp_path,
        "inputs/trades.jsonl",
        [
            {
                "asset": asset,
                "contract_symbol": symbol,
                "source_event_ts_ns": START + NSEC,
                "price": "30",
                "size": 1,
            }
            for asset, symbol in (("SI", "SIZ6"), ("SIL", "SILZ6"))
        ],
    )
    proxy_path = _write(
        tmp_path,
        "inputs/proxy.jsonl",
        [{"metal": "silver", "timestamp_seconds": START // NSEC, "close": "30"}],
    )

    with pytest.raises(SettlementProxyPilotError, match="settlement_eligible=false"):
        run_settlement_proxy_pilot(
            tmp_path,
            labels_path,
            trades_path,
            proxy_path,
            SettlementProxyPilotConfig(
                start_ts_ns=START,
                end_ts_ns=START + 900 * NSEC,
            ),
        )
