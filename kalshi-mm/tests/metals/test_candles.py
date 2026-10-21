import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.metals.candles import (
    CandleMarket,
    backfill_kalshi_candlesticks,
    normalize_kalshi_candlestick,
)
from src.metals.config import load_metals_config
from src.metals.kalshi import ReceivedResponse


def _config(tmp_path: Path):
    config = load_metals_config("configs/metals.example.toml")
    return replace(
        config,
        storage=replace(config.storage, root=str(tmp_path / "metals")),
    )


def _response(path, payload, receive_ts_ns=30_000_000_000):
    return ReceivedResponse(
        path=path,
        params={},
        receive_ts_ns=receive_ts_ns,
        local_monotonic_ts_ns=123,
        payload=payload,
        raw_body=json.dumps(payload),
    )


class FakeCandleClient:
    def __init__(self):
        self.paths = []
        self.started = False
        self.closed = False

    async def start(self):
        self.started = True

    async def close(self):
        self.closed = True

    async def get(self, path, params=None):
        self.paths.append((path, dict(params or {})))
        if path == "/historical/cutoff":
            return _response(
                path,
                {"market_settled_ts": "1970-01-01T00:00:10Z"},
            )
        end_seconds = int(params["end_ts"])
        payload = {
            "candlesticks": [
                {
                    "end_period_ts": end_seconds,
                    "yes_bid": {
                        "open_dollars": "0.40",
                        "low_dollars": "0.39",
                        "high_dollars": "0.41",
                        "close_dollars": "0.40",
                    },
                    "yes_ask": {
                        "open_dollars": "0.42",
                        "low_dollars": "0.41",
                        "high_dollars": "0.43",
                        "close_dollars": "0.42",
                    },
                    "price": {"close_dollars": "0.41"},
                    "volume_fp": "12.5",
                    "open_interest_fp": "4",
                }
            ]
        }
        return _response(path, payload, receive_ts_ns=31_000_000_000)


def test_normalizer_supports_historical_field_names():
    market = CandleMarket("TICKER", "SERIES", "silver", 0, 60_000_000_000, 70)
    response = _response("/historical/test", {})
    candle = {
        "end_period_ts": 60,
        "yes_bid": {"open": "0.20", "close": "0.21"},
        "yes_ask": {"open": "0.23", "close": "0.24"},
        "price": {"mean": "0.22", "previous": "0.19"},
        "volume": "5.5",
        "open_interest": "9",
    }

    row = normalize_kalshi_candlestick(candle, market, 1, response, 7)

    assert row["end_period_ts_ns"] == 60_000_000_000
    assert row["yes_bid_close_dollars"] == "0.21"
    assert row["yes_ask_close_dollars"] == "0.24"
    assert row["trade_mean_dollars"] == "0.22"
    assert row["volume_fp"] == "5.5"
    assert row["ingestion_sequence"] == 7


@pytest.mark.asyncio
async def test_backfill_routes_cutoff_persists_and_resumes(tmp_path):
    markets = [
        CandleMarket(
            "OLD", "SERIES", "silver", 1_000_000_000, 2_000_000_000, 5_000_000_000
        ),
        CandleMarket(
            "NEW",
            "SERIES",
            "silver",
            11_000_000_000,
            12_000_000_000,
            20_000_000_000,
        ),
    ]
    first_client = FakeCandleClient()

    counts = await backfill_kalshi_candlesticks(
        _config(tmp_path),
        "silver",
        markets=markets,
        client=first_client,
    )

    assert counts == {
        "markets_considered": 2,
        "markets_requested": 2,
        "markets_skipped": 0,
        "markets_failed": 0,
        "candles": 2,
    }
    assert first_client.started and first_client.closed
    assert first_client.paths[1][0] == "/historical/markets/OLD/candlesticks"
    assert first_client.paths[2][0] == (
        "/series/SERIES/markets/NEW/candlesticks"
    )
    candle_paths = list(
        (tmp_path / "metals" / "normalized" / "kalshi_candlestick").glob(
            "**/*.jsonl"
        )
    )
    assert len(candle_paths) == 1

    second_client = FakeCandleClient()
    resumed = await backfill_kalshi_candlesticks(
        _config(tmp_path),
        "silver",
        markets=markets,
        client=second_client,
    )

    assert resumed["markets_requested"] == 0
    assert resumed["markets_skipped"] == 2
    assert resumed["candles"] == 0
    assert second_client.paths == []
    assert second_client.closed


@pytest.mark.asyncio
async def test_max_markets_applies_after_resume_filter(tmp_path):
    markets = [
        CandleMarket(
            ticker,
            "SERIES",
            "silver",
            index * 1_000_000_000,
            (index + 1) * 1_000_000_000,
            (index + 2) * 1_000_000_000,
        )
        for index, ticker in enumerate(("ONE", "TWO", "THREE"), start=1)
    ]
    await backfill_kalshi_candlesticks(
        _config(tmp_path),
        "silver",
        max_markets=1,
        markets=markets,
        client=FakeCandleClient(),
    )

    second_client = FakeCandleClient()
    counts = await backfill_kalshi_candlesticks(
        _config(tmp_path),
        "silver",
        max_markets=1,
        markets=markets,
        client=second_client,
    )

    assert counts["markets_considered"] == 3
    assert counts["markets_skipped"] == 1
    assert counts["markets_requested"] == 1
    assert second_client.paths[1][0].endswith("/TWO/candlesticks")
