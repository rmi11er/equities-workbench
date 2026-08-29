"""Pyth Pro History access for exact settlement-index candles."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import aiohttp

from .config import MetalsConfig
from .storage import PartitionedRawStore, update_run_manifest, write_run_manifest


class PythAccessError(RuntimeError):
    pass


@dataclass(frozen=True)
class PythHistoryResponse:
    receive_ts_ns: int
    local_monotonic_ts_ns: int
    raw_body: str
    payload: Dict[str, Any]
    params: Dict[str, Any]


def normalize_tradingview_history(
    payload: Dict[str, Any], symbol: str, receive_ts_ns: int
) -> List[Dict[str, Any]]:
    """Convert TradingView parallel arrays without guessing `t` boundary meaning."""

    status = payload.get("s")
    if status not in (None, "ok"):
        if status == "no_data":
            return []
        raise PythAccessError("Pyth History returned status {!r}".format(status))
    columns = {name: payload.get(name, []) for name in ("t", "o", "h", "l", "c", "v")}
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise PythAccessError("Pyth History arrays have inconsistent lengths")

    rows = []
    for index in range(next(iter(lengths), 0)):
        rows.append(
            {
                "pyth_symbol": symbol,
                "timestamp_seconds": int(columns["t"][index]),
                "open": str(columns["o"][index]),
                "high": str(columns["h"][index]),
                "low": str(columns["l"][index]),
                "close": str(columns["c"][index]),
                "volume": str(columns["v"][index]),
                "resolution_minutes": 1,
                "timestamp_semantics": "unconfirmed_tradingview_t",
                "receive_ts_ns": receive_ts_ns,
            }
        )
    return rows


class PythHistoryClient:
    def __init__(self, endpoint: str, api_key: str, timeout_seconds: float = 30.0):
        if not api_key:
            raise PythAccessError("Pyth API key is empty")
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    async def get_history(
        self, symbol: str, start_seconds: int, end_seconds: int
    ) -> PythHistoryResponse:
        if end_seconds < start_seconds:
            raise ValueError("Pyth History end must be at or after start")
        params = {
            "symbol": symbol,
            "from": start_seconds,
            "to": end_seconds,
            "resolution": 1,
        }
        headers = {"Authorization": "Bearer {}".format(self.api_key)}
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        url = self.endpoint + "/fixed_rate@200ms/history"
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params, headers=headers) as response:
                raw_body = await response.text()
                receive_ts_ns = time.time_ns()
                local_monotonic_ts_ns = time.monotonic_ns()
                try:
                    payload = json.loads(raw_body, parse_float=Decimal)
                except json.JSONDecodeError as error:
                    raise PythAccessError(
                        "Pyth History returned non-JSON HTTP {}".format(response.status)
                    ) from error
                if response.status >= 400:
                    raise PythAccessError(
                        "Pyth History HTTP {}: {}".format(response.status, payload)
                    )
        return PythHistoryResponse(
            receive_ts_ns=receive_ts_ns,
            local_monotonic_ts_ns=local_monotonic_ts_ns,
            raw_body=raw_body,
            payload=payload,
            params=params,
        )


async def backfill_pyth_history(
    config: MetalsConfig,
    metal: str,
    start_seconds: int,
    end_seconds: int,
) -> Dict[str, int]:
    product = config.products.get(metal)
    if product is None or not product.pyth_symbol:
        raise ValueError("No Pyth symbol configured for {}".format(metal))
    api_key = os.environ.get(config.pyth.api_key_env, "").strip()
    if not api_key:
        raise PythAccessError(
            "Set {} to a Pyth Pro trial or licensed key".format(
                config.pyth.api_key_env
            )
        )

    raw_store = PartitionedRawStore(
        Path(config.storage.root),
        fsync_every_records=config.storage.fsync_every_records,
        tier="raw",
    )
    normalized_store = PartitionedRawStore(
        Path(config.storage.root),
        run_id=raw_store.run_id,
        fsync_every_records=config.storage.fsync_every_records,
        tier="normalized",
    )
    write_run_manifest(
        Path(config.storage.root), raw_store.run_id, config.public_snapshot()
    )
    client = PythHistoryClient(config.pyth.endpoint, api_key)
    counts = {"requests": 0, "candles": 0}
    cursor = start_seconds
    failure: Optional[BaseException] = None
    try:
        while cursor <= end_seconds:
            chunk_end = min(
                end_seconds,
                cursor + config.pyth.history_chunk_seconds - 1,
            )
            response = await client.get_history(
                product.pyth_symbol, cursor, chunk_end
            )
            raw_store.write(
                source="pyth",
                stream="history_1m",
                metal=metal,
                receive_ts_ns=response.receive_ts_ns,
                record={
                    "source": "pyth",
                    "stream": "history_1m",
                    "receive_ts_ns": response.receive_ts_ns,
                    "local_monotonic_ts_ns": response.local_monotonic_ts_ns,
                    "params": response.params,
                    "raw_body": response.raw_body,
                },
            )
            rows = normalize_tradingview_history(
                response.payload, product.pyth_symbol, response.receive_ts_ns
            )
            for row in rows:
                normalized_store.write(
                    source="pyth_candle",
                    stream="pyth_candle",
                    metal=metal,
                    receive_ts_ns=response.receive_ts_ns,
                    record=row,
                )
            counts["requests"] += 1
            counts["candles"] += len(rows)
            cursor = chunk_end + 1
    except BaseException as error:
        failure = error
        raise
    finally:
        close_failure: Optional[BaseException] = None
        for store in (raw_store, normalized_store):
            try:
                store.close()
            except BaseException as close_error:
                if close_failure is None:
                    close_failure = close_error
        update_run_manifest(
            Path(config.storage.root),
            raw_store.run_id,
            status=(
                "failed"
                if failure is not None or close_failure is not None
                else "completed"
            ),
            counts=counts,
            output_paths=raw_store.paths + normalized_store.paths,
            errors=[
                item
                for item in (failure, close_failure)
                if item is not None
            ],
        )
        if failure is None and close_failure is not None:
            raise close_failure
    return counts
