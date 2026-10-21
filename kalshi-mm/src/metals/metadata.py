"""Market metadata normalization and rule-relevant version tracking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

from .timeutil import iso8601_to_ns
from .types import MarketMetadataVersion


def _first(raw: Dict[str, Any], *names: str) -> Any:
    for name in names:
        value = raw.get(name)
        if value is not None and value != "":
            return value
    return None


def _string(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _integer(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _boolean(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
    return bool(value)


RULE_FIELDS = (
    "market_ticker",
    "event_ticker",
    "series_ticker",
    "open_ts_ns",
    "close_ts_ns",
    "determination_ts_ns",
    "target_price",
    "round_digits",
    "strike_type",
    "settlement_source_name",
    "settlement_source_url",
    "rules_primary",
    "rules_secondary",
    "pyth_symbol",
    "pyth_feed_id",
    "price_level_structure",
    "price_ranges_json",
    "fractional_enabled",
    "exchange_index",
)


def normalize_market_metadata(
    raw: Dict[str, Any],
    metal: str,
    receive_ts_ns: int,
    metadata_version: int,
    fractional_capability: Optional[bool] = None,
) -> MarketMetadataVersion:
    """Normalize known fields and expose absent audit fields instead of guessing."""

    ticker = _string(_first(raw, "ticker", "market_ticker")) or ""
    event_ticker = _string(raw.get("event_ticker")) or ""
    series_ticker = _string(raw.get("series_ticker")) or ""
    price_ranges = _first(raw, "price_ranges", "valid_price_ranges")
    price_ranges_json = json.dumps(
        price_ranges if price_ranges is not None else [],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    custom_strike = raw.get("custom_strike") or {}
    if not isinstance(custom_strike, dict):
        custom_strike = {}
    explicit_fractional = _boolean(
        _first(
            raw,
            "fractional_enabled",
            "fractional_trading_enabled",
            "fractional_contracts_enabled",
        )
    )

    values: Dict[str, Any] = {
        "market_ticker": ticker,
        "event_ticker": event_ticker,
        "series_ticker": series_ticker,
        "metal": metal,
        "open_ts_ns": iso8601_to_ns(_first(raw, "open_time", "open_ts", "start_time")),
        "close_ts_ns": iso8601_to_ns(
            _first(raw, "close_time", "expiration_time", "close_ts")
        ),
        # expected_expiration_time is not determination time. Populate only from
        # an actual prospective lifecycle determination event.
        "determination_ts_ns": iso8601_to_ns(
            _first(raw, "determination_time", "determination_ts")
        ),
        "target_price": _string(
            _first(
                raw,
                "target_price",
                "strike_value",
                "floor_strike",
                "functional_strike",
            )
        ),
        "round_digits": _integer(
            _first(raw, "round_digits", "settlement_round_digits")
            or custom_strike.get("round_digits")
        ),
        "strike_type": _string(
            _first(raw, "strike_type", "comparison", "functional_strike")
        ),
        "settlement_source_name": _string(
            _first(raw, "settlement_source_name", "settlement_source")
        ),
        "settlement_source_url": _string(
            _first(raw, "settlement_source_url", "settlement_url")
        ),
        "rules_primary": _string(raw.get("rules_primary")),
        "rules_secondary": _string(raw.get("rules_secondary")),
        "pyth_symbol": _string(_first(raw, "pyth_symbol", "price_source_symbol")),
        "pyth_feed_id": _string(_first(raw, "pyth_feed_id", "price_source_id")),
        "price_level_structure": _string(
            _first(raw, "price_level_structure", "price_level_structure_name")
        ),
        "price_ranges_json": price_ranges_json,
        "fractional_enabled": (
            explicit_fractional
            if explicit_fractional is not None
            else fractional_capability
        ),
        "exchange_index": _integer(
            _first(raw, "exchange_index", "execution_lane", "shard")
        ),
    }

    required = {
        "market_ticker": values["market_ticker"],
        "event_ticker": values["event_ticker"],
        "series_ticker": values["series_ticker"],
        "open_ts_ns": values["open_ts_ns"],
        "close_ts_ns": values["close_ts_ns"],
        "target_price": values["target_price"],
        "round_digits": values["round_digits"],
        "strike_type": values["strike_type"],
        "settlement_source_name": values["settlement_source_name"],
        "settlement_source_url": values["settlement_source_url"],
        "rules_primary": values["rules_primary"],
        "rules_secondary": values["rules_secondary"],
        "price_level_structure": values["price_level_structure"],
        "fractional_enabled": values["fractional_enabled"],
    }
    missing = sorted(name for name, value in required.items() if value in (None, ""))
    hash_payload = {name: values[name] for name in RULE_FIELDS}
    metadata_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    return MarketMetadataVersion(
        **values,
        metadata_version=metadata_version,
        metadata_hash=metadata_hash,
        metadata_receive_ts_ns=receive_ts_ns,
        missing_required_fields=missing,
    )


class MetadataRegistry:
    """Version market metadata when any rule-relevant field changes."""

    def __init__(self, fractional_capability: Optional[bool] = None) -> None:
        self._versions: Dict[str, Tuple[str, int]] = {}
        self._fractional_capability = fractional_capability

    def observe(
        self, raw: Dict[str, Any], metal: str, receive_ts_ns: int
    ) -> Tuple[MarketMetadataVersion, bool]:
        ticker = str(_first(raw, "ticker", "market_ticker") or "")
        previous_hash, previous_version = self._versions.get(ticker, ("", 0))
        candidate = normalize_market_metadata(
            raw,
            metal=metal,
            receive_ts_ns=receive_ts_ns,
            metadata_version=previous_version + 1,
            fractional_capability=self._fractional_capability,
        )
        if candidate.metadata_hash == previous_hash:
            return (
                MarketMetadataVersion(
                    **{
                        **asdict(candidate),
                        "metadata_version": previous_version,
                    }
                ),
                False,
            )
        self._versions[ticker] = (candidate.metadata_hash, candidate.metadata_version)
        return candidate, True
