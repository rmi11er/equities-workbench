"""Stable types shared by collection, normalization, and replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


class SequenceStatus(str, Enum):
    FIRST = "first"
    OK = "ok"
    DUPLICATE = "duplicate"
    GAP = "gap"


@dataclass(frozen=True)
class SequenceDecision:
    status: SequenceStatus
    expected: Optional[int]
    received: Optional[int]


@dataclass
class SourceEnvelope:
    """Lossless receive-time envelope around one source message."""

    source: str
    stream: str
    ingestion_sequence: int
    local_monotonic_ts_ns: int
    receive_ts_ns: int
    raw: Dict[str, Any]
    raw_text: Optional[str] = None
    source_sequence: Optional[int] = None
    source_event_ts_ns: Optional[int] = None
    source_publish_ts_ns: Optional[int] = None
    connection_id: Optional[str] = None
    subscription_id: Optional[str] = None
    sequence_status: Optional[str] = None
    gap_flag: bool = False
    previous_monotonic_ts_ns: Optional[int] = None
    monotonic_regression: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketMetadataVersion:
    market_ticker: str
    event_ticker: str
    series_ticker: str
    metal: str
    open_ts_ns: Optional[int]
    close_ts_ns: Optional[int]
    determination_ts_ns: Optional[int]
    target_price: Optional[str]
    round_digits: Optional[int]
    strike_type: Optional[str]
    settlement_source_name: Optional[str]
    settlement_source_url: Optional[str]
    rules_primary: Optional[str]
    rules_secondary: Optional[str]
    pyth_symbol: Optional[str]
    pyth_feed_id: Optional[str]
    price_level_structure: Optional[str]
    price_ranges_json: str
    fractional_enabled: Optional[bool]
    exchange_index: Optional[int]
    metadata_version: int
    metadata_hash: str
    metadata_receive_ts_ns: int
    missing_required_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SettlementCheck:
    market_ticker: str
    target_price: str
    raw_final_value: str
    rounded_final_value: str
    round_digits: int
    tie_flag: bool
    predicted_result: str
    exchange_result: Optional[str] = None
    agreement: Optional[bool] = None
    discrepancy_classification: Optional[str] = None

    @property
    def target_decimal(self) -> Decimal:
        return Decimal(self.target_price)
