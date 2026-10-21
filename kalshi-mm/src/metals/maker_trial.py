"""An expiring operator research assumption, never a provider license approval.

Only the frozen-model Silver maker shadow may use this record. Raw-data retention
has a review deadline, not a promise of indefinite rights or automatic deletion.
The content hash detects mutation; it is not a provider signature or permission.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any

from .config import MetalsConfig
from .storage import _atomic_write_json

NSEC = 1_000_000_000
DAY_NS = 86400 * NSEC
SCHEMA_VERSION = "maker-trial-research/v1"
PUBLIC_SOURCES = (
    "https://docs.pyth.network/price-feeds/pro/mcp",
    "https://docs.pyth.network/price-feeds/pro/mcp-skills",
    "https://docs.pyth.network/price-feeds/pro/pyth-terminal",
    "https://docs.pyth.network/price-feeds/pro/subscribe-to-prices",
    "https://www.dourolabs.xyz/Subscription-Terms.pdf",
)
OFFICIAL_ENDPOINTS = frozenset(
    f"wss://pyth-lazer-{number}.dourolabs.app/v1/stream" for number in range(3)
)
REQUIRED_FIELDS = frozenset({
    "schema_version", "accepted_at_ns", "collection_not_after_ns",
    "retention_review_at_ns", "accepted_by", "acceptance_text", "provider",
    "feed_id", "symbol", "purpose", "no_orders", "research_only",
    "no_redistribution", "provider_permission_confirmed",
    "provider_trial_expires_at_ns", "sources",
})


def _timestamp(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer timestamp")
    return value


def _validate_payload(payload: dict[str, Any]) -> None:
    if type(payload) is not dict or set(payload) != REQUIRED_FIELDS:
        raise ValueError("trial research fields do not match the required schema")
    constants = {
        "schema_version": SCHEMA_VERSION, "accepted_by": "project_owner",
        "provider": "pyth", "feed_id": 345, "symbol": "Metal.XAG/USD",
        "purpose": "frozen_model_maker_shadow", "no_orders": True,
        "research_only": True, "no_redistribution": True,
        "provider_permission_confirmed": False,
    }
    for key, expected in constants.items():
        if type(payload[key]) is not type(expected) or payload[key] != expected:
            raise ValueError(f"trial research scope requires {key}={expected!r}")
    if type(payload["acceptance_text"]) is not str or not payload["acceptance_text"].strip():
        raise ValueError("acceptance_text must preserve the operator's statement")
    accepted = _timestamp(payload["accepted_at_ns"], "accepted_at_ns")
    deadline = _timestamp(payload["collection_not_after_ns"], "collection_not_after_ns")
    review = _timestamp(payload["retention_review_at_ns"], "retention_review_at_ns")
    if not accepted < deadline <= accepted + DAY_NS:
        raise ValueError("collection deadline must be within 24 hours of acceptance")
    if not deadline < review <= accepted + 7 * DAY_NS:
        raise ValueError("retention review must follow collection and be within 7 days")
    expiry = payload["provider_trial_expires_at_ns"]
    if expiry is not None and _timestamp(expiry, "provider_trial_expires_at_ns") <= accepted:
        raise ValueError("known trial expiry must be after acceptance")
    sources = payload["sources"]
    if (type(sources) is not list or any(type(value) is not str for value in sources)
            or len(sources) != len(PUBLIC_SOURCES) or set(sources) != set(PUBLIC_SOURCES)):
        raise ValueError("trial research sources must be the fixed public documentation")


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class TrialResearchPermit:
    _payload_json: str
    sha256: str
    path: Path | None = None

    def __post_init__(self) -> None:
        payload = json.loads(self._payload_json)
        _validate_payload(payload)
        canonical = _canonical(payload)
        if (self._payload_json != canonical
                or self.sha256 != hashlib.sha256(canonical.encode()).hexdigest()):
            raise ValueError("trial research record hash mismatch")
        if self.path is not None and not isinstance(self.path, Path):
            raise ValueError("trial research artifact path must be a Path")

    @classmethod
    def from_dict(cls, value: dict[str, Any], path: Path | None = None) -> TrialResearchPermit:
        if type(value) is not dict:
            raise ValueError("trial research record must be an object")
        payload = dict(value)
        claimed = payload.pop("record_sha256", None)
        _validate_payload(payload)
        canonical = _canonical(payload)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        if "record_sha256" in value and claimed != digest:
            raise ValueError("trial research record hash mismatch")
        return cls(canonical, digest, path)

    @classmethod
    def load(cls, path: Path | str) -> TrialResearchPermit:
        resolved = Path(path).resolve()
        value = json.loads(resolved.read_text(encoding="utf-8"))
        if type(value) is not dict or "record_sha256" not in value:
            raise ValueError("trial research artifact requires a content hash")
        return cls.from_dict(value, path=resolved)

    @property
    def trial_research_id(self) -> str:
        return self.sha256

    def public_dict(self) -> dict[str, Any]:
        return {**json.loads(self._payload_json), "record_sha256": self.sha256}

    def require_current(self, now_ns: int | None = None) -> None:
        now = time.time_ns() if now_ns is None else _timestamp(now_ns, "now_ns")
        value = self.public_dict()
        if now < value["accepted_at_ns"]:
            raise ValueError("trial research acceptance is not yet effective")
        expiry = value["provider_trial_expires_at_ns"]
        if expiry is not None and now >= expiry:
            raise ValueError("known provider trial has expired")
        if now >= value["collection_not_after_ns"]:
            raise ValueError("local trial research collection deadline has expired")

    def validate_scope(self, config: MetalsConfig, metal: str) -> None:
        if (config.shadow.enabled is not True or config.shadow.execution_mode != "shadow"
                or config.shadow.allow_live_orders is not False):
            raise ValueError("trial research permits shadow execution only, with no live orders")
        if config.pyth.enabled is not True or config.cme.enabled is not False:
            raise ValueError("trial research permits only the enabled Pyth source")
        if set(config.enabled_products) != {metal}:
            raise ValueError("trial research requires exactly one enabled XAG product")
        product = config.products[metal]
        if (product.enabled is not True or type(product.pyth_lazer_id) is not int
                or product.pyth_lazer_id != 345 or product.pyth_symbol != "Metal.XAG/USD"):
            raise ValueError("trial research requires Metal.XAG/USD feed 345")
        endpoints = config.pyth.lazer_ws_endpoints
        if (len(endpoints) != 3 or any(type(endpoint) is not str for endpoint in endpoints)
                or set(endpoints) != OFFICIAL_ENDPOINTS
                or config.pyth.endpoint != "https://pyth.dourolabs.app/v1"):
            raise ValueError("trial research requires the official three Pyth endpoints")

    def require_duration(self, seconds: int | float | Decimal, now_ns: int | None = None) -> None:
        if type(seconds) not in (int, float, Decimal):
            raise ValueError("trial research duration must be a finite positive number")
        duration = Decimal(str(seconds))
        if not duration.is_finite() or not 0 < duration <= 900:
            raise ValueError("trial research duration must be in (0, 900] seconds")
        now = time.time_ns() if now_ns is None else _timestamp(now_ns, "now_ns")
        self.require_current(now)
        value = self.public_dict()
        deadlines = [value["collection_not_after_ns"]]
        if value["provider_trial_expires_at_ns"] is not None:
            deadlines.append(value["provider_trial_expires_at_ns"])
        duration_ns = int((duration * NSEC).to_integral_value(rounding=ROUND_CEILING))
        if now + duration_ns + 3 * NSEC > min(deadlines):
            raise ValueError("trial research duration and 3-second drain exceed collection deadline")


def materialize_trial_research(path: Path | str, payload: dict[str, Any]) -> TrialResearchPermit:
    """Exclusively publish reviewed operator facts; never overwrite a record."""
    permit = TrialResearchPermit.from_dict(payload)
    target = Path(path)
    _atomic_write_json(target, permit.public_dict(), create=True)
    return TrialResearchPermit.load(target)


materialize = materialize_trial_research
