"""Fixed-point Kalshi Yes/No order-book reconstruction."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any, Dict, Iterable, Optional, Tuple


def decimal_string(value: Any) -> str:
    return format(Decimal(str(value)), "f")


def _parse_level(level: Any) -> Tuple[str, str]:
    if isinstance(level, dict):
        price = level.get("price_dollars", level.get("price"))
        quantity = level.get("quantity_fp", level.get("quantity", level.get("size")))
    else:
        price, quantity = level[0], level[1]
    return decimal_string(price), decimal_string(quantity)


class FixedPointOrderBook:
    """Store only native Yes and No bids; derive asks through complementarity."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.yes_bids: Dict[str, str] = {}
        self.no_bids: Dict[str, str] = {}
        self.valid = False
        self.recovery_snapshot_id: Optional[str] = None

    def invalidate(self) -> None:
        self.valid = False

    def apply_snapshot(self, message: Dict[str, Any], snapshot_id: str) -> None:
        self.yes_bids = {
            price: size
            for price, size in map(
                _parse_level,
                message.get("yes_dollars_fp", message.get("yes", [])),
            )
        }
        self.no_bids = {
            price: size
            for price, size in map(
                _parse_level,
                message.get("no_dollars_fp", message.get("no", [])),
            )
        }
        self.valid = self._structurally_valid()
        self.recovery_snapshot_id = snapshot_id

    def apply_delta(self, message: Dict[str, Any]) -> None:
        if not self.valid:
            return
        side = str(message.get("side", "")).lower()
        if side not in ("yes", "no"):
            self.invalidate()
            return
        price_value = message.get("price_dollars", message.get("price"))
        delta_value = message.get(
            "delta_fp", message.get("delta", message.get("quantity_delta"))
        )
        if price_value is None or delta_value is None:
            self.invalidate()
            return
        price = decimal_string(price_value)
        delta = Decimal(str(delta_value))
        book = self.yes_bids if side == "yes" else self.no_bids
        new_quantity = Decimal(book.get(price, "0")) + delta
        if new_quantity <= 0:
            book.pop(price, None)
        else:
            book[price] = format(new_quantity, "f")
        self.valid = self._structurally_valid()

    def best_yes_bid(self) -> Optional[Decimal]:
        return max(map(Decimal, self.yes_bids)) if self.yes_bids else None

    def best_yes_ask(self) -> Optional[Decimal]:
        if not self.no_bids:
            return None
        return Decimal("1") - max(map(Decimal, self.no_bids))

    def _structurally_valid(self) -> bool:
        bid = self.best_yes_bid()
        ask = self.best_yes_ask()
        if bid is None or ask is None:
            return False
        # A zero-spread (locked) complementary book can occur near resolution.
        # It remains reconstructable even though quote policy must not join it.
        return Decimal("0") < bid <= ask < Decimal("1")

    def state_hash(self) -> str:
        payload = {
            "ticker": self.ticker,
            "yes_bids": sorted(self.yes_bids.items()),
            "no_bids": sorted(self.no_bids.items()),
            "valid": self.valid,
            "recovery_snapshot_id": self.recovery_snapshot_id,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
