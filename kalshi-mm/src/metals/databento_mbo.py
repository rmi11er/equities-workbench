"""Strict Databento MBO reconstruction for historical SI counterfactuals.

The reconstructor follows Databento's documented MBO actions. Trade, Fill, and
None records do not mutate the book because the associated Cancel records carry
the size change. A book observation is emitted only on ``F_LAST`` so multi-record
publisher events are never inspected halfway through an update.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

F_LAST = 128
F_TOB = 64
UNDEFINED_PRICE_RAW = (1 << 63) - 1


class MboReconstructionError(ValueError):
    """Raised when a normalized MBO stream cannot produce a trustworthy book."""


@dataclass(frozen=True)
class RestingOrder:
    side: str
    price_raw: int
    size: int


@dataclass(frozen=True)
class MboBboSnapshot:
    instrument_id: int
    contract_symbol: str
    ts_event_ns: int
    ts_recv_ns: int
    source_publish_ts_ns: int | None
    sequence: int
    bid_price: Decimal
    ask_price: Decimal
    bid_size: int
    ask_size: int
    bid_order_count: int
    ask_order_count: int

    @property
    def midpoint(self) -> Decimal:
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> Decimal:
        return self.ask_price - self.bid_price

    @property
    def imbalance(self) -> Decimal:
        total = self.bid_size + self.ask_size
        return (
            Decimal(self.bid_size - self.ask_size) / Decimal(total)
            if total
            else Decimal(0)
        )

    @property
    def microprice(self) -> Decimal:
        total = self.bid_size + self.ask_size
        if not total:
            return self.midpoint
        return (
            self.ask_price * self.bid_size + self.bid_price * self.ask_size
        ) / Decimal(total)


@dataclass(frozen=True)
class MboReconstructionResult:
    snapshots: tuple[MboBboSnapshot, ...]
    input_records: int
    selected_records: int
    emitted_events: int
    duplicate_adds: int
    unknown_cancels: int
    unknown_modifies: int
    crossed_or_locked_events: int


class _Book:
    def __init__(self) -> None:
        self.orders: dict[int, RestingOrder] = {}
        self.level_size: dict[str, dict[int, int]] = {"B": {}, "A": {}}
        self.level_count: dict[str, dict[int, int]] = {"B": {}, "A": {}}
        self.bid_heap: list[int] = []
        self.ask_heap: list[int] = []
        self.duplicate_adds = 0
        self.unknown_cancels = 0
        self.unknown_modifies = 0

    def clear(self) -> None:
        self.orders.clear()
        self.level_size = {"B": {}, "A": {}}
        self.level_count = {"B": {}, "A": {}}
        self.bid_heap.clear()
        self.ask_heap.clear()

    def _change_level(self, side: str, price_raw: int, size: int, count: int) -> None:
        sizes = self.level_size[side]
        counts = self.level_count[side]
        previous_size = sizes.get(price_raw, 0)
        previous_count = counts.get(price_raw, 0)
        next_size = previous_size + size
        next_count = previous_count + count
        if next_size < 0 or next_count < 0 or (next_size == 0) != (next_count == 0):
            raise MboReconstructionError("MBO price-level size/count became invalid")
        if next_size == 0:
            sizes.pop(price_raw, None)
            counts.pop(price_raw, None)
            return
        sizes[price_raw] = next_size
        counts[price_raw] = next_count
        if previous_count == 0:
            heapq.heappush(
                self.bid_heap if side == "B" else self.ask_heap,
                -price_raw if side == "B" else price_raw,
            )

    def _remove_order(self, order_id: int) -> RestingOrder | None:
        order = self.orders.pop(order_id, None)
        if order is not None:
            self._change_level(order.side, order.price_raw, -order.size, -1)
        return order

    def _clear_side(self, side: str) -> None:
        for order_id in [
            order_id for order_id, order in self.orders.items() if order.side == side
        ]:
            self._remove_order(order_id)

    def apply(self, row: Mapping[str, Any]) -> None:
        action = str(row.get("action", ""))
        side = str(row.get("side", ""))
        if action in {"T", "F", "N"}:
            return
        if action == "R":
            self.clear()
            return
        if action not in {"A", "M", "C"}:
            raise MboReconstructionError(f"unsupported MBO action: {action!r}")
        if side not in {"B", "A"}:
            raise MboReconstructionError(f"invalid MBO side: {side!r}")
        try:
            order_id = int(row["order_id"])
            price_raw = int(row["price_raw"])
            size = int(row["size"])
            flags = int(row.get("flags", 0))
        except (KeyError, TypeError, ValueError) as error:
            raise MboReconstructionError("malformed normalized MBO order") from error
        if size < 0:
            raise MboReconstructionError("MBO size cannot be negative")

        if action == "A":
            if flags & F_TOB:
                self._clear_side(side)
                if price_raw == UNDEFINED_PRICE_RAW:
                    return
            if order_id in self.orders:
                self.duplicate_adds += 1
                self._remove_order(order_id)
            order = RestingOrder(side, price_raw, size)
            self.orders[order_id] = order
            self._change_level(side, price_raw, size, 1)
            return

        existing = self.orders.get(order_id)
        if existing is None:
            if action == "C":
                self.unknown_cancels += 1
            else:
                self.unknown_modifies += 1
            raise MboReconstructionError(
                f"{action} references unknown MBO order {order_id}"
            )
        if existing.side != side:
            raise MboReconstructionError("MBO update changed an order's side")
        if action == "C":
            if size > existing.size:
                raise MboReconstructionError("MBO cancel exceeds resting size")
            if size == existing.size:
                self._remove_order(order_id)
            elif size:
                self._change_level(side, existing.price_raw, -size, 0)
                self.orders[order_id] = RestingOrder(
                    side, existing.price_raw, existing.size - size
                )
            return

        self._remove_order(order_id)
        modified = RestingOrder(side, price_raw, size)
        self.orders[order_id] = modified
        self._change_level(side, price_raw, size, 1)

    def _best(self, side: str) -> tuple[int, int, int] | None:
        heap = self.bid_heap if side == "B" else self.ask_heap
        sizes = self.level_size[side]
        counts = self.level_count[side]
        while heap:
            raw = -heap[0] if side == "B" else heap[0]
            if raw in sizes:
                return raw, sizes[raw], counts[raw]
            heapq.heappop(heap)
        return None

    def bbo(self) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
        bid = self._best("B")
        ask = self._best("A")
        return (bid, ask) if bid is not None and ask is not None else None


def reconstruct_mbo_bbo(
    rows: Iterable[Mapping[str, Any]],
    *,
    instrument_id: int,
    contract_symbol: str | None = None,
) -> MboReconstructionResult:
    """Reconstruct changed BBO states for one instrument from normalized MBO."""

    if instrument_id < 0:
        raise ValueError("instrument_id must be nonnegative")
    book = _Book()
    snapshots: list[MboBboSnapshot] = []
    input_records = 0
    selected_records = 0
    emitted_events = 0
    crossed = 0
    last_state: tuple[int, int, int, int, int, int] | None = None
    observed_symbol = contract_symbol or ""
    for row in rows:
        input_records += 1
        if int(row.get("instrument_id", -1)) != instrument_id:
            continue
        selected_records += 1
        row_symbol = str(row.get("contract_symbol", ""))
        if row_symbol:
            if observed_symbol and observed_symbol != row_symbol:
                raise MboReconstructionError("one instrument ID mapped to two symbols")
            observed_symbol = row_symbol
        book.apply(row)
        flags = int(row.get("flags", 0))
        if not flags & F_LAST:
            continue
        emitted_events += 1
        bbo = book.bbo()
        if bbo is None:
            continue
        bid, ask = bbo
        state = (bid[0], ask[0], bid[1], ask[1], bid[2], ask[2])
        if bid[0] >= ask[0]:
            crossed += 1
            continue
        if state == last_state:
            continue
        last_state = state
        source_publish = row.get("source_publish_ts_ns")
        snapshots.append(
            MboBboSnapshot(
                instrument_id=instrument_id,
                contract_symbol=observed_symbol,
                ts_event_ns=int(row["ts_event"]),
                ts_recv_ns=int(row["ts_recv"]),
                source_publish_ts_ns=(
                    int(source_publish) if source_publish is not None else None
                ),
                sequence=int(row.get("sequence", 0)),
                bid_price=Decimal(bid[0]).scaleb(-9),
                ask_price=Decimal(ask[0]).scaleb(-9),
                bid_size=bid[1],
                ask_size=ask[1],
                bid_order_count=bid[2],
                ask_order_count=ask[2],
            )
        )
    return MboReconstructionResult(
        snapshots=tuple(snapshots),
        input_records=input_records,
        selected_records=selected_records,
        emitted_events=emitted_events,
        duplicate_adds=book.duplicate_adds,
        unknown_cancels=book.unknown_cancels,
        unknown_modifies=book.unknown_modifies,
        crossed_or_locked_events=crossed,
    )


def iter_hash_chained_jsonl(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    """Stream and verify one independent hash chain per normalized JSONL file."""

    for path in sorted(paths):
        expected_previous: str | None = None
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise MboReconstructionError(
                        f"invalid JSON in {path}:{line_number}"
                    ) from error
                if not isinstance(row, dict):
                    raise MboReconstructionError(
                        f"non-object JSON in {path}:{line_number}"
                    )
                record_hash = row.get("record_sha256")
                payload = dict(row)
                payload.pop("record_sha256", None)
                canonical = json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    default=str,
                )
                computed = hashlib.sha256(canonical.encode()).hexdigest()
                if record_hash != computed:
                    raise MboReconstructionError(
                        f"record hash mismatch in {path}:{line_number}"
                    )
                if row.get("previous_record_sha256") != expected_previous:
                    raise MboReconstructionError(
                        f"record chain mismatch in {path}:{line_number}"
                    )
                expected_previous = str(record_hash)
                yield row


def select_active_outright_contract(
    trade_rows: Iterable[Mapping[str, Any]],
    *,
    asset: str = "SI",
) -> dict[str, Any]:
    """Select the outright future with greatest observed trade quantity."""

    totals: dict[tuple[int, str], Decimal] = {}
    counts: dict[tuple[int, str], int] = {}
    for row in trade_rows:
        if str(row.get("asset", "")) != asset:
            continue
        if str(row.get("instrument_class", "")) != "F":
            continue
        instrument_id = int(row.get("instrument_id", -1))
        symbol = str(row.get("contract_symbol", ""))
        if instrument_id < 0 or not symbol:
            continue
        key = (instrument_id, symbol)
        totals[key] = totals.get(key, Decimal(0)) + Decimal(int(row.get("size", 0)))
        counts[key] = counts.get(key, 0) + 1
    if not totals:
        raise MboReconstructionError(f"no outright {asset} trades were observed")
    selected = max(totals, key=lambda key: (totals[key], counts[key], key[1]))
    ranking = sorted(
        (
            {
                "instrument_id": key[0],
                "contract_symbol": key[1],
                "trade_quantity": format(totals[key], "f"),
                "trade_records": counts[key],
            }
            for key in totals
        ),
        key=lambda row: (
            Decimal(row["trade_quantity"]),
            int(row["trade_records"]),
            row["contract_symbol"],
        ),
        reverse=True,
    )
    return {
        "instrument_id": selected[0],
        "contract_symbol": selected[1],
        "selection_rule": "maximum_observed_trade_quantity",
        "ranking": ranking,
    }
