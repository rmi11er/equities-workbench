"""Settlement-audit primitives for exact opening target and final value checks."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Mapping, Optional

from .settlement import evaluate_settlement, round_price
from .types import SettlementCheck


@dataclass(frozen=True)
class WindowAuditResult:
    market_ticker: str
    reconstructed_target: str
    exchange_target: str
    target_agreement: bool
    settlement: SettlementCheck
    usable: bool
    exclusion_reason: Optional[str] = None


def _close(candle: Mapping[str, Any]) -> Decimal:
    value = candle.get("close", candle.get("c"))
    if value is None:
        raise ValueError("Pyth candle is missing a close value")
    return Decimal(str(value))


def audit_window(
    market_ticker: str,
    exchange_target: Decimal,
    opening_candle: Mapping[str, Any],
    settlement_candle: Mapping[str, Any],
    round_digits: int,
    rounding_mode: str,
    exchange_result: str,
) -> WindowAuditResult:
    """Audit one window after exact candle-boundary selection is complete."""

    reconstructed_target = round_price(
        _close(opening_candle), round_digits, rounding_mode
    )
    settlement = evaluate_settlement(
        market_ticker=market_ticker,
        target_price=exchange_target,
        final_value=_close(settlement_candle),
        round_digits=round_digits,
        rounding_mode=rounding_mode,
        exchange_result=exchange_result,
    )
    target_agreement = reconstructed_target == exchange_target
    usable = target_agreement and settlement.agreement is True
    if not target_agreement:
        reason = "opening_target_mismatch"
    elif settlement.agreement is False:
        reason = settlement.discrepancy_classification or "settlement_mismatch"
    else:
        reason = None
    return WindowAuditResult(
        market_ticker=market_ticker,
        reconstructed_target=format(reconstructed_target, "f"),
        exchange_target=format(exchange_target, "f"),
        target_agreement=target_agreement,
        settlement=settlement,
        usable=usable,
        exclusion_reason=reason,
    )
