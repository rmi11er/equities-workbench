from decimal import Decimal

from src.metals.audit import audit_window


def test_window_audit_requires_target_and_result_agreement():
    result = audit_window(
        market_ticker="KXSILVER15M-TEST",
        exchange_target=Decimal("66.516"),
        opening_candle={"close": "66.5156"},
        settlement_candle={"close": "66.5201"},
        round_digits=3,
        rounding_mode="half_up",
        exchange_result="yes",
    )

    assert result.reconstructed_target == "66.516"
    assert result.target_agreement is True
    assert result.settlement.predicted_result == "yes"
    assert result.usable is True


def test_window_audit_blocks_target_mismatch():
    result = audit_window(
        market_ticker="KXSILVER15M-TEST",
        exchange_target=Decimal("66.516"),
        opening_candle={"close": "66.5140"},
        settlement_candle={"close": "66.5201"},
        round_digits=3,
        rounding_mode="half_up",
        exchange_result="yes",
    )

    assert result.target_agreement is False
    assert result.usable is False
    assert result.exclusion_reason == "opening_target_mismatch"
