"""RFQ Risk Manager - exposure tracking and limit enforcement."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .config import RiskConfig
from .types import RFQ, ActiveQuote, QuoteStatus

logger = logging.getLogger(__name__)


@dataclass
class Exposure:
    """Current exposure state."""
    total_dollars: float = 0.0
    active_quote_count: int = 0
    exposure_by_market: dict[str, int] = field(default_factory=dict)  # ticker -> contracts


class RiskManager:
    """
    Manages risk limits for RFQ responses.

    Tracks:
    - Total dollar exposure across pending quotes
    - Position by underlying market
    - Number of active (unconfirmed) quotes
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self._exposure = Exposure()
        self._active_quotes: dict[str, ActiveQuote] = {}  # quote_id -> ActiveQuote
        self._executed_positions: dict[str, int] = {}  # market_ticker -> contracts

    @property
    def exposure(self) -> Exposure:
        """Current exposure state (read-only view)."""
        return self._exposure

    @property
    def active_quote_count(self) -> int:
        """Number of active (non-executed, non-expired) quotes."""
        return self._exposure.active_quote_count

    def can_quote(self, rfq: RFQ, theo_value: float) -> tuple[bool, str]:
        """
        Check if we can safely quote this RFQ.

        Performs pre-flight risk checks before sending a quote.

        Args:
            rfq: The RFQ to check
            theo_value: Theoretical fair value (probability 0-1)

        Returns:
            (allowed, reason) tuple - allowed is True if we can quote,
            reason explains why if not allowed.
        """
        # Check active quote count
        if self._exposure.active_quote_count >= self.config.max_active_quotes:
            return False, f"at max active quotes ({self.config.max_active_quotes})"

        # Calculate dollar exposure for this RFQ
        rfq_dollars = rfq.contracts * theo_value

        # Check single RFQ limit
        if rfq_dollars > self.config.max_single_rfq_dollars:
            return False, (
                f"RFQ exposure ${rfq_dollars:.2f} > "
                f"max ${self.config.max_single_rfq_dollars:.2f}"
            )

        # Check total exposure
        new_total = self._exposure.total_dollars + rfq_dollars
        if new_total > self.config.max_exposure_dollars:
            return False, (
                f"total exposure ${new_total:.2f} > "
                f"max ${self.config.max_exposure_dollars:.2f}"
            )

        # Check contracts limit
        if rfq.contracts > self.config.max_contracts_per_rfq:
            return False, (
                f"contracts {rfq.contracts} > "
                f"max {self.config.max_contracts_per_rfq}"
            )

        # Check per-market position limits for parlay legs
        for leg in rfq.mve_selected_legs:
            ticker = leg.market_ticker
            current_pending = self._exposure.exposure_by_market.get(ticker, 0)
            current_executed = self._executed_positions.get(ticker, 0)
            total = current_pending + current_executed + rfq.contracts

            if total > self.config.position_limit_per_market:
                return False, (
                    f"position limit for {ticker}: "
                    f"{total} > {self.config.position_limit_per_market}"
                )

        return True, ""

    def register_quote(self, quote: ActiveQuote) -> None:
        """
        Register a newly sent quote, reserving exposure.

        Called after successfully sending a quote to the API.

        Args:
            quote: The active quote to register
        """
        self._active_quotes[quote.quote_id] = quote
        self._exposure.active_quote_count += 1
        self._exposure.total_dollars += quote.dollar_exposure

        # Track per-market exposure
        for leg in quote.rfq.mve_selected_legs:
            ticker = leg.market_ticker
            current = self._exposure.exposure_by_market.get(ticker, 0)
            self._exposure.exposure_by_market[ticker] = current + quote.rfq.contracts

        logger.info(
            f"Registered quote {quote.quote_id}: "
            f"${quote.dollar_exposure:.2f} exposure, "
            f"{self._exposure.active_quote_count} active quotes"
        )

    def release_quote(self, quote_id: str) -> Optional[ActiveQuote]:
        """
        Release exposure when quote expires or is rejected.

        Called when a quote is no longer active but was not executed.

        Args:
            quote_id: ID of the quote to release

        Returns:
            The released ActiveQuote, or None if not found
        """
        quote = self._active_quotes.pop(quote_id, None)
        if quote is None:
            return None

        self._exposure.active_quote_count -= 1
        self._exposure.total_dollars -= quote.dollar_exposure

        # Release per-market exposure
        for leg in quote.rfq.mve_selected_legs:
            ticker = leg.market_ticker
            current = self._exposure.exposure_by_market.get(ticker, 0)
            new_value = max(0, current - quote.rfq.contracts)
            if new_value == 0:
                self._exposure.exposure_by_market.pop(ticker, None)
            else:
                self._exposure.exposure_by_market[ticker] = new_value

        logger.info(
            f"Released quote {quote_id}: "
            f"${quote.dollar_exposure:.2f} freed, "
            f"{self._exposure.active_quote_count} active quotes remaining"
        )

        return quote

    def confirm_quote(self, quote_id: str) -> Optional[ActiveQuote]:
        """
        Mark quote as confirmed (will execute shortly).

        Called when we successfully confirm a quote after acceptance.
        The quote remains in tracking until executed.

        Args:
            quote_id: ID of the quote that was confirmed

        Returns:
            The confirmed ActiveQuote, or None if not found
        """
        quote = self._active_quotes.get(quote_id)
        if quote is None:
            logger.warning(f"Attempted to confirm unknown quote: {quote_id}")
            return None

        quote.status = QuoteStatus.CONFIRMED
        quote.confirmed_at = datetime.now()

        logger.info(f"Confirmed quote {quote_id}")
        return quote

    def execute_quote(self, quote_id: str) -> Optional[ActiveQuote]:
        """
        Mark quote as executed (trade completed).

        Called when we receive execution confirmation. Moves exposure
        from pending quotes to executed positions.

        Args:
            quote_id: ID of the quote that was executed

        Returns:
            The executed ActiveQuote, or None if not found
        """
        quote = self._active_quotes.pop(quote_id, None)
        if quote is None:
            logger.warning(f"Attempted to execute unknown quote: {quote_id}")
            return None

        quote.status = QuoteStatus.EXECUTED
        quote.executed_at = datetime.now()

        # Move from pending to executed positions
        self._exposure.active_quote_count -= 1
        self._exposure.total_dollars -= quote.dollar_exposure

        for leg in quote.rfq.mve_selected_legs:
            ticker = leg.market_ticker
            # Release from pending
            current_pending = self._exposure.exposure_by_market.get(ticker, 0)
            new_pending = max(0, current_pending - quote.rfq.contracts)
            if new_pending == 0:
                self._exposure.exposure_by_market.pop(ticker, None)
            else:
                self._exposure.exposure_by_market[ticker] = new_pending

            # Add to executed positions
            current_executed = self._executed_positions.get(ticker, 0)
            self._executed_positions[ticker] = current_executed + quote.rfq.contracts

        logger.info(
            f"Executed quote {quote_id}: "
            f"{quote.rfq.contracts} contracts @ theo={quote.response.theo_value:.4f}"
        )

        return quote

    def get_quote(self, quote_id: str) -> Optional[ActiveQuote]:
        """Get an active quote by ID."""
        return self._active_quotes.get(quote_id)

    def get_active_quotes(self) -> list[ActiveQuote]:
        """Get all active quotes."""
        return list(self._active_quotes.values())

    def get_pending_quotes(self) -> list[ActiveQuote]:
        """Get quotes in PENDING status (awaiting acceptance)."""
        return [
            q for q in self._active_quotes.values()
            if q.status == QuoteStatus.PENDING
        ]

    def get_executed_position(self, ticker: str) -> int:
        """Get executed position for a market."""
        return self._executed_positions.get(ticker, 0)

    def cleanup_expired_quotes(self, max_age_seconds: float) -> list[ActiveQuote]:
        """
        Clean up quotes that have exceeded max age.

        Called periodically to release exposure from stale quotes.

        Args:
            max_age_seconds: Maximum age before considering a quote expired

        Returns:
            List of expired quotes that were cleaned up
        """
        expired: list[ActiveQuote] = []
        now = datetime.now()

        for quote_id, quote in list(self._active_quotes.items()):
            if quote.status == QuoteStatus.PENDING:
                age = (now - quote.sent_at).total_seconds()
                if age > max_age_seconds:
                    quote.status = QuoteStatus.EXPIRED
                    released = self.release_quote(quote_id)
                    if released:
                        expired.append(released)
                        logger.debug(f"Cleaned up expired quote: {quote_id}")

        return expired

    def reset(self) -> None:
        """Reset all exposure tracking (for testing or restart)."""
        self._exposure = Exposure()
        self._active_quotes.clear()
        self._executed_positions.clear()
        logger.info("Risk manager reset")
