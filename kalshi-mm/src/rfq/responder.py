"""RFQ Responder - main orchestrator for responding to RFQ requests."""

import asyncio
import logging
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..connector import KalshiConnector, WSMessage, APIError
from ..config import Config, VolatilityConfig
from ..orderbook import OrderBookManager
from .config import RFQConfig, FilterConfig
from .types import RFQ, RFQLeg, ActiveQuote, QuoteStatus, RFQStatus
from .pricing import PricingEngine, BBOPriceSource
from .risk import RiskManager
from .decision_logger import (
    RFQDecisionLogger,
    create_decision_record,
    create_outcome_record,
)

logger = logging.getLogger(__name__)


class RFQResponder:
    """
    RFQ Responder - responds to Kalshi RFQ requests for combos/parlays.

    Event flow:
    1. Subscribe to 'communications' WebSocket channel
    2. Receive RFQCreated events
    3. Price the RFQ using PricingEngine
    4. Check filters and risk limits
    5. POST quote via REST API
    6. Handle QuoteAccepted -> PUT confirm
    7. Handle QuoteExecuted -> update positions
    """

    def __init__(self, config: RFQConfig):
        self.config = config
        self._running = False

        # Build connector with adapted config
        connector_config = self._build_connector_config()
        self.connector = KalshiConnector(connector_config)

        # OrderBook manager for leg pricing
        vol_config = VolatilityConfig()
        self.orderbook_manager = OrderBookManager(vol_config)

        # Pricing engine with BBO source (with liquidity thresholds)
        bbo_source = BBOPriceSource(
            self.orderbook_manager,
            min_liquidity_dollars=config.pricing.min_bbo_liquidity_dollars,
            max_spread_cents=config.pricing.max_bbo_spread_cents,
        )
        self.pricing_engine = PricingEngine(config.pricing, bbo_source)

        # Risk manager
        self.risk_manager = RiskManager(config.risk)

        # Decision logger (JSONL output for post-session analysis)
        self.decision_logger = RFQDecisionLogger(config.logging.base_log_dir)

        # Track dynamically subscribed markets (when leg_tickers is empty)
        self._subscribed_markets: set[str] = set()

        # Background tasks
        self._tasks: list[asyncio.Task] = []

    def _build_connector_config(self) -> Config:
        """Build a connector-compatible config from RFQConfig."""
        # Create a minimal Config that the connector can use
        from ..config import (
            CredentialsConfig as MMCredentialsConfig,
            RateLimitConfig,
            LoggingConfig as MMLoggingConfig,
        )

        return Config(
            environment=self.config.environment,
            credentials=MMCredentialsConfig(
                api_key_id=self.config.credentials.api_key_id,
                private_key_path=self.config.credentials.private_key_path,
            ),
            rate_limit=RateLimitConfig(read_rate=20, write_rate=10),
            logging=MMLoggingConfig(
                log_level=self.config.logging.log_level,
                base_log_dir=self.config.logging.base_log_dir,
            ),
        )

    async def start(self) -> None:
        """Start the RFQ responder."""
        mode_str = "SHADOW MODE" if self.config.is_shadow_mode else "LIVE MODE"
        logger.info(f"Starting RFQ Responder ({mode_str})")

        # Log liquidity requirements
        liq_cfg = self.config.pricing
        if liq_cfg.min_bbo_liquidity_dollars > 0:
            logger.info(
                f"Liquidity filter: ${liq_cfg.min_bbo_liquidity_dollars} min per side, "
                f"{liq_cfg.max_bbo_spread_cents}c max spread"
            )

        self._running = True

        # Setup logging directory
        log_dir = Path(self.config.logging.base_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Start decision logger
        self.decision_logger.start()

        # Start connector
        await self.connector.start()

        # Register message handlers
        self.connector.on_message(self._handle_ws_message)
        self.connector.on_reconnect(self._handle_reconnect)

        # Connect WebSocket
        await self.connector.connect_ws()

        # Subscribe to orderbooks for leg markets (static list)
        if self.config.leg_tickers:
            logger.info(f"Subscribing to {len(self.config.leg_tickers)} pre-configured leg tickers")
            await self.connector.subscribe_orderbook(self.config.leg_tickers)
            self._subscribed_markets.update(self.config.leg_tickers)
        else:
            logger.info("No pre-configured leg_tickers - will subscribe dynamically as RFQs arrive")

        # Subscribe to communications channel (RFQs)
        await self._subscribe_communications()

        # Start background monitors
        self._tasks = [
            asyncio.create_task(self._quote_expiry_monitor()),
        ]

        # Run main WebSocket loop
        try:
            await self.connector.run_ws_loop()
        finally:
            await self.shutdown()

    async def _subscribe_communications(self) -> int:
        """Subscribe to the communications channel for RFQ events."""
        cmd_id = self.connector._next_cmd_id
        self.connector._next_cmd_id += 1

        msg = {
            "id": cmd_id,
            "cmd": "subscribe",
            "params": {
                "channels": ["communications"],
            },
        }

        await self.connector._send_ws(msg)
        self.connector._subscriptions[cmd_id] = msg["params"]

        logger.info("Subscribed to communications channel")
        return cmd_id

    def _handle_ws_message(self, ws_msg: WSMessage) -> None:
        """Dispatch WebSocket messages."""
        msg_type = ws_msg.type

        # Orderbook updates (for leg pricing)
        if msg_type in ("orderbook_snapshot", "orderbook_delta"):
            self.orderbook_manager.handle_message(ws_msg)

        # RFQ events from communications channel
        elif msg_type == "rfq_created":
            asyncio.create_task(self._handle_rfq_created(ws_msg.msg))
        elif msg_type == "rfq_deleted":
            self._handle_rfq_deleted(ws_msg.msg)
        elif msg_type == "quote_accepted":
            asyncio.create_task(self._handle_quote_accepted(ws_msg.msg))
        elif msg_type == "quote_executed":
            self._handle_quote_executed(ws_msg.msg)
        elif msg_type == "quote_expired":
            self._handle_quote_expired(ws_msg.msg)
        elif msg_type == "quote_deleted":
            self._handle_quote_deleted(ws_msg.msg)

        # Subscription confirmations and acknowledgements
        elif msg_type in ("subscribed", "ok", "quote_created"):
            logger.debug(f"WS acknowledgement ({msg_type}): {ws_msg.msg}")

        # Log unrecognized message types for debugging
        elif msg_type not in ("orderbook_snapshot", "orderbook_delta", "heartbeat"):
            logger.info(f"Unhandled WS message type: {msg_type}, msg={ws_msg.msg}")

    def _handle_reconnect(self) -> None:
        """Handle WebSocket reconnection."""
        logger.info("WebSocket reconnected, re-subscribing to channels")

    async def _ensure_leg_subscriptions(self, rfq: RFQ) -> bool:
        """
        Ensure we're subscribed to orderbooks for all legs in the RFQ.

        When leg_tickers is empty, we dynamically subscribe to leg markets
        as we see them in RFQs. This allows filtering by allowed_sports
        without pre-defining every possible market ticker.

        Returns:
            True if all legs have orderbook data (or we just subscribed and should retry)
            False if subscription failed
        """
        new_tickers = []
        for leg in rfq.mve_selected_legs:
            ticker = leg.market_ticker
            if ticker and ticker not in self._subscribed_markets:
                new_tickers.append(ticker)

        if not new_tickers:
            return True  # Already subscribed to all legs

        # Subscribe to new markets
        logger.info(f"Dynamically subscribing to {len(new_tickers)} new leg markets: {new_tickers}")
        try:
            await self.connector.subscribe_orderbook(new_tickers)
            self._subscribed_markets.update(new_tickers)

            # Wait briefly for orderbook snapshots to arrive
            # This is a tradeoff - we might miss this RFQ but will catch future ones
            await asyncio.sleep(0.5)
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to leg markets: {e}")
            return False

    async def _handle_rfq_created(self, msg: dict) -> None:
        """
        Handle incoming RFQ request.

        1. Parse RFQ
        2. Check leg filters first (before pricing)
        3. Ensure leg market subscriptions (dynamic mode)
        4. Compute fair value
        5. Check dollar-based filters (after pricing)
        6. Check risk limits
        7. Send quote (with shadow mode spread if enabled)
        """
        rfq = self._parse_rfq(msg)
        shadow_mode = self.config.is_shadow_mode
        spread_mult = self.config.shadow.spread_multiplier if shadow_mode else 1.0

        # Log RFQ with target cost if available (requester's desired price)
        target_info = ""
        if rfq.target_cost_dollars is not None:
            target_info = f", target=${rfq.target_cost_dollars:.2f}"
        logger.info(
            f"RFQ received: {rfq.id}, contracts={rfq.contracts}, "
            f"legs={rfq.leg_count}{target_info}"
        )

        # Check pre-pricing filters (leg count, collections, sports)
        passed, reason = self._passes_pre_filters(rfq)
        if not passed:
            # Only log pre-filter rejections if log_all_rfqs is enabled
            if self.config.shadow.log_all_rfqs:
                self._log_decision(
                    rfq, "filtered", reason,
                    filter_name="pre_filter",
                    shadow_mode=shadow_mode,
                    spread_multiplier=spread_mult,
                )
            logger.debug(f"RFQ {rfq.id} filtered (pre): {reason}")
            return

        # Ensure we're subscribed to leg markets (dynamic subscription mode)
        await self._ensure_leg_subscriptions(rfq)

        # Compute fair value
        theo = self.pricing_engine.compute_fair_value(rfq)
        leg_prices = self.pricing_engine.get_leg_prices(rfq)

        if theo is None:
            missing = [t for t, p in leg_prices.items() if p is None]
            self._log_decision(
                rfq, "skipped", f"missing leg prices: {missing}",
                filter_name="pricing",
                leg_prices=leg_prices,
                shadow_mode=shadow_mode,
                spread_multiplier=spread_mult,
            )
            logger.info(f"RFQ {rfq.id} skipped: cannot price (missing: {missing})")
            return

        logger.info(f"RFQ {rfq.id} theo: {theo:.4f}")

        # Check post-pricing filters (dollar size)
        passed, reason = self._passes_post_filters(rfq, theo)
        if not passed:
            self._log_decision(
                rfq, "filtered", reason,
                filter_name="post_filter",
                theo_value=theo,
                leg_prices=leg_prices,
                shadow_mode=shadow_mode,
                spread_multiplier=spread_mult,
            )
            logger.info(f"RFQ {rfq.id} filtered (post): {reason}")
            return

        # Check risk limits
        allowed, reason = self.risk_manager.can_quote(rfq, theo)
        if not allowed:
            self._log_decision(
                rfq, "filtered", reason,
                filter_name="risk",
                theo_value=theo,
                leg_prices=leg_prices,
                shadow_mode=shadow_mode,
                spread_multiplier=spread_mult,
            )
            logger.info(f"RFQ {rfq.id} rejected (risk): {reason}")
            return

        # Create quote response (apply shadow spread multiplier)
        response = self._create_quote_with_spread(rfq, theo, spread_mult)

        # Check for negative edge on BOTH sides - this shouldn't happen if
        # yes_bid + no_bid < 1, but reject if somehow both edges are negative
        if response.yes_edge < 0 and response.no_edge < 0:
            self._log_decision(
                rfq, "filtered",
                f"both edges negative: yes={response.yes_edge:.4f}, no={response.no_edge:.4f}",
                filter_name="edge",
                theo_value=theo,
                yes_bid=response.yes_bid,
                no_bid=response.no_bid,
                edge=response.edge,
                yes_edge=response.yes_edge,
                no_edge=response.no_edge,
                leg_prices=leg_prices,
                shadow_mode=shadow_mode,
                spread_multiplier=spread_mult,
            )
            logger.info(
                f"RFQ {rfq.id} rejected (both edges negative): "
                f"yes_edge={response.yes_edge:.4f}, no_edge={response.no_edge:.4f}"
            )
            return

        # Log when one side has negative edge (informational, not a blocker)
        if response.yes_edge < 0:
            logger.debug(
                f"RFQ {rfq.id}: YES edge negative ({response.yes_edge:.4f}) but "
                f"NO edge positive ({response.no_edge:.4f}) - quoting anyway"
            )

        # Check quote cost against balance limit (max of yes/no bid * contracts)
        yes_cost = float(response.yes_bid) * rfq.contracts
        no_cost = float(response.no_bid) * rfq.contracts
        max_cost = max(yes_cost, no_cost)
        max_allowed = self.config.risk.max_quote_cost_dollars

        if max_cost > max_allowed:
            self._log_decision(
                rfq, "filtered", f"quote cost ${max_cost:.2f} > max ${max_allowed:.2f}",
                filter_name="quote_cost",
                theo_value=theo,
                yes_bid=response.yes_bid,
                no_bid=response.no_bid,
                edge=response.edge,
                yes_edge=response.yes_edge,
                no_edge=response.no_edge,
                leg_prices=leg_prices,
                shadow_mode=shadow_mode,
                spread_multiplier=spread_mult,
            )
            logger.info(
                f"RFQ {rfq.id} rejected (quote cost): ${max_cost:.2f} > ${max_allowed:.2f} "
                f"(yes=${yes_cost:.2f}, no=${no_cost:.2f})"
            )
            return

        # Send quote via REST
        # Use one-sided quotes when configured and one side has negative edge
        use_one_sided = (
            self.config.risk.one_sided_quotes
            and response.is_one_sided
        )
        if use_one_sided:
            logger.info(
                f"RFQ {rfq.id}: using one-sided quote ({response.profitable_side} only), "
                f"yes_edge={response.yes_edge:.4f}, no_edge={response.no_edge:.4f}"
            )

        quote_id, error_msg = await self._send_quote(response, one_sided=use_one_sided)
        if quote_id is None:
            self._log_decision(
                rfq, "error", f"API error: {error_msg}",
                theo_value=theo,
                yes_bid=response.yes_bid,
                no_bid=response.no_bid,
                edge=response.edge,
                yes_edge=response.yes_edge,
                no_edge=response.no_edge,
                leg_prices=leg_prices,
                shadow_mode=shadow_mode,
                spread_multiplier=spread_mult,
            )
            logger.error(f"Failed to send quote for RFQ {rfq.id}: {error_msg}")
            return

        # Register with risk manager
        active_quote = ActiveQuote(
            quote_id=quote_id,
            rfq_id=rfq.id,
            rfq=rfq,
            response=response,
            sent_at=datetime.now(),
            status=QuoteStatus.PENDING,
        )
        self.risk_manager.register_quote(active_quote)

        # Log decision
        self._log_decision(
            rfq, "quoted",
            theo_value=theo,
            quote_id=quote_id,
            yes_bid=response.yes_bid,
            no_bid=response.no_bid,
            edge=response.edge,
            yes_edge=response.yes_edge,
            no_edge=response.no_edge,
            leg_prices=leg_prices,
            shadow_mode=shadow_mode,
            spread_multiplier=spread_mult,
        )

        mode_tag = " [SHADOW]" if shadow_mode else ""
        logger.info(
            f"Quote sent{mode_tag}: {quote_id} for RFQ {rfq.id}, "
            f"YES bid={response.yes_bid}, NO bid={response.no_bid}, "
            f"theo={theo:.4f}, yes_edge={response.yes_edge:.4f}, no_edge={response.no_edge:.4f}"
        )

    def _create_quote_with_spread(self, rfq: RFQ, theo: float, spread_mult: float):
        """
        Create quote response with liquidity-based spread.

        Spread is computed from underlying leg liquidity, then optionally
        multiplied for shadow mode to make quotes uncompetitive.
        """
        from .types import QuoteResponse

        # Get liquidity-based spread from pricing engine
        # (this uses the leg spreads cached during pricing)
        base_spread = self.pricing_engine.compute_combo_spread(rfq.mve_selected_legs)

        # Apply multiplier (for shadow mode, this makes quotes uncompetitive)
        effective_spread = base_spread * spread_mult

        # In shadow mode, don't clamp to max (we want wide spreads)
        # In live mode, clamp to config bounds
        if spread_mult <= 1.0:
            min_spread = self.config.pricing.min_spread_cents / 100.0
            max_spread = self.config.pricing.max_spread_cents / 100.0
            effective_spread = max(min_spread, min(effective_spread, max_spread))

        # Apply spread symmetrically
        yes_bid_prob = max(0.01, min(0.99, theo - effective_spread / 2))
        no_bid_prob = max(0.01, min(0.99, (1.0 - theo) - effective_spread / 2))

        # Round to whole cents (Kalshi requires 2 decimal places)
        # Round DOWN for bids (conservative)
        yes_bid_prob = int(yes_bid_prob * 100) / 100.0
        no_bid_prob = int(no_bid_prob * 100) / 100.0

        # Ensure minimum 1 cent after rounding
        yes_bid_prob = max(0.01, yes_bid_prob)
        no_bid_prob = max(0.01, no_bid_prob)

        yes_bid = f"{yes_bid_prob:.2f}"
        no_bid = f"{no_bid_prob:.2f}"

        # Calculate edge for both sides
        # yes_edge: our profit if they sell us YES (we buy YES)
        # no_edge: our profit if they sell us NO (we buy NO, they buy parlay)
        yes_edge = theo - yes_bid_prob
        no_edge = (1.0 - theo) - no_bid_prob

        # Log spread components for analysis
        leg_spreads = self.pricing_engine.get_last_leg_spreads()
        logger.debug(
            f"Spread calc: base={base_spread:.4f}, mult={spread_mult}, "
            f"effective={effective_spread:.4f}, leg_spreads={leg_spreads}, "
            f"yes_edge={yes_edge:.4f}, no_edge={no_edge:.4f}"
        )

        return QuoteResponse(
            rfq_id=rfq.id,
            yes_bid=yes_bid,
            no_bid=no_bid,
            theo_value=theo,
            yes_edge=yes_edge,
            no_edge=no_edge,
        )

    def _passes_pre_filters(self, rfq: RFQ) -> tuple[bool, str]:
        """Check filters that don't require pricing."""
        filters = self.config.filters

        # Must have contracts
        if rfq.contracts <= 0:
            return False, f"invalid contracts: {rfq.contracts}"

        # Leg count
        if rfq.leg_count < filters.min_legs:
            return False, f"legs {rfq.leg_count} < min {filters.min_legs}"
        if rfq.leg_count > filters.max_legs:
            return False, f"legs {rfq.leg_count} > max {filters.max_legs}"

        # Collection whitelist
        if filters.allowed_collections:
            if rfq.mve_collection_ticker not in filters.allowed_collections:
                return False, f"collection {rfq.mve_collection_ticker} not in whitelist"

        # Collection blacklist
        if rfq.mve_collection_ticker in filters.blocked_collections:
            return False, f"collection {rfq.mve_collection_ticker} blocked"

        # Sport filtering - check if any leg event ticker matches allowed sports
        # Kalshi tickers use format like "KXNFLGAME-...", "KXNBAGAME-...", etc.
        if filters.allowed_sports:
            sport_match = False
            for leg in rfq.mve_selected_legs:
                event_upper = (leg.event_ticker or "").upper()
                for sport in filters.allowed_sports:
                    sport_upper = sport.upper()
                    # Check for sport code in ticker (e.g., "NFL" in "KXNFLGAME-...")
                    # Handle both "KXNFL..." and "NFL..." formats
                    if sport_upper in event_upper:
                        sport_match = True
                        break
                if sport_match:
                    break

            if not sport_match:
                leg_events = [leg.event_ticker for leg in rfq.mve_selected_legs]
                return False, f"no legs match allowed sports {filters.allowed_sports}: {leg_events}"

        return True, ""

    def _passes_post_filters(self, rfq: RFQ, theo: float) -> tuple[bool, str]:
        """Check filters that require pricing (dollar-based)."""
        filters = self.config.filters

        # Dollar-based size filtering
        rfq_dollars = rfq.contracts * theo

        if rfq_dollars < filters.min_dollars:
            return False, f"${rfq_dollars:.2f} < min ${filters.min_dollars:.2f}"
        if rfq_dollars > filters.max_dollars:
            return False, f"${rfq_dollars:.2f} > max ${filters.max_dollars:.2f}"

        return True, ""

    async def _send_quote(self, response, one_sided: bool = False) -> tuple[Optional[str], Optional[str]]:
        """Send quote via REST API.

        Args:
            response: QuoteResponse to send
            one_sided: If True, only include the side with positive edge

        Returns:
            (quote_id, error_message) - quote_id if successful, error_message if failed
        """
        try:
            data = response.to_api_payload(one_sided=one_sided)
            logger.debug(f"Sending quote: {data}")
            resp = await self.connector._request(
                "POST",
                "/communications/quotes",
                data=data,
                is_write=True,
            )
            quote_id = resp.get("id") or resp.get("quote_id")
            return quote_id, None
        except APIError as e:
            error_msg = e.response.get("error", {}).get("message", str(e.response))
            logger.error(f"Quote creation failed: {error_msg}, payload={response.to_api_payload(one_sided=one_sided)}")
            return None, error_msg
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"Unexpected error sending quote: {error_msg}")
            return None, error_msg

    async def _cancel_quote(self, quote_id: str) -> bool:
        """Cancel an active quote via REST API.

        Args:
            quote_id: ID of the quote to cancel

        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            await self.connector._request(
                "DELETE",
                f"/communications/quotes/{quote_id}",
                is_write=True,
            )
            logger.info(f"Quote cancelled: {quote_id}")
            return True
        except APIError as e:
            error_msg = e.response.get("error", {}).get("message", str(e.response))
            logger.warning(f"Quote cancellation failed: {quote_id}, error: {error_msg}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error cancelling quote {quote_id}: {e}")
            return False

    async def _handle_quote_accepted(self, msg: dict) -> None:
        """
        Handle quote acceptance - must confirm quickly.

        PUT /communications/quotes/{id}/confirm
        """
        quote_id = msg.get("id") or msg.get("quote_id")
        accepted_side = msg.get("accepted_side")

        logger.info(f"Quote accepted: {quote_id}, side={accepted_side}")

        quote = self.risk_manager.get_quote(quote_id)
        if quote is None:
            logger.warning(f"Unknown quote accepted: {quote_id}")
            return

        quote.status = QuoteStatus.ACCEPTED
        quote.accepted_at = datetime.now()
        quote.accepted_side = accepted_side

        # Log acceptance outcome
        self.decision_logger.log_outcome(create_outcome_record(
            quote_id=quote_id,
            rfq_id=quote.rfq_id,
            event_type="accepted",
            accepted_side=accepted_side,
        ))

        # Confirm immediately
        try:
            await self.connector._request(
                "PUT",
                f"/communications/quotes/{quote_id}/confirm",
                data={},
                is_write=True,
            )
            self.risk_manager.confirm_quote(quote_id)
            logger.info(f"Quote confirmed: {quote_id}")

            # Log confirmation
            self.decision_logger.log_outcome(create_outcome_record(
                quote_id=quote_id,
                rfq_id=quote.rfq_id,
                event_type="confirmed",
                accepted_side=accepted_side,
            ))

        except APIError as e:
            logger.error(f"Quote confirmation failed: {e}")
            quote.status = QuoteStatus.FAILED

    def _handle_quote_executed(self, msg: dict) -> None:
        """Handle successful trade execution."""
        quote_id = msg.get("id") or msg.get("quote_id")
        execution_price = msg.get("execution_price") or msg.get("price")
        logger.info(f"Quote executed: {quote_id}")

        quote = self.risk_manager.execute_quote(quote_id)
        if quote:
            # Log execution outcome
            self.decision_logger.log_outcome(create_outcome_record(
                quote_id=quote_id,
                rfq_id=quote.rfq_id,
                event_type="executed",
                accepted_side=quote.accepted_side,
                execution_price=float(execution_price) if execution_price else None,
            ))

            logger.info(
                f"Trade complete: {quote.rfq.contracts} contracts @ "
                f"theo={quote.response.theo_value:.4f}, "
                f"side={quote.accepted_side}"
            )

    def _handle_quote_expired(self, msg: dict) -> None:
        """Handle quote expiration (user didn't accept)."""
        quote_id = msg.get("id") or msg.get("quote_id")
        logger.info(f"Quote expired: {quote_id}")

        released = self.risk_manager.release_quote(quote_id)
        if released:
            released.status = QuoteStatus.EXPIRED

            # Log expiry outcome
            self.decision_logger.log_outcome(create_outcome_record(
                quote_id=quote_id,
                rfq_id=released.rfq_id,
                event_type="expired",
            ))

    def _handle_quote_deleted(self, msg: dict) -> None:
        """Handle quote deletion."""
        quote_id = msg.get("id") or msg.get("quote_id")
        logger.info(f"Quote deleted: {quote_id}")

        released = self.risk_manager.release_quote(quote_id)
        if released:
            released.status = QuoteStatus.REJECTED

            # Log deletion outcome
            self.decision_logger.log_outcome(create_outcome_record(
                quote_id=quote_id,
                rfq_id=released.rfq_id,
                event_type="deleted",
            ))

    def _handle_rfq_deleted(self, msg: dict) -> None:
        """Handle RFQ deletion (expired or cancelled by creator).

        This tells us the RFQ is no longer active. If we had a pending quote,
        it should be released. The RFQ may have:
        - Expired without any fills
        - Been filled by someone else (target_cost_dollars shows execution price)
        - Been cancelled by the creator
        """
        rfq_id = msg.get("id")
        deleted_ts = msg.get("deleted_ts")
        target_cost_dollars = msg.get("target_cost_dollars")

        # Parse market execution price (what the RFQ actually traded at, if filled)
        market_price = None
        if target_cost_dollars is not None:
            try:
                market_price = float(target_cost_dollars)
            except (ValueError, TypeError):
                pass

        # Check if we had a quote on this RFQ and release it
        quote = self.risk_manager.find_quote_by_rfq(rfq_id)
        if quote:
            released = self.risk_manager.release_quote(quote.quote_id)
            if released:
                released.status = QuoteStatus.EXPIRED

                # Log outcome with comparison data
                self.decision_logger.log_outcome(create_outcome_record(
                    quote_id=quote.quote_id,
                    rfq_id=rfq_id,
                    event_type="rfq_deleted",
                    market_execution_price=market_price,
                    our_yes_bid=quote.response.yes_bid,
                    our_no_bid=quote.response.no_bid,
                    our_theo=quote.response.theo_value,
                ))

                # Log analysis: compare our quote vs market execution
                if market_price is not None and market_price > 0:
                    our_yes = float(quote.response.yes_bid)
                    our_no = float(quote.response.no_bid)
                    theo = quote.response.theo_value

                    # Determine if market bought YES or NO based on price
                    # If market_price < 0.50, they likely bought YES
                    # If market_price > 0.50, they likely bought NO (1 - price for YES)
                    if market_price <= 0.50:
                        # They bought YES at market_price
                        diff = market_price - our_yes
                        logger.info(
                            f"RFQ {rfq_id} filled (YES side): market={market_price:.2f}, "
                            f"our_yes_bid={our_yes:.2f}, diff={diff:+.2f}, theo={theo:.4f}"
                        )
                    else:
                        # They bought NO, so YES was sold at (1 - market_price)
                        implied_yes = 1.0 - market_price
                        diff = our_yes - implied_yes
                        logger.info(
                            f"RFQ {rfq_id} filled (NO side): market_no={market_price:.2f}, "
                            f"our_no_bid={our_no:.2f}, our_yes={our_yes:.2f}, theo={theo:.4f}"
                        )
                else:
                    logger.info(
                        f"RFQ {rfq_id} deleted (no fill): our_yes={quote.response.yes_bid}, "
                        f"our_no={quote.response.no_bid}, theo={quote.response.theo_value:.4f}"
                    )
        else:
            # We didn't quote this RFQ
            if market_price is not None and market_price > 0:
                logger.debug(f"RFQ {rfq_id} filled at ${market_price:.2f} (we didn't quote)")
            else:
                logger.debug(f"RFQ {rfq_id} deleted (we didn't quote)")

    def _parse_rfq(self, msg: dict) -> RFQ:
        """Parse RFQ from WebSocket message."""
        # Debug: log raw message to see field names
        logger.debug(f"Raw RFQ message keys: {list(msg.keys())}")
        if msg.get("contracts", 0) == 0:
            logger.debug(f"Zero contracts RFQ - raw msg: {msg}")

        legs = []
        for leg_data in msg.get("mve_selected_legs", []):
            legs.append(RFQLeg(
                event_ticker=leg_data.get("event_ticker", ""),
                market_ticker=leg_data.get("market_ticker", ""),
                side=leg_data.get("side", "yes"),
                yes_settlement_value_dollars=leg_data.get(
                    "yes_settlement_value_dollars", "1.00"
                ),
            ))

        # Parse status
        status_str = msg.get("status", "open").lower()
        status = RFQStatus.OPEN if status_str == "open" else RFQStatus.CLOSED

        return RFQ(
            id=msg.get("id", ""),
            market_ticker=msg.get("market_ticker"),
            contracts=msg.get("contracts", 0),
            mve_collection_ticker=msg.get("mve_collection_ticker"),
            mve_selected_legs=legs,
            target_cost_centi_cents=msg.get("target_cost_centi_cents"),
            created_at=datetime.now(),
            status=status,
            creator_id=msg.get("creator_id"),
            rest_remainder=msg.get("rest_remainder", False),
        )

    def _log_decision(
        self,
        rfq: RFQ,
        action: str,
        reason: Optional[str] = None,
        filter_name: Optional[str] = None,
        theo_value: Optional[float] = None,
        quote_id: Optional[str] = None,
        yes_bid: Optional[str] = None,
        no_bid: Optional[str] = None,
        edge: Optional[float] = None,
        yes_edge: Optional[float] = None,
        no_edge: Optional[float] = None,
        leg_prices: Optional[dict[str, float]] = None,
        shadow_mode: bool = False,
        spread_multiplier: float = 1.0,
    ) -> None:
        """Log a decision to JSONL for post-session analysis."""
        record = create_decision_record(
            rfq=rfq,
            action=action,
            theo_value=theo_value,
            quote_id=quote_id,
            yes_bid=yes_bid,
            no_bid=no_bid,
            edge=edge,
            yes_edge=yes_edge,
            no_edge=no_edge,
            filter_reason=reason,
            filter_name=filter_name,
            leg_prices=leg_prices,
            shadow_mode=shadow_mode,
            spread_multiplier=spread_multiplier,
        )
        self.decision_logger.log_decision(record)

    async def _quote_expiry_monitor(self) -> None:
        """Monitor and proactively cancel stale quotes.

        Best practice: Cancel quotes proactively rather than letting them expire.
        We cancel quotes 5 seconds before TTL to ensure clean expiry.
        """
        CANCEL_BUFFER_SECONDS = 5.0  # Cancel this many seconds before TTL

        while self._running:
            await asyncio.sleep(1.0)

            # Get pending quotes that are approaching expiry
            pending = self.risk_manager.get_pending_quotes()
            cancel_threshold = self.config.quote_ttl_seconds - CANCEL_BUFFER_SECONDS

            for quote in pending:
                if quote.age_seconds >= cancel_threshold:
                    # Proactively cancel via API
                    logger.info(
                        f"Proactively cancelling stale quote: {quote.quote_id} "
                        f"(age={quote.age_seconds:.1f}s, ttl={self.config.quote_ttl_seconds}s)"
                    )
                    cancelled = await self._cancel_quote(quote.quote_id)
                    if cancelled:
                        self.risk_manager.release_quote(quote.quote_id)

            # Also clean up any quotes that somehow exceeded TTL
            expired = self.risk_manager.cleanup_expired_quotes(
                self.config.quote_ttl_seconds
            )
            for quote in expired:
                logger.debug(f"Cleaned up expired quote: {quote.quote_id}")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down RFQ Responder...")
        self._running = False

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Stop decision logger
        self.decision_logger.stop()

        # Disconnect
        await self.connector.disconnect_ws()
        await self.connector.stop()

        logger.info(
            f"Shutdown complete. Logged {self.decision_logger.decision_count} decisions, "
            f"{self.decision_logger.outcome_count} outcomes"
        )


async def run_rfq_responder(config_path: Optional[str] = None) -> int:
    """Run the RFQ responder (entry point)."""
    from .config import load_rfq_config

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = load_rfq_config(config_path)
    responder = RFQResponder(config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(responder.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await responder.start()
        return 0
    except Exception as e:
        logger.exception(f"RFQ Responder error: {e}")
        return 1
