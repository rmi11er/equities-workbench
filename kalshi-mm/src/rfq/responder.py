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

        # Pricing engine with BBO source
        bbo_source = BBOPriceSource(self.orderbook_manager)
        self.pricing_engine = PricingEngine(config.pricing, bbo_source)

        # Risk manager
        self.risk_manager = RiskManager(config.risk)

        # Decision logger (JSONL output for post-session analysis)
        self.decision_logger = RFQDecisionLogger(config.logging.base_log_dir)

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

        # Subscribe to orderbooks for leg markets
        if self.config.leg_tickers:
            logger.info(f"Subscribing to {len(self.config.leg_tickers)} leg tickers")
            await self.connector.subscribe_orderbook(self.config.leg_tickers)

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
        elif msg_type == "quote_accepted":
            asyncio.create_task(self._handle_quote_accepted(ws_msg.msg))
        elif msg_type == "quote_executed":
            self._handle_quote_executed(ws_msg.msg)
        elif msg_type == "quote_expired":
            self._handle_quote_expired(ws_msg.msg)
        elif msg_type == "quote_deleted":
            self._handle_quote_deleted(ws_msg.msg)

        # Subscription confirmations
        elif msg_type == "subscribed":
            logger.debug(f"Subscription confirmed: {ws_msg.msg}")

    def _handle_reconnect(self) -> None:
        """Handle WebSocket reconnection."""
        logger.info("WebSocket reconnected, re-subscribing to channels")

    async def _handle_rfq_created(self, msg: dict) -> None:
        """
        Handle incoming RFQ request.

        1. Parse RFQ
        2. Check leg filters first (before pricing)
        3. Compute fair value
        4. Check dollar-based filters (after pricing)
        5. Check risk limits
        6. Send quote (with shadow mode spread if enabled)
        """
        rfq = self._parse_rfq(msg)
        shadow_mode = self.config.is_shadow_mode
        spread_mult = self.config.shadow.spread_multiplier if shadow_mode else 1.0

        logger.info(
            f"RFQ received: {rfq.id}, contracts={rfq.contracts}, "
            f"legs={rfq.leg_count}"
        )

        # Check pre-pricing filters (leg count, collections)
        passed, reason = self._passes_pre_filters(rfq)
        if not passed:
            self._log_decision(
                rfq, "filtered", reason,
                filter_name="pre_filter",
                shadow_mode=shadow_mode,
            )
            logger.info(f"RFQ {rfq.id} filtered (pre): {reason}")
            return

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
            )
            logger.info(f"RFQ {rfq.id} rejected (risk): {reason}")
            return

        # Create quote response (apply shadow spread multiplier)
        response = self._create_quote_with_spread(rfq, theo, spread_mult)

        # Send quote via REST
        quote_id = await self._send_quote(response)
        if quote_id is None:
            self._log_decision(
                rfq, "error", "API error",
                theo_value=theo,
                leg_prices=leg_prices,
                shadow_mode=shadow_mode,
            )
            logger.error(f"Failed to send quote for RFQ {rfq.id}")
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
            leg_prices=leg_prices,
            shadow_mode=shadow_mode,
            spread_multiplier=spread_mult,
        )

        mode_tag = " [SHADOW]" if shadow_mode else ""
        logger.info(
            f"Quote sent{mode_tag}: {quote_id} for RFQ {rfq.id}, "
            f"YES bid={response.yes_bid}, NO bid={response.no_bid}, "
            f"theo={theo:.4f}, edge={response.edge:.4f}"
        )

    def _create_quote_with_spread(self, rfq: RFQ, theo: float, spread_mult: float):
        """Create quote response, optionally with widened spread for shadow mode."""
        from .types import QuoteResponse

        # Get base spread from config
        base_spread = self.config.pricing.default_spread_pct

        # Apply multiplier (for shadow mode, this makes quotes uncompetitive)
        effective_spread = base_spread * spread_mult

        # Clamp spread to bounds (convert cents to probability)
        min_spread = self.config.pricing.min_spread_cents / 100.0
        max_spread = self.config.pricing.max_spread_cents / 100.0

        # In shadow mode, we want wider spreads, so use higher max
        if spread_mult > 1.0:
            max_spread = max(max_spread, effective_spread)

        spread = max(min_spread, min(effective_spread, max_spread))

        # Apply spread symmetrically
        yes_bid_prob = max(0.01, min(0.99, theo - spread / 2))
        no_bid_prob = max(0.01, min(0.99, (1.0 - theo) - spread / 2))

        yes_bid = f"{yes_bid_prob:.4f}"
        no_bid = f"{no_bid_prob:.4f}"
        edge = theo - yes_bid_prob

        return QuoteResponse(
            rfq_id=rfq.id,
            yes_bid=yes_bid,
            no_bid=no_bid,
            theo_value=theo,
            edge=edge,
        )

    def _passes_pre_filters(self, rfq: RFQ) -> tuple[bool, str]:
        """Check filters that don't require pricing."""
        filters = self.config.filters

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

    async def _send_quote(self, response) -> Optional[str]:
        """Send quote via REST API."""
        try:
            data = response.to_api_payload()
            resp = await self.connector._request(
                "POST",
                "/communications/quotes",
                data=data,
                is_write=True,
            )
            return resp.get("id") or resp.get("quote_id")
        except APIError as e:
            logger.error(f"Quote creation failed: {e}")
            return None

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

    def _parse_rfq(self, msg: dict) -> RFQ:
        """Parse RFQ from WebSocket message."""
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
            filter_reason=reason,
            filter_name=filter_name,
            leg_prices=leg_prices,
            shadow_mode=shadow_mode,
            spread_multiplier=spread_multiplier,
        )
        self.decision_logger.log_decision(record)

    async def _quote_expiry_monitor(self) -> None:
        """Monitor and clean up expired quotes."""
        while self._running:
            await asyncio.sleep(1.0)

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
