"""
v3 Quoter - Passive Market Maker main loop.

Simple tick loop:
1. Update orderbook state
2. Generate quote decision via strategy
3. Reconcile quotes with exchange
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import ConfigV3
from .strategy import PassiveStrategy, QuoteDecision

# Reuse shared components from main src/
from ..connector import KalshiConnector, APIError
from ..orderbook import OrderBookManager
from ..config import VolatilityConfig, StrategyConfig
from ..execution import ExecutionEngine
from ..types import Position, StrategyOutput

logger = logging.getLogger(__name__)


@dataclass
class MarketContext:
    """Per-market state."""
    ticker: str
    position: Position = field(default_factory=lambda: Position(ticker=""))

    def __post_init__(self):
        if self.position.ticker == "":
            self.position = Position(ticker=self.ticker)


class PassiveQuoter:
    """
    Passive Market Maker v3.

    Simple, defensive liquidity provision:
    - Only quote markets with sufficient spread
    - Always join existing depth
    - Gently fade inventory
    """

    def __init__(self, config: ConfigV3):
        self.config = config
        self._running = False
        self._tick_count = 0

        # Initialize shared components
        self.connector = KalshiConnector(self._build_connector_config())
        vol_config = VolatilityConfig()  # Use defaults
        self.orderbook_manager = OrderBookManager(vol_config)
        self.strategy = PassiveStrategy(config)

        # Build a StrategyConfig for ExecutionEngine (it needs debounce settings etc)
        strategy_config = self._build_strategy_config()
        self.execution = ExecutionEngine(strategy_config, self.connector)

        # Per-market state
        self._markets: dict[str, MarketContext] = {}

        # Create market contexts
        for ticker in config.tickers:
            self._markets[ticker] = MarketContext(ticker=ticker)

        # Setup logging
        self._setup_logging()

    def _build_strategy_config(self) -> StrategyConfig:
        """Build a StrategyConfig for ExecutionEngine compatibility."""
        return StrategyConfig(
            max_order_size_dollars=self.config.sizing.max_order_size_dollars,
            max_inventory_dollars=self.config.sizing.max_inventory_dollars,
            quote_size_dollars=self.config.sizing.quote_size_dollars,
            # Disable features we don't use in v3
            min_join_depth_dollars=0,  # We handle this in strategy
            debounce_cents=2,
            debounce_seconds=5.0,
        )

    def _build_connector_config(self):
        """Build a connector-compatible config object."""
        # The connector expects certain attributes - create a minimal adapter
        class RateLimitConfig:
            read_rate = 20
            write_rate = 10

        class ConnectorConfig:
            def __init__(self, cfg: ConfigV3):
                self.environment = cfg.environment
                self.credentials = cfg.credentials
                self.rate_limit = RateLimitConfig()

        return ConnectorConfig(self.config)

    def _setup_logging(self):
        """Configure logging for this run."""
        log_dir = Path(self.config.logging.base_log_dir)
        run_dir = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # File handler
        file_handler = logging.FileHandler(run_dir / "ops.log")
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
        )

        # Configure root logger
        root = logging.getLogger()
        root.setLevel(getattr(logging, self.config.logging.log_level))
        root.addHandler(file_handler)

        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        root.addHandler(console)

        self._run_dir = run_dir
        logger.info(f"Logging to {run_dir}")

    async def start(self):
        """Start the quoter."""
        logger.info(f"Starting Passive Quoter v3 with {len(self._markets)} markets")
        logger.info(f"Environment: {self.config.environment.value}")
        logger.info(f"Quoting config: width>={self.config.quoting.min_market_width}c, "
                    f"depth>=${self.config.quoting.min_join_depth_dollars}")

        self._running = True

        try:
            # Start REST client
            await self.connector.start()

            # Load existing positions
            await self._load_positions()

            # Register message handler before connecting
            self.connector.on_message(self._handle_ws_message)

            # Connect websocket and subscribe
            await self.connector.connect_ws()
            tickers = list(self._markets.keys())
            await self.connector.subscribe_orderbook(tickers)
            await self.connector.subscribe_fills()

            # Run main loops
            await asyncio.gather(
                self.connector.run_ws_loop(),
                self._strategy_loop(),
            )

        except asyncio.CancelledError:
            logger.info("Quoter cancelled")
        except Exception as e:
            logger.exception(f"Quoter error: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown - cancel all orders."""
        logger.info("Shutting down...")
        self._running = False

        # Cancel all orders using ExecutionEngine
        for ticker in self._markets:
            await self.execution.cancel_all(ticker)

        await self.connector.disconnect_ws()
        await self.connector.stop()
        logger.info("Shutdown complete")

    async def _load_positions(self):
        """Load existing positions from exchange."""
        try:
            resp = await self.connector.get_positions()
            positions = resp.get("market_positions", [])

            our_tickers = set(self._markets.keys())
            for pos in positions:
                ticker = pos.get("ticker")
                if ticker in our_tickers:
                    net = pos.get("position", 0)
                    ctx = self._markets[ticker]
                    # Set yes/no contracts based on net position sign
                    if net >= 0:
                        ctx.position.yes_contracts = net
                        ctx.position.no_contracts = 0
                    else:
                        ctx.position.yes_contracts = 0
                        ctx.position.no_contracts = abs(net)
                    if net != 0:
                        logger.warning(f"[{ticker}] LOADED POSITION: {net} contracts")

        except APIError as e:
            logger.error(f"Failed to load positions: {e}")

    def _handle_ws_message(self, ws_msg):
        """Dispatch websocket messages by type."""
        msg_type = ws_msg.type
        msg = ws_msg.msg

        if msg_type == "orderbook_snapshot":
            self._handle_orderbook_snapshot(msg, ws_msg)
        elif msg_type == "orderbook_delta":
            self._handle_orderbook_delta(msg, ws_msg)
        elif msg_type == "fill":
            self._handle_fill(msg)
        elif msg_type == "RESET_BOOK":
            logger.warning(f"Book reset requested: {msg.get('reason')}")

    def _handle_orderbook_snapshot(self, msg: dict, ws_msg):
        """Handle orderbook snapshot from websocket."""
        ticker = msg.get("market_ticker")
        if ticker in self._markets:
            # Pass the original WSMessage to orderbook_manager
            self.orderbook_manager.handle_message(ws_msg)
            book = self.orderbook_manager.get(ticker)
            if book:
                logger.info(f"OrderBook snapshot: {ticker}, mid={book.mid_price()}")

    def _handle_orderbook_delta(self, msg: dict, ws_msg):
        """Handle orderbook delta from websocket."""
        ticker = msg.get("market_ticker")
        if ticker in self._markets:
            self.orderbook_manager.handle_message(ws_msg)

    def _handle_fill(self, msg: dict):
        """Handle fill notification."""
        ticker = msg.get("market_ticker")
        if ticker not in self._markets:
            return

        ctx = self._markets[ticker]
        side = msg.get("side")
        count = msg.get("count", 0)
        price = msg.get("yes_price", 0)
        order_id = msg.get("order_id", "")

        # Store inventory before fill for ExecutionEngine
        inventory_before = ctx.position.net_position

        # Update position (yes_contracts and no_contracts)
        if side == "yes":
            ctx.position.yes_contracts += count
        else:
            ctx.position.no_contracts += count

        logger.info(f"[{ticker}] FILL: {side} {count}@{price}, "
                    f"position now {ctx.position.net_position}")

        # Let ExecutionEngine handle order state tracking
        from ..constants import OrderSide
        order_side = OrderSide.BUY if side == "yes" else OrderSide.SELL
        self.execution.handle_fill(
            ticker=ticker,
            order_id=order_id,
            side=order_side,
            price=price,
            size=count,
            inventory_before=inventory_before,
        )

    async def _strategy_loop(self):
        """Main strategy tick loop."""
        interval = 0.2  # 200ms between ticks

        while self._running:
            tick_start = time.monotonic()

            for ticker, ctx in self._markets.items():
                try:
                    await self._tick_market(ctx)
                except Exception as e:
                    logger.error(f"[{ticker}] Tick error: {e}")

            self._tick_count += 1

            # Pace to interval
            elapsed = time.monotonic() - tick_start
            await asyncio.sleep(max(0, interval - elapsed))

    async def _tick_market(self, ctx: MarketContext):
        """Process one tick for a market."""
        ticker = ctx.ticker
        book = self.orderbook_manager.get(ticker)

        if book is None or not book.is_valid():
            return

        # Get book state
        best_bid = book.best_yes_bid()
        best_ask = book.best_yes_ask()
        volatility = self.orderbook_manager.get_volatility(ticker)
        inventory = ctx.position.net_position

        if best_bid is None or best_ask is None:
            return

        # Get raw book data for depth analysis
        bid_book = dict(book.yes_bids)  # {price: size}
        ask_book = dict(book.yes_asks)

        # Generate quote decision
        decision = self.strategy.generate_quotes(
            best_bid=best_bid,
            best_ask=best_ask,
            bid_book=bid_book,
            ask_book=ask_book,
            inventory=inventory,
            volatility=volatility,
        )

        # Update market state in ExecutionEngine for validation
        self.execution.set_market_state(
            ticker=ticker,
            best_bid=best_bid,
            best_ask=best_ask,
            book_depth=bid_book | ask_book,  # Merge both sides
        )

        # Convert QuoteDecision to StrategyOutput for ExecutionEngine
        should_bid = decision.bid_price is not None
        should_ask = decision.ask_price is not None

        mid = (best_bid + best_ask) / 2.0
        target = StrategyOutput(
            bid_price=decision.bid_price if should_bid else 1,
            ask_price=decision.ask_price if should_ask else 99,
            bid_size=decision.bid_size if should_bid else 0,
            ask_size=decision.ask_size if should_ask else 0,
            reservation_price=decision.reservation_price,
            spread=best_ask - best_bid,
            inventory_skew=mid - decision.reservation_price,
        )
        # ExecutionEngine handles diffing - will cancel existing orders if should_bid/ask is False
        await self.execution.update_quotes(ticker, target, should_bid, should_ask)

        # Periodic logging
        if self._tick_count % 50 == 0:
            mid = (best_bid + best_ask) / 2
            logger.info(
                f"[{ticker}] mid={mid:.1f}, "
                f"bid={decision.bid_price}, ask={decision.ask_price}, "
                f"inv={inventory}, res={decision.reservation_price:.1f}"
            )
            if decision.bid_price is None or decision.ask_price is None:
                logger.info(f"[{ticker}] {decision.reason}")

