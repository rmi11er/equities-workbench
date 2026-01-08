"""Main market maker orchestrator."""

import asyncio
import logging
import time
from typing import Optional

from .config import Config
from .connector import KalshiConnector, WSMessage
from .constants import STALE_DATA_THRESHOLD_SEC
from .execution import ExecutionEngine
from .logger import LogManager
from .orderbook import OrderBookManager
from .strategy import StoikovStrategy, AlphaEngine
from .types import Position

logger = logging.getLogger(__name__)


class MarketMaker:
    """
    Main market maker class.

    Orchestrates all components:
    - Connector: Exchange I/O
    - OrderBook: Market state
    - Strategy: Quote generation
    - Execution: Order management
    - Logger: Observability
    """

    def __init__(self, config: Config):
        self.config = config
        self.ticker = config.market_ticker

        # Core components
        self.connector = KalshiConnector(config)
        self.orderbook_manager = OrderBookManager(config.volatility)
        self.strategy = StoikovStrategy(config.strategy)
        self.execution = ExecutionEngine(config.strategy, self.connector)
        self.alpha_engine = AlphaEngine()
        self.log_manager = LogManager(config.logging)

        # State
        self._running = False
        self._position = Position(ticker=self.ticker)
        self._last_tick_time = 0.0
        self._tick_count = 0

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the market maker."""
        logger.info(f"Starting market maker for {self.ticker}")

        # Start logging
        self.log_manager.start()

        # Start connector
        await self.connector.start()

        # Connect WebSocket
        await self.connector.connect_ws()

        # Register message handler
        self.connector.on_message(self._handle_message)
        self.connector.on_reconnect(self._handle_reconnect)

        # Subscribe to orderbook
        await self.connector.subscribe_orderbook([self.ticker])

        # Subscribe to fills
        await self.connector.subscribe_fills()

        # Start alpha engine
        await self.alpha_engine.start()

        self._running = True
        logger.info("Market maker started")

    async def stop(self) -> None:
        """Stop the market maker."""
        logger.info("Stopping market maker...")
        self._running = False

        # Cancel all orders
        await self.execution.cancel_all(self.ticker)

        # Stop components
        await self.alpha_engine.stop()
        await self.connector.stop()
        self.log_manager.stop()

        logger.info("Market maker stopped")

    async def run(self) -> None:
        """
        Main run loop.

        Runs until interrupted or error.
        """
        try:
            await self.start()

            # Run WebSocket loop and strategy loop concurrently
            await asyncio.gather(
                self.connector.run_ws_loop(),
                self._strategy_loop(),
                self._stale_data_monitor(),
            )

        except asyncio.CancelledError:
            logger.info("Market maker cancelled")
        except Exception as e:
            logger.exception(f"Market maker error: {e}")
        finally:
            await self.stop()

    # -------------------------------------------------------------------------
    # Message Handling
    # -------------------------------------------------------------------------

    def _handle_message(self, msg: WSMessage) -> None:
        """Handle incoming WebSocket messages."""
        self._last_tick_time = time.monotonic()

        if msg.type in ("orderbook_snapshot", "orderbook_delta"):
            self.orderbook_manager.handle_message(msg)

        elif msg.type == "fill":
            self._handle_fill(msg)

        elif msg.type == "RESET_BOOK":
            # Sequence gap detected - pull quotes and wait for new snapshot
            asyncio.create_task(self.execution.pull_quotes(self.ticker))

        elif msg.type == "subscribed":
            logger.info(f"Subscribed to {msg.msg}")

        elif msg.type == "error":
            logger.error(f"WebSocket error: {msg.msg}")

    def _handle_fill(self, msg: WSMessage) -> None:
        """Handle a fill notification."""
        fill_data = msg.msg
        # TODO: Parse fill data and update position
        logger.info(f"Fill received: {fill_data}")

    def _handle_reconnect(self) -> None:
        """Handle WebSocket reconnection."""
        logger.warning("WebSocket reconnected - resetting state")
        self.orderbook_manager.reset()
        # Orders will be cancelled on reconnect via CANCEL_ALL strategy

    # -------------------------------------------------------------------------
    # Strategy Loop
    # -------------------------------------------------------------------------

    async def _strategy_loop(self) -> None:
        """
        Main strategy loop.

        Runs at fixed interval to:
        1. Get current market state
        2. Generate quotes
        3. Update execution
        4. Record to tape
        """
        interval = 0.1  # 100ms tick

        while self._running:
            tick_start = time.monotonic()

            try:
                await self._tick()
            except Exception as e:
                logger.exception(f"Strategy tick error: {e}")

            # Sleep for remainder of interval
            elapsed = time.monotonic() - tick_start
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def _tick(self) -> None:
        """Single strategy tick."""
        tick_start = time.monotonic()

        # Get order book
        book = self.orderbook_manager.get(self.ticker)
        if book is None:
            return

        # Get mid price
        mid = book.mid_price()
        if mid is None:
            return

        # Get volatility
        volatility = self.orderbook_manager.get_volatility(self.ticker)

        # Get external skew
        external_skew = self.alpha_engine.get_external_skew(self.ticker)

        # Get current inventory
        inventory = self._position.net_position

        # Generate quotes
        quotes = self.strategy.generate_quotes(
            mid_price=mid,
            inventory=inventory,
            volatility=volatility,
            external_skew=external_skew,
        )

        # Check if we should quote each side
        should_bid, should_ask = self.strategy.should_quote(inventory)

        # Update execution
        await self.execution.update_quotes(
            ticker=self.ticker,
            target=quotes,
            should_bid=should_bid,
            should_ask=should_ask,
        )

        # Calculate latency
        latency_ms = (time.monotonic() - tick_start) * 1000

        # Record to tape
        state = self.execution.get_quote_state(self.ticker)
        self.log_manager.record_tick(
            ticker=self.ticker,
            mid=mid,
            my_bid=state.last_bid_price,
            my_ask=state.last_ask_price,
            inventory=inventory,
            unrealized_pnl=self.execution.get_unrealized_pnl(
                self.ticker, mid, inventory
            ),
            realized_pnl=self.execution.realized_pnl,
            latency_ms=latency_ms,
            volatility=volatility,
        )

        self._tick_count += 1

        # Periodic logging
        if self._tick_count % 100 == 0:
            logger.info(
                f"Tick {self._tick_count}: mid={mid:.1f}, "
                f"bid={quotes.bid_price}, ask={quotes.ask_price}, "
                f"inv={inventory}, vol={volatility:.2f}, "
                f"latency={latency_ms:.1f}ms"
            )

    # -------------------------------------------------------------------------
    # Monitoring
    # -------------------------------------------------------------------------

    async def _stale_data_monitor(self) -> None:
        """Monitor for stale data and pull quotes if detected."""
        while self._running:
            await asyncio.sleep(1.0)

            if self._last_tick_time == 0:
                continue

            elapsed = time.monotonic() - self._last_tick_time

            if elapsed > STALE_DATA_THRESHOLD_SEC:
                logger.warning(f"Stale data detected: {elapsed:.1f}s since last message")
                await self.execution.pull_quotes(self.ticker)
                self._last_tick_time = time.monotonic()  # Reset to avoid spam
