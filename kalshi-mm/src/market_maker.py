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
from .pegged import PeggedStrategy
from .taker import ImpulseEngine, BailoutAction
from .types import Position, StrategyOutput
from .liquidity import analyze_liquidity, compute_adaptive_params
from .lip import LIPManager, LIPConstraints

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
        self.lip_manager = LIPManager(self.connector, max_tick_cap=config.lip.max_tick_cap)

        # V2 components
        self.pegged_strategy = PeggedStrategy(config.pegged_mode, config.strategy)
        self.impulse_engine = ImpulseEngine(
            config.impulse,
            config.risk,
            config.strategy,
        )

        # State
        self._running = False
        self._position = Position(ticker=self.ticker)
        self._last_tick_time = 0.0
        self._tick_count = 0
        self._market_expiry_ts: Optional[float] = None  # Unix timestamp of market expiry
        self._last_reservation_price: float = 50.0  # Track reservation for impulse checks

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

        # Fetch market expiry timestamp for dynamic time horizon
        await self._fetch_market_expiry()

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

        # Load LIP programs
        await self.lip_manager.refresh_programs(force=True)
        lip_program = self.lip_manager.get_program(self.ticker)
        if lip_program:
            logger.info(
                f"LIP active for {self.ticker}: "
                f"target_size={lip_program.target_size}, "
                f"discount={lip_program.discount_factor:.0%}, "
                f"reward=${lip_program.period_reward/100:.2f}"
            )
        else:
            logger.info(f"No active LIP program for {self.ticker}")

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

    async def _fetch_market_expiry(self) -> None:
        """Fetch market details to get expiry timestamp for dynamic time horizon."""
        try:
            resp = await self.connector._request("GET", f"/markets/{self.ticker}")
            market = resp.get("market", {})

            # Parse expiration_time (ISO 8601 format)
            expiry_str = market.get("expiration_time") or market.get("close_time")
            if expiry_str:
                from datetime import datetime
                # Parse ISO format: "2024-12-31T23:59:59Z"
                expiry_str = expiry_str.replace("Z", "+00:00")
                expiry_dt = datetime.fromisoformat(expiry_str)
                self._market_expiry_ts = expiry_dt.timestamp()
                logger.info(f"Market {self.ticker} expires at {expiry_str} (T-t normalization active)")
            else:
                logger.warning(f"No expiry found for {self.ticker}, using infinite time horizon")
                self._market_expiry_ts = None

        except Exception as e:
            logger.warning(f"Failed to fetch market expiry: {e}, using infinite time horizon")
            self._market_expiry_ts = None

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

    def _apply_lip_constraints(
        self,
        quotes: StrategyOutput,
        constraints: LIPConstraints,
        book,
    ) -> StrategyOutput:
        """
        Apply LIP constraints to ensure minimum compliance.

        Ensures:
        1. Size >= target_size (to qualify for points)
        2. Price within max_distance of best (to get meaningful score)
        """
        best_bid = book.best_yes_bid()
        best_ask = book.best_yes_ask()

        new_bid_price = quotes.bid_price
        new_ask_price = quotes.ask_price
        new_bid_size = quotes.bid_size
        new_ask_size = quotes.ask_size

        # Ensure minimum size
        new_bid_size = max(new_bid_size, constraints.min_size)
        new_ask_size = max(new_ask_size, constraints.min_size)

        # Cap at max order size
        max_size = self.config.strategy.max_order_size
        new_bid_size = min(new_bid_size, max_size)
        new_ask_size = min(new_ask_size, max_size)

        # Ensure within max_distance of best price (if best exists)
        if best_bid is not None:
            min_bid = best_bid - constraints.max_distance
            new_bid_price = max(new_bid_price, min_bid, 1)

        if best_ask is not None:
            max_ask = best_ask + constraints.max_distance
            new_ask_price = min(new_ask_price, max_ask, 99)

        # Ensure no cross
        if new_bid_price >= new_ask_price:
            mid = (new_bid_price + new_ask_price) // 2
            new_bid_price = max(1, mid - 1)
            new_ask_price = min(99, mid + 1)

        return StrategyOutput(
            bid_price=new_bid_price,
            ask_price=new_ask_price,
            bid_size=new_bid_size,
            ask_size=new_ask_size,
            reservation_price=quotes.reservation_price,
            spread=float(new_ask_price - new_bid_price),
            inventory_skew=quotes.inventory_skew,
        )

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

        # 1. UPDATE METRICS
        # Analyze liquidity
        liq_metrics = analyze_liquidity(book)
        adaptive = compute_adaptive_params(liq_metrics)

        # Get best bid/ask for impulse checks
        best_bid = book.best_yes_bid()
        best_ask = book.best_yes_ask()

        # Get effective pricing (depth-based)
        min_depth = self.config.strategy.effective_depth_contracts
        eff_bid, eff_ask = book.get_effective_quote(min_depth)
        effective_mid = (eff_bid + eff_ask) / 2.0
        effective_spread = float(eff_ask - eff_bid)

        # Get mid price - use effective mid, fall back to simple mid, then 50
        mid = effective_mid
        if liq_metrics.is_empty:
            mid = 50.0

        if liq_metrics.is_empty and self._tick_count % 100 == 0:
            logger.info("Empty orderbook - quoting wide at 1/99")
        elif self._tick_count % 100 == 0:
            logger.info(
                f"Liquidity score={liq_metrics.liquidity_score:.2f}, "
                f"spread_mult={adaptive.spread_multiplier:.1f}, "
                f"size_mult={adaptive.size_multiplier:.1f}, "
                f"eff_mid={effective_mid:.1f}, eff_spread={effective_spread:.0f}"
            )

        # Get volatility
        volatility = self.orderbook_manager.get_volatility(self.ticker)

        # Get external skew
        external_skew = self.alpha_engine.get_external_skew(self.ticker)

        # Get current inventory
        inventory = self._position.net_position

        # Determine regime
        regime = "PEGGED" if self.config.pegged_mode.enabled else "STANDARD"

        # 2. CHECK IMPULSE (Priority 1)
        if self.config.impulse.enabled:
            bailout_action = self.impulse_engine.check_bailout(
                inventory=inventory,
                reservation_price=self._last_reservation_price,
                best_bid=best_bid,
                best_ask=best_ask,
                regime=regime,
            )

            if bailout_action:
                logger.warning(f"IMPULSE TRIGGERED: {bailout_action}")
                await self._execute_bailout(bailout_action)
                # Skip quoting this tick to process the dump
                self._tick_count += 1
                return

        # 3. SELECT STRATEGY (Priority 2)
        if self.config.pegged_mode.enabled:
            # Use Pegged Strategy (Fixed Price, Max Size)
            quote_state = self.execution.get_quote_state(self.ticker)
            current_bid_size = quote_state.bid_order.remaining if quote_state.bid_order else 0
            current_ask_size = quote_state.ask_order.remaining if quote_state.ask_order else 0

            quotes = self.pegged_strategy.generate_quotes(
                inventory=inventory,
                current_bid_size=current_bid_size,
                current_ask_size=current_ask_size,
            )
            should_bid, should_ask = self.pegged_strategy.should_quote(inventory)

        else:
            # Use Stoikov Strategy with Effective Depth Pricing
            if liq_metrics.is_empty:
                # Empty book: quote as wide as possible (1/99) with full size
                # Max loss per contract is 1¢, potential gain is huge
                quotes = StrategyOutput(
                    bid_price=1,
                    ask_price=99,
                    bid_size=self.config.strategy.max_order_size,
                    ask_size=self.config.strategy.max_order_size,
                    reservation_price=50.0,
                    spread=98.0,
                    inventory_skew=0.0,
                )
            else:
                # Generate base quotes using effective mid price
                quotes = self.strategy.generate_quotes(
                    mid_price=effective_mid,  # Use effective mid for depth-based pricing
                    inventory=inventory,
                    volatility=volatility,
                    external_skew=external_skew,
                    expiry_ts=self._market_expiry_ts,
                    effective_spread=effective_spread,
                )

                # Apply liquidity-adaptive adjustments
                adjusted_spread = (quotes.ask_price - quotes.bid_price) * adaptive.spread_multiplier
                half_spread = int(adjusted_spread / 2)

                # Recalculate prices around reservation
                new_bid = max(1, int(quotes.reservation_price - half_spread))
                new_ask = min(99, int(quotes.reservation_price + half_spread))

                # Ensure no cross
                if new_bid >= new_ask:
                    new_bid = max(1, int(quotes.reservation_price) - 1)
                    new_ask = min(99, int(quotes.reservation_price) + 1)

                # Adjust sizes
                new_bid_size = max(1, int(quotes.bid_size * adaptive.size_multiplier))
                new_ask_size = max(1, int(quotes.ask_size * adaptive.size_multiplier))

                # Cap at max order size
                max_size = self.config.strategy.max_order_size
                new_bid_size = min(new_bid_size, max_size)
                new_ask_size = min(new_ask_size, max_size)

                quotes = StrategyOutput(
                    bid_price=new_bid,
                    ask_price=new_ask,
                    bid_size=new_bid_size,
                    ask_size=new_ask_size,
                    reservation_price=quotes.reservation_price,
                    spread=float(new_ask - new_bid),
                    inventory_skew=quotes.inventory_skew,
                )

            # Check if we should quote each side (Stoikov limits)
            should_bid, should_ask = self.strategy.should_quote(inventory)

        # Store reservation price for next impulse check
        self._last_reservation_price = quotes.reservation_price

        # Apply LIP constraints if active (both modes)
        lip_constraints = self.lip_manager.get_constraints(self.ticker)
        if lip_constraints and lip_constraints.is_active:
            quotes = self._apply_lip_constraints(quotes, lip_constraints, book)

        # 4. EXECUTE
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

        # Record LIP snapshot (every ~1 second, matching Kalshi's snapshot rate)
        if self._tick_count % 10 == 0:  # Every 10 ticks = ~1 second
            self.lip_manager.record_snapshot(
                ticker=self.ticker,
                our_bid_size=quotes.bid_size if should_bid else 0,
                our_ask_size=quotes.ask_size if should_ask else 0,
                our_bid_price=quotes.bid_price if should_bid else None,
                our_ask_price=quotes.ask_price if should_ask else None,
                best_bid=best_bid,
                best_ask=best_ask,
            )

        # Periodic logging
        if self._tick_count % 100 == 0:
            mode_str = "PEGGED" if self.config.pegged_mode.enabled else "STOIKOV"
            logger.info(
                f"Tick {self._tick_count} [{mode_str}]: mid={mid:.1f}, "
                f"bid={quotes.bid_price}, ask={quotes.ask_price}, "
                f"inv={inventory}, vol={volatility:.2f}, "
                f"latency={latency_ms:.1f}ms"
            )

            # Log LIP status if active
            lip_status = self.lip_manager.get_status(self.ticker)
            if lip_status.get("active"):
                logger.info(
                    f"LIP: uptime={lip_status['uptime_pct']:.1%}, "
                    f"qualifying={lip_status['snapshots_qualifying']}/{lip_status['snapshots_total']}, "
                    f"est_score={lip_status['estimated_score']:.0f}"
                )

    async def _execute_bailout(self, action: BailoutAction) -> None:
        """
        Execute a bailout action via IOC simulation.

        Args:
            action: The bailout action to execute
        """
        logger.warning(f"Executing bailout: {action}")

        result = await self.execution.send_ioc_simulation(
            ticker=self.ticker,
            side=action.side,
            price=action.aggressive_price,
            count=action.quantity,
        )

        if result.get("success"):
            logger.info(f"Bailout IOC sent: order_id={result.get('order_id')}")
        else:
            logger.error(f"Bailout IOC failed: {result.get('error')}")

    # -------------------------------------------------------------------------
    # Monitoring
    # -------------------------------------------------------------------------

    async def _stale_data_monitor(self) -> None:
        """Monitor for stale data and pull quotes if detected."""
        stale_count = 0

        while self._running:
            await asyncio.sleep(1.0)

            if self._last_tick_time == 0:
                continue

            elapsed = time.monotonic() - self._last_tick_time

            if elapsed > STALE_DATA_THRESHOLD_SEC:
                stale_count += 1

                # Only warn occasionally, not every 5 seconds
                if stale_count == 1 or stale_count % 12 == 0:  # First time, then every minute
                    logger.warning(f"Stale data: {elapsed:.1f}s since last message (market may be illiquid)")

                # Only pull quotes if we actually have orders out
                state = self.execution.get_quote_state(self.ticker)
                if state.bid_order or state.ask_order:
                    await self.execution.pull_quotes(self.ticker)

                self._last_tick_time = time.monotonic()
            else:
                stale_count = 0
