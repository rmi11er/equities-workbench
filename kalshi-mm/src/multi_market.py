"""Multi-market orchestrator for simultaneous quoting on multiple markets.

This module enables a single process to quote on 10-20 markets simultaneously,
sharing components like the WebSocket connection, rate limiter, and strategy config.

Key design decisions:
- Sequential tick loop (not parallel) to avoid rate limit collisions
- Shared components: connector, orderbook_manager, strategy, execution
- Per-market state: position, expiry timestamp, circuit breaker
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING

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
from .decision_log import (
    DecisionLogger, LatencyTimer, TickLatencyTracker,
    QuoteDecision, ImpulseEvent, FillEvent,
    MarketState, PositionState, LatencyBreakdown,
)

if TYPE_CHECKING:
    from .run_context import RunContext

logger = logging.getLogger(__name__)


@dataclass
class MarketContext:
    """
    Per-market state for multi-market mode.

    Each market maintains its own:
    - Position tracking
    - Expiry timestamp for time decay
    - Circuit breaker state
    - Reservation price for impulse checks
    """
    ticker: str
    position: Position = field(default_factory=lambda: Position(ticker=""))
    market_expiry_ts: Optional[float] = None
    last_reservation_price: float = 50.0
    peak_pnl: float = 0.0
    circuit_breaker_triggered: bool = False
    last_tick_time: float = 0.0

    def __post_init__(self):
        if not self.position.ticker:
            self.position = Position(ticker=self.ticker)


class MultiMarketOrchestrator:
    """
    Manages multiple markets with shared components.

    Runs a sequential tick loop over all markets, using shared:
    - KalshiConnector (single WebSocket connection)
    - OrderBookManager (already supports multiple tickers)
    - ExecutionEngine (already supports multiple tickers)
    - StoikovStrategy (stateless, same params for all)
    """

    def __init__(self, config: Config, run_context: Optional["RunContext"] = None):
        self.config = config
        self.run_context = run_context

        # Validate we have tickers
        if not config.tickers:
            raise ValueError("No tickers configured. Set market_ticker or market_tickers in config.")

        # Per-market contexts
        self._markets: dict[str, MarketContext] = {}
        for ticker in config.tickers:
            self._markets[ticker] = MarketContext(ticker=ticker)

        # Determine decision log path
        if run_context and run_context.decisions_path:
            decisions_path = run_context.decisions_path
        else:
            decisions_path = config.logging.ops_log_path.replace("ops.log", "decisions.jsonl")

        # Shared components
        self.decision_logger = DecisionLogger(log_path=decisions_path)
        self.connector = KalshiConnector(config)
        self.orderbook_manager = OrderBookManager(config.volatility)
        self.strategy = StoikovStrategy(config.strategy)
        self.execution = ExecutionEngine(
            config.strategy,
            self.connector,
            decision_logger=self.decision_logger,
        )
        self.alpha_engine = AlphaEngine()
        self.log_manager = LogManager(config.logging, run_context=run_context)
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
        self._tick_count = 0

    @property
    def tickers(self) -> list:
        """Get list of tickers being traded."""
        return list(self._markets.keys())

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the multi-market orchestrator."""
        logger.info(f"Starting multi-market orchestrator for {len(self._markets)} markets")
        logger.info(f"Markets: {', '.join(self.tickers)}")

        # Start logging
        self.log_manager.start()
        self.decision_logger.start()

        # Start connector
        await self.connector.start()

        # Fetch market expiry timestamps for all markets
        await self._fetch_all_market_expiries()

        # Connect WebSocket
        await self.connector.connect_ws()

        # Register message handler
        self.connector.on_message(self._handle_message)
        self.connector.on_reconnect(self._handle_reconnect)

        # Subscribe to orderbook for ALL tickers at once
        await self.connector.subscribe_orderbook(self.tickers)

        # Subscribe to fills
        await self.connector.subscribe_fills()

        # Start alpha engine
        await self.alpha_engine.start()

        # Load LIP programs for all markets
        await self.lip_manager.refresh_programs(force=True)
        for ticker in self.tickers:
            lip_program = self.lip_manager.get_program(ticker)
            if lip_program:
                logger.info(
                    f"LIP active for {ticker}: "
                    f"target_size={lip_program.target_size}, "
                    f"discount={lip_program.discount_factor:.0%}"
                )

        self._running = True
        logger.info("Multi-market orchestrator started")

    async def stop(self) -> None:
        """Stop the multi-market orchestrator."""
        logger.info("Stopping multi-market orchestrator...")
        self._running = False

        # Cancel all orders on all markets
        for ticker in self.tickers:
            try:
                await self.execution.cancel_all(ticker)
            except Exception as e:
                logger.error(f"Error cancelling orders for {ticker}: {e}")

        # Stop components
        try:
            await self.alpha_engine.stop()
        except Exception as e:
            logger.error(f"Error stopping alpha engine: {e}")

        try:
            await self.connector.stop()
        except Exception as e:
            logger.error(f"Error stopping connector: {e}")

        self.decision_logger.stop()
        self.log_manager.stop()

        logger.info("Multi-market orchestrator stopped")

    async def run(self) -> None:
        """Main run loop."""
        try:
            await self.start()

            # Run WebSocket loop and strategy loop concurrently
            await asyncio.gather(
                self.connector.run_ws_loop(),
                self._strategy_loop(),
                self._stale_data_monitor(),
            )

        except asyncio.CancelledError:
            logger.info("Multi-market orchestrator cancelled")
        except Exception as e:
            logger.exception(f"Multi-market orchestrator error: {e}")
        finally:
            await self.stop()

    async def _fetch_all_market_expiries(self) -> None:
        """Fetch market expiry timestamps for all markets."""
        for ticker in self.tickers:
            ctx = self._markets[ticker]
            try:
                resp = await self.connector._request("GET", f"/markets/{ticker}")
                market = resp.get("market", {})

                expiry_str = market.get("expiration_time") or market.get("close_time")
                if expiry_str:
                    expiry_str = expiry_str.replace("Z", "+00:00")
                    expiry_dt = datetime.fromisoformat(expiry_str)
                    ctx.market_expiry_ts = expiry_dt.timestamp()
                    logger.info(f"Market {ticker} expires at {expiry_str}")
                else:
                    logger.warning(f"No expiry found for {ticker}")

            except Exception as e:
                logger.warning(f"Failed to fetch expiry for {ticker}: {e}")

    # -------------------------------------------------------------------------
    # Message Handling
    # -------------------------------------------------------------------------

    def _handle_message(self, msg: WSMessage) -> None:
        """Handle incoming WebSocket messages."""
        # Update tick time for the relevant market
        ticker = msg.ticker if hasattr(msg, 'ticker') and msg.ticker else None

        if msg.type in ("orderbook_snapshot", "orderbook_delta"):
            self.orderbook_manager.handle_message(msg)
            # Update last tick time for this market
            if ticker and ticker in self._markets:
                self._markets[ticker].last_tick_time = time.monotonic()

        elif msg.type == "fill":
            self._handle_fill(msg)

        elif msg.type == "RESET_BOOK":
            # Sequence gap detected - pull quotes for affected market
            if ticker and ticker in self._markets:
                asyncio.create_task(self.execution.pull_quotes(ticker))

        elif msg.type == "subscribed":
            logger.info(f"Subscribed to {msg.msg}")

        elif msg.type == "error":
            logger.error(f"WebSocket error: {msg.msg}")

    def _handle_fill(self, msg: WSMessage) -> None:
        """Handle a fill notification."""
        fill_data = msg.msg
        logger.info(f"Fill received: {fill_data}")

    def _handle_reconnect(self) -> None:
        """Handle WebSocket reconnection."""
        logger.warning("WebSocket reconnected - resetting state")
        self.orderbook_manager.reset()

    # -------------------------------------------------------------------------
    # Strategy Loop
    # -------------------------------------------------------------------------

    async def _strategy_loop(self) -> None:
        """
        Main strategy loop - sequential tick over all markets.

        We iterate through markets sequentially rather than in parallel because:
        1. API rate limit is shared (10 writes/sec)
        2. WebSocket delivers updates sequentially
        3. Strategy calc is fast (~100us)

        We add a small delay between markets to pace order creation and
        avoid exhausting the rate limit bucket on startup.
        """
        interval = 0.1  # 100ms between full cycles
        # Pace order creation: ~100ms between markets ensures we stay under 10 writes/sec
        # (2 orders per market = 20 orders/sec max if no delay)
        inter_market_delay = 0.12  # 120ms between markets

        while self._running:
            tick_start = time.monotonic()

            for ctx in self._markets.values():
                if ctx.circuit_breaker_triggered:
                    continue

                try:
                    await self._tick_market(ctx)
                except Exception as e:
                    logger.exception(f"Strategy tick error for {ctx.ticker}: {e}")

                # Pace between markets to respect rate limits
                await asyncio.sleep(inter_market_delay)

            # Sleep for remainder of interval (if any)
            elapsed = time.monotonic() - tick_start
            min_cycle_time = len(self._markets) * inter_market_delay
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)

            self._tick_count += 1

    async def _tick_market(self, ctx: MarketContext) -> None:
        """Single strategy tick for one market."""
        ticker = ctx.ticker
        latency_tracker = TickLatencyTracker()

        # Get order book
        book = self.orderbook_manager.get(ticker)
        if book is None:
            return

        # Validate book state
        if not book.is_valid():
            logger.error(f"INVALID ORDERBOOK for {ticker} - refusing to quote")
            await self.execution.pull_quotes(ticker)
            return

        # Update metrics
        with LatencyTimer() as t:
            liq_metrics = analyze_liquidity(book)
            adaptive = compute_adaptive_params(liq_metrics)
            best_bid = book.best_yes_bid()
            best_ask = book.best_yes_ask()
        latency_tracker.record_orderbook_update(t.elapsed_us)

        # Effective pricing
        with LatencyTimer() as t:
            min_depth = self.config.strategy.effective_depth_contracts
            eff_bid, eff_ask = book.get_effective_quote(min_depth)
            effective_mid = (eff_bid + eff_ask) / 2.0
            effective_spread = float(eff_ask - eff_bid)
        latency_tracker.record_effective_quote(t.elapsed_us)

        simple_mid = liq_metrics.mid_price if liq_metrics.mid_price else 50.0
        mid = effective_mid if not liq_metrics.is_empty else 50.0

        # Get metrics
        volatility = self.orderbook_manager.get_volatility(ticker)
        external_skew = self.alpha_engine.get_external_skew(ticker)
        inventory = ctx.position.net_position
        ofi = book.get_ofi(levels=3)

        regime = "PEGGED" if self.config.pegged_mode.enabled else "STANDARD"

        # Build state for logging
        market_state = MarketState(
            best_bid=best_bid,
            best_ask=best_ask,
            effective_bid=eff_bid,
            effective_ask=eff_ask,
            effective_mid=effective_mid,
            effective_spread=effective_spread,
            simple_mid=simple_mid,
            volatility=volatility,
            ofi=ofi,
            liquidity_score=liq_metrics.liquidity_score,
        )

        position_state = PositionState(
            inventory=inventory,
            unrealized_pnl=self.execution.get_unrealized_pnl(ticker, mid, inventory),
            realized_pnl=self.execution.realized_pnl,
        )

        # Check circuit breaker
        if self._check_circuit_breaker(ctx, position_state.realized_pnl, position_state.unrealized_pnl):
            await self._shutdown_market(ctx, "Circuit breaker triggered")
            return

        # Check API error circuit breaker
        if self.execution.should_halt_on_errors():
            await self._emergency_shutdown("Too many consecutive API errors")
            return

        # Check impulse
        with LatencyTimer() as t:
            bailout_action = None
            if self.config.impulse.enabled:
                bailout_action = self.impulse_engine.check_bailout(
                    inventory=inventory,
                    reservation_price=ctx.last_reservation_price,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    regime=regime,
                )
        latency_tracker.record_impulse_check(t.elapsed_us)

        if bailout_action:
            logger.warning(f"IMPULSE TRIGGERED for {ticker}: {bailout_action}")
            await self._execute_bailout(ticker, bailout_action)
            return

        # Generate quotes
        with LatencyTimer() as t:
            if self.config.pegged_mode.enabled:
                quote_state = self.execution.get_quote_state(ticker)
                current_bid_size = quote_state.bid_order.remaining if quote_state.bid_order else 0
                current_ask_size = quote_state.ask_order.remaining if quote_state.ask_order else 0

                quotes = self.pegged_strategy.generate_quotes(
                    inventory=inventory,
                    current_bid_size=current_bid_size,
                    current_ask_size=current_ask_size,
                )
                should_bid, should_ask = self.pegged_strategy.should_quote(inventory)
            else:
                if liq_metrics.is_empty:
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
                    quotes = self.strategy.generate_quotes(
                        mid_price=effective_mid,
                        inventory=inventory,
                        volatility=volatility,
                        external_skew=external_skew,
                        expiry_ts=ctx.market_expiry_ts,
                        effective_spread=effective_spread,
                    )

                    # Apply liquidity-adaptive adjustments
                    adjusted_spread = (quotes.ask_price - quotes.bid_price) * adaptive.spread_multiplier
                    half_spread = int(adjusted_spread / 2)

                    new_bid = max(1, int(quotes.reservation_price - half_spread))
                    new_ask = min(99, int(quotes.reservation_price + half_spread))

                    if new_bid >= new_ask:
                        new_bid = max(1, int(quotes.reservation_price) - 1)
                        new_ask = min(99, int(quotes.reservation_price) + 1)

                    new_bid_size = max(1, int(quotes.bid_size * adaptive.size_multiplier))
                    new_ask_size = max(1, int(quotes.ask_size * adaptive.size_multiplier))

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

                should_bid, should_ask = self.strategy.should_quote(inventory)
        latency_tracker.record_strategy_calc(t.elapsed_us)

        # Store reservation price
        ctx.last_reservation_price = quotes.reservation_price

        # Apply LIP constraints
        lip_adjusted = False
        lip_constraints = self.lip_manager.get_constraints(ticker)
        with LatencyTimer() as t:
            if lip_constraints and lip_constraints.is_active:
                quotes = self._apply_lip_constraints(quotes, lip_constraints, book)
                lip_adjusted = True
        latency_tracker.record_lip_adjustment(t.elapsed_us)

        # Validate quote pair
        is_valid, error_msg = self.execution.validate_quote_pair(
            quotes.bid_price if should_bid else None,
            quotes.ask_price if should_ask else None,
        )
        if not is_valid:
            logger.error(f"QUOTE PAIR BLOCKED for {ticker}: {error_msg}")
            return

        # Execute
        with LatencyTimer() as t:
            self.execution.set_market_state(ticker, best_bid, best_ask)
            await self.execution.update_quotes(
                ticker=ticker,
                target=quotes,
                should_bid=should_bid,
                should_ask=should_ask,
            )
        latency_tracker.record_execution(t.elapsed_us)

        latency_breakdown = latency_tracker.finalize()

        # Update market context
        self.execution.update_market_context(ticker, mid)

        # Record to tape
        latency_ms = latency_breakdown.total_tick_us / 1000.0
        state = self.execution.get_quote_state(ticker)
        self.log_manager.record_tick(
            ticker=ticker,
            mid=mid,
            my_bid=state.last_bid_price,
            my_ask=state.last_ask_price,
            inventory=inventory,
            unrealized_pnl=position_state.unrealized_pnl,
            realized_pnl=position_state.realized_pnl,
            latency_ms=latency_ms,
            volatility=volatility,
        )

        # Periodic logging (less frequent in multi-market mode)
        if self._tick_count % 50 == 0:
            logger.info(
                f"[{ticker}] mid={mid:.1f}, bid={quotes.bid_price}, ask={quotes.ask_price}, "
                f"inv={inventory}, vol={volatility:.2f}"
            )

    def _apply_lip_constraints(
        self,
        quotes: StrategyOutput,
        constraints: LIPConstraints,
        book,
    ) -> StrategyOutput:
        """Apply LIP constraints to ensure minimum compliance."""
        best_bid = book.best_yes_bid()
        best_ask = book.best_yes_ask()

        new_bid_price = quotes.bid_price
        new_ask_price = quotes.ask_price
        new_bid_size = max(quotes.bid_size, constraints.min_size)
        new_ask_size = max(quotes.ask_size, constraints.min_size)

        max_size = self.config.strategy.max_order_size
        new_bid_size = min(new_bid_size, max_size)
        new_ask_size = min(new_ask_size, max_size)

        if best_bid is not None:
            min_bid = best_bid - constraints.max_distance
            new_bid_price = max(new_bid_price, min_bid, 1)

        if best_ask is not None:
            max_ask = best_ask + constraints.max_distance
            new_ask_price = min(new_ask_price, max_ask, 99)

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

    async def _execute_bailout(self, ticker: str, action: BailoutAction) -> None:
        """Execute a bailout action."""
        logger.warning(f"Executing bailout for {ticker}: {action}")

        result = await self.execution.send_ioc_simulation(
            ticker=ticker,
            side=action.side,
            price=action.aggressive_price,
            count=action.quantity,
        )

        if result.get("success"):
            logger.info(f"Bailout IOC sent for {ticker}: order_id={result.get('order_id')}")
        else:
            logger.error(f"Bailout IOC failed for {ticker}: {result.get('error')}")

    def _check_circuit_breaker(
        self,
        ctx: MarketContext,
        realized_pnl: float,
        unrealized_pnl: float,
    ) -> bool:
        """Check if circuit breaker should trigger for a market."""
        if ctx.circuit_breaker_triggered:
            return True

        total_pnl = realized_pnl + unrealized_pnl

        if total_pnl > ctx.peak_pnl:
            ctx.peak_pnl = total_pnl

        max_loss = self.config.risk.max_loss_cents
        if max_loss > 0 and realized_pnl < -max_loss:
            logger.critical(f"CIRCUIT BREAKER for {ctx.ticker}: Max loss exceeded!")
            ctx.circuit_breaker_triggered = True
            return True

        max_dd = self.config.risk.max_drawdown_cents
        if max_dd > 0:
            drawdown = ctx.peak_pnl - total_pnl
            if drawdown > max_dd:
                logger.critical(f"CIRCUIT BREAKER for {ctx.ticker}: Max drawdown exceeded!")
                ctx.circuit_breaker_triggered = True
                return True

        return False

    async def _shutdown_market(self, ctx: MarketContext, reason: str) -> None:
        """Shutdown quoting for a single market."""
        logger.warning(f"Shutting down market {ctx.ticker}: {reason}")
        ctx.circuit_breaker_triggered = True

        try:
            await self.execution.cancel_all(ctx.ticker)
        except Exception as e:
            logger.error(f"Failed to cancel orders for {ctx.ticker}: {e}")

    async def _emergency_shutdown(self, reason: str) -> None:
        """Emergency shutdown - cancel all orders on all markets and stop."""
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")

        for ticker in self.tickers:
            try:
                await self.execution.cancel_all(ticker)
            except Exception as e:
                logger.error(f"Failed to cancel orders for {ticker}: {e}")

        self._running = False

    # -------------------------------------------------------------------------
    # Monitoring
    # -------------------------------------------------------------------------

    async def _stale_data_monitor(self) -> None:
        """Monitor for stale data across all markets."""
        stale_logged: dict[str, bool] = {t: False for t in self.tickers}

        while self._running:
            await asyncio.sleep(5.0)

            for ticker, ctx in self._markets.items():
                if ctx.last_tick_time == 0:
                    continue

                elapsed = time.monotonic() - ctx.last_tick_time

                if elapsed > STALE_DATA_THRESHOLD_SEC:
                    if not stale_logged[ticker]:
                        logger.warning(
                            f"No orderbook updates for {ticker} in {elapsed:.0f}s - "
                            f"market may be illiquid"
                        )
                        stale_logged[ticker] = True
                    elif elapsed > 300 and int(elapsed) % 300 < 5:
                        logger.warning(f"Still no updates for {ticker} ({elapsed/60:.1f} min)")
                else:
                    stale_logged[ticker] = False
