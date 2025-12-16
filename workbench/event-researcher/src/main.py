#!/usr/bin/env python3
"""Main entry point for Event Researcher CLI."""

import argparse
import asyncio
import sys
from datetime import date, timedelta

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.data.database import Database
from src.utils.config import get_settings
from src.utils.logging import setup_logging, get_logger


console = Console()
logger = get_logger(__name__)


async def cmd_status():
    """Show current status - watchlist count, price data coverage, upcoming events."""
    db = Database()
    await db.initialize()

    console.print("\n[bold]Event Researcher Status[/bold]\n")

    # Watchlist
    watchlist = await db.get_watchlist()
    console.print(f"Watchlist: [cyan]{len(watchlist)}[/cyan] symbols")

    if len(watchlist) > 0:
        # Price coverage
        symbols_with_prices = 0
        for symbol in watchlist["symbol"].to_list():
            latest = await db.get_latest_price_date(symbol)
            if latest:
                symbols_with_prices += 1

        console.print(f"Price data: [cyan]{symbols_with_prices}[/cyan] symbols have data")

    # Upcoming events
    today = date.today()
    next_week = today + timedelta(days=7)
    events = await db.get_events(start_date=today, end_date=next_week)
    console.print(f"Upcoming events (7 days): [cyan]{len(events)}[/cyan]\n")

    await db.close()


async def cmd_watchlist():
    """Display the current watchlist."""
    db = Database()
    await db.initialize()

    watchlist = await db.get_watchlist()

    if len(watchlist) == 0:
        console.print("[yellow]Watchlist is empty. Run seed_watchlist.py to populate.[/yellow]")
        return

    table = Table(title="Watchlist")
    table.add_column("Symbol", style="cyan")
    table.add_column("Name")
    table.add_column("Sector")
    table.add_column("Market Cap")

    for row in watchlist.iter_rows(named=True):
        market_cap = row["market_cap"]
        if market_cap:
            if market_cap >= 1e12:
                mc_str = f"${market_cap/1e12:.1f}T"
            elif market_cap >= 1e9:
                mc_str = f"${market_cap/1e9:.1f}B"
            else:
                mc_str = f"${market_cap/1e6:.1f}M"
        else:
            mc_str = "-"

        table.add_row(
            row["symbol"],
            row["name"] or "-",
            row["sector"] or "-",
            mc_str,
        )

    console.print(table)
    await db.close()


async def cmd_events(days: int = 7):
    """Display upcoming events."""
    db = Database()
    await db.initialize()

    today = date.today()
    end_date = today + timedelta(days=days)

    events = await db.get_events(start_date=today, end_date=end_date)

    if len(events) == 0:
        console.print(f"[yellow]No events in the next {days} days.[/yellow]")
        console.print("Run refresh_data.py to fetch earnings calendar.")
        return

    table = Table(title=f"Upcoming Events (Next {days} Days)")
    table.add_column("Date", style="cyan")
    table.add_column("Symbol", style="green")
    table.add_column("Type")
    table.add_column("Time")
    table.add_column("Title")

    for row in events.iter_rows(named=True):
        table.add_row(
            str(row["event_date"]),
            row["symbol"] or "-",
            row["event_type"],
            row["event_time"] or "-",
            row["title"] or "-",
        )

    console.print(table)
    await db.close()


async def cmd_prices(symbol: str, days: int = 30):
    """Display recent prices for a symbol."""
    db = Database()
    await db.initialize()

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    prices = await db.get_prices([symbol], start_date, end_date)

    if len(prices) == 0:
        console.print(f"[yellow]No price data for {symbol}.[/yellow]")
        console.print("Run backfill_prices.py to fetch historical data.")
        return

    table = Table(title=f"{symbol} Prices (Last {days} Days)")
    table.add_column("Date", style="cyan")
    table.add_column("Open", justify="right")
    table.add_column("High", justify="right")
    table.add_column("Low", justify="right")
    table.add_column("Close", justify="right")
    table.add_column("Volume", justify="right")

    # Show last 10 rows
    for row in prices.tail(10).iter_rows(named=True):
        timestamp = row["timestamp"]
        date_str = timestamp.strftime("%Y-%m-%d") if hasattr(timestamp, "strftime") else str(timestamp)[:10]

        volume = row["volume"]
        if volume >= 1e6:
            vol_str = f"{volume/1e6:.1f}M"
        else:
            vol_str = f"{volume/1e3:.1f}K"

        table.add_row(
            date_str,
            f"{row['open']:.2f}",
            f"{row['high']:.2f}",
            f"{row['low']:.2f}",
            f"{row['close']:.2f}",
            vol_str,
        )

    console.print(table)

    # Show summary stats
    closes = prices["close"].to_list()
    if len(closes) >= 2:
        first_close = float(closes[0])
        last_close = float(closes[-1])
        change_pct = ((last_close - first_close) / first_close) * 100

        console.print(f"\n{days}d change: ", end="")
        if change_pct >= 0:
            console.print(f"[green]+{change_pct:.2f}%[/green]")
        else:
            console.print(f"[red]{change_pct:.2f}%[/red]")

    await db.close()


async def cmd_earnings(symbol: str):
    """Display earnings history for a symbol."""
    db = Database()
    await db.initialize()

    earnings = await db.get_earnings(symbols=[symbol], with_actuals_only=True)

    if len(earnings) == 0:
        console.print(f"[yellow]No earnings data for {symbol}.[/yellow]")
        return

    table = Table(title=f"{symbol} Earnings History")
    table.add_column("Date", style="cyan")
    table.add_column("Quarter")
    table.add_column("EPS Est", justify="right")
    table.add_column("EPS Act", justify="right")
    table.add_column("Surprise", justify="right")

    for row in earnings.head(8).iter_rows(named=True):
        eps_est = row["eps_estimate"]
        eps_act = row["eps_actual"]
        surprise = row["eps_surprise_pct"]

        eps_est_str = f"{eps_est:.2f}" if eps_est else "-"
        eps_act_str = f"{eps_act:.2f}" if eps_act else "-"

        if surprise:
            if float(surprise) >= 0:
                surprise_str = f"[green]+{float(surprise):.1f}%[/green]"
            else:
                surprise_str = f"[red]{float(surprise):.1f}%[/red]"
        else:
            surprise_str = "-"

        table.add_row(
            str(row["event_date"]),
            row["fiscal_quarter"] or "-",
            eps_est_str,
            eps_act_str,
            surprise_str,
        )

    console.print(table)
    await db.close()


async def cmd_monitor(days: int = 14, high_only: bool = False):
    """Display surfaced events with interest scoring."""
    from src.monitor import EventSurfacer, InterestTier

    db = Database()
    await db.initialize()

    surfacer = EventSurfacer(db)
    min_tier = InterestTier.HIGH if high_only else InterestTier.LOW

    with console.status("[bold blue]Analyzing events...", spinner="dots"):
        events = await surfacer.get_surfaced_events(
            lookahead_days=days,
            min_tier=min_tier,
        )

    if len(events) == 0:
        console.print(f"[yellow]No surfaced events in the next {days} days.[/yellow]")
        await db.close()
        return

    # Group by tier
    grouped = await surfacer.get_events_by_tier(lookahead_days=days)

    # Display high interest events
    if grouped[InterestTier.HIGH]:
        console.print("\n[bold red]High Interest Events[/bold red]")
        table = Table(show_header=True, header_style="bold red")
        table.add_column("Days")
        table.add_column("Symbol")
        table.add_column("Type")
        table.add_column("Score")
        table.add_column("Flags")

        for event in grouped[InterestTier.HIGH]:
            days_str = "TODAY" if event.days_until == 0 else f"{event.days_until}d"
            flags_str = ", ".join(f.name for f in event.flags) if event.flags else "-"
            table.add_row(
                days_str,
                event.symbol,
                event.event_type,
                str(event.interest_score),
                flags_str,
            )
        console.print(table)

    # Display standard events
    if grouped[InterestTier.STANDARD] and not high_only:
        console.print("\n[bold yellow]Standard Interest Events[/bold yellow]")
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("Days")
        table.add_column("Symbol")
        table.add_column("Type")
        table.add_column("Score")
        table.add_column("Flags")

        for event in grouped[InterestTier.STANDARD]:
            days_str = "TODAY" if event.days_until == 0 else f"{event.days_until}d"
            flags_str = ", ".join(f.name for f in event.flags) if event.flags else "-"
            table.add_row(
                days_str,
                event.symbol,
                event.event_type,
                str(event.interest_score),
                flags_str,
            )
        console.print(table)

    # Display low interest events (summary only)
    if grouped[InterestTier.LOW] and not high_only:
        console.print(f"\n[dim]Low interest: {len(grouped[InterestTier.LOW])} events[/dim]")

    # Summary
    console.print(f"\n[bold]Total:[/bold] {len(events)} events surfaced")

    await db.close()


def cmd_ui():
    """Launch the Textual TUI application."""
    from src.ui.app import run_app
    run_app()


async def cmd_sessions():
    """List saved research sessions."""
    db = Database()
    await db.initialize()

    sessions = await db.list_sessions()

    if len(sessions) == 0:
        console.print("[yellow]No saved sessions.[/yellow]")
        await db.close()
        return

    table = Table(title="Research Sessions")
    table.add_column("ID", style="dim")
    table.add_column("Title")
    table.add_column("Symbol")
    table.add_column("Status")
    table.add_column("Updated")

    for row in sessions.iter_rows(named=True):
        session_id = row["session_id"][:8] + "..."
        title = row["title"] or "Untitled"
        if len(title) > 40:
            title = title[:37] + "..."

        updated = row["updated_at"]
        if updated:
            updated_str = str(updated)[:16]
        else:
            updated_str = "-"

        table.add_row(
            session_id,
            title,
            row["target_symbol"] or "-",
            row["status"],
            updated_str,
        )

    console.print(table)
    console.print("\n[dim]Use 'chat --resume <session_id>' to continue a session[/dim]")
    await db.close()


async def cmd_chat_with_session(session_id: str | None = None):
    """Interactive chat with optional session resume."""
    from src.agent.tools import registry  # noqa: F401 - registers tools
    from src.agent.orchestrator import AgentOrchestrator

    settings = get_settings()

    if not settings.anthropic_api_key:
        console.print(
            "[red]Error: ANTHROPIC_API_KEY not configured.[/red]\n"
            "Set it in your .env file or environment."
        )
        return

    db = Database()
    await db.initialize()

    orchestrator = AgentOrchestrator(db=db)

    # Resume session if specified
    if session_id:
        # Support partial session IDs
        sessions = await db.list_sessions()
        matching = None
        for row in sessions.iter_rows(named=True):
            if row["session_id"].startswith(session_id):
                matching = row["session_id"]
                break

        if matching:
            success = await orchestrator.resume_session(matching)
            if success:
                console.print(f"[green]Resumed session: {matching[:8]}...[/green]\n")
            else:
                console.print(f"[yellow]Could not find session: {session_id}[/yellow]")
        else:
            console.print(f"[yellow]No session matching: {session_id}[/yellow]")

    console.print(
        Panel(
            "[bold]Event Researcher Chat[/bold]\n\n"
            "Ask questions about events, prices, and earnings.\n"
            "Type 'quit' or 'exit' to end the session.\n"
            "Type 'clear' to reset conversation history.\n"
            "Type 'save' to save the current session.",
            title="Welcome",
            border_style="blue",
        )
    )

    tools = registry.list_tools()
    console.print(f"\n[dim]Available tools: {', '.join(tools)}[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                # Auto-save on exit if there are messages
                if orchestrator.get_message_count() > 0:
                    session_id = await orchestrator.save_session()
                    console.print(f"\n[dim]Session saved: {session_id[:8]}...[/dim]")
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "clear":
                orchestrator.reset_conversation()
                console.print("[dim]Conversation reset. New session started.[/dim]\n")
                continue

            if user_input.lower() == "save":
                session_id = await orchestrator.save_session()
                console.print(f"[green]Session saved: {session_id[:8]}...[/green]\n")
                continue

            if user_input.lower() == "help":
                console.print(
                    "\n[bold]Commands:[/bold]\n"
                    "  quit, exit, q - Exit chat (auto-saves)\n"
                    "  clear - Reset conversation, start new session\n"
                    "  save - Save current session\n"
                    "  help - Show this help\n"
                )
                continue

            # Process with agent
            console.print()
            with console.status("[bold blue]Thinking...", spinner="dots"):
                response = await orchestrator.chat(user_input)

            # Display response with markdown rendering
            console.print("[bold blue]Assistant:[/bold blue]")
            console.print(Markdown(response))
            console.print()

        except KeyboardInterrupt:
            console.print("\n\n[dim]Interrupted. Type 'quit' to exit.[/dim]\n")
            continue
        except Exception as e:
            logger.exception("Chat error")
            console.print(f"[red]Error: {e}[/red]\n")

    await db.close()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Event Researcher - Event-driven trading research platform"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with research agent")
    chat_parser.add_argument(
        "--resume", "-r", type=str, help="Resume a session by ID (partial match supported)"
    )

    # status command
    subparsers.add_parser("status", help="Show current status")

    # watchlist command
    subparsers.add_parser("watchlist", help="Display watchlist")

    # events command
    events_parser = subparsers.add_parser("events", help="Display upcoming events")
    events_parser.add_argument(
        "--days", type=int, default=7, help="Number of days to look ahead"
    )

    # prices command
    prices_parser = subparsers.add_parser("prices", help="Display prices for a symbol")
    prices_parser.add_argument("symbol", help="Stock symbol")
    prices_parser.add_argument(
        "--days", type=int, default=30, help="Number of days of history"
    )

    # earnings command
    earnings_parser = subparsers.add_parser("earnings", help="Display earnings history")
    earnings_parser.add_argument("symbol", help="Stock symbol")

    # monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Display surfaced events with scoring")
    monitor_parser.add_argument(
        "--days", type=int, default=14, help="Number of days to look ahead"
    )
    monitor_parser.add_argument(
        "--high-only", action="store_true", help="Show only high interest events"
    )

    # ui command
    subparsers.add_parser("ui", help="Launch the Textual TUI application")

    # sessions command
    subparsers.add_parser("sessions", help="List saved research sessions")

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)

    if args.command is None:
        # Default to status
        asyncio.run(cmd_status())
    elif args.command == "chat":
        asyncio.run(cmd_chat_with_session(getattr(args, "resume", None)))
    elif args.command == "sessions":
        asyncio.run(cmd_sessions())
    elif args.command == "status":
        asyncio.run(cmd_status())
    elif args.command == "watchlist":
        asyncio.run(cmd_watchlist())
    elif args.command == "events":
        asyncio.run(cmd_events(args.days))
    elif args.command == "prices":
        asyncio.run(cmd_prices(args.symbol.upper(), args.days))
    elif args.command == "earnings":
        asyncio.run(cmd_earnings(args.symbol.upper()))
    elif args.command == "monitor":
        asyncio.run(cmd_monitor(args.days, args.high_only))
    elif args.command == "ui":
        cmd_ui()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
