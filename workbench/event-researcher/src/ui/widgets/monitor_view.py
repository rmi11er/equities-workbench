"""Monitor view widget - displays upcoming events with interest scoring."""

from datetime import date
from typing import ClassVar

from rich.text import Text
from textual import on
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import DataTable, Label, Static

from src.monitor.surfacer import EventSurfacer, InterestTier, SurfacedEvent
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EventRow(Static):
    """A single event row in the monitor."""

    def __init__(self, event: SurfacedEvent, **kwargs):
        super().__init__(**kwargs)
        self.event = event

    def compose(self):
        tier_colors = {
            InterestTier.HIGH: "red",
            InterestTier.STANDARD: "yellow",
            InterestTier.LOW: "dim",
        }
        color = tier_colors.get(self.event.tier, "white")

        # Format the event row
        days = self.event.days_until
        if days == 0:
            days_str = "TODAY"
        elif days == 1:
            days_str = "1 day"
        else:
            days_str = f"{days} days"

        text = Text()
        text.append(f"{self.event.symbol:6} ", style=f"bold {color}")
        text.append(f"{self.event.event_type:10} ", style=color)
        text.append(f"{days_str:8} ", style="cyan")
        text.append(f"[{self.event.interest_score}] ", style="magenta")

        # Add flags summary
        if self.event.flags:
            flags_str = ", ".join(f.name for f in self.event.flags[:2])
            text.append(flags_str, style="dim")

        yield Label(text)


class MonitorView(Widget):
    """Widget displaying upcoming events grouped by interest tier."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("j", "cursor_down", "Down"),
        Binding("k", "cursor_up", "Up"),
        Binding("enter", "select_event", "Select"),
        Binding("h", "toggle_high_only", "High Only"),
    ]

    DEFAULT_CSS = """
    MonitorView {
        background: $surface;
        padding: 1;
    }

    MonitorView #monitor-title {
        text-style: bold;
        color: $text;
        padding-bottom: 1;
    }

    MonitorView #events-table {
        height: 1fr;
    }

    MonitorView DataTable {
        height: 100%;
    }

    MonitorView DataTable > .datatable--cursor {
        background: $accent;
    }
    """

    class EventSelected(Message):
        """Message sent when an event is selected."""

        def __init__(
            self,
            symbol: str,
            event_type: str,
            event_date: date,
            event_id: str,
        ):
            super().__init__()
            self.symbol = symbol
            self.event_type = event_type
            self.event_date = event_date
            self.event_id = event_id

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.surfacer: EventSurfacer | None = None
        self.events: list[SurfacedEvent] = []
        self._show_high_only = False

    def compose(self):
        yield Label("Upcoming Events", id="monitor-title")
        with VerticalScroll(id="events-table"):
            yield DataTable(id="events-data")

    def on_mount(self) -> None:
        """Set up the data table on mount."""
        table = self.query_one("#events-data", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("Days", "Symbol", "Type", "Score", "Flags")

    def set_surfacer(self, surfacer: EventSurfacer) -> None:
        """Set the event surfacer."""
        self.surfacer = surfacer

    async def refresh_events(self) -> None:
        """Refresh the event list from the database."""
        if not self.surfacer:
            return

        try:
            min_tier = InterestTier.HIGH if self._show_high_only else InterestTier.LOW
            self.events = await self.surfacer.get_surfaced_events(min_tier=min_tier)
            self._update_table()
            logger.info(f"Loaded {len(self.events)} events")
        except Exception as e:
            logger.error(f"Failed to refresh events: {e}")
            self.notify(f"Failed to load events: {e}", severity="error")

    def _update_table(self) -> None:
        """Update the data table with current events."""
        table = self.query_one("#events-data", DataTable)
        table.clear()

        tier_styles = {
            InterestTier.HIGH: "bold red",
            InterestTier.STANDARD: "yellow",
            InterestTier.LOW: "dim",
        }

        for event in self.events:
            style = tier_styles.get(event.tier, "")

            # Format days until
            if event.days_until == 0:
                days_str = "TODAY"
            elif event.days_until == 1:
                days_str = "1d"
            else:
                days_str = f"{event.days_until}d"

            # Format flags
            flags_str = ", ".join(f.name for f in event.flags[:2]) if event.flags else ""

            table.add_row(
                Text(days_str, style="cyan" if event.days_until <= 1 else ""),
                Text(event.symbol, style=style),
                Text(event.event_type, style=style),
                Text(str(event.interest_score), style="magenta"),
                Text(flags_str, style="dim"),
                key=event.event_id,
            )

    def action_cursor_down(self) -> None:
        """Move cursor down."""
        table = self.query_one("#events-data", DataTable)
        table.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move cursor up."""
        table = self.query_one("#events-data", DataTable)
        table.action_cursor_up()

    def action_select_event(self) -> None:
        """Select the current event."""
        table = self.query_one("#events-data", DataTable)
        if table.cursor_row is not None and self.events:
            try:
                row_key = table.get_row_at(table.cursor_row)
                # Find matching event
                for event in self.events:
                    if table.get_row_at(table.cursor_row):
                        idx = table.cursor_row
                        if idx < len(self.events):
                            selected = self.events[idx]
                            self.post_message(
                                self.EventSelected(
                                    symbol=selected.symbol,
                                    event_type=selected.event_type,
                                    event_date=selected.event_date,
                                    event_id=selected.event_id,
                                )
                            )
                        break
            except Exception as e:
                logger.error(f"Failed to select event: {e}")

    async def action_toggle_high_only(self) -> None:
        """Toggle showing only high-interest events."""
        self._show_high_only = not self._show_high_only
        status = "high interest only" if self._show_high_only else "all events"
        self.notify(f"Showing {status}")
        await self.refresh_events()

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in the data table."""
        if event.row_key and self.events:
            # Find event by ID
            for ev in self.events:
                if ev.event_id == str(event.row_key.value):
                    self.post_message(
                        self.EventSelected(
                            symbol=ev.symbol,
                            event_type=ev.event_type,
                            event_date=ev.event_date,
                            event_id=ev.event_id,
                        )
                    )
                    break
