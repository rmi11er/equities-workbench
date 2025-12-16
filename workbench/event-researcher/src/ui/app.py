"""Main Textual application for Event Researcher."""

import asyncio
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.widgets import Footer, Header

from src.agent.orchestrator import AgentOrchestrator
from src.data.database import Database
from src.monitor.surfacer import EventSurfacer
from src.ui.widgets.chat_view import ChatView
from src.ui.widgets.monitor_view import MonitorView
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EventResearcherApp(App):
    """Event Researcher TUI application."""

    TITLE = "Event Researcher"
    CSS = """
    Screen {
        layout: horizontal;
    }

    #main-container {
        width: 100%;
        height: 100%;
    }

    #monitor-pane {
        width: 40%;
        height: 100%;
        border-right: solid $primary;
    }

    #chat-pane {
        width: 60%;
        height: 100%;
    }

    MonitorView {
        height: 100%;
    }

    ChatView {
        height: 100%;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit"),
        Binding("tab", "switch_pane", "Switch Pane"),
        Binding("r", "refresh_events", "Refresh"),
        Binding("?", "show_help", "Help"),
        Binding("ctrl+l", "clear_chat", "Clear Chat"),
    ]

    def __init__(self):
        super().__init__()
        self.db: Database | None = None
        self.agent: AgentOrchestrator | None = None
        self.surfacer: EventSurfacer | None = None
        self._active_pane = "chat"

    async def _init_services(self) -> None:
        """Initialize database and agent services."""
        settings = get_settings()
        self.db = Database(settings.database_path)
        await self.db.initialize()
        self.agent = AgentOrchestrator(db=self.db)
        self.surfacer = EventSurfacer(db=self.db)
        logger.info("Services initialized")

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()
        with Horizontal(id="main-container"):
            with Container(id="monitor-pane"):
                yield MonitorView(id="monitor-view")
            with Container(id="chat-pane"):
                yield ChatView(id="chat-view")
        yield Footer()

    async def on_mount(self) -> None:
        """Handle mount event - initialize services and load data."""
        await self._init_services()

        # Load initial events
        monitor_view = self.query_one("#monitor-view", MonitorView)
        monitor_view.set_surfacer(self.surfacer)
        await monitor_view.refresh_events()

        # Set up chat view with agent
        chat_view = self.query_one("#chat-view", ChatView)
        chat_view.set_agent(self.agent)

        # Focus on chat input by default
        chat_view.focus_input()

    async def on_unmount(self) -> None:
        """Clean up on unmount."""
        if self.db:
            await self.db.close()

    def action_switch_pane(self) -> None:
        """Switch focus between monitor and chat panes."""
        if self._active_pane == "chat":
            self._active_pane = "monitor"
            monitor_view = self.query_one("#monitor-view", MonitorView)
            monitor_view.focus()
        else:
            self._active_pane = "chat"
            chat_view = self.query_one("#chat-view", ChatView)
            chat_view.focus_input()

    async def action_refresh_events(self) -> None:
        """Refresh the event list."""
        monitor_view = self.query_one("#monitor-view", MonitorView)
        await monitor_view.refresh_events()

    def action_clear_chat(self) -> None:
        """Clear chat history."""
        chat_view = self.query_one("#chat-view", ChatView)
        chat_view.clear_messages()
        if self.agent:
            self.agent.reset_conversation()

    def action_show_help(self) -> None:
        """Show help dialog."""
        self.notify(
            "Tab: Switch pane | Enter: Send | R: Refresh | Q: Quit",
            title="Keyboard Shortcuts",
            timeout=5,
        )

    async def on_monitor_view_event_selected(
        self, event: "MonitorView.EventSelected"
    ) -> None:
        """Handle event selection from monitor view."""
        chat_view = self.query_one("#chat-view", ChatView)

        # Add context message about selected event
        context_msg = (
            f"I'm interested in the {event.event_type} event for {event.symbol} "
            f"on {event.event_date}. Can you provide analysis?"
        )

        # Send to chat
        await chat_view.send_message(context_msg)

        # Switch to chat pane
        self._active_pane = "chat"
        chat_view.focus_input()


def run_app() -> None:
    """Run the TUI application."""
    app = EventResearcherApp()
    app.run()


if __name__ == "__main__":
    run_app()
