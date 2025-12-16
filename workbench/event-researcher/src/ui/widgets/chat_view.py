"""Chat view widget - research interface with the Claude agent."""

import asyncio
from typing import ClassVar

from rich.markdown import Markdown
from rich.text import Text
from textual import work
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, Label, Static

from src.agent.orchestrator import AgentOrchestrator
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ChatMessage(Static):
    """A single chat message."""

    DEFAULT_CSS = """
    ChatMessage {
        padding: 1;
        margin-bottom: 1;
    }

    ChatMessage.user {
        background: $primary-darken-2;
        border-left: thick $primary;
    }

    ChatMessage.assistant {
        background: $surface;
        border-left: thick $secondary;
    }

    ChatMessage.system {
        background: $warning-darken-3;
        border-left: thick $warning;
    }

    ChatMessage .message-role {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def __init__(self, role: str, content: str, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.add_class(role)

    def compose(self):
        role_colors = {
            "user": "cyan",
            "assistant": "green",
            "system": "yellow",
        }
        color = role_colors.get(self.role, "white")
        role_label = self.role.capitalize()

        yield Label(Text(f"{role_label}:", style=f"bold {color}"), classes="message-role")

        # Render content as markdown for assistant, plain text for others
        if self.role == "assistant":
            yield Static(Markdown(self.content))
        else:
            yield Static(self.content)


class ChatView(Widget):
    """Widget for chat-based research interface."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "blur_input", "Blur"),
    ]

    DEFAULT_CSS = """
    ChatView {
        layout: vertical;
        background: $surface;
    }

    ChatView #chat-title {
        text-style: bold;
        color: $text;
        padding: 1;
        dock: top;
    }

    ChatView #messages-scroll {
        height: 1fr;
        padding: 1;
    }

    ChatView #input-container {
        dock: bottom;
        height: auto;
        padding: 1;
        background: $surface-darken-1;
    }

    ChatView #chat-input {
        width: 100%;
    }

    ChatView #status-label {
        color: $text-muted;
        text-style: italic;
        padding: 0 1;
    }
    """

    class MessageSent(Message):
        """Message sent when user sends a chat message."""

        def __init__(self, content: str):
            super().__init__()
            self.content = content

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent: AgentOrchestrator | None = None
        self._is_processing = False

    def compose(self):
        yield Label("Research Chat", id="chat-title")
        yield VerticalScroll(id="messages-scroll")
        with Vertical(id="input-container"):
            yield Label("", id="status-label")
            yield Input(placeholder="Ask about events, earnings, price moves...", id="chat-input")

    def on_mount(self) -> None:
        """Add welcome message on mount."""
        self._add_message(
            "system",
            "Welcome to Event Researcher. Select an event from the monitor or ask a question to begin research.",
        )

    def set_agent(self, agent: AgentOrchestrator) -> None:
        """Set the agent orchestrator."""
        self.agent = agent

    def focus_input(self) -> None:
        """Focus the chat input."""
        input_widget = self.query_one("#chat-input", Input)
        input_widget.focus()

    def action_blur_input(self) -> None:
        """Blur the input field."""
        self.screen.focus_next()

    def _add_message(self, role: str, content: str) -> None:
        """Add a message to the chat."""
        scroll = self.query_one("#messages-scroll", VerticalScroll)
        message = ChatMessage(role=role, content=content)
        scroll.mount(message)
        # Scroll to bottom
        scroll.scroll_end(animate=False)

    def _set_status(self, status: str) -> None:
        """Set the status label."""
        label = self.query_one("#status-label", Label)
        label.update(status)

    def _clear_status(self) -> None:
        """Clear the status label."""
        self._set_status("")

    def clear_messages(self) -> None:
        """Clear all chat messages."""
        scroll = self.query_one("#messages-scroll", VerticalScroll)
        scroll.remove_children()
        self._add_message(
            "system",
            "Chat cleared. Ready for new research.",
        )

    async def send_message(self, message: str) -> None:
        """Send a message to the agent."""
        if not message.strip():
            return

        if self._is_processing:
            self.notify("Please wait for the current response", severity="warning")
            return

        self._add_message("user", message)
        self._process_message(message)

    @work(exclusive=True)
    async def _process_message(self, message: str) -> None:
        """Process a message with the agent (runs in background)."""
        if not self.agent:
            self._add_message("system", "Agent not initialized")
            return

        self._is_processing = True
        self._set_status("Thinking...")

        try:
            response = await self.agent.chat(message)
            self._add_message("assistant", response)
        except Exception as e:
            logger.error(f"Agent error: {e}")
            self._add_message("system", f"Error: {e}")
        finally:
            self._is_processing = False
            self._clear_status()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        message = event.value.strip()
        if message:
            event.input.clear()
            await self.send_message(message)
