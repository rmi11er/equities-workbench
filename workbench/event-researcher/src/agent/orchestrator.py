"""Agent orchestrator - main agent loop using Claude API with tools."""

import json
from typing import Any, AsyncGenerator
from uuid import uuid4

import anthropic

from src.agent.prompts import get_system_prompt
from src.agent.tools.registry import registry
from src.agent.tools import scenarios as scenarios_module
from src.data.database import Database
from src.data.models import ConversationMessage
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AgentOrchestrator:
    """Orchestrates agent interactions with Claude API and tools."""

    def __init__(
        self,
        db: Database,
        session_id: str | None = None,
        session_context: str = "",
    ):
        self.db = db
        self.session_id = session_id or str(uuid4())
        self.session_context = session_context

        settings = get_settings()
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.agent.default_model
        self.max_tokens = settings.agent.max_tokens
        self.temperature = settings.agent.temperature

        self._conversation_history: list[dict[str, Any]] = []

    @property
    def system_prompt(self) -> str:
        """Get the current system prompt."""
        return get_system_prompt(self.session_context)

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Get available tools in Anthropic format."""
        return registry.get_anthropic_tools()

    async def load_conversation_history(self) -> None:
        """Load conversation history from database."""
        messages = await self.db.get_conversation(self.session_id)
        self._conversation_history = []

        for msg in messages:
            if msg.role == "user":
                self._conversation_history.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                # Reconstruct assistant message with tool calls if present
                if msg.tool_calls:
                    self._conversation_history.append(
                        {"role": "assistant", "content": msg.tool_calls}
                    )
                else:
                    self._conversation_history.append(
                        {"role": "assistant", "content": msg.content}
                    )

    async def _save_message(self, role: str, content: str, tool_calls: Any = None) -> None:
        """Save a message to the database."""
        message = ConversationMessage(
            message_id=str(uuid4()),
            session_id=self.session_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
        )
        await self.db.add_message(message)

    async def _call_api(
        self, messages: list[dict[str, Any]]
    ) -> anthropic.types.Message:
        """Make a single API call with error handling."""
        try:
            return await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                tools=self.tools,
                messages=messages,
            )
        except anthropic.RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}")
            raise RuntimeError(
                "API rate limit reached. Please wait a moment and try again."
            ) from e
        except anthropic.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            raise RuntimeError(
                "API authentication failed. Please check your ANTHROPIC_API_KEY."
            ) from e
        except anthropic.APIConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise RuntimeError(
                "Could not connect to the API. Please check your internet connection."
            ) from e
        except anthropic.APIStatusError as e:
            logger.error(f"API error: {e}")
            raise RuntimeError(f"API error: {e.message}") from e

    async def _process_tool_calls(
        self, tool_use_blocks: list[Any]
    ) -> list[dict[str, Any]]:
        """Process tool calls and return results."""
        results = []
        for block in tool_use_blocks:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                tool_id = block.id

                logger.info(f"Calling tool: {tool_name}")

                try:
                    result = await registry.execute(tool_name, tool_input)
                except Exception as e:
                    logger.error(f"Tool {tool_name} failed: {e}")
                    result = {
                        "error": str(e),
                        "tool": tool_name,
                        "message": "Tool execution failed. The error has been logged.",
                    }

                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": json.dumps(result),
                    }
                )
        return results

    async def chat(self, user_message: str) -> str:
        """Process a user message and return the agent's response.

        This handles the full agentic loop including tool calls.
        """
        # Add user message to history
        self._conversation_history.append({"role": "user", "content": user_message})
        await self._save_message("user", user_message)

        # Agentic loop - keep calling until we get a final response
        while True:
            response = await self._call_api(self._conversation_history)

            # Check if we have tool calls
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if tool_use_blocks:
                # Add assistant message with tool calls
                self._conversation_history.append(
                    {"role": "assistant", "content": response.content}
                )

                # Process tool calls
                tool_results = await self._process_tool_calls(tool_use_blocks)

                # Add tool results to history
                self._conversation_history.append(
                    {"role": "user", "content": tool_results}
                )

                # Continue loop to get next response
                continue

            # No tool calls - we have a final response
            final_text = "\n".join(b.text for b in text_blocks)

            # Add to history and save
            self._conversation_history.append(
                {"role": "assistant", "content": final_text}
            )
            await self._save_message("assistant", final_text)

            return final_text

    async def chat_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        """Process a user message and stream the response.

        Yields text chunks as they arrive. Tool calls are processed
        internally and their results are not streamed.
        """
        # Add user message to history
        self._conversation_history.append({"role": "user", "content": user_message})
        await self._save_message("user", user_message)

        accumulated_text = ""

        while True:
            # Use streaming for the API call
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                tools=self.tools,
                messages=self._conversation_history,
            ) as stream:
                response = await stream.get_final_message()

            # Check for tool calls
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if tool_use_blocks:
                # Add assistant message with tool calls
                self._conversation_history.append(
                    {"role": "assistant", "content": response.content}
                )

                # Yield indicator that tools are being called
                tool_names = [b.name for b in tool_use_blocks]
                yield f"\n[Calling: {', '.join(tool_names)}]\n"

                # Process tool calls
                tool_results = await self._process_tool_calls(tool_use_blocks)

                # Add tool results
                self._conversation_history.append(
                    {"role": "user", "content": tool_results}
                )
                continue

            # Final text response
            final_text = "\n".join(b.text for b in text_blocks)
            accumulated_text = final_text

            # For simplicity, yield the full response
            # (true streaming would require different handling)
            yield final_text

            # Save and exit
            self._conversation_history.append(
                {"role": "assistant", "content": final_text}
            )
            await self._save_message("assistant", final_text)
            break

    def clear_history(self) -> None:
        """Clear conversation history (in memory only)."""
        self._conversation_history = []

    def reset_conversation(self) -> None:
        """Reset conversation and start fresh session."""
        self._conversation_history = []
        self.session_id = str(uuid4())

    async def save_session(self, title: str | None = None) -> str:
        """Save the current session to the database.

        Returns the session ID.
        """
        from src.data.models import ResearchSession

        # Generate title from first message if not provided
        if not title and self._conversation_history:
            first_user_msg = next(
                (m for m in self._conversation_history if m.get("role") == "user"),
                None,
            )
            if first_user_msg:
                content = first_user_msg.get("content", "")
                if isinstance(content, str):
                    title = content[:50] + "..." if len(content) > 50 else content

        # Get current scenarios from scenario module
        current_scenarios = scenarios_module.get_scenarios()

        session = ResearchSession(
            session_id=self.session_id,
            title=title or "Untitled Session",
            status="active",
            context_summary=self.session_context,
            scenarios=current_scenarios,
        )

        # Check if session exists
        existing = await self.db.get_session(self.session_id)
        if existing:
            session.target_symbol = existing.target_symbol
            session.target_event_id = existing.target_event_id
            await self.db.update_session(session)
        else:
            await self.db.create_session(session)

        logger.info(f"Saved session: {self.session_id}")
        return self.session_id

    async def resume_session(self, session_id: str) -> bool:
        """Resume an existing session.

        Returns True if session was found and loaded.
        """
        session = await self.db.get_session(session_id)
        if not session:
            return False

        self.session_id = session_id
        self.session_context = session.context_summary or ""

        # Load conversation history
        await self.load_conversation_history()

        # Load scenarios into the scenario module
        scenarios_module.set_session(session_id, session.scenarios or {})

        logger.info(f"Resumed session: {session_id} with {len(self._conversation_history)} messages")
        return True

    def get_message_count(self) -> int:
        """Get the number of messages in current conversation."""
        return len(self._conversation_history)
