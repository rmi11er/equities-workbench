"""Tool registry and dispatch system for agent tools."""

from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolDefinition:
    """Definition of an agent tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters
    handler: Callable[..., Coroutine[Any, Any, dict[str, Any]]]
    required_params: list[str] = field(default_factory=list)

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_params,
            },
        }


class ToolRegistry:
    """Registry for agent tools."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        required: list[str] | None = None,
    ) -> Callable:
        """Decorator to register a tool handler."""

        def decorator(
            func: Callable[..., Coroutine[Any, Any, dict[str, Any]]]
        ) -> Callable[..., Coroutine[Any, Any, dict[str, Any]]]:
            tool = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                handler=func,
                required_params=required or [],
            )
            self._tools[name] = tool
            logger.debug(f"Registered tool: {name}")
            return func

        return decorator

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_anthropic_tools(self) -> list[dict[str, Any]]:
        """Get all tools in Anthropic format."""
        return [tool.to_anthropic_format() for tool in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool by name with given arguments."""
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}

        try:
            logger.debug(f"Executing tool {name} with args: {arguments}")
            result = await tool.handler(**arguments)
            logger.debug(f"Tool {name} returned: {type(result)}")
            return result
        except TypeError as e:
            logger.error(f"Tool {name} argument error: {e}")
            return {"error": f"Invalid arguments for {name}: {str(e)}"}
        except Exception as e:
            logger.error(f"Tool {name} execution error: {e}")
            return {"error": f"Tool execution failed: {str(e)}"}


# Global registry instance
registry = ToolRegistry()
