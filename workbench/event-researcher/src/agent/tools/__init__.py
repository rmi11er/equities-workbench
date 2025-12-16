"""Agent tools for data access and computation.

Import this module to register all tools with the registry.
"""

from src.agent.tools.registry import registry

# Import tool modules to register their tools
from src.agent.tools import local_data
from src.agent.tools import compute
from src.agent.tools import external
from src.agent.tools import scenarios

__all__ = ["registry"]
