"""Scenario management tools for saving and retrieving analysis scenarios."""

from typing import Any

from src.agent.tools.registry import registry
from src.utils.logging import get_logger

logger = get_logger(__name__)

# In-memory scenario storage (will be persisted via session)
_current_scenarios: dict[str, dict[str, Any]] = {}
_session_id: str | None = None


def set_session(session_id: str, scenarios: dict[str, Any]) -> None:
    """Set the current session and load its scenarios."""
    global _session_id, _current_scenarios
    _session_id = session_id
    _current_scenarios = scenarios.copy()


def get_scenarios() -> dict[str, Any]:
    """Get all scenarios for the current session."""
    return _current_scenarios.copy()


def clear_scenarios() -> None:
    """Clear all scenarios."""
    global _current_scenarios
    _current_scenarios = {}


@registry.register(
    name="save_scenario",
    description="Save an analysis scenario for future reference. Use this to record potential outcomes, trade ideas, or analytical conclusions that the user may want to review later.",
    parameters={
        "name": {
            "type": "string",
            "description": "Short name for the scenario (e.g., 'nvda_beat_scenario', 'aapl_guidance_bull')",
        },
        "description": {
            "type": "string",
            "description": "Detailed description of the scenario including assumptions and expected outcomes",
        },
        "symbol": {
            "type": "string",
            "description": "The stock symbol this scenario relates to",
        },
        "scenario_type": {
            "type": "string",
            "description": "Type of scenario: 'bull', 'bear', 'base', or 'conditional'",
            "enum": ["bull", "bear", "base", "conditional"],
        },
        "expected_move": {
            "type": "string",
            "description": "Expected price movement (e.g., '+5% to +8%', '-3%')",
        },
        "probability": {
            "type": "number",
            "description": "Estimated probability (0-100) if applicable",
        },
        "notes": {
            "type": "string",
            "description": "Additional notes or context",
        },
    },
    required=["name", "description", "symbol", "scenario_type"],
)
async def save_scenario(
    name: str,
    description: str,
    symbol: str,
    scenario_type: str,
    expected_move: str | None = None,
    probability: float | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Save a scenario to the current session."""
    global _current_scenarios

    scenario = {
        "name": name,
        "description": description,
        "symbol": symbol.upper(),
        "type": scenario_type,
        "expected_move": expected_move,
        "probability": probability,
        "notes": notes,
    }

    _current_scenarios[name] = scenario

    logger.info(f"Saved scenario: {name} for {symbol}")

    return {
        "status": "saved",
        "scenario_name": name,
        "total_scenarios": len(_current_scenarios),
    }


@registry.register(
    name="list_scenarios",
    description="List all saved scenarios in the current session. Use this to review previously saved analysis scenarios.",
    parameters={
        "symbol": {
            "type": "string",
            "description": "Filter by symbol (optional)",
        },
        "scenario_type": {
            "type": "string",
            "description": "Filter by scenario type (optional)",
            "enum": ["bull", "bear", "base", "conditional"],
        },
    },
    required=[],
)
async def list_scenarios(
    symbol: str | None = None,
    scenario_type: str | None = None,
) -> dict[str, Any]:
    """List all scenarios in the current session."""
    scenarios = list(_current_scenarios.values())

    if symbol:
        scenarios = [s for s in scenarios if s["symbol"] == symbol.upper()]

    if scenario_type:
        scenarios = [s for s in scenarios if s["type"] == scenario_type]

    return {
        "count": len(scenarios),
        "scenarios": scenarios,
    }


@registry.register(
    name="get_scenario",
    description="Get details of a specific saved scenario by name.",
    parameters={
        "name": {
            "type": "string",
            "description": "Name of the scenario to retrieve",
        },
    },
    required=["name"],
)
async def get_scenario(name: str) -> dict[str, Any]:
    """Get a specific scenario by name."""
    if name not in _current_scenarios:
        return {
            "error": f"Scenario '{name}' not found",
            "available": list(_current_scenarios.keys()),
        }

    return {
        "scenario": _current_scenarios[name],
    }


@registry.register(
    name="delete_scenario",
    description="Delete a saved scenario by name.",
    parameters={
        "name": {
            "type": "string",
            "description": "Name of the scenario to delete",
        },
    },
    required=["name"],
)
async def delete_scenario(name: str) -> dict[str, Any]:
    """Delete a scenario by name."""
    if name not in _current_scenarios:
        return {
            "error": f"Scenario '{name}' not found",
        }

    del _current_scenarios[name]

    return {
        "status": "deleted",
        "scenario_name": name,
        "remaining_scenarios": len(_current_scenarios),
    }
