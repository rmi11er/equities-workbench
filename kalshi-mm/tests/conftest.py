"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    from src.config import Config, CredentialsConfig, StrategyConfig, VolatilityConfig

    return Config(
        market_ticker="TEST-TICKER",
        credentials=CredentialsConfig(
            api_key_id="test-key",
            private_key_path="/dev/null",
        ),
        strategy=StrategyConfig(
            risk_aversion=0.05,
            time_horizon=1.0,
            max_inventory=500,
            max_order_size=100,
            base_spread=2.0,
            quote_size=10,
            debounce_cents=2,
            debounce_seconds=5.0,
        ),
        volatility=VolatilityConfig(
            ema_halflife_sec=60.0,
            min_volatility=0.1,
            initial_volatility=5.0,
        ),
    )


@pytest.fixture
def sample_orderbook_snapshot():
    """Sample orderbook snapshot message."""
    return {
        "market_ticker": "TEST-TICKER",
        "yes": [[30, 100], [35, 200], [40, 150]],  # Asks for YES
        "no": [[55, 100], [60, 200], [65, 150]],   # Asks for NO
    }


@pytest.fixture
def sample_orderbook_delta():
    """Sample orderbook delta message."""
    return {
        "market_ticker": "TEST-TICKER",
        "price": 35,
        "delta": -50,
        "side": "yes",
    }
