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
    """Sample orderbook snapshot message.

    Kalshi format: yes/no arrays contain BID orders (resting buy orders).
    - yes: [[price, size], ...] = bids to buy YES contracts
    - no: [[price, size], ...] = bids to buy NO contracts

    Derived prices:
    - YES asks come from NO bids: if NO bid at 55, YES ask at 45
    - NO asks come from YES bids: if YES bid at 40, NO ask at 60

    This fixture creates:
    - YES bids: 35, 38, 40 → best YES bid = 40
    - NO bids: 55, 57, 59 → best YES ask = 100-59 = 41
    - Spread = 1, Mid = 40.5
    """
    return {
        "market_ticker": "TEST-TICKER",
        "yes": [[35, 100], [38, 200], [40, 150]],  # YES bids
        "no": [[55, 100], [57, 200], [59, 150]],   # NO bids → YES asks at 45, 43, 41
    }


@pytest.fixture
def sample_orderbook_delta():
    """Sample orderbook delta message.

    Deltas update BID orders. This delta reduces the YES bid at price 38
    from 200 to 150 contracts.
    """
    return {
        "market_ticker": "TEST-TICKER",
        "price": 38,
        "delta": -50,
        "side": "yes",
    }
