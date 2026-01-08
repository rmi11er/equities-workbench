"""Constants and enums for the market maker."""

from enum import Enum, auto


class Environment(Enum):
    DEMO = auto()
    PRODUCTION = auto()


class Side(Enum):
    YES = "yes"
    NO = "no"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    RESTING = "resting"
    PENDING = "pending"
    CANCELED = "canceled"
    EXECUTED = "executed"


# API Endpoints
WS_URLS = {
    Environment.DEMO: "wss://demo-api.kalshi.co/trade-api/ws/v2",
    Environment.PRODUCTION: "wss://api.elections.kalshi.com/trade-api/ws/v2",
}

REST_URLS = {
    Environment.DEMO: "https://demo-api.kalshi.co/trade-api/v2",
    Environment.PRODUCTION: "https://api.elections.kalshi.com/trade-api/v2",
}

# Price bounds for binary markets (cents)
MIN_PRICE = 1
MAX_PRICE = 99

# Rate limiting (Basic tier defaults)
DEFAULT_READ_RATE = 20  # req/sec
DEFAULT_WRITE_RATE = 10  # req/sec
CANCEL_COST = 0.2  # cancels count as 0.2 transactions

# WebSocket
WS_HEARTBEAT_INTERVAL = 10  # seconds
WS_PATH = "/trade-api/ws/v2"

# Reconnection
MAX_RECONNECT_ATTEMPTS = 10
INITIAL_BACKOFF_SEC = 1.0
MAX_BACKOFF_SEC = 60.0

# Stale data threshold
STALE_DATA_THRESHOLD_SEC = 5.0
