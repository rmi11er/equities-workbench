# Kalshi Market Maker (KMM-v1)

A low-latency, asynchronous market-making bot for the Kalshi exchange supporting both Demo and Production environments.

## Overview

KMM-v1 implements inventory-aware market making using the Avellaneda-Stoikov model, adapted for binary prediction markets ($0–$100). The architecture is designed for high-throughput websocket data with sub-5ms internal processing latency.

**Core Strategy:** Avellaneda-Stoikov with inventory skew, featuring modular slots for external oracle ("Alpha") integration.

## Architecture

The system follows an event-driven architecture—it reacts to data rather than polling.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Ingest    │───▶│  Normalize  │───▶│   Decide    │───▶│   Execute   │
│ (Websocket) │    │ (OrderBook) │    │ (Strategy)  │    │  (Diffing)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
                                                         ┌─────────────┐
                                                         │   Record    │
                                                         │  (Logger)   │
                                                         └─────────────┘
```

### Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.10+ | Ecosystem, rapid iteration |
| Concurrency | `asyncio` | Single-threaded event loop avoids GIL context switching |
| REST | `aiohttp` | Async HTTP client |
| Streaming | `websockets` | Native async websocket support |
| Data Structures | Native dicts | O(1) order book access |
| Testing | `pytest`, `pytest-asyncio` | Async-aware test framework |

## Module Specifications

### Connector (`kalshi_connector.py`)

Manages raw I/O with Kalshi.

**Responsibilities:**
- RSA key-signing authentication with cached signature generator
- Websocket management with auto-reconnect (exponential backoff)
- Heartbeat monitoring—force reconnect if no message received within threshold
- Sequence checking—trigger `RESET_BOOK` on gap detection
- Client-side rate limiting via TokenBucket (prioritize `CANCEL` orders)

### OrderBook Manager (`orderbook.py`)

Maintains authoritative market state.

```python
class OrderBook:
    bids: dict[int, int]  # {price: volume}
    asks: dict[int, int]  # {price: volume}
    version: int
```

**Update Logic:**
- `delta.size == 0` → Delete price level
- Otherwise → Set `bids[price] = delta.size`

**Outputs:** `get_mid_price()`, `get_ofi()` (Order Flow Imbalance)

### Strategy Engine (`stoikov.py`)

The mathematical brain of the system.

**Inputs:**
- Mid Price (S)
- Inventory (q)
- Volatility (σ)
- Risk Aversion (γ)

**Model:**

*Reservation Price* — skews mid-price based on inventory:

```
r = S - q · γ · σ² · (T - t)
```

*Quote Generation* — bracket around reservation price:

```
Bid = floor(r - δ/2)
Ask = ceil(r + δ/2)
```

**Sanity Checks:** Quotes never cross (`Bid < Ask`), stay within bounds `[1, 99]`.

### Execution Engine (`execution.py`)

Minimizes API calls while maintaining target quotes.

**Diff Engine Example:**
```
Target: Bid 200 @ 21¢
Actual: Bid 200 @ 20¢
Action: CANCEL_REPLACE
```

**Debouncing:** Prevents quote flickering. Updates only trigger when `change > threshold` (X cents or Y seconds elapsed).

### Alpha Engine (`alpha_engine.py`)

Expansion slot for external signal ingestion.

| State | Behavior |
|-------|----------|
| Current | Returns `0.0` |
| Future | Polls external APIs (DraftKings, Polymarket) |

**Interface:** `get_external_skew(ticker) -> float` (±cents, added to reservation price)

## Observability

Observability follows a fire-and-forget principle.

### Logger (`logger.py`)

Uses `QueueListener` pattern—main thread pushes to queue, background thread writes to disk.

**Operations Log** (`ops.log`):
```
[INFO] Connection established.
[WARNING] High latency detected: 150ms.
```

**Data Tape** (`tape.csv`):

| Column | Description |
|--------|-------------|
| `ts` | Timestamp |
| `ticker` | Market identifier |
| `mid` | Mid price |
| `my_bid` | Current bid quote |
| `my_ask` | Current ask quote |
| `inventory` | Position |
| `unrealized_pnl` | Mark-to-market P&L |
| `realized_pnl` | Closed P&L |
| `latency_ms` | Processing latency |

Purpose: Replay for equity curve generation and post-trade analysis.

### Error Handling

| Error | Trigger | Action |
|-------|---------|--------|
| API 429 | Rate limit hit | Sleep & backoff: pause new orders 2s, allow cancels |
| WS Disconnect | Network/server | Panic: assume stuck orders → reconnect → `CANCEL_ALL` → restart |
| Stale Data | No ticks 5s | Pull: cancel all quotes until data resumes |
| Fat Finger | Size > max | Reject: pre-flight check before API call |

## Testing Strategy

### Unit Tests (`/tests/unit`)

- **Math Validation:** Stoikov formula produces correct integer cents from float inputs
- **Inventory Logic:** Buying Yes/No correctly nets out or updates inventory counter

### System Mock (`/tests/mock`)

- **MockKalshi:** Replicates `place_order` and `ws_connect` methods
- **Event Loop Tests:** Run bot against mock, feed synthetic scenarios (e.g., price crash), assert correct order placement

### Production Probe (`probe.py`)

Connectivity validation sequence:
1. Connect to Demo
2. Place 1 contract Buy @ 1¢
3. Confirm Order ID received
4. Cancel Order
5. Confirm Cancellation

## Implementation Roadmap

| Phase | Milestone | Deliverable |
|-------|-----------|-------------|
| 1 | The Listener | WS connection, local OrderBook, MidPrice logging |
| 2 | The Dumb Maker | Static orders (Mid ± 2¢) on Demo with API signing |
| 3 | The Stoikov | Inventory module integration, price skewing on fills |
| 4 | The Tape | Full CSV logging and P&L tracking |
| 5 | The Oracle | DraftKings API connection to AlphaEngine |

## Getting Started

### Prerequisites

- Python 3.10+
- RSA keys for Kalshi Demo environment

### Installation

```bash
git clone <repository-url>
cd kalshi-mm-v1
pip install -r requirements.txt
```

### Dependencies

```
aiohttp
websockets
cryptography
pandas
pytest
pytest-asyncio
```

### Configuration

1. Generate RSA keys for Demo environment
2. Configure credentials (see `config.example.yaml`)
3. Set environment variables or update config file

### Running

```bash
# Run connectivity probe
python probe.py

# Start market maker (Demo)
python main.py --env demo

# Run tests
pytest tests/
```

## Project Structure

```
kalshi-mm-v1/
├── kalshi_connector.py    # Exchange I/O management
├── orderbook.py           # Market state maintenance
├── stoikov.py             # Avellaneda-Stoikov strategy
├── execution.py           # Order diffing and execution
├── alpha_engine.py        # External signal integration
├── logger.py              # Async logging infrastructure
├── probe.py               # Connectivity testing
├── main.py                # Entry point
├── requirements.txt
├── config.example.yaml
└── tests/
    ├── unit/
    └── mock/
```

## License

[Specify license]

## Contributing

[Contribution guidelines]
