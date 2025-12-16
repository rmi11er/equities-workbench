# Event-Driven Research Platform — Project Specification

## Quick Start for Implementation

**What this is**: An agent-augmented research environment for discovering and analyzing tradeable events (earnings, conferences, macro) in equity and crypto markets.

**Core interaction model**: Chat-first interface where user queries a Claude-powered agent that has tools to access local data (prices, events, earnings) and fetch external data (transcripts, news). The agent orchestrates analysis and presents findings conversationally.

**Tech stack**: Python 3.11+, DuckDB, Claude API, Textual (TUI), Polars, plotext

**Key principle**: Flexibility over structure. The agent adapts to the user's research process.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [User Workflow](#2-user-workflow)
3. [Architecture](#3-architecture)
4. [Database Schema](#4-database-schema)
5. [Agent Tools](#5-agent-tools)
6. [Event Monitoring](#6-event-monitoring)
7. [Data Acquisition](#7-data-acquisition)
8. [Project Structure](#8-project-structure)
9. [Configuration](#9-configuration)
10. [Implementation Phases](#10-implementation-phases)
11. [Example Interactions](#11-example-interactions)

---

## 1. Project Overview

### 1.1 Purpose

Enable systematic discovery and analysis of tradeable events by:
- Surfacing upcoming catalysts (earnings, conferences, macro releases) with relevant context
- Providing agent-assisted research to explore historical analogs, price patterns, and fundamental context
- Supporting scenario-based expected value analysis

### 1.2 Non-Goals

- Automated trade execution
- Options pricing / Greeks calculation
- Real-time streaming data
- Portfolio management / P&L tracking
- Position sizing (user does this externally)

### 1.3 Design Principles

1. **Agent as interface**: Natural language is primary interaction mode
2. **Local-first data**: Price/event data stored locally in DuckDB; external APIs supplement on-demand
3. **Flexibility**: System adapts to user's research process, not vice versa
4. **Decision support**: Provides data and analysis; user forms views and trades externally

---

## 2. User Workflow

### 2.1 Primary Loop

```
SURFACE → TRIAGE → EXPLORE → HYPOTHESIZE → VALIDATE → DECIDE

1. SURFACE
   Event monitor shows upcoming catalysts matching user filters
   
2. TRIAGE  
   User scans events, selects one to research
   Opens new research context for that event
   
3. EXPLORE (iterative, agent-assisted)
   User asks questions:
   - "How has this name moved around past earnings?"
   - "Show me analogs where they reported during a sector downturn"
   - "What did competitors guide this quarter?"
   
4. HYPOTHESIZE
   User forms scenario distribution (manually):
   - Beat + raise: +8%, 40% probability
   - Inline: -2%, 35% probability  
   - Miss: -15%, 25% probability
   
5. VALIDATE
   User stress-tests with agent:
   - "What's my EV given these scenarios?"
   - "How often has implied move been this high historically?"
   
6. DECIDE (external to platform)
   User compares EV to market pricing, sizes trade externally
```

### 2.2 Research Context Model

- Each research session is isolated by default
- Context includes: target event, accumulated data/analyses, conversation history
- Sessions can be saved and resumed
- User can explicitly share context across sessions if needed

---

## 3. Architecture

### 3.1 High-Level Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│                   (Textual TUI App)                         │
│  ┌────────────────────┬────────────────────────────────┐   │
│  │   Monitor View     │      Research Chat             │   │
│  │   - Upcoming       │      - NL queries              │   │
│  │     events         │      - Agent responses         │   │
│  │   - Filtered       │      - Inline charts           │   │
│  │   - Actionable     │      - Session context         │   │
│  └────────────────────┴────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   AGENT ORCHESTRATOR                        │
│                  (Claude API + Tools)                       │
│                                                             │
│  - Parse user intent                                        │
│  - Select and invoke tools                                  │
│  - Synthesize results                                       │
│  - Maintain conversation context                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      TOOL LAYER                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Local Data  │  │ External    │  │ Compute             │ │
│  │ Tools       │  │ Fetch Tools │  │ Tools               │ │
│  │             │  │             │  │                     │ │
│  │ query_      │  │ fetch_      │  │ compute_event_      │ │
│  │   prices    │  │   transcript│  │   response          │ │
│  │ query_      │  │ fetch_news  │  │ find_analogs        │ │
│  │   events    │  │ web_search  │  │ compute_stats       │ │
│  │ query_      │  │             │  │ calculate_ev        │ │
│  │   earnings  │  │             │  │ generate_chart      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER (DuckDB)                      │
│                                                             │
│  prices │ events │ earnings │ watchlist │ sessions │ cache │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Technology Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Language | Python 3.11+ | |
| Database | DuckDB | Local, analytical, fast |
| Agent LLM | Claude API | Sonnet for most queries, Opus for complex |
| HTTP | httpx + asyncio | Async data fetching |
| Data processing | Polars | Fast DataFrames, DuckDB-compatible |
| TUI | Textual | Rich terminal interfaces |
| Charts | plotext | Terminal-native plotting |
| Scheduling | APScheduler | Data refresh jobs |
| Config | TOML | Human-readable |

---

## 4. Database Schema

### 4.1 Core Tables

```sql
-- Price data (daily + hourly)
CREATE TABLE prices (
    symbol VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    timeframe VARCHAR NOT NULL,  -- 'daily' or 'hourly'
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume BIGINT,
    source VARCHAR,
    PRIMARY KEY (symbol, timestamp, timeframe)
);

-- Event calendar
CREATE TABLE events (
    event_id VARCHAR PRIMARY KEY,  -- UUID
    symbol VARCHAR,
    event_type VARCHAR NOT NULL,   -- 'earnings', 'conference', 'analyst_day', 'macro'
    event_date DATE NOT NULL,
    event_time VARCHAR,            -- 'BMO', 'AMC', 'DMH', or specific time
    title VARCHAR,
    description TEXT,
    metadata JSON,                 -- Flexible event-specific data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Earnings-specific data
CREATE TABLE earnings (
    event_id VARCHAR PRIMARY KEY REFERENCES events(event_id),
    symbol VARCHAR NOT NULL,
    fiscal_quarter VARCHAR,        -- 'Q1 2024', etc.
    fiscal_year INTEGER,
    eps_estimate DECIMAL(10,4),
    eps_actual DECIMAL(10,4),
    eps_surprise_pct DECIMAL(8,4),
    revenue_estimate BIGINT,
    revenue_actual BIGINT,
    revenue_surprise_pct DECIMAL(8,4),
    guidance_direction VARCHAR,    -- 'raised', 'lowered', 'maintained', 'none', NULL
    reported_at TIMESTAMP
);

-- Watchlist
CREATE TABLE watchlist (
    symbol VARCHAR PRIMARY KEY,
    name VARCHAR,
    sector VARCHAR,
    industry VARCHAR,
    market_cap BIGINT,
    is_crypto BOOLEAN DEFAULT FALSE,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- Research sessions
CREATE TABLE research_sessions (
    session_id VARCHAR PRIMARY KEY,
    target_symbol VARCHAR,
    target_event_id VARCHAR REFERENCES events(event_id),
    title VARCHAR,
    status VARCHAR DEFAULT 'active',  -- 'active', 'archived'
    context_summary TEXT,              -- Agent-generated summary for resumption
    scenarios JSON,                    -- User's scenario distribution
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Conversation history within sessions
CREATE TABLE conversation_history (
    message_id VARCHAR PRIMARY KEY,
    session_id VARCHAR REFERENCES research_sessions(session_id),
    role VARCHAR NOT NULL,             -- 'user', 'assistant'
    content TEXT NOT NULL,
    tool_calls JSON,                   -- If assistant used tools
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cached transcripts
CREATE TABLE transcripts (
    transcript_id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    event_id VARCHAR REFERENCES events(event_id),
    event_date DATE,
    source VARCHAR,                    -- 'fmp', 'seeking_alpha', etc.
    content TEXT,
    summary TEXT,                      -- Agent-generated summary
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Generic cache for external data
CREATE TABLE cache (
    cache_key VARCHAR PRIMARY KEY,
    data JSON,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4.2 Indexes

```sql
CREATE INDEX idx_prices_symbol_time ON prices(symbol, timestamp DESC);
CREATE INDEX idx_events_date ON events(event_date);
CREATE INDEX idx_events_symbol ON events(symbol);
CREATE INDEX idx_events_type_date ON events(event_type, event_date);
CREATE INDEX idx_earnings_symbol ON earnings(symbol);
CREATE INDEX idx_sessions_status ON research_sessions(status);
CREATE INDEX idx_conversations_session ON conversation_history(session_id, created_at);
```

---

## 5. Agent Tools

### 5.1 Local Data Tools

#### query_prices
```python
def query_prices(
    symbols: list[str],
    start_date: str,           # ISO format
    end_date: str,             # ISO format  
    timeframe: str = "daily",  # "daily" or "hourly"
    fields: list[str] = None   # Default: all OHLCV
) -> dict:
    """
    Retrieve price data for symbols over date range.
    
    Returns: {
        "data": [{"symbol": str, "timestamp": str, "open": float, ...}, ...],
        "symbols_found": list[str],
        "symbols_missing": list[str]
    }
    """
```

#### query_events
```python
def query_events(
    symbols: list[str] = None,      # Filter by symbols, None = all
    event_types: list[str] = None,  # Filter by type
    start_date: str = None,
    end_date: str = None,
    include_past: bool = False      # Include events before today
) -> dict:
    """
    Retrieve events matching filters.
    
    Returns: {
        "events": [{"event_id": str, "symbol": str, "event_type": str, ...}, ...],
        "count": int
    }
    """
```

#### query_earnings
```python
def query_earnings(
    symbols: list[str] = None,
    start_date: str = None,
    end_date: str = None,
    with_actuals_only: bool = False,  # Only reported earnings
    min_surprise_pct: float = None    # Filter by surprise magnitude
) -> dict:
    """
    Retrieve earnings data with estimates and actuals.
    
    Returns: {
        "earnings": [{"symbol": str, "fiscal_quarter": str, "eps_estimate": float, ...}, ...],
        "count": int
    }
    """
```

#### get_watchlist
```python
def get_watchlist(
    sectors: list[str] = None,
    include_crypto: bool = True
) -> dict:
    """
    Retrieve current watchlist.
    
    Returns: {
        "symbols": [{"symbol": str, "name": str, "sector": str, ...}, ...],
        "count": int
    }
    """
```

#### search_transcripts
```python
def search_transcripts(
    query: str,                    # Keyword search
    symbols: list[str] = None,
    limit: int = 10
) -> dict:
    """
    Search cached transcripts by keyword.
    
    Returns: {
        "results": [{"transcript_id": str, "symbol": str, "snippet": str, ...}, ...],
        "count": int
    }
    """
```

### 5.2 External Fetch Tools

#### fetch_transcript
```python
def fetch_transcript(
    symbol: str,
    event_date: str = None,        # Specific date, or None for most recent
    quarter: str = None            # Alternative: "Q3 2024"
) -> dict:
    """
    Fetch earnings transcript, cache locally.
    Sources: FMP (primary), Seeking Alpha (fallback)
    
    Returns: {
        "transcript_id": str,
        "symbol": str,
        "event_date": str,
        "content": str,
        "source": str,
        "cached": bool  # True if already had it
    }
    """
```

#### fetch_news
```python
def fetch_news(
    query: str,                    # Symbol or topic
    days_back: int = 7,
    limit: int = 10
) -> dict:
    """
    Fetch recent news articles.
    
    Returns: {
        "articles": [{"title": str, "url": str, "published": str, "snippet": str}, ...],
        "count": int
    }
    """
```

#### web_search
```python
def web_search(
    query: str,
    limit: int = 10
) -> dict:
    """
    General web search for context.
    
    Returns: {
        "results": [{"title": str, "url": str, "snippet": str}, ...],
        "count": int
    }
    """
```

### 5.3 Compute Tools

#### compute_event_response
```python
def compute_event_response(
    symbol: str,
    event_type: str,               # 'earnings', 'conference', etc.
    lookback_events: int = 8,      # Number of past events to analyze
    window_days: tuple = (-1, 5),  # Days relative to event
    conditions: dict = None        # Optional filters (see below)
) -> dict:
    """
    Calculate price response statistics around events.
    
    conditions example: {
        "eps_surprise": "positive",     # 'positive', 'negative', 'any'
        "guidance": "raised",           # 'raised', 'lowered', 'maintained', 'any'
        "market_regime": "up",          # Based on SPY trend
        "vix_above": 20
    }
    
    Returns: {
        "symbol": str,
        "event_type": str,
        "n_events": int,
        "responses": [
            {"event_date": str, "move_1d": float, "move_5d": float, "details": dict},
            ...
        ],
        "statistics": {
            "mean_1d": float,
            "median_1d": float,
            "std_1d": float,
            "mean_5d": float,
            "positive_rate": float,
            "max_move": float,
            "min_move": float
        }
    }
    """
```

#### find_analogs
```python
def find_analogs(
    symbol: str,
    target_event_date: str,
    criteria: dict,
    max_results: int = 10
) -> dict:
    """
    Find historical events similar to target.
    
    criteria example: {
        "same_symbol": True,              # Include same company's history
        "same_sector": True,              # Include sector peers
        "stock_momentum_90d": (0.2, 0.6), # Range: 20-60% up in 90 days
        "sector_vs_spy_30d": (0.05, None),# Sector outperforming by >5%
        "vix_range": (15, 25),
        "event_type": "earnings"
    }
    
    Returns: {
        "analogs": [
            {
                "symbol": str,
                "event_date": str,
                "similarity_score": float,
                "matching_criteria": list[str],
                "outcome": {"move_1d": float, "move_5d": float, "details": dict}
            },
            ...
        ],
        "count": int
    }
    """
```

#### compute_conditional_stats
```python
def compute_conditional_stats(
    base_symbol: str,
    target_symbol: str,
    condition: dict,
    lookback_days: int = 365
) -> dict:
    """
    Compute statistics for target conditioned on base asset behavior.
    
    condition example: {
        "base_move_threshold": 0.05,   # Base asset moved >5%
        "base_direction": "up",        # 'up', 'down', 'any'
        "window_days": 1               # Over what period
    }
    
    Use case: "How does MSTR move when BTC is up >5%?"
    
    Returns: {
        "base_symbol": str,
        "target_symbol": str,
        "condition": dict,
        "n_occurrences": int,
        "target_responses": [{"date": str, "base_move": float, "target_move": float}, ...],
        "statistics": {
            "mean_response": float,
            "median_response": float,
            "beta": float,             # Regression coefficient
            "correlation": float,
            "positive_rate": float
        }
    }
    """
```

#### calculate_ev
```python
def calculate_ev(
    scenarios: list[dict]
) -> dict:
    """
    Calculate expected value from scenario distribution.
    
    scenarios example: [
        {"name": "Beat + raise", "move_pct": 10.0, "probability": 0.35},
        {"name": "Inline", "move_pct": 2.0, "probability": 0.40},
        {"name": "Miss", "move_pct": -12.0, "probability": 0.25}
    ]
    
    Returns: {
        "expected_value": float,
        "scenarios": [
            {"name": str, "move_pct": float, "probability": float, "contribution": float},
            ...
        ],
        "probability_sum": float,  # Should be 1.0
        "variance": float,
        "std_dev": float
    }
    """
```

#### generate_chart
```python
def generate_chart(
    chart_type: str,               # 'price', 'distribution', 'event_responses', 'comparison'
    data: dict,                    # Chart-specific data
    title: str = None
) -> str:
    """
    Generate terminal-native chart using plotext.
    
    Returns: ASCII chart as string for display
    """
```

---

## 6. Event Monitoring

### 6.1 Surfacing Logic

Events are surfaced based on configurable filters:

**Temporal Filters**
- `lookahead_days`: How far ahead to look (default: 7)
- `lookback_regime_days`: Context window for regime analysis (default: 30)

**Universe Filters**
- Must be in watchlist
- Optional: sector filter, crypto inclusion

**Event Type Priority**
- Configurable weights per event type
- Higher weight = more prominent in display

**Interest Flags** (each adds to "interest score")
- Stock momentum: Up/down >X% into event
- Sector relative: Over/underperforming SPY by >Y%
- VIX regime: High (>25) or low (<15)
- Short interest: Above Nth percentile
- Streak: N consecutive beats/misses
- Leadership change: New CEO/CFO within N days
- Guidance revision: Recent guidance change

**Surfacing Tiers**
- HIGH INTEREST: Interest score >= threshold (e.g., 3+ flags)
- STANDARD: Has upcoming event, 1-2 flags
- LOW: Event exists but no notable flags

### 6.2 Monitor Output Format

```
╔══════════════════════════════════════════════════════════════════╗
║  UPCOMING EVENTS (Next 7 Days)                     Dec 15, 2024  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ★ HIGH INTEREST                                                 ║
║  ┌────────────────────────────────────────────────────────────┐ ║
║  │ NVDA │ Earnings │ Thu Dec 19 AMC │ Q3 FY25                 │ ║
║  │ +45% into event │ Hist avg ±8.2% │ Sector outperforming    │ ║
║  └────────────────────────────────────────────────────────────┘ ║
║                                                                  ║
║  STANDARD                                                        ║
║  • ORCL │ Earnings │ Mon Dec 16 AMC │ 3 consecutive beats       ║
║  • COST │ Earnings │ Tue Dec 17 AMC │                           ║
║  • FOMC │ Macro    │ Wed Dec 18 2pm │                           ║
║                                                                  ║
║  [Enter] Research selected  [f] Filters  [w] Watchlist  [q] Quit║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 7. Data Acquisition

### 7.1 Data Sources

| Data Type | Primary Source | Fallback | Cost |
|-----------|---------------|----------|------|
| Daily OHLCV | yfinance | Alpha Vantage | Free |
| Hourly OHLCV | Alpha Vantage | Polygon.io | Free / $29mo |
| Earnings calendar | FMP | Alpha Vantage | Free |
| Earnings estimates/actuals | FMP | Polygon.io | Free |
| Transcripts | FMP | Seeking Alpha | Free |
| Crypto prices | CoinGecko | Binance | Free |
| News | Web search | - | Free |
| SEC filings | SEC EDGAR | - | Free |

### 7.2 Data Refresh Strategy

| Data | Frequency | Trigger |
|------|-----------|---------|
| Daily prices | Daily 6pm ET | Scheduled |
| Hourly prices | On-demand | Agent request, cached 1hr |
| Earnings calendar | Daily | Scheduled |
| Earnings actuals | Day after event | Event-triggered |
| Transcripts | On-demand | Agent request, cached permanently |
| Watchlist prices | Session start | App launch |

### 7.3 Caching Policy

- Transcripts: Permanent (never expire)
- Hourly prices: 1 hour TTL
- News: 15 minute TTL
- Web search: 5 minute TTL
- Earnings calendar: 24 hour TTL

---

## 8. Project Structure

```
event-researcher/
├── README.md
├── pyproject.toml
├── .env.example                    # API keys template (checked in)
├── .gitignore                      # Includes .env, data/*.duckdb
│
├── config/
│   ├── settings.toml               # App settings
│   ├── watchlist.toml              # Tracked symbols
│   └── filters.toml                # Event surfacing filters
│
├── data/
│   ├── .gitkeep
│   └── (researcher.duckdb)         # Created at runtime, gitignored
│
├── src/
│   ├── __init__.py
│   ├── main.py                     # CLI entry point
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── orchestrator.py         # Main agent loop, Claude API
│   │   ├── context.py              # Session/conversation management
│   │   ├── prompts.py              # System prompts
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── registry.py         # Tool registration and dispatch
│   │       ├── local_data.py       # query_prices, query_events, etc.
│   │       ├── external.py         # fetch_transcript, fetch_news, etc.
│   │       └── compute.py          # compute_event_response, find_analogs, etc.
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── database.py             # DuckDB connection management
│   │   ├── schema.py               # Table creation, migrations
│   │   ├── models.py               # Pydantic models for data types
│   │   └── sources/
│   │       ├── __init__.py
│   │       ├── base.py             # Abstract base for data sources
│   │       ├── yfinance_source.py
│   │       ├── alphavantage.py
│   │       ├── fmp.py
│   │       ├── coingecko.py
│   │       └── seekingalpha.py
│   │
│   ├── monitor/
│   │   ├── __init__.py
│   │   ├── surfacer.py             # Event surfacing logic
│   │   ├── filters.py              # Filter implementations
│   │   └── flags.py                # Interest flag calculations
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py                  # Main Textual application
│   │   ├── monitor_view.py         # Event monitor widget
│   │   ├── chat_view.py            # Research chat widget
│   │   └── components/
│   │       ├── __init__.py
│   │       ├── event_card.py
│   │       └── chart_display.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Config loading
│       ├── logging.py              # Logging setup
│       └── datetime_utils.py       # Market hours, date helpers
│
├── scripts/
│   ├── init_db.py                  # Initialize database schema
│   ├── seed_watchlist.py           # Populate initial watchlist
│   ├── backfill_prices.py          # Historical price load
│   └── refresh_data.py             # Manual data refresh
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_agent/
    ├── test_data/
    ├── test_monitor/
    └── fixtures/
```

---

## 9. Configuration

### 9.1 Environment Variables (.env)

```bash
# .env.example - Copy to .env and fill in values

# Claude API (required)
ANTHROPIC_API_KEY=sk-ant-...

# Data sources (at least one required for prices)
ALPHA_VANTAGE_API_KEY=
FMP_API_KEY=
POLYGON_API_KEY=

# Optional
LOG_LEVEL=INFO
```

### 9.2 Application Settings (config/settings.toml)

```toml
[app]
name = "Event Researcher"
data_dir = "data"
database_name = "researcher.duckdb"

[agent]
default_model = "claude-sonnet-4-20250514"
complex_model = "claude-opus-4-5-20250514"  # For multi-step reasoning
max_tokens = 4096
temperature = 0.3

[data]
price_history_years = 5
hourly_cache_ttl_minutes = 60
transcript_source_priority = ["fmp", "seeking_alpha"]

[ui]
theme = "dark"
monitor_refresh_seconds = 300
```

### 9.3 Watchlist (config/watchlist.toml)

```toml
[settings]
auto_update = false  # Future: auto-update by volume ranking

[[equities]]
# Top names by liquidity - manually curated
symbols = [
    # Mega tech
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
    # Semiconductors
    "AMD", "AVGO", "QCOM", "MU", "INTC", "ARM", "MRVL",
    # Software/Cloud
    "CRM", "NOW", "ORCL", "ADBE", "SNOW", "PLTR", "NET",
    # Fintech/Payments
    "SQ", "PYPL", "COIN", "SHOP", "AFRM",
    # Crypto miners
    "MARA", "RIOT", "CLSK",
    # Consumer
    "NFLX", "DIS", "NKE", "SBUX", "MCD", "COST",
    # Other high-interest
    "GME", "AMC", "SMCI"
    # Add more as needed...
]

[[crypto]]
symbols = ["BTC", "ETH", "SOL", "XRP", "HYPE"]
source = "coingecko"
```

### 9.4 Surfacing Filters (config/filters.toml)

```toml
[temporal]
lookahead_days = 7
lookback_regime_days = 30

[event_types]
# Priority weights (higher = more prominent)
earnings = 100
conference = 80
analyst_day = 70
macro = 60

[flags]
# Stock momentum
momentum_enabled = true
momentum_threshold_pct = 30
momentum_window_days = 90

# Sector relative performance
sector_relative_enabled = true
sector_relative_threshold_pct = 10
sector_relative_window_days = 30

# VIX regime
vix_enabled = true
vix_high = 25
vix_low = 15

# Short interest
short_interest_enabled = false  # Requires additional data source
short_interest_percentile = 80

# Consecutive results
streak_enabled = true
streak_threshold = 3

# Leadership change
leadership_change_enabled = true
leadership_change_lookback_days = 180

[thresholds]
high_interest_min_flags = 3
standard_min_flags = 1
```

---

## 10. Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Goals**: Project scaffolding, database, basic data ingestion

**Tasks**:
- [ ] Initialize project structure with pyproject.toml
- [ ] Set up DuckDB with schema from Section 4
- [ ] Implement config loading (settings.toml, watchlist.toml)
- [ ] Create yfinance data source for daily prices
- [ ] Create FMP data source for earnings calendar
- [ ] Write init_db.py and seed_watchlist.py scripts
- [ ] Basic CLI to verify data loading

**Deliverable**: Can load watchlist, fetch prices, store in DuckDB

### Phase 2: Agent Core (Week 2-3)

**Goals**: Claude integration, core tools, basic chat

**Tasks**:
- [ ] Implement Claude API client with tool calling
- [ ] Create tool registry and dispatch system
- [ ] Implement local data tools: query_prices, query_events, query_earnings
- [ ] Implement compute tools: compute_event_response, find_analogs
- [ ] Create basic CLI chat loop (no TUI yet)
- [ ] Implement conversation history storage

**Deliverable**: Can chat with agent, ask about historical earnings moves

### Phase 3: External Data & Compute (Week 3-4)

**Goals**: Transcript fetching, remaining tools, charting

**Tasks**:
- [ ] Implement fetch_transcript (FMP + Seeking Alpha fallback)
- [ ] Implement fetch_news and web_search tools
- [ ] Implement compute_conditional_stats
- [ ] Implement calculate_ev
- [ ] Implement generate_chart with plotext
- [ ] Add caching layer for external fetches

**Deliverable**: Full tool suite working via CLI chat

### Phase 4: Monitoring & TUI (Week 4-5)

**Goals**: Event surfacing, full Textual UI

**Tasks**:
- [ ] Implement surfacing logic (filters, flags, scoring)
- [ ] Build Textual app shell with split layout
- [ ] Implement monitor view (event list)
- [ ] Implement chat view (research interface)
- [ ] Connect views (select event → open research session)
- [ ] Add keyboard navigation

**Deliverable**: Full TUI application with monitor + research

### Phase 5: Polish (Week 5-6)

**Goals**: Session persistence, UX improvements, robustness

**Tasks**:
- [ ] Implement session save/resume
- [ ] Add scenario storage in sessions
- [ ] Improve error handling and recovery
- [ ] Add loading indicators and async UX
- [ ] Write README with usage instructions
- [ ] Add basic test coverage for critical paths

**Deliverable**: Production-ready for personal use

---

## 11. Example Interactions

### 11.1 Starting a Research Session

```
User: Research NVDA earnings on Dec 19

Agent: Opening research session for NVDA Q3 FY25 earnings (Thu Dec 19, AMC).

Quick context:
• Price: $142.50 (+47% YTD, +12% last 30d)
• Last 4 earnings moves: +6.2%, -3.1%, +14.8%, +9.5%
• Historical average move: ±8.4%
• Sector (SMH): +35% YTD, outperforming SPY by 18%

What would you like to explore?
```

### 11.2 Conditional Analysis

```
User: How have they moved when up >30% into earnings and semis are outperforming?

Agent: Analyzing NVDA earnings where:
• Stock +30%+ in prior 90 days
• SMH outperforming SPY in prior 30 days

Found 5 instances since 2020:

 Date       │ 90d Prior │ 1d Move │ Result
────────────┼───────────┼─────────┼─────────────
 Nov 2023   │ +48%      │ +2.1%   │ Beat, raised
 May 2023   │ +89%      │ +24.3%  │ Beat, raised
 Feb 2024   │ +52%      │ +9.5%   │ Beat, raised
 Aug 2024   │ +41%      │ +6.2%   │ Beat, raised
 Nov 2024   │ +35%      │ -3.1%   │ Beat, inline

Statistics:
• Mean 1d move: +7.8%
• All instances were beats
• The negative move (Nov '24) had inline guidance vs raised

The distribution is right-skewed—big beats drive large moves.
```

### 11.3 Scenario EV Calculation

```
User: Calculate EV for these scenarios:
- Beat + strong Blackwell: +10%, 35%
- Beat + cautious: +2%, 40%
- Miss or guide down: -12%, 25%

Agent: Scenario Analysis:

 Scenario               │ Move   │ Prob  │ Contribution
────────────────────────┼────────┼───────┼──────────────
 Beat + strong Blackwell│ +10.0% │  35%  │ +3.50%
 Beat + cautious        │  +2.0% │  40%  │ +0.80%
 Miss or guide down     │ -12.0% │  25%  │ -3.00%
────────────────────────┼────────┼───────┼──────────────
 Expected Value         │        │ 100%  │ +1.30%

• Probability sum: 100% ✓
• Standard deviation: 8.1%

Your distribution implies modest positive edge assuming 
the market prices a symmetric ±8% move.
```

### 11.4 Cross-Asset Analysis

```
User: How does MSTR typically move when BTC is up >5% in a day?

Agent: Analyzing MSTR response to BTC daily moves >+5%:

Lookback: 365 days
Occurrences: 23 days where BTC >+5%

MSTR response statistics:
• Mean move: +11.2%
• Median move: +9.8%
• Beta to BTC (on these days): 2.1x
• Positive rate: 96% (22/23)

      MSTR Move Distribution (BTC >+5% days)
    │
 15%┤                    ██
    │                 ██ ██
 10%┤              ██ ██ ██ ██
    │           ██ ██ ██ ██ ██
  5%┤        ██ ██ ██ ██ ██ ██ ██
    │     ██ ██ ██ ██ ██ ██ ██ ██
    └────────────────────────────────
         0%   5%  10%  15%  20%  25%

Notable: The convexity increases with BTC move size.
For BTC >+10%, MSTR averaged +22% (4 instances).
```

---

## Appendix A: System Prompt for Agent

```
You are a research assistant for an event-driven trading researcher. 
Your role is to help analyze upcoming market events (earnings, conferences, 
macro releases) and provide data-driven insights.

CAPABILITIES:
- Query local database for historical prices, events, and earnings data
- Fetch external data: transcripts, news, SEC filings
- Compute statistics: event responses, historical analogs, conditional analysis
- Calculate expected values from scenario distributions
- Generate terminal-based charts

INTERACTION STYLE:
- Be concise but thorough
- Lead with data, then interpretation
- When showing tables, use clean ASCII formatting
- Proactively surface relevant context without being asked
- Ask clarifying questions when requests are ambiguous

CONSTRAINTS:
- You provide analysis to support decisions; you do not make trade recommendations
- When data is insufficient, say so clearly
- Distinguish between historical patterns and forward predictions

CURRENT SESSION:
{session_context}
```

---

## Appendix B: Key Dependencies

```toml
[project]
name = "event-researcher"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "anthropic>=0.18.0",      # Claude API
    "duckdb>=0.10.0",         # Database
    "polars>=0.20.0",         # DataFrames
    "httpx>=0.27.0",          # HTTP client
    "textual>=0.50.0",        # TUI framework
    "plotext>=5.2.0",         # Terminal charts
    "yfinance>=0.2.36",       # Yahoo Finance
    "pydantic>=2.6.0",        # Data validation
    "pydantic-settings>=2.2.0", # Settings management
    "apscheduler>=3.10.0",    # Task scheduling
    "tomli>=2.0.0",           # TOML parsing (Python <3.11)
    "rich>=13.7.0",           # Rich text (used by Textual)
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.3.0",
]
```

---

*End of Specification*
