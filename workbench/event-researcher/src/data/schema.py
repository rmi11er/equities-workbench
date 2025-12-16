"""Database schema definitions and migrations."""

SCHEMA_VERSION = 1

CREATE_TABLES_SQL = """
-- Price data (daily + hourly)
CREATE TABLE IF NOT EXISTS prices (
    symbol VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    timeframe VARCHAR NOT NULL,
    open DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    close DECIMAL(12,4),
    volume BIGINT,
    source VARCHAR,
    PRIMARY KEY (symbol, timestamp, timeframe)
);

-- Event calendar
CREATE TABLE IF NOT EXISTS events (
    event_id VARCHAR PRIMARY KEY,
    symbol VARCHAR,
    event_type VARCHAR NOT NULL,
    event_date DATE NOT NULL,
    event_time VARCHAR,
    title VARCHAR,
    description TEXT,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Earnings-specific data
CREATE TABLE IF NOT EXISTS earnings (
    event_id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    fiscal_quarter VARCHAR,
    fiscal_year INTEGER,
    eps_estimate DECIMAL(10,4),
    eps_actual DECIMAL(10,4),
    eps_surprise_pct DECIMAL(8,4),
    revenue_estimate BIGINT,
    revenue_actual BIGINT,
    revenue_surprise_pct DECIMAL(8,4),
    guidance_direction VARCHAR,
    reported_at TIMESTAMP
);

-- Watchlist
CREATE TABLE IF NOT EXISTS watchlist (
    symbol VARCHAR PRIMARY KEY,
    name VARCHAR,
    sector VARCHAR,
    industry VARCHAR,
    market_cap BIGINT,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- Research sessions
CREATE TABLE IF NOT EXISTS research_sessions (
    session_id VARCHAR PRIMARY KEY,
    target_symbol VARCHAR,
    target_event_id VARCHAR,
    title VARCHAR,
    status VARCHAR DEFAULT 'active',
    context_summary TEXT,
    scenarios JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Conversation history within sessions
CREATE TABLE IF NOT EXISTS conversation_history (
    message_id VARCHAR PRIMARY KEY,
    session_id VARCHAR,
    role VARCHAR NOT NULL,
    content TEXT NOT NULL,
    tool_calls JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cached transcripts
CREATE TABLE IF NOT EXISTS transcripts (
    transcript_id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    event_id VARCHAR,
    event_date DATE,
    source VARCHAR,
    content TEXT,
    summary TEXT,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Generic cache for external data
CREATE TABLE IF NOT EXISTS cache (
    cache_key VARCHAR PRIMARY KEY,
    data JSON,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_prices_symbol_time ON prices(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_prices_timeframe ON prices(timeframe, symbol);
CREATE INDEX IF NOT EXISTS idx_events_date ON events(event_date);
CREATE INDEX IF NOT EXISTS idx_events_symbol ON events(symbol);
CREATE INDEX IF NOT EXISTS idx_events_type_date ON events(event_type, event_date);
CREATE INDEX IF NOT EXISTS idx_earnings_symbol ON earnings(symbol);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON research_sessions(status);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversation_history(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_transcripts_symbol ON transcripts(symbol);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at);
"""


def get_schema_sql() -> str:
    """Get the full schema SQL."""
    return CREATE_TABLES_SQL + "\n" + CREATE_INDEXES_SQL
