"""DuckDB database connection and operations."""

import asyncio
from contextlib import asynccontextmanager
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import duckdb
import polars as pl

from src.data.models import (
    CacheEntry,
    ConversationMessage,
    Earnings,
    Event,
    Price,
    ResearchSession,
    Transcript,
    WatchlistItem,
)
from src.data.schema import CREATE_INDEXES_SQL, CREATE_TABLES_SQL, SCHEMA_VERSION
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Database:
    """Async-friendly DuckDB database wrapper.

    DuckDB doesn't have native async support, so we run queries in a thread pool
    to avoid blocking the event loop.
    """

    def __init__(self, db_path: Path | str | None = None):
        """Initialize database connection.

        Args:
            db_path: Path to database file. If None, uses settings.
                    Use ':memory:' for in-memory database.
        """
        if db_path is None:
            db_path = get_settings().database_path

        self._db_path = Path(db_path) if db_path != ":memory:" else db_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._lock = asyncio.Lock()

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get the database connection, creating if necessary."""
        if self._conn is None:
            if self._db_path != ":memory:":
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(self._db_path))
        return self._conn

    async def initialize(self) -> None:
        """Initialize database schema."""
        await self._execute(CREATE_TABLES_SQL)
        await self._execute(CREATE_INDEXES_SQL)

        # Check/set schema version
        result = await self._fetchone(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        )
        if result is None:
            await self._execute(
                "INSERT INTO schema_version (version) VALUES (?)", [SCHEMA_VERSION]
            )
            logger.info(f"Initialized database schema version {SCHEMA_VERSION}")
        else:
            current_version = result[0]
            if current_version < SCHEMA_VERSION:
                logger.warning(
                    f"Database schema version {current_version} is older than "
                    f"expected {SCHEMA_VERSION}. Migration may be needed."
                )

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def _execute(self, sql: str, params: list[Any] | None = None) -> None:
        """Execute SQL in thread pool."""

        def _run():
            if params:
                self.conn.execute(sql, params)
            else:
                self.conn.execute(sql)

        async with self._lock:
            await asyncio.to_thread(_run)

    async def _fetchone(self, sql: str, params: list[Any] | None = None) -> tuple | None:
        """Fetch one row."""

        def _run():
            if params:
                return self.conn.execute(sql, params).fetchone()
            return self.conn.execute(sql).fetchone()

        async with self._lock:
            return await asyncio.to_thread(_run)

    async def _fetchall(self, sql: str, params: list[Any] | None = None) -> list[tuple]:
        """Fetch all rows."""

        def _run():
            if params:
                return self.conn.execute(sql, params).fetchall()
            return self.conn.execute(sql).fetchall()

        async with self._lock:
            return await asyncio.to_thread(_run)

    async def _fetchdf(self, sql: str, params: list[Any] | None = None) -> pl.DataFrame:
        """Fetch results as Polars DataFrame."""

        def _run():
            if params:
                return self.conn.execute(sql, params).pl()
            return self.conn.execute(sql).pl()

        async with self._lock:
            return await asyncio.to_thread(_run)

    # ==================== Price Operations ====================

    async def insert_prices(self, prices: list[Price]) -> int:
        """Insert price records, updating on conflict.

        Returns number of rows affected.
        """
        if not prices:
            return 0

        def _run():
            data = [
                (
                    p.symbol,
                    p.timestamp,
                    p.timeframe,
                    float(p.open),
                    float(p.high),
                    float(p.low),
                    float(p.close),
                    p.volume,
                    p.source,
                )
                for p in prices
            ]
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO prices
                (symbol, timestamp, timeframe, open, high, low, close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                data,
            )
            return len(data)

        async with self._lock:
            return await asyncio.to_thread(_run)

    async def get_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        timeframe: str = "daily",
    ) -> pl.DataFrame:
        """Get price data for symbols in date range."""
        symbol_list = ", ".join(f"'{s}'" for s in symbols)
        sql = f"""
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM prices
            WHERE symbol IN ({symbol_list})
              AND timestamp >= ?
              AND timestamp <= ?
              AND timeframe = ?
            ORDER BY symbol, timestamp
        """
        return await self._fetchdf(sql, [start_date, end_date, timeframe])

    async def get_latest_price_date(self, symbol: str, timeframe: str = "daily") -> date | None:
        """Get the most recent price date for a symbol."""
        result = await self._fetchone(
            """
            SELECT MAX(timestamp)::DATE
            FROM prices
            WHERE symbol = ? AND timeframe = ?
            """,
            [symbol, timeframe],
        )
        return result[0] if result and result[0] else None

    # ==================== Event Operations ====================

    async def insert_event(self, event: Event) -> None:
        """Insert or update an event."""
        import json

        await self._execute(
            """
            INSERT OR REPLACE INTO events
            (event_id, symbol, event_type, event_date, event_time, title, description, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                event.event_id,
                event.symbol,
                event.event_type,
                event.event_date,
                event.event_time,
                event.title,
                event.description,
                json.dumps(event.metadata),
            ],
        )

    async def insert_events(self, events: list[Event]) -> int:
        """Insert multiple events."""
        for event in events:
            await self.insert_event(event)
        return len(events)

    async def get_events(
        self,
        symbols: list[str] | None = None,
        event_types: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pl.DataFrame:
        """Get events matching filters."""
        conditions = []
        params = []

        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            conditions.append(f"symbol IN ({placeholders})")
            params.extend(symbols)

        if event_types:
            placeholders = ", ".join("?" for _ in event_types)
            conditions.append(f"event_type IN ({placeholders})")
            params.extend(event_types)

        if start_date:
            conditions.append("event_date >= ?")
            params.append(start_date)

        if end_date:
            conditions.append("event_date <= ?")
            params.append(end_date)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
            SELECT event_id, symbol, event_type, event_date, event_time,
                   title, description, metadata, created_at
            FROM events
            WHERE {where_clause}
            ORDER BY event_date, symbol
        """
        return await self._fetchdf(sql, params if params else None)

    # ==================== Earnings Operations ====================

    async def insert_earnings(self, earnings: Earnings) -> None:
        """Insert or update earnings data."""
        await self._execute(
            """
            INSERT OR REPLACE INTO earnings
            (event_id, symbol, fiscal_quarter, fiscal_year, eps_estimate, eps_actual,
             eps_surprise_pct, revenue_estimate, revenue_actual, revenue_surprise_pct,
             guidance_direction, reported_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                earnings.event_id,
                earnings.symbol,
                earnings.fiscal_quarter,
                earnings.fiscal_year,
                float(earnings.eps_estimate) if earnings.eps_estimate else None,
                float(earnings.eps_actual) if earnings.eps_actual else None,
                float(earnings.eps_surprise_pct) if earnings.eps_surprise_pct else None,
                earnings.revenue_estimate,
                earnings.revenue_actual,
                float(earnings.revenue_surprise_pct) if earnings.revenue_surprise_pct else None,
                earnings.guidance_direction,
                earnings.reported_at,
            ],
        )

    async def get_earnings(
        self,
        symbols: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        with_actuals_only: bool = False,
    ) -> pl.DataFrame:
        """Get earnings data with event info."""
        conditions = []
        params = []

        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            conditions.append(f"e.symbol IN ({placeholders})")
            params.extend(symbols)

        if start_date:
            conditions.append("ev.event_date >= ?")
            params.append(start_date)

        if end_date:
            conditions.append("ev.event_date <= ?")
            params.append(end_date)

        if with_actuals_only:
            conditions.append("e.eps_actual IS NOT NULL")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
            SELECT e.*, ev.event_date, ev.event_time
            FROM earnings e
            JOIN events ev ON e.event_id = ev.event_id
            WHERE {where_clause}
            ORDER BY ev.event_date DESC
        """
        return await self._fetchdf(sql, params if params else None)

    # ==================== Watchlist Operations ====================

    async def add_to_watchlist(self, item: WatchlistItem) -> None:
        """Add or update a watchlist item."""
        await self._execute(
            """
            INSERT OR REPLACE INTO watchlist
            (symbol, name, sector, industry, market_cap, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                item.symbol,
                item.name,
                item.sector,
                item.industry,
                item.market_cap,
                item.notes,
            ],
        )

    async def get_watchlist(self, sectors: list[str] | None = None) -> pl.DataFrame:
        """Get watchlist items."""
        if sectors:
            placeholders = ", ".join("?" for _ in sectors)
            sql = f"SELECT * FROM watchlist WHERE sector IN ({placeholders}) ORDER BY symbol"
            return await self._fetchdf(sql, sectors)
        return await self._fetchdf("SELECT * FROM watchlist ORDER BY symbol")

    async def remove_from_watchlist(self, symbol: str) -> None:
        """Remove a symbol from the watchlist."""
        await self._execute("DELETE FROM watchlist WHERE symbol = ?", [symbol])

    # ==================== Session Operations ====================

    async def create_session(self, session: ResearchSession) -> None:
        """Create a new research session."""
        import json

        await self._execute(
            """
            INSERT INTO research_sessions
            (session_id, target_symbol, target_event_id, title, status,
             context_summary, scenarios)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                session.session_id,
                session.target_symbol,
                session.target_event_id,
                session.title,
                session.status,
                session.context_summary,
                json.dumps(session.scenarios),
            ],
        )

    async def get_session(self, session_id: str) -> ResearchSession | None:
        """Get a research session by ID."""
        import json

        result = await self._fetchone(
            "SELECT * FROM research_sessions WHERE session_id = ?", [session_id]
        )
        if not result:
            return None

        return ResearchSession(
            session_id=result[0],
            target_symbol=result[1],
            target_event_id=result[2],
            title=result[3],
            status=result[4],
            context_summary=result[5],
            scenarios=json.loads(result[6]) if result[6] else {},
            created_at=result[7],
            updated_at=result[8],
        )

    async def update_session(self, session: ResearchSession) -> None:
        """Update an existing session."""
        import json

        await self._execute(
            """
            UPDATE research_sessions
            SET target_symbol = ?, target_event_id = ?, title = ?, status = ?,
                context_summary = ?, scenarios = ?, updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
            """,
            [
                session.target_symbol,
                session.target_event_id,
                session.title,
                session.status,
                session.context_summary,
                json.dumps(session.scenarios),
                session.session_id,
            ],
        )

    async def list_sessions(
        self, status: str | None = None, limit: int = 20
    ) -> pl.DataFrame:
        """List research sessions, ordered by most recent."""
        conditions = []
        params = []

        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
            SELECT session_id, target_symbol, title, status, context_summary,
                   created_at, updated_at
            FROM research_sessions
            WHERE {where_clause}
            ORDER BY updated_at DESC
            LIMIT ?
        """
        params.append(limit)
        return await self._fetchdf(sql, params)

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and its conversation history."""
        await self._execute(
            "DELETE FROM conversation_history WHERE session_id = ?", [session_id]
        )
        await self._execute(
            "DELETE FROM research_sessions WHERE session_id = ?", [session_id]
        )

    # ==================== Conversation Operations ====================

    async def add_message(self, message: ConversationMessage) -> None:
        """Add a message to conversation history."""
        import json

        await self._execute(
            """
            INSERT INTO conversation_history
            (message_id, session_id, role, content, tool_calls)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                message.message_id,
                message.session_id,
                message.role,
                message.content,
                json.dumps(message.tool_calls) if message.tool_calls else None,
            ],
        )

    async def get_conversation(self, session_id: str) -> list[ConversationMessage]:
        """Get conversation history for a session."""
        import json

        rows = await self._fetchall(
            """
            SELECT message_id, session_id, role, content, tool_calls, created_at
            FROM conversation_history
            WHERE session_id = ?
            ORDER BY created_at
            """,
            [session_id],
        )
        return [
            ConversationMessage(
                message_id=r[0],
                session_id=r[1],
                role=r[2],
                content=r[3],
                tool_calls=json.loads(r[4]) if r[4] else None,
                created_at=r[5],
            )
            for r in rows
        ]

    # ==================== Transcript Operations ====================

    async def save_transcript(self, transcript: Transcript) -> None:
        """Save a transcript."""
        await self._execute(
            """
            INSERT OR REPLACE INTO transcripts
            (transcript_id, symbol, event_id, event_date, source, content, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                transcript.transcript_id,
                transcript.symbol,
                transcript.event_id,
                transcript.event_date,
                transcript.source,
                transcript.content,
                transcript.summary,
            ],
        )

    async def get_transcript(
        self, symbol: str, event_date: date | None = None
    ) -> Transcript | None:
        """Get a transcript for a symbol, optionally by date."""
        if event_date:
            result = await self._fetchone(
                "SELECT * FROM transcripts WHERE symbol = ? AND event_date = ?",
                [symbol, event_date],
            )
        else:
            result = await self._fetchone(
                "SELECT * FROM transcripts WHERE symbol = ? ORDER BY event_date DESC LIMIT 1",
                [symbol],
            )

        if not result:
            return None

        return Transcript(
            transcript_id=result[0],
            symbol=result[1],
            event_id=result[2],
            event_date=result[3],
            source=result[4],
            content=result[5],
            summary=result[6],
            fetched_at=result[7],
        )

    # ==================== Cache Operations ====================

    async def set_cache(
        self, key: str, data: dict[str, Any], ttl_seconds: int | None = None
    ) -> None:
        """Set a cache entry."""
        import json
        from datetime import timedelta

        expires_at = None
        if ttl_seconds is not None:
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

        await self._execute(
            """
            INSERT OR REPLACE INTO cache (cache_key, data, expires_at)
            VALUES (?, ?, ?)
            """,
            [key, json.dumps(data), expires_at],
        )

    async def get_cache(self, key: str) -> dict[str, Any] | None:
        """Get a cache entry if not expired."""
        import json

        result = await self._fetchone(
            """
            SELECT data FROM cache
            WHERE cache_key = ?
              AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            [key],
        )
        if result and result[0]:
            return json.loads(result[0])
        return None

    async def clear_expired_cache(self) -> int:
        """Clear expired cache entries. Returns count of deleted rows."""
        result = await self._fetchone(
            "SELECT COUNT(*) FROM cache WHERE expires_at <= CURRENT_TIMESTAMP"
        )
        count = result[0] if result else 0
        await self._execute("DELETE FROM cache WHERE expires_at <= CURRENT_TIMESTAMP")
        return count


@lru_cache
def get_database() -> Database:
    """Get cached database instance."""
    return Database()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[Database, None]:
    """Context manager for database operations."""
    db = get_database()
    try:
        yield db
    finally:
        pass  # Connection is cached, don't close
