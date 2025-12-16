"""Pydantic models for data types."""

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, Field


class Price(BaseModel):
    """Price data model."""

    symbol: str
    timestamp: datetime
    timeframe: str  # 'daily' or 'hourly'
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    source: Optional[str] = None


class Event(BaseModel):
    """Event calendar model."""

    event_id: str
    symbol: Optional[str] = None
    event_type: str  # 'earnings', 'conference', 'analyst_day', 'macro'
    event_date: date
    event_time: Optional[str] = None  # 'BMO', 'AMC', 'DMH', or specific time
    title: Optional[str] = None
    description: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None


class Earnings(BaseModel):
    """Earnings-specific data model."""

    event_id: str
    symbol: str
    fiscal_quarter: Optional[str] = None
    fiscal_year: Optional[int] = None
    eps_estimate: Optional[Decimal] = None
    eps_actual: Optional[Decimal] = None
    eps_surprise_pct: Optional[Decimal] = None
    revenue_estimate: Optional[int] = None
    revenue_actual: Optional[int] = None
    revenue_surprise_pct: Optional[Decimal] = None
    guidance_direction: Optional[str] = None  # 'raised', 'lowered', 'maintained', 'none'
    reported_at: Optional[datetime] = None


class WatchlistItem(BaseModel):
    """Watchlist item model."""

    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[int] = None
    added_at: Optional[datetime] = None
    notes: Optional[str] = None


class ResearchSession(BaseModel):
    """Research session model."""

    session_id: str
    target_symbol: Optional[str] = None
    target_event_id: Optional[str] = None
    title: Optional[str] = None
    status: str = "active"  # 'active', 'archived'
    context_summary: Optional[str] = None
    scenarios: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ConversationMessage(BaseModel):
    """Conversation history message model."""

    message_id: str
    session_id: str
    role: str  # 'user', 'assistant'
    content: str
    tool_calls: Optional[dict[str, Any]] = None
    created_at: Optional[datetime] = None


class Transcript(BaseModel):
    """Earnings transcript model."""

    transcript_id: str
    symbol: str
    event_id: Optional[str] = None
    event_date: Optional[date] = None
    source: Optional[str] = None  # 'fmp', 'seeking_alpha', etc.
    content: Optional[str] = None
    summary: Optional[str] = None
    fetched_at: Optional[datetime] = None


class CacheEntry(BaseModel):
    """Generic cache entry model."""

    cache_key: str
    data: dict[str, Any]
    expires_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
