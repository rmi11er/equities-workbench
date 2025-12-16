"""External data fetching tools."""

from datetime import date, datetime, timedelta
from typing import Any
from uuid import uuid4

import httpx

from src.agent.tools.registry import registry
from src.data.database import get_database
from src.data.models import Transcript
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@registry.register(
    name="fetch_transcript",
    description="""Fetch an earnings call transcript for a symbol.
Searches Seeking Alpha for the transcript and caches it locally.
If no year/quarter specified, fetches the most recent available.""",
    parameters={
        "symbol": {
            "type": "string",
            "description": "Stock symbol",
        },
        "year": {
            "type": "integer",
            "description": "Fiscal year (e.g., 2024). Optional.",
        },
        "quarter": {
            "type": "integer",
            "description": "Fiscal quarter (1-4). Optional.",
        },
    },
    required=["symbol"],
)
async def fetch_transcript(
    symbol: str,
    year: int | None = None,
    quarter: int | None = None,
) -> dict[str, Any]:
    """Fetch earnings transcript, with caching."""
    db = get_database()
    symbol = symbol.upper()

    # Check cache first
    cache_key = f"transcript:{symbol}:{year}:{quarter}"
    cached = await db.get_cache(cache_key)
    if cached:
        logger.info(f"Using cached transcript for {symbol}")
        return {
            "symbol": symbol,
            "cached": True,
            **cached,
        }

    # Also check transcripts table for permanent storage
    existing = await db.get_transcript(symbol)
    if existing and existing.content:
        # Check if it matches requested year/quarter
        if not year or (existing.event_date and existing.event_date.year == year):
            return {
                "symbol": symbol,
                "cached": True,
                "transcript_id": existing.transcript_id,
                "content": existing.content[:5000] + "..." if len(existing.content) > 5000 else existing.content,
                "summary": existing.summary,
                "source": existing.source,
            }

    # Fetch from Seeking Alpha
    from src.data.sources.seekingalpha import SeekingAlphaSource

    source = SeekingAlphaSource()

    try:
        result = await source.fetch_transcript(symbol, year=year, quarter=quarter)
    except Exception as e:
        logger.error(f"Failed to fetch transcript for {symbol}: {e}")
        return {
            "symbol": symbol,
            "error": f"Failed to fetch transcript: {str(e)}",
        }
    finally:
        await source.close()

    if not result:
        return {
            "symbol": symbol,
            "error": "No transcript found",
            "year": year,
            "quarter": quarter,
        }

    # Save to database (permanent storage)
    transcript = Transcript(
        transcript_id=result["transcript_id"],
        symbol=symbol,
        event_date=datetime.strptime(result["published_date"], "%Y-%m-%d").date() if result.get("published_date") else None,
        source="seeking_alpha",
        content=result["content"],
    )
    await db.save_transcript(transcript)

    # Truncate content for response (full content is in DB)
    content_preview = result["content"]
    if len(content_preview) > 5000:
        content_preview = content_preview[:5000] + "...\n[Truncated - full transcript saved to database]"

    return {
        "symbol": symbol,
        "cached": False,
        "transcript_id": result["transcript_id"],
        "title": result.get("title"),
        "content": content_preview,
        "source": "seeking_alpha",
        "url": result.get("url"),
    }


@registry.register(
    name="fetch_news",
    description="""Fetch recent news articles about a symbol or topic.
Returns headlines, snippets, and URLs for recent news.""",
    parameters={
        "query": {
            "type": "string",
            "description": "Search query (symbol or topic)",
        },
        "days_back": {
            "type": "integer",
            "description": "Number of days to look back (default: 7)",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results (default: 10)",
        },
    },
    required=["query"],
)
async def fetch_news(
    query: str,
    days_back: int = 7,
    limit: int = 10,
) -> dict[str, Any]:
    """Fetch recent news articles."""
    db = get_database()

    # Check cache
    cache_key = f"news:{query}:{days_back}"
    cached = await db.get_cache(cache_key)
    if cached:
        return {"query": query, "cached": True, **cached}

    # Use DuckDuckGo news search (no API key required)
    search_url = "https://html.duckduckgo.com/html/"

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            response = await client.post(
                search_url,
                data={"q": f"{query} news", "t": "h_", "ia": "news"},
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                },
            )
            response.raise_for_status()
            html = response.text
        except Exception as e:
            logger.error(f"News search failed: {e}")
            return {"query": query, "error": f"Search failed: {str(e)}"}

    # Parse results
    import re

    articles = []

    # Find result blocks
    result_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
    snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>([^<]+)</a>'

    links = re.findall(result_pattern, html)
    snippets = re.findall(snippet_pattern, html)

    for i, (url, title) in enumerate(links[:limit]):
        # Decode URL
        if "uddg=" in url:
            url_match = re.search(r'uddg=([^&]+)', url)
            if url_match:
                from urllib.parse import unquote
                url = unquote(url_match.group(1))

        snippet = snippets[i] if i < len(snippets) else ""

        articles.append({
            "title": title.strip(),
            "url": url,
            "snippet": snippet.strip(),
        })

    result = {
        "articles": articles,
        "count": len(articles),
    }

    # Cache for 15 minutes
    await db.set_cache(cache_key, result, ttl_seconds=900)

    return {"query": query, "cached": False, **result}


@registry.register(
    name="web_search",
    description="""Perform a general web search for information.
Useful for finding context, recent developments, or specific facts.""",
    parameters={
        "query": {
            "type": "string",
            "description": "Search query",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results (default: 10)",
        },
    },
    required=["query"],
)
async def web_search(
    query: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Perform a web search."""
    db = get_database()

    # Check cache
    cache_key = f"search:{query}"
    cached = await db.get_cache(cache_key)
    if cached:
        return {"query": query, "cached": True, **cached}

    # Use DuckDuckGo search
    search_url = "https://html.duckduckgo.com/html/"

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            response = await client.post(
                search_url,
                data={"q": query, "t": "h_"},
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                },
            )
            response.raise_for_status()
            html = response.text
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {"query": query, "error": f"Search failed: {str(e)}"}

    # Parse results
    import re

    results = []

    # Find result blocks
    result_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
    snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>([^<]+)</a>'

    links = re.findall(result_pattern, html)
    snippets = re.findall(snippet_pattern, html)

    for i, (url, title) in enumerate(links[:limit]):
        # Decode URL
        if "uddg=" in url:
            url_match = re.search(r'uddg=([^&]+)', url)
            if url_match:
                from urllib.parse import unquote
                url = unquote(url_match.group(1))

        snippet = snippets[i] if i < len(snippets) else ""

        results.append({
            "title": title.strip(),
            "url": url,
            "snippet": snippet.strip(),
        })

    result = {
        "results": results,
        "count": len(results),
    }

    # Cache for 5 minutes
    await db.set_cache(cache_key, result, ttl_seconds=300)

    return {"query": query, "cached": False, **result}


@registry.register(
    name="search_transcripts",
    description="""Search cached transcripts by keyword.
Only searches transcripts that have already been fetched and stored locally.""",
    parameters={
        "query": {
            "type": "string",
            "description": "Keyword to search for",
        },
        "symbols": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Filter by symbols (optional)",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum results (default: 10)",
        },
    },
    required=["query"],
)
async def search_transcripts(
    query: str,
    symbols: list[str] | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search locally cached transcripts."""
    db = get_database()

    # Build SQL query
    sql = """
        SELECT transcript_id, symbol, event_date, source, content
        FROM transcripts
        WHERE content LIKE ?
    """
    params = [f"%{query}%"]

    if symbols:
        placeholders = ", ".join("?" for _ in symbols)
        sql += f" AND symbol IN ({placeholders})"
        params.extend(symbols)

    sql += f" LIMIT {limit}"

    results = []
    rows = await db._fetchall(sql, params)

    for row in rows:
        content = row[4] or ""
        # Find snippet around the query
        query_lower = query.lower()
        content_lower = content.lower()
        idx = content_lower.find(query_lower)

        if idx >= 0:
            start = max(0, idx - 100)
            end = min(len(content), idx + len(query) + 100)
            snippet = "..." + content[start:end] + "..."
        else:
            snippet = content[:200] + "..."

        results.append({
            "transcript_id": row[0],
            "symbol": row[1],
            "event_date": str(row[2]) if row[2] else None,
            "source": row[3],
            "snippet": snippet,
        })

    return {
        "query": query,
        "results": results,
        "count": len(results),
    }
