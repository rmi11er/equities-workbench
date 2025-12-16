"""Seeking Alpha data source for earnings transcripts."""

import re
from datetime import date, datetime
from typing import Any
from uuid import uuid4

from src.data.sources.base import DataSource


class SeekingAlphaSource(DataSource):
    """Data source for Seeking Alpha transcripts.

    Note: This uses web scraping. Be respectful of rate limits.
    Transcripts are cached permanently after fetching.
    """

    BASE_URL = "https://seekingalpha.com"

    def __init__(self):
        super().__init__()
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    async def fetch_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        timeframe: str = "daily",
    ) -> list[dict[str, Any]]:
        """Not implemented - use YFinance for prices."""
        raise NotImplementedError("Use YFinanceSource for price data")

    async def fetch_company_info(self, symbol: str) -> dict[str, Any] | None:
        """Not implemented - use YFinance for company info."""
        raise NotImplementedError("Use YFinanceSource for company info")

    async def fetch_transcript_list(
        self,
        symbol: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Fetch list of available transcripts for a symbol.

        Returns list of transcript metadata (not full content).
        """
        url = f"{self.BASE_URL}/symbol/{symbol}/earnings/transcripts"

        self.logger.debug(f"Fetching transcript list for {symbol}")

        try:
            response = await self.client.get(url, headers=self._headers)
            response.raise_for_status()
            html = response.text
        except Exception as e:
            self.logger.error(f"Failed to fetch transcript list for {symbol}: {e}")
            return []

        # Parse transcript links from the page
        # Looking for patterns like /article/XXXXX-symbol-fy-2024-q3-earnings-call-transcript
        pattern = r'/article/(\d+[^"\'>\s]+earnings-call-transcript[^"\'>\s]*)'
        matches = re.findall(pattern, html, re.IGNORECASE)

        # Deduplicate and limit
        seen = set()
        transcripts = []

        for match in matches:
            article_path = f"/article/{match}"
            if article_path not in seen:
                seen.add(article_path)

                # Try to extract date/quarter from URL
                quarter_match = re.search(r'fy-?(\d{4})-?q(\d)', match, re.IGNORECASE)
                fiscal_year = None
                fiscal_quarter = None

                if quarter_match:
                    fiscal_year = int(quarter_match.group(1))
                    fiscal_quarter = f"Q{quarter_match.group(2)}"

                transcripts.append({
                    "article_path": article_path,
                    "url": f"{self.BASE_URL}{article_path}",
                    "fiscal_year": fiscal_year,
                    "fiscal_quarter": fiscal_quarter,
                })

                if len(transcripts) >= limit:
                    break

        self.logger.info(f"Found {len(transcripts)} transcripts for {symbol}")
        return transcripts

    async def fetch_transcript(
        self,
        symbol: str,
        article_url: str | None = None,
        year: int | None = None,
        quarter: int | None = None,
    ) -> dict[str, Any] | None:
        """Fetch a specific earnings transcript.

        Args:
            symbol: Stock symbol
            article_url: Direct URL to transcript article (if known)
            year: Fiscal year to find
            quarter: Fiscal quarter to find (1-4)

        Returns:
            Dict with transcript content, or None if not found
        """
        # If no direct URL, search for the transcript
        if not article_url:
            transcripts = await self.fetch_transcript_list(symbol, limit=20)

            if not transcripts:
                return None

            # Find matching transcript by year/quarter if specified
            if year and quarter:
                for t in transcripts:
                    if t.get("fiscal_year") == year and t.get("fiscal_quarter") == f"Q{quarter}":
                        article_url = t["url"]
                        break

            # Default to most recent
            if not article_url and transcripts:
                article_url = transcripts[0]["url"]

        if not article_url:
            return None

        self.logger.debug(f"Fetching transcript from {article_url}")

        try:
            response = await self.client.get(article_url, headers=self._headers)
            response.raise_for_status()
            html = response.text
        except Exception as e:
            self.logger.error(f"Failed to fetch transcript: {e}")
            return None

        # Extract transcript content
        content = self._extract_transcript_content(html)

        if not content:
            self.logger.warning(f"Could not extract content from {article_url}")
            return None

        # Extract metadata
        title = self._extract_title(html)
        pub_date = self._extract_date(html)

        return {
            "transcript_id": str(uuid4()),
            "symbol": symbol,
            "url": article_url,
            "title": title,
            "content": content,
            "published_date": pub_date,
            "source": "seeking_alpha",
        }

    def _extract_transcript_content(self, html: str) -> str | None:
        """Extract the transcript text from HTML."""
        # Look for the article body content
        # Seeking Alpha uses various div structures

        # Try to find content between common markers
        patterns = [
            # Article body pattern
            r'<div[^>]*data-test-id="article-content"[^>]*>(.*?)</div>\s*</div>\s*</div>',
            # Alternative pattern
            r'<div[^>]*class="[^"]*article-content[^"]*"[^>]*>(.*?)</div>',
            # Transcript specific
            r'<div[^>]*id="[^"]*transcript[^"]*"[^>]*>(.*?)</div>',
        ]

        content = None
        for pattern in patterns:
            match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1)
                break

        if not content:
            # Fallback: try to find paragraphs with speaker patterns
            # Earnings calls typically have "Operator", "CEO", etc.
            speaker_pattern = r'<p[^>]*>(<strong>)?[A-Z][a-z]+\s+[A-Z][a-z]+.*?</p>'
            paragraphs = re.findall(speaker_pattern, html, re.DOTALL)
            if len(paragraphs) > 10:  # Likely a transcript
                content = "\n".join(paragraphs)

        if not content:
            return None

        # Clean HTML tags
        content = re.sub(r'<[^>]+>', ' ', content)
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()

        # Basic validation - transcripts should be substantial
        if len(content) < 1000:
            return None

        return content

    def _extract_title(self, html: str) -> str | None:
        """Extract article title from HTML."""
        patterns = [
            r'<h1[^>]*data-test-id="post-title"[^>]*>([^<]+)</h1>',
            r'<h1[^>]*>([^<]*transcript[^<]*)</h1>',
            r'<title>([^<]+)</title>',
        ]

        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                # Clean up
                title = re.sub(r'\s*\|\s*Seeking Alpha.*', '', title)
                return title

        return None

    def _extract_date(self, html: str) -> str | None:
        """Extract publication date from HTML."""
        patterns = [
            r'<time[^>]*datetime="([^"]+)"',
            r'"datePublished"\s*:\s*"([^"]+)"',
        ]

        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                date_str = match.group(1)
                # Try to parse and normalize
                try:
                    if 'T' in date_str:
                        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        return dt.strftime("%Y-%m-%d")
                    return date_str[:10]
                except ValueError:
                    return date_str

        return None
