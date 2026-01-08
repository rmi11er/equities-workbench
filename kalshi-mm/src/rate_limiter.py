"""Token bucket rate limiter for API calls."""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenBucket:
    """
    Token bucket rate limiter.

    Tokens are added at a fixed rate up to a maximum capacity.
    Requests consume tokens; if insufficient tokens, the request waits.
    """
    rate: float           # tokens per second
    capacity: float       # maximum tokens
    tokens: float = 0.0
    last_update: float = 0.0

    def __post_init__(self):
        self.tokens = self.capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    async def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            True if tokens acquired, False if timed out
        """
        start = time.monotonic()

        async with self._lock:
            while True:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                # Calculate wait time
                needed = tokens - self.tokens
                wait_time = needed / self.rate

                # Check timeout
                if timeout is not None:
                    elapsed = time.monotonic() - start
                    if elapsed + wait_time > timeout:
                        return False
                    wait_time = min(wait_time, timeout - elapsed)

                # Release lock while waiting
                self._lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self._lock.acquire()

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens without waiting.

        Returns:
            True if tokens acquired, False otherwise
        """
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    """
    Combined rate limiter for Kalshi API.

    Maintains separate buckets for read and write operations.
    Write operations include: CreateOrder, CancelOrder, AmendOrder, etc.
    Cancel operations have reduced cost (0.2 per cancel).
    """

    def __init__(self, read_rate: int = 20, write_rate: int = 10):
        # Capacity = 1 second worth of tokens (allows small bursts)
        self.read_bucket = TokenBucket(rate=read_rate, capacity=read_rate)
        self.write_bucket = TokenBucket(rate=write_rate, capacity=write_rate)

    async def acquire_read(self, timeout: Optional[float] = None) -> bool:
        """Acquire a read token."""
        return await self.read_bucket.acquire(1.0, timeout)

    async def acquire_write(self, timeout: Optional[float] = None) -> bool:
        """Acquire a write token (for order operations)."""
        return await self.write_bucket.acquire(1.0, timeout)

    async def acquire_cancel(self, timeout: Optional[float] = None) -> bool:
        """Acquire a cancel token (0.2 cost)."""
        return await self.write_bucket.acquire(0.2, timeout)

    def can_write(self) -> bool:
        """Check if a write is possible without waiting."""
        return self.write_bucket.tokens >= 1.0

    def can_cancel(self) -> bool:
        """Check if a cancel is possible without waiting."""
        return self.write_bucket.tokens >= 0.2
