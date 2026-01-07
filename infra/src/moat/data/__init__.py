"""
Data abstraction layer for The Moat.

Provides a unified interface for fetching OHLCV data from multiple sources.
"""

from moat.data.local_ingestor import LocalIngestionEngine
from moat.data.manager import DataManager
from moat.data.provider import DataProvider
from moat.data.yfinance_adapter import YFinanceAdapter

__all__ = [
    "DataProvider",
    "DataManager",
    "YFinanceAdapter",
    "LocalIngestionEngine",
]
