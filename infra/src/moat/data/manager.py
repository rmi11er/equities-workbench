"""
Data Manager - Unified interface for all data sources.

Orchestrates multiple data providers and presents a single
interface for fetching OHLCV data regardless of source.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

from moat.data.local_ingestor import LocalIngestionEngine
from moat.data.provider import DataNotFoundError, DataProvider
from moat.data.yfinance_adapter import YFinanceAdapter

logger = logging.getLogger(__name__)


class DataManager:
    """Unified interface for fetching market data.

    Aggregates multiple data providers (YFinance, local files)
    and provides a single get() method that routes to the
    appropriate source.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        """Initialize DataManager with configured providers.

        Args:
            config_path: Path to config.yaml. Defaults to config/config.yaml.
            base_dir: Base directory for relative paths. Defaults to cwd.
        """
        load_dotenv()

        self._base_dir = Path(base_dir) if base_dir else Path.cwd()
        self._config = self._load_config(config_path)

        # Initialize providers
        self._providers: dict[str, DataProvider] = {}
        self._init_providers()

    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = self._base_dir / "config" / "config.yaml"

        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}, using defaults")
            return self._default_config()

        with open(config_path) as f:
            return yaml.safe_load(f) or self._default_config()

    def _default_config(self) -> dict:
        """Default configuration."""
        return {
            "paths": {
                "data_incoming": "data/incoming",
                "data_processed": "data/processed",
            },
            "data": {
                "default_source": "yfinance",
                "cache_days": 7,
            },
        }

    def _init_providers(self) -> None:
        """Initialize data providers based on config."""
        paths = self._config.get("paths", {})
        data_config = self._config.get("data", {})

        # YFinance adapter with caching
        cache_dir = self._base_dir / paths.get("data_processed", "data/processed") / ".cache"
        self._providers["yfinance"] = YFinanceAdapter(
            cache_dir=cache_dir,
            cache_days=data_config.get("cache_days", 7),
        )

        # Local ingestion engine
        self._providers["local"] = LocalIngestionEngine(
            incoming_dir=self._base_dir / paths.get("data_incoming", "data/incoming"),
            processed_dir=self._base_dir / paths.get("data_processed", "data/processed"),
            schema_map_path=self._base_dir / "config" / "schema_map.yaml",
        )

        self._default_source = data_config.get("default_source", "yfinance")

    def get(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol.

        Args:
            symbol: Ticker symbol or local dataset name.
            start: Start date (inclusive).
            end: End date (inclusive).
            source: Force specific source ('yfinance', 'local').
                    If None, tries local first, then default source.

        Returns:
            DataFrame with standardized OHLCV columns, indexed by date.

        Raises:
            DataNotFoundError: If symbol not found in any source.
        """
        # If source specified, use it directly
        if source:
            if source not in self._providers:
                raise ValueError(f"Unknown source: {source}")
            return self._providers[source].get(symbol, start, end)

        # Try local first (for proprietary data)
        local_provider = self._providers.get("local")
        if local_provider:
            try:
                return local_provider.get(symbol, start, end)
            except DataNotFoundError:
                pass  # Fall through to default source

        # Try default source
        default_provider = self._providers.get(self._default_source)
        if default_provider:
            return default_provider.get(symbol, start, end)

        raise DataNotFoundError(f"Symbol not found: {symbol}")

    def list_available(self, source: Optional[str] = None) -> dict[str, list[str]]:
        """List available symbols by source.

        Args:
            source: Filter to specific source. None returns all.

        Returns:
            Dict mapping source name to list of available symbols.
        """
        result = {}
        for name, provider in self._providers.items():
            if source is None or name == source:
                result[name] = provider.list_available()
        return result

    def ingest(self, filename: str, schema_name: Optional[str] = None) -> str:
        """Ingest a local CSV file.

        Convenience method that delegates to LocalIngestionEngine.

        Args:
            filename: CSV filename in incoming directory.
            schema_name: Schema mapping to use.

        Returns:
            Dataset name.
        """
        local = self._providers.get("local")
        if not isinstance(local, LocalIngestionEngine):
            raise RuntimeError("Local ingestion not configured")
        return local.ingest(filename, schema_name)

    def ingest_all(self) -> list[str]:
        """Ingest all pending CSV files.

        Returns:
            List of ingested dataset names.
        """
        local = self._providers.get("local")
        if not isinstance(local, LocalIngestionEngine):
            raise RuntimeError("Local ingestion not configured")
        return local.ingest_all_pending()

    @property
    def providers(self) -> dict[str, DataProvider]:
        """Access to underlying providers."""
        return self._providers
