"""
Local data ingestion engine for proprietary data sources.

Monitors a directory for incoming CSV files, normalizes them
according to schema mappings, and stores as Parquet.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
import yaml

from moat.data.provider import DataFetchError, DataNotFoundError, DataProvider

logger = logging.getLogger(__name__)


class LocalIngestionEngine(DataProvider):
    """Data provider for locally ingested files.

    Monitors an incoming directory for CSV files, translates column
    names according to schema_map.yaml, and stores processed data
    as Parquet files for fast retrieval.
    """

    def __init__(
        self,
        incoming_dir: Path,
        processed_dir: Path,
        schema_map_path: Path,
    ) -> None:
        """Initialize local ingestion engine.

        Args:
            incoming_dir: Directory to watch for incoming CSVs.
            processed_dir: Directory for processed Parquet files.
            schema_map_path: Path to schema mapping YAML file.
        """
        self._incoming_dir = Path(incoming_dir)
        self._processed_dir = Path(processed_dir)
        self._schema_map_path = Path(schema_map_path)

        # Create directories
        self._incoming_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)

        # Load schema mappings
        self._schema_map = self._load_schema_map()

    @property
    def source_name(self) -> str:
        """Return identifier for this data source."""
        return "local"

    def _load_schema_map(self) -> dict:
        """Load schema mappings from YAML file."""
        if not self._schema_map_path.exists():
            logger.warning(f"Schema map not found: {self._schema_map_path}")
            return {"generic": self._default_mapping()}

        with open(self._schema_map_path) as f:
            return yaml.safe_load(f) or {"generic": self._default_mapping()}

    def _default_mapping(self) -> dict:
        """Default column mapping for simple CSVs."""
        return {
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "adjusted_close": "adj_close",
        }

    def ingest(self, filename: str, schema_name: Optional[str] = None) -> str:
        """Ingest a CSV file from the incoming directory.

        Args:
            filename: Name of CSV file in incoming directory.
            schema_name: Schema mapping to use. Auto-detected if None.

        Returns:
            Name of the processed dataset.

        Raises:
            DataFetchError: If ingestion fails.
        """
        csv_path = self._incoming_dir / filename
        if not csv_path.exists():
            raise DataFetchError(f"File not found: {csv_path}")

        # Determine schema
        base_name = csv_path.stem
        if schema_name is None:
            schema_name = base_name if base_name in self._schema_map else "generic"

        mapping = self._schema_map.get(schema_name, self._default_mapping())

        try:
            # Use polars for fast CSV reading
            df_pl = pl.read_csv(csv_path, try_parse_dates=True)

            # Convert to pandas for compatibility
            df = df_pl.to_pandas()

            # Apply column mapping
            df = self._apply_mapping(df, mapping)

            # Validate
            df = self.validate_dataframe(df)

            # Save as parquet
            output_path = self._processed_dir / f"{base_name}.parquet"
            df.to_parquet(output_path)

            logger.info(f"Ingested {filename} -> {output_path} ({len(df)} rows)")
            return base_name

        except Exception as e:
            logger.error(f"Failed to ingest {filename}: {e}")
            raise DataFetchError(f"Ingestion failed for {filename}: {e}") from e

    def _apply_mapping(self, df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        """Apply column name mapping to DataFrame."""
        # Build reverse mapping (source -> standard)
        rename_map = {}
        for standard_name, source_name in mapping.items():
            if source_name is None:
                continue
            # Case-insensitive matching
            for col in df.columns:
                if col.lower() == source_name.lower():
                    rename_map[col] = standard_name
                    break

        df = df.rename(columns=rename_map)
        return df

    def get(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch processed local data.

        Args:
            symbol: Dataset name (filename without extension).
            start: Start date filter.
            end: End date filter.

        Returns:
            DataFrame with standardized OHLCV columns.

        Raises:
            DataNotFoundError: If dataset not found.
        """
        parquet_path = self._processed_dir / f"{symbol}.parquet"

        # Check if we need to ingest first
        if not parquet_path.exists():
            csv_path = self._incoming_dir / f"{symbol}.csv"
            if csv_path.exists():
                self.ingest(f"{symbol}.csv")
            else:
                raise DataNotFoundError(f"Dataset not found: {symbol}")

        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            raise DataFetchError(f"Failed to read {symbol}: {e}") from e

        # Apply date filters
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]

        logger.info(f"Loaded {len(df)} bars for {symbol} from local storage")
        return df

    def list_available(self) -> list[str]:
        """List available processed datasets."""
        processed = [f.stem for f in self._processed_dir.glob("*.parquet")]
        pending = [f.stem for f in self._incoming_dir.glob("*.csv")]
        return list(set(processed + pending))

    def ingest_all_pending(self) -> list[str]:
        """Ingest all CSV files in the incoming directory.

        Returns:
            List of successfully ingested dataset names.
        """
        ingested = []
        for csv_file in self._incoming_dir.glob("*.csv"):
            try:
                name = self.ingest(csv_file.name)
                ingested.append(name)
            except DataFetchError as e:
                logger.error(f"Skipping {csv_file.name}: {e}")
        return ingested

    def refresh(self, symbol: str) -> None:
        """Re-ingest a dataset from the incoming CSV.

        Args:
            symbol: Dataset name to refresh.
        """
        csv_path = self._incoming_dir / f"{symbol}.csv"
        if csv_path.exists():
            self.ingest(f"{symbol}.csv")
        else:
            raise DataNotFoundError(f"No CSV found for {symbol}")
