"""
v3 Configuration - Clean, minimal config for passive market making.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

from ..constants import Environment


@dataclass
class CredentialsConfig:
    """API credentials."""
    api_key_id: str
    private_key_path: str

    def load_private_key(self) -> bytes:
        path = Path(self.private_key_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Private key not found: {path}")
        return path.read_bytes()


@dataclass
class QuotingConfig:
    """Core quoting parameters."""
    min_market_width: int = 3           # Only quote if BBO spread >= this (cents)
    min_join_depth_dollars: float = 1000.0  # Hide behind this much depth
    min_edge_for_reduce: int = 2        # Can be alone on reduce side if edge >= this


@dataclass
class SizingConfig:
    """Position sizing parameters."""
    quote_size_dollars: float = 30.0    # Base quote size in dollars
    max_order_size_dollars: float = 50.0  # Fat finger protection
    max_inventory_dollars: float = 250.0  # Per-market exposure limit


@dataclass
class InventoryConfig:
    """Inventory management parameters."""
    gamma: float = 0.01                 # Fade intensity: reservation = mid - q*γ*σ²


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    base_log_dir: str = "logs"


@dataclass
class ConfigV3:
    """
    Root configuration for Passive Market Maker v3.

    Intentionally minimal - only params that matter for passive MM.
    """
    environment: Environment = Environment.DEMO
    market_tickers: list = field(default_factory=list)

    credentials: CredentialsConfig = field(default_factory=lambda: CredentialsConfig("", ""))
    quoting: QuotingConfig = field(default_factory=QuotingConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    inventory: InventoryConfig = field(default_factory=InventoryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def tickers(self) -> list:
        """Alias for market_tickers."""
        return self.market_tickers

    @classmethod
    def from_toml(cls, path: str) -> "ConfigV3":
        """Load configuration from TOML file."""
        if tomllib is None:
            raise ImportError("TOML parsing requires Python 3.11+ or 'tomli' package")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "ConfigV3":
        """Build config from dictionary."""
        env_str = data.get("environment", "demo").upper()
        environment = Environment[env_str]

        creds_data = data.get("credentials", {})
        credentials = CredentialsConfig(
            api_key_id=creds_data.get("api_key_id", ""),
            private_key_path=creds_data.get("private_key_path", ""),
        )

        quoting_data = data.get("quoting", {})
        quoting = QuotingConfig(
            min_market_width=quoting_data.get("min_market_width", 3),
            min_join_depth_dollars=quoting_data.get("min_join_depth_dollars", 1000.0),
            min_edge_for_reduce=quoting_data.get("min_edge_for_reduce", 2),
        )

        sizing_data = data.get("sizing", {})
        sizing = SizingConfig(
            quote_size_dollars=sizing_data.get("quote_size_dollars", 30.0),
            max_order_size_dollars=sizing_data.get("max_order_size_dollars", 50.0),
            max_inventory_dollars=sizing_data.get("max_inventory_dollars", 250.0),
        )

        inv_data = data.get("inventory", {})
        inventory = InventoryConfig(
            gamma=inv_data.get("gamma", 0.01),
        )

        log_data = data.get("logging", {})
        logging_cfg = LoggingConfig(
            log_level=log_data.get("log_level", "INFO"),
            base_log_dir=log_data.get("base_log_dir", "logs"),
        )

        return cls(
            environment=environment,
            market_tickers=data.get("market_tickers", []),
            credentials=credentials,
            quoting=quoting,
            sizing=sizing,
            inventory=inventory,
            logging=logging_cfg,
        )


def load_config_v3(path: Optional[str] = None) -> ConfigV3:
    """Load v3 configuration from file."""
    if path is None:
        path = "config_v3.toml"
    return ConfigV3.from_toml(path)
