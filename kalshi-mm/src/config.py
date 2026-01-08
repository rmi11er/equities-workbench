"""Configuration loading and management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import sys

# tomllib is built-in from Python 3.11+, use tomli for older versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

from .constants import Environment


@dataclass
class CredentialsConfig:
    """API credentials configuration."""
    api_key_id: str = "TODO"
    private_key_path: str = "TODO"

    def load_private_key(self) -> bytes:
        """Load the private key from file."""
        path = Path(self.private_key_path)
        if not path.exists():
            raise FileNotFoundError(f"Private key not found: {self.private_key_path}")
        return path.read_bytes()


@dataclass
class StrategyConfig:
    """Strategy parameters."""
    # Avellaneda-Stoikov parameters
    risk_aversion: float = 0.05  # gamma
    time_horizon: float = 1.0    # T-t (constant)

    # Position limits
    max_inventory: int = 500
    max_order_size: int = 100    # fat finger protection

    # Quote generation
    base_spread: float = 2.0     # base spread in cents
    quote_size: int = 10         # default quote size

    # Debouncing
    debounce_cents: int = 2
    debounce_seconds: float = 5.0


@dataclass
class VolatilityConfig:
    """Volatility estimation parameters."""
    ema_halflife_sec: float = 60.0  # 60 second half-life
    min_volatility: float = 0.1     # floor
    initial_volatility: float = 5.0 # starting estimate (cents)

    @property
    def ema_alpha(self) -> float:
        """Calculate EMA alpha from half-life."""
        # alpha = 1 - exp(-ln(2) / halflife)
        # For tick-based: we'll compute per-tick decay
        import math
        return 1 - math.exp(-math.log(2) / self.ema_halflife_sec)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    read_rate: int = 20   # requests per second
    write_rate: int = 10  # requests per second


@dataclass
class LoggingConfig:
    """Logging configuration."""
    ops_log_path: str = "logs/ops.log"
    tape_csv_path: str = "logs/tape.csv"
    log_level: str = "INFO"


@dataclass
class Config:
    """Root configuration object."""
    environment: Environment = Environment.DEMO
    market_ticker: str = "TODO"

    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a TOML file."""
        if tomllib is None:
            raise ImportError(
                "TOML parsing requires Python 3.11+ or 'tomli' package. "
                "Install with: pip install tomli"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Build Config from a dictionary."""
        env_str = data.get("environment", "demo").upper()
        environment = Environment[env_str]

        credentials = CredentialsConfig(
            api_key_id=data.get("credentials", {}).get("api_key_id", "TODO"),
            private_key_path=data.get("credentials", {}).get("private_key_path", "TODO"),
        )

        strat_data = data.get("strategy", {})
        strategy = StrategyConfig(
            risk_aversion=strat_data.get("risk_aversion", 0.05),
            time_horizon=strat_data.get("time_horizon", 1.0),
            max_inventory=strat_data.get("max_inventory", 500),
            max_order_size=strat_data.get("max_order_size", 100),
            base_spread=strat_data.get("base_spread", 2.0),
            quote_size=strat_data.get("quote_size", 10),
            debounce_cents=strat_data.get("debounce_cents", 2),
            debounce_seconds=strat_data.get("debounce_seconds", 5.0),
        )

        vol_data = data.get("volatility", {})
        volatility = VolatilityConfig(
            ema_halflife_sec=vol_data.get("ema_halflife_sec", 60.0),
            min_volatility=vol_data.get("min_volatility", 0.1),
            initial_volatility=vol_data.get("initial_volatility", 5.0),
        )

        rate_data = data.get("rate_limit", {})
        rate_limit = RateLimitConfig(
            read_rate=rate_data.get("read_rate", 20),
            write_rate=rate_data.get("write_rate", 10),
        )

        log_data = data.get("logging", {})
        logging_cfg = LoggingConfig(
            ops_log_path=log_data.get("ops_log_path", "logs/ops.log"),
            tape_csv_path=log_data.get("tape_csv_path", "logs/tape.csv"),
            log_level=log_data.get("log_level", "INFO"),
        )

        return cls(
            environment=environment,
            market_ticker=data.get("market_ticker", "TODO"),
            credentials=credentials,
            strategy=strategy,
            volatility=volatility,
            rate_limit=rate_limit,
            logging=logging_cfg,
        )


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from file or return defaults."""
    if path is None:
        path = Path("config.toml")

    if Path(path).exists():
        return Config.from_toml(path)

    return Config()
