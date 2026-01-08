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
    gamma: float = 0.05          # alias for risk_aversion (V2)

    # Position limits
    max_inventory: int = 500
    max_order_size: int = 100    # fat finger protection

    # Quote generation
    base_spread: float = 2.0     # base spread in cents
    min_absolute_spread: float = 2.0  # minimum spread floor (safety net below Stoikov math)
    quote_size: int = 10         # default quote size

    # Depth-based pricing (V2)
    effective_depth_contracts: int = 100  # contracts required to define "real" price

    # Time horizon for expiry urgency
    time_normalization_sec: float = 86400.0  # 1 day - expiries beyond this treated as T-t=1.0

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
class LIPConfig:
    """LIP (Liquidity Incentive Program) configuration."""
    max_tick_cap: int = 20  # never quote more than this many cents away for LIP


@dataclass
class RiskConfig:
    """Risk management configuration (V2)."""
    hard_stop_ratio: float = 1.2      # Panic dump at this multiple of max_inventory
    bailout_threshold: int = 1        # Hysteresis cents for reservation crossing

    def get_hard_stop_inventory(self, max_inventory: int) -> int:
        """Calculate the hard stop inventory level."""
        return int(max_inventory * self.hard_stop_ratio)


@dataclass
class ImpulseConfig:
    """Impulse control configuration (V2)."""
    enabled: bool = True              # Master switch for impulse control
    taker_fee_cents: int = 7          # Fee per contract when crossing spread
    slippage_buffer: int = 5          # Max cents to cross spread for IOC simulation
    ofi_window_sec: float = 10.0      # Rolling window for Order Flow Imbalance
    ofi_threshold: int = 500          # Net contract imbalance to trigger toxicity bailout


@dataclass
class MicrostructureConfig:
    """Microstructure tracking configuration (V2)."""
    queue_tracking_enabled: bool = False  # Enable queue position estimation
    max_queue_depth: int = 5000           # Contracts ahead before considering refresh


@dataclass
class PeggedModeConfig:
    """Pegged mode configuration for solved markets (V2)."""
    enabled: bool = False             # Toggle for solved markets mode
    fair_value: int = 50              # Fixed center price
    max_exposure: int = 2000          # Higher limit for pegged mode
    reload_threshold: float = 0.8     # % of max size triggering a refresh


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
    lip: LIPConfig = field(default_factory=LIPConfig)
    # V2 configs
    risk: RiskConfig = field(default_factory=RiskConfig)
    impulse: ImpulseConfig = field(default_factory=ImpulseConfig)
    microstructure: MicrostructureConfig = field(default_factory=MicrostructureConfig)
    pegged_mode: PeggedModeConfig = field(default_factory=PeggedModeConfig)

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
        # gamma can be set explicitly or fall back to risk_aversion
        gamma_val = strat_data.get("gamma", strat_data.get("risk_aversion", 0.05))
        strategy = StrategyConfig(
            risk_aversion=strat_data.get("risk_aversion", 0.05),
            gamma=gamma_val,
            max_inventory=strat_data.get("max_inventory", 500),
            max_order_size=strat_data.get("max_order_size", 100),
            base_spread=strat_data.get("base_spread", 2.0),
            min_absolute_spread=strat_data.get("min_absolute_spread", 2.0),
            quote_size=strat_data.get("quote_size", 10),
            effective_depth_contracts=strat_data.get("effective_depth_contracts", 100),
            time_normalization_sec=strat_data.get("time_normalization_sec", 86400.0),
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

        lip_data = data.get("lip", {})
        lip_cfg = LIPConfig(
            max_tick_cap=lip_data.get("max_tick_cap", 20),
        )

        # V2 configs
        risk_data = data.get("risk", {})
        risk_cfg = RiskConfig(
            hard_stop_ratio=risk_data.get("hard_stop_ratio", 1.2),
            bailout_threshold=risk_data.get("bailout_threshold", 1),
        )

        impulse_data = data.get("impulse", {})
        impulse_cfg = ImpulseConfig(
            enabled=impulse_data.get("enabled", True),
            taker_fee_cents=impulse_data.get("taker_fee_cents", 7),
            slippage_buffer=impulse_data.get("slippage_buffer", 5),
            ofi_window_sec=impulse_data.get("ofi_window_sec", 10.0),
            ofi_threshold=impulse_data.get("ofi_threshold", 500),
        )

        micro_data = data.get("microstructure", {})
        micro_cfg = MicrostructureConfig(
            queue_tracking_enabled=micro_data.get("queue_tracking_enabled", False),
            max_queue_depth=micro_data.get("max_queue_depth", 5000),
        )

        pegged_data = data.get("pegged_mode", {})
        pegged_cfg = PeggedModeConfig(
            enabled=pegged_data.get("enabled", False),
            fair_value=pegged_data.get("fair_value", 50),
            max_exposure=pegged_data.get("max_exposure", 2000),
            reload_threshold=pegged_data.get("reload_threshold", 0.8),
        )

        return cls(
            environment=environment,
            market_ticker=data.get("market_ticker", "TODO"),
            credentials=credentials,
            strategy=strategy,
            volatility=volatility,
            rate_limit=rate_limit,
            logging=logging_cfg,
            lip=lip_cfg,
            risk=risk_cfg,
            impulse=impulse_cfg,
            microstructure=micro_cfg,
            pegged_mode=pegged_cfg,
        )


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from file or return defaults."""
    if path is None:
        path = Path("config.toml")

    if Path(path).exists():
        return Config.from_toml(path)

    return Config()
