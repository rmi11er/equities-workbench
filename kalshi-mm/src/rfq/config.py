"""RFQ Responder configuration."""

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
        """Load the private key from file."""
        path = Path(self.private_key_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Private key not found: {path}")
        return path.read_bytes()


@dataclass
class PricingConfig:
    """Pricing engine configuration."""
    default_spread_pct: float = 0.05   # 5% spread on theo
    min_spread_cents: int = 2          # Minimum spread floor (in cents)
    max_spread_cents: int = 15         # Maximum spread cap (in cents)
    use_bbo_mid: bool = True           # Use BBO mid as primary price source


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_exposure_dollars: float = 1000.0    # Total $ exposure across all active quotes
    max_single_rfq_dollars: float = 200.0   # Max $ on any single RFQ
    max_contracts_per_rfq: int = 5000       # Hard limit on contracts per RFQ
    max_active_quotes: int = 10             # Concurrent unconfirmed quotes
    position_limit_per_market: int = 1000   # Contracts per underlying market


@dataclass
class FilterConfig:
    """RFQ filtering configuration."""
    min_dollars: float = 1.0               # Minimum RFQ size in dollars (contracts * theo)
    max_dollars: float = 10000.0           # Maximum RFQ size in dollars
    min_legs: int = 1                      # Minimum parlay legs
    max_legs: int = 10                     # Maximum parlay legs
    allowed_collections: list[str] = field(default_factory=list)  # Whitelist (empty = all)
    blocked_collections: list[str] = field(default_factory=list)  # Blacklist
    allowed_sports: list[str] = field(default_factory=list)       # e.g., ["NFL", "NBA"]


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    base_log_dir: str = "logs/rfq"


@dataclass
class ShadowConfig:
    """Shadow/observation mode configuration."""
    enabled: bool = False              # Enable shadow mode (quote but don't expect fills)
    spread_multiplier: float = 10.0    # Multiply spread by this factor (10x = very wide)
    log_all_rfqs: bool = True          # Log all RFQs even if filtered out
    track_market_prices: bool = True   # Poll trades API for market execution prices


@dataclass
class RFQConfig:
    """Root RFQ responder configuration."""
    environment: Environment = Environment.DEMO

    credentials: CredentialsConfig = field(default_factory=lambda: CredentialsConfig("", ""))
    pricing: PricingConfig = field(default_factory=PricingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)

    # Markets to monitor for leg pricing (subscribe to orderbooks)
    leg_tickers: list[str] = field(default_factory=list)

    # Response timing
    quote_ttl_seconds: float = 30.0             # How long our quotes are valid
    confirmation_timeout_seconds: float = 5.0   # Time to confirm after acceptance

    @property
    def is_shadow_mode(self) -> bool:
        """Check if running in shadow/observation mode."""
        return self.shadow.enabled

    @classmethod
    def from_toml(cls, path: str) -> "RFQConfig":
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
    def _from_dict(cls, data: dict) -> "RFQConfig":
        """Build config from dictionary."""
        env_str = data.get("environment", "demo").upper()
        environment = Environment[env_str]

        # Credentials
        creds_data = data.get("credentials", {})
        credentials = CredentialsConfig(
            api_key_id=creds_data.get("api_key_id", ""),
            private_key_path=creds_data.get("private_key_path", ""),
        )

        # Pricing
        pricing_data = data.get("pricing", {})
        pricing = PricingConfig(
            default_spread_pct=pricing_data.get("default_spread_pct", 0.05),
            min_spread_cents=pricing_data.get("min_spread_cents", 2),
            max_spread_cents=pricing_data.get("max_spread_cents", 15),
            use_bbo_mid=pricing_data.get("use_bbo_mid", True),
        )

        # Risk
        risk_data = data.get("risk", {})
        risk = RiskConfig(
            max_exposure_dollars=risk_data.get("max_exposure_dollars", 1000.0),
            max_single_rfq_dollars=risk_data.get("max_single_rfq_dollars", 200.0),
            max_contracts_per_rfq=risk_data.get("max_contracts_per_rfq", 5000),
            max_active_quotes=risk_data.get("max_active_quotes", 10),
            position_limit_per_market=risk_data.get("position_limit_per_market", 1000),
        )

        # Filters
        filters_data = data.get("filters", {})
        filters = FilterConfig(
            min_dollars=filters_data.get("min_dollars", 1.0),
            max_dollars=filters_data.get("max_dollars", 10000.0),
            min_legs=filters_data.get("min_legs", 1),
            max_legs=filters_data.get("max_legs", 10),
            allowed_collections=filters_data.get("allowed_collections", []),
            blocked_collections=filters_data.get("blocked_collections", []),
            allowed_sports=filters_data.get("allowed_sports", []),
        )

        # Logging
        log_data = data.get("logging", {})
        logging_cfg = LoggingConfig(
            log_level=log_data.get("log_level", "INFO"),
            base_log_dir=log_data.get("base_log_dir", "logs/rfq"),
        )

        # Shadow mode
        shadow_data = data.get("shadow", {})
        shadow = ShadowConfig(
            enabled=shadow_data.get("enabled", False),
            spread_multiplier=shadow_data.get("spread_multiplier", 10.0),
            log_all_rfqs=shadow_data.get("log_all_rfqs", True),
            track_market_prices=shadow_data.get("track_market_prices", True),
        )

        return cls(
            environment=environment,
            credentials=credentials,
            pricing=pricing,
            risk=risk,
            filters=filters,
            logging=logging_cfg,
            shadow=shadow,
            leg_tickers=data.get("leg_tickers", []),
            quote_ttl_seconds=data.get("quote_ttl_seconds", 30.0),
            confirmation_timeout_seconds=data.get("confirmation_timeout_seconds", 5.0),
        )


def load_rfq_config(path: Optional[str] = None) -> RFQConfig:
    """Load RFQ configuration from file."""
    if path is None:
        path = "config_rfq.toml"
    return RFQConfig.from_toml(path)
