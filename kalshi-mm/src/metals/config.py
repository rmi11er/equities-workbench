"""Configuration for the isolated metals research pipeline."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by the repository's Python 3.9 runtime
    import tomli as tomllib


PRODUCTION_REST_URL = "https://external-api.kalshi.com/trade-api/v2"
PRODUCTION_WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"


@dataclass(frozen=True)
class ProductConfig:
    """Discovery hints for one metal; contract metadata remains authoritative."""

    enabled: bool
    series_tickers: List[str] = field(default_factory=list)
    ticker_prefixes: List[str] = field(default_factory=list)
    title_keywords: List[str] = field(default_factory=list)
    pyth_symbol: str = ""
    pyth_feed_id: str = ""
    pyth_lazer_id: Optional[int] = None
    cme_symbols: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class StorageConfig:
    root: str = "data/metals"
    fsync_every_records: int = 1


@dataclass(frozen=True)
class KalshiConfig:
    rest_url: str = PRODUCTION_REST_URL
    ws_url: str = PRODUCTION_WS_URL
    credentials_config: str = "config.toml"
    api_key_id_env: str = "KALSHI_API_KEY_ID"
    private_key_path_env: str = "KALSHI_PRIVATE_KEY_PATH"
    page_limit: int = 1000
    discovery_interval_seconds: float = 15.0
    request_timeout_seconds: float = 15.0
    fractional_contracts_enabled: bool = True
    fractional_contract_quantum: str = "0.01"
    upcoming_markets_per_product: int = 4
    max_read_requests_per_second: float = 18.0
    max_read_retries: int = 5
    read_retry_base_seconds: float = 0.25


@dataclass(frozen=True)
class PythConfig:
    enabled: bool = False
    api_key_env: str = "PYTH_API_KEY"
    endpoint: str = "https://pyth.dourolabs.app/v1"
    lazer_ws_endpoints: List[str] = field(
        default_factory=lambda: [
            "wss://pyth-lazer-0.dourolabs.app/v1/stream",
            "wss://pyth-lazer-1.dourolabs.app/v1/stream",
            "wss://pyth-lazer-2.dourolabs.app/v1/stream",
        ]
    )
    history_chunk_seconds: int = 86_400


@dataclass(frozen=True)
class CmeConfig:
    enabled: bool = False
    provider: str = "databento"
    api_key_env: str = "DATABENTO_API_KEY"
    dataset: str = "GLBX.MDP3"
    stype_in: str = "parent"
    schemas: List[str] = field(
        default_factory=lambda: [
            "mbo",
            "trades",
            "definition",
            "status",
            "statistics",
        ]
    )
    mbo_snapshot: bool = True
    reconnect_policy: str = "reconnect"
    include_ts_out: bool = True


@dataclass(frozen=True)
class ClockConfig:
    """Local clock-health evidence required by synchronized research runs."""

    enabled: bool = False
    provider: str = "chrony"
    chronyc_path: str = "chronyc"
    sample_interval_seconds: float = 1.0
    probe_timeout_seconds: float = 2.0
    max_probe_elapsed_delta_ns: int = 1_000_000


@dataclass(frozen=True)
class GovernanceConfig:
    """Content-addressed approval required before licensed provider collection."""

    provider_approval_path: str = ""
    pyth_historical_research_assumption_path: str = ""
    databento_historical_research_assumption_path: str = ""


@dataclass(frozen=True)
class ShadowConfig:
    """Research-only execution envelope; live order routing is prohibited."""

    enabled: bool = True
    execution_mode: str = "shadow"
    allow_live_orders: bool = False
    config_version: str = "metals-shadow-v1"
    bankroll_min_dollars: str = "5000"
    bankroll_max_dollars: str = "10000"
    max_contracts_per_order: str = "250"
    max_net_contracts_per_market: str = "1000"
    max_aggregate_worst_case_loss_dollars: str = "3000"
    max_loss_per_market_dollars: str = "1000"
    daily_loss_limit_dollars: str = "3000"


@dataclass(frozen=True)
class MetalsConfig:
    environment: str = "production"
    storage: StorageConfig = field(default_factory=StorageConfig)
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    pyth: PythConfig = field(default_factory=PythConfig)
    cme: CmeConfig = field(default_factory=CmeConfig)
    clock: ClockConfig = field(default_factory=ClockConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)
    products: Dict[str, ProductConfig] = field(default_factory=dict)

    @property
    def enabled_products(self) -> Dict[str, ProductConfig]:
        return {name: cfg for name, cfg in self.products.items() if cfg.enabled}

    def public_snapshot(self) -> Dict[str, Any]:
        """Return a manifest-safe config snapshot containing no secret values."""

        return asdict(self)


@dataclass(frozen=True)
class ResolvedKalshiCredentials:
    api_key_id: str
    private_key_path: str


def _load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_kalshi_credentials(config: KalshiConfig) -> ResolvedKalshiCredentials:
    """Resolve Kalshi credentials from environment, then an ignored local TOML."""

    api_key_id = os.environ.get(config.api_key_id_env, "").strip()
    private_key_path = os.environ.get(config.private_key_path_env, "").strip()

    if not api_key_id or not private_key_path:
        credential_path = Path(config.credentials_config).expanduser()
        if credential_path.exists():
            data = _load_toml(credential_path)
            credentials = data.get("credentials", {})
            api_key_id = api_key_id or str(credentials.get("api_key_id", "")).strip()
            private_key_path = (
                private_key_path or str(credentials.get("private_key_path", "")).strip()
            )

    if not api_key_id or api_key_id == "TODO":
        raise ValueError(
            "Kalshi API key ID is missing; set the configured environment variable "
            "or provide an ignored credentials_config TOML"
        )
    if not private_key_path or private_key_path == "TODO":
        raise ValueError(
            "Kalshi private-key path is missing; set the configured environment "
            "variable or provide an ignored credentials_config TOML"
        )

    key_path = Path(private_key_path).expanduser()
    if not key_path.is_file():
        raise FileNotFoundError("Kalshi private key not found: {}".format(key_path))

    return ResolvedKalshiCredentials(
        api_key_id=api_key_id,
        private_key_path=str(key_path),
    )


def _product_from_dict(data: Dict[str, Any]) -> ProductConfig:
    return ProductConfig(
        enabled=bool(data.get("enabled", False)),
        series_tickers=[str(value).upper() for value in data.get("series_tickers", [])],
        ticker_prefixes=[
            str(value).upper() for value in data.get("ticker_prefixes", [])
        ],
        title_keywords=[str(value).lower() for value in data.get("title_keywords", [])],
        pyth_symbol=str(data.get("pyth_symbol", "")),
        pyth_feed_id=str(data.get("pyth_feed_id", "")),
        pyth_lazer_id=(
            int(data["pyth_lazer_id"])
            if data.get("pyth_lazer_id") is not None
            else None
        ),
        cme_symbols=[str(value).upper() for value in data.get("cme_symbols", [])],
    )


def load_metals_config(path: str) -> MetalsConfig:
    raw = _load_toml(Path(path))

    storage_raw = raw.get("storage", {})
    kalshi_raw = raw.get("kalshi", {})
    pyth_raw = raw.get("pyth", {})
    cme_raw = raw.get("cme", {})
    clock_raw = raw.get("clock", {})
    governance_raw = raw.get("governance", {})
    shadow_raw = raw.get("shadow", {})

    products = {
        name.lower(): _product_from_dict(value)
        for name, value in raw.get("products", {}).items()
    }
    if not products:
        raise ValueError("At least one [products.<metal>] section is required")
    if not any(product.enabled for product in products.values()):
        raise ValueError("At least one metals product must be enabled")

    shadow = ShadowConfig(
        enabled=bool(shadow_raw.get("enabled", True)),
        execution_mode=str(shadow_raw.get("execution_mode", "shadow")).lower(),
        allow_live_orders=bool(shadow_raw.get("allow_live_orders", False)),
        config_version=str(shadow_raw.get("config_version", "metals-shadow-v1")),
        bankroll_min_dollars=str(shadow_raw.get("bankroll_min_dollars", "5000")),
        bankroll_max_dollars=str(shadow_raw.get("bankroll_max_dollars", "10000")),
        max_contracts_per_order=str(shadow_raw.get("max_contracts_per_order", "250")),
        max_net_contracts_per_market=str(
            shadow_raw.get("max_net_contracts_per_market", "1000")
        ),
        max_aggregate_worst_case_loss_dollars=str(
            shadow_raw.get("max_aggregate_worst_case_loss_dollars", "3000")
        ),
        max_loss_per_market_dollars=str(
            shadow_raw.get("max_loss_per_market_dollars", "1000")
        ),
        daily_loss_limit_dollars=str(
            shadow_raw.get("daily_loss_limit_dollars", "3000")
        ),
    )
    if shadow.execution_mode != "shadow":
        raise ValueError("Metals execution_mode must remain 'shadow'")
    if shadow.allow_live_orders:
        raise ValueError("Metals shadow configuration cannot allow live orders")

    clock = ClockConfig(
        enabled=bool(clock_raw.get("enabled", False)),
        provider=str(clock_raw.get("provider", "chrony")).strip().lower(),
        chronyc_path=str(clock_raw.get("chronyc_path", "chronyc")).strip(),
        sample_interval_seconds=float(clock_raw.get("sample_interval_seconds", 1.0)),
        probe_timeout_seconds=float(clock_raw.get("probe_timeout_seconds", 2.0)),
        max_probe_elapsed_delta_ns=int(
            clock_raw.get("max_probe_elapsed_delta_ns", 1_000_000)
        ),
    )
    if clock.provider != "chrony":
        raise ValueError("Metals clock.provider must be 'chrony'")
    if not clock.chronyc_path or "\x00" in clock.chronyc_path:
        raise ValueError("Metals clock.chronyc_path must be non-empty")
    if (
        not math.isfinite(clock.sample_interval_seconds)
        or clock.sample_interval_seconds <= 0
    ):
        raise ValueError("Metals clock sample interval must be positive and finite")
    if (
        not math.isfinite(clock.probe_timeout_seconds)
        or not 0 < clock.probe_timeout_seconds <= 30
    ):
        raise ValueError("Metals clock probe timeout must be in (0, 30] seconds")
    if clock.max_probe_elapsed_delta_ns < 0:
        raise ValueError("Metals clock elapsed-delta limit cannot be negative")

    governance = GovernanceConfig(
        provider_approval_path=str(
            governance_raw.get("provider_approval_path", "")
        ).strip(),
        pyth_historical_research_assumption_path=str(
            governance_raw.get("pyth_historical_research_assumption_path", "")
        ).strip(),
        databento_historical_research_assumption_path=str(
            governance_raw.get("databento_historical_research_assumption_path", "")
        ).strip(),
    )
    if any("\x00" in value for value in asdict(governance).values()):
        raise ValueError("Metals governance artifact path is invalid")

    return MetalsConfig(
        environment=str(raw.get("environment", "production")),
        storage=StorageConfig(
            root=str(storage_raw.get("root", "data/metals")),
            fsync_every_records=int(storage_raw.get("fsync_every_records", 1)),
        ),
        kalshi=KalshiConfig(
            rest_url=str(kalshi_raw.get("rest_url", PRODUCTION_REST_URL)).rstrip("/"),
            ws_url=str(kalshi_raw.get("ws_url", PRODUCTION_WS_URL)),
            credentials_config=str(kalshi_raw.get("credentials_config", "config.toml")),
            api_key_id_env=str(kalshi_raw.get("api_key_id_env", "KALSHI_API_KEY_ID")),
            private_key_path_env=str(
                kalshi_raw.get("private_key_path_env", "KALSHI_PRIVATE_KEY_PATH")
            ),
            page_limit=int(kalshi_raw.get("page_limit", 1000)),
            discovery_interval_seconds=float(
                kalshi_raw.get("discovery_interval_seconds", 15.0)
            ),
            request_timeout_seconds=float(
                kalshi_raw.get("request_timeout_seconds", 15.0)
            ),
            fractional_contracts_enabled=bool(
                kalshi_raw.get("fractional_contracts_enabled", True)
            ),
            fractional_contract_quantum=str(
                kalshi_raw.get("fractional_contract_quantum", "0.01")
            ),
            upcoming_markets_per_product=int(
                kalshi_raw.get("upcoming_markets_per_product", 4)
            ),
            max_read_requests_per_second=float(
                kalshi_raw.get("max_read_requests_per_second", 18.0)
            ),
            max_read_retries=int(kalshi_raw.get("max_read_retries", 5)),
            read_retry_base_seconds=float(
                kalshi_raw.get("read_retry_base_seconds", 0.25)
            ),
        ),
        pyth=PythConfig(
            enabled=bool(pyth_raw.get("enabled", False)),
            api_key_env=str(pyth_raw.get("api_key_env", "PYTH_API_KEY")),
            endpoint=str(
                pyth_raw.get("endpoint", "https://pyth.dourolabs.app/v1")
            ).rstrip("/"),
            lazer_ws_endpoints=[
                str(value)
                for value in pyth_raw.get(
                    "lazer_ws_endpoints",
                    [
                        "wss://pyth-lazer-0.dourolabs.app/v1/stream",
                        "wss://pyth-lazer-1.dourolabs.app/v1/stream",
                        "wss://pyth-lazer-2.dourolabs.app/v1/stream",
                    ],
                )
            ],
            history_chunk_seconds=int(pyth_raw.get("history_chunk_seconds", 86_400)),
        ),
        cme=CmeConfig(
            enabled=bool(cme_raw.get("enabled", False)),
            provider=str(cme_raw.get("provider", "databento")),
            api_key_env=str(cme_raw.get("api_key_env", "DATABENTO_API_KEY")),
            dataset=str(cme_raw.get("dataset", "GLBX.MDP3")),
            stype_in=str(cme_raw.get("stype_in", "parent")),
            schemas=[
                str(value)
                for value in cme_raw.get(
                    "schemas",
                    ["mbo", "trades", "definition", "status", "statistics"],
                )
            ],
            mbo_snapshot=bool(cme_raw.get("mbo_snapshot", True)),
            reconnect_policy=str(cme_raw.get("reconnect_policy", "reconnect")),
            include_ts_out=bool(cme_raw.get("include_ts_out", True)),
        ),
        clock=clock,
        governance=governance,
        shadow=shadow,
        products=products,
    )
