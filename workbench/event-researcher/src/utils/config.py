"""Configuration loading from TOML files."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import tomllib
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_toml(filename: str) -> dict[str, Any]:
    """Load a TOML file from the config directory."""
    config_path = _get_project_root() / "config" / filename
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "rb") as f:
        return tomllib.load(f)


class AppConfig(BaseModel):
    """Application configuration."""

    name: str = "Event Researcher"
    data_dir: str = "data"
    database_name: str = "researcher.duckdb"


class AgentConfig(BaseModel):
    """Agent configuration."""

    default_model: str = "claude-sonnet-4-20250514"
    complex_model: str = "claude-opus-4-5-20250514"
    max_tokens: int = 4096
    temperature: float = 0.3


class DataConfig(BaseModel):
    """Data configuration."""

    price_history_years: int = 5
    hourly_cache_ttl_minutes: int = 60
    transcript_source_priority: list[str] = Field(default_factory=lambda: ["seeking_alpha", "fmp"])


class UIConfig(BaseModel):
    """UI configuration."""

    theme: str = "dark"
    monitor_refresh_seconds: int = 300


class WatchlistConfig(BaseModel):
    """Watchlist configuration."""

    auto_update: bool = False
    symbols: list[str] = Field(default_factory=list)


class TemporalFilters(BaseModel):
    """Temporal filter configuration."""

    lookahead_days: int = 7
    lookback_regime_days: int = 30


class EventTypeWeights(BaseModel):
    """Event type priority weights."""

    earnings: int = 100
    conference: int = 80
    analyst_day: int = 70
    macro: int = 60


class FlagConfig(BaseModel):
    """Interest flag configuration."""

    momentum_enabled: bool = True
    momentum_threshold_pct: float = 30
    momentum_window_days: int = 90
    sector_relative_enabled: bool = True
    sector_relative_threshold_pct: float = 10
    sector_relative_window_days: int = 30
    vix_enabled: bool = True
    vix_high: float = 25
    vix_low: float = 15
    short_interest_enabled: bool = False
    short_interest_percentile: int = 80
    streak_enabled: bool = True
    streak_threshold: int = 3
    leadership_change_enabled: bool = True
    leadership_change_lookback_days: int = 180


class ThresholdConfig(BaseModel):
    """Threshold configuration."""

    high_interest_min_flags: int = 3
    standard_min_flags: int = 1


class FiltersConfig(BaseModel):
    """Event surfacing filters configuration."""

    temporal: TemporalFilters = Field(default_factory=TemporalFilters)
    event_types: EventTypeWeights = Field(default_factory=EventTypeWeights)
    flags: FlagConfig = Field(default_factory=FlagConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)


class Settings(BaseSettings):
    """Main settings class combining environment variables and config files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment variables
    anthropic_api_key: str = ""
    fmp_api_key: str = ""
    log_level: str = "INFO"

    # Loaded from TOML
    app: AppConfig = Field(default_factory=AppConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    watchlist: WatchlistConfig = Field(default_factory=WatchlistConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_toml_configs()

    def _load_toml_configs(self) -> None:
        """Load configuration from TOML files."""
        try:
            settings_data = load_toml("settings.toml")
            if "app" in settings_data:
                self.app = AppConfig(**settings_data["app"])
            if "agent" in settings_data:
                self.agent = AgentConfig(**settings_data["agent"])
            if "data" in settings_data:
                self.data = DataConfig(**settings_data["data"])
            if "ui" in settings_data:
                self.ui = UIConfig(**settings_data["ui"])
        except FileNotFoundError:
            pass

        try:
            watchlist_data = load_toml("watchlist.toml")
            settings = watchlist_data.get("settings", {})
            equities = watchlist_data.get("equities", {})
            self.watchlist = WatchlistConfig(
                auto_update=settings.get("auto_update", False),
                symbols=equities.get("symbols", []),
            )
        except FileNotFoundError:
            pass

        try:
            filters_data = load_toml("filters.toml")
            self.filters = FiltersConfig(
                temporal=TemporalFilters(**filters_data.get("temporal", {})),
                event_types=EventTypeWeights(**filters_data.get("event_types", {})),
                flags=FlagConfig(**filters_data.get("flags", {})),
                thresholds=ThresholdConfig(**filters_data.get("thresholds", {})),
            )
        except FileNotFoundError:
            pass

    @property
    def database_path(self) -> Path:
        """Get the full path to the database file."""
        return _get_project_root() / self.app.data_dir / self.app.database_name


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
