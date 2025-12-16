# Event Researcher

Agent-augmented research environment for event-driven trading analysis.

## Quick Start

```bash
# Install dependencies
uv sync --dev

# Initialize database
uv run python scripts/init_db.py

# Seed watchlist from config
uv run python scripts/seed_watchlist.py

# Backfill historical prices
uv run python scripts/backfill_prices.py

# Check status
uv run researcher status
```

## CLI Commands

```bash
# Show status
uv run researcher status

# View watchlist
uv run researcher watchlist

# View upcoming events
uv run researcher events --days 14

# View price history
uv run researcher prices NVDA --days 30

# View earnings history
uv run researcher earnings NVDA
```

## Data Refresh

```bash
# Refresh prices and earnings calendar
uv run python scripts/refresh_data.py

# Refresh prices only
uv run python scripts/refresh_data.py --prices-only

# Refresh earnings calendar only
uv run python scripts/refresh_data.py --earnings-only
```

## Configuration

- `config/settings.toml` - Application settings
- `config/watchlist.toml` - Tracked symbols
- `config/filters.toml` - Event surfacing filters

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
ANTHROPIC_API_KEY=sk-ant-...
FMP_API_KEY=...
```

## Running Tests

```bash
uv run pytest
```
