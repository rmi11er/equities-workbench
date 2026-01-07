# The Moat - Development Log

## Project Overview
**Name:** The Moat - Quantitative Research & Execution Infrastructure
**Started:** 2026-01-07
**Status:** MVP Complete (All 4 Phases Implemented)

## Quick Reference for Future Sessions

### Project Purpose
High-performance, modular, vendor-agnostic infrastructure for quantitative finance strategy validation.

### Key Design Decisions
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Project Layout | `src/moat/` | Standard Python packaging |
| Dependency Management | `uv` | Fast, modern Python tooling |
| Data Validation | Pydantic | Strict data contracts |
| Local Storage | Parquet | Fast read/write, columnar |
| Config | `.env` (secrets) + `config.yaml` (paths) | Separation of concerns |
| Testing | pytest + hypothesis, TDD-style | Written alongside modules |

### Architecture (3 Layers)
```
[Clients: Agents / Humans]
         ↓
Layer 3: Interface API (StrategyRunner, Sandbox)
         ↓
Layer 2: Core Engines (VectorizedBacktester, RobustnessTestSuite)
         ↓
Layer 1: Data Abstraction (DataProvider → YFinanceAdapter, LocalIngestionEngine)
         ↓
[Cloud APIs] [Local Data Folder]
```

### Directory Structure (Actual)
```
infra/
├── src/moat/
│   ├── __init__.py
│   ├── schemas.py              # Pydantic OHLCV models
│   ├── stats.py                # Performance metrics (Sharpe, Sortino, etc.)
│   ├── stress_test.py          # Robustness testing suite
│   ├── data/
│   │   ├── __init__.py
│   │   ├── provider.py         # DataProvider ABC
│   │   ├── manager.py          # DataManager (unified interface)
│   │   ├── yfinance_adapter.py # YFinance API adapter
│   │   └── local_ingestor.py   # CSV → Parquet pipeline
│   ├── engines/
│   │   ├── __init__.py
│   │   └── vector_engine.py    # Vectorized backtester
│   └── sandbox/
│       ├── __init__.py
│       ├── sandbox.py          # Restricted execution environment
│       └── dsl_wrapper.py      # Safe financial primitives
├── tests/
│   ├── test_data_layer.py      # 4 tests
│   ├── test_backtest_engine.py # 10 tests
│   ├── test_stress_test.py     # 8 tests
│   └── test_sandbox.py         # 19 tests
├── config/
│   ├── config.yaml             # Application config
│   └── schema_map.yaml         # Column name mappings
├── data/
│   ├── incoming/               # Drop CSVs here
│   └── processed/              # Parquet storage
├── pyproject.toml
├── .env.example
└── DEVELOPMENT_LOG.md
```

---

## Development Phases

### Phase 1: Skeleton & Hybrid Data Layer ✅
**Goal:** Define data contracts and enable data flow from both cloud and local sources.

**Deliverables:**
- [x] `schemas.py` - OHLCV Pydantic models
- [x] `data/provider.py` - DataProvider ABC
- [x] `data/yfinance_adapter.py` - Cloud data fetching
- [x] `data/local_ingestor.py` - Local CSV → Parquet pipeline
- [x] `config/schema_map.yaml` - Column name translations

**Definition of Done:** ✅ Can drop a CSV into `data/incoming/`, run `data.get('symbol')`, and receive a clean DataFrame.

### Phase 2: Vectorized Engine ✅
**Goal:** Enable rapid hypothesis testing with matrix operations.

**Deliverables:**
- [x] `engines/vector_engine.py` - Vectorized backtester with indicators
- [x] `stats.py` - Sharpe, Sortino, Drawdown, Calmar, and more

**Definition of Done:** ✅ 5-line MA strategy returns performance report in <0.5s (tested: 0.01s).

### Phase 3: Robustness Suite ✅
**Goal:** Stress-test strategies for overfitting detection.

**Deliverables:**
- [x] `stress_test.py` - Monte Carlo, noise injection, parameter scan

**Definition of Done:** ✅ Generates "Robustness Score" (0-100) for any strategy.

### Phase 4: Agent Sandbox ✅
**Goal:** Safe execution of untrusted strategy code.

**Deliverables:**
- [x] `sandbox/sandbox.py` - Restricted execution environment
- [x] `sandbox/dsl_wrapper.py` - Allowed primitives only

**Definition of Done:** ✅ `sma(close, 50) > close` executes; `import sys` fails.

---

## Session Log

### 2026-01-07 - Session 1: Full MVP Implementation
**Context:** Starting from empty directory with project specification (.docx)

**Decisions Made:**
1. Using `uv` for dependency management
2. Implementing both YFinance + local ingestion in Phase 1
3. Vectorized backtester only (event-driven deferred)
4. Using AAPL YFinance data as initial test fixture
5. TDD-style testing alongside implementation
6. Config: `.env` for secrets, `config.yaml` for paths

**Work Completed:**
- [x] Project structure initialized
- [x] Phase 1 implementation (Data Layer)
- [x] Phase 2 implementation (Vectorized Engine)
- [x] Phase 3 implementation (Robustness Suite)
- [x] Phase 4 implementation (Agent Sandbox)

**Test Results:** 41 tests passing

---

## Resumption Instructions for AI Agents

If you are an AI agent resuming work on this project:

1. **Read this file first** to understand context and current state
2. **All 4 phases are complete** - the MVP is functional
3. **Run `uv sync`** to ensure environment is ready
4. **Run `uv run pytest`** to verify current state (expect 41 passing)

### Key Files to Understand the Codebase
- `src/moat/schemas.py` - Data contracts (start here)
- `src/moat/data/manager.py` - Unified data interface
- `src/moat/engines/vector_engine.py` - Backtesting core
- `src/moat/sandbox/sandbox.py` - Safe execution
- `tests/` - Examples of expected behavior

### Common Commands
```bash
uv sync                    # Install dependencies
uv run pytest              # Run all tests (41 expected)
uv run pytest -v           # Verbose test output

# Quick data test
uv run python -c "
from moat.data import DataManager
dm = DataManager()
print(dm.get('AAPL').tail())
"

# Quick backtest test
uv run python -c "
from moat.data import DataManager
from moat.engines import VectorizedBacktester
from moat.engines.vector_engine import sma

dm = DataManager()
data = dm.get('AAPL')

def ma_strategy(df):
    return (sma(df['close'], 20) > sma(df['close'], 50)).astype(float)

bt = VectorizedBacktester()
result = bt.run(data, ma_strategy, 'MA_20_50', 'AAPL')
print(f'Sharpe: {result.sharpe_ratio:.2f}')
print(f'Return: {result.total_return:.2%}')
"

# Quick sandbox test
uv run python -c "
from moat.sandbox import StrategyRunner
from moat.data import DataManager

dm = DataManager()
data = dm.get('AAPL')
runner = StrategyRunner()

code = '''
result = (sma(close, 20) > sma(close, 50)).astype(float)
'''
signals = runner.run(code, data)
print(f'Generated {len(signals)} signals')
"
```

---

## Future Development Ideas

### Deferred from MVP
- Event-driven backtester (tick-by-tick simulation)
- File watcher for automatic ingestion (watchdog configured but not integrated)

### Potential Enhancements
- More data providers (Polygon, Alpha Vantage, etc.)
- Strategy templating system
- Performance visualization (equity curves, drawdown charts)
- Portfolio-level backtesting (multiple symbols)
- Walk-forward optimization
- Machine learning integration hooks
