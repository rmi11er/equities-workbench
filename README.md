# equities-workbench

A collection of quantitative trading and research tooling, built mostly with Claude Code as a copilot.

| Directory | What it is |
|---|---|
| [`kalshi-mm/`](kalshi-mm/) | Async market maker for Kalshi event contracts (Avellaneda-Stoikov with inventory skew), RFQ responder, and a read-only metals research/shadow-trading harness. |
| [`workbench/event-researcher/`](workbench/event-researcher/) | Agent-augmented research environment for event-driven trading analysis. |
| [`infra/`](infra/) | Shared infrastructure and sandboxed code-execution experiments. |

Each subproject is self-contained and uses [uv](https://docs.astral.sh/uv/):

```bash
cd kalshi-mm && uv sync --all-extras && uv run pytest
```

Credentials are never committed: copy the `*.example.*` config/env files and fill in your own values.

> Research code, not investment advice. The market maker defaults to demo/read-only modes.

Licensed under MIT.
