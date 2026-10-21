# Historical exploratory-assumption gate

This gate records an operator's temporary, documented assumptions for exploratory
historical research. It is intentionally not a provider license or production-access
decision.

## Scope

One artifact can authorize exactly one of these operations:

- `pyth` / `exact_history`
- `databento` / `historical`

It cannot authorize Pyth Lazer, Databento live data, synchronized collection, shadow
or live execution, settlement certification, or a passing daily-readiness result.
Those paths continue to require the existing content-addressed provider-approval
artifact and, where applicable, clock evidence. Real orders remain prohibited by the
metals configuration.

The historical collector uses the production provider evidence when
`[governance].provider_approval_path` is configured. It never falls back from a bad,
expired, or insufficient configured production artifact to a research assumption.
Only when that path is empty can the matching operation-specific assumption be used.

Use a separate `[storage].root` (for example `data/metals-exploratory`) for every run
made under this gate. Exact-history archives are resumable and content-addressed; an
archive first acquired under a research assumption must never be re-labeled later by
running a production-approved wrapper over the same checkpoint. The settlement audit
therefore fails closed on assumption-owned exact-history runs in a production root.
Production settlement acquisition and daily readiness must use a clean production
storage root containing only provider-approved acquisition provenance.

## Create the reviewed input

Copy `configs/historical_research_assumption.example.json` to an ignored local file.
For Databento, change the exact pair to `"provider": "databento"` and
`"operation": "historical"`.

Before setting the account preflight status to `passed`:

1. Run the provider's bounded historical preflight and save its secret-safe JSON.
2. Save or export each primary technical document used for the assumptions.
3. Compute SHA-256 hashes for those evidence files.
4. State every unresolved license, entitlement, timestamp, correction, retention, and
   fitness-for-purpose limitation in `known_limitations`.
5. Set a short, explicit `valid_until`; review and re-materialize a new version after
   expiry or any semantic change.

The schema requires credential-free HTTPS source URLs, evidence hashes, a reviewed
technical-semantics version, the account preflight status/time/hash, all three safety
flags set exactly to `true`, operator identity/version/time, an expiration, and at
least one known limitation. `failed` and `not_checked` preflight states can be frozen
for auditability but cannot authorize a download.

## Materialize and check offline

```bash
uv run --locked python -m src.metals.cli --config configs/metals.toml \
  materialize-historical-research-assumption \
  --assumption reviewed-pyth-history-assumption.json
```

Copy the returned `assumption_path` into exactly one matching field:

```toml
[governance]
provider_approval_path = ""
pyth_historical_research_assumption_path = "data/metals-exploratory/policies/historical_research_assumption/assumption=<sha256>/assumption.json"
databento_historical_research_assumption_path = ""
```

Then verify without network access:

```bash
uv run --locked python -m src.metals.cli --config configs/metals.toml \
  check-historical-research-assumption \
  --provider pyth --operation exact_history
```

Success reports `research_only`, `no_live`, and `no_orders` as `true`. This command
does not prove provider access. The bounded historical preflight must already have
passed and its output hash must be frozen in the artifact.

## Download controls remain separate

The Pyth exact-history command still requires its explicit bounded request. Bulk Pyth
settlement history still requires `--allow-network-download`. Databento still requires
a fresh cost estimate, `--max-cost-usd`, and `--allow-billable-download` for every new
retrieval. Creating or checking this artifact never makes a network request and never
authorizes spend.

Runs made under this gate are marked with
`historical_research_assumption_id`, `research_only=true`, `no_live=true`, and
`no_orders=true`. They intentionally omit `provider_approval_id`, so production
settlement provenance and daily-readiness verification reject them.
