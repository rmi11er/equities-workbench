import asyncio
import json
from dataclasses import replace
from datetime import datetime
from decimal import Decimal

import pytest

from src.metals.pyth_exact_history import (
    PYTH_EXACT_HISTORY_CHANNEL,
    PythExactHistoryRequest,
    run_pyth_exact_history,
)
from src.metals.settlement_audit import (
    SEMANTICS_SCHEMA_VERSION,
    UNIVERSE_SCHEMA_VERSION,
    SemanticsConfirmation,
    SettlementAuditError,
    SettlementAuditSemantics,
    SettlementUniverse,
    run_local_settlement_audit,
    verify_settlement_audit,
)
from src.metals.storage import HashChainedJsonlWriter
from tests.metals.historical_research_assumption_fixtures import (
    materialize_historical_assumption_path,
)
from tests.metals.provider_approval_fixtures import (
    materialize_approved_provider_path,
)

EVIDENCE = "a" * 64
FEED_ID = 3_154
RULES_SECONDARY = (
    "The market is determined using the Pyth Silver price, rounded to the nearest "
    "three decimal places. The price at a time is the close at the end of the "
    "immediately preceding minute. If no data is available, the most recently "
    "published price is used."
)
RULES_PRIMARY = "The closing Pyth Silver price is at least its opening price."


def _ns(value):
    return int(datetime.fromisoformat(value).timestamp() * 1_000_000_000)


def _semantics(confirmed=True):
    confirmation = SemanticsConfirmation(
        version="provider-rule-2026-08-29",
        confirmed=confirmed,
        evidence_sha256=(EVIDENCE if confirmed else None),
    )
    return SettlementAuditSemantics(
        timestamp=confirmation,
        correction=confirmation,
        rounding=confirmation,
        timestamp_field="publish_timestamp_us",
        timestamp_meaning="last_publication_strictly_before_boundary",
        timestamp_source_label=("publish_time_and_feed_update_timestamp_microseconds"),
        correction_policy="last_publication_before_boundary_v1",
        rounding_modes_by_series=(("KXSILVER15M", "half_up"),),
    )


def _universe(*tickers):
    return SettlementUniverse(
        version="kalshi-completed-markets-2026-08-29",
        confirmed=True,
        evidence_sha256="b" * 64,
        completed_market_tickers=tuple(sorted(tickers)),
    )


def _write_stream(root, stream, rows, suffix="one"):
    path = (
        root
        / "normalized"
        / stream
        / "date=2026-11-01"
        / "metal=silver"
        / "run=test"
        / f"{stream}-{suffix}.jsonl"
    )
    with HashChainedJsonlWriter(path) as writer:
        for row in rows:
            writer.write(row)
    return path


class _Response:
    status = 200

    def __init__(self, body):
        self.body = body

    async def read(self):
        return self.body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _Session:
    def __init__(self, body):
        self.body = body
        self.calls = 0

    def get(self, url, *, params, headers):
        self.calls += 1
        return _Response(self.body)


def _exact_price(price, publish_us, *, feed_update_us=None):
    if feed_update_us is None:
        feed_update_us = publish_us
    return {
        "channel": 3,
        "channel_name": PYTH_EXACT_HISTORY_CHANNEL,
        "price_feed_id": FEED_ID,
        "publish_time": publish_us,
        "price": int(Decimal(price) * 10_000),
        "exponent": -4,
        "feed_update_timestamp": feed_update_us,
        "publisher_count": 7,
        "confidence": None,
        "best_bid_price": None,
        "best_ask_price": None,
        "ema_price": None,
        "ema_confidence": None,
        "funding_rate": None,
        "funding_timestamp": None,
        "funding_rate_interval": None,
        "market_session": "Regular",
    }


def _write_exact_range(
    root,
    boundary_ns,
    rows,
    *,
    tag,
    start_offset_us=60_000_000,
    end_offset_us=0,
    historical_research_assumption_path=None,
):
    boundary_us = boundary_ns // 1_000
    request = PythExactHistoryRequest(
        metal="silver",
        feed_ids=(FEED_ID,),
        start_timestamp_us=boundary_us - start_offset_us,
        end_timestamp_us=boundary_us - end_offset_us,
        base_url=f"https://history.example/{tag}",
    )
    body = json.dumps({"data": rows}, separators=(",", ":")).encode()
    session = _Session(body)
    governance = (
        {"historical_research_assumption_path": (historical_research_assumption_path)}
        if historical_research_assumption_path is not None
        else {"provider_approval_path": materialize_approved_provider_path(root)}
    )
    run = asyncio.run(
        run_pyth_exact_history(
            root,
            request,
            api_key_env="TEST_PYTH_KEY",
            environ={"TEST_PYTH_KEY": "synthetic"},
            session=session,
            **governance,
        )
    )
    result = run.archive
    assert result.completed is True
    assert session.calls == 1
    return result


def _fixture(
    root, *, final_close="66.5156", exchange_final="66.516", include_final=True
):
    ticker = "KXSILVER15M-TEST"
    # Both UTC boundaries remain exact through the US daylight-saving fallback.
    open_ns = _ns("2026-11-01T06:00:00+00:00")
    close_ns = _ns("2026-11-01T06:15:00+00:00")
    _write_stream(
        root,
        "market_dim",
        [
            {
                "market_ticker": ticker,
                "event_ticker": "KXSILVER15M-EVENT",
                "series_ticker": "KXSILVER15M",
                "metal": "silver",
                "open_ts_ns": open_ns,
                "close_ts_ns": close_ns,
                "target_price": "66.516",
                "round_digits": 3,
                "strike_type": "greater_or_equal",
                "settlement_source_name": "Pyth - Silver",
                "settlement_source_url": "https://example.test/pyth-silver",
                "rules_primary": RULES_PRIMARY,
                "rules_secondary": RULES_SECONDARY,
                "rules_secondary_source_id": "c" * 64,
                "pyth_symbol": "Metal.Index.SILVER/USD",
                "pyth_feed_id": str(FEED_ID),
                "metadata_receive_ts_ns": close_ns + 1,
                "metadata_version": 1,
            }
        ],
    )
    _write_stream(
        root,
        "kalshi_settlement",
        [
            {
                "market_ticker": ticker,
                "event_ticker": "KXSILVER15M-EVENT",
                "series_ticker": "KXSILVER15M",
                "expiration_value": exchange_final,
                "result": "yes",
                "status": "finalized",
                "settlement_ts_ns": close_ns + 2,
                "receive_ts_ns": close_ns + 3,
            }
        ],
    )
    open_us = open_ns // 1_000
    close_us = close_ns // 1_000
    _write_exact_range(
        root,
        open_ns,
        [_exact_price("66.5155", open_us - 200_000)],
        tag="opening",
    )
    if include_final:
        _write_exact_range(
            root,
            close_ns,
            [_exact_price(final_close, close_us - 200_000)],
            tag="settlement",
        )
    return ticker, open_ns, close_ns


def _details(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_exact_audit_freezes_full_rows_but_needs_sample_or_complete_universe(tmp_path):
    ticker, _, _ = _fixture(tmp_path)
    first = run_local_settlement_audit(tmp_path, _semantics())
    second = run_local_settlement_audit(tmp_path, _semantics())

    assert first.audit_id == second.audit_id
    assert first.directory == second.directory
    assert first.summary["usable_windows"] == 1
    assert first.summary["exact_target_agreements"] == 1
    assert first.summary["exact_settlement_value_agreements"] == 1
    assert first.summary["result_agreements"] == 1
    assert first.summary["usable_window_agreement_gate_passed"] is True
    assert first.summary["minimum_1000_or_all_available_met"] is False
    assert first.summary["settlement_research_gate_passed"] is False
    assert first.summary["expected_count"] == 1
    assert first.summary["usable_count"] == 1
    assert first.summary["excluded_count"] == 0
    assert first.summary["missing_count"] == 0
    assert first.summary["conflicting_count"] == 0

    row = _details(first.details_path)[0]
    assert row["market_ticker"] == ticker
    assert row["reconstructed_opening_target"] == "66.516"
    assert row["reconstructed_target"] == "66.516"
    assert row["reconstructed_final_value"] == "66.516"
    assert row["tie_flag"] is True
    assert row["predicted_result"] == "yes"
    assert row["agreement"] is True
    assert row["usable"] is True
    assert len(row["opening_pyth_source_id"]) == 64
    assert len(row["settlement_pyth_source_id"]) == 64
    assert row["opening_source_update_id"] == row["opening_pyth_source_id"]
    assert row["settlement_source_update_id"] == row["settlement_pyth_source_id"]
    assert row["stale_source_fallback"] is False
    assert row["source_age_ns"] == 200_000_000
    assert row["exclusion_reason"] is None
    assert row["rules_secondary"] == RULES_SECONDARY
    assert row["rules_secondary_source_id"] == "c" * 64
    assert (
        verify_settlement_audit(tmp_path, first.manifest_path)["summary"]
        == first.summary
    )


def test_explicit_exact_feed_id_does_not_depend_on_exchange_metadata_field(tmp_path):
    ticker, _, _ = _fixture(tmp_path)
    market_path = next((tmp_path / "normalized" / "market_dim").glob("**/*.jsonl"))
    market = _details(market_path)[0]
    market.pop("record_sha256")
    market.pop("previous_record_sha256")
    market["pyth_feed_id"] = None
    market_path.unlink()
    with HashChainedJsonlWriter(market_path) as writer:
        writer.write(market)

    artifact = run_local_settlement_audit(
        tmp_path,
        _semantics(),
        universe=_universe(ticker),
        exact_feed_id=FEED_ID,
    )
    row = _details(artifact.details_path)[0]

    assert row["usable"] is True
    assert row["market_pyth_feed_id"] is None
    assert row["pyth_feed_id"] == FEED_ID
    assert json.loads(artifact.manifest_path.read_text())["exact_feed_id"] == FEED_ID


def test_exact_archive_without_completed_approved_run_is_ineligible(tmp_path):
    _fixture(tmp_path)
    for manifest_path in (tmp_path / "manifests").glob("run=*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config", {}).get("collector", {}).get("component") == (
            "pyth_exact_history"
        ):
            manifest_path.unlink()

    with pytest.raises(SettlementAuditError, match="not bound to a completed approved"):
        run_local_settlement_audit(tmp_path, _semantics())


def test_research_assumption_exact_archive_cannot_satisfy_settlement_audit(tmp_path):
    _, _, close_ns = _fixture(tmp_path, include_final=False)
    close_us = close_ns // 1_000
    assumption_path = materialize_historical_assumption_path(tmp_path)
    _write_exact_range(
        tmp_path,
        close_ns,
        [_exact_price("66.5156", close_us - 200_000)],
        tag="research-only-settlement",
        historical_research_assumption_path=assumption_path,
    )

    with pytest.raises(SettlementAuditError, match="provider approval"):
        run_local_settlement_audit(tmp_path, _semantics())


def test_missing_rule_text_is_a_classified_exclusion(tmp_path):
    ticker, _, _ = _fixture(tmp_path)
    market_path = next((tmp_path / "normalized" / "market_dim").glob("**/*.jsonl"))
    market = _details(market_path)[0]
    market.pop("record_sha256")
    market.pop("previous_record_sha256")
    market["rules_secondary"] = None
    market_path.unlink()
    with HashChainedJsonlWriter(market_path) as writer:
        writer.write(market)

    artifact = run_local_settlement_audit(
        tmp_path, _semantics(), universe=_universe(ticker)
    )
    row = _details(artifact.details_path)[0]

    assert row["usable"] is False
    assert "missing_rules_secondary" in row["exclusion_reasons"]
    assert artifact.summary["excluded_count"] == 1


def test_confirmed_complete_universe_satisfies_fewer_than_1000_rule(tmp_path):
    ticker, _, _ = _fixture(tmp_path)
    artifact = run_local_settlement_audit(
        tmp_path, _semantics(), universe=_universe(ticker)
    )

    assert artifact.summary["universe_complete_and_confirmed"] is True
    assert artifact.summary["all_confirmed_universe_windows_usable"] is True
    assert artifact.summary["minimum_1000_or_all_available_met"] is True
    assert artifact.summary["settlement_research_gate_passed"] is True


@pytest.mark.parametrize("field", ["timestamp", "correction", "rounding"])
def test_unconfirmed_semantics_can_never_make_a_window_usable(tmp_path, field):
    ticker, _, _ = _fixture(tmp_path)
    semantics = _semantics()
    semantics = replace(
        semantics,
        **{field: SemanticsConfirmation(version="still-unconfirmed", confirmed=False)},
    )
    artifact = run_local_settlement_audit(
        tmp_path, semantics, universe=_universe(ticker)
    )
    row = _details(artifact.details_path)[0]

    assert row["usable"] is False
    assert f"{field}_semantics_unconfirmed" in row["exclusion_reasons"]
    assert artifact.summary["usable_windows"] == 0
    assert artifact.summary["settlement_research_gate_passed"] is False


def test_missing_exact_boundary_is_excluded_without_nearest_or_stale_guess(tmp_path):
    ticker, _, close_ns = _fixture(tmp_path, include_final=False)
    # TradingView candles are not settlement evidence, even at the boundary.
    _write_stream(
        tmp_path,
        "pyth_candle",
        [
            {
                "pyth_symbol": "Metal.Index.SILVER/USD",
                "timestamp_seconds": close_ns // 1_000_000_000 - 60,
                "resolution_minutes": 1,
                "timestamp_semantics": "unconfirmed_tradingview_t",
                "close": "66.516",
                "receive_ts_ns": close_ns + 99,
            }
        ],
        suffix="nearby",
    )
    artifact = run_local_settlement_audit(
        tmp_path, _semantics(), universe=_universe(ticker)
    )
    row = _details(artifact.details_path)[0]

    assert row["settlement_pyth_source_id"] is None
    assert row["stale_source_fallback_flag"] is False
    assert "missing_settlement_exact_history_coverage" in row["exclusion_reasons"]
    assert row["exclusion_reason"] == "missing_settlement_exact_history_coverage"
    assert row["usable"] is False
    assert artifact.summary["usable_count"] == 0
    assert artifact.summary["excluded_count"] == 1
    assert artifact.summary["missing_count"] == 1
    manifest = json.loads(artifact.manifest_path.read_text())
    assert all(item["stream"] != "pyth_candle" for item in manifest["inputs"])


def test_last_publication_strictly_before_boundary_is_deterministic(tmp_path):
    ticker, _, close_ns = _fixture(
        tmp_path, final_close="66.5000", exchange_final="66.516"
    )
    close_us = close_ns // 1_000
    _write_exact_range(
        tmp_path,
        close_ns,
        [
            _exact_price("66.5156", close_us - 100_000),
            _exact_price("99.9999", close_us),
        ],
        tag="later-publication",
    )
    artifact = run_local_settlement_audit(
        tmp_path, _semantics(), universe=_universe(ticker)
    )
    row = _details(artifact.details_path)[0]

    assert row["reconstructed_final_value"] == "66.516"
    assert row["settlement_publish_timestamp_us"] == close_us - 100_000
    assert row["correction_policy"] == "last_publication_before_boundary_v1"
    assert row["agreement"] is True


def test_exact_mismatch_remains_usable_and_is_an_unresolved_hard_stop(tmp_path):
    ticker, _, _ = _fixture(tmp_path, final_close="66.5000", exchange_final="66.516")
    artifact = run_local_settlement_audit(
        tmp_path, _semantics(), universe=_universe(ticker)
    )
    row = _details(artifact.details_path)[0]

    assert row["usable"] is True
    assert row["agreement"] is False
    assert row["discrepancy_classification"] == "unresolved_discrepancy"
    assert artifact.summary["discrepancy_counts"] == {"unresolved_discrepancy": 1}
    assert artifact.summary["usable_window_agreement_gate_passed"] is False
    assert artifact.summary["settlement_research_gate_passed"] is False


def test_partial_request_does_not_prove_the_preceding_minute(tmp_path):
    ticker, _, close_ns = _fixture(tmp_path, include_final=False)
    close_us = close_ns // 1_000
    _write_exact_range(
        tmp_path,
        close_ns,
        [_exact_price("66.5156", close_us - 100_000)],
        tag="partial-settlement",
        start_offset_us=30_000_000,
    )

    artifact = run_local_settlement_audit(
        tmp_path, _semantics(), universe=_universe(ticker)
    )
    row = _details(artifact.details_path)[0]

    assert row["settlement_pyth_source_id"] is None
    assert row["settlement_exact_history_coverage_request_ids"]
    assert "missing_settlement_exact_history_coverage" in row["exclusion_reasons"]
    assert artifact.summary["minimum_1000_or_all_available_met"] is False


def test_feed_update_carry_forward_and_source_age_are_preserved(tmp_path):
    ticker, _, close_ns = _fixture(tmp_path)
    close_us = close_ns // 1_000
    _write_exact_range(
        tmp_path,
        close_ns,
        [
            _exact_price(
                "66.5156",
                close_us - 100_000,
                feed_update_us=close_us - 10_000_000,
            )
        ],
        tag="carried-settlement",
    )

    artifact = run_local_settlement_audit(
        tmp_path, _semantics(), universe=_universe(ticker)
    )
    row = _details(artifact.details_path)[0]

    assert row["usable"] is True
    assert row["settlement_carried_forward"] is True
    assert row["stale_source_fallback_flag"] is True
    assert row["stale_source_fallback"] is True
    assert row["settlement_source_age_ns"] == 10_000_000_000
    assert row["source_age_ns"] == 10_000_000_000


def test_conflicting_exact_publications_are_classified_and_excluded(tmp_path):
    ticker, _, close_ns = _fixture(tmp_path)
    close_us = close_ns // 1_000
    _write_exact_range(
        tmp_path,
        close_ns,
        [_exact_price("66.4000", close_us - 200_000)],
        tag="conflicting-settlement",
    )

    artifact = run_local_settlement_audit(
        tmp_path, _semantics(), universe=_universe(ticker)
    )
    row = _details(artifact.details_path)[0]

    assert row["usable"] is False
    assert "conflicting_settlement_exact_history" in row["exclusion_reasons"]
    assert artifact.summary["usable_count"] == 0
    assert artifact.summary["conflicting_count"] == 1
    assert artifact.summary["minimum_1000_or_all_available_met"] is False


def test_tampered_input_hash_chain_is_rejected_before_output(tmp_path):
    _fixture(tmp_path)
    path = next((tmp_path / "normalized" / "pyth_exact_history").glob("**/*.jsonl"))
    contents = path.read_text()
    path.write_text(contents.replace("66.5155", "66.4000", 1))

    with pytest.raises(SettlementAuditError, match="completed-request verification"):
        run_local_settlement_audit(tmp_path, _semantics())
    assert not (tmp_path / "derived" / "settlement_check").exists()


def test_tampered_immutable_output_is_rejected(tmp_path):
    _fixture(tmp_path)
    artifact = run_local_settlement_audit(tmp_path, _semantics())
    artifact.summary_path.write_text("{}\n")

    with pytest.raises(SettlementAuditError, match="output changed"):
        verify_settlement_audit(tmp_path, artifact.directory)
    with pytest.raises(SettlementAuditError, match="output changed"):
        run_local_settlement_audit(tmp_path, _semantics())


def test_mapping_parsers_require_versioned_confirmation_evidence():
    semantics = _semantics().to_dict()
    semantics["schema_version"] = SEMANTICS_SCHEMA_VERSION
    parsed = SettlementAuditSemantics.from_mapping(semantics)
    assert parsed.all_confirmed is True

    semantics["timestamp"]["evidence_sha256"] = None
    with pytest.raises(ValueError, match="evidence SHA-256"):
        SettlementAuditSemantics.from_mapping(semantics)

    universe = _universe("M").to_dict()
    universe["schema_version"] = UNIVERSE_SCHEMA_VERSION
    assert SettlementUniverse.from_mapping(universe).completed_market_tickers == ("M",)
