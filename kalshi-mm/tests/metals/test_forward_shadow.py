import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from src.metals.baseline_fair import (
    SCHEMA_VERSION,
    USAGE_CLASSIFICATION,
    BaselineFairError,
    BaselineVolModel,
    load_baseline_vol_model,
)
from src.metals.book import FixedPointOrderBook
from src.metals.config import load_metals_config
from src.metals.forward_shadow import (
    ExactPythObservation,
    ForwardShadowEngine,
    ForwardShadowError,
    ForwardShadowSettings,
    compute_online_fair,
)
from src.metals.shadow_session import ShadowSession
from src.metals.types import MarketMetadataVersion

D = Decimal
NSEC = 1_000_000_000


def _model():
    return BaselineVolModel(
        global_variance_15m=0.000004,
        winsor_absolute_log_return=0.01,
        time_of_day_variance={bucket: 0.000004 for bucket in range(96)},
        time_of_day_observations={bucket: 20 for bucket in range(96)},
        ewma_variance_at_training_end=0.000004,
        ewma_lambda=0.97,
        ewma_ratio_floor=0.25,
        ewma_ratio_ceiling=4.0,
        standardized_residual_scale=1.0,
        standardized_residuals=(-1.0, 0.0, 1.0),
        probability_floor=0.005,
    )


def _metadata(*, receive_ts_ns=1_000 * NSEC):
    return MarketMetadataVersion(
        market_ticker="KXSILVER15M-TEST",
        event_ticker="KXSILVER15M-EVENT",
        series_ticker="KXSILVER15M",
        metal="silver",
        open_ts_ns=1_000 * NSEC,
        close_ts_ns=1_900 * NSEC,
        determination_ts_ns=None,
        target_price="67.000",
        round_digits=3,
        strike_type="greater_or_equal",
        settlement_source_name="Pyth - Silver",
        settlement_source_url="https://app.pyth.com/explore/Metal.Index.SILVER%2FUSD",
        rules_primary="close at least opening target",
        rules_secondary="round to three decimal places",
        pyth_symbol="Metal.Index.SILVER/USD",
        pyth_feed_id=None,
        price_level_structure="tapered_deci_cent",
        price_ranges_json=json.dumps(
            [
                {"start": "0.000", "end": "0.100", "step": "0.001"},
                {"start": "0.100", "end": "0.900", "step": "0.010"},
                {"start": "0.900", "end": "1.000", "step": "0.001"},
            ]
        ),
        fractional_enabled=True,
        exchange_index=0,
        metadata_version=1,
        metadata_hash="metadata-hash",
        metadata_receive_ts_ns=receive_ts_ns,
        missing_required_fields=[],
    )


def _pyth(*, value="67", source_ts_ns=1_300 * NSEC, receive_ts_ns=None):
    return ExactPythObservation(
        symbol="Metal.Index.SILVER/USD",
        value=D(value),
        source_ts_ns=source_ts_ns,
        receive_ts_ns=receive_ts_ns or source_ts_ns + 400_000_000,
        ingestion_sequence=10,
    )


def _book():
    book = FixedPointOrderBook("KXSILVER15M-TEST")
    book.apply_snapshot(
        {
            "yes_dollars_fp": [["0.45", "100"]],
            "no_dollars_fp": [["0.45", "100"]],
        },
        "snapshot-1",
    )
    return book


def _underdog_book():
    book = FixedPointOrderBook("KXSILVER15M-TEST")
    book.apply_snapshot(
        {
            "yes_dollars_fp": [["0.20", "75"]],
            "no_dollars_fp": [["0.78", "125"]],
        },
        "snapshot-underdog",
    )
    return book


def _session(tmp_path: Path):
    config = load_metals_config("configs/metals.example.toml")
    config = replace(
        config,
        storage=replace(config.storage, root=str(tmp_path / "metals")),
    )
    return ShadowSession(config)


def test_frozen_baseline_loader_preserves_study_identity(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "study_id": "study-id",
                "usage_classification": USAGE_CLASSIFICATION,
                "model": _model().public_dict(),
            }
        ),
        encoding="utf-8",
    )

    model, study_id = load_baseline_vol_model(
        path, expected_study_id="study-id"
    )

    assert study_id == "study-id"
    assert model.global_variance_15m == _model().global_variance_15m
    with pytest.raises(BaselineFairError, match="does not match"):
        load_baseline_vol_model(path, expected_study_id="other-study")


def test_online_fair_uses_exact_symbol_and_tod_only_variance():
    pyth = _pyth()
    fair = compute_online_fair(
        _model(),
        _metadata(),
        pyth,
        decision_ts_ns=pyth.receive_ts_ns,
        max_pyth_age_ms=2_000,
    )

    assert D("0") < fair.yes_probability < D("1")
    assert fair.seconds_remaining == 600
    assert fair.maximum_input_ingestion_sequence == 10
    assert D("0") < fair.yes_probability_empirical < D("1")
    assert fair.public_dict()["variance_state"] == "time_of_day_only"
    assert fair.public_dict()["pyth_source_ts_ns"] == pyth.source_ts_ns
    assert fair.public_dict()["pyth_receive_ts_ns"] == pyth.receive_ts_ns
    assert fair.public_dict()["round_digits"] == 3
    assert fair.public_dict()["settlement_symbol_aligned"] is True
    assert fair.public_dict()["trading_eligible"] is False


def test_online_fair_rejects_stale_or_wrong_settlement_feed():
    pyth = _pyth()
    with pytest.raises(ForwardShadowError, match="stale"):
        compute_online_fair(
            _model(),
            _metadata(),
            pyth,
            decision_ts_ns=pyth.source_ts_ns + 3 * NSEC,
            max_pyth_age_ms=2_000,
        )

    with pytest.raises(ForwardShadowError, match="does not match"):
        compute_online_fair(
            _model(),
            replace(_metadata(), pyth_symbol="Metal.XAG/USD"),
            pyth,
            decision_ts_ns=pyth.receive_ts_ns,
            max_pyth_age_ms=2_000,
        )


def test_engine_places_paired_quotes_and_only_fills_strict_penetration(tmp_path):
    session = _session(tmp_path)
    settings = ForwardShadowSettings(
        uncertainty_charge=D("0.01"),
        adverse_selection_charge=D("0.01"),
        fee_charge_per_contract=D("0.01"),
    )
    engine = ForwardShadowEngine(_model(), "study-id", settings, session)
    metadata = _metadata()
    pyth = _pyth()
    engine.observe_market(metadata)
    engine.observe_pyth(pyth)

    fair = engine.evaluate(
        metadata.market_ticker,
        _book(),
        decision_ts_ns=pyth.receive_ts_ns,
        input_ingestion_sequence=11,
        force=True,
    )

    assert fair is not None
    assert set(engine.active_quotes) == {
        (metadata.market_ticker, "yes"),
        (metadata.market_ticker, "no"),
    }
    yes_id = engine.active_quotes[(metadata.market_ticker, "yes")]
    yes_order = session.engine.resting_orders[yes_id]
    assert yes_order.intent.price == D("0.46")

    engine.observe_trade(
        "silver",
        {
            "trade_id": "touch",
            "market_ticker": metadata.market_ticker,
            "yes_price_dollars": "0.46",
            "no_price_dollars": "0.54",
            "quantity_fp": "1",
            "taker_outcome_side": "no",
            "receive_ts_ns": pyth.receive_ts_ns + 1,
            "ingestion_sequence": 12,
        },
    )
    assert yes_id in session.engine.resting_orders
    assert engine.counts["touches_not_filled"] == 1

    engine.observe_trade(
        "silver",
        {
            "trade_id": "penetration",
            "market_ticker": metadata.market_ticker,
            "yes_price_dollars": "0.45",
            "no_price_dollars": "0.55",
            "quantity_fp": "1",
            "taker_outcome_side": "no",
            "receive_ts_ns": pyth.receive_ts_ns + 2,
            "ingestion_sequence": 13,
        },
    )
    assert yes_id not in session.engine.resting_orders
    assert engine.counts["strict_penetration_fills"] == 1
    assert session.counts["shadow_fill_attributions"] == 1
    engine.cancel_all(pyth.receive_ts_ns + 3)
    session.close()


def test_invalid_book_cancels_existing_quotes(tmp_path):
    session = _session(tmp_path)
    engine = ForwardShadowEngine(
        _model(), "study-id", ForwardShadowSettings(), session
    )
    metadata = _metadata()
    pyth = _pyth()
    book = _book()
    engine.observe_market(metadata)
    engine.observe_pyth(pyth)
    assert engine.evaluate(
        metadata.market_ticker,
        book,
        decision_ts_ns=pyth.receive_ts_ns,
        input_ingestion_sequence=11,
        force=True,
    )

    book.invalidate()
    assert engine.evaluate(
        metadata.market_ticker,
        book,
        decision_ts_ns=pyth.receive_ts_ns + 1,
        input_ingestion_sequence=12,
        force=True,
    ) is None
    assert engine.active_quotes == {}
    session.close()


def test_market_rollover_cancels_quotes_from_inactive_window(tmp_path):
    session = _session(tmp_path)
    engine = ForwardShadowEngine(
        _model(), "study-id", ForwardShadowSettings(), session
    )
    metadata = _metadata()
    pyth = _pyth()
    engine.observe_market(metadata)
    engine.observe_pyth(pyth)
    assert engine.evaluate(
        metadata.market_ticker,
        _book(),
        decision_ts_ns=pyth.receive_ts_ns,
        input_ingestion_sequence=11,
        force=True,
    )

    engine.cancel_inactive({"KXSILVER15M-NEXT"}, pyth.receive_ts_ns + 1)

    assert engine.active_quotes == {}
    assert engine.counts["market_rollover_cancels"] == 2
    assert engine.counts["quote_cancels"] == 2
    session.close()


def test_long_vol_log_retains_lineage_rejections_and_persistence(tmp_path):
    session = _session(tmp_path)
    settings = ForwardShadowSettings(
        uncertainty_charge=D("0"),
        adverse_selection_charge=D("0"),
        fee_charge_per_contract=D("0.01"),
    )
    engine = ForwardShadowEngine(_model(), "study-id", settings, session)
    metadata = _metadata()
    book = _underdog_book()
    first_pyth = _pyth(value="67.2")
    engine.observe_market(metadata)
    engine.observe_pyth(first_pyth)

    first_decision = first_pyth.receive_ts_ns
    assert engine.evaluate(
        metadata.market_ticker,
        book,
        decision_ts_ns=first_decision,
        input_ingestion_sequence=11,
        trigger_source="kalshi_book_delta",
        book_receive_ts_ns=first_decision,
        book_local_monotonic_ts_ns=900,
        book_ingestion_sequence=11,
        force=True,
    )

    second_decision = first_decision + NSEC
    second_pyth = _pyth(
        value="67.2",
        source_ts_ns=first_pyth.source_ts_ns + NSEC,
        receive_ts_ns=second_decision,
    )
    engine.observe_pyth(second_pyth)
    assert engine.evaluate(
        metadata.market_ticker,
        book,
        decision_ts_ns=second_decision,
        input_ingestion_sequence=12,
        trigger_source="kalshi_exact_pyth",
        book_receive_ts_ns=second_decision,
        book_local_monotonic_ts_ns=1_000,
        book_ingestion_sequence=12,
        force=True,
    )
    session.close()

    paths = list(
        (tmp_path / "metals" / "normalized" / "long_vol_candidate").glob(
            "**/*.jsonl"
        )
    )
    assert len(paths) == 1
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["screen_eligible"] is True
    assert rows[0]["shadow_policy_eligible"] is False
    assert "persistence_not_met" in rows[0]["shadow_policy_rejection_reasons"]
    assert rows[0]["taker_fee_per_contract"] == "0.0121"
    assert rows[0]["book"]["yes_bid_quantity_fp"] == "75"
    assert rows[0]["book"]["yes_ask_quantity_fp"] == "125"
    assert rows[0]["exact_index"]["ingestion_sequence"] == 10
    assert rows[1]["displayed_edge_persistence_ms"] == "1000"
    assert rows[1]["shadow_policy_eligible"] is True
    assert rows[1]["execution_eligible"] is False
    assert rows[0]["candidate_id"] != rows[1]["candidate_id"]
