from decimal import Decimal

import pytest
import duckdb

from src.metals.toxicity_diagnostics import (
    ToxicityConfig,
    _paired_window_pnl,
    _trimmed,
    action_category,
    load_catalog_coverage,
    load_diagnostics,
    price_bucket,
    summarize_pairs,
)


@pytest.mark.parametrize(
    "price, expected",
    [
        ("0", "0-10c"),
        ("0.099", "0-10c"),
        ("0.10", "10-20c"),
        ("0.20", "20-35c"),
        ("0.35", "35-65c"),
        ("0.65", "65-80c"),
        ("0.80", "80-90c"),
        ("0.90", "90-100c"),
        ("1", "90-100c"),
    ],
)
def test_price_buckets_have_deterministic_left_closed_boundaries(price, expected):
    assert price_bucket(Decimal(price)) == expected


@pytest.mark.parametrize(
    "action, midpoint, expected",
    [
        ("buy_yes", "0.60", "buying_market_favorite"),
        ("sell_yes", "0.60", "selling_market_favorite"),
        ("buy_yes", "0.40", "buying_market_underdog"),
        ("sell_yes", "0.40", "selling_market_underdog"),
        ("buy_yes", "0.50", None),
    ],
)
def test_yes_side_action_cut_preserves_favorite_orientation(
    action, midpoint, expected
):
    assert action_category(action, Decimal(midpoint)) == expected


def test_one_percent_trim_removes_both_extremes_without_winsorizing():
    values = [Decimal(index) for index in range(100)]

    assert _trimmed(values, Decimal("0.01")) == values[1:-1]


@pytest.mark.parametrize(
    "result,yes_filled,no_filled,expected_state,expected_gross,expected_net",
    [
        ("yes", True, True, "both", "0.20", "0.18"),
        ("yes", True, False, "yes_only", "0.60", "0.59"),
        ("no", True, False, "yes_only", "-0.40", "-0.41"),
        ("no", False, True, "no_only", "0.60", "0.59"),
        ("yes", False, True, "no_only", "-0.40", "-0.41"),
        ("yes", False, False, "neither", "0", "0"),
    ],
)
def test_paired_window_pnl_accounts_for_every_fill_state(
    result,
    yes_filled,
    no_filled,
    expected_state,
    expected_gross,
    expected_net,
):
    state, gross, net = _paired_window_pnl(
        result=result,
        yes_bid=Decimal("0.40"),
        yes_offer=Decimal("0.60"),
        yes_filled=yes_filled,
        no_filled=no_filled,
        maker_fee_per_contract=Decimal("0.01"),
    )

    assert state == expected_state
    assert gross == Decimal(expected_gross)
    assert net == Decimal(expected_net)


def test_generic_diagnostic_anchors_both_sides_and_builds_paired_legs(tmp_path):
    catalog = tmp_path / "metals.duckdb"
    second = 1_000_000_000
    with duckdb.connect(str(catalog)) as connection:
        connection.execute(
            "CREATE TABLE market_dim (market_ticker VARCHAR, open_ts_ns BIGINT, "
            "close_ts_ns BIGINT, metal VARCHAR)"
        )
        connection.execute(
            "INSERT INTO market_dim VALUES ('M', 0, ?, 'silver')",
            [900 * second],
        )
        connection.execute(
            "CREATE TABLE kalshi_candlestick (market_ticker VARCHAR, "
            "end_period_ts_ns BIGINT, yes_bid_close_dollars VARCHAR, "
            "yes_ask_close_dollars VARCHAR, metal VARCHAR, "
            "period_interval_minutes INTEGER)"
        )
        connection.execute(
            "INSERT INTO kalshi_candlestick VALUES "
            "('M', ?, '0.40', '0.60', 'silver', 1)",
            [60 * second],
        )
        connection.execute(
            "CREATE TABLE kalshi_settlement (market_ticker VARCHAR, result VARCHAR)"
        )
        connection.execute("INSERT INTO kalshi_settlement VALUES ('M', 'yes')")
        connection.execute(
            "CREATE TABLE kalshi_trade (market_ticker VARCHAR, "
            "source_event_ts_ns BIGINT, trade_id VARCHAR, "
            "yes_price_dollars VARCHAR, quantity_fp VARCHAR, "
            "taker_outcome_side VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO kalshi_trade VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("M", 61 * second, "a", "0.39", "2", "no"),
                ("M", 62 * second, "b", "0.61", "3", "yes"),
            ],
        )
        connection.execute(
            "CREATE TABLE kalshi_trade_request "
            "(market_ticker VARCHAR, status VARCHAR)"
        )
        connection.execute(
            "INSERT INTO kalshi_trade_request VALUES ('M', 'completed')"
        )

    markouts, pairs, counts = load_diagnostics(
        catalog,
        ToxicityConfig(test_start_ts_ns=0, bootstrap_samples=10),
    )

    assert counts == {
        "quote_candidates": 2,
        "analytical_markets": 1,
        "two_sided_quote_decisions": 1,
        "neutral_candidates": 2,
        "anchored_fills": 4,
        "markout_rows": 24,
    }
    assert len(markouts) == 24
    assert len(pairs) == 2
    assert all(pair.both_filled for pair in pairs)
    assert {pair.fill_state for pair in pairs} == {"both"}
    assert {pair.fill_sequence for pair in pairs} == {"yes_then_no"}
    assert {pair.locked_gross_pnl for pair in pairs} == {Decimal("0.20")}
    assert {pair.window_net_pnl for pair in pairs} == {Decimal("0.20")}

    pair_summary = summarize_pairs(
        pairs,
        ToxicityConfig(test_start_ts_ns=0, bootstrap_samples=10),
    )[10]
    assert pair_summary["split"] == "test"
    assert pair_summary["model"] == "touch"
    assert pair_summary["time_bucket"] == "all"
    assert pair_summary["two_sided_quote_decisions"] == 1
    assert pair_summary["both_legs_fill"] == 1
    assert pair_summary["single_leg_fill"] == 0
    assert pair_summary["neither_fills"] == 0
    assert pair_summary["average_window_net_pnl_dollars"] == "0.200000"

    gated_markouts, gated_pairs, gated_counts = load_diagnostics(
        catalog,
        ToxicityConfig(
            test_start_ts_ns=0,
            bootstrap_samples=10,
            require_completed_trade_request=True,
        ),
    )
    assert gated_counts == counts
    assert gated_markouts == markouts
    assert gated_pairs == pairs

    coverage = load_catalog_coverage(catalog, "silver")
    assert coverage["completed_trade_request_markets"] == 1
    assert coverage["completed_trade_request_utc_days"] == 1
