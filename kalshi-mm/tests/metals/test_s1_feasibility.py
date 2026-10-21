from decimal import Decimal

import duckdb

from src.metals.s1_feasibility import (
    CandleRow,
    S1FeasibilityConfig,
    Signal,
    analyze_s1_feasibility,
    build_signals,
    load_trade_markouts,
    summarize_trade_markout_study,
)


def _candle(
    ticker,
    open_seconds,
    end_seconds,
    result,
    *,
    bid_open="0.79",
    bid_low="0.78",
    bid_high="0.83",
    bid_close="0.79",
    ask_open="0.81",
    ask_low="0.80",
    ask_high="0.82",
    ask_close="0.81",
    volume="10",
):
    return CandleRow(
        market_ticker=ticker,
        market_open_ts_ns=open_seconds * 1_000_000_000,
        market_close_ts_ns=(open_seconds + 900) * 1_000_000_000,
        end_period_ts_ns=end_seconds * 1_000_000_000,
        yes_bid_open=Decimal(bid_open),
        yes_bid_low=Decimal(bid_low),
        yes_bid_high=Decimal(bid_high),
        yes_bid_close=Decimal(bid_close),
        yes_ask_open=Decimal(ask_open),
        yes_ask_low=Decimal(ask_low),
        yes_ask_high=Decimal(ask_high),
        yes_ask_close=Decimal(ask_close),
        volume=Decimal(volume),
        result=result,
    )


def _market_pair(ticker, open_seconds, result):
    decision_end = open_seconds + 300
    return [
        _candle(ticker, open_seconds, decision_end, result),
        _candle(
            ticker,
            open_seconds,
            decision_end + 60,
            result,
            bid_open="0.80",
            bid_high="0.83",
        ),
    ]


def test_analysis_uses_chronological_split_and_three_explicit_fill_proxies():
    candles = []
    for index, result in enumerate(("no", "yes", "yes", "yes")):
        candles.extend(_market_pair("M{}".format(index), 1_000 + index * 1_000, result))
    config = S1FeasibilityConfig(bootstrap_samples=50)

    report, signals, executions = analyze_s1_feasibility(candles, config)

    assert len(signals) == 4
    assert {row.time_bucket for row in signals} == {"10-15m"}
    assert {row.underdog_side for row in signals} == {"no"}
    assert {row.passive_bid for row in signals} == {Decimal("0.19")}
    assert [row.split for row in signals] == ["train", "train", "test", "test"]
    assert len(executions) == 12
    assert report["decision"] == "NO_GO_FOR_TRADING"
    assert (
        report["research_conclusion"]["status"]
        == "PRELIMINARY_REJECT_UNFILTERED_S1"
    )
    assert report["chronological_split"]["train_market_count"] == 2
    assert report["chronological_split"]["test_market_count"] == 2

    test_results = report["execution_results"]["test"]
    assert test_results["aggressive_next_open_proxy"]["fills"] == 2
    assert test_results["passive_touch_proxy"]["fills"] == 2
    assert test_results["passive_penetration_proxy"]["fills"] == 2
    assert (
        test_results["aggressive_next_open_proxy"]["average_fee_dollars"]
        == "0.011200"
    )
    assert (
        test_results["passive_penetration_proxy"]["average_net_pnl_dollars"]
        == "-0.190000"
    )
    assert report["gate_assessment"]["final_q0_q1_q2_available"] is False


def test_signal_requires_exact_next_candle_and_ignores_one_sided_quotes():
    candles = [
        _candle("GAP", 1_000, 1_300, "no"),
        _candle("GAP", 1_000, 1_421, "no"),
        _candle(
            "INVALID",
            3_000,
            3_300,
            "yes",
            bid_close="0",
        ),
        _candle("INVALID", 3_000, 3_360, "yes"),
    ]

    assert build_signals(candles, S1FeasibilityConfig()) == []


def test_no_favorite_is_converted_to_yes_underdog_consistently():
    candles = [
        _candle(
            "NO_FAVORITE",
            5_000,
            5_300,
            "yes",
            bid_open="0.19",
            bid_low="0.18",
            bid_high="0.20",
            bid_close="0.19",
            ask_open="0.21",
            ask_low="0.20",
            ask_high="0.22",
            ask_close="0.21",
        ),
        _candle(
            "NO_FAVORITE",
            5_000,
            5_360,
            "yes",
            bid_open="0.19",
            bid_low="0.18",
            bid_high="0.20",
            bid_close="0.19",
            ask_open="0.20",
            ask_low="0.18",
            ask_high="0.21",
            ask_close="0.20",
        ),
    ]

    signals = build_signals(candles, S1FeasibilityConfig())

    assert len(signals) == 1
    assert signals[0].underdog_side == "yes"
    assert signals[0].passive_bid == Decimal("0.19")
    assert signals[0].next_underdog_ask_open == Decimal("0.20")
    assert signals[0].next_underdog_ask_low == Decimal("0.18")
    assert signals[0].underdog_won is True


def test_trade_markouts_anchor_touch_and_penetration_without_calling_them_midpoints(
    tmp_path,
):
    catalog = tmp_path / "metals.duckdb"
    decision = 1_000_000_000_000
    with duckdb.connect(str(catalog)) as connection:
        connection.execute(
            "CREATE TABLE kalshi_trade (market_ticker VARCHAR, "
            "source_event_ts_ns BIGINT, trade_id VARCHAR, "
            "yes_price_dollars VARCHAR, no_price_dollars VARCHAR, "
            "quantity_fp VARCHAR, taker_outcome_side VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO kalshi_trade VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "MARKET",
                    decision + 500_000_000,
                    "a",
                    "0.80",
                    "0.20",
                    "2",
                    "yes",
                ),
                (
                    "MARKET",
                    decision + 2_000_000_000,
                    "b",
                    "0.82",
                    "0.18",
                    "3",
                    "yes",
                ),
            ],
        )
    signal = Signal(
        market_ticker="MARKET",
        market_open_ts_ns=decision - 300_000_000_000,
        market_close_ts_ns=decision + 600_000_000_000,
        decision_ts_ns=decision,
        seconds_to_close=600,
        time_bucket="10-15m",
        price_bucket="0.15-0.25",
        underdog_side="no",
        underdog_mid=Decimal("0.20"),
        passive_bid=Decimal("0.20"),
        underdog_won=True,
        next_underdog_ask_open=Decimal("0.21"),
        next_underdog_ask_low=Decimal("0.18"),
        next_volume=Decimal("5"),
        split="test",
    )
    config = S1FeasibilityConfig(bootstrap_samples=10)

    markouts = load_trade_markouts(catalog, [signal], config)
    study = summarize_trade_markout_study(markouts, [signal], config)

    assert len(markouts) == 12
    touch_1s = next(
        row
        for row in markouts
        if row.model == "passive_touch_proxy" and row.horizon == "1s"
    )
    penetration_1s = next(
        row
        for row in markouts
        if row.model == "passive_penetration_proxy" and row.horizon == "1s"
    )
    assert touch_1s.fill_delay_seconds == Decimal("0.5")
    assert touch_1s.net_markout_pnl == Decimal("0.00")
    assert penetration_1s.fill_delay_seconds == Decimal("2")
    assert penetration_1s.net_markout_pnl == Decimal("-0.02")
    touch_summary = study["results"]["test"]["passive_touch_proxy"]["1s"]
    assert touch_summary[
        "average_market_implied_binary_variance_dollars_squared"
    ] == "0.160000"
    assert touch_summary[
        "average_market_implied_terminal_pnl_stddev_dollars"
    ] == "0.400000"
    assert study["classification"] == "last_trade_price_proxy_not_midpoint_markout"
    assert study["vega"]["true_digital_option_vega"].startswith("not_estimable")
