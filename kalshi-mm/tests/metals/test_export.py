import json

import duckdb

from src.metals.export import build_duckdb_catalog, export_normalized_parquet


def test_export_creates_zstd_parquet_and_queryable_view(tmp_path):
    source = (
        tmp_path
        / "normalized"
        / "kalshi_trade"
        / "date=2026-08-29"
        / "metal=silver"
        / "run=test"
        / "segment.jsonl"
    )
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps({"trade_id": "one", "quantity_fp": "1.25"}) + "\n"
    )

    counts = export_normalized_parquet(tmp_path)
    catalog = build_duckdb_catalog(tmp_path)

    assert counts == {"exported": 1, "skipped": 0}
    with duckdb.connect(str(catalog), read_only=True) as connection:
        row = connection.execute(
            "SELECT trade_id, quantity_fp, metal FROM kalshi_trade"
        ).fetchone()
    assert row == ("one", "1.25", "silver")


def _write_normalized(tmp_path, stream, run, records):
    source = (
        tmp_path
        / "normalized"
        / stream
        / "date=2026-08-29"
        / "metal=silver"
        / "run={}".format(run)
        / "segment.jsonl"
    )
    source.parent.mkdir(parents=True)
    source.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_catalog_deduplicates_repeated_runs_and_retains_provenance(tmp_path):
    _write_normalized(
        tmp_path,
        "kalshi_trade",
        "run-1",
        [
            {
                "trade_id": "duplicate",
                "market_ticker": "SILVER-1",
                "quantity_fp": "1",
                "receive_ts_ns": 100,
                "record_sha256": "a",
            },
            {
                "trade_id": None,
                "market_ticker": "SILVER-1",
                "quantity_fp": "missing-id-old",
                "receive_ts_ns": 100,
                "record_sha256": "b",
            },
        ],
    )
    _write_normalized(
        tmp_path,
        "kalshi_trade",
        "run-2",
        [
            {
                "trade_id": "duplicate",
                "market_ticker": "SILVER-1",
                "quantity_fp": "2",
                "receive_ts_ns": 100,
                "record_sha256": "c",
            },
            {
                "trade_id": None,
                "market_ticker": "SILVER-1",
                "quantity_fp": "missing-id-new",
                "receive_ts_ns": 200,
                "record_sha256": "d",
            },
        ],
    )
    _write_normalized(
        tmp_path,
        "kalshi_settlement",
        "run-1",
        [
            {
                "market_ticker": "SILVER-1",
                "result": "no",
                "settlement_ts_ns": 100,
                "receive_ts_ns": 100,
                "record_sha256": "a",
            }
        ],
    )
    _write_normalized(
        tmp_path,
        "kalshi_settlement",
        "run-2",
        [
            {
                "market_ticker": "SILVER-1",
                "result": "yes",
                "settlement_ts_ns": 100,
                "receive_ts_ns": 100,
                "record_sha256": "b",
            }
        ],
    )

    export_normalized_parquet(tmp_path)
    catalog = build_duckdb_catalog(tmp_path)

    with duckdb.connect(str(catalog), read_only=True) as connection:
        assert connection.execute(
            "SELECT quantity_fp FROM kalshi_trade "
            "WHERE trade_id = 'duplicate'"
        ).fetchall() == [("2",)]
        assert connection.execute(
            "SELECT count(*) FROM kalshi_trade WHERE trade_id IS NULL"
        ).fetchone() == (2,)
        assert connection.execute(
            "SELECT count(*) FROM kalshi_trade_all_runs"
        ).fetchone() == (4,)
        assert connection.execute(
            "SELECT count(*) FROM kalshi_trade_raw"
        ).fetchone() == (4,)
        assert connection.execute(
            "SELECT result FROM kalshi_settlement"
        ).fetchall() == [("yes",)]
        assert connection.execute(
            "SELECT count(*) FROM kalshi_settlement_all_runs"
        ).fetchone() == (2,)


def test_catalog_stabilizes_nullable_market_metadata_types_across_segments(tmp_path):
    _write_normalized(
        tmp_path,
        "market_dim",
        "null-segment",
        [
            {
                "market_ticker": "SILVER-NULL",
                "metadata_version": 1,
                "metadata_receive_ts_ns": 100,
                "pyth_symbol": None,
                "pyth_feed_id": None,
            }
        ],
    )
    _write_normalized(
        tmp_path,
        "market_dim",
        "string-segment",
        [
            {
                "market_ticker": "SILVER-STRING",
                "metadata_version": 1,
                "metadata_receive_ts_ns": 200,
                "pyth_symbol": "Metal.Index.SILVER/USD",
                "pyth_feed_id": "feed-id",
            }
        ],
    )

    export_normalized_parquet(tmp_path)
    catalog = build_duckdb_catalog(tmp_path)

    with duckdb.connect(str(catalog), read_only=True) as connection:
        assert connection.execute(
            "SELECT market_ticker, pyth_symbol, pyth_feed_id FROM market_dim "
            "ORDER BY market_ticker"
        ).fetchall() == [
            ("SILVER-NULL", None, None),
            ("SILVER-STRING", "Metal.Index.SILVER/USD", "feed-id"),
        ]
