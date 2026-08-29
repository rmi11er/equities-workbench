"""Lossless JSONL-to-Parquet export and local DuckDB catalog creation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence


_DEDUPLICATION_RULES = {
    "kalshi_trade": (
        ("trade_id",),
        ("source_event_ts_ns", "receive_ts_ns", "run", "record_sha256"),
    ),
    "kalshi_settlement": (
        ("market_ticker",),
        ("settlement_ts_ns", "receive_ts_ns", "run", "record_sha256"),
    ),
    "kalshi_settlement_source": (
        ("market_ticker",),
        ("settlement_ts_ns", "receive_ts_ns", "run", "record_sha256"),
    ),
    "market_dim": (
        ("market_ticker",),
        (
            "metadata_version",
            "metadata_receive_ts_ns",
            "run",
            "record_sha256",
        ),
    ),
    "pyth_candle": (
        ("pyth_symbol", "timestamp_seconds", "resolution_minutes"),
        ("receive_ts_ns", "run", "record_sha256"),
    ),
}


def _duckdb():
    try:
        import duckdb
    except ImportError as error:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "Parquet export requires duckdb; install requirements.txt"
        ) from error
    return duckdb


def _sql_literal(path: Path) -> str:
    return str(path).replace("'", "''")


def _identifier(value: str) -> str:
    if not value.replace("_", "").isalnum():
        raise ValueError("Unsafe stream name: {}".format(value))
    return '"{}"'.format(value.replace('"', '""'))


def _available_columns(connection, view_name: str) -> set:
    return {
        row[0]
        for row in connection.execute(
            "DESCRIBE SELECT * FROM {}".format(_identifier(view_name))
        ).fetchall()
    }


def _deduplicated_select(
    all_runs_view: str,
    keys: Sequence[str],
    order_candidates: Iterable[str],
    available_columns: set,
) -> str:
    if not set(keys).issubset(available_columns):
        return "SELECT * FROM {}".format(_identifier(all_runs_view))
    order_columns = [
        column for column in order_candidates if column in available_columns
    ]
    if not order_columns:
        order_columns = list(keys)
    partition = ", ".join(_identifier(column) for column in keys)
    order = ", ".join(
        "{} DESC NULLS LAST".format(_identifier(column))
        for column in order_columns
    )
    # A missing identity is not evidence that two records are the same. Keep
    # all such rows instead of collapsing them into a single NULL partition.
    missing_identity = " OR ".join(
        "{} IS NULL".format(_identifier(column)) for column in keys
    )
    return (
        "SELECT * FROM {all_runs} QUALIFY ({missing}) OR "
        "row_number() OVER (PARTITION BY {partition} ORDER BY {order}) = 1"
    ).format(
        all_runs=_identifier(all_runs_view),
        missing=missing_identity,
        partition=partition,
        order=order,
    )


def export_normalized_parquet(root: Path) -> Dict[str, int]:
    """Export each immutable normalized JSONL segment exactly once."""

    root = Path(root)
    source_root = root / "normalized"
    destination_root = root / "normalized_parquet"
    counts = {"exported": 0, "skipped": 0}
    duckdb = _duckdb()
    connection = duckdb.connect()
    try:
        for source in sorted(source_root.glob("**/*.jsonl")):
            relative = source.relative_to(source_root)
            destination = (destination_root / relative).with_suffix(".parquet")
            if destination.exists():
                counts["skipped"] += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            connection.execute(
                "COPY (SELECT * FROM read_json_auto('{}', "
                "format='newline_delimited', union_by_name=true, "
                "maximum_object_size=104857600)) TO '{}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)".format(
                    _sql_literal(source), _sql_literal(destination)
                )
            )
            counts["exported"] += 1
    finally:
        connection.close()
    return counts


def build_duckdb_catalog(root: Path) -> Path:
    """Create provenance and deduplicated analytical views over Parquet.

    ``<stream>_all_runs`` and its ``<stream>_raw`` alias expose every stored
    record. The familiar ``<stream>`` name is the analytical view and applies a
    deterministic identity rule for streams whose source has a stable key.
    """

    root = Path(root)
    parquet_root = root / "normalized_parquet"
    catalog_path = root / "metals.duckdb"
    duckdb = _duckdb()
    connection = duckdb.connect(str(catalog_path))
    try:
        stream_directories = sorted(
            path for path in parquet_root.iterdir() if path.is_dir()
        )
        for stream_dir in stream_directories:
            view_name = stream_dir.name
            _identifier(view_name)
            pattern = stream_dir / "**" / "*.parquet"
            all_runs_view = "{}_all_runs".format(view_name)
            raw_view = "{}_raw".format(view_name)
            connection.execute(
                "CREATE OR REPLACE VIEW {} AS SELECT * FROM "
                "read_parquet('{}', hive_partitioning=true, union_by_name=true)".format(
                    _identifier(all_runs_view), _sql_literal(pattern)
                )
            )
            connection.execute(
                "CREATE OR REPLACE VIEW {} AS SELECT * FROM {}".format(
                    _identifier(raw_view), _identifier(all_runs_view)
                )
            )
            rule = _DEDUPLICATION_RULES.get(view_name)
            if rule is None:
                analytical_select = "SELECT * FROM {}".format(
                    _identifier(all_runs_view)
                )
            else:
                analytical_select = _deduplicated_select(
                    all_runs_view,
                    keys=rule[0],
                    order_candidates=rule[1],
                    available_columns=_available_columns(connection, all_runs_view),
                )
            connection.execute(
                "CREATE OR REPLACE VIEW {} AS {}".format(
                    _identifier(view_name), analytical_select
                )
            )
    finally:
        connection.close()
    return catalog_path
