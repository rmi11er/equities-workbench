import json
from copy import deepcopy

import pytest

from src.metals.provisional_kalshi_labels import (
    LABEL_AUTHORITY,
    RESEARCH_USE,
    TARGET_SERIES,
    ProvisionalKalshiLabelError,
    materialize_provisional_kalshi_labels,
    verify_provisional_kalshi_labels,
)
from src.metals.storage import (
    HashChainedJsonlWriter,
    update_run_manifest,
    write_run_manifest,
)

OPEN_NS = 1_788_278_400_000_000_000
CLOSE_NS = OPEN_NS + 15 * 60 * 1_000_000_000


def _market(ticker="KXSILVER15M-26SEP010000", **changes):
    value = {
        "market_ticker": ticker,
        "event_ticker": "KXSILVER15M-26SEP010000",
        "series_ticker": TARGET_SERIES,
        "metal": "silver",
        "open_ts_ns": OPEN_NS,
        "close_ts_ns": CLOSE_NS,
        "determination_ts_ns": None,
        "target_price": "39.125",
        "round_digits": 3,
        "strike_type": "greater_or_equal",
        "price_level_structure": "single_strike",
        "price_ranges_json": "[]",
        "fractional_enabled": True,
        "metadata_version": 1,
        "metadata_hash": "a" * 64,
        "metadata_receive_ts_ns": CLOSE_NS + 1,
    }
    value.update(changes)
    return value


def _settlement(ticker="KXSILVER15M-26SEP010000", **changes):
    value = {
        "market_ticker": ticker,
        "event_ticker": "KXSILVER15M-26SEP010000",
        "series_ticker": TARGET_SERIES,
        "status": "finalized",
        "result": "yes",
        "expiration_value": "39.126",
        "settlement_value_dollars": "39.1260",
        "settlement_ts_ns": CLOSE_NS + 2,
        "receive_ts_ns": CLOSE_NS + 3,
        "raw_metadata_hash": "a" * 64,
    }
    value.update(changes)
    return value


def _write_stream(root, run_id, stream, rows):
    path = (
        root
        / "normalized"
        / stream
        / "date=2026-09-01"
        / "metal=silver"
        / f"run={run_id}"
        / f"{stream}-synthetic.jsonl"
    )
    with HashChainedJsonlWriter(path) as writer:
        for row in rows:
            writer.write(row)
    return path


def _completed_run(root, run_id, markets, settlements, *, status="completed"):
    paths = [
        _write_stream(root, run_id, "market_dim", markets),
        _write_stream(root, run_id, "kalshi_settlement", settlements),
    ]
    manifest = write_run_manifest(
        root,
        run_id,
        {
            "kalshi": {
                "rest_url": "https://synthetic.invalid",
                "read_only": True,
            }
        },
    )
    update_run_manifest(
        root,
        run_id,
        status=status,
        counts={"markets": len(settlements)},
        output_paths=paths,
    )
    return manifest, paths


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_materializes_one_strictly_provisional_label_with_full_provenance(tmp_path):
    manifest, paths = _completed_run(
        tmp_path, "synthetic-complete", [_market()], [_settlement()]
    )

    first = materialize_provisional_kalshi_labels(tmp_path, [manifest])
    second = materialize_provisional_kalshi_labels(tmp_path, [manifest])

    assert first.artifact_id == second.artifact_id
    assert first.directory == second.directory
    assert first.directory.name == f"artifact={first.artifact_id}"
    assert verify_provisional_kalshi_labels(tmp_path, first.manifest_path) == first
    labels = _rows(first.labels_path)
    assert len(labels) == 1
    label = labels[0]
    assert label["market_ticker"] == _market()["market_ticker"]
    assert label["event_ticker"] == _market()["event_ticker"]
    assert label["open_ts_ns"] == OPEN_NS
    assert label["close_ts_ns"] == CLOSE_NS
    assert label["settlement_ts_ns"] == CLOSE_NS + 2
    assert label["target_price"] == "39.125"
    assert label["strike_type"] == "greater_or_equal"
    assert label["round_digits"] == 3
    assert label["expiration_value"] == "39.126"
    assert label["result"] == "yes"
    assert label["label_authority"] == LABEL_AUTHORITY
    assert label["independent_pyth_verified"] is False
    assert label["research_use"] == RESEARCH_USE
    assert (
        label["market_source"]["record_id"] == label["market_source"]["record_sha256"]
    )
    assert (
        label["settlement_source"]["record_id"]
        == label["settlement_source"]["record_sha256"]
    )
    assert label["market_source"]["manifest_path"] == str(
        manifest.relative_to(tmp_path)
    )
    assert label["market_source"]["jsonl_path"] == str(paths[0].relative_to(tmp_path))
    assert first.summary["settlement_verification"] is False
    assert first.summary["included_labels"] == 1
    assert first.summary["excluded_candidates"] == 0


def test_excludes_and_classifies_incomplete_or_malformed_without_imputation(tmp_path):
    valid_ticker = _market()["market_ticker"]
    market_only = "KXSILVER15M-MARKET-ONLY"
    settlement_only = "KXSILVER15M-SETTLEMENT-ONLY"
    malformed = "KXSILVER15M-MALFORMED"
    wrong_series = "KXGOLD15M-IGNORED"
    markets = [
        _market(),
        _market(market_only, event_ticker="EVENT-MARKET-ONLY"),
        _market(
            malformed,
            event_ticker="EVENT-MALFORMED",
            close_ts_ns=CLOSE_NS + 1,
            target_price=None,
            round_digits=None,
            strike_type=None,
            metadata_hash="d" * 64,
        ),
        _market(
            wrong_series,
            event_ticker="KXGOLD15M-EVENT",
            series_ticker="KXGOLD15M",
            metal="gold",
            metadata_hash="e" * 64,
        ),
    ]
    settlements = [
        _settlement(),
        _settlement(
            settlement_only,
            event_ticker="EVENT-SETTLEMENT-ONLY",
            raw_metadata_hash=None,
        ),
        _settlement(
            malformed,
            event_ticker="EVENT-MALFORMED",
            result="pending",
            expiration_value=None,
            raw_metadata_hash="d" * 64,
        ),
        _settlement(
            wrong_series,
            event_ticker="KXGOLD15M-EVENT",
            series_ticker="KXGOLD15M",
            raw_metadata_hash="e" * 64,
        ),
    ]
    manifest, _ = _completed_run(tmp_path, "synthetic-invalid", markets, settlements)

    artifact = materialize_provisional_kalshi_labels(tmp_path, [manifest])

    assert [row["market_ticker"] for row in _rows(artifact.labels_path)] == [
        valid_ticker
    ]
    exclusions = {row["candidate_id"]: row for row in _rows(artifact.exclusions_path)}
    assert exclusions[market_only]["exclusion_reasons"] == ["missing_settlement"]
    assert exclusions[settlement_only]["exclusion_reasons"] == [
        "missing_market_metadata"
    ]
    assert {
        "invalid_exchange_result",
        "invalid_expiration_value",
        "non_15_minute_window",
        "unaligned_window_boundary",
    }.issubset(exclusions[malformed]["exclusion_reasons"])
    assert wrong_series not in exclusions
    assert artifact.summary["ignored_non_target_records"] == 2
    assert artifact.summary["included_labels"] == 1
    assert artifact.summary["excluded_candidates"] == 3


def test_settlement_metadata_hash_selects_bound_revision_not_newer_revision(tmp_path):
    older = _market(metadata_hash="a" * 64, metadata_version=1)
    newer = _market(
        metadata_hash="b" * 64,
        metadata_version=2,
        metadata_receive_ts_ns=CLOSE_NS + 10,
        target_price="40.000",
    )
    manifest, _ = _completed_run(
        tmp_path,
        "synthetic-revisions",
        [older, newer],
        [_settlement(raw_metadata_hash="a" * 64)],
    )

    artifact = materialize_provisional_kalshi_labels(tmp_path, [manifest])
    label = _rows(artifact.labels_path)[0]

    assert label["target_price"] == "39.125"
    assert label["market_source"]["metadata_hash"] == "a" * 64
    assert label["market_revision_count"] == 2


def test_conflicting_latest_revision_is_classified_not_selected(tmp_path):
    first = _market(metadata_hash="a" * 64, target_price="39.125")
    second = _market(metadata_hash="b" * 64, target_price="40.000")
    settlement = _settlement(raw_metadata_hash=None)
    manifest, _ = _completed_run(
        tmp_path, "synthetic-conflict", [first, second], [settlement]
    )

    artifact = materialize_provisional_kalshi_labels(tmp_path, [manifest])

    assert _rows(artifact.labels_path) == []
    exclusion = _rows(artifact.exclusions_path)[0]
    assert "ambiguous_latest_market_revision" in exclusion["exclusion_reasons"]


def test_rejects_noncompleted_nonkalshi_and_unmanifested_sources(tmp_path):
    running_manifest, _ = _completed_run(
        tmp_path, "synthetic-failed", [_market()], [_settlement()], status="failed"
    )
    with pytest.raises(ProvisionalKalshiLabelError, match="not completed"):
        materialize_provisional_kalshi_labels(tmp_path, [running_manifest])

    other_root = tmp_path / "not-kalshi"
    manifest, _ = _completed_run(
        other_root, "synthetic-not-kalshi", [_market()], [_settlement()]
    )
    value = json.loads(manifest.read_text())
    value["config"] = {"pyth": {"enabled": False}}
    value["config_sha256"] = "0" * 64
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(
        ProvisionalKalshiLabelError, match="not a Kalshi|config checksum"
    ):
        materialize_provisional_kalshi_labels(other_root, [manifest])

    unlisted_root = tmp_path / "unlisted"
    market_path = _write_stream(unlisted_root, "unlisted", "market_dim", [_market()])
    settlement_path = _write_stream(
        unlisted_root, "unlisted", "kalshi_settlement", [_settlement()]
    )
    unlisted_manifest = write_run_manifest(
        unlisted_root, "unlisted", {"kalshi": {"read_only": True}}
    )
    update_run_manifest(
        unlisted_root,
        "unlisted",
        status="completed",
        output_paths=[market_path],
    )
    assert settlement_path.is_file()
    with pytest.raises(ProvisionalKalshiLabelError, match="kalshi_settlement"):
        materialize_provisional_kalshi_labels(unlisted_root, [unlisted_manifest])


def test_source_and_output_tampering_are_detected(tmp_path):
    manifest, paths = _completed_run(
        tmp_path, "synthetic-tamper", [_market()], [_settlement()]
    )
    artifact = materialize_provisional_kalshi_labels(tmp_path, [manifest])

    source_rows = _rows(paths[0])
    source_rows[0]["target_price"] = "999"
    paths[0].write_text(json.dumps(source_rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(
        ProvisionalKalshiLabelError, match="record hash mismatch|inputs changed"
    ):
        verify_provisional_kalshi_labels(tmp_path, artifact.directory)

    other_root = tmp_path / "output-tamper"
    other_manifest, _ = _completed_run(
        other_root, "synthetic-output-tamper", [_market()], [_settlement()]
    )
    other = materialize_provisional_kalshi_labels(other_root, [other_manifest])
    label = _rows(other.labels_path)[0]
    label["independent_pyth_verified"] = True
    other.labels_path.write_text(json.dumps(label) + "\n", encoding="utf-8")
    with pytest.raises(ProvisionalKalshiLabelError, match="checksum"):
        verify_provisional_kalshi_labels(other_root, other.directory)


def test_source_manifest_checksum_and_content_address_are_enforced(tmp_path):
    manifest, _ = _completed_run(
        tmp_path, "synthetic-manifest", [_market()], [_settlement()]
    )
    artifact = materialize_provisional_kalshi_labels(tmp_path, [manifest])

    artifact_manifest = json.loads(artifact.manifest_path.read_text())
    changed = deepcopy(artifact_manifest)
    changed["label_policy"]["settlement_verification"] = True
    artifact.manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ProvisionalKalshiLabelError, match="inputs changed"):
        verify_provisional_kalshi_labels(tmp_path, artifact.directory)
