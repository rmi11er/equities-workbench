import hashlib
import json

import pytest

from src.metals.storage import (
    HashChainedJsonlWriter,
    PartitionedRawStore,
    update_run_manifest,
    write_run_manifest,
)


def _expected_hash(record):
    payload = dict(record)
    payload.pop("record_sha256", None)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_hash_chained_writer_links_records(tmp_path):
    path = tmp_path / "segment.jsonl"
    with HashChainedJsonlWriter(path) as writer:
        first_hash = writer.write({"value": "1.2300"})
        second_hash = writer.write({"value": "2.3400"})

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["previous_record_sha256"] is None
    assert records[0]["record_sha256"] == first_hash == _expected_hash(records[0])
    assert records[1]["previous_record_sha256"] == first_hash
    assert records[1]["record_sha256"] == second_hash == _expected_hash(records[1])


def test_segment_creation_never_overwrites(tmp_path):
    path = tmp_path / "segment.jsonl"
    writer = HashChainedJsonlWriter(path)
    writer.close()

    with pytest.raises(FileExistsError):
        HashChainedJsonlWriter(path)


def test_partitioned_store_uses_utc_date_and_tier(tmp_path):
    store = PartitionedRawStore(
        tmp_path, run_id="safe-run", tier="normalized", fsync_every_records=1
    )
    store.write(
        source="market_dim",
        stream="market_dim",
        metal="silver",
        receive_ts_ns=1_788_000_000_000_000_000,
        record={"ticker": "TEST"},
    )
    store.close()

    path = store.paths[0]
    assert "normalized/market_dim/date=" in str(path)
    assert "metal=silver" in str(path)
    assert "run=safe-run" in str(path)


def test_run_manifest_tracks_atomic_lifecycle_and_redacts_secrets(tmp_path):
    path = write_run_manifest(
        tmp_path,
        "safe-run",
        {
            "endpoint": "https://example.test?token=hidden",
            "api_key": "do-not-store",
            "api_key_env": "PYTH_API_KEY",
        },
        counts={"records": 1},
    )

    running = json.loads(path.read_text())
    assert running["manifest_version"] == 2
    assert running["status"] == "running"
    assert running["started_ts"]
    assert running["completed_ts"] is None
    assert running["config"]["api_key"] == "[REDACTED]"
    assert running["config"]["api_key_env"] == "PYTH_API_KEY"
    canonical_config = json.dumps(
        running["config"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    assert running["config_sha256"] == hashlib.sha256(
        canonical_config.encode("utf-8")
    ).hexdigest()
    assert "hidden" not in path.read_text()

    output = tmp_path / "normalized" / "part.jsonl"
    update_run_manifest(
        tmp_path,
        "safe-run",
        counts={"records": 3, "markets": 2},
        output_paths=[output, output],
    )
    update_run_manifest(
        tmp_path,
        "safe-run",
        status="completed",
        errors=[],
    )

    completed = json.loads(path.read_text())
    assert completed["status"] == "completed"
    assert completed["completed_ts"]
    assert completed["counts"] == {"markets": 2, "records": 3}
    assert completed["output_paths"] == [str(output)]
    assert completed["errors"] == []
    assert list(path.parent.glob(".manifest-*.tmp")) == []


def test_run_manifest_records_sanitized_failure(tmp_path):
    path = write_run_manifest(tmp_path, "failed-run", {})
    update_run_manifest(
        tmp_path,
        "failed-run",
        status="failed",
        errors=[RuntimeError("Authorization: Bearer should-not-survive")],
    )

    manifest = json.loads(path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["errors"] == [
        {"message": "Authorization: [REDACTED]", "type": "RuntimeError"}
    ]
    assert "should-not-survive" not in path.read_text()


def test_run_manifest_preserves_create_only_compatibility(tmp_path):
    write_run_manifest(tmp_path, "one-run", {})
    with pytest.raises(FileExistsError):
        write_run_manifest(tmp_path, "one-run", {})
