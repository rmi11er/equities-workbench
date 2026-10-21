import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.metals.databento_history import (
    BillableDownloadNotAuthorized,
    DatabentoHistoricalAccessError,
    DatabentoHistoricalArchive,
    DatabentoHistoricalError,
    DatabentoHistoricalRequest,
)
from src.metals.historical_research_assumption import require_historical_operation
from src.metals.provider_approval import verify_provider_approval
from tests.metals.historical_research_assumption_fixtures import (
    materialize_historical_assumption_path,
)
from tests.metals.provider_approval_fixtures import (
    materialize_approved_provider_path,
)


@dataclass
class Header:
    instrument_id: int
    ts_event: int
    publisher_id: int = 1
    rtype: int = 0
    length: int = 48


class Definition:
    def __init__(self, instrument_id=42, symbol=b"SIU6\x00"):
        self.hd = Header(instrument_id, 1_777_000_000_000_000_000, rtype=19)
        self.ts_recv = 1_777_000_000_000_000_100
        self.raw_symbol = symbol
        self.asset = b"SI\x00"
        self.expiration = 1_780_000_000_000_000_000
        self.instrument_class = "F"


class Trade:
    def __init__(self, instrument_id=42):
        self.hd = Header(instrument_id, 1_777_000_000_000_000_020)
        self.ts_recv = 1_777_000_000_000_000_200
        self.ts_in_delta = 50
        self.price = 72_345_000_001
        self.size = 2
        self.sequence = 7
        self.action = "T"
        self.side = "A"


class FakeStore:
    def __init__(self, records, schema):
        self.records = list(records)
        self.metadata = {"schema": schema, "api_key": "must-not-persist"}
        self.symbology = {"symbols": ["SI.FUT"], "mappings": {}}

    def replay(self, callback):
        for record in self.records:
            callback(record)


class FakeMetadata:
    def __init__(self, costs=None, error=None):
        self.costs = costs or {}
        self.error = error
        self.calls = []

    def get_cost(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.costs.get(kwargs["schema"], 0.25)


class FakeTimeseries:
    def __init__(self, stores):
        self.stores = stores
        self.calls = []

    def get_range(self, **kwargs):
        self.calls.append(kwargs)
        Path(kwargs["path"]).write_bytes((kwargs["schema"] + "-native").encode())
        return self.stores[kwargs["schema"]]


class FakeBatch:
    def __init__(self, stores=None):
        self.submissions = []
        self.stores = stores or {}

    def submit_job(self, **kwargs):
        self.submissions.append(kwargs)
        return {
            "id": "JOB-{}".format(kwargs["schema"]),
            "state": "queued",
            "api_key": "must-not-persist",
            "user_id": "private-account",
        }

    def get_job_details(self, job_id):
        schema = job_id.removeprefix("JOB-")
        return {
            "id": job_id,
            "state": "done",
            "record_count": len(self.stores[schema].records),
            "cost_usd": 0.1,
            "api_key": "must-not-persist",
        }

    def list_files(self, job_id):
        schema = job_id.removeprefix("JOB-")
        contents = (schema + "-batch-native").encode()
        return [
            {
                "filename": "{}.dbn.zst".format(schema),
                "size": len(contents),
                "hash": "sha256:" + hashlib.sha256(contents).hexdigest(),
            }
        ]

    def download(self, job_id, output_dir):
        schema = job_id.removeprefix("JOB-")
        job_dir = Path(output_dir) / job_id
        job_dir.mkdir(parents=True)
        path = job_dir / "{}.dbn.zst".format(schema)
        path.write_bytes((schema + "-batch-native").encode())
        return [path]


class FakeClient:
    def __init__(self, stores, costs=None, metadata_error=None):
        self.metadata = FakeMetadata(costs, metadata_error)
        self.timeseries = FakeTimeseries(stores)
        self.batch = FakeBatch(stores)


def request(schemas=("definition", "trades")):
    return DatabentoHistoricalRequest(
        metal="silver",
        start=datetime(2026, 4, 25, tzinfo=timezone.utc),
        end=datetime(2026, 4, 26, tzinfo=timezone.utc),
        schemas=schemas,
    )


def stores():
    return {
        "definition": FakeStore([Definition()], "definition"),
        "trades": FakeStore([Trade()], "trades"),
        "mbo": FakeStore([], "mbo"),
        "status": FakeStore([], "status"),
        "statistics": FakeStore([], "statistics"),
    }


def test_request_requires_exact_bounds_known_metal_and_unique_nonempty_schemas():
    with pytest.raises(ValueError, match="strictly after"):
        DatabentoHistoricalRequest("silver", 10, 10)
    with pytest.raises(ValueError, match="timezone"):
        DatabentoHistoricalRequest("silver", datetime(2026, 1, 1), datetime(2026, 1, 2))
    with pytest.raises(ValueError, match="Unsupported metal"):
        DatabentoHistoricalRequest("platinum", 1, 2)
    assert DatabentoHistoricalRequest("silver", 1, 2, schemas=("trades",)).schemas == (
        "trades",
    )
    with pytest.raises(ValueError, match="At least one"):
        DatabentoHistoricalRequest("silver", 1, 2, schemas=())
    with pytest.raises(ValueError, match="unique"):
        DatabentoHistoricalRequest("silver", 1, 2, schemas=("trades", "TRADES"))


def test_schema_subset_is_canonical_and_part_of_request_fingerprint():
    lean = DatabentoHistoricalRequest(
        "silver", 1, 2, schemas=("statistics", "definition", "trades")
    )
    reordered = DatabentoHistoricalRequest(
        "silver", 1, 2, schemas=("trades", "statistics", "definition")
    )
    with_mbo = DatabentoHistoricalRequest(
        "silver", 1, 2, schemas=("definition", "mbo", "trades", "statistics")
    )

    assert lean.schemas == ("definition", "trades", "statistics")
    assert lean.request_id == reordered.request_id
    assert lean.request_id != with_mbo.request_id
    assert lean.public_dict()["schemas"] == ["definition", "trades", "statistics"]


def test_preflight_uses_parent_roots_and_refuses_download_without_explicit_flag(
    tmp_path,
):
    client = FakeClient(stores(), {"definition": 0.1, "trades": 1.2})
    archive = DatabentoHistoricalArchive(tmp_path, client=client)

    with pytest.raises(BillableDownloadNotAuthorized) as raised:
        archive.download_stream(request())

    assert raised.value.estimate.total_cost_usd == pytest.approx(1.3)
    assert raised.value.estimate.warnings == (
        "The requested range is inclusive at start and exclusive at end.",
        "Definition records may include UTC-day reference-data snapshots; snapshot "
        "emission is not an intraday instrument change.",
    )
    assert client.timeseries.calls == []
    assert {call["schema"] for call in client.metadata.calls} == {
        "definition",
        "trades",
    }
    assert all(
        call["symbols"] == ["SI.FUT", "SIL.FUT", "SIC.FUT"]
        for call in client.metadata.calls
    )
    assert all(call["stype_in"] == "parent" for call in client.metadata.calls)


def test_resume_rejects_changed_schema_set_before_any_additional_provider_call(
    tmp_path,
):
    client = FakeClient(stores(), {"definition": 0.1, "trades": 0.2, "status": 0.1})
    archive = DatabentoHistoricalArchive(tmp_path, client=client)
    original = request(("definition", "trades"))
    changed = request(("definition", "trades", "status"))
    with pytest.raises(BillableDownloadNotAuthorized):
        archive.download_stream(original)
    metadata_calls = len(client.metadata.calls)

    with pytest.raises(DatabentoHistoricalError, match="resume schema set"):
        archive.download_stream(changed, allow_billable_download=True)

    assert len(client.metadata.calls) == metadata_calls
    assert client.timeseries.calls == []


def test_stream_archives_native_normalizes_identity_and_resumes_without_api(tmp_path):
    fake_stores = stores()
    client = FakeClient(fake_stores, {"definition": 0.1, "trades": 0.2})
    approval = verify_provider_approval(
        tmp_path, materialize_approved_provider_path(tmp_path)
    )
    approval_id = approval.approval_id
    archive = DatabentoHistoricalArchive(
        tmp_path,
        client=client,
        dbn_loader=lambda path: fake_stores[path.name.split(".")[0]],
        provider_approval=approval,
    )
    current_request = request()

    result = archive.download_stream(
        current_request, allow_billable_download=True, max_cost_usd=1.0
    )

    assert result.completed is True
    assert result.record_counts == {"definition": 1, "trades": 1}
    assert len(client.timeseries.calls) == 2
    native = [path for path in result.output_paths if path.name == "trades.dbn.zst"]
    assert native and native[0].read_bytes() == b"trades-native"
    normalized_path = next(
        path
        for path in result.output_paths
        if path.name == "trades.jsonl" and "normalized" in path.parts
    )
    trade = json.loads(normalized_path.read_text().splitlines()[0])
    assert trade["price_raw"] == 72_345_000_001
    assert trade["contract_symbol"] == "SIU6"
    assert trade["contract_identity_status"] == "resolved_daily_definition"
    assert trade["historical_request_schemas"] == ["definition", "trades"]
    assert trade["vendor_timestamps_ns"] == {
        "ts_event": 1_777_000_000_000_000_020,
        "ts_in_delta": 50,
        "ts_recv": 1_777_000_000_000_000_200,
    }
    assert trade["source_publish_ts_ns"] == 1_777_000_000_000_000_150
    manifest_path = (
        tmp_path
        / "manifests"
        / f"run=databento-history-stream-{current_request.request_id[:16]}"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config"]["provider_approval_id"] == approval_id
    assert set(manifest["config"]["provider_approval_paths"]) == {
        str(approval.approval_path.resolve()),
        str(approval.manifest_path.resolve()),
    }
    assert {
        str(approval.approval_path.resolve()),
        str(approval.manifest_path.resolve()),
    }.issubset(manifest["output_paths"])

    calls_before_resume = len(client.timeseries.calls)
    estimates_before_resume = len(client.metadata.calls)
    resumed = archive.download_stream(current_request)
    assert resumed.completed is True
    assert set(resumed.skipped_schemas) == {"definition", "trades"}
    assert len(client.timeseries.calls) == calls_before_resume
    assert len(client.metadata.calls) == estimates_before_resume

    conflicting = DatabentoHistoricalArchive(
        tmp_path,
        client=client,
        dbn_loader=lambda path: fake_stores[path.name.split(".")[0]],
        provider_approval_id="b" * 64,
    )
    with pytest.raises(DatabentoHistoricalError, match="approval does not match"):
        conflicting.download_stream(current_request)
    assert len(client.timeseries.calls) == calls_before_resume
    assert len(client.metadata.calls) == estimates_before_resume


def test_stream_manifest_preserves_research_only_assumption_provenance(tmp_path):
    fake_stores = stores()
    client = FakeClient(fake_stores, {"definition": 0.1, "trades": 0.2})
    assumption_path = materialize_historical_assumption_path(
        tmp_path, provider="databento", operation="historical"
    )
    authority = require_historical_operation(
        tmp_path,
        "",
        assumption_path,
        "databento",
        "historical",
    )
    archive = DatabentoHistoricalArchive(
        tmp_path,
        client=client,
        dbn_loader=lambda path: fake_stores[path.name.split(".")[0]],
        historical_authority=authority,
    )
    current_request = request()

    result = archive.download_stream(
        current_request, allow_billable_download=True, max_cost_usd=1.0
    )
    manifest_path = (
        tmp_path
        / "manifests"
        / f"run=databento-history-stream-{current_request.request_id[:16]}"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    assert result.completed is True
    assert config["historical_research_assumption_id"] == authority.authority_id
    assert config["research_only"] is True
    assert config["no_live"] is True
    assert config["no_orders"] is True
    assert config["provider_approval_id"] is None
    assert str(assumption_path.resolve()) in manifest["output_paths"]


def test_manifests_and_metadata_do_not_persist_api_key_shaped_vendor_fields(tmp_path):
    secret = "db-secret-value"
    client = FakeClient(stores())
    archive = DatabentoHistoricalArchive(tmp_path, client=client)
    archive.download_stream(request(), allow_billable_download=True)

    contents = "\n".join(
        path.read_text(errors="ignore")
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert secret not in contents
    assert "must-not-persist" not in contents
    assert "api_key" not in contents.lower()


def test_entitlement_error_is_actionable_and_does_not_echo_vendor_message(tmp_path):
    class Forbidden(Exception):
        status_code = 403

    vendor_error = Forbidden("request included db-do-not-echo")
    archive = DatabentoHistoricalArchive(
        tmp_path, client=FakeClient(stores(), metadata_error=vendor_error)
    )

    with pytest.raises(DatabentoHistoricalAccessError) as raised:
        archive.preflight(request())

    assert "not entitled" in str(raised.value)
    assert "db-do-not-echo" not in str(raised.value)


def test_cost_preflight_rejects_empty_duplicate_or_out_of_request_schemas(tmp_path):
    archive = DatabentoHistoricalArchive(tmp_path, client=FakeClient(stores()))
    current_request = request()

    with pytest.raises(ValueError, match="At least one"):
        archive.preflight(current_request, ())
    with pytest.raises(ValueError, match="unique"):
        archive.preflight(current_request, ("trades", "TRADES"))
    with pytest.raises(ValueError, match="outside the bounded request"):
        archive.preflight(current_request, ("mbo",))


def test_batch_submission_is_cost_gated_resumable_and_whitelists_job_metadata(tmp_path):
    client = FakeClient(stores(), {"definition": 0.1, "trades": 0.2})
    archive = DatabentoHistoricalArchive(tmp_path, client=client)
    current_request = request()

    with pytest.raises(BillableDownloadNotAuthorized):
        archive.submit_batch(current_request)
    assert client.batch.submissions == []

    submitted = archive.submit_batch(
        current_request, allow_billable_download=True, max_cost_usd=1.0
    )
    assert submitted.completed is False
    assert [call["schema"] for call in client.batch.submissions] == [
        "definition",
        "trades",
    ]
    assert all(call["encoding"] == "dbn" for call in client.batch.submissions)
    assert all(call["compression"] == "zstd" for call in client.batch.submissions)
    checkpoint = submitted.output_paths[0].read_text()
    assert "must-not-persist" not in checkpoint
    assert "private-account" not in checkpoint
    assert "api_key" not in checkpoint

    submission_count = len(client.batch.submissions)
    resumed = archive.submit_batch(current_request)
    assert len(client.batch.submissions) == submission_count
    assert set(resumed.skipped_schemas) == {"definition", "trades"}


def test_batch_resume_rejects_checkpoint_with_different_schema_set(tmp_path):
    client = FakeClient(stores(), {"definition": 0.1, "trades": 0.2})
    archive = DatabentoHistoricalArchive(tmp_path, client=client)
    current_request = request()
    submitted = archive.submit_batch(
        current_request, allow_billable_download=True, max_cost_usd=1.0
    )
    checkpoint_path = submitted.output_paths[0]
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["request"]["schemas"] = ["definition", "mbo", "trades"]
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    submission_count = len(client.batch.submissions)

    with pytest.raises(DatabentoHistoricalError, match="schema set"):
        archive.submit_batch(current_request)

    assert len(client.batch.submissions) == submission_count


def test_completed_batch_downloads_native_dbn_verifies_hash_and_normalizes(tmp_path):
    fake_stores = stores()
    client = FakeClient(fake_stores)
    archive = DatabentoHistoricalArchive(
        tmp_path,
        client=client,
        dbn_loader=lambda path: fake_stores[path.name.split(".")[0]],
    )
    current_request = request()
    archive.submit_batch(current_request, allow_billable_download=True)

    result = archive.ingest_submitted_batch(current_request)

    assert result.completed is True
    assert result.record_counts == {"definition": 1, "trades": 1}
    assert result.batch_states == {"definition": "done", "trades": "done"}
    native = next(
        path
        for path in result.output_paths
        if path.name == "trades.dbn.zst" and "batch" in path.parts
    )
    assert native.read_bytes() == b"trades-batch-native"
    normalized = next(
        path
        for path in result.output_paths
        if path.name == "trades.jsonl" and "normalized" in path.parts
    )
    trade = json.loads(normalized.read_text().splitlines()[0])
    assert trade["contract_symbol"] == "SIU6"

    # Completed markers make the next resume local and idempotent.
    resumed = archive.ingest_submitted_batch(current_request)
    assert resumed.completed is True


def test_completion_marker_detects_tampering_and_rebuilds_from_native(tmp_path):
    fake_stores = stores()
    client = FakeClient(fake_stores)
    archive = DatabentoHistoricalArchive(
        tmp_path,
        client=client,
        dbn_loader=lambda path: fake_stores[path.name.split(".")[0]],
    )
    current_request = request()
    result = archive.download_stream(current_request, allow_billable_download=True)
    normalized_path = next(
        path
        for path in result.output_paths
        if path.name == "trades.jsonl" and "normalized" in path.parts
    )
    normalized_path.write_text("tampered\n")
    downloads = len(client.timeseries.calls)

    repaired = archive.download_stream(current_request)

    assert repaired.completed is True
    assert len(client.timeseries.calls) == downloads
    repaired_hash = hashlib.sha256(normalized_path.read_bytes()).hexdigest()
    marker = next(
        json.loads(path.read_text())
        for path in tmp_path.rglob("trades.stream.complete.json")
    )
    normalized_entry = next(
        item for item in marker["artifacts"] if item["path"].endswith("trades.jsonl")
    )
    assert repaired_hash == normalized_entry["sha256"]
