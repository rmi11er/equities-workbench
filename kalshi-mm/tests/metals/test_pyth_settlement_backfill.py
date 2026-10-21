import hashlib
import json
from pathlib import Path

import pytest

from src.metals.historical_research_assumption import (
    HistoricalResearchAssumptionError,
)
from src.metals.pyth_exact_history import PythExactHistoryAccessError
from src.metals.pyth_settlement_backfill import (
    PLAN_SCHEMA_VERSION,
    PythSettlementBackfillError,
    create_pyth_settlement_backfill_plan,
    run_pyth_settlement_backfill,
    verify_pyth_settlement_backfill_plan,
)
from src.metals.storage import HashChainedJsonlWriter
from tests.metals.provider_approval_fixtures import (
    materialize_approved_provider_path,
)

FEED_ID = 3_154
BOUNDARY_NS = 1_800_000_000_000_000_000
FIFTEEN_MINUTES_NS = 15 * 60 * 1_000_000_000


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode()


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_row(ticker, open_ns, close_ns, *, exclusions=None):
    return {
        "market_ticker": ticker,
        "metal": "silver",
        "pyth_feed_id": FEED_ID,
        "open_ts_ns": open_ns,
        "close_ts_ns": close_ns,
        "exclusion_reasons": exclusions
        if exclusions is not None
        else [
            "timestamp_semantics_unconfirmed",
            "correction_semantics_unconfirmed",
            "rounding_semantics_unconfirmed",
            "missing_opening_exact_history_coverage",
            "missing_settlement_exact_history_coverage",
        ],
    }


def _write_audit(tmp_path: Path, rows):
    root = tmp_path / "metals"
    identity = {
        "audit_schema_version": "metals-settlement-audit/v2",
        "audit_code_version": "settlement-audit-2",
        "metal": "silver",
        "exact_feed_id": FEED_ID,
        "semantics": {},
        "universe": None,
        "inputs": [],
    }
    audit_id = hashlib.sha256(_canonical(identity)).hexdigest()
    directory = root / "derived" / "settlement_check" / f"audit={audit_id}"
    directory.mkdir(parents=True)
    windows_path = directory / "windows.jsonl"
    with HashChainedJsonlWriter(windows_path) as writer:
        for row in rows:
            writer.write({"audit_id": audit_id, **row})
    summary_path = directory / "summary.json"
    summary_path.write_text(
        json.dumps({"audit_id": audit_id}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        **identity,
        "audit_id": audit_id,
        "outputs": [
            {
                "path": "windows.jsonl",
                "size_bytes": windows_path.stat().st_size,
                "sha256": _sha(windows_path),
                "records": len(rows),
            },
            {
                "path": "summary.json",
                "size_bytes": summary_path.stat().st_size,
                "sha256": _sha(summary_path),
            },
        ],
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root, directory


def _plan(tmp_path: Path):
    rows = [
        _audit_row("SILVER-A", BOUNDARY_NS, BOUNDARY_NS + FIFTEEN_MINUTES_NS),
        _audit_row(
            "SILVER-B",
            BOUNDARY_NS + FIFTEEN_MINUTES_NS,
            BOUNDARY_NS + 2 * FIFTEEN_MINUTES_NS,
        ),
        _audit_row(
            "SILVER-INVALID",
            BOUNDARY_NS + 3 * FIFTEEN_MINUTES_NS,
            BOUNDARY_NS + 4 * FIFTEEN_MINUTES_NS,
            exclusions=["missing_rules_primary"],
        ),
    ]
    root, audit = _write_audit(tmp_path, rows)
    artifact = create_pyth_settlement_backfill_plan(
        root,
        audit,
        metal="silver",
        feed_id=FEED_ID,
        base_url="https://history.example/v1",
        max_pages=2,
    )
    return root, audit, artifact


def test_plan_is_offline_deduplicated_content_addressed_and_source_bound(tmp_path):
    root, audit, artifact = _plan(tmp_path)

    assert artifact.value["schema_version"] == PLAN_SCHEMA_VERSION
    assert artifact.value["window_count"] == 3
    assert artifact.value["planned_window_count"] == 2
    assert artifact.value["boundary_count"] == 3
    assert artifact.value["maximum_provider_pages"] == 6
    assert artifact.value["invalid_windows"] == [
        {
            "market_ticker": "SILVER-INVALID",
            "reasons": ["non_exact_audit_exclusions:missing_rules_primary"],
        }
    ]
    middle = artifact.value["requests"][1]
    assert middle["uses"] == [
        {"boundary_role": "settlement", "market_ticker": "SILVER-A"},
        {"boundary_role": "opening", "market_ticker": "SILVER-B"},
    ]
    assert (
        middle["request"]["end_timestamp_us"] - middle["request"]["start_timestamp_us"]
        == 59_999_999
    )
    assert (
        middle["request"]["end_timestamp_us"] == middle["boundary_ts_ns"] // 1_000 - 1
    )

    repeated = create_pyth_settlement_backfill_plan(
        root,
        audit / "manifest.json",
        metal="silver",
        feed_id=FEED_ID,
        base_url="https://history.example/v1",
        max_pages=2,
    )
    assert repeated.plan_id == artifact.plan_id
    assert (
        verify_pyth_settlement_backfill_plan(root, artifact.plan_path).plan_id
        == artifact.plan_id
    )


def test_plan_tampering_is_rejected(tmp_path):
    root, _, artifact = _plan(tmp_path)
    value = json.loads(artifact.plan_path.read_text(encoding="utf-8"))
    value["boundary_count"] = 999
    artifact.plan_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        PythSettlementBackfillError, match="content hash|output changed"
    ):
        verify_pyth_settlement_backfill_plan(root, artifact.directory)


class _Response:
    def __init__(self, body, *, status=200):
        self.status = status
        self._body = body

    async def read(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url, *, params, headers):
        self.calls.append({"url": url, "params": params, "headers": headers})
        publish_time = params["end_timestamp"]
        body = json.dumps(
            {
                "data": [
                    {
                        "channel": 3,
                        "channel_name": "fixed_rate@200ms",
                        "price_feed_id": FEED_ID,
                        "publish_time": publish_time,
                        "price": 6_651_560,
                        "exponent": -5,
                        "feed_update_timestamp": publish_time,
                    }
                ]
            }
        ).encode()
        return _Response(body)


class _FailingSession(_Session):
    def get(self, url, *, params, headers):
        if len(self.calls) == 1:
            self.calls.append({"url": url, "params": params, "headers": headers})
            return _Response(b'{"error":"must-not-persist"}', status=403)
        return super().get(url, params=params, headers=headers)


@pytest.mark.asyncio
async def test_bounded_executor_skips_verified_ranges_and_resumes_next_boundary(
    tmp_path,
):
    root, _, artifact = _plan(tmp_path)
    approval_path = materialize_approved_provider_path(root)
    first_session = _Session()

    first = await run_pyth_settlement_backfill(
        root,
        artifact.plan_path,
        environ={"PYTH_API_KEY": "secret-that-must-not-persist"},
        allow_network_download=True,
        max_boundaries=2,
        request_interval_seconds=0,
        session=first_session,
        provider_approval_path=approval_path,
    )

    assert len(first_session.calls) == 2
    assert first.completed_before == 0
    assert first.attempted_boundaries == 2
    assert first.completed_this_run == 2
    assert first.completed_after == 2
    assert first.remaining_boundaries == 1
    assert first.provider_requests_made == 2
    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert first_manifest["status"] == "completed"
    assert first_manifest["config"]["collector"]["maximum_provider_pages_this_run"] == 4

    second_session = _Session()
    second = await run_pyth_settlement_backfill(
        root,
        artifact.directory,
        environ={"PYTH_API_KEY": "another-secret"},
        allow_network_download=True,
        max_boundaries=2,
        request_interval_seconds=0,
        session=second_session,
        provider_approval_path=approval_path,
    )

    assert len(second_session.calls) == 1
    assert second.completed_before == 2
    assert second.attempted_boundaries == 1
    assert second.completed_this_run == 1
    assert second.completed_after == 3
    assert second.remaining_boundaries == 0
    assert second.public_dict()["plan_completed"] is True
    assert b"secret-that-must-not-persist" not in b"\n".join(
        path.read_bytes() for path in root.rglob("*") if path.is_file()
    )


@pytest.mark.asyncio
async def test_executor_requires_explicit_network_acknowledgement(tmp_path):
    root, _, artifact = _plan(tmp_path)

    with pytest.raises(ValueError, match="allow-network-download"):
        await run_pyth_settlement_backfill(
            root,
            artifact.plan_path,
            environ={"PYTH_API_KEY": "secret"},
            session=_Session(),
        )

    assert not (root / "manifests").exists()


@pytest.mark.asyncio
async def test_executor_requires_frozen_approval_before_key_storage_or_http(tmp_path):
    root, _, artifact = _plan(tmp_path)
    session = _Session()

    with pytest.raises(HistoricalResearchAssumptionError, match="path is required"):
        await run_pyth_settlement_backfill(
            root,
            artifact.plan_path,
            environ={"PYTH_API_KEY": "must-not-be-used"},
            allow_network_download=True,
            max_boundaries=1,
            request_interval_seconds=0,
            session=session,
            provider_approval_path="",
        )

    assert not (root / "manifests").exists()
    assert session.calls == []


@pytest.mark.asyncio
async def test_failed_slice_manifest_keeps_incremental_call_and_completion_counts(
    tmp_path,
):
    root, _, artifact = _plan(tmp_path)
    approval_path = materialize_approved_provider_path(root)
    session = _FailingSession()

    with pytest.raises(PythExactHistoryAccessError, match="not entitled"):
        await run_pyth_settlement_backfill(
            root,
            artifact.plan_path,
            environ={"PYTH_API_KEY": "secret"},
            allow_network_download=True,
            max_boundaries=2,
            request_interval_seconds=0,
            session=session,
            provider_approval_path=approval_path,
        )

    manifests = list((root / "manifests").glob("run=*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["counts"]["attempted_boundaries"] == 2
    assert manifest["counts"]["completed_this_run"] == 1
    assert manifest["counts"]["completed_after"] == 1
    assert manifest["counts"]["remaining_boundaries"] == 2
    assert manifest["counts"]["provider_requests_made"] == 2
    assert manifest["config"]["collector"]["provider_approval_id"]
    assert {
        str(Path(approval_path)),
        str(Path(approval_path).parent / "manifest.json"),
    }.issubset(manifest["output_paths"])
    assert b"must-not-persist" not in b"\n".join(
        path.read_bytes() for path in root.rglob("*") if path.is_file()
    )
