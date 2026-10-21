import json
from pathlib import Path

import pytest

from src.metals.historical_research_assumption import (
    HistoricalResearchAssumptionError,
)
from src.metals.pyth_exact_history import (
    PYTH_EXACT_HISTORY_CHANNEL,
    PythExactHistoryAccessError,
    PythExactHistoryArchive,
    PythExactHistoryIntegrityError,
    PythExactHistoryProtocolError,
    PythExactHistoryRequest,
    backfill_pyth_exact_history,
    normalize_pyth_exact_price,
    run_pyth_exact_history,
)
from tests.metals.historical_research_assumption_fixtures import (
    materialize_historical_assumption_path,
)
from tests.metals.provider_approval_fixtures import (
    materialize_approved_provider_path,
)

START_US = 1_788_000_000_000_000
FEED_ID = 3_154


def _request(**changes):
    values = {
        "metal": "silver",
        "feed_ids": (FEED_ID,),
        "start_timestamp_us": START_US,
        "end_timestamp_us": START_US + 60_000_000,
        "limit": 1_000,
        "base_url": "https://history.example/v1",
    }
    values.update(changes)
    return PythExactHistoryRequest(**values)


def _price(
    *,
    feed_id=FEED_ID,
    publish_time=START_US,
    feed_update_timestamp=None,
    price=6_651_560,
    exponent=-5,
    channel_name=PYTH_EXACT_HISTORY_CHANNEL,
):
    if feed_update_timestamp is None:
        feed_update_timestamp = publish_time
    return {
        "channel": 3,
        "channel_name": channel_name,
        "price_feed_id": feed_id,
        "publish_time": publish_time,
        "price": price,
        "exponent": exponent,
        "feed_update_timestamp": feed_update_timestamp,
        "publisher_count": 7,
        "confidence": 2_000,
        "best_bid_price": 6_650_000,
        "best_ask_price": 6_653_000,
        "ema_price": None,
        "ema_confidence": None,
        "funding_rate": None,
        "funding_timestamp": None,
        "funding_rate_interval": None,
        "market_session": "Regular",
    }


def _body(rows, *, next_after_marker="absent"):
    payload = {"data": rows}
    if next_after_marker != "absent":
        payload["next"] = next_after_marker
    return json.dumps(payload, separators=(",", ":")).encode()


class _FakeResponse:
    def __init__(self, status, body):
        self.status = status
        self.body = body

    async def read(self):
        return self.body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _FakeSession:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.closed = False

    def get(self, url, *, params, headers):
        self.calls.append({"url": url, "params": params, "headers": headers})
        result = self.script.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    async def close(self):
        self.closed = True


class _NoRequestSession(_FakeSession):
    def __init__(self):
        super().__init__([])

    def get(self, url, *, params, headers):  # pragma: no cover - assertion path
        raise AssertionError("resume must not issue an HTTP request")


def _all_file_bytes(root: Path) -> bytes:
    return b"\n".join(path.read_bytes() for path in root.rglob("*") if path.is_file())


def test_request_is_numeric_bounded_inclusive_and_content_addressed():
    request = _request(feed_ids=(3_155, FEED_ID), end_timestamp_us=START_US)

    assert request.feed_ids == (FEED_ID, 3_155)
    assert request.query_params() == {
        "ids": [FEED_ID, 3_155],
        "start_timestamp": START_US,
        "end_timestamp": START_US,
        "limit": 1_000,
    }
    assert request.query_params("opaque+/==")["after"] == "opaque+/=="
    assert (
        request.request_id
        == _request(feed_ids=(FEED_ID, 3_155), end_timestamp_us=START_US).request_id
    )
    assert request.url.endswith("/fixed_rate@200ms/price/range")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"feed_ids": ()}, "At least one"),
        ({"feed_ids": (FEED_ID, FEED_ID)}, "unique"),
        ({"feed_ids": ("3154",)}, "numeric integer"),
        ({"feed_ids": (True,)}, "numeric integer"),
        ({"feed_ids": tuple(range(101))}, "at most 100"),
        (
            {"end_timestamp_us": START_US + 60_000_001},
            "may not exceed 60 seconds",
        ),
        ({"end_timestamp_us": START_US - 1}, "at or after"),
        ({"limit": 0}, "documented integer range"),
        ({"limit": 1_001}, "documented integer range"),
        ({"max_pages": 0}, "documented integer range"),
        ({"channel": "real_time"}, "locked"),
        ({"base_url": "http://history.example/v1"}, "HTTPS"),
        ({"base_url": "https://key@history.example/v1"}, "no credentials"),
    ],
)
def test_request_rejects_ambiguous_or_unbounded_inputs(changes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _request(**changes)


def test_normalizer_preserves_exact_integer_and_freshness_semantics():
    publish_time = START_US + 200_000
    row, key = normalize_pyth_exact_price(
        _price(
            publish_time=publish_time,
            feed_update_timestamp=START_US,
        ),
        request=_request(),
    )

    assert key == (FEED_ID, publish_time)
    assert row["price_mantissa"] == "6651560"
    assert row["exponent"] == -5
    assert row["price"] == "66.51560"
    assert row["publish_timestamp_us"] == publish_time
    assert row["feed_update_timestamp_us"] == START_US
    assert row["source_event_ts_ns"] == publish_time * 1_000
    assert row["source_publish_ts_ns"] == START_US * 1_000
    assert row["is_carried_forward"] is True
    assert row["freshness_age_us"] == 200_000


@pytest.mark.asyncio
async def test_paginated_download_preserves_exact_bodies_cursor_and_hash_chain(
    tmp_path,
):
    secret = "pyth-key-must-not-persist"
    cursor = "opaque+/cursor=="
    first_body = (
        b'{ "data" : ['
        + json.dumps(_price()).encode()
        + b'], "next" : "opaque+/cursor==" }\n'
    )
    second_body = _body(
        [_price(publish_time=START_US + 200_000)], next_after_marker=None
    )
    session = _FakeSession(
        [_FakeResponse(200, first_body), _FakeResponse(200, second_body)]
    )
    request = _request()
    archive = PythExactHistoryArchive(tmp_path, api_key=secret, session=session)

    result = await archive.download(request)

    assert result.completed is True
    assert result.page_count == 2
    assert result.row_count == 2
    assert result.requests_made == 2
    assert result.resumed_pages == 0
    assert result.raw_paths[0].read_bytes() == first_body
    assert result.raw_paths[1].read_bytes() == second_body
    assert session.calls[0]["params"]["ids"] == [FEED_ID]
    assert all(isinstance(value, int) for value in session.calls[0]["params"]["ids"])
    assert "after" not in session.calls[0]["params"]
    assert session.calls[1]["params"]["after"] == cursor
    assert session.calls[0]["headers"]["Accept-Encoding"] == "identity"

    first_row = json.loads(result.normalized_paths[0].read_text().splitlines()[0])
    second_row = json.loads(result.normalized_paths[1].read_text().splitlines()[0])
    assert first_row["previous_record_sha256"] is None
    assert second_row["previous_record_sha256"] == first_row["record_sha256"]
    assert first_row["raw_page_sha256"] != second_row["raw_page_sha256"]
    checkpoint = json.loads(result.checkpoint_path.read_text())
    assert checkpoint["completed"] is True
    assert checkpoint["next_after"] is None
    assert checkpoint["page_count"] == 2
    assert checkpoint["row_count"] == 2
    assert secret.encode() not in _all_file_bytes(tmp_path)
    assert result.public_dict()["request_id"] == request.request_id


@pytest.mark.asyncio
async def test_complete_resume_is_local_and_reverifies_all_artifacts(tmp_path):
    request = _request()
    first_session = _FakeSession([_FakeResponse(200, _body([_price()]))])
    await PythExactHistoryArchive(
        tmp_path, api_key="secret", session=first_session
    ).download(request)

    result = await PythExactHistoryArchive(
        tmp_path, api_key="another-secret", session=_NoRequestSession()
    ).download(request)

    assert result.completed is True
    assert result.requests_made == 0
    assert result.resumed_pages == 1


@pytest.mark.asyncio
async def test_prepare_resume_is_local_and_distinguishes_absent_from_complete(tmp_path):
    request = _request()
    archive = PythExactHistoryArchive(
        tmp_path,
        api_key="secret",
        session=_FakeSession([_FakeResponse(200, _body([_price()]))]),
    )

    assert archive.prepare_resume(request) is None
    await archive.download(request)
    prepared = archive.prepare_resume(request)

    assert prepared is not None
    assert prepared.completed is True
    assert prepared.requests_made == 0


@pytest.mark.asyncio
async def test_resume_continues_from_last_opaque_cursor_after_http_failure(tmp_path):
    request = _request()
    first_session = _FakeSession(
        [
            _FakeResponse(200, _body([_price()], next_after_marker="next-token")),
            _FakeResponse(503, b'provider failure with "token":"do-not-save"'),
        ]
    )
    with pytest.raises(PythExactHistoryAccessError, match="unavailable"):
        await PythExactHistoryArchive(
            tmp_path, api_key="secret", session=first_session
        ).download(request)

    second_session = _FakeSession(
        [
            _FakeResponse(
                200,
                _body([_price(publish_time=START_US + 200_000)]),
            )
        ]
    )
    result = await PythExactHistoryArchive(
        tmp_path, api_key="secret", session=second_session
    ).download(request)

    assert result.completed is True
    assert result.resumed_pages == 1
    assert result.requests_made == 1
    assert second_session.calls[0]["params"]["after"] == "next-token"
    assert b"do-not-save" not in _all_file_bytes(tmp_path)


@pytest.mark.asyncio
async def test_stale_checkpoint_is_recovered_only_from_verified_page_artifacts(
    tmp_path,
):
    request = _request()
    partial_session = _FakeSession(
        [
            _FakeResponse(200, _body([_price()], next_after_marker="cursor-two")),
            _FakeResponse(503, b"discarded"),
        ]
    )
    archive = PythExactHistoryArchive(
        tmp_path, api_key="secret", session=partial_session
    )
    with pytest.raises(PythExactHistoryAccessError):
        await archive.download(request)
    stale_checkpoint = archive._checkpoint_path(request).read_bytes()

    await PythExactHistoryArchive(
        tmp_path,
        api_key="secret",
        session=_FakeSession(
            [_FakeResponse(200, _body([_price(publish_time=START_US + 200_000)]))]
        ),
    ).download(request)
    archive._checkpoint_path(request).write_bytes(stale_checkpoint)

    recovered = await PythExactHistoryArchive(
        tmp_path, api_key="secret", session=_NoRequestSession()
    ).download(request)

    assert recovered.completed is True
    assert recovered.resumed_pages == 2
    assert recovered.requests_made == 0


@pytest.mark.asyncio
async def test_raw_only_interrupted_page_rebuilds_projection_and_checkpoint(tmp_path):
    request = _request()
    archive = PythExactHistoryArchive(
        tmp_path,
        api_key="secret",
        session=_FakeSession([_FakeResponse(503, b"discarded")]),
    )
    with pytest.raises(PythExactHistoryAccessError):
        await archive.download(request)

    raw_body = _body([_price()])
    raw_path = archive._raw_path(request, 0)
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(raw_body)

    recovered = await PythExactHistoryArchive(
        tmp_path, api_key="secret", session=_NoRequestSession()
    ).download(request)

    assert recovered.completed is True
    assert recovered.resumed_pages == 1
    assert recovered.requests_made == 0
    assert recovered.normalized_paths[0].is_file()
    assert json.loads(recovered.checkpoint_path.read_text())["page_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404, 429, 500])
async def test_http_errors_are_classified_without_persisting_provider_body(
    tmp_path, status
):
    provider_body = b'{"error":"provider-secret-body"}'
    session = _FakeSession([_FakeResponse(status, provider_body)])

    with pytest.raises(PythExactHistoryAccessError) as raised:
        await PythExactHistoryArchive(
            tmp_path, api_key="api-key-secret", session=session
        ).download(_request())

    if status == 403:
        assert "not entitled" in str(raised.value)
    assert "provider-secret-body" not in str(raised.value)
    assert provider_body not in _all_file_bytes(tmp_path)
    assert b"api-key-secret" not in _all_file_bytes(tmp_path)


@pytest.mark.asyncio
async def test_transport_error_text_is_not_exposed_or_persisted(tmp_path):
    session = _FakeSession([RuntimeError("provider URL has secret-query-value")])

    with pytest.raises(PythExactHistoryAccessError) as raised:
        await PythExactHistoryArchive(
            tmp_path, api_key="api-key-secret", session=session
        ).download(_request())

    assert "secret-query-value" not in str(raised.value)
    assert b"secret-query-value" not in _all_file_bytes(tmp_path)
    assert b"api-key-secret" not in _all_file_bytes(tmp_path)


@pytest.mark.asyncio
async def test_cli_hook_requires_key_before_storage_or_http(tmp_path):
    session = _NoRequestSession()
    approval_path = materialize_approved_provider_path(tmp_path)

    with pytest.raises(PythExactHistoryAccessError, match="TEST_PYTH_KEY"):
        await backfill_pyth_exact_history(
            tmp_path,
            _request(),
            api_key_env="TEST_PYTH_KEY",
            environ={},
            session=session,
            provider_approval_path=approval_path,
        )

    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "manifests").exists()
    assert session.calls == []


@pytest.mark.asyncio
async def test_cli_hook_requires_frozen_approval_before_key_storage_or_http(tmp_path):
    session = _NoRequestSession()

    with pytest.raises(HistoricalResearchAssumptionError, match="path is required"):
        await backfill_pyth_exact_history(
            tmp_path,
            _request(),
            api_key_env="TEST_PYTH_KEY",
            environ={"TEST_PYTH_KEY": "must-not-be-used"},
            session=session,
            provider_approval_path="",
        )

    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "manifests").exists()
    assert session.calls == []


@pytest.mark.asyncio
async def test_run_wrapper_writes_completed_secret_safe_manifest(tmp_path):
    approval_path = materialize_approved_provider_path(tmp_path)
    result = await run_pyth_exact_history(
        tmp_path,
        _request(),
        api_key_env="TEST_PYTH_KEY",
        environ={"TEST_PYTH_KEY": "manifest-secret"},
        session=_FakeSession([_FakeResponse(200, _body([_price()]))]),
        provider_approval_path=approval_path,
    )

    manifest = json.loads(result.manifest_path.read_text())
    assert result.archive.completed is True
    assert manifest["status"] == "completed"
    assert manifest["counts"]["rows"] == 1
    assert manifest["config"]["collector"]["provider_approval_id"]
    assert set(manifest["output_paths"]) == {
        str(tmp_path / artifact["path"])
        for page in json.loads(result.archive.checkpoint_path.read_text())["pages"]
        for artifact in (page["raw"], page["normalized"])
    } | {
        str(result.archive.checkpoint_path),
        str(
            tmp_path
            / "checkpoints"
            / "pyth_exact_history"
            / f"request={result.archive.request_id}"
            / "request.json"
        ),
        str(Path(approval_path)),
        str(Path(approval_path).parent / "manifest.json"),
    }
    assert b"manifest-secret" not in _all_file_bytes(tmp_path)


@pytest.mark.asyncio
async def test_run_wrapper_accepts_research_assumption_but_marks_it_ineligible(
    tmp_path,
):
    assumption_path = materialize_historical_assumption_path(tmp_path)
    result = await run_pyth_exact_history(
        tmp_path,
        _request(),
        api_key_env="TEST_PYTH_KEY",
        environ={"TEST_PYTH_KEY": "manifest-secret"},
        session=_FakeSession([_FakeResponse(200, _body([_price()]))]),
        historical_research_assumption_path=assumption_path,
    )

    manifest = json.loads(result.manifest_path.read_text())
    collector = manifest["config"]["collector"]
    assert collector["historical_research_assumption_id"]
    assert collector["research_only"] is True
    assert collector["no_live"] is True
    assert collector["no_orders"] is True
    assert "provider_approval_id" not in collector
    assert str(assumption_path) in manifest["output_paths"]


@pytest.mark.asyncio
async def test_run_wrapper_records_sanitized_failure_manifest(tmp_path):
    approval_path = materialize_approved_provider_path(tmp_path)
    with pytest.raises(PythExactHistoryAccessError):
        await run_pyth_exact_history(
            tmp_path,
            _request(),
            api_key_env="TEST_PYTH_KEY",
            environ={"TEST_PYTH_KEY": "manifest-secret"},
            session=_FakeSession([_FakeResponse(403, b"provider-private-body")]),
            provider_approval_path=approval_path,
        )

    manifests = list((tmp_path / "manifests").glob("**/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "failed"
    assert manifest["errors"][0]["type"] == "PythExactHistoryAccessError"
    assert b"manifest-secret" not in _all_file_bytes(tmp_path)
    assert b"provider-private-body" not in _all_file_bytes(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"[]",
        b'{"data":{}}',
        b'{"data":[],"next":"cursor"}',
        b'{"data":[],"next":7}',
        b'{"data":[],"data":[]}',
        _body([_price(price=66.5)]),
        _body([_price(price=True)]),
        _body([_price(feed_id=99)]),
        _body([_price(publish_time=START_US - 1)]),
        _body(
            [
                _price(
                    publish_time=START_US,
                    feed_update_timestamp=START_US + 1,
                )
            ]
        ),
        _body([_price(channel_name="real_time")]),
        _body([_price(), _price()]),
    ],
)
async def test_malformed_nonmatching_or_duplicate_page_fails_before_raw_persist(
    tmp_path, body
):
    session = _FakeSession([_FakeResponse(200, body)])
    archive = PythExactHistoryArchive(tmp_path, api_key="secret", session=session)

    with pytest.raises(PythExactHistoryProtocolError):
        await archive.download(_request())

    assert not list((tmp_path / "raw").glob("**/*.body.json"))


@pytest.mark.asyncio
async def test_duplicate_row_across_pages_and_cursor_cycle_fail_closed(tmp_path):
    request = _request()
    duplicate_session = _FakeSession(
        [
            _FakeResponse(200, _body([_price()], next_after_marker="cursor-a")),
            _FakeResponse(200, _body([_price()])),
        ]
    )

    with pytest.raises(PythExactHistoryProtocolError, match="duplicate"):
        await PythExactHistoryArchive(
            tmp_path / "duplicate", api_key="secret", session=duplicate_session
        ).download(request)
    assert len(list((tmp_path / "duplicate" / "raw").glob("**/*.body.json"))) == 1

    cycle_session = _FakeSession(
        [
            _FakeResponse(200, _body([_price()], next_after_marker="cursor-a")),
            _FakeResponse(
                200,
                _body(
                    [_price(publish_time=START_US + 200_000)],
                    next_after_marker="cursor-a",
                ),
            ),
        ]
    )
    with pytest.raises(PythExactHistoryProtocolError, match="cursor cycle"):
        await PythExactHistoryArchive(
            tmp_path / "cycle", api_key="secret", session=cycle_session
        ).download(request)
    assert len(list((tmp_path / "cycle" / "raw").glob("**/*.body.json"))) == 1


@pytest.mark.asyncio
async def test_max_pages_is_a_hard_pagination_bound(tmp_path):
    request = _request(max_pages=1)
    session = _FakeSession(
        [_FakeResponse(200, _body([_price()], next_after_marker="more"))]
    )

    with pytest.raises(PythExactHistoryProtocolError, match="max_pages"):
        await PythExactHistoryArchive(
            tmp_path, api_key="secret", session=session
        ).download(request)

    assert len(session.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact", ["raw", "normalized", "checkpoint", "request"])
async def test_resume_fails_closed_on_archived_tampering(tmp_path, artifact):
    request = _request()
    archive = PythExactHistoryArchive(
        tmp_path,
        api_key="secret",
        session=_FakeSession([_FakeResponse(200, _body([_price()]))]),
    )
    result = await archive.download(request)
    paths = {
        "raw": result.raw_paths[0],
        "normalized": result.normalized_paths[0],
        "checkpoint": result.checkpoint_path,
        "request": archive._request_path(request),
    }
    paths[artifact].write_bytes(paths[artifact].read_bytes() + b"tampered")

    with pytest.raises(PythExactHistoryIntegrityError):
        await PythExactHistoryArchive(
            tmp_path, api_key="secret", session=_NoRequestSession()
        ).download(request)


@pytest.mark.asyncio
async def test_checkpointed_missing_normalized_page_is_tampering(tmp_path):
    request = _request()
    result = await PythExactHistoryArchive(
        tmp_path,
        api_key="secret",
        session=_FakeSession([_FakeResponse(200, _body([_price()]))]),
    ).download(request)
    result.normalized_paths[0].unlink()

    with pytest.raises(PythExactHistoryIntegrityError, match="artifact is missing"):
        await PythExactHistoryArchive(
            tmp_path, api_key="secret", session=_NoRequestSession()
        ).download(request)
