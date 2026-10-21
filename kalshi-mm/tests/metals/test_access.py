import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.metals import access
from src.metals.access import (
    preflight_databento,
    preflight_databento_live,
    preflight_kalshi,
    preflight_pyth,
    preflight_pyth_lazer_live,
)
from src.metals.config import CmeConfig, MetalsConfig, ProductConfig, PythConfig

SECRET = "secret-that-must-never-appear"


def _config():
    return MetalsConfig(
        pyth=PythConfig(api_key_env="TEST_PYTH_KEY"),
        cme=CmeConfig(api_key_env="TEST_DATABENTO_KEY"),
        products={
            "silver": ProductConfig(
                enabled=True,
                pyth_symbol="Metal.Index.SILVER/USD",
                pyth_lazer_id=3154,
                cme_symbols=["SI", "SIL", "SIC"],
            )
        },
    )


class _FakeResponse:
    def __init__(self, status):
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _FakeSession:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.requests = []

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        return _FakeResponse(self.statuses.pop(0))


class _FakeKalshiClient:
    def __init__(self, errors=None):
        self.errors = dict(errors or {})
        self.paths = []

    async def get(self, path):
        self.paths.append(path)
        error = self.errors.get(path)
        if error is not None:
            raise error
        return {"sensitive_account_payload": SECRET}


@pytest.mark.asyncio
async def test_kalshi_preflight_is_bounded_and_discards_account_payloads():
    client = _FakeKalshiClient()

    report = await preflight_kalshi(_config(), client=client)

    assert report["ok"] is True
    assert report["credential_status"] == "accepted"
    assert client.paths == ["/account/limits", "/account/endpoint_costs"]
    assert SECRET not in json.dumps(report)


@pytest.mark.asyncio
async def test_kalshi_preflight_sanitizes_authentication_failures():
    class RejectedCredential(Exception):
        status = 401

    client = _FakeKalshiClient(
        {
            "/account/limits": RejectedCredential("body " + SECRET),
            "/account/endpoint_costs": RejectedCredential("body " + SECRET),
        }
    )

    report = await preflight_kalshi(_config(), client=client)

    assert report["ok"] is False
    assert report["credential_status"] == "rejected"
    assert {check["reason"] for check in report["checks"]} == {"credential_rejected"}
    assert SECRET not in json.dumps(report)


@pytest.mark.asyncio
async def test_pyth_preflight_is_bounded_and_secret_safe():
    session = _FakeSession([200, 200])

    report = await preflight_pyth(
        _config(),
        metal="silver",
        history_at_seconds=1_780_000_000,
        session=session,
        environ={"TEST_PYTH_KEY": SECRET},
    )

    assert report["ok"] is True
    assert report["credential_status"] == "accepted"
    assert [check["name"] for check in report["checks"]] == [
        "history_symbol",
        "lazer_feed",
    ]
    assert len(session.requests) == 2
    history_request = session.requests[0]
    assert history_request[0] == "GET"
    assert history_request[2]["params"]["symbol"] == "Metal.Index.SILVER/USD"
    assert (
        history_request[2]["params"]["to"] - history_request[2]["params"]["from"] + 1
        == access.MAX_PYTH_HISTORY_WINDOW_SECONDS
    )
    lazer_request = session.requests[1]
    assert lazer_request[0] == "POST"
    assert lazer_request[2]["json"]["priceFeedIds"] == [3154]
    assert SECRET not in json.dumps(report)


@pytest.mark.asyncio
async def test_pyth_preflight_distinguishes_bad_credentials_and_entitlements():
    rejected = await preflight_pyth(
        _config(),
        metal="silver",
        history_at_seconds=1_780_000_000,
        session=_FakeSession([401, 401]),
        environ={"TEST_PYTH_KEY": SECRET},
    )
    denied = await preflight_pyth(
        _config(),
        metal="silver",
        history_at_seconds=1_780_000_000,
        session=_FakeSession([200, 403]),
        environ={"TEST_PYTH_KEY": SECRET},
    )

    assert rejected["credential_status"] == "rejected"
    assert {check["reason"] for check in rejected["checks"]} == {"credential_rejected"}
    assert denied["credential_status"] == "accepted"
    assert denied["checks"][1]["reason"] == "entitlement_denied"
    assert denied["ok"] is False


class _LiveWebSocket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def send(self, value):
        self.sent.append(value)

    async def recv(self):
        if self.messages:
            return self.messages.pop(0)
        await asyncio.Event().wait()


class _LiveConnect:
    def __init__(self, scripts=None, error=None):
        self.scripts = scripts or {}
        self.error = error
        self.calls = []
        self.websockets = []

    def __call__(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        if self.error is not None:
            raise self.error
        websocket = _LiveWebSocket(self.scripts.get(endpoint, []))
        self.websockets.append(websocket)
        return websocket


def _pyth_live_update():
    return {
        "type": "streamUpdated",
        "subscriptionId": 1,
        "parsed": {
            "timestampUs": "1788000000200000",
            "priceFeeds": [
                {
                    "priceFeedId": 3154,
                    "price": "6651560",
                    "feedUpdateTimestamp": 1788000000000000,
                    "exponent": -5,
                    "confidence": "2000",
                    "publisherCount": 7,
                    "marketSession": "regular",
                }
            ],
        },
    }


@pytest.mark.asyncio
async def test_pyth_live_preflight_checks_every_endpoint_without_retention():
    endpoints = ("wss://pyth-0.example/stream", "wss://pyth-1.example/stream")
    config = _config()
    config = replace(
        config,
        pyth=replace(config.pyth, lazer_ws_endpoints=list(endpoints)),
    )
    scripts = {
        endpoint: [
            json.dumps({"type": "subscribed", "subscriptionId": 1}),
            json.dumps(_pyth_live_update()),
        ]
        for endpoint in endpoints
    }
    connect = _LiveConnect(scripts)

    report = await preflight_pyth_lazer_live(
        config,
        metal="silver",
        duration_seconds=0.1,
        allow_live_probe=True,
        environ={"TEST_PYTH_KEY": SECRET},
        connect_factory=connect,
    )

    assert report["ok"] is True
    assert report["credential_status"] == "accepted"
    assert report["retained_messages"] == 0
    assert len(report["checks"]) == len(endpoints)
    assert all(check["status"] == "passed" for check in report["checks"])
    assert all(websocket.sent for websocket in connect.websockets)
    assert SECRET not in json.dumps(report)


@pytest.mark.asyncio
async def test_pyth_live_preflight_sanitizes_endpoint_failure():
    class Forbidden(Exception):
        status_code = 403

    report = await preflight_pyth_lazer_live(
        _config(),
        metal="silver",
        duration_seconds=0.01,
        allow_live_probe=True,
        environ={"TEST_PYTH_KEY": SECRET},
        connect_factory=_LiveConnect(error=Forbidden("body " + SECRET)),
    )

    assert report["ok"] is False
    assert {check["reason"] for check in report["checks"]} == {"entitlement_denied"}
    assert SECRET not in json.dumps(report)


@pytest.mark.asyncio
async def test_pyth_live_preflight_timeout_is_not_a_pass():
    report = await preflight_pyth_lazer_live(
        _config(),
        metal="silver",
        duration_seconds=0.001,
        allow_live_probe=True,
        environ={"TEST_PYTH_KEY": SECRET},
        connect_factory=_LiveConnect(),
    )

    assert report["ok"] is False
    assert all(check["status"] == "not_checked" for check in report["checks"])


@pytest.mark.asyncio
async def test_pyth_live_preflight_requires_explicit_acknowledgement():
    with pytest.raises(ValueError, match="allow_live_probe"):
        await preflight_pyth_lazer_live(
            _config(),
            metal="silver",
            environ={"TEST_PYTH_KEY": SECRET},
            connect_factory=_LiveConnect(),
        )


class _Metadata:
    def list_datasets(self):
        return ["GLBX.MDP3"]

    def list_schemas(self, dataset):
        assert dataset == "GLBX.MDP3"
        return ["mbo", "trades", "definition", "status", "statistics"]

    def get_dataset_range(self, dataset):
        assert dataset == "GLBX.MDP3"
        return {
            "start": "2010-06-06T00:00:00Z",
            "end": "2026-08-29T00:00:00Z",
            "schema": {
                schema: {
                    "start": "2017-05-21T00:00:00Z",
                    "end": "2026-08-29T00:00:00Z",
                }
                for schema in ("mbo", "trades", "definition", "status", "statistics")
            },
        }


class _Symbology:
    def __init__(self):
        self.kwargs = None

    def resolve(self, **kwargs):
        self.kwargs = kwargs
        return {"result": {}, "partial": [], "not_found": []}


def _databento_client():
    return SimpleNamespace(metadata=_Metadata(), symbology=_Symbology())


def test_databento_preflight_uses_only_metadata_and_symbology():
    client = _databento_client()

    report = preflight_databento(
        _config(),
        metal="silver",
        probe_date="2026-08-28",
        environ={"TEST_DATABENTO_KEY": SECRET},
        client=client,
    )

    assert report["ok"] is True
    assert report["credential_status"] == "accepted"
    assert report["scope"] == "historical_metadata_only"
    assert report["checks"][-1] == {
        "name": "live_dataset_entitlement",
        "status": "not_checked",
        "reason": "requires_bounded_live_stream_probe",
    }
    assert client.symbology.kwargs["symbols"] == ["SI.FUT", "SIL.FUT", "SIC.FUT"]
    assert client.symbology.kwargs["start_date"] == "2026-08-28"
    assert client.symbology.kwargs["end_date"] == "2026-08-29"
    assert not hasattr(client, "timeseries")
    assert SECRET not in json.dumps(report)


class _FakeLiveCollector:
    def __init__(self, *, record=None, error=None):
        self.record = record
        self.error = error
        self.started = False
        self.stopped = False
        self.close_timeout = None

    def start_callbacks(self, callback, exception_callback=None):
        self.started = True
        if self.error is not None:
            exception_callback(self.error)
        elif self.record is not None:
            callback(self.record)

    def stop(self):
        self.stopped = True

    def block_for_close(self, timeout=None):
        self.close_timeout = timeout


def test_databento_live_preflight_proves_entitlement_without_retaining_record():
    collector = _FakeLiveCollector(record={"provider_body": SECRET})

    report = preflight_databento_live(
        _config(),
        metal="silver",
        duration_seconds=0.01,
        allow_live_probe=True,
        environ={"TEST_DATABENTO_KEY": SECRET},
        collector=collector,
    )

    assert report["ok"] is True
    assert report["credential_status"] == "accepted"
    assert report["records_observed"] == 1
    assert report["retained_records"] == 0
    assert report["checks"] == [
        {
            "name": "live_mbo_snapshot",
            "status": "passed",
            "reason": "record_received_and_discarded",
        }
    ]
    assert collector.started and collector.stopped
    assert collector.close_timeout == 2.0
    assert SECRET not in json.dumps(report)


def test_databento_live_preflight_sanitizes_entitlement_failure():
    class Forbidden(Exception):
        status_code = 403

    collector = _FakeLiveCollector(error=Forbidden("body " + SECRET))
    report = preflight_databento_live(
        _config(),
        metal="silver",
        duration_seconds=0.01,
        allow_live_probe=True,
        environ={"TEST_DATABENTO_KEY": SECRET},
        collector=collector,
    )

    assert report["ok"] is False
    assert report["checks"][0]["reason"] == "entitlement_denied"
    assert SECRET not in json.dumps(report)


def test_databento_live_preflight_is_indeterminate_without_a_record():
    report = preflight_databento_live(
        _config(),
        metal="silver",
        duration_seconds=0.001,
        allow_live_probe=True,
        environ={"TEST_DATABENTO_KEY": SECRET},
        collector=_FakeLiveCollector(),
    )

    assert report["ok"] is False
    assert report["records_observed"] == 0
    assert report["checks"][0]["status"] == "not_checked"


@pytest.mark.parametrize("duration", [0, -1, 31, float("nan"), float("inf")])
def test_databento_live_preflight_rejects_unbounded_duration(duration):
    with pytest.raises(ValueError, match="finite"):
        preflight_databento_live(
            _config(),
            metal="silver",
            duration_seconds=duration,
            allow_live_probe=True,
            environ={"TEST_DATABENTO_KEY": SECRET},
            collector=_FakeLiveCollector(),
        )


def test_databento_live_preflight_requires_explicit_acknowledgement():
    with pytest.raises(ValueError, match="allow_live_probe"):
        preflight_databento_live(
            _config(),
            metal="silver",
            duration_seconds=0.01,
            environ={"TEST_DATABENTO_KEY": SECRET},
            collector=_FakeLiveCollector(record={}),
        )


def test_databento_preflight_distinguishes_optional_dependency_absence(monkeypatch):
    def missing_dependency(name):
        assert name == "databento"
        raise ImportError("not installed")

    monkeypatch.setattr(access.importlib, "import_module", missing_dependency)

    report = preflight_databento(
        _config(),
        metal="silver",
        probe_date="2026-08-28",
        environ={"TEST_DATABENTO_KEY": SECRET},
    )

    assert report["credential_status"] == "unknown"
    assert report["checks"] == [
        {
            "name": "python_dependency",
            "status": "failed",
            "reason": "optional_dependency_missing",
            "dependency": "databento",
        }
    ]
    assert report["ok"] is False


def test_databento_preflight_does_not_expose_provider_error_text():
    class FakeAuthError(Exception):
        status_code = 401

    class MetadataWithRejectedKey(_Metadata):
        def list_datasets(self):
            raise FakeAuthError("response body included " + SECRET)

    client = SimpleNamespace(metadata=MetadataWithRejectedKey(), symbology=_Symbology())

    report = preflight_databento(
        _config(),
        metal="silver",
        probe_date="2026-08-28",
        environ={"TEST_DATABENTO_KEY": SECRET},
        client=client,
    )

    assert report["credential_status"] == "rejected"
    assert report["checks"][-1]["reason"] == "credential_rejected"
    assert SECRET not in json.dumps(report)


def test_databento_preflight_reports_missing_schema_entitlement():
    class MetadataMissingMbo(_Metadata):
        def get_dataset_range(self, dataset):
            response = super().get_dataset_range(dataset)
            del response["schema"]["mbo"]
            return response

    client = SimpleNamespace(metadata=MetadataMissingMbo(), symbology=_Symbology())

    report = preflight_databento(
        _config(),
        metal="silver",
        probe_date="2026-08-28",
        environ={"TEST_DATABENTO_KEY": SECRET},
        client=client,
    )

    schema_check = next(
        check
        for check in report["checks"]
        if check["name"] == "required_schema_entitlements"
    )
    assert schema_check["status"] == "failed"
    assert schema_check["missing_schemas"] == ["mbo"]
    assert schema_check["unavailable_on_probe_date"] == []
    assert report["ok"] is False


def test_databento_preflight_checks_probe_date_against_schema_ranges():
    report = preflight_databento(
        _config(),
        metal="silver",
        probe_date="2016-08-28",
        environ={"TEST_DATABENTO_KEY": SECRET},
        client=_databento_client(),
    )

    schema_check = next(
        check
        for check in report["checks"]
        if check["name"] == "required_schema_entitlements"
    )
    assert schema_check["status"] == "failed"
    assert schema_check["missing_schemas"] == []
    assert schema_check["unavailable_on_probe_date"] == [
        "mbo",
        "trades",
        "definition",
        "status",
        "statistics",
    ]
    assert report["ok"] is False


def test_preflight_rejects_malformed_or_unbounded_inputs_before_requests():
    with pytest.raises(ValueError, match="ISO date"):
        preflight_databento(
            _config(),
            metal="silver",
            probe_date="not-a-date",
            environ={"TEST_DATABENTO_KEY": SECRET},
            client=_databento_client(),
        )
