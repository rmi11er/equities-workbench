"""Secret-safe, read-only source-access preflight checks.

The functions in this module deliberately make only bounded metadata or single-price
requests.  Reports contain status classifications, never credentials, response bodies,
third-party exception messages, or account objects.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from typing import Any

import aiohttp
import websockets

from .config import MetalsConfig, resolve_kalshi_credentials
from .databento import DatabentoLiveCollector, glbx_metals_subscriptions
from .kalshi import KalshiReadClient
from .pyth_lazer import (
    PYTH_LAZER_MAX_QUEUE,
    PYTH_LAZER_SUBSCRIPTION_ID,
    PythLazerProtocolError,
    build_pyth_lazer_subscription,
    normalize_pyth_lazer_update,
    parse_pyth_lazer_message,
)

PYTH_LAZER_REST_ENDPOINT = "https://pyth-lazer.dourolabs.app"
MAX_PYTH_HISTORY_WINDOW_SECONDS = 300


async def _safe_kalshi_get(client: Any, name: str, path: str) -> dict[str, Any]:
    """Classify one authenticated GET without retaining its account payload."""

    try:
        await client.get(path)
    except Exception as error:  # noqa: BLE001 - provider/auth versions vary
        return _provider_failure(name, error)
    return _check(name, "passed", "request_accepted")


async def preflight_kalshi(
    config: MetalsConfig,
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    """Check bounded authenticated Kalshi reads without exposing account objects."""

    report: dict[str, Any] = {
        "source": "kalshi",
        "scope": "authenticated_read_metadata",
        "credential_status": "unknown",
        "checks": [],
        "ok": False,
    }
    owns_client = client is None
    if client is None:
        try:
            credentials = resolve_kalshi_credentials(config.kalshi)
        except (ValueError, FileNotFoundError) as error:
            report["credential_status"] = "missing"
            report["checks"] = [
                _check(
                    "credential_files",
                    "failed",
                    "credential_missing",
                    error_type=type(error).__name__,
                )
            ]
            return report
        try:
            client = KalshiReadClient(config.kalshi, credentials)
            await client.start()
        except Exception as error:  # noqa: BLE001 - key parser versions vary
            if client is not None:
                await client.close()
            report["checks"] = [_provider_failure("client_start", error)]
            return report

    try:
        checks = [
            await _safe_kalshi_get(
                client, "authenticated_read_limits", "/account/limits"
            ),
            await _safe_kalshi_get(
                client,
                "authenticated_endpoint_costs",
                "/account/endpoint_costs",
            ),
        ]
    finally:
        if owns_client:
            await client.close()

    report["checks"] = checks
    report["credential_status"] = _credential_status_from_http(checks)
    report["ok"] = all(check["status"] == "passed" for check in checks)
    return report


def _check(name: str, status: str, reason: str, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "status": status,
        "reason": reason,
    }
    result.update(details)
    return result


def _http_check(name: str, status_code: int) -> dict[str, Any]:
    """Classify an HTTP response without inspecting or exposing its body."""

    if 200 <= status_code < 300:
        return _check(name, "passed", "request_accepted", http_status=status_code)
    if status_code == 401:
        reason = "credential_rejected"
    elif status_code == 403:
        reason = "entitlement_denied"
    elif status_code in (400, 404, 422):
        reason = "feed_or_request_rejected"
    elif status_code == 429:
        reason = "rate_limited"
    elif status_code >= 500:
        reason = "provider_unavailable"
    else:
        reason = "http_error"
    return _check(name, "failed", reason, http_status=status_code)


async def _request_status(
    session: Any,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    params: Mapping[str, Any] | None = None,
    json_body: Mapping[str, Any] | None = None,
) -> int:
    """Return only the status code and discard the response without reading it."""

    async with session.request(
        method,
        url,
        headers=dict(headers),
        params=dict(params) if params is not None else None,
        json=dict(json_body) if json_body is not None else None,
    ) as response:
        return int(response.status)


async def _safe_pyth_request(
    session: Any,
    name: str,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str],
    params: Mapping[str, Any] | None = None,
    json_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        status_code = await _request_status(
            session,
            method,
            url,
            headers=headers,
            params=params,
            json_body=json_body,
        )
    except (TimeoutError, aiohttp.ClientError) as error:
        # Third-party exception text can include URLs, headers, or response data.
        # The exception class is sufficient for a secret-safe operational report.
        return _check(
            name,
            "failed",
            "transport_error",
            error_type=type(error).__name__,
        )
    return _http_check(name, status_code)


def _credential_status_from_http(checks: Sequence[Mapping[str, Any]]) -> str:
    if any(check.get("status") == "passed" for check in checks):
        return "accepted"
    if checks and all(check.get("reason") == "credential_rejected" for check in checks):
        return "rejected"
    return "unknown"


async def preflight_pyth(
    config: MetalsConfig,
    *,
    metal: str,
    history_at_seconds: int,
    lazer_rest_endpoint: str = PYTH_LAZER_REST_ENDPOINT,
    timeout_seconds: float = 10.0,
    session: Any | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Check Pyth History auth and one Pyth Pro feed with at most two requests.

    A successful History request proves that the credential is accepted by that
    service for the configured symbol.  A successful ``latest_price`` request proves
    that the configured numeric feed is accepted by the Pyth Pro Router API.  The
    WebSocket stream itself is intentionally not opened by this preflight.
    """

    if history_at_seconds < 0:
        raise ValueError("history_at_seconds must be non-negative")
    if timeout_seconds <= 0 or timeout_seconds > 60:
        raise ValueError("timeout_seconds must be in (0, 60]")

    product = config.products.get(metal.lower())
    if product is None:
        raise ValueError(f"Unknown metals product: {metal}")
    if not product.pyth_symbol:
        raise ValueError(f"No Pyth symbol configured for {metal}")
    if product.pyth_lazer_id is None:
        raise ValueError(f"No Pyth Pro feed ID configured for {metal}")

    environment = os.environ if environ is None else environ
    api_key = environment.get(config.pyth.api_key_env, "").strip()
    report: dict[str, Any] = {
        "source": "pyth",
        "scope": "history_and_lazer_rest",
        "metal": metal.lower(),
        "credential_env": config.pyth.api_key_env,
        "credential_status": "missing" if not api_key else "unknown",
        "checks": [],
        "ok": False,
    }
    if not api_key:
        report["checks"] = [_check("credential", "failed", "credential_missing")]
        return report

    headers = {"Authorization": f"Bearer {api_key}"}
    history_params = {
        "symbol": product.pyth_symbol,
        "from": history_at_seconds,
        "to": history_at_seconds + MAX_PYTH_HISTORY_WINDOW_SECONDS - 1,
        "resolution": 1,
    }
    latest_price_body = {
        "priceFeedIds": [product.pyth_lazer_id],
        "properties": ["price", "feedUpdateTimestamp"],
        "formats": ["solana"],
        "deliveryFormat": "json",
        "channel": "fixed_rate@200ms",
        "jsonBinaryEncoding": "hex",
    }

    owns_session = session is None
    if session is None:
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_seconds)
        )
    try:
        checks = [
            await _safe_pyth_request(
                session,
                "history_symbol",
                "GET",
                config.pyth.endpoint.rstrip("/") + "/fixed_rate@200ms/history",
                headers=headers,
                params=history_params,
            ),
            await _safe_pyth_request(
                session,
                "lazer_feed",
                "POST",
                lazer_rest_endpoint.rstrip("/") + "/v1/latest_price",
                headers=headers,
                json_body=latest_price_body,
            ),
        ]
    finally:
        if owns_session:
            await session.close()

    report["checks"] = checks
    report["credential_status"] = _credential_status_from_http(checks)
    report["ok"] = all(check["status"] == "passed" for check in checks)
    return report


async def preflight_pyth_lazer_live(
    config: MetalsConfig,
    *,
    metal: str,
    duration_seconds: float = 5.0,
    allow_live_probe: bool = False,
    environ: Mapping[str, str] | None = None,
    connect_factory: Any | None = None,
) -> dict[str, Any]:
    """Probe every configured Lazer endpoint without retaining source messages."""

    if allow_live_probe is not True:
        raise ValueError("allow_live_probe=True is required")
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(duration_seconds)
        or not 0 < duration_seconds <= 30
    ):
        raise ValueError("duration_seconds must be finite and in (0, 30]")
    product = config.products.get(metal.lower())
    if product is None:
        raise ValueError(f"Unknown metals product: {metal}")
    if product.pyth_lazer_id is None or not product.pyth_symbol:
        raise ValueError(f"No Pyth Lazer feed configured for {metal}")
    endpoints = tuple(config.pyth.lazer_ws_endpoints)
    if not endpoints or any(
        not endpoint.startswith("wss://") for endpoint in endpoints
    ):
        raise ValueError("Pyth Lazer requires configured wss:// endpoints")

    environment = os.environ if environ is None else environ
    api_key = environment.get(config.pyth.api_key_env, "").strip()
    report: dict[str, Any] = {
        "source": "pyth",
        "scope": "bounded_lazer_websocket_no_retention",
        "metal": metal.lower(),
        "credential_env": config.pyth.api_key_env,
        "credential_status": "missing" if not api_key else "unknown",
        "duration_seconds": float(duration_seconds),
        "endpoint_count": len(endpoints),
        "retained_messages": 0,
        "checks": [],
        "ok": False,
    }
    if not api_key:
        report["checks"] = [_check("credential", "failed", "credential_missing")]
        return report

    connect = connect_factory or websockets.connect
    subscription = json.dumps(
        build_pyth_lazer_subscription(int(product.pyth_lazer_id)),
        separators=(",", ":"),
    )

    async def probe(index: int, endpoint: str) -> dict[str, Any]:
        name = f"lazer_endpoint_{index}"
        subscribed = False
        try:
            async with asyncio.timeout(float(duration_seconds)):
                connection = connect(
                    endpoint,
                    additional_headers={"Authorization": f"Bearer {api_key}"},
                    ping_interval=10,
                    ping_timeout=20,
                    max_queue=PYTH_LAZER_MAX_QUEUE,
                )
                async with connection as websocket:
                    await websocket.send(subscription)
                    while True:
                        message = parse_pyth_lazer_message(await websocket.recv())
                        message_type = message["type"]
                        if message_type == "subscribed":
                            if (
                                message.get("subscriptionId")
                                != PYTH_LAZER_SUBSCRIPTION_ID
                            ):
                                raise PythLazerProtocolError(
                                    "Pyth acknowledged an unexpected subscription"
                                )
                            subscribed = True
                            continue
                        if message_type == "streamUpdated" and subscribed:
                            normalize_pyth_lazer_update(
                                message,
                                expected_feed_id=int(product.pyth_lazer_id),
                                pyth_symbol=product.pyth_symbol,
                                metal=metal.lower(),
                                endpoint=f"endpoint-{index}",
                                connection_id="preflight",
                                receive_ts_ns=0,
                                local_monotonic_ts_ns=0,
                                ingestion_sequence=1,
                                active_endpoint=f"endpoint-{index}",
                            )
                            return _check(
                                name,
                                "passed",
                                "feed_update_received_and_discarded",
                                messages_observed=1,
                            )
                        if message_type in {
                            "error",
                            "subscriptionError",
                            "subscribedWithInvalidFeedIdsIgnored",
                            "unsubscribed",
                        }:
                            raise PythLazerProtocolError(
                                "Pyth rejected or ended the preflight subscription"
                            )
        except TimeoutError:
            return _check(
                name,
                "not_checked",
                (
                    "no_feed_update_within_bound"
                    if subscribed
                    else "no_subscription_ack_within_bound"
                ),
                messages_observed=0,
            )
        except Exception as error:  # noqa: BLE001 - transport versions vary
            return _provider_failure(name, error)

    checks = await asyncio.gather(
        *(probe(index, endpoint) for index, endpoint in enumerate(endpoints))
    )
    report["checks"] = list(checks)
    report["ok"] = bool(checks) and all(check["status"] == "passed" for check in checks)
    report["credential_status"] = _credential_status_from_http(checks)
    if report["ok"]:
        report["credential_status"] = "accepted"
    return report


def _exception_status_code(error: BaseException) -> int | None:
    for name in ("status_code", "status", "http_status"):
        value = getattr(error, name, None)
        if isinstance(value, int):
            return value
    return None


def _provider_failure(name: str, error: BaseException) -> dict[str, Any]:
    status_code = _exception_status_code(error)
    if status_code is not None:
        result = _http_check(name, status_code)
        result["error_type"] = type(error).__name__
        return result
    error_name = type(error).__name__
    lowered = error_name.lower()
    reason = (
        "credential_rejected"
        if "auth" in lowered or "credential" in lowered
        else "provider_error"
    )
    return _check(name, "failed", reason, error_type=error_name)


def _call_provider(
    name: str, function: Any, **kwargs: Any
) -> tuple[Any, dict[str, Any]]:
    try:
        value = function(**kwargs)
    # Provider client versions expose several unrelated exception hierarchies.
    except Exception as error:  # noqa: BLE001
        return None, _provider_failure(name, error)
    return value, _check(name, "passed", "request_accepted")


def _parent_symbols(symbols: Iterable[str]) -> list[str]:
    return [
        value if "." in value else value + ".FUT"
        for value in (str(symbol).strip().upper() for symbol in symbols)
        if value
    ]


def _response_member(response: Any, name: str, default: Any) -> Any:
    if isinstance(response, Mapping):
        return response.get(name, default)
    return getattr(response, name, default)


def _iso_date(value: Any) -> date | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def preflight_databento(
    config: MetalsConfig,
    *,
    metal: str,
    probe_date: str,
    environ: Mapping[str, str] | None = None,
    databento_module: Any | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Check Databento historical metadata, ranges, schemas, and symbology.

    The function never requests time-series records or submits a batch job.  Live
    entitlement is reported as not checked because Databento has no equivalent
    metadata-only live subscription probe.
    """

    product = config.products.get(metal.lower())
    if product is None:
        raise ValueError(f"Unknown metals product: {metal}")
    if not product.cme_symbols:
        raise ValueError(f"No CME symbols configured for {metal}")
    try:
        start_date = date.fromisoformat(probe_date)
    except ValueError as error:
        raise ValueError("probe_date must be an ISO date (YYYY-MM-DD)") from error
    end_date = start_date + timedelta(days=1)

    environment = os.environ if environ is None else environ
    api_key = environment.get(config.cme.api_key_env, "").strip()
    report: dict[str, Any] = {
        "source": "databento",
        "scope": "historical_metadata_only",
        "metal": metal.lower(),
        "dataset": config.cme.dataset,
        "credential_env": config.cme.api_key_env,
        "credential_status": "missing" if not api_key else "unknown",
        "checks": [],
        "ok": False,
    }

    dependency_check: dict[str, Any]
    if client is not None:
        dependency_check = _check("python_dependency", "passed", "injected_client")
    else:
        if databento_module is None:
            try:
                databento_module = importlib.import_module("databento")
            except ImportError:
                databento_module = None
        dependency_check = (
            _check("python_dependency", "passed", "available")
            if databento_module is not None
            else _check(
                "python_dependency",
                "failed",
                "optional_dependency_missing",
                dependency="databento",
            )
        )
    report["checks"].append(dependency_check)
    if not api_key:
        report["checks"].append(_check("credential", "failed", "credential_missing"))
    if dependency_check["status"] != "passed" or not api_key:
        return report

    if client is None:
        try:
            client = databento_module.Historical(api_key)
        # Construction may perform validation in provider-version-specific code.
        except Exception as error:  # noqa: BLE001
            failure = _provider_failure("historical_client", error)
            report["checks"].append(failure)
            if failure["reason"] == "credential_rejected":
                report["credential_status"] = "rejected"
            return report

    datasets, auth_check = _call_provider(
        "credential_and_dataset_catalog", client.metadata.list_datasets
    )
    report["checks"].append(auth_check)
    if auth_check["status"] != "passed":
        if auth_check["reason"] == "credential_rejected":
            report["credential_status"] = "rejected"
        return report
    report["credential_status"] = "accepted"

    dataset_present = config.cme.dataset in set(datasets or [])
    dataset_check = _check(
        "dataset_catalog",
        "passed" if dataset_present else "failed",
        "dataset_listed" if dataset_present else "dataset_not_listed",
    )
    report["checks"].append(dataset_check)
    if not dataset_present:
        return report

    schemas, schema_catalog_check = _call_provider(
        "schema_catalog",
        client.metadata.list_schemas,
        dataset=config.cme.dataset,
    )
    report["checks"].append(schema_catalog_check)

    dataset_range, range_check = _call_provider(
        "historical_dataset_entitlement",
        client.metadata.get_dataset_range,
        dataset=config.cme.dataset,
    )
    report["checks"].append(range_check)

    missing_schemas: list[str] = []
    unavailable_on_probe_date: list[str] = []
    if schema_catalog_check["status"] == "passed" and range_check["status"] == "passed":
        catalog_schemas = set(schemas or [])
        schema_ranges = _response_member(dataset_range, "schema", {})
        if not isinstance(schema_ranges, Mapping):
            schema_ranges = {}
        missing_schemas = [
            schema
            for schema in config.cme.schemas
            if schema not in catalog_schemas or schema not in schema_ranges
        ]
        for schema in config.cme.schemas:
            if schema in missing_schemas:
                continue
            schema_range = schema_ranges[schema]
            if not isinstance(schema_range, Mapping):
                unavailable_on_probe_date.append(schema)
                continue
            range_start = _iso_date(schema_range.get("start"))
            range_end = _iso_date(schema_range.get("end"))
            if (
                range_start is None
                or range_end is None
                or start_date < range_start
                or start_date >= range_end
            ):
                unavailable_on_probe_date.append(schema)
        schemas_available = not missing_schemas and not unavailable_on_probe_date
        schema_check = _check(
            "required_schema_entitlements",
            "passed" if schemas_available else "failed",
            "schemas_available" if schemas_available else "schemas_unavailable",
            missing_schemas=missing_schemas,
            unavailable_on_probe_date=unavailable_on_probe_date,
        )
    else:
        schema_check = _check(
            "required_schema_entitlements",
            "skipped",
            "prerequisite_failed",
            missing_schemas=[],
            unavailable_on_probe_date=[],
        )
    report["checks"].append(schema_check)

    symbols = _parent_symbols(product.cme_symbols)
    resolution, symbol_request_check = _call_provider(
        "symbol_resolution_request",
        client.symbology.resolve,
        dataset=config.cme.dataset,
        symbols=symbols,
        stype_in=config.cme.stype_in,
        stype_out="instrument_id",
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )
    report["checks"].append(symbol_request_check)
    if symbol_request_check["status"] == "passed":
        not_found = set(_response_member(resolution, "not_found", []) or [])
        unresolved = [symbol for symbol in symbols if symbol in not_found]
        symbol_check = _check(
            "configured_symbols",
            "passed" if not unresolved else "failed",
            "symbols_resolved" if not unresolved else "symbols_not_found",
            unresolved_symbols=unresolved,
        )
    else:
        symbol_check = _check(
            "configured_symbols",
            "skipped",
            "prerequisite_failed",
            unresolved_symbols=[],
        )
    report["checks"].append(symbol_check)
    report["checks"].append(
        _check(
            "live_dataset_entitlement",
            "not_checked",
            "requires_bounded_live_stream_probe",
        )
    )

    required_names = {
        "python_dependency",
        "credential_and_dataset_catalog",
        "dataset_catalog",
        "schema_catalog",
        "historical_dataset_entitlement",
        "required_schema_entitlements",
        "symbol_resolution_request",
        "configured_symbols",
    }
    report["ok"] = all(
        check["status"] == "passed"
        for check in report["checks"]
        if check["name"] in required_names
    )
    return report


def preflight_databento_live(
    config: MetalsConfig,
    *,
    metal: str,
    duration_seconds: float = 5.0,
    allow_live_probe: bool = False,
    environ: Mapping[str, str] | None = None,
    collector: Any | None = None,
) -> dict[str, Any]:
    """Probe one bounded MBO snapshot subscription without retaining records."""

    if allow_live_probe is not True:
        raise ValueError("allow_live_probe=True is required")
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(duration_seconds)
        or not 0 < duration_seconds <= 30
    ):
        raise ValueError("duration_seconds must be finite and in (0, 30]")
    product = config.products.get(metal.lower())
    if product is None:
        raise ValueError(f"Unknown metals product: {metal}")
    if not product.cme_symbols:
        raise ValueError(f"No CME symbols configured for {metal}")

    environment = os.environ if environ is None else environ
    api_key = environment.get(config.cme.api_key_env, "").strip()
    report: dict[str, Any] = {
        "source": "databento",
        "scope": "bounded_live_mbo_snapshot_no_retention",
        "metal": metal.lower(),
        "dataset": config.cme.dataset,
        "credential_env": config.cme.api_key_env,
        "credential_status": "missing" if not api_key else "unknown",
        "duration_seconds": float(duration_seconds),
        "records_observed": 0,
        "retained_records": 0,
        "checks": [],
        "ok": False,
    }
    if not api_key:
        report["checks"] = [_check("credential", "failed", "credential_missing")]
        return report

    if collector is None:
        try:
            subscriptions = glbx_metals_subscriptions(
                product.cme_symbols,
                stype_in=config.cme.stype_in,
                schemas=("mbo",),
                mbo_snapshot=True,
            )
            collector = DatabentoLiveCollector(
                api_key,
                subscriptions,
                reconnect_policy="none",
                include_ts_out=config.cme.include_ts_out,
            )
        except Exception as error:  # noqa: BLE001 - dependency/vendor versions vary
            report["checks"] = [_provider_failure("live_client", error)]
            return report

    activity = threading.Event()
    errors: list[BaseException] = []
    record_count = 0

    def discard_record(record: Any) -> None:
        nonlocal record_count
        record_count += 1
        activity.set()

    def capture_error(error: BaseException) -> None:
        errors.append(error)
        activity.set()

    started = False
    shutdown_error: BaseException | None = None
    try:
        collector.start_callbacks(
            discard_record,
            exception_callback=capture_error,
        )
        started = True
        activity.wait(timeout=float(duration_seconds))
    except Exception as error:  # noqa: BLE001 - provider/auth versions vary
        errors.append(error)
    finally:
        if started:
            try:
                collector.stop()
                collector.block_for_close(timeout=2.0)
            except Exception as error:  # noqa: BLE001 - provider versions vary
                shutdown_error = error

    report["records_observed"] = record_count
    if errors:
        check = _provider_failure("live_mbo_snapshot", errors[0])
        report["checks"] = [check]
        if check["reason"] == "credential_rejected":
            report["credential_status"] = "rejected"
        return report
    if shutdown_error is not None:
        report["checks"] = [_provider_failure("live_shutdown", shutdown_error)]
        return report
    if record_count < 1:
        report["checks"] = [
            _check(
                "live_mbo_snapshot",
                "not_checked",
                "no_record_observed_within_bound",
            )
        ]
        return report

    report["credential_status"] = "accepted"
    report["checks"] = [
        _check("live_mbo_snapshot", "passed", "record_received_and_discarded")
    ]
    report["ok"] = True
    return report
