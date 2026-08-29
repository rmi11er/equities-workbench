import pytest

from src.metals.config import KalshiConfig, ResolvedKalshiCredentials
from src.metals.kalshi import KalshiReadClient, ReadOnlyViolation


@pytest.mark.asyncio
async def test_client_rejects_order_methods_before_network_or_key_loading():
    client = KalshiReadClient(
        KalshiConfig(),
        ResolvedKalshiCredentials(
            api_key_id="not-used", private_key_path="/does/not/exist"
        ),
    )

    with pytest.raises(ReadOnlyViolation):
        await client.request("POST", "/portfolio/orders")
    with pytest.raises(ReadOnlyViolation):
        await client.request("DELETE", "/portfolio/orders")
