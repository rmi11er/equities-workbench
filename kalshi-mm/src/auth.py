"""RSA authentication utilities for Kalshi API."""

import base64
import time
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend


class KalshiAuth:
    """Handles RSA-PSS signing for Kalshi API authentication."""

    def __init__(self, api_key_id: str, private_key_path: str):
        self.api_key_id = api_key_id
        self._private_key: Optional[rsa.RSAPrivateKey] = None
        self._private_key_path = private_key_path

    def _load_private_key(self) -> rsa.RSAPrivateKey:
        """Load and cache the private key."""
        if self._private_key is None:
            path = Path(self._private_key_path)
            if not path.exists():
                raise FileNotFoundError(f"Private key not found: {path}")

            key_data = path.read_bytes()
            self._private_key = serialization.load_pem_private_key(
                key_data,
                password=None,
                backend=default_backend(),
            )
        return self._private_key

    def sign(self, timestamp_ms: int, method: str, path: str) -> str:
        """
        Generate signature for Kalshi API request.

        The signature format is: "{timestamp}{method}{path}"
        Signed with RSA-PSS using SHA256.

        Args:
            timestamp_ms: Unix timestamp in milliseconds
            method: HTTP method (GET, POST, DELETE)
            path: API path (e.g., /trade-api/ws/v2)

        Returns:
            Base64-encoded signature
        """
        private_key = self._load_private_key()

        message = f"{timestamp_ms}{method}{path}"
        message_bytes = message.encode("utf-8")

        signature = private_key.sign(
            message_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        return base64.b64encode(signature).decode("utf-8")

    def get_auth_headers(self, method: str, path: str) -> dict[str, str]:
        """
        Generate authentication headers for an API request.

        Args:
            method: HTTP method
            path: API path

        Returns:
            Dictionary of headers to include in the request
        """
        timestamp_ms = int(time.time() * 1000)
        signature = self.sign(timestamp_ms, method, path)

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        }
