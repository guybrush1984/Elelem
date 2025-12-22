"""
Token providers for API authentication.

All providers implement the same interface:
- StaticKeyProvider: reads API key from env var
- GoogleVertexProvider: refreshes Google Cloud tokens automatically
- Future: AWS, Azure, etc.
"""

import os
import json
import base64
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Type
from dataclasses import dataclass

import httpx


@dataclass
class ProbeResult:
    """Result of an endpoint probe."""
    success: bool
    retryable: bool = True
    reason: str = ""


class TokenProvider(ABC):
    """Abstract interface for authentication providers."""

    @abstractmethod
    def get_token(self) -> str:
        """Get current token/key, refreshing if needed."""
        pass

    def get_endpoint(self) -> Optional[str]:
        """Get API endpoint URL. None means use config endpoint."""
        return None

    def probe(self, endpoint: str, timeout: float, logger: logging.Logger) -> ProbeResult:
        """Probe endpoint to verify it's accessible."""
        try:
            probe_url = f"{endpoint.rstrip('/')}/models"
            headers = {"Authorization": f"Bearer {self.get_token()}"}

            with httpx.Client(timeout=timeout) as client:
                response = client.get(probe_url, headers=headers)
                status = response.status_code

                if status == 200:
                    logger.debug(f"Endpoint probe successful: {endpoint}")
                    return ProbeResult(success=True)
                elif status in (401, 403):
                    logger.debug(f"Endpoint probe failed (auth): {endpoint} (status: {status})")
                    return ProbeResult(success=False, retryable=False, reason=f"auth_failed_{status}")
                else:
                    logger.debug(f"Endpoint probe failed: {endpoint} (status: {status})")
                    return ProbeResult(success=False, retryable=True, reason=f"http_{status}")

        except (httpx.TimeoutException, httpx.ConnectError, httpx.ConnectTimeout) as e:
            logger.warning(f"Endpoint probe failed: {endpoint} ({type(e).__name__})")
            return ProbeResult(success=False, retryable=True, reason=type(e).__name__)
        except Exception as e:
            logger.warning(f"Endpoint probe failed: {endpoint} ({type(e).__name__}: {e})")
            return ProbeResult(success=False, retryable=True, reason=type(e).__name__)


class StaticKeyProvider(TokenProvider):
    """Provider for static API keys from environment variables."""

    def __init__(self, provider_name: str, provider_config: Dict, logger: logging.Logger):
        base_provider = provider_config.get("base_provider")
        if base_provider:
            env_var = f"{base_provider.upper()}_API_KEY"
        else:
            env_var = f"{provider_name.upper()}_API_KEY"

        self._api_key = os.getenv(env_var)
        if not self._api_key:
            raise ValueError(f"No API key found (env var: {env_var})")

        logger.debug(f"[{provider_name}] Loaded API key from {env_var}")

    def get_token(self) -> str:
        return self._api_key


class GoogleVertexProvider(TokenProvider):
    """Token provider for Google Vertex AI with auto-refresh."""

    def __init__(self, provider_name: str, provider_config: Dict, logger: logging.Logger):
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request

        self._logger = logger
        self._provider_name = provider_name
        self._request = Request

        creds_json = os.getenv("VERTEX_CREDENTIALS_JSON")
        if not creds_json:
            raise ValueError("VERTEX_CREDENTIALS_JSON not set")

        # Support both base64-encoded and raw JSON
        try:
            decoded = base64.b64decode(creds_json).decode('utf-8')
            # Verify it's valid JSON before using decoded version
            json.loads(decoded)
            creds_json = decoded
        except Exception:
            pass  # Already raw JSON

        info = json.loads(creds_json)
        self._credentials = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self._project_id = info.get("project_id")
        self._location = os.getenv("VERTEX_LOCATION", "global")

        # Get initial token
        self._credentials.refresh(self._request())
        logger.info(f"[{provider_name}] Loaded credentials for project: {self._project_id}")

    def get_token(self) -> str:
        if not self._credentials.valid:
            self._credentials.refresh(self._request())
            self._logger.debug(f"[{self._provider_name}] Token refreshed")
        return self._credentials.token

    def get_endpoint(self) -> str:
        # Global endpoint has different format
        if self._location == "global":
            return (
                f"https://aiplatform.googleapis.com/v1/"
                f"projects/{self._project_id}/locations/global/endpoints/openapi"
            )
        return (
            f"https://{self._location}-aiplatform.googleapis.com/v1/"
            f"projects/{self._project_id}/locations/{self._location}/endpoints/openapi"
        )

    def probe(self, endpoint: str, timeout: float, logger: logging.Logger) -> ProbeResult:
        # Cloud providers don't have /models endpoint
        # Credentials were validated during init, so assume accessible
        logger.debug(f"[{self._provider_name}] Skipping probe (cloud provider)")
        return ProbeResult(success=True)


# Registry of cloud token providers (require auth_type in config)
CLOUD_PROVIDERS: Dict[str, Type[TokenProvider]] = {
    "google_vertex": GoogleVertexProvider,
    # Future: "aws_bedrock": AWSBedrockProvider,
    # Future: "azure_openai": AzureOpenAIProvider,
}


def create_token_provider(
    provider_name: str,
    provider_config: Dict,
    logger: logging.Logger
) -> Optional[TokenProvider]:
    """
    Create a token provider for the given provider.

    Uses auth_type from config to select cloud provider,
    or falls back to StaticKeyProvider for standard API keys.

    Args:
        provider_name: Name of the provider
        provider_config: Provider configuration dict
        logger: Logger instance

    Returns:
        TokenProvider instance or None if unavailable
    """
    auth_type = provider_config.get("auth_type")

    if auth_type:
        # Cloud provider with token refresh
        provider_class = CLOUD_PROVIDERS.get(auth_type)
        if not provider_class:
            logger.error(f"[{provider_name}] Unknown auth_type: {auth_type}")
            return None
    else:
        # Standard API key
        provider_class = StaticKeyProvider

    try:
        return provider_class(provider_name, provider_config, logger)
    except ImportError as e:
        logger.warning(f"[{provider_name}] Missing dependencies: {e}")
        return None
    except ValueError as e:
        logger.debug(f"[{provider_name}] {e}")
        return None
    except Exception as e:
        logger.error(f"[{provider_name}] Failed to initialize auth: {e}")
        return None
