"""
Pytest configuration for Elelem tests.
Manages Docker Compose services for integration testing.
"""

import os
import sys
import time
import pytest
import subprocess
import requests
import logging
from pathlib import Path

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test (no external dependencies)")
    config.addinivalue_line("markers", "integration: mark test as integration test (requires Docker)")
    config.addinivalue_line("markers", "server: mark test as server mode test (requires Elelem API server)")
    config.addinivalue_line("markers", "slow: mark test as slow running")


@pytest.fixture(scope="session")
def docker_compose(request):
    """
    Session-scoped fixture that manages Docker Compose services.
    Only starts services if integration/server tests are being run.
    """
    # Check if we're running integration tests by looking at the request session
    session = request.session
    has_integration_tests = False

    # Check all collected items for integration/server markers
    for item in session.items:
        if item.get_closest_marker("integration") or item.get_closest_marker("server"):
            has_integration_tests = True
            break

    if not has_integration_tests:
        # No integration tests, skip Docker setup
        yield None
        return

    # Change to tests directory
    tests_dir = Path(__file__).parent
    os.chdir(tests_dir)

    # Check if Docker is available
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("Docker not available, skipping integration tests")

    # Check if services are already running
    result = subprocess.run(
        ["docker", "compose", "ps", "--services", "--filter", "status=running"],
        capture_output=True, text=True
    )
    already_running = bool(result.stdout.strip())

    if already_running:
        logger.info("ℹ️  Docker services already running, using existing containers")
        yield "existing"
        return

    logger.info("🚀 Starting Docker Compose services...")

    try:
        # Build images
        subprocess.run(
            ["docker", "compose", "build"],
            capture_output=True, check=True
        )

        # Start services
        subprocess.run(
            ["docker", "compose", "up", "-d"],
            capture_output=True, check=True
        )

        # Wait for services to be healthy
        logger.info("⏳ Waiting for services to be healthy...")
        _wait_for_services()

        logger.info("✅ Docker services ready!")
        yield "started"

    finally:
        # Only stop if we started them
        if not already_running:
            logger.info("🛑 Stopping Docker Compose services...")
            subprocess.run(
                ["docker", "compose", "down"],
                capture_output=True
            )
            logger.info("✅ Docker services stopped")


def _wait_for_services(timeout=60):
    """Wait for all services to be healthy."""
    start_time = time.time()

    services_to_check = [
        ("Elelem Server", "http://localhost:8000/health"),
        ("Dashboard", "http://localhost:8501/_stcore/health"),
        # PostgreSQL is checked via Elelem server health
    ]

    while time.time() - start_time < timeout:
        all_healthy = True

        for service_name, url in services_to_check:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code != 200:
                    all_healthy = False
                    break
            except requests.RequestException:
                all_healthy = False
                break

        if all_healthy:
            return

        time.sleep(1)

    raise TimeoutError(f"Services did not become healthy within {timeout} seconds")


@pytest.fixture(scope="session")
def elelem_server_url(docker_compose):
    """Provides the Elelem server URL for tests."""
    if docker_compose is None:
        pytest.skip("Integration tests require Docker")
    return "http://localhost:8000"


@pytest.fixture(scope="session")
def dashboard_url(docker_compose):
    """Provides the dashboard URL for tests."""
    if docker_compose is None:
        pytest.skip("Integration tests require Docker")
    return "http://localhost:8501"


@pytest.fixture
def elelem_client():
    """Provides a direct Elelem client for unit tests."""
    from elelem import Elelem
    return Elelem()


@pytest.fixture
def api_client(elelem_server_url):
    """Provides an API client for server mode tests."""
    from openai import OpenAI
    return OpenAI(
        api_key="test-key",
        base_url=f"{elelem_server_url}/v1"
    )


# Removed autouse fixture - using session-based marker detection instead


# Optional: Add fixture for cleaning metrics between tests
@pytest.fixture
def clean_metrics():
    """Clean metrics store before test (for unit tests)."""
    from elelem.metrics import MetricsStore
    store = MetricsStore()
    # If there's a method to clear metrics, use it here
    yield store
    # Cleanup after test if needed