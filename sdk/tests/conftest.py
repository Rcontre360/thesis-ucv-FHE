"""Pytest configuration and shared fixtures."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires compiled _backend and CUDA GPU")


@pytest.fixture(scope="session")
def built_context():
    """
    A fully built FHEContext using default parameters.
    Requires the compiled _backend extension (run scripts/run_tests.sh to build).
    """
    try:
        from fhe_sdk.context import FHEContext
    except ImportError as e:
        pytest.skip(f"_backend not compiled: {e}")
    return FHEContext.default()
