"""
Pytest configuration and shared fixtures.

When the compiled _backend is absent or stale (pre-rewrite .so that lacks
create_ckks_context_with_security), a minimal mock is injected into sys.modules
so pure-Python tests (parameter validation, builder logic) can still run.

Tests that require actual homomorphic operations use the `built_context` fixture,
which skips automatically when the real backend is not present.
"""

import sys
import pytest
from unittest.mock import MagicMock


def _install_backend_mock() -> None:
    """Override sys.modules["fhe_sdk._backend"] with a minimal stub."""

    class _SecurityLevel:
        SEC128 = "sec128"
        SEC192 = "sec192"
        SEC256 = "sec256"

    mock = MagicMock()
    mock.SecurityLevel = _SecurityLevel

    for name in [
        "create_ckks_context",
        "create_ckks_context_with_security",
        "CKKSEncoder",
        "CKKSEncryptor",
        "CKKSDecryptor",
        "CKKSKeyGenerator",
        "CKKSSecretkey",
        "CKKSPublickey",
        "CKKSRelinkey",
        "CKKSGaloiskey",
        "CKKSPlaintext",
        "CKKSCiphertext",
        "CKKSOperator",
        "BootstrappingType",
        "BootstrappingConfig",
    ]:
        setattr(mock, name, MagicMock())

    # Override any previously loaded (stale) version.
    sys.modules["fhe_sdk._backend"] = mock


def _backend_is_current() -> bool:
    """Return True only if the loaded _backend has the rewritten API."""
    try:
        import fhe_sdk._backend as b  # noqa: F401
        return hasattr(b, "create_ckks_context_with_security")
    except ImportError:
        return False


if _backend_is_current():
    HAS_BACKEND = True
else:
    _install_backend_mock()
    HAS_BACKEND = False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires compiled _backend and CUDA GPU")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def built_context():
    """
    A fully built FHEContext using default parameters.
    Skipped automatically when the real _backend is not compiled.
    """
    if not HAS_BACKEND:
        pytest.skip("_backend not compiled — skipping GPU test")
    from fhe_sdk.context import FHEContext
    return FHEContext.default()
