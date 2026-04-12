"""Pytest configuration and shared fixtures."""

import glob
import importlib.util
import os
import sys

import pytest

# Pre-load _backend from site-packages before redirecting fhe_sdk to src/.
# This is necessary because _backend.so lives in site-packages (installed by
# run_tests.sh via cmake) while the Python sources live in src/.
def _preload_backend() -> None:
    import site
    candidates = site.getsitepackages() + [site.getusersitepackages()]
    for sp in candidates:
        so_files = glob.glob(os.path.join(sp, "fhe_sdk", "_backend*.so"))
        if so_files:
            spec = importlib.util.spec_from_file_location("fhe_sdk._backend", so_files[0])
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["fhe_sdk._backend"] = mod
                spec.loader.exec_module(mod)
            return

_preload_backend()

# Point imports at development sources so tests always use the latest .py files.
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, _src)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires compiled _backend and CUDA GPU")


@pytest.fixture(scope="session")
def built_context():
    try:
        from fhe_sdk.context import FHEContext
    except ImportError as e:
        pytest.skip(f"_backend not compiled: {e}")
    return FHEContext.default()
