import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires compiled _backend and CUDA GPU")


@pytest.fixture(scope="session")
def built_context():
    try:
        from api.context import FHEContext
    except ImportError as e:
        pytest.skip(f"SDK not installed: {e}")
    return FHEContext.default()
