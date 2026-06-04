import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires compiled _backend and CUDA GPU")


@pytest.fixture(scope="session")
def built_context():
    try:
        from fhe_ml import FHEConfig, FHEContext
    except ImportError as e:
        pytest.skip(f"SDK not installed: {e}")
    config = FHEConfig(
        log_n=14,
        coeff_modulus_bit_sizes=[60] + [40] * 6 + [60],
        log_scale=40,
    )
    return FHEContext(config).build()
