import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires compiled _backend and CUDA GPU")


@pytest.fixture(scope="session")
def built_context():
    """Server-side evaluator context (no secret key)."""
    try:
        from fhe_ml import FHEConfig, FHEContext
    except ImportError as e:
        pytest.skip(f"SDK not installed: {e}")
    config = FHEConfig(
        log_n=14,
        coeff_modulus_bit_sizes=[60] + [40] * 6 + [60],
        log_scale=40,
    )
    return FHEContext(config)


@pytest.fixture
def client(built_context):
    """Key owner sharing the server context's backend; its relin key is loaded
    into the server so ciphertext-ciphertext ops work. Tests that need rotation
    keys call `client.generate_rotation_keys(...)` then
    `built_context.set_client_params(client.key_params())`."""
    from fhe_ml import Client

    c = Client(built_context)
    built_context.set_client_params(c.key_params())
    return c
