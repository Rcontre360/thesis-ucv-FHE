import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires compiled _backend and CUDA GPU")


@pytest.fixture(scope="session")
def built_context():
    try:
        from api.context import FHEContext
    except ImportError as e:
        pytest.skip(f"SDK not installed: {e}")
    # 6 usable levels — enough for Linear -> ReLU(degrees=(3,)) -> Linear
    # (depth 1+4+1=6) and within SEC128 at N=16384.
    return (
        FHEContext()
        .set_poly_modulus_degree(16384)
        .set_coeff_modulus_bit_sizes([60] + [40] * 6 + [60])
        .set_scale(2**40)
        .build()
    )
