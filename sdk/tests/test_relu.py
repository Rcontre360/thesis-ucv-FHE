import pytest

pytest.importorskip("core._backend", reason="Run scripts/run_tests.sh to build _backend first")

from api.functions.activations import ReLU  # noqa: E402

EPSILON = 1e-2


class TestReLU:
    def test_positive_value_passes_through(self, built_context):
        relu = ReLU()
        ct = built_context.encrypt([0.5])
        result = relu(ct).decrypt()[0]
        expected = 0.125 * 0.5**2 + 0.5 * 0.5 + 0.375
        assert abs(result - expected) < EPSILON

    def test_zero_gives_point375(self, built_context):
        relu = ReLU()
        ct = built_context.encrypt([0.0])
        result = relu(ct).decrypt()[0]
        assert abs(result - 0.375) < EPSILON

    def test_negative_value_suppressed(self, built_context):
        relu = ReLU()
        ct = built_context.encrypt([-0.5])
        result = relu(ct).decrypt()[0]
        expected = 0.125 * 0.25 + 0.5 * (-0.5) + 0.375
        assert abs(result - expected) < EPSILON

    def test_batch_output_size(self, built_context):
        relu = ReLU()
        values = [0.1, -0.3, 0.7, -0.9]
        ct = built_context.encrypt(values)
        result = relu(ct)
        assert result.size == len(values)
