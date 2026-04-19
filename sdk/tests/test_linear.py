import pytest

pytest.importorskip("core._backend", reason="Run scripts/run_tests.sh to build _backend first")

from api.layers.linear import Linear  # noqa: E402

EPSILON = 1e-2


class TestLinear:
    def test_identity_matrix_passthrough(self, built_context):
        W = [[1.0, 0.0], [0.0, 1.0]]
        layer = Linear(2, 2, W)
        ct = built_context.encrypt([0.3, -0.7])
        result = layer(ct).decrypt()[:2]
        assert abs(result[0] - 0.3) < EPSILON
        assert abs(result[1] - (-0.7)) < EPSILON

    def test_output_size(self, built_context):
        W = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        layer = Linear(3, 2, W)
        ct = built_context.encrypt([0.1, 0.2, 0.3])
        assert layer(ct).size == 2

    def test_scale_row(self, built_context):
        W = [[2.0, 0.0], [0.0, 3.0]]
        layer = Linear(2, 2, W)
        ct = built_context.encrypt([0.5, 0.4])
        result = layer(ct).decrypt()[:2]
        assert abs(result[0] - 1.0) < EPSILON
        assert abs(result[1] - 1.2) < EPSILON

    def test_bias_added(self, built_context):
        W = [[1.0, 0.0], [0.0, 1.0]]
        b = [0.1, -0.2]
        layer = Linear(2, 2, W, bias=b)
        ct = built_context.encrypt([0.3, 0.5])
        result = layer(ct).decrypt()[:2]
        assert abs(result[0] - 0.4) < EPSILON
        assert abs(result[1] - 0.3) < EPSILON

    def test_wrong_input_size_raises(self, built_context):
        layer = Linear(2, 2, [[1.0, 0.0], [0.0, 1.0]])
        ct = built_context.encrypt([0.1, 0.2, 0.3])
        with pytest.raises(ValueError):
            layer(ct)
