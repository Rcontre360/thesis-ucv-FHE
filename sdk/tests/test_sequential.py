import pytest

pytest.importorskip("core._backend", reason="Run scripts/run_tests.sh to build _backend first")

from api.sequential import Sequential  # noqa: E402
from api.layers.linear import Linear  # noqa: E402
from api.functions.activations import ReLU  # noqa: E402

EPSILON = 1e-2


class TestSequential:
    def test_single_linear(self, built_context):
        W = [[1.0, 0.0], [0.0, 1.0]]
        model = Sequential([Linear(2, 2, W)])
        ct = built_context.encrypt([0.3, -0.7])
        result = model(ct).decrypt()[:2]
        assert abs(result[0] - 0.3) < EPSILON
        assert abs(result[1] - (-0.7)) < EPSILON

    def test_linear_then_relu(self, built_context):
        W = [[1.0, 0.0], [0.0, 1.0]]
        model = Sequential([Linear(2, 2, W), ReLU()])
        ct = built_context.encrypt([0.5, -0.5])
        result = model(ct).decrypt()[:2]
        expected_pos = 0.125 * 0.5**2 + 0.5 * 0.5 + 0.375
        expected_neg = 0.125 * 0.25 + 0.5 * (-0.5) + 0.375
        assert abs(result[0] - expected_pos) < EPSILON
        assert abs(result[1] - expected_neg) < EPSILON

    def test_output_size_matches_last_layer(self, built_context):
        W1 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        W2 = [[1.0, 0.0], [0.0, 1.0]]
        model = Sequential([Linear(3, 2, W1), ReLU(), Linear(2, 2, W2)])
        ct = built_context.encrypt([0.1, 0.2, 0.3])
        assert model(ct).size == 2

    def test_empty_layers_raises(self):
        with pytest.raises(ValueError):
            Sequential([])
