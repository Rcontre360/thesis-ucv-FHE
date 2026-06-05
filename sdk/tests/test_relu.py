import pytest

pytest.importorskip("fhe_ml.backend._backend", reason="Run scripts/run_tests.sh to build _backend first")

import numpy as np  # noqa: E402

from fhe_ml.layers.relu import ReLU  # noqa: E402


def _relu(degrees):
    r = ReLU()
    r.set_degrees(degrees)
    return r


class TestReLU:
    def test_encrypted_far_from_kink_positive(self, built_context):
        # At inputs near +/-1 the step polynomial is well-approximated;
        # the encrypted output matches real ReLU within a coarse tolerance.
        relu = _relu((3,))
        ct = built_context.encrypt([1.0])
        result = relu(ct).decrypt()[0]
        assert abs(result - 1.0) < 0.2

    def test_encrypted_far_from_kink_negative(self, built_context):
        relu = _relu((3,))
        ct = built_context.encrypt([-1.0])
        result = relu(ct).decrypt()[0]
        assert abs(result - 0.0) < 0.2

    def test_batch_output_size(self, built_context):
        relu = _relu((3,))
        ct = built_context.encrypt([0.1, -0.3, 0.7, -0.9])
        assert relu(ct).size == 4

    def test_mult_depth_matches_degrees(self):
        # Per polynomial: (d+3)/2 levels via the x^2 substitution; +1 for the
        # final x*step.
        assert _relu((3,)).mult_depth() == 4
        assert _relu((3, 3)).mult_depth() == 7
        assert _relu((15, 15, 27)).mult_depth() == 34
