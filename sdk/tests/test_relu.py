import pytest

pytest.importorskip("core._backend", reason="Run scripts/run_tests.sh to build _backend first")

import numpy as np  # noqa: E402

from api.functions.relu import ReLU  # noqa: E402


class TestReLU:
    def test_forward_plain_is_real_relu(self):
        # forward_plain (used by calibration) is the true ReLU — exact.
        relu = ReLU(degrees=(3,))
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        out = relu.forward_plain(x)
        assert np.allclose(out, np.maximum(0.0, x))

    def test_encrypted_far_from_kink_positive(self, built_context):
        # At inputs near +/-1 the step polynomial is well-approximated;
        # the encrypted output matches real ReLU within a coarse tolerance.
        relu = ReLU(degrees=(3,))
        ct = built_context.encrypt([1.0])
        result = relu(ct).decrypt()[0]
        assert abs(result - 1.0) < 0.2

    def test_encrypted_far_from_kink_negative(self, built_context):
        relu = ReLU(degrees=(3,))
        ct = built_context.encrypt([-1.0])
        result = relu(ct).decrypt()[0]
        assert abs(result - 0.0) < 0.2

    def test_batch_output_size(self, built_context):
        relu = ReLU(degrees=(3,))
        ct = built_context.encrypt([0.1, -0.3, 0.7, -0.9])
        assert relu(ct).size == 4

    def test_mult_depth_matches_degrees(self):
        # Per polynomial: (d+3)/2 levels via the x^2 substitution; +1 for the
        # final x*step.
        assert ReLU(degrees=(3,)).mult_depth() == 4
        assert ReLU(degrees=(3, 3)).mult_depth() == 7
        assert ReLU(degrees=(15, 15, 27)).mult_depth() == 34
