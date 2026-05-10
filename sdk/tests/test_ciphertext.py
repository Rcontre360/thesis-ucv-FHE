import pytest

pytest.importorskip("core._backend", reason="Run scripts/run_tests.sh to build _backend first")

from api.ciphertext import EncryptedVector  # noqa: E402
from api.tensor import PlaintextTensor      # noqa: E402

EPSILON = 1e-2


class TestEncryptedVectorArithmetic:
    def test_add_scalar(self, built_context):
        ct = built_context.encrypt([1.0, 2.0, 3.0])
        result = built_context.decrypt(ct + 1.0)
        for expected, actual in zip([2.0, 3.0, 4.0], result):
            assert abs(expected - actual) < EPSILON

    def test_add_ciphertext(self, built_context):
        a = built_context.encrypt([1.0, 2.0])
        b = built_context.encrypt([3.0, 4.0])
        result = built_context.decrypt(a + b)
        for expected, actual in zip([4.0, 6.0], result):
            assert abs(expected - actual) < EPSILON

    def test_mul_scalar(self, built_context):
        ct = built_context.encrypt([2.0, 3.0])
        result = built_context.decrypt(ct * 2.0)
        for expected, actual in zip([4.0, 6.0], result):
            assert abs(expected - actual) < EPSILON

    def test_mul_ciphertext(self, built_context):
        a = built_context.encrypt([2.0, 3.0])
        b = built_context.encrypt([4.0, 5.0])
        result = built_context.decrypt(a * b)
        for expected, actual in zip([8.0, 15.0], result):
            assert abs(expected - actual) < EPSILON

    def test_sub_ciphertext(self, built_context):
        a = built_context.encrypt([5.0, 6.0])
        b = built_context.encrypt([3.0, 2.0])
        result = built_context.decrypt(a - b)
        for expected, actual in zip([2.0, 4.0], result):
            assert abs(expected - actual) < EPSILON

    def test_depth_mismatch_raises(self, built_context):
        ct = built_context.encrypt([1.0, 2.0])
        ct_deep = ct * 1.0
        pt_fresh = built_context.encode([1.0, 1.0])
        with pytest.raises(ValueError, match="Depth mismatch"):
            _ = ct_deep + pt_fresh


class TestEncryptedVectorRotate:
    def test_rotate_zero_is_identity(self, built_context):
        values = [1.0, 2.0, 3.0, 4.0]
        ct = built_context.encrypt(values)
        result = built_context.decrypt(ct.rotate(0))[:4]
        for expected, actual in zip(values, result):
            assert abs(expected - actual) < EPSILON

    def test_rotate_shifts_values(self, built_context):
        ct = built_context.encrypt([1.0, 2.0, 3.0, 4.0])
        result = built_context.decrypt(ct.rotate(1))
        assert abs(result[0] - 2.0) < EPSILON
        assert abs(result[1] - 3.0) < EPSILON
        assert abs(result[2] - 4.0) < EPSILON


class TestEncryptedVectorDot:
    def test_dot_result_size_is_one(self, built_context):
        ct = built_context.encrypt([1.0, 2.0, 3.0])
        assert ct.dot([1.0, 1.0, 1.0]).size == 1

    def test_dot_general(self, built_context):
        ct = built_context.encrypt([1.0, 2.0, 3.0])
        # 1*4 + 2*5 + 3*6 = 32
        assert abs(built_context.decrypt(ct.dot([4.0, 5.0, 6.0]))[0] - 32.0) < EPSILON

    def test_dot_all_ones_weights_equals_sum(self, built_context):
        values = [2.0, 3.0, 4.0, 5.0]
        ct = built_context.encrypt(values)
        assert abs(built_context.decrypt(ct.dot([1.0] * 4))[0] - 14.0) < EPSILON

    def test_dot_zero_weights(self, built_context):
        ct = built_context.encrypt([1.0, 2.0, 3.0])
        assert abs(built_context.decrypt(ct.dot([0.0, 0.0, 0.0]))[0]) < EPSILON

    def test_dot_negative_weights(self, built_context):
        ct = built_context.encrypt([3.0, 3.0])
        # 3*1 + 3*(-1) = 0
        assert abs(built_context.decrypt(ct.dot([1.0, -1.0]))[0]) < EPSILON

    def test_dot_single_element(self, built_context):
        ct = built_context.encrypt([5.0])
        assert abs(built_context.decrypt(ct.dot([2.0]))[0] - 10.0) < EPSILON

    def test_dot_size_mismatch_raises(self, built_context):
        ct = built_context.encrypt([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="weights length"):
            ct.dot([1.0, 2.0])


class TestEncryptedVectorMatmul:
    def test_matmul_identity(self, built_context):
        x = built_context.encrypt([3.0, 5.0])
        W = PlaintextTensor([[1.0, 0.0], [0.0, 1.0]])
        result = x.matmul(W)
        assert result.size == 2
        dec = built_context.decrypt(result)
        assert abs(dec[0] - 3.0) < EPSILON
        assert abs(dec[1] - 5.0) < EPSILON

    def test_matmul_projection(self, built_context):
        x = built_context.encrypt([2.0, 4.0, 6.0])
        W = PlaintextTensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = x.matmul(W)
        assert result.size == 2
        dec = built_context.decrypt(result)
        assert abs(dec[0] - 2.0) < EPSILON
        assert abs(dec[1] - 4.0) < EPSILON

    def test_matmul_wrong_ndim_raises(self, built_context):
        x = built_context.encrypt([1.0, 2.0])
        T = PlaintextTensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
        with pytest.raises(ValueError, match="2D PlaintextTensor"):
            x.matmul(T)

    def test_matmul_shape_mismatch_raises(self, built_context):
        x = built_context.encrypt([1.0, 2.0, 3.0])
        W = PlaintextTensor([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="columns"):
            x.matmul(W)

    def test_matmul_wrong_type_raises(self, built_context):
        x = built_context.encrypt([1.0, 2.0])
        with pytest.raises(TypeError, match="PlaintextTensor"):
            x.matmul([[1.0, 0.0], [0.0, 1.0]])  # type: ignore[arg-type]

    def test_matmul_tall_expansion(self, built_context):
        # out > next_pow2(in): rectangular Halevi–Shoup must zero-pad W to
        # s × s with s = max(m_padded, n_padded) = 8 and lift x's tile period.
        x = built_context.encrypt([1.0, 2.0, 3.0, 4.0])
        W = PlaintextTensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ])
        result = x.matmul(W)
        assert result.size == 8
        assert result.period == 8
        dec = result.decrypt()
        expected = [1.0, 2.0, 3.0, 4.0, 3.0, 5.0, 7.0, 5.0]
        for e, a in zip(expected, dec):
            assert abs(e - a) < EPSILON

    def test_matmul_period_preserves_through_chain(self, built_context):
        # First matmul fixes period at s=4, second wide matmul keeps the same period
        # (no shrinking) so the chained result stays correct.
        x = built_context.encrypt([1.0, 2.0, 3.0, 4.0])
        W1 = PlaintextTensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        y = x.matmul(W1)                   # size=2, period=4
        assert y.period == 4
        W2 = PlaintextTensor([[1.0, 1.0]])  # 1x2: out=1, in=2
        z = y.matmul(W2)                    # size=1
        # period should not shrink below x's period (which is 4 here)
        assert z.period == 4
        assert abs(z.decrypt()[0] - 3.0) < EPSILON
