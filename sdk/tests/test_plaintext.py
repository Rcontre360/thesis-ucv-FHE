import pytest

pytest.importorskip("fhe_sdk._backend", reason="Run scripts/run_tests.sh to build _backend first")

from fhe_sdk.plaintext import PlaintextVector  # noqa: E402

EPSILON = 1e-2


class TestPlaintextVectorProperties:
    def test_size(self, built_context):
        pt = built_context.encode([1.0, 2.0, 3.0])
        assert pt.size == 3

    def test_repr_contains_class_name(self, built_context):
        pt = built_context.encode([1.0])
        assert "PlaintextVector" in repr(pt)

    def test_repr_contains_size(self, built_context):
        pt = built_context.encode([1.0, 2.0])
        assert "size=2" in repr(pt)


class TestPlaintextVectorDecode:
    def test_decode_returns_list(self, built_context):
        pt = built_context.encode([1.0, 2.0])
        assert isinstance(pt.decode(), list)

    def test_decode_roundtrip(self, built_context):
        values = [1.5, 2.5, 3.5]
        pt = built_context.encode(values)
        decoded = pt.decode()
        for expected, actual in zip(values, decoded):
            assert abs(expected - actual) < EPSILON


class TestPlaintextVectorArithmetic:
    def test_add_scalar(self, built_context):
        pt = built_context.encode([1.0, 2.0, 3.0])
        result = (pt + 1.0).decode()
        for expected, actual in zip([2.0, 3.0, 4.0], result):
            assert abs(expected - actual) < EPSILON

    def test_add_list(self, built_context):
        pt = built_context.encode([1.0, 2.0])
        result = (pt + [3.0, 4.0]).decode()
        for expected, actual in zip([4.0, 6.0], result):
            assert abs(expected - actual) < EPSILON

    def test_add_plaintext_vector(self, built_context):
        a = built_context.encode([1.0, 2.0])
        b = built_context.encode([3.0, 4.0])
        result = (a + b).decode()
        for expected, actual in zip([4.0, 6.0], result):
            assert abs(expected - actual) < EPSILON

    def test_sub_scalar(self, built_context):
        pt = built_context.encode([5.0, 3.0])
        result = (pt - 1.0).decode()
        for expected, actual in zip([4.0, 2.0], result):
            assert abs(expected - actual) < EPSILON

    def test_sub_list(self, built_context):
        pt = built_context.encode([5.0, 6.0])
        result = (pt - [2.0, 3.0]).decode()
        for expected, actual in zip([3.0, 3.0], result):
            assert abs(expected - actual) < EPSILON

    def test_mul_scalar(self, built_context):
        pt = built_context.encode([2.0, 3.0])
        result = (pt * 2.0).decode()
        for expected, actual in zip([4.0, 6.0], result):
            assert abs(expected - actual) < EPSILON

    def test_mul_list(self, built_context):
        pt = built_context.encode([2.0, 3.0])
        result = (pt * [4.0, 5.0]).decode()
        for expected, actual in zip([8.0, 15.0], result):
            assert abs(expected - actual) < EPSILON

    def test_radd_scalar(self, built_context):
        pt = built_context.encode([1.0, 2.0])
        result = (1.0 + pt).decode()
        for expected, actual in zip([2.0, 3.0], result):
            assert abs(expected - actual) < EPSILON

    def test_rsub_scalar(self, built_context):
        pt = built_context.encode([1.0, 2.0])
        result = (5.0 - pt).decode()
        for expected, actual in zip([4.0, 3.0], result):
            assert abs(expected - actual) < EPSILON

    def test_rmul_scalar(self, built_context):
        pt = built_context.encode([2.0, 3.0])
        result = (3.0 * pt).decode()
        for expected, actual in zip([6.0, 9.0], result):
            assert abs(expected - actual) < EPSILON
