import numpy as np
import pytest

pytest.importorskip("fhe_ml.backend._backend", reason="Run scripts/run_tests.sh to build _backend first")

from fhe_ml.ckks.containers.tensor import PlaintextTensor  # noqa: E402


class TestPlaintextTensor2D:
    def test_shape_and_ndim(self):
        t = PlaintextTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert t.shape == (2, 3)
        assert t.ndim == 2

    def test_len(self):
        t = PlaintextTensor([[1.0], [2.0], [3.0]])
        assert len(t) == 3

    def test_repr(self):
        t = PlaintextTensor([[1.0, 2.0]])
        assert "PlaintextTensor" in repr(t)
        assert "(1, 2)" in repr(t)


class TestPlaintextTensorMeta:
    def test_meta_holds_shape_and_factorization(self):
        t = PlaintextTensor([[float(c) for c in range(16)] for _ in range(4)])
        assert t.meta.shape == (4, 16)
        # n1 = nearest power of two to sqrt(16) = 4; n2 = ceil(16 / 4) = 4
        assert (t.meta.n1, t.meta.n2) == (4, 4)
        assert t.meta.n1 * t.meta.n2 >= t.meta.shape[1]

    def test_meta_property_is_read_only(self):
        t = PlaintextTensor([[1.0, 2.0]])
        with pytest.raises(AttributeError):
            t.meta = None  # type: ignore[misc]

    def test_meta_fields_are_immutable(self):
        t = PlaintextTensor([[1.0, 2.0]])
        with pytest.raises(AttributeError):
            t.meta.n1 = 99  # type: ignore[misc]


class TestPlaintextTensorValidation:
    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            PlaintextTensor([1.0, 2.0, 3.0])

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            PlaintextTensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

    def test_4d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            PlaintextTensor([[[[1.0]]]])

    def test_empty_outer_raises(self):
        with pytest.raises(ValueError):
            PlaintextTensor([])

    def test_inconsistent_row_length_raises(self):
        with pytest.raises(ValueError, match="Dimension mismatch"):
            PlaintextTensor([[1.0, 2.0], [3.0]])

    def test_non_numeric_leaf_raises(self):
        with pytest.raises(ValueError, match="numeric"):
            PlaintextTensor([["a", "b"], ["c", "d"]])

    def test_zero_dimension_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            PlaintextTensor([[], []])


class TestPlaintextTensorFromNumpy:
    def test_from_numpy_2d(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = PlaintextTensor.from_numpy(arr)
        assert t.shape == (2, 2)

    def test_from_non_array_raises(self):
        with pytest.raises(TypeError):
            PlaintextTensor.from_numpy("not_an_array")
