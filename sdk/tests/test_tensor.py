import pytest

pytest.importorskip("fhe_sdk._backend", reason="Run scripts/run_tests.sh to build _backend first")

from fhe_sdk.tensor import PlaintextTensor  # noqa: E402


class TestPlaintextTensor2D:
    def test_shape_and_ndim(self):
        t = PlaintextTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert t.shape == (2, 3)
        assert t.ndim == 2

    def test_len(self):
        t = PlaintextTensor([[1.0], [2.0], [3.0]])
        assert len(t) == 3

    def test_getitem_returns_list(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        t = PlaintextTensor(data)
        assert t[0] == [1.0, 2.0]
        assert t[1] == [3.0, 4.0]

    def test_getitem_out_of_range_raises(self):
        t = PlaintextTensor([[1.0, 2.0]])
        with pytest.raises(IndexError):
            _ = t[5]

    def test_repr(self):
        t = PlaintextTensor([[1.0, 2.0]])
        assert "PlaintextTensor" in repr(t)
        assert "(1, 2)" in repr(t)


class TestPlaintextTensor3D:
    def test_shape_and_ndim(self):
        data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        t = PlaintextTensor(data)
        assert t.shape == (2, 2, 2)
        assert t.ndim == 3

    def test_getitem_returns_2d_tensor(self):
        data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        t = PlaintextTensor(data)
        slice_ = t[0]
        assert isinstance(slice_, PlaintextTensor)
        assert slice_.shape == (2, 2)
        assert slice_.ndim == 2

    def test_getitem_chained(self):
        data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        t = PlaintextTensor(data)
        assert t[1][0] == [5.0, 6.0]


class TestPlaintextTensorValidation:
    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2D or 3D"):
            PlaintextTensor([1.0, 2.0, 3.0])

    def test_4d_raises(self):
        with pytest.raises(ValueError, match="2D or 3D"):
            PlaintextTensor([[[[1.0]]]])

    def test_empty_outer_raises(self):
        with pytest.raises(ValueError):
            PlaintextTensor([])

    def test_inconsistent_row_length_raises(self):
        with pytest.raises(ValueError, match="Dimension mismatch"):
            PlaintextTensor([[1.0, 2.0], [3.0]])

    def test_inconsistent_depth_row_raises(self):
        with pytest.raises(ValueError, match="Dimension mismatch"):
            PlaintextTensor([[[1.0, 2.0], [3.0]], [[4.0, 5.0], [6.0, 7.0]]])

    def test_non_numeric_leaf_raises(self):
        with pytest.raises(ValueError, match="numeric"):
            PlaintextTensor([["a", "b"], ["c", "d"]])

    def test_zero_dimension_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            PlaintextTensor([[], []])


class TestPlaintextTensorFromNumpy:
    def test_from_numpy_2d(self):
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not installed")
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = PlaintextTensor.from_numpy(arr)
        assert t.shape == (2, 2)

    def test_from_non_array_raises(self):
        with pytest.raises(TypeError):
            PlaintextTensor.from_numpy("not_an_array")
