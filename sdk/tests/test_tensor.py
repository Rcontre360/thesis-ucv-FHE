import pytest

pytest.importorskip("core._backend", reason="Run scripts/run_tests.sh to build _backend first")

from api.tensor import PlaintextTensor  # noqa: E402


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


class TestPlaintextTensorGetDiagonal2D:
    # Matrix used across tests:
    #   [[1, 2, 3],
    #    [4, 5, 6],
    #    [7, 8, 9]]
    DATA = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

    def test_main_diagonal(self):
        # k=0, 3x3 → min(3,3)=3 elements
        t = PlaintextTensor(self.DATA)
        assert t.get_diagonal(0) == [1.0, 5.0, 9.0]

    def test_upper_diagonal(self):
        # k=1, c_offset=1: (0,1),(1,2),(2,0) → 3 elements
        t = PlaintextTensor(self.DATA)
        assert t.get_diagonal(1) == [2.0, 6.0, 7.0]

    def test_lower_diagonal(self):
        # k=-1, r_offset=1: (1,0),(2,1),(0,2) → 3 elements
        t = PlaintextTensor(self.DATA)
        assert t.get_diagonal(-1) == [4.0, 8.0, 3.0]

    def test_max_size_limits_output(self):
        t = PlaintextTensor(self.DATA)
        assert len(t.get_diagonal(0, max_size=2)) == 2

    def test_max_size_extends_cyclically(self):
        # max_size > min(rows,cols) → cyclic extension up to n_rows*n_cols
        t = PlaintextTensor(self.DATA)
        assert t.get_diagonal(0, max_size=6) == [1.0, 5.0, 9.0, 1.0, 5.0, 9.0]

    def test_max_size_capped_at_n_rows_times_n_cols(self):
        t = PlaintextTensor(self.DATA)
        assert len(t.get_diagonal(0, max_size=100)) == 9

    def test_rectangular_matrix(self):
        # 2x3 matrix: min(2,3)=2 elements by default
        t = PlaintextTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert t.get_diagonal(0) == [1.0, 5.0]

    def test_rectangular_matrix_extended(self):
        # max_size=6 visits all 6 elements cyclically
        t = PlaintextTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert t.get_diagonal(0, max_size=6) == [1.0, 5.0, 3.0, 4.0, 2.0, 6.0]

    def test_axis_and_slice_ignored_for_2d(self):
        t = PlaintextTensor(self.DATA)
        assert t.get_diagonal(0) == t.get_diagonal(0, axis=1, slice_index=2)


class TestPlaintextTensorGetDiagonal3D:
    # Cube shape (2, 3, 3):
    #   layer 0: [[1,2,3],[4,5,6],[7,8,9]]
    #   layer 1: [[10,11,12],[13,14,15],[16,17,18]]
    DATA = [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
    ]

    def test_axis0_slice0_main_diagonal(self):
        # axis=0, slice=0 → data[0,:,:] = 3x3, min(3,3)=3 elements
        t = PlaintextTensor(self.DATA)
        assert t.get_diagonal(0, axis=0, slice_index=0) == [1.0, 5.0, 9.0]

    def test_axis0_slice1_main_diagonal(self):
        # axis=0, slice=1 → data[1,:,:] = [[10,11,12],[13,14,15],[16,17,18]]
        t = PlaintextTensor(self.DATA)
        assert t.get_diagonal(0, axis=0, slice_index=1) == [10.0, 14.0, 18.0]

    def test_axis1_slice1_main_diagonal(self):
        # axis=1, slice=1 → data[:,1,:] = [[4,5,6],[13,14,15]] shape (2,3), min=2
        t = PlaintextTensor(self.DATA)
        assert t.get_diagonal(0, axis=1, slice_index=1) == [4.0, 14.0]

    def test_axis2_slice0_main_diagonal(self):
        # axis=2, slice=0 → data[:,:,0] = [[1,4,7],[10,13,16]] shape (2,3), min=2
        t = PlaintextTensor(self.DATA)
        assert t.get_diagonal(0, axis=2, slice_index=0) == [1.0, 13.0]

    def test_axis_out_of_range_raises(self):
        t = PlaintextTensor(self.DATA)
        with pytest.raises(ValueError):
            t.get_diagonal(0, axis=3)

    def test_slice_index_out_of_range_raises(self):
        t = PlaintextTensor(self.DATA)
        with pytest.raises(IndexError):
            t.get_diagonal(0, axis=0, slice_index=5)


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
