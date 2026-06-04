import pytest

pytest.importorskip("fhe_ml.backend._backend", reason="Run scripts/run_tests.sh to build _backend first")

from fhe_ml.utils.errors import FHESDKError, ShapeError, LayerConfigError  # noqa: E402
from fhe_ml.utils.validate import check_array  # noqa: E402
from fhe_ml.layers.conv2d import Conv2D  # noqa: E402
from fhe_ml.layers.linear import Linear  # noqa: E402


class TestErrorHierarchy:
    def test_shape_error_is_fhesdk_and_value_error(self):
        assert issubclass(ShapeError, FHESDKError)
        assert issubclass(ShapeError, ValueError)

    def test_layer_config_error_is_fhesdk_and_value_error(self):
        assert issubclass(LayerConfigError, FHESDKError)
        assert issubclass(LayerConfigError, ValueError)

    def test_shape_error_catchable_as_value_error(self):
        with pytest.raises(ValueError):
            raise ShapeError("x")

    def test_both_catchable_as_fhesdk_error(self):
        with pytest.raises(FHESDKError):
            raise ShapeError("x")
        with pytest.raises(FHESDKError):
            raise LayerConfigError("x")


class TestCheckArray:
    def test_returns_float_ndarray(self):
        arr = check_array([[1, 2], [3, 4]])
        assert arr.dtype == float
        assert arr.shape == (2, 2)

    def test_ndim_mismatch_raises_shape_error(self):
        with pytest.raises(ShapeError, match="must be 1-D"):
            check_array([[1, 2], [3, 4]], ndim=1, name="weight")

    def test_shape_mismatch_raises_shape_error(self):
        with pytest.raises(ShapeError, match=r"shape \(2, 2\) != \(3, 3\)"):
            check_array([[1, 2], [3, 4]], shape=(3, 3))

    def test_ragged_input_raises_shape_error(self):
        with pytest.raises(ShapeError, match="numeric and rectangular"):
            check_array([[1, 2], [3]])

    def test_non_numeric_raises_shape_error(self):
        with pytest.raises(ShapeError):
            check_array([["a", "b"]])


class TestLayerErrorTypes:
    def test_conv_bad_stride_raises_layer_config_error(self):
        with pytest.raises(LayerConfigError, match="stride"):
            Conv2D(1, 1, 2, (3, 3), [[[[1.0, 0.0], [0.0, 1.0]]]], stride=0)

    def test_conv_kernel_does_not_fit_raises_layer_config_error(self):
        with pytest.raises(LayerConfigError, match="does not fit"):
            Conv2D(1, 1, 5, (3, 3), [[[[1.0] * 5 for _ in range(5)]]])

    def test_conv_bad_weight_shape_raises_shape_error(self):
        with pytest.raises(ShapeError, match="weight shape"):
            Conv2D(1, 1, 2, (3, 3), [[[[1.0, 0.0]]]])

    def test_linear_bad_weight_shape_raises_shape_error(self):
        with pytest.raises(ShapeError, match="weight shape"):
            Linear(3, 2, [[1.0, 0.0], [0.0, 1.0]])
