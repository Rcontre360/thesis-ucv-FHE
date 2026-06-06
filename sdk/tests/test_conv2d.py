import pytest

pytest.importorskip("fhe_ml.backend._backend", reason="Run scripts/run_tests.sh to build _backend first")

from fhe_ml.layers.conv2d import Conv2D  # noqa: E402
from fhe_ml.layers.linear import Linear  # noqa: E402
from fhe_ml.sequential import Sequential  # noqa: E402
from fhe_ml.layers.input import Input  # noqa: E402

EPSILON = 1e-2


class TestConv2D:
    def test_single_channel_identity_corners(self, built_context):
        # 2x2 kernel with K[0,0]=1, K[1,1]=1, others 0 → each output is
        # input[i,j] + input[i+1,j+1].
        weight = [[[[1.0, 0.0], [0.0, 1.0]]]]
        layer = Conv2D(
            in_channels=1, out_channels=1,
            kernel_size=2, input_shape=(3, 3),
            weight=weight,
        )
        layer._weight.encode(built_context)
        x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        flat = [v for row in x for v in row]
        ct = built_context.encrypt(flat)
        result = layer(ct).decrypt()
        expected = [1 + 5, 2 + 6, 4 + 8, 5 + 9]  # [6, 8, 12, 14]
        assert len(result) == 4
        for a, b in zip(result, expected):
            assert abs(a - b) < EPSILON

    def test_bias_applied(self, built_context):
        weight = [[[[1.0, 0.0], [0.0, 1.0]]]]
        bias = [0.5]
        layer = Conv2D(
            in_channels=1, out_channels=1,
            kernel_size=2, input_shape=(3, 3),
            weight=weight, bias=bias,
        )
        layer._weight.encode(built_context)
        flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        ct = built_context.encrypt(flat)
        result = layer(ct).decrypt()
        expected = [6.5, 8.5, 12.5, 14.5]
        for a, b in zip(result, expected):
            assert abs(a - b) < EPSILON

    def test_multi_output_channels(self, built_context):
        # 2 output channels, both 2x2 kernels.
        weight = [
            [[[1.0, 0.0], [0.0, 1.0]]],  # diagonal
            [[[0.0, 1.0], [1.0, 0.0]]],  # anti-diagonal
        ]
        layer = Conv2D(
            in_channels=1, out_channels=2,
            kernel_size=2, input_shape=(3, 3),
            weight=weight,
        )
        layer._weight.encode(built_context)
        flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        ct = built_context.encrypt(flat)
        result = layer(ct).decrypt()
        # channel 0: input[i,j] + input[i+1,j+1] = [6, 8, 12, 14]
        # channel 1: input[i,j+1] + input[i+1,j] = [2+4, 3+5, 5+7, 6+8] = [6, 8, 12, 14]
        expected = [6, 8, 12, 14, 6, 8, 12, 14]
        for a, b in zip(result, expected):
            assert abs(a - b) < EPSILON

    def test_multi_input_channels(self, built_context):
        # 2 input channels, 1 output channel. Output = sum across in-channels.
        weight = [
            [
                [[1.0, 0.0], [0.0, 0.0]],  # picks input[0][i,j]
                [[0.0, 0.0], [0.0, 1.0]],  # picks input[1][i+1,j+1]
            ]
        ]
        layer = Conv2D(
            in_channels=2, out_channels=1,
            kernel_size=2, input_shape=(2, 2),
            weight=weight,
        )
        layer._weight.encode(built_context)
        # ch0 = [[1,2],[3,4]], ch1 = [[10,20],[30,40]]
        flat = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
        ct = built_context.encrypt(flat)
        result = layer(ct).decrypt()
        # Output is 1x1: ch0[0,0] + ch1[1,1] = 1 + 40 = 41
        assert len(result) == 1
        assert abs(result[0] - 41.0) < EPSILON

    def test_stride(self, built_context):
        # 4x4 input, 2x2 kernel, stride 2 → 2x2 output.
        weight = [[[[1.0, 1.0], [1.0, 1.0]]]]
        layer = Conv2D(
            in_channels=1, out_channels=1,
            kernel_size=2, input_shape=(4, 4),
            weight=weight, stride=2,
        )
        layer._weight.encode(built_context)
        flat = [float(v) for v in range(1, 17)]  # 1..16
        ct = built_context.encrypt(flat)
        result = layer(ct).decrypt()
        # Window sums at strided positions (0,0), (0,2), (2,0), (2,2)
        expected = [1 + 2 + 5 + 6, 3 + 4 + 7 + 8, 9 + 10 + 13 + 14, 11 + 12 + 15 + 16]
        for a, b in zip(result, expected):
            assert abs(a - b) < EPSILON

    def test_wrong_input_size_raises(self, built_context):
        weight = [[[[1.0, 0.0], [0.0, 1.0]]]]
        layer = Conv2D(
            in_channels=1, out_channels=1,
            kernel_size=2, input_shape=(3, 3),
            weight=weight,
        )
        layer._weight.encode(built_context)
        ct = built_context.encrypt([0.1, 0.2, 0.3])
        with pytest.raises(ValueError):
            layer(ct)

    def test_kernel_does_not_fit(self):
        weight = [[[[1.0] * 5 for _ in range(5)]]]
        with pytest.raises(ValueError):
            Conv2D(
                in_channels=1, out_channels=1,
                kernel_size=5, input_shape=(3, 3),
                weight=weight,
            )


class TestConv2DInSequential:
    def test_conv_then_linear_via_input(self, built_context):
        # Conv 2x2 identity diagonal: out has 4 values from 3x3 input.
        conv_w = [[[[1.0, 0.0], [0.0, 1.0]]]]
        # Linear summing all 4 conv outputs into a single value.
        lin_w = [[1.0, 1.0, 1.0, 1.0]]
        model = Sequential([
            Conv2D(1, 1, 2, (3, 3), conv_w),
            Linear(4, 1, lin_w),
        ]).compile(built_context)
        inp = model.input(built_context, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        assert isinstance(inp, Input)
        out = model(inp).decrypt()
        # conv outputs [6, 8, 12, 14]; linear sums to 40.
        assert abs(out[0] - 40.0) < EPSILON

    def test_input_size_check(self, built_context):
        conv_w = [[[[1.0, 0.0], [0.0, 1.0]]]]
        model = Sequential([Conv2D(1, 1, 2, (3, 3), conv_w)])
        with pytest.raises(ValueError):
            model.input(built_context, [1.0, 2.0])  # wrong size

    def test_input_accepts_nested_lists(self, built_context):
        conv_w = [[[[1.0, 0.0], [0.0, 1.0]]]]
        model = Sequential([Conv2D(1, 1, 2, (3, 3), conv_w)])
        # Both flat and nested should produce equivalent Inputs.
        inp_flat = model.input(built_context, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        inp_nested = model.input(built_context, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        assert inp_flat.size == inp_nested.size == 9

    def test_activation_as_first_layer_raises(self):
        from fhe_ml.layers.relu import ReLU
        from fhe_ml.utils.errors import LayerConfigError
        # an activation cannot be the first layer — rejected at construction
        with pytest.raises(LayerConfigError, match="must sit between two weighted layers"):
            Sequential([ReLU(), Linear(2, 2, [[1.0, 0.0], [0.0, 1.0]])])

    def test_non_layer_in_sequential_raises(self):
        with pytest.raises(TypeError, match="must inherit from `Layer`"):
            Sequential([lambda x: x])  # plain callable, not a Layer

    def test_input_dispatches_on_first_layer(self, built_context):
        # Linear-first model: nested input must be rejected.
        lin_model = Sequential([Linear(3, 1, [[1.0, 1.0, 1.0]])])
        with pytest.raises(ValueError):
            lin_model.input(built_context, [[1.0, 2.0, 3.0]])
        # Flat input works.
        inp = lin_model.input(built_context, [1.0, 2.0, 3.0])
        assert inp.size == 3

        # Conv-first model: 3-D (C, H, W) input also works.
        conv_w = [
            [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]
        ]
        conv_model = Sequential([Conv2D(2, 1, 2, (2, 2), conv_w)])
        inp = conv_model.input(
            built_context,
            [[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]],
        )
        assert inp.size == 8
