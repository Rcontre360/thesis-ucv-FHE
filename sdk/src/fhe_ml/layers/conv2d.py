from typing import List, Optional, Tuple, Union

import numpy as np
import torch.nn as nn

from fhe_ml.utils.errors import LayerConfigError, ShapeError
from fhe_ml.layers.base import AffineLayer
from fhe_ml.utils.validate import check_array
from fhe_ml.ckks.containers.tensor import PlaintextTensor


class Conv2D(AffineLayer):
    """2-D convolution as a single plaintext-matrix matmul."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        input_shape: Tuple[int, int],
        weight: object,
        bias: Optional[object] = None,
        stride: int = 1,
    ) -> None:
        if isinstance(kernel_size, int):
            k_h = k_w = kernel_size
        else:
            k_h, k_w = kernel_size
        if stride < 1:
            raise LayerConfigError(f"stride must be >= 1, got {stride}")
        H, W = input_shape
        H_out = (H - k_h) // stride + 1
        W_out = (W - k_w) // stride + 1
        if H_out <= 0 or W_out <= 0:
            raise LayerConfigError(
                f"Kernel {k_h}x{k_w} with stride {stride} does not fit input {H}x{W}"
            )

        weight = check_array(
            weight, shape=(out_channels, in_channels, k_h, k_w), name="weight"
        )
        if bias is not None:
            bias = check_array(bias, shape=(out_channels,), name="bias")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (k_h, k_w)
        self.input_shape = (H, W)
        self.output_shape = (H_out, W_out)
        self.stride = stride

        in_features = in_channels * H * W
        out_features = out_channels * H_out * W_out
        self.in_features = in_features
        self.out_features = out_features
        bias_list: Optional[List[float]] = (
            np.repeat(bias, H_out * W_out).tolist() if bias is not None else None
        )
        super().__init__(
            in_features,
            out_features,
            PlaintextTensor.from_numpy(self._build_conv_matrix(weight)),
            bias_list,
        )

    def _build_conv_matrix(self, weight: np.ndarray) -> np.ndarray:
        H, W = self.input_shape
        k_h, k_w = self.kernel_size
        s = self.stride
        P = self.output_shape[0] * self.output_shape[1]

        # For every output position, the flat input indices its window covers.
        idx = np.arange(self.in_features).reshape(self.in_channels, H, W)
        win = np.lib.stride_tricks.sliding_window_view(idx, (k_h, k_w), axis=(1, 2))
        win = win[:, ::s, ::s].reshape(self.in_channels, P, k_h * k_w)

        M = np.zeros((self.out_features, self.in_features))
        rows = np.arange(P)[:, None]
        for oc in range(self.out_channels):
            block = M[oc * P:(oc + 1) * P]
            for ic in range(self.in_channels):
                block[rows, win[ic]] = weight[oc, ic].reshape(-1)
        return M

    def prepare_input(self, raw_data: object) -> List[float]:
        arr = check_array(raw_data, name="Conv2D input")
        H, W = self.input_shape
        C = self.in_channels
        if arr.ndim == 1:
            if arr.size != self.in_features:
                raise ShapeError(
                    f"flat input size {arr.size} != in_features {self.in_features}"
                )
        elif arr.ndim == 2:
            if C != 1:
                raise ShapeError(f"2-D input requires in_channels=1, but layer has {C}")
            if arr.shape != (H, W):
                raise ShapeError(f"2-D input shape {arr.shape} != ({H}, {W})")
        elif arr.ndim == 3:
            if arr.shape != (C, H, W):
                raise ShapeError(f"3-D input shape {arr.shape} != ({C}, {H}, {W})")
        else:
            raise ShapeError(f"expected 1-D/2-D/3-D input, got {arr.ndim}-D")
        return arr.reshape(-1).tolist()

    @classmethod
    def from_torch(
        cls,
        module: nn.Conv2d,
        input_shape: Tuple[int, ...],
    ) -> Tuple["Conv2D", Tuple[int, int, int]]:
        if len(input_shape) != 3:
            raise ShapeError(
                f"Conv2D expects (C, H, W) input shape, got {input_shape}"
            )
        C_in, H, W = input_shape
        if C_in != module.in_channels:
            raise ShapeError(
                f"Conv2d expects {module.in_channels} input channels, got {C_in}"
            )
        stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
        bias = module.bias.detach().numpy() if module.bias is not None else None
        layer = cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            input_shape=(H, W),
            weight=module.weight.detach().numpy(),
            bias=bias,
            stride=stride,
        )
        H_out, W_out = layer.output_shape
        return layer, (module.out_channels, H_out, W_out)
