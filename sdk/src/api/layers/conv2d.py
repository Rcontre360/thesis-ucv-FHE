from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

from api._validate import check_array
from api.errors import LayerConfigError, ShapeError
from api.tensor import PlaintextTensor
from api.layers.base import Layer

if TYPE_CHECKING:
    from api.ciphertext import EncryptedVector


class Conv2D(Layer):
    """2-D convolution as a single plaintext-matrix matmul.

    Materialises the conv at construction time as a sparse matrix of shape
    (C_out·H_out·W_out, C_in·H·W) and runs it through the Halevi-Shoup matmul,
    so output is replicated and chains directly into Linear/Conv2D. Costs
    O(min(in, out)) plain-muls vs `1 + log₂(k²)` for im2col, traded for
    generality over any H, W, k, stride.

    weight has shape (C_out, C_in, k_h, k_w); bias has length C_out. I/O is
    flat row-major: [c0·H·W slots | c1·H·W slots | ...].
    """

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
        self.in_features = in_channels * H * W
        self.out_features = out_channels * H_out * W_out

        self._weight = PlaintextTensor.from_numpy(self._build_conv_matrix(weight))
        self._bias: Optional[List[float]] = (
            np.repeat(bias, H_out * W_out).tolist() if bias is not None else None
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
        """Reshape flat / 2-D (H,W) / 3-D (C,H,W) data (or numpy) to flat C·H·W."""
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

    def __call__(self, x: "EncryptedVector") -> "EncryptedVector":
        if x.size != self.in_features:
            raise ShapeError(
                f"input size {x.size} != in_features {self.in_features}"
            )
        out = x.matmul(self._weight)
        if self._bias is not None:
            out = out + self._bias
        return out
