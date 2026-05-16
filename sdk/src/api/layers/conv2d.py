from typing import TYPE_CHECKING, List, Optional, Tuple, Union

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
        weight: List,
        bias: Optional[List[float]] = None,
        stride: int = 1,
    ) -> None:
        if isinstance(kernel_size, int):
            k_h = k_w = kernel_size
        else:
            k_h, k_w = kernel_size
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        H, W = input_shape
        H_out = (H - k_h) // stride + 1
        W_out = (W - k_w) // stride + 1
        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"Kernel {k_h}x{k_w} with stride {stride} does not fit input {H}x{W}"
            )

        if len(weight) != out_channels:
            raise ValueError(
                f"weight must have {out_channels} output channels, got {len(weight)}"
            )
        for oc, oc_w in enumerate(weight):
            if len(oc_w) != in_channels:
                raise ValueError(
                    f"weight[{oc}] must have {in_channels} input channels, got {len(oc_w)}"
                )
            for ic, ic_w in enumerate(oc_w):
                if len(ic_w) != k_h or any(len(row) != k_w for row in ic_w):
                    raise ValueError(
                        f"weight[{oc}][{ic}] must be {k_h}x{k_w}"
                    )
        if bias is not None and len(bias) != out_channels:
            raise ValueError(
                f"bias length {len(bias)} != out_channels {out_channels}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (k_h, k_w)
        self.input_shape = (H, W)
        self.output_shape = (H_out, W_out)
        self.stride = stride
        self.in_features = in_channels * H * W
        self.out_features = out_channels * H_out * W_out

        self._weight = PlaintextTensor(self._build_conv_matrix(weight))
        self._bias = self._expand_bias(bias) if bias is not None else None

    def _build_conv_matrix(self, weight: List) -> List[List[float]]:
        H, W = self.input_shape
        H_out, W_out = self.output_shape
        k_h, k_w = self.kernel_size
        s = self.stride
        in_size = self.in_features
        out_size = self.out_features

        M: List[List[float]] = [[0.0] * in_size for _ in range(out_size)]
        for oc in range(self.out_channels):
            for oh in range(H_out):
                for ow in range(W_out):
                    out_idx = (oc * H_out + oh) * W_out + ow
                    for ic in range(self.in_channels):
                        for kh in range(k_h):
                            ih = oh * s + kh
                            for kw in range(k_w):
                                iw = ow * s + kw
                                in_idx = (ic * H + ih) * W + iw
                                M[out_idx][in_idx] = float(weight[oc][ic][kh][kw])
        return M

    def _expand_bias(self, bias: List[float]) -> List[float]:
        H_out, W_out = self.output_shape
        per_map = H_out * W_out
        expanded: List[float] = []
        for oc in range(self.out_channels):
            expanded.extend([float(bias[oc])] * per_map)
        return expanded

    def prepare_input(self, raw_data: object) -> List[float]:
        """Reshape flat / 2-D (H,W) / 3-D (C,H,W) data (or numpy) to flat C·H·W."""
        if hasattr(raw_data, "tolist"):
            raw_data = raw_data.tolist()
        if not isinstance(raw_data, (list, tuple)):
            raise TypeError(
                f"Conv2D.prepare_input expects a list/array, got {type(raw_data).__name__}"
            )

        H, W = self.input_shape
        C = self.in_channels

        if raw_data and isinstance(raw_data[0], (int, float)):
            flat = [float(v) for v in raw_data]
            if len(flat) != self.in_features:
                raise ValueError(
                    f"flat input size {len(flat)} != in_features {self.in_features}"
                )
            return flat

        first = raw_data[0]
        is_2d = isinstance(first, (list, tuple)) and (
            len(first) == 0 or isinstance(first[0], (int, float))
        )

        if is_2d:
            if C != 1:
                raise ValueError(
                    f"2-D input requires in_channels=1, but layer has {C}"
                )
            self._check_shape_2d(raw_data, H, W)
            return [float(v) for row in raw_data for v in row]

        if len(raw_data) != C:
            raise ValueError(
                f"3-D input has {len(raw_data)} channels, expected {C}"
            )
        flat: List[float] = []
        for ic, channel in enumerate(raw_data):
            self._check_shape_2d(channel, H, W, channel_idx=ic)
            for row in channel:
                flat.extend(float(v) for v in row)
        return flat

    @staticmethod
    def _check_shape_2d(data: object, H: int, W: int, channel_idx: int = -1) -> None:
        tag = "" if channel_idx < 0 else f"channel {channel_idx} "
        if not isinstance(data, (list, tuple)) or len(data) != H:
            raise ValueError(f"{tag}expected {H} rows, got {len(data)}")
        for r, row in enumerate(data):
            if not isinstance(row, (list, tuple)) or len(row) != W:
                raise ValueError(
                    f"{tag}row {r} expected {W} cols, got {len(row)}"
                )

    def __call__(self, x: "EncryptedVector") -> "EncryptedVector":
        if x.size != self.in_features:
            raise ValueError(
                f"input size {x.size} != in_features {self.in_features}"
            )
        out = x.matmul(self._weight)
        if self._bias is not None:
            out = out + self._bias
        return out
