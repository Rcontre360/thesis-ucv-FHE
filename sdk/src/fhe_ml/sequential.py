from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import torch.nn as nn

from fhe_ml.utils.errors import LayerConfigError
from fhe_ml.layers.base import AffineLayer, Layer
from fhe_ml.utils import to_numpy
from fhe_ml.layers.relu import ReLU
from fhe_ml.layers.input import Input
from fhe_ml.layers.conv2d import Conv2D
from fhe_ml.layers.linear import Linear
from fhe_ml.ckks.containers.tensor import PlaintextTensor

if TYPE_CHECKING:
    from fhe_ml.ckks.containers.ciphertext import EncryptedVector
    from fhe_ml.ckks.context import FHEContext


class Sequential:
    """Chain of layers for encrypted-vector inference."""

    def __init__(self, layers: List[Layer]) -> None:
        if not layers:
            raise ValueError("Sequential requires at least one layer")
        for i, layer in enumerate(layers):
            if not isinstance(layer, Layer):
                raise TypeError(
                    f"layers[{i}] is {type(layer).__name__}; must inherit from `Layer`."
                )
            if isinstance(layer, AffineLayer):
                continue
            flanked = (
                0 < i < len(layers) - 1
                and isinstance(layers[i - 1], AffineLayer)
                and isinstance(layers[i + 1], AffineLayer)
            )
            if not flanked:
                raise LayerConfigError(
                    f"layers[{i}] ({type(layer).__name__}) is an activation; it "
                    "must sit between two weighted layers (Linear/Conv2D)."
                )
        self._layers = list(layers)
        self._context: Optional["FHEContext"] = None
        self._activation_ranges: Dict[int, np.ndarray] = {}

    @classmethod
    def from_torch(
        cls,
        model: nn.Module,
        input_shape: Tuple[int, ...],
    ) -> "Sequential":
        """Build from a torch model. `input_shape` is one sample's shape."""
        torch_map = {
            nn.Linear: Linear,
            nn.Conv2d: Conv2D,
            nn.ReLU: ReLU,
        }
        layers: List[Layer] = []
        shape = tuple(input_shape)
        for m in model.cpu():
            if isinstance(m, nn.Flatten):
                shape = (int(np.prod(shape)),)
                continue
            sdk_cls = torch_map.get(type(m))
            if sdk_cls is None:
                raise TypeError(f"unsupported torch layer: {type(m).__name__}")
            layer, shape = sdk_cls.from_torch(m, shape)
            layers.append(layer)
        return cls(layers)

    def input(self, context: "FHEContext", raw_data: object) -> Input:
        flat = self._layers[0].prepare_input(raw_data)
        return Input(context, flat)

    def compile(
        self, context: "FHEContext", calibration_data: object = None
    ) -> "Sequential":
        self._context = context
        for layer in self._layers:
            if isinstance(layer, ReLU) and layer._degrees is None:
                layer._set_degrees(context.config.relu_degrees)
        if calibration_data is not None:
            self._calibrate(calibration_data)
            self._fold_calibration()

        total_depth = sum(l.mult_depth() for l in self._layers)
        if total_depth > context._usable_levels():
            context._setup_bootstrapping()
        return self

    @property
    def activation_ranges(self) -> Dict[int, np.ndarray]:
        return dict(self._activation_ranges)

    def forward_plain(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        for layer in self._layers:
            x = layer.forward_plain(x)
        return x

    def _calibrate(self, calibration_data: object) -> None:
        act_idx = {
            i
            for i, layer in enumerate(self._layers)
            if not isinstance(layer, AffineLayer)
        }
        ranges: Dict[int, np.ndarray] = {}
        for batch in self._iter_batches(calibration_data):
            x = np.atleast_2d(batch)
            for i, layer in enumerate(self._layers):
                if i in act_idx:
                    peak = np.abs(x).max(axis=0)
                    ranges[i] = (
                        peak if i not in ranges else np.maximum(ranges[i], peak)
                    )
                x = layer.forward_calibration(x)
        self._activation_ranges = ranges

    def _fold_calibration(self) -> None:
        for i, b in self._activation_ranges.items():
            pre, post = self._layers[i - 1], self._layers[i + 1]
            b = np.maximum(np.asarray(b, dtype=float), 1e-12)
            pre._weight = PlaintextTensor.from_numpy(pre._weight.to_numpy() / b[:, None])
            if pre._bias is not None:
                pre._bias = (np.asarray(pre._bias, dtype=float) / b).tolist()
            post._weight = PlaintextTensor.from_numpy(post._weight.to_numpy() * b[None, :])

    @staticmethod
    def _iter_batches(calibration_data: object):
        if hasattr(calibration_data, "ndim") and calibration_data.ndim == 2:
            yield to_numpy(calibration_data)
        else:
            for batch in calibration_data:
                yield to_numpy(batch)

    def __call__(self, x: Union[Input, "EncryptedVector"]) -> "EncryptedVector":
        if isinstance(x, Input):
            x = x.ciphertext
        for layer in self._layers:
            x = layer(x)
        return x
