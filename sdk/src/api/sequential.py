from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

from core.errors import LayerConfigError
from core.layer import AffineLayer, Layer
from core.utils import to_numpy
from api.input import Input
from api.tensor import PlaintextTensor

if TYPE_CHECKING:
    from api.ciphertext import EncryptedVector
    from api.context import FHEContext


class Sequential:
    """Chain of layers and activation functions for encrypted vectors.

    Each element must be a `Layer` (defined in `core.layer`). Use
    `model.input(context, raw_data)` to validate, reshape, and encrypt raw
    user data into an `Input` in the layout this model's first layer expects.
    The model's `__call__` accepts either an `Input` or an `EncryptedVector`.

    Call `model.compile(context)` once before inference: it sizes the network
    against the context's level budget and, only if the network is too deep,
    enables SLIM bootstrapping and plans where to refresh. Models that fit the
    budget never bootstrap.

    Example:
        model = Sequential([
            Conv2D(1, 4, 3, (28, 28), conv_w, bias=conv_b),
            ReLU(),
            Linear(4 * 26 * 26, 10, lin_w),
        ])
        model.compile(context)
        inp = model.input(context, image_28x28)
        out = model(inp)
    """

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
            # An activation must be flanked by weighted layers, so calibration
            # and folding can always assume affine neighbours.
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
        # Calibrated per-neuron max|input| per activation-layer index;
        # populated by compile() when it is given calibration data.
        self._activation_ranges: Dict[int, np.ndarray] = {}

    def input(self, context: "FHEContext", raw_data: object) -> Input:
        """Validate, reshape, and encrypt `raw_data` via the first layer's `prepare_input`."""
        flat = self._layers[0].prepare_input(raw_data)
        return Input(context, flat)

    def compile(
        self, context: "FHEContext", calibration_data: object = None
    ) -> "Sequential":
        """Bind the context, optionally calibrate activation ranges, set up bootstrapping.

        `calibration_data` (optional): any iterable of input batches — a PyTorch
        `DataLoader`, a single 2-D array/tensor, etc. A plaintext pass measures
        each activation layer's input range, stored in `activation_ranges`.

        If the network's total multiplicative depth exceeds the context's level
        budget, SLIM bootstrapping is set up. The refreshes themselves are
        applied lazily by the layers during inference (`FHEContext._prepare_for`)
        — including mid-layer, so an activation deeper than one bootstrap cycle
        still runs. A network that fits the budget never bootstraps.
        """
        self._context = context
        if calibration_data is not None:
            self._calibrate(calibration_data)
            self._fold_calibration()

        total_depth = sum(l.mult_depth() for l in self._layers)
        if total_depth > context._usable_levels():
            context._setup_bootstrapping()
        return self

    @property
    def activation_ranges(self) -> Dict[int, np.ndarray]:
        """Calibrated per-neuron max|input| per activation-layer index.

        Maps an activation layer's index to a vector of one bound per neuron.
        Empty until `compile` is given calibration data.
        """
        return dict(self._activation_ranges)

    def forward_plain(self, x: np.ndarray) -> np.ndarray:
        """Plaintext numpy forward pass through every layer — reference/calibration."""
        x = np.asarray(x, dtype=float)
        for layer in self._layers:
            x = layer.forward_plain(x)
        return x

    def _calibrate(self, calibration_data: object) -> None:
        """Measure the per-neuron max|input| at each activation layer.

        Activations are the non-affine layers; their input is the output of the
        preceding weighted layer. Each layer's range is a vector (one bound per
        neuron), accumulated over all batches and stored in `activation_ranges`.
        """
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
                    peak = np.abs(x).max(axis=0)  # per-neuron, over the batch
                    ranges[i] = (
                        peak if i not in ranges else np.maximum(ranges[i], peak)
                    )
                x = layer.forward_plain(x)
        self._activation_ranges = ranges

    def _fold_calibration(self) -> None:
        """Fold per-neuron calibration ranges into the surrounding Linear layers.

        For each activation at index i with range B: the preceding weighted
        layer is rescaled so its output (the activation input) is divided by B,
        and the following weighted layer multiplies it back by B. ReLU's
        positive-homogeneity makes this exact — the network is unchanged, but
        the activation now only ever sees inputs normalized to about [-1, 1].
        `__init__` guarantees both neighbours are weighted layers.
        """
        for i, b in self._activation_ranges.items():
            pre, post = self._layers[i - 1], self._layers[i + 1]
            # Floor avoids dividing by zero on a dead (always-0) neuron.
            b = np.maximum(np.asarray(b, dtype=float), 1e-12)
            pre._weight = PlaintextTensor.from_numpy(pre._weight.to_numpy() / b[:, None])
            if pre._bias is not None:
                pre._bias = (np.asarray(pre._bias, dtype=float) / b).tolist()
            post._weight = PlaintextTensor.from_numpy(post._weight.to_numpy() * b[None, :])

    @staticmethod
    def _iter_batches(calibration_data: object):
        """Yield input batches as float ndarrays.

        A single 2-D array/tensor is one batch; anything else is iterated.
        Batch coercion (torch detach, `(inputs, targets)` unwrap) is delegated
        to `core.utils.to_numpy`.
        """
        if hasattr(calibration_data, "ndim") and calibration_data.ndim == 2:
            yield to_numpy(calibration_data)
        else:
            for batch in calibration_data:
                yield to_numpy(batch)

    def __call__(self, x: Union[Input, "EncryptedVector"]) -> "EncryptedVector":
        # Each layer self-manages bootstrapping via FHEContext._prepare_for —
        # no schedule here; refreshes fire lazily as levels run low.
        if isinstance(x, Input):
            x = x.ciphertext
        for layer in self._layers:
            x = layer(x)
        return x
