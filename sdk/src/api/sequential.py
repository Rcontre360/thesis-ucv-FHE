from typing import TYPE_CHECKING, List, Union

from core.layer import Layer
from api.input import Input

if TYPE_CHECKING:
    from api.ciphertext import EncryptedVector
    from api.context import FHEContext


class Sequential:
    """Chain of layers and activation functions for encrypted vectors.

    Each element must be a `Layer` (defined in `core.layer`). Use
    `model.input(context, raw_data)` to validate, reshape, and encrypt raw
    user data into an `Input` in the layout this model's first layer expects.
    The model's `__call__` accepts either an `Input` or an `EncryptedVector`.

    Example:
        model = Sequential([
            Conv2D(1, 4, 3, (28, 28), conv_w, bias=conv_b),
            ReLU(),
            Linear(4 * 26 * 26, 10, lin_w),
        ])
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
        self._layers = list(layers)

    def input(self, context: "FHEContext", raw_data: object) -> Input:
        """Validate, reshape, and encrypt `raw_data` via the first layer's `prepare_input`."""
        flat = self._layers[0].prepare_input(raw_data)
        return Input(context, flat)

    def __call__(self, x: Union[Input, "EncryptedVector"]) -> "EncryptedVector":
        if isinstance(x, Input):
            x = x.ciphertext
        for layer in self._layers:
            x = layer(x)
        return x
