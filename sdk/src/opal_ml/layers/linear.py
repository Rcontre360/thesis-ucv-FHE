import numpy as np
import torch.nn as nn

from opal_ml.ckks.containers.tensor import PlaintextTensor
from opal_ml.layers.base import AffineLayer
from opal_ml.utils.errors import ShapeError
from opal_ml.utils.validate import check_array


class Linear(AffineLayer):
    """Fully-connected layer y = W @ x + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: object,
        bias: object | None = None,
    ) -> None:
        weight = check_array(weight, shape=(out_features, in_features), name="weight")
        bias_list: list[float] | None = None
        if bias is not None:
            bias_list = check_array(bias, shape=(out_features,), name="bias").tolist()
        super().__init__(
            in_features, out_features, PlaintextTensor.from_numpy(weight), bias_list
        )

    def prepare_input(self, raw_data: object) -> list[float]:
        arr = check_array(raw_data, name="Linear input")
        if arr.ndim != 1:
            raise ShapeError(
                f"Linear input must be 1-D, got {arr.ndim}-D — "
                "did you mean to use Conv2D as the first layer?"
            )
        if arr.size != self.in_features:
            raise ShapeError(f"input size {arr.size} != in_features {self.in_features}")
        return arr.tolist()

    @classmethod
    def from_torch(
        cls,
        module: nn.Linear,
        input_shape: tuple[int, ...],
    ) -> tuple["Linear", tuple[int, ...]]:
        expected = int(np.prod(input_shape))
        if expected != module.in_features:
            raise ShapeError(
                f"Linear expects {module.in_features} features but the previous "
                f"layer outputs {expected} (shape={input_shape})"
            )
        bias = module.bias.detach().numpy() if module.bias is not None else None
        layer = cls(
            module.in_features,
            module.out_features,
            module.weight.detach().numpy(),
            bias,
        )
        return layer, (module.out_features,)
