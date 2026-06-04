from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch.nn as nn

from fhe_ml.utils.errors import ShapeError

if TYPE_CHECKING:
    from fhe_ml.ckks.containers.ciphertext import EncryptedVector
    from fhe_ml.ckks.containers.tensor import PlaintextTensor


class Layer(ABC):
    """Sequential element: EncryptedVector -> EncryptedVector."""

    @abstractmethod
    def __call__(self, x: "EncryptedVector") -> "EncryptedVector":
        ...

    @abstractmethod
    def mult_depth(self) -> int:
        ...

    @abstractmethod
    def forward_plain(self, x: np.ndarray) -> np.ndarray:
        ...

    @classmethod
    @abstractmethod
    def from_torch(
        cls,
        module: nn.Module,
        input_shape: Tuple[int, ...],
    ) -> Tuple["Layer", Tuple[int, ...]]:
        ...

    def forward_calibration(self, x: np.ndarray) -> np.ndarray:
        return self.forward_plain(x)

    def prepare_input(self, raw_data: object) -> List[float]:
        raise NotImplementedError(
            f"{type(self).__name__} cannot be a model's input layer; "
            "use a weighted layer (Linear, Conv2D, ...) as the first element."
        )


class AffineLayer(Layer):
    """y = W @ x + b with plaintext W and b."""

    in_features: int
    out_features: int
    _weight: "PlaintextTensor"
    _bias: Optional[List[float]]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: "PlaintextTensor",
        bias: Optional[List[float]],
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self._weight = weight
        self._bias = bias

    def __call__(self, x: "EncryptedVector") -> "EncryptedVector":
        if x.size != self.in_features:
            raise ShapeError(
                f"input size {x.size} != in_features {self.in_features}"
            )
        x = x._context._prepare_for(x, 1)
        out = x.matmul(self._weight)
        if self._bias is not None:
            out = out + self._bias
        return out

    def mult_depth(self) -> int:
        return 1

    def forward_plain(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=float) @ self._weight.to_numpy().T
        if self._bias is not None:
            out = out + np.asarray(self._bias, dtype=float)
        return out
